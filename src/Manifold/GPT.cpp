#include "console.h"
#include "GPT.hpp"
#include "Ganglia.hpp"

WIKI::WIKI()  {
    // hGPT gpt = std::make_shared<GeneratOnPrompt>(params);       //cys
    // gpt->Init();
    // gpt->Generate();
}

GeneratOnPrompt::GeneratOnPrompt(struct gpt_params&par_,int flag) : params(par_)      {
    std::mt19937 rng(params.seed);
    sparams = params.sparams;
}

void GeneratOnPrompt::DisplayEmbd(bool input_echo,llama_context *ctx,int n_consumed,int flag)   {
    if (input_echo && display) {
        for (auto id : embd) {
            const std::string token_str = llama_token_to_piece(ctx, id);
            printf("%s", token_str.c_str());
            if (embd.size() > 1) {
                input_tokens.push_back(id);
            } else {
                output_tokens.push_back(id);
                output_ss << token_str;
            }
        }
        fflush(stdout);
    }
    // reset color to default if there is no pending user input
    if (input_echo && (int) embd_inp.size() == n_consumed) {
        console::set_display(console::reset);
        display = true;
    }
}

int GeneratOnPrompt::Generate(int flag) {
    Tokenize(flag);
    const int n_ctx = llama_n_ctx(ctx);                     //ctx->cparams.n_ctx;
    const int n_ctx_train = llama_n_ctx_train(model);       //model->hparams.n_ctx_train;
    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
    //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
    //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_TEE("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;
    int ga_i = 0;
    
    ctx_sampling = llama_sampling_init(sparams);
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        int iRet = UpdateEmbed(n_past,n_remain,n_consumed,n_session_consumed,n_past_guidance,ga_i,0x0);
        if( iRet==0)    
            break;
        if( iRet==1)    
            return 1;

        embd.clear();
        embd_guidance.clear();
        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
                LOG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            llama_sampling_accept(ctx_sampling, ctx, id, true);
            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());
            embd.push_back(id);
            // echo this to console
            input_echo = true;
            // decrement remaining sampling budget
            --n_remain;
            LOG("n_remain: %d\n", n_remain);
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }
        DisplayEmbd(input_echo,ctx,n_consumed);           

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = llama_sampling_last(ctx_sampling);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of text token in interactive mode
            if (llama_sampling_last(ctx_sampling) == llama_token_eos(model)) {
                LOG("found EOS token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    is_interacting = true;
                    printf("\n");
                } else if (params.instruct || params.chatml) {
                    is_interacting = true;
                }
            }

            if (n_past > 0 && is_interacting) {
                LOG("waiting for user input\n");

                if (params.instruct || params.chatml) {
                    printf("\n> ");
                }

                if (params.input_prefix_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                std::string buffer;
                if (!params.input_prefix.empty()) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    printf("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(console::user_input);
                display = params.display_prompt;

                std::string line;
                bool another_line = true;
                do {
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(console::reset);
                display = true;

                // Add tokens to embd only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    // instruct mode: insert instruction prefix
                    if (params.instruct && !is_antiprompt) {
                        LOG("inserting instruction prefix\n");
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                    }
                    // chatml mode: insert user chat prefix
                    if (params.chatml && !is_antiprompt) {
                        LOG("inserting chatml prefix\n");
                        n_consumed = embd_inp.size();
                        embd_inp.insert(embd_inp.end(), cml_pfx.begin(), cml_pfx.end());
                    }
                    if (params.escape) {
                        process_escapes(buffer);
                    }

                    const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = ::llama_tokenize(ctx, buffer,              false, false);
                    const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

                    LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    // instruct mode: insert response suffix
                    if (params.instruct) {
                        LOG("inserting instruction suffix\n");
                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                    }
                    // chatml mode: insert assistant chat suffix
                    if (params.chatml) {
                        LOG("inserting chatml suffix\n");
                        embd_inp.insert(embd_inp.end(), cml_sfx.begin(), cml_sfx.end());
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        output_tokens.push_back(token);
                        output_ss << llama_token_to_piece(ctx, token);
                    }

                    n_remain -= line_inp.size();
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = false;
            }
        }

        // end of text token
        if (!embd.empty() && embd.back() == llama_token_eos(model) && !(params.instruct || params.interactive || params.chatml)) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    // if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
    //     LOG_TEE("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
    //     llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    // }
    return 0x0;
}

int GeneratOnPrompt::UpdateEmbed(int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag)  {
    const int n_ctx = llama_n_ctx(ctx);                     //ctx->cparams.n_ctx;
    const int n_ctx_train = llama_n_ctx_train(model);       //model->hparams.n_ctx_train;
    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (!embd.empty()) {
        // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
        // --prompt or --file which uses the same value.
        int max_embd_size = n_ctx - 4;

        // Ensure the input doesn't exceed the context size by truncating embd if necessary.
        if ((int) embd.size() > max_embd_size) {
            const int skipped_tokens = (int) embd.size() - max_embd_size;
            embd.resize(max_embd_size);

            console::set_display(console::error);
            printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            console::set_display(console::reset);
            fflush(stdout);
        }

        if (ga_n == 1) {
            // infinite text generation via context shifting
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int) embd.size() + std::max<int>(0, guidance_offset) > n_ctx) {
                if (params.n_predict == -2) {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    return 0;
                }

                const int n_left    = n_past - params.n_keep;
                const int n_discard = n_left/2;

                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                n_past -= n_discard;

                if (ctx_guidance) {
                    n_past_guidance -= n_discard;
                }

                LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

                LOG("embd: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

                LOG("clear session path\n");
                path_session.clear();
            }
        } else {
            // context extension via Self-Extend
            while (n_past >= ga_i + ga_w) {
                const int ib = (ga_n*ga_i)/ga_w;
                const int bd = (ga_w/ga_n)*(ga_n - 1);
                const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                LOG("\n");
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                n_past -= bd;

                ga_i += ga_w/ga_n;

                LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
            }
        }

        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
        if (n_session_consumed < (int) session_tokens.size()) {
            size_t i = 0;
            for ( ; i < embd.size(); i++) {
                if (embd[i] != session_tokens[n_session_consumed]) {
                    session_tokens.resize(n_session_consumed);
                    break;
                }

                n_past++;
                n_session_consumed++;

                if (n_session_consumed >= (int) session_tokens.size()) {
                    ++i;
                    break;
                }
            }
            if (i > 0) {
                embd.erase(embd.begin(), embd.begin() + i);
            }
        }

        // evaluate tokens in batches
        // embd is typically prepared beforehand to fit within a batch, but not always
        assert (ctx_guidance==nullptr); 

        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = (int) embd.size() - i;
            if (n_eval > params.n_batch) {
                n_eval = params.n_batch;
            }

            LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

            if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                LOG_TEE("%s : failed to eval\n", __func__);
                return 1;
            }

            n_past += n_eval;

            LOG("n_past = %d\n", n_past);
            // Display total tokens alongside total time
            if (params.n_print > 0 && n_past % params.n_print == 0) {
                LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
            }
        }

        if (!embd.empty() && !path_session.empty()) {
            session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
            n_session_consumed = session_tokens.size();
        }
    }
    return 2;
}