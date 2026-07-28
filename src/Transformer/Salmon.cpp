/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Acknowledgement: https://github.com/andrewkchan/deepseek.cpp
 *
 *  \brief Scoring(返璞) model
 *  \author Yingshi Chen
 */
#include "../Manifold/gLLM.hpp"
#include "../Utils/GST_Application.hpp"

Salmon::Salmon(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_SCORE_);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
    // config.model.isQKVBias    = false;
    config.model.isQKVBias = true;  // for https://huggingface.co/fredzzp/open-dcoder-0.5B
    // config.model.isFFNGate    = false;
    config.model.norm_rms_eps = 1.0e-6;
    config.model.preLogits_dB = -1;

    if (isTrain()) {
        config.model.qkv4dnn = QKV_PACK::QQKKVV;
    } else {
    }
    // DEBUG.cmd_p1 = 1;

    config.model.isSeparateQKV = true;
    // config.scheduling.strategy = MEM_STRATEGY::MEM_SWAP_GUOKE;
    // config.scheduling.strategy     = MEM_STRATEGY::PRE_ALLOC_HOST_MAP;
    config.model.isQKNormal = false;
    config.model.sLayer     = "layers.";
    config.model.sEmbed = "embed_tokens", config.model.sInvEmbed = "lm_head";
    config.model.isBqkv        = false;  //  0.6B has no bias!
    config.model.isCausalMask  = false;
    config.fuyou.filter_reload = {"mlp", "self_attn"};  //  {"mlp", "self_attn"};
}

int Salmon::ZhuoMo(int flag) { return 0x0; }

int Salmon::Chat(int enable_thinking, LIFE_PHASE outer_phase, int flag) {
    // Statistic(0x100);

    int seq_len           = config.chat_sampler.seq_len;
    int num_prompt_tokens = 0, user_turn = 1, next, token, generated_tokens = 0, nRound = 0;  // pos = 0,
    TOKENS prompt_tokens;
    hTokenizer tokenizer = GetTokenizer();
    double start_time = 0, eval = 0;
    string cur_answer, rendered_prompt;
    hChater gopt       = GetGenerator();
    Head4Token* header = GetNeuron<Head4Token>("Head4Token", 0);
    header->dump_flag  = -1;  // only for debug
    hBATCH hBatch      = GetCurBatch(true);
    assert(hBatch->hostToken->ne[0] >= seq_len);  // batch = hBatch->hostToken->ne[1] may >1
    GST_Application* hApp = GST_Application::GetInstance();
    // DEBUG.T_generate_most_layer = 1;
    DEBUG.verGenerate = DEBUG.cmd_p1;  // use this flag to comparse accu/time of different version
    // DEBUG.verGenerate     = 1;
    DEBUG.T_cuQK          = 0;
    DEBUG.T_kvcache_quant = 0;
    // g_dump_level          = -1;
    gopt->Prepare4N( );
    while (hApp->iRunning() > 0) {
        if (user_turn) {
            num_prompt_tokens = hBatch->FillPrompt(this, DEBUG.prompts, {}, nRound);
            generated_tokens  = 0;
            cur_answer        = "";
            user_turn         = 0, nRound++;
        }
        if (hApp->iRunning() <= 0) {
            _WARN("\n%s[APP] Stop running! code=%d%s\t", COLOR_YELLOW, hApp->iRunning(), COLOR_RESET);
            break;
        }

        start_time = GST_ms();
        SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
        eval = Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
        gopt->OnLogits();
        hBatch->tok_pos++;
        // K_EXIT(KOIFISH_EXIT_DEBUG);

        // _INFO(" %d[%d->%d]", pos, token, next), fflush(stdout);

        token = gopt->Sample(hBatch);  // 3347
        generated_tokens++;
        if (token == tokenizer->S.eos || hBatch->tok_pos >= seq_len) {  //  stop generation if get EOS token
            double elapsed_s = (double)(GST_ms() - start_time) / 1000.0;
            double tps       = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
            if (hBatch->tok_pos >= seq_len) {
                if (outer_phase == P_TRAIN)
                    return 0x0;
                _WARN("%scontext window full!%s\t", COLOR_YELLOW, COLOR_RESET);
            }
            _INFO("\n%s[%.2f tk/s, %d tokens in %.2fs(qkv=%.3fs ffn=%.3fs PreLogits=%.3fs X=%.3fs)]%s\n===================================\n", COLOR_GREEN, tps,
                  generated_tokens - 1, elapsed_s, SUM::tQKV_forw / 1.0e6, SUM::tFFN / 1.0e6, SUM::tPreLogits / 1.0e6, SUM::tX1 / 1.0e6, COLOR_RESET);

            user_turn = 1;
            cur_answer += "\t\t" + SUM::sQuantInfo;
            STR2FILE("chat.csv", cur_answer, nRound == 1 ? std::ofstream::out : std::ofstream::app);
            // OnEOS(shared_from_this());
            if (nRound == DEBUG.prompts.size()) {  // only for debug
                return 0x0;
            }
            continue;
        }
        hBatch->Set(hBatch->tok_pos, 0, 0, 0, token);

        static int in_thinking_section = 0;
        static int in_bold_section     = 0;
        if (hBatch->tok_pos == num_prompt_tokens) {  // first token of the response
            in_thinking_section = enable_thinking;   // reset thinking state
            in_bold_section     = 0;                 // reset bold state
            if (in_thinking_section) {
                _INFO(COLOR_YELLOW);
            }
        }

        const char* piece = tokenizer->T2STR(token).c_str();  // decode(tokenizer, token);
        if (strcmp(piece, "</think>") == 0) {
            in_thinking_section = 0;
            if (!in_bold_section) {
                _INFO(COLOR_RESET);
            }
        } else {
            const char *current_pos = piece, *marker;
            while ((marker = strstr(current_pos, "**")) != NULL) {
                // print the text before the marker
                fwrite(current_pos, 1, marker - current_pos, stdout);

                // flip the bold state and change colour accordingly
                in_bold_section = !in_bold_section;
                if (in_bold_section) {
                    _INFO(COLOR_BOLD_RED);
                } else if (in_thinking_section) {
                    _INFO(COLOR_YELLOW);
                } else {
                    _INFO(COLOR_RESET);
                }
                current_pos = marker + 2;  // Move past the "**"
            }
            // print any remaining text after the last marker
            if (token != tokenizer->S.eos) {
                _INFO("%s", current_pos);
                cur_answer += current_pos;
            }
        }

        fflush(stdout);
    }
    // free(prompt_tokens);
    return 0x0;
}

std::string Salmon::NN2NAME(const std::string& prefix, tpNEURON4NAME neuron, const std::string& suffix, int flag) {
    size_t pos   = 0x0;
    string tName = "";
    switch (neuron) {
        case ATTN_PRE_NORMAL:
            pos   = prefix.rfind(".");
            tName = prefix.substr(0, pos) + ".input_layernorm";  //   model.layers.0.self_attn => model.layers.0.input_layernorm
            break;
        case FFN_PRE_NORMAL:
            pos   = prefix.rfind(".");
            tName = prefix.substr(0, pos) + ".post_attention_layernorm";
            break;
        case ATTN_Q_NORM:
            tName = prefix + ".q_norm";
            break;
        case ATTN_K_NORM:
            tName = prefix + ".k_norm";
            break;
        case ATTN_Q:
            tName = prefix + ".q_proj";
            break;
        case ATTN_K:
            tName = prefix + ".k_proj";
            break;
        case ATTN_V:
            tName = prefix + ".v_proj";
            break;
        case ATTN_OUT:
            tName = prefix + ".o_proj";
            break;
        case LN_RSTD:
            tName = prefix + ".rstd";
            break;
        case FFN_UP:
            tName = prefix + ".up_proj";
            break;  //  ".w1"
        case FFN_RELU:
            return prefix + "_relu";
        case FFN_DOWN:
            tName = prefix + ".down_proj";
            break;  //  ".w2"
        case FFN_GATE:
            tName = prefix + ".gate_proj";
            break;  //  ".w3"
        default:
            assert(0);
    }
    if (!suffix.empty())
        tName += suffix;
    return tName;
}