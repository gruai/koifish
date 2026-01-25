/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  Some user cases
 *     bubble ./scripts/qwen3.json
 *
 *  \brief bubble - chat,answer some questions,...
 *  \author Yingshi Chen
 */
#include <filesystem>
#include <iostream>
#include <string>

#include "./CLI_params.hpp"
#include "./Device/Pipe.hpp"
#include "./Manifold/Fish.hpp"
#include "./Utils/GST_os.hpp"
#include "GoPT.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

std::string UserPrompt(hFISH fish, int pos, int nRound, int flag = 0x0) {
    char* cli_user_prompt = nullptr;
    char* system_prompt   = nullptr;
    int szBuffer          = fish->config.chat_sampler.szBuffer;
    char user_prompt[szBuffer], rendered_prompt[szBuffer];
    if (cli_user_prompt != NULL) {
        if (pos > 0)
            return "";
        strcpy(user_prompt, cli_user_prompt);
    } else {
        if (nRound < DEBUG.prompts.size())
            strcpy(user_prompt, DEBUG.prompts[nRound].c_str());  //
        else
            read_stdin("\n>> ", user_prompt, sizeof(user_prompt));
        if (!user_prompt[0])
            return "";  // exit on empty prompt
    }

    // render the prompt with the correct template
    if (pos == 0 && system_prompt) {
        sprintf(rendered_prompt, fish->config.model.system_prompt_template.c_str(), system_prompt, user_prompt);
    } else {
        sprintf(rendered_prompt, fish->config.model.prompt_template.c_str(), user_prompt);
    }
    return rendered_prompt;
}

//  Keeping the prefill process uncompressed is crucial for performance maintenance
int Prefill(hFISH fish, int enable_thinking) { return 0x0; }

int OnEOS(hFISH fish, int flag = 0x0) {
    hChater gopt = fish->GetGenerator();
    _INFO("[MEMORY] %s\t%s\n", SUM::GPU_Info(0x0).c_str(), SUM::CPU_Info(0x0).c_str());
    _INFO("\t quant=%s\t DEBUG_switch={generate=%d %d} QKV=%d FFN=%d\n", SUM::sQuantInfo.c_str(), DEBUG.verGenerate, DEBUG.T_cuQK, DEBUG.verInferQKV, DEBUG.verInferFFN);
    _INFO("\n");  // next turn
    return 0x0;
}

/*
    hello
        Hello! How can I assist you today?
    More questions:
        1. Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?
        2. just keep asking 142857*7, would get some strange answers!
        3. prime factorization of 996   99997 failed @ 4B

        4. How many games did Arsenal FC go unbeaten during the 2003-2004 season of the English Premier League         correct_answer: "38"
        5. Write a function to print the Fibonacci sequence to the nth digit, but write and comment it like a pirate
        6. I get out on the top floor (third floor) at street level. How many stories is the building above the ground?
        7. "无知觉明",无是指什么? 知是指什么? 觉是指什么? 明是指什么?
        8. 天命玄鸟,降而生生. 玄鸟是什么鸟?
        9. 尚书·商书·胤征

    Common sense
        What is the capital of Shanghai?
        Who wrote the play Romeo and Juliet?
        In which year did the Titanic sink?
        What is the chemical symbol for the element gold?
        What is the longest river in the world?
*/
int Chat(hFISH fish, int enable_thinking) {
    fish->Statistic(0x100);

    int seq_len           = fish->config.chat_sampler.seq_len;
    int num_prompt_tokens = 0, user_turn = 1, next, token, generated_tokens = 0, nRound = 0;  // pos = 0,
    TOKENS prompt_tokens;
    hTokenizer tokenizer = fish->GetTokenizer();
    double start_time    = 0;
    string cur_answer, rendered_prompt;
    floatLogits* logits = nullptr;
    hChater gopt        = fish->GetGenerator();
    hBATCH hBatch       = fish->GetCurBatch(true);
    assert(hBatch->size() == seq_len);
    QWEN3_PIPE qwen_pipe(fish, 0x0);

    // DEBUG.T_generate_most_layer = 1;
    DEBUG.verGenerate     = DEBUG.cmd_p1;  // use this flag to comparse accu/time of different version
    DEBUG.T_cuQK          = 0;
    DEBUG.T_kvcache_quant = 0;
    g_dump_level          = 1;

    while (1) {
        if (user_turn) {
            rendered_prompt   = UserPrompt(fish, hBatch->tok_pos, nRound);
            prompt_tokens     = tokenizer->Encode(rendered_prompt);
            num_prompt_tokens = prompt_tokens.size();
            generated_tokens  = 0;
            cur_answer        = "";
            user_turn         = 0, nRound++;
            hBatch->Reset(prompt_tokens);  // No BOS at sequence start!
            // hLoader->InitOneSamp(rendered_prompt, nullptr, fish.get(), 0x110);
            _INFO("\n");
        }

        if (hBatch->tok_pos == num_prompt_tokens - 1) {
            start_time = GST_ms();
            SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
        }
        if (hBatch->tok_pos == 1) {  // nRound == 2
            DEBUG_HERE;
            // exit(KOIFISH_EXIT_DEBUG);
        }
        if (DEBUG.verGenerate) {  //  much slower with some bug
            float eval = fish->Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
        } else {  // qwen_pipe.UpdatePos(token);
            logits = T_generate_(fish, &qwen_pipe, fish->config.model.tpActivation, 1);
        }
        if (!gopt->VerifyLogits()) {
            _INFO("\n%s(Invalid logits!)%s\n", COLOR_RED, COLOR_RESET);
        }
        hBatch->tok_pos++;
        // K_EXIT(KOIFISH_EXIT_DEBUG);

        // _INFO(" %d[%d->%d]", pos, token, next), fflush(stdout);
        if (hBatch->tok_pos >= num_prompt_tokens) {
            token = gopt->Sample(-1);
            generated_tokens++;
            if (token == tokenizer->eos_id || hBatch->tok_pos >= seq_len) {  //  stop generation if get EOS token
                double elapsed_s = (double)(GST_ms() - start_time) / 1000.0;
                double tps       = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
                if (hBatch->tok_pos >= seq_len) {
                    _WARN("%scontext window full!%s\t", COLOR_YELLOW, COLOR_RESET);
                }
                _INFO("\n%s[%.2f tk/s, %d tokens in %.2fs(qkv=%.3fs ffn=%.3fs PreLogits=%.3fs X=%.3fs)]%s\n===================================\n", COLOR_GREEN,
                      tps, generated_tokens - 1, elapsed_s, SUM::tQKV_forw / 1.0e6, SUM::tFFN / 1.0e6, SUM::tPreLogits / 1.0e6, SUM::tX1 / 1.0e6, COLOR_RESET);

                user_turn = 1;
                cur_answer += "\t\t" + SUM::sQuantInfo;
                STR2FILE("chat.csv", cur_answer, nRound == 1 ? std::ofstream::out : std::ofstream::app);
                OnEOS(fish);
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

                    // flip the bold state and change color accordingly
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
                if (token != tokenizer->eos_id) {
                    _INFO("%s", current_pos);
                    cur_answer += current_pos;
                }
            }

            fflush(stdout);
        }
    }
    // free(prompt_tokens);
    return 0x0;
}

int main(int argc, char* argv[]) {
    try {
        assert(argc >= 2);
        std::string arg_prefix = "--", exec_name = EXE_name(), jsPath = "", eval_metric = "";
        CLI_params config;
        config.phase = LIFE_PHASE::P_GENERATE;  // DEBUG.test_quant = 1;
        if (!config.parse(argc, argv)) {
            return KOIFISH_INVALID_ARGS;
        }
        config.OnArch();

        config.isOnlyGPT                = true;
        config.chat_sampler.mode        = config.model.enable_thinking ? CHAT_MODE::CHATML_THINK : CHAT_MODE::CHATML_ASSIST;
        config.chat_sampler.isSampleCPU = true;
        config.model.preLogits_dB       = 1;
        config.model.sparse.method      = -1;
        // config.quant.T_errQ             = 0.3;
        // config.quant.isNormalFloat = true, config.quant.isSymmetric = false;       //use_double_quant
        // config.quant.default_bits       = 2;
        config.dumpSwitch.tensor_load  = 0;
        config.dumpSwitch.nn_structure = 0;
        // SUM::nMinTensorAlloc = 1;
        // config.quant.filter_MIQ         = {"mlp"};                   //  mlp.down_proj.weight "mlp"
        // config.quant.filter_KVcache = {"0.self_attn"};    //   "layers.27.mlp" model.blk.0.attn

        DEBUG.verCuda = 1, DEBUG.T_cpu = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
        DEBUG.verInferQKV = 0, DEBUG.verInferFFN = 0;
        config.Dump(0x100);
        hFISH fish = Fish::MakeInstance("PPL_", config, {}, Fish::ROLE_TYPE::COMMON, 0x110);
        SUM::MemoryInfo(0x0, 0x0);
        if (fish == nullptr)
            return KOIFISH_NULL_FISH;

        Chat(fish, config.model.enable_thinking);
        return 0x0;
    } catch (const SafeExit& ex) {
        std::string info = ex.getFormattedInfo();
        _WARN("KOIFISH exit: %s %s", ex.what(), info.c_str());
        // std::cerr << info << std::endl;
        // cleanup();
        return ex.getExitCode();
    } catch (const std::exception& e) {
        _INFO("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char* info) {
        _INFO("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}

#if (defined _WINDOWS) || (defined WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    char str_version[1000];
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            GRUAI_KOIFISH_VERSION(str_version);
            _INFO("%s", str_version);
            break;
        case DLL_THREAD_ATTACH:
            break;
        default:
            break;
    }

    return TRUE;
}
#else
// https://stackoverflow.com/questions/22763945/dll-main-on-windows-vs-attribute-constructor-entry-points-on-linux
__attribute__((constructor)) void dllLoad() {
    char str_version[1000];
    GRUAI_KOIFISH_VERSION(str_version, 0x0);
    _INFO("%s", str_version);
    _INFO("\n");
}

__attribute__((destructor)) void dllUnload() {}
#endif