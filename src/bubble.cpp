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
#include "./g_stddef.hpp"
#include "GoPT.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

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
        7. why "无知觉明" ?
        8. 天命玄鸟,降而生生. 玄鸟是什么鸟?

    Common sense
        What is the capital of Shanghai?
        Who wrote the play Romeo and Juliet?
        In which year did the Titanic sink?
        What is the chemical symbol for the element gold?
        What is the longest river in the world?
*/
int Chat(hFISH fish, char* cli_user_prompt, char* system_prompt, int enable_thinking) {
    int PROMPT_BUFFER_SIZE = fish->config.model.PROMPT_BUFFER_SIZE;
    int seq_len            = fish->config.model.SEQ_LEN;
    char user_prompt[PROMPT_BUFFER_SIZE], rendered_prompt[PROMPT_BUFFER_SIZE];
    int num_prompt_tokens = 0, user_turn = 1, next, token, pos = 0, generated_tokens = 0, nRound = 0;
    TOKENS prompt_tokens;
    hTokenizer tokenizer = fish->GetTokenizer();
    double start_time    = 0;
    string cur_answer;
    g_dump_level        = 1;
    hGOPT gopt          = fish->GetGOPT();
    hBATCH hBatch       = fish->GetCurBatch(true);

    while (1) {
        if (pos >= seq_len) {
            printf("\n%s(context window full, clearing)%s\n", COLOR_YELLOW, COLOR_RESET);
            user_turn = 1;
            pos       = 0;
        }

        if (user_turn) {
            if (cli_user_prompt != NULL) {
                if (pos > 0)
                    break;
                strcpy(user_prompt, cli_user_prompt);
            } else {
                if (nRound < DEBUG.prompts.size())
                    strcpy(user_prompt, DEBUG.prompts[nRound].c_str());  //
                else
                    read_stdin("\n>> ", user_prompt, sizeof(user_prompt));
                if (!user_prompt[0])
                    break;  // exit on empty prompt
            }

            // render the prompt with the correct template
            if (pos == 0 && system_prompt) {
                sprintf(rendered_prompt, fish->config.model.system_prompt_template.c_str(), system_prompt, user_prompt);
            } else {
                sprintf(rendered_prompt, fish->config.model.prompt_template.c_str(), user_prompt);
            }

            // encode the prompt & reset the position for the new sequence
            prompt_tokens     = tokenizer->Encode(rendered_prompt);
            num_prompt_tokens = prompt_tokens.size();
            pos               = 0;
            user_turn         = 0, nRound++;
            hBatch->Reset(prompt_tokens);
            // hLoader->InitOneSamp(rendered_prompt, nullptr, fish.get(), 0x110);
            printf("\n");
        }

        if (pos == num_prompt_tokens) {
            start_time       = GST_ms();
            generated_tokens = 0;
            cur_answer       = "";
        }
        if (nRound == 2) {  // pos == 13
            int debug = 0;
            // g_dump_level = 0, printf("\n");
        }
        if (pos < num_prompt_tokens)
            token = prompt_tokens[pos];
        else {
            token = next;
            hBatch->Set(pos, 0, 0, 0, token);
        }
        if(0)
            float eval = fish->Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
        else
            T_generate_(fish, token, fish->config.model.tpActivation, -666);
        pos++;
        next = gopt->Sample(pos);
        // printf(" %d[%d->%d]", pos, token, next), fflush(stdout);
        if (pos > num_prompt_tokens) {
            generated_tokens++;

            static int in_thinking_section = 0;
            static int in_bold_section     = 0;

            if (pos == num_prompt_tokens + 1) {
                // first token of the response
                in_thinking_section = enable_thinking;  // reset thinking state
                in_bold_section     = 0;                // reset bold state
                if (in_thinking_section) {
                    printf(COLOR_YELLOW);
                }
            }

            const char* piece = tokenizer->T2STR(token).c_str();  // decode(tokenizer, token);
            if (strcmp(piece, "</think>") == 0) {
                in_thinking_section = 0;
                if (!in_bold_section) {
                    printf(COLOR_RESET);
                }
            } else {
                const char *current_pos = piece, *marker;
                while ((marker = strstr(current_pos, "**")) != NULL) {
                    // print the text before the marker
                    fwrite(current_pos, 1, marker - current_pos, stdout);

                    // flip the bold state and change color accordingly
                    in_bold_section = !in_bold_section;
                    if (in_bold_section) {
                        printf(COLOR_BOLD_RED);
                    } else if (in_thinking_section) {
                        printf(COLOR_YELLOW);
                    } else {
                        printf(COLOR_RESET);
                    }
                    current_pos = marker + 2;  // Move past the "**"
                }
                // print any remaining text after the last marker
                printf("%s", current_pos);
                cur_answer += current_pos;
            }

            fflush(stdout);

            // stop generation if we sample an EOS token
            if (next == tokenizer->eos_id) {
                double elapsed_s = (double)(GST_ms() - start_time) / 1000.0;
                double tps       = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
                printf("\n\n%s[%.2f tk/s, %d tokens in %.2fs]%s", COLOR_GREEN, tps, generated_tokens - 1, elapsed_s, COLOR_RESET);
                printf("\n===================================\n");
                user_turn = 1;
                if (nRound == DEBUG.prompts.size()) {  // only for debug
                    STR2FILE("chat.csv", cur_answer);
                    return 0x0;
                }
                continue;
            }
        }
    }
    // free(prompt_tokens);
    return 0x0;
}

int main(int argc, char* argv[]) {
    try {
        assert(argc >= 2);
        int enable_thinking    = 0;
        std::string arg_prefix = "--", exec_name = EXE_name(), jsPath = "", eval_metric = "";
        CLI_params config;
        config.phase = LIFE_PHASE::P_GENERATE;
        if (!config.parse(argc, argv)) {
            return KOIFISH_INVALID_ARGS;
        }
        config.OnArch();

        config.isOnlyGPT              = true;
        config.chat_mode              = enable_thinking ? CHAT_MODE::CHATML_THINK : CHAT_MODE::CHATML_ASSIST;
        config.model.preLogits_dB     = 1;
        config.model.sparse.method    = -1;
        config.dumpSwitch.tensor_load = 1;
        SUM::nMinTensorAlloc          = 1;  // g_dump_level = -1;

        DEBUG.T_cuda_ver = 1, DEBUG.T_cpu = 0, DEBUG.cmd_p1 = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
        config.Dump(0x100);

        // fish->isLocalInfer = flag == 0x110;
        hFISH fish      = Fish::MakeInstance("PPL_", config, {}, Fish::ROLE_TYPE::COMMON, 0x110);
        hOptimizer hOPT = fish->GetOptimizer();
        if (hOPT->val_loaders.empty())
            return KOIFISH_DATALOADER_EMPTY;

        char *prompt = NULL, *system_prompt = NULL;

        Chat(fish, prompt, system_prompt, enable_thinking);

        return 0x0;
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