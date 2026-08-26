/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Application
 *  \author Yingshi Chen
 */

#include "GST_Application.hpp"

#include <iostream>

#include "GST_MemBuffer.hpp"
#include "GST_util.hpp"

namespace fs = std::filesystem;

void CUDA_cleanup();
// A safe guard for cuda,supa, or other vendor device lib
class DeviceGuard {
   public:
    DeviceGuard() {}

    //  !!! Do not​ try to clean up CUDA in catch (...), static/ global destruction, ...
    ~DeviceGuard() { CUDA_cleanup(); }

   private:
};
std::string g_sAppName, g_sAppPath;

std::string Time2String(const std::chrono::system_clock::time_point& now, int flag = 0x0) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_buffer;
#ifdef _WIN32
    localtime_s(&tm_buffer, &time);
#else
    localtime_r(&time, &tm_buffer);
#endif
    char buf[1024];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        std::cout << "App name: " << basename(buf) << std::endl;
    }
    g_sAppPath = buf;
    g_sAppName = basename(buf);

    std::stringstream ss;
    ss << std::put_time(&tm_buffer, "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

/*
https://github.com/jermp/cmd_line_parser

void configure(cmd_line_parser::parser& parser) {
    parser.add("perc",                 // name
               "A percentage value.",  // description
               "-p",                   // shorthand
               true,                   // required argument
               false                   // not boolean option (default is false)
    );
    parser.add("input_filename", "An input file name.", "-i", true);

    parser.add("output_filename",       // name
               "An output file name.",  // description
               "-o",                    // shorthand
               false, false);
    parser.add("num_trials", "Number of trials.", "-n", false, false);

    parser.add("sorted", "Sort output.", "--sort", false,
               true  // boolean option: a value is not expected after the shorthand
    );
    parser.add("buffered", "Buffer input.", "--buffer", false, true);

    parser.add("ram", "Amount of ram to use.", "--ram", false, false);
}


*/

bool CLI_params::parse(int argc, char** argv) {
    std::string arg_prefix      = "--", key, value;
    exec_name                   = EXE_name();
    string sExt                 = argc > 1 ? FILE_EXT(argv[1]) : "", jPath;
    FILE_FORMAT_TYPE ckp_format = FILE_JSON;  // bool isHF              = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (fs::exists(arg)) {  //*.json
            jPath = arg;
            if (!LoadJConfig(arg)) {
                return false;
            }
        } else if (arg == "--version") {
        } else if (arg == "p0") {
        } else if (arg == "p1") {
            DEBUG.cmd_p1 = 1;
            _INFO("******************* DEBUG.cmd_p1=%d ******************************\n", DEBUG.cmd_p1);
        } else if (arg == "p2") {
            DEBUG.cmd_p2 = 1;
        } else if (arg == "--quant") {
            sscanf(argv[++i], "%d", &DEBUG.quant_UserMode);
        } else if (arg == "--seq_len") {
            sscanf(argv[++i], "%d", &chat_sampler.nSeqRecommend);
        } else if (arg == "--md_method") {
            chat_sampler.tpZhuomo = strcmp(argv[++i], "dilate") == 0 ? CHAT_SAMPLER::MD_DILATE : CHAT_SAMPLER::MD_LINEAR_TRANSFER;
        } else if (arg == "--hellaswag") {
            eval_metric = "hellaswag";
            assert(i + 1 < argc);
            JSON jEval;
            jEval["type"] = "hellaswag", jEval["glob"] = argv[++i];
            jEval["samp"] = 1.0;
            // jEval["glob"] = argv[i++];
            jConfig["datasets_new"]["eval"] = jEval;
        } else if (arg == "--hf") {  // directory of hf model
            assert(i + 1 < argc);
            model.sCardPath = model.pathCheckPoint = argv[++i];
            if (!VERIFY_DIR_EXIST(model.sCardPath, false)) {
                K_EXIT(KOIFISH_INVALID_ARGS_MODEL);
            }
            ckp_format = CKP_HF;
        } else if (arg == "--fish") {  // directory of hf model
            assert(i + 1 < argc);
            model.pathCheckPoint = argv[++i];
            ckp_format           = CKP_KOIFISH;
        } else if (arg == "--prompts") {  // directory of hf model
            assert(i + 1 < argc);
            string sPrompt = argv[++i];
            if (sPrompt.empty()) {
                DEBUG.prompts = {"hello",
                                 "What is the capital of Shanghai?",
                                 "Who wrote the play Romeo and Juliet?",
                                 "In which year did the Titanic sink?",
                                 "What is the chemical symbol for the element gold?",
                                 "What is the longest river in the world?",
                                 "Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
                                 "How many games did Arsenal FC go unbeaten during the 2003-2004 season of the English Premier League",
                                 "I get out on the top floor (third floor) at street level. How many stories is the building above the ground?",
                                 "天命玄鸟,降而生生. 玄鸟是什么鸟?"};
            } else
                DEBUG.prompts = {sPrompt};
        } else if (arg == "--tokenizer") {  // directory of tokenizer
            assert(i + 1 < argc);
            model.sTokenBinPath = argv[++i];
        } else if (arg == "--step") {
            eval_metric = "hellaswag";
            assert(i + 1 < argc);
            sscanf(argv[++i], "%f", &step);
        } else {
            _ERROR("invalid parameter for argument: %s\n", arg.c_str());
            exit(1);
        }
    }
    DEBUG.T_GEMM  = -1;  //  so many version of gemm
    std::string s = jConfig.dump();
    switch (ckp_format) {
        case CKP_KOIFISH:
            if (!LoadJConfig(model.pathCheckPoint)) {
                return false;
            }
            if (jConfig.contains("datasets")) {
                jConfig.erase("datasets");
            }
            if (jConfig.contains("checkpoint_in")) {  // this would cause many confliction!
                jConfig.erase("checkpoint_in");
            }
            if (jConfig.contains("sft")) {  // this would cause many confliction!
                jConfig.erase("sft");
            }
            model.sTokenBinPath = "./assets/tokenizer_151936.bin";

            model.pad_vocab_size = 151936;
            if (!InitJConfig())
                return false;
            break;
        case CKP_HF: {
            if (!model.InitHugFace(this, jConfig, false, 0x0))
                return false;
            break;
        }
        default:
            assert(!jConfig.empty());
            if (!InitJConfig())
                return false;
            if (!isValid(jPath, "CLI_params::parse"))
                return false;
            break;
    }
    chat_sampler.InitPrefillTemplate(this);
    // Dump(0x100);

    switch (phase) {
        case P_CHAT_1:
            break;
        case P_CHAT_N:
            break;
        case P_EVAL_:
            InitChekcpoints(argc, argv, "checkpoint_in");
            break;
        default:
            InitChekcpoints(argc, argv, "checkpoint_in");
            InitChekcpoints(argc, argv, "checkpoint_out");
            InitAllStates(0x0);
            break;
    }
    OnArch();
    return true;
}

GST_Application::GST_Application(int argc, char* argv[]) {
#ifdef _WIN32
    system("chcp 65001");  // Ensures that Unicode characters (Chinese, emojis, etc.) are displayed correctly in the console output.
#endif
    //  register a cleanup function that will be automatically called when the program exits
    // atexit(CUDA_cleanup);    //CUDA runtime unloading may happen BEFORE your atexit()!   Do not​ try to clean up CUDA in program exits
    g_instance = this;
    SetupSignalHandlers();

    start_time = std::chrono::system_clock::now();
    _INFO("🐠 started at: %s\n\n", Time2String(start_time).c_str());

    if (!params.parse(argc, argv)) {
        std::ostringstream oss;
        for (int i = 0; i < argc; ++i) {
            oss << argv[i];
            if (i != argc - 1)
                oss << " ";
        }
        _ERROR("[APP] exit now! It failed to parse arguments={%s}\n", oss.str().c_str());
        std::exit(EXIT_FAILURE);
    }
}

GST_Application ::~GST_Application() {
    end_time = std::chrono::system_clock::now();
    _INFO("🐠 end at: %s\n", Time2String(end_time).c_str());
}

void GST_Application ::Cleanup() {
    _INFO("[APP] %s Cleanup...\n", name.c_str());
    gBUFF     = nullptr;
    g_running = false;

    // Cleanup Linux-specific resources
    CleanupLogging();
}

int GST_Application ::Run() {
    try {
        if (!Initialize()) {
            throw std::runtime_error("Initialization failed");
        }
        DeviceGuard guard;
        Swim();
        Cleanup();

        return KOIFISH_OK;
    } catch (const SafeExit& e) {
        _ERROR("%s %s", e.what(), e.getFormattedInfo().c_str());
        return e.getExitCode();
    } catch (const std::exception& e) {
        _ERROR("%s", e.what());
        return KOIFISH_INTERNAL_EXCEPTION;
    } catch (const char* info) {
        _ERROR("%s", info);
        return KOIFISH_INTERNAL_EXCEPTION;
    } catch (...) {
        _ERROR("%s  Unknown exception !!!", __func__);
        return KOIFISH_INTERNAL_ERR;
    }
}

bool GST_Application::g_running              = false;
GST_Application* GST_Application::g_instance = nullptr;

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