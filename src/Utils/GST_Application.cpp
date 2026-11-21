/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Application
 *  \author Yingshi Chen
 */

#include "GST_Application.hpp"

#include <filesystem>
#include <iostream>

#include "../Shoal/Neuron.hpp"
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

/*std::string arg_prefix = "--";
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
        } else if (arg == "--temperature") {
            sscanf(argv[++i], "%f", &chat_sampler.temperature);
        } else if (arg == "--md_method") {
            string sType = "MD_";
            STR2ENUM(sType + argv[++i], chat_sampler.tpZhuomo);
            // strcmp(argv[++i], "dilate") == 0 ? CHAT_SAMPLER::MD_DILATE : CHAT_SAMPLER::MD_SNR;
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
            // model.isPaddedCls = true;
            if (!VERIFY_DIR_EXIST(model.sCardPath, false)) {
                K_EXIT(KOIFISH_INVALID_ARGS_MODEL);
            }
            ckp_format = CKP_HF;
        } else if (arg == "--strata") {  // directory of hf model
            assert(i + 1 < argc);
            model.sCardPath = model.pathCheckPoint = argv[++i];
            if (!VERIFY_DIR_EXIST(model.sCardPath, false)) {
                K_EXIT(KOIFISH_INVALID_ARGS_MODEL);
            }
            ckp_format = CKP_GGUF;
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
        } else if (arg == "--model") {  // directory of tokenizer
            assert(i + 1 < argc);
            value = argv[++i];
        } else if (arg == "--coral") {
            assert(i + 1 < argc);
            ++i;  // only for debug
            phase                      = P_EVAL_CHAT;
            chat_sampler.nSeqRecommend = 1024;
            jCoral                     = {{"eval", {{"glob", "/all_datasets/Jev/boolq/data/train*.parquet"}, {"name", "boolq_train"}, {"type", "boolq"}}}};
        } else if (arg == "--step") {
            eval_metric = "hellaswag";
            assert(i + 1 < argc);
            sscanf(argv[++i], "%f", &step);
        } else {
            _ERROR("invalid parameter for argument: %s\n", arg.c_str());
            exit(1);
        }
    }*/

bool CLI_params::OnCmdLine(CmdLineParser* parser, int flag) {
    assert(parser != nullptr);
    cli_parser = parser;  //
    exec_name  = EXE_name();
    // string sExt                 = argc > 1 ? FILE_EXT(argv[1]) : "";
    FILE_FORMAT_TYPE ckp_format = FILE_JSON;  //
    std::string path, jPath, key, value;
    parser->get<std::string>("config_file", jPath, true);
    if (!jPath.empty() && !LoadJConfig(jPath)) {
        return false;
    }
    if (!(path = parser->get<std::string>("strata", path)).empty()) {
        model.sCardPath = model.pathCheckPoint = path;
        ckp_format                             = CKP_STRATA;
    } else if (!(path = parser->get<std::string>("gguf", path)).empty()) {
        model.sCardPath = model.pathCheckPoint = path;
        ckp_format                             = CKP_GGUF;
    } else if (!(path = parser->get<std::string>("hf", path)).empty()) {
        model.sCardPath = model.pathCheckPoint = path;
        ckp_format                             = CKP_HF;
    }
    parser->get<std::string>("prompts", value);
    if (value.empty()) {
    } else
        DEBUG.prompts = {value};

    DEBUG.T_GEMM = -1;  //  so many version of gemm
    // std::string s = jConfig.dump();
    J_dump("./log/_koifish_jApp_0.json", jConfig);
    switch (ckp_format) {
        case CKP_KOIFISH:
            if (!LoadJConfig(model.pathCheckPoint)) {
                return false;
            }
#ifdef _nlohmann_json_
            if (jConfig.contains("datasets")) {
                jConfig.erase("datasets");
            }
            if (jConfig.contains("checkpoint_in")) {  // this would cause many confliction!
                jConfig.erase("checkpoint_in");
            }
            if (jConfig.contains("sft")) {  // this would cause many confliction!
                jConfig.erase("sft");
            }
#endif
            model.sTokenBinPath  = "./assets/tokenizer_151936.bin";  // Deprecated
            model.pad_vocab_size = 151936;
            if (!InitJConfig())
                return false;
            break;
        case CKP_HF: {
            if (!model.InitHugFace(this, jConfig, false, 0x0))
                return false;
            break;
        }
        case CKP_GGUF: {
            // jConfig["model"]["arch"] = "";
        } break;
        case CKP_STRATA: {
            // jConfig["model"]["arch"] = "strata_qwen38";
            jConfig = JSON::parse(
                R"({"model": {
                    "arch": "strata_qwen38",
                    "vocab_size": 248320,
                    "parameter": {
                        "Layer": 48,
                        "transformer": {
                            "Ctx": 1024,"Embed": 1024,"Ffn": 3072,"Head": 24,"KVHead": 2,"head_dim": 256
                        },
                        "tie_word_embeddings": false,
                        "max_pos_embeddings": 32768
                    },
                    "backbone": {"embed_tokens": {"Embedding": []}  }
                }                
                }           )");
            InitJConfig();
        } break;
        default:
            assert(!J_isEmpty(jConfig));  // jConfig.empty()
            if (!InitJConfig())
                return false;
            if (!isValid(jPath, "CLI_params::parse"))
                return false;
            break;
    }
    J_dump("./log/_koifish_jApp_1.json", jConfig);
    chat_sampler.InitPromptTemplate(this);
    // Dump(0x100);

    switch (phase) {
        case P_CHAT_1:
            break;
        case P_CHAT_N:
            break;
        case P_EVAL_:
            InitChekcpoints(parser->m_argc, parser->m_argv, "checkpoint_in");
            break;
        default:
            InitChekcpoints(parser->m_argc, parser->m_argv, "checkpoint_in");
            InitChekcpoints(parser->m_argc, parser->m_argv, "checkpoint_out");
            InitAllStates(0x0);
            break;
    }
    OnArch();
    return true;
}

GST_Application::GST_Application(const string& _name, int argc, char* argv[], int flag) : name(_name), cli_parser(argc, argv) {
#ifdef _WIN32
    system("chcp 65001");  // Ensures that Unicode characters (Chinese, emojis, etc.) are displayed correctly in the console output.
#endif
    //  register a cleanup function that will be automatically called when the program exits
    // atexit(CUDA_cleanup);    //CUDA runtime unloading may happen BEFORE your atexit()!   Do not​ try to clean up CUDA in program exits
    g_instance = this;
    SetupSignalHandlers();
    const bool g_dbg = std::getenv("KOIFISH_VERIFY_DEBUG") != nullptr;
    start_time       = std::chrono::system_clock::now();
    _INFO("🐠 started at: %s\n\n", Time2String(start_time).c_str());
}

GST_Application::~GST_Application() {
    end_time = std::chrono::system_clock::now();
    _INFO("🐠 end at: %s\n", Time2String(end_time).c_str());
}

void GST_Application::Cleanup() {
    _INFO("[APP] %s Cleanup...\n", name.c_str());
    gBUFF     = nullptr;
    g_running = false;

    // Cleanup Linux-specific resources
    CleanupLogging();
}

bool InitRegisterOnce(int flag) {
    static std::once_flag once;
    static bool result = false;
    std::call_once(once, [&] { result = GeNeuron::InitRegister(flag); });
    return result;
}

bool GST_Application::Initialize() {
    try {
        // run_tail();
#ifndef NDEBUG
        std::setvbuf(stdout, nullptr, _IONBF, 0);
#endif
        if (!InitRegisterOnce(0x1))
            return false;

    } catch (const std::exception& e) {
        std::cerr << "init failed: " << e.what() << "\n";
        return 1;
    }

    char path[PATH_MAX] = "\0";
    ssize_t len         = readlink("/proc/self/exe", path, sizeof(path) - 1);  // getpid()
    _INFO("------------------------------------------------------------------------------\n");
    _INFO("[APP] %s Initialization.\tProcess=%s\n", name.c_str(), path);
    KOIFISH_Device_Info(0x0);

    // Linux-specific initializations
    if (!InitializeLogging()) {
        std::cerr << "Logging initialization failed" << std::endl;
        return false;
    }
    _INFO("------------------------------------------------------------------------------\n");

    if (!ParseCmdline()) {
        _ERROR("[APP] exit now! It failed to parse arguments\n");
        cli_parser.help();
        std::exit(EXIT_FAILURE);
    }
    if (!params.OnCmdLine(&cli_parser)) {
        _ERROR("[APP] exit now! It failed to parse arguments\n");
        cli_parser.help();
        std::exit(EXIT_FAILURE);
    }
    /*if (!params.parse(cli_parser)) {
    std::ostringstream oss;
    for (int i = 0; i < argc; ++i) {
        oss << argv[i];
        if (i != argc - 1)
            oss << " ";
    }
    _ERROR("[APP] exit now! It failed to parse arguments={%s}\n", oss.str().c_str());
    std::exit(EXIT_FAILURE);
    }*/
    g_running = true;
    return true;
}

int GST_Application::Run() {
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

bool CmdLineParser::parse() {
    assert(m_argc > 0);
    if (m_argc - 1 < m_required + m_required_pos)
        return abort();

    int num_required = 0;      // required *options* seen
    size_t next_pos  = 0;      // next positional slot to fill
    bool options_end = false;  // set after a bare "--"
    UNORDER_SET_<std::string> parsed_shorthands;
    parsed_shorthands.reserve(m_argc);

    for (int i = 1; i != m_argc; ++i) {
        std::string parsed(m_argv[i]);
        if (!options_end) {
            if (parsed == "-h" || parsed == "--help")
                return abort();
            if (parsed == "--") {  // everything after this is positional
                options_end = true;
                continue;
            }
        }
        // ---- positional argument: anything that does not look like an option ----
        if (options_end || !looks_like_option(parsed)) {
            if (next_pos == m_pos_names.size()) {
                std::cerr << "== error: unexpected argument '" + parsed + "'" << std::endl;
                return abort();
            }
            auto& cmd = m_cmds[m_pos_names[next_pos++]];
            cmd.value = parsed, cmd.isUserSet = true;
            continue;
        }
        // ---- option (your original logic, unchanged except the new error message) ----
        int id = 0;
        if (const auto it = m_shorthands.find(parsed); it == m_shorthands.end()) {
            std::cerr << "== error: shorthand '" + parsed + "' not found" << std::endl;
            return abort();
        } else {
            if (const auto it = parsed_shorthands.find(parsed); it != parsed_shorthands.end()) {
                std::cerr << "== error: shorthand '" + parsed + "' already parsed" << std::endl;
                return abort();
            }
            parsed_shorthands.emplace(parsed);
            id = (*it).second;
        }
        assert(static_cast<size_t>(id) < m_names.size());
        auto const& name = m_names[id];
        auto& cmd        = m_cmds[name];
        if (cmd.is_required)
            num_required += 1;
        if (cmd.is_boolean) {
            parsed = "true";
        } else {
            ++i;
            if (i == m_argc) {
                std::cerr << "== error: option '" << m_argv[i - 1] << "' needs a value" << std::endl;
                return abort();
            }
            parsed = m_argv[i];
        }
        cmd.value     = parsed;
        cmd.isUserSet = true;
    }

    if (next_pos < static_cast<size_t>(m_required_pos)) {
        std::cerr << "== error: missing required argument '" + m_pos_names[next_pos] + "'" << std::endl;
        return abort();
    }
    if (num_required != m_required)
        return abort();
    return true;
}

void CmdLineParser::help() const {
    std::cerr << "Usage: " << m_argv[0];
    for (auto const& n : m_pos_names) std::cerr << (m_cmds.at(n).is_required ? " <" : " [") << n << (m_cmds.at(n).is_required ? ">" : "]");
    std::cerr << " [-h,--help]";
    const auto print = [this](bool with_description) {  // unchanged
        for (size_t i = 0; i != m_names.size(); ++i) {
            auto const& cmd = m_cmds.at(m_names[i]);
            std::cerr << " [" << cmd.shorthand;
            if (!cmd.is_boolean)
                std::cerr << " " << m_names[i];
            std::cerr << "]";
            if (with_description)
                std::cerr << "\n\t" << (cmd.is_required ? "REQUIRED: " : "") << cmd.descr << "\n\n";
        }
    };
    print(false);
    std::cerr << "\n\n";
    for (auto const& n : m_pos_names) {  // positional descriptions first
        auto const& cmd = m_cmds.at(n);
        std::cerr << " " << (cmd.is_required ? "<" : "[") << n << (cmd.is_required ? ">" : "]") << "\n\t" << (cmd.is_required ? "REQUIRED: " : "") << cmd.descr
                  << "\n\n";
    }
    print(true);
    std::cerr << " [-h,--help]\n\tPrint this help text and silently exits." << std::endl;
}