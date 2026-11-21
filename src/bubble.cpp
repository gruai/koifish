/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
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

#include "./Shoal/gLLM.hpp"
#include "./Utils/GST_Application.hpp"
#include "./Utils/GST_log.hpp"
#include "./g_def_x.hpp"

// struct ArgsFirst {
//     Args m_args;
//     explicit ArgsFirst(int argc, char* argv[]) : m_args(ParseCmdlineStatic(argc, argv)) {
//         if (!m_args.ok) { cli_parser.help(); std::exit(EXIT_FAILURE); }
//     }
// };
bool CmdLineParser::bubble_configure(int flag) {
    add_positional("config_file", "path of the config file", false);
    add("hf", "path of hugging face directory", "--hf", false);
    add("prompts", "A string array contain 1 or some prompts", "--prompts", false);

    add("seq_len", "KV/state capacity (default 4096)", "--seq_len", false);
    add("max-context", "KV/state capacity (default 4096)", "--max-context", false);
    add("temperature", "A string array contain 1 or some prompts", "--temperature", false);
    add("md_method", "method of mask-diffusion", "--md_method", false);

    bool success = parse();
    return success;
}

bool CmdLineParser::strata_configure(int flag) {
    add_positional("config_file", "path of the config file", false);
    add("strata", "directory of strata-data ", "--strata", false);
    add("pack_folder", "directory of strata-pack ", "--pack", false);
    add("mtp_folder", "directory of mtp ", "--mtp", false);
    // add("hf", "path of hugging face directory", "--hf", false);
    add("prompts", "A string array contain 1 or some prompts", "--prompts", false);

    add("seq_len", "KV/state capacity (default 4096)", "--seq_len", false);
    add("max-context", "KV/state capacity (default 4096)", "--max-context", false);
    add("temperature", "A string array contain 1 or some prompts", "--temperature", false);
    add("md_method", "method of mask-diffusion", "--md_method", false);

    bool success = parse();
    return success;
}

class BubbleApp : public GST_Application {
   protected:
    hFISH fish           = nullptr;
    LIFE_PHASE chatPhase = LIFE_PHASE::P_CHAT_1;
    vector<hWIKI> wikis;  // reserved for hybrid llm training
    bool ParseCmdline(int flag = 0x0) override {
        bool success = cli_parser.bubble_configure();
        if (!success)
            return false;

        cli_parser.get<int>("seq_len", params.chat_sampler.nSeqRecommend);
        cli_parser.get<float>("temperature", params.chat_sampler.temperature);
        string value = "";
        cli_parser.get<std::string>("md_method", value);
        STR2ENUM("MD_" + value, params.chat_sampler.tpZhuomo);
        return true;
    }

   public:
    BubbleApp(int argc, char* argv[]) : GST_Application("Bubble", argc, argv) {}
    virtual ~BubbleApp() {}

    bool Initialize() override {
        if (!GST_Application::Initialize())
            return false;
        // DEBUG.test_quant = 1;
        auto arch = params.ModelArch();
        params.OnArch();
        if (params.phase == P_EVAL_CHAT) {
            DEBUG.sPromptGuide =
                "You are a strict yes/no classifier. Answer ONLY with the single word 'true' or 'false'. Do not explain, do not add punctuation, do not add "
                "any other text.";
            chatPhase            = P_EVAL_CHAT;
            DEBUG.N_mostEvalSamp = 100;
        } else
            chatPhase = (arch == MD_QWEN || arch == MD_AR_QWEN) ? LIFE_PHASE::P_CHAT_N : LIFE_PHASE::P_CHAT_1;
        params.OnPhase(chatPhase);
        params.chat_sampler.mode = params.chat_sampler.enable_thinking ? CHAT_MODE::CHATML_THINK : CHAT_MODE::CHATML_ASSIST;
        // params.chat_sampler.isSampleCPU = true;
        params.model.preLogits_dB  = 1;
        params.model.sparse.method = -1;
        params.kernels.verHeadLoss = KERNEL_LIB_TYPE::TL_CUDA;
        // params.chat_sampler.seq_len     = 32;
        //
        // params.quant.T_errQ             = 0.3;
        // params.quant.isNormalFloat = true;
        // params.quant.default_bits       = 2;
        params.dumpSwitch.tensor_load = 0;
        // params.dumpSwitch.nn_structure = 0;
        DEBUG.verCuda = 1, DEBUG.T_cpu = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
        DEBUG.verInferQKV = 0, DEBUG.verInferFFN = 0;

        // DEBUG.dump_TensorDetail = 1;

        // config.quant.filter_KVcache = {"0.self_attn"};    //   "layers.27.mlp" model.blk.0.attn
        params.Dump(0x100);
        return true;
    }
    int Swim() override {
        fish = Fish::MakeInstance("Bubble_", params, wikis, MODEL_ROLE::COMMON, 0x110);
        if (fish == nullptr) {
            _ERROR("[APP] %s is nullptr!!!", name.c_str());
            return KOIFISH_NULL_FISH;
        }
        hDataToken coral = nullptr;
        if (fish->isAtPhase({P_EVAL_CHAT})) {
            assert(!fish->tsEval.empty());
            coral = fish->tsEval[0];
            fish->ChatOnCoral(coral);
        } else
            fish->Chat(chatPhase);
        // while (iRunning() > 0) { // no need this loop
        //     usleep(100000);
        // }
        return KOIFISH_OK;
    }
};
int main_demo(int argc, char** argv);
class SwardFish : public BubbleApp {
   public:
    SwardFish(int argc, char* argv[]) : BubbleApp(argc, argv) { name = "SwardFish"; }
    virtual ~SwardFish() {}

    bool ParseCmdline(int flag = 0x0) override {
        bool success = cli_parser.strata_configure();
        if (!success)
            return false;

        cli_parser.get<int>("seq_len", params.chat_sampler.nSeqRecommend);
        cli_parser.get<float>("temperature", params.chat_sampler.temperature);
        string value = "";
        cli_parser.get<std::string>("md_method", value);
        STR2ENUM("MD_" + value, params.chat_sampler.tpZhuomo);
        return true;
    }

    int Swim() override {
#ifdef _USE_STRATA_
//  '--config', 'strata-iq3_s.json'        
std::vector<std::string> args = {params.exec_name, "--tokens", "1,2,3", "--prefill", "auto", "--spec", "4", "--pack",
                                         "/home/cys/Github/infer/Strata-data/packs/iq3_s",  // coder-iq1_m
                                         "--ple-gguf",
                                         // "/all_datasets/strata-data/qwen/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",
                                         "/all_datasets/strata-data/qwen/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf", "--native",
                                         // "/all_datasets/strata-data/qwen/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
                                         "/all_datasets/strata-data/qwen/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf", "--expert-profile",
                                         "/home/cys/Github/infer/strata/data/expert-profile.bin", "--expert-cache", "auto", "--mtp",
                                         "/home/cys/Github/infer/Strata-data/mtp/rt", "--max-context", "65536", "--kv", "int8", "--vision", "--spec-min-p",
                                         "0.70", "--pcie-frac", "0.20"};

        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& s : args) argv.push_back(s.data());  // C++17: char* from std::string
        argv.push_back(nullptr);
        // main_demo(args.size(), argv.data());
/*        LoadStrataPack(paths[0], fish.get());  //  native_preset is always 001-of-002
        for (auto path : paths) {
            LoadGGUF(path, true, 0x0);
        }*/
#else
        NOT_IMPLEMENTED;
#endif
        fish = Fish::MakeInstance("Sward_", params, wikis, MODEL_ROLE::COMMON, 0x110);
        if (fish == nullptr) {
            _ERROR("[APP] %s is nullptr!!!", name.c_str());
            return KOIFISH_NULL_FISH;
        }

        std::vector<std::string> paths = FilesOfDir(params.model.sCardPath, {"gguf"}, 0x0);
        assert(paths.size() >= 2);

        return KOIFISH_OK;
    }
};

int main(int argc, char* argv[]) {
    std::shared_ptr<BubbleApp> hApp = nullptr;
    
    if (strcmp(argv[1], "--strata") == 0)
        hApp = std::make_shared<SwardFish>(argc, argv);
    else
        hApp = std::make_shared<BubbleApp>(argc, argv);
    int iRet = hApp->Run();
    return iRet;
}
