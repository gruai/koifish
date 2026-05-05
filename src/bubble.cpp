/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
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

#include "./Manifold/gLLM.hpp"
#include "./Utils/GST_Application.hpp"
#include "./Utils/GST_log.hpp"
#include "./g_def_x.hpp"

class BubbleApp : public GST_Application {
   protected:
    hFISH fish = nullptr;

   public:
    BubbleApp(int argc, char* argv[]) : GST_Application(argc, argv) {
        name = "Bubble";
        // DEBUG.test_quant = 1;
        params.OnArch();
        params.OnPhase(LIFE_PHASE::P_GENERATE);        
        params.isOnlyGPT = true;
        if (0) {    // 20260428 hack
            params.OnPhase(LIFE_PHASE::P_EVAL_);  // only for debug
            params.common.n_ctx = 512;  
            params.isOnlyGPT    = false;
        }

        params.chat_sampler.mode = params.model.enable_thinking ? CHAT_MODE::CHATML_THINK : CHAT_MODE::CHATML_ASSIST;
        // params.chat_sampler.isSampleCPU = true;
        params.model.preLogits_dB  = 1;
        params.model.sparse.method = -1;
        // params.chat_sampler.seq_len     = 32;
        //
        // params.quant.T_errQ             = 0.3;
        // params.quant.isNormalFloat = true;
        // params.quant.default_bits       = 2;
        params.dumpSwitch.tensor_load  = 0;
        params.dumpSwitch.nn_structure = 0;
        DEBUG.verCuda = 1, DEBUG.T_cpu = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
        DEBUG.verInferQKV = 0, DEBUG.verInferFFN = 0;
        // DEBUG.dump_TensorDetail = 1;

        // config.quant.filter_KVcache = {"0.self_attn"};    //   "layers.27.mlp" model.blk.0.attn
        params.Dump(0x100);
    }
    virtual ~BubbleApp() {}

    int Swim() override {
        vector<hWIKI> wikis;  // reserved for hybrid llm training
        fish = Fish::MakeInstance("Bubble_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x110);
        if (fish == nullptr) {
            _ERROR("[APP] %s is nullptr!!!", name.c_str());
            return KOIFISH_NULL_FISH;
        }
        fish->Chat(params.model.enable_thinking, P_GENERATE);
        // while (iRunning() > 0) { // no need this loop
        //     usleep(100000);
        // }
        return KOIFISH_OK;
    }
};

int main(int argc, char* argv[]) {
    BubbleApp app(argc, argv);
    int iRet = app.Run();
    return iRet;
}
