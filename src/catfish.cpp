/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  Some user cases
 *     catfish ./scripts/qwen3.json
 *
 *  \brief catfish - answer some questions
 *  \author Yingshi Chen
 */
#include <filesystem>
#include <iostream>
#include <string>

#include "./CLI_params.hpp"
#include "./Manifold/Fish.hpp"
#include "./Utils/GST_os.hpp"
#include "./g_stddef.hpp"
#include "GoPT.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

int main(int argc, char *argv[]) {
    try {
        assert(argc >= 2);
        std::string arg_prefix = "--", exec_name = EXE_name(), jsPath = "", eval_metric = "";
        CLI_params config;
        config.phase = LIFE_PHASE::P_GENERATE;
        if (!config.parse(argc, argv)) {
            return -1;
        }
        config.OnArch();
        if (config.jConfig["datasets"].empty())
            return KOIFISH_DATASET_EMPTY;

        config.isOnlyGPT           = true;
        config.model.preLogits_dB  = 1;
        config.model.sparse.method = -1;
        SUM::nMinTensorAlloc       = 1;  // g_dump_level = -1;

        DEBUG.T_cuda_ver = 1, DEBUG.T_cpu = 0, DEBUG.cmd_p1 = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
        config.Dump(0x100);

        // fish->isLocalInfer = flag == 0x110;
        hFISH fish      = Fish::MakeInstance("PPL_", config, {}, Fish::ROLE_TYPE::COMMON, 0x110);
        hOptimizer hOPT = fish->GetOptimizer();
        if (hOPT->val_loaders.empty())
            return KOIFISH_DATALOADER_EMPTY;

        std::string prompt = LoadSomeText("/home/cys/rnd/lic/models/TinyStories-valid.txt", 64 * 1024);  // shakespeare.txt
        int nVocab = fish->nClass(), _nctx = fish->config.n_ctx(), i, j, nz = 0;
        hSampLoader hLoader = hOPT->val_loaders[0];
        if (hLoader->num_batches <= 0) {
            hLoader->InitOneSamp(prompt, nullptr, fish.get(), 0x110);
        }
        SUM::tX = 0;
        _INFO("\n====== %s: %s FLOAT=%s @%s\n[CATFISH]\t", __func__, DEBUG.T_cpu == 0 ? "CUDA" : "CPU", cNameOf(config.model.tpActivation), cDATE);

        OutCLS *hCLS  = fish->GetNeuron<OutCLS>("OutCLS", 0);
        hCLS->hLoader = hLoader;
        double sum = 0, ss = 0, tps = 0, t0 = GST_ms(), tAll = 0, eval, delta = 0;
        vector<TOKEN_ID> &tokens = hLoader->GetTokens();
        hOPT->SetPhase(LIFE_PHASE::P_GENERATE);
        SUM::Reset("memory");
        //  fish = nullptr;     return KOIFISH_EXIT_DEBUG;

        int nTokens = hLoader->nMostToken;
        nTokens     = DEBUG.T_cpu == 0 ? 1024 : 200;  // assert(nTokens <= _nctx);
        double ppl = 0, pplerr = 0;
        for (i = 0; i + 1 < nTokens; i++) {
            /* T_generate_(fish, i, config.model.tpActivation, -666);
            double logprob = log(P_softmax(tokens[i + 1], logits, nVocab));

            sum += logprob;
            ss += logprob * logprob;
            nz++;
            ppl    = exp(-sum / nz);
            pplerr = ppl * sqrt((ss - sum * sum / nz) / nz / nz);*/
        }
        eval = ppl;

        tAll = (GST_ms() - t0) / 1000.0;
        tps  = nz / tAll / 1.0e3;
        Fish::stat.Dump(0x0);
        _INFO("[CATFISH] Upload=%.3gG Throughput=%.3g GByte/s \n", SUM::szUpload / 1.0e9, SUM::szUpload / 1.0e6 / SUM::tRemater);
        if (fish->config.model.isSparse()) {
            // fish->Sparsing();
        }

        return 0x0;
    } catch (const std::exception &e) {
        _INFO("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char *info) {
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