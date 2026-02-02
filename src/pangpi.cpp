/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  鳑鲏 不仅仅是一种色彩艳丽的小观赏鱼，是​​生态系统的指示剂​​：它的存在意味着水域环境健康。
 *
 *  Some user cases
 *      1.  pangpi ./hy-tmp/checkpoint/gpt_1.fish --hellaswag ./cases/datasets/hellaswag_val.bin
 *
 *  \brief 鳑鲏 - Some evaluation cases(Hellaswag,...,)
 *  \author Yingshi Chen
 */
#include <filesystem>
#include <iostream>
#include <string>

#include "./CLI_params.hpp"
#include "./Manifold/Fish.hpp"
#include "./Utils/GST_os.hpp"
#include "GoPT.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"
#include "./Utils/GST_Application.hpp"

class PangpiApp : public GST_Application {
   protected:
    hFISH fish = nullptr;

   public:
    PangpiApp(int argc, char* argv[]) : GST_Application(argc, argv) { name = "Koifish"; }
    virtual ~PangpiApp() {
        fish  = nullptr;
        gBUFF = nullptr;
    }

    int Swim() override {
        if (params.n_swarm > 1) {
            fish = Fish::MakeSwarm("Fish_", params, 0x0);
        } else {
            vector<hWIKI> wikis = WIKI::MakeInstance("", params, 0x0);
            if (wikis.size() == 0) {
                // _INFO("====== NO WIKI !!! ======\n");       return;
            } else if (params.wiki_actor == "copy") {
                wikis[0]->CopyParams(params);
            }
            fish = Fish::MakeInstance("Fish_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x0);
        }

        if (fish && fish->isTrain())
            fish->Train();
        return KOIFISH_OK;
    }
};

int main(int argc, char* argv[]) {
    try {
        assert(argc >= 3);
        std::string arg_prefix = "--", exec_name = EXE_name(), jsPath = "", eval_metric = "";
        CLI_params config;
        config.phase = LIFE_PHASE::P_EVAL_;
        if (!config.parse(argc, argv)) {
            return -1;
        }
        config.OnArch();
        JSON old_datasets = config.jConfig["datasets"], new_data = config.jConfig["datasets_new"];
        if (!new_data.empty()) {
            config.jConfig["datasets"] = new_data;
        }
        if (config.jConfig["datasets"].empty())
            return KOIFISH_DATASET_EMPTY;

        config.wiki_actor = "copy";
        //  if isOnlyGPT, batch_size is always 1 !
        config.isOnlyGPT = config.eval_metric != "hellaswag";  // true;
        // config.common.n_batch      = 1;
        config.model.preLogits_dB     = config.eval_metric == "hellaswag" ? config.model.preLogits_dB : 1;
        config.model.sparse.method    = -1;
        config.dumpSwitch.tensor_load = 1;
        SUM::nMinTensorAlloc          = 1;  // g_dump_level = -1;

        DEBUG.verCuda = 1, DEBUG.T_cpu = 0, DEBUG.cmd_p1 = 0, DEBUG.graph_dump = 0, DEBUG.Time_most = 60;
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
        _INFO("\n====== %s: %s FLOAT=%s @%s\n[BUBBLE]\t", __func__, DEBUG.T_cpu == 0 ? "CUDA" : "CPU", cNameOf(config.model.tpActivation), cDATE);

        OutCLS* hCLS  = fish->GetNeuron<OutCLS>("OutCLS", 0);
        hCLS->hLoader = hLoader;
        double sum = 0, ss = 0, tps = 0, t0 = GST_ms(), tAll = 0, eval, delta = 0;
        vector<TOKEN_ID>& tokens = hLoader->GetTokens();
        fish->SetPhase(LIFE_PHASE::P_EVAL_);
        // SUM::Reset("memory");
        //  fish = nullptr;     return KOIFISH_EXIT_DEBUG;
        if (config.eval_metric == "hellaswag") {  //[eval]  Loss@"HellaSwag"=0.300 nBranch=2
            eval = hLoader->Evaluate(SAMPLEofSHARD, 0x0);
            nz   = hLoader->nEvalTokens;
        } else {
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
        }
        tAll = (GST_ms() - t0) / 1000.0;
        tps  = nz / tAll / 1.0e3;
        // Fish::stat.Dump(0x0);
        _INFO("[BUBBLE] Upload=%.3gG Throughput=%.3g GByte/s \n", SUM::szUpload / 1.0e9, SUM::szUpload / 1.0e6 / SUM::tRemater);
        if (fish->config.model.isSparse()) {
            // fish->Sparsing();
        }

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

