/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Generate some nonsense on Prompt
 *  \author Yingshi Chen
 */
#pragma once

#include <float.h>
#include <inttypes.h>
#include <stdio.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <map>
#include <memory>
#include <regex>
#include <stack>
#include <thread>
#include <typeinfo>
#include <vector>
using namespace std;
#include "../TokenSet/DataLoader.hpp"
#include "../g_float.hpp"
#include "WIKI.hpp"

class Fish;

// multiple arr design for GPU version
template <typename Typ>
struct LogitsInfo {
    void* d_temp  = nullptr;
    size_t szTemp = 0;

    int dim    = -1;
    bool isCPU = true;
    //  cls->preLogits->host_data = _logits
    Typ *logits = nullptr, *logits_sorted = nullptr;
    int *index = nullptr, *index_sorted = nullptr;
    float maxLogit     = 0.f;
    hGensor hClsLogits = nullptr;

    virtual void Swap(int i, int j) { std::swap(logits[i], logits[j]), std::swap(index[i], index[j]); }
    virtual bool Init(int n_vocab, bool isCPU_, hGensor hClsLogits_, int flag = 0x0) {
        isCPU      = isCPU_;
        dim        = n_vocab;
        hClsLogits = hClsLogits_;
        // assert(cls->preLogits->host_data == nullptr);
        index = new int[n_vocab];
        for (int i = 0; i < n_vocab; i++) {
            index[i] = i;
        }
        if (isCPU) {
            logits = new Typ[n_vocab];
            // hClsLogits->host_data = logits;
        } else {
            logits          = TO<Typ>(hClsLogits);
            int* host_index = index;
            cudaCheck(cudaMalloc(&index, n_vocab * sizeof(int)));
            H2D(index, host_index, n_vocab * sizeof(int));
            delete[] host_index;

            cudaCheck(cudaMalloc(&index_sorted, n_vocab * sizeof(int)));
            cudaCheck(cudaMalloc(&logits_sorted, n_vocab * sizeof(Typ)));
        }
        return true;
    }
    virtual void UpdateLogits(int flag = 0x0) {
        void* src = hClsLogits->host_data;
        assert(src != nullptr);
        switch (hClsLogits->type) {
            case typNUMBER::BF16:
                for (int i = 0; i < dim; i++) {
                    float a = T2Float<bf16>((bf16*)src + i);
                    logits[i] = Float2T<Typ>(&a);
                    index[i]  = i;
                }
                break;
            default:
                assert(0);
                break;
        }
    }
    virtual void quick_select(int n, int k) {
        int l = 0, r = n - 1;
        while (l < r) {
            // ProbIndex pivot = arr[k];
            float pivot = (float)logits[k];
            int i = l, j = r;
            do {
                while ((float)logits[i] > pivot) i++;
                while ((float)logits[j] < pivot) j--;
                if (i <= j) {
                    std::swap(logits[i], logits[j]), std::swap(index[i], index[j]);
                    i++;
                    j--;
                }
            } while (i <= j);

            if (j < k)
                l = i;
            if (i > k)
                r = j;
        }
        maxLogit = (float)logits[0];
        for (int i = 1; i < k; i++) {
            if ((float)logits[i] > maxLogit) {
                maxLogit = (float)logits[i];
            }
        }
    }
    virtual void SortPair(int nPick, int flag = 0x0) {
        if (nPick <= 0)
            nPick = dim;
        if (isCPU) {
            // qsort(probindex, n_cands, sizeof(ProbIndex), compare_prob_desc);
            assert(nPick <= dim);
            for (int i = 0; i < nPick; i++) {
                for (int j = 1; j < nPick; j++) {
                    if ((float)logits[i] < (float)logits[j]) {
                        Swap(i, j);
                    }
                }
            }
        } else {
            assert(0);
            /*if (d_temp == nullptr) {
                cub::DeviceRadixSort::SortPairs(d_temp, szTemp, logits, logits_sorted, index, index_sorted, nPick);
                cudaCheck(cudaMalloc(&d_temp, szTemp));  //
            }
            CU_init_i<<<CEIL_DIV(nPick, CU_T4B_SMALL), CU_T4B_SMALL>>>(index, nPick);
            // cub::DeviceRadixSort::SortKeys(d_temp, szTemp, logits, logits, nPick);
            //  In-place operations are not supported. There must be no overlap between any of the provided ranges!!!
            // cudaMemcpy(index_out, index, sizeof(int) * nPick, cudaMemcpyDeviceToDevice);
            // cudaMemcpy(logits_out, logits, sizeof(Typ) * nPick, cudaMemcpyDeviceToDevice);
            cub::DeviceRadixSort::SortPairs(d_temp, szTemp, logits, logits_sorted, index, index_sorted, nPick);
            PrintTensor<Typ>("sort_logits", logits_sorted, true, nPick, 1, 1, 1, 0);
            PrintTensor<int>("sort_index", index_sorted, true, nPick, 1, 1, 1, 0);*/
        }
    }
    virtual ~LogitsInfo() {
        if (isCPU) {
            // hClsLogits would release this! since hClsLogits->host_data = logits; @GeneratOnPrompt::GeneratOnPrompt=>cpuLogits.Init
            // FREE_a(logits);
            FREE_a(index);
        }
    }
};

/*
    Ref     https://github.com/karpathy/llama2.c
*/
class GeneratOnPrompt {
    // GeneratOnPrompt(const GeneratOnPrompt&);
    // GeneratOnPrompt& operator=(const GeneratOnPrompt&);

   protected:
    CLI_params config;
    CHAT_SAMPLER samp_params;

    LogitsInfo<float> cpuLogits;
    LogitsInfo<floatLogits> gpuLogits;
    // ProbIndex* probindex = nullptr;

    //  std::vector<float> x_logits;
    float delta_max = 0, delta_a = 0;
    bool display = true;

    MODEL_ARCH _arch = MODEL_ARCH::_X_;

    int ga_n = -1, ga_w = -1;
    int32_t bos = 1, eos = 2;
    int n_predict = 32, n_batch = 2048, n_keep;
    bool is_antiprompt = false;
    // bool input_echo           = true;
    bool isCTXSampling = true;
    int n_ctx = -1, n_ctx_train = -1;
    int nCanTopK = -1;

    // std::string path_session = params.path_prompt_cache;
    std::vector<TOKEN_ID> session_tokens;
    std::vector<TOKEN_ID> embd_inp;
    std::string GetPrompt(int flag = 0x0);
    hSampLoader dialogs;
    std::vector<int> input_tokens, output_tokens;
    std::ostringstream output_ss;
    bool is_interacting = false;
    hWIKI wiki0         = nullptr;
    arrHWIKI wikis;
    const Fish* fish_0 = nullptr;
    Fish* fish_1       = nullptr;
    // shared_ptr<Fish> fish_1 = nullptr;        //for generate, only 1 input

    virtual std::string T2STR(TOKEN_ID tok, int flag = 0x0);

    virtual void Clear();
    uint64_t rng_state;
    virtual void OnAntiPrompt(int flag);
    virtual bool Inference(hSAMP samp, int& nPast, int flag = 0x0);
    // virtual void OnInteractive(int& n_past,int& n_consumed,int& n_remain,int flag);
   public:
    GeneratOnPrompt() {}
    GeneratOnPrompt(struct gpt_params& par_, int flag);
    GeneratOnPrompt(CLI_params& cp_, arrHWIKI& wiki_, const Fish* hG_, int flag);

    CHAT_MODE ChatMode(int flag = 0x0) const;

    static shared_ptr<GeneratOnPrompt> MakeInstance(struct CLI_params& params, arrHWIKI& wiki, const Fish*, int flag);

    virtual ~GeneratOnPrompt() { Clear(); }
    virtual bool Init(const std::string& prompt_, int flag = 0x0);

    std::vector<TOKEN_ID> guidance_inp;
    std::vector<TOKEN_ID> inp_pfx, inp_sfx, cml_pfx, cml_sfx;
    int guidance_offset     = 0;
    int original_prompt_len = 0;
    virtual void InitInput(int flag = 0x0);

    virtual int Tokenize(int flag);

    std::vector<TOKEN_ID> tokens;
    // std::vector<TOKEN_ID> embd_guidance;
    // tokenized antiprompts
    std::vector<std::vector<TOKEN_ID>> antiprompt_ids;

    virtual void TokenEmbed(int flag = 0x0) {
        assert(0);  // Deprecated
        // antiprompt_ids.reserve(params.antiprompt.size());
        // for (const std::string & antiprompt : params.antiprompt) {
        //     antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
        // }
    }

    // virtual int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0);

    virtual int Generate(int nJob, int flag = 0x0);
    virtual int Generate_v0(int nJob, int flag = 0x0);
    virtual TOKEN_ID Sample_cpu(int idx = -1, bool isSorted = false);
    virtual TOKEN_ID Sample(int idx = -1, bool is_resampling = false);
    virtual bool VerifyLogits(int flag = 0x0);
    virtual void DisplayEmbd(bool input_echo, int n_consumed, int flag = 0x0);
};
typedef shared_ptr<GeneratOnPrompt> hGENERATOR;
using hChater = hGENERATOR;

class GOPT_infinite : public GeneratOnPrompt {
   protected:
    // int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0) override;

   public:
    GOPT_infinite(struct gpt_params& par_, int flag) : GeneratOnPrompt(par_, flag) { ; }
};

class GOPT_Metropolis : public GeneratOnPrompt {
   protected:
    TOKEN_ID Sample(int idx = -1, bool is_resampling = false) override;

   public:
    GOPT_Metropolis(struct gpt_params& par_, int flag) : GeneratOnPrompt(par_, flag) { isCTXSampling = false; }
    GOPT_Metropolis(CLI_params& cp_, arrHWIKI& wikis_, const Fish* hG_, int flag) : GeneratOnPrompt(cp_, wikis_, hG_, flag) { isCTXSampling = false; }

    virtual ~GOPT_Metropolis() { Clear(); }

    // int Generate(int nJob,int flag=0x0) override;
};
