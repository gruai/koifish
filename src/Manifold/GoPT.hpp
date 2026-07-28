/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
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

#include <limits>
#include <queue>
#include <vector>

/**
 * 1. supoort multi-thread
 */
struct TOPK_heap {
    int tid = -1;  // thread id, each thread for one tensor
    int dim = -1, nPick = -1;

    float maxLogit  = 0.f;
    float lastLogit = 0.f;  // k-th largest value (smallest in heap)
    virtual bool isLarge(int i, int k, int flag = 0x0) { return false; }
    std::priority_queue<int> heap;
    std::vector<int> picks;

    virtual int Select(int nPick, bool isOrder = false, int flag = 0x0);
    virtual float ValueAt(int k) {
        assert(0);
        return 0.0;
    }
};

struct LogitsInfo : TOPK_heap {
    int ver        = 0;
    int posInBatch = 0;
    bool isCPU     = true;

    float* logits       = nullptr;
    floatX* src         = nullptr;  //  cls->preLogits->host_data
    hGTensor hClsLogits = nullptr;
    uint64_t rng_state;
    TOKEN_ID qu;  // 非我无所取(qu)
    float confidence;

    LogitsInfo(int id, const Fish* hG_, hGTensor hClsLogits_, int flag = 0x0);
    // virtual void Swap(int i, int j) { std::swap(logits[i], logits[j]), std::swap(index[i], index[j]); }
    // virtual bool Init(int n_vocab, hGTensor hClsLogits_, int flag = 0x0);
    // BF16->float
    virtual void UpdateLogits(const CHAT_SAMPLER& samp_params, int flag = 0x0);

    bool isLarge(int i, int k, int flag = 0x0) override {
        assert(i >= 0 && i < dim);
        assert(k >= 0 && k < dim);
        float a = T2Float(src + i);
        float b = T2Float(src + k);
        return a > b;
    }
    float ValueAt(int k) override {
        assert(k >= 0 && k < dim);
        float a = T2Float(src + k);
        return a;
    }

    virtual void TopK(int k, int flag = 0x0);
    virtual float TopP(float top_p, int k, int flag = 0x0);
    virtual int Qu_FlipCoin(int flag = 0x0);

    virtual void SortPair(int nPick, int flag = 0x0) { assert(0); }
    virtual ~LogitsInfo() { FREE_a(logits); }
};
typedef std::shared_ptr<LogitsInfo> hLogitsInfo;

template <typename Typ>
struct LogitsInfo_GPU : public LogitsInfo {
    void* d_temp  = nullptr;
    size_t szTemp = 0;

    virtual bool Init(int n_vocab, hGTensor hClsLogits_, int flag = 0x0) {
        this->isCPU = false;
        assert(0);
        /*dim        = n_vocab;
        hClsLogits = hClsLogits_;
        // assert(cls->preLogits->host_data == nullptr);
        index = new int[n_vocab];
        for (int i = 0; i < n_vocab; i++) {
            index[i] = i;
        }

        logits          = TO<Typ>(hClsLogits);
        int* host_index = index;
        cudaCheck(cudaMalloc(&index, n_vocab * sizeof(int)));
        H2D(index, host_index, n_vocab * sizeof(int));
        delete[] host_index;

        cudaCheck(cudaMalloc(&index_sorted, n_vocab * sizeof(int)));
        cudaCheck(cudaMalloc(&logits_sorted, n_vocab * sizeof(Typ)));*/

        return false;
    }

    void SortPair(int nPick, int flag = 0x0) override {
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
};

/*

*/
class GeneratOnPrompt {
    // GeneratOnPrompt(const GeneratOnPrompt&);
    // GeneratOnPrompt& operator=(const GeneratOnPrompt&);

   protected:
    CLI_params config;
    CHAT_SAMPLER samp_params;
    hBATCH hBatch       = nullptr;
    hGTensor hClsLogits = nullptr;
    // LogitsInfo cpuLogits;
    std::vector<hLogitsInfo> arrLogit;
    // LogitsInfo_GPU<floatLogits> gpuLogits;   // [todo]

    float delta_max = 0, delta_a = 0;
    // 0.1 – 0.5Mild reduction in repetition; >1.5 Risk of language mixing and degraded quality
    float presence_penalty = 0.0;
    bool display           = true;

    MODEL_ARCH _arch = MODEL_ARCH::_X_;

    int ga_n = -1, ga_w = -1;
    int32_t bos = 1, eos = 2;
    int n_predict = 32, n_batch = 2048, n_keep;
    bool is_antiprompt = false;
    int nCTX_(int type=0x0);    // default return ctx_recommend
    // int n_ctx = -1, n_ctx_train = -1;
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
    virtual void TopK(int idx = -1, int flag = 0x0);

   public:
    GeneratOnPrompt() {}
    GeneratOnPrompt(struct gpt_params& par_, int flag);
    GeneratOnPrompt(CLI_params& cp_, arrHWIKI& wiki_, const Fish* hG_, int flag);

    static shared_ptr<GeneratOnPrompt> MakeInstance(struct CLI_params& params, arrHWIKI& wiki, const Fish*, int flag);

    virtual ~GeneratOnPrompt() { Clear(); }
    virtual bool Init(const std::string& prompt_, int flag = 0x0);

    std::vector<TOKEN_ID> guidance_inp;
    std::vector<TOKEN_ID> inp_pfx, inp_sfx, cml_pfx, cml_sfx;
    int guidance_offset     = 0;
    int original_prompt_len = 0;

    virtual void InitInput(int flag = 0x0);
    virtual void Prepare4N(int flag = 0x0);  // Prepare for CHAT_N(diffusion model)

    virtual int Tokenize(int flag);

    std::vector<TOKEN_ID> tokens;
    std::vector<std::vector<TOKEN_ID>> antiprompt_ids;

    virtual int Generate(int nJob, int flag = 0x0);
    virtual int Generate_v0(int nJob, int flag = 0x0);
    virtual TOKEN_ID Sample_cpu(int idx = -1, bool isSorted = false);
    virtual TOKEN_ID Sample(hBATCH hBatch, bool is_resampling = false);
    virtual int SampleOnBatch(hBATCH hBatch, float* hostLoss, int B, int T, SampLoader* hLoader, int flag = 0x0);
    virtual bool OnLogits(int flag = 0x0);
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
    TOKEN_ID Sample(hBATCH hBatch, bool is_resampling = false) override;

   public:
    GOPT_Metropolis(struct gpt_params& par_, int flag) : GeneratOnPrompt(par_, flag) {}
    GOPT_Metropolis(CLI_params& cp_, arrHWIKI& wikis_, const Fish* hG_, int flag) : GeneratOnPrompt(cp_, wikis_, hG_, flag) {}

    virtual ~GOPT_Metropolis() { Clear(); }

    // int Generate(int nJob,int flag=0x0) override;
};
