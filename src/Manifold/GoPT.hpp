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
#include "../Dynamics/Langevin.hpp"
#include "../TokenSet/DataLoader.hpp"
#include "../g_float.hpp"
#include "Scheduler.hpp"
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
    int dim = -1;

    float maxLogit  = 0.f;
    float lastLogit = 0.f;  // k-th largest value (smallest in heap)
    virtual bool isLarge(int i, int k, int flag = 0x0) { return false; }
    std::priority_queue<int> heap;
    std::vector<int> picks;

    virtual int Select(int nPick, SORT_BY sort = SORT_BY::NO_SORT, int flag = 0x0);
    virtual float ValueAt(int k) {
        assert(0);
        return 0.0;
    }
};

// 非我无所取(qu)
struct QU {
    int posOfTarget = -1;
    TOKEN_ID token  = 0;
    float confi     = 0.;  // confidense
};

struct LogitsInfo : TOPK_heap {
    bool isHeaviside = false;
    int ver          = 0;
    int posInBatch = 0, posOfTarget = 0;
    bool isCPU = true;

    float* logits       = nullptr;
    floatX* src         = nullptr;  //  cls->preLogits->host_data
    hGTensor hClsLogits = nullptr;
    float T_coin        = 0.0f;

    QU qu;

    LogitsInfo(int id, const Fish* hG_, hGTensor hClsLogits_, float T_x_, int flag = 0x0);
    // virtual void Swap(int i, int j) { std::swap(logits[i], logits[j]), std::swap(index[i], index[j]); }
    // virtual bool Init(int n_vocab, hGTensor hClsLogits_, int flag = 0x0);
    // BF16->float
    virtual void UpdateProbability(const CHAT_SAMPLER& samp_params, int flag = 0x0);

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

    virtual void Dump(int type, int flag = 0x0);

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
    hBATCH hBatch     = nullptr;  // for chat(1 batch with prompt & mask/pad tokens)
    hDataToken tsChat = nullptr;

    std::vector<std::string> some_prompts, some_answers;

    hGTensor hClsLogits = nullptr;
    hHuaPLAN planner    = nullptr;  // planner to decide which masked tokens to reveal at each sample step
    // LogitsInfo cpuLogits;
    std::vector<hLogitsInfo> originLogits, maskLogits;
    std::vector<hLogitsInfo> candLogit;  // candidate of SampFromLogits
    // LogitsInfo_GPU<floatLogits> gpuLogits;   // [todo]

    float delta_max = 0, delta_a = 0;
    // 0.1 – 0.5Mild reduction in repetition; >1.5 Risk of language mixing and degraded quality
    float presence_penalty = 0.0;
    bool display           = true;

    MODEL_ARCH _arch = MODEL_ARCH::_X_;

    int ga_n = -1, ga_w = -1;
    int32_t bos = 1, eos = 2;
    // int n_predict = 32, n_batch = 2048, n_keep;
    bool is_antiprompt = false;

    // int n_ctx = -1, n_ctx_train = -1;
    int nCanTopK = -1, nGenerate = 0;

    std::string fResult, sResult, cur_answer;  // path of file to save result
    // std::string path_session = params.path_prompt_cache;
    std::vector<TOKEN_ID> session_tokens;
    std::vector<TOKEN_ID> embd_inp;
    std::vector<QU> arrQu;
    std::string GetPrompt(int flag = 0x0);
    hSampNanny dialogs;
    std::vector<int> input_tokens, output_tokens;
    std::ostringstream output_ss;
    bool is_interacting = false;
    hWIKI wiki0         = nullptr;
    arrHWIKI wikis;
    Fish* fish_0 = nullptr;
    Fish* fish_1 = nullptr;
    // shared_ptr<Fish> fish_1 = nullptr;        //for generate, only 1 input

    virtual std::string T2STR(TOKEN_ID tok, int flag = 0x0);

    virtual void Clear();
    uint64_t rng_state;
    GRander rand_coin;

    virtual void OnAntiPrompt(int flag);
    virtual bool Inference(hSAMP samp, int& nPast, int flag = 0x0);
    virtual void TopK(int idx = -1, int flag = 0x0);
    virtual void SampFromLogits(int step, int flag = 0x0);

   public:
    GeneratOnPrompt() {}
    // GeneratOnPrompt(struct gpt_params& par_, int flag);
    GeneratOnPrompt(CLI_params& cp_, arrHWIKI& wiki_, Fish* hG_, int flag);

    static shared_ptr<GeneratOnPrompt> MakeInstance(struct CLI_params& params, arrHWIKI& wiki, Fish*, int flag);

    virtual ~GeneratOnPrompt() { Clear(); }

    virtual bool InitCoral(const std::string& prompt_, Fish* hG_, int flag = 0x0);
    // Deprecated
    virtual bool Init_0(const std::string& prompt_, int flag = 0x0);

    std::vector<TOKEN_ID> guidance_inp;
    std::vector<TOKEN_ID> inp_pfx, inp_sfx, cml_pfx, cml_sfx;
    int guidance_offset     = 0;
    int original_prompt_len = 0;

    virtual void InitInput(int flag = 0x0);
    virtual void Prepare4N(int iter, hBATCH hBatch, int flag = 0x0);  // Prepare for CHAT_N(diffusion model)

    virtual int Tokenize(int flag);

    std::vector<TOKEN_ID> tokens;
    std::vector<std::vector<TOKEN_ID>> antiprompt_ids;

    virtual int Generate_v0(int nJob, int flag = 0x0);

    // virtual TOKEN_ID Sample_cpu(int idx = -1, bool isSorted = false);
    virtual TOKEN_ID Sample(hBATCH hBatch, bool is_resampling = false);
    virtual int SampleOnBatch(hBATCH hBatch, float* hostLoss, int B, int T, SampNanny* hLoader, int flag = 0x0);
    virtual bool OnLogits(int flag = 0x0);
    virtual void DisplayEmbd(bool input_echo, int n_consumed, int flag = 0x0);

    virtual void AfterSample(int iter, double elapsed_s, int flag = 0x0);

    friend class Fish;
};
typedef shared_ptr<GeneratOnPrompt> hGENERATOR;
using hChater = hGENERATOR;

// for mask(denoising) models(diffusion LM, ...)
class GOPT_Diffusion : public GeneratOnPrompt {
   protected:
    int nCurMask = 0, nToMask = 0;
    TOKEN_ID mask_id;
    TOKEN_ID Sample(hBATCH hBatch, bool is_resampling = false) override;

   public:
    GOPT_Diffusion(CLI_params& cp_, arrHWIKI& wikis_, Fish* hG_, int flag);
    bool OnLogits(int flag = 0x0) override;
    virtual ~GOPT_Diffusion() {}
};

class GOPT_Metropolis : public GeneratOnPrompt {
   protected:
    TOKEN_ID Sample(hBATCH hBatch, bool is_resampling = false) override;

   public:
    // GOPT_Metropolis(struct gpt_params& par_, int flag) : GeneratOnPrompt(par_, flag) {}
    GOPT_Metropolis(CLI_params& cp_, arrHWIKI& wikis_, Fish* hG_, int flag) : GeneratOnPrompt(cp_, wikis_, hG_, flag) {}

    virtual ~GOPT_Metropolis() { Clear(); }

    // int Generate(int nJob,int flag=0x0) override;
};
