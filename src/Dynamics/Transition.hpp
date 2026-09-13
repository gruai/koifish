
/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  HUA(化)  庄周化蝶, 蝶胥也化而为虫, 虫千日为鸟, 鸟扶摇而上九万⾥化鹏, 鹏六月以去北冥，物化为鲲
 *
 *  Langevin dynamics = Score(log‑probability)‑driven stochastic gradient flow + Gaussian noise, annealed over time.
 *  Langevin dynamics is the Markov process whose infinitesimal generator is the Fokker–Planck operator corresponding to diffusion in a potential field.
 *  Chat is just sampling process on Langevin dynamics
 *
 *  \brief HUA & Langevin dynamics
 *  \author Yingshi Chen
 */

#pragma once

#include "../CLI_params.hpp"
#include "../Utils/GST_rander.hpp"
#include "../g_def_x.hpp"

class Fish;
// each state in Langevin samping process
typedef std::vector<int> lState;

//  A sampling trajectory {x(t)}, where each transition x(t)->x(t+1) ​is a single Langevin/diffusion step.
struct TRAJECTORY {
    std::vector<lState> lPath;  // A sampling trajectory
};

enum HUA_STATE { X = 0x0, TOKEN, MASK, DENOISE, PAD };

/**
 *  化而(Transition)
 *  x(t)->x(t+1) each state in Langevin samping process
 */
struct HUAER {
    std::vector<HUA_STATE> flow;
    std::vector<float> noise;
    int nPrefill = 0, nDenoise = 0, nMask = 0;

    HUAER() {}
    HUAER(std::vector<HUA_STATE>& _flow, int flag = 0x0) : flow(_flow) {}
};

//  化而(Transitions)
typedef std::vector<std::shared_ptr<HUAER>> HUAERs;

struct HUA_Token : public HUAER {
    HUA_Token() {}
    HUA_Token(std::vector<HUA_STATE>& _TOKEN_TRANSs, int nPre, int flag = 0x0);

    virtual void Init(int seq_len, int flag = 0x0);
    inline HUA_STATE operator[](const int pos) const {
        assert(pos >= 0 && pos < flow.size());
        return flow[pos];
    }
};

struct BATCH_INPUT;
struct SAMP;
// Planner of sampling of LM(mask-denoising, ...), transition in chat(token-sequence) space
struct TOKEN_Planner {
    static std::shared_ptr<TOKEN_Planner> MakeInstance(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);

    enum BIT_FLAG {
        F_ABSOLUTE_ID   = 0x100,  //  @Init4Prefill_ ABSOLUTE: id=id0[k], Relative: id=k        
    };

    float* fNoise    = nullptr;
    std::string name = "planner";
    // typedef std::vector<int> Group;
    std::vector<lState> arrGroup;
    HUAERs huaers;
    std::vector<HUA_STATE> samp_flow;  // flow for one samp
    std::vector<float> arrRatio;       // ratios for each step
    hRANDER hMaskRander = nullptr;

    int nMostStep = 0, context_len = 0, samp_len = 0;
    // int seq_len = -1;  // would change, context_len is fix as config
    // SKDU_POLICY policy;
    CHAT_SAMPLER _params;
    TOKEN_Planner(const std::string& nam_, CHAT_SAMPLER& _params, int nMostStep, int seq_len, int flag = 0x0);
    virtual ~TOKEN_Planner();
    virtual void Init(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0);
    hRANDER hPickRander = nullptr;
    float t_base = 0.0, t_previous = 0.0, t_final = 0.0;
    // how much noise was removed between cur & last step.
    virtual float RelativeRate(int64_t step, int flag = 0x0);
    // virtual lState PickGroup(int64_t step, int nSamp, int type, int flag = 0x0);

    virtual bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0);
    virtual bool Transition4Samp(SAMP* hSamp, int flag = 0x0) { NOT_IMPLEMENTED; }
    // For fixed prefill, get all trainsitions
    virtual bool Init4Prefill(int minPrefill, int type, int flag = 0x0) { NOT_IMPLEMENTED; }
    virtual bool InitAllTrainsitions(int minPrefill, int mostPrefill, int flag = 0x0);
    virtual void Dump(int flag = 0x0);
};
typedef std::shared_ptr<TOKEN_Planner> hHuaPLAN;
/*  Deprecated
struct TPLAN_random : public TOKEN_Planner {
    TPLAN_random(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);
    bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};*/

struct TPLAN_SNR : public TOKEN_Planner {
    TPLAN_SNR(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);
    bool Init4Prefill(int minPrefill, int type, int flag = 0x0) override;
    bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};

struct TPLAN_PUMA : public TOKEN_Planner {
    TPLAN_PUMA(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);
    bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};

struct TPLAN_SNR_sequence : public TPLAN_SNR {
    TPLAN_SNR_sequence(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);
    // bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};

struct TPLAN_Dilate : public TOKEN_Planner {
    TPLAN_Dilate(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT>, int flag = 0x0);
    bool Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
    bool Transition4Samp(SAMP* hSamp, int flag = 0x0) override;

    // bool InitAllTrainsitions(int minPrefill, int mostPrefill, int flag = 0x0) override;
    bool Init4Prefill(int minPrefill, int type, int flag = 0x0) override;
    void Dump(int flag = 0x0) override;
};