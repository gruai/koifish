
/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Why batch is so important?
 *
 *  \brief A batch of random samples
 *  \author Yingshi Chen
 */

#pragma once

#include "../CLI_params.hpp"
#include "../Dynamics/Transition.hpp"
#include "TokenSet.hpp"

enum MASK_FLAG {
    F_PAD         = 0x100,
    F_CAUSAL      = 0x200,
    F_SCORE_NOISE = 0x400,

    F_IGNORE_LOSS = 0x10000,  //  -100 of torch's cross_entropy
};

struct TOKEN_Planner;

/**
 * Each tokenset has a unique samp_loader; Each samp_loader has a unique batch
 * Each batch has many samples, each sample is a just transition from x(t) to x(t+1) (just like our life flow from begin to end)
 *  */
struct BATCH_INPUT : public std::enable_shared_from_this<BATCH_INPUT> {
    LIFE_PHASE phasb = P_X;  // batch only has one phase(never change in lifecycle!)

    std::shared_ptr<TOKEN_Planner> hHuaPLAN = nullptr;
    std::shared_ptr<TOKEN_Planner> hPLAN_1  = nullptr;  // fro one samp

    enum NOISE_TYPE {
        NO_NOISE,
        MASK_NOISE,
    };
    NOISE_TYPE tpNoise = NO_NOISE;

    shared_ptr<GTensor> hostLabel = nullptr;
    shared_ptr<GTensor> hostToken = nullptr, hostMask = nullptr, devMask = nullptr, hostLen = nullptr;

    int *host_toks = nullptr, *mask32 = nullptr;  //  mask32 = TO<int>(hostMask);
    int* tmp32 = nullptr;
    // float* fNoise    = nullptr;  // float noise for each tokens
    HUA_Token transiX;
    HUAERs huaers;  // each sample is a just transition from x(t) to x(t+1) (just like our life flow from begin to end)

    int nValidTokens = 0, nMaskToken = -1, nNoiseToken = -1, nPadToken = -1;
    int nMostSample = 0, ldT = 0;  //  B,T
    int dB4Logit = -1, iter = -1;
    size_t nPrefill = 0, nFill = 0;
    std::vector<int> arrTic0, arrTic1;
    std::vector<TOKENS_SECTION> section_metas;  // tokens of each sample contain sections, each section may has different role
    Fish* hFish      = nullptr;
    hTokenizer hDict = nullptr;
    //  always point to last token when P_CHAT_1
    int tok_pos = -1;
    //  just return host_toks[tok_pos], only for P_CHAT_1
    int CurToken() {
        assert(tok_pos >= 0 && host_toks != nullptr);
        // assert(host_toks[pos] < embed->nVocab);
        return host_toks[tok_pos];
    }
    bool onlyLogits = false;  // If true, logits is enough, no need to set label & get loss!

    BATCH_INPUT(Fish* hFish, SHAPE sp, LIFE_PHASE phasb, int flag = 0x0);
    virtual ~BATCH_INPUT() {
        FREE_a(tmp32);
        // FREE_a(fNoise);
    }
    virtual void Init(int flag = 0x0);

    virtual int nTokens(int flag = 0x0);  // return hostToken->size();
    // virtual void Update(hGTensor batch,int flag=0x0);
    virtual int FillPrompt(Fish* hFish, const std::vector<std::string>& Prompt, const std::vector<std::string>& answers, int nRound, int flag = 0x0);
    virtual void FillTokens(int k, const std::vector<TOKEN_ID>& tokens, int i_off, int flag);
    virtual void FillOneSamp(SampNanny* hNanny, int idSamp, hSAMP samp, const CLI_params& params, float T_x, int flag = 0x0);
    // 1. Deprecated, replace by FillTokens 2. No BOS at sequence start!
    virtual void Reset(const std::vector<TOKEN_ID>& tokens, int flag = 0x0);
    virtual void SetLen(int i0, int len) {
        if (hostLen != nullptr)
            hostLen->Set(i0, 0, 0, 0, len);
    }
    virtual void SetToken(int i0, int i1, int i2, int i3, int tok, HUA_STATE ti = HUA_STATE::X);
    virtual void SetMask(int i0, int i1, int i2, int i3, int tok) { hostMask->Set(i0, i1, i2, i3, tok); }
    // Label may be noised(>0) in some models(mask-noise diffusion model)
    virtual bool SetLabel(int label0, int i_target, int k, HUA_STATE ti = HUA_STATE::X, int flag = 0x0);
    virtual TOKEN_ID GetLabel(int i_target, int k, int flag = 0x0);

    virtual size_t nFillTokens() {
        if (nFill > 0)
            return nFill;
        else {
            assert(nFill <= hostToken->size());
            return hostToken->size();
        }
    }
    virtual bool BeforeCollate(int iter, int flag = 0x0);
    virtual bool UpdatePadMask(const std::vector<hSAMP>& samps, int iter, TOKEN_ID* tokens, int* labels, int flag = 0x0);
    virtual bool PickTransitions(const std::vector<hSAMP>& samps, int iter, int flag = 0x0);

    virtual void DumpX(TOKEN_ID* tokens, float* hostLoss, int flag = 0x0);
};
typedef shared_ptr<BATCH_INPUT> hBATCH;

// Batch for denoising diffusion model
struct BATCH_Denoise : public BATCH_INPUT {
    BATCH_Denoise(Fish* hFish, SHAPE sp, LIFE_PHASE phasb, int flag = 0x0);
    void FillTokens(int k, const std::vector<TOKEN_ID>& tokens, int i_off, int flag) override;
};
