/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Generate samples for train/Eval
 *
 *  \brief samples from multiple dataset/tokenset/...
 *  \author Yingshi Chen
 */

#ifndef DATALOADER_H
#define DATALOADER_H

struct DictVAE;
#include "../CLI_params.hpp"
#include "../Tensor/GTensor.hpp"
#include "../Utils/GST_obj.hpp"
#include "../Utils/GST_rander.hpp"
#include "TokenSet.hpp"
using namespace std;
// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// The head of preTokenFile by koifish is always 256 int(1024 byte)!
#define K_SHARD_HEADER_SIZE 256

class WIKI;
class Fish;
// struct train_state;
class Optimizer;
class NLP_AutoRegressive;
class SampNanny;

// the type of update each batch
enum DL_BATCH_UPATE {
    SAMPLEofSHARD,    //  SampNanny::Samp2Batch -> hBatch->Set(...);
    BATCHofEMBED = 1  //  TokenEmbed::hBatch
};

struct StepInfos {
    string sTokenSet = "", sRoot = "./";
    Optimizer* hOpt = nullptr;
    std::vector<hGTensor> csvTensors;  // only dump these tensors to .csv file

    struct STEP {
        float loss, lr, gNorm, tX, dt, gMax, wMax;
        int nValidLoss = 0;
        int iter, epoch;
        std::vector<float> nrmG, nrmW;
        std::map<std::string, std::vector<string>> details;
        string gMaxName, wMaxName;
        virtual string Info(int flag);

        STEP(float los_, int it_, int epo_, float lr_ = 0, float g_ = 0, float tX_ = 0, float dt_ = 0)
            : loss(los_), iter(it_), epoch(epo_), lr(lr_), gNorm(g_), tX(tX_), dt(dt_) {}
    };
    vector<STEP> steps;
    // vector<float> curve;
    int best_id     = -1;
    bool isAccuracy = false;
    virtual void Init(Optimizer* hO, int flag = 0x0);
    float Last() { return steps.empty() ? FLT_MAX : steps[steps.size() - 1].loss; }
    virtual void AfterStep(int iter, int flag = 0x0);
    virtual bool SaveToCSV(const string& sPath, int flag = 0x0);
    virtual bool SaveColorsToCSV(const string& sPath, int flag = 0x0);
    float Best() const;

    void Add(STEP step, int flag = 0x0);
};

enum MASK_FLAG {
    F_PAD         = 0x100,
    F_CAUSAL      = 0x200,
    F_SCORE_NOISE = 0x400,

    F_IGNORE_LOSS = 0x10000,  //  -100 of torch's cross_entropy
};

// Each tokenset has a unique samp_loader; Each samp_loader has a unique batch
struct BATCH_INPUT {
    LIFE_PHASE phasb              = P_X;  // batch only has one phase(never change in lifecycle!)
    shared_ptr<GTensor> hostLabel = nullptr;
    enum NOISE_TYPE {
        NO_NOISE,
        MASK_NOISE,
    };
    NOISE_TYPE tpNoise = NO_NOISE;
    
    shared_ptr<GTensor> hostToken = nullptr, hostMask = nullptr, devMask = nullptr, hostLen = nullptr;
    hRANDER hMaskRander = nullptr;
    int *host_toks = nullptr, *mask32 = nullptr;  //  mask32 = TO<int>(hostMask);
    int* tmp32       = nullptr;
    float* fNoise    = nullptr;  // float noise for each tokens
    int nValidTokens = 0, nNoiseToken = -1, nPadToken = -1;
    int nMostSample = 0, ldT = 0;  //  B,T
    int dB4Logit    = -1, iter = -1;
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
        FREE_a(fNoise);
    }
    virtual int nTokens(int flag = 0x0);  // return hostToken->size();
    // virtual void Update(hGTensor batch,int flag=0x0);
    virtual int FillPrompt(Fish* hFish, const std::vector<std::string>& Prompt, const std::vector<std::string>& answers, int nRound, int flag = 0x0);
    virtual void FillTokens(int k, const std::vector<TOKEN_ID>& tokens, int i_off, int flag);
    // 1. Deprecated, replace by FillTokens 2. No BOS at sequence start!
    virtual void Reset(const std::vector<TOKEN_ID>& tokens, int flag = 0x0);
    virtual void SetLen(int i0, int len) {
        if (hostLen != nullptr)
            hostLen->Set(i0, 0, 0, 0, len);
    }
    virtual void SetToken(int i0, int i1, int i2, int i3, int tok);
    virtual void SetMask(int i0, int i1, int i2, int i3, int tok) { hostMask->Set(i0, i1, i2, i3, tok); }
    // Label may be noised(>0) in some models(mask-noise diffusion model)
    virtual bool SetLabel(int label0, int i_target, int k, int tpNoise = -1, int flag = 0x0);
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
    virtual bool AddSomeNoise(const std::vector<hSAMP>& samps, int iter, int flag = 0x0);

    virtual void DumpX(TOKEN_ID* tokens, float* hostLoss, int flag = 0x0);
};
typedef shared_ptr<BATCH_INPUT> hBATCH;

// Batch for denoising diffusion model
struct BATCH_Denoise : public BATCH_INPUT {
    BATCH_Denoise(Fish* hFish, SHAPE sp, LIFE_PHASE phasb, int flag = 0x0);
    void FillTokens(int k, const std::vector<TOKEN_ID>& tokens, int i_off, int flag) override;
};

/**
 * 1. Each tokenset has a unique samp_nanny
 * 2. samp_nanny load each samp from tokenset, records its' train/eval/chat infos
 */
class SampNanny : public std::enable_shared_from_this<SampNanny> {
   protected:
    typedef std::string mt19937_state;

    Distri_ARRAY iiLoss;
    // ppl(Perplexity) is the exponential of the average cross entropy; or geometric mean of the inverse probabilities of each token
    Distri_ARRAY iiPPL;
    float* T_mask_probs = nullptr;
    //  Store tokens from source.  always in CPU
    int eval_every = -1, tokens_per_iter = 0;
    TRAIN_CARD _params;  // only need seed & force_reshuffle

    std::string fp_data;
    std::string sentence = "";
    std::vector<TOKEN_ID> samp_toks;
    std::vector<hSAMP> shard_samps;

    size_t nShard() { return shard_samps.size(); }
    // data_token_set is much complex than train/val datasets
    hDataToken hDaTokens = nullptr;
    hTokenizer hDict     = nullptr;

    // bool isTarget_1 = false;
    bool isRecycle = true, isLastShard = false;
    bool isFixEvalSample = false;  // Need fix this to do some experiments
    bool isMask          = false;
    mt19937_state shuffle_rng_state_current;
    mt19937_state shuffle_rng_state_next;
    size_t shuffle_sample_count = 0, next_sample = 0, shuffle_samples_hash = 0x0;
    hBATCH hBatch               = nullptr;
    NLP_AutoRegressive* dolphin = nullptr;

    /**
     * 1. Most open-source LLMs use BOS at sequence start;
     *      some papers analyzing the “attention sink” phenomenon explicitly note that the first token is almost always a BOS token
     * 2. diffusion LMs don’t develop sink heads
     * 3. chat(InitOneSamp) no need bos
     *  */
    bool isAddBOS = true;
    bool isNoShiftLabel = false;

   public:
    StepInfos stepis;                 // info of each step on train/evaluate/...
    std::string tpBatchSample, name;  //
    std::vector<hSAMP> cur_samps;
    int nMostToken  = -1;
    int num_batches = -1;        // number of batchs in each epoch
    int B = -1, T = -1, C = -1;  // number of samples in each batch,  number of tokens in each sample
    size_t nEvalTokens = 0;
    int StepOfEvaluate(int flag = 0x0);  //  smaple to reduce eval time

    int64_t len() { return shard_samps.size(); }
    bool empty() { return len() == 0; }

    // Derprecated, only for PPL
    size_t nTokens() { return hDaTokens->tokens.size(); }
    // Derprecated, only for PPL
    vector<TOKEN_ID>& GetTokens() { return hDaTokens->tokens; }

    int nLeastCTX(int flag = 0x0);
    hSAMP SampAt(size_t idx_) {
        assert(idx_ < nShard());
        return shard_samps[idx_];
    }
    virtual void ClearII() {
        iiLoss.Clear(); 
        iiPPL.Clear();
    }
    virtual double LossOnResult(Fish *hFish, int flag = 0x0);
    virtual float UpdateII(float mean_loss, int flag);
    hBATCH GetCurBatch(int flag = 0x0) const {
        assert(hBatch != nullptr);
        return hBatch;
    }
    virtual bool isEval(int t, int flag = 0x0);
    virtual hSAMP Next(bool isLoop = true);
    virtual bool NextEpoch(int flag = 0x0);
    virtual string IterInfo(int flag = 0x0);
    virtual string sTokenSet(int flag = 0x0);

    // May be pad_id/mask_id
    TOKEN_ID TokenAt(size_t pos, hSAMP samp, int flag = 0x0);
    bool MaskAt(size_t pos, TOKEN_ID& mask);
    // Deprecated!!!
    bool isHostMask(size_t pos, int flag = 0x0);
    std::vector<std::string> curDeTexts;

    // 1. prompt=>tokens 2. hTokens->tokens=tokens 3.Samp2Batch 4. hBatch->Set(i, token)
    virtual hSAMP InitOneSamp(const string& prompt, hGTensor input, Fish* hFish, int flag = 0x0);
    virtual double DecodeVerify(hSAMP samp, hGTensor tokens, hGTensor logits, int flag = 0x0);
    void Samp2Batch(int k, hSAMP samp, const CLI_params& params, float T_mask, int flag = 0x0);

    DT_TYPE type = DT_TRAIN;

    Optimizer* hOPT = nullptr;

    SampNanny() {}
    SampNanny(Fish* g_, const string& n, bool isNewTS, int flag = 0x0);
    virtual ~SampNanny() {
        if (!shard_samps.empty()) {
        }
    }

    virtual int PickSomeTokens(GRander& rander, int nSample, std::vector<int>& samps, int flag = 0x0);
    virtual bool Prepare(Optimizer* hO, hDataToken hT, int flag = 0x0);
    virtual bool SetOPT(Optimizer* hO, int flag = 0x0);
    virtual void UpdateStepInfos(float mean_loss, int nB, int flag = 0x0);
    virtual size_t CollateBatch(int next_id, Fish* fish);
    virtual double Evaluate(DL_BATCH_UPATE tpBatch, int flag = 0x0);

#ifdef _DATA_LOADER_LITE_
#else
    virtual bool Serialize(const std::string& path, bool isSave, int flag = 0x0);
    virtual void SetSamples(std::vector<size_t>& begin_, std::vector<size_t>& size_, bool isTrain, CLI_params& train_params, int flag = 0x0);
    void Shuffle(int flag = 0x0);
    bool TopoOrder(std::vector<size_t>& ids, std::mt19937& rng, int flag = 0x0);
#endif
    virtual void Dump(int typ);
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class Fish;
    friend class GeneratOnPrompt;
    friend class Head4Token;
    friend class DataTokenSet;
    friend class GlobTokenset;
};
typedef shared_ptr<SampNanny> hSampNanny;

//  one batch may contain many smales

// class DataLoader_3D : public SampNanny  {
// protected:
// public:
//     int64_t CollateBatch(int next_id,Fish* fish)    override;
// };
#endif  // DATALOADER_H