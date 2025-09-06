/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Generate samples for train/Eval
 *
 *  \brief samples from multiple dataset/tokenset/...
 *  \author Yingshi Chen
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "../ggex/GG_util.hpp"
struct DictVAE;
#include "../CLI_params.hpp"
#include "../Utils/GST_rander.hpp"
#include "../g_stddef.hpp"
#include "TokenSet.hpp"
using namespace std;
// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define SHARD_HEADER_SIZE 256

class WIKI;
class Fish;
// struct train_state;
class Optimizer;
class NLP_AutoRegressive;
class SampLoader;

struct StepInfos {
    string name = "", sRoot = "./";
    Optimizer *hOpt = nullptr;
    struct STEP {
        float loss, lr, gNorm, tX, dt, gMax, wMax;
        int iter, epoch;
        std::vector<float> nrmG, nrmW;
        string gMaxName, wMaxName;
        virtual string Info(int flag);

        STEP(float los_, int it_, int epo_, float lr_ = 0, float g_ = 0, float tX_ = 0, float dt_ = 0)
            : loss(los_), iter(it_), epoch(epo_), lr(lr_), gNorm(g_), tX(tX_), dt(dt_) {}
    };
    vector<STEP> steps;
    // vector<float> curve;
    int best_id     = -1;
    bool isAccuracy = false;
    virtual void Init(Optimizer *hO, int flag = 0x0);
    float Last() { return steps.empty() ? FLT_MAX : steps[steps.size() - 1].loss; }
    virtual bool SaveToCSV(const string &sPath, int flag = 0x0);
    float Best() const;

    void Add(STEP step, int flag = 0x0);
};

struct BATCH_INPUT {
    shared_ptr<GTensor> hostToken = nullptr, hostMask = nullptr;

    int *host = nullptr, *mask = nullptr;
    void *dev = nullptr;
    int pos   = -1;

    BATCH_INPUT(SHAPE sp, int flag = 0x0);
    // virtual void Update(hGTensor batch,int flag=0x0);
    int CurToken() {
        assert(pos >= 0 && host != nullptr);
        // assert(host[pos] < embed->nVocab);
        return host[pos];
    }
    virtual void Set(int i0, int i1, int i2, int i3, int tok) { hostToken->Set(i0, i1, i2, i3, tok); }
    virtual void SetMask(int i0, int i1, int i2, int i3, int tok) { hostMask->Set(i0, i1, i2, i3, tok); }
    virtual size_t size() { return hostToken->size(); }
};
typedef shared_ptr<BATCH_INPUT> hBATCH;
class SampLoader : public std::enable_shared_from_this<SampLoader> {
   protected:
    typedef std::string mt19937_state;

    Distri_ARRAY iiLoss;
    // ppl(Perplexity) is the exponential of the average cross entropy; or geometric mean of the inverse probabilities of each token
    Distri_ARRAY iiPPL;

    //  Store tokens from source.  always in CPU
    int eval_every = -1, tokens_per_iter = 0;
    // CLI_params config;
    std::string fp_data;
    std::string sentence = "";
    std::vector<TOKEN_ID> samp_toks;
    std::vector<hSAMP> shard_samps;
    // std::vector<size_t> idcs;      //would change epoch by epoch(shuffle,subsampling...)
    size_t nShard() { return shard_samps.size(); }
    // std::shared_ptr<DictVAE> hDictVAE;
    hDataToken hTokens = nullptr;
    hTokenizer hDict   = nullptr;

    bool isNeedBOS = true;
    bool sample_separation_eos, sample_separation_bos;
    bool isTarget_1 = false;
    bool isRecycle = true, isLast = false;
    bool isFixEvalSample = false;  // Need fix this to do some experiments
    mt19937_state shuffle_rng_state_current;
    mt19937_state shuffle_rng_state_next;
    size_t shuffle_sample_count = 0, next_sample = 0, shuffle_samples_hash = 0x0;
    hBATCH hBatch               = nullptr;
    NLP_AutoRegressive *dolphin = nullptr;

   public:
    StepInfos stepis;                 // info of each step on train/evaluate/...
    std::string tpBatchSample, name;  //
    std::vector<hSAMP> cur_samps;
    int nMostToken  = -1;
    int num_batches = -1;        // number of batchs in each epoch
    int B = -1, T = -1, C = -1;  // number of samples in each batch,  number of tokens in each sample
    size_t nEvalTokens = 0;
    int StepOfEvaluate(int flag = 0x0);  //  smaple to reduce eval time

    shared_ptr<GTensor> hostTargetProbs = nullptr;  // hostBatch=nullptr,hostBatchMask=nullptr,

    int64_t len() { return shard_samps.size(); }
    bool empty() { return len() == 0; }
    size_t nTokens() { return hTokens->tokens.size(); }
    int nLeastCTX(int flag = 0x0);
    hSAMP SampAt(size_t idx_) {
        assert(idx_ < nShard());
        return shard_samps[idx_];
    }
    virtual void ClearII() {
        iiLoss.Clear();
        iiPPL.Clear();
    }
    virtual void UpdateII() {
        iiLoss.Stat();
        iiPPL.Stat();
    }
    hBATCH GetCurBatch(int flag = 0x0) const { return hBatch; }
    virtual bool isEval(int t, int flag = 0x0);
    virtual hSAMP Next(bool isLoop = true);
    virtual bool NextEpoch(int flag = 0x0);
    virtual string IterInfo(int flag = 0x0);
    virtual string sTokenSet(int flag = 0x0);
    vector<TOKEN_ID> &GetTokens() { return hTokens->tokens; }

    TOKEN_ID TokenAt(size_t pos) { return hTokens->At(pos); }
    bool MaskAt(size_t pos, TOKEN_ID &mask);
    bool isHostMask(size_t pos, int flag = 0x0);
    std::vector<std::string> curDeTexts;
    virtual hSAMP InitOneSamp(const string &prompt, hGensor input, Fish *hFish, int flag = 0x0);
    virtual double DecodeVerify(hSAMP samp, hGensor tokens, hGensor logits, int flag = 0x0);
    void Samp2Batch(int k, hSAMP samp, TRAIN_CARD &params, int flag = 0x0);

    enum TYPE { DT_TRAIN = 1, DT_EVAL, DT_PREDICT, DT_MERGE };
    TYPE type = DT_TRAIN;

    Optimizer *hOPT = nullptr;

    SampLoader() {}
    SampLoader(Fish *g_, const string &n, bool isNewTS, int flag = 0x0);
    virtual ~SampLoader() {
        if (!shard_samps.empty()) {
        }
    }

    virtual int PickSomeTokens(Grusoft::GRander &rander, int nSample, std::vector<int> &samps, int flag = 0x0);
    virtual bool Prepare(Optimizer *hO, hDataToken hT, int flag = 0x0);
    virtual void UpdateStepInfos(float mean_loss, int nB, int flag = 0x0);
    virtual size_t UpdateBatch(int next_id, Fish *fish);
    virtual double Evaluate(int flag = 0x0);

#ifdef _DATA_LOADER_LITE_
#else
    virtual bool Serialize(const std::string &path, bool isSave, int flag = 0x0);
    virtual void SetSamples(std::vector<size_t> &begin_, std::vector<size_t> &size_, bool isTrain, CLI_params &train_params, int flag = 0x0);
    void Shuffle(int flag = 0x0);
    bool TopoOrder(std::vector<size_t> &ids, std::mt19937 &rng, int flag = 0x0);
#endif
    virtual void Dump(int typ);
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class Fish;
    friend class GeneratOnPrompt;
    friend class OutCLS;
    friend class DataTokenSet;
    friend class GlobTokenset;
};
typedef shared_ptr<SampLoader> hSampLoader;

//  one batch may contain many smales

// class DataLoader_3D : public SampLoader  {
// protected:
// public:
//     int64_t UpdateBatch(int next_id,Fish* fish)    override;
// };
#endif  // DATALOADER_H