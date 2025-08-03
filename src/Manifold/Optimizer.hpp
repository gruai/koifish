/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <threads.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <map>
#include <memory>
#include <regex>
#include <stack>
#include <typeinfo>
#include <vector>
using namespace std;

#include "../Device/EDevice.hpp"
#include "../TokenSet/DataLoader.hpp"
#include "../Utils/Cache.hpp"
#include "../ggex/GG_util.hpp"
#include "Scheduler.hpp"
#include "TGraph.hpp"

class Fish;

class Optimizer : public std::enable_shared_from_this<Optimizer> {
    Optimizer(const Optimizer&);
    Optimizer& operator=(const Optimizer&);

   protected:
    std::string title = "Optimizer";
    // std::map<hGensor, GENSOR_INFO> gimap;

    void* _ctx = nullptr;
    std::vector<hGensor> opt_ps;
    size_t nParams = 0, nMostParam = 0;
    float* _tmp           = nullptr;
    bool just_initialized = false, isAdaptiveSched = false, isGlobalGrad = true;
    bool isConverge = false, isDumpOnce = false;

    bool isBackward = false;

    int past = 0, nGradAccum = 0, tpSign = 0;
    int warmup_iters = 0;
    // gradient clipping
    double gclip = 1.0;
    // schedule multiplier (fixed, decay or warmup)
    // float sched = 1.0f;

    hGensor grad = nullptr;  // current gradient
    vector<float> fx_best;
    vector<float> fx_prev;
    int n_no_improvement = 0;
    float loss_before = 0, loss_after = 0, tokens_per_second = 0, ema_tps = 0, last_lr = 0;
    int first_epoch = 0, iter_at_last_epoch = -1, first_iter = -1, iter = -1;
    uint64_t train_its = 0, train_samples = 0, train_tokens = 0, train_epochs = 0, max_epoch = 0;
    double last_time, tData, tUpdate;
    double millis_per_iter          = 0;
    std::vector<string> adam_filter = {"output", "norm"};  //{"token_embd","output","norm"};

    // void *app_ctx = nullptr;
    double zmuv_0 = 0.0, zmuv_1 = 0.0, g_step = 0.0, g_ll = 0, g2_sum = 0;

    hGensor hLoss();
    hGensor hTargetProbs();
    hGensor hPreLogits();
    hGensor GradOf(hGensor node, int flag = 0);
    float* fGrad(hGensor node, int flag = 0) {
        auto grad = GradOf(node);
        assert(grad != nullptr);
        float* g = (float*)(grad->data);
        return g;
    }
    float fLoss() {
        float* val = (float*)(hLoss()->data);
        return *val;
    }

    // Learning rate hLR
    hLearnSKDU hLR     = nullptr;
    bool isStopImprove = false;
    bool isPreGStep    = false;

    virtual void Clear() {}
    // update sched & dump some info
    virtual float UpdateLossCurve(int flag = 0x0);
    virtual bool AfterLoadBatch(int accum_step, int flag = 0x0);
    virtual bool OnNextShard(int flag = 0x0);
    virtual bool OnNextEpoch(int flag = 0x0);
    virtual int SignStochastic(int nx, CLI_params& config, int flag = 0x0);
    virtual float gClip(int nx, floatX* g, hGensor hP, int flag = 0x0);
    virtual void UpdateParams(int nx, CLI_params& config, int flag);
    // virtual void AdamMiniV(float gnorm,int nx,CLI_params& config,int flag);
    virtual bool BatchGrad(int iter, float& fx, int flag = 0x0) {
        assert(0);
        return false;
    }
    bool OnLogits(int flag = 0x0);

   public:
    GD_METHOD tpGD = ADAMw;
    enum RESULT {
        OK = 0,
        DID_NOT_CONVERGE,
        NO_CONTEXT,
        INVALID_WOLFE,
        FAIL,
        CANCEL,
    };

    LIFE_PHASE phase = LIFE_PHASE::P_TRAIN;

    struct STAGE {
        string name;
        LIFE_PHASE phase;
        double lr, w_decay;
    };
    vector<STAGE> stages;
    // PIPE_Optimizer pipe;
    Grusoft::GRander rRounding;  // stochastic rounding
    hSampLoader train_loader = nullptr;
    StepInfos& trainInfos() {
        assert(train_loader != nullptr);
        return train_loader->stepis;
    }
    std::vector<hSampLoader> val_loaders;
    size_t shuffle_samples_hash = 0x0;  // hack

    Fish* _fish     = nullptr;  // ref only
    hEDevices hEDS  = nullptr;  // ref only
    hKVCache hCache = nullptr;
    struct train_params_ TrainParams();

    Optimizer(NLP_AutoRegressive* g_, CLI_params& params_, int flag = 0x0);
    // Deprecated need refactor!!!       9/30/2024
    virtual double GraphCompute(hSampLoader loader, hTGraph, int flag = 0x0);
    virtual bool SetPhase(LIFE_PHASE phase_, int flag = 0x0);
    virtual float Evaluate(hSampLoader loader, int iter, int flag = 0x0);
    // virtual float Prefill(hSampLoader loader,int iter,int flag=0x0);
    virtual int GetITER(int flag = 0x0) const;
    virtual float LearningRate(int flag = 0x0) { return hLR->LearningRate(iter); }
    virtual void UpdateTrainLoss(int x, float loss, int flag = 0x0);  //
    virtual double UpdateTensorParam(hGensor hP, floatX* g, float gnorm) { return 0.0; }
    virtual bool isStopImproving() { return isStopImprove; }

    virtual void Dump(int typ);
    virtual void AfterBuild(int flag = 0x0);
    virtual void BeforeTrain(hGensor tokens_input, int flag);
    virtual void InitOnCUDA(int flag);
    virtual void ClearOnCUDA(int flag);
    virtual bool PrepareData(CLI_params& config, int flag);
    virtual void Shuffle(int n_vocab, struct train_params_& train_params, int flag = 0x0) { assert(0); }

    virtual bool isSpike(int flag = 0x0);

    virtual void Prepare(size_t nx, int flag = 0x0);

    // virtual void InitOpt(struct train_params_& params_,int flag=0x0);

    virtual ~Optimizer() {
        // ggml_free(_ctx);
        if (_tmp != nullptr)
            delete[] _tmp;
    }

    RESULT Search(void* ctx, hGensor loss_, hGensor target_, CLI_params& config);

    friend class Fish;
    friend class NLP_AutoRegressive;
    friend class GeNeuron;
    friend class SampLoader;
    friend class SAMP;
    friend class TGraph;
    friend class StepInfos;
};
typedef shared_ptr<Optimizer> hOptimizer;

class OPT_Adam : public Optimizer {
   protected:
    ADAM_params_* adam = nullptr;  // may be modified
    float p_decay      = 0;
    float beta1h, beta2h;

    void Prepare(size_t nx, int flag = 0x0) override;
    // compute grad on batchs
    bool BatchGrad(int iter, float& fx, int flag = 0x0) override;
    double UpdateTensorParam(hGensor hP, floatX* g, float gnorm) override;
    void UpdateParams_V0(int nx, CLI_params& config, int flag);

   public:
    OPT_Adam(NLP_AutoRegressive* g_, CLI_params& params_, int flag = 0x0);
    void Dump(int typ) override;
};

class OPT_Muon : public Optimizer {
   public:
    OPT_Muon(NLP_AutoRegressive* g_, CLI_params& params_, int flag = 0x0);
};