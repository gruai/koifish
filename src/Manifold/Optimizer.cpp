/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  1. Spike analysis
 *      Some cause:   1).Large lr  2) ROPE
 *
 *  \brief Optimizer
 *  \author Yingshi Chen
 */
#include "Optimizer.hpp"

#include "../Device/Pipe.hpp"
#include "../TokenSet/Dictionary.hpp"
#include "../ggex/GG_util.hpp"
#include "gLLM.hpp"

int tpFuseCu = 1;
struct train_params_ Optimizer::TrainParams() { return _fish->config.common; }

Optimizer::Optimizer(NLP_AutoRegressive *g_, CLI_params &config, int flag) : _fish(g_) {
    // adam_filter =  {"output","norm","embd"};
    adam_filter = {"output", "norm"};
    rRounding.Init(1314);
    tpSign            = config.Get({"train", "optimizatioin", "sign"}, 0, false);
    string method     = config.Get({"train", "optimizatioin", "method"}, string("adamw"), false);
    auto train_params = TrainParams();
    /*
        Although many people think "ADAMw is much better than SGD for attention models"  https://arxiv.org/pdf/2310.01082
        But may litter slower. For example:   jModel_SGD_v.info
    */
    tpGD = method == "adamw" ? ADAMw : method == "sgdv" ? SGD_v : method == "sgd" ? SGD : method == "hsgd" ? SGD_HYBRID : method == "lion" ? LION : ADAM_spike;

    nGradAccum   = std::max(1, train_params.n_gradient_accumulation);
    isGlobalGrad = nGradAccum > 1;  // Nearly same alloc grad or not
    train_loader = std::make_shared<SampLoader>(_fish, "Train", false);
    train_loader->Prepare(this, _fish->tsTrain);
    for (auto tsE : _fish->tsEval) {
        auto loader  = std::make_shared<SampLoader>(_fish, "Eval", false);
        loader->type = SampLoader::TYPE::DT_EVAL;
        loader->Prepare(this, tsE);
        val_loaders.push_back(loader);
    }

    if (_fish->isTrain()) {
        hLR = std::make_shared<DiscreteSchedule>(_fish->config.common);
        // train_loader->Init(g_,"Train");

    } else {
    }
}

bool Optimizer::SetPhase(LIFE_PHASE phase_, int flag) {
    phase = phase_;
    _fish->GetScheduler<RLS_BP>()->SetPhase(phase);
    return true;
}

hGensor Optimizer::hLoss() {
    assert(_fish != nullptr);
    if (_fish->loss != nullptr) {
        // auto a = _fish->loss->Item();
        // assert(ggml_is_scalar(G(_fish->loss)));
    }

    return _fish->loss;
}

hGensor Optimizer::hTargetProbs() { return _fish->target_probs; }
hGensor Optimizer::hPreLogits() { return _fish->hCLS->preLogits; }

hGensor Optimizer::GradOf(hGensor node, int flag) {
    // assert(0);
    // auto cgraph = _fish->GetBackRaw();
    return nullptr;  //::GradOf(cgraph,node,flag);
}

/* Xoshiro256PlusSIMD       https://github.com/stephanfr/Xoshiro256PlusSIMD/blob/main/README.md
 * https://www.lomont.org/papers/2008/Lomont_PRNG_2008.pdf
 */
/*
// ggml_compute_forward_cross_entropy_loss_f32
    float max = -INFINITY;
    ggml_vec_max_f32(nc, &max, s0);
    ggml_float sum = ggml_vec_soft_max_f32(nc, st, s0, max);        assert(sum > 0.0);
    sum = (1.0 - eps) / sum;        //eps = 1e-9;
    // avoid log(0) by rescaling from [0..1] to [eps..1]
    ggml_vec_scale_f32(nc, st, sum);
    ggml_vec_add1_f32(nc, st, st, eps);
    ggml_vec_log_f32(nc, st, st);
    ggml_vec_mul_f32(nc, st, st, s1);

    float st_sum = 0;
    ggml_vec_sum_f32(nc, &st_sum, st);
    sums[ith] += st_sum;
*/
bool Optimizer::OnLogits(int flag) {
    auto pTarget = hTargetProbs();
    size_t nz = tELEM(pTarget), nToken = pTarget->ne[0], n_ctx = pTarget->ne[1];

    auto pLogits = hPreLogits();
    assert(pLogits != nullptr);
    assert(pLogits->type == typNUMBER::F32);

    float *p = (float *)(pLogits->data), sum = 0;  // target_probs->data

    for (int k = 0; k < n_ctx; k++) {
        sum = 0;
        for (int i = 0; i < nToken; i++) sum += p[i];
        // assert(fabs(sum-1.0)<1.0e-6);
    }

    return true;
}

bool cuClearGrad(std::vector<hGTensor> tensors, int flag);
bool OPT_Adam::BatchGrad(int iter, float &fx, int flag) {
    fx           = 0;
    auto loss    = hLoss();
    float *fLoss = (float *)(loss->data), *g = nullptr, accum_norm = 1.0f / (float)nGradAccum;
    OutCLS *cls  = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    cls->hLoader = train_loader;
    if (grad != nullptr) {
        ZERO_(grad);
        g = (float *)grad->data;
    }
    bool bench = false;

    for (int accum_step = 0; accum_step < 1 /*nGradAccum*/; ++accum_step) {
        auto now      = GST_ms();
        int64_t nSamp = train_loader->UpdateBatch(-1, _fish);
        tData         = GST_ms() - now;
        if (nSamp == 0) {
            _WARN("<%s> Failed to get next batch!!!\n", __func__);
            return false;
        }

        train_samples += nSamp;
        if (!AfterLoadBatch(accum_step)) {
            return false;
        }

#ifdef _TENSOR_G_
        // cuClearGrad(opt_ps,0x0);    //  reset grad online, no need this
        // double a = tNormOf(opt_ps,0x0);
        GraphCompute(train_loader, _fish->hBackTG);
#else
        auto grad = GradOf(loss);
        tSET(grad, 1.0f);  // ggml_set_f32      (grad, 1.0f);
        if (bench) {       //  only for performance benchmark
            GST_TIC(t0);
            // ggml_graph_comp0(_gf,0x0);
            // ggml_graph_comp0(_gb,0x0);
            _INFO("gb_compute %s T=%.3g", "", GST_TOC(t0));
            exit(-666);
        } else {
            GraphCompute(train_loader, _fish->hBackTG);
        }

        OnLogits();
        if (isGlobalGrad) {
            ggml_opt_acc_grad(opt_ps.size(), opt_ps.data(), g, accum_norm);
        } else {  //  g[i++] += ggml_get_f32_1d(ps[p]->grad, j) * scale;
        }
        fx += tGET(loss, 0);
        UpdateTrainLoss(-1, fx);
#endif

        if (accum_step == 0) {
            // sched = UpdateSchedule(flag);
        }
    }
    fx *= accum_norm;

    return true;
}

bool isGensor(hGensor gensor, vector<string> keys, int flag = 0x0) {
    string name = gensor->name;
    for (auto key : keys) {
        if (name.find(key) != std::string::npos)
            return true;
    }
    return false;
}

int Optimizer::SignStochastic(int nx, CLI_params &config, int flag) {
    if (tpSign <= 0)
        return tpSign;
    if (grad == nullptr) {
        for (auto hP : opt_ps) {
            size_t ne = tELEM(hP);
            float *g  = (float *)(GradOf(hP)->data);
            for (int64_t i = 0; i < ne; ++i) {
                g[i] = g[i] > 0 ? 1 : -1;
            }
        }
    } else {
        float *g   = (float *)grad->data;
        double sum = 0.0, norm = 0;
        for (int64_t i = 0; i < nx; ++i) {
            g[i] = g[i] > 0 ? 1 : -1;  // signed
            sum += (g[i] * g[i]);
        }
        norm = sqrt(sum);
        assert(norm < FLT_MAX);
    }

    return 0x0;
}

/**
 * 1. LARS/LAMB  trus_ratio - ratio between the norm of the layer weights and norm of gradients update
 */
void Optimizer::UpdateParams(int nx, CLI_params &config, int flag) {}

void OPT_Adam::UpdateParams_V0(int nx, CLI_params &config, int flag) {
    floatX *g = nullptr;

    float clip = 0.f, fx = 0, sum, sched;
    g2_sum = 0;
    if (!isAdaptiveSched)  // params.adam->sched;
        sched = 1.0;
    if (isPreGStep)
        g_step = tNormsOf(opt_ps, 0x0);
    if (DEBUG.train_hyperparams == 1) {
        adam->decay = 0;
        sched       = 1.0;
    }

    if (grad != nullptr) {
        g    = (floatX *)grad->data;  // gradients
        clip = gClip(nParams, g, nullptr);
    }
    auto now = GST_ms();
    beta1h   = sched / (1.0f - powf(adam->beta1, iter));
    beta2h   = 1.0f / (1.0f - powf(adam->beta2, iter));
    size_t i = 0;
    zmuv_0 = DBL_MAX, zmuv_1 = 0.0;
    if (iter == 1)
        _INFO("clip=%g(%g) lr=%g beta1=%g , beta2=%g,  eps=%g , weight_decay=%g\n", 1.0e-6, 0.0, adam->alpha, adam->beta1, adam->beta2, adam->eps, adam->decay);
    {
        for (auto t : opt_ps) {
            // p_decay = ((tDIM(t) >= adam->decay_min_ndim) ? adam->decay : 0.0f) * sched;
            if (t->isRefer() || !t->isParam())
                continue;
            if (DEBUG.check_tensor_norm) {
                // assert(hP->nrm<1000.0);     //  only for debug
            }
            UpdateTensorParam(t, g == nullptr ? nullptr : g + i, clip);
            i += tELEM(t);
        }
    }
    assert(i == nParams);

    tUpdate = GST_ms() - now;
}

float Optimizer::gClip(int ne, floatX *g, hGensor hP, int flag) {
    float clip = 1.0f, a, a1 = -FLT_MAX;
    double sum = 0.0;

    for (int64_t i = 0; i < ne; ++i, ++g) {
        a = T2Float(g);
        sum += a * a;
        a1 = std::max(fabs(a), a1);
    }

    g_ll = sum;
    g2_sum += g_ll;
    double norm = sqrt(sum), avg = sqrt(sum / ne);
    if (gclip > 0.0f) {  // Gradient clipping maybe superior than L2 Regularization in terms of gaining results and easiness in implementation
        // gradient clipping
        assert(norm < FLT_MAX);
        if (norm > gclip) {
            clip = (float)(gclip / norm);
        }
    }
    assert(clip > 0);

    if (hP != nullptr) {
        if (fabs(a1) < 1.0e-10) {
            _INFO("\t|g| = 0@%s!", hP->name);
        }
        if (isnan(avg)) {
            _INFO("\tNAN  |g|@%s!!", hP->name);
        }
    }

    return clip;
}

inline bool isStrMatch(const string &target, const vector<string> &words) {
    for (auto w : words) {
        if (target.find(w) != std::string::npos)
            return true;
    }
    return false;
}

template <typename Tp, typename Tmv>
void Optimizer_update(PIPE_Optimizer<Tp, Tmv> &pipe, cudaStream_t stream);
int GTensor::Dogleg(int flag) {
    hOptimizer hOPT = hFish->GetOptimizer();
    int iter        = hOPT->GetITER();
    if (iter == last_iter)  // may try dogleg more than once in one optimization step
        return 0x0;

    ADAM_params_ adam   = hOPT->TrainParams().adam;
    float learning_rate = hOPT->LearningRate(), beta1 = adam.beta1, beta2 = adam.beta2, eps = adam.eps, wd = adam.decay;
    size_t nEle = size();

    if (shape.size() == 1)  // we only want to weight decay the 2D tensors and leave all 1D tensors alone
        wd = 0;

    Length(1);
    // if(flag==-1)    return 0x0;
    if (gnorm > adam.gclip) {
        // _INFO("\tdelta|%s|=%g scale=%g\n",name,grad_norm,adam.gclip/grad_norm);
    }

    float grad_scale = (gnorm > adam.gclip) ? adam.gclip / gnorm : 1.0f;
    if (hFish->config.lars_ratio > 0) {
        grad_scale = rLARS(grad_scale, hFish->config.lars_ratio, 0x0);
    }
    seed = hOPT->rRounding.RandInt32();
    PIPE_Optimizer<floatX, floatMV> pipe(nEle, nEle, nEle, nEle, flags, learning_rate, beta1, beta2, iter, eps, wd, grad_scale, gnorm, seed);
    pipe.Update(this);
    Optimizer_update(pipe, main_stream);

    if (flag == -1) {
        // Print(name, 1, -1);
    }
    last_iter = iter;
    return 0;
}

int UpdateTensorParam_cuda(hGTensor tensor, Optimizer *hOPT, float &grad_norm, int flag);
double OPT_Adam::UpdateTensorParam(hGensor hP, floatX *gX, float clip) {
    // assert(gimap.find(hP)!=gimap.end());
    float alpha = adam->alpha, beta1 = adam->beta1, beta2 = adam->beta2, eps = adam->eps, grad_norm = g_step;
    auto &im = _fish->GetGensorInfo(hP);  // gimap[hP];
    float *m = im.gm, *v = im.gv;
    bool isToHost    = false;  // out of GPU memory!
    const int64_t ne = tELEM(hP);
    floatX *paramX   = (floatX *)(hP->data), *paramX0, *gX0;
    float mh, vh, g0, x, x0, x00;
#ifdef __USE_CUDA__
    if (isToHost) {
        assert(_tmp != nullptr);
        paramX  = (floatX *)_tmp;
        gX      = paramX + hP->szData;
        paramX0 = paramX, gX0 = gX;
        hP->SerialGP(paramX, gX, hP->szData, true);
        x00 = T2Float(paramX);
        if (DEBUG.train_hyperparams == 1) {
            clip = 1.0e-6;
        } else {
            clip = gClip(ne, gX, hP);
        }
    }
#else
    if (gX == nullptr) {
        gX   = fGrad(hP);
        clip = gClip(ne, gX, hP);
    } else {
    }
    assert(hP->type == typNUMBER::F32);
    if (config.lars_ratio > 0) {  // lars/lamb
        float wnorm = 0, norm = 0, trust_ratio;
        for (int64_t j = 0; j < ne; ++j) {
            float x = ggml_get_f32_1d(opt_ps[p], j);
            wnorm += x * x;
        }
        for (int64_t j = i; j < i + ne; ++j) {
            norm += g[j] * g[j];
        }
        trust_ratio = sqrt(wnorm) / sqrt(norm + eps);
        trust_ratio = std::min(trust_ratio, config.lars_ratio);
        gnorm       = trust_ratio;
    }
#endif
#ifdef __USE_CUDA__
    hP->Dogleg(0x0);
    grad_norm = hP->gnorm;
    // UpdateTensorParam_cuda(hP, this, grad_norm, 0x0);

    if (!isPreGStep)
        g_step += grad_norm * grad_norm;
    if (isToHost) {
        x  = T2Float(paramX0);
        g0 = x - x00;
        hP->SerialGP(paramX0, gX0, hP->szData, false);
    }
#else
    GD_METHOD tpCurGD = tpGD;
    if (tpGD == SGD_HYBRID) {
        tpCurGD = im.isAdam ? ADAMw : SGD;
    }
    switch (tpCurGD) {
        case SGD:  //  converge very slow  !!!
            for (int64_t j = 0; j < ne; ++j, v++, gX++, paramX++) {
                g0      = *gX * clip;
                x       = T2Float(paramX);
                *paramX = x - alpha * (g0);
            }
            break;
        case SGD_v:  // why v is so important?
            for (int64_t j = 0; j < ne; ++j, v++, gX++, paramX++) {
                g0      = *gX * clip;
                x       = T2Float(paramX);
                *v      = *v * beta2 + g0 * g0 * (1.0f - beta2);
                vh      = sqrtf(*v * beta2h) + eps;
                *paramX = x - alpha * (g0) / vh;  //  beta1h = learning rate
            }
            break;
        case SGD_blk_v: {                                           // why v is so important?
            double vb = *v * beta2 + (g_ll / ne) * (1.0f - beta2);  // *v = *v*beta2 + g0*g0*(1.0f - beta2);
            *v        = vb;
            v++;
            vh = sqrtf(vb * beta2h) + eps;
            for (int64_t j = 0; j < ne; ++j, gX++, paramX++) {
                g0      = *gX * clip;
                x       = T2Float(paramX);
                *paramX = x - alpha * (g0) / vh;  //  beta1h = learning rate
            }
        } break;
        case ADAMw:
        default:

            for (int64_t j = 0; j < ne; ++j, m++, v++, gX++, paramX++) {
                float g = T2Float(gX);
                x0 = x = T2Float(paramX);
                // float g0 = isZMUV ? *gX : *gX*clip;
                g0      = g * clip;
                *m      = *m * beta1 + g0 * (1.0f - beta1);       // linear interpolations of the gradients
                *v      = *v * beta2 + g0 * g0 * (1.0f - beta2);  // linear interpolations of their variances
                mh      = *m * beta1h;                            // debiasing
                vh      = *v * beta2h;
                vh      = sqrtf(vh) + eps;
                x       = x * (1.0f - p_decay) - alpha * mh / vh;  // ormalize step(mh) by its standard deviation(vh)
                *paramX = x;                                       // ggml_set_f32_1d(hP, j, x);
                //  !!!update our low precision version of the parameters using stochastic rounding!!!
            }

            break;
    }
#endif

    return 0.0;
}

void Optimizer::UpdateTrainLoss(int x, float loss, int flag) {
    int step = iter;
    if (!fx_prev.empty()) {
        float last    = hLR->Last();
        isStopImprove = step > 0 ? (loss > last * 1.1) : false;
        if (isStopImprove) {
            // _INFO("%s_%d: StopImprove\n", __func__, iter);
        }
        fx_prev[0] = loss;
    } else if (fx_prev.empty()) {
        fx_prev.push_back(loss);
        fx_best.push_back(loss);  //   only 1 element  ???
        loss_before = loss;
    }
    loss_after = loss;
    hLR->Append(loss);

    isConverge = false; /*fabsf(loss_after - fx_prev[0])/loss_after < train_params.adam->eps_loss*/
    // check for improvement
    /*if (train_params.opt_max_no_improvement > 0) {
        if (fx_best[0] > fx) {
            fx_best[0] = fx;
            n_no_improvement = 0;
        } else {
            ++n_no_improvement;

            if (n_no_improvement >= train_params.opt_max_no_improvement) {
                result = OK;
            }
        }
    } */
    // fx_prev[0] = fx;
}

int Optimizer::GetITER(int flag) const {
    //  first_iter             = iter;
    //  iter = iter0 + t + 1;
    return iter;
}

bool Optimizer::isSpike(int flag) { return false; }

int RAW_update(std::vector<hGTensor> &tensors, Optimizer *hOPT, float &grad_norm, int alg, int flag);
/*
 */
Optimizer::RESULT Optimizer::Search(void *ctx, hGensor loss_, hGensor target_, CLI_params &config) {
    hEDS = _fish->hEDS;
    assert(hEDS != nullptr);
    auto train_params = TrainParams();

    last_time                = GST_ms();
    Optimizer::RESULT result = DID_NOT_CONVERGE;
    RLS_BP *hRLS             = _fish->hEDS->GetScheduler<RLS_BP>();
    bool cancel = false, isWarmup = false;
    string suf, pref;
    Dump(0x0);
    _INFO("\t%s@<%s> %s device=[%s] \n", __func__, _fish->hBackTG->name.c_str(), _fish->isLoadCheckpoint ? config.checkpoint.in.c_str() : "",
          hEDS->__repr__(suf, pref, 0).c_str());
    _INFO("\t Accumulation=%d AdaptiveSched=%d GRAP=%p rZMUV=%g rLARS=%g \n", nGradAccum, (int)isAdaptiveSched, grad, config.ZMUV_ratio, config.lars_ratio);
    // tpGD=SGD_HYBRID;    //ADAMw      SGD_v    SGD_HYBRID        SGD_blk_v
    _INFO("\tDECENT=%d(%s) SIGN=%d tpFuseCu=%d\n\n", tpGD, GD_NAME[tpGD].c_str(), tpSign, tpFuseCu);
    DEBUG.Dump(0);

    float a = 0, val_loss = 0, grad_norm = 0;
    if (isWarmup && !BatchGrad(0, a, 0x0))  //  warmup
        return CANCEL;

    if (just_initialized) {
        n_no_improvement = 0;
        just_initialized = false;
    }
    // g_dump_level = 0;
    int iter0 = 0, t;  // opt->iter;
    for (t = 0; t < train_params.nMostIter; ++t) {
        _fish->BeforeNextStep(t, 0x0);
        if (t == train_params.nMostIter - 1) {
            if (train_loader != nullptr) {
                train_loader->isLast = true;
                isDumpOnce           = true;
            }
        }
        iter = iter0 + t + 1;
        SUM::Reset("time");
        SetPhase(LIFE_PHASE::P_TRAIN);
        if (!BatchGrad(iter, a, 0x0))
            return CANCEL;
        // if(t==1)    exit(KOIFISH_EXIT_DEBUG);
        // const int64_t t_start_wall = ggml_time_us(),t_start_cpu = ggml_cycles();
        float lr = train_params.LearningRate();
        lr       = hLR->LearningRate(iter);
        last_lr  = lr;

        SignStochastic(nParams, config);
        if (_fish->config.scheduling.isUpdateParamV0()) {
            UpdateParams(nParams, config, 0x0);
        } else {
            for (auto t : opt_ps) {
                if (t->isRefer())
                    continue;
                // assert(t->last_stp==GetITER());
            }
        }
        if (!isPreGStep)
            g_step = sqrt(g_step);
        _fish->AfterNextStep(t, 0x0);

        UpdateLossCurve(0x0);
        // throw "DEBUG exit@";        //only for debug

        if (train_params.save_every > 0 && t % train_params.save_every == 0) {
            _fish->SaveTrain("");
        }
        if (t % 100 == 0)
            trainInfos().SaveToCSV("_info_.csv");
        SetPhase(LIFE_PHASE::P_EVAL_);
        for (auto vl : val_loaders) {
            if (vl->isEval(t)) {
                val_loss = Evaluate(vl, iter);  //  0.272727281
                //val_loss = vl->hTokens->Evaluate(_fish,vl,0x0);                // 
            }
        }
#ifdef _TENSOR_G_
#else
        if (config.common.gpt_every > 0 && t % config.common.gpt_every == 0) {
            _fish->GenSentence(1);
        }
#endif

        if (_fish->hDistler != nullptr)
            _fish->hDistler->UpdateSigma(iter);
        // check convergence
        if (isConverge) {
            _INFO("[search] Converged!!!\n");
            result = OK;
        }

        result = DID_NOT_CONVERGE;
    }
    double b = t / 1.0e6 * config.nTokenInBatch();
    _INFO("[train]: End of all epochs. nEpoch=%d nIter=%d(%d) nToken=%.5g(M)\n", train_epochs + 1, t, iter0, b);
    return result;
}

void Fish::Train(int flag) {}

/*
    1 Forward/Backward on neuron graph
    2 Forward/Backward on tensor graph
*/
double Optimizer::GraphCompute(hSampLoader hLoader, hTGraph hTG, int flag) {
    double now = GST_ms(), mean_loss = 0.0;
    int nThread = TrainParams().n_threads, no = 0, nAccum = TrainParams().n_gradient_accumulation;
    bool isOnlyEvaluate = !hTG->isBackward;

    OutCLS *cls       = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    TokenEmbed *embed = _fish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    if (phase != LIFE_PHASE::P_GENERATE && phase != LIFE_PHASE::P_PREFILL)
        embed->hBatch = hLoader->hBatch;

    isBackward = false;
    _fish->ForwardOnRLS(iter, 0x0);
    if (phase == LIFE_PHASE::P_GENERATE || phase == LIFE_PHASE::P_PREFILL) {
    } else {
        mean_loss = hLoader->hTokens->LossOnResult(hLoader, cls);
        if (isOnlyEvaluate) {
            return mean_loss;
        } else {
            isBackward = true;
            g_step     = 0;
            _fish->BackwardOnRLS(iter, 0x0);
            UpdateTrainLoss(-1, mean_loss);
            isBackward = false;
        }
    }
    // SUM::tX1 += GST_ms()-now;
    return 0.0;
}

/*
    Multiple purpose(Need refactor!)
    1. Get loss on some evaluate set in training procee
    2. Get loss on some evaluate set in unit-testing
    3. Prefill stage of Inference
    4. Generation stage of Inference
*/
float Optimizer::Evaluate(hSampLoader loader, int iter, int flag) {
    if (loader->num_batches == 0) {
        assert(0);
        return 0;
    }
    assert(loader->num_batches > 0);
    switch (phase) {
        case LIFE_PHASE::P_EVAL_:
            _INFO("[eval] ");
            break;
        case LIFE_PHASE::P_PREFILL:
            // _INFO("[prefill] " );
            assert(loader->num_batches == 1);
            break;
        case LIFE_PHASE::P_GENERATE:
            // _INFO("[generate] " );
            assert(loader->num_batches == 1);
            break;
        default:
            assert(0);
    }

    GST_TIC(tic);
    OutCLS *cls  = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    cls->hLoader = loader;
    auto loss    = hLoss();
    double l2, delta_max = 0, delta_ = 0, a, mean_loss = 0, ee = 0, tX = 0;
    auto tokens_input = _fish->Input();
    int i, nB = 0, step = loader->StepOfEvaluate();
    size_t nz           = 0, j;
    TOKEN_ID tMost      = (TOKEN_ID)(_fish->nClass() - 1);
    hSAMP samp          = nullptr;
    const float *wLog   = nullptr;
    loader->next_sample = 0;  // fix this to keep same acc on each experiment
    for (i = 0; i < loader->num_batches; i += step) {
        if (tokens_input != nullptr && (phase != LIFE_PHASE::P_PREFILL && phase != LIFE_PHASE::P_GENERATE)) {  // in some debug mode, tokens_input maybe null
            TIMING_ms(loader->UpdateBatch(i, _fish), tX);
            samp = loader->cur_samps[i];
        }
        if (wLog != nullptr) {
            for (l2 = 0, j = 0; j < nz; j++) {
                a = wLog[j];
                l2 += a * a;
            }
            l2        = sqrt(l2) / nz;
            delta_max = max(delta_max, l2);
            delta_ += l2;
        }
        mean_loss += GraphCompute(loader, _fish->hForwTG);
        nB++;
        if (_fish->config.isOnlyGPT) {
            return ee;
        }
#ifdef _TENSOR_G_
#else
        a  = ((float *)hPreLogits()->data)[0];  //  -6.60046101     -4.3040733
        ee = loader->DecodeVerify(samp, tokens_input, _fish->preLogits);
        mean_loss += loss == nullptr ? 0 : ((float *)(loss->data))[0];  // float *fLoss = (float*)(loss->data)
        break;
#endif
        if (_fish->wikis.size() > 0)  // too long
            break;
    }
    mean_loss /= nB;
    if (iter == -666) {  // hack
        return i;
    }

    float last = loader->stepis.Last();  //[eval]   Loss@Evaluation=7.302641 T=0.232s ======
    auto stp   = StepInfos::STEP(mean_loss, iter, train_epochs);
    loader->stepis.Add(stp);
    float delta = last - mean_loss, best = loader->stepis.Best();
    bool isOverfit = delta < 0 && abs(mean_loss - best) > best / 10;
    // if(isOverfit)   {
    //     _INFO(" !OVERFIT! ");
    // }
    a = nB * TrainParams().nTokenInBatch() / 1.0e6;
    _INFO(" Loss@\"%s\"=%.3f(%.2g) nToken=%.3gM best=%.4f(eval_%d) E2T=%.3g T=%g(%.3g)s x=%.3g\n", loader->sTokenSet().c_str(), mean_loss, delta, a, best,
          loader->stepis.best_id, mean_loss - trainInfos().Last(), GST_TOC(tic), tX / 1000.0, ee);  //

    if (wLog == nullptr) {
    } else
        _INFO("\t Loss@Evaluation=%f delta=%g(%.5g) T=%gs\n", mean_loss, delta_max, delta_ / nB, GST_TOC(tic));
    string sX = "_loss=" + std::to_string(mean_loss);
    // if(delta>0)
    //     _fish->SaveTrain(sX);

    loader->stepis.SaveToCSV("_info_.csv");

    return mean_loss;
}

string StepInfos::STEP::Info(int flag) {
    char buffer[256] = "\0";
    // _INFO("loss=%f |g|=%g\tlr=%.2e", loss,gNorm,lr);
    return buffer;
}

float Optimizer::UpdateLossCurve(int flag) {
    struct train_params_ _params = TrainParams();
    int n_batch = _params.n_batch, n_ctx = _params.n_ctx;
    double now = GST_ms();
    if (now > last_time && iter > first_iter) {
        double dt = (double)(now - last_time);
        if (millis_per_iter == 0.0) {
            millis_per_iter = dt;
        } else {
            const double gain = 0.7;
            millis_per_iter   = millis_per_iter * (1.0 - gain) + dt * gain;
        }
    }
    double remaining_millis = 0.0;
    if (millis_per_iter > 0.0) {
        // const int n_iter = _params.adam.n_iter,done_iter = iter - first_iter,remaining_iter = n_iter - done_iter;
        remaining_millis = (_params.nMostIter - (iter - first_iter)) * millis_per_iter;
    }

    last_time = GST_ms();

    int impr_plot = -(int)(1 + (loss_before - loss_after) * 10.0f + 0.5f);
    if (impr_plot > 0)
        impr_plot = 0;
    if (std::isnan(loss_before) || std::isnan(loss_after))
        impr_plot = 0;
    trainInfos().Add(StepInfos::STEP(loss_after, iter, train_epochs, last_lr, g_step, SUM::tX1, millis_per_iter));
    if ((iter - 1) % _params.dump_every == 0 || isDumpOnce) {
        isDumpOnce = false;
        _INFO("[epoch_%d]_%-6d loss=%f |g|=%.3g\tlr=%.2e | %s ", train_epochs, iter, loss_after, g_step, last_lr,
              train_loader->IterInfo().c_str());  //,zmuv_0,zmuv_1
        if (millis_per_iter > 0) {
            _TIME_INFO(" T=", millis_per_iter);
            _TIME_INFO("(data=", tData);
            SUM::TimeInfo();  // _TIME_INFO(" R=",SUM::tRemater);
            _TIME_INFO(" X=", SUM::tX1);
            _TIME_INFO(") eta=", remaining_millis);
        }
        size_t tokens_processed = _fish->config.nTokensPerGrad();  //(size_t) * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / millis_per_iter * 1000.0f;
        ema_tps                 = iter == 1 ? tokens_per_second : 0.95f * ema_tps + 0.05f * tokens_per_second;
        _INFO(" | %.1fK token/s | %s", ema_tps / 1000.0, _fish->DebugInfo().c_str());
        _INFO("\n");
    }
    float improvement = loss_before - loss_after;
    return improvement;
}

// Maybe more operation in the future
bool Optimizer::AfterLoadBatch(int accum_step, int flag) {
    auto _params = TrainParams();
    int n_batch = _params.n_batch, n_ctx = _params.n_ctx;

    if (accum_step == 0) {
    }
    const bool last_epoch_reached = (_params.n_epochs > 0 && train_epochs - first_epoch >= _params.n_epochs);
    if (last_epoch_reached) {
    }
    return true;
}

bool Optimizer::OnNextShard(int flag) {
    float a      = trainInfos().Best();
    string sRoot = "./";
    trainInfos().SaveToCSV("_info_.csv");
    for (auto vl : val_loaders) {
        float b = vl->stepis.Best();
        vl->stepis.SaveToCSV("_info_.csv");
    }
    // _fish->SaveTrain("");
    // Fish_ppl();

    return true;
}

bool Optimizer::OnNextEpoch(int flag) {
    train_epochs++;
    if (hLR->policy == LearnSKDU::COSINE_EPOCH) {
        // _INFO("-------- End of all shards of epoch_%d! -------- \n");
    }
    return true;
}

void Optimizer::AfterBuild(int flag) {
    if (_fish->isLocalInfer)
        hCache = std::make_shared<KVCache>(_fish);
}

void Optimizer::BeforeTrain(hGensor tokens_, int flag) {
    first_iter = iter;

    auto &adam        = _fish->config.common.adam;  // TrainParams().
    adam.n_parameters = nParams;

    opt_ps        = _fish->optParams;
    nMostParam    = 0;
    size_t offset = 0x0;
    for (auto t : opt_ps) {  //  Init @AfterBuild
        t->offset = nMostParam;
        nMostParam += tELEM(t);
        // ps->offset = offset;
        offset += t->nByte();
    }
    if (tpGD == GD_METHOD::LION) {  //  Based on our experience, a suitable learning rate for Lion is typically 3-10x smaller than that for AdamW
        adam.alpha /= 10;           // little better than /3
        adam.decay *= 10;
        adam.beta1 = 0.9;  //  the default values for β1 and β2 are discovered through the program search process and set as 0.9 and 0.99
        adam.beta2 = 0.99;
        _INFO("\tLION alpha=%g decay=%g\n", adam.alpha, adam.decay);
    }
    // adam.decay = 0.1;      //very high for GPT2 Model
    InitOnCUDA(1);
    assert(nMostParam >= nParams);
    assert(_fish->config.common.nMostIter > 0);
}

size_t TGraph::Prepare4Train(void *ctx_, GD_METHOD tpGD, int flag) {
    hOptimizer hOpt = hFish->hOPT;
    assert(hOpt != nullptr);
    size_t nP = 0, nz = 0, nzAll = 0, id = 0, n1 = hFish->gensors.size();
    for (auto &gi : hFish->gensors.infos) {
        auto gensor = gi.first;
        int nParam  = (int)tELEM(gensor);
        if (strcmp(gensor->name, "output.weight") == 0) {
            int xxx = 0;
        }
        // auto& im = gi.second;
        id++;
        if (!(gensor->flags & GTensor::F_PARAM))
            continue;
        nzAll += nParam;
        if (tpGD == GD_METHOD::SGD_HYBRID) {
            gi.second.isAdam = isStrMatch(gensor->name, hOpt->adam_filter);
        }
        gi.second.isAdam = isStrMatch(gensor->name, hOpt->adam_filter);
        if (!gi.second.isAdam)
            continue;

        nP++;
        nz += nParam;
#ifdef _TENSOR_G_
#else
        gi.second.gm  = new float[nParam]();
        gi.second.gv  = new float[nParam]();
        gi.second.gpf = new float[nParam]();
#endif
    }

    _INFO("[TGraph::%s] AdamTensor=(%d,%.3g%%) filter={", __func__, nP, nz * 100.0 / nzAll);
    for (auto f : hOpt->adam_filter) {
        _INFO("\"%s\" ", f.c_str());
    }
    _INFO("}\n");

    return nz;
}

void Optimizer::Prepare(size_t nx_, int flag) {
    // assert(nx_>=0);
    iter                        = 0;
    nParams                     = nx_;
    just_initialized            = true;
    double s                    = 0;
    size_t nz                   = 0;
    NLP_AutoRegressive *dolphin = dynamic_cast<NLP_AutoRegressive *>(_fish);
    if (nParams > 0) {
#ifdef __USE_GGML__
        struct ggml_init_params ctx_opt_params;
        ctx_opt_params.mem_size = GGML_MEM_ALIGN * 3 + ggml_tensor_overhead() * 3 + BPE(typNUMBER::F32) * nParams * 3;
        if (past > 0) {
            ctx_opt_params.mem_size += GGML_MEM_ALIGN + ggml_tensor_overhead() + BPE(typNUMBER::F32) * past;
        }
        ctx_opt_params.mem_buffer = NULL;
        ctx_opt_params.no_alloc   = false;
        _ctx                      = ggml_init(ctx_opt_params);
#endif
        if (_fish->isTrain()) {
            if (isGlobalGrad) {
                assert(nParams < INT_MAX);
                grad = GT(_fish, typNUMBER::F32, {(int)(nParams)});
            }

            nz = _fish->hBackTG->Prepare4Train(_ctx, tpGD);
            s  = nz * 1.0 / nParams;
            // gimap = _fish->hBackTG->gimap;
        }
        // train_loader->Init(dolphin,"Train",false);     //train_loader->Prepare(this);
    }
#ifdef _TENSOR_G_
    if (_fish->isTrain())
        _tmp = new float[nParams]();
#else
#endif
    // val_loader->Init(dolphin,"Eval",false);            //val_loader->Prepare(this);
}

bool Optimizer::PrepareData(CLI_params &config, int flag) {
    GST_TIC(tic);

    bool isLoadOK = false;
    string root = _fish->tsTrain->serial_root, spTrain = root + ".train", spEval = root + ".eval";
    if (root.empty()) {
        train_loader->Shuffle();
        return true;
    }
    hSampLoader val_loader = val_loaders.empty() ? nullptr : val_loaders[0];

    if (1) {
        if (train_loader->Serialize(spTrain, false) && val_loader->Serialize(spEval, false)) {
            if (train_loader->len() > 0) {
                // hDictVAE->nUniqueToken = train_loader->n_unique_tokens;
                // _INFO("%s: nTrain=%zu nEval=%zu tpBatchSample=%s T=%.3g\n", __func__, train_loader->len(),val_loader->len(),GST_TOC(tic));
                isLoadOK = true;
            }
        }
    }
    if (!isLoadOK) {
        hDataToken hTokenset = train_loader->hTokens;
        assert(hTokenset != nullptr);

        std::vector<size_t> samples_begin, samples_size;
        // auto train_params = config.common;
        size_t nUnique = hTokenset->nUnique, nVocab = hTokenset->nVocab;
        // int n_ctx_tokens = config.n_ctx;
        if (hTokenset->InitSamps(config.common.n_ctx, samples_begin, samples_size)) {
        } else {
            _INFO("%s: NULL Samps!!!    tpBatchSample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader->tpBatchSample.c_str(), train_loader->len(),
                  val_loader->len(), GST_TOC(tic));
            return false;
        }

        train_loader->SetSamples(samples_begin, samples_size, true, config);
        val_loader->SetSamples(samples_begin, samples_size, false, config);

        // assert(val_loader->n_unique_tokens <= nUnique && train_loader->n_unique_tokens <= nUnique);
        // val_loader->n_unique_tokens = nUnique;
        // train_loader->n_unique_tokens = nUnique;
        shuffle_samples_hash = train_loader->shuffle_samples_hash;
        train_loader->Serialize(spTrain, true);
        // train_loader->Serialize(spTrain,false);      //only for debug
        val_loader->Serialize(spEval, true);
    }

    // assert(train_samples_begin.size() == train_samples_size.size());
    _INFO("%s: tpBatchSample=%s nTrain=%zu nEval=%zu T=%.3g\n", __func__, train_loader->tpBatchSample.c_str(), train_loader->len(), val_loader->len(),
          GST_TOC(tic));
    return true;
}

void OPT_Adam::Prepare(size_t nx_, int flag) {
    Optimizer::Prepare(nx_, flag);
    if (_fish->isTrain()) {
        if (grad != nullptr) {
        } else {
        }
    }
}

/*
    https://github.com/KellerJordan/muon
    https://x.com/Kimi_Moonshot/status/1897929976948965870
*/
OPT_Muon::OPT_Muon(NLP_AutoRegressive *g_, CLI_params &params_, int flag) : Optimizer(g_, params_, flag) {}

OPT_Adam::OPT_Adam(NLP_AutoRegressive *g_, CLI_params &params_, int flag) : Optimizer(g_, params_, flag) {
    // auto train_params = TrainParams();
    //  0.9f, 0.95f, 1e-8f      decay=0.1

    adam                      = &(_fish->config.common.adam);
    adam->clip_alg            = 1;  // clip_alg=0 little better
    adam->gclip               = adam->gclip / _fish->config.nLayer();
    _fish->config.Fuse_Normal = 0;  //
    if (g_->isTrain()) {
        trainInfos().Init(this);
    } else {  // may different
        _fish->config.common.remater_ffn = 1;
        // _fish->config.common.remater_qkv = 1;
    }
    for(auto loader : val_loaders){
        loader->stepis.Init(this);
    }

    // sched              = 1.0f;
}

void Optimizer::Dump(int typ) {
    if (NOT_DUMP(1))
        return;
    _INFO("========\n");
    fflush(stdout);
    RLS_BP *hRLS = _fish->hEDS->GetScheduler<RLS_BP>();
    hRLS->Dump(typ);
    _INFO("\tType weight=%s activation=%s", cNameOf(_fish->config.model.tpWeight), cNameOf(_fish->config.model.tpActivation));
    fflush(stdout);

    size_t sz         = 0x0;
    const char *title = "OPT";  //__func__
    if (_ctx != nullptr)
        _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, sz, (float)sz / (1024.0f * 1024.0f));
    _INFO("%s: iter = %d\n", title, iter);

    if (typ == 1) {
        _INFO("%s: SAMP_HASH=%llu total train_iterations=%llu train_samples=%llu train_tokens=%llu completed_epochs=%llu\n", title, shuffle_samples_hash,
              train_its, train_samples, train_tokens, train_epochs);
    }
}

void OPT_Adam::Dump(int typ) {
    Optimizer::Dump(typ);

    /*
        printf("+-----------------------+----------------------------------------------------+\n");
        printf("| Parameter             | Value                                              |\n");
        printf("+-----------------------+----------------------------------------------------+\n");
        printf("| train data pattern    | %-50s |\n", train_data_pattern);
        printf("| val data pattern      | %-50s |\n", val_data_pattern);
        printf("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
        printf("| checkpoint_every      | %-50d |\n", checkpoint_every);
        printf("| resume                | %-50d |\n", resume);
        printf("| micro batch size B    | %-50d |\n", B);
        printf("| sequence length T     | %-50d |\n", T);
        printf("| total batch size      | %-50d |\n", total_batch_size);
        printf("| LR scheduler          | %-50s |\n", lr_scheduler_type);
        printf("| learning rate (LR)    | %-50e |\n", learning_rate);
        printf("| warmup iterations     | %-50d |\n", warmup_iterations);
        printf("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
        printf("| weight decay          | %-50e |\n", weight_decay);
        printf("| skip update lossz     | %-50f |\n", skip_update_lossz);
        printf("| skip update gradz     | %-50f |\n", skip_update_gradz);
        printf("| max_steps             | %-50d |\n", max_steps);
        printf("| val_loss_every        | %-50d |\n", val_loss_every);
        printf("| val_max_steps         | %-50d |\n", val_max_steps);
        printf("| sample_every          | %-50d |\n", sample_every);
        printf("| genT                  | %-50d |\n", genT);
        printf("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
        printf("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
        printf("| gelu_fusion           | %-50d |\n", gelu_fusion);
        printf("| recompute             | %-50d |\n", recompute);*/
    // if(NOT_DUMP())  return;
    _INFO("[OPT_Adam]\tsRESI=%g s_rounding=%d alloc_w=%d remater[ffn=%d ]\n", TrainParams().residual_scale, TrainParams().opt_alloc_weight,
          TrainParams().opt_alloc_weight, TrainParams().remater_ffn);
    adam->Dump(typ);
    if (hLR != nullptr)
        hLR->Dump(typ);
    _INFO("\tnParams = %ld(%.6gM)\n", nParams, nParams / 1.0e6);
    fflush(stdout);
}