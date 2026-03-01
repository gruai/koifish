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
#include "gLLM.hpp"

int PackedN_bug = 0;
int tpFuseCu = 1;
TRAIN_CARD Optimizer::TrainParams() { return _fish->config.common; }

Optimizer::Optimizer(NLP_AutoRegressive* g_, CLI_params& config, int flag) : _fish(g_) {
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

    if (!_fish->isLocalInfer) {
        tpGD         = method == "adamw"   ? ADAMw
                       : method == "adams" ? ADAM_S
                       : method == "sgdv"  ? SGD_v
                       : method == "sgd"   ? SGD
                       : method == "hsgd"  ? SGD_HYBRID
                       : method == "lion"  ? LION
                       : method == "muon"  ? MUON
                                           : ADAMw;
        nGradAccum   = std::max(1, train_params.n_gradient_accumulation);
        isGlobalGrad = nGradAccum > 1;  // Nearly same alloc grad or not
        train_loader = std::make_shared<SampLoader>(_fish, "Train", false);
        train_loader->Prepare(this, _fish->tsTrain);
    }
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

bool Fish::SetPhase(LIFE_PHASE phase_, int flag) {
    phase = phase_;
    // _fish->GetScheduler<RLS_BP>()->SetPhase(phase);

    switch (phase) {
        case LIFE_PHASE::P_EVAL_:
            _INFO("[eval] ");
            break;
        case LIFE_PHASE::P_PREFILL:
            // _INFO("[prefill] " );
            // assert(loader->num_batches == 1);
            break;
        case LIFE_PHASE::P_GENERATE:
            _INFO("[generate] ");
            // assert(loader->num_batches == 1);
            break;
        default:
            break;
    }
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

    float *p = (float*)(pLogits->data), sum = 0;  // target_probs->data

    for (int k = 0; k < n_ctx; k++) {
        sum = 0;
        for (int i = 0; i < nToken; i++) sum += p[i];
        // assert(fabs(sum-1.0)<1.0e-6);
    }

    return true;
}

bool cuClearGrad(std::vector<hGTensor> tensors, int flag);
bool Optimizer::BatchGrad(int iter, float& fx, int flag) {
    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();
    fx           = 0;
    auto loss    = hLoss();
    float *fLoss = (float*)(loss->data), *g = nullptr, accum_norm = 1.0f / (float)nGradAccum;
    OutCLS* cls  = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    cls->hLoader = train_loader;
    train_loader->ClearII();
    if (grad != nullptr) {
        ZERO_(grad);
        g = (float*)grad->data;
    }
    bool bench = false;

    for (int accum_step = 0; accum_step < 1; ++accum_step) {
        auto now = GST_ms();
        if (hRLS->isUpdateBatch(GetITER())) {
            int64_t nSamp = train_loader->UpdateBatch(-1, _fish);
            SUM::tData    = GST_ms() - now;
            if (nSamp == 0) {
                _WARN("<%s> Failed to get next batch!!!\n", __func__);
                return false;
            }

            train_samples += nSamp;
            if (!AfterLoadBatch(accum_step)) {
                return false;
            }
        }

        GraphCompute(train_loader, _fish->hBackTG);

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

// [Experimental]
int Optimizer::SignStochastic(int nx, CLI_params& config, int flag) {
    if (tpSign <= 0)
        return tpSign;
    if (grad == nullptr) {
        for (auto hP : opt_ps) {
            size_t ne = tELEM(hP);
            float* g  = (float*)(GradOf(hP)->data);
            for (int64_t i = 0; i < ne; ++i) {
                g[i] = g[i] > 0 ? 1 : -1;
            }
        }
    } else {
        float* g   = (float*)grad->data;
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
void Optimizer::UpdateParams(int nx, CLI_params& config, int flag) {}

void OPT_Adam::UpdateParams_V0(int nx, CLI_params& config, int flag) {
    floatX* g = nullptr;

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
        g    = (floatX*)grad->data;  // gradients
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

float Optimizer::gClip(int ne, floatX* g, hGensor hP, int flag) {
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

int GTensor::Dogleg(int flag) {
    if (isStrMatch(name, {"blk.2.attn.wq"})) {  // model.blk.11.attn.wo.weight_a
        DEBUG_HERE;
        flag = -2;
    }
    assert(needUpdateParam);

    hOptimizer hOPT = hFish->GetOptimizer();
    int iter        = hOPT->GetITER();
    if (iter == last_iter)  // may try dogleg more than once in one optimization step
        return 0x0;

    ADAM_params_ adam   = hOPT->TrainParams().adam;
    float learning_rate = hOPT->LearningRate(), beta1 = adam.beta1, beta2 = adam.beta2, eps = adam.eps, wd = adam.decay;
    size_t nEle = size();
    //  Weight decay is typically disabled for 1D parameters (like biases and LayerNorm weights) and enabled for all other parameters.
    if (shape.size() < adam.decay_min_ndim)
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
    seed            = hOPT->rRounding.RandInt32();
    hPipeOpt hPipe0 = hOPT->hPipe;
    // hPipeOpt hPipe  = std::make_shared<PIPE_Adamw<floatX, floatMV>>(hOPT.get(), nEle, nEle, nEle, nEle, flags, learning_rate, beta1, beta2, iter, eps, wd);
    // // PIPE_Adamw<floatX, floatMV> pipe(nEle, nEle, nEle, nEle, flags, learning_rate, beta1, beta2, iter, eps, wd, grad_scale, gnorm, seed);
    // hPipe->Update(this, wd, grad_scale, seed);
    hPipe0->Update(this, wd, grad_scale, seed);
    hPipe0->CU_core(main_stream);
    // Optimizer_update(pipe, main_stream);
    if (1) {  // fuyou
        // for(auto t : fuyous)
    }

    if (flag == -2) {
        // Print(name, 1, -1);
    }
    last_iter = iter;
    SUM::nDogLeg++;
    return 0;
}

double Optimizer::UpdateTensorParam(hGensor hP, floatX* g, float gnorm) {
    float grad_norm = g_step;
    hP->Dogleg(0x0);
    grad_norm = hP->gnorm;
    if (!isPreGStep)
        g_step += grad_norm * grad_norm;

    return 0.0;
}

int UpdateTensorParam_cuda(hGTensor tensor, Optimizer* hOPT, float& grad_norm, int flag);
//  Always call Optimizer::UpdateTensorParam          Deprecated
double OPT_Adam::UpdateTensorParam(hGensor hP, floatX* gX, float clip) {
    // return Optimizer::UpdateTensorParam(hP, gX, clip);

    // assert(gimap.find(hP)!=gimap.end());
    float alpha = adam->alpha, beta1 = adam->beta1, beta2 = adam->beta2, eps = adam->eps, grad_norm = g_step;
    auto& im = _fish->GetGensorInfo(hP);  // gimap[hP];
    float *m = im.gm, *v = im.gv;
    bool isToHost    = false;  // out of GPU memory!
    const int64_t ne = tELEM(hP);
    floatX *paramX   = (floatX*)(hP->data), *paramX0, *gX0;
    float mh, vh, g0, x, x0, x00;
#ifdef __USE_CUDA__
    if (isToHost) {
        assert(_tmp != nullptr);
        paramX  = (floatX*)_tmp;
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
    RLS_BP* hRLS = _fish->hEDS->GetScheduler<RLS_BP>();
    if (hRLS->afu != nullptr)
        hRLS->afu->loss = loss;
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

bool Optimizer::isAtLongtail(int flag) {
    bool isPass = false;
    float fLT   = DEBUG.fLongTail;
    if (fLT <= 0)
        return false;
    int nMostIter = TrainParams().nMostIter, T_iter = fLT < 1.0 ? nMostIter * fLT : (int)fLT;  //
    if (GetITER() >= T_iter) {
        if (GetITER() == T_iter) {
            _INFO("[Longtail] iter=%d(%g)\n", T_iter, fLT);
        }
        return true;
    }
    return isPass;
}

bool Optimizer::isSpike(int flag) { return false; }

int RAW_update(std::vector<hGTensor>& tensors, Optimizer* hOPT, float& grad_norm, int alg, int flag);

void Optimizer::CheckExitSearch(int t, int flag) {
    bool isExit = false;
    if (DEBUG.N_mostiter > 0 && t > DEBUG.N_mostiter) {  // only for debug
        isExit = true;
    }
    if (isExit) {
        trainInfos().SaveToCSV("_info_.csv");
        // release more resource here
        K_EXIT(KOIFISH_EXIT_DEBUG);
    }
}
/*
    10/04/2025  行走于天地之间，所见大美，皆为人心
 */
Optimizer::RESULT Optimizer::Search(void* ctx, hGensor loss_, hGensor target_, CLI_params& config) {
    hEDS = _fish->hEDS;
    assert(hEDS != nullptr);
    auto train_params = TrainParams();
    // auto fish_in      = _fish->config.ckp_in[0];_fish->isLoadCheckpoint ? fish_in.sDir.c_str() : ""

    last_time                = GST_ms();
    Optimizer::RESULT result = DID_NOT_CONVERGE;
    RLS_BP* hRLS             = _fish->hEDS->GetScheduler<RLS_BP>();
    bool cancel = false, isWarmup = false;
    string suf, pref;
    Dump(0x0);
    _INFO("\t%s@<%s> %s device=[%s] \n", __func__, _fish->hBackTG->name.c_str(), "", hEDS->__repr__(suf, pref, 0).c_str());
    _INFO("\t Accumulation=%d AdaptiveSched=%d GRAP=%p rZMUV=%g rLARS=%g \n", nGradAccum, (int)isAdaptiveSched, grad, config.ZMUV_ratio, config.lars_ratio);
    // tpGD=SGD_HYBRID;    //ADAMw    ADAM_S  SGD_v    SGD_HYBRID        SGD_blk_v
    _INFO("\tDECENT=%d(%s) SIGN=%d tpFuseCu=%d filter=%d\n\n", tpGD, GD_NAME[tpGD].c_str(), tpSign, tpFuseCu, _fish->config.filter_tmp_grad.size());
    DEBUG.Dump(0);

    float a = 0, grad_norm = 0;
    if (_fish->isLoadCheckpoint) {
        Evaluate(1);
    }
    if (isWarmup && !BatchGrad(0, a, 0x0))  //  warmup
        return CANCEL;

    if (just_initialized) {
        n_no_improvement = 0;
        just_initialized = false;
    }

    int iter0 = 0, t;
    for (t = 0; t < train_params.nMostIter; ++t) {
        NvtxRange range("step", t);
        CheckExitSearch(t);
        if (_fish->isModel({NLP_QWEN2, NLP_QWEN3})) {
            // g_dump_level = -1;
        }
        _fish->BeforeNextStep(t, 0x0);
        if (t == train_params.nMostIter - 1) {
            if (train_loader != nullptr) {
                train_loader->isLast = true;
                isDumpOnce           = true;
            }
        }
        iter = iter0 + t + 1;
        SUM::Reset("time");
        _fish->SetPhase(LIFE_PHASE::P_TRAIN);
        if (!BatchGrad(iter, a, 0x0))
            return CANCEL;
        // if(t==1)    K_EXIT(KOIFISH_EXIT_DEBUG);

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
        for (auto ck : _fish->config.ckp_out) {
            if (ck.needSave(t))
                _fish->SaveTrain(ck);
        }
        if (t % 100 == 0)
            trainInfos().SaveToCSV("_info_.csv");

        Evaluate();

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

    OutCLS* cls       = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    TokenEmbed* embed = _fish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    if (_fish->phase != LIFE_PHASE::P_GENERATE && _fish->phase != LIFE_PHASE::P_PREFILL)
        embed->hBatch = hLoader->hBatch;

    isBackward = false;
    _fish->ForwardOnRLS(iter, 0x0);
    if (_fish->phase == LIFE_PHASE::P_GENERATE || _fish->phase == LIFE_PHASE::P_PREFILL) {
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

bool Optimizer::Evaluate(int type, int flag) {
    OutCLS* cls = _fish->GetNeuron<OutCLS>("OutCLS", 0);

    int iter       = GetITER();
    float val_loss = 0;
    if (type == 1) {
        assert(_fish->isLoadCheckpoint);
        _INFO("[checkpoint] Evaluate the checkpoint of \"%s\"\n", "");  //_fish->config.fish_in.sDir.c_str()
    }
    for (auto vl : val_loaders) {
        if (type == 1 || vl->isEval(iter + 1)) {
            _fish->SetPhase(LIFE_PHASE::P_EVAL_);
            cls->hLoader = vl;
            // for(int i=0;i<10;i++)
            // val_loss = EvaluateSamps(vl, iter), a = val_loss;       //  0.272727281
            val_loss = vl->Evaluate(SAMPLEofSHARD, 0x0);
        }
    }
    // K_EXIT(KOIFISH_EXIT_DEBUG);

    // if (config.common.gpt_every > 0 && t % config.common.gpt_every == 0) {
    //     _fish->GenSentence(1);
    // }
    return true;
}

/*
    Multiple purpose(Need refactor!)
    1. Get loss on some evaluate set in training procee
    2. Get loss on some evaluate set in unit-testing
    3. Prefill stage of Inference
    4. Generation stage of Inference
*/
float Optimizer::EvaluateSamps(hSampLoader loader, int iter, int flag) {
    if (loader->num_batches == 0) {
        assert(0);
        return 0;
    }
    assert(loader->num_batches > 0);
    assert(0);  // Deprecated due to code refactor

    GST_TIC(tic);
    OutCLS* cls  = _fish->GetNeuron<OutCLS>("OutCLS", 0);
    cls->hLoader = loader;
    auto loss    = hLoss();
    double l2, delta_max = 0, delta_ = 0, a, mean_loss = 0, ee = 0;
    auto tokens_input = _fish->Input();
    int i, nB = 0, step = loader->StepOfEvaluate();
    size_t nz           = 0, j;
    TOKEN_ID tMost      = (TOKEN_ID)(_fish->nClass() - 1);
    hSAMP samp          = nullptr;
    const float* wLog   = nullptr;
    loader->next_sample = 0;  // fix this to keep same acc on each experiment
    for (i = 0; i < loader->num_batches; i += step) {
        if (tokens_input != nullptr &&
            (_fish->phase != LIFE_PHASE::P_PREFILL && _fish->phase != LIFE_PHASE::P_GENERATE)) {  // in some debug mode, tokens_input maybe null
            TIMING_ms(loader->UpdateBatch(i, _fish), SUM::tLoadData);
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
        a  = ((float*)hPreLogits()->data)[0];  //  -6.60046101     -4.3040733
        ee = loader->DecodeVerify(samp, tokens_input, _fish->preLogits);
        mean_loss += loss == nullptr ? 0 : ((float*)(loss->data))[0];
        break;
#endif
        if (_fish->wikis.size() > 0)  // too long
            break;
    }
    mean_loss /= nB;
    if (iter == -666) {  // hack
        return i;
    }
    SUM::tEval_1 = GST_TOC(tic);
    loader->UpdateStepInfos(mean_loss, nB);

    return mean_loss;
}

string StepInfos::STEP::Info(int flag) {
    char buffer[256] = "\0";

    return buffer;
}

float Optimizer::UpdateLossCurve(int flag) {
    TRAIN_CARD _params = TrainParams();
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
            SUM::TimeInfo(_fish->config.dumpSwitch.train_time);
            _TIME_INFO(" eta=", remaining_millis);
        }
        size_t tokens_processed = _fish->config.nTokensPerGrad();  //(size_t) * B * T * grad_accum_steps;
        float tokens_per_second = tokens_processed / millis_per_iter * 1000.0f;
        ema_tps                 = iter == 1 ? tokens_per_second : 0.95f * ema_tps + 0.05f * tokens_per_second;
        _INFO(" | %.1fK token/s | %s", ema_tps / 1000.0, _fish->DebugInfo().c_str());
        _INFO(" x=%d\n", SUM::nUpdateParam);
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
    // if (_fish->isLocalInfer)
    //     _fish->hCache = std::make_shared<KVCache>(_fish);
}

void Optimizer::BeforeTrain(hGensor tokens_, int flag) {
    first_iter = iter;

    auto& adam        = _fish->config.common.adam;  // TrainParams().
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

bool MUON_params_::isAdamW(void* hUserData, int flag) {
    GTensor* tensor = (GTensor*)(hUserData);
    if (!tensor->isWMAT())
        return true;
    if (G_Has_(tensor->name, {"embd", "embed"}))
        return true;

    int64_t m = tensor->ne[0], n = tensor->ne[1];
    // if(isStrMatch(tensor->name,{"model.blk.0.ffn_down.weight"})){ //  model.blk.11.ffn_down.weight
    //     return false;        //only for debug
    // }else
    //     return true;
    if (m >= n)  // m <= n
        return false;
    return true;
}

void OPT_Muon::BeforeTrain(hGensor tokens_input, int flag) {
    Optimizer::BeforeTrain(tokens_input, flag);
    int version        = 1;
    ADAM_params_* adam = &(_fish->config.common.adam);
    MUON_params_* muon = &(_fish->config.common.muon);
    muon->ldAB         = 0;
    string sNames;
    for (auto t : opt_ps) {
        if (muon->isAdamW(t.get()))
            continue;
        tMuons.push_back(t);
        nmParams += t->size();
        sNames += t->name + string(" ");
        int n      = min(t->ne[0], t->ne[1]);
        muon->ldAB = max(muon->ldAB, n);
    }
    if (version == 0)
        hPipe = std::make_shared<PIPE_Adamw<floatX, floatMV>>(this, flag, adam->alpha, adam->beta1, adam->beta2, adam->eps, adam->decay);  // only for debug
    else
        hPipe = std::make_shared<PIPE_Muon<floatX, floatMV>>(this, flag, adam->alpha, adam->beta1, adam->beta2, adam->eps, adam->decay);
    _INFO("[Muon] version=%d filter=\"%s\"", version, sNames.c_str());
}

size_t TGraph::Prepare4Train(void* ctx_, GD_METHOD tpGD, int flag) {
    hOptimizer hOpt = hFish->hOPT;
    assert(hOpt != nullptr);
    size_t nP = 0, nz = 0, nzAll = 0, id = 0, n1 = hFish->gensors.size();
    for (auto& gi : hFish->gensors.infos) {
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
    NLP_AutoRegressive* dolphin = dynamic_cast<NLP_AutoRegressive*>(_fish);
    if (nParams > 0) {
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

    if (_fish->isTrain())
        _tmp = new float[nParams]();
    // val_loader->Init(dolphin,"Eval",false);            //val_loader->Prepare(this);
}

bool Optimizer::PrepareData(CLI_params& config, int flag) {
    GST_TIC(tic);

    bool isLoadOK = false;
    string root = _fish->tsTrain->serial_root, spTrain = root + ".train", spEval = root + ".eval";
    if (root.empty()) {
        train_loader->Shuffle();
        assert(train_loader->shuffle_sample_count > 0);
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

void OPT_Muon::Prepare(size_t nx_, int flag) { Optimizer::Prepare(nx_, flag); }

/*
    https://github.com/KellerJordan/muon
    https://x.com/Kimi_Moonshot/status/1897929976948965870
*/
OPT_Muon::OPT_Muon(NLP_AutoRegressive* g_, CLI_params& params_, int flag) : Optimizer(g_, params_, flag) {
    ADAM_params_* adam        = &(_fish->config.common.adam);
    adam->clip_alg            = 1;  // clip_alg=0 little better
    adam->gclip               = adam->gclip / _fish->config.nLayer();
    _fish->config.Fuse_Normal = 0;  //
    if (g_->isTrain()) {
        trainInfos().Init(this);
    } else {  // may different
        _fish->config.common.remater_ffn = 1;
        // _fish->config.common.remater_qkv = 1;
    }
    for (auto loader : val_loaders) {
        loader->stepis.Init(this);
    }
}

OPT_Adam::OPT_Adam(NLP_AutoRegressive* g_, CLI_params& params_, int flag) : Optimizer(g_, params_, flag) {
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
    for (auto loader : val_loaders) {
        loader->stepis.Init(this);
    }
    hPipe = std::make_shared<PIPE_Adamw<floatX, floatMV>>(this, flag, adam->alpha, adam->beta1, adam->beta2, adam->eps, adam->decay);
    // hPipe = std::make_shared<PIPE_Muon<floatX, floatMV>>(this, flag, adam->alpha, adam->beta1, adam->beta2, adam->eps, adam->decay);    //only for debug
    // sched              = 1.0f;
}

void Optimizer::Dump(int typ) {
    if (NOT_DUMP(1))
        return;
    _fish->memBuffer->Dump(0x0);

    auto train_params = TrainParams();
    _INFO("======== nEopch=%d most_iter=%d\n", train_params.n_epochs, train_params.nMostIter);  //,train_params.nEpochIter
    fflush(stdout);
    RLS_BP* hRLS = _fish->hEDS->GetScheduler<RLS_BP>();
    hRLS->Dump(typ);
    _INFO("\tType weight=%s activation=%s", cNameOf(_fish->config.model.tpWeight), cNameOf(_fish->config.model.tpActivation));
    fflush(stdout);

    size_t sz         = 0x0;
    const char* title = "OPT";  //__func__
    if (_ctx != nullptr)
        _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, sz, (float)sz / (1024.0f * 1024.0f));
    _INFO("%s: iter = %d\n", title, iter);

    if (typ == 1) {
        _INFO("%s: SAMP_HASH=%llu total train_iterations=%llu train_samples=%llu train_tokens=%llu completed_epochs=%llu\n", title, shuffle_samples_hash,
              train_its, train_samples, train_tokens, train_epochs);
    }

    // string path = _fish->config.checkpoint.out;
    // if (path.empty()) {
    //     _INFO("[Save] path is empty! To save model, please set the key of \"checkpoint-out\" in json config file(\"%s\").\n", _fish->config.jsPath.c_str());
    // } else {
    //     if (VERIFY_DIR_EXIST(path, true))
    //         _INFO("[Save] path=\"%s\", save_every=%d\n", path.c_str(), train_params.save_every);
    //     else {
    //         _INFO("[Save] Invalid path@\"%s\"!\n", path.c_str());
    //     }
    // }
    if (!HIERARCH_LoRA::sNeurons.empty()) {  // HIERARCH_LoRA::tpLORA == LORA_ADAPT_W::AB ? "AB" : "W_AB",
        _INFO("[H_LORA] neurons={%s}\n", HIERARCH_LoRA::sNeurons.c_str());
    }

    // if(NOT_DUMP())  return;
    if (train_loader != nullptr)
        train_loader->Dump(typ);
    for (auto vl : val_loaders) {
        vl->Dump(typ);
    }

    if (hLR != nullptr)
        hLR->Dump(typ);
    int nG0 = 0;
    for (auto t : opt_ps) {
        if (BIT_TEST(t->flags, GTensor::F_TMP_GRAD)) {
            nG0++;
        }
    }
    _INFO("\tnParams = %ld(%.6gM, nT=%ld nG0=%d)\n", nParams, nParams / 1.0e6, opt_ps.size(), nG0);
    fflush(stdout);
}

void OPT_Adam::Dump(int typ) {
    Optimizer::Dump(typ);
    adam->Dump(typ);
    _INFO("[OPT_Adam]\tsRESI=%g s_rounding=%d alloc_w=%d remater[ffn=%d ]\n", TrainParams().residual_scale, TrainParams().opt_alloc_weight,
          TrainParams().opt_alloc_weight, TrainParams().remater_ffn);

    fflush(stdout);
}

void OPT_Muon::Dump(int typ) {
    Optimizer::Dump(typ);
    auto muon = TrainParams().muon;

    _INFO("[OPT_Muon]\tnT=%ld muon_params=%ld(%.3g)\n", tMuons.size(), nmParams, nmParams * 1.0 / nParams);
    muon.Dump(0x0);
    fflush(stdout);
}