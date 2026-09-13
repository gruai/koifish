/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Function of "Langevin dynamics"
 *  \author Yingshi Chen
 */

#include "Langevin.hpp"

#include "../Manifold/Fish.hpp"
#include "../TokenSet/Batch.hpp"

hHuaPLAN TOKEN_Planner::MakeInstance(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> hBatch, int flag) {
    hHuaPLAN hPlan = nullptr;
    switch (hFish->config.chat_sampler.tpZhuomo) {
        case CHAT_SAMPLER::MD_DILATE:
            hPlan = std::make_shared<TPLAN_Dilate>(nam_, hFish, hBatch);
            break;
        case CHAT_SAMPLER::MD_SNR:
            hPlan = std::make_shared<TPLAN_SNR>(nam_, hFish, hBatch);
            break;
        case CHAT_SAMPLER::MD_PUMA:
            hPlan = std::make_shared<TPLAN_PUMA>(nam_, hFish, hBatch);
            break;
        case CHAT_SAMPLER::TEMPERATURE:
            NOT_IMPLEMENTED;
            break;
        default:
            hPlan = std::make_shared<TPLAN_SNR>(nam_, hFish, hBatch);
            break;
    }
    hPlan->Dump();
    return hPlan;
}

TOKEN_Planner::TOKEN_Planner(const std::string& nam_, CHAT_SAMPLER& user_params, int nMostStep_, int len_, int flag)
    : name(nam_), _params(user_params), nMostStep(nMostStep_), context_len(len_) {
    // policy = LINEAR_DECAY;
    samp_len = context_len;

    hPickRander = std::make_shared<GRanderTorch>(803);
    switch (_params.tpZhuomo) {
        case CHAT_SAMPLER::MD_DILATE:

            break;
        default:
            break;
    }

    t_base     = 1.0;
    t_previous = t_base;
    t_final    = 0.001;
}

void TOKEN_Planner::Init(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag) { t_previous = t_base; }

float TOKEN_Planner::RelativeRate(int64_t step, int flag) {
    // assert()
    float delta = (t_final - t_base) / nMostStep, t_cur = t_base + (step + 1) * delta;
    float r    = t_cur / t_previous;
    t_previous = t_cur;
    return r;
}
/*
std::vector<int> TOKEN_Planner::PickGroup(int64_t step, int nSamp, int x, int flag) {
    std::vector<int> picks;
    // hPickRander->Init(20260713);  // only for debug
    float T_confi = 0.f;
    switch (_params.tpZhuomo) {
        case CHAT_SAMPLER::MD_DILATE:
            picks = arrGroup[step];
            break;
        default: {
            float scale = RelativeRate(step);
            T_confi     = 1.0 - scale;
            for (int i = 0; i < nSamp; i++) {
                float a = hPickRander->NextFloat_01();
                if (a < T_confi)
                    picks.push_back(i);
            }
        } break;
    }
    if (picks.size() == nSamp) {
        DEBUG_HERE;
    }
    return picks;
}*/

void TOKEN_Planner::Dump(int flag) {
    _INFO("[Huaer]_\"%s\" N=%d(iter=%d)", name.c_str(), samp_len, nMostStep);
    if (!arrRatio.empty()) {
        _INFO("\tratio=[%g,%g]", arrRatio[0], arrRatio[arrRatio.size() - 1]);
    }
    _INFO("\n");
}
void TPLAN_Dilate::Dump(int flag) { _INFO("[Huaer]_\"%s\" Dilate N=%d(iter=%d)\n", name.c_str(), samp_len, nMostStep); }

TOKEN_Planner::~TOKEN_Planner() { FREE_a(fNoise); }

HUA_Token::HUA_Token(std::vector<HUA_STATE>& flow_, int nPre, int flag) : HUAER(flow_, flag) {
    nPrefill = nPre;
    nDenoise = 0, nMask = 0;
    for (auto ti : flow) {
        if (ti == MASK)
            nMask++;
        if (ti == DENOISE)
            nDenoise++;
    }
    assert(nDenoise >= 1);
}

void HUA_Token::Init(int seq_len, int flag) {
    flow.resize(seq_len);
    for (int i = 0; i < seq_len; i++) flow[i] = HUA_STATE::X;
}
/**
 * 1. MDLM forward process defines noise as a function of the ratio between timesteps

TPLAN_random::TPLAN_random(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> _hB, int flag)
    : TOKEN_Planner(nam_, hFish->config.chat_sampler, -1, _hB->ldT, flag) {
    // hBatch      = _hB;
    int seed    = hFish->config.XI.mask_seed;
    hMaskRander = std::make_shared<GRanderTorch>(seed);
    fNoise      = new float[_hB->nMostSample * seq_len]();
}

*/

/**
 * signal_noise_ratio based token planner
 */
TPLAN_SNR::TPLAN_SNR(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> _hB, int flag)
    : TOKEN_Planner(nam_ + "_SNR", hFish->config.chat_sampler, -1, _hB->ldT, flag) {
    int seed    = hFish->config.XI.mask_seed;
    hMaskRander = std::make_shared<GRanderTorch>(seed);
    fNoise      = new float[_hB->nMostSample * samp_len]();
    nMostStep   = hFish->config.chat_sampler.most_hua;
    for (int step = 0; step < nMostStep; step++) {
        float scale = RelativeRate(step);
        arrRatio.push_back(1.0 - scale);
    }
    if(DEBUG.verHuaSNR<0){
        name += "_random noise on training";
    }
    assert(nMostStep >= 1);
}

bool TOKEN_Planner::Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag) {
    assert(hMaskRander != nullptr && hBatch != nullptr);
    assert(hBatch->ldT <= samp_len);

    huaers.clear();
    std::vector<float> T_sample;
    hMaskRander->RandFloat(hBatch->nMostSample, T_sample);
    //  ensuring at least 5% context remains, and at least 5% is masked
    for (int i = 0; i < T_sample.size(); i++) {
        T_sample[i] = 0.05 + 0.9 * T_sample[i];
    }
    hMaskRander->RandNoise_MN(hBatch->nMostSample, hBatch->ldT, T_sample, fNoise);
    float* noise   = fNoise;
    int minPrefill = 1;
    for (int samp = 0; samp < hBatch->nMostSample; samp++) {
        std::vector<HUA_STATE> current;
        current.resize(hBatch->ldT);
        for (int j = 0; j < hBatch->ldT; j++, noise++) {
            if (j < minPrefill) {
                current[j] = HUA_STATE::TOKEN;
            } else
                current[j] = (*noise) > 0 ? HUA_STATE::DENOISE : HUA_STATE::TOKEN;
        }
        auto hT = std::make_shared<HUA_Token>(current, -1);
        huaers.push_back(hT);
    }
    hBatch->huaers = huaers;
    return true;
};

// Sequence Plan(from left to right) on signal_noise_ratio
TPLAN_SNR_sequence::TPLAN_SNR_sequence(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> _hB, int flag) : TPLAN_SNR(nam_, hFish, _hB, flag) {}

TPLAN_Dilate::TPLAN_Dilate(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> _hB, int flag)
    : TOKEN_Planner(nam_ + "_dialate", hFish->config.chat_sampler, -1, _hB->ldT, flag) {
    // hBatch   = _hB;
    int seed = hFish->config.XI.mask_seed;
    nMostStep = -1;
}

bool TPLAN_Dilate::Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag) {
    assert(hPickRander != nullptr);
    if (huaers.empty()) {
        int mostPrefill = samp_len / 10, minPrefill = std::max(6, samp_len / 100);
        InitAllTrainsitions(minPrefill, mostPrefill);
    }

    size_t n = hBatch->nMostSample;
    // assert(n < huaers.size());
    if (n >= huaers.size()) {
        if (iter == 0)
            _WARN("[Plan] n(%d)>=huaers(%ld), oversampling\n", n, huaers.size());
        for (int i = 0; i < n; i++) {
            int no = i % huaers.size();
            hBatch->huaers.push_back(huaers[no]);
        }
        std::mt19937 g(20260903 + iter);  // hPickRander = std::make_shared<GRanderTorch>(803);
        std::shuffle(huaers.begin(), huaers.end(), g);
    } else {
        auto picks = hPickRander->kSampleInN(n, huaers.size());
        for (int i = 0; i < n; i++) {
            int no = picks[i];
            hBatch->huaers.push_back(huaers[no]);
        }
    }
    return true;
}

bool TPLAN_Dilate::Transition4Samp(SAMP* hSamp, int flag) {
    huaers.clear();
    samp_len = hSamp->len;           // 601
    if (hSamp->len > context_len) {  //
        samp_len = context_len;
    }
    int mostPrefill = samp_len / 10, minPrefill = std::max(6, samp_len / 100), prefill = 0;
    prefill = (mostPrefill + minPrefill) / 2;
    InitAllTrainsitions(prefill, prefill + 1);
    assert(huaers.size() > 0);
    auto picks = hPickRander->kSampleInN(1, huaers.size());
    // std::random_device rd;
    // std::mt19937 g(rd());
    // std::shuffle(huaers.begin(), huaers.end(), g);
    samp_flow = huaers[picks[0]]->flow;
    for (int i = samp_len; i < context_len; i++) {  // pad
        samp_flow.push_back(HUA_STATE::PAD);
    }
    assert(samp_flow.size() == context_len);
    return true;
}
