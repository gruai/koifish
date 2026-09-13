/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Function of HUA(化)
 *  \author Yingshi Chen
 */

#include "Transition.hpp"

#include "../Manifold/Fish.hpp"
#include "../TokenSet/Batch.hpp"

/**
 * signal_noise_ratio based token planner
 */
TPLAN_PUMA::TPLAN_PUMA(const std::string& nam_, Fish* hFish, std::shared_ptr<BATCH_INPUT> _hB, int flag)
    : TOKEN_Planner(nam_ + "_puma", hFish->config.chat_sampler, -1, _hB->ldT, flag) {
    int seed    = hFish->config.XI.mask_seed;
    hMaskRander = std::make_shared<GRanderTorch>(seed);
    fNoise      = new float[_hB->nMostSample * samp_len]();
}
bool TPLAN_PUMA::Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag) {
    NOT_IMPLEMENTED;
    return true;
};

bool TPLAN_SNR::Init4Prefill(int prefill, int type, int flag) {
    arrGroup.clear();

    // seq_len     = samp_len - prefill;
    int cur_len = samp_len - prefill;

    std::vector<int> ids0, ids, status;
    status.resize(cur_len);
    bool isFixID = BIT_TEST(flag, F_ABSOLUTE_ID);
    for (int i = 0; i < cur_len; i++) ids0.push_back(i);
    for (int step = 0; step < nMostStep; step++) {
        // float scale = RelativeRate(step), s;
        float s = arrRatio[step];  // 1.0 - scale;
        assert(ids0.size() == cur_len);
        lState group;
        for (int i = 0; i < cur_len; i++) {
            float a     = hPickRander->NextFloat_01();
            bool isPick = a < s || step == nMostStep - 1;
            int id      = isFixID ? ids0[i] : i;
            if (isPick) {
                group.push_back(id);
                assert(status[ids0[i]] == 0);
                status[ids0[i]] = 1;
            } else {
                ids.push_back(ids0[i]);
            }
        }
        if (!group.empty()) {
            arrGroup.push_back(group);
            cur_len -= group.size();
        }
        ids0 = ids;
        ids.clear();
    }
    assert(cur_len == 0);
    if (isFixID) {
    }
    int nGroup = arrGroup.size();
    return nGroup > 0;
}

bool TPLAN_Dilate::Init4Prefill(int prefill, int type, int flag) {
    arrGroup.clear();
    int basis = 2, i, j, at = basis, st, B = samp_len - prefill, nz = 0;
    int* mask = new int[B]();
    if (samp_len == 633)
        DEBUG_HERE;
    int mostStep = (int)ceil(log(samp_len - prefill) / log(basis));
    for (i = 0; i < mostStep; i++) {
        st = (int)floor(B * 1.0 / at);
        if (st == 0)
            break;
        lState group;
        for (j = 0; j < B; j++) {
            if (mask[j] != 0)
                continue;
            if (j % st == 0) {
                mask[j] = 1;
                group.push_back(j);
            }
        }
        if (group.empty())
            break;
        arrGroup.push_back(group);
        nz += group.size();
        at *= basis;
    }
    assert(nz == B);
    delete[] mask;

    mostStep = arrGroup.size();
    return mostStep > 0;
}

bool TPLAN_SNR::Transition4Batch(int iter, std::shared_ptr<BATCH_INPUT> hBatch, int flag) {
    if (DEBUG.verHuaSNR < 0)  // hack random noiser is better than path-based noiser, so strange!
        return TOKEN_Planner::Transition4Batch(iter, hBatch, flag);
    assert(hPickRander != nullptr);

    hBatch->huaers.clear();
    int mostPrefill = samp_len / 10, minPrefill = std::max(6, samp_len / 100);
    size_t nSamp = hBatch->nMostSample;
    while (hBatch->huaers.size() < nSamp) {
        int prefill = minPrefill + hMaskRander->RandInt32() % (mostPrefill - minPrefill);
        InitAllTrainsitions(prefill, prefill + 1, flag | F_ABSOLUTE_ID);
        int nKeep = std::min(nSamp - hBatch->huaers.size(), huaers.size());
        hBatch->huaers.insert(hBatch->huaers.begin(), huaers.begin(), huaers.begin() + nKeep);
    }
    assert(hBatch->huaers.size() == nSamp);

    std::mt19937 g(20260903 + iter);  // hPickRander = std::make_shared<GRanderTorch>(803);
    std::shuffle(hBatch->huaers.begin(), hBatch->huaers.end(), g);
    int nzPrefill = 0, nzDenoise = 0;
    for (auto hua : hBatch->huaers) {
        nzPrefill += hua->nPrefill, nzDenoise += hua->nDenoise;
        // hua->Dump();
    }
    double rLos = nzDenoise * 1.0 / samp_len / nSamp, rFil = nzPrefill * 1.0 / samp_len / nSamp;
    return true;
};

bool TOKEN_Planner::InitAllTrainsitions(int minPrefill, int mostPrefill, int flag) {
    // int samp_len = seq_len;
    int nzPrefill = 0, nzDenoise = 0;
    std::vector<HUA_STATE> current;
    current.resize(samp_len);
    for (int i = 0; i < samp_len; i++) current[i] = HUA_STATE::MASK;
    // so strange, remove mask would make train stable
    // for (int i = 0; i < samp_len; i++) current[i] = HUA_STATE::TOKEN;
    // for (int i = 0; i < samp_len; i++) current[i] = HUA_STATE::PAD;
    bool isOnly2 = false;  //  only for debug
    huaers.clear();
    for (int prefill = minPrefill; prefill < mostPrefill; prefill++) {
        arrGroup.clear();
        int cur_len = samp_len - prefill;
        for (int i = 0; i < prefill; i++) current[i] = HUA_STATE::TOKEN;
        Init4Prefill(prefill, 0x0, flag);
        for (auto group : arrGroup) {
            for (auto pos : group) {
                current[prefill + pos] = HUA_STATE::DENOISE;
            }
            auto hT = std::make_shared<HUA_Token>(current, prefill);
            assert(hT->flow.size() == samp_len);
            huaers.push_back(hT);
            if (isOnly2) {
                for (int k = 0; k < arrGroup.size() - 1; k++) {  // only for debug
                    huaers.push_back(hT);
                }
                break;
            }
            for (auto pos : group) {
                current[prefill + pos] = HUA_STATE::TOKEN;
            }
        }
        for (int i = 0; i < samp_len; i++) {
            if (!isOnly2)
                assert(current[i] == HUA_STATE::TOKEN || current[i] == HUA_STATE::DENOISE);
            current[i] = HUA_STATE::MASK;
        }
    }
    for (auto hT : huaers) {
        nzPrefill += hT->nPrefill, nzDenoise += hT->nDenoise;
    }
    int N = huaers.size();
    if (mostPrefill - minPrefill > 1)
        _INFO("[Transition] N=%d seqlen=%d nPrefill=%g nDenoise=%g\n", N, samp_len, nzPrefill * 1.0 / N, nzDenoise * 1.0 / N);
    return true;
};