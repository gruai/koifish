/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */

#include "Prober.hpp"

#include "Fish.hpp"
#include "Neuron.hpp"

INSPECT::INSPECT(GeNeuron* hN_, const std::string& info, int typ, int flag) : hN(hN_) {
    if (hN != nullptr)
        hN->OnDebug(info, typ, flag);
}

INSPECT::~INSPECT() {
    if (hN != nullptr)
        hN->ExitDebug();
}

#include "../Manifold/HotPicker.hpp"

bool PROBER::Inject(std::shared_ptr<Fish> hFish_, int flag) {
    hFish   = hFish_;
    neurons = hFish->GetNeurons(sNeuron, nid_0, nid_1);
    assert(!neurons.empty());
    int ldH = hFish_->config.nEmbed();
    switch (tpFeat) {
        case PARTIAL_PRELOGITS:
            ldH = hFish_->nClass();
            break;
        default:
            break;
    }
    hPicker = std::make_shared<FeatPicker>(name, ldH);
    for (auto nn : neurons) {
        nn->probers.push_back(this);
    }

    return true;
}

void PROBER::Dump(int typ, int flag) const {
    _INFO("[Prober_%s] nInject=%ld @\"%s\" of \"%s\"\n", name.c_str(), neurons.size(), sNeuron.c_str(), hFish->Name().c_str());
}

PROBER::PROBER(CSTRING& nam_, CSTRING& sN, int n0, int n1, FEAT_TYPE tpF, int flag) : name(nam_), sNeuron(sN), nid_0(n0), nid_1(n1), tpFeat(tpF) {
    if (tpFeat == PARTIAL_PRELOGITS) {
        stage = IN_FORWARD;
    }
}

bool PROBER::CollectFeats(GeNeuron* nn, hGTensor tnsor, int flag) {
    hGTensor fTensor = tnsor;
    if (fTensor == nullptr)
        fTensor = nn->out;
    assert(hPicker != nullptr);
    hBATCH hBatch     = hFish->curBatch(0x0);
    hGTensor raw_feat = hPicker->raw_feat;
    int sampInBatch = fTensor->ne[2], ldT = fTensor->ne[1], ldH = fTensor->ne[0], i, samp, nSamp = raw_feat->ne[0];
    int pid           = hPicker->nPickSamp;
    floatX *host_data = new floatX[ldH], *fv;
    assert(ldH <= raw_feat->ne[1]);
    for (samp = 0; samp < sampInBatch; samp++, pid++, hPicker->nPickSamp++) {
        hSAMP hSamp = hBatch->curSamps[samp];
        assert(pid <= nSamp);
        size_t off = fTensor->Offset(0, hSamp->nPrefill - 1, 0, 0), off_1 = fTensor->size() - hSamp->last_pad_len * ldH;
        D2H(fTensor->data + off, host_data, sizeof(floatX) * ldH);  // last hidden feat, nPrefill=176
        // PrintTensor<floatX>("host_data", host_data, false, 1, ldH, 1, 1, -1);
        for (i = 0; i < hPicker->dimY; i++) {  // i_th target
            float a = hBatch->user_answer[samp * hPicker->dimY + i];
            assert(a == 1.0f || a == 0.0f);
            raw_feat->Set(pid, ldH + i, 0, 0, a);
        }
        for (i = 0; i < ldH; i++) {
            float a = T2Float(host_data + i);
            raw_feat->Set(pid, i, 0, 0, a);
        }
    }
    // PrintTensor<float>("Target", (float*)(raw_feat->data) + ldH * nSamp, false, nSamp, 1, 1, 1, -1);
    // PrintTensor<float>("feature", (float*)(raw_feat->data), false, nSamp, ldH, 1, 1, -1);
    delete[] host_data;
    return true;
}