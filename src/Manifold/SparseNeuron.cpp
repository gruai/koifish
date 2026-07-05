/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Neurons with sparsing activation/weight
 *  \author Yingshi Chen
 */
#include <memory>

#include "Fish.hpp"
#include "HotPicker.hpp"
#include "Optimizer.hpp"
#ifdef __USE_GBDT__
#include "../GBDT/data_fold/Histogram.hpp"
using namespace Grusoft;
#endif
#include "../Utils/GST_rander.hpp"
#include "../lenda/kernel/SVD.hpp"

CS_Picker::CS_Picker(hFISH hFish, int flag) {
    int nEmbed = hFish->config.nEmbed();  //
    dim        = hFish->config.n_ff();
    hot        = new int[dim * 2]();
    for (int i = 0; i < dim; i++) hot[i] = 1;
    dTemp = new float[dim + nEmbed * dim];

    T_hot  = 0.2;
    T_zero = 1.0e-3;
}

//  Picker should much fast than dot!
double CS_Picker::tPick = 0.0;

int CS_Picker::Update(int level, float* hb, int flag) {
    return -1;
    double t0 = GST_us();
    if (level > 0)
        return nLastHot;
    int nz = 0, i = 0, nHot = std::max((int)(dim * T_hot), 16), nEx = nHot, id;
    nEx = ((nHot * 2) / 16) * 16 - nHot;
    // nEx = ((nHot)/16)*16-nHot;
    float *tmp = dTemp, prev = FLT_MAX, a;
    int* map = hot + dim;
    // for(i=0; i<dim; i++)    hot[i]=1;   return dim;

    for (i = 0; i < dim; i++) {
        if (hot[i] == 0) {
            // assert(hb[i]==0);	continue;
        }
        hot[i] = 0;
        if (fabs(hb[i]) < T_zero)
            continue;
        // if(hb[i]<T_zero)		continue;
        map[nz]   = i;
        tmp[nz++] = fabs(hb[i]);
    }
    if (nz < nHot)
        return -1;
#ifdef __USE_GBDT__
    vector<tpSAMP_ID> idx;
    sort_indexes(nz, tmp, idx);
    for (i = 0; i < nHot; i++) {
        id      = map[idx[nz - 1 - i]];
        hot[id] = 1;
        a       = fabs(hb[id]);
        assert(prev >= a);
        prev = a;
    }
    i = 0;
    while (i < nEx) {
        id = rand() % dim;
        if (hot[id] == 0) {
            hot[id] = 1;
            nHot++;
            i++;
        }
    }
    assert(nHot % 16 == 0);

    nLastHot = nHot;
    if (isMerge) {
        for (nHot = 0, i = 0; i < dim; i++) {
            if (hot[i] == 0)
                continue;
            hot[nHot++] = i;
        }
        assert(nHot == nLastHot);
    }
#endif
    tPick += GST_us() - t0;
    return nHot;
}

HotPicker::HotPicker(SparseNeuron* n, int flag) {
    name = n->name;
    // config.num_trees = 256;
}

string HotPicker::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "sparse_%s", "GBDT");
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
}

int HotPicker::Predict(int nPoint, floatI* data, int* hot, int flag) { return 0x0; }

bool HotPicker::SerialModel(const std::string& sPath, bool isSave, int flag) { return false; }

int HotPicker::Train(int flag) {
#ifdef __USE_GBDT__
    string title    = name + "_GBDT";
    ExploreDA* edaX = new ExploreDA(config, title, flag);
    hTrainData      = std::make_shared<FeatsOnFold>(config, edaX, title, flag);  //  from X,Y
    size_t nSamp_   = arrX.size();
    hTrainData->InitMost(nSamp_);
    if (hTrainData == nullptr)
        return -1;

    int nTree = config.num_trees;
    hGBRT     = std::make_shared<GBRT>(hTrainData.get(), nullptr, 0.333, BoostingForest::CLASIFY, nTree);
    hGBRT->Train("", 0, 0);
    SerialModel("", true);
#endif
    return 0x0;
}
int HotPicker::Eval(int flag) { return 0x0; }

SparseNeuron::SparseNeuron(const std::string& key_, JSON::const_iterator jit, Fish* hG_, int flag) : GeNeuron(key_, jit, hG_, flag) {
    if (BIT_TEST(flag, F_HOTPICK)) {
        isSparse = true;
    }
    if (isSparse) {
        method  = hG_->config.model.sparse.method;
        hPicker = std::make_shared<HotPicker>(this);
    }
}

SparseNeuron::SparseNeuron(const std::string& key_, Fish* hG_, int flag) : GeNeuron(key_, hG_, flag) {
    if (BIT_TEST(flag, F_HOTPICK)) {
        isSparse = true;
    }
    if (isSparse) {
        method  = hG_->config.model.sparse.method;
        hPicker = std::make_shared<HotPicker>(this);
    }
}

void SparseNeuron::SetEmbed(TokenEmbed* embd_, int type, int flag) {
    assert(embd_ != nullptr);
    subw      = embd_;
    samp_type = type;
    if (samp_type == 0) {
        w->SetRefer(embd_->w);
    } else {
        w->SetRefer(embd_->wInv);
    }
}

// TODO: Weighted sampling
void SparseNeuron::UpdateSamps(int seed, int flag) {
    assert(hSamps != nullptr);
    float* weight = nullptr;  // TODO: Weighted sampling
    int nVocab = hFish->nClass(), nSample = hSamps->size();
    std::vector<int> samps;
    Grusoft::GRander rander(seed);
    if (1) {  // nearly same
        hSampLoader sloader = hFish->GetOptimizer()->train_loader;
        assert(sloader != nullptr);
        sloader->PickSomeTokens(rander, nSample, samps);
    } else {
        samps = rander.kSampleInN(nSample, nVocab);
    }
    assert(samps.size() == nSample);
    assert(0);  //    hSamps->SerialGP(samps.data(), nullptr, sizeof(int) * nSample, false);
}

/*
    EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation
*/
bool SparseNeuron::InitSVD(int flag) {
    assert(hSVD == nullptr);
    int nIn = w->ne[1], nOut = w->ne[0], rank = min(256, min(nIn, nOut) / 5);
    rank = (int)(rank / 16) * 16;
    assert(rank >= 16);
    size_t i, nz = w->size();
    assert(nz == nIn * nOut);
    float *A = new float[nIn * nOut], tol_ = 0;  // 1.0e-3
    f8e5* src = (f8e5*)(w->data);
    for (i = 0; i < nz; i++) A[i] = T2Float(src + i);  // fp8_to_float(src[i]);
    hSVD = std::make_shared<LoSVD<float>>(name, A, nIn, nOut, rank, tol_, typNUMBER::F32);
    if (!hSVD->Build()) {
        compression = SKIP;
    } else {
        if (compression == SVD_a) {  // keep same graph
            float* approx = hSVD->Approx();
        } else {
        }
    }
    delete[] A;
    return true;
}

string HIERARCH_LorAB::sNeurons = "";

//  Forward: rhs = b*(a*inp)
HIERARCH_LorAB::HIERARCH_LorAB(SparseNeuron* neuron, hGensor w_, const std::string& title_, int r_, int flag)
    : wBase(w_), rank(r_), title(title_), spNeuron(neuron) {
    hFish = neuron->hFish;
    config = hFish->config.loAB;

    nInA = w_->ne[1], nOutB = w_->ne[0];
    if (rank == -1)  // only for debug
        rank = nInA;
    else {
        assert(rank * 10 < hFish->config.nEmbed());  // neuron->C   low rank
    }
    assert(w_->isWMAT() && rank > 0);
    //  nIn = shape[1], nOut = shape[0] @SLP::Build

    B = neuron->B, T = neuron->T;

    // string title = w_->name;
    //  spQ      = {q_dim, n_embd};
    a = GT(hFish, w_->type, {rank, nInA}, flag | GTensor::F_LORA_A, title + "_a");
    b = GT(hFish, w_->type, {nOutB, rank}, flag | GTensor::F_LORA_B, title + "_b");
    hFish->InitGensor(nullptr, "", a, true);
    hFish->InitGensor(nullptr, "", b, true);

    // tmp          = gBUFF->bt4c;
    // assert(m * rank <= tmp->size() && n * rank <= tmp->size());
    huTensor* ta   = dynamic_cast<huTensor*>(a.get());
    size_t szAlloc = ta->Alloc_1(&Ax, false, title + "_ax", sizeof(floatX) * B * T * rank);
    szAlloc += ta->Alloc_1(&Adelta, false, title + "_delta", sizeof(floatX) * B * T * rank);

    UpdateAdapt(flag);
    // _INFO("[H_LoAB]");
}
HIERARCH_LorAB::~HIERARCH_LorAB() {
    huTensor* ta = dynamic_cast<huTensor*>(a.get());
    ta->Free_1(&Ax, "");
    ta->Free_1(&Adelta, "");
}

void HIERARCH_LorAB::UpdateAdapt(int flag) {
    /*switch (spNeuron->tpLORW) {
        case LoAB_CARD::W0:  //  x = W*x
            break;
        case LoAB_CARD::AB:  //  x = (BA)*x = B*(Ax)
            // beta_F        = 0.0f;
            isAccumuDelta = false;
            break;
        case LoAB_CARD::W_AB:  //  x = (W+AB)*x
            // beta_F        = 1.0f;
            isAccumuDelta = true;
            break;
        case LoAB_CARD::SHADOW_AB:  //  Foreward:    x = (W+AB)*x    Backward: delta'=>(AB)delta
            // beta_F        = 1.0f;
            isAccumuDelta = false;
            break;
        default:
            assert(0);
            break;
    }*/
}
/*
    Since low rank, no need to quantize/ramater/guoke...
*/
bool SparseNeuron::InitLoRA(LoAB_CARD::typW tpLora, const std::string& wname, int flag) {
    assert(w->isWMAT());  // ne[0] > 1 && ne[1] > 1 && ne[2] == 1 && ne[3] == 1
    tpLORW = tpLora;
    if (tpLORW == LoAB_CARD::typW::NO_LORW || tpLORW == LoAB_CARD::typW::W0)
        return false;
    HIERARCH_LorAB::sNeurons = HIERARCH_LorAB::sNeurons + "" + name + ",";
    int rank                 = 32;  // 32 -1
    H_LoAB lora              = std::make_shared<HIERARCH_LorAB>(this, w, wname, rank);
    wLoABs.push_back(lora);

    _INFO("[H_LoAB] rank=%d adapt=%d x=%d @\"%s\"\n", rank, tpLORW, 0, w->name);
    return true;
}

bool SparseNeuron::Sparsing(int flag) {
    if (hPicker == nullptr)
        return false;
    int iRet = hPicker->Train(flag);
    return iRet;
};

bool SparseNeuron::GetHotIndex(int nPoint, floatI* data, int* hot, int flag) {
    if (hPicker == nullptr)
        return false;
    hPicker->Predict(nPoint, data, hot);
    return true;
}

bool SparseNeuron::OnData(hGTensor X, hGTensor Y, int* hot, int flag) {
    if (hPicker == nullptr)
        return false;
    if (method == 1) {
        hPicker->arrX.push_back(X);
        hPicker->arrY.push_back(Y);
    } else if (method == -1) {
        int i, dim = Y->shape[0];
    }

    return true;
}

void Fish::Sparsing(int flag) {
    for (auto neuron : neurons) {
        if (!neuron->isSparse)
            continue;
        neuron->Sparsing(flag);
    }
}
