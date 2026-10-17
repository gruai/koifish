/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Neurons with sparsing activation/weight
 *  \author Yingshi Chen
 */
#include <memory>

#include "../Shoal/Fish.hpp"
#include "../Utils/GST_rander.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "HotPicker.hpp"
#include "Optimizer.hpp"

#ifdef __USE_GBDT__
#include "../GBDT/data_fold/DataFold.hpp"
#include "../GBDT/data_fold/Distribution.hpp"
#include "../GBDT/data_fold/Histogram.hpp"
#include "../GBDT/data_fold/Loss.hpp"
#include "../GBDT/python/pyMORT_DLL.h"
#include "../GBDT/tree/GBRT.hpp"
using namespace Grusoft;
#endif

FeatPicker::FeatPicker(hFISH hFish, int flag) {
    assert(0);                            // Deprecated
    int nEmbed = hFish->config.nEmbed();  //
    dimX       = hFish->config.n_ff();
    hot        = new int[dimX * 2]();
    for (int i = 0; i < dimX; i++) hot[i] = 1;
    dTemp = new float[dimX + nEmbed * dimX];

    T_hot  = 0.2;
    T_zero = 1.0e-3;
}

FeatPicker::FeatPicker(const std::string& key_, int dimX_, int dimY_, int flag) : name(key_), dimX(dimX_), dimY(dimY_) {
    sKey = key_;
#ifdef __USE_GBDT__
    // PY_ITEM params[] = {{"feature_quanti", 8}, {"verbose", 1000}, {"objective", 0, "binary"}, {"n_estimators", 32}, {"adaptive", 0, "weight"}};
    PY_ITEM params[] = {{"feature_quanti", 128}, {"verbose", 1000},         {"objective", 0, "regression"}, {"n_estimators", 3200},
                        {"learning_rate", 0.1},  {"adaptive", 0, "weight"}, {"metric", 0, "rmse"},          {"early_stopping_rounds", 5000}};
    DORT_wrap* dort  = static_cast<DORT_wrap*>(LiteMORT_init(params, sizeof(params) / sizeof(PY_ITEM), nullptr, 0x0));
    config           = dort->config;
    name             = name + "_GBDT";
#endif
}

bool FeatPicker::SerialFeatures(Fish* hFish, const std::filesystem::path& path0, bool isSave, int flag) {
    std::filesystem::path path = path0;

    if (isSave) {
        assert(dimX > 0 && nPickSamp > 0 && dimY > 0);
        FSerial S(path.c_str(), isSave, 0x0);
        CHECK_(S.Serial(nPickSamp, isSave, flag));
        CHECK_(S.Serial(dimX, isSave, flag));
        CHECK_(S.Serial(dimY, isSave, flag));
        CHECK_(S.Serial<float>((float*)(raw_feat->data), raw_feat->size(), isSave, flag));
        _WARN("\n>>>>>> FeatPicker save [%d,%d+%d] to \"%s\" ... OK\n", nPickSamp, dimX, dimY, path.c_str());
    } else {
        if (std::filesystem::exists(path)) {
            FSerial S(path.c_str(), isSave, 0x0);
            CHECK_(S.Serial(nPickSamp, isSave, flag));
            CHECK_(S.Serial(dimX, isSave, flag));
            CHECK_(S.Serial(dimY, isSave, flag));
            assert(dimX > 0 && nPickSamp > 0 && dimY > 0);
            SHAPE spMost = {nPickSamp, dim()};
            raw_feat     = std::make_shared<GTensor>(hFish, spMost, typNUMBER::F32, true, GTensor::F_DEBUG);
            raw_feat->Alloc();
            CHECK_(S.Serial<float>((float*)(raw_feat->data), raw_feat->size(), isSave, flag));
            _WARN("<<<<<< FeatPicker load [%d,%d+%d] from \"%s\" .... OK\n", nPickSamp, dimX, dimY, path.c_str());
        } else {
            return false;
        }
    }
    return true;
}

bool FeatPicker::BeforeCollate(hDataToken coral, int flag) {
    nPickSamp = 0x0;
    return true;
}
bool FeatPicker::CleanFeats(int flag) {
    int nFeat = dim();
    assert(raw_feat->ne[0] >= nPickSamp && raw_feat->ne[1] == nFeat);
    if (raw_feat->ne[0] == nPickSamp)
        return true;
    int n0 = raw_feat->ne[0];
    raw_feat->Shrink(nPickSamp, 0x0, 0);
    _INFO("[Feature] Shrink from %d=>%d @%s\n", n0, nPickSamp, name.c_str());
    return true;
}

hTabularData FeatPicker::Coral2Tabular(hDataToken coral, hFISH backbone, MODEL_ARCH model, int flagD, int flag) {
    assert(model == DECISION_MAISUI);
    assert(dimY == 1);
    //  BeforeCollate(coral);
    nPickSamp = 0;

    backbone->CollateHidden(coral, sKey);  // KOIFISH_FEATURE_NULL

#ifdef __USE_GBDT__
    // PY_ITEM params[] = {{"feature_quanti", 8}, {"verbose", 1000}, {"objective", 0, "binary"}, {"n_estimators", 32}, {"adaptive", 0, "weight"}};
    // PY_ITEM params[] = {{"feature_quanti", 8},  {"verbose", 1000},         {"objective", 0, "regression"},
    //                     {"n_estimators", 3200}, {"adaptive", 0, "weight"}, {"metric", 0, "rmse"}};
    // DORT_wrap* dort  = static_cast<DORT_wrap*>(LiteMORT_init(params, sizeof(params) / sizeof(PY_ITEM), nullptr, 0x0));
    // config           = dort->config;

    // _INFO("\n[DORT] nSamp=%d,nFeat_0=%d %s hExDA=%p********* \n\n", nSamp, nFeat_0, dort->merge_info.c_str(), dort->hEDA_train);
    // FeatsOnFold* hFold = Feats2DORT(config, dort, hidden_feats, flag | FeatsOnFold::DF_TRAIN);

    //  flagD = flag | FeatsOnFold::DF_TRAIN,
    int nRow = raw_feat->ne[1], nFeat = nRow - dimY;
    ExploreDA* hEDA_train = new ExploreDA(config, "DORT_wrap", flag);
    assert(nPickSamp <= raw_feat->ne[0]);
    if (nFeat <= 0) {
        _ERROR("[Feature] Feature raw is empty(nPick=%d)!\n", nPickSamp);
        return nullptr;
    }
    hTabularData hData = Tensor2Tabular(config, hEDA_train, raw_feat.get(), nFeat, nPickSamp, flagD);
    hData->nam         = coral->Name() + "_" + std::to_string(nPickSamp) + "x" + std::to_string(nFeat);
    // hTrainData->nClas = 2;
#else
    NOT_IMPLEMENTED;
#endif
    // hData.reset();     return -1;  //only for debug
    return hData;
}

int FeatPicker::Train(int flag) {
#ifdef __USE_GBDT__
    if (hTrainData == nullptr)
        return -1;

    int nTree   = config.num_trees;
    double sOOB = 0.0;
    assert(sOOB == 0.0);  // otherwise would crash
    BoostingForest::MODEL cr = config.objective == "regression" ? BoostingForest::REGRESSION : BoostingForest::CLASIFY;
    FeatsOnFold* hEval       = hValData == nullptr ? nullptr : hValData.get();
    hGBRT                    = std::make_shared<GBRT>(hTrainData.get(), hEval, sOOB, cr, nTree);
    hGBRT->Train("", 0, 0);
    // SerialModel("", true);
#endif
    return 0x0;
}

int HotPicker::Train(int flag) {
#ifdef __USE_GBDT__
    string title    = name + "_hot";
    ExploreDA* edaX = new ExploreDA(config, title, flag);
    hTrainData      = std::make_shared<FeatsOnFold>(config, edaX, title, flag);  //  from X,Y
    size_t nSamp_   = arrX.size();
    hTrainData->InitMost(nSamp_);
    if (hTrainData == nullptr)
        return -1;

    int nTree = config.num_trees;
    hGBRT     = std::make_shared<GBRT>(hTrainData.get(), nullptr, 0, BoostingForest::CLASIFY, nTree);
    hGBRT->Train("", 0, 0);
    SerialModel("", true);
#endif
    return 0x0;
}
int HotPicker::Eval(int flag) { return 0x0; }

//  Picker should much fast than dot!
double FeatPicker::tPick = 0.0;

int FeatPicker::Update(int level, float* hb, int flag) {
    return -1;
    double t0 = GST_us();
    if (level > 0)
        return nLastHot;
    int nz = 0, i = 0, nHot = std::max((int)(dimX * T_hot), 16), nEx = nHot, id;
    nEx = ((nHot * 2) / 16) * 16 - nHot;
    // nEx = ((nHot)/16)*16-nHot;
    float *tmp = dTemp, prev = FLT_MAX, a;
    int* map = hot + dimX;
    // for(i=0; i<dim; i++)    hot[i]=1;   return dim;

    for (i = 0; i < dimX; i++) {
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
        id = rand() % dimX;
        if (hot[id] == 0) {
            hot[id] = 1;
            nHot++;
            i++;
        }
    }
    assert(nHot % 16 == 0);

    nLastHot = nHot;
    if (isMerge) {
        for (nHot = 0, i = 0; i < dimX; i++) {
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

HotPicker::HotPicker(SparseNeuron* n, int flag) : FeatPicker(n->name, -1) {
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

SparseNeuron::SparseNeuron(const std::string& key_, const JSON& jit, Fish* hG_, int flag) : GeNeuron(key_, jit, hG_, flag) {
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
        assert(0);
        // w->SetRefer(embd_->wInv);
    }
}

// TODO: Weighted sampling
void SparseNeuron::UpdateSamps(int seed, int flag) {
    assert(hSamps != nullptr);
    float* weight = nullptr;  // TODO: Weighted sampling
    int nVocab = hFish->nClass(), nSample = hSamps->size();
    std::vector<int> samps;
    GRander rander(seed);
    if (1) {  // nearly same
        hSampNanny sloader = hFish->GetOptimizer()->train_loader;
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
HIERARCH_LorAB::HIERARCH_LorAB(SparseNeuron* neuron, hGTensor w_, const std::string& title_, int r_, int flag)
    : wBase(w_), rank(r_), title(title_), spNeuron(neuron) {
    hFish  = neuron->hFish;
    config = &(hFish->config.loAB);
    rank   = config->rLatent;
    nInA = w_->ne[1], nOutB = w_->ne[0];
    if (rank == -1)  // only for debug
        rank = nInA;
    else {
        assert(rank * 2 < hFish->config.nEmbed());  // neuron->C   low rank
    }
    assert(w_->isWMAT() && rank > 0);
    //  nIn = shape[1], nOut = shape[0] @SLP::Build

    B = neuron->B, T = neuron->T;

    // string title = w_->name;
    //  spQ      = {q_dim, n_embd};
    a = GT(hFish, w_->type, {rank, nInA}, flag | GTensor::F_LORA_A, title + "_a.w");
    b = GT(hFish, w_->type, {nOutB, rank}, flag | GTensor::F_LORA_B, title + "_b.w");
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

void HIERARCH_LorAB::UpdateAdapt(int flag) {}
/*
    Since low rank, no need to quantize/ramater/guoke...
*/
bool SparseNeuron::InitLoRA(LoAB_CARD::typW tpLora, const std::string& wname, int flag) {
    assert(w->isWMAT());  // ne[0] > 1 && ne[1] > 1 && ne[2] == 1 && ne[3] == 1
    tpLORW = tpLora;
    if (tpLORW == LoAB_CARD::typW::NO_LORW || tpLORW == LoAB_CARD::typW::W0)
        return false;
    HIERARCH_LorAB::sNeurons = HIERARCH_LorAB::sNeurons + "" + name + ",";

    H_LoAB lora = std::make_shared<HIERARCH_LorAB>(this, w, wname, -1);
    wLoABs.push_back(lora);
    auto info = lora->config->Dump(0x0);
    _INFO("[H_LoAB] %s @\"%s\"\n", info.c_str(), w->name);
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

bool GeNeuron::InitFeatPicker(const std::string& key, int dim_, int flag) {
    hPicker = std::make_shared<FeatPicker>(key, dim_);
    // hGTensor hidden_feats = nullptr;
    // std::vector<Grusoft::FeatVector*> hidden_feats;return true;
    hFish->Pickers[key] = hPicker;
    return true;
}

bool GeNeuron::AppendFeats(hGTensor fTensor, int x, int flag) {
    assert(hPicker != nullptr);
    hBATCH hBatch     = hFish->curBatch(0x0);
    hGTensor raw_feat = hPicker->raw_feat;
    int sampInBatch = fTensor->ne[2], ldT = fTensor->ne[1], ldH = fTensor->ne[0], i, samp, nSamp = raw_feat->ne[0];
    int pid           = hPicker->nPickSamp;
    const floatX* dev = ToX(fTensor);
    floatX *host_data = new floatX[ldH], *fv;
    assert(ldH <= raw_feat->ne[1]);
    for (samp = 0; samp < sampInBatch; samp++, pid++, hPicker->nPickSamp++) {
        hSAMP hSamp = hBatch->curSamps[samp];
        assert(pid <= nSamp);
        D2H(dev + fTensor->size() - hSamp->pad_len * ldH, host_data, sizeof(floatX) * ldH);  // last hidden feat
        // PrintTensor<floatX>("host_data", host_data, false, 1, ldH, 1, 1, -1);
        for (i = 0; i < hPicker->dimY; i++) {  // i_th target
            float a = hBatch->user_answer[samp * hPicker->dimY + i];
            assert(a == 1.0f || a == 0.0f);
            raw_feat->Set(pid, ldH + i, 0, 0, a);
        }
        for (i = 0; i < ldH; i++) {
            float a = T2Float(host_data + i);
            raw_feat->Set(pid, i, 0, 0, a);

            /*  @MAT_ROW2FEAT_
            Wrong -  each FeatVector need all samp's element, so not available here!
            Grusoft::FeatVector* hFV    = nullptr;
            Grusoft::FeatsOnFold* hFold = nullptr;
            int flagF = flag, id = hidden_feats.size();
            string desc = "feat_" + std::to_string(id);
            assert(!BIT_TEST(flagF, Grusoft::FeatVector::VAL_REFER));
            switch (feat->type) {
                case typNUMBER::BF16:
                    hFV = new FeatVec_T<float>(hFold, nSamp_, id, desc, flagF);
                    hFV->Set_bf16(nSamp_, (bf16*)samp_data);
                    break;
                case typNUMBER::F32:
                    hFV = new FeatVec_T<float>(hFold, nSamp_, id, desc, flagF);
                    hFV->Set(nSamp_, samp_data);
                    break;
                default:
                    NOT_IMPLEMENTED;
                    break;
            }
            hidden_feats.push_back(hFV);
            */
        }
    }
    // PrintTensor<float>("Target", (float*)(raw_feat->data) + ldH * nSamp, false, nSamp, 1, 1, 1, -1);
    // PrintTensor<float>("feature", (float*)(raw_feat->data), false, nSamp, ldH, 1, 1, -1);
    delete[] host_data;
    return true;
}

/*
    v0.1	cys
        11/17/2025
*/
FeatVector* MAT_ROW2FEAT(const FeatsOnFold* hFold, typNUMBER tpN, void* hData, size_t nSamp_, int id, bool isMergePivot, int flag) {
    string desc       = "feat_" + std::to_string(id);
    FeatVector* hFeat = nullptr;
    int flagF         = flag;
    // if (hFold->isTrain() && !isMergePivot) {
    //     hFold->edaX->AddDistri(nullptr, id);  // col
    // }
    assert(!BIT_TEST(flagF, FeatVector::VAL_REFER));
    switch (tpN) {
        case typNUMBER::BF16:
            hFeat = new FeatVec_T<float>(hFold, nSamp_, id, desc, flagF);
            hFeat->Set_bf16(nSamp_, (bf16*)hData);
            break;
        case typNUMBER::F32:
            hFeat = new FeatVec_T<float>(hFold, nSamp_, id, desc, flagF);
            // hFeat = new FeatVec_T<float>(hFold, nSamp_, id, desc, flagF | FeatVector::VAL_REFER);
            hFeat->Set(nSamp_, (float*)hData);
            break;
        default:
            assert(0);
            throw "MAT_ROW2FEAT_ col->dtype is XXX";
            break;
    }
    /*
    hFeat->PY  = col;
    hFeat->nam = col->type_x;
    hFeat->nam += col->name;
    hFeat->UpdateType();
    */
    hFeat->nam = desc;
    assert(hFold->edaX != nullptr);  // hFold->edaX == nullptr ? nullptr : &(hFold->edaX->arrDistri[i]);
    Distribution* tDistri = hFold->edaX->GetDistri(id);
    hFeat->InitDistri(hFold, tDistri, nullptr, true, 0x0);

    return hFeat;
}

//  PY_DATASET *dataset_
hTabularData Tensor2Tabular(LiteBOM_Config config, ExploreDA* edaX, void* hOBJ, int nMostFeat, int nSamp, int flag) {
    GTensor* hRawData = static_cast<GTensor*>(hOBJ);
    clock_t t0        = GST_ms();
    assert(BIT_IS(flag, FeatsOnFold::DF_FLAG));
    if (config.eda_Normal == LiteBOM_Config::NORMAL_gaussian) {
        edaX = nullptr;
        _INFO("\n---- !!! Data normalization by gaussian So edaX is nullptr@FeatsOnFold_InitInstance !!!----\n");
    }

    bool isTrain   = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
    bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
    bool isMerge   = BIT_TEST(flag, FeatsOnFold::DF_MERGE);
    assert(!isMerge);
    double sparse = 0, nana = 0;
    size_t nMostQ = config.feat_quanti, nConstFeat = 0, nMergedFeat = 0, nLocalConst = 0, nQuant = 0;
    // FeatsOnFold* hFold = new FeatsOnFold(config, edaX, "T2DORT", flag);
    hTabularData hFold = std::make_shared<FeatsOnFold>(config, edaX, "T2DORT", flag);
    hFold->InitRanders();
    hFold->InitMost(nSamp);
    // hFold->nMost = nSamp_;
    int rnd_seed = 0, nThread = config.num_threads, flagF = flag | FeatVector::VAL_REFER;
    for (int i = 0; i < nMostFeat; i++) {  //
        hFold->edaX->AddDistri("", i);
    }
    hFold->feats.resize(nMostFeat);
    assert(hRawData->type == typNUMBER::F32);
    // #pragma omp parallel for
    for (int i = 0; i < nMostFeat; i++) {
        void* hData     = hRawData->data + hRawData->Offset(0, i, 0, 0);
        hFold->feats[i] = MAT_ROW2FEAT(hFold.get(), hRawData->type, hData, nSamp, i, false, flag);
    }

    if (!isMerge) {  // lossy and importance
        hFold->importance = new Feat_Importance(hFold.get());
        hFold->lossy->Init_T<tpDOWN>(hFold.get(), nSamp, 0x0, rnd_seed, flag);
        if (isPredict) {
        } else {
            FeatVector* Y = hFold->GetY();
            void* hData   = hRawData->data + hRawData->Offset(0, nMostFeat, 0, 0);
            Y->Set(nSamp, ((float*)hData));
            hFold->InitTags();
        }
        hFold->lossy->EDA(nullptr, flag);
    }

    GST_TIC(t1);
    int nFeat = hFold->nFeat();
    if (false) {  // Ϊ�˵��� ���Ǻ�FeatVector::Feat(int no)��ͻ��
        std::sort(hFold->feats.begin(), hFold->feats.end(), FeatVector::OrderByName);
        for (int i = 0; i < nFeat; i++) {
            // hFold->feats[i]->id = i;		ָ��eda->distribution�����ܸ�
        }
    }

    // #pragma omp parallel for num_threads(nThread) schedule(dynamic) reduction(+ : sparse,nana,nConstFeat,nLocalConst,nQuant)
    for (int feat = 0; feat < nFeat; feat++) {
        FeatVector* hFQ   = nullptr;
        FeatVector* hFeat = hFold->Feat(feat);
        if (feat == 77)
            feat = 77;
        Distribution* fDistri = hFeat->myDistri();
        sparse += fDistri->rSparse * nSamp;
        nana += fDistri->rNA * nSamp;
        // if (BIT_TEST(hFeat->type, FeatVector::V_ZERO_DEVIA)) {
        if (fDistri->isPass()) {
            // printf("%d\n", feat);
            nConstFeat++;
            nLocalConst++;
        } else if (config.objective == "quant") {
        } else {  //  FeatVecQ only support uint8_t/uint16_t
            if (hFold->isQuanti || hFeat->isCategory()) {
                hFold->feats[feat] = hFQ = FeatVecQ_InitInstance(hFold.get(), hFeat, 0x0);  // new FeatVec_Q<short>(hFold, hFeat, nMostQ);
                nQuant++;                                                                   // delete hFeat;
            }
        }
    }
    // FeatsOnFold::stat.tX += GST_TOC(t1);
    // if (hFold->isQuanti) {
    //     hFold->Feature_Bundling();  //  just return
    // }
    int nTotalBins = 0;
    for (int feat = 0; feat < hFold->nFeat(); feat++) {
        FeatVector* hFeat = hFold->Feat(feat);
        if (config.feat_selector != nullptr) {
            hFeat->select.user_rate = config.feat_selector[feat];
            printf("%d(%.3g)\t", feat, hFeat->select.user_rate);
        }
        if (BIT_TEST(hFeat->type, FeatVector::REPRESENT_)) {
            PY_COLUMN* col = hFeat->PY;  // dataset_->columnX + feat;
            assert(hFeat->select.isPick = true);
            hFeat->select.isPick = false;
            hFold->present.Append(hFeat, col->representive);
        }
        if (hFold->config.verbose > 666) {
            hFeat->myDistri()->Dump(feat, false, flag);
        }
        if (!isMerge) {
            Distribution* hDistri = hFold->histoDistri(hFeat);
            nTotalBins += hDistri->nHistoBin();
        }
    }

    sparse /= (nSamp * hFold->nFeat());
    nana /= (nSamp * hFold->nFeat());
    // assert(nana == 0.0);
    if (config.verbose > 0) {
        printf("\r********* Fold_[%s] nSamp=%d nFeat=%ld(const=%ld) QUANT=%ld Total Bins=%d\n\tsparse=%g NAN=%g nLocalConst=%ld time=%g sec\r\n",
               hFold->nam.c_str(), nSamp, hFold->nFeat(), nConstFeat, nQuant, nTotalBins, sparse, nana, nLocalConst, (GST_ms() - t0) / 1000.0);
    }
    // if(nLocalConst>0)
    //	printf("\t!!! [%s] nLocalConst=%ld !!! \n",hFold->nam.c_str(), nLocalConst);
    return hFold;
}