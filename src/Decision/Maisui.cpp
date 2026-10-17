/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "Maisui.hpp"

#include "../Manifold/HotPicker.hpp"
#include "../Manifold/Optimizer.hpp"
#include "../Shoal/gLLM.hpp"

Maisui::Maisui(const std::string& nam_, CLI_params params, int flag) : Fish(nam_, params) {
    //  @Fish::MakeSwarm
    int nSwarm = params.n_swarm, i, nEnbed = config.nEmbed();
    for (i = 0; i < nSwarm; i++) {  // support multiple backbone
        MODEL_ROLE role       = BACKBONE_LOGITS;
        string title          = "Backbone_";
        CLI_params paramSwarm = params;
        assert(paramSwarm.sft.baseArch == NTP_QWEN3);
        paramSwarm.common.SetOnlyInfer();
        paramSwarm.phase            = LIFE_PHASE::P_EVAL_;
        paramSwarm.common.isPadMask = true;
        hFISH fish                  = Fish::MakeInstance(title, paramSwarm, wikis, role, 0x0);
        Head4Token* hCLS            = fish->GetNeuron<Head4Token>("Head4Token", 0);
        hCLS->InitFeatPicker(sPickerName, nEnbed);

        // fish->isLocalInfer = true;
        Fish::swarm.push_back(fish);
        backbone = fish;
    }
    assert(backbone->hDict != nullptr);
    hDict = backbone->hDict;
}

int Fish::CollateHidden(hDataToken coral, const std::string& sPicker, int flag) {
    assert(coral != nullptr);
    // LayerNormal* lastNorm = GetNeuron<LayerNormal>("norm", 0);
    int nEmbed = config.nEmbed(), nSamp = coral->shard_samps.size();
    assert(curTokenSet == nullptr);
    curTokenSet = coral;
    auto picker = Pickers[sPicker];
    assert(picker != nullptr);
    DEBUG.Time_most   = 60 * 1000;
    bool isCache      = !DEBUG.fHiddenFeature.empty();
    std::string sPath = DEBUG.fHiddenFeature + coral->name + "_" + std::to_string(nSamp) + ".dat";
    if (isCache && picker->SerialFeatures(this, sPath, false)) {
    } else {
        //  off = i0 + i1 * ne[0] + i2 * ne[0] * ne[1] + i3 * ne[0] * ne[1] * ne[2]
        SHAPE spMost = {nSamp, picker->dim()};  // extra line for target
        if (picker->raw_feat != nullptr) {
            picker->raw_feat.reset();
        }
        picker->raw_feat = std::make_shared<GTensor>(this, spMost, typNUMBER::F32, true, GTensor::F_DEBUG);
        picker->raw_feat->Alloc();
        // hidden_feats.clear();

        SetPhase(P_EVAL_);
        _INFO("\n[Feature] Collate %s @\"%s\" ...", picker->sFeatDesc.c_str(), coral->name.c_str());
        // Evaluate({coral}, SAMPLEofSHARD);
        coral->loader->Evaluate(SAMPLEofSHARD, 0x0);
        picker->CleanFeats();  // Shrink
        if (isCache) {
            picker->SerialFeatures(this, sPath, true);
        }
    }
    curTokenSet = nullptr;
    if (picker->empty()) {
        _ERROR("[Feature] Failed to pick anything!\n");
        return KOIFISH_HIDDEN_NULL;
    }
    return 0x0;
}

void Maisui::Train(int flag) {
    InitSpecialTokensOfDecision(GetTokenizer());

    assert(backbone != nullptr);
    hFeatPicker picker = backbone->Pickers[sPickerName];
    assert(picker != nullptr);

    hDataToken coral = backbone->tsTrain;
    assert(coral != nullptr);
    picker->hTrainData = picker->Coral2Tabular(coral, backbone, arch, FeatsOnFold::DF_TRAIN);
    if (!backbone->tsEval.empty()) {
        coral            = backbone->tsEval[0];
        picker->hValData = picker->Coral2Tabular(coral, backbone, arch, FeatsOnFold::DF_EVAL);
    }

    picker->Train();
#ifdef __USE_GBDT__
/*
    int nTree          = 1;                                                                                                       // just grow leaf
    dort->hGBRT        = new GBRT(hFold, nullptr, 0.0, flag == 0 ? BoostingForest::REGRESSION : BoostingForest::CLASIFY, nTree);  //
    dort->hGBRT->Train("", 0x0);
    // memory
    dort->hGBRT->ClearData(), dort->hGBRT->ClearHisto();
    delete hFold;
    delete dort;*/
#endif
}

bool Fish::Init(const vector<hWIKI>& wikis_, int flag) {
    wikis = wikis_;
    hEDS  = EDGE_DEVICES::GetInstance(config);
    if (hDict == nullptr) {
        if (!InitTokenCoral())  //
            return false;
    }
    if (config.common.method == "gbdt") {
    }

    if (wikis.size() == 0) {
        _INFO("[WIKI] OFF\n");  // return false;
    } else {
        teach = wikis[0]->teach;
    }

    hDistler = nullptr;  // config.sigma=="" ? nullptr : std::make_shared<Distillation>(this,config,0x0);     //ADD SIGMA

    // InitModel(flag);
    config.Dump();

    if (hOPT != nullptr) {
        // hOPT->Dump(1);
        if (/*config.isOnlyGPT ||*/ !isTrain())
            return true;
        if (!hOPT->PrepareData(config, flag))
            return false;
    }
    return true;
}

bool Maisui::Init(const vector<hWIKI>& wikis_, int flag) {
    bool bRet = Fish::Init(wikis_, flag);
    return bRet;
}

size_t Maisui::nClass() const { return 2; }

bool Maisui::InitInput(void* ctx_build, bool isMask, int flag) { return true; }

bool Maisui::AllocBuffer(int flag) { return true; };

bool Maisui::Build(int flag) { return true; }
