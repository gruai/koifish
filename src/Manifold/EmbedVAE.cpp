/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Perceptrons
 *
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */
#include <set>

#include "../g_stddef.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "Fish.hpp"
#include "HotPicker.hpp"
#include "Neuron.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

TokenEmbed::TokenEmbed(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_, jit, hG_, flag) {
    auto dims = hFish->config.token_embeds;
    nVocab    = hG_->nClass();
    latent    = hFish->config.nEmbed(-1);
    shape.clear();
    hostID = new int[nVocab];
    rRounding.Init(20250903);
    /*if(jvals.size()==2){
        shape={(int)(jvals[0]),(int)(jvals[1])};
    }else{
        shape = {nCls,C};
    }    */
    assert(latent > 0);
    isAddPos = type_info[type_info.length() - 1] == '+';
    if (hG_->config.model.Rope_version > 0) {
        isAddPos = false;
    }
}
TokenEmbed::~TokenEmbed() {
    FREE_a(workload_indices);
    FREE_a(bucket_info);
    FREE_a(hostID);
}

bool TokenEmbed::SetMAEC(hMAEC mae_, int flag) {
    maec = mae_;
    // latent = maec->nIn;
    for (auto ac : maec->codes) {  // ugly design
        SLP &up = ac->up, &down = ac->down;
        out->AddSrc({up.w, up.b, up.out, down.w, down.b, down.out});
    }
    return true;
}

bool TokenEmbed::Build(int flag) {
    void *ctx     = hFish->GetGGCTX(1);
    hFish->hEmbed = this;
    int flagW = flag, n = nVocab;

    // InitMAC();
    assert(latent > 0);
    typNUMBER tpData = tpWeight;

    bool isTrain = hFish->isTrain();
    string sw = name + MODEL_CARD::sWeight, sb = name + ".pos";

    if (hFish->config.model.isPaddedCls) {
        flagW |= GTensor::F_PADDED;
        padded_nCls = ceil(n / 128.0) * 128;
    } else
        padded_nCls = n;
    // w = GT(hFish, tpData, {latent, padded_nCls}, flagW);  // padded_nCls
    w = GT(hFish, tpData, {padded_nCls, latent}, flagW);
    hFish->InitGensor(ctx, sw.c_str(), w, true);
    if (!hFish->config.model.isEmbedWeightTying) {
        wInv = GT(hFish, tpData, {padded_nCls, latent}, flagW);
        if (padded_nCls > n) {
            wInv->x_shape = {n, latent};
        }
        // sw += ".inv";
        sw = "embed_inv.weight";
        hFish->InitGensor(ctx, sw.c_str(), wInv, true);
    } else {
        wInv = w;
    }
    if (padded_nCls > n) {
        w->x_shape = {latent, n};
    }

    if (isAddPos) {
        n  = hFish->config.n_ctx();
        b  = GT(hFish, tpData, {latent, n});
        sb = "position_embd.weight";
        hFish->InitGensor(ctx, sb.c_str(), b, true);
    }
    if (hFish->isModel({NLP_GUPPY})) {
        // lnW.BuildX(name+MODEL_CARD::sNorm,{padded_nCls},hFish,flag);
        // if(wInv!=w)
        //     lnWInv.BuildX(name+MODEL_CARD::sNorm+".inv",{padded_nCls},hFish,flag);
    }

    SHAPE s3 = {B, T, latent};
    out      = std::make_shared<huTensor>(hFish, name + ".batch", s3, w->type, false);
    // hFish->InitGensor(ctx,name+".batch",out,false);

    return true;
}
/*
   batch(tokens) embeddings from glob token embedding(w)
*/
hGensor TokenEmbed::Ming(RLS_BP *ctx_, hGensor tokens, int flag) {
    // GeNeuron::BeforeMing(ctx_,tokens,flag);
    int seed    = 0;  //  rRounding.RandInt32();       //Nearly same as 0
    string sw   = name + "_rows";
    hGensor cur = nullptr;
    if (hFish->isSymbolic()) {
        assert(tokens->type == typNUMBER::I32);
        if (wInv != w)
            out->AddSrc({w, wInv, tokens, b, lnW.w, lnWInv.w});
        else
            out->AddSrc({w, tokens, b, lnW.w});
        cur = out;
    } else {
        cur = OnEmbed(tokens, seed);
    }

    cur = AfterMing(ctx_, cur, flag);
    return cur;
}
string TokenEmbed::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    bool isSym      = hFish->config.model.isEmbedWeightTying;
    sprintf(buf + strlen(buf), "%s {EMBED n=%d %s} %s", tab, nVocab, isAddPos ? "+POS" : "", isSym ? "SYM" : "");
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

hGensor VarCoder::ENC(const hGensor x0) {
    /*hGensor x = encode*x0;
    switch(tpNorm){
    case 0:
        x = x->Relu();
        break;
    case 1:
        x = x->Silu();
        break;
    case 2:
        x = x->Norm(1.0e-5);
        break;
    }

    if(isResi)
        resi = x;

    return x;*/
    return nullptr;
}

hGensor VarCoder::DEC(hGensor x) {
    if (down.Empty())  // decode==nullptr
        return x;
    if (resi != nullptr) {
        x += resi;
    }
    // x = decode*x;
    switch (tpNorm) {
        case 0:
            x = x->Relu();
            break;
        case 1:
            x = x->Silu();
            break;
        case 2:
            assert(0);  // x = x->Norm(1.0e-5);
            break;
    }

    return x;
}

string VarCoder::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    sprintf(buf + strlen(buf), "\n%s\t[%d=>%d]%s resi=%d tpNorm=%d", tab, nTop, nBottom, prefix.c_str(), isResi, tpNorm);
    // _T_repr_(encode,tab,buf);
    // _T_repr_(decode,tab,buf);
    // _T_repr_(norm,tab,buf);
    // _T_repr_(resi,tab,buf);
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
}

VarCoder::VarCoder(Fish *hG_, std::vector<int> &dims, int level, bool isR, bool isB, int tpN, int flag)
    : nTop(dims[level]), nBottom(dims[level + 1]), isResi(isR), tpNorm(tpN) {
    isBias = isB;
    name   = "AE_" + std::to_string(level);
    Init(hG_, 0x0);
    assert(nTop >= nBottom && nBottom > 0);
    Build(flag);
}

MAEC::MAEC(Fish *hG_, const std::string &key_, int flag) {
    name = "MAEC_";  //+key;
    Init(hG_, 0x0);
    auto dims = hFish->config.token_embeds;
    if (dims.size() == 1)
        return;

    nIn           = dims[0];
    nOut          = dims[dims.size() - 1];
    int reserve_x = 0;
    int nMap = dims.size() - 1, tpNorm = -1;
    assert(nMap > 0);
    bool isSymmetric = hFish->config.model.isEmbedWeightTying;

    isBias = false;
    codes.clear();
    for (int i = 0; i < nMap; i++) {
        hVarCoder hCoder = std::make_shared<VarCoder>(hFish, dims, i, reserve_x, isBias, tpNorm);
        codes.push_back(hCoder);
    }
    hVarCoder first = codes[0], last = codes[codes.size() - 1];
    /*if(0){
        normE.BuildX(name+"norm_E",{nIn},hFish,flag);
        first->down.out->AddSrc({normE.w,normE.b,normE.out,normE.rstd,normE.mean});
        normD.BuildX(name+"norm_D",{nIn},hFish,flag);
        last->up.out->AddSrc({normD.w,normD.b,normD.out,normD.rstd,normD.mean});
        normE.delta = GTensor::delta;
        normD.delta = GTensor::delta;
    }*/

    return;
}
hGensor MAEC::ENC(hGensor cur, int flag) {
    if (isForward()) {
        if (!normE.Empty()) {
            normE.cuTrain(cur);
            cur = normE.out;
        }
        for (auto ac : codes) {
            SLP &down = ac->down;
            //    cur->PrintX<floatX>("ac_in",0,-1);
            down.Forw(down.out, cur);
            cur = down.out;  // cur->PrintX<floatX>("ac_out",0,-1);
        }
    } else {
        for (auto it = codes.rbegin(); it != codes.rend(); ++it) {
            hVarCoder ac = *it;  // cur->PrintX<floatX>("ac_in",0,-1);
            SLP &down    = ac->down;
            assert(down.inp != nullptr);
            down.Back(down.delta, down.inp, cur, nullptr, 0);
            cur = down.delta;
        }
        if (!normE.Empty()) {
            normE.cuTrain(cur);
            cur = normE.delta;
        }
    }
    return cur;
}

hGensor MAEC::DEC(hGensor cur, bool isForw, int flag) {
    if (isForw) {  //  !=isForward()
        for (auto ac : codes) {
            SLP &up = ac->up;
            //    cur->PrintX<floatX>("ac_in",0,-1);
            up.Forw(up.out, cur);
            cur = up.out;  // cur->PrintX<floatX>("ac_out",0,-1);
        }
    } else {
        for (auto it = codes.rbegin(); it != codes.rend(); ++it) {
            hVarCoder ac = *it;  // cur->PrintX<floatX>("ac_in",0,-1);
            SLP &up      = ac->up;
            assert(up.inp != nullptr);
            up.Back(up.delta, up.inp, cur, nullptr, 0);
            cur = up.delta;
        }
    }
    return cur;
}
string MAEC::__repr__(string &suffix, string &prefix, int flag) {
    string p    = prefix + "\t";
    string info = p + name + "{ bias=" + std::to_string(isBias);
    info += "\n" + normE.__repr__(suffix, p, flag);
    for (auto ac : codes) {
        info += ac->__repr__(suffix, p, flag);
    }
    info += "\n" + normD.__repr__(suffix, p, flag);
    info += prefix + "\t}\n";
    return info;
};

VarCoder::VarCoder(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_, jit, hG_, flag) {
    if (jvals.size() >= 2) {
        shape = {(int)(jvals[0]), (int)(jvals[1])};
    } else {
        int n_ff = hFish->config.n_ff();
        shape    = {C, n_ff};
    }
    assert(shape[0] > 0 && shape[1] > 0);
    nBottom = shape[0], nTop = shape[1];
}

FFN::FFN(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : VarCoder(hG_, key_, jit, flag) {
    remater_ffn = hFish->config.common.remater_ffn;  // false;
    tpWeight = TYPE_<floatFFN>(), tpActivation = tpWeight, tpGradient = tpWeight;
    up.SetDType(tpWeight, tpActivation, tpGradient);
    down.SetDType(tpWeight, tpActivation, tpGradient);
    gate.SetDType(tpWeight, tpActivation, tpGradient);
    if (hG_->config.model.isFFNWeightTying) {
        // isSymmetric = true;      Need more time to study its effect
    }
    if (hFish->config.ModelArch() == NLP_GUPPY) {  //->config.model.isFFNShareParam;
        int nSample = nTop;
        int nVocab  = hFish->nClass();
        if (nSample == nVocab) {
            // compression = SAMPLE;
            // for(int i=0;i<nSample;i++){
            //     samples[i] = i;
            // }
        } else {
            compression = SAMPLE;
        }
        hSamps = GT(hFish, typNUMBER::I32, {nSample}, 0x0);
        hSamps->Alloc();
        // GT({nSample},samples,typNUMBER::I32,0x0);
        isShareParam = true;
        isBias       = false;
    }
    tpNorm = 2;
}
bool VarCoder::Build(int flag_0) {
    int flagSLP = flag_0 | F_DELTA;

    if (tpNorm > 0)
        norm.BuildX(name + MODEL_CARD::sNorm, {nBottom}, hFish, flag_0 | F_DELTA);
    if (hFish->isModel({NLP_QWEN2})) {
        up.BuildX(name + ".w1", {nBottom, nTop}, hFish, flagSLP);
        gate.BuildX(name + ".w3", {nBottom, nTop}, hFish, flagSLP);
        down.BuildX(name + ".w2", {nTop, nBottom}, hFish, flagSLP);
    } else if (hFish->isModel({NLP_QWEN3})) {
        up.BuildX(name + ".w1", {nBottom, nTop}, hFish, flagSLP);
        gate.BuildX(name + ".w3", {nBottom, nTop}, hFish, flagSLP);
        down.BuildX(name + ".w2", {nTop, nBottom}, hFish, flagSLP);
    } else {
        down.BuildX(name + "_down", {nTop, nBottom}, hFish, flagSLP);
        up.BuildX(name + "_up", {nBottom, nTop}, hFish, flagSLP);
        if (isSymmetric) {
            down.w        = nullptr;
            down.w        = up.w;
            down.isTransW = true;
        }
    }
    up.SetGanglia(this);
    gate.SetGanglia(this);
    down.SetGanglia(this);

    if (!isBias) {
        up.b   = nullptr;
        down.b = nullptr;
        if (!gate.Empty())
            gate.b = nullptr;
    }

    return true;
}

FFN *FFN::first = nullptr;
bool FFN::Build(int flag_0) {
    delta        = GTensor::delta;
    SHAPE sp     = {shape[0]}, sp3, sp2;
    void *ctx_   = hFish->GetGGCTX(1);
    bool isTrain = hFish->isTrain();
    int flag     = flag_0;
    latent       = shape[1];

    // flag |= GeNeuron::F_BIAS;
    assert(C == shape[0]);
    sp3 = {B, T, latent};
    sp2 = {B, T, C};
    // dump_flag = -1;
    // relu.out = std::make_shared<huTensor>(name+"_relu",sp3,tpWeight,false);
    // gate.BuildX(name + "_gate", {shape[0], shape[1]}, hFish, flag);

    VarCoder::Build(flag);
    if (isShareParam) {
        TokenEmbed *embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
        down.SetEmbed(embed, 0);  // down.isTransW = true;
        up.SetEmbed(embed, 1);
        if (layer == 1) {
            first = this;
        } else {
        }
    }

    if (GTensor::tmpFF1 != nullptr) {
        assert(GTensor::tmpFF1->size() >= up.out->size());
        // gelu_fusion = 1;     //  0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
        if (remater_ffn) {
            BIT_SET(up.out->flags, GTensor::F_NOALLOC);
            if (!gate.Empty())
                BIT_SET(gate.out->flags, GTensor::F_NOALLOC);
            BIT_SET(down.out->flags, GTensor::F_NOALLOC);
        } else {
            // out = norm.out;     ???
        }
        out = std::make_shared<huTensor>(hFish, name + "_out", sp2, tpWeight, false);
        if (hFish->config.isShareLayerOut()) {  //  ???
            out->SetRefer(GTensor::outL);
        }
    }

    up.w->residual_scale = hFish->config.common.residual_scale;
    if (layer > 6) {  //  Gradient would explode!
        // up.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);
        // down.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);       //  down.Back(GTensor::bt4c, tGelu, GTensor::delta, up_out);
    }

    return true;
}

std::vector<GeNeuron *> FFN::SubNeurons(int flag) {
    std::vector<GeNeuron *> neurons = {&up, &down, &norm};  // gate
    if(!gate.Empty()){
        neurons.push_back(&gate);
    }
    return neurons;
}

hGensor FFN::Ming(RLS_BP *ctx_, hGensor inpL, int flag) {
    GeNeuron::BeforeMing(ctx_, inpL, flag);

    hGensor cur      = inpL;
    hGensor lastResi = inpL;
    if (hFish->isSymbolic()) {
        // out = inpL >> up >> relu >> down >> norm;
        inpL >> up >> down >> gate >> norm >> this;
        cur = out;
    } else {  //  high performance fused operator
        cur = cuTrain(cur, 0x0);
    }

    cur = AfterMing(ctx_, cur, flag);
    return cur;
}
string FFN::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    string sS       = "";
    int n           = 0;
    switch (compression) {
        case SAMPLE:
            n  = hSamps->size();
            sS = "SAMP_" + std::to_string(samp_1);
            break;
        default:
            sS = isSparse ? hPicker->__repr__(suffix, prefix, flag) : "";
            break;
    }

    sprintf(buf + strlen(buf), "%s %s {hidden=%d} %s %s", tab, name.c_str(), shape[1], isSymmetric ? "SYM" : "", sS.c_str());
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};