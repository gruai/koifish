/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "TGraph.hpp"

#include <sched.h>

#include "../Tensor/GeQuant.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "Fish.hpp"
#include "llama_cys.h"
#ifdef __USE_GGML__
#include "ggml-impl.h"
#endif

// hRope SelfAttention::rope = nullptr;
hGensor ROPE::KQ_pos = nullptr, ROPE::hSin = nullptr, ROPE::hCos = nullptr;

hGensor Fish::AddTensor(const std::string& key_, typNUMBER tp, const SHAPE& shape, int flag) {
    auto ctx          = GetGGCTX();
    hGensor gg_tensor = nullptr;
    if (shape.size() == 4) {
        gg_tensor = GT(this, tp, shape);
    } else if (shape.size() == 2) {
        gg_tensor = GT(this, tp, shape);
    } else if (shape.size() == 1) {
        gg_tensor = GT(this, tp, shape);
    } else {
        assert(0);
    }
    // gensors.Insert(gg_tensor);

    return gg_tensor;
}

size_t SLP::nElem() {
    size_t nX = 0;
    nX += tELEM(w);
    if (b != nullptr)
        nX += tELEM(b);
    return nX;
}

SelfAttention::SelfAttention(Fish* hG_, const std::string& key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_, jit, hG_, flag) {
    assert(hFish != nullptr);

    delta = gBUFF->delta;

    auto& config     = hG_->config;
    f_max_alibi_bias = config.f_max_alibi_bias;
    n_embd_gqa       = config.n_embd_v_gqa();
    KQ_mask = hFish->KQ_mask, KQ_pos = hFish->KQ_pos;
    ID = 0;
    // remater_qkv = hFish->config.common.remater_qkv;

    n_embd = config.nEmbed();
    n_ff      = hFish->config.n_ff(ID);
    n_head_kv = config.n_head_kv(ID);
    n_head    = config.n_head(ID);
    head_dim  = config.head_dim(ID);
    q_dim = config.Q_dim(layid - 1), kv_dim = config.KV_dim(layid - 1);    
    assert(kv_dim <= q_dim);  // config.nEmbed();
    // assert(n_embd_head * n_head == C);
    C_qkv = q_dim;  //C;
    /**/
    if (jvals.size() >= 3) {
        shape = {(int)(jvals[0]), (int)(jvals[1]), (int)(jvals[2])};
    } else {  //"attn":{"QKV":[]},
        if (config.model.qkv_embeds.size() > 1) {
            assert(0);
            // C_qkv = config.qkv_embeds[1];
        }
        //  hidden_dim shoud be less than 256 and hidden_dim should be multiple of 8
        assert(C_qkv % n_head == 0 && (C_qkv / n_head) % 8 == 0);
        shape = {q_dim, C_qkv, n_head};
    }
    isSeparateQKV = hFish->config.model.isSeparateQKV;
    isBqkv        = hFish->config.model.isBqkv;

    isQKNormal = hFish->config.model.isQKNormal;

    // dump_flag = -1;
    tpNormal = DEBUG.SelfAttention_noraml;
    spQ      = {q_dim, n_embd};
    spKV     = {kv_dim, n_embd};
}

bool SelfAttention::Build(int flag_0) {
    hCache  = hFish->hCache;  // may nullptr
    qkv4dnn = hFish->config.model.qkv4dnn;

    // SHAPE sp = {shape[0], shape[1]};
    int flag = flag_0, nTokens = nBatchToken();
    norm.BuildX(_NAME(name, ATTN_PRE_NORMAL), {n_embd}, hFish, flag);
    if (isQKNormal) {
        normQ.nHead = n_head, normK.nHead = n_head_kv;
        normQ.BuildX(_NAME(name, ATTN_Q_NORM), {head_dim}, hFish, flag);
        normK.BuildX(_NAME(name, ATTN_K_NORM), {head_dim}, hFish, flag);
    }
    
    if (isSeparateQKV) {
        int flagQKV = hFish->config.model.isQKVBias ? flag | F_BIAS | F_DELTA : flag | F_DELTA;
        Q.BuildX(_NAME(name, ATTN_Q), spQ, hFish, flagQKV);
        K.BuildX(_NAME(name, ATTN_K), spKV, hFish, flagQKV);
        V.BuildX(_NAME(name, ATTN_V), spKV, hFish, flagQKV);
        if (isBqkv) {
            bqkv = std::make_shared<huTensor>(hFish, name + ".wqkv.bias", spQ, tpWeight, false);  //  model.layers.0.attn.wqkv.bias
            hFish->InitGensor(nullptr, bqkv->name, bqkv, true);
        }
    } else {
        Q.BuildX(name + "_qkv", {shape[0], shape[1] * 3}, hFish, flag);
    }
    if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE)) {
        attn = std::make_shared<huTensor>(hFish, name + ".attn", (SHAPE){n_head, hFish->config.chat_sampler.seq_len}, tpWeight, false);
    } else {
        attn = std::make_shared<huTensor>(hFish, name + ".attn", (SHAPE){B, T, q_dim}, tpWeight, false);  // B * T * C
    }
#ifdef ENABLE_CUDNN
    transition = GT(hFish, typNUMBER::F32, {B, n_head, T}, 0x0, name + ".trans");  // (B, Hq, T)
#else
    transition = GT(hFish, tpWeight, {B, n_head, T * T}, 0X0, name + ".trans");  //  too much memory!
#endif
    SHAPE spOut = {B, T, (int)hFish->config.nEmbed()};  //,T
    out = std::make_shared<huTensor>(hFish, name + ".out", spOut, tpWeight, false);
    if (hFish->config.isShareLayerOut()) {
        out->SetRefer(gBUFF->outL);
    }

    proj_cat.BuildX(_NAME(name, ATTN_OUT), (SHAPE){spQ[1], spQ[0]}, hFish, flag | F_DELTA);

    BIT_SET(proj_cat.out->flags, GTensor::F_NOALLOC);  // memory trick as kGPT
    proj_cat.w->residual_scale = hFish->config.common.residual_scale;
    if (layid > 6) {
        // Q.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);
        // K.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);
        // Lora of V+proj would slow convergence
        // V.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);
        // proj_cat.InitCompression(COMPRESSIVE_SENSING::LORA, hFish->config.tpLORA);
    }
    // if (remater_qkv) {
    if (isSeparateQKV) {
        BIT_SET(Q.out->flags, GTensor::F_NOALLOC), BIT_SET(K.out->flags, GTensor::F_NOALLOC), BIT_SET(V.out->flags, GTensor::F_NOALLOC);
    } else
        BIT_SET(Q.out->flags, GTensor::F_NOALLOC);
    //}
    if (!hFish->isModel({NLP_GPT2, NLP_GPT2_char})) {
        assert(isSeparateQKV);
        rope = std::make_shared<ROPE>(this, name + ".ROPE");
        // rope->BuildX(name + ".ROPE", spQ, hFish, flag);
    }
    _devQKV(0x0);

    // QUANT_CARD quant_params = hFish->config.quant;
    quant_params.Init4Neuron(name, hFish->config.jQuant);
    if (layid > quant_params.nPassLayer) {
        quant_params.spMost       = Q.w->shape;
        quant_params.default_bits = layid > 0 ? 3 : 4;
        hQuant                    = GeQuant::MakeInstance(this, name, quant_params, {Q.w, K.w, V.w, proj_cat.w}, 0x0);  //  {Q.w,proj_cat.w}
    }
    // tpTrans = RELU2;
    // moe.BuildX(name+".moe",sp,hFish,flag);        //  why this would slow converge???
    return true;
}

//
bool SelfAttention::UpdateQKVPack(int flag) {
    devQ = Q.out->data, devK = K.out->data, devV = V.out->data;
    if (qkv4dnn == qkvPack) {
        return true;
    }

    hBITARR hQKV = TO<BIT_8>(gBUFF->bt4c), hQ = (hBITARR)devQ, hK = (hBITARR)devK, hV = (hBITARR)devV;
    size_t offset = 0x0, nT = B * T, ldQ = sizeof(floatX) * q_dim, ldKV = sizeof(floatX) * kv_dim;
    assert(gBUFF->bt4c->nByte() >= nT * (ldQ + ldKV * 2));
    switch (qkv4dnn) {
        case QKV_PACK::QKVQKV:
            for (int r = 0; r < nT; r++) {
                D2D(hQKV + offset, hQ + ldQ * r, ldQ), offset += ldQ;
                D2D(hQKV + offset, hK + ldKV * r, ldKV), offset += ldKV;
                D2D(hQKV + offset, hV + ldKV * r, ldKV), offset += ldKV;
            }
            devQ = hQKV, devK = hQKV + ldQ, devV = hQKV + ldQ + ldKV;
            break;
        default:
            break;
    }

    return true;
}
bool SelfAttention::_devQKV(int stage, int flag) {
    hGensor tmpQKV = gBUFF->tmpFF1;  
    if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE)) {
        assert(hCache != nullptr);
        int pos           = stage;
        floatX *key_cache = (floatX*)hCache->Get(KVCache::KV_KEY, layid - 1, 0), *val_cache = (floatX*)hCache->Get(KVCache::KV_VAL, layid - 1, 0);
        K.out->data = key_cache + (size_t)pos * kv_dim, V.out->data = val_cache + (size_t)pos * kv_dim;
        Q.out->data = ToX(tmpQKV);
        return true;
    }

    assert(hCache == nullptr);
    bool isTrain = hFish->isTrain();
    // devQ = ToX(tmpQKV);
    // assert(devQ != nullptr);
    // devK = (char*)devQ + Q.out->nByte(), devV = (char*)devK + K.out->nByte();
    // assert((char*)(devV) + V.out->nByte() - (char*)(tmpQKV->data) <= tmpQKV->nByte());  // may fail!!!
    Q.out->data = ToX(tmpQKV);
    K.out->data = ToX(Q.out) + Q.out->size();
    V.out->data = ToX(K.out) + K.out->size();
    qkvPack     = QQKKVV;

    // offset = Q.out->nByte();
    size_t offset = 0;
    if (isTrain) {
        assert(kv_dim <= q_dim);
        deltaQ = gBUFF->bt4c->Partial("partialDeltaQ", 0, {B, T, q_dim}), offset += deltaQ->size();
        deltaK = gBUFF->bt4c->Partial("partialDeltaK", B * T * q_dim, {B, T, kv_dim}), offset += deltaK->size();
        deltaV = gBUFF->bt4c->Partial("partialDeltaV", B * T * (q_dim + kv_dim), {B, T, kv_dim}), offset += deltaV->size();
        assert(B * T * (q_dim + kv_dim * 2) * sizeof(floatX) <= gBUFF->bt4c->nByte());
        // devDeltaQ = ToX(gBUFF->bt4c), devDeltaK = (char*)devDeltaQ + Q.out->nByte(), devDeltaV = (char*)devDeltaK + K.out->nByte();
        devDeltaQ = ToX(deltaQ), devDeltaK = ToX(deltaK), devDeltaV = ToX(deltaV);
    } else {
    }

    return true;
}

string SelfAttention::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    string a, sRope = rope == nullptr ? "" : rope->__repr__(a, a, flag);
    sprintf(buf + strlen(buf), "{%s QKV%s%s E%d H%d x=%d trans=%d %s}", tab, moe.Empty() ? "" : "+moe", sRope.c_str(), n_embd, n_head, tpNormal, tpTrans,
            bqkv == nullptr ? "" : "bqkv");
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

std::vector<GeNeuron*> SelfAttention::SubNeurons(int flag) {
    std::vector<GeNeuron*> neurons = {&Q, &K, &V, &proj_cat, &norm};
    if (isQKNormal) {
        neurons.push_back(&normQ), neurons.push_back(&normK);
        /*auto gensors = normQ.PickGensors();
        for(auto t : gensors){  //only for debug
            t->DumpX(0x0);
        }*/
    }
    return neurons;
}

bool SelfAttention::BeforeForward(int iter, int op, int flag) {
    if (rope != nullptr) {
        rope_seed = rope->rRounding.RandU32();
    }
    return true;
}

hGensor SelfAttention::Ming(RLS_BP* hRLS, hGensor inpL, int flag) {
    GeNeuron::BeforeMing(hRLS, inpL, flag);

    hGensor cur = inpL, lastResi = inpL;
    if (hFish->isSymbolic()) {
        if (isSeparateQKV)
            attn->AddSrc({inpL, Q.w, Q.b, Q.out, K.w, K.b, K.out, V.w, V.b, V.out, transition, bqkv});
        else {
            inpL >> Q;
            attn->AddSrc({Q.out, transition});
        }
        if (isQKNormal) {
            // attn->AddSrc({normQ.rstd, normK.rstd});
            attn >> proj_cat >> norm >> normQ >> normK >> this;
        } else
            attn >> proj_cat >> norm >> this;
        out->AddSrc({transition, bqkv, attn});  // duplicate!
        cur = out;
        // gTN0(cur,"%s_+",name.c_str());
    } else if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE)) {
        cur = cuInfer(cur, flag);
    } else {  // high performace fused operator
        cur = cuFlow(cur, flag);
    }

    cur = AfterMing(hRLS, cur, flag);
    return cur;
}

hGensor SelfAttention::MyAttention(RLS_BP* ctx_, hGensor cur, int flag) {
    float kq_scale = 1.0f / sqrtf(float(head_dim)), s;

    hGensor q, k, kq = nullptr;  //  assert(KQ_mask!=nullptr);
    hGensor Qcur = Q.Ming(ctx_, cur, 0x0);
    hGensor Kcur = K.Ming(ctx_, cur, 0x0);
#ifdef _TENSOR_G_
    return cur;
#else
    // cb(Qcur, "Qcur", il);        cb(Kcur, "Kcur", il);        cb(Vcur, "Vcur", il);
    if (isAttOnBC) {  // attenion on all tokens, memory would explode!
        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head, n_tokens);
        // Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head, n_tokens);
    } else /**/ {
        Qcur = ggml_reshape_4d(ctx_, Qcur, n_embd_head, n_head, n_ctx, n_batch);
        Kcur = ggml_reshape_4d(ctx_, Kcur, n_embd_head, n_head, n_ctx, n_batch);
        // Vcur = ggml_reshape_4d(ctx_, Vcur, n_embd_head, n_head, n_ctx,n_batch);
    }
    if (!rope->Empty()) {
        rope->sT = "Q";
        Qcur     = rope->Ming(ctx_, Qcur, 0x1);
        rope->sT = "K";
        Kcur     = rope->Ming(ctx_, Kcur, 0x1);
    }
    if (tpNormal == 1 /*&& n_embd_head>=1*/) {
        Qcur = ggml_rms_norm(ctx_, Qcur, 1.0e-5);
        Kcur = ggml_rms_norm(ctx_, Kcur, 1.0e-5);  // Vcur=ggml_rms_norm(ctx_,Vcur,1.0e-5);
        gTN(Qcur, "%s.Q04", name.c_str());
        gTN(Kcur, "%s.K04", name.c_str());  // gTN(Vcur,"%s.V4",name.c_str());
    }

    if (tpTrans == LINEAR && 0) {  //  ???
        attn_k   = Permute(ctx_, Kcur, 1, 2, 0, 3);
        attn_q   = Permute(ctx_, Qcur, 0, 2, 1, 3);
        isLinear = true;
        return nullptr;
    }
    q  = Permute(ctx_, Qcur, 0, 2, 1, 3);
    k  = Permute(ctx_, Kcur, 0, 2, 1, 3);
    kq = ggml_mul_mat(ctx_, k, q);  // cb(kq, "kq", il);
    gTN(kq, "%s.kxq", name.c_str());
    switch (tpTrans) {  // Get markov transition matrix from KQ
        case LINEAR:
            kq = ggml_scale(ctx_, kq, kq_scale);
            gTN(kq, "%s.kqs", name.c_str());
            break;
        case RELU2:  // same name of grad!
            kq = ggml_silu(ctx_, kq);
            gTN(kq, "%s.r2_0", name.c_str());
            kq = ggml_mul(ctx_, kq, kq);
            gTN(kq, "%s.r2_1", name.c_str());
            kq = ggml_scale(ctx_, kq, (1.0f / n_embd_head));
            gTN(kq, "%s.r2_2", name.c_str());
            break;
        case RELU_:  // slower <0.38->0.68>@Epoch_161
            kq = ggml_silu(ctx_, kq);
            gTN(kq, "%s.r_0", name.c_str());
            kq = ggml_scale(ctx_, kq, kq_scale);
            gTN(kq, "%s.r_1", name.c_str());
            break;
        case SIGMOID:
            kq = ggml_sigmoid(ctx_, kq);
            gTN(kq, "%s.s_0", name.c_str());
            kq = ggml_scale(ctx_, kq, 1.0 / float(n_embd_head + 1.0));
            gTN(kq, "%s.s_1", name.c_str());
            break;
        case SOFT_MAX:
        default:                                                                        //
            if (KQ_mask != nullptr) {                                                   //     may crash in some case!
                kq = ggml_soft_max_ext(ctx_, kq, KQ_mask, kq_scale, f_max_alibi_bias);  // would
            } else {                                                                    // wouls slow converge,why?
                hGensor t16_1 = tSCAL(ctx_, kq, kq_scale);
                gTN(t16_1, "%s.161", name.c_str());
                // hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx_, t16_1, 0);    gTN(t16_2,"%s.162",name.c_str());
                /*
                    ggml_diag_mask_inf  实现对输入张量进行对角线以下部分的掩码操作，将其设置为负无穷。  n_past：过去的时间步数
                    REF @ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
                */
                hGensor t16_2 = ggml_diag_mask_inf(ctx_, t16_1, 0);
                gTN(t16_2, "%s.162", name.c_str());
                kq = ggml_soft_max(ctx_, t16_2);
            }
            break;
    }
#endif
    gTN(kq, "%s.kq_attn", name.c_str());  // cb(kq, "kq_soft_max_ext", il);

    return kq;
}

BROWN_attn::BROWN_attn(Fish* hG_, const std::string& key_, JSON::const_iterator jit, int flag) : SelfAttention(hG_, key_, jit, flag) {
    auto& config    = hG_->config;
    n_rot           = config.head_dim();
    rope_freq_base  = config.model.rope_freq_base;
    rope_freq_scale = config.model.rope_freq_scale;
}
bool BROWN_attn::Build(int flag) {
    // SelfAttention::Build(flag);
    SHAPE sp = {shape[0], shape[1]};
    norm.BuildX(name + ".norm", {shape[0]}, hFish, 0x0);
    Q.BuildX(name + ".tmp", {T, T, n_head, B}, hFish, flag);  // transition as property
    proj_cat.BuildX(name + ".proj", sp, hFish, flag);
    // moe.BuildX(name+".moe",sp,hFish,flag);
    return true;
}
hGensor BROWN_attn::Ming(RLS_BP* ctx_, hGensor teb, int flag) {
    assert_shape_2d(teb, n_embd, T * B);
    hGensor cur = BeforeMing(ctx_, teb, flag);
    if (cur == nullptr)
        return cur;

    cur                  = norm.Ming(ctx_, cur, 0x0);
    const float kq_scale = 1.0f / sqrtf(head_dim);
    int N = T, n_past = 0;
    ;
    hGensor v = cur, v3 = nullptr, v4 = nullptr, wv = nullptr, kqv_out = nullptr, prob;
#ifdef _TENSOR_G_
#else
    hGensor v_rope = ggml_reshape_4d(ctx_, cur, n_embd_head, n_head, N, n_batch);
    gTN(v_rope, "%s.4", name.c_str());
    if (!Rope_version) {
        v_rope = ggml_permute(ctx_, v_rope, 1, 2, 0, 3);
        gTN(v_rope, "%s.4p", name.c_str());  //  [ctx, E/H, H, n_batch); ]
        v = ggml_cont(ctx_, v_rope);
    } else {
        if (0)
            ;  // v = W_rope(ctx_,cur,V.w,KQ_pos,{n_embd_head, n_head, N, n_batch},"v",0x1);   //24,6,32,3
        else {
            v_rope = ggml_rope_ext(ctx_, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            gTN(v_rope, "%s.rope_ext", name.c_str());
            v_rope = ggml_permute(ctx_, v_rope, 1, 2, 0, 3);  //  [ctx, E/H, H, n_batch); ]
            v      = ggml_cont(ctx_, v_rope);
        }
    }
    gTN(v, "%s.v4", name.c_str());
    if (KQ_mask != nullptr) {
        prob = ggml_soft_max_ext(ctx_, Q.w, KQ_mask, kq_scale, f_max_alibi_bias);  // would crash!
    } else {
        hGensor t16_1 = tSCAL(ctx_, Q.w, kq_scale);
        hGensor t16_2 = ggml_diag_mask_inf_inplace(ctx_, t16_1, n_past);
        prob          = ggml_soft_max_inplace(ctx_, t16_2);
    }
    // [32,24,6,3]x[32,32,6,3]  => [24,32,6,3]
    wv = ggml_mul_mat(ctx_, v, prob);
    gTN(wv, "%s.wv", name.c_str());
    // experts mechanism
    if (!moe.Empty()) {
        // v4 = ggml_reshape_4d   (ctx_, teb, n_embd_head, n_head, N, n_batch);
        // v4 = ggml_permute(ctx_, v4, 0,2,1,3);
        // v4 = ggml_cont(ctx_,v4);
        // wv = moe.Forward2(ctx_,wv,v4);
        wv = moe.Ming(ctx_, wv);
    }

    kqv_out = ggml_permute(ctx_, wv, 0, 2, 1, 3);  //
    assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);
    // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    gTN(kqv_out, "%s.kqv_out_rope", name.c_str());

    kqv_out = ggml_cont(ctx_, kqv_out);
    gTN(kqv_out, "%s.kqv_merged_cont", name.c_str());
    kqv_out = ggml_reshape_2d(ctx_, kqv_out, C, N * n_batch);  // [768,17,1]
    // if(isOnlinePush) ggml_build_forward_expand(gf_,kqv_out);
    hGensor t20 = proj_cat.Ming(ctx_, kqv_out);
    gTN(t20, "%s.kqv_out", name.c_str());
    assert_shape_2d(t20, C, N * n_batch);
    cur = ggml_add(ctx_, t20, teb); /**/
#endif
    cur = AfterMing(ctx_, cur, flag);
    return cur;
}

string BROWN_attn::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "%s BROWN_attn %s", tab, moe.Empty() ? "" : "+moe");
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

GatedAttention::GatedAttention(Fish* hG_, const std::string& key_, JSON::const_iterator jit, int flag) : SelfAttention(hG_, key_, jit, flag) {
    auto& config = hG_->config;
    shape        = {n_embd, n_ff};
    tpTrans      = T_RELU2;
    // tpTrans = LINEAR;
    if (jvals.size() > 0)
        tpTrans = (TRANSITION_MODE)(jvals[0]);
}
bool GatedAttention::Build(int flag) {
    norm.BuildX(name + ".norm", {shape[0]}, hFish, 0x0);  // layer->ffn_norm.sT="f";
    upU.BuildX(name + ".upU", {shape[0], shape[1]}, hFish, flag);
    upV.BuildX(name + ".upV", {shape[0], shape[1]}, hFish, flag);
    down.BuildX(name + ".down", {shape[1], shape[0]}, hFish, flag);
    if (attn_mode > 0) {
        assert(0);
        // Q.BuildX(name + ".Q", sp, hFish, flag);
        // K.BuildX(name + ".K", sp, hFish, flag);
        // rope->BuildX(name + ".rope", sp, hFish, flag);
    }

    return true;
}
hGensor GatedAttention::Ming(RLS_BP* ctx_, hGensor inpL, int flag) {
    if (inpL == nullptr) {  // symbolic analysis
        return GeNeuron::BeforeMing(ctx_, nullptr, flag);
    }

    hGensor cur = norm.Ming(ctx_, inpL, 0x0), attn = nullptr;
#ifdef _TENSOR_G_
#else
    gTN(cur, "%s.gau_norm", name.c_str());  // cb(cur, _NAM_("ffn_norm"), il);
    if (attn_mode > 0)
        attn = MyAttention(ctx_, cur, 0x0);  //  [c,c,H,B]

    hGensor Ucur = upU.Ming(ctx_, cur, 0x0);
    hGensor Vcur = upV.Ming(ctx_, cur, 0x0);

    hGensor u = ggml_silu(ctx_, Ucur);
    gTN(u, "%s.u", name.c_str());
    hGensor v = ggml_silu(ctx_, Vcur);
    gTN(v, "%s.v", name.c_str());

    v = vXattn(ctx_, v, attn, 0x100);

    hGensor uv = ggml_mul(ctx_, u, v);
    cur        = down.Ming(ctx_, uv, 0x0);
    cur        = ggml_add(ctx_, cur, inpL);  // add the input
#endif
    cur = AfterMing(ctx_, cur, flag);
    return cur;
}
string GatedAttention::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "%s {GatedAttention attn_mode=(%d trans=%d)}", tab, attn_mode, tpTrans);
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

bool cuAttention::Build(int flag) {
#ifdef _TENSOR_G_

#endif
    return true;
}
hGensor cuAttention::Ming(RLS_BP* ctx_, hGensor inpL, int flag) {
    if (inpL == nullptr) {  // symbolic analysis
        return GeNeuron::BeforeMing(ctx_, nullptr, flag);
    }
    hGensor cur = norm.Ming(ctx_, inpL, 0x0), attn = nullptr;
#ifdef _TENSOR_G_

#endif
    cur = AfterMing(ctx_, cur, flag);
    return cur;
}
string cuAttention::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "%s {cuAttention attn_mode=(%d trans=%d)}", tab, attn_mode, tpTrans);
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

hGensor SelfAttention::vXattn(void* ctx_, hGensor v, hGensor attn, int flag) {
    float kq_scale = 1.0f / sqrtf(float(head_dim)), s;
    if (attn == nullptr) {
        if (isLinear) {
            assert(attn_k != nullptr && attn_q != nullptr);
        } else
            return v;
    }
#ifdef _TENSOR_G_
#else
    if (flag = 0x100) {  // 2d=>4d
        v = ggml_reshape_4d(ctx_, v, n_embd_head, n_head, n_ctx, n_batch);
        v = ggml_cont(ctx_, ggml_permute(ctx_, v, 1, 2, 0, 3));
    }
    gTN(v, "%s.v4", name.c_str());
    hGensor attnv = nullptr;
    if (attn == nullptr) {
        hGensor vk = ggml_mul_mat(ctx_, v, attn_k);
        gTN(vk, "%s.vk", name.c_str());
        vk    = ggml_scale(ctx_, vk, kq_scale);
        vk    = ggml_cont(ctx_, vk);
        attnv = ggml_mul_mat(ctx_, vk, attn_q);

    } else
        attnv = ggml_mul_mat(ctx_, v, attn);
    gTN(v, "%s.vattn", name.c_str());
    hGensor v_merged = ggml_permute(ctx_, attnv, 0, 2, 1, 3);  // eh,h,ctx,b
    gTN0(v_merged, "%s.vehcb", name.c_str());                  // cb(kqv_merged, "kqv_merged", il);
    if (0) {                                                   //  back gradient is zero
        // cur = ggml_cont_2d(ctx_, kqv_merged, n_embd_head_v*n_head, n_tokens);
    } else {
        hGensor kqv_out = ggml_cont(ctx_, v_merged);
        v               = ggml_reshape_2d(ctx_, kqv_out, C, n_tokens);
    }
#endif
    gTN0(v, "%s.kqv_merged_cont", name.c_str());  // cb(cur, "kqv_merged_cont", il);
    return v;
}

/*
BROWN_v0::BROWN_v0(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : SelfAttention(hG_,key_,jit,flag)     {
    auto& config = hG_->config;
    n_rot = config.n_rot;
    rope_freq_base  = config.model.rope_freq_base;
    rope_freq_scale = config.model.rope_freq_scale;
}
bool BROWN_v0::Build(int flag)   {
    // SelfAttention::Build(flag);
    SHAPE sp={shape[0],shape[1]};
    norm.BuildX(name+".norm",{shape[0]},hFish,0x0);
    Q.BuildX(name+".Q",sp,hFish,flag);
    if(Transfer_1)
        V.BuildX(name+".V",{shape[0],1},hFish,flag);  //w = GT(this, typNUMBER::F32, C, 1);
    // K.BuildX(name+".K",sp,hFish,flag);              V.BuildX(name+".V",sp,hFish,flag);
    proj_cat.BuildX(name+".proj",sp,hFish,flag);

    return true;
}
hGensor BROWN_v0::Ming(RLS_BP* ctx_,hGensor teb,int flag)    {
    hGensor cur=BeforeMing(ctx_,teb,flag);

    const float kq_scale = 1.0f/sqrtf(float(C)/n_head);
    int rope = 1,N = n_ctx,n_past=0;;
    hGensor v = teb,v3=nullptr,v4=nullptr, t14 = nullptr, kqv_out=nullptr;
    assert_shape_2d(teb, C, N*n_batch);
    if(0)
        v = W_rope(ctx_,cur,V.w,KQ_pos,{n_embd_head, n_head, N, n_batch},"v",0x1);   //24,6,32,3
    else{
        hGensor v_rope = ggml_reshape_4d   (ctx_, teb, n_embd_head, n_head, N, n_batch);
        gTN(v_rope,"%s.teb",name.c_str());
        v_rope = ggml_rope_ext(ctx_, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        gTN(v_rope,"%s.rope_ext",name.c_str());
        v_rope = ggml_reshape_2d   (ctx_, v_rope, n_embd_head*n_head, N*n_batch);
        v = ggml_mul_mat(ctx_, v_rope, Q.w);    //[144,96]x[144,144]=>[96,144]
    }
    gTN(v,"%s.rope_wq",name.c_str());
    v3 = ggml_reshape_3d(ctx_, v, N, n_batch, C);        gTN(v3, "%s.v3",name.c_str());
    // experts mechanism
    hGensor probs = nullptr;
    if(V.w!=nullptr)   {
        hGensor w_trans = V.w;
        hGensor w_ = ggml_mul_mat(ctx_, w_trans,teb ); //ggml_reshape_2d(ctx,v3,N, n_batch*C)
        gTN(w_,"%s.wvte",name.c_str());
        w_ = ggml_reshape_2d(ctx_, w_, N,n_batch);
        // if(isSiLU){ //maybe useful
        //     w_ = ggml_silu(ctx,w_);
        // }
        probs = ggml_soft_max(ctx_,w_);              gTN(probs,"%s.probs",name.c_str());
        probs = ggml_repeat(ctx_, probs, v3);
    }else
        probs = ggml_soft_max(ctx_,v3);
    hGensor expert = v3;    //ggml_reshape_2d(ctx,v3,n_vocab,n_ctx*n_batch);
    // [32,3,144]x[32,3,144,1]
    hGensor kqv = ggml_mul(ctx_,expert,probs);       gTN(kqv,"%s.kqv",name.c_str());
    v4 = ggml_reshape_4d   (ctx_, kqv,N, n_batch,n_embd_head, n_head);
    kqv_out = ggml_permute(ctx_, v4, 2, 3, 0, 1);       // [24,6,512,32]
    assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);
    // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    gTN(kqv_out, "%s.kqv_out_rope",name.c_str());

    kqv_out = ggml_cont(ctx_, kqv_out);
    gTN(kqv_out, "%s.kqv_merged_cont",name.c_str());
    kqv_out = ggml_reshape_2d   (ctx_, kqv_out, C, N*n_batch);   // [768,17,1]
    // if(isOnlinePush) ggml_build_forward_expand(gf_,kqv_out);
    hGensor t20 = proj_cat.Ming(ctx_,kqv_out);
    gTN(t20, "%s.kqv_out",name.c_str());     assert_shape_2d(t20, C, N*n_batch);
    cur = ggml_add          (ctx_, t20, teb);

    cur = AfterMing(ctx_,cur,flag);
    return cur;
}

string BROWN_v0::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s BROWN_v0",tab);
    if(flag>0)
        _INFO("%s",buf);
    return buf;
};*/

/*
struct ggml_compute_state_shared {
    const struct ggml_cgraph * cgraph=nullptr;
    const struct ggml_cplan  * cplan=nullptr;
    int64_t perf_node_start_cycles=0;
    int64_t perf_node_start_time_us=0;
    const int n_threads=0;
    // synchronization primitives
    atomic_int n_active;  // num active threads
    atomic_int node_n=-1;    // active graph node
    atomic_int node_task=GGML_TASK_TYPE_FINALIZE; // active graph node task phase

    ggml_abort_callback abort_callback=NULL; // abort ggml_graph_compute when true
    void * abort_callback_data=NULL;
    ggml_compute_state_shared() {}
    ggml_compute_state_shared(const struct ggml_cgraph * cg,const struct ggml_cplan  * cp,int nt,int flag=0x0)
        :   cgraph(cg),cplan(cp),n_threads(nt),n_active(nt)  {

    }

};

typedef void * thread_ret_t;
typedef pthread_t ggml_thread_t;
#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

struct ggml_compute_state {
    ggml_thread_t thrd;
    int ith;
    struct ggml_compute_state_shared * shared;
};

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512
struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

struct ggml_context_container {
    bool used;

    struct ggml_context context;
};

struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
    struct ggml_numa_nodes numa;
};

// global state
static struct ggml_state g_state;
// static atomic_int g_state_barrier = 0;

#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static void ggml_graph_compute_thread_sync_node(int * node_n, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_node_n = * node_n;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * node_n = atomic_load(&state->shared->node_n);
        if (* node_n != last_node_n) break;
    }
}
static void ggml_graph_compute_thread_sync_task(int * task_phase, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_task_phase = * task_phase;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * task_phase = atomic_load(&state->shared->node_task);
        if (* task_phase != last_task_phase) break;
    }
}*/

bool GENSOR_TOPU::has(hGensor gensor) {
    assert(nag.size() == infos.size());
    bool b1 = nag.find(gensor->name) != nag.end(), b2 = infos.find(gensor) != infos.end();
    if (b1 != b2) {
        std::vector<hGTensor> gensors_1, gensors_2;
        for (auto gt : nag) {
            gensors_1.push_back(gt.second);
        }
        for (auto gt : infos) {
            gensors_2.push_back(gt.first);
        }
        std::sort(gensors_1.begin(), gensors_1.end(), [](const hGTensor& a, const hGTensor& b) { return strcmp(a->name, b->name) < 0; });
        std::sort(gensors_2.begin(), gensors_2.end(), [](const hGTensor& a, const hGTensor& b) { return strcmp(a->name, b->name) < 0; });
        Gensors2File(gensors_1, "~/gset_1.info");
        Gensors2File(gensors_2, "~/gset_2.info");
        exit(KOIFISH_INVALID_NAG);
    }
    assert(b1 == b2);
    return b2;
}

void GENSOR_TOPU::Insert(hGensor gensor, const GENSOR_INFO& gi, int flag) {
    auto key = gensor->name;
    if (strcmp(key, "model.norm.weight") == 0) {
        DEBUG_HERE;
    }
    // assert(strlen(key)>0);
    assert(nag.find(key) == nag.end());
    nag[key] = gensor;

    assert(infos.find(gensor) == infos.end());
    infos[gensor]    = gi;
    infos[gensor].sX = gensor->name;

    gensor->ginfo = &(infos[gensor]);
}
/*

*/
bool TGraph::TopoOrder(int flag) {
    if (hFish->graph_update > 0) {
        int xxx = 0;
    }

    hFish->gensors.Clear();
    topo_nodes.clear();
    int pos = -1, nDup = 0, i, no = 0, nNode = 0, nLeaf = 0;
#ifdef _TENSOR_G_
    nNode = gset.size();
#else
    nNode = cgraph->n_nodes, nLeaf = cgraph->n_leafs;
    assert(nNode > 0 && nLeaf > 0);
#endif

    hGensor cur, son;

    assert(sinks.size() > 0);
    for (auto r : sinks) {
        topo_nodes.push_back(r);
        hFish->gensors.Insert(r, GENSOR_INFO(0, 0, -1, -1));
        // gimap[r] = GENSOR_INFO(0,0,-1,-1);
        // gimap[r].sX = r->name;
    }
    nNeedGrad = 0;
    while (++pos < topo_nodes.size()) {
        cur = topo_nodes[pos];
        if ((cur->flags & GTensor::F_PARAM) || (cur->flags & GTensor::F_LOSS)) {
            nNeedGrad++;
        }

        if (strcmp(cur->name, "result_output'") == 0) {  // only for debug
            int xxx = 0;
        }
        auto info = hFish->GetGensorInfo(cur);  // gimap[cur];

        for (auto son_ : cur->src) {
            son = son_->_t;
            if (strcmp(son->name, "loss'") == 0) {  // only for debug
                int xxx = 0;
            }
            if (!hFish->gensors.has(son)) {  // gimap.find(son) == gimap.end()
                hFish->gensors.Insert(son, GENSOR_INFO(topo_nodes.size(), info.level + 1, pos, no++));
                // gimap[son] = GENSOR_INFO(topo_nodes.size(),info.level+1,pos,no++);
                // gimap[son].sX = son->name;
                topo_nodes.push_back(son);
            } else {
                nDup++;
            }
        }
    }
    size_t nT = hFish->gensors.size();
    if (isBackward)
        nT += 1;                  //"Loss"
    assert(nT == nNode + nLeaf);  //  271=211+60

    return true;
}

string TGraph::__repr__(string& suffix, string& prefix, hGensor root_0, int flag) {
    const char* tab      = prefix.c_str();
    string root_name     = "";
    const size_t MAX_BUF = 640 * 1024;
    char buf[MAX_BUF]    = "\0";

#ifdef _TENSOR_G_
    if (DEBUG.graph_dump == 1) {
        for (auto gensor : gset) {
            _T_repr_(gensor, tab, buf, hFish->GetGensorInfo(gensor));
            assert(strlen(buf) < MAX_BUF);
        }
        sprintf(buf + strlen(buf), "%s", suffix.c_str());
        _INFO("%s", buf);
    }
    return buf;
#else
    sprintf(buf + strlen(buf), "\n CGRAPH_%s x=%d nodes=%d leafs=%d forward=(%d,%d)\n", name.c_str(), -1, cgraph->n_nodes, cgraph->n_leafs, nForwN, nForwL);
    if (empty()) {
        _INFO("CGRAPH_%s is empty! root=%s", name.c_str(), root_0 == nullptr ? "" : root_0->name);
        return "";
    }

    // the output is always the last tensor in the graph
    int pos = -1, nDup = 0, i, no, nNode = cgraph->n_nodes, nLeaf = cgraph->n_leafs, root_id = root_0 == nullptr ? cgraph->n_nodes - 1 : -1;
    hGensor root = root_0;
    if (root_0 == nullptr) {
        if (isBackward) {
        } else
            root = cgraph->nodes[nNode - 1];
    }
#if !defined(NDEBUG)
#endif
    if (!root_name.empty()) {
        for (int i = 0; i < nNode; i++) {                                  // pick root
            if (strcmp(cgraph->nodes[i]->name, root_name.c_str()) == 0) {  //    l_out-1    inp_embd
                root    = cgraph->nodes[i];
                root_id = i;
                break;
            }
        }
    }
    assert(root != nullptr || sinks.size() > 0);

    hGensor cur, son;
    std::vector<hGensor> all_nodes;
    for (int i = 0; i < nNode; i++) {
        if (strcmp(cgraph->nodes[i]->name, "loss") == 0)
            continue;
        all_nodes.push_back(cgraph->nodes[i]);
    }
    for (int i = 0; i < nLeaf; i++) all_nodes.push_back(cgraph->leafs[i]);

    for (auto gensor : topo_nodes) {
        if (DEBUG.graph_dump == 0)
            _T_repr_(gensor, tab, buf, hFish->GetGensorInfo(gensor));
        assert(strlen(buf) < MAX_BUF);
        if (!hFish->GetGensor(gensor->name)) {
            assert(0);
        }
    }

    sprintf(buf + strlen(buf), "%s", suffix.c_str());
    _INFO("%s", buf);
    int nMiss = all_nodes.size() - topo_nodes.size();
    _INFO("CGRAPH_%s root=%d(%d) nPass=%ld(%d) nMiss=%d", name.c_str(), root_id, sinks.size(), topo_nodes.size(), nDup, nMiss);
    if (CHECK_SAME_TENSORS(name, topo_nodes, all_nodes) != 0x0) {  //       "loss"
        assert(has("loss") >= 0);
        assert(0);
    }
#endif
    return buf;
}

TGraph::TGraph(Fish* hF_, const string& nam_, void* ctx_, bool isGrad, int flag) : hFish(hF_), ctx(ctx_), name(nam_) {
    size_t nVisi = gset.size();
#ifdef _TENSOR_G_
#else
    cgraph = ggml_new_graph_custom(ctx, LLAMA_TRAIN_MAX_NODES, isGrad);
    nodes  = cgraph->nodes;
    grads  = cgraph->grads;
    leafs  = cgraph->leafs;
#endif
}

bool TGraph::empty() {
    // return cgraph==nullptr || cgraph->n_nodes==0;
    return size == 0 || topo_nodes.size() == 0;
}

void TGraph::PushBack(hGensor node, int flag) {
    assert(node != nullptr && strlen(node->name) > 0);
    const char* name = node->name;

    if (gset.find(node) != gset.end())
        return;
    gset.insert(node);
    for (auto child_ : node->src) {
        auto child = child_->_t;
        assert(strlen(child->name) >= 0);
        // gTN(node->src[k], "%s_%d",node->name,k);
        // if(DUMP()) _INFO("\t%s@@@%s\n",child->name,node->name);
        PushBack(child);
    }
    return;
}

bool TGraph::isSink(hGensor node, int flag) {
    for (auto n : sinks) {
        if (n == node)
            return true;
    }
    return false;
}

int Fish::BuildComputeGraph(int order, void* ctx, int flag) {
    const int N = config.n_ctx(), n_past = 0;
    if (N == 19) {
        DEBUG_HERE;
    }

    if (order >= 0) {  // order<0: we have build it in other way
        hForwTG->PushBack(out_node);
        hForwTG->sinks.push_back(out_node);
    }
    if (!hForwTG->isSink(out_node)) {
        _INFO("%s %s is not Sink!!!", __func__, out_node->name);
        return -1;
    }
    assert(hForwTG->isValid());
    hForwTG->TopoOrder();
    hForwTG->__repr__(out_node);

    size_t sz2 = hEDS->AfterBuild(hForwTG, ctx_build);
    if (!isLocalInfer) {
        hBackTG             = std::make_shared<TGraph>(this, hForwTG->name + ".Backward", ctx_build, true);
        hBackTG->isBackward = true;
#ifdef __USE_GGML__
        if (rnd == nullptr) {  // InitModel
            rnd = init_random_normal_distribution(config.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
        }
        assert(ctx == ctx_build);
        struct ggml_cgraph *gf = hForwTG->raw(), *gb = nullptr;
        if (order >= 0) {  // order<0: we have build it in other way
            if (gf != nullptr)
                gf->order = (enum ggml_cgraph_eval_order)order;
        }
        gb = hBackTG->BuildBackward(ctx, hForwTG);
        // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
        int n_leafs_before = gb->n_leafs, n_nodes_before = gb->n_nodes;

#ifndef GG_V12
        auto grad = GradOf(gb, out_node);  // out_node->grad
        ggml_set_input(grad);
#endif
        hBackTG->PushBack(tSCAL(ctx, KQ_pos, 1.0f));

        bool isReforward = false;  // why???
        for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
            gb->leafs[i] = NULL;
        }
        for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
            gb->nodes[i] = NULL;
        }
        gb->n_leafs = n_leafs_before;
        gb->n_nodes = n_nodes_before;
#endif
        // hBackTG->__repr__(out_node);
    }

    // auto leaf0=gf->nodes[0];

    // if (false) { //train_params.use_checkpointing
    //     if(gb!=nullptr) {
    //         gb_tmp = train_params.use_checkpointing ? ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true) : NULL;
    //         // ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    //     }
    // }

    /*if(isLocalInfer){  //gb=nullptr
        assert(gb==nullptr);
        hEDS->AllocGraph(hForwTG);
    } else {

    }*/
    for (auto nn : backbons) {
        for (auto t : nn->PickGensors()) {
            auto& ginfo  = GetGensorInfo(t);
            ginfo.hNeron = nn;
        }
    }

    return 0x0;
}

int TGraph::has(const string& name, int flag) {
    std::map<std::string, int> msg;
#ifdef __USE_GGML__
    int nLeaf = cgraph->n_leafs, nNode = cgraph->n_nodes, no = 1, nDup = 0;
    std::vector<hGensor> gensors, all_nodes;
    for (int i = 0; i < nNode; i++) {
        if (name == cgraph->nodes[i]->name) {
            return i;
        }
        all_nodes.push_back(NEW_(cgraph->nodes[i]));
    }
    for (int i = 0; i < nLeaf; i++) {
        all_nodes.push_back(NEW_(cgraph->leafs[i]));
        if (name == cgraph->leafs[i]->name) {
            return i + nNode;
        }
    }
#endif
    return -1;
}

bool TGraph::isValid() {
    char buf[5012] = "\0";
    std::map<std::string, int> msg;
    std::vector<hGensor> gensors, all_nodes;
    int no = 1, nDup = 0, nNull = 0;

    std::copy(gset.begin(), gset.end(), std::back_inserter(all_nodes));
#ifdef __USE_GGML__
    int nLeaf = cgraph->n_leafs, nNode = cgraph->n_nodes;
    if (nLeaf == 0 && nNode == 0)
        return false;

    for (int i = 0; i < nNode; i++) all_nodes.push_back(NEW_(cgraph->nodes[i]));
    for (int i = 0; i < nLeaf; i++) all_nodes.push_back(NEW_(cgraph->leafs[i]));
#endif
    // for(auto tA:all_nodes){
    for (no = 0; no < all_nodes.size(); no++) {  //
        auto tA = all_nodes[no];
        // if(tA->op==GGML_OP_NONE)    nNull++;
        if (no == 75 || no == 77) {
            // printf("\n%s\n",tA->name);  //grad for block.0.gattn.r2_0
            int only_debug = 0;
        }
        if (msg.find(tA->name) != msg.end()) {
            int j      = msg[tA->name];
            hGensor tB = all_nodes[j];
            assert(strcmp(tA->name, tB->name) == 0);
            _INFO("\tAA_[%d=%d]=\"%s\" !!!\n", j, no, tA->name);
            buf[0] = '\0';
            _T_repr_(tA, "", buf);
            _T_repr_(tB, "", buf);
            _INFO("%s", buf);
            //_pt_cys_("",tA,0);          _pt_cys_("",tB,0);
            nDup++;
        }
        msg[tA->name] = no;
        // no++;
    }
    // if(nDup>0)
    //     return false;
    bool any_params = false, any_loss = false;
    for (auto gensor : all_nodes) {
        any_params = any_params || (gensor->flags & GTensor::F_PARAM);
        any_loss   = any_loss || (gensor->flags & GTensor::F_LOSS);
        if (gensor->flags & GTensor::F_INPUT) {
            nInput++;
        }
    }
    assert(nInput && "No input nodes!");
    if (hFish->isTrain()) {
        assert(any_params && "no trainable parameters found, did you forget to set F_PARAM?");
#ifdef GG_V12
        if (!any_loss) {
            _INFO("Invalid TGraph,no training loss node found, did you forget to set F_LOSS?\n\n");
        }
#endif
    }

    return true;
}
// extern "C" void ggml_compute_backward(void * ctx, struct ggml_cgraph * cgraph, int i, bool * grads_needed);
struct ggml_cgraph* TGraph::BuildBackward(void* ctx_0, hTGraph hFore, bool accumulate, int flag) {
#ifdef __USE_GGML__
    struct ggml_context* ctx_ = (struct ggml_context*)ctx_0;
    auto gf                   = hFore->raw();
    nForwN = gf->n_nodes, nForwL = gf->n_leafs;
    int n_grad = 0, n_param = 0, n_p0 = 0;
    size_t sz                       = gf->size;
    struct ggml_cgraph* gb          = GG_dup_graph(ctx_, gf);
    cgraph                          = gb;  //  ggml_build_backward_expand
    const size_t size_meta          = (3 * hFore->nNeedGrad + 9) * ggml_tensor_overhead();
    struct ggml_context* ctx_static = ggml_init({size_meta, nullptr, true});
    if (DEBUG.back_graph_version == 1) {
        ggml_build_backward_expand(ctx_static, ctx_, gb, accumulate);
        return gb;
    }

    assert(cgraph->n_nodes > 0);
    assert(cgraph->grads);
    assert(cgraph->grad_accs);
    const int n_nodes_f = cgraph->n_nodes, nHash = cgraph->visited_hash_set.size;
    memset(cgraph->grads, 0, cgraph->visited_hash_set.size * sizeof(hGensor));
    memset(cgraph->grad_accs, 0, cgraph->visited_hash_set.size * sizeof(hGensor));
    bool* grads_needed = new bool[nHash]();  // calloc(cgraph->visited_hash_set.size, sizeof(bool));
    // bool accumulate = false;
    for (int i = 0; i < n_nodes_f; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (typNUMBER(node->type) == typNUMBER::I32) {
            continue;
        }

        bool node_needs_grad          = (node->flags & GTensor::F_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS);
        bool ignore_src[GGML_MAX_SRC] = {false};
        switch (node->op) {
            // gradients in node->src[0] for one reason or another have no effect on output gradients
            case GGML_OP_IM2COL:       // only used for its shape
            case GGML_OP_IM2COL_BACK:  // same as IM2COL
                ignore_src[0] = true;
                break;
            case GGML_OP_UNARY: {
                const enum ggml_unary_op uop = ggml_get_unary_op(node);
                // SGN and STEP unary ops are piecewise constant
                if (uop == GGML_UNARY_OP_SGN || uop == GGML_UNARY_OP_STEP) {
                    ignore_src[0] = true;
                }
            } break;

            // gradients in node->src[1] for one reason or another have no effect on output gradients
            case GGML_OP_CPY:            // gradients in CPY target are irrelevant
            case GGML_OP_GET_ROWS:       // row indices not differentiable
            case GGML_OP_GET_ROWS_BACK:  // same as for GET_ROWS
            case GGML_OP_ROPE:           // positions not differentiable
                ignore_src[1] = true;
                break;

            default:
                break;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || ignore_src[j] || !grads_needed[ggml_hash_find(&cgraph->visited_hash_set, node->src[j])]) {
                continue;
            }
            assert(node->src[j]->type == GGML_TYPE_F32 || node->src[j]->type == GGML_TYPE_F16);
            node_needs_grad = true;
            break;
        }
        if (!node_needs_grad) {
            continue;
        }

        // inplace operations are currently not supported
        assert(!node->view_src || node->op == GGML_OP_CPY || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
               node->op == GGML_OP_TRANSPOSE);

        const size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, node);
        assert(igrad != GGML_HASHSET_FULL);
        assert(ggml_bitset_get(cgraph->visited_hash_set.used, igrad));
        if (((node->flags & GTensor::F_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            auto grad = ggml_dup_tensor(ctx_, node);
            // cgraph->grad_accs[igrad] = ggml_dup_tensor(ctx_static, node);
            cgraph->grads[igrad] = grad;  // cgraph->grad_accs[igrad];
            ggml_format_name(grad, "%s\"", node->name);
            // ggml_format_name(cgraph->grad_accs[igrad], "grad acc for %s", node->name);
            if (node->flags & GTensor::F_PARAM) {
                // sinks.push_back(grad);
                n_param++;
            }
        }
        grads_needed[igrad] = true;
        n_grad++;
    }

    for (int i = n_nodes_f - 1; i >= 0; --i) {
        // inplace operations to add gradients are not created by ggml_compute_backward except for gradient accumulation
        // use allocator to automatically make inplace operations
        struct ggml_tensor* node = cgraph->nodes[i];
        // auto grad = GradOf(cgraph,node);
        // if(grad==nullptr)   continue;
        // ggml_compute_backward(ctx_, cgraph, i, grads_needed);
    }

    delete[] grads_needed;
    assert(isValid());
    auto root_b = gb->nodes[gb->n_nodes - 1];
    // for (int i = 0; i < gf->n_nodes; i++) {
    //     hGensor  node = gf->nodes[i];
    //     auto gra_0 = ggml_graph_get_grad(gf,node),grad=ggml_graph_get_grad(cgraph,node);
    //     if (node->flags & GTensor::F_PARAM) {
    //         assert( !(grad->flags & GTensor::F_PARAM) );
    //         gTN(node,"");
    //         PushBack(grad);
    //         sinks.push_back(grad);
    //         n_param++;
    //     }
    // }
    int n_bleafs = gb->n_leafs, n_bnodes = gb->n_nodes;
    // int n_bleafs = gb->n_leafs,n_bnodes = gb->n_nodes;
    // ggml_hash_set_free(&zero_table);

    TopoOrder();
    __repr__();

    return gb;
#else
    return nullptr;
#endif
}

void TGraph::Traverse(int flag) {
#ifdef __USE_GGML__
    bool GGML_OP_HAS_INIT[GGML_OP_COUNT]     = {0};
    bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = {0};
    int node_n                               = -1;
    while (true) {
        if (node_n != -1) { /* FINALIZE */
            struct ggml_tensor* node = cgraph->nodes[node_n];
            if (GGML_OP_HAS_FINALIZE[node->op]) {
                // params.nth = ggml_get_n_tasks(node, n_threads);
            }
        }

        while (++node_n < cgraph->n_nodes) {  // distribute new work or execute it direct if 1T
            _INFO("%s: %d/%d\n", __func__, node_n, cgraph->n_nodes);
            struct ggml_tensor* node = cgraph->nodes[node_n];
            if (1) {
                if (GGML_OP_HAS_INIT[node->op]) {
                    // params.type = GGML_TASK_TYPE_INIT;
                }
                if (GGML_OP_HAS_FINALIZE[node->op]) {
                    // params.type = GGML_TASK_TYPE_FINALIZE;
                    //  ggml_compute_forward(&params, node);
                }
                // ggml_graph_compute_perf_stats_node(node, state->shared);
            } else {
                break;
            }
        }
    }
#endif
}

int TGraph::compute_on_plan(struct ggml_cplan* cplan, int flag) {
    return -1;
    /*return ggml_graph_compute(cgraph, cplan);
    int compute_status = GGML_EXIT_ABORTED;
    assert(cplan);
    assert(cplan->n_threads > 0);
    if (cplan->work_size > 0) {
        assert(cplan->work_data);
    }
    GST_TIC(t0);
#ifdef GGML_USE_VULKAN
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_preallocate_buffers_graph_cpu_assist(cgraph->nodes[i]);
    }
    ggml_vk_preallocate_buffers_cpu_assist();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_build_graph_cpu_assist(cgraph->nodes[i], i == cgraph->n_nodes - 1);
    }
#endif

    const int n_threads = cplan->n_threads;

    struct ggml_compute_state_shared state_shared(cgraph,cplan,n_threads);

    struct ggml_compute_state * workers = (struct ggml_compute_state *)alloca(sizeof(struct ggml_compute_state)*n_threads);

    // create thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; ++j) {
            workers[j].thrd=0;      workers[j].ith=j;   workers[j].shared=&state_shared;
            workers[j] = (struct ggml_compute_state) {
                .thrd   = 0,
                .ith = j,
                .shared = &state_shared,
            };
            if(isOnlySymbol){
                const int rc = pthread_create(&workers[j].thrd, NULL, _graph_pass_thread, &workers[j]);
                assert(rc == 0);                UNUSED(rc);
                // _graph_pass_thread(&workers[j]);
            }   else{
                // const int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
                const int rc = pthread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
                assert(rc == 0);                UNUSED(rc);
            }
        }
    }

    workers[0].ith = 0;
    workers[0].shared = &state_shared;

    const int64_t perf_start_cycles  = ggml_perf_cycles();
    const int64_t perf_start_time_us = ggml_perf_time_us();
    if(isOnlySymbol)
        _graph_pass_thread(&workers[0]);
    else
        compute_status = (size_t) ggml_graph_compute_thread(&workers[0]);

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    // join or kill thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; j++) {
            const int rc = ggml_thread_join(workers[j].thrd, NULL);
            assert(rc == 0);
        }
    }

#ifdef GGML_USE_VULKAN
    ggml_vk_graph_cleanup_cpu_assist();
#endif

    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

        cgraph->perf_runs++;
        cgraph->perf_cycles  += perf_cycles_cur;
        cgraph->perf_time_us += perf_time_us_cur;

        _INFO("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, cgraph->perf_runs,
                (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
                (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
    }
    tCompute = GST_TOC(t0);
    return compute_status;*/
}

void s2layerinfo(struct CLI_params& config, const string& root, const string& jkey, std::vector<string>& lays) {
    lays.clear();
    const char* seps = " ,:;{}()\t=";
    string nam_0;
    char* token = strtok((char*)jkey.c_str(), seps);
    int no = 0, nLay = 1;
    while (token != NULL) {
        if (no == 0) {
            nam_0 = token;
            if (strcasecmp(token, "Layer") == 0) {
                nLay = config.nLayer();
            }
        } else {
            if (token[0] == '*') {
                if (sscanf(token + 1, "%d", &nLay) == 1) {
                } else {
                }
            }
        }
        token = strtok(NULL, seps);
        no++;
    }
    // const string sL=".";  //".L"

    if (nLay > 1) {
        for (int i = 0; i < nLay; i++) {
            // string nam_ = config.NameOnArch(nam_0);
            string name = config.model.sLayer + std::to_string(i);
            lays.push_back(name);
        }
    } else {
        lays.push_back(nam_0);
    }
}

int TGraph::curLayer = -1;
hNEURON Fish::J2Neuron(void* ctx_, string& dad, int level, const JConfig& jconfig, int flag) {
    hNEURON hN = nullptr, cur = nullptr;
    std::vector<hNEURON> vNN;
    string k, nam_, prefix;
    std::vector<string> lay_names;
    int i, nLay;

    for (JSON::const_iterator it = jconfig.js.begin(); it != jconfig.js.end(); ++it) {
        k = it.key();
        if (!k.empty() && k[0] == '#')
            continue;
        auto v = it.value();
        if (it->is_array()) {
        } else if (it->is_structured()) {
            s2layerinfo(config, dad, k, lay_names);
            int lay = 0;
            for (auto nam_ : lay_names) {
                if (level == 0) {
                    TGraph::curLayer++;
                }
                JConfig jLay(*it, lay++);
                prefix = dad.empty() ? nam_ : dad + "." + nam_;  //  ,  //nam_
                cur    = J2Neuron(ctx_, prefix, level + 1, jLay, flag);
                vNN.push_back(cur);
            }
            continue;
        } else {
            assert(0);
        }
        cur        = GeNeuron::MakeInstance(this, ctx_, dad, it, flag);
        cur->ID    = jconfig.ID;
        cur->level = level + 1;
        neurons.push_back(cur);
        vNN.push_back(cur);
    }
    assert(vNN.size() > 0);

    if (vNN.size() > 1) {
        hN        = std::make_shared<Ganglia>(this, dad, vNN, flag);
        hN->level = level;
        neurons.push_back(hN);
    } else {
        assert(cur != nullptr);
        hN = cur;
    }
    assert(hN->isValid());
    return hN;
}

/*

*/
int Fish::jToGraph(void* ctx_, bool isBuild, int flag) {
    JConfig js(config.jBackBone);
    // JConfig js(config.jModel);
    string sRoot = "model";  //  prefix = dad.empty() ? nam_ : dad + "." + nam_;

    AllocBuffer();
    int L = config.nLayer();
    J2Neuron(ctx_, sRoot, 0, js, flag);
    L = config.nLayer();
    for (auto n : neurons) {
        if (dynamic_cast<Ganglia*>(n.get()) != nullptr)
            continue;
        backbons.push_back(n);
    }
    assert(backbons.size() == 2 * L + 3);
    // RLScheduling
    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();

    // FFN* last_ffn = GetNeuron<FFN>("FFN", L - 1);
    // for (int l = L - 1; l >= 0; l--) {
    //     SelfAttention* QKV = GetNeuron<SelfAttention>("SelfAttention", l);
    //     FFN* ffn           = GetNeuron<FFN>("FFN", l);
    //     ffn->delta         = nullptr;
    // }

    hCLS = GetNeuron<OutCLS>("OutCLS", 0);
    assert(hCLS != nullptr);
    if (config.model.token_embeds.size() > 1) {
        auto nn = std::make_shared<MAEC>(this, "MAEC", flag);
        neurons.push_back(nn);
        TokenEmbed* embed = GetNeuron<TokenEmbed>("TokenEmbed", 0);
        embed->SetMAEC(nn);
        hCLS->maec = nn;
    }

    // Only symbolic analysis
    string suffix, prefix;
    int no      = 0;
    hGensor cur = in_node;     // tBatch;
    for (auto nn : neurons) {  // Symbolic Ming
        no++;
        assert(cur != nullptr);
        if (nn->name != "model.output_norm") {  //"model.output_norm"     "model.blk.31.attn"
            int only_for_debug = 0;
        }
        double t0 = GST_ms(), a;
        cur       = nn->Ming(hRLS, cur);
        if (nn->isGang()) {
        } else {
            // _INFO("%d\t%s\n",no,nn->__repr__(suffix,prefix).c_str());
        }
        if ((a = (GST_ms() - t0)) > 1000)  // why take so long?
            _INFO("\t%s T=%.3gs\n", nn->name.c_str(), a / 1000.0);
    }

    hRLS->Init(this, backbons);

    return 0x0;
}
