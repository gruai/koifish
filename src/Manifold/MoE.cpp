/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Mixture of Expert
 *
 *  \brief Mixture of Expert
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

LLM_MOE::LLM_MOE(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_MOE);

    tpFFN = FFN_TYPE::GATE_CYS;
    // tpFFN = FFN_TYPE::SMOE;
}

size_t LLM_MOE::MostMemSize(int flag) {
    int n_layer = config.nLayer();
    int nHead   = hDictVAE != nullptr ? hDictVAE->nLevel * 3 + 2 + 6 : 6;
    size_t sz0 = GTensor::MostOverhead(), sz = sz0 * 2 * (nHead + n_layer * 18);
    return sz;
}

hGensor MixOfModels::Forward(void *ctx, hGensor cur, hGensor w) {
#ifdef _TENSOR_G_
    return nullptr;
#else
    int n_vocab = cur->ne[0], n_ctx = cur->ne[1], n_batch = cur->ne[2];
    hGensor expert = ggml_reshape_2d(ctx, cur, n_vocab, n_ctx * n_batch);
    hGensor wB     = _repeat(ctx, w, expert);
    gTN(wB, "%s_r", w->name);
    return ggml_mul(ctx, expert, wB);
#endif
}

void MixOfSwarm::Init(tpSWARM &swarm, void *ctx, int n_embd, int flag) {
    hGensor a = nullptr, b = nullptr;
    for (auto fish : swarm) {
        b = fish->Output();
        // assert(b!=nullptr && b->data!=nullptr);
        exs.push_back(b);
        if (a == nullptr) {
            a = b;
        } else {
            // assert(ggml_are_same_shape(a, b));
        }
    }
    gat_ = GT(nullptr, typNUMBER::F32, {n_embd, (int)(swarm.size() + 1)});
}

hGensor MixOfSwarm::Build(CLI_params &config, void *ctx, hGensor cur, int flag) {
    hGensor ouput = nullptr;
#ifdef _TENSOR_G_
#else
    // return cur;
    int n_batch = config.common.n_batch, n_ctx = config.common.n_ctx, n_embd = config.nEmbed();
    int i = 0, nSwarm = exs.size() + 1;
    size_t offset = gat_->nb[0];
    assert(offset == 4);
    size_t N0 = cur->ne[0], ld1 = (nSwarm)*offset, nToken = cur->ne[1], N1 = nToken;
    assert(nSwarm == gat_->ne[1]);

    //[nWiki+1, n_tokens] := [n_embed,nWiki+1]x[n_embed,n_tokens]
    hGensor w_ = ggml_mul_mat(ctx, gat_, ggml_reshape_2d(ctx, cur, N0, nToken));
    if (isSiLU) {  // maybe useful
        w_ = ggml_silu(ctx, w_);
    }
    hGensor probs = ggml_soft_max(ctx, w_);
    // hGensor tA = ggml_reshape_2d(ctx,curlogits,n_vocab,n_ctx*n_batch);      //[nVocab, n_ctx*nBatch]
    hGensor wA = ggml_view_2d(ctx, probs, 1, nToken, ld1, 0);  //[1, n_ctx*nBatch] ne0,ne1,nb1,offset
    //  wA = _repeat(ctx,wA,tA);
    ouput = Forward(ctx, cur, wA);  // ggml_mul(ctx,tA,wA);
    for (auto tB : exs) {
        i++;
        // float *logistB = (float*)(tB->data);    //always nullptr
        hGensor wB     = ggml_view_2d(ctx, probs, 1, nToken, ld1, offset * i);  // ne0,ne1,nb1,offset
        hGensor expert = Forward(ctx, tB, wB);
        // wB = _repeat(ctx,wB,expert);
        ouput = ggml_add(ctx, ouput, expert);  // ggml_mul(ctx,expert,wB)
    }

    // ouput = ggml_reshape_3d        (ctx, ouput, N0, n_ctx, n_batch);
    if (isRes) {
        ouput = ggml_add(ctx, ouput, cur);
    }
#endif
    return ouput;
}

hGensor NLP_AutoRegressive::build_gate(void *ctx, hGensor cur, hGensor curlogits, int flag) {
    hGensor ouput = nullptr;
#ifdef _TENSOR_G_
#else
    bool isRes = true, isSiLU = false;

    int n_vocab = hDictVAE->tVocab(), n_batch = config.common.n_batch, n_ctx = config.common.n_ctx, n_embd = config.nEmbed();
    int nWiki = wikis.size(), i = 0;
    CHILD_0909_WIKIS
    size_t N0 = cur->ne[0], ld1 = (nWiki + 1) * cur->nb[0];
    assert(nWiki + 1 == mom.embed2w->ne[1]);

    hGensor gat_ = mom.embed2w;
    //[nWiki+1, n_tokens] := [n_embed,nWiki+1]x[n_embed,n_tokens]
    hGensor w_ = ggml_mul_mat(ctx, mom.embed2w, ggml_reshape_2d(ctx, cur, N0, n_ctx * n_batch));
    if (isSiLU) {  // maybe useful
        w_ = ggml_silu(ctx, w_);
    }
    hGensor probs = ggml_soft_max(ctx, w_);
    gTN(probs, "gate_probs");
    size_t offset = probs->nb[0];
    // hGensor tA = ggml_reshape_2d(ctx,curlogits,n_vocab,n_ctx*n_batch);      //[nVocab, n_ctx*nBatch]
    hGensor wA  = ggml_view_2d(ctx, probs, 1, n_ctx * n_batch, ld1, 0);  //[1, n_ctx*nBatch] ne0,ne1,nb1,offset
    wA->name[0] = '\0';
    gTN(wA, "gate_probs_000");
    //  wA = _repeat(ctx,wA,tA);
    ouput = mom.Forward(ctx, curlogits, wA);  // ggml_mul(ctx,tA,wA);
    gTN(ouput, "gate_ouput_0");
    for (auto wiki : wikis) {
        i++;
        hGensor tB     = wiki->exLogits;
        float *logistB = (float *)(tB->data);
        if (wiki->t2t != nullptr) {
            tB = ggml_mul_mat(ctx, wiki->t2t, tB);
            gTN(tB, "gate_tB_%d", i);
        } else {
            // tB = ggml_soft_max(ctx,tB);  maybe bad choice
            gTN(tB, "gate_exLogits_%d", i);
        }
        hGensor wB  = ggml_view_2d(ctx, probs, 1, n_ctx * n_batch, ld1, offset * i);  // ne0,ne1,nb1,offset
        wB->name[0] = '\0';
        gTN(wB, "gate_probs_%d", i);
        hGensor expert = mom.Forward(ctx, tB, wB);
        gTN(expert, "gate_expert_%d", i);
        // wB = _repeat(ctx,wB,expert);
        ouput = ggml_add(ctx, ouput, expert);  // ggml_mul(ctx,expert,wB)
        gTN(ouput, "gate_ouput_%d", i);
    }

    ouput = ggml_reshape_3d(ctx, ouput, n_vocab, n_ctx, n_batch);
    if (isRes) {
        ouput = ggml_add(ctx, ouput, curlogits);
        gTN(ouput, "gate_ouput+");
    }
#endif
    return ouput;
}
