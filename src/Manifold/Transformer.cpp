#include "../Device/Pipe.hpp"
#include "../g_float.hpp"
#include "../g_float_cpu.hpp"
#include "Fish.hpp"
// we only support fp16 kv cache by default; this can be changed to float with a recompile
typedef __gcc_fp16 kvtype_t;
typedef f8e5m2_t tpEMBED;  //	f8e5m2_t or __gcc_fp16
typedef f8e5m2_t tpW;

/*
inline __gcc_fp16 fp82half(unsigned char v) {
    union {
        unsigned short u;
        __gcc_fp16 f;
    } u;
    u.u = v << 8;
    return u.f;
}


    5 exponent bits and 2 mantissa bits (plus an implicit leading 1 for normalized numbers)

inline float fp8e5m2_to_float(f8e5m2_t x) {
    __gcc_fp16 val = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(&val, &x, sizeof(f8e5m2_t));
#else
    memcpy((char*)&val + sizeof(f8e5m2_t), &x, sizeof(f8e5m2_t));
#endif
    return half_to_float(val);
}*/
inline float gf4_ff(uint32_t v, int k) {
    f8e5m2_t a = v & 0xff;
    float s    = T2Float(&a);  // fp82half(v & 0xff)
    s          = s / -4.f;
    return ((int)((v >> (8 + k * 3)) & 7) - 4) * s;
}

void attn(float *xout, float *atth, float *qh, kvtype_t *kh, kvtype_t *vh, int head_dim, int kv_dim, int kv_len) {
    float score_max = -FLT_MAX;

    // calculate attention scores as dot products of q and k; also track score max for this head
    for (int t = 0; t < kv_len; ++t) {
        float score = 0.0f;
        for (int j = 0; j < head_dim; ++j) {
            score += qh[j] * kh[t * kv_dim + j];
        }
        score /= sqrtf(head_dim);
        score_max = (score_max < score) ? score : score_max;
        atth[t]   = score;
    }

    // softmax the scores to get attention weights over [0..kv_len)
    float score_sum = 0.f;
    for (int t = 0; t < kv_len; ++t) {
        atth[t] = expf(atth[t] - score_max);
        score_sum += atth[t];
    }

    // mix values with attention weights
    for (int j = 0; j < head_dim; ++j) {
        float res = 0.f;
        for (int t = 0; t < kv_len; ++t) {
            res += (atth[t] / score_sum) * vh[t * kv_dim + j];
        }
        xout[j] = res;
    }
}

// int Hot_Update(int dim,float *hb,int *hot,int flag=0x0);

//	transfer data between device & host
KERNEL_PIPE<uint32_t, void> *tPipe = nullptr;

int FFN::CPU_v0(void *ctx, int layer, int flag) {
    auto config                        = hFish->config;
    KERNEL_PIPE<uint32_t, void> *tPipe = (KERNEL_PIPE<uint32_t, void> *)(ctx);
    int wbit = BitPE(config.model.tpWeight), dim = tPipe->dim, nHot = -1, hidden_dim = tPipe->hidden_dim;
    dotprod_t dotprod = fnDot(config.model.tpWeight);  // wbit == 4 ? dotprod_gf4 : (wbit == 8 ? dotprod_fp8 : dotprod_fp16);
    floatX *x = tPipe->x, *xb = tPipe->xb, *xb2 = tPipe->xb2, *exp = tPipe->exp, *hb = tPipe->hb, *hb2 = tPipe->hb2;
    float rmsscale, val, t0, t1;
    const CoopLayer<tpW> *w = (const CoopLayer<tpW> *)(tPipe->cLayers + layer);
    hCSPicker hPicker       = nullptr;  // std::make_shared<CS_Picker>(hFish);
    floatX *moe_weights     = exp + tPipe->n_experts;
    int *moe_experts        = (int *)moe_weights + (tPipe->n_experts_ac ? tPipe->n_experts_ac : 1);
    assert(tPipe->n_experts == 0);
    if (tPipe->n_experts) {
        assert(0);  // moe gate
                    // matmul(exp, xb, w->moegate, NULL, dim, tPipe->n_experts, dotprod);
                    // moe_gate(moe_weights, moe_experts, exp, tPipe->n_experts, tPipe->n_experts_ac);
    } else {
        moe_weights[0] = 1.0f;
        moe_experts[0] = 0;
    }

    for (int e = 0; e < (tPipe->n_experts_ac ? tPipe->n_experts_ac : 1); ++e) {
        t0           = GST_us();
        size_t esize = dim * tPipe->hidden_dim * (size_t)wbit / 8;
        // matmul(hb, xb, (char*)w->w1 + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);
        up.Forw(hb, xb, 0x0);
        // matmul(hb2, xb, (char*)w->w3 + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);
        gate.Forw(hb2, xb, 0x0);
        if (tPipe->act_gelu) {  // GEGLU non-linearity
            for (int i = 0; i < hidden_dim; i++) {
                hb[i] = gelu(hb[i]) * hb2[i];
            }
        } else {  // SwiGLU non-linearity
            for (int i = 0; i < hidden_dim; i++) {
                hb[i] = silu(hb[i]) * hb2[i];
            }
        }
        nHot = hPicker == nullptr ? 0 : hPicker->Update(layer, hb);
        // ffn->OnData(GT({dim},xb,typNUMBER::F32),GT({hidden_dim},hb,typNUMBER::F32),hot);
        // matmul(xb2, hb, (char*)w->w2 + moe_experts[e] * esize, NULL, hidden_dim, dim, dotprod);
        down.Forw(xb2, hb, 0x0);
        Fish::stat.tFFN += GST_us() - t0;
        for (int i = 0; i < dim; i++) {
            x[i] += xb2[i] * moe_weights[e];
        }
    }
    return 0x0;
}
int T_generate_cpu(hFISH hFish, bool isOnlyUpdateKV, unsigned flags) {
    assert(hFish != nullptr);
    auto config       = hFish->config;
    TokenEmbed *embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    int token = embed->hBatch->CurToken(), pos = embed->hBatch->pos++;
    if (tPipe == nullptr) {
        tPipe = new KERNEL_PIPE<uint32_t, void>(hFish, 0x0);
    } else
        tPipe->UpdatePos(pos);

    int wbit = BitPE(config.model.tpWeight), dim = tPipe->dim, kv_dim = tPipe->kv_dim, q_dim = tPipe->q_dim, hidden_dim = tPipe->hidden_dim;
    int kv_mul = tPipe->n_heads / tPipe->n_kv_heads, kv_pos = tPipe->kv_pos, kv_len = tPipe->kv_len, kv_sink = 0, nHot = -1;
    hCSPicker hPicker = std::make_shared<CS_Picker>(hFish);
    // int *hot=new int[hidden_dim*2]();
    // for(int i=0;i<hidden_dim;i++)	hot[i] = 1;
    // float *dTemp = new float[hidden_dim+dim*hidden_dim];
    dotprod_t dotprod              = wbit == 4 ? dotprod_gf4 : (wbit == 8 ? dotprod_fp8 : dotprod_fp16);
    tpEMBED *token_embedding_table = TO<tpEMBED>(embed->w);
    OutCLS *cls                    = hFish->GetNeuron<OutCLS>("OutCLS", 0);
    LayerNormal *lnf               = hFish->GetNeuron<LayerNormal>("LayerNormal", 0);
    float *logits = TO<float>(cls->preLogits), *rms_final_weight = TO<float>(lnf->w);  // (dim,);
    float *x = tPipe->x, *xb = tPipe->xb, *xb2 = tPipe->xb2, *exp = tPipe->exp, *hb = tPipe->hb, *hb2 = tPipe->hb2, rmsscale, val, t0, t1;
    // copy the token embedding into x
    char *content_row = (char *)token_embedding_table + token * dim * (size_t)wbit / 8;
    if (wbit == 4) {
        for (int i = 0; i < dim; i += 8) {
            uint32_t wg = ((uint32_t *)content_row)[i / 8];
            for (int k = 0; k < 8; ++k) {
                x[i + k] = gf4_ff(wg, k);
            }
        }
    } else {
        for (int i = 0; i < dim; ++i) {
            x[i] = wbit == 8 ? T2Float(content_row + i) : ((__gcc_fp16 *)content_row)[i];
        }
    }
    assert(isValidF(dim, x));
    const CoopLayer<tpW> *w0 = (const CoopLayer<tpW> *)(tPipe->cLayers);

    for (int lay = 0; lay < tPipe->n_layers; lay++) {
        int curLay              = lay;  //(lay/2)*2;
        SelfAttention *QKV      = hFish->GetNeuron<SelfAttention>("SelfAttention", curLay);
        FFN *ffn                = hFish->GetNeuron<FFN>("FFN", curLay);
        const CoopLayer<tpW> *w = (const CoopLayer<tpW> *)(tPipe->cLayers + curLay);
        // attention rmsnorm
        rmsscale = rmsnorm(xb, x, w->rms_att_weight, dim, tPipe->norm_eps, tPipe->norm_ln);
        // key and value point to the kv cache
        size_t loff  = (size_t)lay * tPipe->seq_len * kv_dim;  // kv cache layer offset for convenience
        kvtype_t *kb = (kvtype_t *)tPipe->key_cache + loff;
        kvtype_t *vb = (kvtype_t *)tPipe->val_cache + loff;
        if (curLay == -1) {  // only for debug
            PrintT<tpW>("wk", w0->wk, dim, 1);
            PrintT<float>("xb", xb, dim, 1);
            val = dotprod(w0->wk, dim, 0, xb);
            PrintT<float>("q", tPipe->q, q_dim, 1);
            PrintT<float>("k", tPipe->k, kv_dim, 1);
        }  // qkv matmuls for this position
        D_matvec(tPipe->q, xb, w->wq, w->bqkv, dim, q_dim, dotprod);
        D_matvec(tPipe->k, xb, w->wk, w->bqkv ? w->bqkv + q_dim : NULL, dim, kv_dim, dotprod);
        D_matvec(tPipe->v, xb, w->wv, w->bqkv ? w->bqkv + q_dim + kv_dim : NULL, dim, kv_dim, dotprod);
        for (int i = 0; i < q_dim; i++) {  // some models require clipping qkv values
            tPipe->q[i] = clip(tPipe->q[i], tPipe->qkv_clip);
        }
        for (int i = 0; i < kv_dim; i++) {
            tPipe->k[i] = clip(tPipe->k[i], tPipe->qkv_clip);
            tPipe->v[i] = clip(tPipe->v[i], tPipe->qkv_clip);
        }

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        rope(tPipe->q, q_dim, tPipe->head_dim, pos, tPipe->rope_theta, tPipe->rotary_dim);
        rope(tPipe->k, kv_dim, tPipe->head_dim, pos, tPipe->rope_theta, tPipe->rotary_dim);
        // update kv cache
        for (int i = 0; i < kv_dim; i++) {
            kb[kv_pos * kv_dim + i] = tPipe->k[i];
            vb[kv_pos * kv_dim + i] = tPipe->v[i];
        }

        // rotate sink tokens forward to keep pace with non-sink tokens
        for (int r = 0; r < kv_sink; r++) {
            for (int i = 0; i < kv_dim; i++) {
                tPipe->k[i] = kb[r * kv_dim + i];
            }

            rope(tPipe->k, kv_dim, tPipe->head_dim, 1, tPipe->rope_theta, tPipe->rotary_dim);

            for (int i = 0; i < kv_dim; i++) {
                kb[r * kv_dim + i] = tPipe->k[i];
            }
        }
        t0 = GST_us();
        // multihead attention. iterate over all heads
        int h;
#pragma omp parallel for private(h)
        for (h = 0; h < tPipe->n_heads; h++) {
            float *qh    = tPipe->q + h * tPipe->head_dim;
            float *atth  = tPipe->att + h * tPipe->seq_len;
            kvtype_t *kh = kb + (h / kv_mul) * tPipe->head_dim;
            kvtype_t *vh = vb + (h / kv_mul) * tPipe->head_dim;

            attn(xb2 + h * tPipe->head_dim, atth, qh, kh, vh, tPipe->head_dim, kv_dim, kv_len);
        }
        Fish::stat.tQKV += GST_us() - t0;
        if (curLay == -1) {
            float val = dotprod(w->wo, q_dim, 0, xb2);    //	-0.165075183
            PrintT<f8e5m2_t>("wout", w->wo, q_dim, dim);  //	only for debug
        }
        D_matvec(hb, xb2, w->wo, NULL, q_dim, dim, dotprod);  // final matmul to get the output of the attention
        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += hb[i];  //	-0.170934558
        }

        if (!tPipe->norm_par) {  // ffn rmsnorm
            rmsnorm(xb, x, w->rms_ffn_weight, dim, tPipe->norm_eps, tPipe->norm_ln);
        }
        if (DEBUG.T_cpu == 2) {
            ffn->CPU_v0(tPipe, curLay);
        } else {
            float *moe_weights = exp + tPipe->n_experts;
            int *moe_experts   = (int *)moe_weights + (tPipe->n_experts_ac ? tPipe->n_experts_ac : 1);
            assert(tPipe->n_experts == 0);
            if (tPipe->n_experts) {
                assert(0);  // moe gate
                            // matmul(exp, xb, w->moegate, NULL, dim, tPipe->n_experts, dotprod);
                            // moe_gate(moe_weights, moe_experts, exp, tPipe->n_experts, tPipe->n_experts_ac);
            } else {
                moe_weights[0] = 1.0f;
                moe_experts[0] = 0;
            }
            // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
            for (int e = 0; e < (tPipe->n_experts_ac ? tPipe->n_experts_ac : 1); ++e) {
                size_t esize = dim * hidden_dim * (size_t)wbit / 8;
                D_matvec(hb, xb, (char *)w->w1 + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);
                D_matvec(hb2, xb, (char *)w->w3 + moe_experts[e] * esize, NULL, dim, hidden_dim, dotprod);
                if (tPipe->act_gelu) {  // GEGLU non-linearity
                    for (int i = 0; i < hidden_dim; i++) {
                        hb[i] = gelu(hb[i]) * hb2[i];
                    }
                } else {  // SwiGLU non-linearity
                    for (int i = 0; i < hidden_dim; i++) {
                        hb[i] = silu(hb[i]) * hb2[i];
                    }
                }
                D_matvec(xb2, hb, (char *)w->w2 + moe_experts[e] * esize, NULL, hidden_dim, dim, dotprod);
                for (int i = 0; i < dim; i++) {
                    x[i] += xb2[i] * moe_weights[e];
                }
            }
        }

        int checkpoint = 0;
    }

    if (isOnlyUpdateKV) {  // flags & FF_UPDATE_KV_ONLY
        // only update kv cache and don't output logits
        return 0x0;
    }

    // final rmsnorm
    rmsnorm(x, x, rms_final_weight, dim, tPipe->norm_eps, tPipe->norm_ln);
    // classifier into logits
    D_matvec(logits, x, TO<f8e5m2_t>(tPipe->out_weight), NULL, tPipe->dim, tPipe->vocab_size, dotprod);
    // PrintTensor<float>("logits",logits,tPipe->vocab_size,1);//only for debug

    return 0x0;
}