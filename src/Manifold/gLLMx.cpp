/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  More models derived from NLP_AutoRegressive
 *
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

LLM_MAMBA::LLM_MAMBA(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_MAMBA);
    bool worst_case = true;
    // isLoadTokenEmbed = true;
    // config.common.adam.alpha = 0.0001;     //
}

hGensor LLM_MAMBA::BuildTarget(void *ctx, hGensor cur, int flag) { return nullptr; }

Guppy::Guppy(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_GUPPY);
    // config.model.isFFNShareParam = true;
    // config.model.isEmbedWeightTying = false;
    // isBias = config.model.isBias;    //   if true, converge much slower
}

string Guppy::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    string sBasic   = NLP_AutoRegressive::__repr__(suffix, prefix, flag);
    sprintf(buf + strlen(buf), "%s", sBasic.c_str());
    _INFO("Guppy:    Bias=(normal=%d,slp=%d) AttOnBC=%d\n========\n", config.model.isNormalBias, config.model.isSLPBias, isAttOnBC);
    return buf;
}

// static uint64_t rng_seed = 42;

bool Guppy::BeforeNextStep(int iter, int flag) {
    int nLayer = config.nLayer(), l;
    for (l = 0; l < nLayer; l++) {
        FFN *ffn = GetNeuron<FFN>("FFN", l);
        ffn->UpdateSamps(iter * nLayer + l);
        // SelfAttention *qkv = GetNeuron<SelfAttention>("FFN",l);
    }
    return true;
}

string Guppy::DebugInfo(int type, int flag) {
    char buf[5012] = "\0";
    sprintf(buf + strlen(buf), "|gw|=(%.2f,%.2f)", hEmbed->w->gnorm, hEmbed->wInv->gnorm);
    return buf;
}

GPT2::GPT2(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_GPT2 || arch == MODEL_ARCH::NLP_GPT2_char);
    // isBias = config.model.isBias;    //   if true, converge much slower
}

/*
GPT(
  (transformer): ModuleDict(
    (wte): Embedding(65, 768)
    (wpe): Embedding(256, 768)
    (drop): Dropout(p=0, inplace=False)
    (h): ModuleList(
      (0): Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=False)
          (c_proj): Linear(in_features=768, out_features=768, bias=False)
          (attn_dropout): Dropout(p=0, inplace=False)
          (resid_dropout): Dropout(p=0, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=3072, out_features=768, bias=False)
          (dropout): Dropout(p=0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=768, out_features=65, bias=False)
)

*/
/*
int DTS_GPT2::STR2T(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag)    {
    // bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, const gpt_params & params) {
    gpt_vocab vocab;
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);
    return 0x0;
}*/

CDict_GPT2::CDict_GPT2(Fish *nlp_, int flag) : DictVAE(nlp_, flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();
    // n_vocab=50257;
}

CDict_CHAR::CDict_CHAR(Fish *nlp_, int flag) : DictVAE(nlp_, flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();
    // n_vocab=256;
}
int CDict_CHAR::InitMAEC(void *ctx_build, const std::vector<int> &dims_, int flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();

    assert(gensors.size() == 0);
    return 0x0;
}
int CDict_CHAR::STR2T(const char *txt, int txt_len, std::vector<TOKEN_ID> &btch, int flag) {
    int n_tokens = 0, nMost = btch.size();
    assert(txt_len <= nMost);
    unsigned char *a = (unsigned char *)(txt);
    for (int i = 0; i < txt_len; i++, a++) {
        TOKEN_ID t = (TOKEN_ID)(*a);
        // assert(t>=0 && t<n_vocab);
        btch[i] = t;
        n_tokens++;
    }
    return n_tokens;
}
std::string CDict_CHAR::T2STR(TOKEN_ID tok, int flag) {
    assert(tok >= 0 && tok < 256);
    string a = string(1, (char)tok);
    return a;
};

int CDict_GPT2::InitMAEC(void *ctx_build, const std::vector<int> &dims_, int flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();

    // tok_embeddings = dolphin->AddTensor(ctx_build,_NAM_("token_embd.weight"),typNUMBER::F32,{n_embd, n_vocab},true,0x0);
    // _norm.BuildX("output_norm", {n_embd},dolphin,0x0);
    // _output.isBias = false;
    // _output.BuildX("output", {n_embd, n_vocab},dolphin,0x0);
    // assert(gensors.size()==0);
    return 0x0;
}

#ifdef _TENSOR_G_
#else
static void cb(hGensor cur, const char *name, int il) {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }

    /*if (!lctx.cparams.offload_kqv) {
        if (strcmp(name, "kqv_merged_cont") == 0) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
        }
    }*/

    // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
    // FIXME: fix in ggml_backend_sched
    const bool full_offload = false;  // lctx.n_gpu_layers > (int)lctx.config.n_layer;
    if (full_offload) {               // batch.n_tokens < 32 ||
        if (il != -1 && strcmp(name, "norm") == 0) {
            /*for (auto * backend : lctx.backends) {
                if (ggml_backend_supports_buft(backend, lctx.buft_layer[il].buft) &&
                    (ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur))) {
                    ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                    break;
                }
            }*/
        }
    }
};
int GPT2::cRawGraph(void *ctx_build, bool isBuild, int flag) {
    bool isOnlinePush     = true;  // push nodes last or online(QKV)
    const int n_embd_head = config.n_embd_head_v, n_embd_gqa = config.n_embd_v_gqa();
    assert(n_embd_head == config.n_embd_head_k);
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed(), n_vocab = hDictVAE->n_vocab;
    int n_tokens = n_ctx * n_batch, n_head, n_embd_head_v = config.n_embd_head_v, n_ff;
    const uint32_t n_layer = config.nLayer();
    for (int i = 0; i < n_layer; i++) {
        auto layer    = std::make_shared<QKV_LAY>(this, i);
        layer->isLast = i == n_layer - 1;
        layers.push_back(layer);
    }
    hGensor cur = nullptr, pos = nullptr, q, k, v, Qcur, Kcur, Vcur;
    float kq_scale = 1.0f / sqrtf(float(n_embd_head));
    // hDictVAE->InitMAEC(ctx_build,{n_embd},0x0);
    hGensor tok_embeddings = AddTensor(ctx_build, _NAM_("token_embd.weight"), typNUMBER::F32, {n_embd, n_vocab}, true, 0x0);
    _norm.BuildX("output_norm", {n_embd}, this, 0x0);
    _output.BuildX("output", {n_embd, n_vocab}, this, 0x0);

    pos_embd = AddTensor(ctx_build, _NAM_("position_embed.weight"), typNUMBER::F32, {n_embd, n_ctx}, true,
                         0x0);  // ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});
    // llm_build_inp_embd(ctx_build, lctx, config, batch, tok_embd, cb);

    hGensor inpL = ggml_get_rows(ctx_build, tok_embeddings, in_node);
    gTN(inpL, "inp_embd");

    if (0) {
        hGensor inp_pos = KQ_pos;
        pos             = ggml_get_rows(ctx_build, pos_embd, KQ_pos);
        cb(pos, "pos_embd", -1);
        float *fpos = (float *)(pos->data);
        inpL        = ggml_add(ctx_build, inpL, pos);
    } else {
        inpL = ggml_add(ctx_build, inpL, pos_embd);
    }
    cb(inpL, "inp_pe", -1);
    auto gf = GetForwRaw();
    // layers.clear();     cur = inpL;        xn=cur;      xxn=cur->grad;   //only for debug
    for (int il = 0; il < layers.size(); ++il) {
        n_head = config.n_head(il);
        assert(n_embd_head * n_head == n_embd);
        n_ff = config.n_ff(il);
        // sprintf(nam_,"block.%d.attn_norm",il);
        //  llm_build_norm(ctx_build, inpL, config,layers[il].attn_norm,layers[il].attn_norm_b,LLM_NORM, cb, il);
        auto layer      = dynamic_pointer_cast<QKV_LAY>(layers[il]);
        hGensor ffn_inp = inpL;
        if (1) {  // pass self attention module,  loss would stop at 2.48
            layer->att_norm.BuildX(_NAM_("block.%d.attn_norm", il), {n_embd}, this, 0x0);
            layer->att_norm.sT = "a";
            cur                = layer->att_norm.Ming(ctx_build, inpL, 0x0);
            cb(cur, _NAM_("attn_norm"), il);
            if (0) {
                layer->Q.BuildX(_NAM_("block.%d.attn_qkv", il), {n_embd, n_embd + 2 * n_embd_gqa}, this, 0x0);
                cur = layer->Q.Ming(ctx_build, cur, 0x0);
                cb(cur, "wqkv", il);
                hGensor Q2 = ggml_view_2d(ctx_build, cur, n_embd, n_tokens, cur->nb[1], 0 * sizeof(float) * (n_embd));
                hGensor K2 = ggml_view_2d(ctx_build, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd));
                hGensor V2 = ggml_view_2d(ctx_build, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
                cb(Q2, "Q2", il);
                cb(K2, "K2", il);
                cb(V2, "V2", il);
                Qcur = ggml_cont(ctx_build, Q2), Kcur = ggml_cont(ctx_build, K2), Vcur = ggml_cont(ctx_build, V2);
            } else {  // Gradient of Q K would vanish,but the training curve nearly same!
                layer->Q.isBias = false;
                layer->K.isBias = false;
                layer->Q.BuildX(_NAM_("block.%d.Q", il), {n_embd, n_embd}, this, 0x0);
                layer->Q.sT = "q";
                layer->K.BuildX(_NAM_("block.%d.K", il), {n_embd, n_embd_gqa}, this, 0x0);
                layer->K.sT = "k";
                layer->V.BuildX(_NAM_("block.%d.V", il), {n_embd, n_embd_gqa}, this, 0x0);
                layer->V.sT = "v";
                Qcur        = layer->Q.Ming(ctx_build, cur, 0x0);
                Kcur        = layer->K.Ming(ctx_build, cur, 0x0);
                Vcur        = layer->V.Ming(ctx_build, cur, 0x0);
            }
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
            if (isAttOnBC) {  // attenion on all tokens, memory would explode!
                Qcur = ggml_reshape_3d(ctx_build, Qcur, n_embd_head, n_head, n_tokens);
                Kcur = ggml_reshape_3d(ctx_build, Kcur, n_embd_head, n_head, n_tokens);
                Vcur = ggml_reshape_3d(ctx_build, Vcur, n_embd_head, n_head, n_tokens);
            } else {
                Qcur = ggml_reshape_4d(ctx_build, Qcur, n_embd_head, n_head, n_ctx, n_batch);
                Kcur = ggml_reshape_4d(ctx_build, Kcur, n_embd_head, n_head, n_ctx, n_batch);
                Vcur = ggml_reshape_4d(ctx_build, Vcur, n_embd_head, n_head, n_ctx, n_batch);
            }

            q = ggml_permute(ctx_build, Qcur, 0, 2, 1, 3);  // eh,ctx,h,b
            k = ggml_permute(ctx_build, Kcur, 0, 2, 1, 3);
            v = ggml_cont(ctx_build, ggml_permute(ctx_build, Vcur, 1, 2, 0, 3));
            if (isOnlinePush) {
                ggml_build_forward_expand(gf, q);
                ggml_build_forward_expand(gf, k);
                ggml_build_forward_expand(gf, v);
            }

            hGensor kq = ggml_mul_mat(ctx_build, k, q);
            cb(kq, "kq", il);
            if (0) {                                                                                // nearly same
                kq = ggml_soft_max_ext(ctx_build, kq, KQ_mask, kq_scale, config.f_max_alibi_bias);  // would
            } else {
                hGensor t16_1 = tSCAL(ctx_build, kq, kq_scale);
                hGensor t16_2 = ggml_diag_mask_inf_inplace(ctx_build, t16_1, 0);
                kq            = ggml_soft_max_inplace(ctx_build, t16_2);
            }
            cb(kq, "kq_soft_max_ext", il);

            hGensor kqv = ggml_mul_mat(ctx_build, v, kq);  // eh,ctx,h,b
            cb(kqv, "kqv", il);
            hGensor kqv_merged = ggml_permute(ctx_build, kqv, 0, 2, 1, 3);  // eh,h,ctx,b
            cb(kqv_merged, "kqv_merged", il);
            if (0)  //  back gradient is zero
                cur = ggml_cont_2d(ctx_build, kqv_merged, n_embd_head_v * n_head, n_tokens);
            else {
                hGensor kqv_out = ggml_cont(ctx_build, kqv_merged);
                cur             = ggml_reshape_2d(ctx_build, kqv_out, n_embd, n_tokens);
            }
            cb(cur, "kqv_merged_cont", il);
            layer->proj.BuildX(_NAM_("block.%d.attn_proj", il), {n_embd, n_embd}, this, 0x0);  // why proj is useful?
            cur = layer->proj.Ming(ctx_build, cur, 0x0);
            cb(cur, "attn_proj", il);

            if (isOnlinePush)
                ggml_build_forward_expand(gf, cur);

            if (layer->isLast) {  // skip computing output for unused tokens
                // hGensor inp_out_ids = nullptr;  //build_inp_out_ids();
                // cur  = ggml_get_rows(ctx_build,  cur, inp_out_ids);
                // inpL = ggml_get_rows(ctx_build, inpL, inp_out_ids);
            }
            // add the input
            ffn_inp = ggml_add(ctx_build, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);
            cur = ffn_inp;
        }
        if (1) {  // pass self attention module            loss would stop at 2.4
            layer->ffn_norm.BuildX(_NAM_("block.%d.ffn.norm", il), {n_embd}, this, 0x0);
            layer->ffn_norm.sT = "f";
            cur                = layer->ffn_norm.Ming(ctx_build, inpL, 0x0);  //  ??? cur = layer->ffn_norm.Ming(ctx_build,inpL,0x0);
            cb(cur, _NAM_("ffn.norm"), il);
            layer->up.BuildX(_NAM_("block.%d.ffn.up", il), {n_embd, n_ff}, this, 0x0);
            cur = layer->up.Ming(ctx_build, cur, 0x0);
            cb(cur, "ffn_up", il);
            // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
            cur = ggml_silu(ctx_build, cur);
            cb(cur, "ffn.silu", il);
            layer->down.BuildX(_NAM_("block.%d.ffn.down", il), {n_ff, n_embd}, this, 0x0);
            cur = layer->down.Ming(ctx_build, cur, 0x0);
            cb(cur, "ffn.down", il);
            cur = ggml_add(ctx_build, cur, ffn_inp);
            // cur = lctx.cvec.apply_to(ctx_build, cur, il);
            cb(cur, "l_out", il);
        }
        // input for next layer
        inpL = cur;
    }

    cur = _norm.Ming(ctx_build, inpL, 0x0);
    cb(cur, "result_norm", -1);
    cur = _output.Ming(ctx_build, cur, 0x0);
    cb(cur, "result_output", -1);
    preLogits = cur;

    // if(0){  //only for debug
    //     string suffix="", prefix="\t";
    //     __repr__(suffix,prefix);
    //     TGraph(this,"gpt_raw",gf,true).__repr__(suffix,prefix,cur);
    // }

    return 0x0;
}
#endif
#ifdef __USE_GGML__
struct ggml_cgraph *GPT2::BuildRawGraph(void *ctx_build, bool isBuild, int flag) {
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_ctx_train = config.n_ctx_train, n_embd = config.nEmbed();
    bool isJModel = !config.jModel.empty();
    hForwTG       = std::make_shared<TGraph>(this, isJModel ? "gptJ" : "gpt_raw", ctx_build, true);
    auto gf       = nullptr;  // GetForwRaw();
    // ctx = ctx_build;
    InitInput(ctx_build, false);

    if (isJModel)
        jToGraph(ctx_build, isBuild, flag);
    // else
    //     cRawGraph(ctx_build,isBuild,flag);
    assert(preLogits != nullptr);
    BuildLoss(ctx_build, preLogits);

#ifndef _TENSOR_G_
    if (rnd == nullptr)
        rnd = init_random_normal_distribution(config.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    size_t sz2 = hEDS->Alloc(hForwTG, ctx_build);
#endif
    return gf;
}
#endif

string GPT2::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    string sBasic   = NLP_AutoRegressive::__repr__(suffix, prefix, flag);
    sprintf(buf + strlen(buf), "%s", sBasic.c_str());
    _INFO("GPT2:    Bias=(normal=%d,slp=%d) AttOnBC=%d\n========\n", config.model.isNormalBias, config.model.isSLPBias, isAttOnBC);
    return buf;
}

std::string CDict_GPT2::T2STR(TOKEN_ID tok, int flag) {
#ifdef _TENSOR_G_
    // [todo] call gpt2 tokenizer in next version
#else
    assert(0);
#endif
    return "";
};
int CDict_GPT2::STR2T(const char *txt, int txt_len, std::vector<TOKEN_ID> &btch, int flag) {
    //  https://github.com/wangkuiyi/huggingface-tokenizer-in-cxx
    //  https://www.daoplays.org/blog/gpt2_p1
    btch = {12518, 262, 7523, 318, 1016, 866, 11};  //  "when the smoke is going down"
    return btch.size();
    assert(0);
    int n_tokens = 0, nMost = btch.size();
    assert(txt_len <= nMost);
    unsigned char *a = (unsigned char *)(txt);
    for (int i = 0; i < txt_len; i++, a++) {
        TOKEN_ID t = (TOKEN_ID)(*a);
        // assert(t>=0 && t<n_vocab);
        btch[i] = t;
        n_tokens++;
    }
    return n_tokens;
}

/*
bool CDict_GPT2::LoadTokenizer(const char *filename,int flag) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        _WARN("%s: Failed to open the tokenizer file @%s\n", filename);
        init_ok = 0;
        return false;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    n_vocab = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(n_vocab == 50257); // let's be defensive here
        eot_token = 50256;
    } else if (version == 2) {
        eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    token_table = (char **)mallocCheck(n_vocab * sizeof(char *));
    for (uint32_t i = 0; i < n_vocab; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    init_ok = true;
    return init_ok;
}*/