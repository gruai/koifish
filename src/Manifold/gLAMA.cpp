/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General LLama model  
 * 
 *  \brief LLaMeta Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#include "gLAMA.hpp"

// gguf constants (sync with gguf.py)
static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";

static const char * LLM_KV_GENERAL_NAME                = "general.name";
static const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
static const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";

static const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
static const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";
static const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
static const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";

static const char * LLM_KV_TOKENIZER_MODEL             = "tokenizer.ggml.model";
static const char * LLM_KV_TOKENIZER_LIST              = "tokenizer.ggml.tokens";
static const char * LLM_KV_TOKENIZER_TOKEN_TYPE        = "tokenizer.ggml.token_type";
static const char * LLM_KV_TOKENIZER_SCORES            = "tokenizer.ggml.scores";
static const char * LLM_KV_TOKENIZER_MERGES            = "tokenizer.ggml.merges";
static const char * LLM_KV_TOKENIZER_BOS_ID            = "tokenizer.ggml.bos_token_id";
static const char * LLM_KV_TOKENIZER_EOS_ID            = "tokenizer.ggml.eos_token_id";
static const char * LLM_KV_TOKENIZER_UNK_ID            = "tokenizer.ggml.unknown_token_id";
static const char * LLM_KV_TOKENIZER_SEP_ID            = "tokenizer.ggml.seperator_token_id";
static const char * LLM_KV_TOKENIZER_PAD_ID            = "tokenizer.ggml.padding_token_id";

void LLaMeta::save_gguf(const char * filename, int flag) {
    printf("%s: saving to %s ......\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    const char * arch = "llama";
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::vector<char> keybuf;
    keybuf.resize(512);
    auto kv = [arch, &keybuf](const char * key) -> const char * {
        snprintf(keybuf.data(), keybuf.size(), key, arch);
        return keybuf.data();
    };

    // set arch
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);

    // set hparams
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              hparams.n_ctx                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            hparams.n_embd                 );
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         hparams.n_ff                   );
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        hparams.n_head                 );
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 hparams.n_layer                );
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        hparams.n_rot                  );

    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), hparams.f_norm_rms_eps         );
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              hparams.rope_freq_base         ); // TODO load in llama.cpp
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           1.0f / hparams.rope_freq_scale );
    const char*fn_model_base="";
    // set vocab by copying from vocab_model gguf file
    {
        struct gguf_init_params params = {
            /*.no_alloc = */ false,
            /*.ctx      = */ NULL,
        };
        struct gguf_context * vctx = gguf_init_from_file(fn_model_base, params);

        const int token_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_LIST));
        if (token_idx == -1) {
            die("cannot find tokenizer vocab in model file");
        }
        const uint32_t n_vocab = gguf_get_arr_n(vctx, token_idx);

        const int score_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_SCORES));
        if (score_idx == -1) {
            die("cannot find tokenizer scores in model file");
        }

        const float * scores = (const float * ) gguf_get_arr_data(vctx, score_idx);

        const int toktype_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
        if (toktype_idx == -1) {
            die("cannot find token type list in GGUF file");
        }

        const int * toktypes = (const int * ) gguf_get_arr_data(vctx, toktype_idx);

        std::string tokenizer_name;
        GGUF_GET_KEY(vctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));

        gguf_set_val_str(fctx, kv(LLM_KV_TOKENIZER_MODEL), tokenizer_name.c_str());
        gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, scores, n_vocab);
        gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, toktypes, n_vocab);

        int32_t special_bos_id = 1;
        int32_t special_eos_id = 2;
        int32_t special_unk_id = 0;
        int32_t special_sep_id = -1;
        int32_t special_pad_id = -1;
        if (tokenizer_name == "llama") {
            // default special tokens
            special_bos_id = 1;
            special_eos_id = 2;
            special_unk_id = 0;
            special_sep_id = -1;
            special_pad_id = -1;
        } else if (tokenizer_name == "gpt2") {
            // read and copy bpe merges
            const int merges_keyidx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_MERGES));
            if (merges_keyidx == -1) {
                die("cannot find tokenizer merges in model file");
            }

            const int n_merges = gguf_get_arr_n(vctx, merges_keyidx);

            std::vector<const char*> merges;
            merges.resize(n_merges);
            for (int i = 0; i < n_merges; i++) {
                merges[i] = gguf_get_arr_str(vctx, merges_keyidx, i);
            }
            gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_MERGES), merges.data(), n_merges);

            // default special tokens
            special_bos_id = 11;
            special_eos_id = 11;
            special_unk_id = -1;
            special_sep_id = -1;
            special_pad_id = -1;
        } else {
            fprintf(stderr, "%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
            fprintf(stderr, "%s: using default tokenizer: 'llama'", __func__);
        }

        std::vector<const char*> tokens;
        tokens.resize(n_vocab);
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokens[i] = gguf_get_arr_str(vctx, token_idx, i);
        }
        gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), tokens.data(), n_vocab);

        GGUF_GET_KEY(vctx, special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
        GGUF_GET_KEY(vctx, special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
        GGUF_GET_KEY(vctx, special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
        GGUF_GET_KEY(vctx, special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
        GGUF_GET_KEY(vctx, special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_BOS_ID), special_bos_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_EOS_ID), special_eos_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_UNK_ID), special_unk_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_SEP_ID), special_sep_id);
        gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_PAD_ID), special_pad_id);

        gguf_free(vctx);
    }

    // add tensors
    gguf_add_tensor(fctx, hDict->tok_embeddings);
    gguf_add_tensor(fctx, hDict->norm);
    gguf_add_tensor(fctx, hDict->output);
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        auto layer = dynamic_pointer_cast<lama_layer>(layers[i]); //layers[i];

        gguf_add_tensor(fctx, layer->attention_norm);
        gguf_add_tensor(fctx, layer->wq);
        gguf_add_tensor(fctx, layer->wk);
        gguf_add_tensor(fctx, layer->wv);
        gguf_add_tensor(fctx, layer->wo);
        gguf_add_tensor(fctx, layer->ffn_norm);
        gguf_add_tensor(fctx, layer->ffn_gate);
        gguf_add_tensor(fctx, layer->ffn_down);
        gguf_add_tensor(fctx, layer->ffn_up);
    }

    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

static void save_checkpoint_file(const char * filename, const char * fn_model_base, struct llama_model * model, struct train_state * train) {
    printf("%s: saving to %s\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    // save_checkpoint_gguf(fctx, fn_model_base, model, train);
    gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE, LLM_KV_TRAINING_TYPE_TRAIN_MODEL);
    // save_llama_model_gguf(fctx, fn_model_base, model);
    save_train_state_gguf(fctx, train);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

// void save_llama_model_file(const char * filename, const char * fn_model_base, struct llama_model * model) {
//     printf("%s: saving to %s\n", __func__, filename);
//     struct gguf_context * fctx = gguf_init_empty();

//     save_llama_model_gguf(fctx, fn_model_base, model);

//     // write file
//     const bool only_meta = false;
//     gguf_write_to_file(fctx, filename, only_meta);
//     gguf_free(fctx);
// }

/*
    // optionally save the session on first sample (for faster prompt loading next time)
    if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
        need_to_save_session = false;
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

        LOG("saved session to %s\n", path_session.c_str());
    }
*/
void LLaMeta::save_train_(struct save_train_files_data * data, struct train_state * train) { 
    int64_t iter = train->opt->iter;
    if (strlen(data->fn_checkpoint_out) > 0) {
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model, train);
        save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_model_base, data->model, train);
    }
    if (strlen(data->fn_model_out) > 0) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        save_gguf(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(),0x0);
    }
    // if (strlen(data->fn_checkpoint_out) > 0) {
    //     save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->model, data->lora, train);
    //     save_checkpoint_lora_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->model, data->lora, train);
    // }
    // if (strlen(data->fn_model_out) > 0) {
    //     save_as_llama_lora(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->lora);
    //     save_as_llama_lora(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->lora);
    // }
    return;
}

//n_embd_head, n_head_kv
hGensor  LLaMeta::build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor  KQ_pos,/*hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down,*/ int flag) {
    auto train_params = hparams.common;
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;  
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    const int n_past = 0, n_head_kv=hparams.n_head_kv,n_embd_head = hparams.n_embd_head();
    hGensor wq = UpdateGensor (layer->wq->name);                     
    hGensor wk = UpdateGensor (layer->wk->name);
    hGensor wv = UpdateGensor (layer->wv->name);
    hGensor wo = UpdateGensor (layer->wo->name);
    hGensor attention_norm = UpdateGensor (layer->attention_norm->name);    
    hGensor ffn_norm = layer->ffn_norm==nullptr ? nullptr : UpdateGensor (layer->ffn_norm->name); 
    hGensor ffn_up = nullptr,ffn_gate=nullptr,ffn_down=nullptr;
    if(layer->ffn_up!=nullptr){                
        ffn_up = UpdateGensor (layer->ffn_up->name);
        ffn_gate = UpdateGensor (layer->ffn_gate->name);
        ffn_down = UpdateGensor (layer->ffn_down->name);                
    }  
    /*  LORA
        hGensor wq = n_rank_wq ==0 ? nullptr : UpdateGensor (layer->wq->name);                     
        hGensor wk = n_rank_wk ==0 ? nullptr : UpdateGensor (layer->wk->name);
        hGensor wv = n_rank_wv ==0 ? nullptr : UpdateGensor (layer->wv->name);
        hGensor wo = UpdateGensor (layer->wo->name);
        hGensor a_norm = UpdateGensor (layer->attention_norm->name);   
        hGensor ffn_norm = UpdateGensor (layer->ffn_norm->name);
        hGensor ffn_up = UpdateGensor (layer->ffn_up->name);
        hGensor ffn_gate = UpdateGensor (layer->ffn_gate->name);
        hGensor ffn_down = UpdateGensor (layer->ffn_down->name);
    */

    //  rms_norm:   Root Mean Square Layer Normalization
    hGensor  t02 = ggml_rms_norm     (ctx_compute, cur, f_norm_rms_eps);                    set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
    hGensor  t03 = ggml_repeat       (ctx_compute, attention_norm, t02);              set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
    hGensor  t04 = ggml_mul          (ctx_compute, t03, t02);                               set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
    // QKV_Motion qkv(wq,wk,wv,n_embd,n_head,  N, n_batch,n_rot, n_ctx,n_head_kv,f_norm_rms_eps,rope_freq_base,rope_freq_scale);
    hBrownMotion hBrown = CreateBrownMotion(wq, wk, wv);                              
    hGensor t16 = hBrown->Build(ctx_compute , t04,  KQ_pos,  train_params.use_flash);        
    hGensor  t17 = ggml_permute      (ctx_compute, t16, 0, 2, 1, 3);                        set_name(t17, "t17");     assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
    hGensor  t18 = ggml_cont         (ctx_compute, t17);                                    set_name(t18, "t18");     assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
    hGensor  t19 = ggml_reshape_2d   (ctx_compute, t18, n_embd, N*n_batch);                 set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
    hGensor  t20 = ggml_mul_mat      (ctx_compute, wo, t19);                          set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
    hGensor  t21 = ggml_add          (ctx_compute, t20, cur);                               set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
    hGensor  ffn = nullptr;
    switch(tpFFN)   {
    case SWIGLU:    {
        hGensor  t22 = ggml_rms_norm     (ctx_compute, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
        hGensor  t24 = ggml_mul          (ctx_compute, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
        hGensor  t25 = ggml_mul_mat      (ctx_compute, ffn_up, t24);                      set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
        hGensor  t26 = ggml_mul_mat      (ctx_compute, ffn_gate, t24);                    set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
        hGensor  t27 = ggml_silu         (ctx_compute, t26);                                    set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
        hGensor  t28 = ggml_mul          (ctx_compute, t27, t25);                               set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
        hGensor  t29 = ggml_mul_mat      (ctx_compute, ffn_down, t28);                    set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
        hGensor  t30 = ggml_add          (ctx_compute, t29, t21);                               set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        return t30;   
    }
    case VAR_0:
    case ONLY_LNormal:
    case ONLY_RMSNormal:    {
        assert(ffn_up==nullptr);
        hGensor  t22 = tpFFN==ONLY_LNormal ? ggml_norm(ctx_compute, t21, f_norm_rms_eps) :
            ggml_rms_norm(ctx_compute, t21, f_norm_rms_eps);
        set_name(t22, "t22");               assert_shape_2d(t22, n_embd, N*n_batch);   
        if(tpFFN==VAR_0)     {
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_compute, layer->eps, 0.01);
            ffn = ggml_add          (ctx_compute, t22,noise);             
        }else{
            hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            ffn = ggml_mul          (ctx_compute, t23, t22);
        }
        return ffn;
    }
    default:
        TO_DO;
        break;
    }
    if(ffn_up==nullptr)   {        
        
        /*trick v0
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu        
        //  
        // hGensor  tMu = ggml_mul          (ctx_compute, layer->w_mu, t21);
        // hGensor  tVar = ggml_mul          (ctx_compute, layer->w_var, t21);*/
        //  trick v1
                              
            // set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        return ffn;
    }else{
        
    }                 
}

/*
virtual void LoadModel(const char * fn_model, int flag=0x0) {
        std::vector<char> tn_buf;
        tn_buf.resize(GGML_MAX_NAME);
        int nLayer0 = hparams.n_layer;        
        {
            struct gguf_init_params params = { false, NULL,            };
            struct gguf_context * mctx = gguf_init_from_file(fn_model, params);
            const char *expected_arch = "llama";
            // load_model_hparams_gguf(mctx, &hparams, "llama");
            std::string arch;
            GGUF_GET_KEY(mctx, arch, gguf_get_val_str, GGUF_TYPE_STRING, true, LLM_KV_GENERAL_ARCHITECTURE);
            if (expected_arch != NULL) {
                if (arch != expected_arch) {
                    _INFO("%s: arch=%s expected_arch=%s\n", __func__, arch.c_str(), expected_arch);
                }
                GGML_ASSERT(arch == expected_arch);
            }

            std::vector<char> keybuf;
            keybuf.resize(512);
            auto kv = [&arch, &keybuf](const char * key) -> const char * {
                snprintf(keybuf.data(), keybuf.size(), key, arch.c_str());
                return keybuf.data();
            };

            GGUF_GET_KEY(mctx, hparams.n_embd,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_EMBEDDING_LENGTH));
            GGUF_GET_KEY(mctx, hparams.n_ctx,          gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_CONTEXT_LENGTH));
            GGUF_GET_KEY(mctx, hparams.n_ff,           gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_FEED_FORWARD_LENGTH));
            GGUF_GET_KEY(mctx, hparams.n_head,         gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_ATTENTION_HEAD_COUNT));
            GGUF_GET_KEY(mctx, hparams.n_layer,        gguf_get_val_u32, GGUF_TYPE_UINT32,  true, kv(LLM_KV_BLOCK_COUNT));
            GGUF_GET_KEY(mctx, hparams.n_head_kv,      gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV));
            float rope_freq_scale = 1.0f;
            GGUF_GET_KEY(mctx, hparams.f_norm_rms_eps, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS));
            GGUF_GET_KEY(mctx, hparams.rope_freq_base, gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_FREQ_BASE));
            GGUF_GET_KEY(mctx, rope_freq_scale,         gguf_get_val_f32, GGUF_TYPE_FLOAT32, false, kv(LLM_KV_ROPE_SCALE_LINEAR));
            if (rope_freq_scale != 1.0f) {
                hparams.rope_freq_scale = 1.0f / rope_freq_scale;
            }

            gguf_free(mctx);
        }
        struct llama_model * input = lmodel;
        hparams.n_vocab = llama_n_vocab(input);
        if( hparams.n_layer != nLayer0 )        {   //cys
            hparams.n_layer = nLayer0;
        };    
        // hparams.n_ctx = n_ctx;
        nParams = 0;
        // get tensors from llama_model (possibly mmapped)
        tok_embeddings = llama_get_model_tensor(input, TN(LLM_TENSOR_TOKEN_EMBD));      nParams+=ggml_nelements(tok_embeddings);
        norm           = llama_get_model_tensor(input, TN(LLM_TENSOR_OUTPUT_NORM));     nParams+=ggml_nelements(norm);
        output         = llama_get_model_tensor(input, TN(LLM_TENSOR_OUTPUT));          nParams+=ggml_nelements(output);
        // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32); 

        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);
        
        // layers.resize(hparams.n_layer);
        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
            // auto layer = dynamic_pointer_cast<lama_layer>(layers[i]);
            auto layer = std::make_shared<lama_layer>( );
            layers.push_back(layer);        
            layer->attention_norm = llama_get_model_tensor(input, TN(LLM_TENSOR_ATTN_NORM, i));     nParams+=ggml_nelements(layer->attention_norm);
            layer->wq             = llama_get_model_tensor(input, TN(LLM_TENSOR_ATTN_Q, i));        nParams+=ggml_nelements(layer->wq);
            layer->wk             = llama_get_model_tensor(input, TN(LLM_TENSOR_ATTN_K, i));        nParams+=ggml_nelements(layer->wk);
            layer->wv             = llama_get_model_tensor(input, TN(LLM_TENSOR_ATTN_V, i));        nParams+=ggml_nelements(layer->wv);
            layer->wo             = llama_get_model_tensor(input, TN(LLM_TENSOR_ATTN_OUT, i));      nParams+=ggml_nelements(layer->wo);
            layer->ffn_norm       = llama_get_model_tensor(input, TN(LLM_TENSOR_FFN_NORM, i));      nParams+=ggml_nelements(layer->ffn_norm);
            layer->ffn_gate       = llama_get_model_tensor(input, TN(LLM_TENSOR_FFN_GATE, i));      nParams+=ggml_nelements(layer->ffn_gate);
            layer->ffn_down       = llama_get_model_tensor(input, TN(LLM_TENSOR_FFN_DOWN, i));      nParams+=ggml_nelements(layer->ffn_down);
            layer->ffn_up         = llama_get_model_tensor(input, TN(LLM_TENSOR_FFN_UP, i));        nParams+=ggml_nelements(layer->ffn_up); 

            assert_shape_1d(layer->attention_norm, hparams.n_embd);
            assert_shape_2d(layer->wq,             hparams.n_embd, hparams.n_embd);
            assert_shape_2d(layer->wk,             hparams.n_embd, hparams.n_embd_gqa());
            assert_shape_2d(layer->wv,             hparams.n_embd, hparams.n_embd_gqa());
            assert_shape_2d(layer->wo,             hparams.n_embd, hparams.n_embd);
            assert_shape_1d(layer->ffn_norm,       hparams.n_embd);
            assert_shape_2d(layer->ffn_gate,       hparams.n_embd, hparams.n_ff);
            assert_shape_2d(layer->ffn_down,       hparams.n_ff,   hparams.n_embd);
            assert_shape_2d(layer->ffn_up,         hparams.n_embd, hparams.n_ff);
        }
        if(hparams.n_layer==40){
            assert(nParams==nParamsGGUF);
        }
    }*/