/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General LLama model  
 * 
 *  \brief LLaMeta Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#include "gLAMA.hpp"
#include "../LLAMA/unicode.h"

// gguf constants (sync with gguf.py)
static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";

static const char * LLM_KV_GENERAL_NAME                = "general.name";
static const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
static const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";
static const char * LLM_KV_VOCAB_SIZE                  = "%s.vocab_size";
static const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
static const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";

static const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
static const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";
static const char * LLM_KV_ATTENTION_HEAD_COUNT_KV     = "%s.attention.head_count_kv";

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

    // { LLM_KV_DICT_VAE_LAYERS,               "dict.vae.layers"       },
    // { LLM_KV_DICT_LATENT_DIM,                  "%s.dict_latent_dim"},
static const char * LLM_KV_DICT_VAE_LAYERS             = "dict.vae.layers";
static const char * LLM_KV_DICT_LATENT_DIM             = "%s.dict_latent_dim";

static const char * arch = "gruai";
static char keybuf[512];
const char * kv(const char * key)   {    
    snprintf(keybuf, 512, key, arch);
    return keybuf;
};

void Ganglia::save_train_(struct save_train_model * data, struct train_state * train) { 
    int64_t iter = train->opt->iter;
    printf("%s: iter_%ld\n", __func__, iter);
    string sBaseName = get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, -1  );
    if (strlen(data->fn_checkpoint_out) > 0) {
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model, train);
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_model_base, data->model, train);
    }
    if (strlen(data->fn_model_out) > 0) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        string sOut = "g_" + sBaseName; 
        save_gguf(sOut.c_str(),0x0);
    }
    if(1){  //only for debug
        arch = "llama";
        string sOut = "l_" + sBaseName;     //hack        
        save_gguf(sOut.c_str(),0x0);
    }

    return;
}

ConsiceDict::ConsiceDict(LLaMeta *lama_,int flag) : VariationaAE(),hLM(lama_)   {
    assert(hLM->isValid());
    hparams = hLM->hparams;
    reserve_x = true;
    lama_embed = hparams.n_embd;
    if(hLM->hparams.nabla==3){
        // dims = {hparams.n_embd, 64};
        // dims = {hparams.n_embd, 512, 128};
        dims = {hparams.n_embd, 512, 256,64};
        nLevel = dims.size()-1;   
        latent_dim = dims[nLevel];
    }   else if(hLM->hparams.nabla>3)
        assert(0);         
    _INFO("%s resi=%d tpNorm=%d nLevel=%d dims={%d...%d}",__func__,(int)(reserve_x),tpNorm,nLevel,nLevel>0?dims[0]:-1,nLevel>0?latent_dim:-1);           
}

void ConsiceDict::InitVAE(int flag)  {
    assert(nLevel>0);   
    if(nLevel>=1){
        isLoadTokenEmbed = true;
        InitMAEC(hLM->ctx,dims);
        // hMultiCoder hCoder = std::make_shared<MutliCoder>(hLM->ctx, hparams.n_embd, latent_dim);
        // MAEC.push_back(hCoder);
        // encoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, hparams.n_embd, latent_dim);     
        // decoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, latent_dim, hparams.n_embd); 
    }    
    hLM->hparams.n_embd = latent_dim;        
}

/**
 * Only for compare & debug
 * 
 * llm_load_vocab

void ConsiceDict::LoadVocab_v1(const char*fn_model_base,struct cwd_params& params,llama_model & model,int flag) {
    llama_model_loader ml(fn_model_base, true,false,nullptr);
    // model.hparams.vocab_only = params.vocab_only;
    assert(model.arch==LLM_ARCH_GRUAI);
    

    auto & vocab = model.vocab;
    bpe_ranks = vocab.bpe_ranks;
}*/

void ConsiceDict::CreateEmbeddings(struct random_normal_distribution * rnd,int flag){
    if(isLoadTokenEmbed) {
        return;
    }
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab;
    auto lama = hLM->GetRawModel( );  
    auto ctx = hLM->ctx;
    tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);  
}

void ConsiceDict::Update_0(struct random_normal_distribution * rnd,int flag){
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab;
    auto lama = hLM->GetRawModel( );  
    if(isLoadTokenEmbed) {
        bool isParam = false;
        // get tensors from llama_model (possibly mmapped)
        tok_embeddings = llama_get_model_tensor(lama, TN(LLM_TENSOR_TOKEN_EMBD));      
        if(isParam) nParams+=ggml_nelements(tok_embeddings);
        norm           = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT_NORM));     
        if(isParam) nParams+=ggml_nelements(norm);
        output         = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT));          
        if(isParam) nParams+=ggml_nelements(output);
    }   else   {
        auto ctx = hLM->ctx;
        /*tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
        norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); */ 

        hLM->InitGensor(ctx,tok_embeddings, TN(LLM_TENSOR_TOKEN_EMBD), rnd);
        hLM->InitGensor(ctx,norm,           TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        hLM->InitGensor(ctx,output,         TN(LLM_TENSOR_OUTPUT), rnd);
    }
    // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);              
    }else{

    }      
}

void ConsiceDict::Update_1(struct random_normal_distribution * rnd,int flag) {
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab;

    bool isParam = false;
    // get tensors from llama_model (possibly mmapped)
    auto lmodel = hLM->GetRawModel( );  
    tok_embeddings = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_TOKEN_EMBD) );        //TN(LLM_TENSOR_TOKEN_EMBD)
    if(isParam) hLM->nParams+=ggml_nelements(tok_embeddings);
    norm           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );       //  
    if(isParam) hLM->nParams+=ggml_nelements(norm);
    output         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  );            //
    if(isParam) hLM->nParams+=ggml_nelements(output);
    assert(tok_embeddings!=nullptr && norm!=nullptr && output!=nullptr);

    // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);              
    }
    int i = 0;
    for(auto map : MAEC){
        std::string name = TN(LLM_DICT_DOWN, i);    //"dict.0.down.weight"
        hLM->InitGensor(hLM->ctx, map->encode,    TN(LLM_DICT_DOWN, i),     rnd); 
        hLM->InitGensor(hLM->ctx, map->decode,    TN(LLM_DICT_UP, i),       rnd);    
        i++;            
    }
    //ggml_set_param(hLM->ctx, norm);              hLM->nParams+=ggml_nelements(norm);
    //output is Q6k would fail @float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i)
    //ggml_set_param(hLM->ctx, output);            hLM->nParams+=ggml_nelements(output);

    hLM->tensors[ggml_get_name(tok_embeddings)] = tok_embeddings;
    hLM->tensors[ggml_get_name(norm)] = norm;
    hLM->tensors[ggml_get_name(output)] = output;  
    assert(tensors.size()==0);          
}

void ConsiceDict::LoadVocab(const char*fn_model_base,int flag)     {
    string word;
    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;   //LLAMA_FTYPE_ALL_F32;
    struct gguf_init_params params = {        false,NULL,    };
    struct gguf_context * vctx = gguf_init_from_file(fn_model_base, params);

    token_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_LIST));
    if (token_idx == -1) {
        die("cannot find tokenizer vocab in model file");
    }
    n_vocab = gguf_get_arr_n(vctx, token_idx);
    int nTT = gguf_get_arr_n(vctx, token_idx);          assert(n_vocab==nTT);
    score_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_SCORES));
    if (score_idx == -1) {
        die("cannot find tokenizer scores in model file");
    }
    scores = new float[nTT];
    memcpy(scores,gguf_get_arr_data(vctx, score_idx),sizeof(float)*nTT);    

    toktype_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
    if (toktype_idx == -1) {
        die("cannot find token type list in GGUF file");
    }
    assert( nTT == gguf_get_arr_n(vctx, toktype_idx));
    toktypes = new int[nTT];
    memcpy(toktypes,gguf_get_arr_data(vctx, toktype_idx),sizeof(int)*nTT);    
    GGUF_GET_KEY(vctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));
    if (tokenizer_name == "llama") {
        // default special tokens
        special_bos_id = 1;
        special_eos_id = 2;
        special_unk_id = 0;
        special_sep_id = -1;
        special_pad_id = -1;
    } else if (tokenizer_name == "gpt2") {
        // read and copy bpe merges
        merges_keyidx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_MERGES));
        if (merges_keyidx == -1) {
            die("cannot find tokenizer merges in model file");
        }
        n_merges = gguf_get_arr_n(vctx, merges_keyidx);
        // std::vector<const char*> merges;
        merges.resize(n_merges);
        for (int i = 0; i < n_merges; i++) {
            merges[i] = strdup(gguf_get_arr_str(vctx, merges_keyidx, i));
            word = merges[i];
            GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);
            std::string first,second;
            const size_t pos = word.find(' ', 1);
            if (pos != std::string::npos) {
                first  = word.substr(0, pos);
                second = word.substr(pos + 1);
            }
            bpe_ranks.emplace(std::make_pair(first, second), i);
        }
        word = merges[0];
        // gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_MERGES), merges.data(), n_merges);
        // default special tokens
        special_bos_id = 11;        special_eos_id = 11;        special_unk_id = -1;
        special_sep_id = -1;        special_pad_id = -1;
    } else {
        fprintf(stderr, "%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
        fprintf(stderr, "%s: using default tokenizer: 'llama'", __func__);
    }

    // std::vector<const char*> tokens;
    tokens.resize(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
        tokens[i] = strdup(gguf_get_arr_str(vctx, token_idx, i));
    }
    // gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), tokens.data(), n_vocab);
    GGUF_GET_KEY(vctx, special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
    GGUF_GET_KEY(vctx, special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
    GGUF_GET_KEY(vctx, special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
    GGUF_GET_KEY(vctx, special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
    GGUF_GET_KEY(vctx, special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

    gguf_free(vctx);
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
            // ffn = t22;        
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

void VariationaAE::save_gguf(struct gguf_context *fctx, int flag)   {
    if(MAEC.size()==0)
        return;
    int nLay = MAEC.size()+1;       assert(nLay>=2);
    gguf_set_arr_data(fctx, kv(LLM_KV_DICT_VAE_LAYERS), GGUF_TYPE_INT32, dims.data(), nLay);  
    for(auto coder:MAEC){
        gguf_add_tensor(fctx, coder->encode);
        gguf_add_tensor(fctx, coder->decode);
    }   
}

void LLaMeta::save_gguf(const char * filename, int flag) {
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;       //LLAMA_FTYPE_MOSTLY_Q2_K
    printf("%s: saving to %s ftype=%d ......\n", __func__, filename,ftype);
    struct gguf_context * fctx = gguf_init_empty();
    int keyidx = -1;    
    
    // set arch
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_str(fctx, LLM_KV_GENERAL_NAME, ".");
    gguf_set_val_u32(fctx, kv(LLM_KV_VOCAB_SIZE), n_vocab);
    int llm_embd = hDict->lama_embed,latent_dim=hDict->latent_dim;        //hparams.n_embd
    if(hDict->nLevel>0)    assert(llm_embd>latent_dim && latent_dim>0);
    // set hparams
    const char*str = kv(LLM_KV_CONTEXT_LENGTH);
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              hparams.n_ctx                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            llm_embd                       );
    
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 hparams.n_layer                );
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         hparams.n_ff                   );
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        hparams.n_rot                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        hparams.n_head                 );    
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV),     hparams.n_head_kv              );

    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), hparams.f_norm_rms_eps         );
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              hparams.rope_freq_base         ); // TODO load in llama.cpp
    
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);
    // set vocab by copying from vocab_model gguf file
    gguf_set_val_str(fctx, kv(LLM_KV_TOKENIZER_MODEL), hDict->tokenizer_name.c_str());

    gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), hDict->tokens.data(), n_vocab);
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, hDict->scores, n_vocab);    
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, hDict->toktypes, n_vocab);
    if (hDict->tokenizer_name == "gpt2"){
        const char* sMERGES = kv(LLM_KV_TOKENIZER_MERGES);
        gguf_set_val_u32(fctx, sMERGES,  hDict->merges_keyidx              );
        keyidx = gguf_find_key(fctx, sMERGES);      //only for debug
        assert(hDict->merges.size()==hDict->n_merges);
        string word = hDict->merges[0];
        gguf_set_arr_str(fctx, sMERGES, hDict->merges.data(), hDict->n_merges);
        /*for (int i = 0; i < hDict->n_merges; i++) {        //only for debug
            const std::string word = gguf_get_arr_str(fctx, keyidx, i);
            GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);            
        }*/
    }   
    
    
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_BOS_ID), hDict->special_bos_id);
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_EOS_ID), hDict->special_eos_id);
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_UNK_ID), hDict->special_unk_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_SEP_ID), hDict->special_sep_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_PAD_ID), hDict->special_pad_id);      -1
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           1.0f / hparams.rope_freq_scale );
    gguf_set_val_u32(fctx, kv(LLM_KV_DICT_LATENT_DIM),             latent_dim                       );
    //more maybe from llama_chat_apply_template
    // add tensors
    gguf_add_tensor(fctx, hDict->tok_embeddings);   //4096*128256
    gguf_add_tensor(fctx, hDict->norm);             //4096
    gguf_add_tensor(fctx, hDict->output);           //4096*128256
    hDict->save_gguf(fctx, flag);    

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        auto layer = dynamic_pointer_cast<lama_layer>(layers[i]); //layers[i];

        gguf_add_tensor(fctx, layer->attention_norm);
        gguf_add_tensor(fctx, layer->wq);
        gguf_add_tensor(fctx, layer->wk);
        gguf_add_tensor(fctx, layer->wv);
        gguf_add_tensor(fctx, layer->wo);
        if(layer->ffn_norm!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_norm);
        if(layer->ffn_gate!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_gate);
        if(layer->ffn_down!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_down);
        if(layer->ffn_up!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_up);
    }

    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}