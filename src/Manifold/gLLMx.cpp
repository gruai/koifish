/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  More models derived from NLP_AutoRegressive
 * 
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"


LLM_MAMBA::LLM_MAMBA( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag) : NLP_AutoRegressive(nam_,params,role,flag)  {
    assert(arch==MODEL_ARCH::NLP_MAMBA);
    bool worst_case = true;
    // isLoadTokenEmbed = true;
    // hparams.common.adam_alpha = 0.0001;     // 
}

hGensor LLM_MAMBA::build_layer_( int N,struct ggml_context *ctx_compute,hGensor inpL,std::shared_ptr<QKV_LAY> layer,hGensor KQ_pos,int flag) {
    int il = layer->id,nLay=layers.size();
    LAMA *lam = lama();      assert(lam!=nullptr);    
    
    /*  [4096,512] [4096]    */
    hGensor cur = ggml_rms_norm(ctx_compute, inpL, hparams.f_norm_rms_eps);     set_name(cur, "norm");
    hGensor t11 = ggml_repeat(ctx_compute, layer->att_norm.w, cur);          
    set_name(t11, "t11");     //assert_shape_2d(t03, n_embd, N*n_batch);
    cur = ggml_mul(ctx_compute, cur, t11);                    set_name(cur, "attn_norm");
//  cur = mamba_build_layer(ctx_compute,lctx,gf,cur,inpL,il,n_layer,n_tokens,kv_head,n_kv,n_outputs);
    cur = mamba_build_layer(ctx_compute, *(lam->_ctx), gf, cur,inpL, il, nLay,512);
    return cur;
}


hGensor LLM_MAMBA::BuildTarget( struct ggml_context * ctx,hGensor cur,int flag)  {
    /*int n_vocab = tVocab(),n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    auto train_params = hparams.common;
    preLogits = ggml_reshape_3d(ctx, cur, n_vocab, n_ctx, n_batch);             set_name(preLogits, "preLogits");     
    assert_shape_3d(preLogits, n_vocab, n_ctx, n_batch);
    if(hparams.is({"model","target"},string("OneHot")))
        loss = ggml_cross_entropy_loss_1(ctx, preLogits, target_probs);
    else
        loss = ggml_cross_entropy_loss(ctx, preLogits, target_probs);            
                   
    set_name(loss, "loss");     assert_shape_1d(loss, 1);
    ggml_build_forward_expand(gf, loss);
    if (train_params.use_checkpointing) {
        if(gb!=nullptr) 
            ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        if(gb!=nullptr){
            ggml_graph_cpy(gf, gb);
            ggml_build_backward_expand(ctx, gf, gb, true);            
        }
    } */
   return nullptr;   
}


GPT2::GPT2( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag) : NLP_AutoRegressive(nam_,params,role,flag)  {
    assert(arch==MODEL_ARCH::NLP_GPT2);
    isBias = true;
    
}

static void cb(struct ggml_tensor * cur, const char * name, int il)    {
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
    const bool full_offload = false;    //lctx.n_gpu_layers > (int)lctx.hparams.n_layer;
    if(full_offload) { //batch.n_tokens < 32 ||
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

/*
    case LLM_ARCH_GPT2:
                {
                    model.tok_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
                    model.pos_embd = ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});

                    // output
                    {
                        model.output_norm   = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
                        model.output_norm_b = ml.create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
                        model.output        = ml.create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
                    }

                    for (int i = 0; i < n_layer; ++i) {
                        ggml_context * ctx_layer = ctx_for_layer(i);
                        ggml_context * ctx_split = ctx_for_layer_split(i);

                        auto & layer = model.layers[i];

                        layer.attn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
                        layer.attn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

                        layer.wqkv = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
                        layer.bqkv = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

                        layer.wo   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
                        layer.bo   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

                        layer.ffn_norm   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
                        layer.ffn_norm_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

                        layer.ffn_down   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
                        layer.ffn_down_b = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

                        layer.ffn_up     = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
                        layer.ffn_up_b   = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
                    }
                } break;
*/

int DTS_GPT2::stream2token(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag)    {
    // // bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, const gpt_params & params) {
    // gpt_vocab vocab;
    // std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);
    return 0x0;
}

CDict_GPT2::CDict_GPT2(NLP_AutoRegressive *nlp_,int flag)   : ConsiceDict(nlp_,flag)   {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;
    n_vocab=50257;    
}

int CDict_GPT2::InitMAEC(struct ggml_context *ctx_compute,const std::vector<int>& dims_,int flag)  {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;

    tok_embeddings = hLM->AddTensor(ctx_compute,_NAM_("token_embd.weight"),GGML_TYPE_F32,{n_embd, n_vocab},true,0x0);  
    _norm.Build("output_norm", {n_embd},0x0);
    _output.Build("output", {n_embd, n_vocab},0x0);  
    assert(gensors.size()==0);
    return 0x0;     
}


struct ggml_cgraph *GPT2::GetRawGraph( struct ggml_context *ctx_compute,bool isBuild,int flag)   { 
    ctx = ctx_compute;
    gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);  //llama_model_max_nodes(model)
    size_t sz2 = 0;
    const int n_embd_head = hparams.n_embd_head_v,n_embd_gqa  = hparams.n_embd_v_gqa();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);    
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;
    // if(n_ctx_train!=n_ctx)      n_ctx_train = n_ctx;
    int n_tokens=n_ctx*n_batch,n_head,n_embd_head_v=hparams.n_embd_head_v,n_ff;
    const uint32_t n_layer = hparams.nLayer();
    for(int i=0;i<n_layer;i++){
        auto  layer = std::make_shared<QKV_LAY>(this,i);
        layer->isLast = i==n_layer-1;
        layers.push_back(layer); 
    }
    hGensor cur = nullptr,pos = nullptr;
    float kq_scale = 1.0f/sqrtf(float(n_embd_head));
    hDict->InitMAEC(ctx_compute,{n_embd},0x0);    

    InitEntryTensors();
    pos_embd = AddTensor(ctx_compute,_NAM_("position_embed.weight"),GGML_TYPE_F32,{n_embd, n_ctx_train},true,0x0); //ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});
    //llm_build_inp_embd(ctx_compute, lctx, hparams, batch, tok_embd, cb);
    hGensor  inp_tokens = tokens_input;    
    if(n_batch>1){
        inp_tokens = ggml_reshape_1d(ctx_compute, tokens_input, n_ctx*n_batch);   set_name(inp_tokens, "inp_tokens_1d");
    }
    
    hGensor  inpL = ggml_get_rows(ctx_compute, hDict->tok_embeddings, inp_tokens);    set_name(inpL, "inp_embd"); 
    build_inp_KQ_(ctx_compute,false);     // init KQ_pos
    hGensor inp_pos = KQ_pos;           //build_inp_pos();    
    // hGensor KQ_mask = build_inp_KQ_mask();
    pos = ggml_get_rows(ctx_compute, pos_embd, KQ_pos);
    cb(pos, "pos_embd", -1);
    inpL = ggml_add(ctx_compute, inpL, pos);
    cb(inpL, "inpL", -1);

    for (int il = 0; il < layers.size(); ++il) {  
        n_head=hparams.n_head(il);          assert(n_embd_head*n_head==n_embd);
        n_ff=hparams.n_ff(il);
        // sprintf(nam_,"block.%d.attn_norm",il);
        //  llm_build_norm(ctx_compute, inpL, hparams,layers[il].attn_norm,layers[il].attn_norm_b,LLM_NORM, cb, il);
        auto layer = dynamic_pointer_cast<QKV_LAY>(layers[il]);    
        layer->att_norm.Build(_NAM_("block.%d.attn_norm",il),{n_embd},0x0);
        cur = layer->att_norm.Build_2(ctx_compute,inpL,0x0);
        cb(cur, _NAM_("attn_norm"), il);
        // // cur = llm_build_lora_mm(lctx, ctx_compute, layers[il].wqkv, cur);
        layer->Q.Build(_NAM_("block.%d.attn_qkv",il),{n_embd, n_embd + 2*n_embd_gqa},0x0);
        cur = layer->Q.Build_2(ctx_compute,cur,0x0);
        cb(cur, "wqkv", il);      
        hGensor Q2= ggml_view_2d(ctx_compute, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd));
        hGensor K2= ggml_view_2d(ctx_compute, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd));
        hGensor V2= ggml_view_2d(ctx_compute, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa));
        cb(Q2, "Q2", il);        cb(K2, "K2", il);        cb(V2, "V2", il);
        hGensor Qcur = ggml_cont(ctx_compute,Q2 );
        hGensor Kcur = ggml_cont(ctx_compute,K2 );
        hGensor Vcur = ggml_cont(ctx_compute,V2 );
        cb(Qcur, "Qcur", il);        cb(Kcur, "Kcur", il);        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx_compute, Qcur, n_embd_head, n_head, n_tokens);
        ggml_build_forward_expand(gf, Qcur);    ggml_build_forward_expand(gf, Kcur);    ggml_build_forward_expand(gf, Vcur);
        struct ggml_tensor * q = ggml_permute(ctx_compute, Qcur, 0, 2, 1, 3);   
        struct ggml_tensor * k =  ggml_reshape_3d(ctx_compute, Kcur, n_embd_head, n_head, n_tokens);
        k = ggml_permute(ctx_compute, k, 0, 2, 1, 3);
        struct ggml_tensor * kq = ggml_mul_mat(ctx_compute, k, q);
        cb(kq, "kq", il);
        kq = ggml_soft_max_ext(ctx_compute, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max_ext", il);
        struct ggml_tensor * v =  ggml_reshape_3d(ctx_compute, Vcur, n_embd_head, n_head, n_tokens);
        v = ggml_cont(ctx_compute,ggml_permute(ctx_compute, v, 1, 2, 0, 3));
        struct ggml_tensor * kqv = ggml_mul_mat(ctx_compute, v, kq);
        cb(kqv, "kqv", il);
        struct ggml_tensor * kqv_merged = ggml_permute(ctx_compute, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);
        cur = ggml_cont_2d(ctx_compute, kqv_merged, n_embd_head_v*n_head, n_tokens);
        cb(cur, "kqv_merged_cont", il);
        ggml_build_forward_expand(gf, cur);        

        if (layer->isLast) {            // skip computing output for unused tokens
            // hGensor inp_out_ids = nullptr;  //build_inp_out_ids();
            // cur  = ggml_get_rows(ctx_compute,  cur, inp_out_ids);
            // inpL = ggml_get_rows(ctx_compute, inpL, inp_out_ids);
        }
        // add the input
        hGensor ffn_inp = ggml_add(ctx_compute, cur, inpL);        cb(ffn_inp, "ffn_inp", il);
        layer->ffn_norm.Build(_NAM_("block.%d.ffn_norm",il),{n_embd},0x0);
        cur = layer->ffn_norm.Build_2(ctx_compute,inpL,0x0);
        cb(cur, _NAM_("ffn_norm"), il);
        layer->up.Build(_NAM_("block.%d.ffn_up",il),{n_embd, n_ff},0x0);
        cur = layer->up.Build_2(ctx_compute,cur,0x0);
        cb(cur, "ffn_up", il);
        // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
        cur = ggml_silu(ctx, cur);                cb(cur, "ffn_silu", il); 
        layer->down.Build(_NAM_("block.%d.ffn_down",il),{n_ff, n_embd},0x0);
        cur = layer->down.Build_2(ctx_compute,cur,0x0);
        cb(cur, "ffn_down", il);
        cur = ggml_add(ctx_compute, cur, ffn_inp);
        // cur = lctx.cvec.apply_to(ctx_compute, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    
    cur = BuildTarget(ctx_compute,cur);
    ggml_build_forward_expand(gf, cur);
    
    if(0){  //only for debug
        string suffix="", prefix="\t"; 
        __repr__(suffix,prefix);
        TGraph("gpt_raw",gf,true).__repr__(suffix,prefix,cur);  
    }
    if(rnd==nullptr)
        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    if(back_data==nullptr){
        back_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx_compute, ggml_backend_cpu_buffer_type());
        sz2=ggml_backend_buffer_get_size(back_data); 
    }
    return gf;
}

hGensor GPT2::BuildTarget(struct ggml_context * ctx_compute,hGensor inpL,int flag) {
    assert(hDict!=nullptr);
    hGensor cur = hDict->_norm.Build_2(ctx_compute,inpL,0x0);                    cb(cur, "result_norm", -1);
    cur = hDict->_output.Build_2(ctx_compute,cur,0x0);                         cb(cur, "result_output", -1);
    
    /*hGensor _tNorm = hDict->norm;
    const float rms_norm_eps = hparams.f_norm_rms_eps; 
    hGensor mw,t31 = ggml_rms_norm(ctx_compute, inpL, rms_norm_eps);                   cb(t31, "t31", -1);
    if(isTrain()){
        mw = ggml_repeat(ctx_compute, _tNorm, t31);   //_tNorm shoud same shape as t31 if has grad!
        set_name(mw, "_tNorm.repeat");     
    }else
        mw = _tNorm ;  
    hGensor cur = ggml_mul(ctx_compute, t31, mw);             cb(cur, "result_norm", -1);
    cb(cur, "result_output", -1);*/
    return cur;
}
