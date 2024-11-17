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
    hGensor cur = ggml_rms_norm(ctx_compute, inpL, hparams.f_norm_rms_eps);     gTN(cur, "norm");
    hGensor t11 = ggml_repeat(ctx_compute, layer->att_norm.w, cur);          
    gTN(t11, "t11");     //assert_shape_2d(t03, n_embd, N*n_batch);
    cur = ggml_mul(ctx_compute, cur, t11);                    gTN(cur, "attn_norm");
//  cur = mamba_build_layer(ctx_compute,lctx,gf,cur,inpL,il,n_layer,n_tokens,kv_head,n_kv,n_outputs);
    cur = mamba_build_layer(ctx_compute, *(lam->_ctx), ForwarGraph(), cur,inpL, il, nLay,512);
    return cur;
}


hGensor LLM_MAMBA::BuildTarget( struct ggml_context * ctx,hGensor cur,int flag)  {
    /*int n_vocab = tVocab(),n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    auto train_params = hparams.common;
    preLogits = ggml_reshape_3d(ctx, cur, n_vocab, n_ctx, n_batch);             gTN(preLogits, "preLogits");     
    assert_shape_3d(preLogits, n_vocab, n_ctx, n_batch);
    if(hparams.is({"model","target"},string("OneHot")))
        loss = ggml_cross_entropy_loss_1(ctx, preLogits, target_probs);
    else
        loss = ggml_cross_entropy_loss(ctx, preLogits, target_probs);            
                   
    gTN(loss, "loss");     assert_shape_1d(loss, 1);
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
    isBias = false;    //   if true, converge much slower
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
int DTS_GPT2::stream2token(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag)    {
    // bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, const gpt_params & params) {
    gpt_vocab vocab;
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);
    return 0x0;
}*/

CDict_GPT2::CDict_GPT2(NLP_AutoRegressive *nlp_,int flag)   : ConsiceDict(nlp_,flag)   {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;
    n_vocab=50257;    
}

CDict_CHAR::CDict_CHAR(NLP_AutoRegressive *nlp_,int flag)   : ConsiceDict(nlp_,flag)   {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;
    n_vocab=256;    
}
int CDict_CHAR::InitMAEC(struct ggml_context *ctx_compute,const std::vector<int>& dims_,int flag)  {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;

    tok_embeddings = hLM->AddTensor(ctx_compute,_NAM_("token_embd.weight"),GGML_TYPE_F32,{n_embd, n_vocab},true,0x0);  
    _norm.Build("output_norm", {n_embd},0x0);
    _output.isBias = false;
    _output.Build("output", {n_embd, n_vocab},0x0);  
    assert(gensors.size()==0);
    return 0x0;     
}
int CDict_CHAR::stream2token(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag){
    int n_tokens = 0, nMost = btch.size(); 
    assert(txt_len<=nMost);
    unsigned char *a = (unsigned char*)(txt);
    for(int i=0;i<txt_len;i++,a++)  {
        TOKEN_ID t=(TOKEN_ID)(*a);
        assert(t>=0 && t<n_vocab);
        btch[i] = t;
        n_tokens++;
    }
    return n_tokens;
}

int CDict_GPT2::InitMAEC(struct ggml_context *ctx_compute,const std::vector<int>& dims_,int flag)  {
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;

    tok_embeddings = hLM->AddTensor(ctx_compute,_NAM_("token_embd.weight"),GGML_TYPE_F32,{n_embd, n_vocab},true,0x0);  
    _norm.Build("output_norm", {n_embd},0x0);
    _output.isBias = false;
    _output.Build("output", {n_embd, n_vocab},0x0);  
    assert(gensors.size()==0);
    return 0x0;     
}

hNeuron GPT2::J2Neuron(struct ggml_context *ctx_compute,string dad,const JSON& j,int flag){
    hNeuron hN=nullptr,cur=nullptr;
    std::vector<hNeuron> gang;    
    string k;
    for(auto it = j.begin(); it != j.end(); ++it)    {
        k =it.key();        
        auto v=it.value();
        if(it->is_array()){
            cur = GeNeuron::MakeInstance(this,ctx_compute,dad,k,v,flag);        
        }else if(it->is_structured())        {
            cur =  J2Neuron(ctx_compute,k,*it,flag);            
        }
        else        {
            cur = GeNeuron::MakeInstance(this,ctx_compute,dad,k,v,flag);              
        }
        neurons.push_back(cur);
        gang.push_back(cur);
    }
    assert(gang.size()>0);
    if(gang.size()>1)   {
        hN = std::make_shared<Ganglia>(ctx_compute,dad,gang,flag);
        neurons.push_back(hN);  
    }else{
        assert(cur!=nullptr);
        hN = cur;
    }
    assert(hN->isValid());
    return hN;
}

struct ggml_cgraph *GPT2::jRawGraph( struct ggml_context *ctx_compute,bool isBuild,int flag)   {
    J2Neuron(ctx_compute,"GPT2",hparams.jModel,flag);

    hGensor cur = nullptr;
    for(auto nn : neurons){
        cur = nn->Forward(ctx_compute,cur);
    }
    
    return nullptr;
}

struct ggml_cgraph *GPT2::GetRawGraph( struct ggml_context *ctx_compute,bool isBuild,int flag)   { 
    return jRawGraph(ctx_compute,isBuild,flag);

    bool isOnlinePush = true;      // push nodes last or online(QKV)
    ctx = ctx_compute;
    // gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);  
    hForwTG = std::make_shared<TGraph>(this,"gpt_raw",ctx_compute,true);
    auto gf = hForwTG->raw();
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
    hGensor cur = nullptr,pos = nullptr,q,k,v,Qcur,Kcur,Vcur;
    float kq_scale = 1.0f/sqrtf(float(n_embd_head));
    hDict->InitMAEC(ctx_compute,{n_embd},0x0);    

    InitEntryTensors();
    
    pos_embd = AddTensor(ctx_compute,_NAM_("position_embed.weight"),GGML_TYPE_F32,{n_embd, n_ctx},true,0x0); //ml.create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});
    //llm_build_inp_embd(ctx_compute, lctx, hparams, batch, tok_embd, cb);
    hGensor  inp_tokens = tokens_input;    
    if(n_batch>1){
        inp_tokens = ggml_reshape_1d(ctx_compute, tokens_input, n_ctx*n_batch);   gTN(inp_tokens, "inp_tokens_1d");
    }    
    hGensor  inpL = ggml_get_rows(ctx_compute, hDict->tok_embeddings, inp_tokens);    gTN(inpL, "inp_embd");    
    build_inp_KQ_(ctx_compute,false);     // init KQ_pos 
    if(1)   {        
        hGensor inp_pos = KQ_pos;           
        pos = ggml_get_rows(ctx_compute, pos_embd, KQ_pos);    cb(pos, "pos_embd", -1);
        float *fpos = (float*)(pos->data);
        inpL = ggml_add(ctx_compute, inpL, pos);     
    }else{
        inpL = ggml_add(ctx_compute, inpL, pos_embd);     
    }   
    cb(inpL, "inp_pe", -1);
    // layers.clear();     cur = inpL;        xn=cur;      xxn=cur->grad;   //only for debug
    for (int il = 0; il < layers.size(); ++il) {  
        n_head=hparams.n_head(il);          assert(n_embd_head*n_head==n_embd);
        n_ff=hparams.n_ff(il);
        // sprintf(nam_,"block.%d.attn_norm",il);
        //  llm_build_norm(ctx_compute, inpL, hparams,layers[il].attn_norm,layers[il].attn_norm_b,LLM_NORM, cb, il);
        auto layer = dynamic_pointer_cast<QKV_LAY>(layers[il]);    
        layer->att_norm.Build(_NAM_("block.%d.attn_norm",il),{n_embd},0x0);     layer->att_norm.sT="a";
        cur = layer->att_norm.Forward(ctx_compute,inpL,0x0);
        cb(cur, _NAM_("attn_norm"), il);
        if(1){  
            layer->Q.Build(_NAM_("block.%d.attn_qkv",il),{n_embd, n_embd + 2*n_embd_gqa},0x0);
            cur = layer->Q.Forward(ctx_compute,cur,0x0);            cb(cur, "wqkv", il);      
            hGensor Q2= ggml_view_2d(ctx_compute, cur, n_embd,     n_tokens, cur->nb[1], 0*sizeof(float)*(n_embd));
            hGensor K2= ggml_view_2d(ctx_compute, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd));
            hGensor V2= ggml_view_2d(ctx_compute, cur, n_embd_gqa, n_tokens, cur->nb[1], 1*sizeof(float)*(n_embd + n_embd_gqa));
            cb(Q2, "Q2", il);        cb(K2, "K2", il);        cb(V2, "V2", il);
            Qcur = ggml_cont(ctx_compute,Q2 ),Kcur = ggml_cont(ctx_compute,K2 ),Vcur = ggml_cont(ctx_compute,V2 );  
        }else{  // Gradient of Q K would vanish,but the training curve nearly same!
            layer->Q.isBias=false;          layer->K.isBias=false;
            layer->Q.Build(_NAM_("block.%d.Q",il),{n_embd, n_embd},0x0);                layer->Q.sT="q";
            layer->K.Build(_NAM_("block.%d.K",il),{n_embd, n_embd_gqa},0x0);            layer->K.sT="k";
            layer->V.Build(_NAM_("block.%d.V",il),{n_embd, n_embd_gqa},0x0);            layer->V.sT="v";
            Qcur = layer->Q.Forward(ctx_compute,cur,0x0);       
            Kcur = layer->K.Forward(ctx_compute,cur,0x0);       
            Vcur = layer->V.Forward(ctx_compute,cur,0x0);
        }
        cb(Qcur, "Qcur", il);        cb(Kcur, "Kcur", il);        cb(Vcur, "Vcur", il);
        if(isAttOnBC){  // attenion on all tokens, memory would explode!
            Qcur = ggml_reshape_3d(ctx_compute, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx_compute, Kcur, n_embd_head, n_head, n_tokens);
            Vcur = ggml_reshape_3d(ctx_compute, Vcur, n_embd_head, n_head, n_tokens);
        }else{
            Qcur = ggml_reshape_4d(ctx_compute, Qcur, n_embd_head, n_head, n_ctx,n_batch);
            Kcur = ggml_reshape_4d(ctx_compute, Kcur, n_embd_head, n_head, n_ctx,n_batch);
            Vcur = ggml_reshape_4d(ctx_compute, Vcur, n_embd_head, n_head, n_ctx,n_batch);
        }
        
        q = ggml_permute(ctx_compute, Qcur, 0, 2, 1, 3);   //eh,ctx,h,b
        k = ggml_permute(ctx_compute, Kcur, 0, 2, 1, 3);  
        v = ggml_cont(ctx_compute,ggml_permute(ctx_compute, Vcur, 1, 2, 0, 3));
        if(isOnlinePush)    {
            ggml_build_forward_expand(gf, q);    ggml_build_forward_expand(gf, k);    ggml_build_forward_expand(gf, v);
        }    

        struct ggml_tensor * kq = ggml_mul_mat(ctx_compute, k, q);        cb(kq, "kq", il);        
        if(1)      {    // nearly same     
            kq = ggml_soft_max_ext(ctx_compute, kq, KQ_mask, kq_scale, hparams.f_max_alibi_bias);       //would 
        }else{
            hGensor  t16_1 = ggml_scale_inplace        (ctx, kq, kq_scale);   
            hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, 0);     
            kq = ggml_soft_max_inplace     (ctx, t16_2);             
        }   
        cb(kq, "kq_soft_max_ext", il);
        
        struct ggml_tensor * kqv = ggml_mul_mat(ctx_compute, v, kq);        // eh,ctx,h,b
        cb(kqv, "kqv", il);
        struct ggml_tensor * kqv_merged = ggml_permute(ctx_compute, kqv, 0, 2, 1, 3); // eh,h,ctx,b
        cb(kqv_merged, "kqv_merged", il);
        if(0)   //  back gradient is zero
            cur = ggml_cont_2d(ctx_compute, kqv_merged, n_embd_head_v*n_head, n_tokens);
        else{
            hGensor kqv_out = ggml_cont(ctx_compute, kqv_merged);              
            cur = ggml_reshape_2d(ctx_compute, kqv_out, n_embd, n_tokens);              
        }        
        cb(cur, "kqv_merged_cont", il);
        layer->proj.Build(_NAM_("block.%d.attn_proj",il),{n_embd, n_embd},0x0);     //why proj is useful?
        cur = layer->proj.Forward(ctx_compute,cur,0x0);            cb(cur, "attn_proj", il); 
        
        if(isOnlinePush)            ggml_build_forward_expand(gf, cur);        

        if (layer->isLast) {            // skip computing output for unused tokens
            // hGensor inp_out_ids = nullptr;  //build_inp_out_ids();
            // cur  = ggml_get_rows(ctx_compute,  cur, inp_out_ids);
            // inpL = ggml_get_rows(ctx_compute, inpL, inp_out_ids);
        }
        // add the input
        hGensor ffn_inp = ggml_add(ctx_compute, cur, inpL);        cb(ffn_inp, "ffn_inp", il);
        layer->ffn_norm.Build(_NAM_("block.%d.ffn_norm",il),{n_embd},0x0);      layer->ffn_norm.sT="f";
        cur = layer->ffn_norm.Forward(ctx_compute,inpL,0x0);
        cb(cur, _NAM_("ffn_norm"), il);
        layer->up.Build(_NAM_("block.%d.ffn_up",il),{n_embd, n_ff},0x0);
        cur = layer->up.Forward(ctx_compute,cur,0x0);
        cb(cur, "ffn_up", il);
        // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
        cur = ggml_silu(ctx, cur);                cb(cur, "ffn_silu", il); 
        layer->down.Build(_NAM_("block.%d.ffn_down",il),{n_ff, n_embd},0x0);
        cur = layer->down.Forward(ctx_compute,cur,0x0);
        cb(cur, "ffn_down", il);
        cur = ggml_add(ctx_compute, cur, ffn_inp);
        // cur = lctx.cvec.apply_to(ctx_compute, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    
    cur = BuildTarget(ctx_compute,cur);
    hForwTG->PushBack(cur);         //0x7fffdfaaba60    
    // ggml_build_forward_expand(gf, cur);
    
    if(0){  //only for debug
        string suffix="", prefix="\t"; 
        __repr__(suffix,prefix);
        TGraph(this,"gpt_raw",gf,true).__repr__(suffix,prefix,cur);  
    }
    if(rnd==nullptr)
        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    sz2 = InitBackEnd(ctx_compute);
    
    return gf;
}

hGensor GPT2::BuildTarget(struct ggml_context * ctx_compute,hGensor inpL,int flag) {
    assert(hDict!=nullptr);
    hGensor cur = hDict->_norm.Forward(ctx_compute,inpL,0x0);                    cb(cur, "result_norm", -1);
    cur = hDict->_output.Forward(ctx_compute,cur,0x0);                         cb(cur, "result_output", -1);
    preLogits = cur;
    out_node = BuildLoss(ctx_compute,preLogits);

    return cur;
}

string GPT2::__repr__( string& suffix,string& prefix,int flag) {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    string sBasic = NLP_AutoRegressive::__repr__(suffix,prefix,flag);
    sprintf(buf+strlen(buf),"%s",sBasic.c_str()); 
    _INFO("GPT2:    Bias=%d AttOnBC=%d\n========\n",isBias,isAttOnBC); 
    return buf;
}