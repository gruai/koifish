/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  Mixture of Expert
 * 
 *  \brief Mixture of Expert
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

bool LLaMeta::lama_layer::CreateFFN(const CLI_params&hparams, ggml_context *ctx, FFN_TYPE tpFFN, int flag)  {
    const uint32_t n_embd = hparams.n_embd, n_ctx = hparams.n_ctx(), n_ff = hparams.n_ff, n_batch = hparams.n_batch();  
    const uint32_t n_expert = hparams.n_expert;
    switch(tpFFN){
    case VAR_LAST:
    case SWIGLU:
        if(hparams.ZMUV_ratio>0)
            ffn_norm = nullptr;  
        else
            ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff); 
        if(tpFFN==VAR_LAST && isLast){/*i==n_layer-1*/
            eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_batch*n_ctx);   
        }
        break;
    case ONLY_LNormal:
    case ONLY_RMSNormal:
        ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        break;
    case VAR_0:
        eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_batch*n_ctx);   
        break;  
    case GATE_CYS:
        assert(0);
        break;
    case SMOE:{// MoE branch
        assert(n_expert>0 && hparams.n_expert_used>0) ; 
        ffn_gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_expert);
        // ffn_gate_inp = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});        
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / hparams.n_expert_used;
        ffn_gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp, n_expert);
        // ffn_gate_exps = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        ffn_down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff_exp,   n_embd, n_expert); 
        // ffn_down_exps = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
        ffn_up_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp, n_expert); 
        // ffn_up_exps   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

        // Shared expert branch
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;
        ffn_gate_inp_shexp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); 
        // ffn_gate_inp_shexp = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd});
        ffn_gate_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff_shexp);
        // ffn_gate_shexp = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp});
        ffn_down_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff_shexp, n_embd);
        // ffn_down_shexp = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd});
        ffn_up_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff_shexp);
        // ffn_up_shexp   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp});
    }
        break;
    default:
        TO_DO;
    } 
    return true;  
}

LLM_MOE::LLM_MOE( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag) : LLaMeta(nam_,params,role,flag)  {
    assert(arch==MODEL_ARCH::NLP_MOE);
    
    tpFFN = FFN_TYPE::GATE_CYS;    
    // tpFFN = FFN_TYPE::SMOE;    
}

size_t LLM_MOE::MostMemSize(int flag)  {
    int n_layer = nLayerX<=0 ? hparams.n_layer : nLayerX;
    int nHead = hDict!=nullptr ? hDict->nLevel*3+2+6 : 6; 
    size_t sz0 = ggml_tensor_overhead(),sz = sz0*2*(nHead + n_layer*18);
    return sz;
}    



hGensor LLM_MOE::build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor  KQ_pos,/*hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down,*/ int flag) {
    auto train_params = hparams.common;
    int n_vocab = hDict->n_vocab,n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd,n_head = hparams.n_head,n_ff = hparams.n_ff;
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;  
    const float kv_scale = 1.0f/sqrtf(float(hparams.n_embd)/hparams.n_head);
    const int n_past = 0, n_head_kv=hparams.n_head_kv,n_embd_head = hparams.n_embd_head();
    hGensor wq = UpdateGensor (layer->wq->name);                     
    hGensor wk = layer->wk==nullptr ? nullptr : UpdateGensor (layer->wk->name);
    hGensor wv = layer->wv==nullptr ? nullptr : UpdateGensor (layer->wv->name);
    hGensor wo = UpdateGensor (layer->wo->name);
    hGensor attention_norm = UpdateGensor (layer->attention_norm->name);    
    hGensor ffn_norm = layer->ffn_norm==nullptr ? nullptr : UpdateGensor (layer->ffn_norm->name); 
    hGensor ffn_up = nullptr,ffn_gate=nullptr,ffn_down=nullptr;
    if(layer->ffn_up!=nullptr){                
        ffn_up = UpdateGensor (layer->ffn_up->name);
        ffn_gate = UpdateGensor (layer->ffn_gate->name);
        ffn_down = UpdateGensor (layer->ffn_down->name);                
    }  

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
    case VAR_LAST:
    case SWIGLU:    {
        hGensor  t22 = ggml_rms_norm     (ctx_compute, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        ffn = t22;
        if(ffn_norm!=nullptr)       {
            hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            hGensor  t24 = ggml_mul          (ctx_compute, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch); 
            ffn = t24;                 
        }
          
        if(ffn_up!=nullptr){
            // hGensor  t22 = ggml_rms_norm     (ctx_compute, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
            // hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            // hGensor  t24 = ggml_mul          (ctx_compute, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
            hGensor  t24 = ffn;
            hGensor  t25 = ggml_mul_mat      (ctx_compute, ffn_up, t24);                      set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
            hGensor  t26 = ggml_mul_mat      (ctx_compute, ffn_gate, t24);                    set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
            hGensor  t27 = ggml_silu         (ctx_compute, t26);                                    set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
            hGensor  t28 = ggml_mul          (ctx_compute, t27, t25);                               set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
            hGensor  t29 = ggml_mul_mat      (ctx_compute, ffn_down, t28);                    set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
            hGensor  t30 = ggml_add          (ctx_compute, t29, t21);                               set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
            ffn = t30;
        }
        if(layer->eps!=nullptr){
            // hGensor  t300 = ffn!=nullptr ? ggml_rms_norm(ctx_compute, ffn, f_norm_rms_eps) : ggml_rms_norm(ctx_compute, t21, f_norm_rms_eps);
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_compute, layer->eps, 0.001);
            ffn = ggml_add          (ctx_compute, ffn,noise);     
        }else{
        }
        return ffn;  
    } 
    case GATE_CYS:{
    }
        break;
    case SMOE:{
        LAMA *lam = lama();      assert(lam!=nullptr); 
        //  build_qwen2moe:     LLM_FFN_SILU,false,false,
        ffn = moe_build_ffn(ctx_compute, *(lam->_ctx), cur,
                    layer->ffn_gate_inp,layer->ffn_up_exps,layer->ffn_gate_exps,layer->ffn_down_exps,
                    n_expert, n_expert_used,false,false, 0.0,layer->id);
        }
        return ffn;  
        break;
    case VAR_0:
    case ONLY_LNormal:
    case ONLY_RMSNormal:    {
        assert(ffn_up==nullptr);
        hGensor  t22 = tpFFN==ONLY_LNormal ? ggml_norm(ctx_compute, t21, f_norm_rms_eps) :
            ggml_rms_norm(ctx_compute, t21, f_norm_rms_eps);
        set_name(t22, "t22");               assert_shape_2d(t22, n_embd, N*n_batch);   
        if(tpFFN==VAR_0)     {
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_compute, layer->eps, 0.001);
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
    return nullptr;             
}

hGensor MixOfModels::Forward(struct ggml_context * ctx,hGensor cur,hGensor w){
    int n_vocab=cur->ne[0],n_ctx=cur->ne[1],n_batch=cur->ne[2];
    hGensor expert = ggml_reshape_2d(ctx,cur,n_vocab,n_ctx*n_batch);
    hGensor wB = _repeat(ctx,w,expert);
    return ggml_mul(ctx,expert,wB);
}

void MixOfSwarm::Init(tpSWARM&swarm,struct ggml_context *ctx,int n_embd,int flag){
    hGensor a=nullptr,b=nullptr;
    for(auto fish:swarm){
        b = fish->Output();
        // assert(b!=nullptr && b->data!=nullptr);
        exs.push_back(b);
        if(a==nullptr){
            a = b;
        }else{
            assert(ggml_are_same_shape(a, b));
        }
    }
    gat_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, swarm.size()+1);
}

hGensor MixOfSwarm::Build(CLI_params&hparams,struct ggml_context * ctx,hGensor cur,int flag )  { 
    // return cur;
    int n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    int i=0,nSwarm=exs.size()+1;      
    size_t offset = gat_->nb[0];       assert(offset==4);  
    size_t N0=cur->ne[0],ld1=(nSwarm)*offset,nToken=cur->ne[1],N1=nToken;    
    assert(nSwarm == gat_->ne[1]);

//[nWiki+1, n_tokens] := [n_embed,nWiki+1]x[n_embed,n_tokens]  
    hGensor w_ = ggml_mul_mat(ctx, gat_, ggml_reshape_2d(ctx,cur,N0,nToken));           
    if(isSiLU){ //maybe useful
        w_ = ggml_silu(ctx,w_);
    }
    hGensor probs = ggml_soft_max(ctx,w_);        
    // hGensor tA = ggml_reshape_2d(ctx,curlogits,n_vocab,n_ctx*n_batch);      //[nVocab, n_ctx*nBatch] 
    hGensor wA = ggml_view_2d(ctx, probs, 1, nToken,ld1, 0);         //[1, n_ctx*nBatch] ne0,ne1,nb1,offset
    //  wA = _repeat(ctx,wA,tA);   
    hGensor ouput = Forward(ctx,cur,wA);     //ggml_mul(ctx,tA,wA);
    for(auto tB : exs){
        i++;  
        // float *logistB = (float*)(tB->data);    //always nullptr    
        hGensor wB = ggml_view_2d(ctx, probs, 1, nToken,ld1, offset*i);  //ne0,ne1,nb1,offset
        hGensor expert = Forward(ctx,tB,wB);          
        // wB = _repeat(ctx,wB,expert);
        ouput = ggml_add(ctx,ouput,expert);       //ggml_mul(ctx,expert,wB)
    }
       
    // ouput = ggml_reshape_3d        (ctx, ouput, N0, n_ctx, n_batch);
    if(isRes){
        ouput = ggml_add(ctx,ouput,cur);   
    }
    return ouput;
}

hGensor LLaMeta::build_gate(struct ggml_context * ctx,hGensor cur,hGensor curlogits, int flag )  {
    bool isRes = true,isSiLU=false;

    int n_vocab = hDict->tVocab(),n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    int nWiki = wikis.size(),i=0;        CHILD_0909_WIKIS
    size_t N0=cur->ne[0],ld1=(nWiki+1)*cur->nb[0];    
    assert(nWiki+1 == mom.embed2w->ne[1]);
      
    hGensor gat_ = mom.embed2w;
//[nWiki+1, n_tokens] := [n_embed,nWiki+1]x[n_embed,n_tokens]  
    hGensor w_ = ggml_mul_mat(ctx, mom.embed2w, ggml_reshape_2d(ctx,cur,N0,n_ctx*n_batch));           
    if(isSiLU){ //maybe useful
        w_ = ggml_silu(ctx,w_);
    }
    hGensor probs = ggml_soft_max(ctx,w_);    
    size_t offset = probs->nb[0];
    // hGensor tA = ggml_reshape_2d(ctx,curlogits,n_vocab,n_ctx*n_batch);      //[nVocab, n_ctx*nBatch] 
    hGensor wA = ggml_view_2d(ctx, probs, 1, n_ctx*n_batch,ld1, 0);         //[1, n_ctx*nBatch] ne0,ne1,nb1,offset
    //  wA = _repeat(ctx,wA,tA);   
    hGensor ouput = mom.Forward(ctx,curlogits,wA);     //ggml_mul(ctx,tA,wA);
    for(auto wiki : wikis){
        i++;  
        hGensor tB = wiki->exLogits;
        float *logistB = (float*)(tB->data);
        if(wiki->t2t!=nullptr){
            tB = ggml_mul_mat(ctx, wiki->t2t, tB);     
        }else {
            // tB = ggml_soft_max(ctx,tB);  maybe bad choice
        }
        hGensor wB = ggml_view_2d(ctx, probs, 1, n_ctx*n_batch,ld1, offset*i);  //ne0,ne1,nb1,offset
        hGensor expert = mom.Forward(ctx,tB,wB);          
        // wB = _repeat(ctx,wB,expert);
        ouput = ggml_add(ctx,ouput,expert);       //ggml_mul(ctx,expert,wB)
    }
       
    ouput = ggml_reshape_3d        (ctx, ouput, n_vocab, n_ctx, n_batch);
    if(isRes){
        ouput = ggml_add(ctx,ouput,curlogits);   
    }
    return ouput;
}
 /*if(0)   {  //yun!!! GGML_OP_CONCAT don't support backward                
        hGensor t101 = ggml_concat(ctx,tA,tB,0);
        hGensor t102 = ggml_mul_mat(ctx,t101,gate);
        //probs     [n_expert, n_tokens]
        //weights   [1, n_expert_used, n_tokens]
        hGensor weight = ggml_soft_max(ctx,t102);   
        // hGensor t103 = ggml_reshape_3d(ctx, t101, n_vocab,2, N*n_batch);
        wA=ggml_get_rows(ctx,weight,gate_id0),wB=ggml_get_rows(ctx,weight,gate_id1);
    }else{
        wA = ggml_mul_mat(ctx,tA,gate);     wB = ggml_mul_mat(ctx,tB,gate);
    }
    wA = _repeat(ctx,ggml_reshape_2d(ctx, wA, 1,n_ctx*n_batch),tA);
    // tA=ggml_mul(ctx,tA,wA);       
    wB = _repeat(ctx,ggml_reshape_2d(ctx, wB, 1,n_ctx*n_batch),tB);
    // tB=ggml_mul(ctx,tB,wB); */