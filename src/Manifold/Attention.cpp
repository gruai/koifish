#include "Fish.hpp"
#include "TGraph.hpp"
#include "Cache.hpp"

bool BROWN_Motion::Transfer_1 = true;  //if true,  wQ is large & need more time to train
//  bool BROWN_Motion::Transfer_1 = false;  

void inline set_name(struct ggml_tensor * t, const char * n) {
    ggml_set_name(t, n);
    if (t->grad) {
        ggml_format_name(t->grad, "%s->grad", n);
    }
};

BROWN_Motion::BROWN_Motion(Fish *hF_,hGensor _wq, hGensor _wv,struct CLI_params& hparams,hLQKV lQKV,int flags) : hFish_(hF_),wq(_wq),wv(_wv),lay(lQKV)   {
    int layer_id = lay->id;
    version = hparams.Get({"model","attention","version"},version,false);
    isTrain = _wq->grad!=nullptr;
    f_norm_rms_eps  = hparams.f_norm_rms_eps;
    rope_freq_base  = hparams.rope_freq_base;
    rope_freq_scale = hparams.rope_freq_scale;  
    f_max_alibi_bias = hparams.f_max_alibi_bias;
    attn_soft_cap = hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f;
    use_flash = hparams.isFlashAtten();
    // float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    // int n_embd_gqa = hparams.n_embd_gqa();
    // int n_embd_head = hparams.n_embd_head( );
    n_head_kv=hparams.n_head_kv(layer_id);
    // n_vocab = hparams.n_vocab;          
    n_batch  = hparams.common.n_batch;          
    n_ctx_orig = hparams.n_ctx_orig();                  n_ctx = hparams.n_ctx();            
    // n_tokens = n_ctx*n_batch;             
    n_embd = hparams.n_embd;
    n_head = hparams.n_head(layer_id),                  n_rot = hparams.n_rot,              n_ff = hparams.n_ff(layer_id);
    n_embd_head = hparams.n_embd_head(layer_id);
    
    rope_type = hparams.rope_type;
    N = n_ctx;

    n_embd_head_k = hparams.n_embd_head_k;
    n_embd_k_gqa  = hparams.n_embd_k_gqa(layer_id);
    n_embd_head_v = hparams.n_embd_head_v;
    n_embd_v_gqa  = hparams.n_embd_v_gqa(layer_id);    

    kv = hF_->hCache;
}

/*
    Rotary Position Embedding
*/
hGensor BROWN_Motion::W_rope(struct ggml_context *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,int flag)   {
    /*  ggml_rope_impl(
        ctx, a, b, NULL, n_dims, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false
    */
    hGensor  t05 = w==nullptr ? cur : ggml_mul_mat      (ctx, w, cur);                          
    //set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
    hGensor  t06 = ggml_reshape_4d   (ctx, t05,shape[0],shape[1],shape[2],shape[3]); //n_embd_head, n_head, N, n_batch
    //set_name(t06, "t06");     
    assert_shape_4d(t06, shape[0],shape[1],shape[2],shape[3]);
    //                                                  32      64          10000           1        
    const int rope_mode = 0;
    hGensor  t07 = n_embd_head==1 ? t06 :
        ggml_rope_ext(ctx, t06, KQ_pos, nullptr, n_rot, rope_mode, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    // CYS_0826 hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f);
    hGensor  t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);    //  [24,6,512,32] => [24,512,6,32]
    return t13;
}

hGensor BROWN_Motion::DiffusionOnEmbed(struct ggml_context *ctx, hGensor teb, hGensor KQ_pos){
    assert(Transfer_1==false);
    const float kq_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
    auto gf_ = hFish_->ForwarGraph();
    hGensor  t13 = nullptr, t14 = nullptr, kqv_out=nullptr;
    auto shape=teb->ne;
    assert_shape_2d(teb, n_embd, N*n_batch);
    assert(wq!=nullptr);
    
    hGensor wq_2 = wq,v = teb,v4 = nullptr,v3=nullptr;
    if(!Transfer_1){  //scale wq
        hGensor  t16_1 = ggml_scale_inplace        (ctx, wq, kq_scale);  
        wq_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past); 
    }
    hGensor  kq = ggml_soft_max_inplace     (ctx, wq_2);  
    ggml_build_forward_expand(gf_,kq); 
    if(wv!=nullptr){
        v = ggml_mul_mat(ctx, teb, wv); 
        v4 = ggml_reshape_4d(ctx, v, N, n_batch, n_embd_head, n_head_kv);   
    }else{
        hGensor v_rope = ggml_reshape_4d   (ctx, teb, n_embd_head, n_head, N, n_batch);
        v_rope = ggml_rope_ext(ctx, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        v_rope = ggml_reshape_2d   (ctx, v_rope, n_embd_head*n_head, N*n_batch);
        v = ggml_mul_mat(ctx, v_rope, wq); 
        v3 = ggml_reshape_3d(ctx, v, N, n_batch, n_embd);        set_name(v3, "v3");   
    }
                
    if(!Transfer_1){
        hGensor  t05 = ggml_mul_mat      (ctx, kq, v);       //[144,144,1,1]x[144,16384,1,1] = [4096,256,1,1]                   
        set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
        hGensor  t06 = ggml_reshape_4d   (ctx, t05, n_embd_head, n_head, N, n_batch); 
        set_name(t06, "t06");     assert_shape_4d(t06, n_embd_head, n_head, N, n_batch);      //[128,32,64,4]   
        hGensor  t07 =  ggml_rope_ext(ctx, t06, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                //ggml_rope_custom   (ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f); 
        set_name(t07, "t07");     assert_shape_4d(t07, n_embd_head, n_head, N, n_batch);      // [128,6,32,8]    
        kqv_out = t07;
    }else if(v3!=nullptr){
        // if(isSiLU){ //maybe useful
        //     v3 = ggml_silu(ctx,v3);
        // }  
        // v3 = ggml_scale_inplace        (ctx, v3, kq_scale);  
        // v3 = ggml_diag_mask_inf_inplace(ctx, v3, n_past);       
        hGensor probs = ggml_soft_max(ctx,v3); 
        hGensor expert = v3;    //ggml_reshape_2d(ctx,v3,n_vocab,n_ctx*n_batch);
        // hGensor wB = _repeat(ctx,probs,expert);
        hGensor kqv = ggml_mul(ctx,expert,probs);
        v4 = ggml_reshape_4d   (ctx, kqv,N, n_batch,n_embd_head, n_head);
        kqv_out = ggml_permute(ctx, v4, 2, 3, 0, 1);       // [24,6,512,32]  
        assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);  
        // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        set_name(kqv_out, "kqv_out_rope");     
        
    } else{     //  Deprecated
        assert(v4!=nullptr);
        set_name(v4, "v4");     
        v = ggml_permute      (ctx, v4, 0, 3, 1, 2); 

        // hGensor v4 = ggml_reshape_4d   (ctx, v, n_embd_head, n_head, N, n_batch); 
        // hGensor  v_rope =  ggml_rope_ext(ctx, v4, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        // set_name(v_rope, "v_rope");     assert_shape_4d(v_rope, n_embd_head, n_head, N, n_batch);      // [128,6,32,8] 
        // v = ggml_permute(ctx, v_rope, 1, 2, 0, 3); 
        hGensor  kqv = ggml_mul_mat      (ctx, v, kq);       //  [512,24,6,32] x [512,512,6,32]    => [24,512,6,32]  
        set_name(kqv, "kqv");     
        kqv_out = ggml_permute(ctx, kqv, 0, 2, 1, 3);       // [24,6,512,32]  
        kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        set_name(kqv_out, "kqv_out_rope");     assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);      
    }      

    kqv_out = ggml_cont(ctx, kqv_out);              
    sprintf(nam_,"kqv_merged_cont-%d",lay->id);    set_name(kqv_out, nam_);     
    kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1]  
    ggml_build_forward_expand(gf_,kqv_out);  
    return kqv_out;   
}

hGensor BROWN_Motion::DiffusionOnToken(struct ggml_context *ctx, hGensor teb, hGensor KQ_pos)   {
    assert(Transfer_1 && wq!=nullptr);
    const float kq_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
    auto gf_ = hFish_->ForwarGraph();
    hGensor v = teb,v3=nullptr,v4=nullptr, t14 = nullptr, kqv_out=nullptr;
    assert_shape_2d(teb, n_embd, N*n_batch);
      
    hGensor v_rope = ggml_reshape_4d   (ctx, teb, n_embd_head, n_head, N, n_batch);
    v_rope = ggml_rope_ext(ctx, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    v_rope = ggml_reshape_2d   (ctx, v_rope, n_embd_head*n_head, N*n_batch);
    v = ggml_mul_mat(ctx, v_rope, wq); 
    v3 = ggml_reshape_3d(ctx, v, N, n_batch, n_embd);        set_name(v3, "v3");       
  
    hGensor probs = nullptr;
    if(wv!=nullptr)   {
        hGensor w_trans = wv;
        hGensor w_ = ggml_mul_mat(ctx, w_trans,teb ); //ggml_reshape_2d(ctx,v3,N, n_batch*n_embd)  
        w_ = ggml_reshape_2d(ctx, w_, N,n_batch);   
        // if(isSiLU){ //maybe useful
        //     w_ = ggml_silu(ctx,w_);
        // } 
        probs = ggml_soft_max(ctx,w_); 
        probs = ggml_repeat(ctx, probs, v3); 
    }else
        probs = ggml_soft_max(ctx,v3); 
    hGensor expert = v3;    //ggml_reshape_2d(ctx,v3,n_vocab,n_ctx*n_batch);
    // hGensor wB = _repeat(ctx,probs,expert);
    hGensor kqv = ggml_mul(ctx,expert,probs);
    v4 = ggml_reshape_4d   (ctx, kqv,N, n_batch,n_embd_head, n_head);
    kqv_out = ggml_permute(ctx, v4, 2, 3, 0, 1);       // [24,6,512,32]  
    assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);  
    // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    set_name(kqv_out, "kqv_out_rope");     

    kqv_out = ggml_cont(ctx, kqv_out);              
    sprintf(nam_,"kqv_merged_cont-%d",lay->id);    set_name(kqv_out, nam_);     
    kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1]  
    ggml_build_forward_expand(gf_,kqv_out);  
    return kqv_out;   
}

//  teb=tokens*embed*batch
hGensor BROWN_Motion::Build(struct ggml_context *ctx ,hGensor teb, hGensor KQ_pos)    {
    hGensor kqv_out=nullptr;
    if(Transfer_1)
        kqv_out = DiffusionOnToken(ctx,teb,KQ_pos);
    else
        kqv_out = DiffusionOnEmbed(ctx,teb,KQ_pos);
    return kqv_out;
    /*const float kq_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
    auto gf_ = hFish_->ForwarGraph();
    hGensor  t13 = nullptr, t14 = nullptr, kqv_out=nullptr;
    auto shape=teb->ne;
    assert_shape_2d(teb, n_embd, N*n_batch);
    assert(wq!=nullptr);
    
    hGensor wq_2 = wq,v = teb,v4 = nullptr,v3=nullptr;
    if(!Transfer_1){  //scale wq
        hGensor  t16_1 = ggml_scale_inplace        (ctx, wq, kq_scale);  
        wq_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past); 
    }
    hGensor  kq = ggml_soft_max_inplace     (ctx, wq_2);  
    ggml_build_forward_expand(gf_,kq); 
    if(wv!=nullptr){
        v = ggml_mul_mat(ctx, teb, wv); 
        v4 = ggml_reshape_4d(ctx, v, N, n_batch, n_embd_head, n_head_kv);   
    }else{
        hGensor v_rope = ggml_reshape_4d   (ctx, teb, n_embd_head, n_head, N, n_batch);
        v_rope = ggml_rope_ext(ctx, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        v_rope = ggml_reshape_2d   (ctx, v_rope, n_embd_head*n_head, N*n_batch);
        v = ggml_mul_mat(ctx, v_rope, wq); 
        v3 = ggml_reshape_3d(ctx, v, N, n_batch, n_embd);        set_name(v3, "v3");   
    }
                
    if(!Transfer_1){
        hGensor  t05 = ggml_mul_mat      (ctx, kq, v);       //[144,144,1,1]x[144,16384,1,1] = [4096,256,1,1]                   
        set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
        hGensor  t06 = ggml_reshape_4d   (ctx, t05, n_embd_head, n_head, N, n_batch); 
        set_name(t06, "t06");     assert_shape_4d(t06, n_embd_head, n_head, N, n_batch);      //[128,32,64,4]   
        hGensor  t07 =  ggml_rope_ext(ctx, t06, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                //ggml_rope_custom   (ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f); 
        set_name(t07, "t07");     assert_shape_4d(t07, n_embd_head, n_head, N, n_batch);      // [128,6,32,8]    
        kqv_out = t07;
    }else if(v3!=nullptr){
        // if(isSiLU){ //maybe useful
        //     v3 = ggml_silu(ctx,v3);
        // }  
        // v3 = ggml_scale_inplace        (ctx, v3, kq_scale);  
        // v3 = ggml_diag_mask_inf_inplace(ctx, v3, n_past);       
        hGensor probs = ggml_soft_max(ctx,v3); 
        hGensor expert = v3;    //ggml_reshape_2d(ctx,v3,n_vocab,n_ctx*n_batch);
        // hGensor wB = _repeat(ctx,probs,expert);
        hGensor kqv = ggml_mul(ctx,expert,probs);
        v4 = ggml_reshape_4d   (ctx, kqv,N, n_batch,n_embd_head, n_head);
        kqv_out = ggml_permute(ctx, v4, 2, 3, 0, 1);       // [24,6,512,32]  
        assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);  
        // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        set_name(kqv_out, "kqv_out_rope");     
        
    } else{     //  Deprecated
        assert(v4!=nullptr);
        set_name(v4, "v4");     
        v = ggml_permute      (ctx, v4, 0, 3, 1, 2); 

        // hGensor v4 = ggml_reshape_4d   (ctx, v, n_embd_head, n_head, N, n_batch); 
        // hGensor  v_rope =  ggml_rope_ext(ctx, v4, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        // set_name(v_rope, "v_rope");     assert_shape_4d(v_rope, n_embd_head, n_head, N, n_batch);      // [128,6,32,8] 
        // v = ggml_permute(ctx, v_rope, 1, 2, 0, 3); 
        hGensor  kqv = ggml_mul_mat      (ctx, v, kq);       //  [512,24,6,32] x [512,512,6,32]    => [24,512,6,32]  
        set_name(kqv, "kqv");     
        kqv_out = ggml_permute(ctx, kqv, 0, 2, 1, 3);       // [24,6,512,32]  
        kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        set_name(kqv_out, "kqv_out_rope");     assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);      
    }      

    kqv_out = ggml_cont(ctx, kqv_out);              
    sprintf(nam_,"kqv_merged_cont-%d",lay->id);    set_name(kqv_out, nam_);     
    kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1]  
    ggml_build_forward_expand(gf_,kqv_out);  
    return kqv_out;   */
}

hGensor QKV_Motion::vXkq(struct ggml_context *ctx, hGensor v,hGensor kq,int layer_id){
    char nam_[128];
    hGensor kqv = ggml_mul_mat(ctx, v, kq);         assert_shape_4d(kqv, n_embd_head, N, n_head, n_batch); 
    sprintf(nam_,"kqv-%d",layer_id);    set_name(kqv, nam_);
    hGensor kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);        assert_shape_4d(kqv_merged, n_embd_head,n_head,N,n_batch); 
    sprintf(nam_,"kqv_merged-%d",layer_id);    set_name(kqv_merged, nam_);
    // kqv_out = ggml_cont_2d(ctx, kqv_merged, n_embd, n_ctx*n_batch);
    hGensor kqv_out = nullptr;
    if(0){  // stuck at local minima
        kqv_out = ggml_cont_2d(ctx, kqv_merged, n_embd, n_ctx*n_batch);
        sprintf(nam_,"kqv_merged_cont-%d",layer_id);    set_name(kqv_out, nam_);         
    }else{      ///home/cys/rnd/lic/log/wiki/MOM_1010.info
        kqv_out = ggml_cont         (ctx, kqv_merged);              // [128,6,32,8]      
        sprintf(nam_,"kqv_merged_cont-%d",layer_id);    set_name(kqv_out, nam_);     
        kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);     
    }
  
    // sprintf(nam_,"kqv-%d",layer_id);            set_name(kqv_out, nam_); 
    return kqv_out;
}

/**
 *  llm_build_kqv   ???
 *  https://github.com/ggerganov/llama.cpp/pull/5021
*/
hGensor QKV_Motion::Build(struct ggml_context *ctx, hGensor teb, hGensor KQ_pos)    {
    use_cache = false;
    char nam_[128];
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1,layer_id=lay->id,il=layer_id,n_kv=kv->n_kv();  
    // int kv_head=0,kv_size = kv->size;  // index of where we store new KV data in the cache     
    float kq_scale = 1.0f/sqrtf(float(n_embd_head));        //0.0883883461
    hGensor  q=nullptr,k=nullptr,v=nullptr,kqv_out=nullptr; //t13 = nullptr, t14 = nullptr, t16=nullptr;
    assert(wk!=nullptr && wv!=nullptr) ;
    auto gf_ = hFish_->ForwarGraph();
    if(version == 0)   {
        q = W_rope(ctx,teb,wq,KQ_pos,{n_embd_head, n_head, N, n_batch});        
        set_name(q, "q");     assert_shape_4d(q, n_embd_head, N, n_head, n_batch);
        k = W_rope(ctx ,teb,wk,KQ_pos,{n_embd_head, n_head_kv, N, n_batch});        
        set_name(k, "k");     assert_shape_4d(k, n_embd_head, N, n_head_kv, n_batch);                
        v = ggml_mul_mat      (ctx, teb, wv);        //[4096,256,1,1]x[4096,1024,1,1]= [256,1024,1,1]                   
        set_name(v, "v");     //assert_shape_2d(t11, N*n_batch, n_embd_gqa);    
        ggml_build_forward_expand(gf_,q);           ggml_build_forward_expand(gf_,k);           ggml_build_forward_expand(gf_,v);           
        hGensor  v4 = ggml_reshape_4d   (ctx, v, N, n_batch, n_embd_head, n_head_kv);      // [64,4,128,8,] 
        set_name(v4, "t12");     assert_shape_4d(v4, N, n_batch, n_embd_head, n_head_kv);
        hGensor  t15 = ggml_permute      (ctx, v4, 0, 3, 1, 2);                                // [64,128,8,4,] 
        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd_head, n_head_kv, n_batch);        
        hGensor  kq = ggml_mul_mat              (ctx, k, q);      
        sprintf(nam_,"kq-%d",layer_id);     set_name(kq, nam_);         assert_shape_4d(kq, N, N, n_head, n_batch);
        if(0)      {
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask, kq_scale, f_max_alibi_bias);       //would crash!
        }else{
            hGensor  t16_1 = ggml_scale_inplace        (ctx, kq, kq_scale);          
                    // set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
            hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);            
                    // set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
            kq = ggml_soft_max_inplace     (ctx, t16_2);                    
                    // set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);            
        }   
        kqv_out = vXkq(ctx,t15,kq,layer_id);    //  [512,24,6,32]x[512,512,6,32]
        /*hGensor kqv = ggml_mul_mat(ctx, t15, t16_3);         assert_shape_4d(kqv, n_embd_head, N, n_head, n_batch);                                             
                // set_name(t16, "t16");     assert_shape_4d(t16, n_embd_head, N, n_head, n_batch);
        hGensor  t17 = ggml_permute      (ctx, kqv, 0, 2, 1, 3);    // [128,6,17,1]                      
        set_name(t17, "t17");     assert_shape_4d(t17, n_embd_head, n_head, N, n_batch);
        kqv_out = ggml_cont         (ctx, t17);                                    
        // set_name(t18, "t18");     assert_shape_4d(t18, n_embd_head, n_head, N, n_batch);
        kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1] */
        
    }else{        
        hGensor  Qcur = ggml_mul_mat      (ctx, wq, teb);     
        sprintf(nam_,"Qcur-%d",layer_id);    set_name(Qcur, nam_);  
        hGensor  Kcur = ggml_mul_mat      (ctx, wk, teb);     
        sprintf(nam_,"Kcur-%d",layer_id);    set_name(Kcur, nam_); 
        hGensor  Vcur = ggml_mul_mat      (ctx, wv, teb);
        sprintf(nam_,"Vcur-%d",layer_id);    set_name(Vcur, nam_); 
        hGensor  rope_factors = lay->rope_(false);         
        Qcur = ggml_rope_ext(
                ctx, ggml_reshape_4d(ctx, Qcur, n_embd_head, n_head, n_ctx,n_batch), KQ_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,ext_factor, attn_factor, beta_fast, beta_slow
            );
        sprintf(nam_,"Qcur_rope-%d",layer_id);    set_name(Qcur, nam_);     assert_shape_4d(Qcur, n_embd_head, n_head, N, n_batch);
        Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_4d(ctx, Kcur, n_embd_head, n_head_kv, n_ctx,n_batch), KQ_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,ext_factor, attn_factor, beta_fast, beta_slow
                );
        sprintf(nam_,"Kcur_rope-%d",layer_id);      set_name(Kcur, nam_);     assert_shape_4d(Kcur, n_embd_head, n_head, N, n_batch);   
        ggml_build_forward_expand(gf_, Qcur);       ggml_build_forward_expand(gf_, Kcur);        ggml_build_forward_expand(gf_, Vcur);
        k = kv->SerialK(ctx,Kcur,il,true);          v = kv->SerialV(ctx,Vcur,il,true);          
        ggml_build_forward_expand(gf_,k);           ggml_build_forward_expand(gf_,v);
        /*if(use_cache){            //  llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);
            size_t nzK = n_ctx*n_batch*n_embd_k_gqa;
            struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv->k_l[il], nzK, ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa)*kv_head);
            sprintf(nam_,"k_cache_view-%d",layer_id);    set_name(k_cache_view, nam_);         
            ggml_build_forward_expand(gf_, ggml_cpy(ctx, Kcur, k_cache_view));        //
            // assert(Vcur->ne[0] == n_embd_v_gqa && Vcur->ne[1] == n_ctx);
            size_t nzV = n_ctx*n_batch*n_embd_v_gqa;
            // struct ggml_tensor * v_cache_view = ggml_view_1d(ctx, kv->v_l[il], nzV, ggml_row_size(kv->v_l[il]->type, n_embd_v_gqa)*kv_head);        
            struct ggml_tensor *v_cache_view = ggml_view_2d(ctx, kv->v_l[il], n_ctx*n_batch, n_embd_v_gqa,(  n_ctx)*ggml_element_size(kv->v_l[il]),(kv_head)*ggml_element_size(kv->v_l[il]));
            sprintf(nam_,"v_cache_view-%d",layer_id);    set_name(v_cache_view, nam_);         //cb(v_cache_view, "v_cache_view", il);
            Vcur = ggml_transpose(ctx, Vcur);
            ggml_build_forward_expand(gf_, ggml_cpy(ctx, Vcur, v_cache_view));      //copy Vcur to v_cache_view
        }else{
            ggml_build_forward_expand(gf_,Kcur);            ggml_build_forward_expand(gf_,Vcur);
        }*/
        
        q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);        
        sprintf(nam_,"q-%d",layer_id);    set_name(q, nam_);    
        k = kv->SerialK(ctx,Kcur,il,false);        
        if(isTrain && k->type!=GGML_TYPE_F32)
            k = ggml_cast(ctx, k, GGML_TYPE_F32);             
        /*if(use_cache){             
            k = ggml_view_3d(ctx, kv->k_l[il],n_embd_head_k, n_kv, n_head_kv,
                    ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa),ggml_row_size(kv->k_l[il]->type, n_embd_head_k),0);
            // assert(k->nb[0]==4 && k->nb[1]==4*k->ne[0]);
            
        }else{
            k = Kcur;
        }*/
        sprintf(nam_,"k-%d",layer_id);    set_name(k, nam_);
        if(!use_flash)  {
            hGensor kq = ggml_mul_mat(ctx, k, q);           //assert_shape_4d(kq, N, N, n_head, n_batch);   
            sprintf(nam_,"kq-%d",layer_id);    set_name(kq, nam_);
            kq = ggml_soft_max_ext(ctx, kq, KQ_mask, kq_scale, f_max_alibi_bias);
            sprintf(nam_,"kq_soft_max_ext-%d",layer_id);    set_name(kq, nam_);
            v = kv->SerialV(ctx,Vcur,il,false);    
            /*if(use_cache){ 
                v = ggml_view_3d(ctx, kv->v_l[il],n_kv, n_embd_head_v, n_head_kv,
                        ggml_element_size(kv->v_l[il])*kv_size,ggml_element_size(kv->v_l[il])*kv_size*n_embd_head_v,0);
            }else{
                v = Vcur;
            }*/
            // assert(v->nb[0]==4 && v->nb[1]==4*v->ne[0]);
            if(isTrain ){
                if( v->type!=GGML_TYPE_F32 )
                    v = ggml_cast(ctx, v, GGML_TYPE_F32);
                // v = ggml_repeat(ctx, v, kq);  
            }
            sprintf(nam_,"v-%d",layer_id);    set_name(v, nam_);    
            kqv_out = vXkq(ctx,v, kq,layer_id);   
            // sprintf(nam_,"v-%d",layer_id);    set_name(v, nam_);             
            // hGensor kqv = ggml_mul_mat(ctx, v, kq);
            // sprintf(nam_,"kqv-%d",layer_id);    set_name(kqv, nam_);
            // hGensor kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            // sprintf(nam_,"kqv_merged-%d",layer_id);    set_name(kqv_merged, nam_);
            // kqv_out = ggml_cont_2d(ctx, kqv_merged, n_embd, n_ctx*n_batch);
            // sprintf(nam_,"kqv_merged_cont-%d",layer_id);    set_name(kqv_out, nam_);     
   
        }else{      //  
            assert(use_flash);
            v = kv->SerialV(ctx,Vcur,il,false);
            if(isTrain && v->type!=GGML_TYPE_F32)
                v = ggml_cast(ctx, v, GGML_TYPE_F32);
            sprintf(nam_,"v-%d",layer_id);    set_name(v, nam_);
            kqv_out = ggml_flash_attn_ext(ctx, q, k, v, KQ_mask, kq_scale, f_max_alibi_bias,attn_soft_cap);
            // if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_GEMMA2) {
            //     ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
            // }   
            kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1] 
            sprintf(nam_,"kqv-%d",layer_id);            set_name(kqv_out, nam_); 
        }   
    }    
    ggml_build_forward_expand(gf_, kqv_out);
    assert(ggml_nelements(teb)==ggml_nelements(kqv_out));
    return kqv_out;
}



