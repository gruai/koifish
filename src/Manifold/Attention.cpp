#include "Fish.hpp"
#include "TGraph.hpp"

void inline set_name(struct ggml_tensor * t, const char * n) {
    ggml_set_name(t, n);
    if (t->grad) {
        ggml_format_name(t->grad, "%s->grad", n);
    }
};

/*
    Rotary Position Embedding
*/
hGensor BROWN_Motion::QKV_rope(struct ggml_context *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,int flag)   {
    /*  struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);
            }
    */
    hGensor  t05 = w==nullptr ? cur : ggml_mul_mat      (ctx, w, cur);                          
    //set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
    hGensor  t06 = ggml_reshape_4d   (ctx, t05,shape[0],shape[1],shape[2],shape[3]); //n_embd/n_head, n_head, N, n_batch
    //set_name(t06, "t06");     
    assert_shape_4d(t06, shape[0],shape[1],shape[2],shape[3]);
    //                                                  32      64          10000           1        
    const int rope_mode = 0;
    hGensor  t07 = ggml_rope_ext(
            ctx, t06, KQ_pos, nullptr, n_rot, rope_mode, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f
        );
    // CYS_0826 hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f);
    hGensor  t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);
    return t13;
}

hGensor BROWN_Motion::Build(struct ggml_context *ctx ,hGensor t04, hGensor KQ_pos,const KV_CACHE& kv)    {
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
   
    hGensor  t13 = nullptr, t14 = nullptr, t16=nullptr;
    
    assert(wq!=nullptr);
    auto shape=t04->ne;
    hGensor  tP = ggml_soft_max_inplace     (ctx, wq);  
    hGensor  t05 = ggml_mul_mat      (ctx, tP, t04);       //[4096,4096,1,1]x[4096,256,1,1] = [4096,256,1,1]                   
    set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
    hGensor  t06 = ggml_reshape_4d   (ctx, t05, n_embd/n_head, n_head, N, n_batch); 
    set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);      //[128,32,64,4]   
    assert(0);  // CYS_0826
    hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f); 
    set_name(t07, "t07");     assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
    t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);         //[128,64,32,4] 
    assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);
    if(0){
        hGensor t16_1 = ggml_reshape_4d   (ctx, t13, shape[0],shape[1],shape[2],shape[3]); //crash @ggml_is_contiguous
        hGensor t16_2 = ggml_mul_mat(ctx, wq, t16_1); 
        t16 = ggml_reshape_4d   (ctx, t16_2, n_embd/n_head, n_head, N, n_batch);   
        assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);  
    }else{
        t16 = t13; 
    }           
    return t16;   
}

/**
 *  llm_build_kqv   ???
 *  https://github.com/ggerganov/llama.cpp/pull/5021
*/
hGensor QKV_Motion::Build(struct ggml_context *ctx ,hGensor t04, hGensor KQ_pos,const KV_CACHE& kv)    {
    char nam_[128];
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1,layer_id=lay->id,il=layer_id,n_kv=kv.n;  
    int kv_head=0;  // index of where we store new KV data in the cache     
    
    hGensor  q=nullptr,k=nullptr,v=nullptr,kqv_out=nullptr; //t13 = nullptr, t14 = nullptr, t16=nullptr;
    assert(wk!=nullptr && wv!=nullptr) ;
    if(version == 0)   {
        q = QKV_rope(ctx,t04,wq,KQ_pos,{n_embd/n_head, n_head, N, n_batch});        
        set_name(q, "t13");     assert_shape_4d(q, n_embd/n_head, N, n_head, n_batch);
        k = QKV_rope(ctx ,t04,wk,KQ_pos,{n_embd/n_head, n_head_kv, N, n_batch});        
        set_name(k, "t14");     assert_shape_4d(k, n_embd/n_head, N, n_head_kv, n_batch);        
        hGensor  t11 = ggml_mul_mat      (ctx, t04, wv);        //[4096,256,1,1]x[4096,1024,1,1]= [256,1024,1,1]                   
        set_name(t11, "t11");     //assert_shape_2d(t11, N*n_batch, n_embd_gqa);    
        hGensor  t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd/n_head, n_head_kv);      // [64,4,128,8,] 
        set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head_kv);
        hGensor  t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                                // [64,128,8,4,] 
        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd/n_head, n_head_kv, n_batch);
        hGensor  t16_0 = ggml_mul_mat              (ctx, k, q);                 
                // set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
        hGensor  t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);          
                // set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
        hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);            
                // set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
        hGensor  t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                    
                // set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
        hGensor t16 = ggml_mul_mat(ctx, t15, t16_3);         assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);                                             
                // set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        hGensor  t17 = ggml_permute      (ctx, t16, 0, 2, 1, 3);    // [128,6,17,1]                      
        set_name(t17, "t17");     assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
        kqv_out = ggml_cont         (ctx, t17);                                    
        // set_name(t18, "t18");     assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
        kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1] 
    }else{
        float kq_scale = 1.0f/sqrtf(float(n_embd_head));        //0.0883883461
        hGensor  Qcur = ggml_mul_mat      (ctx, wq, t04);     
        sprintf(nam_,"Qcur-%d",layer_id);    set_name(Qcur, nam_);  
        hGensor  Kcur = ggml_mul_mat      (ctx, wk, t04);     
        sprintf(nam_,"Kcur-%d",layer_id);    set_name(Kcur, nam_); 
        hGensor  Vcur = ggml_mul_mat      (ctx, wv, t04);
        sprintf(nam_,"Vcur-%d",layer_id);    set_name(Vcur, nam_); 
        hGensor  rope_factors = lay->rope_(false);         
        Qcur = ggml_rope_ext(
                ctx, ggml_reshape_3d(ctx, Qcur, n_embd_head, n_head, n_tokens), KQ_pos, rope_factors,
                n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,ext_factor, attn_factor, beta_fast, beta_slow
            );
        sprintf(nam_,"Qcur_rope-%d",layer_id);    set_name(Qcur, nam_);     assert_shape_4d(Qcur, n_embd_head, n_head, N, n_batch);
        Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, n_embd_head, n_head_kv, n_tokens), KQ_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, rope_freq_base, rope_freq_scale,ext_factor, attn_factor, beta_fast, beta_slow
                );
        sprintf(nam_,"Kcur_rope-%d",layer_id);    set_name(Kcur, nam_);     assert_shape_4d(Kcur, n_embd_head, n_head, N, n_batch);        
        //  llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);
        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, kv.k_l[il], n_tokens*n_embd_k_gqa, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa)*kv_head);
        sprintf(nam_,"k_cache_view-%d",layer_id);    set_name(k_cache_view, nam_);         
        // ggml_build_forward_expand(graph, ggml_cpy(ctx, k_cur, k_cache_view));        assert(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);
        struct ggml_tensor * v_cache_view = ggml_view_1d(ctx, kv.v_l[il], n_tokens*n_embd_v_gqa, ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa)*kv_head);        
        sprintf(nam_,"v_cache_view-%d",layer_id);    set_name(v_cache_view, nam_);         //cb(v_cache_view, "v_cache_view", il);
        // ggml_build_forward_expand(graph, ggml_cpy(ctx, v_cur, v_cache_view));

        q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);        
        sprintf(nam_,"q-%d",layer_id);    set_name(q, nam_);               
        k = ggml_view_3d(ctx, kv.k_l[il],n_embd_head_k, n_kv, n_head_kv,
                ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),ggml_row_size(kv.k_l[il]->type, n_embd_head_k),0);
        sprintf(nam_,"k-%d",layer_id);    set_name(k, nam_);
        if(!use_flash)  {
            hGensor kq = ggml_mul_mat(ctx, k, q);
            sprintf(nam_,"kq-%d",layer_id);    set_name(kq, nam_);
            kq = ggml_soft_max_ext(ctx, kq, inp_KQ_mask, kq_scale, f_max_alibi_bias);
            sprintf(nam_,"kq_soft_max_ext-%d",layer_id);    set_name(kq, nam_);
            v = ggml_view_3d(ctx, kv.v_l[il],n_kv, n_embd_head_v, n_head_kv,
                    ggml_element_size(kv.v_l[il])*n_ctx,ggml_element_size(kv.v_l[il])*n_ctx*n_embd_head_v,0);
            sprintf(nam_,"v-%d",layer_id);    set_name(v, nam_);             
            hGensor kqv = ggml_mul_mat(ctx, v, kq);
            sprintf(nam_,"kqv-%d",layer_id);    set_name(kqv, nam_);
            hGensor kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            sprintf(nam_,"kqv_merged-%d",layer_id);    set_name(kqv_merged, nam_);
            kqv_out = ggml_cont_2d(ctx, kqv_merged, n_embd, n_tokens);
            sprintf(nam_,"kqv_merged_cont-%d",layer_id);    set_name(kqv_out, nam_);        
        }else{      //  
            v = ggml_view_3d(ctx, kv.v_l[il],
                        n_embd_head_v, n_kv, n_head_kv,
                        ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),ggml_row_size(kv.v_l[il]->type, n_embd_head_v),0);
            sprintf(nam_,"v-%d",layer_id);    set_name(v, nam_);
            kqv_out = ggml_flash_attn_ext(ctx, q, k, v, inp_KQ_mask, kq_scale, f_max_alibi_bias,attn_soft_cap);
            // if (model.arch == LLM_ARCH_PHI2 || model.arch == LLM_ARCH_PHI3 || model.arch == LLM_ARCH_GPTNEOX || model.arch == LLM_ARCH_GEMMA2) {
            //     ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
            // }   
            kqv_out = ggml_reshape_2d   (ctx, kqv_out, n_embd, N*n_batch);   // [768,17,1] 
        }   
    }    
    sprintf(nam_,"kqv-%d",layer_id);            set_name(kqv_out, nam_);     
    assert(ggml_nelements(t04)==ggml_nelements(kqv_out));
    return kqv_out;
}



/*
    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(lctx.sched, cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (batch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_buft_supports_backend(lctx.model.buft_layer[il].buft, backend)) {
                        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                        break;
                    }
                }
            }
        }
    };
*/