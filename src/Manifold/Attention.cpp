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
    hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    hGensor  t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);
    return t13;
}

/*
    hGensor  t05 = ggml_mul_mat      (ctx_compute, wq, t04);                          set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
    hGensor  t06 = ggml_reshape_4d   (ctx_compute, t05, n_embd/n_head, n_head, N, n_batch); set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);
    hGensor  t07 = ggml_rope_custom(ctx_compute,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f); //rope              (t06);                                         
            set_name(t07, "t07");     assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
    hGensor  t08 = ggml_mul_mat      (ctx_compute, wk, t04);                          
            set_name(t08, "t08");     assert_shape_2d(t08, n_embd_gqa, N*n_batch);
    hGensor  t09 = ggml_reshape_4d   (ctx_compute, t08, n_embd/n_head, n_head_kv, N, n_batch); set_name(t09, "t09");     
        assert_shape_4d(t09, n_embd/n_head, n_head_kv, N, n_batch);
    hGensor  t10 = ggml_rope_custom(ctx_compute,t09, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);  //rope              (t09);                                         
            set_name(t10, "t10");     assert_shape_4d(t10, n_embd/n_head, n_head_kv, N, n_batch);
    hGensor  t11 = ggml_mul_mat      (ctx_compute, t04, wv);                          set_name(t11, "t11");     assert_shape_2d(t11, N*n_batch, n_embd_gqa);    
    hGensor  t12 = ggml_reshape_4d   (ctx_compute, t11, N, n_batch, n_embd/n_head, n_head_kv); set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head_kv);
    hGensor  t13 = ggml_permute      (ctx_compute, t07, 0, 2, 1, 3);                        set_name(t13, "t13");     assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);
    hGensor  t14 = ggml_permute      (ctx_compute, t10, 0, 2, 1, 3);                        set_name(t14, "t14");     assert_shape_4d(t14, n_embd/n_head, N, n_head_kv, n_batch);
    hGensor  t15 = ggml_permute      (ctx_compute, t12, 0, 3, 1, 2);                        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd/n_head, n_head_kv, n_batch);
    hGensor  t16;
    if (train_params.use_flash) {
        t16 = ggml_flash_attn(ctx_compute, t13, t14, t15, true);                                        set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
    } else {
        hGensor  t16_0 = ggml_mul_mat              (ctx_compute, t14, t13);                 set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
        hGensor  t16_1 = ggml_scale_inplace        (ctx_compute, t16_0, kv_scale);          set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
        hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx_compute, t16_1, n_past);            set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
        hGensor  t16_3 = ggml_soft_max_inplace     (ctx_compute, t16_2);                    set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
        t16 = ggml_mul_mat(ctx_compute, t15, t16_3);                                                    set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
    }
*/
hGensor BROWN_Motion::Build(struct ggml_context *ctx ,hGensor t04, hGensor KQ_pos, bool use_flash)    {
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
    use_flash = true;      //only for debug
    hGensor  t13 = nullptr, t14 = nullptr, t16=nullptr;
    
    assert(wq!=nullptr);
    auto shape=t04->ne;
    hGensor  tP = ggml_soft_max_inplace     (ctx, wq);  
    hGensor  t05 = ggml_mul_mat      (ctx, tP, t04);       //[4096,4096,1,1]x[4096,256,1,1] = [4096,256,1,1]                   
    set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
    hGensor  t06 = ggml_reshape_4d   (ctx, t05, n_embd/n_head, n_head, N, n_batch); 
    set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);      //[128,32,64,4]   
    hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f); 
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
hGensor QKV_Motion::Build(struct ggml_context *ctx ,hGensor t04, hGensor KQ_pos, bool use_flash)    {
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1;       
    use_flash = true;      //only for debug
    hGensor  t13 = nullptr, t14 = nullptr, t16=nullptr;
    assert(wk!=nullptr && wv!=nullptr) ;
    {
        if(rope==1)
           t13 = QKV_rope(ctx,t04,wq,KQ_pos,{n_embd/n_head, n_head, N, n_batch});
        else{
            hGensor  t05 = ggml_mul_mat      (ctx, wq, t04);       //[4096,4096,1,1]x[4096,256,1,1] = [4096,256,1,1]                   
            set_name(t05, "t05");     assert_shape_2d(t05, n_embd, N*n_batch);
            hGensor  t06 = ggml_reshape_4d   (ctx, t05, n_embd/n_head, n_head, N, n_batch); 
            set_name(t06, "t06");     assert_shape_4d(t06, n_embd/n_head, n_head, N, n_batch);      //[128,32,64,4]   
            hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f); 
            set_name(t07, "t07");     assert_shape_4d(t07, n_embd/n_head, n_head, N, n_batch);
            t13 = ggml_permute      (ctx, t07, 0, 2, 1, 3);         //[128,64,32,4] 
        }
        set_name(t13, "t13");     assert_shape_4d(t13, n_embd/n_head, N, n_head, n_batch);

        if(rope==1)
            t14 = QKV_rope(ctx ,t04,wk,KQ_pos,{n_embd/n_head, n_head_kv, N, n_batch});
        else{
            hGensor  t08 = ggml_mul_mat      (ctx, wk, t04);      //[4096,1024,1,1]x[4096,256,1,1] = [1024,256,1,1]                         
            set_name(t08, "t08");     //assert_shape_2d(t08, n_embd_gqa, N*n_batch);
            hGensor  t09 = ggml_reshape_4d   (ctx, t08, n_embd/n_head, n_head_kv, N, n_batch);      // [128,8,64,4]    
            set_name(t09, "t09");                 assert_shape_4d(t09, n_embd/n_head, n_head_kv, N, n_batch);
            hGensor  t10 = ggml_rope_custom(ctx,t09, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);  //rope              (t09);                                         
            set_name(t10, "t10");     assert_shape_4d(t10, n_embd/n_head, n_head_kv, N, n_batch);
            t14 = ggml_permute      (ctx, t10, 0, 2, 1, 3);         // [128,64,8,4]  
        }
        set_name(t14, "t14");     assert_shape_4d(t14, n_embd/n_head, N, n_head_kv, n_batch);
        
        hGensor  t11 = ggml_mul_mat      (ctx, t04, wv);        //[4096,256,1,1]x[4096,1024,1,1]= [256,1024,1,1]                   
        set_name(t11, "t11");     //assert_shape_2d(t11, N*n_batch, n_embd_gqa);    
        hGensor  t12 = ggml_reshape_4d   (ctx, t11, N, n_batch, n_embd/n_head, n_head_kv);      // [64,4,128,8,] 
        set_name(t12, "t12");     assert_shape_4d(t12, N, n_batch, n_embd/n_head, n_head_kv);
        hGensor  t15 = ggml_permute      (ctx, t12, 0, 3, 1, 2);                                // [64,128,8,4,] 
        set_name(t15, "t15");     assert_shape_4d(t15, N, n_embd/n_head, n_head_kv, n_batch);
        
        if (use_flash) {
            t16 = ggml_flash_attn(ctx, t13, t14, t15, true);
        } else {    //would crash 
            hGensor  t16_0 = ggml_mul_mat              (ctx, t14, t13);                 
                    // set_name(t16_0, "t16_0"); assert_shape_4d(t16_0, N, N, n_head, n_batch);
            hGensor  t16_1 = ggml_scale_inplace        (ctx, t16_0, kv_scale);          
                    // set_name(t16_1, "t16_1"); assert_shape_4d(t16_1, N, N, n_head, n_batch);
            hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx, t16_1, n_past);            
                    // set_name(t16_2, "t16_2"); assert_shape_4d(t16_2, N, N, n_head, n_batch);
            hGensor  t16_3 = ggml_soft_max_inplace     (ctx, t16_2);                    
                    // set_name(t16_3, "t16_3"); assert_shape_4d(t16_3, N, N, n_head, n_batch);
            t16 = ggml_mul_mat(ctx, t15, t16_3);                                                    
                    // set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);
        }
    }
    set_name(t16, "t16");     assert_shape_4d(t16, n_embd/n_head, N, n_head, n_batch);  
    assert(ggml_nelements(t04)==ggml_nelements(t16));
    return t16;
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