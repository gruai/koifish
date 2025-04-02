/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Sparse KVCache
 *  \author Yingshi Chen
 */

#include "Cache.hpp"
#include "../Manifold/gLLM.hpp"

KVCache::KVCache(Fish *hF,int max_batch_size, int max_seq_len, int flag) : _fish(hF){   
    // init_lamakv();
    auto modep = _fish->config.model;
    max_seq_len = std::max(max_seq_len,modep.seq_len);
    int kv_dim = modep.head_dim * modep.n_kv_heads;
    //  GTensor::scratch = std::make_shared<huTensor>("scratch_output",sp0,GTensor::tpFloatX,false);
    SHAPE sp={modep.n_layers,max_seq_len,kv_dim};
    key = std::make_shared<huTensor>("KCache",sp,tpCache,true); 
    value = std::make_shared<huTensor>("VCache",sp,tpCache,true); 
    _INFO("[KVCache] shape=%s\n",CSTR(sp));
    // raw_key = key->data;        raw_val = value->data;
}

void *KVCache::Get(KVCache::CTYPE typ, int flag){
    switch(typ){
    case KV_KEY:
        assert(key!=nullptr);
        return key->data;
    case KV_VAL:
        assert(key!=nullptr);
        return value->data;
    default:
        assert(0);
    }
    return nullptr;
}

int KVCache::n_kv(){
    return kv_n;
}

hGensor KVCache::SerialV(void *ctx,hGensor Vcur,int il,bool isSave){
    hGensor v = nullptr;
    /*llama_kv_cache *kv = (llama_kv_cache *)lamakv;
    if(kv==nullptr)
        return Vcur;

    auto& config = lam_->config;
    int n_ctx = config.n_ctx(),n_batch = config.n_batch(),n_embd_v_gqa = config.n_embd_v_gqa(il),n_embd_head_v = config.n_embd_head_v,n_head_kv=config.n_head_kv(il);    
    int kv_head=0,kv_size = kv->size;   
    
    if(isSave){
        char nam_[128];
        size_t nzV = n_ctx*n_batch*n_embd_v_gqa;
            // hGensor  v_cache_view = ggml_view_1d(ctx, kv->v_l[il], nzV, ggml_row_size(kv->v_l[il]->type, n_embd_v_gqa)*kv_head);        
            hGensor v_cache_view = ggml_view_2d(ctx, kv->v_l[il], n_ctx*n_batch, n_embd_v_gqa,(n_ctx)*ggml_element_size(kv->v_l[il]),(kv_head)*ggml_element_size(kv->v_l[il]));
            sprintf(nam_,"v_cache_view-%d",il);    gTN(v_cache_view, nam_);         //cb(v_cache_view, "v_cache_view", il);
            Vcur = ggml_transpose(ctx, Vcur);
            ggml_cpy(ctx, Vcur, v_cache_view);          v = v_cache_view;            
    }else{
        v = ggml_view_3d(ctx, kv->v_l[il],kv_n, n_embd_head_v, n_head_kv,
                            ggml_element_size(kv->v_l[il])*kv_size,ggml_element_size(kv->v_l[il])*kv_size*n_embd_head_v,0);        
    }*/

    return v;
}

hGensor KVCache::SerialK(void *ctx,hGensor Kcur,int il,bool isSave){
    hGensor k = nullptr;
    /*llama_kv_cache *kv = (llama_kv_cache *)lamakv;
    if(kv==nullptr)
        return Kcur;

    const auto& config = lam_->config;
    int n_ctx = config.n_ctx(),n_batch = config.n_batch(),n_embd_k_gqa  = config.n_embd_k_gqa(il),n_embd_head_k = config.n_embd_head_k,n_head_kv=config.n_head_kv(il);    
    int kv_head=0,kv_size = kv->size;   
    
    if(isSave){
        char nam_[128];
        size_t nzK = n_ctx*n_batch*n_embd_k_gqa;
            hGensor  k_cache_view = ggml_view_1d(ctx, kv->k_l[il], nzK, ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa)*kv_head);
            sprintf(nam_,"k_cache_view-%d",il);    gTN(k_cache_view, nam_);   
        ggml_cpy(ctx, Kcur, k_cache_view);          k = k_cache_view;     
            
    }else{
        k = ggml_view_3d(ctx, kv->k_l[il],n_embd_head_k, kv_n, n_head_kv,
                    ggml_row_size(kv->k_l[il]->type, n_embd_k_gqa),ggml_row_size(kv->k_l[il]->type, n_embd_head_k),0);        
    }*/

    return k;
}

void KVCache::init_lamakv(int n_batch) {
#ifdef __USE_GGML__
    struct LAMA *lama  = la_->lama();  
    if(lama==nullptr){

    }else{
        /*llama_kv_cache *kv = lama->_cache;  
        bool isFlash = false;
        lamakv = kv; 
        const uint32_t pad = isFlash ? 256u : 32u,cell_max=0;     //llama_kv_cache_cell_max(*cache)
        kv_n = std::min(kv->size, std::max(pad, GGML_PAD(cell_max, pad)));      */  
    }

    const auto& config = lam_->config;

    const uint32_t n_ctx   = config.n_ctx();
    const uint32_t n_embd  = config.nEmbed();
    const uint32_t n_layer = config.n_layer_train;

    const int64_t n_mem      = n_layer*n_ctx*n_batch;
    const int64_t n_elements = n_embd*n_mem;

    // cache.buf.resize(2u*n_elements*BPE(wtype) + 2u*MB);

    // struct ggml_init_params params;
    // params.mem_size   = cache.buf.size;
    // params.mem_buffer = cache.buf.addr;
    // params.no_alloc   = false;
    //  llama_kv_cache *cache = (llama_kv_cache *)lamakv;
    /*if (!cache->ctx) {
        struct ggml_init_params params;
        params.mem_size   = 2u*n_elements*BPE(typNUMBER::F32) + 2u*1024*1024;
        params.mem_buffer = NULL;
        params.no_alloc   = false;

        cache->ctx = ggml_init(params);

        if (!cache->ctx) {
            fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
            exit(1);
        }
    }

    cache->k = TENSO(cache->ctx, typNUMBER::F32, n_elements);
    cache->v = TENSO(cache->ctx, typNUMBER::F32, n_elements);*/
#endif
}
