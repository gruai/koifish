/*
#include "llama_cys.inc"

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin/)

struct ggml_object * ggml_new_object(struct ggml_context * ctx, enum ggml_object_type type, size_t size)
size_t ggml_graph_nbytes(size_t size, bool grads)
*/

/*
struct ggml_cgraph * build_mamba() {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, llama_model_max_nodes(model), false);
        bool isV0 = false;
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        // {n_embd, n_tokens}
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

        struct ggml_tensor *state_copy = nullptr, *state_mask = nullptr;
        if(isV0){
            state_copy = build_inp_s_copy();
            state_mask = build_inp_s_mask();            
        }

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);
            if(!isV0){
                cur = mamba_build_layer(ctx0,lctx,gf,cur,inpL,il,n_layer,n_tokens,kv_head,n_kv,n_outputs);
                // cur = layer_cys(     ctx0,lctx,gf,cur,inpL,il,n_layer,n_tokens,kv_head,n_kv,n_outputs);
            }else{
                cur = llm_build_mamba(ctx0, lctx, batch, gf, cur,state_copy, state_mask,kv_head, n_kv, cb, il);
                if (il == n_layer - 1) {
                    // skip computing output for unused tokens
                    struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                    cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
                    inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                }
                // residual
                cur = ggml_add(ctx0, cur, inpL);
                cur = lctx.cvec.apply_to(ctx0, cur, il);
                cb(cur, "l_out", il);                          
            }            
            inpL = cur;      // input for next layer
        }
        if(isV0)   {        // final rmsnorm
            cur = llm_build_norm(ctx0, inpL, hparams,
                    model.output_norm, NULL,LLM_NORM_RMS, cb, -1);
            cb(cur, "result_norm", -1);
            // lm_head
            cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
            cb(cur, "result_output", -1);
        }

        ggml_build_forward_expand(gf, cur);

        return gf;
    }
*/

#include "../../src/GG_params.hpp"
bool llama_get_params(struct llama_model * lmodel,struct llama_hparams& params){
    if(lmodel==nullptr)
        return false;
    params = lmodel->hparams;
    return true;
}

bool llama2params(struct llama_model * lmodel,struct CLI_params& cparam){
    if(lmodel==nullptr)
        return false;
    /*
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
*/
    cparam.n_embd = lmodel->hparams.n_embd;  
    // cparam.n_ctx = lmodel->hparams.n_ctx;     
    cparam.n_ff = lmodel->hparams.n_ff();  
    cparam.n_layer = lmodel->hparams.n_layer;
    cparam.n_head = lmodel->hparams.n_head();
    cparam.n_head_kv = lmodel->hparams.n_head_kv();    
    // cparam.n_vocab = lmodel->hparams.n_vocab;
    cparam.rope_type = lmodel->hparams.rope_type;

    cparam.f_norm_rms_eps = lmodel->hparams.f_norm_rms_eps;  
    cparam.rope_freq_base = lmodel->hparams.rope_freq_base_train; 
    cparam.rope_freq_scale = lmodel->hparams.rope_freq_scale_train; 
    // ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    // ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);
    cparam.n_expert = lmodel->hparams.n_expert;
    cparam.n_expert_used = lmodel->hparams.n_expert_used;

    return true;
}

// hack to get internal data
bool llama_ctx_get_( struct llama_context * ctx,void **hData, int type){
    llama_synchronize(ctx);
    switch(type){
        case 10:
            *hData = ctx->logits;
            break;
        default:
            assert(0);
    };
    return true;
}

bool llama_ctx_set_( struct llama_context * ctx,void *hData, int type){
    assert(ctx->sched!=nullptr);
    llama_synchronize(ctx);
    switch(type){
        case 11:
            ctx->logits_all = *((bool*)(hData));
            break;
        case 42:      
            assert(0);  // CYS_0826      
            // ctx->rng                 = std::mt19937(42);
            break;
        default:
            assert(0);
    };
    return true;
}

struct llama_context *llama_ctx_reset_(struct llama_context * ctx_,struct llama_model * model,struct llama_context_params   params) {    
    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    llama_context * ctx = ctx_;
    // ctx = new llama_context(*model);
    const auto & hparams = model->hparams;
    auto       & cparams = ctx->cparams;   
    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa() + hparams.n_embd_k_s();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa() + hparams.n_embd_v_s();
    const int64_t  n_layer      = hparams.n_layer;

    uint32_t kv_size = cparams.n_ctx;
    ggml_type type_k = params.type_k;
    ggml_type type_v = params.type_v;
    // Mamba only needs a constant number of KV cache cells per sequence
    if (model->arch == LLM_ARCH_MAMBA) {
        // Mamba needs at least as many KV cells as there are sequences kept at any time
        kv_size = std::max((uint32_t) 1, params.n_seq_max);
        // it's probably best to keep as much precision as possible for the states
        type_k = GGML_TYPE_F32; // required by ggml_ssm_conv for Mamba's conv_states
        type_v = GGML_TYPE_F32; // required by ggml_ssm_scan for Mamba's ssm_states
    }

    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);
    // ctx->backends.clear();
    if (!hparams.vocab_only) {
        // if (!llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv)) {
        //     LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
        //     llama_free(ctx);
        //     return nullptr;
        // }
        auto& cache = ctx->kv_self;
        cache.has_shift = false;

        // TODO: find a nicer way to add other recurrent model architectures
        cache.recurrent = model->arch == LLM_ARCH_MAMBA;
        cache.v_trans   = !cparams.flash_attn;

        // TODO: support mixed recurrent Transformer architectures
        // NOTE: (!a || b) is a logical implication (a -> b)
        GGML_ASSERT(!cache.recurrent || n_embd_k_gqa == hparams.n_embd_k_s());
        GGML_ASSERT(!cache.recurrent || n_embd_v_gqa == hparams.n_embd_v_s());
        GGML_ASSERT( cache.recurrent || n_embd_k_gqa == hparams.n_embd_k_gqa());
        GGML_ASSERT( cache.recurrent || n_embd_v_gqa == hparams.n_embd_v_gqa());

        cache.head = 0;        
        cache.used = 0;

        cache.size = kv_size;
        cache.type_k = type_k;
        cache.type_v = type_v;

        cache.cells.clear();
        cache.cells.resize(kv_size);

        if (cache.recurrent) {
            // init state copy sources
            for (uint32_t i = 0; i < cache.size; ++i) {
                cache.cells[i].src = i;
            }
        }
        
        if(cache.ctxs.size()>0){
            // assert(buft_layer_count.size()==cache.ctxs.size());
            // int i=0;
            // for (auto & it : buft_layer_count){
            //     ctx_map[it.first] = cache.ctxs[i++];
            // }            
        }else{       // create a context for each buffer type
            std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
            bool offload = cparams.offload_kqv;
        #ifdef GGML_USE_CLBLAST
            offload = false;
        #endif

            // count used buffer types
            std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
            if (offload) {
                for (int64_t i = 0; i < n_layer; ++i) {
                    buft_layer_count[model->buft_layer[i].buft]++;
                }
            } else {
                buft_layer_count[llama_default_buffer_type_cpu(true)] = n_layer;
            }
            for (auto & it : buft_layer_count) {
                int n_layers = it.second;
                struct ggml_init_params params = {
                    /*.mem_size   =*/ 2u*n_layers*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context * ctx_buf = ggml_init(params);
                if (!ctx_buf) {
                    LLAMA_LOG_ERROR("%s: failed to allocate context for kv cache\n", __func__);
                    return nullptr;
                }
                ctx_map[it.first] = ctx_buf;
                cache.ctxs.push_back(ctx_buf);
            }
            
            cache.k_l.reserve(n_layer);
            cache.v_l.reserve(n_layer);

            for (int i = 0; i < (int) n_layer; i++) {
                struct ggml_context * ctx = cparams.offload_kqv ? ctx_map.at(model->buft_layer[i].buft) : cache.ctxs.front();
                ggml_tensor * k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);
                ggml_tensor * v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
                ggml_format_name(k, "cache_k_l%d", i);
                ggml_format_name(v, "cache_v_l%d", i);
                cache.k_l.push_back(k);
                cache.v_l.push_back(v);
            }        
            // allocate tensors and initialize the buffers to avoid NaNs in the padding
            for (auto it : ctx_map) {
                ggml_backend_buffer_type_t buft = it.first;
                ggml_context * ctx = it.second;
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                if (!buf) {
                    LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
                    return nullptr;
                }
                ggml_backend_buffer_clear(buf, 0);
                LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
                cache.bufs.push_back(buf);
            }   
        }
    }

    return ctx;
}

void GG_set_(struct ggml_tensor * cur, const char * name,int il) {
    // ggml_set_name(t, n);
    // if (t->grad) {
    //     ggml_format_name(t->grad, "%s->grad", n);
    // }
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
};
struct ggml_tensor * _repeat(struct ggml_context * ctx,struct ggml_tensor * a,struct ggml_tensor * b) {
    // return a;
    return ggml_repeat(ctx,a,b);
}

struct ggml_tensor * _v3d_cont(struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int64_t               ne0,
            int64_t               ne1,
            int64_t               ne2,
            size_t                nb1, // row   stride in bytes
            size_t                nb2, // slice stride in bytes
            size_t                offset) {
    struct ggml_tensor  * b = ggml_view_3d(ctx,a,ne0,ne1,ne2,nb1,nb2,offset);
    // b = ggml_cont(ctx,b);        // returning a contiguous tensor
    return b;
}

struct ggml_tensor * mamba_build_layer(
        struct ggml_context * ctx,struct llama_context & lctx,  //  const llama_ubatch & batch,
        struct ggml_cgraph * graph,struct ggml_tensor * curT,struct ggml_tensor * inpL0,int il,int n_layer, int n_tokens,int32_t kv_head,int32_t n_kv,int n_outputs) {
    auto& kv_self = lctx.kv_self;   //[4096,512]
    if(kv_head==-1){
        bool worst_case = true;
        n_kv = (worst_case ? kv_self.size : kv_self.n);
        kv_head = (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head);
        n_outputs = (worst_case ? n_tokens : lctx.n_outputs);        
    }

    struct ggml_tensor *state_copy=lctx.inp_s_copy, *state_mask=lctx.inp_s_mask;
    if( state_copy==nullptr )   {  //  build_inp_s_copy();
        // if(n_tokens>1) LLAMA_LOG_INFO("%s: @state_copy n_tokens=%d n_layer=%d...\n", __func__,n_tokens,n_layer );
        lctx.inp_s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
        GG_set_(lctx.inp_s_copy, "inp_s_copy",-1);        // cb(lctx.inp_s_copy, "inp_s_copy", -1);
        ggml_set_input(lctx.inp_s_copy);
        state_copy = lctx.inp_s_copy;
    }
    if( state_mask==nullptr )   {   //build_inp_s_mask();
        // if(n_tokens>1) LLAMA_LOG_INFO("%s: @state_mask n_tokens=%d n_layer=%d ...\n", __func__,n_tokens,n_layer );
        lctx.inp_s_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_kv);
        GG_set_(lctx.inp_s_mask, "inp_s_mask", -1);  // cb(lctx.inp_s_mask, "inp_s_mask", -1);
        ggml_set_input(lctx.inp_s_mask);
        state_mask = lctx.inp_s_mask;
    }
        
    const llama_model    & model   = lctx.model;
    const llama_hparams  & hparams = model.hparams;
    const llama_kv_cache & kv      = lctx.kv_self;
    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs  = 1;  //batch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;
    // Use the same RMS norm as the final layer norm
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int64_t n_seq_tokens = n_tokens;   //batch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    // GGML_ASSERT(batch.equal_seqs);
    // GGML_ASSERT(batch.n_tokens == n_seq_tokens * n_seqs);

    struct ggml_tensor * conv_states_all = kv.k_l[il];
    struct ggml_tensor * ssm_states_all  = kv.v_l[il];

    // (ab)using the KV cache to store the states
    struct ggml_tensor * conv = llm_build_copy_mask_state(ctx,
            graph, conv_states_all, state_copy, state_mask,
            hparams.n_embd_k_s(), kv.size, kv_head, n_kv, n_seqs);
    conv = ggml_reshape_3d(ctx, conv, d_conv - 1, d_inner, n_seqs);
    struct ggml_tensor * ssm = llm_build_copy_mask_state(ctx,
            graph, ssm_states_all, state_copy, state_mask,
            hparams.n_embd_v_s(), kv.size, kv_head, n_kv, n_seqs);
    ssm = ggml_reshape_3d(ctx, ssm, d_state, d_inner, n_seqs);

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    struct ggml_tensor *cur = ggml_reshape_3d(ctx, curT, curT->ne[0], n_seq_tokens, n_seqs);

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * xz = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor * x = _v3d_cont(ctx, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], 0);
    struct ggml_tensor * z = _v3d_cont(ctx, xz, d_inner, xz->ne[1], xz->ne[2], xz->nb[1], xz->nb[2], d_inner*ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        struct ggml_tensor * conv_x = ggml_concat(ctx, conv, ggml_transpose(ctx, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        struct ggml_tensor * last_conv = _v3d_cont(ctx, conv_x, d_conv - 1, d_inner, n_seqs, conv_x->nb[1], conv_x->nb[2], n_seq_tokens*(conv_x->nb[0]));

        ggml_build_forward_expand(graph,
            ggml_cpy(ctx, last_conv,
                ggml_view_1d(ctx, conv_states_all,
                    (d_conv - 1)*(d_inner)*(n_seqs),
                    kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(conv_states_all))));

        x = ggml_ssm_conv(ctx, conv_x, model.layers[il].ssm_conv1d);     
        x = ggml_add(ctx, x, _repeat(ctx, model.layers[il].ssm_conv1d_b,x));        // bias
        x = ggml_silu(ctx, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        struct ggml_tensor * x_db = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_x, x);
        // split
        struct ggml_tensor * dt = _v3d_cont(ctx, x_db, dt_rank, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], 0);
        struct ggml_tensor * B  = _v3d_cont(ctx, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*dt_rank);
        struct ggml_tensor * C  = _v3d_cont(ctx, x_db, d_state, n_seq_tokens, n_seqs, x_db->nb[1], x_db->nb[2], ggml_element_size(x_db)*(dt_rank+d_state));

        // Some Mamba variants (e.g. FalconMamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms) {
            // dt = ggml_rms_norm(ctx, dt, norm_rms_eps);
            // B = ggml_rms_norm(ctx, B, norm_rms_eps);
            // C = ggml_rms_norm(ctx, C, norm_rms_eps);
            //  https://github.com/ggerganov/llama.cpp/commit/fae826fb56b6f40e73fb4721e11adc1cf2795431
            dt = ggml_rms_norm(ctx, ggml_cont(ctx, dt), norm_rms_eps);
            B = ggml_rms_norm(ctx, ggml_cont(ctx, B), norm_rms_eps);
            C = ggml_rms_norm(ctx, ggml_cont(ctx, C), norm_rms_eps);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx, dt, _repeat(ctx,model.layers[il].ssm_dt_b,dt));

        // Custom operator to optimize the parallel associative scan
        // as described in the Annex D of the Mamba paper.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        struct ggml_tensor * y_ssm = ggml_ssm_scan(ctx, ssm, x, dt, model.layers[il].ssm_a, B, C);

        // store last states
        ggml_build_forward_expand(graph,
            ggml_cpy(ctx,
                ggml_view_1d(ctx, y_ssm, d_state*d_inner*n_seqs, x->nb[3]),
                ggml_view_1d(ctx, ssm_states_all, d_state*d_inner*n_seqs, kv_head*d_state*d_inner*ggml_element_size(ssm_states_all))));

        struct ggml_tensor * y = _v3d_cont(ctx, y_ssm, d_inner, n_seq_tokens, n_seqs, x->nb[1], x->nb[2], 0);

        // TODO: skip computing output earlier for unused tokens

        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        y = ggml_add(ctx, y, ggml_mul(ctx, x, _repeat(ctx,model.layers[il].ssm_d,x)));
        y = ggml_mul(ctx, y, ggml_silu(ctx, ggml_cont(ctx, z)));

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape_2d(ctx, cur, cur->ne[0], n_seq_tokens * n_seqs);
    GG_set_(cur, "mamba_out", il);   // cb(cur, "mamba_out", il);

    if (il == n_layer - 1) {        // skip computing output for unused tokens
        // struct ggml_tensor * inp_out_ids = build_inp_out_ids();
        lctx.inp_out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_outputs);
        GG_set_(lctx.inp_out_ids, "inp_out_ids", -1);    // cb(lctx.inp_out_ids, "inp_out_ids", -1);
        ggml_set_input(lctx.inp_out_ids);
        struct ggml_tensor * inp_out_ids = lctx.inp_out_ids;
        cur  = ggml_get_rows(ctx,  cur, inp_out_ids);
        inpL0 = ggml_get_rows(ctx, inpL0, inp_out_ids);
    }

    // residual
    cur = ggml_add(ctx, cur, inpL0);
    cur = lctx.cvec.apply_to(ctx, cur, il);
    GG_set_(cur, "l_out", il);       // cb(cur, "l_out", il);

    // final rmsnorm
    if (il == n_layer - 1) { 
        cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);             
        struct ggml_tensor *t11 = _repeat(ctx, model.output_norm, cur);   
        cur = ggml_mul(ctx, cur, t11);                     
        GG_set_(cur, "result_norm", -1);    // cb(cur, "result_norm", -1);
        // lm_head
        cur = llm_build_lora_mm(lctx, ctx, model.output, cur);
        GG_set_(cur, "result_output", -1);    // cb(cur, "result_output", -1);
    }
    
    return cur;
}

struct ggml_tensor * moe_build_ffn(struct ggml_context * ctx0,struct llama_context & lctx,
         struct ggml_tensor * cur,struct ggml_tensor * gate_inp,struct ggml_tensor * up_exps,struct ggml_tensor * gate_exps,struct ggml_tensor * down_exps,
                    int64_t   n_expert,int64_t   n_expert_used,bool   norm_w,bool   scale_w,float   w_scale,int   il){
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
        /*const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (batch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_buft_supports_backend(lctx.model.buft_layer[il].buft, backend)) {
                        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, backend);
                        break;
                    }
                }
            }
        }*/
    };
    // ggml_tensor *ffn = llm_build_moe_ffn(ctx0, lctx, cur,gate_inp,up_exps,gate_exps,down_exps,n_expert, n_expert_used,LLM_FFN_SILU, true,false, 0.0,cb, il);   
    auto ctx = ctx0;
    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor * probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
    cb(probs, "ffn_moe_probs", il);
    
    ggml_tensor * selected_experts = ggml_top_k(ctx, probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor * weights = n_expert_used==n_expert ? ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens) :
        ggml_get_rows(ctx,ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]    
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);
        GGML_ASSERT(ggml_are_same_shape(weights, weights_sum));
        weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    }
    if (scale_w) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * up = llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);
    ggml_tensor * gate = llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(gate, "ffn_moe_gate", il);

    switch (LLM_FFN_SILU) {
        case LLM_FFN_SILU:
            {
                gate = ggml_silu(ctx, gate);
                cb(gate, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            {
                gate = ggml_gelu(ctx, gate);
                cb(gate, "ffn_moe_gelu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    ggml_tensor * par = ggml_mul(ctx, up, gate); // [n_ff, n_expert_used, n_tokens]
    cb(par, "ffn_moe_gate_par", il);

    ggml_tensor * experts = llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx, experts, _repeat(ctx,weights,experts) );

    // aggregate experts
    ggml_tensor * moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
        ggml_tensor * cur_expert = ggml_view_2d(ctx, experts, n_embd, n_tokens,experts->nb[2], i*experts->nb[1]);
        if (i == 0) {
            moe_out = cur_expert;
        } else {
            moe_out = ggml_add(ctx, moe_out, cur_expert);
        }
    }

    if (n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx, moe_out);
    }

    return moe_out;             
}

bool llama_model2vocb_( struct llama_model * model,void *hData, int type){
    llama_vocab *hVocab=(llama_vocab *)(hData);
    *hVocab = model->vocab;
        
    return true;
}