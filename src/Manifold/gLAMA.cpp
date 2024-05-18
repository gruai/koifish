/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief A collection of neurons
 *  \author Yingshi Chen
 */


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
    }