/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General LLama model  
 * 
 *  \brief LLaMeta Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#pragma once
#include "../ggex/common-ggml.h" 
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Ganglia.hpp"   
#include "../Manifold/VAE.hpp" 
#include "llama.h"
#include "../Manifold/Dictionary.hpp"
#include "common.h"

#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>

static const char * LLM_TENSOR_TOKEN_EMBD    = "token_embd";
static const char * LLM_TENSOR_OUTPUT_NORM   = "output_norm";
static const char * LLM_TENSOR_OUTPUT        = "output";
static const char * LLM_TENSOR_ATTN_NORM     = "blk.%d.attn_norm";
static const char * LLM_TENSOR_ATTN_Q        = "blk.%d.attn_q";
static const char * LLM_TENSOR_ATTN_K        = "blk.%d.attn_k";
static const char * LLM_TENSOR_ATTN_V        = "blk.%d.attn_v";
static const char * LLM_TENSOR_ATTN_OUT      = "blk.%d.attn_output";
static const char * LLM_TENSOR_FFN_NORM      = "blk.%d.ffn_norm";
static const char * LLM_TENSOR_FFN_GATE      = "blk.%d.ffn_gate";
static const char * LLM_TENSOR_FFN_DOWN      = "blk.%d.ffn_down";
static const char * LLM_TENSOR_FFN_UP        = "blk.%d.ffn_up";
static const char * LLM_DICT_UP              = "dict.%d.up";
static const char * LLM_DICT_DOWN            = "dict.%d.down";

static char buffer[GGML_MAX_NAME];
static const char * TN( const char *format,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GGML_MAX_NAME,format,args );
    va_end(args);
    assert(strlen(buffer)+strlen(".weight")<=GGML_MAX_NAME);
    return strcat(buffer,".weight");
}

static const char * TNs( const char *format,const char *suffix,... )	{
    va_list args;
    va_start( args, format );
    vsnprintf( buffer,GGML_MAX_NAME,format,args );
    va_end(args);
    const char*w = ".weight";
    if(strlen(buffer) > 7 && strcmp(buffer+strlen(buffer)-7, ".weight")==0){
        w = "";
    }
    assert(strlen(buffer)+strlen(w)+strlen(suffix)<=GGML_MAX_NAME);
    strcat(buffer,w);
    strcat(buffer,suffix);        
    return buffer;
}
    
void static set_name(struct ggml_tensor * t, const char * n) {
    ggml_set_name(t, n);
    if (t->grad) {
        ggml_format_name(t->grad, "%s->grad", n);
    }
};

bool llama2params(struct llama_model * lmodel,struct CLI_params& cparam);

struct LLaMeta : public Ganglia {
    enum FFN_TYPE {
        SWIGLU = 0,
        VANILLA,
        ONLY_LNormal,    
        ONLY_RMSNormal,
        VAR_0,
        VAR_LAST,
    };
    enum FFN_TYPE tpFFN = VAR_LAST;   //VAR_LAST;    

    enum ATTENTION_TYPE {
        QKV = 0,
        BROWN,      //little gain on early test, why ???
    };
    enum ATTENTION_TYPE tpATT= QKV;   
    
    int nLayerX = -1;        //user could set it from command-line, mainly for debug
    bool isLoadTokenEmbed = false;
    struct random_normal_distribution * rnd = nullptr;    
    
    hCDICT hDict=nullptr;    

    struct lama_layer : public NeLayer {
        hGensor eps=nullptr;
        hGensor attention_norm=nullptr;
            // attention
        hGensor wq=nullptr,wk=nullptr,wv=nullptr;
        hGensor wo=nullptr;
            // normalization
        hGensor  ffn_norm=nullptr,ffn_gate=nullptr,ffn_down=nullptr,ffn_up=nullptr;  

        lama_layer()    {   name = "lama_layer";   }
        int64_t parameter_count() {
            int64_t nx = 0;
            nx += ggml_nelements(attention_norm);
            nx += ggml_nelements(wq);            nx += ggml_nelements(wk);            nx += ggml_nelements(wv);
            nx += ggml_nelements(wo);
            nx += ggml_nelements(ffn_norm); nx += ggml_nelements(ffn_gate); nx += ggml_nelements(ffn_down); nx += ggml_nelements(ffn_up);            
            return nx;
        }

        string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
    };
    uint32_t n_vocab=0,n_batch=0,n_ctx=0,n_embd=0, n_head=0,n_rot=0,n_ff=0;
    bool measure_only=false;  
    ggml_gallocr_t alloc;
    // struct train_params_common& train_params;
    struct llama_model_params llama_mparams = llama_model_default_params();
    struct llama_context_params cparams = llama_context_default_params();
    std::vector<struct ggml_tensor *> checkpoints;
    ggml_backend_buffer_t data = NULL;

    struct llama_model *lmodel = nullptr;
    struct llama_model *GetRawModel( )    {   
        assert(lmodel!=nullptr);   return lmodel;  
    }

    struct llama_context * lama_ctx = nullptr;      
    struct ggml_init_params ctx_compute_params = {0, NULL,true,};
    struct ggml_context * ctx_compute = NULL;

    hGensor  logits = NULL;
    hGensor  tokens_input  = NULL;
    struct ggml_cgraph * gb_tmp = NULL;

    
    LLaMeta()   {}
    LLaMeta( struct CLI_params params,int flag=0x0) : Ganglia(params) {
        nLayerX = params.n_layer;       
        assert(nLayerX<160);
        hparams.n_rot   = hparams.n_embd / hparams.n_head;
        hparams.n_ctx = hparams.common.n_ctx;
                    
        hparams.n_head_kv = hparams.n_head;     // n_head_kv is optional, default to n_head        
    }

    virtual ~LLaMeta() {
        free_random_normal_distribution(rnd); 
        // ggml_free(lora.ctx);
        llama_free(lama_ctx);
        llama_free_model(lmodel);
    }

 // get tensors from llama_model (possibly mmapped)
    virtual void LoadTensors(struct llama_model * lama,int flag=0x0) {
        assert(lmodel!=nullptr);  
        llama2params(lmodel,hparams);
        if( hparams.n_layer != nLayerX )        {   // from user commad
            hparams.n_layer = nLayerX;
        };    
        
        nParams = 0;
        hDict->Update(nullptr,true);   //UpdateTokEmbed(lama,nullptr,true);
       
        // layers.resize(hparams.n_layer);
        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
            // auto layer = dynamic_pointer_cast<lama_layer>(layers[i]);
            auto layer = std::make_shared<lama_layer>( );
            layers.push_back(layer);        
            layer->attention_norm = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_NORM, i));     nParams+=ggml_nelements(layer->attention_norm);
            layer->wq             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_Q, i));        nParams+=ggml_nelements(layer->wq);
            layer->wk             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_K, i));        nParams+=ggml_nelements(layer->wk);
            layer->wv             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_V, i));        nParams+=ggml_nelements(layer->wv);
            layer->wo             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_OUT, i));      nParams+=ggml_nelements(layer->wo);
            layer->ffn_norm       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_NORM, i));      nParams+=ggml_nelements(layer->ffn_norm);
            layer->ffn_gate       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_GATE, i));      nParams+=ggml_nelements(layer->ffn_gate);
            layer->ffn_down       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_DOWN, i));      nParams+=ggml_nelements(layer->ffn_down);
            layer->ffn_up         = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_UP, i));        nParams+=ggml_nelements(layer->ffn_up); 

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

    virtual void LoadModel(const char * fn_model, int flag=0x0) {
        assert(0);  //  A more practical way:   just use llama_load_model_from_file get parameters directly from gguf file
    }

    virtual hGensor UpdateGensor (const char*name,int flag=0x0){
        return GetGensor(name);
    }

    virtual void InitModel(int flag=0x0){        
        uint32_t n_embd  = hparams.n_embd;        
        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_vocab = hparams.n_vocab;
        const uint32_t n_ff    = hparams.n_ff;
        auto train_params = hparams.common;
        hparams.n_rot = hparams.n_embd / hparams.n_head;    
        _INFO("\nLLaMeta%s: init model embed=%d layer=%d ff=%d tpFFN=%d\n", __func__,n_embd,n_layer,n_ff,tpFFN);  
        _INFO("\t type of FFN=%d\n", tpFFN);  
        for (int i=0;i<n_layer;i++) {
            auto  layer = std::make_shared<lama_layer>();
            layers.push_back(layer);        //typedef shared_ptr<layer> hLayer;
            layer->attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer->wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            if(tpATT==ATTENTION_TYPE::QKV){
                layer->wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                layer->wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);                
            }

            layer->wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            
            switch(tpFFN){
            case VAR_LAST:
            case SWIGLU:
                layer->ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
                layer->ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
                layer->ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
                layer->ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff); 
                if(tpFFN==VAR_LAST && i==n_layer-1){
                    layer->eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   train_params.n_batch*hparams.n_ctx);   
                }
                break;
            case ONLY_LNormal:
            case ONLY_RMSNormal:
                layer->ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
                break;
            case VAR_0:
                layer->eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   train_params.n_batch*hparams.n_ctx);   //tensors[key] = gensor;
                break;  
            default:
                TO_DO;
            }
        }
        
        nParams = 0;
        
        hDict->CreateEmbeddings(rnd,0x0);    //    UpdateTokEmbed(lmodel,rnd,0x0);    
        data = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
        hDict->Update(rnd,0x0);    //    UpdateTokEmbed(lmodel,rnd,0x0);        
            
        for (u_int i=0;i<n_layer;i++) {
            auto layer = dynamic_pointer_cast<lama_layer>(layers[i]);        
            InitGensor(ctx,layer->attention_norm, TN(LLM_TENSOR_ATTN_NORM, i), rnd);
            InitGensor(ctx,layer->wq,             TN(LLM_TENSOR_ATTN_Q, i), rnd);
            if(layer->wk!=nullptr)      InitGensor(ctx,layer->wk,TN(LLM_TENSOR_ATTN_K, i), rnd);
            if(layer->wv!=nullptr)      InitGensor(ctx,layer->wv,TN(LLM_TENSOR_ATTN_V, i), rnd);
            InitGensor(ctx,layer->wo,             TN(LLM_TENSOR_ATTN_OUT, i), rnd);
            if(layer->ffn_norm!=nullptr)
                InitGensor(ctx,layer->ffn_norm,       TN(LLM_TENSOR_FFN_NORM, i), rnd);
            if(layer->ffn_up!=nullptr){                
                InitGensor(ctx,layer->ffn_gate,       TN(LLM_TENSOR_FFN_GATE, i), rnd);
                InitGensor(ctx,layer->ffn_down,       TN(LLM_TENSOR_FFN_DOWN, i), rnd);
                InitGensor(ctx,layer->ffn_up,         TN(LLM_TENSOR_FFN_UP, i), rnd);                
            }
        }
        // free_random_normal_distribution(rnd); 
        
        if (!hparams.only_write_model) {
            ggml_opt_init(hOPT->opt->ctx, hOPT->opt, hOPT->opt->params, nParams);
        }

        szModel = (ggml_used_mem(ctx) + ggml_backend_buffer_get_size(data)), (float) (ggml_used_mem(ctx) + ggml_backend_buffer_get_size(data)) ;
    }

    virtual size_t MostMemSize(int flag)    {
        int n_layer = nLayerX<=0 ? hparams.n_layer : nLayerX;
        int nHead = hDict!=nullptr ? hDict->nLevel*3+2+6 : 6; 
        size_t sz = ggml_tensor_overhead()*2*(nHead + n_layer*18);
        return sz;
    }

    void CreateWiki(int flag=0x0) override  {
        CLI_params wiki_param = hparams;
        wiki_param.only_infer = true;        
        wiki = std::make_shared<LLaMeta>(wiki_param);      
        wiki->Init();
        wiki->BuildGraph( );
    }

        //for tokens_input & target_probs
    virtual void InitEntryTensors(int flag=0x0) {
        auto train_params = hparams.common;
        int n_tokens = hparams.n_ctx,n_vocab = hparams.n_vocab,n_batch = train_params.n_batch;
        struct ggml_init_params ctx_input_params = {// mem_size mem_buffer no_alloc
            ggml_tensor_overhead() * 2, NULL,true,                        
        };
        struct ggml_context * ctx_input = ggml_init(ctx_input_params);
        tokens_input  = ggml_new_tensor_2d(ctx_input, GGML_TYPE_I32, n_tokens, n_batch);
        target_probs  = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
        ggml_backend_buffer_t input_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx_input, ggml_backend_cpu_buffer_type());
        size_t max_input_size = ggml_backend_buffer_get_size(input_data);
        _INFO("%s: input_size(%d,%d) = %zu bytes (%.1f MB)\n", __func__, n_tokens, n_batch, max_input_size, (float) max_input_size / (1024.0f*1024.0f));
        ggml_free(ctx_input);
    }

    virtual void Init(int flag=0x0)     {
        auto train_params = hparams.common;
        if (train_params.seed == LLAMA_DEFAULT_SEED) {
            train_params.seed = time(NULL); 
        }
        llama_mparams.n_gpu_layers = train_params.n_gpu_layers;
        llama_mparams.vocab_only = hparams.train=="scratch";     //need lmodel to tokenize training samples
    
        _INFO("%s: model base = '%s' nEmbd=%d\n", __func__, hparams.fn_model_base.c_str(),hparams.n_embd);
        lmodel = llama_load_model_from_file(hparams.fn_model_base.c_str(), llama_mparams);    
        if(llama_mparams.vocab_only){
        }
        
        // InitEntryTensors(flag);
        if(hparams.only_infer){
            hDict = std::make_shared<ConsiceDict>(this);        hDict->isLoadTokenEmbed = true;
            hDict->LoadVocab(hparams.fn_model_base.c_str(),0x0);
            LoadTensors(lmodel,0x0);
            return;
        }
        //Tokenize need lama_ctx!!!
        lama_ctx = llama_new_context_with_model(lmodel, cparams);          
        nParamsGGUF = 0;
        if(llama_mparams.vocab_only)    {   //not load tensors from fn_model_base
            updateTMap = true;            
        } else { // load all tensors from fn_model_base
            auto tmaps = llama_internal_get_tensor_map(lama_ctx);
            assert(tensors.size()==0);
            for(auto it : tmaps){
                tensors[it.first] = it.second;
                nParamsGGUF += ggml_nelements(it.second);
            }
        }
        hOPT = std::make_shared<Optimizer>(this,train_params,flag);
        hDistler = hparams.sigma=="" ? nullptr : std::make_shared<Distillation>(this,hparams,0x0);     //ADD SIGMA
        hDict = std::make_shared<ConsiceDict>(this);
        hDict->LoadVocab(hparams.fn_model_base.c_str(),0x0);
        // hDict->LoadVocab_v1(hparams.fn_model_base,hparams,*lmodel, 0x0);

        struct ggml_init_params ctx_model_params;
        ctx_model_params.mem_size   = MostMemSize(0x0) ;
        ctx_model_params.mem_buffer = NULL;
        ctx_model_params.no_alloc   = true;
        assert(ctx==nullptr);
        ctx = ggml_init(ctx_model_params);

        InitModel(flag);
        // hGensor tok_0 = UpdateGensor(TN(LLM_TENSOR_TOKEN_EMBD));  //only for debug llama_get_model_tensor
        n_vocab = hparams.n_vocab;          n_batch  = train_params.n_batch;        n_ctx = hparams.n_ctx;        n_embd = hparams.n_embd;
        n_head = hparams.n_head,            n_rot = hparams.n_rot,     n_ff = hparams.n_ff;
        // opt->iter = train->train_its;
        
        InitEntryTensors(flag); 
        hparams.Dump();         //        print_params(&hparams);
        auto train=hOPT->train;
        _INFO("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
        _INFO("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
        _INFO("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
        _INFO("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
        _INFO("%s: nParams=%zu model_size = %zu bytes (%.1f MB)\n", __func__, nParams,szModel,szModel / (1024.0f*1024.0f) );
        _INFO("%s: n_vocab=%d,n_batch=%d,n_ctx=%d,n_embd=%d,n_head=%d,n_rot=%d,n_ff=%d\n", __func__, 
            n_vocab,n_batch,n_ctx,n_embd,n_head,n_rot,n_ff );

        save_data.Init(hparams,lmodel);
        hOPT->Dump(0x0);
    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;

    virtual void Dump(int type,int flag=0x0)    {
        string suffix="\n========\n",prefix;
        __repr__(suffix,prefix);
        hparams.Dump();         //        print_params(&hparams)
        _INFO("====== nParams = %ld(%.6gM) ======\n", nParams,nParams/1.0e6);
        _INFO("====== tensors=%d gf=(%d %d)  gb=(%d %d) ======\n", tensors.size(),gf->n_nodes,gf->n_leafs,gb->n_nodes,gb->n_leafs);
        // _INFO("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
        // _INFO("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
        // _INFO("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
        // _INFO("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
        _INFO("%s: nParams=%zu model_size = %zu bytes (%.1f MB)\n", __func__, nParams,szModel,szModel / (1024.0f*1024.0f) );
        _INFO("%s: n_vocab=%d,n_batch=%d,n_ctx=%d,n_embd=%d,n_head=%d,n_rot=%d,n_ff=%d\n", __func__, 
            n_vocab,n_batch,n_ctx,n_embd,n_head,n_rot,n_ff );
        hOPT->Dump( 1 );
        
    }

    void BuildGraph(int flag=0x0)   override   {        // measure required memory for compute tensors
        int n_tokens = hparams.n_ctx;
        int n_vocab  = hparams.n_vocab;
        auto train_params = hparams.common;
        int n_batch  = train_params.n_batch;
        ctx_compute_params.mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
            (hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
        size_t best_compute_size = SIZE_MAX;
        enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT;  //GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;  //GGML_CGRAPH_EVAL_ORDER_COUNT;
        if(1)   {// find best evaluation order
            for (unsigned order = 0; order < (unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order) {
                ctx_compute = ggml_init(ctx_compute_params);
                ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
                gf->order = (enum ggml_cgraph_eval_order) order;
                gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
                gb_tmp = train_params.use_checkpointing
                    ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
                    : NULL;
                build_finetune(ctx_compute,alloc,true,flag);
                size_t max_compute_size = ggml_gallocr_get_buffer_size(alloc, 0); // FIXME: this will still allocate the buffer
                if (max_compute_size < best_compute_size) {
                    best_compute_size = max_compute_size;
                    best_order = gf->order;
                }
                ggml_gallocr_free(alloc);
                ggml_free(ctx_compute);
            }
        
            size_t max_compute_size = best_compute_size;
            _INFO("%s: compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
            _INFO("%s: evaluation order = %s\n", __func__,
                (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
                (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
                "invalid");
        }
        

        ctx_compute = ggml_init(ctx_compute_params);
        alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        gf = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gf->order = best_order;
        gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
        gb_tmp = train_params.use_checkpointing
            ? ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true)
            : NULL;
        build_finetune(ctx_compute,alloc,false,flag);
        
        Statistic(0x0);
#ifndef NDEBUG
        ggml_graph_print(gf);           ggml_graph_print(gb);       //only for debug
#endif
    }

    virtual void BuildTarget( struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,hGensor cur,hGensor _tNorm,hGensor _tOutput,hGensor KQ_pos, int flag=0x0); 

    virtual hBrownMotion CreateBrownMotion(hGensor wq, hGensor wk, hGensor wv)  {
        hBrownMotion hMotion =  (tpATT==ATTENTION_TYPE::QKV) ? 
            std::make_shared<QKV_Motion> (wq,wk,wv,hparams,0x0) :
            std::make_shared<BROWN_Motion> (wq,hparams,0x0);
        return hMotion;
    }
    
    //n_embd_head, n_head_kv
    // virtual hGensor  build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    //     hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down, int flag=0x0) ;
    virtual hGensor  build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor KQ_pos,int flag=0x0) ;

    virtual void build_finetune(struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,int flag=0x0)   {        // measure required memory for compute tensors
        int n_tokens = hparams.n_ctx;
        int n_vocab  = hparams.n_vocab;
        auto train_params = hparams.common;
        int n_batch  = train_params.n_batch;
        measure_only = m_only;
        ggml_set_scratch(ctx, { 0, 0, nullptr, });      
        const int n_past = 0;
        const int N = n_tokens;
        const int n_ctx      = hparams.n_ctx;
        const int n_embd     = hparams.n_embd;
        const int n_layer    = hparams.n_layer;
        const int n_head     = hparams.n_head;
        const int n_rot      = hparams.n_rot;
        const int n_ff       = hparams.n_ff;
        const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
        const float rope_freq_base  = hparams.rope_freq_base;
        const float rope_freq_scale = hparams.rope_freq_scale;        

        // KQ_pos - contains the positions
        hGensor  KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_input(KQ_pos);
        struct ggml_context   * ctx0= ctx;
        set_name(tokens_input, "tokens_input");
        set_name(target_probs,      "targets");

        GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
        hGensor _tEmbed = UpdateGensor (hDict->tok_embeddings->name); 
        hGensor _tNorm = UpdateGensor (hDict->norm->name); 
        hGensor _tOutput = UpdateGensor (hDict->output->name);       
        hGensor  t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
        hGensor  t01 = ggml_get_rows(ctx, _tEmbed, t00);    set_name(t01, "t01"); 
        hGensor  cur = t01;
        cur = hDict->ENC(ctx,t01);      // cur = ggml_mul_mat(ctx, hDict->encoder, t01 );
        set_name(cur, "embed_encoder");
        
            
        assert_shape_2d(cur, n_embd, N*n_batch);
        checkpoints.clear();
        checkpoints.push_back(tokens_input);
        checkpoints.push_back(target_probs);
        checkpoints.push_back(t00);
        checkpoints.push_back(t01);

        const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        for (auto lay : layers) {
            auto layer = dynamic_pointer_cast<lama_layer>(lay);
            // hGensor a_norm = layer->attention_norm;     assert(a_norm==layer->attention_norm);
            /*hGensor wq = UpdateGensor (layer->wq->name);                     
            hGensor wk = UpdateGensor (layer->wk->name);
            hGensor wv = UpdateGensor (layer->wv->name);
            hGensor wo = UpdateGensor (layer->wo->name);
            hGensor a_norm = UpdateGensor (layer->attention_norm->name);    
            hGensor ffn_norm = UpdateGensor (layer->ffn_norm->name); 
            if(layer->ffn_up!=nullptr){                
                hGensor ffn_up = UpdateGensor (layer->ffn_up->name);
                hGensor ffn_gate = UpdateGensor (layer->ffn_gate->name);
                hGensor ffn_down = UpdateGensor (layer->ffn_down->name);                
            } */ 
                   
            cur = build_layer_( N,ctx, cur, layer, KQ_pos);
           
            checkpoints.push_back(cur);
        }
        BuildTarget(ctx,alloc,m_only,cur,_tNorm,_tOutput,KQ_pos);        
    }

    virtual void Tokenize( int flag=0x0 )   {   
        auto train_params = hparams.common;
        int n_tokens = hparams.n_ctx;

        assert( lama_ctx != nullptr );
        // _INFO("%s: tokenize training data\n", __func__);
        _INFO("%s: tokenize training data from %s\n", __func__, hparams.common.fn_train_data.c_str());
        _INFO("%s: sample-start: %s\n", __func__, hparams.common.sample_start.c_str());
        _INFO("%s: include-sample-start: %s\n", __func__, hparams.common.include_sample_start ? "true" : "false");
        tokenize_file(lama_ctx,
                train_params.fn_train_data.c_str(),
                train_params.sample_start,
                train_params.include_sample_start,
                train_params.overlapping_samples,
                n_tokens,
                hOPT->train_tokens,
                hOPT->train_samples_begin,
                hOPT->train_samples_size);
        GGML_ASSERT(hOPT->train_samples_begin.size() == hOPT->train_samples_size.size());
        _INFO("%s: number of training tokens: %zu\n", __func__, hOPT->train_tokens.size());        
    }

    

    void Loss(int flag=0x0)     override   {

    }

    void Train(int flag=0x0)    override   {
        Tokenize( flag );

        auto train_params = hparams.common;
        hOPT->Shuffle(n_vocab,train_params,flag);
        Dump(0x0);        
        auto opt = hOPT->opt;
        auto adam = opt->params.adam;
        _INFO("%s: sched=%.4g ADAM(lr=%g,%g,[%g-%g])\n", __func__,adam.sched,adam.alpha,adam.decay,adam.beta1,adam.beta2);
        print_build_info();
        auto train=hOPT->train;
        struct train_opt_callback_data opt_cb_data;
        // measure required memory for work buffer
        size_t max_work_size = ggml_graph_plan(gb, train_params.n_threads).work_size + GGML_OBJECT_SIZE;
        _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));

        // context for work buffer
        struct ggml_init_params ctx_work_params = {
            max_work_size, // mem_size
            NULL,          // mem_buffer
            false,         // no_alloc
        };
        struct ggml_context * ctx_work = ggml_init(ctx_work_params);
        hOPT->Init_CallbackData(opt_cb_data,lama_ctx,hparams.common,tokens_input,0x0);
        int64_t t0 = ggml_time_ms();
        // save_train_(&save_data, opt_cb_data.train);      //warmup save
        enum ggml_opt_result result = hOPT->ggml_train(ctx_work, &opt_cb_data,loss,target_probs,gf,gb);       

        ggml_free(ctx_work);
        ggml_free(ctx_compute);
        // ggml_free(ctx_input);
        ggml_gallocr_free(alloc);        

        int64_t t1 = ggml_time_ms();
        _INFO("%s: total training time: ", __func__);
        print_duration((double) (t1 - t0));
        _INFO("\n");
    }
    void save_gguf(const char * fn_model_base, int flag)    override;
    // virtual void save_train_(struct save_train_files_data * data, struct train_state * train);
};
typedef shared_ptr<LLaMeta> hLLAMA;

struct LLAMA_LORA  : public LLaMeta {
    uint32_t _rank_=4;      //lora_r
    uint32_t lora_r = 1;
    uint32_t lora_alpha = 1;
    uint32_t n_rank_attention_norm = 1;
    uint32_t n_rank_wq = _rank_;
    uint32_t n_rank_wk = _rank_;
    uint32_t n_rank_wv = _rank_;
    uint32_t n_rank_wo = _rank_;
    uint32_t n_rank_ffn_norm = 1;
    uint32_t n_rank_ffn_gate = _rank_;
    uint32_t n_rank_ffn_down = _rank_;
    uint32_t n_rank_ffn_up = _rank_;
    uint32_t n_rank_tok_embeddings = _rank_;
    uint32_t n_rank_norm = 1;
    uint32_t n_rank_output = _rank_;

    hGensor  tok_embeddings_a,tok_embeddings_b,norm_a,norm_b,output_a,output_b;
    // struct random_normal_distribution * rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    
    LLAMA_LORA( struct CLI_params params,int flag=0x0)   {
        hparams = params;
        nLayerX = hparams.n_layer;
        hparams.n_rot   = hparams.n_embd / hparams.n_head;
        hparams.n_ctx = hparams.common.n_ctx;
        assert(hparams.lora_r>0);
        n_rank_wq = n_rank_wk = n_rank_wv = n_rank_wo = hparams.lora_r;
        n_rank_ffn_gate = n_rank_ffn_down = n_rank_ffn_up = n_rank_tok_embeddings = hparams.lora_r;
        n_rank_output = hparams.lora_r;

        isLoadTokenEmbed = true;
    }

    virtual ~LLAMA_LORA() {
        free_random_normal_distribution(rnd);
    }  
    
    size_t MostMemSize(int flag)     override   {
        int n_layer = nLayerX<=0 ? hparams.n_layer : nLayerX;
        size_t sz = ggml_tensor_overhead()*2*(9 + n_layer*27);
        return sz;
    }

    void InitModel(int flag=0x0)    override    {
        _INFO("LLAMA_LORA::%s: init model\n", __func__);
        hDict->isLoadTokenEmbed = true;
        
        LoadTensors(lmodel);  
        // LoadModel( hparams.fn_model_base,0 );        
        hparams.n_ctx = hparams.common.n_ctx;
 
        nParams = 0;
        enum ggml_type lora_type  = GGML_TYPE_F32;
        // lora_type = GGML_TYPE_COUNT;
        InitFactor(ctx, hDict->tok_embeddings,lora_type, n_rank_tok_embeddings,0x0);
        // tok_embeddings_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_tok_embeddings, n_embd);
        // tok_embeddings_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_tok_embeddings, n_vocab);
        InitFactor(ctx, hDict->norm,lora_type, n_rank_norm,0x0);
        // norm_a           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_norm, n_embd);
        // norm_b           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_norm, 1);
        InitFactor(ctx, hDict->output,lora_type, n_rank_output,0x0);
        // output_a         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_output, n_embd);
        // output_b         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_output, n_vocab);        
        
        // layers.resize(n_layer);
        for (uint32_t i = 0; i < hparams.n_layer; ++i) {
            auto lay_ = dynamic_pointer_cast<lama_layer>(layers[i]);
            // auto layer = std::make_shared<lora_layer>();
            // lora_layers.push_back(layer);        //typedef shared_ptr<layer> hLayer;
            InitFactor(ctx, lay_->attention_norm,lora_type, n_rank_attention_norm,0x0);
            // layer->attention_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_attention_norm, n_embd);
            // layer->attention_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_attention_norm, 1);
            InitFactor(ctx, lay_->wq,lora_type, n_rank_wq,0x0);
            // layer->wq_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wq, n_embd);
            // layer->wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wq, n_embd);
            InitFactor(ctx, lay_->wk,lora_type, n_rank_wk, 0x0);
            // layer->wk_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wk, n_embd);
            // layer->wk_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wk, n_embd_gqa);
            InitFactor(ctx, lay_->wv,lora_type, n_rank_wv, 0x0);
            // layer->wv_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wv, n_embd);
            // layer->wv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wv, n_embd_gqa);
            InitFactor(ctx, lay_->wo,lora_type, n_rank_wo, 0x0);
            // layer->wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wo, n_embd);
            // layer->wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wo, n_embd);
            InitFactor(ctx, lay_->ffn_norm,lora_type, n_rank_ffn_norm, 0x0);
            // layer->ffn_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_norm, n_embd);
            // layer->ffn_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_norm, 1);
            InitFactor(ctx, lay_->ffn_gate,lora_type, n_rank_ffn_gate, 0x0);
            // layer->ffn_gate_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_gate, n_embd);
            // layer->ffn_gate_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_gate, n_ff);
            InitFactor(ctx, lay_->ffn_down,lora_type, n_rank_ffn_down, 0x0);
            // layer->ffn_down_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_down, n_ff);
            // layer->ffn_down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_down, n_embd);
            InitFactor(ctx, lay_->ffn_up,lora_type, n_rank_ffn_up, 0x0);
            // layer->ffn_up_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_up,   n_embd);
            // layer->ffn_up_b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_up,   n_ff);
        }        
        // allocate data for lora tensors
        data = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        assert(updateTMap==false);
        // size_t sz1=ggml_backend_buffer_get_size(data);      //130872416
        ggml_backend_buffer_type_t buf0 = ggml_backend_cpu_buffer_type(); 
        
        ggml_backend_buffer_type_t buf1 = ggml_backend_cpu_buffer_type();
        // data = ggml_backend_alloc_ctx_tensors_from_buft(ctx,buf1 );
        size_t sz2=ggml_backend_buffer_get_size(data);      //130872416
        if (!hparams.only_write_model) {    //only_write_lora
            ggml_opt_init( hOPT->opt->ctx, hOPT->opt, hOPT->opt->params, nParams);     //nx=16358481
        }
        szModel = ggml_used_mem(ctx) + ggml_backend_buffer_get_size(data);  //580800+65436224

        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }

    void Dump(int type,int flag=0x0)    override   {
        LLaMeta::Dump(type);
        _INFO("%s:%s model=%zu bytes (%.1f MB)\tn_embd=%d n_vocab=%d n_ff=%d n_rank_wq=%d\n", "LLAMA_LORA", hparams.sTune(),szModel, (float) (szModel) / (1024.0f*1024.0f),
            n_embd,n_vocab,n_ff,n_rank_wq);
    }


    CLI_params::TUNE_ALG TuneOnShape(int nIn,int nOut){
        CLI_params::TUNE_ALG tune = hparams.tune;
        if( nIn==1 || nOut==1 || nIn*nOut>5120*5120*4 ) {
            tune = CLI_params::LORA;
        }
        return tune;
    }
    hGensor UpdateGensor( const char*name,int flag=0x0 ) override    {
        hGensor gensor = LLaMeta::UpdateGensor(name);       assert(gensor!=nullptr); 
        // if(ggml_is_quantized(gensor->type))    
        //     ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32); 

        auto shape = gensor->ne;
        int nIn=shape[0], nOut=shape[1];
        CLI_params::TUNE_ALG tune = TuneOnShape(nIn,nOut);  
        hGensor u,v,d,ga,gb;
        if(tune==CLI_params::LORA_SVD){
            u = GetGensor(TNs(name,".u"));    v = GetGensor(TNs(name,".v"));    d = GetGensor(TNs(name,".d"));
            // if(d->grad==nullptr){      //               
            //     ggml_set_param(ctx, d);               
            // }
        }else{
            string na=string(name)+".lo_a",nb=string(name)+".lo_b";
            ga = GetGensor(na.c_str());             
            gb = GetGensor(nb.c_str());
            assert(ga!=nullptr && gb!=nullptr);     //0x555555b4ebb0        0x555555b4ed40
            // if(ga->grad==nullptr){      //
            //     ggml_set_param(ctx, ga);            ggml_set_param(ctx, gb);
            //     randomize_tensor_normal(ga, rnd);            
            //     ggml_set_zero(gb);              
            // }
        }
        hGensor delta = nullptr;
        switch( tune )    {
        case CLI_params::LORA:      //   9.615243
            if(!measure_only){
                randomize_tensor_normal(ga, rnd);       ggml_set_zero(gb); 
            }
            delta = ggml_mul_mat(ctx_compute, ga, gb);  //gensor = add_to_f32(ctx_compute, gensor, ggml_mul_mat(ctx_compute, ga, gb));
        break;
        case CLI_params::OFF:   //would fail because gensor has no grad!!! gensor fixed as load from model file
            assert(0);
            break;
        case CLI_params::LORA_AB:       //9.784955
        //,nHeavy = ga->ne[0], rank = nHeavy
            if(!measure_only)
                Gensor_loab(ctx_compute,gensor,ga->ne[0],ga,gb);
            delta = ggml_mul_mat(ctx_compute, ga, gb);  //gensor = add_to_f32(ctx_compute, gensor, ggml_mul_mat(ctx_compute, ga, gb));
            break;
        case CLI_params::LORA_SVD:  {       //9.784960
                if(!measure_only)
                    Gensor_SVD(ctx_compute,gensor,d->ne[0],u,d,v);
                // hGensor dv = ggml_mul_mat(ctx_compute, ggml_diag(ctx_compute,d),v);  //GGML_OP_DIAG don't support backward!!!
                hGensor dv = ggml_mul_mat(ctx_compute, d,v);
                delta = ggml_mul_mat(ctx_compute, u, dv);  //gensor = add_to_f32(ctx_compute, gensor, ggml_mul_mat(ctx_compute, u, dv));
            }
            break;
        default:
            assert(0);
            break;
        }       
        if(hDistler!=nullptr) 
            gensor = hDistler->UpdateGG(ctx_compute,gensor,delta);
        else
            gensor = add_to_f32(ctx_compute, gensor, delta);
        
        return gensor;
    }

    void InitFactor(struct ggml_context *ctx, hGensor gensor, enum ggml_type typ0, int lo_rank,int flag=0x0){
        if(lo_rank==0)
            return;
        assert(lo_rank>0);
        int nIn=gensor->ne[0], nOut=gensor->ne[1];
        //always A'*B in GGML
        CLI_params::TUNE_ALG tune = TuneOnShape(nIn,nOut);  
        enum ggml_type type = typ0==GGML_TYPE_COUNT?gensor->type : GGML_TYPE_F32;
        if(tune==CLI_params::LORA_SVD){
            hGensor u = ggml_new_tensor_2d(ctx, type, lo_rank, nIn);   
            InitGensor(ctx,TNs(gensor->name,  ".u"), u,true);
            hGensor v = ggml_new_tensor_2d(ctx, type, lo_rank, nOut);
            InitGensor(ctx,TNs(gensor->name,  ".v"), v,true);
            // hGensor d = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, lo_rank); 
            hGensor d = ggml_new_tensor_2d(ctx, type, lo_rank, lo_rank); 
            InitGensor(ctx,TNs(gensor->name,  ".d"), d,true);
        }else{
            assert(type==GGML_TYPE_F32);
            hGensor ga = ggml_new_tensor_2d(ctx, type, lo_rank, gensor->ne[0]);
            hGensor gb = ggml_new_tensor_2d(ctx, type, lo_rank, gensor->ne[1]);            
            InitGensor(ctx,TNs(gensor->name,  ".lo_a"),ga,true);
            InitGensor(ctx,TNs(gensor->name,  ".lo_b"),gb,true);        
            // nParams +=  ggml_nelements(ga);        nParams +=  ggml_nelements(gb);            
        }
    }    

    virtual void build_finetune(struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,int flag=0x0)     {   
        int n_tokens = hparams.n_ctx;
        int n_vocab  = hparams.n_vocab;
        auto train_params = hparams.common;
        int n_batch  = train_params.n_batch;
        measure_only = m_only;
        ggml_set_scratch(ctx, { 0, 0, nullptr, });
        const int n_past = 0;
        const int N = n_tokens;
        const int n_ctx       = hparams.n_ctx;
        // const int n_vocab     = hparams.n_vocab;
        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.n_layer;
        const int n_head      = hparams.n_head;
        const int n_head_kv   = hparams.n_head_kv;
        const int n_ff        = hparams.n_ff;
        const int n_rot       = hparams.n_embd_head();
        const int n_embd_head = hparams.n_embd_head();
        const int n_embd_gqa  = hparams.n_embd_gqa();
        const float rms_norm_eps    = hparams.f_norm_rms_eps;
        const float rope_freq_base  = hparams.rope_freq_base;
        const float rope_freq_scale = hparams.rope_freq_scale;
        GGML_ASSERT((size_t) n_layer == layers.size());
       
        // KQ_pos - contains the positions
        hGensor  KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_input(KQ_pos);
        struct ggml_context   * ctx0= ctx;
        set_name(tokens_input, "tokens_input");
        set_name(target_probs,      "targets");
        auto rope = [ctx, KQ_pos, n_rot, n_ctx, rope_freq_base, rope_freq_scale]
                (struct ggml_tensor * t) -> struct ggml_tensor * {
            // not capturing these, to silcence warnings
            const int rope_mode = 0;

            return ggml_rope_custom(ctx,
                t, KQ_pos, n_rot, rope_mode, n_ctx, 0,
                rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f
            );
        };

        GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
        LLaMeta *base = dynamic_cast<LLaMeta *>(this);
    //    hGensor  _tEmbed = add_to_f32(ctx, base->tok_embeddings, ggml_mul_mat(ctx, tok_embeddings_a, tok_embeddings_b));
    //    hGensor  _tNorm           = add_to_f32(ctx, base->norm, ggml_mul_mat(ctx, norm_a, norm_b));
    //    hGensor  _tOutput         = add_to_f32(ctx, base->output, ggml_mul_mat(ctx, output_a, output_b));
        hGensor _tEmbed = base->hDict->tok_embeddings,_tNorm = base->hDict->norm,_tOutput = base->hDict->output; 
        _tEmbed = UpdateGensor (base->hDict->tok_embeddings->name); 
        _tNorm = UpdateGensor (base->hDict->norm->name); 
        _tOutput = UpdateGensor (base->hDict->output->name);         

        hGensor  t00 = ggml_reshape_1d(ctx, tokens_input, N*n_batch);  set_name(t00, "t00"); assert_shape_1d(t00, N*n_batch);
        hGensor  t01 = ggml_get_rows(ctx, _tEmbed, t00);        set_name(t01, "t01"); assert_shape_2d(t01, n_embd, N*n_batch);
        hGensor  cur = t01;
        checkpoints.clear();
        // std::vector<struct ggml_tensor *> checkpoints;
        if (train_params.use_checkpointing) {
            checkpoints.push_back(tokens_input);    checkpoints.push_back(target_probs);
            checkpoints.push_back(t00);             checkpoints.push_back(t01);
        }

        const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        for (int il = 0; il < n_layer; ++il) {
            auto layer = dynamic_pointer_cast<lama_layer>(layers[il]);
            // auto llayer = dynamic_pointer_cast<lora_layer>(lora_layers[il]); 
            hGensor a_norm = UpdateGensor (layer->attention_norm->name);   
            hGensor wq = n_rank_wq ==0 ? nullptr : UpdateGensor (layer->wq->name);                     
            hGensor wk = n_rank_wk ==0 ? nullptr : UpdateGensor (layer->wk->name);
            hGensor wv = n_rank_wv ==0 ? nullptr : UpdateGensor (layer->wv->name);
            hGensor wo = UpdateGensor (layer->wo->name);
            hGensor ffn_norm = UpdateGensor (layer->ffn_norm->name);
            hGensor ffn_up = UpdateGensor (layer->ffn_up->name);
            hGensor ffn_gate = UpdateGensor (layer->ffn_gate->name);
            hGensor ffn_down = UpdateGensor (layer->ffn_down->name);
            cur = build_layer_( N,ctx,cur, layer, KQ_pos);
            if(wk==nullptr && wv==nullptr){ //markov

            }
            
            if (train_params.use_checkpointing) {
                checkpoints.push_back(cur);
            }     
        }

        BuildTarget(ctx,alloc,m_only,cur,_tNorm,_tOutput,KQ_pos);
    }

};

/**
 *  llama-2-13b.Q2_K.gguf           wq,wk,wv has same shape
 *  Meta-Llama-3-8B.Q2_K.gguf       wq,wk,wv has different shape!
*/
struct LLAMA_Brown  : public LLAMA_LORA {
    LLAMA_Brown( struct CLI_params params,int flag=0x0) 
        : LLAMA_LORA(params,flag)  {
        n_rank_wq = 4;
        n_rank_wk = 0;
        n_rank_wv = 0;
    }

    virtual ~LLAMA_Brown() {
        // free_random_normal_distribution(rnd);
    }  

    hBrownMotion CreateBrownMotion(hGensor wq, hGensor wk, hGensor wv)  override    {
        hBrownMotion hMotion = std::make_shared<BROWN_Motion> (wq,hparams,0x0);
        return hMotion;
    }
};

struct LLAMA_VAE  : public LLaMeta {
    
    int lama_embed = 0,latent_dim = 192;

    LLAMA_VAE( struct CLI_params params,int flag=0x0) 
        : LLaMeta(params,flag)  {
        isLoadTokenEmbed = true;
        // hparams.common.adam_alpha = 0.0001;     // 
    }

    virtual ~LLAMA_VAE() {        
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("LLAMA_VAE%s: init model\n", __func__);

        llama2params(lmodel,hparams);
        hparams.n_head = 8;    
        hparams.n_head_kv = hparams.n_head;     //hack
        hparams.n_layer = nLayerX;

        tensors.clear();
        hDict->InitVAE();       updateTMap = true;

        LLaMeta::InitModel(flag);        
    }     
   
};