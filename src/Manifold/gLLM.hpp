/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General LLM model  
 * 
 *  \brief NLP_AutoRegressive Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#pragma once
#include "../ggex/common-ggml.h" 
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Fish.hpp"   
#include "../Manifold/VAE.hpp" 
#include "llama.h"
#include "llama_cys.h"
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
static const char *LLM_TENSOR_FFN_GATE_INP      = "blk.%d.ffn_gate_inp";
static const char *LLM_TENSOR_FFN_GATE_EXPS     = "blk.%d.ffn_gate_exps";
static const char *LLM_TENSOR_FFN_DOWN_EXPS     = "blk.%d.ffn_down_exps";
static const char *LLM_TENSOR_FFN_UP_EXPS       = "blk.%d.ffn_up_exps";
static const char *LLM_TENSOR_FFN_GATE_INP_SHEXP= "blk.%d.ffn_gate_inp_shexp";
static const char *LLM_TENSOR_FFN_GATE_SHEXP    = "blk.%d.ffn_gate_shexp";
static const char *LLM_TENSOR_FFN_DOWN_SHEXP    = "blk.%d.ffn_down_shexp";
static const char *LLM_TENSOR_FFN_UP_SHEXP      = "blk.%d.ffn_up_shexp";

   
// void static set_name(struct ggml_tensor * t, const char * n) {
//     ggml_set_name(t, n);
//     if (t->grad) {
//         ggml_format_name(t->grad, "%s->grad", n);
//     }
// };
struct LAMA : public WIKI   {    
    struct llama_hparams llama_params;
    struct llama_model_params llama_model_params = llama_model_default_params();
    struct llama_context_params cparams = llama_context_default_params();            
    struct llama_model *lmodel = nullptr;
    struct llama_context *_ctx = nullptr; 
    struct llama_kv_cache *_cache=nullptr;
    ggml_cgraph * gf = nullptr;
    hGensor res = nullptr;  
    
    bool isValid(   )  override 
    {   return lmodel!=nullptr && _ctx!=nullptr;   }

    std::string T2STR(int32_t tok,int flag=0x0 ) override;   
    bool Decode(std::vector<int32_t>&ids,int start,int n_past,bool out_all)  override;
    void Answer(std::vector<int32_t>&ids,int flag)  override;
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    /*
        uint32_t n_ctx;             // text context, 0 = from model_
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
    */
    int nCTX()   override  {   return cparams.n_ctx;    };

    LAMA(CLI_params& hparams,const std::string&model_path);
    void CopyParams(CLI_params& params,int flag=0x0)    override;
    // bool CopyGensors(Fish *hFish,int flag=0x0)          override;

    virtual ~LAMA(){
        if(gf!=nullptr)
            ggml_graph_clear(gf);            
        llama_free(_ctx);
        llama_free_model(lmodel);
    }

    void Reset(int flag=0x0)    override;   

    hGensor P()         override        {   assert(res!=nullptr);   return res; }
    hGensor Target()    override        {   return nullptr; }

    //Hack function, should be called afer decode and the data maybe modified!!!
    const float *GetLogits(int n_vocab,int n_ctx,int idx=-1)   override;
};

struct NLP_AutoRegressive : public Fish {
    enum FFN_TYPE tpFFN = VAR_LAST;   //VAR_LAST;    

    /*
        @"/home/cys/rnd/lic/log/Distiller/07_11_QKV_brown.info"
        BROWN attenion would falls into local trap much earlier than QKV attention.
    */
    enum ATTENTION_TYPE {
        QKV = 0,
        BROWN,      //little gain on early test, why ???
        OFF
    };
    enum ATTENTION_TYPE tpATT = QKV;   
    
    bool isLoadTokenEmbed = false;
    
    hCDICT hDict=nullptr; 
    
    bool isAttOnBC=false;   
        
    // struct train_params_common& train_params;
    
    CHILD_0909_WIKIS
    //Assume wikis[0] is the backbone of FISH
    struct LAMA *lama(int id=0)    {  
        if(wikis.size()==0) 
            return nullptr;
        assert(id>=0 && id<wikis.size());
        LAMA *lama = dynamic_cast<LAMA *>(wikis[id].get());
        assert(lama!=nullptr);
        return lama;  
    }

    struct llama_model *GetRawModel( )    {   
        LAMA *lam = lama( );    //dynamic_cast<LAMA *>(wiki.get());
        assert(lam->lmodel!=nullptr);   
        return lam->lmodel;  
    }    
    
    struct ggml_cgraph *GetRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   override;
    std::string T2STR( const std::vector<TOKEN_ID>& tok,int flag );
    
    std::string Name()  override;    
    hGensor  tokens_input=nullptr;    
    
    NLP_AutoRegressive()   {}
    NLP_AutoRegressive( const std::string& nam_, struct CLI_params params,ROLE_TYPE role_,int flag=0x0);
    NLP_AutoRegressive(const std::string& nam_,const NLP_AutoRegressive* src,struct CLI_params params,int flag=0x0);

    virtual ~NLP_AutoRegressive() {
        free_random_normal_distribution(rnd); 
        // ggml_free(lora.ctx);        
    }
    //number of vocab at target layer
    virtual size_t tVocab();
    size_t nClass()    override     {   return tVocab(); }

 // get gensors from llama_model (possibly mmapped)
    virtual void LoadTensors(struct llama_model * lama,int flag=0x0);   

    virtual void LoadModel(const char * fn_model, int flag=0x0) {
        assert(0);  //  A more practical way:   just use llama_load_model_from_file get parameters directly from gguf file
    }

    virtual hGensor UpdateGensor (const char*name,int flag=0x0){
        return GetGensor(name);
    }

    virtual bool CreateExlogists(hWIKI wiki,uint32_t n_ctx,uint32_t n_batch,int flag=0x0);

    virtual void InitModel(int flag=0x0);
    virtual void InitGensors(int flag=0x0);
    virtual hGensor build_gate(struct ggml_context * ctx,hGensor cur,hGensor cur_logits, int flag );

    virtual size_t MostMemSize(int flag)    {
        //mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +(hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
        int n_layer = hparams.nLayer();
        int nHead = hDict!=nullptr ? hDict->nLevel*3+2+6 : 6; 
        int nMost = LLAMA_TRAIN_MAX_NODES;      //  16384
        assert(nHead*2 + n_layer*18<nMost);
        size_t sz = ggml_tensor_overhead()*2*nMost;
        size_t overhead = GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true);
        sz += (hparams.common.use_checkpointing ? 3 : 2)*overhead;
        return sz;
    }    

    //for tokens_input & target_probs
    virtual void InitInput(struct ggml_context * ctx,bool isMask,int flag=0x0);
    virtual bool InitDictTokenset(int flag=0x0);
    hGensor Input()     override    {   return tokens_input;    }

    bool Init(const vector<hWIKI>& wikis_,int flag=0x0)     override ;
    
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;

    void Dump(int type,int flag=0x0)    override  {
        int n_vocab = hDict->n_vocab,n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
        string suffix="\n========\n",prefix;
        __repr__(suffix,prefix);
        hparams.Dump();         //        print_params(&hparams)
        _INFO("====== nParams = %ld(%.6gM) ======\n", nParams,nParams/1.0e6);
        _INFO("%s: nParams=%zu model_size = %zu bytes (%.1f MB)\n", __func__, nParams,szModel,szModel / (1024.0f*1024.0f) );
        _INFO("%s: n_vocab=%d t_vocab=%d,n_batch=%d,n_ctx=%d,n_embd=%d,n_head=%d,n_rot=%d,n_ff=%d\n", __func__, 
            n_vocab,tVocab(),n_batch,n_ctx,n_embd,hparams.n_head(),hparams.n_rot,hparams.n_ff() );
        _INFO("%s: loader=%s\n", __func__, hparams.batch_sample.c_str() );
        if(hOPT!=nullptr)
            hOPT->Dump( 1 );     
        else{
            _INFO("hOPT is NULL\n");
        }   
        if(hparams.lars_ratio>0)
            _INFO("%s: LARS(t_max=%g)\n", __func__,hparams.lars_ratio);
    }
    
    bool Build(int flag=0x0)   override;
    /*void Build_v0(int flag=0x0)      {    
        _WARN("\n%s Deprecated!!!\n",__func__);    
        InitInput(flag); 
        
        auto train_params = hparams.common;
        int n_batch  = train_params.n_batch;
        ctx_compute_params.mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
            (hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
        size_t best_compute_size = SIZE_MAX;
        enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT;  //GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;  //GGML_CGRAPH_EVAL_ORDER_COUNT;
        if(graph_order==-1)   {// find best evaluation order
            for (unsigned order = 0; order < (unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order) {
                ctx_build = ggml_init(ctx_compute_params);
                ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                gf = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);
                gf->order = (enum ggml_cgraph_eval_order) order;
                if(!isLocalInfer){
                    gb = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);
                    gb_tmp = train_params.use_checkpointing ? ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true) : NULL;
                }
                BuildOperators(ctx_build,alloc,true,flag);
                size_t max_compute_size = ggml_gallocr_get_buffer_size(alloc, 0); // FIXME: this will still allocate the buffer
                if (max_compute_size < best_compute_size) {
                    best_compute_size = max_compute_size;
                    best_order = gf->order;
                }
                ggml_gallocr_free(alloc);
                ggml_free(ctx_build);
            }
        
            size_t max_compute_size = best_compute_size;
            _INFO("%s: compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
            _INFO("%s: evaluation order = %s\n", __func__,
                (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
                (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
                "invalid");
            graph_order = best_order;
        }else{
            assert(GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT<=graph_order && graph_order<=GGML_CGRAPH_EVAL_ORDER_COUNT);
            best_order = (enum ggml_cgraph_eval_order)(graph_order);
        }        

        ctx_build = ggml_init(ctx_compute_params);
        alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        gf = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);
        gf->order = best_order;
        if(!isLocalInfer){
            gb = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);
            gb_tmp = train_params.use_checkpointing ? ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true) : NULL;
        }
        
        BuildOperators(ctx_build,alloc,false,flag);
        
        Statistic(0x0);
#ifndef NDEBUG
        // ggml_graph_print(gf);           ggml_graph_print(gb);       //only for debug
#endif
    }*/

    hGensor BuildTarget( struct ggml_context * ctx,hGensor cur,int flag=0x0)   override; 
    

    virtual hBrownMotion CreateBrownMotion(hGensor wq, hGensor wk, hGensor wv,const std::shared_ptr<QKV_LAY>& layer)  {
        hBrownMotion hMotion =  (tpATT==ATTENTION_TYPE::QKV) ? 
            std::make_shared<QKV_Motion> (this,wq,wk,wv,KQ_mask,hparams,layer,0x0) :
            std::make_shared<BROWN_Motion> (this,wq,wv,hparams,layer,0x0);
        return hMotion;
    }
    // build KQ_pos & KQ_mask
    void build_inp_KQ_(struct ggml_context *ctx,bool isMask,bool causal = true);
    //n_embd_head, n_head_kv
    // virtual hGensor  build_layer_( int N,struct ggml_context *ctx_build,hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    //     hGensor att_norm.w,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down, int flag=0x0) ;
    virtual hGensor  build_layer_( int N,struct ggml_context *ctx_build,hGensor cur,std::shared_ptr<QKV_LAY> layer,hGensor KQ_pos,int flag=0x0) ;

    virtual void BuildOperators(struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,int flag=0x0)   {    
        // gf = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);    
        hForwTG = std::make_shared<TGraph>(this,"Forward",ctx_build,true);         

        auto train_params = hparams.common;     
        int n_batch  = train_params.n_batch;
        measure_only = m_only;
        ggml_set_scratch(ctx, { 0, 0, nullptr, });      
        const int n_ctx = train_params.n_ctx,n_embd = hparams.n_embd,n_layer = hparams.nLayer(),
            n_head = hparams.n_head(),n_rot = hparams.n_rot,n_ff = hparams.n_ff(),n_past=0;
 
    // build_inp_KQ_(ctx,true);      

        GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
        hGensor _tEmbed = UpdateGensor (hDict->tok_embeddings-> name);      //embedding of all tokens
        
        // hGensor _tOutput = UpdateGensor (hDict->_output.w->name);       
        hGensor  t00 = tBatch;  //n_batch==1 ? tokens_input : ggml_reshape_1d(ctx, tokens_input, n_ctx*n_batch);  //gTN(t00, "t00"); 
        assert_shape_1d(t00, n_ctx*n_batch);
        hGensor  t01 = ggml_get_rows(ctx, _tEmbed, t00);    gTN(t01, "inp_embd"); 
        hGensor  cur = t01;        
        cur = hDict->ENC(ctx,t01);      // cur=t01 if hDict->dims is empty;
        if(cur!=t01)        gTN(cur, "embed_encoder");  
        assert_shape_2d(cur, n_embd, n_ctx*n_batch);
        checkpoints.clear();
        checkpoints.push_back(tokens_input);        checkpoints.push_back(target_probs);        
        checkpoints.push_back(t00);        checkpoints.push_back(t01);        
        
        for (auto lay : layers) {
            auto layer = dynamic_pointer_cast<QKV_LAY>(lay);      
            cur = build_layer_( n_ctx,ctx, cur, layer, KQ_pos);           
            checkpoints.push_back(cur);
        }
        BuildTarget(ctx,cur,flag);       
    }

    // virtual bool LoadTokens( int flag=0x0 );

    void CopyWeight(const Fish* src,int flag = 0x0)  override;
    bool LocalFeeling(std::vector<llama_token>&tokens,vector<float>& result)   const   override;

    void Loss(int flag=0x0)     override   {

    }

    void Train(int flag=0x0)    override    ;

    void save_gguf(const char * fn_model_base, int flag)    override;
    // virtual void SaveTrain(struct save_train_files_data * data, struct train_state * train);
};
typedef shared_ptr<NLP_AutoRegressive> hLLAMA;

struct LLAMA_LORA  : public NLP_AutoRegressive {
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
    
    LLAMA_LORA( const std::string& nam_,struct CLI_params params,int flag=0x0)   {
        hparams = params;        
        hparams.n_rot   = hparams.n_embd / hparams.n_head();
        // hparams.n_ctx = hparams.common.n_ctx;
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
        int n_layer = hparams.nLayer();
        size_t sz = ggml_tensor_overhead()*2*(9 + n_layer*27);
        return sz;
    }

    void InitModel(int flag=0x0)    override    {
        _INFO("LLAMA_LORA::%s: init model\n", __func__);
        auto ctx=GetCTX();
        hDict->isLoadTokenEmbed = true;
        
        LoadTensors(GetRawModel());  
        // LoadModel( hparams.fn_model_base,0 );        
        // hparams.n_ctx = hparams.common.n_ctx;
 
        nParams = 0;
        enum ggml_type lora_type  = GGML_TYPE_F32;
        // lora_type = GGML_TYPE_COUNT;
        InitFactor(ctx, hDict->tok_embeddings,lora_type, n_rank_tok_embeddings,0x0);
        // tok_embeddings_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_tok_embeddings, n_embd);
        // tok_embeddings_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_tok_embeddings, n_vocab);
        InitFactor(ctx, hDict->_norm.w,lora_type, n_rank_norm,0x0);
        // norm_a           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_norm, n_embd);
        // norm_b           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_norm, 1);
        InitFactor(ctx, hDict->_output.w,lora_type, n_rank_output,0x0);
        // output_a         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_output, n_embd);
        // output_b         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_output, n_vocab);        
        
        for (uint32_t i = 0; i < hparams.nLayer(); ++i) {
            auto lay_ = dynamic_pointer_cast<QKV_LAY>(layers[i]);
            // auto layer = std::make_shared<lora_layer>();
            // lora_layers.push_back(layer);        //typedef shared_ptr<layer> hLayer;
            InitFactor(ctx, lay_->att_norm.w,lora_type, n_rank_attention_norm,0x0);
            // layer->attention_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_attention_norm, n_embd);
            // layer->attention_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_attention_norm, 1);
            InitFactor(ctx, lay_->Q.w,lora_type, n_rank_wq,0x0);
            // layer->wq_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wq, n_embd);
            // layer->wq_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wq, n_embd);
            InitFactor(ctx, lay_->K.w,lora_type, n_rank_wk, 0x0);
            // layer->wk_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wk, n_embd);
            // layer->wk_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wk, n_embd_gqa);
            InitFactor(ctx, lay_->V.w,lora_type, n_rank_wv, 0x0);
            // layer->wv_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wv, n_embd);
            // layer->wv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wv, n_embd_gqa);
            InitFactor(ctx, lay_->wo,lora_type, n_rank_wo, 0x0);
            // layer->wo_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wo, n_embd);
            // layer->wo_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_wo, n_embd);
            InitFactor(ctx, lay_->ffn_norm.w,lora_type, n_rank_ffn_norm, 0x0);
            // layer->ffn_norm_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_norm, n_embd);
            // layer->ffn_norm_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_norm, 1);
            InitFactor(ctx, lay_->ffn_gate,lora_type, n_rank_ffn_gate, 0x0);
            // layer->ffn_gate_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_gate, n_embd);
            // layer->ffn_gate_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_gate, n_ff);
            InitFactor(ctx, lay_->down.w,lora_type, n_rank_ffn_down, 0x0);
            // layer->ffn_down_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_down, n_ff);
            // layer->ffn_down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_down, n_embd);
            InitFactor(ctx, lay_->up.w,lora_type, n_rank_ffn_up, 0x0);
            // layer->ffn_up_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_up,   n_embd);
            // layer->ffn_up_b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rank_ffn_up,   n_ff);
        }        
        // allocate data for lora_tensors
        // hEDS->Init(this,ctx);   //back_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        assert(updateTMap==false);
        // size_t sz1=ggml_backend_buffer_get_size(data);      //130872416
        ggml_backend_buffer_type_t buf0 = ggml_backend_cpu_buffer_type(); 
        
        ggml_backend_buffer_type_t buf1 = ggml_backend_cpu_buffer_type();
        // data = ggml_backend_alloc_ctx_tensors_from_buft(ctx,buf1 );
        size_t sz2=hEDS->sz;  // ggml_backend_buffer_get_size(back_data);      //130872416
        if (!hparams.only_write_model) {    //only_write_lora
            hOPT->Prepare(nParams);
            // ggml_opt_init( hOPT->opt->ctx, hOPT->opt, hOPT->opt->params, nParams);     //nx=16358481
        }
        szModel = ggml_used_mem(ctx) + sz2; //ggml_backend_buffer_get_size(back_data);  //580800+65436224

        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }

    void Dump(int type,int flag=0x0)    override   {
        NLP_AutoRegressive::Dump(type);
        _INFO("%s:%s model=%zu bytes (%.1f MB)\tn_embd=%d n_vocab=%d n_ff=%d n_rank_wq=%d\n", "LLAMA_LORA", hparams.sTune(),szModel, (float) (szModel) / (1024.0f*1024.0f),
            hparams.n_embd,hDict->n_vocab,hparams.n_ff(),n_rank_wq);
    }


    CLI_params::TUNE_ALG TuneOnShape(int nIn,int nOut){
        CLI_params::TUNE_ALG tune = hparams.tune;
        if( nIn==1 || nOut==1 || nIn*nOut>5120*5120*4 ) {
            tune = CLI_params::LORA;
        }
        return tune;
    }
    hGensor UpdateGensor( const char*name,int flag=0x0 ) override    {
        hGensor gensor = NLP_AutoRegressive::UpdateGensor(name);       assert(gensor!=nullptr); 
        // if(ggml_is_quantized(gensor->type))    
        //     ggml_tensor_dequant(ctx_build,gensor,GGML_TYPE_F32); 

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
            delta = ggml_mul_mat(ctx_build, ga, gb);  //gensor = add_to_f32(ctx_build, gensor, ggml_mul_mat(ctx_build, ga, gb));
        break;
        case CLI_params::OFF:   //would fail because gensor has no grad!!! gensor fixed as load from model file
            assert(0);
            break;
        case CLI_params::LORA_AB:       //9.784955
        //,nHeavy = ga->ne[0], rank = nHeavy
            if(!measure_only)
                Gensor_loab(ctx_build,gensor,ga->ne[0],ga,gb);
            delta = ggml_mul_mat(ctx_build, ga, gb);  //gensor = add_to_f32(ctx_build, gensor, ggml_mul_mat(ctx_build, ga, gb));
            break;
        case CLI_params::LORA_SVD:  {       //9.784960
                if(!measure_only)
                    Gensor_SVD(ctx_build,gensor,d->ne[0],u,d,v);
                // hGensor dv = ggml_mul_mat(ctx_build, ggml_diag(ctx_build,d),v);  //GGML_OP_DIAG don't support backward!!!
                hGensor dv = ggml_mul_mat(ctx_build, d,v);
                delta = ggml_mul_mat(ctx_build, u, dv);  //gensor = add_to_f32(ctx_build, gensor, ggml_mul_mat(ctx_build, u, dv));
            }
            break;
        default:
            assert(0);
            break;
        }       
        if(hDistler!=nullptr) 
            gensor = hDistler->UpdateGG(ctx_build,gensor,delta);
        else
            gensor = add_to_f32(ctx_build, gensor, delta);
        
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
        }
    }    

    virtual void BuildOperators(struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,int flag=0x0)     {   
        auto train_params = hparams.common;
        int n_batch  = train_params.n_batch;
        measure_only = m_only;
        ggml_set_scratch(ctx, { 0, 0, nullptr, });
        const int n_past = 0;
        const int n_ctx = train_params.n_ctx;    //n_tokens;
        const int n_embd      = hparams.n_embd;
        const int n_layer     = hparams.nLayer();
        const int n_head      = hparams.n_head();
        const int n_head_kv   = hparams.n_head_kv();
        const int n_ff        = hparams.n_ff();
        const int n_rot       = hparams.n_embd_head();
        const int n_embd_head = hparams.n_embd_head();
        const int n_embd_gqa  = hparams.n_embd_gqa();
        const float rms_norm_eps    = hparams.f_norm_rms_eps;
        const float rope_freq_base  = hparams.rope_freq_base;
        const float rope_freq_scale = hparams.rope_freq_scale;
        GGML_ASSERT((size_t) n_layer == layers.size());
       
        // KQ_pos - contains the positions
        hGensor  KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ctx);
        ggml_set_input(KQ_pos);
        struct ggml_context   * ctx0= ctx;
        gTN(tokens_input, "inp_tokens");
        gTN(target_probs,      "targets");
        auto rope = [ctx, KQ_pos, n_rot, n_ctx, rope_freq_base, rope_freq_scale]
                (struct ggml_tensor * t) -> struct ggml_tensor * {
            // not capturing these, to silcence warnings
            const int rope_mode = 0;
            assert(0);  // CYS_0826
            return ggml_rope_custom(ctx,
                t, KQ_pos, n_rot, rope_mode, n_ctx, 0,
                rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f
            );
        };

        GGML_ASSERT(tokens_input->type == GGML_TYPE_I32);
        NLP_AutoRegressive *base = dynamic_cast<NLP_AutoRegressive *>(this);
    //    hGensor  _tEmbed = add_to_f32(ctx, base->tok_embeddings, ggml_mul_mat(ctx, tok_embeddings_a, tok_embeddings_b));
    //    hGensor  _tNorm           = add_to_f32(ctx, base->norm, ggml_mul_mat(ctx, norm_a, norm_b));
    //    hGensor  _tOutput         = add_to_f32(ctx, base->output, ggml_mul_mat(ctx, output_a, output_b));
        hGensor _tEmbed = base->hDict->tok_embeddings,_tNorm = base->hDict->_norm.w,_tOutput = base->hDict->_output.w; 
        _tEmbed = UpdateGensor (base->hDict->tok_embeddings->name); 
        // _tNorm = UpdateGensor (base->hDict->_norm.w->name); 
        // _tOutput = UpdateGensor (base->hDict->_output.w->name);         

        hGensor  t00 = tBatch;  //ggml_reshape_1d(ctx, tokens_input, n_ctx*n_batch);  gTN(t00, "t00"); assert_shape_1d(t00, n_ctx*n_batch);
        hGensor  t01 = ggml_get_rows(ctx, _tEmbed, t00);        gTN(t01, "t01"); assert_shape_2d(t01, n_embd, n_ctx*n_batch);
        hGensor  cur = t01;
        checkpoints.clear();
        // std::vector<struct ggml_tensor *> checkpoints;
        if (train_params.use_checkpointing) {
            checkpoints.push_back(tokens_input);    checkpoints.push_back(target_probs);
            checkpoints.push_back(t00);             checkpoints.push_back(t01);
        }

        const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        for (int il = 0; il < n_layer; ++il) {
            auto layer = dynamic_pointer_cast<QKV_LAY>(layers[il]);
            // auto llayer = dynamic_pointer_cast<lora_layer>(lora_layers[il]); 
            hGensor a_norm = UpdateGensor (layer->att_norm.w->name);   
            hGensor wq = n_rank_wq ==0 ? nullptr : UpdateGensor (layer->Q.w->name);                     
            hGensor wk = n_rank_wk ==0 ? nullptr : UpdateGensor (layer->K.w->name);
            hGensor wv = n_rank_wv ==0 ? nullptr : UpdateGensor (layer->V.w->name);
            hGensor wo = UpdateGensor (layer->wo->name);
            hGensor ffn_norm = UpdateGensor (layer->ffn_norm.w->name);
            hGensor ffn_up = UpdateGensor (layer->up.w->name);
            hGensor ffn_gate = UpdateGensor (layer->ffn_gate->name);
            hGensor ffn_down = UpdateGensor (layer->down.w->name);
            cur = build_layer_( n_ctx,ctx,cur, layer, KQ_pos);
            if(wk==nullptr && wv==nullptr){ //markov

            }
            
            if (train_params.use_checkpointing) {
                checkpoints.push_back(cur);
            }     
        }

        BuildTarget(ctx,cur);
    }

};

/**
 *  llama-2-13b.Q2_K.gguf           wq,wk,wv has same shape
 *  Meta-Llama-3-8B.Q2_K.gguf       wq,wk,wv has different shape!
*/
struct LLAMA_Brown  : public LLAMA_LORA {
    LLAMA_Brown( const std::string& nam_,struct CLI_params params,int flag=0x0) 
        : LLAMA_LORA(nam_,params,flag)  {
        n_rank_wq = 4;
        n_rank_wk = 0;
        n_rank_wv = 0;
    }

    virtual ~LLAMA_Brown() {
        // free_random_normal_distribution(rnd);
    }  

    hBrownMotion CreateBrownMotion(hGensor wq, hGensor wk, hGensor wv,const std::shared_ptr<QKV_LAY>& layer)  override    {
        hBrownMotion hMotion = std::make_shared<BROWN_Motion> (this,wq,wv,hparams,layer,0x0);
        return hMotion;
    }
};

struct LLAMA_VAE  : public NLP_AutoRegressive {
    
    int lama_embed = 0,latent_dim = 192;

    LLAMA_VAE( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0) 
        : NLP_AutoRegressive(nam_,params,role,flag)  {
        isLoadTokenEmbed = true;
        // hparams.common.adam_alpha = 0.0001;     // 
    }

    virtual ~LLAMA_VAE() {        
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("LLAMA_VAE%s: init model\n", __func__);

        gensors.clear();
        hDict->InitVAE();       updateTMap = true;

        NLP_AutoRegressive::InitModel(flag);        
    }     
   
};

struct LLM_MAMBA : public NLP_AutoRegressive {
    int32_t n_tokens;
    int32_t n_kv;     // size of KV cache to consider (n_kv <= kv_self.size)
    int32_t n_outputs;
    int32_t n_outputs_enc;
    int32_t kv_head;  // index of where we store new KV data in the cache
    int32_t n_ctx_orig;
 
    LLM_MAMBA( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~LLM_MAMBA() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("MAMBA::%s: init model\n", __func__);

        /*llama2params(GetRawModel(),hparams);
        hparams.n_head() = 8;    
        hparams.n_head_kv = hparams.n_head();     //hack
        hparams.n_layer = nLayerX;

        gensors.clear();
        hDict->InitVAE();       updateTMap = true;*/

        NLP_AutoRegressive::InitModel(flag);        
    }     

    hGensor build_layer_( int N,struct ggml_context *ctx_build,hGensor cur,std::shared_ptr<QKV_LAY> layer,hGensor KQ_pos,int flag=0x0) override;
    hGensor BuildTarget( struct ggml_context * ctx,hGensor cur,int flag=0x0) override; 
};


class GPT2 : public NLP_AutoRegressive {
protected:    
    LayerNormal _norm;
    SLP _output;    
    
    int cRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   ;//override;
public:
    GPT2( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~GPT2() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("GPT2_model::%s: init model\n", __func__);

        // NLP_AutoRegressive::InitModel(flag);         
    }     
    
    struct ggml_cgraph *GetRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   override;
    void InitGensors(int flag=0x0) override    {;}

    // hGensor BuildTarget(struct ggml_context * ctx,hGensor cur,int flag=0x0) override; 
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};

struct TinyLama : public NLP_AutoRegressive { 
};

struct StableLM : public NLP_AutoRegressive { 
};

struct LLM_MOE : public NLP_AutoRegressive { 
    uint32_t n_expert,n_expert_used;
    LLM_MOE( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~LLM_MOE() {          
    }  

    size_t MostMemSize(int flag)  override;

    void InitModel(int flag=0x0)    override    {
        _INFO("LLM_MOE::%s: init model\n", __func__);
       
        n_expert = hparams.n_expert,n_expert_used = hparams.n_expert_used;
        assert( n_expert_used <= n_expert && n_expert <= 160 );      //160:    DeepSeekV2

        NLP_AutoRegressive::InitModel(flag);        
    }     

    hGensor build_layer_( int N,struct ggml_context *ctx_build,hGensor cur,std::shared_ptr<QKV_LAY> layer,hGensor KQ_pos,int flag=0x0) override;
        // hGensor build_gate(struct ggml_context * ctx,hGensor cur,hGensor cur_logits, int flag )    override;
};


