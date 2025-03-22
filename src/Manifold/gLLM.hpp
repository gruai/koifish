/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  General LLM model  
 * 
 *  \brief NLP_AutoRegressive Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#pragma once
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
// #include "../ggex/common-ggml.h" 
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Fish.hpp"   
#include "../Manifold/VAE.hpp" 
#include "llama.h"
#include "llama_cys.h"
#include "../Manifold/Dictionary.hpp"

#define GGML_OBJECT_MAX_SIZE    128

//  GGML_DEFAULT_GRAPH_SIZE/*2048*/

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

   
// void static set_name(hGensor  t, const char * n) {
//     ggml_set_name(t, n);
//     if (t->grad) {
//         ggml_format_name(t->grad, "%s->grad", n);
//     }
// };


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
    
    hCDICT hDictVAE=nullptr; 
    hTokenizer hDict = nullptr;
    
    bool isAttOnBC=false;   
        
   
    CHILD_0909_WIKIS
    //Assume wikis[0] is the backbone of FISH
    struct LAMA *lama(int id=0)    {  
        /*if(wikis.size()==0) 
            return nullptr;
        assert(id>=0 && id<wikis.size());
        LAMA *lama = dynamic_cast<LAMA *>(wikis[id].get());
        assert(lama!=nullptr);
        return lama; */
        return nullptr; 
    }

    struct llama_model *GetRawModel( )    {   
        // LAMA *lam = lama( );    //dynamic_cast<LAMA *>(wiki.get());
        // assert(lam->lmodel!=nullptr);   
        // return lam->lmodel;  
        return nullptr; 
    }    
    
    struct ggml_cgraph *BuildRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   override;
    int GenSentence(int flag=0x0)  override;
    std::string T2STR( const std::vector<TOKEN_ID>& tok,int flag=0x0);
    std::string T2STR( TOKEN_ID tok,int flag=0x0)       { 
        assert(0);  return "";
    }
    
    std::string Name()  override;    
    hGensor  tokens_input=nullptr;    
    
    NLP_AutoRegressive()   {}
    NLP_AutoRegressive( const std::string& nam_, struct CLI_params params,ROLE_TYPE role_,int flag=0x0);
    NLP_AutoRegressive(const std::string& nam_,const NLP_AutoRegressive* src,struct CLI_params params,int flag=0x0);

    virtual ~NLP_AutoRegressive() {
        free(rnd);
        // free_random_normal_distribution(rnd); 
        // ggml_free(lora.ctx);        
    }
    //number of vocab at target layer
    virtual size_t tVocab();
    size_t nClass()    override     {   return tVocab(); }

 // get gg_tensors from llama_model (possibly mmapped)
    // virtual void LoadTensors(struct llama_model * lama,int flag=0x0);   

    virtual void LoadModel(const char * fn_model, int flag=0x0) {
        assert(0);  //  A more practical way:   just use llama_load_model_from_file get parameters directly from gguf file
    }

    virtual hGensor UpdateGensor (const string&name,int flag=0x0){
        return GetGensor(name);
    }

    virtual bool CreateExlogists(hWIKI wiki,uint32_t n_ctx,uint32_t n_batch,int flag=0x0);

    virtual void InitModel(int flag=0x0);
    virtual void InitGensors(int flag=0x0);
    virtual hGensor build_gate(struct ggml_context * ctx,hGensor cur,hGensor cur_logits, int flag );

    size_t MostMemSize(int flag)  override  {
        //mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +(config.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
        int n_layer = config.nLayer();
        int nHead = hDictVAE!=nullptr ? hDictVAE->nLevel*3+2+6 : 6; 
        int nMost = LLAMA_TRAIN_MAX_NODES;      //  16384
        assert(nHead*2 + n_layer*18<nMost);
        size_t sz = ggml_tensor_overhead()*2*nMost;
        size_t overhead = GGML_OBJECT_MAX_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true);
        sz += (config.common.use_checkpointing ? 3 : 2)*overhead;
        return sz;
    }    

    //for tokens_input & target_probs
    bool InitInput(struct ggml_context * ctx,bool isMask,int flag=0x0)  override;
    virtual bool InitDictTokenset(int flag=0x0);
    hGensor Input()     override    {   return tokens_input;    }

    bool Init(const vector<hWIKI>& wikis_,int flag=0x0)     override ;
    
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;

    void Dump(int type,int flag=0x0)    override;
    
    // bool Build(int flag=0x0)   override;    

    hGensor BuildTarget( struct ggml_context * ctx,hGensor cur,int flag=0x0)   override; 
    

    // virtual hBrownMotion CreateBrownMotion(hGensor wq, hGensor wk, hGensor wv,const std::shared_ptr<QKV_LAY>& layer)  {
    //     hBrownMotion hMotion =  (tpATT==ATTENTION_TYPE::QKV) ? 
    //         std::make_shared<QKV_Motion> (this,wq,wk,wv,KQ_mask,config,layer,0x0) :
    //         std::make_shared<BROWN_Motion> (this,wq,wv,config,layer,0x0);
    //     return hMotion;
    // }
    // build KQ_pos & KQ_mask
    void build_inp_KQ_(struct ggml_context *ctx,bool isMask,bool causal = true);
    // Deprecated!!!
    virtual hGensor  build_layer_( int N,struct ggml_context *ctx_build,hGensor cur,std::shared_ptr<QKV_LAY> layer,hGensor KQ_pos,int flag=0x0) {
        return nullptr;
    };

    bool BuildOperators(struct ggml_context * ctx,ggml_gallocr_t alloc,bool m_only,int flag=0x0)   override {    
        // gf = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);    
        hForwTG = std::make_shared<TGraph>(this,"Forward",ctx_build,true);         

        auto train_params = config.common;     
        int n_batch  = train_params.n_batch;
        measure_only = m_only;     
        const int n_ctx = train_params.n_ctx,n_embd = config.nEmbed(),n_layer = config.nLayer();
 
    // build_inp_KQ_(ctx,true);      

        assert(tokens_input->type == typNUMBER::I32);
        hGensor _tEmbed = UpdateGensor (hDictVAE->tok_embeddings->name);      //embedding of all tokens
        
        // hGensor _tOutput = UpdateGensor (hDictVAE->_output.w->name);       
        hGensor  t00 = nullptr; //tBatch;  
        assert_shape_1d(t00, n_ctx*n_batch);
        hGensor  t01 = nullptr; //ggml_get_rows(ctx, _tEmbed, t00);    gTN(t01, "inp_embd"); 
        hGensor  cur = t01;        
        cur = hDictVAE->ENC(ctx,t01);      // cur=t01 if hDictVAE->dims is empty;
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
        return true;
    }

    // virtual bool LoadTokens( int flag=0x0 );
    // void CopyWeight(const Fish* src,int flag = 0x0)  override;
    bool LocalFeeling(hSampLoader hLoader,vector<float>& result,int flag)   override;

    void Loss(int flag=0x0)     override   {

    }

    void Train(int flag=0x0)    override    ;
};
typedef shared_ptr<NLP_AutoRegressive> hLLAMA;

struct LLAMA_VAE  : public NLP_AutoRegressive {
    
    int lama_embed = 0,latent_dim = 192;

    LLAMA_VAE( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0) 
        : NLP_AutoRegressive(nam_,params,role,flag)  {
        isLoadTokenEmbed = true;
        // config.common.adam.alpha = 0.0001;     // 
    }

    virtual ~LLAMA_VAE() {        
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("LLAMA_VAE%s: init model\n", __func__);

        gensors.Clear();
        hDictVAE->InitVAE();       updateTMap = true;

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
        NLP_AutoRegressive::InitModel(flag);        
    }     

    hGensor BuildTarget( struct ggml_context * ctx,hGensor cur,int flag=0x0) override; 
};


class GPT2 : public NLP_AutoRegressive {
protected:    
    LayerNormal _norm;
    SLP _output;    
#ifdef _TENSOR_G_    
#else
    int cRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   ;//override;
#endif
public:
    GPT2( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~GPT2() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("GPT2_model::%s: init model\n", __func__);

        NLP_AutoRegressive::InitModel(flag);         
    }     
    
    struct ggml_cgraph *BuildRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)   override;
    void InitGensors(int flag=0x0) override    {;}

    // hGensor BuildTarget(struct ggml_context * ctx,hGensor cur,int flag=0x0) override; 
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};

class DeepSeek : public NLP_AutoRegressive {
protected:    
    virtual void _forward_cpu(int token, int pos, int flag=0x0);

public:
    DeepSeek( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~DeepSeek() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("DeepSeek::%s: init model\n", __func__);

        NLP_AutoRegressive::InitModel(flag);         
    }     

    // hGensor BuildTarget(struct ggml_context * ctx,hGensor cur,int flag=0x0) override; 
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};

class Mistral : public NLP_AutoRegressive {
protected:    

public:
    Mistral( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~Mistral() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("Mistral::%s: init model\n", __func__);

        NLP_AutoRegressive::InitModel(flag);         
    }  
};

class QWen : public NLP_AutoRegressive {
protected:    

public:
    QWen( const std::string& nam_,struct CLI_params params,ROLE_TYPE role,int flag=0x0);

    virtual ~QWen() {          
    }  

    void InitModel(int flag=0x0)    override    {
        _INFO("QWen::%s: init model\n", __func__);

        NLP_AutoRegressive::InitModel(flag);         
    }  
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
       
        n_expert = config.n_expert,n_expert_used = config.n_expert_used;
        assert( n_expert_used <= n_expert && n_expert <= 160 );      //160:    DeepSeekV2

        NLP_AutoRegressive::InitModel(flag);        
    }     
    
};


