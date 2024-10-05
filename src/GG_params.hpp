/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once
#include <cassert>
#include "../common/train.h" 
#include "./ggex/json.hpp" 

typedef int32_t TOKEN_ID;

/**
 *  All paramters defined here
*/
enum COMPRESSIVE_SENSING    {
    SKIP,
    SVD,
    SVD_a,
    GBTQ,
};

enum MODEL_ARCH {
    _X_,
    NLP_LLAMA,
    NLP_MAMBA,
    NLP_MOE,    //???
    SCORE_,
    SAM_
};

struct CLI_params {
    struct train_params_common common;      
    uint32_t n_ctx()    const    {  return common.n_ctx;  }             //number of tokens in each sample
    uint32_t n_batch()  const    {  return common.n_batch;}             //number of samps in each batch
    uint32_t nTokenInBatch()  const    {  return common.n_batch*common.n_ctx;}

    JSON jConfig;
    MODEL_ARCH arch = MODEL_ARCH::_X_;

    std::string exec_name="",test="",compute_graph="";
    std::vector<std::string> fn_model_base;
    //  std::string fn_vocab_model;
    std::string fn_model_out="",model_title="";
    std::string fp_train_data;
    std::string train="";  //"scratch"
    bool only_write_model = false;
    // uint32_t n_vocab = 0;
    // uint32_t n_ctx   = 0;
    uint32_t n_swarm = 1;
    uint32_t n_embd  = -1;  //4096;
    uint32_t n_head  = 32;          
    uint32_t n_head_kv  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    uint32_t n_ff    = 11008;    
    int nabla = 1;      //cys
    std::string sigma = ""; 
    std::string vae = "";
    std::string prompt = "";
    std::string dict_vae_dims = "",dict_dialect="",dict_logits="";
    int dict_latent_dim = 256;

    std::vector<uint32_t> n_head_arr;
    std::vector<uint32_t> n_head_kv_arr;
    std::vector<uint32_t> n_ff_arr;

    int eval_every=-1,gpt_every=-1;
    // MOE
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms = false;
    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }

    template<typename T>
    bool is(const std::vector<std::string>&keys,const T& t){
        T v0;   
        T val = jKV(jConfig,keys,v0);
        return val==t;
    }

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;
    float f_attn_logit_softcapping = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    //parameters of wiki
    std::string wiki_actor = "",wiki_logits="";

    // bool only_infer = false;

    enum TUNE_ALG   {
        OFF=0,
        LORA,
        LORA_SVD,
        LORA_AB,
        LORA_Q,
        // VARIATIONAL,
    };
    enum TUNE_ALG tune=OFF;
    std::string sTune(int flag=0x0){
        std::string tune_desc[]={
            "","_AB","_SVD","_SVD_AB","_VARIATIONAL",
        };
        return tune_desc[tune];
    }

     //parameters of datasets
    float rSplit = 0.1;
    std::string serial_path,batch_sample;
    
    // float f_norm_eps     = 1e-5f; // falcon
    float f_norm_rms_eps = 1e-5f; // llama
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
    float lars_ratio = 0.0f;
    std::string tpWiki = "";
    //ffn_norm = nullptr if>0 
    float ZMUV_ratio = 0.0;   

    int32_t lora_r=0,lora_alpha=0;
    /*enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE = -1,
        LLAMA_ROPE_TYPE_NORM =  0,  
        LLAMA_ROPE_TYPE_NEOX =  2,
        LLAMA_ROPE_TYPE_GLM  =  4,
    };*/
    int         rope_type               = -1 ;
    /**
     * The GQA model efficiently breaks the query into n_heads, and the key and value are divided into n_kv_heads groups, 
     * enabling multiple key-value heads to share the same query.
    */
    uint32_t n_gqa() const {
        assert(n_head>=n_head_kv);
        return n_head/n_head_kv;
    }

    uint32_t n_embd_head() const {
        return n_embd/n_head;
    }

    uint32_t n_embd_gqa() const {
        return n_embd/n_gqa();
    }

    bool operator!=(const CLI_params& other) const; 

    void Dump( );

    bool parse(int argc, char ** argv);
    virtual bool InitJConfig(const std::string&jPath,int flag=0x0);
    //static bool train_params_parse(int argc, char ** argv, struct CLI_params * params)
};