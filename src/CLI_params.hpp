/**
 *  Copyright 2023-2025 by Grusoft  
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once
#include <cassert>
// #include "train.h" 
#include "./ggex/json.hpp" 

// typedef int32_t TOKEN_ID;
typedef uint16_t TOKEN_ID;
/*
    10 levels of dumps, 0-9. 0 is a full dump,The lower the number the more dump.
*/  
extern int g_dump_level;
inline bool NOT_DUMP(int t=0) {
    if(g_dump_level<=t)    return false;
    return true;
}
inline bool DUMP(int t=0) {
    return !NOT_DUMP(t);
}
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
    NLP_GPT2,       NLP_GPT2_char,
    NLP_LLAMA,
    NLP_MAMBA,
    NLP_MOE,    //???
    
    SCORE_,
    SAM_
};

struct LAY_PARAM{
    uint32_t head,head_kv,ff;
    LAY_PARAM(uint32_t h,uint32_t k,uint32_t f) : head(h),head_kv(k),ff(f) {
        assert(h>0 && k>0 && f>0);
    }

    uint32_t n_head()       const           {   return head;    }
    virtual void SetHead(int nH)    {
        assert(nH>0 && nH<1024*1024);       head=nH;        head_kv=nH;
    }
    virtual void SetFF(int nF)    {
        assert(nF>0 && nF<1024*1024);       ff=nF;        
    }
    uint32_t n_head_kv()    const           {   return head_kv;    }
    uint32_t n_ff()         const           {   return ff;    }    

    /**
     * The GQA model efficiently breaks the query into n_heads, and the key and value are divided into n_kv_heads groups, 
     * enabling multiple key-value heads to share the same query.
    */
    uint32_t n_gqa() const {
        assert(head>=head_kv && head%head_kv==0);
        return head/head_kv;
    }
    uint32_t n_embd_head(int n_embd) const {
        assert(n_embd>0 && n_embd%head==0 && n_embd>=head);
        return n_embd/head;
    }

    uint32_t n_embd_gqa(int n_embd) const {
        int gqa = n_gqa();
        assert(n_embd%gqa==0 && n_embd>=gqa);
        return n_embd/gqa;
    }
};

struct train_params_ {
    const char * fn_train_data;
    const char * fn_checkpoint_in;
    const char * fn_checkpoint_out;
    const char * pattern_fn_it;
    const char * fn_latest;

    bool print_usage;

    int save_every;

    uint32_t seed;

    int n_ctx;
    int n_threads;
    int n_batch;
    int n_gradient_accumulation;
    int n_epochs;
    int n_gpu_layers;

    bool custom_n_ctx;

    bool use_flash;
    bool use_checkpointing;

    std::string sample_start;
    bool include_sample_start;
    bool escape;
    bool overlapping_samples;
    bool fill_with_next_samples;
    bool separate_with_eos;
    bool separate_with_bos;
    bool sample_random_offsets;

    bool force_reshuffle;

    int   warmup;
    int   cos_decay_steps;
    float cos_decay_restart;
    float cos_decay_min;
    bool  enable_restart;

    int   opt_past;
    float opt_delta;
    int   opt_max_no_improvement;

    int   adam_n_iter;
    float adam_alpha;
    float adam_min_alpha;
    float adam_decay;
    int   adam_decay_min_ndim;
    float adam_beta1;
    float adam_beta2;
    float adam_gclip;
    float adam_eps_f;
};

struct CLI_params {
    struct train_params_ common;  
    struct SaveModel {
        std::string checkpoint_in,checkpoint_out;
        std::string model_out,model_base;       
    };    
    SaveModel save;

    struct DEUG_SWITCH{
        int SelfAttention_noraml=1;
        bool NO_loss = false;
        int dict_latent_dim = -1;
        int graph_dump = 0; //  10 levels of dumps, 0-9. 0 is a full dump,The lower the number the more dump.
    };
    DEUG_SWITCH debug;
    //Always false,     GGML don't support back of FLASH_ATTEN !
    bool isFlashAtten()     {   
        common.use_flash=false;  return common.use_flash;  
    }
    uint32_t nThread()  const;
    
    uint32_t nLayer()   {
        if(nLayerX>0)   return nLayerX;
        assert(n_layer_train>0);
        return n_layer_train;
    }
    uint32_t n_ctx()    const    {  return common.n_ctx;  }             //number of tokens in each sample
    uint32_t n_ctx_orig()    const    {
        return n_ctx_orig_yarn != 0 ? n_ctx_orig_yarn : n_ctx_train;
    }
    void SetNCTX(int _nctx)  {
        assert(_nctx>0 && _nctx<1024*1024);
        common.n_ctx = _nctx;
    }
    uint32_t n_batch()  const    {  return common.n_batch;}             //number of samps in each batch
    uint32_t nTokenInBatch()  const    {  return common.n_batch*common.n_ctx;}
    uint32_t n_seq_max,n_ctx_orig_yarn,n_ctx_train=-1;
    bool isLongRope(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        const auto n_ctx_pre_seq = n_ctx() / n_seq_max;
        bool isLong = n_ctx_pre_seq > n_ctx_orig_yarn;
        return isLong;
    }

    JSON jConfig;
    nlohmann::ordered_json jModel;
    MODEL_ARCH ModelArch();
    virtual void OnArch();
    virtual void JModel2Params(int flag);

    std::string exec_name="",test="",compute_graph="";
    std::vector<std::string> fn_model_base;
    //  std::string fn_vocab_model;
    std::string model_title="";
    // std::string fp_train_data;   serial_path
    std::string train="";  //"scratch"

    bool isOnlyGPT = false;
    bool passLoadToken = false;
    bool only_write_model = false;
    bool ffn_use_gate = false;
    // uint32_t n_vocab = 0;
    // uint32_t n_ctx   = 0;
    uint32_t n_swarm = 1;
    // uint32_t n_outputs = 1;
    int n_embd = -1, n_embd_head_k = -1, n_embd_head_v = -1; 
    std::vector<LAY_PARAM> layerps;

    int n_layer_train = -1, nLayerX = -1, nFFX = -1;
    
    uint32_t n_rot = 64;
        
    int nabla = 1;      //cys
    // std::string sigma = ""; 
    std::string vae = "";
    std::string prompt = "";
    std::string dict_vae_dims = "",dict_dialect="",dict_logits="";
    


    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim = 0;
    uint32_t time_decay_extra_dim = 0;
    uint32_t wkv_head_size = 0;

    int eval_every=-1,gpt_every=-1;
    // MOE
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms = false;

    template<typename T>
    bool is(const std::vector<std::string>&keys,const T& t){
        T v0;  
        bool isDump = !NOT_DUMP(0);
        T val = jKV(jConfig,keys,v0,isDump);
        // return jKV_is(jConfig,keys,target);
        return val==t;
    }
    bool is(const std::vector<std::string>&keys,const char* t){
        return is(keys,std::string(t));
    }

    template<typename T>
    T Get(const std::vector<std::string>&keys,const T& t,bool isCLI=true)   const   {
        T val = jKV(jConfig,keys,t,isCLI);
        return val;
    }
    std::string KV(const std::vector<std::string>&keys,const std::string& t="",bool isCLI=true) const   {
        std::string val = Get(keys,t);
        return val;
    }

    // It is also not easy to change keys in a std::map
    template<typename T>
    T Set(const std::vector<std::string>&keys,const T& t){
        //  https://github.com/nlohmann/json/issues/1723
        assert(0);
        T v0;   
        T val = jKV(jConfig,keys,v0,false);
        return val;
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
    std::string batch_sample;
    
    float f_norm_eps = 1e-5f; 
    float f_norm_rms_eps = 1e-5f; 
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

    uint32_t FOMULA_n_ff(int n_mult) {
        const uint32_t n_ff = ((2*(4*n_embd)/3 + n_mult - 1)/n_mult)*n_mult;
        return n_ff;
    }
    void SetHead(uint32_t nH){
        for(int il=0;il<layerps.size();il++){
            assert(layerps[il].head==layerps[il].head_kv);
            layerps[il].head = nH;
            layerps[il].head_kv = nH;
        }
            
    }
    uint32_t n_head(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        return layerps[il].n_head();        
    }
    uint32_t n_head_kv(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        return layerps[il].n_head_kv();
    }

    uint32_t n_ff(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        if(nFFX<=0)
            return layerps[il].n_ff();        
        else
            return nFFX;
    }
    uint32_t n_embd_head(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        return layerps[il].n_embd_head(n_embd);        
    }
    uint32_t n_embd_gqa(uint32_t il = 0) const {
        assert(il>=0 && il<layerps.size());
        return layerps[il].n_embd_gqa(n_embd);        
    }
    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);
        if (n_head_kv == 0) {
            return 0;
        }
        return n_head/n_head_kv;
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_v * n_head_kv;
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        // corresponds to Mamba's conv_states size or RWKV's token_shift states size
        if (wkv_head_size != 0) {
            // for RWKV models
            return 2 * n_embd;
        } else {
            // TODO: maybe support other convolution strides than 1
            // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
            return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
        }
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (wkv_head_size != 0) {
            // corresponds to RWKV's wkv_states size
            return n_embd * wkv_head_size;
        } else {
            // corresponds to Mamba's ssm_states size
            return ssm_d_state * ssm_d_inner;
        }
    }

    bool operator!=(const CLI_params& other) const; 

    void Dump( );

    bool parse(int argc, char ** argv);
    virtual bool InitJConfig(const std::string&jPath,int flag=0x0);
    std::string GetDataPath(const std::string type,int flag=0x0);
    //static bool train_params_parse(int argc, char ** argv, struct CLI_params * params)
};