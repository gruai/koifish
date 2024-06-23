/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once
#include <cassert>
#include "../LLAMA/common/train.h" 
#include "./ggex/json.hpp" 
/**
 *  All paramters defined here
*/
enum COMPRESSIVE_SENSING    {
    SKIP,
    SVD,
    SVD_a,
    GBTQ,
};

struct CLI_params {
    struct train_params_common common;
    JSON jConfig;

    std::string exec_name="";
    std::string fn_model_base="",fn_model_out="";
    std::string train="";  //"scratch"
    bool only_write_model = false;
    uint32_t n_vocab = 32000;
    uint32_t n_ctx   = 512;
    uint32_t n_embd  = 4096;
    uint32_t n_head  = 32;          
    uint32_t n_head_kv  = 32;
    uint32_t n_layer = 32;
    uint32_t n_rot   = 64;
    uint32_t n_ff    = 11008;    
    int nabla = 1;      //cys
    std::string sigma = "";
    bool only_infer = false;

    enum TUNE_ALG   {
        OFF=0,
        LORA,
        LORA_SVD,
        LORA_AB,
        LORA_Q,
        // VARIATIONAL,
    };
    enum TUNE_ALG tune;
    std::string sTune(int flag=0x0){
        std::string tune_desc[]={
            "","_AB","_SVD","_SVD_AB","_VARIATIONAL",
        };
        return tune_desc[tune];
    }

    // float f_norm_eps     = 1e-5f; // falcon
    float f_norm_rms_eps = 1e-5f; // llama
    float rope_freq_base  = 10000.0f;
    float rope_freq_scale = 1.0f;
    float lars_ratio = 0.0f;
    float ZMUV_ratio = 0;            //Default is .01;  0.1 is too big!

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
    virtual void InitJConfig(const std::string&jPath,int flag=0x0);
    //static bool train_params_parse(int argc, char ** argv, struct CLI_params * params)
};