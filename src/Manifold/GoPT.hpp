/**
 *  Copyright 2023-2025 by Grusoft  
 * 
 *  \brief Generate some nonsense on Prompt
 *  \author Yingshi Chen
 */
#pragma once

#include <cassert>
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h> 
#include <regex>
#include <stack>
using namespace std;
#include "common.h"
#include "GG_util.hpp"   
#include "DataLoader.hpp" 
#include "WIKI.hpp"
class Fish;


/*
    Ref     https://github.com/karpathy/llama2.c
*/
class GeneratOnPrompt {
    // GeneratOnPrompt(const GeneratOnPrompt&);
	// GeneratOnPrompt& operator=(const GeneratOnPrompt&);

protected:
    float *_logits = nullptr;
    std::vector<float> x_logits;
    float delta_max = 0,delta_a=0;
    bool display              = true;
    //compatible with LLAMA.cpp
    // gpt_params params;
    CLI_params hparams;
    MODEL_ARCH _arch = MODEL_ARCH::_X_;
    
    int ga_n=-1,ga_w=-1;
    int32_t bos=1,eos=2;  
    int n_predict = 32, n_batch = 2048, n_keep;
    bool is_antiprompt        = false;  
    // bool input_echo           = true;
    bool isCTXSampling = true;
    // struct llama_sampling_params sparams;
    // llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int n_ctx=-1,n_ctx_train=-1;
    llama_context * ctx_guidance = NULL;
    struct llama_sampling_context * ctx_sampling = NULL;
    // std::string path_session = params.path_prompt_cache;
    std::vector<TOKEN_ID> session_tokens;
    std::vector<TOKEN_ID> embd_inp;
    std::string GetPrompt(int flag=0x0);
    hSampLoader dialogs;
    std::vector<int>   input_tokens,output_tokens; 
    std::ostringstream output_ss;     
    bool is_interacting = false;    
    hWIKI wiki0 = nullptr;   
    arrHWIKI wikis;
    const Fish *fish_0 = nullptr;        
    shared_ptr<Fish> fish_1 = nullptr;        //for generate, only 1 input

    virtual TOKEN_ID Sample(int idx = -1,bool is_resampling=false);
    virtual std::string T2STR(TOKEN_ID tok,int flag=0x0 );  
    
    virtual void Clear();
    uint64_t rng_state;
    virtual void OnAntiPrompt(int flag);
    virtual bool Inference(hSAMP samp,int& nPast,int flag=0x0);
    // virtual void OnInteractive(int& n_past,int& n_consumed,int& n_remain,int flag);
public:
    GeneratOnPrompt()    {}
    GeneratOnPrompt(struct gpt_params&par_,int flag);
    GeneratOnPrompt(CLI_params&cp_, arrHWIKI& wiki_,const Fish* hG_,int flag);

    static shared_ptr<GeneratOnPrompt> MakeInstance(struct CLI_params& params,arrHWIKI& wiki,const Fish *,int flag);

    virtual ~GeneratOnPrompt()   {   Clear();    }
    virtual bool Init(const std::string& prompt_,int flag=0x0);    

    std::vector<TOKEN_ID> guidance_inp;
    std::vector<TOKEN_ID> inp_pfx,inp_sfx,cml_pfx,cml_sfx;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    virtual void InitInput(int flag=0x0);

    virtual int Tokenize(int flag);

    std::vector<TOKEN_ID> tokens;
    // std::vector<TOKEN_ID> embd_guidance;
    // tokenized antiprompts
    std::vector<std::vector<TOKEN_ID>> antiprompt_ids;

    virtual void Embed(int flag=0x0)    {
        assert(0);  //Deprecated
        // antiprompt_ids.reserve(params.antiprompt.size());
        // for (const std::string & antiprompt : params.antiprompt) {
        //     antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
        // }
    }

    // virtual int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0); 
    
    virtual int Generate(int nJob,int flag=0x0);
    virtual int Generate_v0(int nJob,int flag=0x0);
    // virtual int Generate_0(int nJob,int flag=0x0);

    virtual void DisplayEmbd(bool input_echo,int n_consumed,int flag=0x0);
            
};
typedef shared_ptr<GeneratOnPrompt> hGOPT;

class GOPT_infinite : public GeneratOnPrompt{
protected:    
    // int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0) override;   

public:
    GOPT_infinite(struct gpt_params&par_,int flag) : GeneratOnPrompt(par_,flag)  {;}
};

class GOPT_Metropolis : public GeneratOnPrompt{
protected:    
    TOKEN_ID Sample(int idx = -1,bool is_resampling=false)   override;
public:
    GOPT_Metropolis(struct gpt_params&par_,int flag) : GeneratOnPrompt(par_,flag)  {
        isCTXSampling = false;
    }
    GOPT_Metropolis(CLI_params&cp_,arrHWIKI& wikis_,const Fish* hG_,int flag): GeneratOnPrompt(cp_,wikis_,hG_,flag)  {
        isCTXSampling = false;
    }

    virtual ~GOPT_Metropolis()   {   Clear();    }

    // int Generate(int nJob,int flag=0x0) override;
};

int GPT_work(CLI_params& params);
int fish_1(CLI_params& hparams);
int GGUF_list(CLI_params& hparams);
int Fish_bubble(CLI_params& hparams);
int Tutor(CLI_params& hparams);