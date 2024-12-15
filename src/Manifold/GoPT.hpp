/**
 *  Copyright 2023-2024 by Grusoft 
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
class Fish;

/*
     always for language model
*/
struct WIKI {
    enum INDUCT_MODE{
                        //  "off"-no wiki
        _OFF,           //  ""-no INDUCT
        _LOGITS,        //  "logits"-INDUCT to logits
        _TARGET,        //  "target"-INDUCT to target
        _LOGITS_SCALE,  //  "logits_scale"
        _LOGITS_GATE,
    };
    std::string title="",tokenizer_name,model_path; 
    shared_ptr<Fish> hFish=nullptr;
    void *vocab = nullptr;
    int32_t bos=1,eos=2;   
    string sBos,sEos;
    INDUCT_MODE teach=_LOGITS;
    bool isOnlyTokenizer = false;
    size_t n_vocab = 0, nOutToken = -1,nEleGGUF = 0;
    std::vector<std::pair<std::string, struct ggml_tensor *>> tmaps;

    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
    hGensor exLogits = nullptr,t2t = nullptr;

    virtual const float *GetLogits(int n_vocab,int n_ctx,int idx=-1)   {   return nullptr; }
    virtual bool isInduct() 
    {   return teach!=_OFF && exLogits!=nullptr; }

    virtual double InductLogits(int nSampInBatch,std::vector<int32_t>& tok_ids,hGensor exLogits,hGensor target_probs,int flag);
    virtual bool isValid(   )   {   return false;   }
    // bool takeRest = false;          //only for debug

    virtual hGensor P()    {   return nullptr; }
    virtual hGensor Target()    {   return nullptr; }
    /*
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
    */ 
    virtual int nCTX()   {   return -1;    };


    virtual std::string T2STR(int32_t tok,int flag=0x0 )                    {   assert(0); return "";       }
    virtual bool Decode(std::vector<int32_t>&ids,int start,int n_past,bool out_all)      {   assert(0); return false;    }
    virtual void Answer(std::vector<int32_t>&ids,int flag=0x0)    {   assert(0); }
    virtual void Reset(int flag=0x0)    {   assert(0); }

    virtual string __repr__( string& suffix,string& prefix,int flag=0x0)    {   assert(0); return "";      };

    WIKI();
    // WIKI(CLI_params& hparams,const std::string&path_,int flag=0x0);
    virtual ~WIKI() {}

    virtual void CopyParams(CLI_params& params,int flag=0x0)     {   assert(0); return;      };
    virtual bool CopyGensors(Fish *hFish,int flag=0x0);

    static std::vector<shared_ptr<WIKI>> MakeInstance(const std::string nam_,struct CLI_params& params,int flag);
};
typedef shared_ptr<WIKI> hWIKI;
typedef std::vector<hWIKI> arrHWIKI;

/*
    Ref     https://github.com/karpathy/llama2.c
*/
class GeneratOnPrompt {
    // GeneratOnPrompt(const GeneratOnPrompt&);
	// GeneratOnPrompt& operator=(const GeneratOnPrompt&);

protected:
    float *_logits = nullptr;
    float delta_max = 0,delta_a=0;
    bool display              = true;
    //compatible with LLAMA.cpp
    gpt_params params;
    CLI_params hparams;
    MODEL_ARCH _arch = MODEL_ARCH::_X_;

    
    int ga_n=-1,ga_w=-1;
    int32_t bos=1,eos=2;  
    int n_predict = 160;
    bool is_antiprompt        = false;  
    bool input_echo           = true;
    bool isCTXSampling = true;
    struct llama_sampling_params sparams;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int n_ctx=-1,n_ctx_train=-1;
    llama_context * ctx_guidance = NULL;
    struct llama_sampling_context * ctx_sampling = NULL;
    std::string path_session = params.path_prompt_cache;
    std::vector<TOKEN_ID> session_tokens;
    std::vector<TOKEN_ID> embd_inp;
    std::string GetPrompt(int flag=0x0);
    SampLoader dialogs;
    std::vector<int>   input_tokens,output_tokens; 
    std::ostringstream output_ss;     
    bool is_interacting = false;    
    // hWIKI wiki = nullptr;   
    arrHWIKI wikis;
    const Fish *fish_0 = nullptr;        
    shared_ptr<Fish> fish_1 = nullptr;        //for generate, only 1 input

    virtual TOKEN_ID Sample(int idx = -1,bool is_resampling=false);

    virtual void Clear();
    uint64_t rng_state;
    virtual void OnAntiPrompt(int flag);
    virtual void OnInteractive(int& n_past,int& n_consumed,int& n_remain,int flag);
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
    std::vector<TOKEN_ID> embd_guidance;
    // tokenized antiprompts
    std::vector<std::vector<TOKEN_ID>> antiprompt_ids;

    virtual void Embed(int flag=0x0)    {
        antiprompt_ids.reserve(params.antiprompt.size());
        for (const std::string & antiprompt : params.antiprompt) {
            antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
        }
    }

    virtual int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0); 

    virtual int Generate(int nJob,int flag=0x0);
    virtual int Generate_v0(int nJob,int flag=0x0);
    // virtual int Generate_0(int nJob,int flag=0x0);

    virtual void DisplayEmbd(bool input_echo,llama_context *ctx,int n_consumed,int flag=0x0);
            
};
typedef shared_ptr<GeneratOnPrompt> hGOPT;

class GOPT_infinite : public GeneratOnPrompt{
protected:    
    int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0) override;   

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