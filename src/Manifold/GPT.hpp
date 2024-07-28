/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief A collection of neurons
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
#include "../LLAMA/common/common.h"
#include "GG_util.hpp"   

class Ganglia;

struct WIKI {
    enum MERGE_MODE{
        OFF,MERGE_P,MERGE_T
    };
    int32_t bos,eos;   
    MERGE_MODE teach=OFF;
    hGensor  logits = NULL;
    float * logits_out = nullptr;

    virtual hGensor P()    {   return nullptr; }
    virtual hGensor Target()    {   return nullptr; }

    virtual std::string T2STR(int32_t tok,int flag=0x0 )                    {   assert(0); return "";    }
    virtual bool Decode(std::vector<int32_t>&ids,int start,int n_past)      {   assert(0); return false;    }
    virtual void Answer(std::vector<int32_t>&ids,int flag=0x0)    {   assert(0); }

    WIKI();
    virtual ~WIKI() {}
};
typedef shared_ptr<WIKI> hWIKI;

/*
    Ref     https://github.com/karpathy/llama2.c
*/
class GeneratOnPrompt {
    // GeneratOnPrompt(const GeneratOnPrompt&);
	// GeneratOnPrompt& operator=(const GeneratOnPrompt&);

protected:
    bool display              = true;
    //compatible with LLAMA.cpp
    gpt_params params;
    CLI_params hparams;

    std::string prompt = "";
    int ga_n,ga_w;
    int n_predict = 16;
    bool is_antiprompt        = false;
    bool input_echo           = true;

    struct llama_sampling_params sparams;
    llama_model * model;
    llama_context * ctx;
    llama_context * ctx_guidance = NULL;
    struct llama_sampling_context * ctx_sampling = NULL;
    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;
    std::vector<llama_token> embd_inp;
    std::vector<int>   input_tokens,output_tokens; 
    std::ostringstream output_ss;     
    bool is_interacting = false;
    hWIKI wiki = nullptr;   
    Ganglia *gang = nullptr;        //only ref

    llama_token Sample(int idx = -1,bool is_resampling=false);

    virtual void Clear()    {
        llama_print_timings(ctx);
        // write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

        if (ctx_guidance) { llama_free(ctx_guidance); }
        if(wiki==nullptr){
            llama_free(ctx);        
            llama_free_model(model);            
        }

        if(ctx_sampling!=nullptr)
            llama_sampling_free(ctx_sampling);
        llama_backend_free();
    }

    virtual void OnAntiPrompt(int flag);
    virtual void OnInteractive(int& n_past,int& n_consumed,int& n_remain,int flag);
public:
    GeneratOnPrompt()    {}
    GeneratOnPrompt(struct gpt_params&par_,int flag);
    GeneratOnPrompt(hWIKI wiki_,Ganglia* hG_,int flag);

    virtual ~GeneratOnPrompt()   {   Clear();    }
    virtual bool Init(const std::string& prompt_,int flag=0x0);    

    std::vector<llama_token> guidance_inp;
    std::vector<llama_token> inp_pfx,inp_sfx,cml_pfx,cml_sfx;
    int guidance_offset = 0;
    int original_prompt_len = 0;

    virtual int Tokenize(int flag=0x0) {
        const int n_ctx = llama_n_ctx(ctx);        

        const bool add_bos = llama_should_add_bos_token(model);
        LOG("add_bos: %d\n", add_bos);
        if (params.interactive_first || params.instruct || params.chatml || !params.prompt.empty() || session_tokens.empty()) {
            LOG("tokenize the prompt\n");
            if (params.chatml) {
                params.prompt = "<|im_start|>system\n" + params.prompt + "<|im_end|>";
            }
            embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
        } else {
            LOG("use session tokens\n");
            embd_inp = session_tokens;
        }
        LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
        LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

        // Should not run without any tokens
        if (embd_inp.empty()) {
            embd_inp.push_back(llama_token_bos(model));
            LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
        }
        // Tokenize negative prompt
        std::vector<llama_token> guidance_inp;
        int guidance_offset = 0;
        int original_prompt_len = 0;
        if (ctx_guidance) {
            LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

            guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, add_bos, true);
            LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

            std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
            LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

            original_prompt_len = original_inp.size();
            guidance_offset = (int)guidance_inp.size() - original_prompt_len;
            LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
            LOG("guidance_offset:     %s", log_tostr(guidance_offset));
        }

        if ((int) embd_inp.size() > n_ctx - 4) {
            LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
            return 1;
        }

        // debug message about similarity of saved session, if applicable
        size_t n_matching_session_tokens = 0;
        if (!session_tokens.empty()) {
            for (llama_token id : session_tokens) {
                if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                    break;
                }
                n_matching_session_tokens++;
            }
            if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
                LOG_TEE("%s: using full prompt from session file\n", __func__);
            } else if (n_matching_session_tokens >= embd_inp.size()) {
                LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
            } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
                LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            } else {
                LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
            }

            // remove any "future" tokens that we might have inherited from the previous session
            llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
        }

        LOGLN(
                "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
                log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

        // if we will use the cache for the full prompt without reaching the end of the cache, force
        // reevaluation of the last token token to recalculate the cached logits
        if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
            LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

            session_tokens.resize(embd_inp.size() - 1);
        }

        // number of tokens to keep when resetting context
        if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
            params.n_keep = (int)embd_inp.size();
        } else {
            params.n_keep += add_bos; // always keep the BOS token
        }
         // prefix & suffix for instruct mode
        inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos, true);
        inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n",    false,   true);

        LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx).c_str());
        LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx).c_str());

        // chatml prefix & suffix
        cml_pfx = ::llama_tokenize(ctx, "\n<|im_start|>user\n", add_bos, true);
        cml_sfx = ::llama_tokenize(ctx, "<|im_end|>\n<|im_start|>assistant\n", false, true);

        LOG("cml_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_pfx).c_str());
        LOG("cml_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_sfx).c_str());
        if (params.instruct) {
            params.interactive_first = true;
            params.antiprompt.emplace_back("### Instruction:\n\n");
        }
        // similar for chatml mode
        else if (params.chatml) {
            params.interactive_first = true;
            params.antiprompt.emplace_back("<|im_start|>user\n");
        }

        // enable interactive mode if interactive start is specified
        if (params.interactive_first) {
            params.interactive = true;
        }

        if (params.verbose_prompt) {
            LOG_TEE("\n");
            LOG_TEE("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
            LOG_TEE("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
            for (int i = 0; i < (int) embd_inp.size(); i++) {
                LOG_TEE("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }

            if (ctx_guidance) {
                LOG_TEE("\n");
                LOG_TEE("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
                LOG_TEE("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
                for (int i = 0; i < (int) guidance_inp.size(); i++) {
                    LOG_TEE("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
                }
            }

            if (params.n_keep > add_bos) {
                LOG_TEE("%s: static prompt based on n_keep: '", __func__);
                for (int i = 0; i < params.n_keep; i++) {
                    LOG_TEE("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
                }
                LOG_TEE("'\n");
            }
            LOG_TEE("\n");
        }
        LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
        LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
        LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, 
            params.n_batch, params.n_predict, params.n_keep);
        // Handle_CtrlC();
        return 0x0;
    }

    std::vector<llama_token> tokens;
    std::vector<llama_token> embd_guidance;
    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    virtual void Embed(int flag=0x0)    {
        antiprompt_ids.reserve(params.antiprompt.size());
        for (const std::string & antiprompt : params.antiprompt) {
            antiprompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true));
        }
    }

    virtual int UpdateEmbed(int nJob,int &n_past,int &n_remain,int &n_consumed,int &n_session_consumed,int &n_past_guidance,int &ga_i,int flag=0x0); 

    virtual int Generate(int nJob,int flag=0x0);
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