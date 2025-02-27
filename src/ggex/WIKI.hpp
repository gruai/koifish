/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT 
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
    bool add_bos = false;
    int32_t bos=1,eos=2;   
    string sBos,sEos;
    INDUCT_MODE teach=_LOGITS;
    bool isOnlyTokenizer = false;
    size_t n_vocab = 0, nOutToken = -1,nEleGGUF = 0;
    std::vector<std::pair<std::string, struct ggml_tensor  *>> tmaps;

    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
    struct ggml_tensor  *exLogits = nullptr,*t2t = nullptr;

    virtual const float *GetLogits(int n_vocab,int n_ctx,int idx=-1)   {   return nullptr; }
    virtual bool isInduct() 
    {   return teach!=_OFF && exLogits!=nullptr; }

    // virtual double InductLogits(int nSampInBatch,std::vector<TOKEN_ID>& tok_ids,struct ggml_tensor *exLogits,struct ggml_tensor *target_probs,int flag);

    virtual bool isValid(   )   const    {   return false;   }
    // bool takeRest = false;          //only for debug

    virtual struct ggml_tensor  * P()    {   return nullptr; }
    virtual struct ggml_tensor  * Target()    {   return nullptr; }
    /*
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
    */ 
    virtual int nCTX()   {   return -1;    };

    virtual int STR2T(const std::string&info,std::vector<TOKEN_ID>&,int flag=0x0 )                    {   assert(0); return -1;       }
    virtual std::string T2STR(int32_t tok,int flag=0x0 )                    {   assert(0); return "";       }
    virtual bool Decode(std::vector<TOKEN_ID>&ids,int start,int n_past,bool out_all)      {   assert(0); return false;    }
    virtual void Answer(std::vector<TOKEN_ID>&ids,int flag=0x0)    {   assert(0); }
    virtual void Reset(int flag=0x0)    {   assert(0); }

    virtual string __repr__( string& suffix,string& prefix,int flag=0x0)    const   {   assert(0); return "";      };

    WIKI( )         {   }
    // WIKI(CLI_params& hparams,const std::string&path_,int flag=0x0);
    virtual ~WIKI() {   }

    virtual void CopyParams(CLI_params& params,int flag=0x0)        {   assert(0); return;      };
    // virtual bool CopyGensors(Fish *hFish,int flag=0x0)              {   return false;   }
    static std::vector<shared_ptr<WIKI>> MakeInstance(const std::string nam_,struct CLI_params& params,int flag);
};
typedef shared_ptr<WIKI> hWIKI;
typedef std::vector<hWIKI> arrHWIKI;

struct LAMA;