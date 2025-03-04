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
inline double NRM_2_( const float *X,size_t dim )		{	
    double sum=0;
    const float *x=X;
    for(size_t i=0;i<dim;i++,x++){
        sum += (*x)*(*x);
    }
    return sqrt(sum);
}
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
    float *exLogits = nullptr,*t2t = nullptr;
    // struct ggml_tensor  *exLogits = nullptr,*t2t = nullptr;

    virtual const float *GetLogits(int n_vocab,int n_ctx,int idx=-1)   {   return nullptr; }
    virtual bool isInduct() 
    {   return teach!=_OFF && exLogits!=nullptr; }

    virtual double InductLogits(const CLI_params&config, int nSampInBatch,std::vector<TOKEN_ID>& tok_ids,struct ggml_tensor *target_probs,int flag) {
        if(!isInduct())
            return -1.0;
        
        Reset();         //Timing bottleneck!!! for the crazy design of llama.cpp
        Decode(tok_ids,0,0x0,true);    
        const float *all_logits = GetLogits(n_vocab,tok_ids.size(),0),*logit; 
        size_t k,j,i;    //exLogits->ne[0];  
        int n_ctx =config.n_ctx(),n_dialect=mapT2T.size(),token;  //target_probs->ne[1],
        double a1,a2,nrm=0;    
        float *p=teach == WIKI::_TARGET ? new float[n_vocab]:nullptr,*target=nullptr;  
        if(flag<0){    //CHILD_0909_WIKIS
            /*struct ggml_tensor * logits = userLogits==nullptr ? exLogits : userLogits;
            assert(logits!=nullptr);
            target = (float*)logits->data+nSampInBatch*n_ctx*ldL;        
            nrm =  NRM_2_(all_logits,n_ctx*ldL)/ldL;   
            if(logits->ne[0]==n_dialect){
                for(i=0; i<n_ctx; i++,target+=n_dialect,all_logits+=n_vocab){
                    for(j=0;j<n_vocab;j++){
                        if(dialect[j]==0)       
                            continue;
                        token = mapT2T[j];
                        target[token] = all_logits[j];
                    }                
                }
            }else*/{
                target = exLogits+nSampInBatch*n_ctx*n_vocab;        
                memcpy((void*)target,(void*)all_logits,sizeof(float)*n_ctx*n_vocab);       //memcpy(g->data+off,(void*)(logits),ld2); 
            }
        }else{    
            assert(0);
            for (k=0; k<nSampInBatch; ++k) {        
                const float *from=all_logits+k*n_vocab;
                a1=NRM_2_((float*)(from),n_ctx*n_vocab);          nrm=max(nrm,a1/n_vocab);     
                if(teach == WIKI::_TARGET){              
                    assert(exLogits==nullptr);             
                    for(j=0;j<n_ctx;j++){
                        logit = from+j*n_vocab;
                        target = (float*)target_probs->data+(k*n_ctx+j)*n_vocab;
                        //  SOFT_MAX_minus(n_vocab,target,logit);
                        // SOFT_MAX(n_vocab,p,logit);
                        for(a1=0,a2=0,i=0;i<n_vocab;i++){
                            a1 += target[i];            a2 += p[i];
                            target[i] -= p[i];
                        }
                        // SOFT_MAX(n_vocab,p,target);     //  !!!No converge!!!   @home/cys/rnd/lic/log/eval/08_21_wiki_target_no_converge.info  
                        memcpy(target,p,sizeof(float)*n_vocab);
                        // todo - cys 20240821: MSE loss 
                    }
                }else{
                    assert(exLogits!=nullptr);                
                    if(exLogits!=from){
                        // target = (float*)exLogits->data+k*n_ctx*n_vocab;          
                        target = (float*)exLogits+k*n_ctx*n_vocab;          
                        memcpy((void*)target,(void*)from,sizeof(float)*n_ctx*n_vocab);         
                    }
                    
                }
            }    
        }
        delete[] p;
        return nrm;
    }

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
    virtual ~WIKI() {   }

    virtual void CopyParams(CLI_params& params,int flag=0x0)        {   assert(0); return;      };
    // virtual bool CopyGensors(Fish *hFish,int flag=0x0)              {   return false;   }
    static std::vector<shared_ptr<WIKI>> MakeInstance(const std::string nam_,struct CLI_params& params,int flag);
};

typedef shared_ptr<WIKI> hWIKI;
typedef std::vector<hWIKI> arrHWIKI;

struct LAMA;