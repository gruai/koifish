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

#include "../ggex/GG_util.hpp"
#include "train.h"
#include "TGraph.hpp"
#include "Scheduler.hpp"
#include "DataLoader.hpp"
#include "../lenda/util/GST_util.hpp"

class Fish;

struct LossCurve    {
    vector<float> curve;
    int best_id=-1;

    float Last()    {
        return curve.empty() ? FLT_MAX : curve[curve.size()-1];
    }
    
    float Best()    {   return best_id==-1 ? FLT_MAX : curve[best_id]; }

    void Add(float a)   {
        curve.push_back(a);
        if(a<Best()){
            best_id = curve.size()-1;
        }
    }
};
class Optimizer : public std::enable_shared_from_this<Optimizer> {
    Optimizer(const Optimizer&);
	Optimizer& operator=(const Optimizer&);

protected:    
    std::string title = "Optimizer";

    LossCurve lcTrain,lcEval;
    struct ggml_context * _ctx=nullptr;
    std::vector<hGensor> opt_ps;
    size_t nParams = 0, nMostParam = 0;
    bool just_initialized = false,isAdaptiveSched = false,isGlobalGrad=true;
    int past=0,nGradAccum=0,tpSign=0;
    // gradient clipping
    double gclip = 1.0;         
    // schedule multiplier (fixed, decay or warmup)   
    float sched = 1.0f;             

    hGensor grad=nullptr;  // current gradient
    vector<float> fx_best;
    vector<float> fx_prev;
    int n_no_improvement=0;
    float loss_before=0,loss_after=0;
    int first_epoch=0,iter_at_last_epoch=-1,first_iter=-1,iter=-1;
    uint64_t train_its=0,train_samples=0,train_tokens=0,train_epochs=0,max_epoch=0;
    int64_t last_time;
    double millis_per_iter;

    // void *app_ctx = nullptr;
    double zmuv_0 = 0.0,zmuv_1 = 0.0,g_step=0.0;    
    llama_token bos,eos;
    // hGensor loss=nullptr, target_probs=nullptr, preLogits=nullptr; 
    hGensor hLoss( );
    hGensor hTargetProbs( );
    hGensor hPreLogits( );
    float fLoss()   {
        float *val = (float*)(hLoss()->data);
        return *val;
    }
    
    hScheduler scheduler = std::make_shared<DiscreteSchedule>( );
    bool isStopImprove = false;

    virtual void Clear()    {}   
    // update sched & dump some info
    virtual float UpdateSchedule(int flag = 0x0);
    virtual bool AfterLoadBatch(SampLoader&loader, int accum_step, int flag = 0x0);
    virtual int SignStochastic(int nx,CLI_params& hparams,int flag=0x0);
    virtual float gClip(int nx,float *g,hGensor hP,int flag=0x0);
    virtual void UpdateParams(int nx,CLI_params& hparams,int flag);
    // virtual void AdamMiniV(float gnorm,int nx,CLI_params& hparams,int flag);
    virtual bool BatchGrad(float&fx,int flag=0x0) {   assert(0);  return false; }
    bool OnLogits(int flag=0x0);
public:
    enum GD_METHOD {
        ADAMw=0x0,          
        SGD,               
        SGD_v,
        SGD_gensor_v,
   
    };
    GD_METHOD tpGD=ADAMw;  

    // typedef bool (*_CALL_BACK_)(void * data, int accum_step, float * sched);    
    SampLoader train_loader, val_loader;
    size_t shuffle_samples_hash = 0x0;  //hack

    Fish* gang=nullptr;      //ref only


    struct train_state      *trainst = init_train_state();
    //  struct ggml_opt_context * opt = train->opt; 
    struct train_params_common train_params;

    Optimizer(NLP_AutoRegressive *g_,struct train_params_common& params_,int flag=0x0);
    //Deprecated need refactor!!!       9/30/2024
    virtual float Compute(std::vector<llama_token>&tokens,bool isForward,int flag=0x0);
    virtual float Evaluate(SampLoader&loader,int iter,int flag=0x0);

    virtual void UpdateLoss(int step,float loss,int flag=0x0){
        if(step>1){
            float last = scheduler->Last();
            isStopImprove = step>0 ? (loss>last*1.1) : false;   
            if(isStopImprove){
                _INFO("%s_%d: StopImprove\n", __func__, iter);
            }            
        }

        scheduler->Append(loss);
    }

    virtual bool isStopImproving( )	{	
        return isStopImprove;	
    }

    virtual void Dump(int typ){
        const char*title = "OPT";   //__func__
        if(_ctx!=nullptr)
            _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, ggml_get_mem_size(_ctx), (float) ggml_get_mem_size(_ctx) / (1024.0f*1024.0f));
        _INFO("%s: iter = %d\n", title, iter);
        
        if(typ==1){
           _INFO("%s: SAMP_HASH=%llu total train_iterations=%llu train_samples=%llu train_tokens=%llu completed_epochs=%llu\n", title, 
                shuffle_samples_hash, train_its,train_samples,train_tokens,train_epochs);            
        }
        /*
        auto train=hOPT->train;
        _INFO("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
        _INFO("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
        _INFO("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
        _INFO("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
        _INFO("%s: nParams=%zu model_size = %zu bytes (%.1f MB)\n", __func__, nParams,szModel,szModel / (1024.0f*1024.0f) );*/
    }
    
    virtual void BeforeTrain(struct llama_context * lctx,struct train_params_common& train_params,hGensor  tokens_input,int flag) ;
    virtual bool PrepareData( CLI_params& hparams,int flag );
    virtual void Shuffle(int n_vocab,struct train_params_common& train_params,int flag=0x0)  {
        assert(0);
    }    

    virtual void Prepare(size_t nx,int flag=0x0);

    // virtual void InitOpt(struct train_params_common& params_,int flag=0x0);
    
    virtual ~Optimizer( ) {
        ggml_free(_ctx);
        // free_train_state(train);
    }

    enum ggml_opt_result Search(struct ggml_context * ctx, hGensor loss_,hGensor target_,CLI_params& hparams);
    
    friend class Fish;
    friend class SampLoader;
    friend class SAMP;

};
typedef shared_ptr<Optimizer> hOptimizer;

class OPT_Adam : public Optimizer  {
protected:    
    float decay = 0.0f; // weight decay for AdamW, use 0.0f to disable
    float p_decay = 0;
    int   decay_min_ndim = 2; // minimum number of tensor dimension to apply weight decay
    float alpha = 0.001f; // learning rate
    float beta1 = 0.9f,beta2 = 0.999f;
    float beta1h,beta2h;
    float eps = 1e-8f;   // epsilon for numerical stability
    float eps_f = 1e-5f; // epsilon for convergence test
    float eps_g = 1e-3f; // epsilon for convergence test
    
    hGensor gm=nullptr;  // first moment
    hGensor gv=nullptr;  // second moment
    hGensor gpf=nullptr; // past function values

    void Prepare(size_t nx,int flag=0x0)   override;
    // compute grad on batchs
    bool BatchGrad(float&fx,int flag=0x0) override;
    virtual double UpdateTensorParam(hGensor hP,float *m,float *v,float *g,float gnorm);
    void UpdateParams(int nx,CLI_params& hparams,int flag)  override;
public:
    OPT_Adam(NLP_AutoRegressive *g_,struct train_params_common& params_,int flag=0x0);
    void Dump(int typ)  override;
};

class OPT_AdamMiniV: public OPT_Adam  {
protected:    
    double UpdateTensorParam(hGensor hP,float *m,float *v,float *g,float gnorm) override;
public:
    OPT_AdamMiniV(NLP_AutoRegressive *g_,struct train_params_common& params_,int flag=0x0) :
        OPT_Adam(g_,params_,flag) {
        title = "OPT_AdamMiniV";
    };
};