/**
 *  Copyright 2023-2025 by Grusoft  
 * 
 *  \brief 
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
#include "../Device/EDevice.hpp"
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
    // std::map<hGensor, GENSOR_INFO> gimap;   
    LossCurve lcTrain,lcEval;
    struct ggml_context * _ctx=nullptr;
    std::vector<hGensor> opt_ps;
    size_t nParams = 0, nMostParam = 0;
    float *_tmp=nullptr;
    bool just_initialized = false,isAdaptiveSched = false,isGlobalGrad=true;
    bool isBackward = false;
    bool isConverge = false;
    int past=0,nGradAccum=0,tpSign=0;
    int warmup_iters=0;
    // gradient clipping
    double gclip = 1.0;         
    // schedule multiplier (fixed, decay or warmup)   
    float sched = 1.0f;             

    hGensor grad=nullptr;  // current gradient
    vector<float> fx_best;
    vector<float> fx_prev;
    int n_no_improvement=0;
    float loss_before=0,loss_after=0,tokens_per_second=0,ema_tps=0;
    int first_epoch=0,iter_at_last_epoch=-1,first_iter=-1,iter=-1;
    uint64_t train_its=0,train_samples=0,train_tokens=0,train_epochs=0,max_epoch=0;
    double last_time,tX=0,tData;
    double millis_per_iter=0;
    std::vector<string> adam_filter =  {"output","norm"};    //{"token_embd","output","norm"};

    // void *app_ctx = nullptr;
    double zmuv_0 = 0.0,zmuv_1 = 0.0,g_step=0.0,g_ll=0,g2_sum=0;    

    hGensor hLoss( );
    hGensor hTargetProbs( );
    hGensor hPreLogits( );
    hGensor GradOf(hGensor node,int flag=0);
    float* fGrad(hGensor node,int flag=0){
        auto grad = GradOf(node);
        assert(grad!=nullptr);
        float *g = (float*)(grad->data);
        return g;
    }
    float fLoss()   {
        float *val = (float*)(hLoss()->data);
        return *val;
    }
    
    hScheduler scheduler = std::make_shared<DiscreteSchedule>( );
    bool isStopImprove = false;

    virtual void Clear()    {}   
    // update sched & dump some info
    virtual float UpdateSchedule(int flag = 0x0);
    virtual bool AfterLoadBatch(int accum_step, int flag = 0x0);
    virtual int SignStochastic(int nx,CLI_params& hparams,int flag=0x0);
    virtual float gClip(int nx,floatX *g,hGensor hP,int flag=0x0);
    virtual void UpdateParams(int nx,CLI_params& hparams,int flag);
    // virtual void AdamMiniV(float gnorm,int nx,CLI_params& hparams,int flag);
    virtual bool BatchGrad(int iter,float&fx,int flag=0x0) {   assert(0);  return false; }
    bool OnLogits(int flag=0x0);
public:
    GD_METHOD tpGD=ADAMw;  
    enum RESULT {
        OK = 0,DID_NOT_CONVERGE,NO_CONTEXT,INVALID_WOLFE,
        FAIL,CANCEL,
    };
  
    hSampLoader train_loader=nullptr, val_loader=nullptr;
    size_t shuffle_samples_hash = 0x0;  //hack

    Fish* _fish=nullptr;         //ref only
    hEDevices hEDS;             //ref only

    struct train_state      *trainst = nullptr; //init_train_state();
    struct train_params_ TrainParams();
    
    Optimizer(NLP_AutoRegressive *g_,CLI_params& params_,int flag=0x0);
    //Deprecated need refactor!!!       9/30/2024
    virtual double GraphCompute(hTGraph,int flag=0x0);
    virtual float Evaluate(hSampLoader loader,int iter,int flag=0x0);

    virtual void UpdateLoss(int x,float loss,int flag=0x0);

    virtual bool isStopImproving( )	{	
        return isStopImprove;	
    }

    virtual void Dump(int typ){
        if(NOT_DUMP(0))    return;

        const char*title = "OPT";   //__func__
        if(_ctx!=nullptr)
            _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, ggml_get_mem_size(_ctx), (float) ggml_get_mem_size(_ctx) / (1024.0f*1024.0f));
        _INFO("%s: iter = %d\n", title, iter);
        
        if(typ==1){
           _INFO("%s: SAMP_HASH=%llu total train_iterations=%llu train_samples=%llu train_tokens=%llu completed_epochs=%llu\n", title, 
                shuffle_samples_hash, train_its,train_samples,train_tokens,train_epochs);            
        }
    }
    
    virtual void BeforeTrain(struct train_params_& train_params,hGensor  tokens_input,int flag) ;
    virtual bool PrepareData( CLI_params& hparams,int flag );
    virtual void Shuffle(int n_vocab,struct train_params_& train_params,int flag=0x0)  {
        assert(0);
    }    

    virtual void Prepare(size_t nx,int flag=0x0);

    // virtual void InitOpt(struct train_params_& params_,int flag=0x0);
    
    virtual ~Optimizer( ) {
        ggml_free(_ctx);
        if(_tmp!=nullptr)
            delete[] _tmp;
    }

    RESULT Search(struct ggml_context * ctx, hGensor loss_,hGensor target_,CLI_params& hparams);
    
    friend class Fish;
    friend class GeNeuron;
    friend class SampLoader;
    friend class SAMP;
    friend class TGraph;
};
typedef shared_ptr<Optimizer> hOptimizer;

class OPT_Adam : public Optimizer  {    
protected:    
    ADAM_params_ adam;      //may be modified
    // float decay = 0.0f; // weight decay for AdamW, use 0.0f to disable
    float p_decay = 0;
    // int   decay_min_ndim = 2; // minimum number of tensor dimension to apply weight decay
    // float alpha = 0.001f; // learning rate
    // float beta1 = 0.9f,beta2 = 0.999f;
    float beta1h,beta2h;

    void Prepare(size_t nx,int flag=0x0)   override;
    // compute grad on batchs
    bool BatchGrad(int iter,float&fx,int flag=0x0) override;
    virtual double UpdateTensorParam(hGensor hP,size_t offset,floatX *g,float gnorm);
    void UpdateParams(int nx,CLI_params& hparams,int flag)  override;
public:
    OPT_Adam(NLP_AutoRegressive *g_,CLI_params& params_,int flag=0x0);
    void Dump(int typ)  override;
};
