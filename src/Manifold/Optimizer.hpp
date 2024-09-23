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
    struct ggml_context * ctx=nullptr;
    size_t nParams = 0;
    bool just_initialized = false,isAdaptiveSched = true;
    int past=0,n_gradient_accumulation=0;
    float gclip = 1.0;            // gradient clipping
    float sched = 1.0f; // schedule multiplier (fixed, decay or warmup)

    hGensor grad=nullptr;  // current gradient
    vector<float> fx_best;
    vector<float> fx_prev;
    int n_no_improvement=0;
    float loss_before=0,loss_after=0;
    int first_epoch=0,iter_at_last_epoch=-1,first_iter=-1,iter=-1;
    int64_t                      last_time;
    double                       millis_per_iter;

    void *app_ctx = nullptr;
    double zmuv_0 = 0.0,zmuv_1 = 0.0;
    struct ggml_cgraph *gf=nullptr,*gb=nullptr;     //only for debug
    llama_token bos,eos;
    // hGensor loss=nullptr, target_probs=nullptr, preLogits=nullptr; 
    hGensor hLoss( );
    hGensor hTargetProbs( );
    hGensor hPreLogits( );
    float fLoss()   {
        float *val = (float*)(hLoss()->data);
        return *val;
    }

    hGensor tokens_input = nullptr; 
    hScheduler scheduler = std::make_shared<DiscreteSchedule>( );
    bool isStopImprove = false;
    // struct ggml_tensor * opt_ps[GGML_MAX_PARAMS]; // these will store the parameters we want to optimize    
    std::vector<hGensor> opt_ps;
    virtual void Clear()    {}   

    virtual bool one_step(struct train_state *train,SampLoader&loader, int accum_step, float *sched, int flag = 0x0);
    virtual float gClip(int nx,CLI_params& hparams,int flag=0x0);
    virtual void UpdateParams(float gnorm,int nx,CLI_params& hparams,int flag);
    // virtual void AdamMiniV(float gnorm,int nx,CLI_params& hparams,int flag);
    virtual bool GradAccumulation(float&fx,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag=0x0) {   assert(0);  return false; }
    bool OnLogits(int flag=0x0);
public:
    // typedef bool (*_CALL_BACK_)(void * data, int accum_step, float * sched);    
    SampLoader train_loader, val_loader;
    Fish* gang=nullptr;      //ref only
    // std::vector<llama_token> train_tokens;
    // std::vector<size_t> train_samples_begin,train_samples_size;
    // std::vector<size_t> train_shuffled_samples_offs;
    // std::vector<size_t> train_shuffled_samples_begin;
    // std::vector<size_t> train_shuffled_samples_size;

    struct train_state      * train = init_train_state();
    //  struct ggml_opt_context * opt = train->opt; 
    struct train_params_common train_params;

    Optimizer(Fish *g_,struct train_params_common& params_,int flag=0x0);

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
        _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, ggml_get_mem_size(ctx), (float) ggml_get_mem_size(ctx) / (1024.0f*1024.0f));
        _INFO("%s: iter = %d\n", title, iter);
        
        if(typ==1){
            _INFO("%s: total train_iterations=%llu train_samples=%llu train_tokens=%llu completed_epochs=%llu\n", title, 
                train->train_its,train->train_samples,train->train_tokens,train->train_epochs);            
        }
        /*
        auto train=hOPT->train;
        _INFO("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
        _INFO("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
        _INFO("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
        _INFO("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
        _INFO("%s: nParams=%zu model_size = %zu bytes (%.1f MB)\n", __func__, nParams,szModel,szModel / (1024.0f*1024.0f) );*/
    }
    
    virtual void Init_CallbackData(struct llama_context * lctx,struct train_params_common& train_params,hGensor  tokens_input,int flag) ;

    virtual void Shuffle(int n_vocab,struct train_params_common& train_params,int flag=0x0)  {
        assert(0);
    }    

    virtual void Init(size_t nx,int flag=0x0);

    virtual void InitOpt(struct train_params_common& params_,int flag=0x0);
    
    virtual ~Optimizer( ) {
        ggml_free(ctx);
        free_train_state(train);
    }

    enum ggml_opt_result ggml_train(struct ggml_context * ctx, hGensor loss_,hGensor target_,
            struct ggml_cgraph * gf_,struct ggml_cgraph * gb_,CLI_params& hparams);
    
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
    
    hGensor gm;  // first moment
    hGensor gv;  // second moment
    hGensor gpf; // past function values

    void Init(size_t nx,int flag=0x0)   override;
    bool GradAccumulation(float&fx,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag=0x0) override;
    virtual void UpdateTensorParam(hGensor hP,float *m,float *v,float *g,float gnorm);
    void UpdateParams(float gnorm,int nx,CLI_params& hparams,int flag)  override;
public:
    OPT_Adam(Fish *g_,struct train_params_common& params_,int flag=0x0);
    void Dump(int typ)  override;
};

class OPT_AdamMiniV: public OPT_Adam  {
protected:    
    void UpdateTensorParam(hGensor hP,float *m,float *v,float *g,float gnorm) override;
public:
    OPT_AdamMiniV(Fish *g_,struct train_params_common& params_,int flag=0x0) :
        OPT_Adam(g_,params_,flag) {
        title = "OPT_AdamMiniV";
    };
};