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
    LossCurve lcTrain,lcEval;

    int first_epoch=0,iter_at_last_epoch=-1,first_iter=-1;
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
    struct ggml_tensor * opt_ps[GGML_MAX_PARAMS]; // these will store the parameters we want to optimize    
    virtual void Clear()    {}   

    virtual bool one_step(struct train_state *train,SampLoader&loader, int accum_step, float *sched, int flag = 0x0);
    virtual void AdamW(int nx,int np,CLI_params& hparams,int flag);
    bool GradAccumulation(float&fx,int np,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag=0x0);
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
    struct ggml_opt_context * opt = train->opt; 
    struct train_params_common train_params;

    Optimizer(Fish *g_,struct train_params_common& params_,int flag=0x0);

    virtual float Compute(std::vector<llama_token>&tokens,bool isForward,int flag=0x0);
    virtual float Evaluate(SampLoader&loader,int iter,int flag=0x0);

    virtual void UpdateLoss(int step,float sched,float loss,int flag=0x0){
        if(step>1){
            float last = scheduler->Last();
            isStopImprove = step>0 ? (loss>last*1.1) : false;   
            if(isStopImprove){
                _INFO("%s_%d: StopImprove\n", __func__, opt->iter);
            }            
        }

        scheduler->Append(loss);
    }

    virtual bool isStopImproving( )	{	
        return isStopImprove;	
    }

    virtual void Dump(int typ){
        const char*title = "OPT";   //__func__
        _INFO("%s: mem_size  = %zu bytes (%.1f MB)\n", title, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));
        _INFO("%s: iter = %d\n", title, opt->iter);
        
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

    virtual void InitOpt(struct train_params_common& params_,int flag=0x0)  {
        opt->iter = train->train_its;

        opt->params = ggml_opt_default_params(GGML_OPT_TYPE_ADAM);
        opt->params.print_forward_graph     = false;
        opt->params.print_backward_graph    = false;
        opt->params.graph_size              = LLAMA_TRAIN_MAX_NODES;
        opt->params.n_threads               = train_params.n_threads;
        opt->params.past                    = train_params.opt_past;
        opt->params.delta                   = train_params.opt_delta;
        opt->params.max_no_improvement      = train_params.opt_max_no_improvement;
        opt->params.n_gradient_accumulation = train_params.n_gradient_accumulation;
        opt->params.adam.n_iter             = train_params.adam_n_iter;
        opt->params.adam.sched              = 1.0f;
        opt->params.adam.alpha              = train_params.adam_alpha;
        opt->params.adam.decay              = train_params.adam_decay;
        opt->params.adam.decay_min_ndim     = train_params.adam_decay_min_ndim;
        opt->params.adam.beta1              = train_params.adam_beta1;
        opt->params.adam.beta2              = train_params.adam_beta2;
        opt->params.adam.gclip              = train_params.adam_gclip;
        opt->params.adam.eps_f              = train_params.adam_eps_f;
    }
    
    virtual ~Optimizer( ) {
        ggml_free(opt->ctx);
        free_train_state(train);
    }

    enum ggml_opt_result ggml_train(struct ggml_context * ctx, hGensor loss_,hGensor target_,
            struct ggml_cgraph * gf_,struct ggml_cgraph * gb_,CLI_params& hparams);
    
    friend class Fish;
    friend class SampLoader;
    friend class SAMP;

};
typedef shared_ptr<Optimizer> hOptimizer;