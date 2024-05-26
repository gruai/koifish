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
#include "../LLAMA/common/train.h"
#include "TGraph.hpp"
#include "Scheduler.hpp"
#include "../lenda/util/GST_util.hpp"

// void save_train_(void * vdata, struct train_state * train);

class Ganglia;
class Optimizer : public std::enable_shared_from_this<Optimizer> {
    Optimizer(const Optimizer&);
	Optimizer& operator=(const Optimizer&);

protected:    
    struct ggml_cgraph *gf=NULL,*gb=NULL;     //only for debug
    hGensor  loss = NULL, target_probs = NULL; 
    hScheduler scheduler = std::make_shared<DiscreteSchedule>( );
    bool isStopImprove = false;
    struct ggml_tensor * opt_ps[GGML_MAX_PARAMS]; // these will store the parameters we want to optimize    
    virtual void Clear()    {}    

    bool GradAccumulation(float&fx,int np,struct train_opt_callback_data *callback_data,struct ggml_cplan *cplan,int flag=0x0);
public:
    // typedef bool (*_CALL_BACK_)(void * data, int accum_step, float * sched);
    
    

    Ganglia* gang=nullptr;      //ref only
    std::vector<llama_token> train_tokens;
    std::vector<size_t> train_samples_begin,train_samples_size;
    std::vector<size_t> train_shuffled_samples_offs;
    std::vector<size_t> train_shuffled_samples_begin;
    std::vector<size_t> train_shuffled_samples_size;

    struct train_state      * train = init_train_state();
    struct ggml_opt_context * opt = train->opt; 

    struct train_params_common train_params;
    Optimizer(Ganglia *g_,struct train_params_common& params_,int flag=0x0) : train_params(params_),gang(g_) {
        InitOpt(train_params,flag);
    }
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
        _INFO("%s: opt_size  = %zu bytes (%.1f MB)\n", __func__, ggml_get_mem_size(opt->ctx), (float) ggml_get_mem_size(opt->ctx) / (1024.0f*1024.0f));
        _INFO("%s: opt iter %d\n", __func__, opt->iter);
        if(typ==1){
            _INFO("%s: total train_iterations %llu\n", __func__, (long long unsigned) train->train_its);
            _INFO("%s: seen train_samples     %llu\n", __func__, (long long unsigned) train->train_samples);
            _INFO("%s: seen train_tokens      %llu\n", __func__, (long long unsigned) train->train_tokens);
            _INFO("%s: completed train_epochs %llu\n", __func__, (long long unsigned) train->train_epochs);
        }
    }
    
    virtual void Init_CallbackData(struct train_opt_callback_data& opt_cb_data,struct llama_context * lctx,struct train_params_common& train_params,hGensor  tokens_input,int flag) {
        opt_cb_data.params                 = &(train_params); //hparams.common
        opt_cb_data.train                  = train;
        opt_cb_data.save_cb                = nullptr;   //&save_train_;
        opt_cb_data.save_data              = nullptr;   //&save_data;
        opt_cb_data.lctx                   = lctx;
        opt_cb_data.last_save_iter         = opt->iter;
        opt_cb_data.tokens_data            = train_tokens.data();
        opt_cb_data.tokens_size            = train_tokens.size();
        opt_cb_data.samples_begin          = train_samples_begin.data();
        opt_cb_data.samples_size           = train_samples_size.data();
        opt_cb_data.shuffled_samples_offs  = train_shuffled_samples_offs.data();
        opt_cb_data.shuffled_samples_begin = train_shuffled_samples_begin.data();
        opt_cb_data.shuffled_samples_size  = train_shuffled_samples_size.data();
        opt_cb_data.samples_count          = train_samples_size.size();
        opt_cb_data.tokens_input           = tokens_input;
        opt_cb_data.target_probs           = target_probs;
        opt_cb_data.first_iter             = opt->iter;
        opt_cb_data.first_epoch            = train->train_epochs;
        opt_cb_data.iter_at_last_epoch     = -1;
        opt_cb_data.last_time              = ggml_time_ms();
        opt_cb_data.millis_per_iter        = 0.0;
    }

    virtual void Shuffle(int n_vocab,struct train_params_common& train_params,int flag=0x0)  {
        std::vector<size_t> token_noccurs;
        token_noccurs.resize(n_vocab, 0);   //params.n_vocab
        for (unsigned int i = 0; i < train_tokens.size(); ++i) {
            ++token_noccurs[train_tokens[i]];
        }
        int n_unique_tokens = 0;
        for (unsigned int i = 0; i < token_noccurs.size(); ++i) {
            if (token_noccurs[i] == 0) continue;
            ++n_unique_tokens;
        }
        _INFO("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);

        size_t shuffle_samples_hash = compute_samples_hash(train_params.fn_train_data, train_samples_begin.data(), train_samples_size.data(), train_samples_size.size());
        const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != train_samples_size.size());
        if (changed_train_data) {
            _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
        }
        if (train_params.force_reshuffle) {
            _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
        }
        if ((train->shuffle_rng_state_current == "") || changed_train_data || train_params.force_reshuffle) {
            train->shuffle_rng_state_current = mt19937_seed_to_state(train_params.seed);
            train->shuffle_sample_count = train_samples_size.size();
            train->shuffle_next_sample = 0;
            train->shuffle_samples_hash = shuffle_samples_hash;
        }

        train_shuffled_samples_offs.resize(train_samples_begin.size());
        train_shuffled_samples_begin.resize(train_samples_begin.size());
        train_shuffled_samples_size.resize(train_samples_size.size());
        train->shuffle_rng_state_next = shuffle_samples(
            train->shuffle_rng_state_current,
            train_shuffled_samples_offs.data(),
            train_shuffled_samples_begin.data(),
            train_shuffled_samples_size.data(),
            train_samples_begin.data(),
            train_samples_size.data(),
            train_samples_size.size());
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

    enum ggml_opt_result ggml_train(struct ggml_context * ctx, struct train_opt_callback_data *callback_data, hGensor loss_,hGensor target_,struct ggml_cgraph * gf_,struct ggml_cgraph * gb_);

};
typedef shared_ptr<Optimizer> hOptimizer;