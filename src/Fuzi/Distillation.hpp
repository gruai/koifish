/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief NLP_AutoRegressive Model(https://llama.meta.com/)
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
#include "../lenda/util/GST_util.hpp"
#include "../Manifold/Scheduler.hpp"

class Fish;
class Distillation {
    float alpha = 0.5;
protected:
    std::vector<hGensor> gensors;
    std::vector<float> sigmas;
    hScheduler scheduler = std::make_shared<DiscreteSchedule>( );
    Fish *hGang=nullptr;
    
public:
    enum ALG   {
        ADD=0,
        SIGMA,
        SIGMA_0,
    };
    ALG alg;    

    Distillation(Fish *hGang_,struct CLI_params&param,int flag)  : hGang(hGang_)   {
        alg = SIGMA;    //param.sigma=="add" ? ADD : SIGMA;
        scheduler = std::make_shared<DiscreteSchedule>();
        sigmas = scheduler->get_sigmas(100);
        _INFO("%s: alg=%d(%s)\n", __func__, alg,alg==SIGMA?"sigma":"add");
    }
    virtual ~Distillation() {

    }

    //  ref:    ggml_scale_impl     ggml_compute_forward_timestep_embedding_f32
    virtual hGensor UpdateGG(struct ggml_context * ctx,hGensor gensor,hGensor delta,int flag=0x0){
        float sigma = 0.5;
        hGensor result=nullptr;
        switch(alg){
        case SIGMA:
            gensor = gg_axpy_f32(ctx,gensor,sigma,delta,1-sigma);
            gensors.push_back(gensor);
            result = gensor;
            break;
        case SIGMA_0:
            gensor = delta;
            break;
        default:
            gensor = add_to_f32(ctx, gensor, delta);
            break;
        }      
        
        return gensor;  
    }

    virtual void UpdateSigma( int step,int flag=0x0);

};
typedef shared_ptr<Distillation> hDistillation;