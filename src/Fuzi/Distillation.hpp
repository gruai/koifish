/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
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
// #include "../lenda/util/GST_util.hpp"
#include "../Manifold/Scheduler.hpp"

class Fish;
class Distillation {
    float alpha = 0.5;
protected:
    std::vector<hGensor> dist_gensors;
    std::vector<float> sigmas;
    hLearnSKDU scheduler = nullptr; 
    Fish *hFish=nullptr;
    
public:
    enum ALG   {
        ADD=0,
        SIGMA,
        SIGMA_0,
    };
    ALG alg;    

    Distillation(Fish *hGang_,struct CLI_params&param,int flag)  : hFish(hGang_)   {
        alg = SIGMA;    //param.sigma=="add" ? ADD : SIGMA;
        scheduler = std::make_shared<DiscreteSchedule>(param.common);
        sigmas = scheduler->get_sigmas(100);
        _INFO("%s: alg=%d(%s)\n", __func__, alg,alg==SIGMA?"sigma":"add");
    }
    virtual ~Distillation() {

    }

    //  ref:    ggml_scale_impl     ggml_compute_forward_timestep_embedding_f32
    virtual hGensor UpdateGG(struct ggml_context * ctx,hGensor gensor,hGensor delta,int flag=0x0){
        // float sigma = 0.5;
        hGensor result=nullptr;
        switch(alg){
        case SIGMA:
            gensor = nullptr;   //gg_axpy_f32(ctx,G(gensor),sigma,G(delta),1-sigma);
            dist_gensors.push_back(gensor);
            result = gensor;
            break;
        case SIGMA_0:
            gensor = delta;
            break;
        default:
            gensor = nullptr;   //add_to_f32(ctx, G(gensor), G(delta));
            break;
        }      
        
        return gensor;  
    }

    virtual void UpdateSigma( int step,int flag=0x0);

};
typedef shared_ptr<Distillation> hDistillation;