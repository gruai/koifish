/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */
#include "Scheduler.hpp"
#include "../ggex/GG_util.hpp"

LearnSKDU::LearnSKDU(struct train_params_& train_params) : _params(train_params) {
    if(_params.lr_restart==1)
        policy = COSINE_EPOCH;
    warmup = _params.warmup,mostIter=_params.nMostIter;
    if(policy == COSINE_EPOCH){
        mostIter = _params.nEpochIter;   
    }
    if(mostIter<warmup){
        warmup = max(1,mostIter/10);       assert(warmup>0);
    }
}

void LearnSKDU::Dump(int typ){
    _INFO("\tSKDU policy=%s warmup=%d@%d\n",policy==COSINE?"COSINE":"COSINE_EPOCH",warmup,mostIter);
}

float LearnSKDU::LearningRate(int64_t step, int flag){
    float lr0 = _params.LearningRate(),lr;
    int final_learning_rate_frac=0;
    if(policy == COSINE_EPOCH){
        step = step%_params.nEpochIter;     
    }
    
    if (step < warmup) {
        lr = lr0 * ((float)(step + 1)) / warmup;
    } else {
        float decay_ratio = ((float)(step - warmup)) / (mostIter - warmup);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = lr0 * final_learning_rate_frac;
        lr = min_lr + coeff * (lr0 - min_lr);
    }
    return lr;
}

/*
float cosine_decay(int64_t step, int64_t decay_steps, float minimum) {
    if (step > decay_steps) {
        step = decay_steps;
    }
    const float cosine_decay = 0.50f*(1.0f + cosf(3.14159265359f*step/decay_steps));
    const float decay = (1 - minimum)*cosine_decay + minimum;
    return decay;
}

float cosine_decay_restart(int64_t step, int64_t decay_steps, float minimum, float restart_step_mult) {
    while (step > decay_steps) {
        step -= decay_steps;
        decay_steps = (int64_t) (restart_step_mult * decay_steps);
    }
    return cosine_decay(step, decay_steps, minimum);
}

float learning_schedule(
    int64_t step,
    int64_t warmup_steps,
    int64_t cos_decay_steps,
    float   learning_rate,
    float   overall_minimum,
    float   cos_decay_minimum,
    float   cos_decay_restart_step_mult,
    bool    enable_restart) {

    float result =
        (step < warmup_steps)
            ? (float) step / (float) warmup_steps
            : enable_restart
                ? cosine_decay_restart(
                    step - warmup_steps,
                    cos_decay_steps,
                    cos_decay_minimum,
                    cos_decay_restart_step_mult)
                : cosine_decay(
                    step,
                    cos_decay_steps,
                    cos_decay_minimum);

    float min = overall_minimum / learning_rate;
    result = min + result * (1.0f - min);
    return result;
}


*/