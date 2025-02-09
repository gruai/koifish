/**
 * @file LearnSKDU.cpp
 * @author your name (you@domain.com)
 * @brief learning rate LearnSKDU
 * @version 0.1
 * @date 2024-09-20
 * 
 * @copyright Copyright (c) 2024
 * 
 * 
 * Warmup-Stable-Decay (WSD)   outperforms the cosine LearnSKDU
 */
#include "Scheduler.hpp"

float LearnSKDU::LearningRate(int64_t step, int flag){
    float lr0 = _params.LearningRate(),lr;
    int warmup = _params.warmup,n_iter=_params.adam.n_iter,final_learning_rate_frac=0;
    if (step < warmup) {
        lr = lr0 * ((float)(step + 1)) / warmup;
    } else {
        float decay_ratio = ((float)(step - warmup)) / (n_iter - warmup);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = lr0 * final_learning_rate_frac;
        lr = min_lr + coeff * (lr0 - min_lr);
    }
    return lr;
}