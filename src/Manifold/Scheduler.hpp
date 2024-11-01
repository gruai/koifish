/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief NLP_AutoRegressive Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */
#pragma once

/*
struct LearnSKDU{		
    int step;
    typedef enum{
        STATIC,TRI_LINE,
    }POLICY;
    Optimizer* hOPT=nullptr;
    float rBase,rMax,rMin,gamma,lr;
    int half,stp0;	//the step number to start the specified policy
    int btch_nz;
    POLICY policy;

    LearnSKDU( );
    void Init( Optimizer *hOPT_,int flag=0x0 );
    void BeforeStep( int flag=0x0  );
    void AfterEpoch( int epoch,int nBatch,bool isTest,int flag );

};*/
/**
 * https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
 * time-based decay, step decay and exponential decay.
*/
struct Scheduler {
    const static int TIMESTEPS = 1000;
    float alphas_cumprod[TIMESTEPS];
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

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

    struct _HISTORY {        
        std::vector<float> vals;
    };
    _HISTORY history;
    
    float Last()            {   assert(history.vals.size()>0);  return history.vals[history.vals.size()-1];  }
    void Append(float a)    {   history.vals.push_back(a);   }

    Scheduler() {}

    virtual std::vector<float> get_sigmas(uint32_t n) = 0;

    float sigma_to_t(float sigma) {
        float log_sigma = std::log(sigma);
        std::vector<float> dists;
        dists.reserve(TIMESTEPS);
        for (float log_sigma_val : log_sigmas) {
            dists.push_back(log_sigma - log_sigma_val);
        }

        int low_idx = 0;
        for (size_t i = 0; i < TIMESTEPS; i++) {
            if (dists[i] >= 0) {
                low_idx++;
            }
        }
        low_idx      = std::min(std::max(low_idx - 1, 0), TIMESTEPS - 2);
        int high_idx = low_idx + 1;

        float low  = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        float w    = (low - log_sigma) / (low - high);
        w          = std::max(0.f, std::min(1.f, w));
        float t    = (1.0f - w) * low_idx + w * high_idx;

        return t;
    }

    float t_to_sigma(float t) {
        int low_idx     = static_cast<int>(std::floor(t));
        int high_idx    = static_cast<int>(std::ceil(t));
        float w         = t - static_cast<float>(low_idx);
        float log_sigma = (1.0f - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx];
        return std::exp(log_sigma);
    }

    float sigma_data = 1.0f;

    std::vector<float> get_scalings(float sigma) {
        float c_out = -sigma;
        float c_in  = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        return {c_out, c_in};
    }
};
typedef shared_ptr<Scheduler> hScheduler;

struct DiscreteSchedule : Scheduler {
    std::vector<float> get_sigmas(uint32_t n) {
        std::vector<float> result;

        int t_max = TIMESTEPS - 1;

        if (n == 0) {
            return result;
        } else if (n == 1) {
            result.push_back(t_to_sigma((float)t_max));
            result.push_back(0);
            return result;
        }

        float step = static_cast<float>(t_max) / static_cast<float>(n - 1);
        for (uint32_t i = 0; i < n; ++i) {
            float t = t_max - step * i;
            result.push_back(t_to_sigma(t));
        }
        result.push_back(0);
        return result;
    }
};

struct KarrasSchedule : Scheduler {
    std::vector<float> get_sigmas(uint32_t n) {
        // These *COULD* be function arguments here,
        // but does anybody ever bother to touch them?
        float sigma_min = 0.1f;
        float sigma_max = 10.f;
        float rho       = 7.f;

        std::vector<float> result(n + 1);

        float min_inv_rho = pow(sigma_min, (1.f / rho));
        float max_inv_rho = pow(sigma_max, (1.f / rho));
        for (uint32_t i = 0; i < n; i++) {
            // Eq. (5) from Karras et al 2022
            result[i] = pow(max_inv_rho + (float)i / ((float)n - 1.f) * (min_inv_rho - max_inv_rho), rho);
        }
        result[n] = 0.;
        return result;
    }
};

