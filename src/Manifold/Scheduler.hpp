/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Scheduling
 *  \author Yingshi Chen
 */
#pragma once
#include <memory>

#include "../CLI_params.hpp"
#include "../Device/EDevice.hpp"
#include "Neuron.hpp"
#include "TGraph.hpp"

class EDGE_DEVICES;

/**
 * https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
 * time-based decay, step decay and exponential decay.
 * 1. A critical limitation of cosine learning rate decay is that it achieves optimal performance only when performing an entire cosine period [4], forcing
 * practitioners to fix the number of steps beforehand, which poses a significant hurdle if we want to continue pre-training later when more data or/and compute
 * becomes available.
 */
struct LearnSKDU {
    typedef enum {
        STATIC,
        TRI_LINE,
        COSINE,
        COSINE_EPOCH,
        WSD,  //  Warmup-Stable-Decay (WSD), might be less stable with spike loss curve
    } POLICY;
    POLICY policy = COSINE;

    TRAIN_CARD _params;
    const static int TIMESTEPS = 1000;
    int warmup = 1, mostIter = 1;
    float alphas_cumprod[TIMESTEPS];
    float sigmas[TIMESTEPS];
    float log_sigmas[TIMESTEPS];

    float LearningRate(int64_t step, int flag = 0x0);
    float cosine_decay(int64_t step, int64_t decay_steps, float minimum) {
        if (step > decay_steps) {
            step = decay_steps;
        }
        const float cosine_decay = 0.50f * (1.0f + cosf(3.14159265359f * step / decay_steps));
        const float decay        = (1 - minimum) * cosine_decay + minimum;
        return decay;
    }

    float cosine_decay_restart(int64_t step, int64_t decay_steps, float minimum, float restart_step_mult) {
        while (step > decay_steps) {
            step -= decay_steps;
            decay_steps = (int64_t)(restart_step_mult * decay_steps);
        }
        return cosine_decay(step, decay_steps, minimum);
    }

    struct _HISTORY {
        std::vector<float> vals;
    };
    _HISTORY history;

    float Last() {
        assert(history.vals.size() > 0);
        return history.vals[history.vals.size() - 1];
    }
    void Append(float a) { history.vals.push_back(a); }

    LearnSKDU(TRAIN_CARD &train_params);
    virtual void Dump(int typ);
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
typedef std::shared_ptr<LearnSKDU> hLearnSKDU;

struct DiscreteSchedule : LearnSKDU {
    DiscreteSchedule(TRAIN_CARD &train_params) : LearnSKDU(train_params) {}
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

struct KarrasSchedule : LearnSKDU {
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

enum TASK_STATUS {
    PASS,
    RESIDENT,
    FLIP,
    UPDATE_PARAM,
};
struct TaskNode {
    string name;
    void *hOBJ         = nullptr;
    double cost        = 0;
    TASK_STATUS status = FLIP;

    TaskNode(const std::string &n, void *h, double v, int flag = 0x0) : name(n), hOBJ(h), cost(v) {}
    bool isOn() { return status == RESIDENT; }
    int begin = -1, end = -1;
};
// typedef vector<TaskNode *> hFuyou;

/*
    Only for back-propagation of neurons

    蜉蝣 - 寄蜉蝣之天地，渺沧海之一粟
*/
class Fuyou {
   protected:
    Fuyou_params params;
    Grusoft::GRander rander;
    uint32_t seed = 42;
    string name;
    float loss = FLT_MAX, loss_0 = FLT_MAX;
    vector<TaskNode *> tasks;
    RLS_BP *hRLS = nullptr;
    Fish *hFish  = nullptr;
    vector<hGensor> fParams;    // from Fish::optParams
    vector<hGensor> tReloads;   //  subset of fParams
   public:
    // Fuyou() {}
    Fuyou(const string &name, RLS_BP *hRL, Fish *hFish, vector<TaskNode *> arrT, int flag = 0x0);
    virtual bool Serialize(bool isSave, int flag = 0x0);

    vector<TaskNode *> Tasks(int flag = 0x0) { return tasks; }
    virtual bool empty() { return tasks.size() == 0; }
    virtual void Clear() { tasks.clear(); }
    virtual void Add(TaskNode *node, int flag = 0x0) { tasks.push_back(node); }
    virtual TaskNode *Last() {
        assert(!empty());
        return tasks[tasks.size() - 1];
    }
    virtual TaskNode *First() {
        assert(!empty());
        return tasks[0];
    }
    virtual bool UpdateFollower(std::shared_ptr<Fuyou> follower, int flag = 0x0);
    virtual bool Backward(hGensor cur, int flag = 0x0);
    // bool Exploitation(hGensor cur, int flag = 0x0);
    virtual bool Exploitation(hGensor tHead, hGensor tNext, int flag = 0x0);
    // virtual void CrossOver(hGensor tHead, hGensor tNext, int flag = 0x0);
    // virtual void Mutation(double T_mut, int flag = 0x0);
    friend class RLSchedule;
    friend class RLS_BP;
    friend class Optimizer;
};
typedef std::shared_ptr<Fuyou> hFuyou;
/*
    Resource limited scheduling / resource planning
    budget/availability/capacity/costs  workload management
    Resource-constrained
*/
class RLSchedule {
   public:
   protected:
    EDGE_DEVICES *hDevices = nullptr;
    Fish *hFish            = nullptr;
    int step               = 0x0;
    bool isPrefill         = true;
    bool isRemater         = false;
    double budget          = 1000;

    vector<hFuyou> fuyouSwarm;
    //  active subset of hRLS->fuyouSwarm,  may varied at different stage & models
    vector<hFuyou> ActiveFuyous(int flag = 0x0);  //
    hFuyou afu      = nullptr;
    int curBranchID = 0;

    LIFE_PHASE phase = LIFE_PHASE::P_TRAIN;
    Grusoft::GRander rand_branch;

    SKDU_params params;
    string resident_list = "";

   public:
    RLSchedule(EDGE_DEVICES *hED, const CLI_params &config, int flag) : hDevices(hED) { rand_branch.Init(654321); }
    virtual ~RLSchedule() {}
    virtual void BeforeStart(int flag = 0x0);
    virtual bool Planning(int flag = 0x0);
    virtual bool Verify(int flag = 0x0) {
        assert(0);
        return false;
    }
    virtual int BeforeNextStep(int flag = 0x0) {
        assert(0);
        return 0x0;
    }
    bool SetPhase(LIFE_PHASE phase_, int flag = 0x0) {
        phase = phase_;
        return true;
    }
    virtual TASK_STATUS GetStatus(int step, void *hObj, int flag) { return FLIP; }
    virtual void Dump(int typ) const {}

    virtual bool isSwitchFuyou(int iter, int flag = 0x0);
    virtual bool ExploreOptimization(int iter, int flag = 0x0);
    // if type==1 return curBraches, otherwise, return allBraches
    virtual int nFuyou(int type) {
        int nB = fuyouSwarm.size();
        if (type == 1) {
            nB = ActiveFuyous().size();
        }
        assert(nB >= 0);
        return nB;
    }

    vector<TaskNode *> curTasks(int flag = 0x0) {
        assert(afu != nullptr);
        return afu->Tasks();
    }
    friend class EDGE_DEVICES;
    friend class NLP_AutoRegressive;
    friend class Fish;
};
typedef shared_ptr<RLSchedule> hRLSchedule;

//  Resource limited scheduling of BackPropagation
class RLS_BP : public RLSchedule {
   protected:
    int T_fore = -1, T_back = -1;
    int nT_guoke   = 0;
    size_t szGuoke = 0;
    std::map<hGensor, enum TASK_STATUS> tMaps;
    virtual bool UpdateBackbone(int iter, int flag = 0x0);
    virtual bool isUpdateBatch(int iter, int flag = 0x0);

   public:
    RLS_BP(EDGE_DEVICES *hED, const CLI_params &config, int flag);
    virtual ~RLS_BP() {}
    virtual void Init(Fish *hF, std::vector<shared_ptr<GeNeuron>> backbons, int flag = 0x0);
    virtual bool InitGUOKE(int flag = 0x0);
    virtual bool InitBranch(int flag = 0x0);
    virtual bool Prepare(int iter, int flag = 0x0);

    virtual TASK_STATUS GetTensorStatus(int step, hGTensor tenosr, int flag = 0x0);
    virtual TASK_STATUS SetTensorStatus(int step, hGTensor tenosr, TASK_STATUS sta, int flag = 0x0);
    TASK_STATUS GetStatus(int step, void *hObj, int flag) override;
    bool isResident(GeNeuron *neuron, int flag = 0x0);

    bool Planning(int flag = 0x0) override;
    bool Verify(int flag = 0x0) override;
    int BeforeNextStep(int flag = 0x0) override;
    void Dump(int typ) const override;
    friend class EDGE_DEVICES;
    friend class GeNeuron;
    friend class Optimizer;
};