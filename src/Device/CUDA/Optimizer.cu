/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 *
 *  \brief cuda kernel of Optimizer
 *  \author Yingshi Chen
 */
#include "./kernel/Operator.cuh"
// #include "./llm_c/sampler.h"
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "./kernel/utils.cuh"

// static void *grads_memory=nullptr;
// static float *m_memory=nullptr;
//*v_memory=nullptr;
//*master_weights=nullptr;
extern unsigned long long rng_state;

typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;
//  reset grad online
template <typename Tp, typename Tg>
__device__ void sgd_update(Tp* params, float* master_params_memory, Tg* grads_memory, size_t num_parameters, float learning_rate, float beta1, float beta2,
                           float beta1_correction, float beta2_correction, float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad      = grad_scale * (float)grads_memory[idx];
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params[idx];
    float param     = old_param - (learning_rate * grad + weight_decay * old_param);
    // stochastic_rounding(param, &params[idx], seed);
    params[idx]       = (Tp)(param);
    grads_memory[idx] = (Tp)(0.0);
    if (master_params_memory != NULL) {
        master_params_memory[idx] = param;
    }
}
//  reset grad online
template <typename Tp, typename Tg>
__global__ void CU_sgd(Tp* params, float* master_params_memory, Tg* grads_memory, size_t num_parameters, ptrdiff_t w_stride, ptrdiff_t g_stride,
                       ptrdiff_t s_stride, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps,
                       float weight_decay, float grad_scale, unsigned int seed) {
    sgd_update(params + blockIdx.y * w_stride, master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL, grads_memory + blockIdx.y * g_stride,
               num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed);
}

template <typename Tp, typename Tg>
__global__ void CU_sgdv(Tp* params, Tg* grads_memory, float* v_memory, size_t num_parameters, float learning_rate, float beta1, float beta2,
                        float beta1_correction, float beta2_correction, float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad    = grad_scale * (float)grads_memory[idx];
    float v       = v_memory[idx];
    v             = lerp(grad * grad, v, beta2);  // beta2*v+(1-beta2)*grad*grad;
    v_memory[idx] = v;
    v /= beta2_correction;  // v_hat
    float old_param = (float)params[idx];
    float param     = old_param - (learning_rate * (grad / (sqrtf(v) + eps) + weight_decay * old_param));
    // stochastic_rounding(param, &params[idx], seed);
    params[idx]       = (Tp)(param);
    grads_memory[idx] = (Tp)(0.0);
}
template <typename Tp, typename Tg>
__global__ void CU_lion_(Tp* params, Tg* grads_memory, float* m_memory, size_t num_parameters, float learning_rate, float beta1, float beta2, float eps,
                         float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad = grad_scale * (float)grads_memory[idx];
    float m    = m_memory[idx], c;
    /*  Cautious LION
        mask = (update * grad > 0).to(grad.dtype)
        mask = mask * (mask.numel() / (mask.sum() + 1))
    */
    c = lerp(grad, m, beta1);  // beta1*m+(1-beta1)*grad;
    // c                 = c > 0 ? 1 : c == 0 ? 0 : -1;
    c                 = c > eps ? 1 : c < -eps ? -1 : 0;  // ternary
    m_memory[idx]     = lerp(grad, m, beta2);             // beta2*m+(1-beta2)*grad;
    float old_param   = CU_T2Float(params + idx);
    params[idx]       = CU_Float2T<Tp>(old_param - learning_rate * (c + weight_decay * old_param), seed);
    grads_memory[idx] = (Tp)(0.0);
}

template <typename Tp, typename Tg>
__global__ void CU_adamw_(Tp* params, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                          float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                          float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad    = grad_scale * CU_T2Float(grads_memory + idx);
    float m       = m_memory[idx];
    float v       = v_memory[idx];
    m             = lerp(grad, m, beta1);
    m_memory[idx] = m;
    v             = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params[idx];
    float param     = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    //  stochastic_rounding(param, &params[idx], seed);
    params[idx] = CU_Float2T<Tp>(param, seed);
    // params[idx] = (Tp)(param);
    grads_memory[idx] = (Tp)(0.0);

    if (master_params_memory != NULL) {
        master_params_memory[idx] = param;
    }
}

template <typename Tp, typename Tg>
void Optimizer_update(Tp* params, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters, ptrdiff_t w_stride,
                      ptrdiff_t g_stride, ptrdiff_t s_stride, int num_slices, float learning_rate, float beta1, float beta2, int t, float eps,
                      float weight_decay, float grad_scale, float grad_norm, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size         = 512;
    int num_blocks         = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    assert(num_slices == 1);    // limited gpu memory
    if (m_memory == nullptr) {  // SGD,SGD_V
        if (v_memory == nullptr) {
            CU_sgd<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params, master_params_memory, grads_memory, num_parameters, w_stride, g_stride,
                                                                            s_stride, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps,
                                                                            weight_decay, grad_scale, seed);
        } else {
            CU_sgdv<<<num_blocks, block_size, 0, stream>>>(params, grads_memory, v_memory, num_parameters, learning_rate, beta1, beta2, beta1_correction,
                                                           beta2_correction, eps, weight_decay, grad_scale, seed);
        }
    } else {  // LION ADAMw
        if (v_memory == nullptr) {
            eps = grad_norm / num_parameters;
            CU_lion_<<<num_blocks, block_size, 0, stream>>>(params, grads_memory, m_memory, num_parameters, learning_rate, beta1, beta2, eps, weight_decay,
                                                            grad_scale, seed);
        } else {
            // CU_sgdv<<<num_blocks, block_size, 0, stream>>>(params, master_params_memory, grads_memory,
            //          v_memory, num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,grad_scale, seed);
            CU_adamw_<<<num_blocks, block_size, 0, stream>>>(params, master_params_memory, grads_memory, m_memory, v_memory, num_parameters, learning_rate,
                                                             beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed);
        }
    }
    cudaCheck(cudaGetLastError());
}

void Optimizer::ClearOnCUDA(int flag) {}
void Optimizer::InitOnCUDA(int flag) {
    ADAM_params_ adam = TrainParams().adam;
    GD_METHOD tpCurGD = tpGD;

    int num_slices = 1, C = _fish->config.nEmbed();
    // if (m_memory == NULL) {
    //     NvtxRange rng("InitOpt");

    //     if(tpGD!=SGD){
    //         _INFO("Optimizer \t cudaMalloc=%zu MiB for v\n", (adam.n_parameters * sizeof(float)) >> 20);
    //         cudaCheck(cudaMalloc((void**)&v_memory, adam.n_parameters * sizeof(float)));
    //         cudaCheck(cudaMemset(v_memory, 0, adam.n_parameters * sizeof(float)));
    //     }
    // }
    // if (TrainParams().opt_alloc_weight==1 && master_weights == NULL) {
    //     _INFO("Optimizer \t cudaMalloc=%zu MiB for parametrs(FP32)\n", (adam.n_parameters * sizeof(float)) >> 20);
    //     cudaCheck(cudaMalloc((void**)&master_weights, adam.n_parameters * sizeof(float)));
    // }
    size_t off = 0;
    for (auto tensor : opt_ps) {
        size_t nP = tensor->size(), grid_size = CEIL_DIV(nP, 512);
        auto& im = _fish->GetGensorInfo(tensor);
        if (tpGD == SGD_HYBRID) {
            tpCurGD = im.isAdam ? ADAMw : SGD;
        }
        // if(tpCurGD==ADAMw){
        //     // _INFO("Optimizer allocating %zu MiB for m\n", (adam.n_parameters * sizeof(float)) >> 20);
        //     cudaCheck(cudaMalloc((void**)&(im.gm), tensor->size() * sizeof(float)));
        //     cudaCheck(cudaMemset(im.gm, 0, tensor->size() * sizeof(float)));
        // }
        // if(master_weights!=nullptr){
        //     copy_and_cast_kernel<<<dim3(grid_size, num_slices), 512, 0, main_stream>>>(master_weights+off, ToX(tensor), nP,nP, nP);
        //     cudaCheck(cudaGetLastError());
        // }
        off += nP;
    }
}

int UpdateTensorParam_cuda(hGTensor tensor, Optimizer* hOPT, float& grad_norm, int flag) {
    CLI_params config = hOPT->_fish->config;
    ADAM_params_ adam = hOPT->TrainParams().adam;
    auto& im          = hOPT->_fish->GetGensorInfo(tensor);
    // GD_METHOD tpCurGD = hOPT->tpGD;
    // if(hOPT->tpGD==SGD_HYBRID){
    //     tpCurGD = im.isAdam ? ADAMw : SGD;
    // }
    float learning_rate = hOPT->LearningRate(), beta1 = adam.beta1, beta2 = adam.beta2, eps = adam.eps;
    int num_slices = 1, iter = hOPT->GetITER();
    unsigned int seed = hOPT->rRounding.RandInt32();  // random_u32(&rng_state);
    const char* name  = tensor->name;
    ShardInfo shard   = {0, tensor->size()};
    float wd          = adam.decay;  // we only want to weight decay the 2D tensors and leave all 1D tensors alone
    if (tensor->shape.size() == 1)
        wd = 0;

    floatX *param_ptr = ToX(tensor), *grad_ptr = ToG(tensor);
    ptrdiff_t opt_state_offset = tensor->offset;                     // multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;
    float *m_ptr = (float*)tensor->gm, *v_ptr = (float*)tensor->gv;  // v_memory==nullptr? nullptr : v_memory + opt_state_offset;
    float* master_ptr = NULL;                                        // why this would slow down converge?
    // if (master_weights != NULL && im.isAdam) {
    //     master_ptr = master_weights + opt_state_offset;
    // }

    if (adam.clip_alg != 0 || config.lars_ratio > 0) {
        grad_norm     = tNormOf(tensor, 0x0);
        tensor->gnorm = grad_norm;
        if (grad_norm > adam.gclip) {
            // _INFO("\tdelta|%s|=%g scale=%g\n",tensor->name,grad_norm,adam.gclip/grad_norm);
        }
    }
    float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
    if (config.lars_ratio > 0) {
        grad_scale = tensor->rLARS(grad_scale, config.lars_ratio, 0x0);
    }

    Optimizer_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr, shard.size, shard.size, shard.size, shard.size,
                     num_slices,  // num_parameters,ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices,
                     learning_rate, beta1, beta2, iter, eps, wd, grad_scale, grad_norm, seed, main_stream);

    cudaCheck(cudaGetLastError());
    return 0x0;
}

/*Deparecated
int RAW_update(std::vector<hGTensor>& tensors, Optimizer* hOPT, float& grad_norm, int alg, int flag) {
    CLI_params config = hOPT->_fish->config;
    ADAM_params_ adam = hOPT->TrainParams().adam;
    if (adam.clip_alg == 0)
        grad_norm = flag == 0x10002 ? 1.0e6 : tNormOf(tensors, 0x0);
    double gnorm_0 = grad_norm, gnorm_1 = 0;
    float learning_rate = hOPT->LearningRate();
    float beta1 = adam.beta1, beta2 = adam.beta2, eps = adam.eps, weight_decay = adam.decay * adam.alpha;
    NVTX_RANGE_FN();
    size_t np      = 0;
    int num_slices = 1, iter = hOPT->GetITER();

    // for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {        // generate a unique seed for each tensor
    for (auto tensor : tensors) {
        if (alg == 0) {
            UpdateTensorParam_cuda(tensor, hOPT, grad_norm, flag);
        } else {
            unsigned int seed = 42;  // random_u32(&rng_state);
            const char* name  = tensor->name;
            ShardInfo shard   = {0, tensor->size()};
            float wd          = weight_decay;  // we only want to weight decay the 2D tensors and leave all 1D tensors alone
            if (tensor->shape.size() == 1)
                wd = 0;
            // ptrdiff_t local_offset_full=0,local_offset_partial=tensor->offset;
            floatX* param_ptr = ToX(tensor);  //(floatX*)params + local_offset_full;
            floatX* grad_ptr  = ToG(tensor);  //(floatX*)grads_memory + local_offset_full;

            ptrdiff_t opt_state_offset = np;  // multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;

            // float* m_ptr = m_memory + opt_state_offset,* v_ptr = v_memory + opt_state_offset;
            float *m_ptr = (float*)tensor->gm, *v_ptr = (float*)tensor->gv;
            float* master_ptr = NULL;
            // if (master_weights != NULL) { master_ptr = master_weights + opt_state_offset; }

            if (adam.clip_alg != 0 || config.lars_ratio > 0) {
                grad_norm = tNormOf(tensor, 0x0);
                gnorm_1 += grad_norm * grad_norm;
            }
            float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
            // if( config.lars_ratio>0 && tensor->shape.size()>1){
            //     grad_scale = tensor->rLARS(config.lars_ratio,0x0);
            // }

            if (flag != 0x10001) {  // some debug
                Optimizer_update(param_ptr, master_ptr, grad_ptr, m_ptr, v_ptr, shard.size, shard.size, shard.size, shard.size,
                                 num_slices,  // num_parameters,ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices,
                                 learning_rate, beta1, beta2, iter, eps, wd, grad_scale, seed, main_stream);
            }
            cudaCheck(cudaGetLastError());
        }
        np += tensor->size();
    }
    // assert(fabs(gnorm_1-gnorm_0*gnorm_0)<1.0e-6*gnorm_1);       // verify
    assert(np == adam.n_parameters);
    cudaCheck(cudaDeviceSynchronize());
    return 0x0;
}
    */