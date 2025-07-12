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
#include "../Pipe.hpp"
#include "./kernel/utils.cuh"

extern unsigned long long rng_state;

typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;
//  reset grad online
template <typename Tp>
__device__ void sgd_update(Tp* params, float* tmp, Tp* grads0, size_t num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction,
                           float beta2_correction, float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad      = grad_scale * (float)grads0[idx];
    float old_param = (tmp != NULL) ? tmp[idx] : (float)params[idx];
    float param     = old_param - (learning_rate * grad + weight_decay * old_param);
    // stochastic_rounding(param, &params[idx], seed);
    params[idx] = (Tp)(param);
    grads0[idx] = (Tp)(0.0);
    if (tmp != NULL) {
        tmp[idx] = param;
    }
}
//  reset grad online
template <typename Tp>
__global__ void CU_sgd(Tp* params, float* tmp, Tp* grads0, size_t num_parameters, ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                       float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                       float grad_scale, unsigned int seed) {
    sgd_update(params + blockIdx.y * w_stride, tmp ? tmp + blockIdx.y * s_stride : NULL, grads0 + blockIdx.y * g_stride, num_parameters, learning_rate, beta1,
               beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale, seed);
}

template <typename Tp, typename Tmv>
__global__ void CU_sgdv(Tp* params, Tp* grads0, Tmv* gv, size_t num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction,
                        float beta2_correction, float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad = grad_scale * (float)grads0[idx];
    float v    = gv[idx];
    v          = lerp(grad * grad, v, beta2);  // beta2*v+(1-beta2)*grad*grad;
    gv[idx]    = v;
    v /= beta2_correction;  // v_hat
    float old_param = (float)params[idx];
    float param     = old_param - (learning_rate * (grad / (sqrtf(v) + eps) + weight_decay * old_param));
    // stochastic_rounding(param, &params[idx], seed);
    params[idx] = (Tp)(param);
    grads0[idx] = (Tp)(0.0);
}
template <typename Tp, typename Tmv>
__global__ void CU_lion_(Tp* params, Tp* grads0, Tmv* gm, size_t num_parameters, float learning_rate, float beta1, float beta2, float eps, float weight_decay,
                         float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad = grad_scale * (float)grads0[idx];
    float m    = gm[idx], c;
    /*  Cautious LION
        mask = (update * grad > 0).to(grad.dtype)
        mask = mask * (mask.numel() / (mask.sum() + 1))
    */
    c = lerp(grad, m, beta1);  // beta1*m+(1-beta1)*grad;
    // c                 = c > 0 ? 1 : c == 0 ? 0 : -1;
    c               = c > eps ? 1 : c < -eps ? -1 : 0;  // ternary
    gm[idx]         = lerp(grad, m, beta2);             // beta2*m+(1-beta2)*grad;
    float old_param = CU_T2Float(params + idx);
    params[idx]     = CU_Float2T<Tp>(old_param - learning_rate * (c + weight_decay * old_param), seed);
    grads0[idx]     = (Tp)(0.0);
}

template <typename Tp, typename Tmv>
__global__ void CU_adamw_(Tp* params, float* tmp, Tp* grads0, Tmv* gm, Tmv* gv, size_t num_parameters, float learning_rate, uint64_t flags, float beta1,
                          float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) {
        return;
    }  // guard

    float grad = grad_scale * CU_T2Float(grads0 + idx), m = gm[idx], v = gv[idx];
    m = lerp(grad, m, beta1), gm[idx] = m;
    v = lerp(grad * grad, v, beta2), gv[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    float old_param = (tmp != NULL) ? tmp[idx] : (float)params[idx];
    float step = m / (sqrtf(v) + eps), T_spike = 50;
    //  Automatic detection of training instability
    // step = step>T_spike ?  T_spike : step<-T_spike ? -T_spike : step;
    float param = old_param - learning_rate * weight_decay * old_param - learning_rate * step;
    //  stochastic_rounding(param, &params[idx], seed);
    params[idx] = CU_Float2T<Tp>(param, seed);
    // params[idx] = (Tp)(param);
    grads0[idx] = (Tp)(0.0);

    if (tmp != NULL) {
        tmp[idx] = param;
    }
}

template <typename Tp, typename Tmv>
__device__ inline float _adamw_idx(float old_param, const PIPE_Optimizer<Tp, Tmv>& pipe, float m, float v, int idx) {
    // float old_param = (float)pipe.params[idx];
    float step = m / (sqrtf(v) + pipe.eps);
    //  Automatic detection of training instability
    // step = step>T_spike ?  T_spike : step<-T_spike ? -T_spike : step;
    float param = old_param - pipe.learning_rate * pipe.weight_decay * old_param - pipe.learning_rate * step;
    //  stochastic_rounding(param, &params[idx], seed);
    param = CU_Float2T<Tp>(param, pipe.seed);
    // params[idx] = (Tp)(param);
    pipe.grads0[idx] = (Tp)(0.0);
    return param;
}

template <typename Tp, typename Tmv>
__global__ void CU_adamw_p(PIPE_Optimizer<Tp, Tmv> pipe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pipe.num_parameters) {
        return;
    }  // guard

    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.gm[idx], v = pipe.gv[idx];
    m = lerp(grad, m, pipe.beta1), pipe.gm[idx] = m;
    v = lerp(grad * grad, v, pipe.beta2), pipe.gv[idx] = v;
    m /= pipe.beta1_correction;  // m_hat
    v /= pipe.beta2_correction;  // v_hat
    pipe.params[idx] = _adamw_idx((float)pipe.params[idx], pipe, m, v, idx);
}

// row-major  slow versioin
template <typename Tp, typename Tmv>
__global__ void CU_adamw_bit(PIPE_Optimizer<Tp, Tmv> pipe) {
    int M = pipe.ne[0], N = pipe.ne[1], tid = threadIdx.x;
    int idrow = blockIdx.x * blockDim.x + tid, offset = idrow * N;
    if (idrow >= M)
        return;  // guard
    float average = pipe.gama_T[idrow];
    Tp ta = (Tp)(average), tb = (Tp)(-average);
    char* terns  = (char*)(pipe.params) + offset / 8;
    Tp* params_x = pipe.paramX + offset;
    for (int k = 0; k < N; k += 8, offset += 8) {
        unsigned char tbyte = terns[k / 8];
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
            int bit         = BYTE_bit(tbyte, kk);  //(tbyte >> (7-kk)) & 0x1;
            float old_param = bit ? ta : tb;
            // CU_Float2T<Tp>(bit ? ta : tb, pipe.seed);      //
            int idx    = offset + kk;
            float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.gm[idx], v = pipe.gv[idx];
            m = lerp(grad, m, pipe.beta1), pipe.gm[idx] = m;
            v = lerp(grad * grad, v, pipe.beta2), pipe.gv[idx] = v;
            // v = lerp(average * average, v, pipe.beta2), pipe.gv[idx] = v;
            m /= pipe.beta1_correction, v /= pipe.beta2_correction;
            params_x[k + kk] = _adamw_idx(old_param, pipe, m, v, idx);
        }
    }
    CU_X2ternary_row(pipe.gama_T + idrow, params_x, terns, N);
    // __syncthreads();
}

template <typename Tp, typename Tmv>
void Optimizer_update(PIPE_Optimizer<Tp, Tmv>& pipe, cudaStream_t stream) {
    cudaError_t err       = cudaSuccess;
    int64_t ne[4]         = {pipe.ne[0], pipe.ne[1], pipe.ne[2], pipe.ne[3]};
    int dBLOCK            = 512;  //  1024?
    int dGRID             = CEIL_DIV(pipe.num_parameters, dBLOCK);
    size_t smemPB         = 1024 * sizeof(float);
    pipe.beta1_correction = 1.0f - powf(pipe.beta1, pipe.iter);
    pipe.beta2_correction = 1.0f - powf(pipe.beta2, pipe.iter);
    void* kernelArgs[]    = {(void*)&pipe};
    if (pipe.gm == nullptr) {  // SGD,SGD_V
        // if (gv == nullptr) {
        //     CU_sgd<<<num_blocks, block_size, 0, stream>>>(params, tmp, grads0, num_parameters, w_stride, g_stride,
        //                                                                     s_stride, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps,
        //                                                                     weight_decay, grad_scale, seed);
        // } else {
        //     CU_sgdv<<<num_blocks, block_size, 0, stream>>>(params, grads0, gv, num_parameters, learning_rate, beta1, beta2, beta1_correction,
        //                                                    beta2_correction, eps, weight_decay, grad_scale, seed);
        // }
    } else {  // LION ADAMw
        if (pipe.gv == nullptr) {
            pipe.eps = pipe.grad_norm / pipe.num_parameters;
            // CU_lion_<<<num_blocks, block_size, 0, stream>>>(params, grads0, gm, num_parameters, learning_rate, beta1, beta2, eps, weight_decay,grad_scale,
            // seed);
        } else {
            if (pipe.isBitParam) {
                // PrintTensor<Tp>("grad", (Tp*)pipe.grads0, true, pipe.num_parameters, 1,1,1,-1);
                if (DEBUG.T_ternary == 1) {
                    CU_adamw_p<<<dGRID, dBLOCK, 0, stream>>>(pipe);
                    CU_ternary_online<<<CEIL_DIV(pipe.ne[0], dBLOCK), dBLOCK, smemPB, stream>>>(pipe.params, pipe.ne[0], pipe.ne[1]);
                    // PrintTensor<floatX>(pipe.tensor->name, (floatX*)pipe.params, true, pipe.ne[0], pipe.ne[1], pipe.ne[2], pipe.ne[3], -1);
                } else {
                    CU_adamw_bit<<<CEIL_DIV(ne[0], dBLOCK), dBLOCK, smemPB, stream>>>(pipe);
                    // pipe.tensor->GetDataX(dump_flag,"w1");
                }
            } else {
                // err = cudaLaunchCooperativeKernel((void*)CU_adamw_<Tp,Tmv>, dGRID, dBLOCK, kernelArgs, smemPB, main_stream);
                // cudaCheck(err);      "too many blocks in cooperative launch"
                CU_adamw_p<<<dGRID, dBLOCK, 0, stream>>>(pipe);
            }
            // {            //  deparecated version result=/home/cys/rnd/lic/log/gpt2/0703_adamw.info
            //     CU_adamw_<<<dGRID, dBLOCK, 0, stream>>>(pipe.params, pipe.tmp, pipe.grads0, pipe.gm, pipe.gv, pipe.num_parameters,
            //     pipe.learning_rate,pipe.flags,
            //                                                  pipe.beta1, pipe.beta2, pipe.beta1_correction, pipe.beta2_correction, pipe.eps,
            //                                                  pipe.weight_decay, pipe.grad_scale, pipe.seed);
            // }
        }
    }
    cudaCheck(cudaGetLastError());
}

void Optimizer::ClearOnCUDA(int flag) {}
void Optimizer::InitOnCUDA(int flag) {
    ADAM_params_ adam = TrainParams().adam;
    GD_METHOD tpCurGD = tpGD;

    int num_slices = 1, C = _fish->config.nEmbed();
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

//  Deprecated
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
    ptrdiff_t opt_state_offset = tensor->offset;                           // multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;
    floatMV *m_ptr = (floatMV*)tensor->gm, *v_ptr = (floatMV*)tensor->gv;  // gv==nullptr? nullptr : gv + opt_state_offset;
    float* master_ptr = NULL;                                              // why this would slow down converge?
    // if (master_weights != NULL && im.isAdam) {
    //     master_ptr = master_weights + opt_state_offset;
    // }

    if (adam.clip_alg != 0 || config.lars_ratio > 0) {
        grad_norm     = tensor->Length(1);      //    tNormOf(tensor, 0x0);
        tensor->gnorm = grad_norm;
        if (fabs(grad_norm) < 1.0e-10) {
            _INFO("\t|g|=0@%s!", tensor->name);
        }
        if (isnan(grad_norm)) {
            _INFO("!!! NAN |g|@%s !!!\n", tensor->name);
        }
        if (grad_norm > adam.gclip) {
            // _INFO("\tdelta|%s|=%g scale=%g\n",tensor->name,grad_norm,adam.gclip/grad_norm);
        }
    }
    float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
    if (config.lars_ratio > 0) {
        grad_scale = tensor->rLARS(grad_scale, config.lars_ratio, 0x0);
    }
    PIPE_Optimizer<floatX, floatMV> pipe(shard.size, shard.size, shard.size, shard.size, tensor->flags, learning_rate, beta1, beta2, iter, eps, wd, grad_scale,
                                         grad_norm, seed);
    pipe.Update(tensor.get());
    Optimizer_update(pipe, main_stream);

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
            floatX* grad_ptr  = ToG(tensor);  //(floatX*)grads0 + local_offset_full;

            ptrdiff_t opt_state_offset = np;  // multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;

            // float* m_ptr = gm + opt_state_offset,* v_ptr = gv + opt_state_offset;
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