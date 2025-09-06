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
__global__ void CU_muon_mG(PIPE_Muon<Tp, Tmv> pipe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pipe.num_parameters) {
        return;
    }  // guard
    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.mG[idx], x2;
    m = lerp(grad, m, pipe.mui);

    float G_norm = pipe.grad_scale;  //
    pipe.mG[idx] = m, x2 = m * m;
    float block_sum = blockReduce_v0<warpReduceSum>(x2, true);
    if (threadIdx.x == 0)
        atomicAdd(pipe.arrNorm, block_sum);
}

template <typename Tp, typename Tmv>
__global__ void CU_muon_update(PIPE_Muon<Tp, Tmv> pipe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pipe.num_parameters) {
        return;
    }  // guard
    float old_param = (float)pipe.params[idx], step = (float)pipe.X[idx];
    float param = old_param - pipe.learning_rate * pipe.weight_decay * old_param - pipe.learning_rate * step, x2 = param * param;
    pipe.params[idx] = CU_Float2T<Tp>(param, pipe.seed);
    pipe.grads0[idx] = (Tp)(0.0);
    float block_sum  = blockReduce_v0<warpReduceSum>(x2, true);
    if (threadIdx.x == 0)
        atomicAdd(pipe.arrNorm, block_sum);
}

template <typename Tp, typename Tmv>
__device__ inline float _adamw_idx(float old_param, const PIPE_Adamw<Tp, Tmv>& pipe, float& m, float& v, int idx) {
    m /= pipe.beta1_correction, v /= pipe.beta2_correction;  // m_hat    v_hat
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
__global__ void CU_adamw_p(PIPE_Adamw<Tp, Tmv> pipe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pipe.num_parameters) {
        return;
    }  // guard

    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.gm[idx], v = pipe.gv[idx];
    m = lerp(grad, m, pipe.beta1), pipe.gm[idx] = m;
    v = lerp(grad * grad, v, pipe.beta2), pipe.gv[idx] = v;
    // m /= pipe.beta1_correction;  // m_hat
    // v /= pipe.beta2_correction;  // v_hat
    float x = _adamw_idx((float)pipe.params[idx], pipe, m, v, idx), x2 = x * x;
    pipe.params[idx] = x;
    float block_sum  = blockReduce_v0<warpReduceSum>(x2, true);
    if (threadIdx.x==0)     //  idx == 0
        atomicAdd(pipe.arrNorm, block_sum);
}

// row-major  slow versioin
template <typename Tp, typename Tmv>
__global__ void CU_adamw_ternary(PIPE_Adamw<Tp, Tmv> pipe) {
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
            // m /= pipe.beta1_correction, v /= pipe.beta2_correction;
            params_x[k + kk] = _adamw_idx(old_param, pipe, m, v, idx);
        }
    }
    CU_X2ternary_row(pipe.gama_T + idrow, params_x, terns, N);
    // __syncthreads();
}

template <typename Tp, typename Tmv>
__global__ void CU_adamw_s(PIPE_Adamw<Tp, Tmv> pipe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pipe.num_parameters) {
        return;
    }  // guard

    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.gm[idx], v;
    v = lerp(grad * grad, m * m, pipe.beta2);
    m = lerp(grad, m, pipe.beta1), pipe.gm[idx] = m;
    // m /= pipe.beta1_correction;  // m_hat
    // v /= pipe.beta2_correction;  // v_hat
    float x = _adamw_idx((float)pipe.params[idx], pipe, m, v, idx), x2 = x * x;
    pipe.params[idx] = x;
    float block_sum  = blockReduce_v0<warpReduceSum>(x2, true);
    if (threadIdx.x == 0)
        atomicAdd(pipe.arrNorm, block_sum);
}

template <typename Tp, typename Tmv>
__global__ static void CU_adamw_Tile_v0(PIPE_Adamw<Tp, Tmv> pipe) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol, M = pipe.ne[0], N = pipe.ne[1], trans = 1;
    idrow = blockIdx.x * TM + tid / TM;
    idcol = blockIdx.y * TN + tid % TM;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), idx = pos, gpos = blockIdx.x * gridDim.y + blockIdx.y;
    float old_param = pipe.gama_T[gpos];
    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx), m = pipe.gm[idx], v = pipe.gv[idx];
    m = lerp(grad, m, pipe.beta1), pipe.gm[idx] = m;
    v = lerp(grad * grad, v, pipe.beta2), pipe.gv[idx] = v;
    // m /= pipe.beta1_correction, v /= pipe.beta2_correction;
    float a   = _adamw_idx(old_param, pipe, m, v, idx);
    float sum = blockReduce_v0<warpReduceSum>(a, true);
    if (tid == 0) {
        pipe.gama_T[gpos] = sum / TM / TN;
    }
}

#define RC2TILE(r, c) (((r) / THREAD_TILE_M) * gridDim.y + ((c) / THREAD_TILE_N))
//  all element in tile has one mv
template <typename Tp, typename Tmv>
__global__ static void CU_adamw_Tile(PIPE_Adamw<Tp, Tmv> pipe) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol, M = pipe.ne[0], N = pipe.ne[1], trans = 1;
    // const int nWrapT = std::min(WARP_SIZE,THREAD_TILE_M*THREAD_TILE_N);
    idrow = blockIdx.x * TM + tid / TM;
    idcol = blockIdx.y * TN + tid % TM;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), idx = pos, gpos = RC2TILE(idrow, idcol);  // blockIdx.x * gridDim.y + blockIdx.y;
    float old_param = pipe.gama_T[gpos], m = pipe.gm[gpos], v = pipe.gv[gpos];
    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx);
    float sum =
        CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(grad);  // nWrapT<=WARP_SIZE ? warpReduceSum<nWrapT>(grad) : blockReduce_v0<warpReduceSum>(grad, true);
    grad = sum / TM / TN;
    m = lerp(grad, m, pipe.beta1), pipe.gm[gpos] = m;
    v = lerp(grad * grad, v, pipe.beta2), pipe.gv[gpos] = v;
    float a = _adamw_idx(old_param, pipe, m, v, idx);
    sum     = CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(a);  // nWrapT<=WARP_SIZE ? warpReduceSum<nWrapT>(a) :blockReduce_v0<warpReduceSum>(a, true);
    if (tid == 0) {
        a                 = sum / TM / TN;
        pipe.gama_T[gpos] = a;  // CU_Float2T<Tp>(a, pipe.seed);   //
        atomicAdd(pipe.arrNorm, a * a * TM * TN);
    }
}

//  all element in tile has one mv
template <typename Tp, typename Tmv>
__global__ static void CU_adamw_Tile_RC(PIPE_Adamw<Tp, Tmv> pipe) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol, idrow_0, idcol_0, M = pipe.ne[0], N = pipe.ne[1], trans = 1;

    idrow = blockIdx.x * TM + tid / TM, idcol = blockIdx.y * TN + tid % TM;
    idrow_0 = blockIdx.x * TM + tid / TM + pipe.tile_r1;
    idcol_0 = blockIdx.y * TN + tid % TM + pipe.tile_c1;
    if (idrow >= M || idcol >= N)
        return;  // guard
    if (idrow_0 < 0)
        idrow_0 = 0;
    if (idrow_0 >= M)
        idrow_0 = M - 1;
    if (idcol_0 < 0)
        idcol_0 = 0;
    if (idcol_0 >= N)
        idcol_0 = N - 1;

    fnPOS pA   = trans == 0 ? fnCR2POS : fnRC2POS;
    int gpos_0 = RC2TILE(idrow_0, idcol_0);
    int pos_0 = pA(idrow_0, idcol_0, M, N), idx_0 = pos_0, gpos = RC2TILE(idrow, idcol);  // blockIdx.x * gridDim.y + blockIdx.y;
    if (gpos_0 != gpos) {
        int debug = 0;
    }
    float old_param = pipe.gama_T[gpos_0], m = pipe.gm[gpos_0], v = pipe.gv[gpos_0];
    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx_0);
    float sum  = CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(grad);  // blockReduce_v0<warpReduceSum>(grad, true);
    grad       = sum / TM / TN;
    m = lerp(grad, m, pipe.beta1), v = lerp(grad * grad, v, pipe.beta2);
    float sum_m = CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(m);  // blockReduce_v0<warpReduceSum>(m, true);
    float sum_v = CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(v);  // blockReduce_v0<warpReduceSum>(v, true);
    float a     = _adamw_idx(old_param, pipe, m, v, idx_0);
    sum         = CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(a);  // blockReduce_v0<warpReduceSum>(a, true);
    if (tid == 0) {
        assert(!(isnan(sum) || isinf(sum)));
        assert(!(isnan(sum_v) || isinf(sum_v)));
        assert(!(isnan(sum_m) || isinf(sum_m)));
        pipe.gama_T[gpos] = sum / TM / TN;
        pipe.gv[gpos]     = sum_v / TM / TN;
        pipe.gm[gpos]     = sum_m / TM / TN;
    }
}
//  each element in tile has different mv
template <typename Tp, typename Tmv>
__global__ static void CU_adamw_Tile_each_mv(PIPE_Adamw<Tp, Tmv> pipe) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol, M = pipe.ne[0], N = pipe.ne[1], trans = 1;
    idrow = blockIdx.x * TM + tid / TM;
    idcol = blockIdx.y * TN + tid % TM;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), idx = pos, gpos = blockIdx.x * gridDim.y + blockIdx.y;
    float old_param = pipe.gama_T[gpos], m = pipe.gm[gpos], v = pipe.gv[gpos];
    float grad = pipe.grad_scale * CU_T2Float(pipe.grads0 + idx);
    m = lerp(grad, m, pipe.beta1), v = lerp(grad * grad, v, pipe.beta2);

    float sum   = blockReduce_v0<warpReduceSum>(grad, true);
    float sum_m = blockReduce_v0<warpReduceSum>(m, true);
    float sum_v = blockReduce_v0<warpReduceSum>(v, true);
    float a     = _adamw_idx(old_param, pipe, m, v, idx);  //  m,v => m_hat,v_hat
    sum         = blockReduce_v0<warpReduceSum>(a, true);
    if (tid == 0) {
        pipe.gama_T[gpos] = sum / TM / TN;
        pipe.gv[gpos]     = sum_v / TM / TN;
        pipe.gm[gpos]     = sum_m / TM / TN;
    }
}

// bool Fuyou::Exploitation(hGensor cur, int flag) {
//     int nP = cur->size(), dT4B = 512 ,nF = cur->fuyous.size();  //
//     int dGRID = CEIL_DIV(nP, dT4B);
//     for (auto t : cur->fuyous) {
//         //  position[i] = alpha*A->position[i] + beta*B->position[i];
//         CU_mix_<<<dGRID, dT4B, 0, main_stream>>>(alpha, ToX(cur), beta, ToX(t),nP);
//     }
//     return true;
// }

bool Fuyou::Exploitation(hGensor tHead, hGensor tNext, int flag) {
    if (!tHead->is2D() || params.algorithm == Fuyou_params::NO_EVOL)
        return false;
    int nParam = tHead->size(), dT4B = 512, M = tHead->ne[0], N = tHead->ne[1], nRander = M;  //
    // int dGRID = CEIL_DIV(nParam, dT4B);
    int mGRID = CEIL_DIV(M, dT4B), pGRID = CEIL_DIV(nParam, dT4B);
    float T_scale = tNext->wnorm == 0 ? 1.0 : tNext->wnorm / M / N;
    curandState* d_states;
    cudaCheck(cudaMalloc(&d_states, nRander * sizeof(curandState)));
    seed = rander.RandU32();
    CU_initrand<<<CEIL_DIV(nRander, 256), 256>>>(d_states, seed, nRander);

    switch (params.algorithm) {
        case Fuyou_params::NO_EVOL:
            break;
        case Fuyou_params::GENE_MIX:
            CU_mix_<<<pGRID, dT4B, 0, main_stream>>>(params.alpha, ToX(tNext), 1.0 - params.alpha, ToX(tHead), nParam);
            break;
        case Fuyou_params::GENE_MUTATION:
            // CU_mutation_<<<mGRID, dT4B, 0, main_stream>>>(d_states, T_mutation, ToX(tNext), nParam, N);
            break;
        case Fuyou_params::PARTICLE_GENETIC:
            CU_PSO_2D<<<mGRID, dT4B, 0, main_stream>>>(d_states, params.alpha, ToX(tNext), params.social, ToX(tHead), nParam, N);
            // CU_mutation_<<<mGRID, dT4B, 0, main_stream>>>(d_states, params.T_mutation,T_scale, ToX(tNext), ToX(tHead), nParam, N);
            // why T_crossover=0.6 is still effective
            CU_crossover_<<<mGRID, dT4B, 0, main_stream>>>(d_states, params.T_crossover, ToX(tNext), ToX(tHead), nParam, N);
            break;
        case Fuyou_params::PARTICLE_SWARM:
        default:
            // CU_mix_<<<dGRID, dT4B, 0, main_stream>>>(alpha, ToX(tNext), beta, ToX(tHead), nP);
            CU_PSO_2D<<<mGRID, dT4B, 0, main_stream>>>(d_states, params.alpha, ToX(tNext), params.social, ToX(tHead), nParam, N);
            break;
    }
    cudaCheck(cudaFree(d_states));

    return true;
}

/*
from https://github.com/KellerJordan/Muon/blob/master/muon.py
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.
*/
template <typename Tp, typename Tmv>
void PIPE_Muon<Tp, Tmv>::CU_core(cudaStream_t stream, int flag) {
    int64_t m = this->ne[0], n = this->ne[1], ldG = m, lda = std::min(m, n);
    bool isAdamw = !this->tensor->is2D() || m > n;
    if (isAdamw) {
        PIPE_Adamw<Tp, Tmv>::CU_core(stream, flag);
        return;
    }

    float alpha = 1.0f, beta = 1.0f, zero = 0.f, one = 1.0f, xNrm = 0;
    int dT4B = 512, dGRID = CEIL_DIV(this->num_parameters, dT4B);
    D20(this->arrNorm, sizeof(float) * 1);
    // D2e(this->arrNorm, xNrm);
    CU_muon_mG<<<dGRID, dT4B, 0, stream>>>(*this);
    D2e(this->arrNorm, xNrm);
    alpha = beta = 1.0 / (xNrm + eps);
    bool isTrans = m > n;  //  ;
    if (isTrans) {
        //     G = G.T
    }

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    //X = G.bfloat16()    X /= (X.norm() + eps)
    cublasAxpyEx(cublas_handle, m * n, &alpha, CUDA_R_16BF, mG, CUDA_R_16BF, 1, X, CUDA_R_16BF, 1, CUDA_R_16BF);
    for (int i = 0; i < most_iter; i++) {
        //     A = X @ X.T
        cublasGemmEx(cublas_handle, opN, opT, m, n, n, &alpha, X, CUDA_R_16BF, m, X, CUDA_R_16BF, n, &zero, A, CUDA_R_16BF, lda, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT);
        //     B = b * A + c * A @ A
        cublasGemmEx(cublas_handle, opN, opN, lda, lda, lda, &c, A, CUDA_R_16BF, lda, A, CUDA_R_16BF, lda, &zero, B, CUDA_R_16BF, lda, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT);
        cublasAxpyEx(cublas_handle, lda * lda, &b, CUDA_R_16BF, A, CUDA_R_16BF, 1, B, CUDA_R_16BF, 1, CUDA_R_16BF);
        //     BX =  B @ X
        cublasGemmEx(cublas_handle, opN, opN, m, n, m, &a, B, CUDA_R_16BF, lda, X, CUDA_R_16BF, lda, &zero, BX, CUDA_R_16BF, ldG, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT);
        //  X = a*X + BX
        cublasAxpyEx(cublas_handle, m * n, &a, CUDA_R_16BF, X, CUDA_R_16BF, 1, BX, CUDA_R_16BF, 1, CUDA_R_16BF);
        cudaMemcpy(BX, X, sizeof(Tmv)*m*n, cudaMemcpyDeviceToDevice);   //cudaMemcpyAsync        
    }
    if (isTrans) {  // if G.size(0) > G.size(1):
        //     X = X.T
    }

    CU_muon_update<<<dGRID, dT4B, 0, stream>>>(*this);
    D2e(this->arrNorm, this->tensor->wnorm, 0x0), assert(isValidF(&(this->tensor->wnorm)));
    this->tensor->wnorm = sqrt(this->tensor->wnorm);
    cudaCheck(cudaGetLastError());
}
template struct PIPE_Muon<floatX, floatMV>;   // Force compilation
template struct PIPE_Adamw<floatX, floatMV>;  // Force compilation

template <typename Tp, typename Tmv>
void PIPE_Adamw<Tp, Tmv>::CU_core(cudaStream_t stream, int flag) {
    // cudaError_t err       = cudaSuccess;
    int64_t ne[4]    = {ne[0], ne[1], ne[2], ne[3]};
    int dT4B         = 512;  //  1024?
    int dGRID        = CEIL_DIV(num_parameters, dT4B);
    size_t smemPB    = 1024 * sizeof(float);
    beta1_correction = 1.0f - powf(beta1, iter);
    beta2_correction = 1.0f - powf(beta2, iter);

    D20(arrNorm, sizeof(float) * 1);
    if (gm == nullptr) {  // SGD,SGD_V
        assert(0);
    } else {  //   ADAM_S LION(locked!!!)
        if (gv == nullptr) {
            CU_adamw_s<<<dGRID, dT4B, 0, stream>>>(*this);
            //  eps = grad_norm / num_parameters;    for lion
            // CU_lion_<<<num_blocks, block_size, 0, stream>>>(params, grads0, gm, num_parameters, learning_rate, beta1, beta2, eps, weight_decay,grad_scale,
            // seed);
        } else {
            if (isBitParam) {
                // PrintTensor<Tp>("grad", (Tp*)grads0, true, num_parameters, 1,1,1,-1);
                switch (tensor->type) {
                    case typNUMBER::T_BINARY_TILE: {
                        dim3 dBlock(THREAD_TILE_M * THREAD_TILE_N), dGrid(CEIL_DIV(ne[0], THREAD_TILE_M), CEIL_DIV(ne[1], THREAD_TILE_N));
                        CU_adamw_Tile<<<dGrid, dBlock, smemPB, stream>>>(*this);
                    } break;
                    default:
                        if (DEBUG.T_ternary == 1) {
                            CU_adamw_p<<<dGRID, dT4B, 0, stream>>>(*this);
                            CU_ternary_online<<<CEIL_DIV(ne[0], dT4B), dT4B, smemPB, stream>>>(params, ne[0], ne[1]);
                            // PrintTensor<floatX>(tensor->name, (floatX*)params, true, ne[0], ne[1], ne[2], ne[3], -1);
                        } else {
                            CU_adamw_ternary<<<CEIL_DIV(ne[0], dT4B), dT4B, smemPB, stream>>>(*this);
                            // tensor->GetDataX(dump_flag,"w1");
                        }
                        break;
                }
            } else {  //  ADAMw
                //  void* kernelArgs[]    = {(void*)&pipe};
                // err = cudaLaunchCooperativeKernel((void*)CU_adamw_<Tp,Tmv>, dGRID, dT4B, kernelArgs, smemPB, main_stream);
                // cudaCheck(err);      "too many blocks in cooperative launch"
                CU_adamw_p<<<dGRID, dT4B, 0, stream>>>(*this);
            }
            D2e(arrNorm, tensor->wnorm, 0x0);
            assert(isValidF(&(tensor->wnorm)));
            tensor->wnorm = sqrt(tensor->wnorm);
            // tensor->Mutation();        //  need more test
        }
    }
    cudaCheck(cudaGetLastError());
}

void Optimizer::ClearOnCUDA(int flag) {}
void Optimizer::InitOnCUDA(int flag) {
    ADAM_params_ adam = TrainParams().adam;
    // GD_METHOD tpCurGD = tpGD;

    int C      = _fish->config.nEmbed();  // num_slices = 1,
    size_t off = 0;
    for (auto tensor : opt_ps) {
        size_t nP = tensor->size();  //, grid_size = CEIL_DIV(nP, 512);
        auto& im  = _fish->GetGensorInfo(tensor);
        if (tpGD == SGD_HYBRID) {
            // tpCurGD = im.isAdam ? ADAMw : SGD;
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
    CLI_params config   = hOPT->_fish->config;
    ADAM_params_ adam   = hOPT->TrainParams().adam;
    auto& im            = hOPT->_fish->GetGensorInfo(tensor);
    float learning_rate = hOPT->LearningRate(), beta1 = adam.beta1, beta2 = adam.beta2, eps = adam.eps;
    int iter          = hOPT->GetITER();              // num_slices = 1,
    unsigned int seed = hOPT->rRounding.RandInt32();  // random_u32(&rng_state);
    const char* name  = tensor->name;
    ShardInfo shard   = {0, tensor->size()};
    float wd          = adam.decay;  // we only want to weight decay the 2D tensors and leave all 1D tensors alone
    if (tensor->shape.size() == 1)
        wd = 0;

    floatX *param_ptr = ToX(tensor), *grad_ptr = ToG(tensor);
    ptrdiff_t opt_state_offset = tensor->offset;                           // multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;
    floatMV *m_ptr = (floatMV*)tensor->gm, *v_ptr = (floatMV*)tensor->gv;  // gv==nullptr? nullptr : gv + opt_state_offset;
    // float* master_ptr = NULL;                                              // why this would slow down converge?
    // if (master_weights != NULL && im.isAdam) {
    //     master_ptr = master_weights + opt_state_offset;
    // }

    if (adam.clip_alg != 0 || config.lars_ratio > 0) {
        grad_norm     = tensor->Length(1);  //    tNormOf(tensor, 0x0);
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
    PIPE_Adamw<floatX, floatMV> pipe(hOPT, tensor->flags, learning_rate, beta1, beta2, eps, wd);
    pipe.Update(tensor.get(), wd, grad_scale, seed);
    pipe.CU_core(main_stream);
    // Optimizer_update(pipe, main_stream);

    cudaCheck(cudaGetLastError());
    return 0x0;
}
