/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Some trial/testing cuda kernels
 *  \author Yingshi Chen
 */
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "../../Manifold/gLLM.hpp"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"

extern cudaStream_t main_stream;

/**
 *  lite version: each thread for one row/head, No need sync!
 *  1. out maybe same as inp
 */
//  CU_rmsnorm_multihead
__global__ void CU_rms_forward_v0(bf16* __restrict__ out, float* __restrict__ rstd, const bf16* __restrict__ inp, const bf16* __restrict__ weight, int nTH,
                                  int ldTH, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTH) {
        return;
    }
    inp += idx * ldTH, out += idx * ldTH;
    float acc = 0.f;
    for (int c = 0; c < ldTH; c++) {
        float a = CU_T2Float(inp + c);
        acc += a * a;
    }
    float s = rsqrtf(acc / ldTH + eps);
    assert(!isnan(s) && !isinf(s));
    for (int c = 0; c < ldTH; c++) {
        // float n = s * (float)inp[c];
        out[c] = inp[c] * (bf16)s * weight[c];  // != inp[c]*weight[c]*(bf16)s
    }

    if (rstd != nullptr)
        rstd[idx] = s;
}

/**
 *  lite version: each thread for one row/head, No need sync!
 *  1. Y = x/(RMS(x)+ϵ)⊙w for each token/head in the forward pass
 *  2. dX0 maybe same as dY0
 */
template <typename Typ>
__global__ void CU_rms_backward_v0(Typ* dX0, Typ* dWeight0, Typ* scratch, const Typ* dY0, const Typ* X0, const Typ* weight0, const float* rstd0, int nTH,
                                   int ldTH, unsigned int seed, int flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTH) {
        return;
    }
    const Typ *w = weight0, *x = X0 + idx * ldTH, *dY = dY0 + idx * ldTH;
    float delta_avg = 0, rstd = rstd0[idx], acc = 0.f;
    for (int i = 0; i < ldTH; i++) {
        Typ xi = x[i] * CU_Float2T<Typ>(rstd, seed);
        Typ dw = dY[i] * xi;  // It is a total gradient, not an average gradient    ???
        // Typ dw = dY[i] * xi / (Typ)nTH;
        scratch[idx * ldTH + i] = dw;
        // atomicAdd(dWeight0 + i, dw);
        delta_avg += (float)(dY[i] * w[i] * xi);

        // float a = CU_T2Float(x + i);
        // acc += a * a;
    }
    // float s = rsqrtf(acc / ldTH + 1.0e-6);
    // assert(s==rstd);

    delta_avg /= ldTH;
    Typ* dX = dX0 + idx * ldTH;
    for (int i = 0; i < ldTH; i++) {
        dX[i] = rstd * ((float)(dY[i] * w[i]) - (float)(x[i]) * rstd * delta_avg);
    }
    // dX[j1] = dy_real,       dX[j2] = dy_imag;
}

template <typename Typ>
__global__ void CU_rms_dw_v0(Typ* dW, const Typ* dW0, int nTH, int ldTH, int flag = 0x0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ldTH) {
        return;
    }
    // dW[idx] = (Typ)0.0;
    float sum      = 0.f;
    const Typ* cur = dW0 + idx;
    for (int i = 0; i < nTH; i++, cur += ldTH) {
        sum += CU_T2Float(cur);
    }
    dW[idx] = (Typ)sum;
}

hGTensor LayerNormal::cuFlow(hGTensor inpDelta, int flag) {  //,hGTensor deltaIn
    NVTX_RANGE_FN();
    const int block_size = 256;  //, N = B * T
    floatX *weight = ToX(w), *bias = ToX0(b), *in = ToX(inpDelta);
    if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE) && nHead == 0) {  //
        CU_rms_v2(ToX(out), ToX(inpDelta), weight, C);
        return out;
    }
    float *_mean = mean == nullptr ? nullptr : TO<float>(mean), *_rstd = TO<float>(rstd);
    if (isForward() || BIT_TEST(flag, F_REMATER)) {
        SetInp4Back(inpDelta);  //        inp            = inpDelta;
        floatX* devOut = isOnline ? in : ToX(out);
        if (isOnline) {
            // inp->Print(name + (BIT_TEST(flag, F_REMATER)? "_remater" : "_forw"), 0x0, -1);
        }
        if (mean == nullptr) {  // RMS
            // auto status = cudaFuncSetAttribute(CU_rms_forward<floatX, floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            // cudaCheck(cudaGetLastError());
            // CU_rms_forward<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _rstd, in, weight, nTH, ldTH, rms_eps);
            CU_rms_forward_v0<<<CEIL_DIV(nTH, block_size), block_size, 0x0, main_stream>>>(devOut, _rstd, in, weight, nTH, ldTH, rms_eps);
        } else {
            const int block_y = block_size / WARP_SIZE, grid_size = CEIL_DIV(nTH, block_y);
            size_t smem = (2 + block_y) * C * sizeof(floatX);
            auto status = cudaFuncSetAttribute(CU_lm_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaCheck(cudaGetLastError());
            if (status == cudaSuccess) {
                CU_lm_forward<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, nTH, ldTH);
            } else {
                // fall back to the version without shared memory
                const int grid_size_fb = CEIL_DIV(nTH, (block_size / WARP_SIZE));
                layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, nTH, ldTH);
            }
        }
        cudaCheck(cudaGetLastError());
        return out;
    } else {
        hGTensor deltaIn = inpDelta;
        assert(deltaIn != nullptr);
        float* scratch = (float*)GTensor::buff;
        deltaIn->Print("LN.delta.in", 0, 0);
        VerifyInp4Back(inp);
        if (mean == nullptr) {  // RMS
            if (isOnline) {     // for debug
                hGTensor deltaY = isOnline ? deltaIn : delta;
                assert(GTensor::buff_len >= nTH * ldTH);
                CU_rms_backward_v0<<<CEIL_DIV(nTH, block_size), block_size, 0x0, main_stream>>>(ToX(deltaY), ToG(w), (floatX*)scratch, ToX(deltaIn), ToX(inp),
                                                                                                ToX(w), _rstd, nTH, ldTH, 42, 0x0);
                CU_rms_dw_v0<<<1, ldTH, 0x0, main_stream>>>(ToG(w), (floatX*)scratch, nTH, ldTH);
                // w->Print(name, 1, -1);
            } else {
                const int block_size = 512, blocks_per_sm = 2, grid_size = blocks_per_sm * deviceProp.multiProcessorCount;
                size_t rounded_C = CEIL_DIV(C, (int)(32 * x128::size)) * (32 * x128::size);
                size_t smem      = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);
                auto status      = cudaFuncSetAttribute(CU_rms_backward<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
                cudaCheck(cudaGetLastError());
                //  dinp, dweight, scratch, dout, inp, weight,
                CU_rms_backward<<<grid_size, block_size, smem, main_stream>>>(ToX(delta), ToG(w), (hBITARR)(scratch), ToX(deltaIn), ToX(inp), ToX(w), _rstd,
                                                                              nullptr, nTH, ldTH);
                if (isOnline) {
                    assert(deltaIn->nByte() == nTH * ldTH * sizeof(floatX));
                    D2D(ToX(deltaIn), ToX(delta), deltaIn->nByte());
                }
            }
        } else
            layernorm_backward(ToX(delta), ToG(w), ToG0(b), scratch, ToX(deltaIn), ToX(inp), ToX(w), _mean, _rstd, B, T, C, main_stream);
        delta->Print("back of normal", 0, 0);
        return delta;
    }
}

/*
    In Qwen2.5/Qwen3 models, the RMS normalization formula for multi-head tensors in attention mechanisms is typically applied per head.
*/
template <typename Typ>
__global__ void CU_rope_rmsnormal_forw(Typ* qk, const Typ* weight, int pos, int qk_head, int head_dim, float theta, float EPS = 1e-6f, int flag = 0x0) {
    int h = blockIdx.z, j = threadIdx.x;
    assert(gridDim.z == qk_head);
    int nzHead = blockIdx.x * (gridDim.y * qk_head) + blockIdx.y * qk_head + h;
    if (h < qk_head && j < head_dim / 2) {
        Typ* q_head = qk + nzHead * head_dim;
        float rstd = 0.0f, x2 = 0.0f;
        int j1 = j, j2 = j + head_dim / 2;
        float q_real = CU_T2Float(q_head + j1);
        float q_imag = CU_T2Float(q_head + j2);
        if (weight != nullptr) {
            x2   = q_real * q_real + q_imag * q_imag;
            rstd = blockReduce_v0<warpReduceSum>(x2);
            rstd = rsqrtf(rstd / head_dim + EPS);
        }

        if (weight != nullptr) {
            float w1 = CU_T2Float(weight + j1), w2 = CU_T2Float(weight + j2);
            q_real *= rstd * w1;
            q_imag *= rstd * w2;
            // if (h == 0 && j == 0) {
            //     printf("rstd=%g (%g,%g)=>%g (%g,%g)=>%g\n", rstd, a1, w1, q_real, a2, w2, q_imag);
            // }
        } else {
            // if (h == 0 && j == 0) {
            //     printf("%g %g\n", q_real, q_imag);
            // }
        }

        float inv_freq = 1.0f / powf(theta, (float)(j * 2) / (float)head_dim);
        if (pos < 0) {  //  (B, T, n_head)
            pos = blockIdx.y;
        }
        float angle = (float)pos * inv_freq, cos, sin;
        sincosf(angle, &sin, &cos);
        q_head[j1] = __float2bfloat16_rn(q_real * cos - q_imag * sin);
        q_head[j2] = __float2bfloat16_rn(q_real * sin + q_imag * cos);
        // if (nzHead == N_HEADS && j == 0) {
        //     nout("\t(%g,%g)=%g %g %g %g@<%d %d %d>\n", CU_T2Float(q_head+j1),CU_T2Float(q_head+j2),q_real, q_imag, sin,
        //     cos,blockIdx.x,blockIdx.y,blockIdx.z);
        // }
    }
}

/*
    Fuse of normal&rope for backpropagation of each head
    1) RMS normal:  Y = x/(RMS(x)+ϵ)⊙w for each head in the forward pass
    2) ROPE:        (y_r,y_i) => (y_r',y_i')

    1. dX0 may same as dY0
*/
hGTensor ROPE::cuInfer(SelfAttention* hQKV, uint32_t seed, int pos, int flag) {
    hFish->GetBTC(B, T, C);
    size_t nToken = B * T;
    assert(nToken == 1);
    floatX *q = ToX(hQKV->Q.out), *k = ToX(hQKV->K.out);
    floatX *qW = hnQ == nullptr ? nullptr : ToX(hnQ->w), *kW = hnK == nullptr ? nullptr : ToX(hnK->w);

    dim3 blocks_q(B, T, n_head), blocks_k(B, T, n_head_kv), blocks(B, T);
    float rstd_eps = 1.0e-6;
    if (fuse_normal == 0 && hnQ != nullptr) {
        // hnQ->w->Print("qnw", 0x0, dump_flag), hnK->w->Print("knw", 0x0, dump_flag);
        hnQ->cuFlow(hQKV->Q.out);  // CU_rms_forward_v0<<<n_head, 1>>>(q, nullptr, q, qW, n_head, head_dim, rstd_eps);
        hnK->cuFlow(hQKV->K.out);  // CU_rms_forward_v0<<<n_head_kv, 1>>>(k, nullptr, k, kW, n_head_kv, head_dim, rstd_eps);
        qW = nullptr, kW = nullptr;
    }
    hQKV->Q.out->Print("Q.norm", 0x0, dump_flag, nToken * q_dim);
    hQKV->K.out->Print("K.norm", 0x0, dump_flag, nToken * kv_dim);

    assert(n_head_kv <= n_head);  // so blocks_k is in blocks_q
    if (fuse_normal == 1) {
        CU_rope_rmsnormal_forw<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, qW, pos, n_head, head_dim, theta);
        CU_rope_rmsnormal_forw<floatX><<<blocks_k, dim3(head_dim / 2, 1, 1)>>>(k, kW, pos, n_head_kv, head_dim, theta);
    } else
        CU_rope2_v0<<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, k, pos, n_head, n_head_kv, head_dim, theta);
    // Q.out->Print("Q.rope", 0x0, dump_flag, nToken * q_dim), K.out->Print("K.rope", 0x0, dump_flag, nToken * kv_dim);
    return nullptr;
}

/*
    Fuse of rope & normal for backpropagation of each head
    1. in the forward pass
        1) RMS normal:  Y = x/(RMS(x)+ϵ)⊙w for each head
        2) ROPE:        (y_r,y_i) => (y_r',y_i')

    2. dX0 may same as dY0
*/
template <typename Typ>
__global__ void CU_rope_rmsnormal_back(Typ* dX0, Typ* dWeight0, const Typ* dY0, const Typ* qk, const Typ* weight0, int pos, int nToken, int qk_head,
                                       int head_dim, float theta, unsigned int seed, int flag = 0x0) {
    int h = blockIdx.z, j = threadIdx.x;
    assert(gridDim.z == qk_head);
    int head_id  = blockIdx.x * (gridDim.y * qk_head) + blockIdx.y * qk_head + h;
    int nAllHead = gridDim.x * gridDim.y * gridDim.z;
    assert(nToken = gridDim.x * gridDim.y);
    if (h < qk_head && j < head_dim / 2) {
        const Typ *dY = dY0 + head_id * head_dim, *x = qk + head_id * head_dim;
        Typ *dX = dX0 + head_id * head_dim, *dW = dWeight0, sw = (Typ)(1.0f / nToken);
        float rstd = 0.0f, x2 = 0.0f, EPS = 1.0e-6;
        int j1 = j, j2 = j + head_dim / 2;
        float dy_real = CU_T2Float(dY + j1), x_real = CU_T2Float(x + j1);
        float dy_imag = CU_T2Float(dY + j2), x_imag = CU_T2Float(x + j2);
        float inv_freq = 1.0f / powf(theta, (float)(j * 2) / (float)head_dim);
        if (pos < 0) {  //  (B, T, n_head)
            pos = blockIdx.y;
        }
        float angle = (float)pos * inv_freq, cos, sin;
        sincosf(angle, &sin, &cos);
        sin    = -sin;  // for back
        dX[j1] = __float2bfloat16_rn(dy_real * cos - dy_imag * sin);
        dX[j2] = __float2bfloat16_rn(dy_real * sin + dy_imag * cos);

        if (weight0 != nullptr) {
            x2   = x_real * x_real + x_imag * x_imag;
            rstd = blockReduce_v0<warpReduceSum>(x2);
            rstd = rsqrtf(rstd / head_dim + EPS);

            const Typ* w = weight0;
            Typ xr = CU_Float2T<Typ>(x_real * rstd, seed), xi = CU_Float2T<Typ>(x_imag * rstd, seed);
            // Typ dy1 = dX[j1] * xr * sw, dy2 = dX[j2] * xi * sw;
            // atomicAdd(dW + j1, dy1);  //  dW[j1] += (dY[j1] * xr * sw);
            // atomicAdd(dW + j2, dy2);  //  dW[j2] += (dY[j2] * xi * sw);

            x2              = dX[j1] * w[j1] * xr + dX[j2] * w[j2] * xi;
            float delta_sum = blockReduce_v0<warpReduceSum>(x2);
            dy_real         = rstd * ((float)(dX[j1] * w[j1]) - (float)(xr) / head_dim * delta_sum);
            dy_imag         = rstd * ((float)(dX[j2] * w[j2]) - (float)(xi) / head_dim * delta_sum);
            // dX[j1] = dy_real,       dX[j2] = dy_imag;
        }
    }
}

// fuse normal to rope, may reduce time
int ROPE::cuFlow(SelfAttention* hQKV, uint32_t seed, bool isFX, int flag) {
    if (hFish == nullptr)  // some models(GPT2) don't need rope
        return 0x0;

    INSPECT_THIS;
    hFish->GetBTC(B, T, C);
    dim3 blocks_q(B, T, n_head), blocks_k(B, T, n_head_kv), blocks(B, T);
    // size_t smemPB = 1024 * sizeof(float);
    floatX *q = ToX(hQKV->Q.out), *k = ToX(hQKV->K.out), *freqs = ToX(hSin);
    floatX *qW = hnQ == nullptr ? nullptr : ToX(hnQ->w), *kW = hnK == nullptr ? nullptr : ToX(hnK->w);
    PrintTensor<floatX>("Q.out", q, true, 1, 1, q_dim, 1, dump_flag);
    if (isForward() || BIT_TEST(flag, F_REMATER)) {
        if (fuse_normal == 0) {
            if (hnQ != nullptr) {
                hnQ->cuFlow(hQKV->Q.out, flag);
                hnK->cuFlow(hQKV->K.out, flag);
                hQKV->Q.out->Print("Q.norm", 0x0, dump_flag, B * T * q_dim);
                hQKV->K.out->Print("K.norm", 0x0, dump_flag, B * T * kv_dim);
            }
            CU_rope2_v0<<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, k, -1, n_head, n_head_kv, head_dim, theta);
        } else {
            CU_rope_rmsnormal_forw<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, qW, -1, n_head, head_dim, theta);
            CU_rope_rmsnormal_forw<floatX><<<blocks_k, dim3(head_dim / 2, 1, 1)>>>(k, kW, -1, n_head_kv, head_dim, theta);
        }
    } else {
        floatX *dQ = ToX(hQKV->deltaQ), *dK = ToX(hQKV->deltaK), *dQY = dQ, *dKY = dK;  // from back of QKV self-attention
        if (fuse_normal == 1) {                                                         // some strange bug
            floatX *dQw = hnQ == nullptr ? nullptr : ToG(hnQ->w), *dKw = hnK == nullptr ? nullptr : ToG(hnK->w);
            if (qW != nullptr) {
                assert(dQw != nullptr && dKw != nullptr);
                if (layid == 28) {
                    // hnQ->w->Print("normQ.w", 0x0, -1), hnQ->w->Print("normQ.w", 1, -1);
                }
            }
            CU_rope_rmsnormal_back<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(dQ, dQw, dQY, q, qW, -1, B * T, n_head, head_dim, theta, 42);
            if (layid == 28 && qW != nullptr) {
                hnQ->w->Print("normQ.w", 1, -1);
            }
            CU_rope_rmsnormal_back<floatX><<<blocks_k, dim3(head_dim / 2, 1, 1)>>>(dK, dKw, dKY, k, kW, -1, B * T, n_head_kv, head_dim, theta, 42);
        } else {
            CU_rope2_v0<<<blocks_q, head_dim / 2>>>(dQ, dK, -1, n_head, n_head_kv, head_dim, theta, 1);
            if (hnQ != nullptr) {
                hnK->cuFlow(hQKV->deltaK);
                hnQ->cuFlow(hQKV->deltaQ);
            }
        }
        //
        // SYNC_DEVICE();
    }
    // hQKV->Q.out->Print("Q.rope", 0x0, -1, C);  hQKV->K.out->Print("K.rope", 0x0, -1);
    PrintTensor<floatX>("q_0.rope", (floatX*)q, true, 1, 1, q_dim, 1, dump_flag);
    PrintTensor<floatX>("k_0.rope", (floatX*)k, true, 1, 1, kv_dim, 1, dump_flag);
    PrintTensor<floatX>("q_1.rope", (floatX*)q + q_dim, true, 1, 1, q_dim, 1, dump_flag);
    PrintTensor<floatX>("k_1.rope", (floatX*)k + kv_dim, true, 1, 1, kv_dim, 1, dump_flag);
    return 0x0;
}
