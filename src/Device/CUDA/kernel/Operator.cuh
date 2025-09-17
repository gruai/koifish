/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief From good job of CALM
 *  \author Yingshi Chen
 */
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <stdint.h>

#include "../Device/EDevice.hpp"
#include "../Tensor/GTensor.hpp"
#include "../cuda_common.h"
#include "./utils.cuh"

template <typename T, int NUM>
__device__ __inline__ T warpReduceMax(T* val, int thread_group_width = 32) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
#pragma unroll
        for (int mask = thread_group_width / 2; mask > 0; mask >>= 1) {
            val[i] = max(val[i], __shfl_xor_sync(0xffffffff, val[i], mask, 32));
        }
    }
    return (T)(0.0f);
}

/*
    It's not deterministic, why?
        atomicAdd in CUDA is not deterministic because it involves race conditions when multiple threads attempt to modify the same memory location
   simultaneously. only support CU_x2_<<<grid_size, block_size, 0, main_stream>>>
*/
template <class T, int NUM_THREADS = 256>
__global__ static void CU_x2_atomic(float* out, const T* x0, size_t N) {
    int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
    if (idx >= N) {
        return;
    }  // guard

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    assert(NUM_WARPS <= WARP_SIZE);
    // __shared__ float reduce_smem[NUM_WARPS];	// keep the data in register is enough for warp operaion.
    float a         = (idx < N) ? (float)(x0[idx]) : 0.0;
    float sum       = a * a;
    float block_sum = CU_BlockSum<NUM_THREADS>(sum, true);  // blockReduce_v0<warpReduceSum>(sum, true);
    if (tid == 0)
        atomicAdd(out, block_sum);
    // __syncthreads();
}

// Performs a deterministic x2
template <class T>
__global__ static void CU_x2_(float* out, const T* x0, size_t N) {
    size_t index      = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ldT        = blockDim.x * gridDim.x;
    float accumulator = 0.f, a, block_sum = 0;
    for (size_t i = index; i < N; i += ldT) {
        a = (float)x0[i];
        accumulator += a * a;
    }
    out[blockIdx.x] = blockReduce_v0<warpReduceSum>(accumulator);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0;
        for (size_t i = 0; i < blockDim.x; i++) {
            sum += out[i];
        }
        *out = sum;
    }
}

// __device__ inline  float4 bf16x4_to_float4(__nv_bfloat16x4 bf16_vec) {
//     float4 res;
//     float2 temp;
//     temp = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bf16_vec.x));
//     res.x = temp.x;    res.y = temp.y;
//     temp = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bf16_vec.z));
//     res.z = temp.x;    res.w = temp.y;

//     return res;
// }

// gf4 decoding: 8 3-bit values + 1 fp8 scale are packed in a 32-bit word
__device__ inline half cu_gf4_ff(uint32_t v, int k) {
    half s = fp8_e5m2_ff(v & 0xff) * half(-0.25f);  // we expect compiler to reuse this across multiple calls
    return half(int((v >> (8 + k * 3)) & 7) - 4) * s;
}

// regular mat*vec; naive and unoptimized (won't reach peak bw or flops)
template <typename T>
__device__ inline float matmul(float* x, T* w, int i, int n) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += float(w[i * n + j]) * x[j];
    }
    return val;
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for half weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, half* w, int i, int n) {
    int lane  = threadIdx.x % warpSize;
    float val = 0.0f;
    for (int j = lane * 2; j < n; j += warpSize * 2) {
        float2 ww = __half22float2(*(half2*)&w[i * n + j]);
        float2 xx = *(float2*)&x[j];
        val += ww.x * xx.x;
        val += ww.y * xx.y;
    }
    return warpreduce_sum(val);
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for fp8 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, __nv_fp8_e5m2* w, int i, int n) {
    int lane  = threadIdx.x % warpSize;
    float val = 0.0f;
    // use 64-bit loads instead of 32-bit loads to increase memory throughput on H100/A100
    // without this we are seeing lower throughput given the limited number of parallel warps in coop kernel
    // this is performance-neutral on 4090 but results in issues with x[] load coalescing (that are benign)
    if (0) {
        for (int j = lane * 8; j < n; j += warpSize * 8) {
            ablock<__nv_fp8x4_e5m2, 2> wwp = *(ablock<__nv_fp8x4_e5m2, 2>*)&w[i * n + j];
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                float4 ww = fp8x4_e5m2_ff(wwp.v[k]);
                float4 xx = *(float4*)&x[j + k * 4];
                val += ww.x * xx.x;
                val += ww.y * xx.y;
                val += ww.z * xx.z;
                val += ww.w * xx.w;
            }
        }
        return warpreduce_sum(val);
    } else {
        for (int j = lane * 4; j < n; j += warpSize * 4) {
            // ablock<__nv_fp8x4_e5m2, 1> wwp = *(ablock<__nv_fp8x4_e5m2, 1>*)&w[i * n + j];
            float4 ww = fp8x4_e5m2_ff((__nv_fp8x4_e5m2*)(w + i * n + j));
            float4 xx = {x[j], x[j + 1], x[j + 2], x[j + 3]};  //*(float4*)&x[j];
            val += ww.x * xx.x;
            val += ww.y * xx.y;
            val += ww.z * xx.z;
            val += ww.w * xx.w;
            // val += x[j] * ww.x + x[j+1] * ww.y + x[j+2] * ww.z + x[j+3] * ww.w;
        }
    }
    // for(int offset = 16; offset > 0; offset >>= 1) {
    //     val += __shfl_down_sync(0xffffffff, sum, offset);
    // }
    return warpreduce_sum(val);
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for gf4 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float matmul_warppar(float* x, uint32_t* w, int i, int n) {
    int lane = threadIdx.x % warpSize;
    if (n % (warpSize * 16) == 0) {
        float val = 0.0f;
        for (int j = lane * 8; j < n; j += warpSize * 16) {
            uint32_t wg0 = w[i * n / 8 + j / 8];
            uint32_t wg1 = w[i * n / 8 + j / 8 + warpSize];

            ablock<float, 8> xx0 = *(ablock<float, 8>*)&x[j];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                val += float(cu_gf4_ff(wg0, k)) * xx0.v[k];
            }

            ablock<float, 8> xx1 = *(ablock<float, 8>*)&x[j + warpSize * 8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                val += float(cu_gf4_ff(wg1, k)) * xx1.v[k];
            }
        }
        return warpreduce_sum(val);
    } else {
        float val = 0.0f;
        for (int j = lane * 8; j < n; j += warpSize * 8) {
            uint32_t wg = w[i * n / 8 + j / 8];

            ablock<float, 8> xx = *(ablock<float, 8>*)&x[j];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                val += float(cu_gf4_ff(wg, k)) * xx.v[k];
            }
        }
        return warpreduce_sum(val);
    }
}

template <typename T>
__device__ inline float embed(T* weight, int idx) {
    return float(weight[idx]);
}

__device__ inline float embed(uint32_t* weight, int idx) { return cu_gf4_ff(weight[idx / 8], idx % 8); }

template <typename T_out, typename T>
__global__ static void kernel_embed(T_out* o, T* weight, int token, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    assert(i < n);

    o[i] = embed(weight, token * n + i);
}

template <typename KVT>
__global__ static void kernel_rotate_sink(uint64_t, int kvd, KVT* key_cache, int head_dim, int kv_sink, float theta_log2, int seq_len, int rotary_dim) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    assert(i < kv_sink * kvd);

    int l = blockIdx.y;

    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : exp2f(-theta_log2 * (float)j_head / (float)rotary_dim);

    // rotate sink tokens forward to keep pace with non-sink tokens
    float fcr, fci;
    sincosf(freq, &fci, &fcr);

    size_t loff = (size_t)l * seq_len * kvd;
    KVT* kb     = key_cache + loff;

    // note: k layout is transposed / tiled to improve attn_score performance
    int t = i / kvd;
    int k = i % kvd;
    int o = t * 16 + seq_len * (k / 16) * 16 + (k % 16);

    float v0 = float(kb[o + 0]);
    float v1 = float(kb[o + 1]);

    float r0 = v0 * fcr - v1 * fci;
    float r1 = v0 * fci + v1 * fcr;

    kb[o + 0] = KVT(r0);
    kb[o + 1] = KVT(r1);
}

__device__ inline float cu_d_gelu(const float x) {
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param  = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanhf(sqrt_param * (x + x * x2mul));
    float dg1   = 0.5f * (1.0f + tan_h);
    float dg2   = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3   = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

template <typename T>
__device__ inline T CU_gelu(T x) {
    // const float sqrt_param = 0.79788456080286535587989211986876f;
    float xf = x, out = 0.5f * xf * (1.0f + tanhf(0.797885f * (xf + 0.044715f * xf * xf * xf)));
    return T(out);  // 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}
template <typename T>
__device__ inline T CU_silu(T x) {
    float xf = x, out = xf / (1.0f + expf(-xf));
    return T(out);
}

__device__ static void moe_gate_warp(float* moe_weights, int* moe_experts, float* weights, int experts, int active) {
    int i = threadIdx.x;

    // (unscaled) softmax across experts
    float w       = (i < experts) ? weights[i] : -FLT_MAX;
    float max_val = warpreduce_max(w);
    w             = expf(w - max_val);

    // weight in top 24 bits, index in bottom 8
    int wi = (__float_as_int(w) & 0xffffff00) | i;

    // top k within warp
    float sumw = 0.f;
    int acti   = -1;

    for (int k = 0; k < active; ++k) {
        int maxi = warpreduce_maxi(wi);

        sumw += __int_as_float(maxi);

        // keeps top weight in thread k, clears weight for thread with max thread to avoid re-selection
        acti = (i == k) ? maxi : acti;
        wi   = (wi == maxi) ? 0 : wi;
    }

    // write normalized weights
    if (i < active) {
        assert(acti >= 0);

        moe_experts[i] = acti & 0xff;
        moe_weights[i] = __int_as_float(acti) / sumw;
    }
}

__device__ inline float4 attn_load4(half* p) {
    ablock<__half2_raw, 2> h = *(ablock<__half2_raw, 2>*)p;
    float2 h0 = __half22float2(h.v[0]), h1 = __half22float2(h.v[1]);
    return {h0.x, h0.y, h1.x, h1.y};
}

__device__ inline float4 attn_load4(__nv_fp8_e5m2* p) { return fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)p); }

template <typename KVT, typename T>
__device__ inline float attn_score(KVT* kht, T* qh, int head_dim, int seq_len, int t, int off) {
    float score = 0.0f;
    for (int j = 0; j < head_dim; j += 16) {
        float4 kk4 = attn_load4(&kht[j * seq_len + t * 16 + off]);
        float4 qq  = {qh[j + off], qh[j + off + 1], qh[j + off + 2], qh[j + off + 3]};  //*(float4*)&qh[j + off];
        score += kk4.x * qq.x;
        score += kk4.y * qq.y;
        score += kk4.z * qq.z;
        score += kk4.w * qq.w;
    }

    return score;
}

template <typename KVT, typename T>
__device__ inline float attn_warpdot(KVT* val, T* atth, int kv_len) {
    // assert(0);
    int kv_len4 = kv_len & ~3;
    int lane    = threadIdx.x % warpSize;

    float res = 0.0f;
    float sum = 0.0f;
    for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
        float4 vv = attn_load4(&val[t]);
        float4 aa = {atth[t], atth[t + 1], atth[t + 2], atth[t + 3]};  //*(float4*)&atth[t];
        res += vv.x * aa.x;
        res += vv.y * aa.y;
        res += vv.z * aa.z;
        res += vv.w * aa.w;
        sum += aa.x + aa.y + aa.z + aa.w;
    }

    if (kv_len4 + lane < kv_len) {
        float a = atth[kv_len4 + lane];
        res += a * float(val[kv_len4 + lane]);
        sum += a;
    }

    res = warpreduce_sum(res);
    sum = warpreduce_sum(sum);

    return res / sum;
}

template <typename T>
__device__ static void CU_softmax_v0(T* xout, T* x, int size) {
    int i = threadIdx.x;

    // find max value per thread (for numerical stability)
    float max_val = -FLT_MAX;
    for (int j = i; j < size; j += blockDim.x) {
        max_val = max(max_val, x[j]);
    }

    // max across threads in block
    max_val = blockreduce_max(max_val);
    T a1    = max_val;
    // exp per thread
    for (int j = i; j < size; j += blockDim.x) {
        xout[j] = expf(x[j] - a1);
    }
}

#include <curand_kernel.h>
//	Initialization of the random generator state generally requires more registers and local memory than random number generation. It may be beneficial to
// separate calls to curand_init() and curand() into separate kernels for maximum performance.
__global__ static void CU_initrand(curandState* state, uint32_t seed, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        curand_init(seed, id, 0, &state[id]);
    }
}

template <typename typ>
__global__ void CU_disti_normal_generate(curandState* state, typ* results, int N, float devia) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        //	Normally distributed float with mean 0.0f and standard deviation 1.0f
        float rand_val = curand_normal(&state[id]) * devia;
        results[id]    = (typ)rand_val;
    }
}

template <typename typ>
__global__ void CU_disti_normal_N(curandState* state, typ* results, int N, int ldB, float devia) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        int no = id * ldB;
        for (int i = 0; i < ldB; i++, no++) {
            if (no >= N)
                continue;
            float rand_val = curand_normal(&state[id]) * devia;
            results[no]    = (typ)rand_val;
        }
    }
}
/*
    cublasGemmEx(cublas_handle, opT, opN, n, m, 1, &one, X, bf16, m, nullptr, bf16, 1, &zero, Xt, bf16, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    cudaMemcpy(X, Xt, sizeof(Tmv) * m * n, cudaMemcpyDeviceToDevice);  // cudaMemcpyAsync
*/
template <typename typ>
__global__ void CU_transpose(const typ* input, typ* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

template <class T>
__global__ static void CU_mix_(float alpha, T* x, float beta, const T* y, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    // size_t grid_width = blockDim.x * gridDim.x;
    if (index >= count)
        return;

    x[index] = (T)alpha * x[index] + (T)beta * y[index];
}

//  kernel of particle-swarm-optimization
template <class T>
__global__ static void CU_PSO_2D(curandState* state, float alpha, T* x, float social, const T* gBest, size_t N, int ldB) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, no = tid * ldB;
    for (int i = 0; i < ldB; i++, no++) {
        if (no >= N)
            continue;
        float r = (curand_normal(&state[tid]) + 3.f) / 6.f;
        if (r <= 0)
            continue;
        if (r > 1)
            r = 1;
        // results[no]    = (typ)rand_val;
        // x[no] = (T)alpha * x[no] + (T)(social*r) * (gBest[no]-x[no]);
        x[no] += (T)(social * r) * (gBest[no] - x[no]);
    }
}

//  todo - try Cauchy Mutation
template <class T>
__global__ static void CU_mutation_(curandState* state, float T_mutation, float T_scale, T* x, const T* y, size_t N, int ldB) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, no = tid * ldB;
    for (int i = 0; i < ldB; i++, no++) {
        if (no >= N)
            continue;

        int max_position = 1000000, pick = curand(&state[tid]) % max_position;
        // if(a < 3.0)     //  0.135%
        //     continue;
        if (pick < T_mutation * max_position) {
            float a = curand_normal(&state[tid]) * T_scale;  //  Gaussian (Normal) Mutation
            x[no] += a;
        }
    }
}

template <class T>
__global__ static void CU_crossover_(curandState* state, float T_cross, T* x, const T* y, size_t N, int ldB) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x, no = tid * ldB;
    for (int i = 0; i < ldB; i++, no++) {
        if (no >= N)
            continue;

        int max_position = 1000, pick = curand(&state[tid]) % max_position;
        // if(a < 3.0)     //  0.135%
        //     continue;
        if (pick < T_cross * max_position) {
            x[no] = y[no];
        }
    }
}

/*
    seed:   curand_init(seed, id, 0, &state[id]);
    results[id] = __float2bfloat16(rand_val); // convert to bf16
*/
template <typename typ>
void CU_disti_normal(int N, typ* out, float devia, uint32_t seed = 42, bool isToHost = false) {
    typ* d_results;
    curandState* d_states;
    if (isToHost) {
        cudaCheck(cudaMalloc(&d_results, N * sizeof(typ)));
    } else {
        d_results = out;
    }
    //  nRander=N use too many space!       sizeof(curandState)=48
    int ldB = 480, nRander = max(CEIL_DIV(N, ldB), 1);
    cudaCheck(cudaMalloc(&d_states, nRander * sizeof(curandState)));

    CU_initrand<<<CEIL_DIV(nRander, 256), 256>>>(d_states, seed, nRander);
    // CU_disti_normal_generate<typ><<<(N + 255) / 256, 256>>>(d_states, d_results, N, devia);
    CU_disti_normal_N<typ><<<CEIL_DIV(nRander, 256), 256>>>(d_states, d_results, N, ldB, devia);
    SYNC_DEVICE("disti_normal");  //    cudaCheck(cudaDeviceSynchronize());
    // Copy back results if needed
    // __nv_bfloat16 *h_results = new __nv_bfloat16[N];
    if (isToHost) {
        cudaMemcpy(out, d_results, N * sizeof(typ), cudaMemcpyDeviceToHost);
        cudaCheck(cudaFree(d_results));
    }

    cudaCheck(cudaFree(d_states));
    return;
}

// row-scaling  2 thrshold
template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_ternary_2thrshold(floatGama* gama, T* mat, int M, int N, int update) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid, ldJ = blockDim.x;
    float ta = 1.0, tb = (-1.0), t0 = 0.0;
    for (int j = tid; j < M; j += ldJ) {
        float sum_1 = 0.0f, sum_2 = 0.0f, a;
        int n_1 = 0, n_2 = 0;
        T* x0 = mat + j * N;
        for (int k = 0; k < N; k++) {
            a = CU_T2Float(x0 + k);
            if (a > 0) {
                sum_1 += a;
                n_1++;
            } else {
                sum_2 -= a;
                n_2++;
            }
        }

        if (update == QUANT_ALG::W_SCALE) {
            // gama[j] = average;
            ta = n_1 == 0 ? t0 : (sum_1 / n_1), tb = n_2 == 0 ? t0 : (-sum_2 / n_2);
        } else {
            gama[j] = 1.0f;
        }
        T xa = (T)ta, xb = (T)tb;
        for (int k = 0; k < N; k++) {
            a = CU_T2Float(x0 + k);
            if (a > ta / 2)
                x0[k] = xa;
            else if (a < tb / 2)
                x0[k] = xb;
            else {
                x0[k] = k % 2 == 0 ? xa : xb;
            }
            // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
        }
    }
    __syncthreads();
}

/*
template <class T>
__device__ inline void CU_ternary_row(floatGama* gama, T* row, int N, int update) {
    T ta = (T)1.0, tb = (T)(-1.0), t0 = (T)(0.0);
    float sum = 0.0f, a, average = 0.0f;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        sum += fabs(a);
    }
    average = (sum / (N)) + 1.0e-5;

    if (update == QUANT_ALG::W_SCALE) {
        // gama[idx] = average;
        ta = (T)(average), tb = (T)(-average);
    } else {
        // gama[idx] = 1.0f;
    }
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        if (a > average / 2)
            row[k] = ta;
        else if (a < -average / 2)
            row[k] = tb;
        else {
            row[k] = k % 2 == 0 ? ta : tb;
        }
        // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
    }
}*/
// row-scaling  & online update
template <class T>
__global__ static void CU_ternary_online(T* mat, int M, int N, int seed = 0x0) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid;
    if (idrow >= M)
        return;

    float sum = 0.0f, a, average = 0.0f;
    T* x0 = mat + idrow * N;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(x0 + k);
        sum += fabs(a);
    }
    average     = sum / N;
    float thrsh = average / 2;
    T ta = CU_Float2T<T>(average, seed), tb = CU_Float2T<T>(-average, seed);
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(x0 + k);
        if (a > thrsh)
            x0[k] = ta;
        else if (a < -thrsh)
            x0[k] = tb;
        else {
            x0[k] = k % 2 == 0 ? ta : tb;
        }
        // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
    }

    // __syncthreads();
}

template <class T>
__device__ inline void CU_X2ternary_row(floatGama* gama, T* row, char* terns, int N, bool isOverwrite = false, float T_zeroRow = 1.0e-5) {
    // T ta = (T)1.0, tb = (T)(-1.0), t0 = (T)(0.0);

    float sum = 0.0f, a, average = 0.0f;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        sum += fabs(a);
    }
    average = (sum / (N));
    if (average < T_zeroRow) {
        *gama = 0.0;
        return;
    }
    *gama = average;
    // ta = (T)(average), tb = (T)(-average);
    for (int k = 0; k < N; k += 8) {
        unsigned char tbyte = 0, bit;
        // #pragma unroll
        for (int bpos = 0; bpos < 8; bpos++, row++) {
            a = CU_T2Float(row);
            // bit = (a < -average / 2 || a > average / 2) ? 1 : 0; // would explode
            bit = (a > average / 2) ? 1 : 0;  // binary quant after Implicit RELU
            // if (a > average / 2)
            //     bit = 1;  // x0[pos] = ta;
            // else if (a < -average / 2)
            //     bit = 0;  // x0[pos] = tb;
            // else {
            //     bit = bpos % 2 == 0;  // x0[pos] = pos%2==0 ? ta : tb;
            // }
            tbyte |= bit << (7 - bpos);
            // x0[pos] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;    }
            if (isOverwrite)
                *row = bit ? (T)(average) : (T)(-average);
        }
        terns[k / 8] = tbyte;
    }
}

// row-scaling  1 thrshold
template <class T>
__global__ static void CU_X2ternary_(floatGama* gama, T* mat0, char* terns, int M, int N, int bpe, bool isOverwrite = false) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard
    CU_X2ternary_row(gama + idrow, mat0 + idrow * N, terns + (idrow * N) / 8, N, isOverwrite);

    // __syncthreads();
}

// row-scaling  1 thrshold
template <class T>
__global__ void CU_ternary2X_(floatGama* gama, const char* terns, T* mat0, int M, int N, int seed = 0x0) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard

    float average = gama[idrow];
    T* x0         = mat0 + idrow * N;
    if (average == 0) {
        memset(x0, 0x0, sizeof(T) * N);
        return;
    }
    T ta = (T)(average), tb = (T)(-average), t0 = (T)(0);
    // T ta = CU_Float2T<T>(average, seed), tb = CU_Float2T<T>(-average, seed);
    const char* tern = terns + (idrow * N) / 8;
    for (int k = 0; k < N; k += 8, tern++) {
        unsigned char tbyte = *tern;  // terns[(idrow * N + k) / 8];
#pragma unroll
        for (int bpos = 0; bpos < 8; bpos++, x0++) {
            // int idx = idrow * N + k + bpos;
            // if (idx == 0) {
            //     int debug = 0;
            // }
            bit = BYTE_bit(tbyte, bpos);  //(tbyte >> (7-bpos)) & 0x1;
            *x0 = bit ? ta : t0;          // binary quant after Implicit RELU
            // *x0 = bit ? ta : tb;
            // *x0 = bit ? (bpos%2==1 ? ta : tb) : t0;      // would explode
        }
    }

    // __syncthreads();
}

template <const int BM, const int BN, class T>
__global__ static void CU_X2Tile_v0(floatGama* gama, T* mat0, float T_x, bool isOverwrite = false, int trans = 1) {
    /*const int TM=THREAD_TILE_M, TN=THREAD_TILE_N;
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;
    int block_row_thread = BN / TN, block_col_thread = BM / TM;
    assert(thread_num == block_row_thread * block_col_thread);
    int tx = (threadIdx.x % block_row_thread) * TN;  // Each thread for [ty:ty+tM,tx:tx+TN]
    int ty = (threadIdx.x / block_row_thread) * TM;
    __shared__ float As[BM * BK];
    fnPOS pA = transA == 0 ? fnCR2POS : fnRC2POS;
    A = A + pA(by * BM, 0, M, K);
    int a_tile_row = threadIdx.x / BK, a_tile_col = threadIdx.x % BK, a_tile_stride = thread_num / BK;  // 32
    int nG2A = (BM * BK / thread_num), curA = threadIdx.x * nG2A;
    int stepA = pA(0, BK, M, K), r, c;

#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA) {
#pragma unroll
        for (int i = curA; i < curA + nG2A; i++) {  //[BM:Bk]
            r = i / BK, c = i % BK;         //r = i % BM, c = i / BM;
            As[i] = A[pA(r, c, M, K)];  // CR2POS(r, c, BM, BK)
        }
        __syncthreads();
        float sum = 0;
        UNROLL for (int j = 0; j < TM; j++) {
            UNROLL for (int l = 0; l < TN; l++) {
                UNROLL for (int i = 0; i < BK; i++) sum += As[RC2POS((ty + j), i, BM, BK)] ;
            }
        }
        gama[RC2POS((by* BM+ty)/TM,(bx * BN+ty)/TN)] = sum/TM/TN;
        if(isOverwrite){

        }
        __syncthreads();
    }*/
}

template <typename T>
__global__ static void CU_X2Tile_(T* A, floatGama* gama, float T_x, int M, int N, int r0 = 0, int c0 = 0, bool isOverwrite = false, int trans = 1) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol;
    // const int nWrapT = std::min(WARP_SIZE,THREAD_TILE_M*THREAD_TILE_N);
    idrow = blockIdx.x * TM + tid / TM + r0;
    idcol = blockIdx.y * TN + tid % TM + c0;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), gpos = blockIdx.x * gridDim.y + blockIdx.y;  // 13825
    float a = A[pos];
    float sum =
        CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(a, true);  // nWrapT<=WARP_SIZE ? warpReduceSum<nWrapT>(a) : blockReduce_v0<warpReduceSum<nWrapT>>(a, true);
    if (tid == 0) {
        // if (idrow == 0 && idcol == 0) {
        //     int debug = 0;  //-0.005210
        // }
        gama[gpos] = sum / TM / TN;
    }
    if (isOverwrite) {
        A[pos] = gama[gpos];
    }
}
template <typename T>
__global__ static void CU_Tile2X_(T* A, floatGama* gama, float T_x, int M, int N, unsigned int seed, int trans = 1) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol;
    idrow   = blockIdx.x * TM + tid / TM;
    idcol   = blockIdx.y * TN + tid % TM;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), gpos = blockIdx.x * gridDim.y + blockIdx.y;
    A[pos] = gama[gpos];
    //  same
    // float fgama = gama[gpos];
    // A[pos] = CU_Float2T<T>(fgama,seed+pos);
}

template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_OFF(float* out, T* A, T* B, size_t N, int flag) {
    int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
    if (idx >= N) {
        return;
    }  // guard

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    assert(NUM_WARPS <= WARP_SIZE);
    float a         = (idx < N) ? (float)(A[idx] - B[idx]) : 0.0;
    float sum       = a * a;
    float block_sum = CU_BlockSum<CU_T4B_SMALL>(sum, true);  // blockReduce_v0<warpReduceSum>(sum, true);
    if (tid == 0)
        atomicAdd(out, block_sum);
}

template <class T>
double OFF_(T* A, T* B, size_t N, bool isCU = true, int flag = 0x0) {
    double res = DBL_MAX;
    if (isCU) {
        float *d_a, a = 0;
        cudaMalloc(&d_a, sizeof(float)), cudaCheck(cudaMemset(d_a, 0, sizeof(float)));
        size_t dBLOCK = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
        CU_OFF<T><<<CEIL_DIV(N, dBLOCK), dBLOCK, smemPB, main_stream>>>(d_a, A, B, N, flag);
        D2H(d_a, &a, sizeof(float), flag);
        res = sqrt(a);
        cudaFree(d_a);
    } else {  //  cpu version
        assert(0);
    }
    return res;
}

void CU_abc(floatX* d, hGTensor gensor, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int transB = 0,
            float beta = 0.0, floatX* pre_gelu = NULL, bool backward = false);
void CU_mm_(floatX* d, hGTensor gensor, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int transB = 0,
            float beta = 0.0, floatX* pre_gelu = NULL, bool backward = false);
void CU_mm_blasLt(floatX* d, const floatX* a, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int transB = 0,
                  float alpha = 0.0, float beta = 0.0, floatX* pre_gelu = NULL, bool backward = false);
void matmul_backward(floatX* delta, floatX* dweight, floatX* dbias, floatX* deltaIn, floatX* inp, floatX* weight, float* dbias_buffer, int B, int T, int C,
                     int OC, cudaStream_t stream, bool isTransW = false, floatX* pre_gelu = NULL, bool isAccumuDelta = false);