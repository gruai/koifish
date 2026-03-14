/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief Some cuda kernels
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
#include "./packedN.cuh"
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

// Kernel to compute row standard deviations of a matrix
template <typename T>
__global__ static void CU_RowStdDev(T* matrix, floatGama* rowStdDev, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x, col;
    if (row >= rows)
        return;
    float sum = 0.f, sum2 = 0.f;
    for (col = 0; col < cols; col++) {
        float a = CU_T2Float(matrix + row * cols + col);
        sum += a, sum2 += a * a;
    }
    float variance = (sum2 / cols) - ((sum / cols) * (sum / cols));
    rowStdDev[row] = (floatGama)(sqrtf(variance));
}
template <typename T>
__global__ static void CU_ColStdDev(T* matrix, floatGama* colStdDev, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x, row;
    if (col >= cols)
        return;
    float sum = 0.f, sum2 = 0.f;
    for (row = 0; row < rows; row++) {
        float a = CU_T2Float(matrix + row * cols + col);
        sum += a, sum2 += a * a;
    }
    float variance = (sum2 / rows) - ((sum / rows) * (sum / rows));
    colStdDev[col] = (floatGama)(sqrtf(variance));
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

// grid: n_heads, block: 1
template <typename T>
__global__ static void CU_softmax_multihead(T* att, int pos, int seq_len) {
    int h = blockIdx.x;

    T* scores = att + (size_t)h * seq_len;
    int len   = pos + 1;

    // find max value for numerical stability
    // float max_val = -HUGE_VALF;
    T max_val = -1e9f;
    for (int i = 0; i < len; i++) {
        if (scores[i] > max_val) {
            max_val = scores[i];
        }
    }

    float sum = 0.0f, a;
    for (int i = 0; i < len; i++) {  // exp and sum
        a         = expf(scores[i] - max_val), sum += a;
        scores[i] = a;
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) {  // normalize
        scores[i] *= inv_sum;
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
    May faster by call blas:
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

    // x[index] = (T)alpha * x[index] + (T)beta * y[index];
    x[index] = (T)(alpha * (float)x[index] + beta * (float)y[index]);
    // x[index] = __hmul((T)alpha, x[index]) + __hmul((T)beta, y[index]);
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

        x[no] = (T)((float)x[no] + (social * r) * ((float)gBest[no] - (float)x[no]));
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
            // x[no] = __hadd(x[no], (T)a);
            x[no] = (T)((float)(x[no]) + a);
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
    typ* d_results = nullptr;
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
    float a = CU_T2Float(A + pos);
    float sum =
        CU_BlockSum<THREAD_TILE_M * THREAD_TILE_N>(a, true);  // nWrapT<=WARP_SIZE ? warpReduceSum<nWrapT>(a) : blockReduce_v0<warpReduceSum<nWrapT>>(a, true);
    if (tid == 0) {
        // if (idrow == 0 && idcol == 0) {
        //     int debug = 0;  //-0.005210
        // }
        gama[gpos] = sum / TM / TN;
    }
    if (isOverwrite) {
        A[pos] = CU_16BF2T<T>(gama + gpos);
    }
}
template <typename T>
__global__ static void CU_Tile2X_(T* A, floatGama* gama, float T_x, int M, int N, unsigned int seed, int trans = 1) {
    const int TM = THREAD_TILE_M, TN = THREAD_TILE_N;  //, thread_num = blockDim.x;
    int tid = threadIdx.x, idrow, idcol;
    idrow   = blockIdx.x * TM + tid / TM;
    idcol   = blockIdx.y * TN + tid % TM;
    if (idrow >= M || idcol >= N)
        return;  // guard
    fnPOS pA = trans == 0 ? fnCR2POS : fnRC2POS;
    int pos = pA(idrow, idcol, M, N), gpos = blockIdx.x * gridDim.y + blockIdx.y;
    A[pos] = CU_16BF2T<T>(gama + gpos);  // gama[gpos];
    //  same
    // float fgama = gama[gpos];
    // A[pos] = CU_Float2T<T>(fgama,seed+pos);
}

template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_OFF(float* out, T* A, T* B, size_t N, int flag) {
    int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
    if (idx >= N) {
        return;  // guard
    }

    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    assert(NUM_WARPS <= WARP_SIZE);
    float a         = (idx < N) ? (float)A[idx] - (float)B[idx] : 0.0;
    float sum       = a * a;
    float block_sum = CU_BlockSum<CU_T4B_SMALL>(sum, true);  // blockReduce_v0<warpReduceSum>(sum, true);
    if (tid == 0)
        atomicAdd(out, block_sum);
}

template <typename T>
__global__ static void CU_Float2F8(const T* src, f8e5* dst, size_t N, int seed, int flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) {
        return;  // guard
    }
    dst[tid] = f8e5(src[tid]);
}
template <>
__global__ void CU_Float2F8<bf16>(const bf16* src, f8e5* dst, size_t N, int seed, int flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) {
        return;  // guard
    }
    dst[tid] = CU_16BF2T<f8e5>(src + tid, seed);
}

template <typename T>
__global__ static void CU_F82Float(const f8e5* src, T* dst, size_t N, int seed, int flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) {
        return;  // guard
    }
    float a  = CU_T2Float<f8e5>(src + tid);
    dst[tid] = T(a);
}

template <class T>
double CU_OFF_(T* A, T* B, size_t N, int flag = 0x0) {
    double res = DBL_MAX;
    float *d_a, a = 0;
    cudaMalloc(&d_a, sizeof(float)), cudaCheck(cudaMemset(d_a, 0, sizeof(float)));
    size_t dBLOCK = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
    CU_OFF<T><<<CEIL_DIV(N, dBLOCK), dBLOCK, smemPB, main_stream>>>(d_a, A, B, N, flag);
    D2H(d_a, &a, sizeof(float), flag);
    res = sqrt(a);
    cudaFree(d_a);

    return res;
}

/**
 *      grid: n_heads, block: pos + 1 (up to 1024)
 *      attention_qk_kernel<<<N_HEADS, qk_threads_per_block>>>(hQwen->att, hQwen->q, layer_key_cache, pos, seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
 */
template <typename tpATT>
__global__ static void attention_qk_kernel(tpATT* att, bf16* q, bf16* k_cache, int pos, int seq_len, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.x, t = threadIdx.x;
    int kv_mul = n_heads / n_kv_heads, KV_DIM = n_kv_heads * head_dim;
    if (t > pos) {
        return;  // guard
    }
    bf16* q_head    = q + h * head_dim;
    int kv_head_idx = h / kv_mul;
    bf16* k_vec     = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * head_dim;
    if (0) {  // try 1-bit quant
        float sum = 0.0f, a, average = 0.0f;
        bf16* row = k_vec;  // q_head;
        for (int k = 0; k < head_dim; k++) {
            a = CU_T2Float(row + k);
            sum += fabs(a);
        }
        average = sum / head_dim;
        // if (average < 1.0e-5) {  // T_zeroRow
        //     att[(size_t)h * seq_len + t] = 0.0;
        //     return;
        // }
        // *gama = average;
        for (int k = 0; k < head_dim; k += 8) {
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
                if (1)                                           // isOverwrite
                    *row = bit ? (bf16)(average) : (bf16)(0.f);  //(bf16)(-average);
            }
            // terns[k / 8] = tbyte;
        }
    }
    float score = 0.0f;
    for (int i = 0; i < head_dim / 2; i++) {
        __nv_bfloat162 q_pair = reinterpret_cast<const __nv_bfloat162*>(q_head)[i];
        __nv_bfloat162 k_pair = reinterpret_cast<const __nv_bfloat162*>(k_vec)[i];

        float2 q_vals = __bfloat1622float2(q_pair);
        float2 k_vals = __bfloat1622float2(k_pair);

        // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
        score = __fmaf_rn(q_vals.x, k_vals.x, score);  //  _rn= Round to nearest even (IEEE-754 default)
        score = __fmaf_rn(q_vals.y, k_vals.y, score);
    }

    score /= sqrtf((float)head_dim);
    att[(size_t)h * seq_len + t] = score;
}

template <typename tpATT>
__global__ static void attention_qk_kernel_v2(tpATT* att, bf16* q, bf16* k_cache, int pos, int seq_len, int n_heads, int n_kv_heads, int head_dim) {
    int hid = blockIdx.x, token = blockIdx.y, tid = threadIdx.x;
    int kv_mul = n_heads / n_kv_heads, kv_head_idx = hid / kv_mul, KV_DIM = n_kv_heads * head_dim;

    bf16* q_head = q + hid * head_dim;
    bf16* k_vec  = k_cache + (size_t)token * KV_DIM + (size_t)kv_head_idx * head_dim;
    float a      = q_head[tid] * k_vec[tid];
    float score  = blockReduce_v0<warpReduceSum>(a, true);
    if (tid == 0) {
        score /= sqrtf((float)head_dim);
        att[hid * seq_len + token] = score;
    }
}

template <typename tpATT>
__global__ static void attention_v_kernel(bf16* out, const tpATT* att, const bf16* v_cache, int pos, int seq_len, int n_heads, int n_kv_heads, int head_dim) {
    // grid: n_heads, block: head_dim
    int h      = blockIdx.x;
    int i      = threadIdx.x;  // idx within the head dimension
    int kv_mul = n_heads / n_kv_heads, KV_DIM = n_kv_heads * head_dim;

    bf16* out_head        = out + (size_t)h * head_dim;
    const tpATT* att_head = att + (size_t)h * seq_len;
    int kv_head_idx       = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const bf16* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * head_dim;

        // weighted_sum += att_head[t] * __bfloat162float(v_vec[i]);
        weighted_sum = __fmaf_rn(att_head[t], __bfloat162float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2bfloat16_rn(weighted_sum);
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

template <class T>
__global__ void CU_ternary_online(T* mat, int M, int N, int seed = 0x0);
template <class T>
__global__ void CU_ternary2X_(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int seed = 0x0);
template <class T>
__global__ void CU_Q42X_lut(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q32X_(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q22X_(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q42X_RTN(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q42X_NF4(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q32X_NF3(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q32X_RTN(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_Q22X_RTN(floatGama* gama, const hBITARR terns, T* mat0, int M, int N, int rc_normal = 0x0, int seed = 0x0);
template <class T>
__global__ void CU_X2ternary_(floatGama* gama, T* mat0, char* terns, int M, int N, int bpe, bool isOverwrite = false);

// rope2 - add rope info to both q&k
//  blockIdx(B, T, n_head)
template <typename Typ>
__global__ void CU_rope2_v0(Typ* q, Typ* k, int pos, int N_HEADS, int N_KV_HEADS, int head_dim, float theta, int seed, int type = 0x0) {
    int h = blockIdx.z, j = threadIdx.x;
    assert(gridDim.z == N_HEADS);
    int nzHead = blockIdx.x * (gridDim.y * N_HEADS) + blockIdx.y * N_HEADS + h;
    if (h < N_HEADS && j < head_dim / 2) {
        Typ* q_head = q + nzHead * head_dim;
        //               1.0f / powf(theta, (float)(2 * tid) / head_dim);
        float inv_freq = 1.0f / powf(theta, (float)(j * 2) / (float)head_dim);
        if (pos < 0) {  //  (B, T, n_head)
            pos = blockIdx.y;
        }

        float angle = (float)pos * inv_freq, cos, sin;
        sincosf(angle, &sin, &cos);
        int j1 = j, j2 = j + head_dim / 2;
        // j1 = 2 * j, j2 = 2 * j + 1;
        if (type == 1) {
            sin = -sin;
        }
        float q_real = CU_T2Float(q_head + j1);
        float q_imag = CU_T2Float(q_head + j2);

        q_head[j1] = CU_Float2T<Typ>(q_real * cos - q_imag * sin, seed);  //__float2bfloat16_rn
        q_head[j2] = CU_Float2T<Typ>(q_real * sin + q_imag * cos, seed);  //__float2bfloat16_rn
        // if (nzHead == N_HEADS && j == 0) {
        //     nout("\t(%g,%g)=%g %g %g %g@<%d %d %d>\n", CU_T2Float(q_head+j1),CU_T2Float(q_head+j2),q_real, q_imag, sin,
        //     cos,blockIdx.x,blockIdx.y,blockIdx.z);
        // }
        if (h < N_KV_HEADS) {
            nzHead       = blockIdx.x * (gridDim.y * N_KV_HEADS) + blockIdx.y * N_KV_HEADS + h;
            Typ* k_head  = k + nzHead * head_dim;
            float k_real = CU_T2Float(k_head + j1), k_imag = CU_T2Float(k_head + j2);
            k_head[j1] = CU_Float2T<Typ>(k_real * cos - k_imag * sin, seed);  //__float2bfloat16_rn
            k_head[j2] = CU_Float2T<Typ>(k_real * sin + k_imag * cos, seed);  //__float2bfloat16_rn
        }
    }
}
/*  Somte ticks
    1.  Tensor cores are used to accelerate GEMMs. But it doesn’t mean there’s enough resources to execute other operation with CUDA cores. There are other
   resources that are shared among the on-chip resources, when kernels are called.
*/
template <typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* deltaIn, int B, int T, int OC, std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d  = (int)threadIdx.x;
    int warp_c  = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * X128::size;  // 64 at BF16

    int local_oc  = warp_c * X128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt     = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[X128::size];
    for (int k = 0; k < X128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if (global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            X128 packed_dout = load128(deltaIn + global_oc + idx * OC);
            for (int k = 0; k < X128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[X128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < X128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if (warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < X128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if (warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = (OutFloat)(a);
            }
        }
    }
}

__global__ void static reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % X128::size == 0);
    if (idx < n) {
        f128 acc;
        for (int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for (int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for (int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for (int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
        }
    }
}

// void CU_mm_blasLt_v0(floatX* d, const floatX* a, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int
// transB = 0,
//                   float alpha = 1.0, float beta = 0.0, bool backward = false)
template <class FloatC, class FloatA, class FloatB, class FloatBias>
void CU_mm_blasLt(FloatC* d, const FloatA* a, const FloatB* b, const FloatBias* bias, int m, int n, int k, const float* scale_a, const float* scale_b,
                  cudaStream_t stream = 0, int transA = 1, int transB = 0, bool accumulate = false, bool backward = false) {
    NVTX_RANGE_FN();
    bool has_bias              = (bias != nullptr);
    hBITARR workspace          = (hBITARR)GTensor::cudnn_workspace;
    std::size_t workspace_size = GTensor::cudnn_workspace_size;
    assert(workspace_size >= 32 * 1024 * 1024);
    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        _ERROR("All CU_mm_blas_ pointers must be aligned!\n");
        exit(KOIFISH_BLAS_UNALIGN);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose   = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, cuLibType<FloatA>, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, cuLibType<FloatA>, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, cuLibType<FloatB>, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, cuLibType<FloatB>, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, cuLibType<FloatC>, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, cuLibType<FloatC>, m, n, m));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias) {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = cuLibType<FloatBias>;  // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if (scale_a) {
        if (sizeof(FloatA) != 1) {
            throw std::runtime_error("Scaling A is only supported for FP8");
        }
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(&scale_a)));
    }
    if (scale_b) {
        if (sizeof(FloatB) != 1) {
            throw std::runtime_error("Scaling B is only supported for FP8");
        }
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(&scale_b)));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout, preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        _ERROR("No cuBLASLt algorithm@%d: m: %d, n: %d, k: %d, bias: %d\n", tpCuBLAS, n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one    = 1.f;
    float zero   = 0.f;
    float* alpha = &one;
    float* beta  = accumulate ? &one : &zero;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout, &heuristic.algo, workspace,
                               workspace_size, stream));
    cudaCheck(cudaGetLastError());

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

template <typename Typ>
__global__ void CU_rope_(Typ* out, Typ* inp, const Typ* freqs, float* stat_info, int B, int T, int Nq, int Nkv, int head_dim, bool isBack = false);

void CU_mm_(floatX* d, hGTensor gensor, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int transB = 0,
            float beta = 0.0, floatX* pre_gelu = NULL, bool backward = false);
//  y=W*x
void CU_mv_(floatX* y, const floatX* W, const floatX* x, int m, int n, float alpha = 1.0f, float beta = 0.0f);
// void matmul_backward(floatX* delta, floatX* dweight, floatX* dbias, floatX* deltaIn, floatX* inp, floatX* weight, float* dbias_buffer, int B, int T, int C,
//                      int OC, cudaStream_t stream, bool isTransW = false, floatX* pre_gelu = NULL, bool isAccumuDelta = false);