// qwen_model.cuh

#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_common.h"


constexpr int THREADS_PER_BLOCK = CU_T4B_SMALL;




template <int THREADS_PER_BLOCK, int HEAD_DIM>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fused_multi_rmsnorm_kernel(bf16* __restrict__ vecs, const bf16* __restrict__ weight, int num_vecs, float inv_head_dim, float EPS = 1e-6f) {
    // each block processes one vector/head
    const int vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs)
        return;

    const int t_idx     = threadIdx.x;
    const int vec_iters = HEAD_DIM / 2;

    bf16* vec_start = vecs + vec_idx * HEAD_DIM;

    const __nv_bfloat162* row_in    = reinterpret_cast<const __nv_bfloat162*>(vec_start);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out         = reinterpret_cast<__nv_bfloat162*>(vec_start);

    // 1. calculate sum of squares
    float lsum = 0.0f;
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        float2 v_fp32         = __bfloat1622float2(v_bf16);
        lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
    }

    // 2. reduce sum within the block
    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) {
        mul_val = rsqrtf(block_sum * inv_head_dim + EPS);
    }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_in_bf16     = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32             = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32         = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

// ================================================================
// RoPE
// ================================================================
__global__ void rope_kernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, int pos, int Q_DIM, int KV_DIM, int HEAD_DIM, float ROPE_THETA ) {
    // grid: Q_DIM / 2, block: THREADS_PER_BLOCK
    // each thread handles one pair of dimensions (i, i+1)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q_DIM / 2) {
        return;
    }

    int head_dim_idx = (i * 2) % HEAD_DIM;
    float freq       = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
    float val        = (float)pos * freq;
    float fcr, fci;
    sincosf(val, &fci, &fcr);

    // rotate Q
    __nv_bfloat162 q_val_bf16               = reinterpret_cast<__nv_bfloat162*>(q)[i];
    float2 q_val_fp32                       = __bfloat1622float2(q_val_bf16);
    float q0                                = q_val_fp32.x * fcr - q_val_fp32.y * fci;
    float q1                                = q_val_fp32.x * fci + q_val_fp32.y * fcr;
    reinterpret_cast<__nv_bfloat162*>(q)[i] = __float22bfloat162_rn(make_float2(q0, q1));

    if (i < KV_DIM / 2) {
        // rotate K
        __nv_bfloat162 k_val_bf16               = reinterpret_cast<__nv_bfloat162*>(k)[i];
        float2 k_val_fp32                       = __bfloat1622float2(k_val_bf16);
        float k0                                = k_val_fp32.x * fcr - k_val_fp32.y * fci;
        float k1                                = k_val_fp32.x * fci + k_val_fp32.y * fcr;
        reinterpret_cast<__nv_bfloat162*>(k)[i] = __float22bfloat162_rn(make_float2(k0, k1));
    }
}

// void rope_gpu_naive(__nv_bfloat16* q, __nv_bfloat16* k, int pos, int N_HEADS, int N_KV_HEADS, int HEAD_DIM, float ROPE_THETA) {
//     dim3 grid(N_HEADS, 1, 1);
//     dim3 block(HEAD_DIM / 2, 1, 1);

//     qwen_naive_rope_kernel<<<grid, block>>>(q, k, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, ROPE_THETA);
// }

// ================================================================
// softmax
// ================================================================




// ================================================================
// add residual
// ================================================================
__global__ void add_residual_kernel(__nv_bfloat16* x, const __nv_bfloat16* residual, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x_fp32      = __bfloat162float(x[i]);
        float res_fp32    = __bfloat162float(residual[i]);
        float result_fp32 = x_fp32 + res_fp32;
        x[i]              = __float2bfloat16_rn(result_fp32);
    }
}

void

add_residual_gpu(
    __nv_bfloat16* x, 
    const __nv_bfloat16* residual, 
    int size)
{
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_residual_kernel<<<grid_size, THREADS_PER_BLOCK>>>(x, residual, size);
}


// fp32_out should not share same address of bf16_in
__global__ void convert_bf16_to_fp32_kernel(__nv_bfloat16* bf16_in, float* fp32_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fp32_out[i] = __bfloat162float(bf16_in[i]);
    }
}
