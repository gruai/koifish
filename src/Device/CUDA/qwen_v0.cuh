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

// ================================================================
// CUDA OPTIMIZED KERNELS
// ================================================================
// RMS Norm
// ================================================================
template <int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    rms_norm_kernel(__nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ weight, size_t D, float inv_dim, float EPS = 1e-6f) {
    const int t_idx     = threadIdx.x;
    const int vec_iters = D / 2;

    const __nv_bfloat162* row_in    = reinterpret_cast<const __nv_bfloat162*>(X);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out         = reinterpret_cast<__nv_bfloat162*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        // convert to fp32 for math
        float2 v_fp32 = __bfloat1622float2(v_bf16);

        // lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    __shared__ float mul_val;
    if (t_idx == 0) {
        float val    = __fmaf_rn(block_sum, inv_dim, EPS);
        float approx = __frsqrt_rn(val);
        // mul_val = approx * (1.5f - 0.5f * val * approx * approx);
        mul_val = rsqrtf(val);
    }
    __syncthreads();

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_in_bf16     = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32             = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32         = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        // convert back to BF16 for storing
        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

void CU_rms_v2(__nv_bfloat16* o, const __nv_bfloat16* x, const __nv_bfloat16* weight, int dim) {
    if (dim % 2 != 0) {
        fprintf(stderr, "FATAL: rmsnorm dim %d is not divisible by 2. Vectorized kernel cannot run.\n", dim);
        exit(EXIT_FAILURE);
    }
    // if dim > (THREADS_PER_BLOCK * some_threshold), a multi-block reduction might be needed,
    // but for typical dimensions up to 8192, a single block is sufficient and simpler.
    const int num_blocks = 1;

    rms_norm_kernel<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(o, x, weight, dim, 1.0f/dim);
}

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

// void qk_norm_fused_gpu(bf16* q, bf16* k, const bf16* q_norm_weight, const bf16* k_norm_weight) {
//     constexpr int QK_NORM_THREADS_PER_BLOCK = 64;

//     // launching ONE kernel for all query heads
//     fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(q, q_norm_weight, N_HEADS);

//     // launching ONE kernel for all key heads
//     fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(k, k_norm_weight, N_KV_HEADS);
// }

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

// void rope_gpu(__nv_bfloat16* q, __nv_bfloat16* k, int pos) {
//     int num_pairs = Q_DIM / 2;
//     int grid_size = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//     rope_kernel<<<grid_size, THREADS_PER_BLOCK>>>(q, k, pos);
// }

__global__ void qwen_naive_rope_kernel(bf16* q, bf16* k_cache_pos, int pos, int N_HEADS, int N_KV_HEADS, int HEAD_DIM, float ROPE_THETA) {
    // `blockIdx.x` will correspond to the head index 'h'
    int h = blockIdx.x;
    // `threadIdx.x` will correspond to the inner loop index 'j'
    int j = threadIdx.x;

    if (h < N_HEADS && j < HEAD_DIM / 2) {
        bf16* q_head = q + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val  = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float q_real = __bfloat162float(q_head[j]);
        float q_imag = __bfloat162float(q_head[j + HEAD_DIM / 2]);

        float q_rotated_real = q_real * fcr - q_imag * fci;
        float q_rotated_imag = q_real * fci + q_imag * fcr;

        q_head[j]                = __float2bfloat16_rn(q_rotated_real);
        q_head[j + HEAD_DIM / 2] = __float2bfloat16_rn(q_rotated_imag);
    }

    if (h < N_KV_HEADS && j < HEAD_DIM / 2) {
        bf16* k_head = k_cache_pos + h * HEAD_DIM;

        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val  = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        float k_real = __bfloat162float(k_head[j]);
        float k_imag = __bfloat162float(k_head[j + HEAD_DIM / 2]);

        // perform rotation in fp32
        float k_rotated_real = k_real * fcr - k_imag * fci;
        float k_rotated_imag = k_real * fci + k_imag * fcr;

        k_head[j]                = __float2bfloat16_rn(k_rotated_real);
        k_head[j + HEAD_DIM / 2] = __float2bfloat16_rn(k_rotated_imag);
    }
}

void rope_gpu_naive(__nv_bfloat16* q, __nv_bfloat16* k, int pos, int N_HEADS, int N_KV_HEADS, int HEAD_DIM, float ROPE_THETA) {
    dim3 grid(N_HEADS, 1, 1);
    dim3 block(HEAD_DIM / 2, 1, 1);

    qwen_naive_rope_kernel<<<grid, block>>>(q, k, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, ROPE_THETA);
}

// ================================================================
// softmax
// ================================================================
__global__ void softmax_kernel(float* att, int pos, int seq_len) {
    // grid: N_HEADS, block: 1
    int h = blockIdx.x;

    float* scores = att + (size_t)h * seq_len;
    int len       = pos + 1;

    // find max value for numerical stability
    // float max_val = -HUGE_VALF;
    float max_val = -1e9f;
    for (int i = 0; i < len; i++) {
        if (scores[i] > max_val) {
            max_val = scores[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    // normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) {
        scores[i] *= inv_sum;
    }
}

// ================================================================
// Attention
// ================================================================
__global__ void attention_qk_kernel(float* att, const bf16* q, const bf16* k_cache, int pos, int seq_len, int N_HEADS, int N_KV_HEADS, int HEAD_DIM) {
    // grid: N_HEADS, block: pos + 1 (up to 1024)
    int h      = blockIdx.x;
    int t      = threadIdx.x;
    int kv_mul = N_HEADS / N_KV_HEADS, KV_DIM = N_KV_HEADS * HEAD_DIM;
    if (t <= pos) {
        const bf16* q_head = q + h * HEAD_DIM;
        int kv_head_idx    = h / kv_mul;
        const bf16* k_vec  = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++) {
            __nv_bfloat162 q_pair = reinterpret_cast<const __nv_bfloat162*>(q_head)[i];
            __nv_bfloat162 k_pair = reinterpret_cast<const __nv_bfloat162*>(k_vec)[i];

            float2 q_vals = __bfloat1622float2(q_pair);
            float2 k_vals = __bfloat1622float2(k_pair);

            // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * seq_len + t] = score;
    }
}
__global__ void attention_v_kernel(bf16* out, const float* att, const bf16* v_cache, int pos, int seq_len, int N_HEADS, int N_KV_HEADS, int HEAD_DIM) {
    // grid: N_HEADS, block: HEAD_DIM
    int h      = blockIdx.x;
    int i      = threadIdx.x;  // idx within the head dimension
    int kv_mul = N_HEADS / N_KV_HEADS, KV_DIM = N_KV_HEADS * HEAD_DIM;

    bf16* out_head        = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * seq_len;
    int kv_head_idx       = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
        const bf16* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        // weighted_sum += att_head[t] * __bfloat162float(v_vec[i]);
        weighted_sum = __fmaf_rn(att_head[t], __bfloat162float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2bfloat16_rn(weighted_sum);
}

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

// ================================================================
// swiGlu
// ================================================================
__global__ void swiglu_kernel(__nv_bfloat16* hb, const __nv_bfloat16* hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val_fp32 = __bfloat162float(hb[i]);
        float hb2_fp32 = __bfloat162float(hb2[i]);

        float silu_val    = val_fp32 * (1.0f / (1.0f + expf(-val_fp32)));
        float result_fp32 = silu_val * hb2_fp32;
        hb[i]             = __float2bfloat16_rn(result_fp32);
    }
}

void swiglu_gpu(__nv_bfloat16* hb, const __nv_bfloat16* hb2, int size) {
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    swiglu_kernel<<<grid_size, THREADS_PER_BLOCK>>>(hb, hb2, size);
}

// ================================================================
// forward
// ================================================================
__global__ void convert_bf16_to_fp32_kernel(__nv_bfloat16* bf16_in, float* fp32_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fp32_out[i] = __bfloat162float(bf16_in[i]);
    }
}
