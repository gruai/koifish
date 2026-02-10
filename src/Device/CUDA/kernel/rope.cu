/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief cuda kernel of ROPE(Rotary Position Embeddings)
 *  \author Yingshi Chen
 */

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

#include "../../../Manifold/Fish.hpp"
#include "../../../Manifold/Neuron.hpp"
// #include "packedN.cuh"
#include "operator.cuh"
#include "utils.cuh"

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])

/*
    1. Each thread is responsible for one head
    2. out may same as inp
*/
template <typename Typ>
__global__ void CU_rope_v0_(Typ* inp, Typ* out, float* scale, int seq_len, int head_dim, float theta, int rotary_dim, int B, int T, int C, uint32_t seed,
                            bool isBack = false) {  //, int N
    int b = blockIdx.x, t = blockIdx.y, j_head = blockIdx.z;
    int c = threadIdx.x, nHead = seq_len / head_dim, h_id = b * T * nHead + t * nHead + j_head;
    int half_hs = head_dim / 2;
    if (c >= half_hs) {
        return;
    }
    // h_id = b * T * nHead + t + j_head * T;      // [batch, seq_len, heads, dim] -> [batch, heads, seq_len, dim]
    // float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    // Why c?  the index of pair - similar to an optical splitter functionality where the light can be broken down to multiple signals
    int c_1 = c;  //  |g| would > 1
    // int c_1 = (c + seed) % head_dim;                               // converge much slower
    float sin_v, cos_v, freq = 1.0f / powf(theta, 2.0f * c_1 / (float)rotary_dim);  //  [1,1.0/theta]
    // float val   = t * freq;
    // float sin_v = sinf(val), cos_v = cosf(val);
    __sincosf(t * freq, &sin_v, &cos_v);
    int idx1 = h_id * head_dim + c, idx2 = idx1 + half_hs;
    assert(idx2 < B * T * C);
    float x1 = inp[idx1], x2 = inp[idx2], out1, out2;
    if (scale != nullptr) {
        if (isBack) {
            x1 /= scale[h_id], x2 /= scale[h_id];
        } else {
            int NUM_WARPS = CEIL_DIV(head_dim, WARP_SIZE);
            assert(NUM_WARPS <= WARP_SIZE);
            out1           = x1 * cos_v - x2 * sin_v;
            out2           = x1 * sin_v + x2 * cos_v;
            float sum      = out1 * out1 + out2 * out2;
            float head_sum = blockReduce_v0<warpReduceSum>(sum, true);
            scale[h_id]    = rsqrtf(head_sum / head_dim + 1.0e-5);
            x1 *= scale[h_id], x2 *= scale[h_id];
        }
    }

    if (isBack) {
        out1 = x1 * cos_v + x2 * sin_v, out2 = x1 * sin_v - x2 * cos_v;
        //  out1 = x2;          out2 = x1;   // testing: transpos has no influence
    } else {
        out1 = x1 * cos_v - x2 * sin_v, out2 = x1 * sin_v + x2 * cos_v;
        //  out1 = x2;          out2 = x1;
    }
    out[idx1] = CU_Float2T<Typ>(out1, seed + c);
    out[idx2] = CU_Float2T<Typ>(out2, seed + c);
    // out[idx] = x1;      out[idx + 1] = x2;
}

static __forceinline__ __device__ void CU_stat(float* __restrict__ stat_info, float* __restrict__ block_max, float thread_max) {
    if (stat_info) {
        // this code is only guaranteed to be correct if it is warp convergent
        // (in theory, ensuring thread 0 hasn't exited would be enough...)
        assert(__activemask() == 0xffffffff);
        auto warp_max = __reduce_max_sync(0xffffffff, __float_as_uint(thread_max));
        if (threadIdx.x % 32 == 0) {
            atomicMax_block(reinterpret_cast<unsigned*>(block_max), warp_max);
        }

        __syncthreads();
        if (threadIdx.x == 0) {
            atomicMax(reinterpret_cast<unsigned int*>(stat_info), __float_as_uint(*block_max));
        }
    }
}

__global__ static void rope_f32x4_pack_kernel(float* x, float* out, int seq_len, int N) {
    float _xita   = 10000.0f;
    int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    float4 x_v    = FLOAT4(x[idx * 4]);
    int token_pos = idx / N;
    int token_idx = idx % N;
    float exp_f_v = 1.0f / powf(_xita, token_idx * 2 / (N * 4));
    float exp_s_v = 1.0f / powf(_xita, ((token_idx * 2) + 1) / (N * 4));
    float sin_f_v = sinf(token_pos / exp_f_v);
    float cos_f_v = cosf(token_pos / exp_f_v);
    float sin_s_v = sinf(token_pos / exp_s_v);
    float cos_s_v = cosf(token_pos / exp_s_v);
    float4 out_v;
    out_v.x              = x_v.x * cos_f_v - x_v.y * sin_f_v;
    out_v.y              = x_v.x * sin_f_v + x_v.y * cos_f_v;
    out_v.z              = x_v.z * cos_s_v - x_v.w * sin_s_v;
    out_v.w              = x_v.z * sin_s_v + x_v.w * cos_s_v;
    FLOAT4(out[idx * 4]) = out_v;
}

struct rope_corr_dims {
    float v[2];
};

struct mrope_dims {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template <bool forward>
static __device__ void rope_yarn(const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
                                 float mscale, float& cos_theta, float& sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta        = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta          = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_norm(const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int32_t* pos,
                                 const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                                 const float theta_scale, const float* freq_factors) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0;
    const int ix   = channel_x * s2 + row_x * s1 + i0;

    const float theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + 1] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_neox(const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int32_t* pos,
                                 const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                                 const float theta_scale, const float* freq_factors) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix   = channel_x * s2 + row_x * s1 + i0 / 2;

    const float theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0]          = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_multi(const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
                                  const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                                  const float theta_scale, const float* freq_factors, const mrope_dims sections) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst * ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix   = channel_x * s2 + row_x * s1 + i0 / 2;

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        theta_base = pos[channel_x] * powf(theta_scale, i0 / 2.0f);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        theta_base = pos[channel_x + ne2 * 1] * powf(theta_scale, i0 / 2.0f);
    } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 2] * powf(theta_scale, i0 / 2.0f);
    } else if (sector >= sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 3] * powf(theta_scale, i0 / 2.0f);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0]          = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
                                   const int32_t* pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                                   const float theta_scale, const float* freq_factors, const mrope_dims sections) {
    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix   = channel_x * s2 + row_x * s1 + i0 / 2;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[channel_x] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[channel_x + ne2] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

#define CUDA_ROPE_BLOCK_SIZE 256
template <bool forward, typename T>
static void rope_norm_cuda(const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr, const int32_t* pos,
                           const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                           const float* freq_factors, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                                                                         theta_scale, freq_factors);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                                                                        theta_scale, freq_factors);
    }
}

template <bool forward, typename T>
static void rope_neox_cuda(const T* x, T* dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr, const int32_t* pos,
                           const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                           const float* freq_factors, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                            corr_dims, theta_scale, freq_factors);
    } else {
        rope_neox<forward, true, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                           corr_dims, theta_scale, freq_factors);
    }
}

template <bool forward, typename T>
static void rope_multi_cuda(const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
                            const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
                            const rope_corr_dims corr_dims, const float* freq_factors, const mrope_dims sections, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                             corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                            corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T* x, T* dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
                             const int32_t* pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
                             const rope_corr_dims corr_dims, const float* freq_factors, const mrope_dims sections, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                              corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                                                             corr_dims, theta_scale, freq_factors, sections);
    }
}
/*
    M-RoPE(Multi-Modal Rotary Positional Embedding) transformation for one token.

    1) MROPE构成：t时间，h位置，w位置.
        RoPE可以理解为:             PE = seq_idx
        2D-ROPE就是：              PE = (h_idx + w_idx)
        3D-rope就是每3个一组去变化:  PE = (time_idx + h_idx + w_idx)
     文本简化为所有都是一样的: (idx, idx, id),图片则是时间一致，h和w不一致：（same_idx， w_idx， h_idx）视频就是图片上，时间也会变化: （t_idx, w_idx, h_idx）
     time的长度（权重16）< h_idx(24) = w_idx(24),这个也可以理解为作者认为：时间的重要程度比空间小。时间只会在视频这一个场景上有作用。

    2) Each thread is responsible for one token.
*/
template <typename Typ>
__global__ void qwen2vl_mrope_kernel(Typ* q,               // [bs*sl, n_qh * hd]
                                     Typ* k,               // [bs*sl, n_kh * hd]
                                     const float* cos,     // shape: [3, bs*sl, hd]
                                     const float* sin,     // shape: [3, bs*sl, hd]
                                     int sl,               // sequence length
                                     int bs,               // batch size
                                     int n_qh,             // number of Q heads
                                     int n_kh,             // number of K heads
                                     int hd,               // head dimension (must be even)
                                     int pad_n_qh,         // padded number of Q heads (assumed = n_qh)
                                     int pad_n_kh,         // padded number of K heads (assumed = n_kh)
                                     int pad_hd,           // padded head dimension (assumed = hd)
                                     int mrope_section_t,  // mrope section “t”    3
                                     int mrope_section_h,  // mrope section “h”    2
                                     bool backward_pass    // if true, perform backward transformation
) {
    //
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_row    = bs * sl;
    if (token_id >= n_row)
        return;

    // Each token's Q and K are stored contiguously:
    // Q: [n_qh, hd] and K: [n_kh, hd].
    Typ* q_token = q + token_id * n_qh * hd;
    Typ* k_token = k + token_id * n_kh * hd;

    // cos and sin arrays are arranged in three contiguous blocks:
    // Section 0: t_cos/t_sin, Section 1: h_cos/h_sin, Section 2: w_cos/w_sin.
    const int token_offset = token_id * hd;
    const float* t_cos     = cos + token_offset;
    const float* h_cos     = cos + bs * sl * hd + token_offset;
    const float* w_cos     = cos + 2 * bs * sl * hd + token_offset;
    const float* t_sin     = sin + token_offset;
    const float* h_sin     = sin + bs * sl * hd + token_offset;
    const float* w_sin     = sin + 2 * bs * sl * hd + token_offset;

    // For the rotary computation we use only the first half of the head dimension.
    int half_hd = hd / 2;
    int h_end   = mrope_section_t + mrope_section_h;  // boundary for second section

    // Process each Q head for this token.
    for (int head = 0; head < n_qh; head++) {
        float* q_head_ptr = q_token + head * hd;
        for (int d = 0; d < half_hd; d++) {
            float q1 = q_head_ptr[d];
            float q2 = q_head_ptr[d + half_hd];

            float cos_val = 0.f, sin_val = 0.f;
            if (d < mrope_section_t) {
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end) {
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd) {
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_q1, new_q2;
            if (!backward_pass) {
                new_q1 = q1 * cos_val - q2 * sin_val;
                new_q2 = q2 * cos_val + q1 * sin_val;
            } else {
                new_q1 = q1 * cos_val + q2 * sin_val;
                new_q2 = q2 * cos_val - q1 * sin_val;
            }
            q_head_ptr[d]           = new_q1;
            q_head_ptr[d + half_hd] = new_q2;
        }
    }

    // Process each K head for this token.
    for (int head = 0; head < n_kh; head++) {
        float* k_head_ptr = k_token + head * hd;
        for (int d = 0; d < half_hd; d++) {
            float k1 = k_head_ptr[d];
            float k2 = k_head_ptr[d + half_hd];

            float cos_val = 0.f, sin_val = 0.f;
            if (d < mrope_section_t) {
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end) {
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd) {
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_k1, new_k2;
            if (!backward_pass) {
                new_k1 = k1 * cos_val - k2 * sin_val;
                new_k2 = k2 * cos_val + k1 * sin_val;
            } else {
                new_k1 = k1 * cos_val + k2 * sin_val;
                new_k2 = k2 * cos_val - k1 * sin_val;
            }
            k_head_ptr[d]           = new_k1;
            k_head_ptr[d + half_hd] = new_k2;
        }
    }
}

template <typename Typ>
__global__ void CU_rope_prenormal_(Typ* inp, Typ* out, float* scale, int seq_len, int head_dim, float theta, int rotary_dim, int B, int T, int C, uint32_t seed,
                                   bool isBack = false) {  //, int N
    int b = blockIdx.x, t = blockIdx.y, j_head = blockIdx.z;
    int c = threadIdx.x, nHead = seq_len / head_dim, h_id = (b * T + t) * nHead + j_head;
    int half_hs = head_dim / 2;
    if (c >= half_hs) {
        return;
    }

    int idx1 = h_id * head_dim + c, idx2 = idx1 + half_hs;
    assert(idx2 < B * T * C);
    float x1 = inp[idx1], x2 = inp[idx2], out1, out2, val, sin_v, cos_v;
    if (scale != nullptr) {
        if (isBack) {
            x1 = x1 / scale[h_id], x2 = x2 / scale[h_id];
        } else {
            int NUM_WARPS = CEIL_DIV(head_dim, WARP_SIZE);
            assert(NUM_WARPS <= WARP_SIZE);
            float sum      = x1 * x1 + x2 * x2;
            float head_sum = blockReduce_v0<warpReduceSum>(sum, true);
            scale[h_id]    = rsqrtf(head_sum / head_dim + 1.0e-5);
            x1 *= scale[h_id], x2 *= scale[h_id];
        }
    }
    // float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    // Why c?  the index of pair - similar to an optical splitter functionality where the light can be broken down to multiple signals
    int c_1    = (c + seed) % head_dim;                               //  c_1 = c
    float freq = 1.0f / powf(theta, 2.0f * c_1 / (float)rotary_dim);  //  [1,1.0/theta]
    // float val  = t * freq;
    // float sin_v = sinf(val), cos_v = cosf(val);
    __sincosf(t * freq, &sin_v, &cos_v);

    if (isBack) {
        out1 = x1 * cos_v + x2 * sin_v;
        out2 = x1 * sin_v - x2 * cos_v;
    } else {
        out1 = x1 * cos_v - x2 * sin_v;
        out2 = x1 * sin_v + x2 * cos_v;
    }
    out[idx1] = CU_Float2T<Typ>(out1, seed + c);
    out[idx2] = CU_Float2T<Typ>(out2, seed + c);
    // out[idx] = x1;      out[idx + 1] = x2;
}

// template __global__ void CU_rope_<bf16>(bf16* out, bf16* inp, const bf16* freqs, float* stat_info, int B, int T, int Nq, int Nkv, int head_dim,
//                                         bool isBack = false);

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
        q_head[j1] = CU_Float2T<Typ>(q_real * cos - q_imag * sin, 42);
        q_head[j2] = CU_Float2T<Typ>(q_real * sin + q_imag * cos, 42);
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
    hFish->GetBT(B, T);
    size_t nToken = B * T;
    assert(nToken == 1);
    floatX *q = ToX(hQKV->Q.out), *k = ToX(hQKV->K.out);
    floatX *qW = hnQ == nullptr ? nullptr : ToX(hnQ->w), *kW = hnK == nullptr ? nullptr : ToX(hnK->w);

    dim3 blocks_q(B, T, n_head), blocks_k(B, T, n_head_kv), blocks(B, T);
    float rstd_eps = 1.0e-6;
    if (fuse_normal == 0 && hnQ != nullptr) {
        // hnQ->w->Print("qnw", 0x0, dump_flag), hnK->w->Print("knw", 0x0, dump_flag);
        hnQ->cuFlow(hQKV->Q.out);
        hnK->cuFlow(hQKV->K.out);
        qW = nullptr, kW = nullptr;
    }
    hQKV->Q.out->Print("Q.norm", 0x0, dump_flag, nToken * q_dim);
    hQKV->K.out->Print("K.norm", 0x0, dump_flag, nToken * kv_dim);

    assert(n_head_kv <= n_head);  // so blocks_k is in blocks_q
    if (fuse_normal == 1) {
        CU_rope_rmsnormal_forw<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, qW, pos, n_head, head_dim, theta);
        CU_rope_rmsnormal_forw<floatX><<<blocks_k, dim3(head_dim / 2, 1, 1)>>>(k, kW, pos, n_head_kv, head_dim, theta);
    } else
        CU_rope2_v0<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, k, pos, n_head, n_head_kv, head_dim, theta, 42);
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
#if defined(USE_FP8_BASELINE)
#else
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
        dX[j1] = CU_Float2T<Typ>(dy_real * cos - dy_imag * sin, seed);
        dX[j2] = CU_Float2T<Typ>(dy_real * sin + dy_imag * cos, seed);

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
#endif
}

// fuse normal to rope, may reduce time
int ROPE::cuFlow(SelfAttention* hQKV, uint32_t seed, bool isFX, int flag) {
    if (hFish == nullptr)  // some models(GPT2) don't need rope
        return 0x0;

    INSPECT_THIS;
    hFish->GetBT(B, T);
    dim3 blocks_q(B, T, n_head), blocks_k(B, T, n_head_kv), blocks(B, T);
    // size_t smemPB = 1024 * sizeof(float);
    floatX *q = ToX(hQKV->Q.out), *k = ToX(hQKV->K.out);  // *freqs = ToX(hSin);
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
            CU_rope2_v0<<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, k, -1, n_head, n_head_kv, head_dim, theta, 42);
        } else {
            CU_rope_rmsnormal_forw<floatX><<<blocks_q, dim3(head_dim / 2, 1, 1)>>>(q, qW, -1, n_head, head_dim, theta);
            CU_rope_rmsnormal_forw<floatX><<<blocks_k, dim3(head_dim / 2, 1, 1)>>>(k, kW, -1, n_head_kv, head_dim, theta);
        }
    } else {
        // hQKV->deltaQ->Print("dQ.rope.0", 0, dump_flag);
        // hQKV->deltaK->Print("dK.rope.0", 0, dump_flag);
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
            CU_rope2_v0<<<blocks_q, head_dim / 2>>>(dQ, dK, -1, n_head, n_head_kv, head_dim, theta, 42, 1);
            // hQKV->deltaQ->Print("dQ.rope.1", 0, dump_flag), hQKV->deltaK->Print("dK.rope.1", 0, dump_flag);
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
