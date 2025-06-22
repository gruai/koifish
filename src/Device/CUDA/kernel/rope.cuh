#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define CUDA_ROPE_BLOCK_SIZE 256
#define _xita  10000.0f

/*
rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
	for (int i = 0; i < d; i += 2) {
		int j_head = i % head_dim;
		float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
		float val = pos * freq;
		float fcr = cosf(val);
		float fci = sinf(val);

		float v0 = vec[i];
		float v1 = vec[i + 1];
		vec[i] = v0 * fcr - v1 * fci;
		vec[i + 1] = v0 * fci + v1 * fcr;
	}
}
*/

// out may same as inp
template<typename typ>
__global__ void CU_rope_(typ* inp, typ* out, int seq_len, int head_dim, float theta, int rotary_dim,int B, int T, int C){ //, int N
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int N = B * T * C;
    if (idx >= N) { return; }
    int b = (idx/C)/T,t = (idx/C)%T,c = idx % C;
    int j_head = c % head_dim;
    // float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float freq =  1.0f / powf(theta, (float)c / (float)C);
    // freq = 0;
    float val = t * freq;
    float sin_v = sinf(val),cos_v = cosf(val);
    float x1 = inp[idx],x2 = inp[idx + 1];
    
    /* int token_pos = idx / N,    token_idx = idx % N;
    float exp_v = 1.0f / powf(_xita, token_idx / (N * 2));
    float sin_v = sinf(token_pos / exp_v);
    float cos_v = cosf(token_pos / exp_v);*/
    
    float out1 = x1 * cos_v - x2 * sin_v;
    float out2 = x1 * sin_v + x2 * cos_v;
    out[idx] = out1;    out[idx + 1] = out2;
    // out[idx * 2] = x1;    out[idx * 2 + 1] = x2;
}
template<typename typ>
__global__ void CU_rope_back(typ* inp, typ* out, int seq_len, int head_dim, float theta, int rotary_dim,int B, int T, int C){ //, int N
    int idx = (blockIdx.x * blockDim.x + threadIdx.x)*2;
    int N = B * T * C;
    if (idx >= N) { return; }
    int b = (idx/C)/T,t = (idx/C)%T,c = idx % C;
    int j_head = c % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    freq = 0;
    float val = t * freq;
    float sin_v = sinf(val),cos_v = cosf(val);
    float x1 = inp[idx],x2 = inp[idx + 1];    
    float out1 = x1 * cos_v - x2 * sin_v;
    float out2 = x1 * sin_v + x2 * cos_v;
    out[idx] = out1;    out[idx + 1] = out2;
}


// another index method of rope.
__global__ static void CU_rope_f32_v2(float* x, float* out, int seq_len, int N){ 
  int token_pos = blockIdx.x;
  int tid = threadIdx.x;
  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];
  float exp_v = 1.0f / powf(_xita, (int)(tid / 2) / (N * 2));
  float sin_v = sinf(token_pos / exp_v);
  float cos_v = cosf(token_pos / exp_v);
  float out1 = x1 * cos_v - x2 * sin_v;
  float out2 = x1 * sin_v + x2 * cos_v;
  out[token_pos * N * 2 + tid * 2] = out1;
  out[token_pos * N * 2 + tid * 2 + 1] = out2;
}

__global__ static void rope_f32x4_pack_kernel(float* x, float* out, int seq_len, int N){ 
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = FLOAT4(x[idx * 4]);
  int token_pos = idx / N; 
  int token_idx = idx % N;
  float exp_f_v = 1.0f / powf(_xita, token_idx * 2 / (N * 4));
  float exp_s_v = 1.0f / powf(_xita, ((token_idx * 2) + 1) / (N * 4));
  float sin_f_v = sinf(token_pos / exp_f_v);
  float cos_f_v = cosf(token_pos / exp_f_v);
  float sin_s_v = sinf(token_pos / exp_s_v);
  float cos_s_v = cosf(token_pos / exp_s_v);
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v; 
  FLOAT4(out[idx * 4]) = out_v;
}

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_norm(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0;
    const int ix   = channel_x*s2 + row_x*s1 + i0;

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    dst[idst + 0] = x0*cos_theta - x1*sin_theta;
    dst[idst + 1] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_neox(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const float theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_multi(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2,
        const int n_dims, const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors, const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row_dst*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        theta_base = pos[channel_x]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        theta_base = pos[channel_x + ne2 * 1]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 2]*powf(theta_scale, i0/2.0f);
    }
    else if (sector >= sec_w + sections.v[2]) {
        theta_base = pos[channel_x + ne2 * 3]*powf(theta_scale, i0/2.0f);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    dst[idst + 0]        = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, bool has_ff, typename T>
static __global__ void rope_vision(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims,
        const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
        const float theta_scale, const float * freq_factors, const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const int row_x     = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst*ne0 + i0/2;
    const int ix   = channel_x*s2 + row_x*s1 + i0/2;

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base = pos[channel_x]*powf(theta_scale, p);
    }
    else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base = pos[channel_x + ne2]*powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template<bool forward, typename T>
static void rope_norm_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

template<bool forward, typename T>
static void rope_neox_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_neox<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    } else {
        rope_neox<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors);
    }
}

template<bool forward, typename T>
static void rope_multi_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template<bool forward, typename T>
static void rope_vision_cuda(
        const T * x, T * dst, const int ne0, const int ne1, const int ne2, const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos, const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float * freq_factors, const mrope_sections sections, cudaStream_t stream) {
    assert(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template<typename typ>
__global__ void apply_rope_backward_kernel1(
    typ *dq, typ *dk, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs) // Guard to handle only half_hs elements for real and imaginary pairs
    {
        int freq_index = t * half_hs + hs;
#if defined(ENABLE_FP8)
#else
        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        // Backprop for q (shape: B, T, num_kv_heads, C/NH)
        if (nh < num_kv_heads) // only the q heads are processed
        {
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + nh * C_per_NH + hs;

            // Gradients from the next layer (dout_q)
            float dq_r = dq[q_index];
            float dq_i = dq[q_index + half_hs];

            // Backpropagation using chain rule
            dq[q_index] = dq_r * cos_val + dq_i * sin_val;           // (df/dq_r)
            dq[q_index + half_hs] = dq_i * cos_val - dq_r * sin_val; // (df/dq_i)
        }

        // Backprop for k (shape: B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Gradients from the next layer
        float dk_r = dk[k_index];
        float dk_i = dk[k_index + half_hs];

        // Backpropagation using chain rule (dout_k)
        dk[k_index] = dk_r * cos_val + dk_i * sin_val;           // (df/dk_r)
        dk[k_index + half_hs] = dk_i * cos_val - dk_r * sin_val; // (df/dk_i)
#endif
    }
}

/**
 * Similar to Kernel-1 but uses Shared Memory for `freqs_cos` and `freqs_sin`
 * It may help us address our limiting perf factor (Memory Bandwidth), since we will be utilizing SRAM (less latency, faster memory)
 */
__global__ static void apply_rope_backward_kernel2(
    float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    extern __shared__ float shared_mem[]; // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;
    float *shared_freqs_sin = shared_mem + blockDim.x;

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;
#if defined(ENABLE_FP8)
#else
    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Each thread loads the necessary cos and sin values into shared memory
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // wait till all threads have loaded the shared memory

    if (hs < half_hs)
    {
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        // Backprop for q (shape: B, T, num_kv_heads, C/NH)
        if (nh < num_kv_heads)
        {
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + nh * C_per_NH + hs;

            // Gradients from the next layer
            float dq_r = dq[q_index];
            float dq_i = dq[q_index + half_hs];

            // Backpropagation using chain rule (dout_q)
            dq[q_index] = dq_r * cos_val + dq_i * sin_val;           // (df/dq_r)
            dq[q_index + half_hs] = dq_i * cos_val - dq_r * sin_val; // (df/dq_i)
        }

        // Backprop for k (shape: B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Gradients from the next layer
        float dk_r = dk[k_index];
        float dk_i = dk[k_index + half_hs];

        // Backpropagation using chain rule (dout_k)
        dk[k_index] = dk_r * cos_val + dk_i * sin_val;           // (df/dk_r)
        dk[k_index + half_hs] = dk_i * cos_val - dk_r * sin_val; // (df/dk_i)
    }
#endif
}

// ----------------------------------------------------------------------------
// kernel launcher

void inline apply_rope_backward1(float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                          int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH); // Parallelizing over B, T, NH
    int threads = C_per_NH / 2;

    apply_rope_backward_kernel1<<<blocks, threads>>>(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void inline apply_rope_backward2(float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                          int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH); // Parallelizes over B, T, NH
    int threads = C_per_NH / 2;
    int shared_mem_size = 2 * (C_per_NH / 2) * sizeof(float); // Shared memory size for freqs_cos and freqs_sin

    apply_rope_backward_kernel2<<<blocks, threads, shared_mem_size>>>(
        dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void inline apply_rope_backward(int kernel_num,
                         float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                         int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    switch (kernel_num)
    {
    case 1:
        apply_rope_backward1(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 2:
        apply_rope_backward2(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}


/**
 * Derived from the CPU PORT of the apply_rope kernel. Utilized Coalesced Memory access for `q`, `k`,
 * - Applies RoPE to `q` and `k` separately.
 * - Each thread handles a real/imaginary pair
 * - Can be optimized more, since we can warp-divergence (because of the if condition), making some threads become idle
 */
__global__ static void apply_rope_forward_kernel1(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Separate indexing for q and k based on their respective shapes
    if (hs < half_hs)
    {
        // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        if (kv_head < num_kv_heads)
        {
            // coalesced memory accesses for `q`
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;

            // Frequency index (T, C/2NH)
            int freq_index = t * half_hs + hs;

            float cos_val = freqs_cos[freq_index];
            float sin_val = freqs_sin[freq_index];

            // Apply RoPE to q (query)
            float q_r = q[q_index];
            float q_i = q[q_index + half_hs];

            q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
            q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
        }

        // Key (k) index for NH shape (B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + kv_head * C_per_NH + hs;

        // Apply RoPE to k (key)
        int freq_index = t * half_hs + hs;
        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];

        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}

// ----------------------------------------------------------------------------

/**
 * Below, in order to remove the warp-divergence, we are separating the kernels for `q` and `k`
 * - Utilizes coalesced memory access for `q`, `k`, `freq_cos`, and `freq_sin`
 * Each thread handles a real/imaginary pair for `q`, and `k`(in their respective kernels)
 */
template<typename typ>
__global__ void apply_rope_forward_q1(
    typ *q, const float *freqs_cos, const float *freqs_sin,int B, int T, int num_kv_heads, int C_per_NH){
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs)
    {
        int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs; // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        int freq_index = t * half_hs + hs;                                                                     // Frequency index (T, C/2NH)
#if defined(ENABLE_FP8)
#else
        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];
        float q_r = q[q_index];
        float q_i = q[q_index + half_hs];

        // Apply RoPE to q (query)
        q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
#endif
    }
}
template<typename typ>
__global__ static void apply_rope_forward_k1(
    typ *k, const float *freqs_cos, const float *freqs_sin,    int B, int T, int NH, int C_per_NH){
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs)    {
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs; // Key (k) index for NH shape (B, T, NH, C/NH)
        int freq_index = t * half_hs + hs;                                            // Frequency index (T, C/2NH)


#if defined(ENABLE_FP8)
#else
        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];
        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];        
        // Apply RoPE to k (key)
        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
#endif
    }
}
// ----------------------------------------------------------------------------

/**
 * Verion-1 and version 2, are of same perf (no as such significant performance increase due to warp-divergence), since the kernels are memory bandwidth bound
 *  These kernels use shared memory to store `freqs_cos` and `freqs_sin` values (frequently accessed in the computation).
 *
 * Each thread loads one cos and one sin value, so the total size of shared memory is 2 * blockDim.x * sizeof(float).
 */

__global__ static void apply_rope_forward_q2(
    float *q, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int C_per_NH)
{
    extern __shared__ float shared_mem[];              // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;              // First part of shared memory for freqs_cos
    float *shared_freqs_sin = shared_mem + blockDim.x; // Second part for freqs_sin

    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;
#if defined(ENABLE_FP8)
#else
    // Load freqs_cos and freqs_sin into shared memory for reuse
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // Ensure all threads have loaded shared memory before proceeding

    if (hs < half_hs)
    {
        // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;

        // Apply RoPE to q (query)
        float q_r = q[q_index];
        float q_i = q[q_index + half_hs];

        // Use shared memory for cos and sin values
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
    }
#endif
}

__global__ static void apply_rope_forward_k2(
    float *k, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int C_per_NH)
{
    extern __shared__ float shared_mem[];              // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;              // First part of shared memory for freqs_cos
    float *shared_freqs_sin = shared_mem + blockDim.x; // Second part for freqs_sin

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;
#if defined(ENABLE_FP8)
#else
    // Load freqs_cos and freqs_sin into shared memory for reuse
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // Ensure all threads have loaded shared memory before proceeding

    if (hs < half_hs)
    {
        // Key (k) index for NH shape (B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Apply RoPE to k (key)
        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];

        // Use shared memory for cos and sin values
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
#endif
}

// ----------------------------------------------------------------------------
// kernel launcher

void inline apply_rope_forward1(float *q, float *k, float *freqs_cos, float *freqs_sin, int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH);
    int threads = C_per_NH / 2;

    apply_rope_forward_kernel1<<<blocks, threads>>>(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void inline apply_rope_forward2(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    // Separate kernel launches for `q` and `k` to avoid warp-divergence

    dim3 blocks_q(B, T, num_kv_heads); // For q (shape: B, T, num_kv_heads, C/NH)
    dim3 blocks_k(B, T, NH);           // For k (shape: B, T, NH, C/NH)

    int block_size = C_per_NH / 2;

    apply_rope_forward_q1<<<blocks_q, block_size>>>(q, freqs_cos, freqs_sin, B, T, num_kv_heads, C_per_NH);
    apply_rope_forward_k1<<<blocks_k, block_size>>>(k, freqs_cos, freqs_sin, B, T, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void inline apply_rope_forward3(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    // Separate kernel launches for `q` and `k` with shared memory for `freqs_cos` and `freqs_sin`

    dim3 blocks_q(B, T, num_kv_heads); // For q (shape: B, T, num_kv_heads, C/NH)
    dim3 blocks_k(B, T, NH);           // For k (shape: B, T, NH, C/NH)

    int block_size = C_per_NH / 2;

    size_t shared_mem_size = 2 * block_size * sizeof(float); // Shared memory for cos and sin values

    apply_rope_forward_q2<<<blocks_q, block_size, shared_mem_size>>>(q, freqs_cos, freqs_sin, B, T, num_kv_heads, C_per_NH);
    apply_rope_forward_k2<<<blocks_k, block_size, shared_mem_size>>>(k, freqs_cos, freqs_sin, B, T, NH, C_per_NH);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void inline apply_rope_forward(int kernel_num,
                        float *q, float *k, float *freqs_cos, float *freqs_sin,
                        int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    switch (kernel_num)
    {
    case 1:
        apply_rope_forward1(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 2:
        apply_rope_forward2(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 3:
        apply_rope_forward3(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}