/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *        
 * 
 *  \brief From good job of CALM
 *  \author Yingshi Chen
 */
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include "./utils.cuh"
#include "../cuda_common.h"

__device__ inline float warpreduce_max(float v) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
	}
	return v;
}
__device__ inline int warpreduce_maxi(int v) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
	}
	return v;
}

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

__device__ inline float blocktranspose(float v, float def) {
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	__shared__ float sm[32];
	sm[warp] = v;
	__syncthreads();

	return lane < blockDim.x / warpSize ? sm[lane] : def;
}

__device__ inline float blockreduce_sum(float v) {
	v = warpreduce_sum(v);
	v = blocktranspose(v, 0.f);
	v = warpreduce_sum(v);
	return v;
}

__device__ inline float blockreduce_max(float v) {
	v = warpreduce_max(v);
	v = blocktranspose(v, -FLT_MAX);
	v = warpreduce_max(v);
	return v;
}

/*
	It's not deterministic, why?
	only support CU_x2_<<<grid_size, block_size, 0, main_stream>>> 
*/
template<class T,int NUM_THREADS = 256>
__global__ static void CU_x2_(float* out, const T* x0,size_t N) {
	int tid = threadIdx.x,warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
	int idx = blockIdx.x * NUM_THREADS + tid;
	constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float reduce_smem[NUM_WARPS];	// keep the data in register is enough for warp operaion.
	float a = (idx < N) ? (float)(x0[idx]) : 0.0;
	float sum = a*a;

	// perform warp sync reduce.
	sum = warpReduceSum<WARP_SIZE>(sum);	//warp_reduce_sum_f32<WARP_SIZE>(sum);
	// warp leaders store the data to shared memory.
	if (lane == 0) reduce_smem[warp] = sum;
	__syncthreads(); // make sure the data is in shared memory.
	sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warpReduceSum<NUM_WARPS>(sum);	
	if (tid == 0) atomicAdd(out, sum);
    /*int lane_id = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;
    int idx = blockIdx.x * num_warps + warp_id,C=n;
    if(idx >= n) { return; } // guard

    // the row of input that this group of threads is responsible for
    const T* x = inp + idx * C;
    // rstd
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float a = (float)x[i];
        sum += a * a;
    }
    *out = warpReduceSum(sum);*/
}

template<class T,int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_ternary_(float* out,T* x0,size_t N) {
	int tid = threadIdx.x,warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
	int idx = blockIdx.x * NUM_THREADS + tid;
	constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
	__shared__ float average;                 
	// __shared__ T Ta,Tb;
	__shared__ float reduce_smem[NUM_WARPS];	// keep the data in register is enough for warp operaion.
	float a = (idx < N) ? (float)(x0[idx]) : 0.0;
	float sum = fabs(a);
	sum = warpReduceSum<WARP_SIZE>(sum);	
	// warp leaders store the data to shared memory.
	if (lane == 0) reduce_smem[warp] = sum;
	__syncthreads(); // make sure the data is in shared memory.
	sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
	if (warp == 0) sum = warpReduceSum<NUM_WARPS>(sum);	
	if (tid == 0) {
		atomicAdd(out, sum);   
		average = (*out/N)+1.0e-5; 		out[1] = average;
		// Ta = -average,	Tb = average;
		average = average/2;
	}	
	__syncthreads();			// wait for s_variance in shared memory to be ready for all threads
	if (idx < N){
		a = float(x0[idx]);
		x0[idx] = a	> average ? (T)1.0 : a<-average ? (T)(-1.0) : (T)(0.0);
		//x0[idx] = a	> average ? Tb : a<-average ? Ta : (T)(0.0);
	}
}

// fast fp8x4 => float4 conversion; drops unnecessary NaN handling from __nv_cvt_fp8_to_halfraw
__device__ inline float4 fp8x4_e5m2_ff(__nv_fp8x4_e5m2 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	return float4(v);
#else
	unsigned int vlo = v.__x, vhi = v.__x >> 16;
	__half2_raw hlo = {(unsigned short)(vlo << 8), (unsigned short)(vlo & 0xff00)};
	__half2_raw hhi = {(unsigned short)(vhi << 8), (unsigned short)(vhi & 0xff00)};
	float2 rlo = __internal_halfraw2_to_float2(hlo);
	float2 rhi = __internal_halfraw2_to_float2(hhi);
	float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
	return res;
#endif
}

// gf4 decoding: 8 3-bit values + 1 fp8 scale are packed in a 32-bit word
__device__ inline half cu_gf4_ff(uint32_t v, int k) {
	half s = fp8_e5m2_ff(v & 0xff) * half(-0.25f); // we expect compiler to reuse this across multiple calls
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
	int lane = threadIdx.x % warpSize;
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
	int lane = threadIdx.x % warpSize;
	float val = 0.0f;
	// use 64-bit loads instead of 32-bit loads to increase memory throughput on H100/A100
	// without this we are seeing lower throughput given the limited number of parallel warps in coop kernel
	// this is performance-neutral on 4090 but results in issues with x[] load coalescing (that are benign)
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

__device__ inline float embed(uint32_t* weight, int idx) {
	return cu_gf4_ff(weight[idx / 8], idx % 8);
}

template <typename T>
__global__ static void kernel_embed(float* o, T* weight, int token, int n) {
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
	KVT* kb = key_cache + loff;

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

__device__ inline float cu_gelu(float x) {
	// const float sqrt_param = 0.79788456080286535587989211986876f;
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}
__device__ inline float cu_d_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;

    float x2mul = x * x * mul_param;
    float tan_h = tanhf(sqrt_param * (x + x * x2mul));
    float dg1 = 0.5f * (1.0f + tan_h);
    float dg2 = x * 0.5f * sqrt_param * (1 - tan_h * tan_h);
    float dg3 = dg2 * 3 * x2mul;
    return (dg1 + dg2 + dg3);
}

__device__ inline float cu_silu(float x) {
	return x / (1.0f + expf(-x));
}

__device__ static void moe_gate_warp(float* moe_weights, int* moe_experts, float* weights, int experts, int active) {
	int i = threadIdx.x;

	// (unscaled) softmax across experts
	float w = (i < experts) ? weights[i] : -FLT_MAX;
	float max_val = warpreduce_max(w);
	w = expf(w - max_val);

	// weight in top 24 bits, index in bottom 8
	int wi = (__float_as_int(w) & 0xffffff00) | i;

	// top k within warp
	float sumw = 0.f;
	int acti = -1;

	for (int k = 0; k < active; ++k) {
		int maxi = warpreduce_maxi(wi);

		sumw += __int_as_float(maxi);

		// keeps top weight in thread k, clears weight for thread with max thread to avoid re-selection
		acti = (i == k) ? maxi : acti;
		wi = (wi == maxi) ? 0 : wi;
	}

	// write normalized weights
	if (i < active) {
		assert(acti >= 0);

		moe_experts[i] = acti & 0xff;
		moe_weights[i] = __int_as_float(acti) / sumw;
	}
}

template <typename T>
__device__ static float CU_rmsnorm(T* o, float* x, float* weight, int size, float eps, bool ln) {
	int i = threadIdx.x;
	int blockSize = blockDim.x;

	float mean = 0.0f;
	if (ln) {
		// calculate sum (per thread)
		float sum = 0.0f;
		for (int j = i; j < size; j += blockSize) {
			sum += x[j];
		}

		// sum across threads in block
		mean = blockreduce_sum(sum) / size;
	}

	// calculate sum of squares (per thread)
	float ss = 0.0f;
	for (int j = i * 2; j < size; j += blockSize * 2) {
		float2 xx = *(float2*)&x[j];
		float2 ww = *(float2*)&weight[j];
		float v0 = xx.x - mean;
		float v1 = xx.y - mean;
		ss += v0 * v0;
		ss += v1 * v1;
		*(ablock<T, 2>*)&o[j] = { v0 * ww.x, v1 * ww.y };
	}

	// sum across threads in block
	ss = blockreduce_sum(ss);

	// caller is responsible for normalization
	return rsqrtf(ss / size + eps);
}

__device__ inline float4 attn_load4(half* p) {
	ablock<__half2_raw, 2> h = *(ablock<__half2_raw, 2>*)p;
	float2 h0 = __half22float2(h.v[0]), h1 = __half22float2(h.v[1]);
	return {h0.x, h0.y, h1.x, h1.y};
}

__device__ inline float4 attn_load4(__nv_fp8_e5m2* p) {
	return fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)p);
}

template <typename KVT>
__device__ inline float attn_score(KVT* kht, float* qh, int head_dim, int seq_len, int t, int off) {
	float score = 0.0f;
	for (int j = 0; j < head_dim; j += 16) {
		float4 kk = attn_load4(&kht[j * seq_len + t * 16 + off]);
		float4 qq = *(float4*)&qh[j + off];
		score += kk.x * qq.x;
		score += kk.y * qq.y;
		score += kk.z * qq.z;
		score += kk.w * qq.w;
	}

	return score;
}


template <typename KVT>
__device__ inline float attn_warpdot(KVT* val, float* atth, int kv_len) {
	int kv_len4 = kv_len & ~3;
	int lane = threadIdx.x % warpSize;

	float res = 0.0f;
	float sum = 0.0f;
	for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
		float4 vv = attn_load4(&val[t]);
		float4 aa = *(float4*)&atth[t];
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

__device__ static void CU_softmax_v0(float* xout, float* x, int size) {
	int i = threadIdx.x;

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < size; j += blockDim.x) {
		max_val = max(max_val, x[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(max_val);

	// exp per thread
	for (int j = i; j < size; j += blockDim.x) {
		xout[j] = expf(x[j] - max_val);
	}
}

template <typename T, typename AT>
__global__ static void kernel_output(uint64_t, float* xout, float* x, T* w, float* rms_weight, int n, int d, float norm_eps, bool norm_ln) {
	extern __shared__ char smem[];

	AT* xs = (AT*)smem;

	float rmsscale = CU_rmsnorm(xs, x, rms_weight, n, norm_eps, norm_ln);

	int io = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int ib = (gridDim.x * blockDim.x) / warpSize;

	for (int j = io; j < d; j += ib) {
		float val = matmul_warppar(xs, w, j, n) * rmsscale;

		// instead of writing one value per block, we transpose the values and write all results from first warp
		val = blocktranspose(val, 0.f);

		if (threadIdx.x < blockDim.x / warpSize) {
			xout[j + threadIdx.x] = val;
		}
	}
}

#include <curand_kernel.h>
//	Initialization of the random generator state generally requires more registers and local memory than random number generation. It may be beneficial to separate calls to curand_init() and curand() into separate kernels for maximum performance.
__global__ static void CU_initCurand(curandState *state, unsigned long seed, int N) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
        curand_init(seed, id, 0, &state[id]);
    }
}

template<typename typ>
__global__ void CU_normal_generate(curandState *state, typ *results, int N, float devia) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N) {
		//	Normally distributed float with mean 0.0f and standard deviation 1.0f
        float rand_val = curand_normal(&state[id])*devia; 
        results[id] = (typ)rand_val; 
    }
}
//	results[id] = __float2bfloat16(rand_val); // convert to bf16
template<typename typ>
void CU_normal(int N,typ* out, float devia,unsigned long seed=42,bool isToHost=false){
    typ *d_results;
    curandState *d_states;
	if(isToHost){
    	cudaMalloc(&d_results, N * sizeof(typ));
	}	else{
		d_results = out;
	}
    cudaMalloc(&d_states, N * sizeof(curandState));

    CU_initCurand<<<(N + 255) / 256, 256>>>(d_states, seed, N);
    CU_normal_generate<typ><<<(N + 255) / 256, 256>>>(d_states, d_results, N, devia);
	cudaCheck(cudaDeviceSynchronize());
    // Copy back results if needed
    // __nv_bfloat16 *h_results = new __nv_bfloat16[N];
	if(isToHost){
		cudaMemcpy(out, d_results, N * sizeof(typ), cudaMemcpyDeviceToHost);
		cudaFree(d_results);
	}

    cudaFree(d_states);
    return;
}