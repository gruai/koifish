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
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <stdint.h>

#include "./utils.cuh"
#include "../cuda_common.h"


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
	only support CU_x2_<<<grid_size, block_size, 0, main_stream>>> 
*/
template<class T,int NUM_THREADS = 256>
__global__ static void CU_x2_(float* out, const T* x0,size_t N) {
	int tid = threadIdx.x, idx = blockIdx.x * NUM_THREADS + tid;
	if(idx >= N) { return; } // guard

	constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
	assert(NUM_WARPS<=WARP_SIZE);
	// __shared__ float reduce_smem[NUM_WARPS];	// keep the data in register is enough for warp operaion.
	float a = (idx < N) ? (float)(x0[idx]) : 0.0;
	float sum = a*a;
	
	float block_sum = blockReduce<warpReduceSum>(sum,true);
	// int wid = tid / WARP_SIZE, lane = tid % WARP_SIZE;		//	laneId = tid & 0x1f;  faster than %
	// sum = warpReduceSum<WARP_SIZE>(sum);	
	// if (lane == 0) reduce_smem[wid] = sum;
	// __syncthreads(); // make sure the data is in shared memory.
	// sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
	// if (wid == 0) sum = warpReduceSum<NUM_WARPS>(sum);	
	if (tid == 0) atomicAdd(out, block_sum);   
	// __syncthreads(); 		
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
	if(0){
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
	}else{
		for(int j = lane * 4; j < n; j += warpSize * 4) {
			//ablock<__nv_fp8x4_e5m2, 1> wwp = *(ablock<__nv_fp8x4_e5m2, 1>*)&w[i * n + j];
			float4 ww = fp8x4_e5m2_ff((__nv_fp8x4_e5m2 *)(w+i*n+j));
			float4 xx = {x[j],x[j+1],x[j+2],x[j+3]};	//*(float4*)&x[j];
			val += ww.x * xx.x;				val += ww.y * xx.y;				val += ww.z * xx.z;				val += ww.w * xx.w;
			//val += x[j] * ww.x + x[j+1] * ww.y + x[j+2] * ww.z + x[j+3] * ww.w;
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

__device__ inline float embed(uint32_t* weight, int idx) {
	return cu_gf4_ff(weight[idx / 8], idx % 8);
}

template <typename T_out,typename T>
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

template<typename T>
__device__ inline T CU_gelu(T x) {
	// const float sqrt_param = 0.79788456080286535587989211986876f;
	float xf = x,out=0.5f*xf*(1.0f + tanhf(0.797885f * (xf + 0.044715f*xf*xf*xf)));
	return T(out);	//0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}
template<typename T>
__device__ inline T CU_silu(T x) {
	float xf = x, out = xf / (1.0f + expf(-xf));
	return T(out);
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

__device__ inline float4 attn_load4(half* p) {
	ablock<__half2_raw, 2> h = *(ablock<__half2_raw, 2>*)p;
	float2 h0 = __half22float2(h.v[0]), h1 = __half22float2(h.v[1]);
	return {h0.x, h0.y, h1.x, h1.y};
}

__device__ inline float4 attn_load4(__nv_fp8_e5m2* p) {
	return fp8x4_e5m2_ff(*(__nv_fp8x4_e5m2*)p);
}

template <typename KVT,typename T>
__device__ inline float attn_score(KVT* kht, T* qh, int head_dim, int seq_len, int t, int off) {
	float score = 0.0f;
	for (int j = 0; j < head_dim; j += 16) {
		float4 kk = attn_load4(&kht[j * seq_len + t * 16 + off]);
		float4 qq = {qh[j+off],qh[j+off+1],qh[j+off+2],qh[j+off+3]};	//*(float4*)&qh[j + off];
		score += kk.x * qq.x;
		score += kk.y * qq.y;
		score += kk.z * qq.z;
		score += kk.w * qq.w;
	}

	return score;
}


template <typename KVT,typename T>
__device__ inline float attn_warpdot(KVT* val, T* atth, int kv_len) {
	// assert(0);
	int kv_len4 = kv_len & ~3;
	int lane = threadIdx.x % warpSize;

	float res = 0.0f;
	float sum = 0.0f;
	for (int t = lane * 4; t < kv_len4; t += warpSize * 4) {
		float4 vv = attn_load4(&val[t]);
		float4 aa = {atth[t],atth[t+1],atth[t+2],atth[t+3]};	//*(float4*)&atth[t];
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

template<typename T>
__device__ static void CU_softmax_v0(T* xout, T* x, int size) {
	int i = threadIdx.x;

	// find max value per thread (for numerical stability)
	float max_val = -FLT_MAX;
	for (int j = i; j < size; j += blockDim.x) {
		max_val = max(max_val, x[j]);
	}

	// max across threads in block
	max_val = blockreduce_max(max_val);
	T a1 = max_val;
	// exp per thread
	for (int j = i; j < size; j += blockDim.x) {
		xout[j] = expf(x[j] - a1);
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