/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 * 
 *  \brief Some Utilities cuda kernels
 *  \author Yingshi Chen
 */
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <assert.h>
// #include <float.h>
#include <stdint.h>
#include <cooperative_groups.h>
#include "../cuda_common.h"
#include "../Tensor/Packed.hpp"

// fused multiply-add: FMA(a, b, c) = a*b + c, where the full product enters into the addition unmodified (neither rounded nor truncated), and there is a single rounding at the end. 
// One FMA instruction thus comprises two floating-point operations.It's the basic floating-point building block of the GPU.
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
// __device__ inline float lerp(float start, float end, float weight) {
//     return fma(weight, end, fma(-weight, start, start));
// }

// only for kernels by cudaLaunchCooperativeKernel
__device__ static void SYNC_GRID() {
    // cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    // grid.sync();  
	volatile unsigned int* barrier = &cooperative_groups::details::get_grid_workspace()->barrier;

	if (threadIdx.x == 0) {
		unsigned int nb = 1;
		if (blockIdx.x == 0) {
			nb = 0x80000000 - (gridDim.x - 1);
		}

		unsigned int old_arrive;
		asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(old_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier), "r"(nb) : "memory");

		unsigned int current_arrive;
		do {
			asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier) : "memory");
		} while (((old_arrive ^ current_arrive) & 0x80000000) == 0);
	}

	__syncthreads();
}

/*
    An “all reduce” within a warp using
*/
__device__ inline float warpreduce_sum(float v) {
#pragma unroll
	for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, mask);
	}
	return v;
}

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

__device__ inline float blocktranspose(float v, float def) {
	int lane = threadIdx.x % warpSize;
	int warp = threadIdx.x / warpSize;

	__shared__ float sm[32];
	sm[warp] = v;
	__syncthreads();

	return lane < blockDim.x / warpSize ? sm[lane] : def;
}



__device__ inline float blockreduce_max(float v) {
	v = warpreduce_max(v);
	v = blocktranspose(v, -FLT_MAX);
	v = warpreduce_max(v);
	return v;
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
// fast fp8x4 => float4 conversion; drops unnecessary NaN handling from __nv_cvt_fp8_to_halfraw
__device__ inline float4 fp8x4_e5m2_ff(__nv_fp8x4_e5m2 *v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	return float4(*v);
#else
	unsigned int vlo = v->__x, vhi = v->__x >> 16;
	__half2_raw hlo = {(unsigned short)(vlo << 8), (unsigned short)(vlo & 0xff00)};
	__half2_raw hhi = {(unsigned short)(vhi << 8), (unsigned short)(vhi & 0xff00)};
	float2 rlo = __internal_halfraw2_to_float2(hlo);
	float2 rhi = __internal_halfraw2_to_float2(hhi);
	float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
	return res;
#endif
}

__device__ inline half fp8_e5m2_ff(uint8_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
	__half_raw h = __nv_cvt_fp8_to_halfraw(v, __NV_E5M2);
#else
	__half_raw h = {(unsigned short)(v << 8)};
#endif
	return h;
}


template <typename T>
__device__ inline float CU_T2Float(const T* a0)   {
    float a = float(*a0);   
    return a;
}
//	Frome smart code of CALM
template <> __device__ inline float CU_T2Float<__nv_fp8_e5m2>(const __nv_fp8_e5m2 *x) {
// For Hopper (SM 90+) and later architectures:
//   asm("cvt.f32.f8.e5m2 %0, %1;" : "=f"(f) : "h"(x));
// For (SM 80/86 ...)  without native FP8 support:
	union {
		unsigned short u;
		half f;   //   IEEE 754-2008 binary16 (1 sign bit, 5 exponent bits, 10 mantissa bits).
	} u;
	u.u = (*(unsigned char*)(x)) << 8;
	float a = u.f;
    return a;
}

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

template<>
__device__ inline float cast_value<float, float>(float val) {
    return val;
}

#if defined(ENABLE_FP16) 
template<>
__device__ inline float cast_value<float, half>(half val) {
    return __half2float(val);
}
#endif

template<>
__device__ inline float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}
    
// ----------------------------------------------------------------------------
// Warp/Block communication primitives

// warp-level reduction for summing values
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warpReduceSum(float val) {
    assert(kWarpSize<=WARP_SIZE);
#pragma unroll
    for (int offset = kWarpSize >> 1; offset >= 1; offset >>= 1) {  //  performs a butterfly reduction pattern
        val += __shfl_xor_sync(0xffffffff, val, offset);            // 1-2 cycle
    }
    return val;
}
// warp-level reduction for finding the maximum value
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ inline void bitonicSwap(float& val, int j, int k) {
    int lane = threadIdx.x % 32;
    float other = __shfl_xor_sync(0xFFFFFFFF, val, j ^ k);
    if ((lane & k) == 0 && val > other)
        val = other;
}

using reduction_func_t = float (*) (float);
/*  requires all 32 threads in the warp to be active, but should work for any block size
    uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
    the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
    but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
*/

template<reduction_func_t warp_reduction> 
/*
    two reductions of up to 1024 threads!
    1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
*/
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) { shared_val[warp_id] = warp_val; }
    __syncthreads();        //  make sure the data is in shared memory.
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads(); // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}
/*
__device__ inline float blockreduce_sum(float v) {
	v = warpreduce_sum(v);
	v = blocktranspose(v, 0.f); //very interesting
      
        int lane = threadIdx.x % warpSize;
        int warp = threadIdx.x / warpSize;
        __shared__ float sm[32];
        sm[warp] = v;
        __syncthreads();
        return lane < blockDim.x / warpSize ? sm[lane] : def;
    
	v = warpreduce_sum(v);
	return v;
}*/

// Performs a _deterministic_ sum reduction. determinism is achieved by requiring that only
// a single block be used.
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // only a single block!
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) {
        *result = reduction;
    }
}

template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{
    constexpr unsigned int PRIME_NUMBER = 198491317u; // Large prime number with non-boring bits
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

#if defined(ENABLE_FP8)
    __device__ __forceinline__ void stochastic_rounding(float in, __nv_fp8_e5m2 *out, unsigned int seed) {
        assert(0);
    }
#endif

#if defined(ENABLE_BF16)
    // stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
    __device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {
        // todo - is this stochastic rounding *too good*? can we cut any corners?
        // makes sure each thread gets a different random number
        unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
        unsigned int threshold = random & 0xFFFF;
        unsigned int float_bits = __float_as_uint(in);
        unsigned int rounded_bits = float_bits & 0x0000FFFF;
        float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
        *out = __float2bfloat16_rn(__uint_as_float(float_bits));
    }
#endif

#if defined(ENABLE_FP16) 
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; // todo - implement this...
}
#endif

__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; // dummy function for when floatX is float (FP32 mode)
}

#endif