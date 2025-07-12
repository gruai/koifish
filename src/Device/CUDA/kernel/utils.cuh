/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Some Utilities cuda kernels
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
// #include <float.h>
#include <cooperative_groups.h>
#include <stdint.h>

#include "../Tensor/Packed.hpp"
#include "../cuda_common.h"

// fused multiply-add: FMA(a, b, c) = a*b + c, where the full product enters into the addition unmodified (neither rounded nor truncated), and there is a single
// rounding at the end. One FMA instruction thus comprises two floating-point operations.It's the basic floating-point building block of the GPU. Reference:
// https://developer.nvidia.com/blog/lerp-faster-cuda
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
    float2 rlo      = __internal_halfraw2_to_float2(hlo);
    float2 rhi      = __internal_halfraw2_to_float2(hhi);
    float4 res      = {rlo.x, rlo.y, rhi.x, rhi.y};
    return res;
#endif
}
// fast fp8x4 => float4 conversion; drops unnecessary NaN handling from __nv_cvt_fp8_to_halfraw
__device__ inline float4 fp8x4_e5m2_ff(__nv_fp8x4_e5m2* v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    return float4(*v);
#else
    unsigned int vlo = v->__x, vhi = v->__x >> 16;
    __half2_raw hlo = {(unsigned short)(vlo << 8), (unsigned short)(vlo & 0xff00)};
    __half2_raw hhi = {(unsigned short)(vhi << 8), (unsigned short)(vhi & 0xff00)};
    float2 rlo      = __internal_halfraw2_to_float2(hlo);
    float2 rhi      = __internal_halfraw2_to_float2(hhi);
    float4 res      = {rlo.x, rlo.y, rhi.x, rhi.y};
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
__device__ inline float CU_T2Float(const T* a0) {
    float a = float(*a0);
    return a;
}
//	Frome smart code of CALM
template <>
__device__ inline float CU_T2Float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* x) {
    // For Hopper (SM 90+) and later architectures:
    //   asm("cvt.f32.f8.e5m2 %0, %1;" : "=f"(f) : "h"(x));
    // For (SM 80/86 ...)  without native FP8 support:
    union {
        unsigned short u;
        half f;  //   IEEE 754-2008 binary16 (1 sign bit, 5 exponent bits, 10 mantissa bits).
    } u;
    u.u     = (*(unsigned char*)(x)) << 8;
    float a = u.f;
    return a;
}

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template <typename Td, typename Ts>
__device__ Td cast_value(Ts val);

template <>
__device__ inline float cast_value<float, float>(float val) {
    return val;
}

#if defined(ENABLE_FP16)
template <>
__device__ inline float cast_value<float, half>(half val) {
    return __half2float(val);
}
#endif

template <>
__device__ inline float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename Td, typename Ts>
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
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warpReduceSum(float val) {
    assert(kWarpSize <= WARP_SIZE);
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
    int lane    = threadIdx.x % 32;
    float other = __shfl_xor_sync(0xFFFFFFFF, val, j ^ k);
    if ((lane & k) == 0 && val > other)
        val = other;
}

using reduction_func_t = float (*)(float);
/*  requires all 32 threads in the warp to be active, but should work for any block size
    uses non-dynamic shared memory so every call increases shared memory requirements by 128 bytes
    the fact it's unique shared memory allows us to avoid an extra __syncthreads() call at the end
    but if called inside a loop, the shared memory will be implicitly reused, so set final_sync to 1
*/

template <reduction_func_t warp_reduction>
/*
    two reductions of up to 1024 threads!
    1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
*/
__device__ inline float blockReduce(float val, bool final_sync = false, float out_of_bounds = 0.0f) {
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id   = threadIdx.x % WARP_SIZE;
    const int warp_id   = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    float warp_val = warp_reduction(val);
    if (lane_id == 0) {
        shared_val[warp_id] = warp_val;
    }
    __syncthreads();  //  make sure the data is in shared memory.
    warp_val        = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    if (final_sync) {
        __syncthreads();  // only needed in loops when effectively reusing shared memory etc.
    }
    return block_val;
}

// Performs a _deterministic_ sum reduction. determinism is achieved by requiring that only a single block be used.
template <class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);  // only a single block!
    float thread_sum = 0;
    for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if (threadIdx.x == 0) {
        *result = reduction;
    }
}

template <class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    assert(count < 1024);
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Random Number Generation used in Stochastic Rounding

// SquirrelNoise5 - Squirrel's Raw Noise utilities (version 5)
// This gives us a random number from threadIdx/blockIdx + a single seed for the entire GPU
// todo - possibly overkill and we don't need such high quality random numbers? (tbd)
// http://eiserloh.net/noise/SquirrelNoise5.hpp
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed) {
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;  // 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;  // 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B;  // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;  // 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;  // 00011011010101101100010011110101
    unsigned int mangledBits              = positionX;
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
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed) {
    constexpr unsigned int PRIME_NUMBER = 198491317u;  // Large prime number with non-boring bits
    unsigned int x                      = static_cast<unsigned int>(indexX);
    unsigned int y                      = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed);
}

// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
static bool isRounding = false;  // only for debug

template <typename T>
__device__ __forceinline__ T CU_Float2T(const float& a0, unsigned int seed) {
    T a = a0;
    return a;
}

template <>
__device__ __forceinline__ __nv_bfloat16 CU_Float2T<__nv_bfloat16>(const float& a0, unsigned int seed) {
    // todo - is this stochastic rounding *too good*? can we cut any corners?
    // makes sure each thread gets a different random number
    unsigned int random       = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold    = random & 0xFFFF;
    unsigned int float_bits   = __float_as_uint(a0);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits                = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits & ~0xFFFF);
    __nv_bfloat16 out         = __float2bfloat16_rn(__uint_as_float(float_bits));
    return out;
}

/*
    1) PCIe/Communication Bottlenecks:
        For data transfer between CPU and GPU, PCIe bandwidth (typically 16-64 GB/s) is far lower than GPU memory bandwidth (e.g., 1 TB/s for high-end GPUs),
   dragging down overall performance . 2) PCIe bandwidth depends on generation (e.g., PCIe 3.0 x16 = ~16 GB/s, PCIe 4.0 x16 = ~32 GB/s). Overhead: Actual
   bandwidth is lower due to protocol overhead (~20% for PCIe 3.0/4.0). Bidirectional Transfer: PCIe is full-duplex, but simultaneous reads/writes may contend
   for bandwidth.
*/
void inline PCIE_test(int flag = 0x0) {
    // char pciBusId;
    // cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id);
    // Use system tools (e.g., `lspci -vv` on Linux) to check PCIe link speed/lanes

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *host_data, *device_data;
    size_t data_size = 1 << 26;  // 64 MB
    cudaMallocHost(&host_data, data_size);
    cudaMalloc(&device_data, data_size);

    cudaEventRecord(start);
    cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double bandwidth_GBs = (data_size / 1e9) / (milliseconds / 1000);

    printf("PCIe Bandwidth: %.2f GB/s\n", bandwidth_GBs);
}

// Helper function determines the maximum number of block sums
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    // NOTE: this needs to be kept in sync with `global_norm_squared` below.
    const int block_size = 512;
    const int grid_size  = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);
    int max_num_block_sums = 0;
    for (int i = 0; i < numel; i++) {
        int num_slices     = num_slices_all[i];
        const int gx       = CEIL_DIV(grid_size, num_slices);
        const int gy       = num_slices;
        max_num_block_sums = max(max_num_block_sums, gx * gy);
    }

    return max_num_block_sums;
}

template <class T>
__global__ inline void CU_X2_partial(float* out,const T* data, size_t count) {
    size_t index      = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f,a;
    for (size_t i = index; i < count; i += grid_width) {
        a = (float)data[i];
        accumulator += a * a;
    }    
    float block_sum = blockReduce<warpReduceSum>(accumulator);
    if (threadIdx.x == 0) {
        size_t out_index = blockIdx.x;
        out[out_index]   = block_sum;
    }
}

template <class T>
__device__ inline float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index      = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f,a;
    for (size_t i = index; i < count; i += grid_width) {
        a = (float)data[i];
        accumulator += a * a;
        // assert(!isnan(a) && !isinf(a));
    }
    // block-level reduce
    return blockReduce<warpReduceSum>(accumulator);
}
template <class T>
__global__ static void global_norm_squared_2D(float* out, const T* data, size_t count, ptrdiff_t stride) {
    float block_sum = global_norm_squared_for_range(data + blockIdx.y * stride, count);
    
    // each block accumulates its partial sum to out[out_index]
    // we want to avoid using atomic add here so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if (threadIdx.x == 0) {
        size_t out_index = blockIdx.y * gridDim.x + blockIdx.x;
        out[out_index]   = out[out_index] + block_sum;
        // assert(!isnan(block_sum) && !isinf(block_sum));
    }
}
template <typename T>
inline float global_norm_squared(const T* values, size_t count, ptrdiff_t stride, int num_slices, bool reset, cudaStream_t stream, int flag = 0) {
    float a, *norm2 = (float*)(GTensor::bt4c->data);
    // int slices[2] = {1, 1}, max_num_block_sums = get_max_num_block_sums(slices, 2);
    constexpr int block_size = 512;  // 256 may be better for shared memory of CU_x2_
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact .
    auto now             = GST_us();
    const int dMaxThread = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount, grid_size = dMaxThread / block_size;
    if (DEBUG.algCuX2 == 0) {   // too complex
        assert(grid_size > 0);  // gives a better error than letting the call below fail
        const int gx = CEIL_DIV(grid_size, num_slices), gy = num_slices;
        assert(gx * gy < 1024);  // we want to later accumulate the block sums in a single block
        if (reset) {
            cudaCheck(cudaMemsetAsync(norm2, 0, grid_size * sizeof(float), stream));
        }
        global_norm_squared_2D<<<dim3(gx, gy), block_size, 0, stream>>>(norm2, values, count, stride);
        cudaCheck(cudaGetLastError());
        global_sum_deterministic(norm2, norm2, grid_size, main_stream);
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        size_t smemPB = 1024 * sizeof(float);
        int dBLOCK = 512, dGRID = dMaxThread / dBLOCK;
        dGRID = 512;
        assert(dGRID < 1024);  //  blockReduce<warpReduceSum>
        cudaCheck(cudaMemset(norm2, 0, sizeof(float)*dGRID));
        CU_x2_<T><<<dGRID, dBLOCK, smemPB, main_stream>>>(norm2, values, count);  //  0.00190092938
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        cudaStreamSynchronize(main_stream);
    }
    // SUM::tX1 += GST_us()-now;
    return a;
}