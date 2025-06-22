/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *        
 *  Some codes from TK/Bitnet
 * 
 *  \brief Mutmul(bit x bit)
 *  
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
  uint const i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; 

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}

template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, 
    __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  constexpr int K_per_loop = 16;
  constexpr int wmma_K = 32;
  constexpr int wmma_N = 16;
  int in_thread_C_local[1];
  signed char A_local[K_per_loop];
  int B_reshape_local[1];
  signed char B_decode_local[K_per_loop];
  int red_buf0[1];
  in_thread_C_local[0] = 0;
  #pragma unroll
  for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) {
    *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
    B_reshape_local[0] = *(int*)(B + 
      (((int)blockIdx.x) * N_block_size * K / 4) + 
      (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
      ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) +
      ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) + 
      ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) + 
      ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4)
      );
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    #pragma unroll
    for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
      in_thread_C_local[0] = __dp4a(*(int *)&A_local[((k_2_0 * 4))],*(int *)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
    }
  }
  red_buf0[0] = in_thread_C_local[0];
  #pragma unroll
  for (int offset = K_block_size/2; offset > 0; offset /= 2) {
    red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
  }
  int out_idx = ((((int)blockIdx.x) * N_block_size) + ((int)threadIdx.y));
  int ws_idx = out_idx / (N / ws_num);
  if (threadIdx.x == 0)
    dtype_transform[out_idx] = (__nv_bfloat16)(((float)red_buf0[0])/(float)s[0]*(float)ws[ws_idx]);
}

inline void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream){
    if (M == 1 && N == 3840 && K == 2560){
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 2560){
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 13824 && K == 2560){
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16><<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 6912){
        ladder_int8xint2_kernel<1, 2560, 6912, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 4800 && K == 3200){
        ladder_int8xint2_kernel<1, 4800, 3200, 6, 8, 16><<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 3200){
        ladder_int8xint2_kernel<1, 3200, 3200, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 20480 && K == 3200){
        ladder_int8xint2_kernel<1, 20480, 3200, 2, 8, 16><<<dim3(1280, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 10240){
        ladder_int8xint2_kernel<1, 3200, 10240, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }    
    else if(M == 1 && N == 5120 && K == 27648){
        ladder_int8xint2_kernel<1, 5120, 27648, 1, 8, 16><<<dim3(320, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 55296 && K == 5120){
        ladder_int8xint2_kernel<1, 55296, 5120, 1, 8, 16><<<dim3(3456, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}


/**
 * @brief Perform the HMMA.16816 operation.
 *
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                        const bf16_2 &a0, const bf16_2 &a1, const bf16_2 &a2, const bf16_2 &a3,
                                        const bf16_2 &b0, const bf16_2 &b1,
                                        const float2 &c0, const float2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
        "{%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, " \
        "{%8, %9}, " \
        "{%10, %11, %12, %13};"

        // D matrix
    :   "+f"(d0.x), "+f"(d0.y),
        "+f"(d1.x), "+f"(d1.y)

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
/**
 * @brief Perform the HMMA.16816 operation.
 *
 *
 * @param[out] d0 The first half of the output half_2 accumulator.
 * @param[out] d1 The second half of the output half_2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the half_2 accumulator matrix C.
 * @param[in] c1 The second half of the half_2 accumulator matrix C.
 */
__device__ static inline void hmma16816(      half_2 &d0,       half_2 &d1,
                                        const half_2 &a0, const half_2 &a1, const half_2 &a2, const half_2 &a3,
                                        const half_2 &b0, const half_2 &b1,
                                        const half_2 &c0, const half_2 &c1                                    ) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
        "{%0, %1}, " \
        "{%2, %3, %4, %5}, " \
        "{%6, %7}, " \
        "{%8, %9};"

        // D matrix
    :   "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))

        // A matrix
    :   "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
        "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

        // C matrix
        "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1))
    );
}

#ifdef CUDA_HOPPER
/**
* @brief Perform the HMMA.16816 operation for FP8 using fp8e4m3_2.
*
* Using mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
* but with fp8e4m3_2 (2 FP8 values) instead of fp8e4m3_4
*/
/**
 * @brief Perform the HMMA.16816 operation for FP8.
 *
 * This function performs the fp8-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0,a1,a2,a3 Input FP8 matrix A values
 * @param[in] b0,b1 Input FP8 matrix B values
 * @param[in] c0,c1 Input float2 accumulator matrix C values
 */
__device__ static inline void hmma16816(      float2 &d0,       float2 &d1,
                                       const fp8e4m3_4 &a0, const fp8e4m3_4 &a1, 
                                       const fp8e4m3_4 &a2, const fp8e4m3_4 &a3,
                                       const fp8e4m3_4 &b0, const fp8e4m3_4 &b1,
                                       const float2 &c0, const float2 &c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        
        // D matrix (output)
        : "+f"(d0.x), "+f"(d0.y),
          "+f"(d1.x), "+f"(d1.y)
        
        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
        
        // B matrix
        "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
        
        // C matrix
        "f"(c0.x), "f"(c0.y),
        "f"(c1.x), "f"(c1.y)
    );
}
#endif
