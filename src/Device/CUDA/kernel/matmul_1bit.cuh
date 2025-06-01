/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *        
 * 
 *  \brief From https://github.com/microsoft/BitNet
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
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
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