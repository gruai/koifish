#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void __launch_bounds__(128, 1) flash_bwd_T64_16_S49152_bfloat16(const float* __restrict__ Delta, const bfloat16_t* __restrict__ K, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V, float* __restrict__ dK, const bfloat16_t* __restrict__ dO, float* __restrict__ dQ, float* __restrict__ dV, const float* __restrict__ lse) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float dv[64];
  float dk[64];
  float qkT[8];
  float dsT[8];
  bfloat16_t qkT_cast[8];
  bfloat16_t dsT_cast[8];
  float dq[16];
  float delta_local_cast_1[2];
  float delta_local_cast_2[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((int)blockIdx.z) * 1048576) + (((int)blockIdx.y) * 65536)) + (i * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = *(uint4*)(V + ((((((((int)blockIdx.z) * 1048576) + (((int)blockIdx.y) * 65536)) + (i_1 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.x) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dv + (i_2 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 16; ++i_3) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dk + (i_3 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::cp_async_wait<0>();
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 8192)])), (&(lse[((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16448)])), (&(dO[((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 131072)) + (i_4 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 10272)])), (&(Delta[((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20608)])), (&(Q[((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 131072)) + (i_5 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::cp_async_wait<0>();
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 8208)])), (&(lse[(((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 16)])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18496)])), (&(dO[(((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 131072)) + (i_6 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
  }
  tl::cp_async_commit();
  tl::cp_async_wait<0>();
  __syncthreads();
  if (((int)threadIdx.x) < 16) {
    tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 10288)])), (&(Delta[(((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 16)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 2; ++i_7) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22656)])), (&(Q[(((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 131072)) + (i_7 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
  }
  tl::cp_async_commit();
  for (int k = (((int)blockIdx.y) * 4); k < 62; ++k) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(qkT + (i_8 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[8];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 2048) + ((ki >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20608)])), (&(B_local[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      float broadcast_var_3 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      float2 __1;
      float2 __2;
        float2 __3;
          float2 v_ = *(float2*)(qkT + (i_9 * 2));
          float2 v__1 = make_float2(broadcast_var_3, broadcast_var_3);
          __3.x = (v_.x*v__1.x);
          __3.y = (v_.y*v__1.y);
        float2 v__2 = *(float2*)(((float*)buf_dyn_shmem) + (((((k & 1) * 16) + ((i_9 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
        __2.x = (__3.x-v__2.x);
        __2.y = (__3.y-v__2.y);
      __1.x = exp2f(__2.x);
      __1.y = exp2f(__2.y);
      *(float2*)(qkT + (i_9 * 2)) = __1;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[((((k & 1) * 16) + ((int)threadIdx.x)) + 8192)])), (&(lse[(((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (k * 16)) + ((int)threadIdx.x)) + 32)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float condval;
      if ((((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_10 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= ((((k * 16) + ((i_10 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_10 & 1)))) {
        condval = qkT[i_10];
      } else {
        condval = 0x0p+0f/*0.000000e+00*/;
      }
      qkT[i_10] = condval;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dsT + (i_11 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_1 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_1[0])));
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 2048) + ((ki_1 >> 2) * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16448)])), (&(B_local_1[0])));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      uint2 __4;
      float4 v__3 = *(float4*)(qkT + (i_12 * 4));
      (reinterpret_cast<__nv_bfloat162*>(&__4))[0] = __float22bfloat162_rn(((float2*)(&v__3))[0]);
      (reinterpret_cast<__nv_bfloat162*>(&__4))[1] = __float22bfloat162_rn(((float2*)(&v__3))[1]);
      *(uint2*)(qkT_cast + (i_12 * 4)) = __4;
    }
    {
      bfloat16_t B_local_2[64];
      for (int i_13 = 0; i_13 < 8; ++i_13) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((k & 1) * 2048) + ((i_13 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16448)])), (&(B_local_2[(i_13 * 8)])));
      }
      for (int j = 0; j < 8; ++j) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + (j * 8)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + (i_14 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16448)])), (&(dO[(((((((((int)blockIdx.z) * 2097152) + (k * 32768)) + (i_14 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      *(float2*)(delta_local_cast_1 + 0) = *(float2*)(((float*)buf_dyn_shmem) + (((((k & 1) * 16) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 10272));
      float broadcast_var_5 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
      uint1 __5;
      float2 __6;
        float2 __7;
          float2 v__4 = *(float2*)(qkT + (i_15 * 2));
          float2 __8;
            float2 v__5 = *(float2*)(dsT + (i_15 * 2));
            float2 v__6 = *(float2*)(delta_local_cast_1 + 0);
            __8.x = (v__5.x-v__6.x);
            __8.y = (v__5.y-v__6.y);
          __7.x = (v__4.x*__8.x);
          __7.y = (v__4.y*__8.y);
        float2 v__7 = make_float2(broadcast_var_5, broadcast_var_5);
        __6.x = (__7.x*v__7.x);
        __6.y = (__7.y*v__7.y);
      (reinterpret_cast<__nv_bfloat162*>(&__5))[0] = __float22bfloat162_rn(((float2*)(&__6))[0]);
      *(uint1*)(dsT_cast + (i_15 * 2)) = __5;
    }
    __syncthreads();
    if (((int)threadIdx.x) < 16) {
      tl::cp_async_gs<4>((&(((float*)buf_dyn_shmem)[((((k & 1) * 16) + ((int)threadIdx.x)) + 10272)])), (&(Delta[(((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (k * 16)) + ((int)threadIdx.x)) + 32)])));
    }
    tl::cp_async_commit();
    {
      bfloat16_t B_local_3[64];
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((k & 1) * 2048) + ((i_16 >> 2) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_16 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_16 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(B_local_3[(i_16 * 8)])));
      }
      for (int j_1 = 0; j_1 < 8; ++j_1) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + (j_1 * 8)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((k & 1) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 1024)) + (i_17 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20608)])), (&(Q[(((((((((int)blockIdx.z) * 2097152) + (k * 32768)) + (i_17 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 384) + ((i_18 & 1) * 192)) + (((((int)threadIdx.x) & 31) >> 2) * 24)) + ((i_18 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24704)) = *(uint1*)(dsT_cast + (i_18 * 2));
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(dq + (i_19 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    {
      bfloat16_t A_local_2[8];
      bfloat16_t B_local_4[16];
      __syncthreads();
      for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 * 384) + (((((int)threadIdx.x) & 31) >> 4) * 192)) + ((((int)threadIdx.x) & 7) * 24)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 24704)])), (&(A_local_2[0])));
        for (int i_20 = 0; i_20 < 2; ++i_20) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 4096) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_20) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_4[(i_20 * 8)])));
        }
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_2 * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 16; ++i_21) {
      AtomicAdd((&(dQ[((((((((((int)blockIdx.z) * 2097152) + (k * 32768)) + (((i_21 & 3) >> 1) * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_21 >> 2) * 64)) + ((i_21 & 1) * 32)) + (((int)threadIdx.x) & 31))])), dq[i_21]);
    }
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(qkT + (i_22 * 4)) = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local_3[8];
    bfloat16_t B_local_5[8];
    for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_3 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_3[0])));
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_3 >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_3 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20608)])), (&(B_local_5[0])));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 0), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 4), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_5 + 4));
    }
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    float broadcast_var_8 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __9;
    float2 __10;
      float2 __11;
        float2 v__8 = *(float2*)(qkT + (i_23 * 2));
        float2 v__9 = make_float2(broadcast_var_8, broadcast_var_8);
        __11.x = (v__8.x*v__9.x);
        __11.y = (v__8.y*v__9.y);
      float2 v__10 = *(float2*)(((float*)buf_dyn_shmem) + ((((i_23 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 8192));
      __10.x = (__11.x-v__10.x);
      __10.y = (__11.y-v__10.y);
    __9.x = exp2f(__10.x);
    __9.y = exp2f(__10.y);
    *(float2*)(qkT + (i_23 * 2)) = __9;
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    float condval_1;
    if ((((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_24 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_24 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_24 & 1)) + 992))) {
      condval_1 = qkT[i_24];
    } else {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    }
    qkT[i_24] = condval_1;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dsT + (i_25 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local_4[8];
    bfloat16_t B_local_6[8];
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_4[0])));
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_4 >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16448)])), (&(B_local_6[0])));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 0), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 4), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_6 + 4));
    }
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    uint2 __12;
    float4 v__11 = *(float4*)(qkT + (i_26 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__12))[0] = __float22bfloat162_rn(((float2*)(&v__11))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__12))[1] = __float22bfloat162_rn(((float2*)(&v__11))[1]);
    *(uint2*)(qkT_cast + (i_26 * 4)) = __12;
  }
  {
    bfloat16_t B_local_7[64];
    for (int i_27 = 0; i_27 < 8; ++i_27) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_27 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_27 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16448)])), (&(B_local_7[(i_27 * 8)])));
    }
    for (int j_3 = 0; j_3 < 8; ++j_3) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + (j_3 * 8)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_7 + (j_3 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_7 + ((j_3 * 8) + 4)));
    }
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 4; ++i_28) {
    *(float2*)(delta_local_cast_2 + 0) = *(float2*)(((float*)buf_dyn_shmem) + ((((i_28 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 10272));
    float broadcast_var_10 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
    uint1 __13;
    float2 __14;
      float2 __15;
        float2 v__12 = *(float2*)(qkT + (i_28 * 2));
        float2 __16;
          float2 v__13 = *(float2*)(dsT + (i_28 * 2));
          float2 v__14 = *(float2*)(delta_local_cast_2 + 0);
          __16.x = (v__13.x-v__14.x);
          __16.y = (v__13.y-v__14.y);
        __15.x = (v__12.x*__16.x);
        __15.y = (v__12.y*__16.y);
      float2 v__15 = make_float2(broadcast_var_10, broadcast_var_10);
      __14.x = (__15.x*v__15.x);
      __14.y = (__15.y*v__15.y);
    (reinterpret_cast<__nv_bfloat162*>(&__13))[0] = __float22bfloat162_rn(((float2*)(&__14))[0]);
    *(uint1*)(dsT_cast + (i_28 * 2)) = __13;
  }
  {
    bfloat16_t B_local_8[64];
    for (int i_29 = 0; i_29 < 8; ++i_29) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_29 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 20608)])), (&(B_local_8[(i_29 * 8)])));
    }
    for (int j_4 = 0; j_4 < 8; ++j_4) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + (j_4 * 8)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_8 + (j_4 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_8 + ((j_4 * 8) + 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 4; ++i_30) {
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 384) + ((i_30 & 1) * 192)) + (((((int)threadIdx.x) & 31) >> 2) * 24)) + ((i_30 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24704)) = *(uint1*)(dsT_cast + (i_30 * 2));
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 4; ++i_31) {
    float broadcast_var_11 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dq + (i_31 * 4)) = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
  }
  {
    bfloat16_t A_local_5[8];
    bfloat16_t B_local_9[16];
    __syncthreads();
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_5 * 384) + (((((int)threadIdx.x) & 31) >> 4) * 192)) + ((((int)threadIdx.x) & 7) * 24)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 24704)])), (&(A_local_5[0])));
      for (int i_32 = 0; i_32 < 2; ++i_32) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 4096) + (ki_5 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_32) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_9[(i_32 * 8)])));
      }
      for (int j_5 = 0; j_5 < 2; ++j_5) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + (j_5 * 8)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_9 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_5 + 0), reinterpret_cast<const unsigned*>(B_local_9 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 16; ++i_33) {
    AtomicAdd((&(dQ[((((((((((int)blockIdx.z) * 2097152) + (((i_33 & 3) >> 1) * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_33 >> 2) * 64)) + ((i_33 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 2031616)])), dq[i_33]);
  }
  #pragma unroll
  for (int i_34 = 0; i_34 < 2; ++i_34) {
    float broadcast_var_12 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(qkT + (i_34 * 4)) = make_float4(broadcast_var_12, broadcast_var_12, broadcast_var_12, broadcast_var_12);
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local_6[8];
    bfloat16_t B_local_10[8];
    for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_6 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_6[0])));
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_6 >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22656)])), (&(B_local_10[0])));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 0), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(qkT + 4), reinterpret_cast<const unsigned*>(A_local_6 + 0), reinterpret_cast<const unsigned*>(B_local_10 + 4));
    }
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 4; ++i_35) {
    float broadcast_var_13 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __17;
    float2 __18;
      float2 __19;
        float2 v__16 = *(float2*)(qkT + (i_35 * 2));
        float2 v__17 = make_float2(broadcast_var_13, broadcast_var_13);
        __19.x = (v__16.x*v__17.x);
        __19.y = (v__16.y*v__17.y);
      float2 v__18 = *(float2*)(((float*)buf_dyn_shmem) + ((((i_35 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 8208));
      __18.x = (__19.x-v__18.x);
      __18.y = (__19.y-v__18.y);
    __17.x = exp2f(__18.x);
    __17.y = exp2f(__18.y);
    *(float2*)(qkT + (i_35 * 2)) = __17;
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 8; ++i_36) {
    float condval_2;
    if ((((((((int)blockIdx.y) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_36 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((i_36 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_36 & 1)) + 1008))) {
      condval_2 = qkT[i_36];
    } else {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    }
    qkT[i_36] = condval_2;
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 2; ++i_37) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dsT + (i_37 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  {
    bfloat16_t A_local_7[8];
    bfloat16_t B_local_11[8];
    for (int ki_7 = 0; ki_7 < 8; ++ki_7) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_7 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 8192)])), (&(A_local_7[0])));
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_7 >> 2) * 1024) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_7 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_7 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18496)])), (&(B_local_11[0])));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 0), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_11 + 0));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dsT + 4), reinterpret_cast<const unsigned*>(A_local_7 + 0), reinterpret_cast<const unsigned*>(B_local_11 + 4));
    }
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 2; ++i_38) {
    uint2 __20;
    float4 v__19 = *(float4*)(qkT + (i_38 * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__20))[0] = __float22bfloat162_rn(((float2*)(&v__19))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__20))[1] = __float22bfloat162_rn(((float2*)(&v__19))[1]);
    *(uint2*)(qkT_cast + (i_38 * 4)) = __20;
  }
  {
    bfloat16_t B_local_12[64];
    for (int i_39 = 0; i_39 < 8; ++i_39) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_39 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_39 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_39 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 18496)])), (&(B_local_12[(i_39 * 8)])));
    }
    for (int j_6 = 0; j_6 < 8; ++j_6) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + (j_6 * 8)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_12 + (j_6 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dv + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(qkT_cast + 0), reinterpret_cast<const unsigned*>(B_local_12 + ((j_6 * 8) + 4)));
    }
  }
  if ((((bool)0 & (bool)0) & (bool)0) & (bool)0) {
    tl::cp_async_wait<0>();
  }
  __syncthreads();
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    *(float2*)(delta_local_cast_2 + 0) = *(float2*)(((float*)buf_dyn_shmem) + ((((i_40 >> 1) * 8) + ((((int)threadIdx.x) & 3) * 2)) + 10288));
    float broadcast_var_15 = 0x1.6a09e667f3bcdp-4f/*8.838835e-02*/;
    uint1 __21;
    float2 __22;
      float2 __23;
        float2 v__20 = *(float2*)(qkT + (i_40 * 2));
        float2 __24;
          float2 v__21 = *(float2*)(dsT + (i_40 * 2));
          float2 v__22 = *(float2*)(delta_local_cast_2 + 0);
          __24.x = (v__21.x-v__22.x);
          __24.y = (v__21.y-v__22.y);
        __23.x = (v__20.x*__24.x);
        __23.y = (v__20.y*__24.y);
      float2 v__23 = make_float2(broadcast_var_15, broadcast_var_15);
      __22.x = (__23.x*v__23.x);
      __22.y = (__23.y*v__23.y);
    (reinterpret_cast<__nv_bfloat162*>(&__21))[0] = __float22bfloat162_rn(((float2*)(&__22))[0]);
    *(uint1*)(dsT_cast + (i_40 * 2)) = __21;
  }
  {
    bfloat16_t B_local_13[64];
    for (int i_41 = 0; i_41 < 8; ++i_41) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((i_41 >> 2) * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_41 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_41 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 22656)])), (&(B_local_13[(i_41 * 8)])));
    }
    for (int j_7 = 0; j_7 < 8; ++j_7) {
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + (j_7 * 8)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_13 + (j_7 * 8)));
      tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dk + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(dsT_cast + 0), reinterpret_cast<const unsigned*>(B_local_13 + ((j_7 * 8) + 4)));
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_42 = 0; i_42 < 4; ++i_42) {
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 384) + ((i_42 & 1) * 192)) + (((((int)threadIdx.x) & 31) >> 2) * 24)) + ((i_42 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 24704)) = *(uint1*)(dsT_cast + (i_42 * 2));
  }
  #pragma unroll
  for (int i_43 = 0; i_43 < 4; ++i_43) {
    float broadcast_var_16 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(dq + (i_43 * 4)) = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
  }
  {
    bfloat16_t A_local_8[8];
    bfloat16_t B_local_14[16];
    __syncthreads();
    for (int ki_8 = 0; ki_8 < 4; ++ki_8) {
      tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_8 * 384) + (((((int)threadIdx.x) & 31) >> 4) * 192)) + ((((int)threadIdx.x) & 7) * 24)) + (((((int)threadIdx.x) & 15) >> 3) * 8)) + 24704)])), (&(A_local_8[0])));
      for (int i_44 = 0; i_44 < 2; ++i_44) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 4096) + (ki_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_44) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(B_local_14[(i_44 * 8)])));
      }
      for (int j_8 = 0; j_8 < 2; ++j_8) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + (j_8 * 8)), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_14 + (j_8 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(dq + ((j_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_8 + 0), reinterpret_cast<const unsigned*>(B_local_14 + ((j_8 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_45 = 0; i_45 < 16; ++i_45) {
    AtomicAdd((&(dQ[((((((((((int)blockIdx.z) * 2097152) + (((i_45 & 3) >> 1) * 16384)) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) >> 5) * 256)) + ((i_45 >> 2) * 64)) + ((i_45 & 1) * 32)) + (((int)threadIdx.x) & 31)) + 2064384)])), dq[i_45]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_46 = 0; i_46 < 32; ++i_46) {
    *(float2*)(((float*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_46 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_46 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(dv + (i_46 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_47 = 0; i_47 < 64; ++i_47) {
    AtomicAdd((&(dV[(((((((int)blockIdx.z) * 1048576) + (((int)blockIdx.y) * 65536)) + (i_47 * 1024)) + ((((int)blockIdx.x) >> 1) * 128)) + ((int)threadIdx.x))])), ((float*)buf_dyn_shmem)[((i_47 * 128) + ((int)threadIdx.x))], 0);
  }
  __syncthreads();
  #pragma unroll
  for (int i_48 = 0; i_48 < 32; ++i_48) {
    *(float2*)(((float*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_48 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_48 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(dk + (i_48 * 2));
  }
  __syncthreads();
  #pragma unroll
  for (int i_49 = 0; i_49 < 64; ++i_49) {
    AtomicAdd((&(dK[(((((((int)blockIdx.z) * 1048576) + (((int)blockIdx.y) * 65536)) + (i_49 * 1024)) + ((((int)blockIdx.x) >> 1) * 128)) + ((int)threadIdx.x))])), ((float*)buf_dyn_shmem)[((i_49 * 128) + ((int)threadIdx.x))], 0);
  }
}
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void __launch_bounds__(128, 1) flash_fwd_T64_32_S36864_bfloat16(const bfloat16_t* __restrict__ K, bfloat16_t* __restrict__ Output, const bfloat16_t* __restrict__ Q, const bfloat16_t* __restrict__ V, float* __restrict__ lse) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[16];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  float scores_max_clear[2];
  bfloat16_t acc_s_cast_local_cast[2];
  float scores_max_clear_1[2];
  bfloat16_t acc_s_cast_local_cast_1[2];
  bfloat16_t Output_local_cast_2[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + (i * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(logsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(scores_max + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[(((((((int)blockIdx.z) * 1048576) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.y) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(V[(((((((int)blockIdx.z) * 1048576) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.y) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < ((((int)blockIdx.x) * 2) + 1); ++k) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      float condval;
      if ((((((k * 32) + ((i_4 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_4 & 1)) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
        condval = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval = -CUDART_INF_F;
      }
      acc_s[i_4] = condval;
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      bfloat16_t A_local[8];
      bfloat16_t B_local[16];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local[0])));
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki >> 2) * 2048) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local[(i_5 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[(((((((((int)blockIdx.z) * 1048576) + (k * 32768)) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.y) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
    }
    tl::cp_async_commit();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      scores_max_clear[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max_clear[i_7] = max(scores_max_clear[i_7], acc_s[((((rv & 3) * 4) + (i_7 * 2)) + (rv >> 2))]);
      }
      scores_max_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear[i_7]);
      scores_max[i_7] = max(scores_max[i_7], scores_max_clear[i_7]);
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 2; ++i_8) {
      scores_max[i_8] = max(scores_max[i_8], scores_max_prev[i_8]);
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_scale[i_9] = exp2f(((scores_max_prev[i_9] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_9] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 64; ++i_10) {
      acc_o[i_10] = (acc_o[i_10] * scores_scale[((i_10 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 16; ++i_11) {
      acc_s[i_11] = exp2f(((acc_s[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_11 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_12 * 2));
      (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_cast_local_cast + 0) = __1;
      *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) >> 5) * 512) + ((i_12 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_12 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_12 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = *(uint1*)(acc_s_cast_local_cast + 0);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      bfloat16_t A_local_1[8];
      bfloat16_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 16384)])), (&(A_local_1[0])));
        for (int i_13 = 0; i_13 < 8; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_13 >> 2) * 2048) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_13 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_13 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      tl::cp_async_gs<16>((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_14 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)])), (&(V[(((((((((int)blockIdx.z) * 1048576) + (k * 32768)) + (i_14 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + ((((int)blockIdx.y) >> 1) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 32768)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      scores_sum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_15] = (scores_sum[i_15] + acc_s[((((rv_1 & 3) * 4) + (i_15 * 2)) + (rv_1 >> 2))]);
      }
      scores_sum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 2; ++i_16) {
      logsum[i_16] = ((logsum[i_16] * scores_scale[i_16]) + scores_sum[i_16]);
    }
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 16; ++i_17) {
    float condval_1;
    if (((((((((int)blockIdx.x) * 64) + ((i_17 >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_17 & 1)) + 32) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_17 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
      condval_1 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_1 = -CUDART_INF_F;
    }
    acc_s[i_17] = condval_1;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    bfloat16_t A_local_2[8];
    bfloat16_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])), (&(A_local_2[0])));
      for (int i_18 = 0; i_18 < 2; ++i_18) {
        tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((((ki_2 >> 2) * 2048) + (i_18 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_2[(i_18 * 8)])));
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
      }
    }
  }
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    scores_max_clear_1[i_19] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      scores_max_clear_1[i_19] = max(scores_max_clear_1[i_19], acc_s[((((rv_2 & 3) * 4) + (i_19 * 2)) + (rv_2 >> 2))]);
    }
    scores_max_clear_1[i_19] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max_clear_1[i_19]);
    scores_max[i_19] = max(scores_max[i_19], scores_max_clear_1[i_19]);
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    scores_max[i_20] = max(scores_max[i_20], scores_max_prev[i_20]);
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    scores_scale[i_21] = exp2f(((scores_max_prev[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 64; ++i_22) {
    acc_o[i_22] = (acc_o[i_22] * scores_scale[((i_22 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 16; ++i_23) {
    acc_s[i_23] = exp2f(((acc_s[i_23] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_23 & 3) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_s + (i_24 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__2))[0] = __float22bfloat162_rn(((float2*)(&v__1))[0]);
    *(uint1*)(acc_s_cast_local_cast_1 + 0) = __2;
    *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) >> 5) * 512) + ((i_24 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_24 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_24 & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384)) = *(uint1*)(acc_s_cast_local_cast_1 + 0);
  }
  {
    bfloat16_t A_local_3[8];
    bfloat16_t B_local_3[64];
    __syncthreads();
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 5) * 512) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki_3) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + 16384)])), (&(A_local_3[0])));
      for (int i_25 = 0; i_25 < 8; ++i_25) {
        tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((i_25 >> 2) * 2048) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_25 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_25 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])), (&(B_local_3[(i_25 * 8)])));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    scores_sum[i_26] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      scores_sum[i_26] = (scores_sum[i_26] + acc_s[((((rv_3 & 3) * 4) + (i_26 * 2)) + (rv_3 >> 2))]);
    }
    scores_sum[i_26] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    logsum[i_27] = ((logsum[i_27] * scores_scale[i_27]) + scores_sum[i_27]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 64; ++i_28) {
    acc_o[i_28] = (acc_o[i_28] / logsum[((i_28 & 3) >> 1)]);
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 32; ++i_29) {
    uint1 __3;
    float2 v__2 = *(float2*)(acc_o + (i_29 * 2));
    (reinterpret_cast<__nv_bfloat162*>(&__3))[0] = __float22bfloat162_rn(((float2*)(&v__2))[0]);
    *(uint1*)(Output_local_cast_2 + 0) = __3;
    *(uint1*)(Output + ((((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + ((((int)threadIdx.x) >> 5) * 32768)) + ((i_29 & 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.y) * 128)) + ((i_29 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(Output_local_cast_2 + 0);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    logsum[i_30] = (log2f(logsum[i_30]) + (scores_max[i_30] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_31 = 0; i_31 < 2; ++i_31) {
      lse[((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.y) * 1024)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (i_31 * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = logsum[i_31];
    }
  }
}
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void __launch_bounds__(128, 1) flash_bwd_preprocess_T32_32_S0_bfloat16(float* __restrict__ Delta, const bfloat16_t* __restrict__ O, const bfloat16_t* __restrict__ dO) {
  float acc[8];
  bfloat16_t o[8];
  bfloat16_t do_1[8];
  float delta[1];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  for (int k = 0; k < 4; ++k) {
    *(uint4*)(o + 0) = *(uint4*)(O + ((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    *(uint4*)(do_1 + 0) = *(uint4*)(dO + ((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.y) * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)));
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      float4 __1;
        float4 v_ = *(float4*)(acc + (i_1 * 4));
        float4 __2;
        uint2 __3;
          uint2 v__1 = *(uint2*)(o + (i_1 * 4));
          uint2 v__2 = *(uint2*)(do_1 + (i_1 * 4));
          *(uint1*)(&(__3.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.x))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.x)))));
          *(uint1*)(&(__3.y)) = tl::to_uint1(tl::mul2(tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__1.y))), tl::from_uint1<__nv_bfloat162>(*(uint1*)(&(v__2.y)))));
        ((float2*)(&__2))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__3))[0]);
        ((float2*)(&__2))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&__3))[1]);
        __1.x = (v_.x+__2.x);
        __1.y = (v_.y+__2.y);
        __1.z = (v_.z+__2.z);
        __1.w = (v_.w+__2.w);
      *(float4*)(acc + (i_1 * 4)) = __1;
    }
  }
  delta[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    delta[0] = (delta[0] + acc[rv]);
  }
  delta[0] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(delta[0]);
  if ((((int)threadIdx.x) % 4) == 0) {
    Delta[((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 32)) + (((int)threadIdx.x) >> 2))] = delta[0];
  }
}
#ifdef ENABLE_BF16
#endif

extern "C" __global__ void __launch_bounds__(128, 1) flash_bwd_postprocess_T64_64_S0_bfloat16(const float* __restrict__ dQ, bfloat16_t* __restrict__ dQ_out) {
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    dQ_out[(((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + (i * 2048)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))] = ((bfloat16_t)dQ[((((((((((int)blockIdx.z) * 2097152) + (((int)blockIdx.x) * 131072)) + ((i >> 3) * 16384)) + (((int)blockIdx.y) * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + ((((int)threadIdx.x) & 1) * 32)) + ((i & 7) * 4)) + ((((int)threadIdx.x) & 7) >> 1))]);
  }
}
