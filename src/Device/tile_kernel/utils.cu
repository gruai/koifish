#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void __launch_bounds__(2048, 1) tl_scale__T128_128_S32768_bfloat16(bfloat16_t* __restrict__ C, int M, int N, float alpha) {
  extern __shared__ __align__(1024) bfloat16_t C_shared[];
  bfloat16_t C_shared_local_cast[8];
  bfloat16_t C_local[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    bfloat16_t condval;
    if (((((((int)blockIdx.y) * 128) + (((int)threadIdx.x) >> 4)) < M) && ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) & 15) * 8)) + i) < N))) {
      condval = C[((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) * ((int64_t)N))) + ((int64_t)i))];
    } else {
      condval = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    C_shared[((((int)threadIdx.x) * 8) + i)] = condval;
  }
  *(uint4*)(C_shared_local_cast + 0) = *(uint4*)(C_shared + (((int)threadIdx.x) * 8));
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    uint2 __1;
    float4 __2;
      float4 __3;
      uint2 v_ = *(uint2*)(C_shared_local_cast + (i_1 * 4));
      ((float2*)(&__3))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
      ((float2*)(&__3))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
      float4 v__1 = make_float4(alpha, alpha, alpha, alpha);
      __2.x = (__3.x*v__1.x);
      __2.y = (__3.y*v__1.y);
      __2.z = (__3.z*v__1.z);
      __2.w = (__3.w*v__1.w);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
    *(uint2*)(C_local + (i_1 * 4)) = __1;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    if ((((((int)blockIdx.y) * 128) + (((int)threadIdx.x) >> 4)) < M) && ((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) & 15) * 8)) + i_2) < N)) {
      C[((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)) + (((((int64_t)((int)blockIdx.y)) * (int64_t)128) + (((int64_t)((int)threadIdx.x)) >> (int64_t)4)) * ((int64_t)N))) + ((int64_t)i_2))] = C_local[i_2];
    }
  }
}
