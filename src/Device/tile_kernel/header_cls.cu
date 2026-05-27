#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void __launch_bounds__(128, 1) header_cls_T128_128_S32768_bfloat16(bfloat16_t* __restrict__ grad_pre_logits, const int* __restrict__ labels, float* __restrict__ losses, const bfloat16_t* __restrict__ pre_logits, int N) {
  float lse[1];
  bfloat16_t z[128];
  bfloat16_t max_z[1];
  float sum_exp[1];
  lse[0] = -CUDART_INF_F;
  int label = labels[((int64_t)((int)blockIdx.x))];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    *(uint4*)(z + (i * 8)) = *(uint4*)(pre_logits + ((((int64_t)((int)blockIdx.x)) * (int64_t)128) + (((int64_t)i) * (int64_t)8)));
  }
  max_z[0] = -std::numeric_limits<bfloat16_t>::infinity();
  #pragma unroll
  for (int rv = 0; rv < 128; ++rv) {
    max_z[0] = cutlass::bfloat16_t(__hmax((max_z[0]).to_nv_bfloat16(), (z[rv]).to_nv_bfloat16()));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    float broadcast_var = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint2 __1;
    float4 __2;
    float4 __3;
      float4 __4;
        float4 __5;
        uint2 v_ = *(uint2*)(z + (i_1 * 4));
        ((float2*)(&__5))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[0]);
        ((float2*)(&__5))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v_))[1]);
        float4 v__1 = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
        __4.x = (__5.x*v__1.x);
        __4.y = (__5.y*v__1.y);
        __4.z = (__5.z*v__1.z);
        __4.w = (__5.w*v__1.w);
      float4 v__2 = make_float4((((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/), (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/));
      __3.x = (__4.x-v__2.x);
      __3.y = (__4.y-v__2.y);
      __3.z = (__4.z-v__2.z);
      __3.w = (__4.w-v__2.w);
    __2.x = exp2f(__3.x);
    __2.y = exp2f(__3.y);
    __2.z = exp2f(__3.z);
    __2.w = exp2f(__3.w);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&__2))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&__2))[1]);
    *(uint2*)(z + (i_1 * 4)) = __1;
  }
  sum_exp[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 128; ++rv_1) {
    sum_exp[0] = (sum_exp[0] + ((float)z[rv_1]));
  }
  lse[0] = ((((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) + log2f((exp2f((lse[0] - (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))) + sum_exp[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 16; ++i_2) {
    bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if (((label < 128) && (0 <= label))) {
      condval = *(uint4*)(pre_logits + ((((int64_t)((int)blockIdx.x)) * (int64_t)128) + (((int64_t)i_2) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(z + (i_2 * 8)) = condval;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    float broadcast_var_2 = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    uint2 __6;
    float4 __7;
    float4 __8;
      float4 __9;
        float4 __10;
        uint2 v__3 = *(uint2*)(z + (i_3 * 4));
        ((float2*)(&__10))[0] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[0]);
        ((float2*)(&__10))[1] = __bfloat1622float2((reinterpret_cast<__nv_bfloat162*>(&v__3))[1]);
        float4 v__4 = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
        __9.x = (__10.x*v__4.x);
        __9.y = (__10.y*v__4.y);
        __9.z = (__10.z*v__4.z);
        __9.w = (__10.w*v__4.w);
      float4 v__5 = make_float4(lse[0], lse[0], lse[0], lse[0]);
      __8.x = (__9.x-v__5.x);
      __8.y = (__9.y-v__5.y);
      __8.z = (__9.z-v__5.z);
      __8.w = (__9.w-v__5.w);
    __7.x = exp2f(__8.x);
    __7.y = exp2f(__8.y);
    __7.z = exp2f(__8.z);
    __7.w = exp2f(__8.w);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[0] = __float22bfloat162_rn(((float2*)(&__7))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__6))[1] = __float22bfloat162_rn(((float2*)(&__7))[1]);
    *(uint2*)(z + (i_3 * 4)) = __6;
  }
  float p_label = ((float)z[(label & 127)]);
  losses[((int64_t)((int)blockIdx.x))] = ((log2f(p_label) * -0x1p+0f/*-1.000000e+00*/) - lse[0]);
  if (0 <= label) {
    if (label < 128) {
      grad_pre_logits[((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((int64_t)label))] = ((bfloat16_t)(p_label - 0x1p+0f/*1.000000e+00*/));
    }
  }
}
