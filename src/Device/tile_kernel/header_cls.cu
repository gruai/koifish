#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void __launch_bounds__(128, 1) header_cls_T64_64_S49152_bfloat16(bfloat16_t* __restrict__ grad_pre_logits, const int* __restrict__ labels, float* __restrict__ losses, bfloat16_t* __restrict__ pre_logits, int N, int nValidToken) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 512));
  float lse[1];
  bfloat16_t z[1];
  bfloat16_t row[1187];
  bfloat16_t max_z[1];
  float sum_exp[1];
  int label = labels[((int64_t)((int)blockIdx.x))];
  int condval;
  if ((0 <= label)) {
    condval = 0;
  } else {
    condval = 1;
  }
  if (condval == 0) {
    lse[0] = -CUDART_INF_F;
    __syncthreads();
    for (int i_v = 0; i_v < 1187; ++i_v) {
      z[0] = pre_logits[(((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + (((int64_t)i_v) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))];
      max_z[0] = -std::numeric_limits<bfloat16_t>::infinity();
      max_z[0] = cutlass::bfloat16_t(__hmax((max_z[0]).to_nv_bfloat16(), (z[0]).to_nv_bfloat16()));
      max_z[0] = tl::AllReduce<tl::MaxOp, 128, 1, 0>::run(max_z[0], (&(((bfloat16_t*)workspace_1)[0])));
      z[0] = ((bfloat16_t)exp2f(((((float)z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))));
      sum_exp[0] = 0x0p+0f/*0.000000e+00*/;
      sum_exp[0] = (sum_exp[0] + ((float)z[0]));
      sum_exp[0] = tl::AllReduce<tl::SumOp, 128, 1, 0>::run(sum_exp[0], (&(((float*)workspace)[0])));
      lse[0] = ((((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) + log2f((exp2f((lse[0] - (((float)max_z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/))) + sum_exp[0])));
    }
    for (int i_v_1 = 0; i_v_1 < 1187; ++i_v_1) {
      z[0] = pre_logits[(((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + (((int64_t)i_v_1) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))];
      z[0] = ((bfloat16_t)(exp2f(((((float)z[0]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/) - lse[0])) / ((float)nValidToken)));
      grad_pre_logits[(((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + (((int64_t)i_v_1) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = z[0];
    }
    __syncthreads();
    bfloat16_t condval_1;
    if (((0 <= label) && (label < 151936))) {
      condval_1 = grad_pre_logits[((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + ((int64_t)label))];
    } else {
      condval_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    bfloat16_t p_label = (condval_1 * ((bfloat16_t)nValidToken));
    float p_label_1 = max(((float)p_label), 0x1.79ca10c924223p-67f/*1.000000e-20*/);
    losses[((int64_t)((int)blockIdx.x))] = (logf(max(((float)p_label), 0x1.79ca10c924223p-67f/*1.000000e-20*/)) * -0x1p+0f/*-1.000000e+00*/);
    if (0 <= label) {
      if (label < 151936) {
        grad_pre_logits[((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + ((int64_t)label))] = ((bfloat16_t)((max(((float)p_label), 0x1.79ca10c924223p-67f/*1.000000e-20*/) - 0x1p+0f/*1.000000e+00*/) / ((float)nValidToken)));
      }
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 1187; ++i) {
      row[i] = bfloat16_t(0x0p+0f/*0.000000e+00*/);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 1187; ++i_1) {
      pre_logits[(((((int64_t)((int)blockIdx.x)) * (int64_t)151936) + (((int64_t)i_1) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = row[i_1];
    }
    losses[((int64_t)((int)blockIdx.x))] = 0x0p+0f/*0.000000e+00*/;
  }
}
