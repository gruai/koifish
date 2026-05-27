#pragma once

#define KOIFISH_TL_INFO "tilelang_0.1.9+cuda.git84c5f812"

#ifdef __USE_TILELANG__

extern "C" __global__ void flash_bwd_T64_16_S49152_bfloat16(const float* __restrict__ Delta, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ V, float* __restrict__ dK, const __nv_bfloat16* __restrict__ dO, float* __restrict__ dQ, float* __restrict__ dV, const float* __restrict__ lse);
extern "C" __global__ void flash_fwd_T64_32_S36864_bfloat16(const __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ Output, const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ V, float* __restrict__ lse);
extern "C" __global__ void flash_bwd_preprocess_T32_32_S0_bfloat16(float* __restrict__ Delta, const __nv_bfloat16* __restrict__ O, const __nv_bfloat16* __restrict__ dO);
extern "C" __global__ void flash_bwd_postprocess_T64_64_S0_bfloat16(const float* __restrict__ dQ, __nv_bfloat16* __restrict__ dQ_out);

#else

#endif