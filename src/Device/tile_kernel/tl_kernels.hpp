#pragma once

#define KOIFISH_TL_INFO "tilelang_0.1.10+cuda.git497b1d45"

#ifdef __USE_TILELANG__



    
extern "C" __global__ void header_cls_T64_64_S49152_bfloat16(__nv_bfloat16* __restrict__ grad_pre_logits, const int* __restrict__ labels, float* __restrict__ losses, __nv_bfloat16* __restrict__ pre_logits, int N, int nValidToken);

#else

#endif