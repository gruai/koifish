/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *      
 *  \brief  From CALM
 *  \author Yingshi Chen
 */
#pragma once
#include <assert.h>
#include <cfloat>
#include <math.h>
#include <float.h>
#include <cstdint>
#include <stdint.h>

/*
    For half-precision floating-point numbers, __FLT16_MANT_DIG__ typically has a value of 11, indicating that the mantissa has 11 bits. This is important for understanding the precision and range of values that can be represented with this type. 
*/
#if defined(__FLT16_MANT_DIG__)
    typedef _Float16 __gcc_fp16;
#else
    typedef short __gcc_fp16;
#endif

// typedef uint16_t __gcc_fp16;
// how define f8 like __nv_fp8_e5m2	?
typedef uint8_t f8e5m2_t;

inline bool isValidF(float *x) {
	return !(isnan(*x) || isinf(*x));
}
inline bool isValidF(size_t n,float *x0,int flag=0x0) {
	float *x = x0;
	for(size_t i =0;i<n;i++,x++){
		if( !isValidF(x)	){
			return false;
		}
	}
	return true;
}

inline float clip(float x, float v) {
	assert(!isnan(x) && !isinf(x));
	return x < -v ? -v : (x > v ? v : x);
}
template<typename T>
inline T gelu(T x) {
	T a = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
	return a;
}

template<typename T>
inline T silu(T x) {
	return T(x / (1.0f + expf(-x)));
}

float half_to_float(__gcc_fp16 x);
__gcc_fp16 float_to_half(float x);

/*
	Lite & Smart conversion from CALM
	1.	__gcc_fp16 is compiler-dependent (GCC/Clang), IEEE 754-2008 binary16 (1 sign bit, 5 exponent bits, 10 mantissa bits).
	2. __nv_fp8x4_e5m2 is NVIDIA-specific.

inline float fp8_to_float(unsigned char v) {
	union {
		unsigned short u;
		__gcc_fp16 f;
	} u;
	u.u = v << 8;
	float a = u.f;
	return a;
}*/


inline f8e5m2_t float_to_fp8e5m2(float x) {
	__gcc_fp16 val = float_to_half(x);
	f8e5m2_t out;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	memcpy(&out, (char*)&val, sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#else
	memcpy(&out, (char*)&val + sizeof(f8e5m2_t), sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#endif
	return out;
}

inline void float_to_fp8e5m2(size_t n,float *x,f8e5m2_t*out,int flag=0x0) {
	float *src=x;
	f8e5m2_t *dst=out;
	for(size_t i =0;i<n;i++,src++,dst++){
		*dst = float_to_fp8e5m2(*src);
	}
}

typedef float (*dotprod_t)(void* w, int n, int i, float* x);
float dotprod_fp32(void* w, int n, int i, float* x);
float dotprod_fp16(void* w, int n, int i, float* x);
float dotprod_fp8(void* w, int n, int i, float* x);
float dotprod_gf4(void* w, int n, int i, float* x);
dotprod_t fnDot(typNUMBER tp);

// W (nOut,nIn) @ x (nIn) -> xout (nOut)
void D_matvec(float* xout, float* x, void* w, float* b, int nIn, int nOut, dotprod_t dotprod);
void D_matmul_sparse(float* xout, float* x, void* w, float* b, int n, int d, int *hot,dotprod_t dotprod);
void D_matmul_sparse_2(float* xout, float* x, void* w, float* b, int n, int d,int nHot,int *hot,float *dTemp, dotprod_t dotprod);



