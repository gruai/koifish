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
inline float gelu(float x) {
	return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
	return x / (1.0f + expf(-x));
}

float half_to_float(__gcc_fp16 x);
__gcc_fp16 float_to_half(float x);

inline __gcc_fp16 fp82half(unsigned char v) {
	union {
		unsigned short u;
		__gcc_fp16 f;
	} u;
	u.u = v << 8;
	return u.f;
}

inline float fp8e5m2_to_float(f8e5m2_t x) {
	__gcc_fp16 val = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
	memcpy(&val, &x, sizeof(f8e5m2_t));
#else
	memcpy((char*)&val + sizeof(f8e5m2_t), &x, sizeof(f8e5m2_t));
#endif
	return half_to_float(val);
}
inline float fp8_to_float(unsigned char v) {
	union {
		unsigned short u;
		__gcc_fp16 f;
	} u;
	u.u = v << 8;
	float a = u.f;
	return a;
}


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

inline float gf4_ff(uint32_t v, int k) {
	float s = fp82half(v & 0xff) ; // we expect compiler to reuse this across multiple calls
	s = s / -4.f;
	return ((int)((v >> (8 + k * 3)) & 7) - 4) * s;
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



