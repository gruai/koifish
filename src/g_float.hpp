/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  Acknowledgement: https://github.com/andrewkchan/deepseek.cpp
 * 
 *  \brief DeepSeek
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
#include <cfloat>
#include <math.h>
#include <float.h>
#include <cstdint>
#include <stdint.h>
#include <string.h>

/*
    Type of floating-point numbers 
*/
enum class tpFloatingPoint : uint8_t {
    FP32, 
    FP16, 
    BF16,       //  1 sign, 8 exponent, and the significand is being stored in 7 bits.
    F8E5M2,     //  1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits
    F8E4M3,     //  1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits
  
    I32,  I16,  I8,  U8,
};
typedef tpFloatingPoint DType;

typedef uint16_t f16_t;
typedef uint8_t f8e5m2_t;

#if defined(__AVX2__) && defined(__F16C__)
    inline float half_to_float(f16_t x) {
    return _cvtsh_ss(x);
    }
    inline f16_t float_to_half(float x) {
    return _cvtss_sh(x, 0);
    }
#else
    inline float half_to_float(f16_t x) {
        assert(false && "float16 not supported on this platform");
        return 0.0f;
    }
    inline f16_t float_to_half(float x) {
        assert(false && "float16 not supported on this platform");
        return 0;
    }
#endif

inline float float8e5m2_to_float(f8e5m2_t x) {
  f16_t val = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&val, &x, sizeof(f8e5m2_t));
#else
  memcpy((char*)&val + sizeof(f8e5m2_t), &x, sizeof(f8e5m2_t));
#endif
  return half_to_float(val);
}
[[maybe_unused]] inline f8e5m2_t float_to_float8e5m2(float x) {
  f16_t val = float_to_half(x);
  f8e5m2_t out;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&out, (char*)&val, sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#else
  memcpy(&out, (char*)&val + sizeof(f8e5m2_t), sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#endif
  return out;
}

void matmul_unscaled(float* xout, float* x, float* w, int n, int d);
void matmul_unscaled(float* xout, float* x, f16_t* w, int n, int d);
void matmul_unscaled(float* xout, float* x, f8e5m2_t* w, int n, int d);