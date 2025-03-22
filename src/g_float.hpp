/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *      
 *  \brief  Numbers:    Floating/Integer/Quants
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
#include <typeinfo>
#include <string>

#undef ENABLE_BF16
#undef ENABLE_FP32
//#define ENABLE_FP32
#define ENABLE_BF16

/*
    Type of Tokens 
    1. 32bit for DEEPSEEK
*/
//typedef uint32_t TOKEN_ID;
typedef uint16_t TOKEN_ID;

/*
    Type of numbers 
*/
enum class typNUMBER : uint8_t {
    F32     = 0,
    F16     = 1,              //  1 sign, 5 exponent, and the significand is being stored in 10 bits.
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, support has been removed
    // Q4_3 = 5, support has been removed
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,           //  1 sign, 8 exponent, and the significand is being stored in 7 bits.
    // Q4_0_4_4 = 31, support has been removed from gguf files
    // Q4_0_4_8 = 32,
    // Q4_0_8_8 = 33,
    TQ1_0   = 34,
    TQ2_0   = 35,
    // IQ4_NL_4_4 = 36,
    // IQ4_NL_4_8 = 37,
    // IQ4_NL_8_8 = 38,
    F8E5M2,     //  1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits
    F8E4M3,     //  1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits

    COUNT   = 39,
};

inline typNUMBER tpNumOf(const std::string&dtype_str){
    typNUMBER type = typNUMBER::F32;
    if (dtype_str == "F32") {
        type = typNUMBER::F32;
    } else if (dtype_str == "F16") {
        type = typNUMBER::F16;
    } else if (dtype_str == "BF16") {
        type = typNUMBER::BF16;
    } else if (dtype_str == "F8_E5M2") {
        type = typNUMBER::F8E5M2;
    } else if (dtype_str == "F8_E4M3") {
        type = typNUMBER::F8E4M3;
    } else if (dtype_str == "I32") {
        type = typNUMBER::I32;
    } else if (dtype_str == "I16") {
        type = typNUMBER::I16;
    } else if (dtype_str == "I8") {
        type = typNUMBER::I8;
    } else {
        std::string sErr = "Invalid typNumber@"+dtype_str;
        assert(0 && sErr.c_str());
    }
    return type;
}
/*
enum PrecisionMode {
    typNUMBER::F32,
    PRECISION_FP16,
    PRECISION_BF16,
    PRECISION_BF8
};

*/

/*
    FP16/BF16/FP8/FP4 from different vendors
*/
#define _USE_CUDA_FLOAT_
#if defined(_USE_CUDA_FLOAT_)
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>

  #if defined(ENABLE_FP32)
      typedef float floatX;
      #define FLOAT_TYPE typNUMBER::F32
  #elif defined(ENABLE_FP8)
      typedef __nv__fp8__e4m3 floatX;
      // typedef __nv__fp8__e5m2 floatX;
      #define FLOAT_TYPE typNUMBER::F8E4M3
  #elif defined(ENABLE_FP16)    
      typedef half floatX;
      #define FLOAT_TYPE typNUMBER::FP16
  #elif defined(ENABLE_BF16) 
      typedef __nv_bfloat16 floatX;
      #define FLOAT_TYPE typNUMBER::BF16
  #endif

  template <typename T>
  inline float T2Float(T* a0)   {
      float a;    
    
      if(typeid(T)==typeid(half)){
          a = __half2float(*(half*)a0);
      } else
      if(typeid(T)==typeid(nv_bfloat16)) {
          a = __bfloat162float(*(nv_bfloat16*)a0);
      } else
      if(typeid(T)==typeid(float)) {
          a = *a0;
      } else
      if(typeid(T)==typeid(int)) {
          a = *a0;
      }else{
          assert(0);
      }
      return a;
  }
#endif

/*
    byte per element of this type
*/
double BPE(typNUMBER type);
size_t BPBlck(typNUMBER type);
const char *cNameOf(typNUMBER type);
std::string NameOf(typNUMBER type);
bool isQuantized(typNUMBER type);

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