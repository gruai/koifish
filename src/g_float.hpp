/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief  Numbers:    Floating/Integer/Quants
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <string>
#include <typeinfo>

/*
    Type of Tokens
    1. 32bit for DEEPSEEK
    2. 32bit for QWEN(151936)
*/
using TOKEN_ID = uint32_t;
// using TOKEN_ID = uint16_t;
const TOKEN_ID TOKEN_MAX = TOKEN_ID(-1);

//
using floatI = float;

// 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits, just like _nv_fp8_e5m2
using f8e5   = uint8_t;  
   

/*
    Type of numbers
*/
enum class typNUMBER : uint8_t {
    F32 = 0,
    F16 = 1,  //  1 sign, 5 exponent, 10 mantissa(significand) bits; 15361 numbers in [0.0, 1.0], endpoints included.  On average, log10(2**11) ~ 3.311 decimal
              //  digits.
    /*Q4_0    = 2,    Q4_1    = 3,    Q5_0    = 6,    Q5_1    = 7,    Q8_0    = 8,    Q8_1    = 9,    Q2_K    = 10,    Q3_K    = 11,    Q4_K    = 12,
    Q5_K    = 13,    Q6_K    = 14,    Q8_K    = 15,    IQ2_XXS = 16,    IQ2_XS  = 17,    IQ3_XXS = 18,    IQ1_S   = 19,    IQ4_NL  = 20,    IQ3_S   = 21, IQ2_S
    = 22, IQ4_XS  = 23,*/
    I8  = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    // IQ1_M   = 29,
    BF16 = 30,  //  1 sign, 8 exponent, and the significand is being stored in 7 bits.

    // TQ1_0   = 34,    TQ2_0   = 35,

    F8E5M2,  //  1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits
    F8E4M3,  //  1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits

    T_SIGN,      //  ternary {-1, 0, 1}
    T_BINARY,    //  binary {-1,  1}
    T_BINARY_3,  //  binary {-1,  1} from three partition
    T_BINARY_TILE,

    COUNT = 39,
};

struct tpBIT2 {};

template <typename T>
inline typNUMBER TYPE_() {
    // if(std::is_same<T, half>::value){   //typeid(T)==typeid(half)    ???
    //     return typNUMBER::F16;
    // } else
    // if(std::is_same<T, nv_bfloat16>::value) {
    //     return typNUMBER::BF16;
    // } else
    if (typeid(T) == typeid(float)) {
        return typNUMBER::F32;
    } else if (typeid(T) == typeid(int)) {
        return typNUMBER::I32;
    } else if (typeid(T) == typeid(f8e5)) {
        return typNUMBER::F8E5M2;
    } else if (typeid(T) == typeid(uint8_t)) {
        return typNUMBER::I8;
    } else {
        assert(0);
    }
    return typNUMBER::F16;
}

struct TYPE_DESC {
    const char* type_name;
    int64_t blck_size;
    int64_t blck_size_interleave;  // interleave elements in blocks
    size_t type_size;
    bool is_quantized;
};

inline typNUMBER tpNumOf(const std::string& dtype_str) {
    std::string sType = dtype_str;
    std::transform(sType.begin(), sType.end(), sType.begin(), ::toupper);
    typNUMBER type = typNUMBER::F32;
    if (sType == "F32") {
        type = typNUMBER::F32;
    } else if (sType == "F16") {
        type = typNUMBER::F16;
    } else if (sType == "BF16") {
        type = typNUMBER::BF16;
    } else if (sType == "F8_E5M2") {
        type = typNUMBER::F8E5M2;
    } else if (sType == "FP8") {
        type = typNUMBER::F8E5M2;
    } else if (sType == "F8_E4M3") {
        type = typNUMBER::F8E4M3;
    } else if (sType == "I32") {
        type = typNUMBER::I32;
    } else if (sType == "I16") {
        type = typNUMBER::I16;
    } else if (sType == "I8") {
        type = typNUMBER::I8;
    } else if (sType == "U8") {
        type = typNUMBER::I8;
    } else if (sType == "TERNARY") {
        type = typNUMBER::T_SIGN;
    } else if (sType == "BINARY") {
        type = typNUMBER::T_BINARY;
    } else {
        std::string sErr = "Invalid typNumber@" + sType;
        assert(0 && sErr.c_str());
    }
    return type;
}

/*
static std::map <typNUMBER, std::vector <std::string> > name2TP = {
    {typNUMBER::T_SIGN, {"ternary"}}, {typNUMBER::T_BINARY, {"binary"}}
    {typNUMBER::F32, {"float32", "fp32"}},
    {typNUMBER::F16, {"float16", "fp16", "half"}},
    {typNUMBER::BF16, {"bfloat32", "bf32"}}, {typNUMBER::I16, {"int16"}},
    {typNUMBER::I8, {"int8"}},
    {typNUMBER::F8E5M2, {"float8", "fp8", "fp8_e5m2"}},{typNUMBER::F8E4M3, {"float8", "fp8", "fp8_e4m3"}},
};*/

#undef ENABLE_BF16
#undef ENABLE_FP32
#undef ENABLE_FP16
#undef ENABLE_FP8
// #define ENABLE_FP32
// #define ENABLE_FP16
#define ENABLE_BF16
// #define ENABLE_FP8

/*
    FP16/BF16/FP8/FP4 from different vendors
    floatX - type of Activatioin/tmp
*/
#define _USE_CUDA_FLOAT_
// #undef _USE_CUDA_FLOAT_
#if defined(_USE_CUDA_FLOAT_)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#if defined(ENABLE_FP32)
typedef float floatX;
#define PARAMS_TYPE typNUMBER::F32
#define tpCuBLAS CUDA_R_32F
typedef float floatGrad;
typedef float floatFFN;
typedef float floatMV;
#elif defined(ENABLE_FP8)
typedef f8e5 floatX;
#define PARAMS_TYPE typNUMBER::F8E5M2
#define tpCuBLAS CUDA_R_8F_E5M2
#elif defined(ENABLE_FP8_1)
typedef __nv__fp8__e4m3 floatX;
#define PARAMS_TYPE typNUMBER::F8E4M3
#define tpCuBLAS CUDA_R_8F_E4M3
#elif defined(ENABLE_FP16)
typedef half floatX;
#define PARAMS_TYPE typNUMBER::F16
#define tpCuBLAS CUDA_R_16F
#define tpCuBLASCOMPUTE CUBLAS_COMPUTE_16F
//  #define tpCuBLASCOMPUTE  CUBLAS_COMPUTE_32F_FAST_16F
#elif defined(ENABLE_BF16)
#define PARAMS_TYPE typNUMBER::BF16
#define tpCuBLAS CUDA_R_16BF
#define tpCuBLASCOMPUTE CUBLAS_COMPUTE_32F
#endif

using floatX      = __nv_bfloat16;
using floatMV     = __nv_bfloat16;
using floatGrad   = __nv_bfloat16;
using floatFFN    = __nv_bfloat16;
using floatLogits = __nv_bfloat16;


using bf16   = __nv_bfloat16;
using bf16_2 = __nv_bfloat162;

using half   = __half;
using half_2 = __half2;

template <>
inline typNUMBER TYPE_<__nv_bfloat16>() {
    return typNUMBER::BF16;
}

#else

#endif


// more datatypes on both floatX & float
using floatGama   = floatX;
// using floatGama   = float;

#include "g_float_cpu.hpp"

template <typename T>
inline float T2Float(const T* a0) {
    float a;

    if (typeid(T) == typeid(half)) {
        a = __half2float(*(half*)a0);
    } else if (typeid(T) == typeid(nv_bfloat16)) {
        a = __bfloat162float(*(nv_bfloat16*)a0);
    } else if (typeid(T) == typeid(float)) {
        a = *a0;
    } else if (typeid(T) == typeid(int)) {
        a = *a0;
    } else {
        assert(0);
    }
    // assert(!isnan(a) && !isinf(a));
    return a;
}

template <typename T>
inline float T2Float(const T* a0, size_t offset) {
    return T2Float(a0 + offset);
}

template <>
inline float T2Float<tpBIT2>(const tpBIT2* a0, size_t offset) {
    assert(0x0);
    return 0.0;
}
/*
    Lite & Smart conversion from CALM
    1.	__gcc_fp16 is compiler-dependent (GCC/Clang), IEEE 754-2008 binary16 (1 sign bit, 5 exponent bits, 10 mantissa bits).
    2. __nv_fp8x4_e5m2 is NVIDIA-specific.
*/
template <>
inline float T2Float<f8e5>(const f8e5* a0) {
    union {
        unsigned short u;
        half f;  //__gcc_fp16 f;
    } u;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    u.u = (*(unsigned char*)(a0));
#else
    u.u = (*(unsigned char*)(a0)) << 8;
#endif
    float a = u.f;
    /*
        uint8_t sign = fp8 >> 7;
        uint8_t exp = (fp8 >> 2) & 0x1F;
        uint8_t mantissa = fp8 & 0x03;

        if (exp == 0) {  // Subnormal or zero
            return (sign ? -1 : 1) * (mantissa / 4.0f) * powf(2.0f, -14);
        } else {         // Normalized
            return (sign ? -1 : 1) * (1.0f + mantissa / 4.0f) * powf(2.0f, exp - 15);
        }*/
    assert(!isnan(a) && !isinf(a));
    return a;
}
template <>
inline float T2Float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* a0) {
    float a = T2Float<f8e5>((const f8e5*)a0);
    return a;
}
template <typename T>
inline void T2Float_arr(const size_t N, const T* in, float* out) {
    for (size_t i = 0; i < N; i++) {
        out[i] = T2Float(in + i);
    }
}

template <typename T>
inline T Float2T(const float* a0) {
    assert(!isnan(*a0) && !isinf(*a0));
    T a = T(*a0);
    /*if(typeid(T)==typeid(half)){
        a = (half)__float2half(*a0);
    } else
    if(typeid(T)==typeid(nv_bfloat16)) {
        a = (nv_bfloat16)__float2bfloat16(*a0);
    }*/
    return a;
}

template <>
inline f8e5 Float2T<f8e5>(const float* a0) {
    assert(!isnan(*a0) && !isinf(*a0));
    __gcc_fp16 val = float_to_half(*a0);
    f8e5 out       = f8e5(0);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(&out, (char*)&val, sizeof(f8e5));  // TODO: round instead of truncate?
#else
    memcpy(&out, (char*)&val + sizeof(f8e5), sizeof(f8e5));  // TODO: round instead of truncate?
#endif
    return out;
}
template <>
inline __nv_fp8_e5m2 Float2T<__nv_fp8_e5m2>(const float* a0) {
    f8e5 a = Float2T<f8e5>(a0);
    return (__nv_fp8_e5m2)(a);
}

//  Deprecated!!!   byte per element of this type,  (maybe decimals rather than integers!)
double BPE(typNUMBER type);
//  bit per element of this type,   (maybe decimals rather than integers!)
double BitPE(typNUMBER type);
// size_t NPBlck(typNUMBER type);
const char* cNameOf(typNUMBER type);
std::string NameOf(typNUMBER type);
bool isQuantized(typNUMBER type);

struct BF16_LUT {
    float table[65536];
};
struct F16_LUT {
    float table[65536];
};

struct FP8E4M3_LUT {
    float table[256] = {
        0.0,          0.001953125,  0.00390625,  0.005859375,  0.0078125,   0.009765625,  0.01171875,  0.013671875,  0.015625,     0.017578125,  0.01953125,
        0.021484375,  0.0234375,    0.025390625, 0.02734375,   0.029296875, 0.03125,      0.03515625,  0.0390625,    0.04296875,   0.046875,     0.05078125,
        0.0546875,    0.05859375,   0.0625,      0.0703125,    0.078125,    0.0859375,    0.09375,     0.1015625,    0.109375,     0.1171875,    0.125,
        0.140625,     0.15625,      0.171875,    0.1875,       0.203125,    0.21875,      0.234375,    0.25,         0.28125,      0.3125,       0.34375,
        0.375,        0.40625,      0.4375,      0.46875,      0.5,         0.5625,       0.625,       0.6875,       0.75,         0.8125,       0.875,
        0.9375,       1.0,          1.125,       1.25,         1.375,       1.5,          1.625,       1.75,         1.875,        2.0,          2.25,
        2.5,          2.75,         3.0,         3.25,         3.5,         3.75,         4.0,         4.5,          5.0,          5.5,          6.0,
        6.5,          7.0,          7.5,         8.0,          9.0,         10.0,         11.0,        12.0,         13.0,         14.0,         15.0,
        16.0,         18.0,         20.0,        22.0,         24.0,        26.0,         28.0,        30.0,         32.0,         36.0,         40.0,
        44.0,         48.0,         52.0,        56.0,         60.0,        64.0,         72.0,        80.0,         88.0,         96.0,         104.0,
        112.0,        120.0,        128.0,       144.0,        160.0,       176.0,        192.0,       208.0,        224.0,        240.0,        256.0,
        288.0,        320.0,        352.0,       384.0,        416.0,       448.0,        480,         -0.0,         -0.001953125, -0.00390625,  -0.005859375,
        -0.0078125,   -0.009765625, -0.01171875, -0.013671875, -0.015625,   -0.017578125, -0.01953125, -0.021484375, -0.0234375,   -0.025390625, -0.02734375,
        -0.029296875, -0.03125,     -0.03515625, -0.0390625,   -0.04296875, -0.046875,    -0.05078125, -0.0546875,   -0.05859375,  -0.0625,      -0.0703125,
        -0.078125,    -0.0859375,   -0.09375,    -0.1015625,   -0.109375,   -0.1171875,   -0.125,      -0.140625,    -0.15625,     -0.171875,    -0.1875,
        -0.203125,    -0.21875,     -0.234375,   -0.25,        -0.28125,    -0.3125,      -0.34375,    -0.375,       -0.40625,     -0.4375,      -0.46875,
        -0.5,         -0.5625,      -0.625,      -0.6875,      -0.75,       -0.8125,      -0.875,      -0.9375,      -1.0,         -1.125,       -1.25,
        -1.375,       -1.5,         -1.625,      -1.75,        -1.875,      -2.0,         -2.25,       -2.5,         -2.75,        -3.0,         -3.25,
        -3.5,         -3.75,        -4.0,        -4.5,         -5.0,        -5.5,         -6.0,        -6.5,         -7.0,         -7.5,         -8.0,
        -9.0,         -10.0,        -11.0,       -12.0,        -13.0,       -14.0,        -15.0,       -16.0,        -18.0,        -20.0,        -22.0,
        -24.0,        -26.0,        -28.0,       -30.0,        -32.0,       -36.0,        -40.0,       -44.0,        -48.0,        -52.0,        -56.0,
        -60.0,        -64.0,        -72.0,       -80.0,        -88.0,       -96.0,        -104.0,      -112.0,       -120.0,       -128.0,       -144.0,
        -160.0,       -176.0,       -192.0,      -208.0,       -224.0,      -240.0,       -256.0,      -288.0,       -320.0,       -352.0,       -384.0,
        -416.0,       -448.0,       -480};
};
struct FP8E5M2_LUT {
    float table[256] = {
        0.0,          0.001953125,  0.00390625,  0.005859375,  0.0078125,   0.009765625,  0.01171875,  0.013671875,  0.015625,     0.017578125,  0.01953125,
        0.021484375,  0.0234375,    0.025390625, 0.02734375,   0.029296875, 0.03125,      0.03515625,  0.0390625,    0.04296875,   0.046875,     0.05078125,
        0.0546875,    0.05859375,   0.0625,      0.0703125,    0.078125,    0.0859375,    0.09375,     0.1015625,    0.109375,     0.1171875,    0.125,
        0.140625,     0.15625,      0.171875,    0.1875,       0.203125,    0.21875,      0.234375,    0.25,         0.28125,      0.3125,       0.34375,
        0.375,        0.40625,      0.4375,      0.46875,      0.5,         0.5625,       0.625,       0.6875,       0.75,         0.8125,       0.875,
        0.9375,       1.0,          1.125,       1.25,         1.375,       1.5,          1.625,       1.75,         1.875,        2.0,          2.25,
        2.5,          2.75,         3.0,         3.25,         3.5,         3.75,         4.0,         4.5,          5.0,          5.5,          6.0,
        6.5,          7.0,          7.5,         8.0,          9.0,         10.0,         11.0,        12.0,         13.0,         14.0,         15.0,
        16.0,         18.0,         20.0,        22.0,         24.0,        26.0,         28.0,        30.0,         32.0,         36.0,         40.0,
        44.0,         48.0,         52.0,        56.0,         60.0,        64.0,         72.0,        80.0,         88.0,         96.0,         104.0,
        112.0,        120.0,        128.0,       144.0,        160.0,       176.0,        192.0,       208.0,        224.0,        240.0,        256.0,
        288.0,        320.0,        352.0,       384.0,        416.0,       448.0,        480,         -0.0,         -0.001953125, -0.00390625,  -0.005859375,
        -0.0078125,   -0.009765625, -0.01171875, -0.013671875, -0.015625,   -0.017578125, -0.01953125, -0.021484375, -0.0234375,   -0.025390625, -0.02734375,
        -0.029296875, -0.03125,     -0.03515625, -0.0390625,   -0.04296875, -0.046875,    -0.05078125, -0.0546875,   -0.05859375,  -0.0625,      -0.0703125,
        -0.078125,    -0.0859375,   -0.09375,    -0.1015625,   -0.109375,   -0.1171875,   -0.125,      -0.140625,    -0.15625,     -0.171875,    -0.1875,
        -0.203125,    -0.21875,     -0.234375,   -0.25,        -0.28125,    -0.3125,      -0.34375,    -0.375,       -0.40625,     -0.4375,      -0.46875,
        -0.5,         -0.5625,      -0.625,      -0.6875,      -0.75,       -0.8125,      -0.875,      -0.9375,      -1.0,         -1.125,       -1.25,
        -1.375,       -1.5,         -1.625,      -1.75,        -1.875,      -2.0,         -2.25,       -2.5,         -2.75,        -3.0,         -3.25,
        -3.5,         -3.75,        -4.0,        -4.5,         -5.0,        -5.5,         -6.0,        -6.5,         -7.0,         -7.5,         -8.0,
        -9.0,         -10.0,        -11.0,       -12.0,        -13.0,       -14.0,        -15.0,       -16.0,        -18.0,        -20.0,        -22.0,
        -24.0,        -26.0,        -28.0,       -30.0,        -32.0,       -36.0,        -40.0,       -44.0,        -48.0,        -52.0,        -56.0,
        -60.0,        -64.0,        -72.0,       -80.0,        -88.0,       -96.0,        -104.0,      -112.0,       -120.0,       -128.0,       -144.0,
        -160.0,       -176.0,       -192.0,      -208.0,       -224.0,      -240.0,       -256.0,      -288.0,       -320.0,       -352.0,       -384.0,
        -416.0,       -448.0,       -480};
};

void matmul_unscaled(float* xout, float* x, float* w, int n, int d);
void matmul_unscaled(float* xout, float* x, __gcc_fp16* w, int n, int d);
void matmul_unscaled(float* xout, float* x, f8e5* w, int n, int d);

void S_matmul(float* xout, float* x, float* w, int n, int d, const int* block_size, float* scale);
void S_matmul(float* xout, float* x, __gcc_fp16* w, int n, int d, const int* block_size, float* scale);
void S_matmul(float* xout, float* x, f8e5* w, int n, int d, const int* block_size, float* scale);

float rmsnorm(float* o, float* x, float* weight, int size, float eps, bool ln);
float rmsnorm(float* o, float* x, float* weight, int size, float eps);

void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim);
void rope(float* buf, float* vec, int d, int head_dim, int pos, float theta);
void rope(float* buf, __gcc_fp16* vec, int d, int head_dim, int pos, float theta);
#if defined(_USE_CUDA_FLOAT_)
#else

#endif
