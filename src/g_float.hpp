/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief  Numbers:    Floating/Integer/Quants
 *  \author Yingshi Chen
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

/*
    Type of Tokens
    1. 32bit for DEEPSEEK
    2. 32bit for QWEN(151936)
*/
using TOKEN_ID = uint32_t;
// using TOKEN_ID = uint16_t;
const TOKEN_ID TOKEN_MAX = TOKEN_ID(-1);

// 8-bit bytes
using hBITARR = uint8_t*;
using BIT_8   = uint8_t;

using SHAPE = std::vector<int>;
inline size_t SHAPE2NZ(const SHAPE& shape){
    size_t nz = 1;
    for(auto n : shape){
        nz *= n;
    }
    return nz;
}

//
using floatI = float;

//  __fp16 is not part of standard C++!
// using half = __fp16;

// 1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits, just like _nv_fp8_e5m2
using f8e5 = uint8_t;

struct FLOAT_META {
    int bis = 0;
    std::string name;
    std::string alias;
    bool isQuantized() { return bis < 8; }
};
/*
    Type of numbers
*/
enum class typNUMBER : uint8_t {
    F32 = 0,
    F16,  //  1 sign, 5 exponent, 10 mantissa(significand) bits; 15361 numbers in [0.0, 1.0], endpoints included.  On average, log10(2**11) ~ 3.311 decimal
          //  digits.
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F64,
    BF16,  //  1 sign, 8 exponent, and the significand is being stored in 7 bits.

    // TQ1_0   = 34,    TQ2_0   = 35,

    F8E5M2,  //  1 sign, 5 exponent, 1 implicit and 2 explicit mantissa bits
    F8E4M3,  //  1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits

    Q4,
    Q3,
    Q2,
    BOOL1,

    T_SIGN,      //  ternary {-1, 0, 1}
    T_BINARY,    //  binary {-1,  1}
    T_BINARY_3,  //  binary {-1,  1} from three partition
    T_BINARY_TILE,

    COUNT = 39,
};

inline FLOAT_META K_FLOATS[] = {
    {32, "F32"},           // F32
    {16, "F16"},           // F16
    {8, "U8"},             // U8
    {8, "I8"},             // I8
    {16, "U16"},           // U16
    {16, "I16"},           // I16
    {32, "U32"},           // U32
    {32, "I32"},           // I32
    {64, "U64"},           // U64
    {64, "I64"},           // I64
    {64, "F64"},           // F64
    {16, "BF16"},          // BF16
    {8, "F8E5M2"},         // F8E5M2
    {8, "F8E4M3"},         // F8E4M3
    {4, "Q4"},             // Q4
    {3, "Q3"},             // Q3
    {2, "Q2"},             // Q2
    {1, "BOOL1"},          // BOOL1
    {1, "T_SIGN"},         // T_SIGN
    {1, "T_BINARY"},       // T_BINARY
    {1, "T_BINARY_3"},     // T_BINARY_3
    {0, "T_BINARY_TILE"},  // T_BINARY_TILE
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
template <>
inline typNUMBER TYPE_<half>() {
    return typNUMBER::F16;
}

#else

#endif

// more datatypes on both floatX & float
using floatGama = floatX;
// using floatGama   = half;
// using floatGama   = float;

#include "g_float_cpu.hpp"

template <typename T>
inline float T2Float(const T* a0) {
    float a;

    if (typeid(T) == typeid(half)) {
        a = __half2float(*(half*)a0);
    } else if (typeid(T) == typeid(float)) {
        a = *a0;
    } else if (typeid(T) == typeid(double)) {
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
template <>
inline float T2Float<nv_bfloat16>(const nv_bfloat16* a0) {
    /*  trick*/
    // uint16_t x   = *((uint16_t*)(a0));
    // uint32_t tmp = static_cast<uint32_t>(x) << 16;
    // float a;
    // memcpy(&a, &tmp, sizeof(a));

    float a =  __bfloat162float(*a0);
    return a;
}
template <typename T>
inline void T2Float_arr(const size_t N, const T* in, float* out) {
    for (size_t i = 0; i < N; i++) {
        out[i] = T2Float(in + i);
    }
}
template <typename T>
inline float T2Float_delta(const T* dat0, const T* dat1, const size_t pos, int flag = 0x0) {
    T delta = dat0[pos] - dat1[pos];
    return T2Float(&delta);
}

template <typename T>
inline T Float2T(const float* a0) {
    assert(!isnan(*a0) && !isinf(*a0));
    T a = T(*a0);
    return a;
}
template <>
inline bf16 Float2T<bf16>(const float* a0) {
    assert(!isnan(*a0) && !isinf(*a0));
    bf16 out = (nv_bfloat16)__float2bfloat16(*a0);
    // bf16 out = (nv_bfloat16)__float2bfloat16_rn(*a0);
    return out;
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

inline void Float2T(typNUMBER typ, void* arr, size_t offset, float a) {
    switch (typ) {
        case typNUMBER::BF16: {
            bf16 out               = Float2T<bf16>(&a);
            ((bf16*)(arr))[offset] = out;
            break;
        }
        case typNUMBER::F32:
            ((float*)(arr))[offset] = a;
            break;
        default:
            assert(0);
            break;
    }
}

//  Deprecated!!!   byte per element of this type,  (maybe decimals rather than integers!)
double BPE(typNUMBER type);
//  bit per element of this type,   (maybe decimals rather than integers!)
double BitPE(typNUMBER type);
// size_t NPBlck(typNUMBER type);
const char* cNameOf(typNUMBER type);
std::string NameOf(typNUMBER type);
// bool isQuantized(typNUMBER type);

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

//  normal-float-4 (NF4)
struct NF4_LUT {
    float table[16] = {-1.0f,
                       -0.6961928009986877f,
                       -0.5250730514526367f,
                       -0.39491748809814453f,
                       -0.28444138169288635f,
                       -0.18477343022823334f,
                       -0.09105003625154495f,
                       0.0f,
                       0.07958029955625534f,
                       0.16093020141124725f,
                       0.24611230194568634f,
                       0.33791524171829224f,
                       0.44070982933044434f,
                       0.5626170039176941f,
                       0.7229568362236023f,
                       1.0f};
    // midpoints between NF4_LUT
    float mids[16] = {-0.8480964004993438f,  -0.6106329262256622f,   -0.4599952697753906f, -0.33967943489551544f, -0.23460715460772705f,
                      -0.13791173315048218f, -0.045525018125772475f, 0.03979014977812767f, 0.1202552504837513f,   0.20352124667733002f,
                      0.2920137718319893f,   0.3893125355243683f,    0.5016634166240692f,  0.6427869200706482f,   0.8614784181118011f};
};

struct NF3_LUT {
    float table[8]     = {-1.0, -0.5350227355957031, -0.2469314038753510, 0.0, 0.1833375245332718, 0.3819939494132996, 0.6229856610298157, 1.0};
    float nf4_scale[8] = {-1.0f, -0.6961928009986877f * 0.7f, -0.5250730514526367f * 0.7f, -0.18477343022823334f * 0.7f,
                          0.0f,  0.18477343022823334f * 0.7f, 0.5250730514526367f * 0.7f,  1.0f};
};

void BIT_SET_k(hBITARR array, size_t offset, BIT_8 elem, int bits);
BIT_8 BIT_GET_k(hBITARR array, size_t offset, int bits);

#if defined(_USE_CUDA_FLOAT_)
#else

#endif
