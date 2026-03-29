/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief  Quants on bitstream
 *  \author Yingshi Chen
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "g_float.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Compiler-specific detection
#ifdef __SIZEOF_INT128__
#define HAS_NATIVE_INT128 1
#else
#define HAS_NATIVE_INT128 0
#endif

// Generic implementation of BIT_128 (works everywhere including CUDA device)
struct alignas(16) Packed128 {
    uint64_t low;
    uint64_t high;

#ifdef __CUDACC__
    __host__ __device__
#endif
    Packed128()
        : low(0), high(0) {
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    Packed128(uint64_t l, uint64_t h = 0)
        : low(l), high(h) {
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    Packed128(uint64_t val)
        : low(val), high(0) {
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
    Packed128(int64_t val)
        : low(static_cast<uint64_t>(val)), high(val < 0 ? 0xFFFFFFFFFFFFFFFFULL : 0) {
    }
};
using BIT_128 = Packed128;

#define UNPACK_32to4_LE(val, arr)                                      \
    {                                                                  \
        do {                                                           \
            uint32_t v = (val);                                        \
            (arr)[0]   = ((v >> 0) & 0xF0) >> 4;  /* Byte0 hi4 bit */  \
            (arr)[1]   = ((v >> 0) & 0x0F);       /* Byte0 low4 bit */ \
            (arr)[2]   = ((v >> 8) & 0xF0) >> 4;  /* Byte1 hi4 bit */  \
            (arr)[3]   = ((v >> 8) & 0x0F);       /* Byte1 low4 bit */ \
            (arr)[4]   = ((v >> 16) & 0xF0) >> 4; /* Byte2 hi4 bit */  \
            (arr)[5]   = ((v >> 16) & 0x0F);      /* Byte2 low4 bit */ \
            (arr)[6]   = ((v >> 24) & 0xF0) >> 4; /* Byte3 hi4 bit */  \
            (arr)[7]   = ((v >> 24) & 0x0F);      /* Byte3 low4 bit */ \
        } while (0);                                                   \
    }
#define UNPACK_32to4_BE(val, arr)                                            \
    {                                                                        \
        do {                                                                 \
            uint32_t v = (val);                                              \
            (arr)[0]   = ((v >> 24) & 0xF0) >> 4; /* Byte0 hi4 bit (MSB) */  \
            (arr)[1]   = ((v >> 24) & 0x0F);      /* Byte0 low4 bit */       \
            (arr)[2]   = ((v >> 16) & 0xF0) >> 4; /* Byte1 hi4 bit */        \
            (arr)[3]   = ((v >> 16) & 0x0F);      /* Byte1 low4 bit */       \
            (arr)[4]   = ((v >> 8) & 0xF0) >> 4;  /* Byte2 hi4 bit */        \
            (arr)[5]   = ((v >> 8) & 0x0F);       /* Byte2 low4 bit */       \
            (arr)[6]   = (v & 0xF0) >> 4;         /* Byte3 hi4 bit */        \
            (arr)[7]   = v & 0x0F;                /* Byte3 low4 bit (LSB) */ \
        } while (0);                                                         \
    }

#define PACK_4to32_LE(arr)                                                                                                      \
    ((((uint32_t)((arr)[0] & 0x0F) << 4 | ((arr)[1] & 0x0F))) | (((uint32_t)((arr)[2] & 0x0F) << 4 | ((arr)[3] & 0x0F)) << 8) | \
     (((uint32_t)((arr)[4] & 0x0F) << 4 | ((arr)[5] & 0x0F)) << 16) | (((uint32_t)((arr)[6] & 0x0F) << 4 | ((arr)[7] & 0x0F)) << 24))

#define PACK_4to32_BE(arr)                                                                                                             \
    ((((uint32_t)((arr)[0] & 0x0F) << 4 | ((arr)[1] & 0x0F)) << 24) | (((uint32_t)((arr)[2] & 0x0F) << 4 | ((arr)[3] & 0x0F)) << 16) | \
     (((uint32_t)((arr)[4] & 0x0F) << 4 | ((arr)[5] & 0x0F)) << 8) | ((uint32_t)((arr)[6] & 0x0F) << 4 | ((arr)[7] & 0x0F)))

#define PACK_4to128_(arr, dst)                                  \
    {                                                           \
        do {                                                    \
            uint64_t high = 0, low = 0;                         \
            /* Pack first 16 values into high 64-bit */         \
            high |= ((uint64_t)((arr)[0] & 0x0F) << 60);        \
            high |= ((uint64_t)((arr)[1] & 0x0F) << 56);        \
            high |= ((uint64_t)((arr)[2] & 0x0F) << 52);        \
            high |= ((uint64_t)((arr)[3] & 0x0F) << 48);        \
            high |= ((uint64_t)((arr)[4] & 0x0F) << 44);        \
            high |= ((uint64_t)((arr)[5] & 0x0F) << 40);        \
            high |= ((uint64_t)((arr)[6] & 0x0F) << 36);        \
            high |= ((uint64_t)((arr)[7] & 0x0F) << 32);        \
            high |= ((uint64_t)((arr)[8] & 0x0F) << 28);        \
            high |= ((uint64_t)((arr)[9] & 0x0F) << 24);        \
            high |= ((uint64_t)((arr)[10] & 0x0F) << 20);       \
            high |= ((uint64_t)((arr)[11] & 0x0F) << 16);       \
            high |= ((uint64_t)((arr)[12] & 0x0F) << 12);       \
            high |= ((uint64_t)((arr)[13] & 0x0F) << 8);        \
            high |= ((uint64_t)((arr)[14] & 0x0F) << 4);        \
            high |= ((uint64_t)((arr)[15] & 0x0F) << 0);        \
            /* Pack last 16 values into low 64-bit */           \
            low |= ((uint64_t)((arr)[16] & 0x0F) << 60);        \
            low |= ((uint64_t)((arr)[17] & 0x0F) << 56);        \
            low |= ((uint64_t)((arr)[18] & 0x0F) << 52);        \
            low |= ((uint64_t)((arr)[19] & 0x0F) << 48);        \
            low |= ((uint64_t)((arr)[20] & 0x0F) << 44);        \
            low |= ((uint64_t)((arr)[21] & 0x0F) << 40);        \
            low |= ((uint64_t)((arr)[22] & 0x0F) << 36);        \
            low |= ((uint64_t)((arr)[23] & 0x0F) << 32);        \
            low |= ((uint64_t)((arr)[24] & 0x0F) << 28);        \
            low |= ((uint64_t)((arr)[25] & 0x0F) << 24);        \
            low |= ((uint64_t)((arr)[26] & 0x0F) << 20);        \
            low |= ((uint64_t)((arr)[27] & 0x0F) << 16);        \
            low |= ((uint64_t)((arr)[28] & 0x0F) << 12);        \
            low |= ((uint64_t)((arr)[29] & 0x0F) << 8);         \
            low |= ((uint64_t)((arr)[30] & 0x0F) << 4);         \
            low |= ((uint64_t)((arr)[31] & 0x0F) << 0);         \
            /* Write to destination (assuming little-endian) */ \
            (dst)->high = high; /* Most significant 64 bits */  \
            (dst)->low  = low;  /* Least significant 64 bits */ \
        } while (0);                                            \
    }

#define UNPACK_128to4_UNSIGNED_(src, arr)    \
    {                                        \
        do {                                 \
            uint64_t high = (src)->high;     \
            uint64_t low  = (src)->low;      \
            /* Unpack from high 64-bit */    \
            (arr)[0]  = (high >> 60) & 0x0F; \
            (arr)[1]  = (high >> 56) & 0x0F; \
            (arr)[2]  = (high >> 52) & 0x0F; \
            (arr)[3]  = (high >> 48) & 0x0F; \
            (arr)[4]  = (high >> 44) & 0x0F; \
            (arr)[5]  = (high >> 40) & 0x0F; \
            (arr)[6]  = (high >> 36) & 0x0F; \
            (arr)[7]  = (high >> 32) & 0x0F; \
            (arr)[8]  = (high >> 28) & 0x0F; \
            (arr)[9]  = (high >> 24) & 0x0F; \
            (arr)[10] = (high >> 20) & 0x0F; \
            (arr)[11] = (high >> 16) & 0x0F; \
            (arr)[12] = (high >> 12) & 0x0F; \
            (arr)[13] = (high >> 8) & 0x0F;  \
            (arr)[14] = (high >> 4) & 0x0F;  \
            (arr)[15] = (high >> 0) & 0x0F;  \
            /* Unpack from low 64-bit */     \
            (arr)[16] = (low >> 60) & 0x0F;  \
            (arr)[17] = (low >> 56) & 0x0F;  \
            (arr)[18] = (low >> 52) & 0x0F;  \
            (arr)[19] = (low >> 48) & 0x0F;  \
            (arr)[20] = (low >> 44) & 0x0F;  \
            (arr)[21] = (low >> 40) & 0x0F;  \
            (arr)[22] = (low >> 36) & 0x0F;  \
            (arr)[23] = (low >> 32) & 0x0F;  \
            (arr)[24] = (low >> 28) & 0x0F;  \
            (arr)[25] = (low >> 24) & 0x0F;  \
            (arr)[26] = (low >> 20) & 0x0F;  \
            (arr)[27] = (low >> 16) & 0x0F;  \
            (arr)[28] = (low >> 12) & 0x0F;  \
            (arr)[29] = (low >> 8) & 0x0F;   \
            (arr)[30] = (low >> 4) & 0x0F;   \
            (arr)[31] = (low >> 0) & 0x0F;   \
        } while (0);                         \
    }

#define PACK_2to128_(arr, dst)                                            \
    {                                                                     \
        do {                                                              \
            uint64_t high = 0, low = 0;                                   \
            for (int i = 0; i < 32; i++) {                                \
                high |= ((uint64_t)((arr)[i] & 0x3) << (62 - 2 * i));     \
            }                                                             \
            for (int i = 0; i < 32; i++) {                                \
                low |= ((uint64_t)((arr)[i + 32] & 0x3) << (62 - 2 * i)); \
            }                                                             \
            (dst)->high = high;                                           \
            (dst)->low  = low;                                            \
        } while (0);                                                      \
    }

#define UNPACK_128to2_UNSIGNED_(src, arr)              \
    {                                                  \
        do {                                           \
            uint64_t high = (src)->high;               \
            uint64_t low  = (src)->low;                \
            int shift;                                 \
            for (int i = 0; i < 32; i++) {             \
                shift         = 62 - 2 * i;            \
                (arr)[i]      = (high >> shift) & 0x3; \
                (arr)[i + 32] = (low >> shift) & 0x3;  \
            }                                          \
        } while (0);                                   \
    }

void BIT_SET_k(hBITARR array, size_t offset, int elem, int bits);
int BIT_GET_k(const hBITARR array, size_t offset, int bits);

void BIT_SET_2(hBITARR array, size_t offset, int elem, int bits);
int BIT_GET_2(const hBITARR array, size_t offset, int bits);