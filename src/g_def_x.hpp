/**
 *  SPDX-FileCopyrightText: 2018-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief globl def & error code
 *  \author Yingshi Chen
 */

#pragma once

//  ERR code of exit
#define KOIFISH_INVALID_ARGS -10

#define KOIFISH_OUTOF_GPUMEMORY -100
#define KOIFISH_OUTOF_CPUMEMORY -101

#define KOIFISH_ZERO_PARAMETERS -200

#define KOIFISH_LOAD_TOKENIZER -300

#define KOIFISH_INVALID_GSET -600
#define KOIFISH_INVALID_NAG -601

#define KOIFISH_UNSUPPORTED_DATATYPE -1000
#define KOIFISH_GRAD_EXPLODE -1100
#define KOIFISH_BLAS_UNALIGN -1200
#define KOIFISH_DATASET_EMPTY -1300
#define KOIFISH_DATALOADER_EMPTY -1310

#define KOIFISH_EXIT_DEBUG -2000
#define KOIFISH_EXIT_SYNC_DEVICE -2100
#define KOIFISH_EXIT_OUT_CLS -2200

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define BIT_SET(val, flag) ((val) |= (flag))
#define BIT_RESET(val, flag) ((val) &= (~(flag)))
#define BIT_TEST(val, flag) (((val) & (flag)) == (flag))
#define BIT_IS(val, flag) (((val) & (flag)) != 0)

#define BYTE_bit(byt, k) (((byt) >> (7 - (k))) & 0x1)

#define BIT_STREAM_SET(char_arr, n) ((char_arr)[(n) / 8] |= (1 << (7 - ((n) % 8))))
#define BIT_STREAM_CLEAR(char_arr, n) ((char_arr)[(n) / 8] &= ~(1 << (7 - ((n) % 8))))
#define BIT_STREAM_GET(char_arr, n) (((char_arr)[(n) / 8] >> (7 - ((n) % 8))) & 1)

#define MEM_CLEAR(mem, size) memset((mem), (0x0), (size))

#define UNUSED(x) (void)(x)

#define CHILD_0909_WIKIS
#define CHILD_1218_GRAD  //

#define CHILD_1012_CACHE true

#ifndef NDEBUG
#define DEBUG_HERE                      \
    do {                                \
        volatile int __debug_break = 0; \
        (void)__debug_break;            \
    } while (0);
#define DEBUG_MARKER(msg)                           \
    do {                                            \
        std::cout << "DEBUG: " << msg << std::endl; \
    } while (0);
#else  // Eliminated in release builds
#define DEBUG_HERE ((void)0);
#define DEBUG_MARKER(msg) ((void)0);
#endif