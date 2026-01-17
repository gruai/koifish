/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief #define for CUDA code
 *  \author Yingshi Chen
 */

#pragma once

// #define THREAD_TILE_M 16U
// #define THREAD_TILE_N 16U
#define THREAD_TILE_M 8U
#define THREAD_TILE_N 8U
// #define THREAD_TILE_M 4U
// #define THREAD_TILE_N 4U
// WarpSize is not a compile time constant, Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// Thread number of each block  - If each thread requires big private memory, then using less threads per block helps but its not infinite so should be
// soft-limited to a minimum like 32 or 64 depending on algorithm. But maximum is hard-limited to 1024 threads per block.
#define CU_T4B_SMALL 256U
#define CU_T4B_MIDDLE 512U
#define CU_T4B_BIG 1024U
//  Q_nThreadOfBlock

#define BLOCKS_PER_SM 2U  // Having 2 blocks/SM(Streaming Multiprocessor) often hits the "sweet spot" for latency hiding

#define CU_DEV_WINDOW 512U

// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// tied to enum PrecisionMode, in a future refactor make them the same
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2