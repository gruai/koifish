/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Bit version of C=AxB+D(A or B is bit tensor)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <mma.h>
using namespace nvcuda;
#include <iostream>

#include "../cuda_common.h"
#include "_bit_utils.cuh"
#include "operator.cuh"
#include "utils.cuh"

//  So slow, so strange!
template <typename Tb, typename Tc, const int BM, const int BN, const int BK, const int TM = 8, const int TN = 8>
__global__ void ABC_bit_v0(char* A, float* gama_T, Tb* B, Tc* C, const Tc* bias, int M, int N, int K, int transA = 0, int transB = 0, int flag = 0x0) {
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;
    const int NPB = 8;  // number of element per byte
    fnPOS pA = transA == 0 ? fnCR2POS : fnRC2POS, pB = transB == 0 ? fnCR2POS : fnRC2POS, pC = fnCR2POS;
    A = A + pA(by * BM, 0, M, K) / NPB, B = B + pB(0, bx * BN, K, N), C = C + pC(by * BM, bx * BN, M, N);

    int block_row_thread = BN / TN, block_col_thread = BM / TM;
    assert(thread_num * TM * TN == BN * BM);                         // each thread <<[TM][TN]>>
    assert(sizeof(Tb) * TM % 16 == 0 && sizeof(Tb) * TN % 16 == 0);  //  ALIGN(16) + 128-bit loads/stores
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;
    __shared__ char As[BM * BK / NPB];  // 128*16/8
    __shared__ Tb Bs[BK * BN];
    int nG2A = (BM * BK / thread_num), nG2B = (BK * BN / thread_num), curA = threadIdx.x * nG2A, curB = threadIdx.x * nG2B;
    int stepA = pA(0, BK, M, K) / NPB, stepB = pB(BK, 0, K, N), r, c;
    char a_frag, bit;
    float tmp[TM][TN] = {0.};
    float b_frag[TN]  = {0.};  // share memory->register

#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA, B += stepB) {
#pragma unroll
        for (int i = curA; i < curA + nG2A; i++) {  //[BM:Bk]
            r = i / BK, c = i % BK;                 // r = i % BM, c = i / BM;
            As[i / NPB] = A[pA(r, c, M, K) / NPB];  // CR2POS(r, c, BM, BK)
        }
#pragma unroll
        for (int i = curB; i < curB + nG2B; i++) {
            r = i / BN, c = i % BN;
            Bs[i] = B[pB(r, c, K, N)];
        }
        __syncthreads();

        UNROLL for (int j = 0; j < TM; j++) {
            a_frag = As[RC2POS((ty + j), 0, BM, BK) / NPB];
            UNROLL for (int l = 0; l < TN; l++) {
                for (int i = 0; i < BK; i++) {
                    b_frag[i] = Bs[RC2POS(i, tx + l, BK, BN)];
                }
                tmp[j][l] += ByteDot(a_frag, b_frag);
                // UNROLL  for (int i = 0; i < BK; i++)
                //         tmp[j][l] += As[RC2POS((ty + j),i,BM,BK)] * Bs[RC2POS(i,tx+l,BK,BN)];
                // }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        float bia = bias != nullptr ? (float)(bias[ty + j]) : 0;
#pragma unroll
        for (int l = 0; l < TN; l++) C[pC(ty + j, tx + l, M, N)] = tmp[j][l] + bia;  //[(ty + j) * N + tx + l]
    }
}

// ~3.3s
template <const int BM, const int BN, const int BK, const int TM, const int TN, typename Ta, typename Tb, typename Tc>
__global__ void ABC_v2(Ta* A, Tb* B, Tc* C, int M, int N, int K, int flag = 0x0) {
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    assert(thread_num * TM == BM * BN);  // Each thread store TM resutls with extra one for Bs[tx + i * BN]
    float tmp[TM + 1] = {0.};
    // sliding block
    A              = A + CR2POS(by * BM, 0, M, K);        // by * BM * K
    B              = B + CR2POS(0, bx * BN, K, N);        // bx * BN
    C              = C + CR2POS(by * BM, bx * BN, M, N);  // by * BM * N + bx * BN
    int a_tile_row = threadIdx.x / BK, a_tile_col = threadIdx.x % BK, a_tile_stride = thread_num / BK;
    int b_tile_row = threadIdx.x / BN, b_tile_col = threadIdx.x % BN, b_tile_stride = thread_num / BN;
    int stepA = CR2POS(0, BK, M, K), stepB = CR2POS(BK, 0, K, N);
#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA, B += stepB) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = AT_(A, a_tile_row + i, a_tile_col, M, K);  // A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = AT_(B, b_tile_row + i, b_tile_col, K, N);  // B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
// A += BK;
// B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
            tmp[TM] = Bs[tx + i * BN];  //
#pragma unroll
            for (int j = 0; j < TM; j++) {
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        C[CR2POS(ty + j, tx, M, N)] = tmp[j];  //(ty + j) * N + tx
    }
}

/*  ~2.45s

*/
template <const int BM, const int BN, const int BK, const int TM, const int TN, typename Ta, typename Tb, typename Tc>
__global__ void ABC_v4(Ta* A, Tb* B, Tc* C, const Tc* bias, int M, int N, int K, int transA = 0, int transB = 0, int flag = 0x0) {
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;
    int block_row_thread = BN / TN, block_col_thread = BM / TM;
    assert(thread_num == block_row_thread * block_col_thread);
    int tx = (threadIdx.x % block_row_thread) * TN;  // Each thread for [ty:ty+tM,tx:tx+TN]
    int ty = (threadIdx.x / block_row_thread) * TM;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    fnPOS pA = transA == 0 ? fnCR2POS : fnRC2POS, pB = transB == 0 ? fnCR2POS : fnRC2POS, pC = fnCR2POS;
    A = A + pA(by * BM, 0, M, K), B = B + pB(0, bx * BN, K, N), C = C + pC(by * BM, bx * BN, M, N);
    int a_tile_row = threadIdx.x / BK, a_tile_col = threadIdx.x % BK, a_tile_stride = thread_num / BK;  // 32
    int b_tile_row = threadIdx.x / BN, b_tile_col = threadIdx.x % BN, b_tile_stride = thread_num / BN;  // 2
    int stepA = pA(0, BK, M, K), stepB = pB(BK, 0, K, N);
    float tmp[TM][TN] = {0.};
    float a_frag[TM] = {0.}, b_frag[TN] = {0.};  // share memory->register
#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA, B += stepB) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[pA(a_tile_row + i, a_tile_col, M, K)];  // A[(a_tile_row + i) * K + a_tile_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[pB(b_tile_row + i, b_tile_col, K, N)];  // B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();
        // A += BK;        B += BK * N;
        // #pragma unroll
        //         for (int i = 0; i < BK; i++) {
        //             UNROLL  for (int j = 0; j < TM; j++) {  //  [TM,1]
        //                 a_frag[j] = As[RC2POS((ty + j),i,BM,BK)];       //  (ty + j) * BK + i
        //             }
        //             UNROLL  for (int l = 0; l < TN; l++) {  //  [1,TN]
        //                 b_frag[l] = Bs[RC2POS(i,tx+l,BK,BN)];       //  tx + l + i * BN
        //             }
        //             UNROLL  for (int j = 0; j < TM; j++) {  //  [TM,1]x[1,TN] => [TM,TN]
        //             UNROLL  for (int l = 0; l < TN; l++) tmp[j][l] += a_frag[j] * b_frag[l];
        //             }
        //         }

        UNROLL for (int j = 0; j < TM; j++) {
            UNROLL for (int l = 0; l < TN; l++) {
                UNROLL for (int i = 0; i < BK; i++) tmp[j][l] += As[RC2POS((ty + j), i, BM, BK)] * Bs[RC2POS(i, tx + l, BK, BN)];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TM; j++) {
        float bia = bias != nullptr ? (float)(bias[ty + j]) : 0;
#pragma unroll
        for (int l = 0; l < TN; l++) C[pC(ty + j, tx + l, M, N)] = tmp[j][l] + bia;  //[(ty + j) * N + tx + l]
    }
}

/*  2.6~2.7s
    1. Each thread for mma of [ty:ty+tM,tx:tx+TN]
    2. Each block load As & Bs
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN, typename Ta, typename Tb, typename Tc>
__global__ void ABC_v5(Ta* A, Tb* B, Tc* C, int M, int N, int K, int flag = 0x0) {
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;
    A = A + CR2POS(by * BM, 0, M, K), B = B + CR2POS(0, bx * BN, K, N), C = C + CR2POS(by * BM, bx * BN, M, N);

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    assert(thread_num * TM * TN == BN * BM);                         // each thread <<[TM][TN]>>
    assert(sizeof(Ta) * TM % 16 == 0 && sizeof(Ta) * TN % 16 == 0);  //  ALIGN(16) + 128-bit loads/stores
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;
    __shared__ Ta As[BM * BK];
    __shared__ Tb Bs[BK * BN];
    int nG2A = (BM * BK / thread_num), nG2B = (BK * BN / thread_num), curA = threadIdx.x * nG2A, curB = threadIdx.x * nG2B;
    int stepA = CR2POS(0, BK, M, K), stepB = CR2POS(BK, 0, K, N), r, c;
    float tmp[TM][TN] = {0.};
    float a_frag[TM] = {0.}, b_frag[TN] = {0.};  // share memory->register
    floatX *tileA = ((floatX*)As) + CR2POS(ty, 0, BM, BK), *tileB = ((floatX*)Bs) + RC2POS(0, tx, BK, BN);
#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA, B += stepB) {
#pragma unroll
        for (int i = curA; i < curA + nG2A; i++) {  //[BM:Bk]
            // r = (i) / BK, c = (i) % BK;
            // As[i] = A[CR2POS(r,c,M,K)];
            r = i % BM, c = i / BM;
            As[i] = A[CR2POS(r, c, M, K)];  // CR2POS(r, c, BM, BK)
        }
#pragma unroll
        for (int i = curB; i < curB + nG2B; i++) {
            r = (i) / BN, c = (i) % BN;
            Bs[i] = B[CR2POS(r, c, K, N)];
        }
        __syncthreads();

        SYNC_AtBC_m8n8k16(a_frag, b_frag, tmp);
        __syncthreads();
    }
    x128 packed_out;  // little faster
#pragma unroll
    for (int l = 0; l < TN; l++) {
#pragma unroll
        for (int j = 0; j < x128::size; j++) {
            packed_out[j] = tmp[j][l];
        }
        store128(C + CR2POS(ty, tx + l, M, N), packed_out);
    }
    // #pragma unroll
    //     for (int j = 0; j < TM; j++) {
    //         #pragma unroll
    //         for (int l = 0; l < TN; l++) C[CR2POS(ty + j, tx + l, M, N)] = tmp[j][l];  //[(ty + j) * N + tx + l]
    //     }
}

/*  ~2.5
    1. Each thread for mma of [ty:ty+tM,tx:tx+TN]
    2. Each block load As & Bs
*/
template <const int BM, const int BN, const int BK, const int TM, const int TN, typename Ta>
__global__ void ABC_v6(Ta* A, floatX* B, floatX* C, int M, int N, int K, int flag = 0x0) {
    int bx = blockIdx.x, by = blockIdx.y, thread_num = blockDim.x;
    A = A + CR2POS(by * BM, 0, M, K), B = B + CR2POS(0, bx * BN, K, N), C = C + CR2POS(by * BM, bx * BN, M, N);
    const int ld128      = 16 / sizeof(Ta);
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    assert(thread_num * TM * TN == BN * BM);                         // each thread <<[TM][TN]>>
    assert(sizeof(Ta) * TM % 16 == 0 && sizeof(Ta) * TN % 16 == 0);  //  ALIGN(16) + 128-bit loads/stores
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;
    __shared__ x128 As[BM * BK / ld128];
    __shared__ x128 Bs[BK * BN / ld128];
    //  n element of Global memory to share memory
    int nG2A = (BM * BK / thread_num), nG2B = (BK * BN / thread_num), curA = threadIdx.x * nG2A, curB = threadIdx.x * nG2B;
    assert(nG2A % ld128 == 0 && nG2B % ld128 == 0);
    int stepA = CR2POS(0, BK, M, K), stepB = CR2POS(BK, 0, K, N), r, c;
    float tmp[TM][TN] = {0.};
    float a_frag[TM] = {0.}, b_frag[TN] = {0.};  // share memory->register
    // floatX *tileA = (floatX *)As, *tileB = (floatX *)Bs;
    floatX *tileA = ((floatX*)As) + CR2POS(ty, 0, BM, BK), *tileB = ((floatX*)Bs) + CR2POS(0, tx, BK, BN);

#pragma unroll
    for (int k = 0; k < K; k += BK, A += stepA, B += stepB) {
#pragma unroll
        for (int i = curA; i < curA + nG2A; i += ld128) {  //[BM:Bk]
            r = i % BM, c = i / BM;
            // tileA[i] = *(A+CR2POS(r, c, M, K));
            As[i / ld128] = load128(A + CR2POS(r, c, M, K));
        }
#pragma unroll
        for (int i = curB; i < curB + nG2B; i += ld128) {
            // r = (i) / BN, c = (i) % BN;
            //  tileB[i] = B[CR2POS(r, c, K, N)];
            r = i % BK, c = i / BK;
            Bs[i / ld128] = load128(B + CR2POS(r, c, K, N));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int j = 0; j < TM; j++) {                // i-th column:   r=ty+j,c=i
                a_frag[j] = tileA[CR2POS(j, i, BM, BK)];  // (ty + j) * BK + i
            }
#pragma unroll
            for (int l = 0; l < TN; l++) {                // i-th row:   r=i,c=tx + l
                b_frag[l] = tileB[CR2POS(i, l, BK, BN)];  // RC2POS(i, tx + l, BK, BN)
            }
#pragma unroll
            for (int j = 0; j < TM; j++) {
#pragma unroll
                for (int l = 0; l < TN; l++) tmp[j][l] += a_frag[j] * b_frag[l];
            }
        }
        __syncthreads();
    }
    x128 packed_out;
#pragma unroll
    for (int l = 0; l < TN; l++) {
#pragma unroll
        for (int j = 0; j < x128::size; j++) {
            packed_out[j] = tmp[j][l];
        }
        store128(C + CR2POS(ty, tx + l, M, N), packed_out);
        // #pragma unroll
        //         for (int j = 0; j < TM; j++) C[CR2POS(ty + j, tx + l, M, N)] = tmp[j][l];  //[(ty + j) * N + tx + l]
    }
}
/*
    c(m,n) = op(a)*op(b) + bias
*/
void CU_abc(floatX* d, hGTensor gensor, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream, int transA, int transB, float beta,
            floatX* pre_gelu, bool backward) {
    NVTX_RANGE_FN();
    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if (((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        _INFO("All CU_abc_ pointers must be aligned!\n");
        exit(KOIFISH_BLAS_UNALIGN);
    }
    static size_t smem_max_size = 40960;
    // std::max((BLOCK_ROWS + BLOCK_COLS) * AB_SMEM_STRIDE * sizeof(floatX), BLOCK_ROWS * C_SMEM_STRIDE * sizeof(floatX));
    // smem_max_size = TM * TM * 2 * sizeof(float);
    assert(deviceProp.sharedMemPerMultiprocessor >= smem_max_size);

    /*switch (gensor->type) {
        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3: {
            const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, nT = BM * BN / TM / TN;  // 256
            dim3 dBlock(nT), dGrid(CEIL_DIV(n, BM), CEIL_DIV(m, BN));
            assert(dBlock.x > BN && dBlock.x >= BK);
            assert(gensor->ne[0] == k);  // length of gama_T is always ne[0]
            ABC_bit_v0<floatX, floatX, BM, BN, BK, TM, TN>
                <<<dGrid, dBlock, smem_max_size, stream>>>((char *)gensor->data,gensor->gama_T, (floatX *)b, d, bias, m, n, k, transA, transB);
            return;
        } break;
        default:

            break;
    }*/

    // assert(batch_count==0);
    bool has_bias = (bias != nullptr), has_gelu = (pre_gelu != nullptr);
    const float alpha = 1.0f;  //, beta = accumulate ? 1.0f : 0.0f;
    assert(pre_gelu == nullptr);
    floatX* a = gensor->GetDataX();
    // [50304,768] x [768,8192] => [50304,8192]         or(transA) [768,50304]' x [768,8192] => [50304,8192]
    int lda = transA ? k : m, ldb = transB ? n : k;

    switch (DEBUG.T_GEMM) {  //  Back of delta: [768,50304] x [50304,8192] => [768,8192]
        case 2: {
            const int BM = 64, BN = 64, BK = 8, TM = 8, TN = 8;
            dim3 dBlock(512), dGrid(CEIL_DIV(n, BM), CEIL_DIV(m, BN));
            assert(k % BK == 0 && dBlock.x == BM * BK && dBlock.x == BK * BN);
            ABC_v2<BM, BN, BK, TM, TN, floatX, floatX, floatX><<<dGrid, dBlock, smem_max_size, stream>>>((floatX*)a, (floatX*)b, d, m, n, k);
            break;
        }
        case 4: {
            const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 dBlock(256), dGrid(CEIL_DIV(n, BM), CEIL_DIV(m, BN));
            ABC_v4<BM, BN, BK, TM, TN, floatX, floatX, floatX>
                <<<dGrid, dBlock, smem_max_size, stream>>>((floatX*)a, (floatX*)b, d, bias, m, n, k, transA, transB);
            break;
        }
        case 5: {
            const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, nT = BM * BN / TM / TN;  // 256
            dim3 dBlock(nT), dGrid(CEIL_DIV(n, BM), CEIL_DIV(m, BN));
            assert(dBlock.x > BN && dBlock.x >= BK);
            ABC_v5<BM, BN, BK, TM, TN, floatX, floatX, floatX><<<dGrid, dBlock, smem_max_size, stream>>>((floatX*)a, (floatX*)b, d, m, n, k);
            break;
        }
        case 6: {
            const int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8, nT = BM * BN / TM / TN;  // 256
            dim3 dBlock(nT), dGrid(CEIL_DIV(n, BM), CEIL_DIV(m, BN));
            assert(sizeof(floatX) * BK % 16 == 0 && TN == x128::size);  // x128
            ABC_v6<BM, BN, BK, TM, TN, floatX><<<dGrid, dBlock, smem_max_size, stream>>>((floatX*)a, (floatX*)b, d, m, n, k);
            break;
        }
        default:
            assert(0);
            break;
    }

    return;
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

//  Deprecated sample code from BitNet
template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1* _i2s, T2* _i8s, const int N = 16) {
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint* i8s = reinterpret_cast<uint*>(_i8s);

    // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
    uint const i2s = *_i2s;

    static constexpr uint immLut               = (0xf0 & 0xcc) | 0xaa;  // 0b11101010
    static constexpr uint BOTTOM_MASK          = 0x03030303;            // 0xf -> 0b11 select 0,3
    static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000;

#pragma unroll
    for (int i = 0; i < (N / 4); i++) {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(i8s[i]) : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsubss4(i8s[i], 0x02020202);
    }
}

template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform,
                                                               __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
    constexpr int K_per_loop = 16;
    constexpr int wmma_K     = 32;
    constexpr int wmma_N     = 16;
    int in_thread_C_local[1], bx = blockIdx.x, by = blockIdx.y;
    signed char A_local[K_per_loop];
    int B_reshape_local[1];
    signed char B_decode_local[K_per_loop];
    int red_buf0[1];
    in_thread_C_local[0] = 0;
#pragma unroll
    for (int k_0 = 0; k_0 < K / (K_per_loop * K_block_size); ++k_0) {
        *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
        B_reshape_local[0]    = *(int*)(B + (((int)bx) * N_block_size * K / 4) + (k_0 * K_block_size * K_per_loop * wmma_N / 4) +
                                     ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) + ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) +
                                     ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) + ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4));
        decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
#pragma unroll
        for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
            in_thread_C_local[0] = __dp4a(*(int*)&A_local[((k_2_0 * 4))], *(int*)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
        }
    }
    red_buf0[0] = in_thread_C_local[0];
#pragma unroll
    for (int offset = K_block_size / 2; offset > 0; offset /= 2) {
        red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
    }
    int out_idx = ((((int)bx) * N_block_size) + ((int)threadIdx.y));
    int ws_idx  = out_idx / (N / ws_num);
    if (threadIdx.x == 0)
        dtype_transform[out_idx] = (__nv_bfloat16)(((float)red_buf0[0]) / (float)s[0] * (float)ws[ws_idx]);
}

inline void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K,
                                cudaStream_t stream) {
    if (M == 1 && N == 3840 && K == 2560) {
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 2560 && K == 2560) {
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 13824 && K == 2560) {
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16><<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 2560 && K == 6912) {
        ladder_int8xint2_kernel<1, 2560, 6912, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 4800 && K == 3200) {
        ladder_int8xint2_kernel<1, 4800, 3200, 6, 8, 16><<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 3200 && K == 3200) {
        ladder_int8xint2_kernel<1, 3200, 3200, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 20480 && K == 3200) {
        ladder_int8xint2_kernel<1, 20480, 3200, 2, 8, 16><<<dim3(1280, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 3200 && K == 10240) {
        ladder_int8xint2_kernel<1, 3200, 10240, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 5120 && K == 27648) {
        ladder_int8xint2_kernel<1, 5120, 27648, 1, 8, 16><<<dim3(320, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else if (M == 1 && N == 55296 && K == 5120) {
        ladder_int8xint2_kernel<1, 55296, 5120, 1, 8, 16><<<dim3(3456, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    } else {
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}

/**
 * @brief Perform the HMMA.16816 operation.
 *
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0 The first half of the first input bf16_2 matrix.
 * @param[in] a1 The second half of the first input bf16_2 matrix.
 * @param[in] a2 The first half of the second input bf16_2 matrix.
 * @param[in] a3 The second half of the second input bf16_2 matrix.
 * @param[in] b0 The first half of the bf16_2 matrix B.
 * @param[in] b1 The second half of the bf16_2 matrix B.
 * @param[in] c0 The first half of the float2 accumulator matrix C.
 * @param[in] c1 The second half of the float2 accumulator matrix C.
 */
__device__ static inline void hmma16816(float2& d0, float2& d1, const bf16_2& a0, const bf16_2& a1, const bf16_2& a2, const bf16_2& a3, const bf16_2& b0,
                                        const bf16_2& b1, const float2& c0, const float2& c1) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"

        // D matrix
        : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)

        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)), "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

          // B matrix
          "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

          // C matrix
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
/**
 * @brief Perform the HMMA.16816 operation.
 *
 *
 * @param[out] d0 The first half of the output half_2 accumulator.
 * @param[out] d1 The second half of the output half_2 accumulator.
 * @param[in] a0 The first half of the first input half_2 matrix.
 * @param[in] a1 The second half of the first input half_2 matrix.
 * @param[in] a2 The first half of the second input half_2 matrix.
 * @param[in] a3 The second half of the second input half_2 matrix.
 * @param[in] b0 The first half of the half_2 matrix B.
 * @param[in] b1 The second half of the half_2 matrix B.
 * @param[in] c0 The first half of the half_2 accumulator matrix C.
 * @param[in] c1 The second half of the half_2 accumulator matrix C.
 */
__device__ static inline void hmma16816(half_2& d0, half_2& d1, const half_2& a0, const half_2& a1, const half_2& a2, const half_2& a3, const half_2& b0,
                                        const half_2& b1, const half_2& c0, const half_2& c1) {
    asm volatile(
        // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};"

        // D matrix
        : "=r"(*(uint32_t*)(&d0)), "=r"(*(uint32_t*)(&d1))

        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)), "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

          // B matrix
          "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

          // C matrix
          "r"(*(uint32_t*)(&c0)), "r"(*(uint32_t*)(&c1)));
}

#ifdef CUDA_HOPPER
/**
 * @brief Perform the HMMA.16816 operation for FP8 using fp8e4m3_2.
 *
 * Using mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 instruction
 * but with fp8e4m3_2 (2 FP8 values) instead of fp8e4m3_4
 */
/**
 * @brief Perform the HMMA.16816 operation for FP8.
 *
 * This function performs the fp8-precision matrix multiply-accumulate operation
 * using the `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` instruction.
 *
 * @param[out] d0 The first half of the output float2 accumulator.
 * @param[out] d1 The second half of the output float2 accumulator.
 * @param[in] a0,a1,a2,a3 Input FP8 matrix A values
 * @param[in] b0,b1 Input FP8 matrix B values
 * @param[in] c0,c1 Input float2 accumulator matrix C values
 */
__device__ static inline void hmma16816(float2& d0, float2& d1, const fp8e4m3_4& a0, const fp8e4m3_4& a1, const fp8e4m3_4& a2, const fp8e4m3_4& a3,
                                        const fp8e4m3_4& b0, const fp8e4m3_4& b1, const float2& c0, const float2& c1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"

        // D matrix (output)
        : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)

        // A matrix
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)), "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),

          // B matrix
          "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),

          // C matrix
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y));
}
#endif
