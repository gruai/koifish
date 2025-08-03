/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief General C=op(A*B)    from many great work of open source commmutiy(TK,llm.c,calm,...)
 *
 */

/**

*/

#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cuda_bf16.hpp>
#include <type_traits>

#include "../cuda_common.h"
#include "gelu.cuh"
#include "utils.cuh"
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

//  从共享内存（shared memory）中加载一个8x8的矩阵块（FP16/BF16 格式），并以4x并行方式分发到 Warp 内的多个线程寄存器中
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

//  Warp Matrix Multiply-Accumulate: A[16,16]xB[16,8]=>C[16,8]
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

//  Volta/Turing don't support BF16 !!!
#define HMMA16816_bf(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                   \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                  \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))

/*  Somte ticks
    1.  Tensor cores are used to accelerate GEMMs. But it doesn’t mean there’s enough resources to execute other operation with CUDA cores. There are other
   resources that are shared among the on-chip resources, when kernels are called.
*/

template <typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* deltaIn, int B, int T, int OC, std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d  = (int)threadIdx.x;
    int warp_c  = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc  = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt     = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if (global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(deltaIn + global_oc + idx * OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];

    // reduce within-warp results
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if (warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // block-wide reductions
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if (warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = (OutFloat)(a);
            }
        }
    }
}

__global__ void static reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size;
    assert(n % x128::size == 0);
    if (idx < n) {
        f128 acc;
        for (int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for (int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for (int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for (int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX)((float)dst[idx + k] + acc[k]);
        }
    }
}

/*  d(m,n) = alpha*a'*b + beta*d + bias
    Wrapper around cublasLtMatmul(https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)
*/
void CU_mm_blas(floatX* d, const floatX* a, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream = 0, int transA = 1, int transB = 0,
                float alpha=1.0, float beta = 0.0, floatX* pre_gelu = NULL, bool backward = false) {
    NVTX_RANGE_FN();
    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        _INFO("All CU_mm_blas_ pointers must be aligned!\n");
        exit(KOIFISH_BLAS_UNALIGN);
    }

    // assert(batch_count==0);
    bool has_bias = (bias != NULL), has_gelu = (pre_gelu != NULL);
    //,const float alpha = 1.0f;    beta = accumulate ? 1.0f : 0.0f;
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    /*if (bias == nullptr && pre_gelu == nullptr) {
        int lda = transA ? k : m, ldb = transB ? n : k;
        if (!transA && !transB) {
            //Back of delta: [768,50304] x [50304,8192] => [768,8192]
            if (DEBUG.T_GEMM >= 0)
                CU_abc(d, a, b, bias, m, n, k, stream, transA, transB, accumulate, pre_gelu, backward);
            else
                cublasGemmEx(cublas_handle, opA, opB, m, n, k, &alpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, d, CUDA_R_16BF, m, CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT);
            // PrintTensor("d=axb", d, true, m, n, 1, 1, -1);
            // exit(KOIFISH_EXIT_DEBUG);
        } else {
            // [50304,768] x [768,8192] => [50304,8192]         or(transA) [768,50304]' x [768,8192] => [50304,8192]
            cublasGemmEx(cublas_handle, opA, opB, m, n, k, &alpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, d, CUDA_R_16BF, m, CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT);  //  CUBLAS_GEMM_DEFAULT_TENSOR_OP[DEPRECATED]
        }

        return;
    }*/

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;

    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));  //(transA)  ? &opTranspose : &opNoTranspose
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));  //(transB) ? &opTranspose   : &opNoTranspose

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, tpCuBLAS, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, tpCuBLAS, m, k, m));
    }
    if (transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, tpCuBLAS, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, tpCuBLAS, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : tpCuBLAS, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, tpCuBLAS, m, n, m));

    // Strided Batched GEMM (used for non-flash attention, equivalent to cublasGemmStridedBatchedEx)
    /*if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }*/

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(
        cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m;  // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias);  // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if (has_bias) {
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : tpCuBLAS;  // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout, preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {  //  CUBLAS_STATUS_SUCCESS
        _INFO("No cuBLASLt algorithm@%d: m: %d, n: %d, k: %d, bias: %d\n", tpCuBLAS, n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // d = alpha*A*B + beta*d + bias
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout, &heuristic.algo,
                               cublaslt_workspace, cublaslt_workspace_size, stream));

    // cleanups
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}

/*
    backward to bias/weight
    backward to input(delta of this layer)
    backward to gelu

    deltaIn is delta of next layer
*/
void matmul_backward(floatX* delta, floatX* dweight, floatX* dbias, floatX* deltaIn, floatX* inp, floatX* weight, float* dbias_buffer, int B, int T, int C,
                     int OC, cudaStream_t stream, bool isTransW = false, floatX* pre_gelu = NULL, bool isAccumuDelta = false) {
    NVTX_RANGE_FN();
    bool transAW    = false;
    int gelu_fusion = 1;  // assert(gelu_fusion==1);
    // if(isTransW)
    //     transAW = true;
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim        = {4, 8, (unsigned)block_size / WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size;  // 64 at BF16

        const int grid_size_x = CEIL_DIV(OC, OC_per_warp);  // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if (grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, deltaIn, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, deltaIn, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL;  // prevent dbias calculation from also being fused in CU_mm_blas_ below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    CU_mm_blas(delta, weight, deltaIn, NULL, C, B * T, OC, stream, transAW, false, 1.0, isAccumuDelta, gelu_fusion >= 2 ? pre_gelu : NULL, true);

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(delta, pre_gelu, B * T * C, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    CU_mm_blas(dweight, inp, deltaIn, NULL /*dbias*/, C, OC, B * T, stream, transAW, true, 1.0, 1 /* accumulate */, NULL, true);
}

// fast fp8x2 => half2 conversion; drops unnecessary NaN handling from __nv_cvt_fp8_to_halfraw
__device__ inline half2 fp8x2_e5m2_ff(unsigned int v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    __nv_fp8x2_e5m2 p;
    p.__x = v;
    return half2(p);
#else
    __half2_raw h = {(unsigned short)(v << 8), (unsigned short)(v & 0xff00)};
    return h;
#endif
}

// gf4 decoding (2 values): 8 3-bit values + 1 fp8 scale are packed in a 32-bit word
__device__ inline half2 cu_gf4x2_ff(uint32_t v, int k) {
    half us    = fp8_e5m2_ff(v & 0xff);  // we expect compiler to reuse this across multiple calls
    half s     = us * half(-0.25f);      // we expect compiler to reuse this across multiple calls
    uint32_t p = v >> (8 + k * 3);
    half2 q    = half2(int(p & 7), int((p >> 3) & 7));
    return __hfma2(q, half2(s, s), half2(us, us));
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for half/fp8/gf4 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float CU_dot16x16_warppar(half* x, half* w, int i, int n) {
    int lane  = threadIdx.x % warpSize;
    half2 val = {0, 0};
    for (int j = lane * 2; j < n; j += warpSize * 2) {
        half2 ww = *(half2*)&w[i * n + j];
        half2 xx = *(half2*)&x[j];
        val      = __hfma2(ww, xx, val);
    }
    return warpreduce_sum(float(val.x + val.y));
}
__device__ inline float CU_dot16x8_warppar(half* x, __nv_fp8_e5m2* w, int i, int n) {
    int lane  = threadIdx.x % warpSize;
    half2 val = {0, 0};
    // use 64-bit loads instead of 32-bit loads to increase memory throughput on H100/A100
    // without this we are seeing lower throughput given the limited number of parallel warps in coop kernel
    // this is performance-neutral on 4090 but results in issues with x[] load coalescing (that are benign)
    for (int j = lane * 8; j < n; j += warpSize * 8) {
        ablock<__nv_fp8x2_e5m2, 4> wwp = *(ablock<__nv_fp8x2_e5m2, 4>*)&w[i * n + j];
        ablock<__half2_raw, 4> xxp     = *(ablock<__half2_raw, 4>*)&x[j];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            half2 ww = fp8x2_e5m2_ff(wwp.v[k].__x);
            half2 xx = xxp.v[k];
            val      = __hfma2(ww, xx, val);
        }
    }
    return warpreduce_sum(float(val.x + val.y));
}
__device__ inline float matmul_warppar(__nv_bfloat16* x, __nv_fp8_e5m2* w, int i, int n) {
    float val = 0.0f;
    int lane  = threadIdx.x % warpSize;
    for (int j = lane * 4; j < n; j += warpSize * 4) {
        // ablock<__nv_fp8x4_e5m2, 1> wwp = *(ablock<__nv_fp8x4_e5m2, 1>*)&w[i * n + j];
        float4 ww = fp8x4_e5m2_ff((__nv_fp8x4_e5m2*)(w + i * n + j));
        float4 xx = {x[j], x[j + 1], x[j + 2], x[j + 3]};  //*(float4*)&x[j];
        val += ww.x * xx.x;
        val += ww.y * xx.y;
        val += ww.z * xx.z;
        val += ww.w * xx.w;
        // val += x[j] * ww.x + x[j+1] * ww.y + x[j+2] * ww.z + x[j+3] * ww.w;
    }
    return warpreduce_sum(val);
}

__device__ inline float CU_dot16x4_warppar(half* x, uint32_t* w, int i, int n) {
    int lane = threadIdx.x % warpSize;
    if (n % (warpSize * 64) == 0) {
        half2 val = {0, 0};
        for (int j = lane * 16; j < n; j += warpSize * 64) {
            ablock<uint32_t, 2> wgp[4] = {
                *(ablock<uint32_t, 2>*)&w[i * n / 8 + j / 8],
                *(ablock<uint32_t, 2>*)&w[i * n / 8 + j / 8 + (warpSize * 16) / 8],
                *(ablock<uint32_t, 2>*)&w[i * n / 8 + j / 8 + (warpSize * 32) / 8],
                *(ablock<uint32_t, 2>*)&w[i * n / 8 + j / 8 + (warpSize * 48) / 8],
            };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                ablock<__half2_raw, 8> xx = *(ablock<__half2_raw, 8>*)&x[j + warpSize * 16 * u];
#pragma unroll
                for (int k = 0; k < 8; k += 2) {
                    val = __hfma2(cu_gf4x2_ff(wgp[u].v[0], k), xx.v[k / 2], val);
                }
#pragma unroll
                for (int k = 0; k < 8; k += 2) {
                    val = __hfma2(cu_gf4x2_ff(wgp[u].v[1], k), xx.v[k / 2 + 4], val);
                }
            }
        }
        return warpreduce_sum(float(val.x + val.y));
    } else {
        half2 val = {0, 0};
        for (int j = lane * 16; j < n; j += warpSize * 16) {
            ablock<uint32_t, 2> wgp = *(ablock<uint32_t, 2>*)&w[i * n / 8 + j / 8];

            ablock<__half2_raw, 8> xx = *(ablock<__half2_raw, 8>*)&x[j];
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                val = __hfma2(cu_gf4x2_ff(wgp.v[0], k), xx.v[k / 2], val);
            }
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                val = __hfma2(cu_gf4x2_ff(wgp.v[1], k), xx.v[k / 2 + 4], val);
            }
        }
        return warpreduce_sum(float(val.x + val.y));
    }
}

// C[m,n] = A[m,k]*B[k,n]       A,B,C are all row-major
template <typename Tw, typename Ta>
__global__ void static tABC_0(const Ta* __restrict__ A, const Tw* __restrict__ B, Ta* __restrict__ C, size_t M, size_t N, size_t k) {
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= M && col >= N) {
        return;
    }
    Ta tmp = (Ta)(0.0);
#if defined(ENABLE_FP8)
#else
#pragma unroll
    for (size_t i = 0; i < k; ++i) {
        tmp += A[row * k + i] * B[i + col * k];
    }
    C[row * N + col] = tmp;
#endif
}
