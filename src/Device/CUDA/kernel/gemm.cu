/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief General C=op(A*B)    from many great work of open source commmutiy(TK,llm.c,calm,...)
 *
 */
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cuda_bf16.hpp>
#include <type_traits>

#include "../../../Tensor/GTensor.hpp"
#include "../../../Tensor/GeQuant.hpp"
#include "../../tile_kernel/tl_kernels.hpp"
#include "../cuda_common.h"
// #include "gelu.cuh"
#include "operator.cuh"
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

/*
    d(m,n) = a'*b
    Wrapper around cublasLtMatmul(https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)

    transA = 1: in cuBLAS, matrices are column-major by default. W matrix is (m, n) in row-major layout, which is (n, m) in column-major.
*/
void CU_mm_blas(floatX* d, const floatX* wX, const floatX* b, int m, int n, int k, float beta = 0.0, int transA = 1, int transB = 0, int flag = 0x0) {
    const float alpha = 1.0f;
    int lda = transA ? k : m, ldb = transB ? n : k;
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasGemmEx(cublas_handle, opA, opB, m, n, k, &alpha, wX, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, d, CUDA_R_16BF, m, CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT);  //  CUBLAS_GEMM_DEFAULT_TENSOR_OP[DEPRECATED]
    /*cublasGemmEx(handle,
                 CUBLAS_OP_T,        // Transpose W (since it's row-major)
                 CUBLAS_OP_N,        // Don't transpose x
                 m,                  // rows of C (output y)
                 1,                  // columns of C (output y is a vector)
                 n,                  // common dimension (k)

                 &alpha,             // host pointer
                 W,                  // A matrix (W)
                 CUDA_R_16BF,        // A datatype
                 n,                  // leading dimension of A

                 x,                  // B matrix (x)
                 CUDA_R_16BF,        // B datatype
                 n,                  // leading dimension of B

                 &beta,              // host pointer
                 y,                  // C matrix (y)
                 CUDA_R_16BF,        // C datatype
                 m,                  // leading dimension of C

                 CUDA_R_32F,         // compute type: use fp32 for precision
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);*/

    return;
}

// y = alpha*W*x + beta*y
void CU_mv_(floatX* y, const floatX* W, const floatX* x, int m, int n, float alpha, float beta) {
    assert(alpha == 1.0f);  //&& beta == 0.0f
    CU_mm_blas(y, W, x, m, 1, n, beta);
}

/*  d(m,n) = alpha*a'*b + beta*d + bias
    Wrapper around cublasLtMatmul(https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)
*/
template <class FloatC, class FloatA, class FloatB, class FloatBias>
void CU_mm_blasLt(FloatC* d, const FloatA* a, const FloatB* b, const FloatBias* bias, TASKA_AxB& taskm, int flag) {
    assert(taskm.isValid());
    assert(d!=nullptr && a!=nullptr && b!=nullptr);
    int m = taskm.m, n = taskm.n, k = taskm.k;
    cudaStream_t stream = (cudaStream_t)taskm.device;
    NVTX_RANGE_FN();
    bool isDone = false;
    // #ifdef __USE_TILELANG__
    //     for (auto K : TL_GEMM_tables) {                       // KOIFISH(torch)   row-major
    //         if (!accumulate && K.isMatch(n, m, k, transA)) {  // (m,n) of CU_mm_ is transpose of output matrix
    //             assert(transB == 0);
    //             K.kernel<<<dim3(K.grid_x, K.grid_y, K.grid_z), dim3(K.block_x, K.block_y, K.block_z), 49152, stream>>>(b, a, d);
    //             isDone = True;
    //             break;
    //         }
    //     }
    //     if (isDone)
    //         return;
    // #endif
    bool has_bias              = (bias != nullptr);
    hBITARR workspace          = (hBITARR)GTensor::qkv_workspace;
    std::size_t workspace_size = GTensor::workspace_size;
    // hBITARR workspace          = (hBITARR)GTensor::buff;
    // std::size_t workspace_size = GTensor::buff_len;
    assert(workspace_size >= 32 * 1024 * 1024);
    // check alignment (some modes work unaligned, but it is always best to be aligned for performance)
    if (((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        _ERROR("All CU_mm_blas_ pointers must be aligned!\n");
        exit(KOIFISH_BLAS_UNALIGN);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose   = CUBLAS_OP_T;
    cublasCheck(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (taskm.transA) ? &opTranspose : &opNoTranspose, sizeof(opTranspose)));
    cublasCheck(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (taskm.transB) ? &opTranspose : &opNoTranspose, sizeof(opNoTranspose)));

    // define matrix layouts
    cublasLtMatrixLayout_t ALayout;
    cublasLtMatrixLayout_t BLayout;
    cublasLtMatrixLayout_t DLayout;
    cublasLtMatrixLayout_t CLayout;
    if (taskm.transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, cuLibType<FloatA>, k, m, k));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, cuLibType<FloatA>, m, k, m));
    }
    if (taskm.transB) {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, cuLibType<FloatB>, n, k, n));
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, cuLibType<FloatB>, k, n, k));
    }
    // cuBLASLt requires C in FP8 mode to be BF16 or FP32... (sigh)
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, cuLibType<FloatC>, m, n, m));
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, cuLibType<FloatC>, m, n, m));

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (has_bias) {
        epilogue = CUBLASLT_EPILOGUE_BIAS;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = cuLibType<FloatBias>;  // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    if (taskm.scale_a) {
        if (sizeof(FloatA) != 1) {
            throw std::runtime_error("Scaling A is only supported for FP8");
        }
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &(taskm.scale_a), sizeof(&(taskm.scale_a))));
    }
    if (taskm.scale_b) {
        if (sizeof(FloatB) != 1) {
            throw std::runtime_error("Scaling B is only supported for FP8");
        }
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &(taskm.scale_b), sizeof(&(taskm.scale_b))));
    }

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout, preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) {
        _ERROR("No cuBLASLt algorithm@%d: m: %d, n: %d, k: %d, bias: %d\n", tpCuBLAS, n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // set whether to accumulate (i.e. D += C) or not - note this isn't considered in algorithm selection (?!)
    float one    = 1.f, zero   = 0.f;
    float* alpha = &(taskm.alpha);  //one;
    float* beta  = &(taskm.beta);   //taskm.isAccumuDelta ? &one : &zero;

    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc, alpha, a, ALayout, b, BLayout, beta, d, CLayout, d, DLayout, &heuristic.algo, workspace,
                               workspace_size, stream));
    cudaCheck(cudaGetLastError());

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
    d = wGensor*b+bias
    wrapper of CU_abc_ & cublasGemmEx & more ...
*/
void CU_mm_(floatX* d, hGTensor wGensor, const floatX* b, const floatX* bias, int m, int n, int k, cudaStream_t stream, int transA, int transB, float beta,
            floatX* pre_gelu) {
    NVTX_RANGE_FN();
    const float alpha = 1.0f;  //, beta = accumulate ? 1.0f : 0.0f;
    hQUANT hQuant     = wGensor->GetDynamicQuant();
    bool isDone       = false;
    if (hQuant != nullptr) {
        transA = hQuant->params.TransA;
    }
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (bias != nullptr) {  //  bias != nullptr || pre_gelu != nullptr
        floatX* wX = wGensor->GetDataX();
        TASKA_AxB taskm(main_stream, m, n, k, transA, transB, beta);
        CU_mm_blasLt(d, wX, b, bias, taskm /*m, n, k, nullptr, nullptr, main_stream, transA, transB, beta*/);
        return;
    }
    bool isBlas = true;
    int lda = transA ? k : m, ldb = transB ? n : k;

    if (isBlas) {
        floatX* wX = wGensor->GetDataX();
        // [2048,1024] x [1024,8192] => [2048,8192]         or(transA) [1024,8192]' x [1024,2048] => [8192,2048]
        // #ifdef __USE_TILELANG__
        //         for (auto K : TL_GEMM_tables) {        // KOIFISH(torch)   row-major
        //             if (K.isMatch(n, m, k, transA)) {  // (m,n) of CU_mm_ is transpose of output matrix
        //                 K.kernel<<<dim3(K.grid_x, K.grid_y, K.grid_z), dim3(K.block_x, K.block_y, K.block_z), 49152, stream>>>(b, wX, d);
        //                 isDone = True;
        //                 break;
        //             }
        //         }
        // #endif
        if (!isDone) {  //  BLAS:   column-major
            cublasGemmEx(cublas_handle, opA, opB, m, n, k, &alpha, wX, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, d, CUDA_R_16BF, m, CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT);  //  CUBLAS_GEMM_DEFAULT_TENSOR_OP[DEPRECATED]
        }

        // PrintTensor<floatX>("d=wGensor*b", d, true, m, n, 1, 1, -1);
    }

    return;
}

template <typename Typ>
struct TASKA_gemm_back {
    floatX *delta = nullptr, *dweight = nullptr, *deltaIn = nullptr, *inp = nullptr, *weight = nullptr;
    floatX* bias = nullptr;
    int B, T, C, OC;
    int block3 = 0, tpb = 512;  //  tpb is the number of threads in one block
    int grid3 = 0, nBlock = 0;  // grid3=(nBlock, 1, 1), nBlock is the total number of blocks
    cudaStream_t stream;
    // TASKA_AxB taskW, taskDelta;
    bool isFixWeight = false;
    TASKA_gemm_back(const GTensor* hTensor, cudaStream_t stream_, int flag = 0x0) {
        // const int grid_size_x = CEIL_DIV(OC, OC_per_warp);  // e.g. 12 horizontal blocks for 768 OCs at BF16
        // const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!
    }

    bool cuda() const {
        NVTX_RANGE_FN();
        bool transAW = false;
        // if(isTransW)
        //     transAW = true;
        // backward to bias, if given, does a +=
        assert(bias == nullptr && "Only support null bias!");
        if (bias != nullptr) {
            /*// Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
            // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

            const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

            dim3 block_dim        = {4, 8, (unsigned)block_size / WARP_SIZE};
            const int OC_per_warp = block_dim.y * X128::size;  // 64 at BF16

            const int grid_size_x = CEIL_DIV(OC, OC_per_warp);  // e.g. 12 horizontal blocks for 768 OCs at BF16
            const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!

            // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
            // and write results directly to the output.
            if (grid_size_y == 1) {
                matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(bias, deltaIn, B, T, OC, False);
                cudaCheck(cudaGetLastError());
            } else {
                // kernel 9 overwrites temp buffer, so no need to memset
                matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, deltaIn, B, T, OC, True);
                cudaCheck(cudaGetLastError());
                reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(bias, dbias_buffer, OC, grid_size_y);
                cudaCheck(cudaGetLastError());
            }
            bias = NULL;  // prevent bias calculation from also being fused in CU_mm_blas_ below (if we enabled fusion)*/
        }

        /*CU_mm_blasLt(delta, weight, deltaIn, bias, C, B * T, OC, nullptr, nullptr, stream, (int)transAW, 0, isAccumuDelta);

        // backward GELU (if it wasn't fused into the matmul above)
        // if (gelu_fusion < 2 && pre_gelu) {
        //     Activation_backward_inplace(delta, pre_gelu, B * T * C, stream);
        // }

        // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
        CU_mm_blasLt(dweight, inp, deltaIn, bias, C, OC, B * T, nullptr, nullptr, stream, transAW, 1, isAccumuDelta);*/
        return true;
    }
};


/*
void matmul_backward(floatX* delta, floatX* dweight, floatX* bias, floatX* deltaIn, floatX* inp, floatX* weight, float* dbias_buffer, int B, int T, int C,
                     int OC, cudaStream_t stream, bool isTransW = false, floatX* pre_gelu = NULL, bool isAccumuDelta = false) {
    NVTX_RANGE_FN();
    bool transAW = false;
    // if(isTransW)
    //     transAW = true;
    // backward to bias, if given, does a +=
    if (bias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim        = {4, 8, (unsigned)block_size / WARP_SIZE};
        const int OC_per_warp = block_dim.y * X128::size;  // 64 at BF16

        const int grid_size_x = CEIL_DIV(OC, OC_per_warp);  // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x));  // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if (grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(bias, deltaIn, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, deltaIn, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(bias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        bias = NULL;  // prevent bias calculation from also being fused in CU_mm_blas_ below (if we enabled fusion)
    }
    

    CU_mm_blasLt(delta, weight, deltaIn, bias, C, B * T, OC, nullptr, nullptr, stream, (int)transAW, 0, isAccumuDelta ? 1.0f : 0.0f, true);

    // backward GELU (if it wasn't fused into the matmul above)
    // if (gelu_fusion < 2 && pre_gelu) {
    //     Activation_backward_inplace(delta, pre_gelu, B * T * C, stream);
    // }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    CU_mm_blasLt(dweight, inp, deltaIn, bias, C, OC, B * T, nullptr, nullptr, stream, transAW, true, 1 , true);

}*/
