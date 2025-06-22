/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *        
 * 
 *  \brief General C=op(A*B)    from many great work of open source commmutiy(TK,llm.c,calm,...)
 *  
 */
#include <assert.h>
#include <type_traits> 
#include <cuda_fp16.h>
#include <cuda_bf16.hpp>
#include <cuda_fp8.h>     
#include "../cuda_common.h"
#include "utils.cuh"
#include "gelu.cuh"
#include "matmul_bit.cuh"

/*  Somte ticks
    1.  Tensor cores are used to accelerate GEMMs. But it doesn’t mean there’s enough resources to execute other operation with CUDA cores. There are other resources that are shared among the on-chip resources, when kernels are called.
*/

template<typename OutFloat, bool UseAuxBuffer>
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* deltaIn, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);

    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;  // 64 at BF16

    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;

    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }

    if(global_oc < OC) {
        // sum up over all bt within registers
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(deltaIn + global_oc + idx*OC);
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
        if(warp_d == 0) {
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
        if(warp_d == 0 && global_oc < OC) {
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
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }

        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}


/*  d(m,n) = a'*b + bias
    Wrapper around cublasLtMatmul(https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) or 
*/
void inline CU_mm_blas(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
                     int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
                     int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
                     bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false) {
    NVTX_RANGE_FN();
    bool has_bias = (bias != NULL), has_gelu = (pre_gelu != NULL);
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if(bias==nullptr && pre_gelu==nullptr && batch_count==0)   {
        // A little faster than cublasLtMatmul ? 
        int lda = transA ? k : m; 
        int ldb = transB ? n : k;          
        cublasGemmEx(cublas_handle,opA,opB, m,n,k,  &alpha,     a, CUDA_R_16BF, lda, 
            b, CUDA_R_16BF, ldb,    &beta,  d, CUDA_R_16BF, m, CUDA_R_32F,CUBLAS_GEMM_DEFAULT );     //  CUBLAS_GEMM_DEFAULT_TENSOR_OP[DEPRECATED]
        return;
    }
    // check alignment (some modes work unaligned but it always best to be aligned for performance)
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
        printf("All cuBLASLt pointers must be aligned!\n");
        exit(EXIT_FAILURE);
    }

    // create the operation descriptor
    cublasLtMatmulDesc_t operationDesc;
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    int returnedResults = 0;
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulHeuristicResult_t heuristic;


    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA,   sizeof(opA)));       //(transA)  ? &opTranspose : &opNoTranspose
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB,   sizeof(opB)));     //(transB) ? &opTranspose   : &opNoTranspose

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
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // create a preference handle with specified max workspace
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    // setup epilogue and associated pointers for bias & gelu
    cublasLtEpilogue_t epilogue;
    if (has_gelu) {
        int64_t gelu_ld = m; // todo - is this affected by anything else?
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS;
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    }
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : tpCuBLAS; // force BF16 bias for FP8 mode
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // set scale type to FP32 (needs to be FP16 if and only if using CUBLAS_COMPUTE_16F, so it's FP32 even for FP8!)
    cublasDataType_t scale_type = CUDA_R_32F;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // find a suitable algorithm (cached internally so shouldn't take much CPU time in practice)
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                                   preference, 1, &heuristic, &returnedResults);
    if (returnedResults == 0) { //  CUBLAS_STATUS_SUCCESS           
        printf("No cuBLASLt algorithm@%d: m: %d, n: %d, k: %d, bias: %d\n", tpCuBLAS, n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }


    // call the matmul
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                               &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                               &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

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
void inline matmul_backward(floatX* delta, floatX* dweight, floatX* dbias,
                     floatX* deltaIn, floatX* inp, floatX* weight,
                     float* dbias_buffer,
                     int B, int T, int C, int OC, cudaStream_t stream,bool isTransW = false,
                     floatX* pre_gelu=NULL, bool isAccumuDelta=false) {
    NVTX_RANGE_FN();
    bool transAW = false;
    int gelu_fusion=1;          //assert(gelu_fusion==1);
    // if(isTransW)
    //     transAW = true;
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        // Each warp is responsible for 8 * "x128::size" = 64 OCs at BF16 (OC must be a multiple of 64!)
        // Block size is 1024 | 768 threads (32|24 warps) and we reduce those values into 1 at the end

        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // If we have enough OC that we don't need cross-block reductions, we can skip the bias_buffer accumulation
        // and write results directly to the output.
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, deltaIn, B, T, OC, False);
            cudaCheck(cudaGetLastError());
        } else {
            // kernel 9 overwrites temp buffer, so no need to memset
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, deltaIn, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        dbias = NULL; // prevent dbias calculation from also being fused in CU_mm_blas below (if we enabled fusion)
    }

    // backward to input, uses = in the backward pass (set the gradient)
    CU_mm_blas(delta, weight, deltaIn, NULL, C, B*T, OC, stream, transAW, false, 0, 0, 0, 0, isAccumuDelta,
                    gelu_fusion >= 2 ? pre_gelu : NULL, true);

    // backward GELU (if it wasn't fused into the matmul above)
    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(delta, pre_gelu, B*T*C, stream);
    }

    // backward to weight, uses += in the backward pass (accumulate the gradient) by setting alpha=one
    CU_mm_blas(dweight, inp, deltaIn, NULL /*dbias*/, C, OC, B*T, stream, transAW, true, 0, 0, 0, 0,
                    true /* accumulate */, NULL, true);
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
	half us = fp8_e5m2_ff(v & 0xff); // we expect compiler to reuse this across multiple calls
	half s = us * half(-0.25f);      // we expect compiler to reuse this across multiple calls
	uint32_t p = v >> (8 + k * 3);
	half2 q = half2(int(p & 7), int((p >> 3) & 7));
	return __hfma2(q, half2(s, s), half2(us, us));
}

// warp-parallel mat*vec; each warp collaboratively computes mat*vec for a single row
// specialized for half/fp8/gf4 weights and ensures that we maximize transaction sizes by reading 4 bytes per thread
__device__ inline float CU_dot16x16_warppar(half* x, half* w, int i, int n) {
	int lane = threadIdx.x % warpSize;
	half2 val = {0, 0};
	for (int j = lane * 2; j < n; j += warpSize * 2) {
		half2 ww = *(half2*)&w[i * n + j];
		half2 xx = *(half2*)&x[j];
		val = __hfma2(ww, xx, val);
	}
	return warpreduce_sum(float(val.x + val.y));
}
__device__ inline float CU_dot16x8_warppar(half* x, __nv_fp8_e5m2* w, int i, int n) {
	int lane = threadIdx.x % warpSize;
	half2 val = {0, 0};
	// use 64-bit loads instead of 32-bit loads to increase memory throughput on H100/A100
	// without this we are seeing lower throughput given the limited number of parallel warps in coop kernel
	// this is performance-neutral on 4090 but results in issues with x[] load coalescing (that are benign)
	for (int j = lane * 8; j < n; j += warpSize * 8) {
		ablock<__nv_fp8x2_e5m2, 4> wwp = *(ablock<__nv_fp8x2_e5m2, 4>*)&w[i * n + j];
		ablock<__half2_raw, 4> xxp = *(ablock<__half2_raw, 4>*)&x[j];
#pragma unroll
		for (int k = 0; k < 4; ++k) {
			half2 ww = fp8x2_e5m2_ff(wwp.v[k].__x);
			half2 xx = xxp.v[k];
			val = __hfma2(ww, xx, val);
		}
	}
	return warpreduce_sum(float(val.x + val.y));
}
__device__ inline float matmul_warppar(__nv_bfloat16* x, __nv_fp8_e5m2* w, int i, int n) {
    float val = 0.0f;
    int lane = threadIdx.x % warpSize;	
    for(int j = lane * 4; j < n; j += warpSize * 4) {
        //ablock<__nv_fp8x4_e5m2, 1> wwp = *(ablock<__nv_fp8x4_e5m2, 1>*)&w[i * n + j];
        float4 ww = fp8x4_e5m2_ff((__nv_fp8x4_e5m2 *)(w+i*n+j));
        float4 xx = {x[j],x[j+1],x[j+2],x[j+3]};	//*(float4*)&x[j];
        val += ww.x * xx.x;				val += ww.y * xx.y;				val += ww.z * xx.z;				val += ww.w * xx.w;
        //val += x[j] * ww.x + x[j+1] * ww.y + x[j+2] * ww.z + x[j+3] * ww.w;
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
/*
#include "common.h"
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

//  C = A*B
template<typename Tw, typename Ta>
__global__ void mmaNaiveKernel(const Tw *__restrict__ A, const Ta *__restrict__ B, Ta *__restrict__ C, size_t M,
                               size_t N, size_t K) {
    const size_t K_tiles = div_ceil(K, MMA_K);

    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;

    if (warp_row >= M || warp_col >= N) {
        return;
    }

    __shared__ Tw A_smem[MMA_M][MMA_K];
    __shared__ Ta B_smem[MMA_N][MMA_K];
    __shared__ Ta C_smem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RC[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&A_smem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&A[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&B_smem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&B[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        uint32_t A_smem_lane_addr = __cvta_generic_to_shared(&A_smem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_smem_lane_addr);

        uint32_t B_smem_lane_addr = __cvta_generic_to_shared(&B_smem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_smem_lane_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        __syncthreads();
    }

    *((uint32_t *)(&C_smem[lane_id / 4][0]) + lane_id % 4) = RC[0];
    *((uint32_t *)(&C_smem[lane_id / 4 + 8][0]) + lane_id % 4) = RC[1];

    __syncthreads();

    if (lane_id < MMA_M) {
        *((int4 *)(&C[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&C_smem[lane_id][0]));
    }
}*/

// C[m,n] = A[m,k]*B[k,n]       A,B,C are all row-major
template<typename Tw, typename Ta>
__global__ void static tABC_0(const Ta *__restrict__ A, const Tw *__restrict__ B, Ta *__restrict__ C, size_t M, size_t N, size_t K) {
    size_t row = threadIdx.y + blockDim.y * blockIdx.y;
    size_t col = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= M && col >= N) {
        return;
    }
    Ta tmp = (Ta)(0.0);
#if defined(ENABLE_FP8)
#else
 #pragma unroll
    for (size_t i = 0; i < K; ++i) {
        tmp += A[row * K + i] * B[i + col * K];
    }
    C[row * N + col] = tmp;
#endif
}

/*
    Forward of Tensor: rhs = GELU(lhs*W+b)
    lhs[m,k]*W[k,n] => rhs[m,n]       lhs,W,rhs are all row-major
*/
template<typename Tw, typename Ta>
void tMM(Ta *lhs,Tw *W, Ta *rhs,Tw *b, size_t M, size_t N, size_t K,bool transAW, cudaStream_t main_stream,int flag=0) {
    assert(typeid(Ta)==typeid(Tw));

    switch(DEBUG.T_GEMM){
    case -1:{
        // PrintTensor<Ta>("rhs",rhs,true,M,N,1,1,-1);
        // PrintTensor<Ta>("W",W,true,K,N,1,1,-1);
        // PrintTensor<Ta>("lhs",lhs,true,M,K,1,1,-1);
        dim3 block(16, 16), grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y));
        tABC_0<Ta,Tw><<<grid, block, 0, main_stream>>>(lhs, W, rhs, M, N, K);        }
        break;
    case 1:{
        // dim3 block(WARP_SIZE), grid(CEIL_DIV(N, MMA_N), CEIL_DIV(M, MMA_M));
        // mmaNaiveKernel<Ta,Tw><<<grid, block>>>(lhs, W, rhs, M, N, K); 
    }        break;
    case 2:{
        const float alpha = 1.0f,beta = 0.0f;
        assert(transAW==true);
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,N, M,K,&alpha, W, CUDA_R_16BF, M,        
                    lhs, CUDA_R_16BF, K,&beta, rhs, CUDA_R_16BF, M, CUDA_R_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP); }
        break;
    default:
        CU_mm_blas(rhs, W, lhs, b, N, M, K, main_stream, transAW, false, 0, 0, 0, 0, false, NULL, false);
        //  CU_mm_blas(rhs, wX, ToX(lhs_), ToX0(b), OC, B*T, IC, main_stream, transAW, false, 0, 0, 0, 0, false, NULL, false);
        break;
    }
    

    // static size_t smem_max_size = initMmaAsyncStage4<Ta>();
    // dim3 block(THREADS_PER_BLOCK);
    // dim3 grid(BLOCK_STRIDE, div_ceil(M, BLOCK_ROWS), div_ceil(N, BLOCK_COLS * BLOCK_STRIDE));
    // mmaAsyncStage4Kernel<Ta><<<grid, block, smem_max_size>>>(W, lhs, rhs, M, N, K);
}
