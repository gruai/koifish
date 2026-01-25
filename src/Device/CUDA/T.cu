/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Some trial/testing cuda kernels
 *  \author Yingshi Chen
 */
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "../../Manifold/gLLM.hpp"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"

extern cudaStream_t main_stream;

template <typename Typ>
__global__ void CU_rms_backward_v1(Typ* dX0, Typ* dweight, float* dW_scratch, const Typ* dY0, const Typ* X0, const Typ* weight, const float* rstd,
                                   TASKA_SWMD<Typ> smp, int nTH, int ldTH, unsigned int seed, int flag) {
    using typ128 = PackedN<Typ, 16 / sizeof(Typ)>;
    using f256   = PackedN<float, typ128::size>;
    assert(f256::size == typ128::size);

    int BLOCK_SIZE = blockDim.x;
    assert(BLOCK_SIZE == smp.block3);
    assert(smp.ldC0 == ldTH && smp.ldC >= ldTH);
    extern __shared__ float shared[];
    int warpId        = threadIdx.x / WARP_SIZE;  // warp index within a block
    int warpThreadIdx = threadIdx.x % WARP_SIZE;  // Thread index within the warp
    int taskId        = blockIdx.x * smp.warpsInBlock + warpId;

    float* dweight_shared = shared + smp.ldC;
    // warp 0 doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dweight_tmp_shared = shared + 2 * smp.ldC + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;
    // init shared memory to zero
    for (int i = threadIdx.x * f128::size; i < smp.ldC; i += BLOCK_SIZE * f128::size) {
        f128::zeros().store(dweight_shared + i);
    }
    __syncthreads();

    if (taskId >= nTH) {
        // make sure we're not reading uninitialized memory below
        f128::zeros().store(dweight_tmp_shared + threadIdx.x * f128::size);
    }
    assert(smp.warpsInGrid == nTH);
    for (int bt = taskId; bt < nTH; bt += smp.warpsInGrid) {
        const Typ *dout_bt = dY0 + bt * ldTH, *inp_bt = X0 + bt * ldTH;
        Typ* dinp_bt          = dX0 + bt * ldTH;
        float dnorm_norm_mean = 0.0f;  // dnorm_mean = 0.0f,
        int wrap_i0           = warpThreadIdx * typ128::size;
        for (int i = wrap_i0; i < ldTH; i += smp.wrap_stride) {
            typ128 dY128(dout_bt + i), xi(inp_bt + i), wi(weight + i);
            for (int k = 0; k < typ128::size; k++) {  //  delta_avg += (float)(dY[i] * w[i] * xi);
                dnorm_norm_mean += (float)wi[k] * (float)dY128[k] * (float)xi[k];
            }
        }
        const float rstd_bt = rstd[bt];
        dnorm_norm_mean     = warpReduceSum(dnorm_norm_mean, smp.wrap_mask) / ldTH * rstd_bt;

        // have to use ldC(>=ldTH), to ensure __syncthreads!
        for (int i8 = wrap_i0; i8 < smp.ldC; i8 += smp.wrap_stride) {
            typ128 dY128 = typ128::zeros(), xi = typ128::zeros(), wi = typ128::zeros();
            f256 dweight_f = f256::zeros();
            if (i8 < ldTH) {
                dY128        = typ128::load_cs(dout_bt + i8);
                xi           = typ128::load_cs(inp_bt + i8);
                typ128 dX128 = typ128::load(dinp_bt + i8);
                wi           = typ128::load(weight + i8);

                for (int i = 0; i < typ128::size; ++i) {
                    float dval     = 0.0f;
                    float norm_bti = ((float)xi[i]) * rstd_bt;
                    dval += (float)wi[i] * (float)dY128[i];  // term 1
                    dval -= norm_bti * dnorm_norm_mean;      // term 2
                    dval *= rstd_bt;                         // final scale
                    dX128[i] = flag == 0x200 ? (Typ)dval : (Typ)((float)dX128[i] + dval);
                }
                // TODO cache hint
                dX128.store(dinp_bt + i8);

                for (int i = 0; i < f256::size; ++i) {
                    dweight_f[i] = ((float)xi[i]) * rstd_bt * (float)dY128[i];
                }
                if (warpId != 0) {
                    dweight_f.store(dweight_tmp_shared + threadIdx.x * f256::size);
                }
                // if (bt == 0 && i8 == 0) {  //  only for debug
                //     printf("*** dnorm_norm_mean=%g rstd=%g dX=%g %g %g %g\n", dnorm_norm_mean, rstd_bt, (float)dX128[0], (float)dX128[1], (float)dX128[2],
                //            (float)dX128[3]);
                // }
            }
            __syncthreads();
            if (warpId == 0 && i8 < ldTH) {                   //&& i8 < ldTH
                for (int j = 1; j < smp.warpsInBlock; j++) {  // ugly code!
                    dweight_f.Add(dweight_tmp_shared + f256::size * (threadIdx.x + j * WARP_SIZE));
                }
                // dweight_f.store(dW_scratch + i8 + f128::size * o + ldTH * blockIdx.x);
                dweight_f.store(dweight_shared + i8);
            }
            __syncthreads();  //    ???
        }  //  loop of x128
    }  //  taskId
    __syncthreads();

    // Each block writes its partial sum to global memory
    // unsigned int* scratchFlag = reinterpret_cast<unsigned int*>(dW_scratch);
    // A cache line​ in CUDA devices (GPUs) is typically 128 bytes​ & cudaMalloc– Already Aligned (Usually)
    // float* scratch_dweight = reinterpret_cast<float*>(dW_scratch);
    for (int i = threadIdx.x * f128::size; i < ldTH;
         i += BLOCK_SIZE * f128::size) {  // Write to global memory in the same "shared memory banking friendly" order
        f128::load(dweight_shared + i).store(dW_scratch + i + ldTH * blockIdx.x);
    }
    // __syncthreads();
    // // that portion of shared memory is no longer used, so we can repurpose it for the scratch_ flag.
    // unsigned int* tmp_flag = (unsigned int*)(shared + 2 * smp.ldC);
    // if (threadIdx.x == 0) {
    //     *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    // }
    // __syncthreads();
}
// lite version: CU_rms_dw_v1_ is much faster!
template <typename Typ>
__global__ void CU_rms_dw_v0(Typ* dW, const float* dW0, int nTH, int ldTH, int flag = 0x0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ldTH) {
        return;
    }
    float sum = 0.f;
    // if (flag == 1) {  // flag==1 dW0 is float array!
    // } else {  // flag==1 dW0 is Typ array!
    //     const Typ* cur = reinterpret_cast<const Typ*>(dW0) + idx;
    //     for (int i = 0; i < nTH; i++, cur += ldTH) {
    //         sum += CU_T2Float(cur);
    //     }
    // }
    const float* cur = reinterpret_cast<const float*>(dW0) + idx;
    for (int i = 0; i < nTH; i++, cur += ldTH) {
        sum = sum + *cur;
    }
    dW[idx] = (Typ)(float)sum;
}
template <typename Typ>
__global__ void CU_rms_dw_v1(Typ* dW, const float* dW_scratch, TASKA_SWMD<Typ> smp, int ldTH, int flag = 0x0) {
    using typ128 = PackedN<Typ, 16 / sizeof(Typ)>;
    extern __shared__ float shared[];

    const float* scratch_dweight = reinterpret_cast<const float*>(dW_scratch);  //, *dweight_shared = shared;
    int tid = threadIdx.x, id = tid * f128::size;
    if (id >= ldTH) {  // gurard
        return;
    }

    // for (int i = threadIdx.x * f128::size; i < ldTH; i += blockDim.x * f128::size) {
    f128 dweight_accum;
    for (int block_id = 0; block_id < smp.grid3; block_id++) {
        int offset = id + ldTH * block_id;
        dweight_accum.Add(scratch_dweight + offset);
    }

    for (int k = 0; k < f128::size; ++k) {
        // dW[i + k] = CU_Float2T<Typ>(dweight_accum[k], 42);   // No need to do stochastic rounding
        dW[id + k] = (Typ)(dweight_accum[k]);
    }
    // dweight_accum.store(dweight_shared + i);
    // }
    // No need to do second loop: just 2*f128 => x128
    /*__syncthreads();
    int warpId        = threadIdx.x / WARP_SIZE;  // warp index within a block
    int warpThreadIdx = threadIdx.x % WARP_SIZE;  // Thread index within the warp
    for (int c = warpId; c < smp.C_nStride; c += smp.warpsInBlock) {
        int i8 = (warpThreadIdx * typ128::size) + (c * smp.wrap_stride);
        if (i8 >= ldTH) {
            break;
        }
        typ128 dW128 = typ128::load(dW + i8);
        dW128.AddFloat(dweight_shared + i8);
        dW128.store(dW + i8);
    } */
}

/*
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void rowSumFast(const float* __restrict__ matrix,
                          float* __restrict__ rowSums,
                          int rows, int cols) {

    __shared__ float block_sum[BLOCK_SIZE_X][BLOCK_SIZE_Y];

    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

    if (row >= rows) return;

    float thread_sum = 0.0f;

    // Each thread processes a single element
    if (col < cols) {
        thread_sum = matrix[row * cols + col];
    }

    block_sum[threadIdx.x][threadIdx.y] = thread_sum;
    __syncthreads();

    // Reduce within warp (warp-level reduction)
    for (int stride = BLOCK_SIZE_X / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            block_sum[threadIdx.x][threadIdx.y] +=
                block_sum[threadIdx.x + stride][threadIdx.y];
        }
        __syncthreads();
    }

    // Write result
    if (threadIdx.x == 0 && row < rows) {
        atomicAdd(&rowSums[row], block_sum[0][threadIdx.y]);
    }
}*/

/**
 *  lite version: each thread for one row/head, No need sync!
 *  1. Y = x/(RMS(x)+ϵ)⊙w for each token/head in the forward pass
 *  2. dX0 maybe same as dY0
 */
template <typename Typ>
__global__ void CU_rms_backward_v0(Typ* dX0, Typ* dWeight0, float* dW_scratch, const Typ* dY0, const Typ* X0, const Typ* weight0, const float* rstd0, int nTH,
                                   int ldTH, unsigned int seed, int flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTH) {
        return;
    }
    const Typ *w = weight0, *x = X0 + idx * ldTH, *dY = dY0 + idx * ldTH;
    float delta_avg = 0, rstd = rstd0[idx], acc = 0.f;
    for (int i = 0; i < ldTH; i++) {
        Typ xi = x[i] * CU_Float2T<Typ>(rstd, seed);
        // Typ dw = dY[i] * xi;  // It is a total gradient, not an average gradient
        dW_scratch[idx * ldTH + i] = (float)x[i] * rstd * (float)dY[i];
        // delta_avg += (float)(dY[i] * w[i] * xi);
        delta_avg += (float)w[i] * (float)dY[i] * (float)x[i];
    }
    delta_avg = delta_avg / ldTH * rstd;
    Typ* dX   = dX0 + idx * ldTH;
    for (int i = 0; i < ldTH; i++) {
        dX[i] = rstd * ((float)(dY[i] * w[i]) - (float)(x[i]) * rstd * delta_avg);
    }
    // if (idx == 0) {  //  only for debug
    //     printf("delta_avg=%g, rstd=%g dX=%g,%g,%g,%g\n", delta_avg, rstd, (float)dX[0], (float)dX[1], (float)dX[2], (float)dX[3]);
    // }
}

hGTensor LayerNormal::cuFlow(hGTensor inpDelta, int flag) {  //,hGTensor deltaIn
    NVTX_RANGE_FN();
    const int block_size = 256, block_y = block_size / WARP_SIZE, grid_size = CEIL_DIV(nTH, block_y);
    int nThread    = X128::nThreadOfBlock(ldTH, 0);
    floatX *weight = ToX(w), *bias = ToX0(b), *in = ToX(inpDelta);
    if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE) && nHead == 0) {  //
        CU_rms_infer(ToX(out), ToX(inpDelta), weight, C);
        // CU_rms_infer(ToX(inpDelta), ToX(inpDelta), weight, C);
        return out;
    }
    float *_mean = mean == nullptr ? nullptr : TO<float>(mean), *_rstd = TO<float>(rstd);
    if (isForward() || BIT_TEST(flag, F_REMATER)) {
        SetInp4Back(inpDelta);  //        inp            = inpDelta;
        floatX* devOut = isOnline ? in : ToX(out);
        if (isOnline) {
            // inp->Print(name + (BIT_TEST(flag, F_REMATER)? "_remater" : "_forw"), 0x0, -1);
        }
        if (mean == nullptr) {                                    // RMS
            size_t smem = (1 + block_y) * ldTH * sizeof(floatX);  // 16128
            // auto status = cudaFuncSetAttribute(CU_rms_forward_v3<floatX, floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            // cudaCheck(cudaGetLastError());
            // CU_rms_forward_v3<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _rstd, in, weight, nTH, ldTH, rms_eps);
            // inpDelta->Print(inpDelta->name, 0, -1, ldTH);
            CU_rms_forward_v2<<<nTH, nThread, 0x0, main_stream>>>(devOut, _rstd, in, weight, nTH, ldTH, rms_eps);
            // out->Print(out->name, 0, -1), rstd->Print(rstd->name, 0, -1);
        } else {
            size_t smem = (2 + block_y) * C * sizeof(floatX);
            auto status = cudaFuncSetAttribute(CU_lm_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaCheck(cudaGetLastError());
            assert(status == cudaSuccess);
            CU_lm_forward<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, nTH, ldTH);
            //     // fall back to the version without shared memory
            //     const int grid_size_fb = CEIL_DIV(nTH, (block_size / WARP_SIZE));
            //     layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, nTH, ldTH);
            // }
        }
        cudaCheck(cudaGetLastError());
        return out;
    } else {
        hGTensor deltaIn = inpDelta;
        assert(deltaIn != nullptr);
        float* dW_scratch = (float*)GTensor::buff;
        assert(nTH * ldTH * sizeof(float) <= GTensor::buff_len);
        // deltaIn->Print("LN.delta.in", 0, 0);
        VerifyInp4Back(inp);
        if (mean == nullptr) {
            TASKA_SWMD<floatX> smp(nTH, ldTH, deviceProp.multiProcessorCount, 0x100);  // RMS

            if (isOnline) { /*some bug in CU_rms_backward_v1_ & CU_rms_back_llmc, very hard to debugg!    */
                hGTensor deltaY = isOnline ? deltaIn : delta;
                // deltaIn->Print(deltaIn->name, 0, -1, nTH * ldTH);
                if (ver_rms_qknormal_ == 0) {
                    CU_rms_backward_v0<<<CEIL_DIV(nTH, block_size), block_size, 0x0, main_stream>>>(ToX(deltaY), ToG(w), dW_scratch, ToX(deltaIn), ToX(inp),
                                                                                                    ToX(w), _rstd, nTH, ldTH, 42, 0x0);
                    CU_rms_dw_v0<<<1, ldTH, 0x0, main_stream>>>(ToG(w), dW_scratch, nTH, ldTH);
                } else {
                    SWMD(CU_rms_backward_v1)(ToX(deltaY), ToG(w), dW_scratch, ToX(deltaIn), ToX(inp), ToX(w), _rstd, smp, nTH, ldTH, 4, 0x200);
                    CU_rms_dw_v1<<<1, ldTH / f128::size, smp.smem, main_stream>>>(ToG(w), dW_scratch, smp, ldTH);
                    // CU_rms_dw_v0<<<1, ldTH, 0x0, main_stream>>>(ToG(w), dW_scratch, smp.grid3, ldTH, 1);
                }
                // deltaY->Print(delta->name, 0, -1, nTH * ldTH), w->Print(w->name, 1, -1);
                assert(GTensor::buff_len >= nTH * ldTH);
            } else {
                auto status = cudaFuncSetAttribute(CU_rms_back_llmc<floatX>, cudaFuncAttributeMaxDynamicSharedMemorySize, smp.smem);
                cudaCheck(cudaGetLastError());
                //  dinp, dweight, dW_scratch, dY0, inp, weight,
                if (0) {  // DEBUG.cmd_p1
                    CU_rms_back_llmc<<<smp.grid3, smp.block3, smp.smem, main_stream>>>(ToX(delta), ToG(w), (hBITARR)(dW_scratch), ToX(deltaIn), ToX(inp),
                                                                                       ToX(w), _rstd, nullptr, nTH, ldTH);
                } else {
                    SWMD(CU_rms_backward_v1)(ToX(delta), ToG(w), dW_scratch, ToX(deltaIn), ToX(inp), ToX(w), _rstd, smp, nTH, ldTH, 42, 0x0);
                    CU_rms_dw_v1<<<1, ldTH / f128::size, smp.smem, main_stream>>>(ToG(w), dW_scratch, smp, ldTH);
                    // CU_rms_dw_v0<<<1, ldTH, 0x0, main_stream>>>(ToG(w), dW_scratch, smp.grid3, ldTH, 1);
                }
                // delta->Print(delta->name, 0, -1), w->Print(w->name, 1, -1);
            }
        } else
            layernorm_backward(ToX(delta), ToG(w), ToG0(b), dW_scratch, ToX(deltaIn), ToX(inp), ToX(w), _mean, _rstd, B, T, C, main_stream);
        delta->Print("back of normal", 0, 0);
        return delta;
    }
}
