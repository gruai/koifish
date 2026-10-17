/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some codes are from llm.c
 *
 *  \brief cuda kernel of QKV self attention
 *  \author Yingshi Chen
 */

// #include <catch2/catch_test_macros.hpp>
#include "../../Manifold/Neuron.hpp"
#include "../../Shoal/Fish.hpp"
#include "./cuda_common.h"
// #include "./kernel/gelu.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"

#define NOMINMAX

static cudaEvent_t cuStart, cuEnd;

/*
void CUBIN_1()  {
    CUmodule module;
    cuModuleLoad(&module, "/home/cys/rnd/lic/src/Device/tile_kernel/flash_bwd.cubin");
    // Get kernel function
    CUfunction kernel;
    CUresult err = cuModuleGetFunction(&kernel, module, "flash_bwd_kernel");
    assert(err==CUDA_SUCCESS);

    // Prepare data
    constexpr int N = 1024;
    size_t bytes = N * sizeof(float);

    float h_A[N], h_B[N], h_C[N];
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // CUdeviceptr d_A, d_B, d_C;
    // cuMemAlloc(&d_A, bytes);
    // cuMemAlloc(&d_B, bytes);
    // cuMemAlloc(&d_C, bytes);

    // cuMemcpyHtoD(d_A, h_A, bytes);
    // cuMemcpyHtoD(d_B, h_B, bytes);

    // Launch config
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // void* args[] = { &d_A, &d_B, &d_C, &N };

    // CHECK_CUDA(
    //     cuLaunchKernel(
    //         kernel,
    //         blocks, 1, 1,          // grid
    //         threadsPerBlock, 1, 1, // block
    //         0, nullptr, args, nullptr
    //     )
    // );
    //  heads, tl.ceildiv(seq_len, block_M), batch, threads=threads:
        //  const float* Delta, const *K, const *Q, const *V, float* dK, const *dO, float* dQ, float* dV, const float* lse
        // cudaFuncSetAttribute(    flash_bwd_T64_16_S49152_bfloat16,    cudaFuncAttributeMaxDynamicSharedMemorySize,    0   );
        // flash_bwd_T64_16_S49152_bfloat16<<<dim3(n_head, T / 64, B), dim3(128, 1, 1), 49152, main_stream>>>(
        //     deltaOdO, (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, deltaQ32, dV_group, devLse
        // //     deltaOdO, (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, dV_group, devLse
        //     // (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, dV_group, devLse
        // );

    // cuCtxSynchronize();
}*/
//

inline int convert_SM_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
                                       {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
                                       {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    _INFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor, nGpuArchCoresPerSM[index - 1].cores);
    return nGpuArchCoresPerSM[index - 1].cores;
}

bool InitCUDNN();

bool InitCUDA(const CLI_params& hparams, EDGE_DEVICES* hDevice, int flag) {
    int local_device_idx = 0;  // override_enable_tf32 = 1;
    cudaError_t err      = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[InitCUDA] failed at cudaSetDevice! ERR=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaGetDeviceProperties(&deviceProp, local_device_idx));
    // int cuda_num_SMs = deviceProp.multiProcessorCount;
    // int cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    // int cuda_arch_major = deviceProp.major;
    // int cuda_arch_minor = deviceProp.minor;

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    // nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cublasCheck(cublasCreate(&cublas_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    // bool enable_tf32 = PARAMS_TYPE == typNUMBER::F32 && deviceProp.major >= 8 && override_enable_tf32;
    // cublas_compute   = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
#ifdef __USE_CUDNN__
    InitCUDNN();
#endif
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    int nCorePP = convert_SM_to_cores(deviceProp.major, deviceProp.minor);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    _INFO("\t君子不器 - \"%s\" \n", deviceProp.name);
    _INFO("\tCUDA driver version: %d.%d, %s runtime version: %d.%d %s\n", driver_version / 1000, (driver_version % 100) / 10, COLOR_ORANGE,
          runtime_version / 1000, (runtime_version % 100) / 10, COLOR_RESET);
    _INFO("\tCUDA capability major/minor version number: %d.%d. ECC=%d\n", deviceProp.major, deviceProp.minor, deviceProp.ECCEnabled);
    _INFO("\t\n");
    _INFO("\t%d multiprocessors, %d CUDA cores/MP, %d CUDA cores\n", deviceProp.multiProcessorCount, nCorePP, nCorePP * deviceProp.multiProcessorCount);

    _INFO("\tGPU max clock rate: %.0f MHz (%0.2f GHz)\n", static_cast<double>(deviceProp.clockRate) * 1e-3, static_cast<double>(deviceProp.clockRate) * 1e-6);
    _INFO("\tPeak bandwidth %.1f GByte/s.\tMemory clock rate: %.0f MHz (%0.2f GHz)\n",
          (double)deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 2 / 1e6, static_cast<double>(deviceProp.memoryClockRate) * 1e-3,
          static_cast<double>(deviceProp.memoryClockRate) * 1e-6);
    _INFO("\tMemory bus width: %d-bit\n", deviceProp.memoryBusWidth);
    _INFO("\tGlobal memory: %.0f MBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalGlobalMem) / 1048576, deviceProp.totalGlobalMem);
    _INFO("\tConstant memory: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalConstMem) / 1024, deviceProp.totalConstMem);
    _INFO("\tShared memory per block: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerBlock) / 1024, deviceProp.sharedMemPerBlock);
    _INFO("\tShared memory per multiprocessor: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerMultiprocessor) / 1024,
          deviceProp.sharedMemPerMultiprocessor);
    _INFO("\tL2 cache size: %.0f KBytes (%d Bytes)\n", static_cast<double>(deviceProp.l2CacheSize) / 1024, deviceProp.l2CacheSize);
    _INFO("\tTotal number of registers available per block: %d\n", deviceProp.regsPerBlock);
    _INFO("\tWarp size: %d, Max number of threads per block: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock);
    _INFO("\tMax number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    _INFO("\tMax dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
    _INFO("\tMax dimension size of a grid size (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    _INFO("+-----------------------+----------------------------------------------------+\n");

    fflush(stdout);
    // Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state
    // available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors.
    hDevice->nCore = deviceProp.multiProcessorCount;
    if (0) {
        cudaCheck(cudaEventCreate(&cuStart));
        cudaCheck(cudaEventCreate(&cuEnd));
        //  https://github.com/jonasmr/microprofile
        cudaCheck(cudaProfilerStart());
    }

    return true;
}

/*
    Even if my program uses only one stream, cudaDeviceSynchronize()may still wait for GPU work launched by other programs.
      cudaStreamSynchronize(stream)/cudaEventSynchronize(event) maybe better

      1. cudaEventSynchronize
        cudaEventSynchronizeis a CUDA function that blocks the host (CPU) until a CUDA event is completed. It's crucial for proper timing, synchronization, and
   debugging in CUDA applications.
      2. cudaEventRecord
        cudaEventRecordis a CUDA function that records a CUDA event at a specific point in a CUDA stream. It's used for timing, synchronization, and
   establishing dependencies between operations.
*/
bool SYNC_DEVICE_heavy(const std::string& sX, int flag) {
#ifdef __USE_CUDA__
    if (main_stream != nullptr) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            _INFO("_SYNC_DEVICE_ err=\"%s\" (%s code=%d)\t%s\n", cudaGetErrorString(error), cudaGetErrorName(error), error, sX.c_str());
            if (flag == 1)
                return false;
            assert(0 && "_SYNC_DEVICE_");
            exit(KOIFISH_EXIT_SYNC_DEVICE);
        }
        return true;
    }
#endif
    return true;
}
bool SYNC_STREAM(const std::string& sX, int flag) {
#ifdef __USE_CUDA__
    if (main_stream != nullptr) {
        cudaError_t error = cudaStreamSynchronize(main_stream);
        if (error != cudaSuccess) {
            _INFO("SYNC_STREAM err=\"%s\" (%s code=%d)\t%s\n", cudaGetErrorString(error), cudaGetErrorName(error), error, sX.c_str());
            if (flag == 1)
                return false;
            assert(0 && "_SYNC_DEVICE_");
            exit(KOIFISH_EXIT_SYNC_STREAM);
        }
        return true;
    }
#endif
    return true;
}

hGTensor SelfAttention::cuInfer(hGTensor inpL, int flag) {
    if (G_Has_(name, {"layers.17.self_attn"})) {
        DEBUG_HERE;  // 74200000
    }
#ifdef USE_FP8_BASELINE
    assert(0 && "FP8 baseline is not implemented yet");
    return nullptr;
#else
    assert(hCache != nullptr && isSeparateQKV);
    hBATCH hBatch = hFish->curBatch(0x0);
    int pos       = hBatch->tok_pos;
    if (pos >= 0)
        _devQKV(pos);  // update k,v
    // // hCache:   (layer, seq_len, kv_dim)
    floatX *key_cache = (floatX*)hCache->Get(KVCache::KV_KEY, layid - 1, 0), *val_cache = (floatX*)hCache->Get(KVCache::KV_VAL, layid - 1, 0);

    hGTensor tmpQKV = gBUFF->tmpFF1;
    floatX* qkvr    = ToX(tmpQKV);  // Q.out/K.out/V.out
    int nToken = nBatchToken(), seq_len = hFish->curChatLen(LIMIT), nEmbed = hFish->config.nEmbed();
    inp = OnInput(inpL);  //  may remater by hIn->SerialData_
    inp->Print("inp", 0x0, dump_flag, nToken * nEmbed);
    gBUFF->residual = inp;
    hGTensor inpQ   = inpL;
    if (fuseNorm == nullptr) {
        inpQ = norm.cuFlow(inpL);
        inpQ->Print("qkv.in", 0x0, dump_flag, nToken * nEmbed);
        // norm.w->Print("qkvn.w",0x0,dump_flag);          norm.b->Print("qkvn.b",0x0,dump_flag);
        // norm.mean->Print("qkvn.mean",0x0,dump_flag);       norm.rstd->Print("qkvn.rstd",0x0,dump_flag);
    }

    inpQ->Quant4A(DEBUG.tpActi);
    if (isSeparateQKV) {
        Q.Forw(Q.out, inpQ);
        // Q.out->Print("Q.out", 0x0, dump_flag, nToken * q_dim);
        K.Forw(K.out, inpQ);
        V.Forw(V.out, inpQ);
    } else {
        Q.Forw(tmpQKV, inpQ);
        // Q.w->Print("Q.w",0x0,dump_flag);     Q.b->Print("Q.b",0x0,dump_flag);
    }
    // tmpQKV->Print("QKV",0x0,dump_flag);
    Q.out->Print("Q.out", 0x0, dump_flag, nToken * q_dim), K.out->Print("K.out", 0x0, dump_flag, nToken * kv_dim),
        V.out->Print("V.out", 0x0, dump_flag, nToken * kv_dim);

    if (rope != nullptr) {
        rope->cuInfer(this, rope_seed, pos);
    }
    if (DEBUG.T_kvcache_quant == 1) {
        // size_t smemPB = head_dim * sizeof(float);
        // CU_X2ternary_multihead<floatX><<<n_head_kv, head_dim, smemPB, main_stream>>>(nullptr, ToX(K.out), (char*)key_cache, n_head_kv, head_dim, true);
    }
    if (1) {
        int qk_threads_per_block = std::min(1024, pos + 1);
        attention_qk_kernel<<<n_head, qk_threads_per_block>>>(ToX(qk_v), ToX(Q.out), key_cache, pos, seq_len, n_head, n_head_kv, head_dim);
        CU_softmax_multihead<<<n_head, 1>>>(ToX(qk_v), pos, seq_len);
        qk_v->Print("qk_v", 0x0, dump_flag, pos + 1);
        attention_v_kernel<<<n_head, head_dim>>>(ToX(Q.out), ToX(qk_v), val_cache, pos, seq_len, n_head, n_head_kv, head_dim);
        Q.out->Print("att_out", 0x0, dump_flag, nToken * q_dim);
    } else {
        // FUSE_qkv(nullptr, nullptr, flag);
        qk_v->Print("l_atty", 0x0, dump_flag);
    }
    if (0)  // ShareLayerOut: inpL/out is same data
        ;   // CU_mv_(ToX(inpL), ToX(proj_cat.w), ToX(Q.out), C, q_dim, 1.0f, 1.0f);
    else {  //       From cuFlow
        proj_cat.Forw(gBUFF->scratch, Q.out);
        gBUFF->scratch->Print("proj_cat", 0x0, dump_flag, nToken * nEmbed);
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        if (!hFish->isRemater()) {
            gBUFF->residual->Print("residual", 0x0, dump_flag, nToken * nEmbed);
            TASKA_1p1<floatX> task_11(nToken * nEmbed, main_stream);
            T1p1(CU_add3<floatX>, task_11, ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), 0x0);
            // residual_forward(ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), nToken * nEmbed, main_stream);
            assert(fuseNorm == nullptr && "Try fuse normal later...");
            {
                // float *mean = TO<float>(fuseNorm->mean), *rstd = TO<float>(fuseNorm->rstd);
                // layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out), ToX(fuseNorm->w), ToX0(fuseNorm->b), nToken, 1, C, main_stream);
            }

        } else {
        }
    }
    out->Print("qkv.out", 0x0, 0, nToken * nEmbed);
    return out;
#endif
};

/*
    Forward:    cur = cuFlow(cur,residual,flag);
    Backward:   QKV->cuFlow(last->out,QKV->norm.out,0x0);
*/
hGTensor SelfAttention::cuFlow(hGTensor inpL, int flag) {
    // NVTX_RANGE_FN();
    hGTensor tmpQKV = gBUFF->tmpFF1;
    floatX* qkvr    = ToX(tmpQKV);  // Q.out/K.out/V.out
    int nEmbed      = hFish->config.nEmbed();
    // bool isAlternate = true;                   // layer%2==1;layer>1;
    _devQKV(0);
    if (isForward()) {  //  data=ToX(QKV->norm.out)
        // NvtxRange range(name.c_str(), 0);
        inp             = OnInput(inpL);  //         inp->Print("inp",0x0,dump_flag);
        gBUFF->residual = inp;            // gBUFF->residual->OverWrite(inp);
        hGTensor inpQ   = inpL;
        if (fuseNorm == nullptr) {
            inpQ = norm.cuFlow(inpL);
            inpQ->Print("qkvn.in", 0x0, dump_flag);
            // norm.w->Print("qkvn.w",0x0,dump_flag);         if(norm.b!=nullptr)  norm.b->Print("qkvn.b",0x0,dump_flag);
            // norm.mean->Print("qkvn.mean",0x0,dump_flag);       norm.rstd->Print("qkvn.rstd",0x0,dump_flag);
        }
        if (isSeparateQKV) {
            Q.Forw(Q.out, inpQ);
            K.Forw(K.out, inpQ);
            V.Forw(V.out, inpQ);
        } else {
            Q.Forw(tmpQKV, inpQ);
            // Q.w->Print("Q.w",0x0,dump_flag);     Q.b->Print("Q.b",0x0,dump_flag);
        }
        Q.out->Print("Q.out", 0x0, dump_flag);
        K.out->Print("K.out", 0x0, dump_flag), V.out->Print("V.out", 0x0, dump_flag);
        // INSPECT_THIS;
        if (rope != nullptr) {
            rope->cuFlow(this, rope_seed);
        }
        // GTensor::tZ->Print(GTensor::tZ->name, 0, dump_flag);
        FUSE_qkv(nullptr, nullptr, flag);
        // proj_cat.w->Print(proj_cat.w->name, 0, dump_flag);
        proj_cat.Forw(gBUFF->scratch, qk_v);  // fuMM(scratch, ToX(qk_v), pw, pb, B, T, C, C, main_stream);
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        if (!hFish->isRemater()) {
            gBUFF->scratch->Print("proj_out", 0x0, dump_flag, B * T * nEmbed);
            gBUFF->residual->Print("residual", 0x0, dump_flag, B * T * nEmbed);
            TASKA_1p1<floatX> task_11(B * T * nEmbed, main_stream);
            T1p1(CU_add3<floatX>, task_11, ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), 0x0);
            // residual_forward(ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), B * T * nEmbed, main_stream);
            assert(fuseNorm == nullptr && "Try fuse normal later...");
            // if (fuseNorm != nullptr) {
            //     float *mean = TO<float>(fuseNorm->mean), *rstd = TO<float>(fuseNorm->rstd);
            //     layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out), ToX(fuseNorm->w), ToX0(fuseNorm->b), B * T, 1, C, main_stream);
            // }
        } else {
        }
        SYNC_STREAM();
        return out;
    } else {  //  Backward
        // NvtxRange range(name.c_str(), 1);
        INSPECT_THIS;
        // Q.w->Print("Qw", 1, dump_flag);
        assert(inpL == gBUFF->delta);
        delta->Print("deltaQKV", 0x0, dump_flag);
        proj_cat.w->Print("proj_cat.w", 0x0, dump_flag);
        proj_cat.Back(gBUFF->tmpDelta, qk_v, gBUFF->delta, nullptr);

        hGTensor delta_qkv = gBUFF->bt4c;
        if (hFish->config.scheduling.strategy != MEM_STRATEGY::MEM_SWAP_GUOKE) {  //    remater_qkv
            hGTensor norm_out = norm.out;
            if (isSeparateQKV) {
                Q.Forw(Q.out, norm_out), K.Forw(K.out, norm_out), V.Forw(V.out, norm_out);
                if (gBUFF->tmpQout != nullptr) {
                    gBUFF->tmpQout->OverWrite(Q.out);
                    gBUFF->tmpKout->OverWrite(K.out);
                }
            } else {
                Q.Forw(tmpQKV, norm_out);
            }
            if (rope != nullptr)
                rope->cuFlow(this, rope_seed, true, F_REMATER);
        }
        gBUFF->tmpDelta->Print("delta_cat", 0x0, dump_flag);
        FUSE_qkv(ToX(delta_qkv), ToX(gBUFF->tmpDelta), flag);
        if (rope != nullptr) {
            if (gBUFF->tmpQout != nullptr) {  //
                Q.out->OverWrite(gBUFF->tmpQout), K.out->OverWrite(gBUFF->tmpKout);
            } else if (rope->hnQ != nullptr) {
                Q.Forw(Q.out, norm.out), K.Forw(K.out, norm.out);
            }
            rope->cuFlow(this, rope_seed);
            // Q.out->Print("Q.rope",0x0,dump_flag);    K.out->Print("K.rope",0x0,dump_flag);
        }
        // float* scratchF = (float*)GTensor::buff;
        if (isSeparateQKV) {
            // Q.w->Print("Qw", 0, dump_flag);  Q.b->Print("Qb", 0, dump_flag);   norm.out->Print("norm.out", 0, dump_flag);
            Q.Back(gBUFF->tmpDelta, norm.out, deltaQ, nullptr);
            // matmul_backward(ToX(gBUFF->tmpDelta), ToG(Q.w), ToG0(Q.b), (floatX*)devDeltaQ, ToX(norm.out), Q.w->GetDataX(), scratchF, B, T, Q.nOut, Q.nIn,
            //                 main_stream, false, NULL, false);
            // gBUFF->tmpDelta->Print("delta_0", 0, dump_flag);
            K.Back(gBUFF->tmpDelta, norm.out, deltaK, nullptr, true);
            // matmul_backward(ToX(gBUFF->tmpDelta), ToG(K.w), ToG0(K.b), (floatX*)devDeltaK, ToX(norm.out), K.w->GetDataX(), scratchF, B, T, K.nIn, K.nOut,
            // main_stream, false, NULL, true);
            // gBUFF->tmpDelta->Print("delta_1", 0, dump_flag);
            // K.w->Print("Kw", 1, dump_flag);
            V.Back(gBUFF->tmpDelta, norm.out, deltaV, nullptr, true);
            // matmul_backward(ToX(gBUFF->tmpDelta), ToG(V.w), ToG0(V.b), (floatX*)devDeltaV, ToX(norm.out), V.w->GetDataX(), scratchF, B, T, V.nIn, V.nOut,
            //                 main_stream, false, NULL, true);
            // gBUFF->tmpDelta->Print("delta_2", 0, dump_flag);
            // V.w->Print("Vw", 1, dump_flag);
        } else {
            Q.Back(gBUFF->tmpDelta, norm.out, deltaQ, nullptr);
            // matmul_backward(ToX(gBUFF->tmpDelta), ToG(Q.w), ToG0(Q.b), ToX(delta_qkv), ToX(norm.out), ToX(Q.w), scratchF, B, T, C_qkv, 3 * C_qkv,
            // main_stream);
            gBUFF->tmpDelta->Print("delta_3", 0, dump_flag);
        }

        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        gBUFF->tmpDelta->Print("DLN1", 0, dump_flag);
        norm.cuFlow(gBUFF->tmpDelta);  // would backpropagatioin to gBUFF->delta
        gBUFF->delta->Print("back of QKV", 0, dump_flag);
        SYNC_STREAM();
        return gBUFF->delta;
    }
    return nullptr;
}
