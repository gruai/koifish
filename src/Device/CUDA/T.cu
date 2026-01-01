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
#include "./kernel/gelu.cuh"
#include "./kernel/operator.cuh"

extern cudaStream_t main_stream;

int Relu::Forw(hGTensor out, hGTensor inp, int flag) {
    size_t nz            = SHAPE2NZ(shape);
    const int block_size = 128;
    const int grid_size  = CEIL_DIV(nz, block_size);
    hGTensor gate        = nullptr;
    switch (fAct) {
        case SWIG:            
            assert(slp_gate != nullptr && slp_gate->tRhs!=nullptr);
            gate = slp_gate->tRhs;
            gate->Print("swig.gate", 0, dump_flag, C);
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                assert(C % x128::size == 0);
                assert((B * T * C) % (block_size * x128::size) == 0);
                const int num_blocks = CEIL_DIV(B * T * C, (int)(block_size * x128::size));
                assert(gate != nullptr);
                // CU_swiglu_v1<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), C);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            }
            out->Print("ffn.swig", 0, dump_flag, B * T * C);
            break;
        case GELU:
            gelu_forward(ToX(out), ToX(inp), nz, main_stream);
            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

template <typename T>
__global__ static void CU_swiglu_back_v0(T* delta_in_out, T* delta_gate, const T* gate, const T* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xiW   = CU_T2Float(gate + idx);
        float xiV   = CU_T2Float(inp + idx);
        float delta = delta_in_out[idx];
        // if(idx==0)    // only for debug
        // {    nout("nout<%d>: gate=%g ffn.up=%g delta=%g\n", idx, xiW,xiV,delta);    }
        float sigW = 1.0f / (1.0f + expf(-xiW));
        delta_gate[idx] = delta * xiV * sigW * (1 + xiW * (1.0f - sigW));

        delta_in_out[idx] = delta * xiW * sigW;  //  delta * swish_out[i];
    }
}

//  delta is both delta_in & delta_out
int Relu::Back(hGTensor delta_in_out, hGTensor pre_gelu, int flag) {
    size_t nz            = SHAPE2NZ(shape);
    const int block_size = 128;
    const int grid_size  = CEIL_DIV(nz, block_size);
    hGTensor gate        = nullptr;
    switch (fAct) {
        case SWIG:
            // dump_flag = hFish->isModel({NLP_QWEN2}) ? -1 : 0;
            assert(slp_gate != nullptr);
            gate = slp_gate->tRhs;
            pre_gelu->Print("ffn.up", 0, dump_flag, B * T * C);
            gate->Print("swig.gate", 0, dump_flag, C);      //-1.890625
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                assert(C % x128::size == 0);
                assert((B * T * C) % (block_size * x128::size) == 0);
                const int num_blocks = CEIL_DIV(B * T * C, (int)(block_size * x128::size));
                assert(gate != nullptr && slp_gate != nullptr);
                CU_swiglu_back_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(delta_in_out), ToX(slp_gate->delta), ToX(gate), ToX(pre_gelu), nz);
            }
            delta_in_out->Print("dUp", 0, dump_flag, C);
            slp_gate->delta->Print("dGate", 0, dump_flag, C);
            break;
        case GELU:
            //  gelu_backward_inplace fused @matmul_backward
            gelu_backward_inplace(ToX(delta_in_out), ToX(pre_gelu), nz, main_stream);

            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

__global__ void CU_rms_forward_v0(bf16* __restrict__ out, float* __restrict__ rstd, const bf16* __restrict__ inp, const bf16* __restrict__ weight, int N, int C,
                                  float eps = 1e-5f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    inp += idx * C;
    out += idx * C;
    float acc = 0.f;
    for (int c = 0; c < C; c++) {
        float a = (float)inp[c];
        acc += a * a;
    }
    float s = rsqrtf(acc / C + eps);
    assert(!isnan(s) && !isinf(s));
    for (int c = 0; c < C; c++) {
        float n = s * (float)inp[c];  // normalized output
        if (idx == 0 && c == 0) {
            DEBUG_HERE;
        }
        out[c] = (bf16)n * (bf16)weight[c];  // scale
    }
    rstd[idx] = s;
}