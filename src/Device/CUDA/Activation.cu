/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Cu kernels of activations
 *
 *  \author Yingshi Chen
 */

#include <assert.h>

#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "./cuda_common.h"
#include "./kernel/packedN.cuh"
#include "./kernel/utils.cuh"
// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void static silu_forward_kernel2(floatX* out, const floatX* inp) {}
__global__ void static silu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
/**
 * Gaussian Error Linear Unit.  GELU(x)=xG(x), where G(x) is the standard Gaussian cumulative distribution function.
 */
__global__ void static gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * X128::size;

    X128 packed_out;
    X128 packed_inp = load128cs(inp + idx);  // load and do not keep in cache
    for (int k = 0; k < packed_inp.size; ++k) {
        float xi      = (float)packed_inp[k];
        float cube    = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void static gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * X128::size;

    X128 packed_dinp;
    X128 packed_inp  = load128cs(inp + idx);
    X128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x          = (float)packed_inp[k];
        float cube       = 0.044715f * x * x * x;
        float tanh_arg   = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out   = tanhf(tanh_arg);
        float coshf_out  = coshf(tanh_arg);
        float sech_out   = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k]   = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

void inline Activation_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = CU_T4B_MIDDLE;
    assert(N % (block_size * X128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * X128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void inline Activation_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * X128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * X128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}

/**
 * SwiGLU(x) = Swish(x') * Gate(x') = Swish(x*W) * (x*V)
 *      swish(x) = x * sigmoid(x) which is equivalent to the Sigmoid Linear Unit or SiLU.
 *
 * activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
 */
template <typename T>
__global__ void CU_swiglu_v0(T* out, const T* gate, const T* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xiW = CU_T2Float(gate + i);
        float xiV = CU_T2Float(inp + i);
        out[i]    = (T)((xiW * xiV) / (1.0f + expf(-xiW)));
    }
}

/**
 *  ReLU²(Gate(x)) ⊙ Up(x)
 *
 */
template <typename Typ>
__global__ void CU_glu2_forw_(TASKA_1p1<Typ> taska, Typ* out, const Typ* gate, const Typ* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx) {
        float xiW = CU_T2Float(gate + idx);
        float xiV = CU_T2Float(inp + idx);
        float g2  = xiW < 0.0 ? 0.0 : xiW * xiW;
        out[idx]  = (Typ)(g2 * xiV);
    }
}

/**
 * y=SiLU(xW)*(xV)
 * z=xW
 * g=xV
 *
 * Using Chain-Rule
 * ∂y/∂x = (σ(z) + z*σ(z)*(1−σ(z)))*g*W + SiLU(z)*V

template <typename T>
__global__ static void CU_swiglu_back_v0(T* dinp, const T* inp, const T* gate, const T* dout, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xW         = CU_T2Float(gate + i);    //(float)inp[i];
        float xV         = CU_T2Float(inp + i);     //(float)gate[i];
        float y          = xW / (1.0f + expf(-xW)) * xV;                // SwiGLU(x)
        float sig_xW     = 1.0f / (1.0f + expf(-xW));                   // Sigmoid(xW)
        float silu_prime = sig_xW + xW * sig_xW * (1.0f - sig_xW);      // SiLU'(xW)
        float grad_xW    = (silu_prime * xV) * dout[i];          // Gradient w.r.t. xW
        float grad_xV    = (xW / (1.0f + expf(-xW))) * dout[i];  // Gradient w.r.t. xV
        dinp[i]          = grad_xW + grad_xV;                           // Sum of gradients
    }
} */

template <typename Typ>
__global__ static void CU_swiglu_v1(Typ* out, const Typ* inp, const Typ* gate, int C) {
    using x128 = PackedN<Typ, 16 / sizeof(Typ)>;

    // thread coordinates
    int idx         = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    floatX* out_ptr = out + idx;
    int bt          = (idx / C);
    int c           = idx % C;

    const floatX* up_ptr   = inp + (bt * C * 2 + c);
    const floatX* gate_ptr = gate + C;

    float thread_max = 0.f;

    x128 packed_out;
    x128 up_inp   = x128::load_cs(up_ptr);
    x128 gate_inp = x128::load_cs(gate_ptr);
    for (int k = 0; k < up_inp.size; ++k) {
        float x1      = (float)up_inp[k];
        float x2      = (float)gate_inp[k];
        packed_out[k] = (floatX)((x1 * x2) / (1.0f + expf(-x2)));
    }
    packed_out.store(out_ptr);

    // handle_absmax_reduction(stat_info, &block_max, thread_max);
}

int Relu::Forw(hGTensor out, hGTensor inp, int flag) {
    int nToken = nBatchToken(), C = hFish->config.n_ff();
    size_t nz  = nToken * C;  // SHAPE2NZ(shape);
    const int block_size = 128,grid_size = CEIL_DIV(nz, block_size);
    // assert(B * T * C == nz);
    hGTensor gate = nullptr;
    if (slp_gate != nullptr && slp_gate->tRhs != nullptr) {
        gate = slp_gate->tRhs;  // assert(gate != nullptr);
        gate->Print("swig.gate", 0, dump_flag, C);
    }
    TASKA_1p1<floatX> task_11(nz, main_stream, false);
    switch (fAct) {
        case GLU2:
            T1p1(CU_glu2_forw_<floatX>, task_11, ToX(out), ToX(gate), ToX(inp), nz);
            break;
        case SWIG:
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                // assert(C % X128::size == 0);
                // assert((nz) % (block_size * X128::size) == 0);
                // const int num_blocks = CEIL_DIV(nz, (int)(block_size * X128::size));
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
                // CU_swiglu_v1<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), C);
            }
            out->Print("ffn.swig", 0, dump_flag, nz);
            break;
        case GELU:
            Activation_forward(ToX(out), ToX(inp), nz, main_stream);
            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

template <typename T>
__global__ static void CU_swiglu_back(T* delta_in_out, T* delta_gate, const T* gate, const T* inp, int N) {
    using typ128 = PackedN<T, 16 / sizeof(T)>;
    using f256   = PackedN<float, typ128::size>;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }  // guard
    f256 xiW, xiV, delta, sigW;
    for (int k = 0; k < f256::size; k++) {
        // a = delta * xiV * sigW * (1 + xiW * (1.0f - sigW);
    }
}

template <typename Typ>
__global__ static void CU_swiglu_back_v0(TASKA_1p1<Typ> taska, Typ* delta_in_out, Typ* delta_gate, const Typ* gate, const Typ* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xiW   = CU_T2Float(gate + idx);
        float xiV   = CU_T2Float(inp + idx);
        float delta = CU_T2Float(delta_in_out + idx);
        // if(idx==0)    // only for debug
        // {    nout("nout<%d>: gate=%g ffn.up=%g delta=%g\n", idx, xiW,xiV,delta);    }
        float sigW      = 1.0f / (1.0f + expf(-xiW));
        delta_gate[idx] = CU_Float2T<Typ>(delta * xiV * sigW * (1 + xiW * (1.0f - sigW)), 42);

        delta_in_out[idx] = CU_Float2T<Typ>(delta * xiW * sigW, 42);  //  delta * swish_out[i];
    }
}

/**
 *  ReLU²(Gate(x)) ⊙ Up(x)
 *  d_up = delta * ReLU²(gate)
 *  d_gate = delta * up * 2 * ReLU(gate)
 */
template <typename Typ>
__global__ void CU_glu2_back_(TASKA_1p1<Typ> taska, Typ* delta_in_out, Typ* delta_gate, const Typ* gate, const Typ* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xiW   = CU_T2Float(gate + idx);
        float xiV   = CU_T2Float(inp + idx);
        float delta = CU_T2Float(delta_in_out + idx);
        float g     = xiW < 0.0 ? 0.0 : xiW;

        delta_gate[idx]   = CU_Float2T<Typ>(delta * xiV * 2 * g, 42);
        delta_in_out[idx] = CU_Float2T<Typ>(delta * g * g, 42);
    }
}

//  delta is both delta_in & delta_out
int Relu::Back(hGTensor delta_in_out, hGTensor pre_gelu, int flag) {
    size_t nz            = SHAPE2NZ(shape);
    const int block_size = 128, C = hFish->config.n_ff();
    assert(B * T * C == nz);
    const int grid_size = CEIL_DIV(nz, block_size);
    hGTensor gate       = nullptr;
    TASKA_1p1<floatX> task_11(nz, main_stream, false);
    if (slp_gate != nullptr) {
        gate = slp_gate->tRhs;
    }
    switch (fAct) {
        case SWIG:
            assert(slp_gate != nullptr);
            gate = slp_gate->tRhs;
            pre_gelu->Print("ffn.up", 0, dump_flag, nz);
            gate->Print("swig.gate", 0, dump_flag, C);  //-1.890625
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                assert(C % X128::size == 0);
                assert(nz % (block_size * X128::size) == 0);
                const int num_blocks = CEIL_DIV(nz, (int)(block_size * X128::size));
                assert(gate != nullptr && slp_gate != nullptr);
                // CU_swiglu_back_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(delta_in_out), ToX(slp_gate->delta), ToX(gate), ToX(pre_gelu), nz);
                T1p1(CU_swiglu_back_v0<floatX>, task_11, ToX(delta_in_out), ToX(slp_gate->delta), ToX(gate), ToX(pre_gelu), nz);
            }
            delta_in_out->Print("dUp", 0, dump_flag, C);
            slp_gate->delta->Print("dGate", 0, dump_flag, C);
            break;
        case GLU2:
            T1p1(CU_glu2_back_<floatX>, task_11, ToX(delta_in_out), ToX(slp_gate->delta), ToX(gate), ToX(pre_gelu), nz);
            break;
        case GELU:
            //  gelu_backward_inplace_ fused @matmul_backward_
            Activation_backward_inplace(ToX(delta_in_out), ToX(pre_gelu), nz, main_stream);

            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}
