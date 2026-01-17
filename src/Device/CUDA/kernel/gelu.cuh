/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "../cuda_common.h"
#include "utils.cuh"

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

void inline gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = CU_T4B_MIDDLE;
    assert(N % (block_size * X128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * X128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void inline gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
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
 * inp = x*W,    gate = x*V
 *
 * activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
 */
template <typename T>
__global__ static void CU_swiglu_v0(T* out, const T* gate, const T* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xiW = CU_T2Float(gate + i);
        float xiV = CU_T2Float(inp + i);
        out[i]    = (T)((xiW * xiV) / (1.0f + expf(-xiW)));
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

template <typename T>
void inline swiglu_forward(T* out, const T* inp, const T* gate, int N, cudaStream_t stream = 0x0) {
    const int block_size = 128;
    const int grid_size  = CEIL_DIV(N, block_size);
    // CU_swiglu_v1<<<grid_size, block_size, 0, stream>>>(out, inp, gate, N);
    CU_swiglu_v0<<<grid_size, block_size, 0, stream>>>(out, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// kernel launcher

// void inline swiglu_backward(float* dinp, const float* inp, const float* gate, const float* dout, const float* W, const float* V, int N) {
//     const int block_size = 128;
//     const int grid_size  = CEIL_DIV(N, block_size);
//     swiglu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, gate, dout, W, V, N);
//     cudaCheck(cudaGetLastError());
// }
