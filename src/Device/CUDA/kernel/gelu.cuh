/*
(Approximate) GeLU non-linearity layer
*/
#include <assert.h>
// llmc internal imports
#include "../cuda_common.h"
#include "utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ void inline silu_forward_kernel2(floatX* out, const floatX* inp) {
    
}
__global__ void inline silu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
/**
 * Gaussian Error Linear Unit.  GELU(x)=xG(x), where G(x) is the standard Gaussian cumulative distribution function.
 */
__global__ void inline gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    // store instead of storecs (without cache streaming) in case it is useful for the
    // data to be in the cache for the next operation after this GeLU
    store128(out + idx, packed_out);
}

__global__ void inline gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp);
}

// ----------------------------------------------------------------------------
// kernel launchers

void inline gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

void inline gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}

__global__ inline void swiglu_forward_kernel(float *out, const float *inp, const float *gate, int N){
    /**
     * SwiGLU(x) = Swish(x) * Gate(x)
     * SwiGLU(x) = SiLU(x*W) * (x*V)
     * SiLU is the Swish activation function.
     * inp = x*W
     * gate = x*V
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xiW = inp[i];
        float xiV = gate[i];
        out[i] = (xiW / (1.0f + expf(-xiW))) * xiV;
    }
}

void inline swiglu_forward(float *out, const float *inp, const float *gate, int N){
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    swiglu_forward_kernel<<<grid_size, block_size>>>(out, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

__global__ inline void swiglu_backward_kernel(float *dinp, const float *inp, const float *gate, const float *dout, const float *W, const float *V, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)    {
        float xW = (float)inp[i];
        float xV = (float)gate[i];
        float y = xW / (1.0f + expf(-xW)) * xV;                     // SwiGLU(x)
        float sig_xW = 1.0f / (1.0f + expf(-xW));                   // Sigmoid(xW)
        float silu_prime = sig_xW + xW * sig_xW * (1.0f - sig_xW);  // SiLU'(xW)
        float grad_xW = (silu_prime * xV * W[i]) * dout[i];         // Gradient w.r.t. xW
        float grad_xV = (xW / (1.0f + expf(-xW))) * V[i] * dout[i]; // Gradient w.r.t. xV
        dinp[i] = grad_xW + grad_xV;                                // Sum of gradients
    }
}

// ----------------------------------------------------------------------------
// kernel launcher
/**
 * y=SiLU(xW)*(xV)
 * z=xW
 * g=xV
 *
 * Using Chain-Rule
 * ∂y/∂x = (σ(z) + z*σ(z)*(1−σ(z)))*g*W + SiLU(z)*V
 */
void inline swiglu_backward(float *dinp, const float *inp, const float *gate, const float *dout, const float *W, const float *V, int N){
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    swiglu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, gate, dout, W, V, N);
    cudaCheck(cudaGetLastError());
}
