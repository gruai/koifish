/*
Fused Classifier:
- Forwards the Cross Entropy Loss
- Never materializes the full normalized logits, only at the target label
- (fusion) Also kicks off the backward pass, because everything is already loaded
*/

#include "../cuda_common.h"
#include "utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// struct SoftmaxParams {
//     float Scale;
//     float Offset;
// };

// each block for one row of inp, i.e. inp[idx, :] of shape (V,)
__device__ inline SoftmaxParams prepare_softmax_blockwide3(const floatX* inp, int V) {
    // same but not float4
    const floatX* x     = inp;  // inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    int i               = (V + X128::size - 1) / X128::size + threadIdx.x - blockDim.x;

    // special-case loop to handle the unaligned elements at the end of the array
    // this lets us skip the bounds check in the main loop below, which improves performance
    while ((i + 1) * X128::size > V) {
        for (int k = 0; k < X128::size; ++k) {
            if (i * X128::size + k >= V) {
                break;  // bounds checking against real V (rather than padded P)
            }
            float v          = (float)x[i * X128::size + k];
            float old_maxval = thread_maxval;
            thread_maxval    = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
        i -= blockDim.x;
    }

    // main loop for the bulk of the iterations (no bounds checking required!)
    for (; i >= 0; i -= blockDim.x) {
        X128 packed_x = load128(x + i * X128::size);  // load and keep in cache until fused_classifier loop
        for (int k = 0; k < X128::size; ++k) {
            float v          = (float)packed_x[k];
            float old_maxval = thread_maxval;
            thread_maxval    = fmaxf(thread_maxval, v);
            thread_sumval *= expf((old_maxval - thread_maxval));
            thread_sumval += expf(v - thread_maxval);
        }
    }

    // Block Max Reduction -> Maths -> Block Sum Reduction
    float block_maxval = blockReduce_v0<warpReduceMax>(thread_maxval, false, -INFINITY);
    thread_sumval *= expf(thread_maxval - block_maxval);
    float block_sumval = blockReduce_v0<warpReduceSum>(thread_sumval);

    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// will _update_ logits to logit gradients
// uses template to decide whether to write logits and probs
// split both loops in "multiple-of-X128-size" and "bounds-checked remainder" parts
// template <bool WriteDLogits = true, bool WriteProbs = false>
__global__ void __launch_bounds__(1024, MAX_1024_THREADS_BLOCKS)
    fused_classifier_kernel5(floatX* logits, float* losses, floatX* probs, const float dloss, const int* targets, int B, int T, int V, int P,
                             bool WriteDLogits = true) {
    // note: idx is small enough that it easily fits into 32 bit;
    // by making it a long here, we ensure that any offsets calculated with it (e.g., idx * P)
    // are done is 64 bit
    int64_t idx              = gridDim.x - (blockIdx.x + 1);  // reverse order for cache hits on matmul data
    int ix                   = targets[idx];
    bool WriteProbs          = probs != nullptr;
    const floatX* logits_vec = logits + idx * P;
    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide3(logits_vec, V);
    // calculate the probability needed for the loss and update (single-threaded)
    if (threadIdx.x == 0) {
        float prob = expf((float)logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] -= logf(prob);
    }

    // without this synchronization point we have a race condition:
    // the logits used above to compute the loss are concurrently (race) modified to carry backward pass grads.
    // since the "logits" are overwritten to be in the [-1, 1] range and sp.Offset is sometimes smaller than -90
    // we errouneously end up computing exp^(90+) which gives us infinities in the loss! this is the fix.
    __syncthreads();

    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging

    for (int i = threadIdx.x; i < V / X128::size; i += blockDim.x) {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // it will be overwritten by the logits gradients which is when we reduce cache persistence
        X128 packed_logits_vec = load128(logits_vec + i * X128::size);  // rely on cs of store128cs
        X128 packed_probs;
        for (int k = 0; k < X128::size; ++k) {
            int element          = i * X128::size + k;
            float prob           = expf((float)packed_logits_vec[k] - sp.Offset) * sp.Scale;
            packed_probs[k]      = (floatX)prob;
            float indicator      = (element == ix) ? 1.0f : 0.0f;
            packed_logits_vec[k] = (floatX)((prob - indicator) * dloss);
        }
        if (WriteDLogits) {
            // reduce cache persistence for the overwritten logits
            // to maximise probability that logits remain in cache between prepare_softmax and here
            store128cs(logits + idx * P + i * X128::size, packed_logits_vec);
        }
        if (WriteProbs) {
            store128(probs + idx * P + i * X128::size, packed_probs);
        }
    }

    // handle remaining elements after the last multiple of X128::size
    // e.g. if V = 8003, and X128::size = 8, we need to handle the last 3 elements
    int unaligned_start = V & ~(X128::size - 1);  // round down to multiple of X128::size
    for (int i = threadIdx.x + unaligned_start; i < V; i++) {
        float prob      = expf((float)logits_vec[i] - sp.Offset) * sp.Scale;
        float indicator = (i == ix) ? 1.0f : 0.0f;
        float dlogit    = (prob - indicator) * dloss;
        if (WriteDLogits) {
            logits[idx * P + i] = (floatX)dlogit;
            // __stcs(logits + idx * P + i, (floatX)dlogit);    //the __stcs intrinsic expects specific data types, and FP8 support in CUDA is version-dependent
            // and requires specific handling.
        }
        if (WriteProbs) {
            probs[idx * P + i] = (floatX)prob;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

// replaces logits with logit gradients
template <typename Type, bool WriteDLogits>
void fused_classifier_v0(Type* logits, float* losses, const float dloss, const int* targets, int B, int T, int V, int P,
                         std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 1024;
    const int N = B * T, grid_size = N;  //  each block for one token
    floatX* probs = NULL;
    fused_classifier_kernel5<<<grid_size, block_size, 0, stream>>>(logits, losses, probs, dloss, targets, B, T, V, P, write_dlogits);
    cudaCheck(cudaGetLastError());
}
