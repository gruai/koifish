/*
LayerNorm CUDA kernel, and also Residual, because sometimes they are fused

Note in llm.c we try to be clever in the backward pass to conserve memory.
All parameters use a += in the backward pass, so we can do gradient accumulation.
But all activations have = instead of += because these are faster (just read, no write).
This is okay for all activations except for those in the residual stream, where the
gradients have to add. We make sure that we do a += as necessary.
E.g., the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <assert.h>

#include <cub/cub.cuh>
// llmc internal imports
#include "../cuda_common.h"
#include "utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

__global__ static void layernorm_forward_kernel3(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd, const floatX* __restrict__ inp,
                                                 const floatX* __restrict__ weight, const floatX* __restrict__ bias, int N, int C) {
    int lane_id   = threadIdx.x % WARP_SIZE;
    int warp_id   = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int idx = blockIdx.x * num_warps + warp_id;
    if (idx >= N) {
        return;
    }  // guard

    // the row of input that this group of threads is responsible for
    const floatX* x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        sum += (float)x[i];
    }
    sum     = warpReduceSum(sum);
    float m = sum / C;
    if (lane_id == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    sum     = warpReduceSum(sum);
    float s = rsqrtf(sum / C + 1e-5f);
    if (lane_id == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    floatX* o = out + idx * C;
    for (int c = lane_id; c < C; c += WARP_SIZE) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * ((float)__ldcs(x + c) - m);
        __stcs(o + c, (floatX)(n * (float)weight[c] + (float)bias[c]));
    }
}

// deprecated
template <typename T, typename Tw>
__device__ static float CU_rmsnorm(T* o, T* x, Tw* weight, int size, float eps, bool ln) {
    int i         = threadIdx.x;
    int blockSize = blockDim.x;

    float mean = 0.0f;
    if (ln) {
        // calculate sum (per thread)
        float sum = 0.0f;
        for (int j = i; j < size; j += blockSize) {
            sum += (float)(x[j]);
        }
        // mean = blockreduce_sum(sum) / size;         // sum across threads in block
        mean = blockReduce_v0<warpReduceSum>(sum) / size;
    }

    // calculate sum of squares (per thread)
    float ss = 0.0f;
    for (int j = i * 2; j < size; j += blockSize * 2) {
        float2 xx = {x[j], x[j + 1]};            //*(float2*)&x[j];
        float2 ww = {weight[j], weight[j + 1]};  //*(float2*)&weight[j];
        float v0  = xx.x - mean;
        float v1  = xx.y - mean;
        ss += v0 * v0;
        ss += v1 * v1;
        *(ablock<T, 2>*)&o[j] = {v0 * ww.x, v1 * ww.y};
    }

    // sum across threads in block
    // ss = blockreduce_sum(ss);
    ss = blockReduce_v0<warpReduceSum>(ss);

    // caller is responsible for normalization
    return rsqrtf(ss / size + eps);
}

template <typename T, typename Tw>
__global__ static void CU_rms_forward(T* __restrict__ out, float* __restrict__ rstd, const T* __restrict__ inp, const Tw* __restrict__ weight, int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    // x128* s_bias = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for (int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i / x128::size] = load128(weight + i);
        // s_bias[i/x128::size] = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= N) {
        return;
    }  // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;
    float sum       = 0.0f;
    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for (int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum     = warpReduceSum(sum);
    float v = 0.f;

    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for (int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k]) * ((float)in_data[k]);
        }
    }

    v       = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w       = s_weight[c / x128::size];
        // const x128 b = s_bias[c / x128::size];
        x128 out_data;
        for (int k = 0; k < x128::size; ++k) {
            float n     = s * ((float)in_data[k]);  // normalized output
            float o     = n * (float)w[k];          // scale it
            out_data[k] = (floatX)o;
        }

        store128cs(out + c, out_data);
    }
    // store the rstd, for the backward pass later
    if (threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

// inline void CU_rms_v2(__nv_bfloat16* o, const __nv_bfloat16* x, const __nv_bfloat16* weight, int dim) {
//     if (dim % 2 != 0) {
//         fprintf(stderr, "FATAL: rmsnorm dim %d is not divisible by 2. Vectorized kernel cannot run.\n", dim);
//         exit(EXIT_FAILURE);
//     }
//     // if dim > (THREADS_PER_BLOCK * some_threshold), a multi-block reduction might be needed,
//     // but for typical dimensions up to 8192, a single block is sufficient and simpler.
//     const int num_blocks = 1;

//     kernel_rms_norm_v2<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(o, x, weight, dim);
// }

__global__ static void layernorm_forward_kernel6(floatX* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd, const floatX* __restrict__ inp,
                                                 const floatX* __restrict__ weight, const floatX* __restrict__ bias, int N, int C) {
    assert(blockDim.x == WARP_SIZE);

    // load weights and biases into shared memory
    // do this before we allow any threads to exit!
    extern __shared__ char* params[];
    // load128/store128 sometimes generated multiple instructions when the types here were floatX*, so
    // let's keep everything as x128
    x128* s_weight = reinterpret_cast<x128*>(params);
    x128* s_bias   = reinterpret_cast<x128*>(params) + (C / x128::size);
    x128* s_in     = reinterpret_cast<x128*>(params) + ((2 + threadIdx.y) * C / x128::size);

    int sidx = (threadIdx.x + WARP_SIZE * threadIdx.y) * x128::size;
    for (int i = sidx; i < C; i += blockDim.y * WARP_SIZE * x128::size) {
        s_weight[i / x128::size] = load128(weight + i);
        s_bias[i / x128::size]   = load128(bias + i);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= N) {
        return;
    }  // guard

    // adjust pointers to current token
    inp += idx * C;
    out += idx * C;

    const float eps = 1e-5f;
    float sum       = 0.0f;
    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = load128cs(inp + c);
        for (int k = 0; k < x128::size; ++k) {
            sum += (float)in_data[k];
        }
        s_in[c / x128::size] = in_data;
    }

    sum     = warpReduceSum(sum);
    float m = sum / C;
    float v = 0.f;

    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        for (int k = 0; k < x128::size; ++k) {
            v += ((float)in_data[k] - m) * ((float)in_data[k] - m);
        }
    }

    v       = warpReduceSum(v) / C;
    float s = rsqrtf(v + eps);

    for (int c = threadIdx.x * x128::size; c < C; c += WARP_SIZE * x128::size) {
        const x128 in_data = s_in[c / x128::size];
        const x128 w       = s_weight[c / x128::size];
        const x128 b       = s_bias[c / x128::size];
        x128 out_data;
        for (int k = 0; k < x128::size; ++k) {
            float n     = s * ((float)in_data[k] - m);    // normalized output
            float o     = n * (float)w[k] + (float)b[k];  // scale and shift it
            out_data[k] = (floatX)o;
        }

        store128cs(out + c, out_data);
    }
    // cache the mean and rstd for the backward pass later
    if (threadIdx.x == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }
    // store the rstd, no need to cache it
    if (threadIdx.x == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }
}

template <typename T>
__global__ static void CU_residual_forward(T* out, const T* inp1, const T* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    out[idx] = inp1[idx] + inp2[idx];
}

__global__ static void residual_forward_x128(floatX* out, const floatX* inp1, const floatX* inp2) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_out;
    x128 packed_inp1 = load128cs(inp1 + idx);
    x128 packed_inp2 = load128cs(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);
}

__global__ static void __launch_bounds__(512, 2)  // todo - any warnings on Turing with only 1024 threads?
    layernorm_backward_kernel10(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch, const floatX* dout, const floatX* inp, const floatX* weight,
                                const float* mean, const float* rstd, int B, int T, int C) {
    int BLOCK_SIZE   = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE;  // number of warps in block
    extern __shared__ float shared[];

    int warpId          = threadIdx.x / WARP_SIZE;  // warp index within a block
    int baseIdx         = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx   = threadIdx.x % WARP_SIZE;  // Thread index within the warp
    int warpsInGrid     = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C    = CEIL_DIV(C, C_per_iteration);  // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C      = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float* dbias_shared   = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared   = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt  = inp + bt * C;
        floatX* dinp_bt       = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean      = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt + i);
            x128 weight128_i = load128(weight + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = mean[bt];
        const float rstd_bt = rstd[bt];
        dnorm_mean          = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean     = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if (global_index < C) {
                dout128   = load128cs(dout_bt + global_index);
                inp128    = load128cs(inp_bt + global_index);
                dinp128   = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for (int o = 0; o < x128::size / f128::size; ++o) {
                f128 dbias_f;
                f128 dweight_f;
                for (int i = 0; i < f128::size; ++i) {
                    int x          = o * f128::size + i;
                    float dout_i   = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i]     = dout_i;
                    dweight_f[i]   = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float)weight128[x] * (float)dout128[x];  // term 1
                    dval -= dnorm_mean;                               // term 2
                    dval -= norm_bti * dnorm_norm_mean;               // term 3
                    dval *= rstd_bt;                                  // final scale
                    dinp128[x] = (floatX)((float)dinp128[x] + dval);
                }

                if (warpId != 0) {
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp   = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for (int i = 0; i < f128::size; ++i) {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for (int i = 0; i < f128::size; ++i) {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if (global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float* scratch_dbias   = scratch;
    float* scratch_dweight = scratch + C;
    for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2 * C * blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2 * C * blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int* tmp_flag = (unsigned int*)(shared + 2 * rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            f128 dbias_accum   = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset      = i + 2 * C * read_block_idx;
                f128 dbias128   = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for (int k = 0; k < f128::size; k++) {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            x128 dbias128   = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int o = 0; o < x128::size / f128::size; ++o) {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for (int i = 0; i < f128::size; ++i) {
                    int x         = o * f128::size + i;
                    dbias128[x]   = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

__global__ static void __launch_bounds__(512, 2)  // todo - any warnings on Turing with only 1024 threads?
    layernorm_backward_kernel11(floatX* dinp, floatX* dweight, float* scratch, const floatX* dout, const floatX* inp, const floatX* weight, const float* mean,
                                const float* rstd, int B, int T, int C) {
    int BLOCK_SIZE   = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE;  // number of warps in block
    extern __shared__ float shared[];

    int warpId          = threadIdx.x / WARP_SIZE;  // warp index within a block
    int baseIdx         = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx   = threadIdx.x % WARP_SIZE;  // Thread index within the warp
    int warpsInGrid     = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C    = CEIL_DIV(C, C_per_iteration);  // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    // float* dbias_shared = shared;
    float* dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float* dbias_tmp_shared   = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float* dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size) {
        // store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid) {
        const floatX* dout_bt = dout + bt * C;
        const floatX* inp_bt  = inp + bt * C;
        floatX* dinp_bt       = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean      = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size) {
            x128 dout128_i   = load128(dout_bt + i);
            x128 inp128_i    = load128(inp_bt + i);
            x128 weight128_i = load128(weight + i);
            for (int k = 0; k < x128::size; k++) {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = mean == nullptr ? 0 : mean[bt];
        const float rstd_bt = rstd[bt];
        dnorm_mean          = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean     = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        for (int c = 0; c < iterations_C; c++) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128   = x128::zeros();
            x128 inp128    = x128::zeros();
            x128 dinp128   = x128::zeros();
            x128 weight128 = x128::zeros();

            if (global_index < C) {
                dout128   = load128cs(dout_bt + global_index);
                inp128    = load128cs(inp_bt + global_index);
                dinp128   = load128(dinp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for (int o = 0; o < x128::size / f128::size; ++o) {
                // f128 dbias_f;
                f128 dweight_f;
                for (int i = 0; i < f128::size; ++i) {
                    int x          = o * f128::size + i;
                    float dout_i   = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    // dbias_f[i] = dout_i;
                    dweight_f[i] = norm_bti * dout_i;

                    float dval = 0.0f;
                    dval += (float)weight128[x] * (float)dout128[x];  // term 1
                    dval -= dnorm_mean;                               // term 2
                    dval -= norm_bti * dnorm_norm_mean;               // term 3
                    dval *= rstd_bt;                                  // final scale
                    dinp128[x] = (floatX)((float)dinp128[x] + dval);
                }

                if (warpId != 0) {
                    // store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0) {
                    for (int j = 1; j < warpsInBlock; j++) {
                        f128 dbias_tmp   = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for (int i = 0; i < f128::size; ++i) {
                            // dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0) {
                    // f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for (int i = 0; i < f128::size; ++i) {
                        // dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    // store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
            if (global_index < C) {
                // cache in L2 as this is read by the next kernel, but bypass L1 to minimise thrashing
                store128cg(dinp_bt + global_index, dinp128);
            }
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int* scratchFlag = (unsigned int*)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    // float* scratch_dbias = scratch;
    float* scratch_dweight = scratch + C;
    for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
        // Write to global memory in the same "shared memory banking friendly" order
        // store128(scratch_dbias + i + 2*C*blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2 * C * blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int* tmp_flag = (unsigned int*)(shared + 2 * rounded_C);
    if (threadIdx.x == 0) {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x - 1) {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size) {
            // f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++) {
                int offset = i + 2 * C * read_block_idx;
                // f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for (int k = 0; k < f128::size; k++) {
                    // dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            // store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock) {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C) {
                break;
            }

            // x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int o = 0; o < x128::size / f128::size; ++o) {
                // f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for (int i = 0; i < f128::size; ++i) {
                    int x = o * f128::size + i;
                    // dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            // store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }
}

void inline layernorm_forward(floatX* out, float* mean, float* rstd, floatX* inp, const floatX* weight, const floatX* bias, int B, int T, int C,
                              cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 256;
    int block_y          = block_size / WARP_SIZE;
    const int N          = B * T;
    const int grid_size  = CEIL_DIV(N, block_y);
    size_t smem          = (2 + block_y) * C * sizeof(floatX);

    // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
    // this may fail, in which case we fall back to the smem free implementation.
    cudaCheck(cudaGetLastError());
    if (mean == nullptr) {  // RMS
        // auto status = cudaFuncSetAttribute(CU_rms_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        // cudaCheck(cudaGetLastError());        assert(status == cudaSuccess);
        CU_rms_forward<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(out, rstd, inp, weight, N, C);
    } else {
        auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaCheck(cudaGetLastError());
        if (status == cudaSuccess) {
            layernorm_forward_kernel6<<<grid_size, dim3(WARP_SIZE, block_y), smem, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
        } else {
            // fall back to the version without shared memory
            const int grid_size_fb = CEIL_DIV(N, (block_size / WARP_SIZE));
            layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
        }
    }
    cudaCheck(cudaGetLastError());
}

void inline residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = N >= 2048 ? 256 : 128;
    if (N % (block_size * x128::size) == 0) {  //  x128 version
        const int grid_size = CEIL_DIV(N, block_size * x128::size);
        residual_forward_x128<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2);
    } else {
        const int grid_size = CEIL_DIV(N, block_size);
        CU_residual_forward<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2, N);
    }
    cudaCheck(cudaGetLastError());
}

void inline layernorm_backward(floatX* delta, floatX* dweight, floatX* dbias, float* scratch, const floatX* deltaIn, const floatX* inp, const floatX* weight,
                               const float* mean, const float* rstd, int B, int T, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size    = 512;
    const int blocks_per_sm = 2;  // supported on every architecture and less cache thrashing than 3
    const int grid_size     = blocks_per_sm * deviceProp.multiProcessorCount;
    size_t rounded_C        = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size  = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    cudaCheck(cudaMemsetAsync(scratch, 0, 1 * sizeof(float), stream));  // only need to reset the flag to 0
    if (dbias == nullptr)
        layernorm_backward_kernel11<<<grid_size, block_size, shared_mem_size, stream>>>(delta, dweight, scratch, deltaIn, inp, weight, mean, rstd, B, T, C);
    else {
        assert(mean != nullptr);
        layernorm_backward_kernel10<<<grid_size, block_size, shared_mem_size, stream>>>(delta, dweight, dbias, scratch, deltaIn, inp, weight, mean, rstd, B, T,
                                                                                        C);
    }

    cudaCheck(cudaGetLastError());
}


__global__ static void CU_rmsnorm_multihead(bf16* __restrict__ vecs, const bf16* __restrict__ weight, int nHead, int HEAD_DIM, float EPS = 1e-6f) {
    // each block processes one vector/head
    const int vec_idx = blockIdx.x;
    if (vec_idx >= nHead)
        return;
    float inv_head_dim = 1.0f/HEAD_DIM;
    const int t_idx     = threadIdx.x;
    const int vec_iters = HEAD_DIM / 2;

    bf16* vec_start = vecs + vec_idx * HEAD_DIM;

    const __nv_bfloat162* row_in    = reinterpret_cast<const __nv_bfloat162*>(vec_start);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out         = reinterpret_cast<__nv_bfloat162*>(vec_start);

    // 1. calculate sum of squares
    float lsum = 0.0f;
    for (int idx = t_idx; idx < vec_iters; idx += blockDim.x) {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        float2 v_fp32         = __bfloat1622float2(v_bf16);
        lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
    }

    // // 2. reduce sum within the block
    // using BlockReduce = cub::BlockReduce<float, nBlk>;
    // __shared__ typename BlockReduce::TempStorage tmp;
    // float block_sum = BlockReduce(tmp).Sum(lsum);
    float block_sum = blockReduce_v0<warpReduceSum>(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) {
        mul_val = rsqrtf(block_sum * inv_head_dim + EPS);
    }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += blockDim.x) {
        __nv_bfloat162 v_in_bf16     = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32             = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32         = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}


template <int THREADS_PER_BLOCK>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK)
    rms_norm_kernel(__nv_bfloat16* __restrict__ Y, const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ weight, size_t D, float inv_dim,
                    float EPS = 1e-6f) {
    const int t_idx     = threadIdx.x;
    const int vec_iters = D / 2;

    const __nv_bfloat162* row_in    = reinterpret_cast<const __nv_bfloat162*>(X);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out         = reinterpret_cast<__nv_bfloat162*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        // convert to fp32 for math
        float2 v_fp32 = __bfloat1622float2(v_bf16);

        // lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    __shared__ float mul_val;
    if (t_idx == 0) {
        float val    = __fmaf_rn(block_sum, inv_dim, EPS);
        float approx = __frsqrt_rn(val);
        // mul_val = approx * (1.5f - 0.5f * val * approx * approx);
        mul_val = rsqrtf(val);
    }
    __syncthreads();

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK) {
        __nv_bfloat162 v_in_bf16     = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32             = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32         = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        // convert back to BF16 for storing
        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

void inline CU_rms_v2(__nv_bfloat16* o, const __nv_bfloat16* x, const __nv_bfloat16* weight, int dim) {
    if (dim % 2 != 0) {
        fprintf(stderr, "FATAL: rmsnorm dim %d is not divisible by 2. Vectorized kernel cannot run.\n", dim);
        exit(EXIT_FAILURE);
    }
    // if dim > (THREADS_PER_BLOCK * some_threshold), a multi-block reduction might be needed,
    // but for typical dimensions up to 8192, a single block is sufficient and simpler.
    constexpr int THREADS_PER_BLOCK = CU_T4B_SMALL, num_blocks = 1;
    rms_norm_kernel<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>(o, x, weight, dim, 1.0f / dim);
}