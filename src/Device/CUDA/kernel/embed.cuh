/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
#include <assert.h>
#include <stdint.h>

#include <algorithm>
#include <unordered_map>
#include <utility>  // std::pair
#include <vector>
// llmc internal imports
#include "../cuda_common.h"
#include "utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels
// out = wte[inp]
__global__ static void encoder_forward_kernel3(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * X128::size;
    int N   = B * T * C;
    if (idx >= N) {
        return;
    }

    int bt = idx / C, b = bt / T, t = bt % T, c = idx % C;
    int ix               = inp[b * T + t];
    floatX* out_btc      = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    const floatX* wpe_tc = wpe == nullptr ? nullptr : wpe + t * C + c;

    X128 packed_out, wte128 = load128cs(wte_ix);
    if (wpe_tc == nullptr) {
        for (int k = 0; k < X128::size; k++) {
            packed_out[k] = (floatX)((float)wte128[k]);
        }
    } else {
        X128 wpe128 = load128cs(wpe_tc);
        for (int k = 0; k < X128::size; k++) {
            packed_out[k] = (floatX)((float)wte128[k] + (float)wpe128[k]);
        }
    }
    store128(out_btc, packed_out);  //  store packed(128-BIT) wte to out_btc(aligned memory address)
}

// __device__ inline float embed_gf4(uint32_t* weight, int idx) { return cu_gf4_ff(weight[idx / 8], idx % 8); }

// template <typename T>
// __device__ inline float embed(T* weight, int idx) {
//     return float(weight[idx]);
// }

template <typename T_out>
__global__ static void CU_embed_forw_q4(floatGama* gamas, hBITARR quants, T_out* o, int token, int M, int N, int rc_normal, unsigned int seed) {
#if defined(USE_FP8_BASELINE)
#else
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= N / 2)  // guard
        return;
    assert(token >= 0 && token < M);
    T_out* x0      = o + 2 * offset;
    BIT_8 q        = quants[(token * N) / 2 + offset];
    floatGama* lut = gamas + M + N + token * 16;  //*gamaCol = nullptr;
    floatGama sR = 1.0, sC = 1.0;
    if (rc_normal > 0) {
        sR = gamas[token];
    }

    BIT_8 id0 = ((q) >> 4) & 0x0F, id1 = (q) & 0x0F;
    floatGama g0 = lut[id0] * sR, g1 = lut[id1] * sR;
    if (rc_normal > 0) {
        // g0 *= gamaCol[2 * k];
        // g1 *= gamaCol[2 * k + 1];
    }
    *x0       = g0;  // CU_Float2T<T>(g0, seed);
    *(x0 + 1) = g1;  // CU_Float2T<T>(g1, seed);
#endif
}
template <typename T_out>
__global__ static void CU_embed_forw_nf4(floatGama* gamas, hBITARR quants, T_out* o, int token, int M, int N, int rc_normal, unsigned int seed) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= N / 2)  // guard
        return;
    /*floatGama nf4[16] = {-1.0f,
                         -0.6961928009986877f,
                         -0.5250730514526367f,
                         -0.39491748809814453f,
                         -0.28444138169288635f,
                         -0.18477343022823334f,
                         -0.09105003625154495f,
                         0.0f,
                         0.07958029955625534f,
                         0.16093020141124725f,
                         0.24611230194568634f,
                         0.33791524171829224f,
                         0.44070982933044434f,
                         0.5626170039176941f,
                         0.7229568362236023f,
                         1.0f};*/
    assert(token >= 0 && token < M);
    T_out* x0      = o + 2 * offset;
    BIT_8 q        = quants[(token * N) / 2 + offset];
    floatGama* lut = gamas + M + N + token * 16;  // *gamaCol = nullptr;
    floatGama sR = 1.0, sC = 1.0;                 // zero = gamas[M + N + token * 2], scale = gamas[M + N + token * 2 + 1];
    ;
    if (rc_normal > 0) {
        sR = gamas[token];
    }

    BIT_8 id0 = ((q) >> 4) & 0x0F, id1 = (q) & 0x0F;
    floatGama g0 = (lut[id0]) * sR, g1 = (lut[id1]) * sR;
    if (rc_normal > 0) {
        // g0 *= gamaCol[2 * k];
        // g1 *= gamaCol[2 * k + 1];
    }
    *x0       = g0;  // CU_Float2T<T>(g0, seed);
    *(x0 + 1) = g1;  // CU_Float2T<T>(g1, seed);
}

template <typename T_out, typename T>
__global__ static void CU_embed_forw_1(T_out* o, T* weight, int token, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    // o[i] = weight[token * n + i];
    float a = CU_T2Float<T>(weight + token * n + i);
    o[i]    = CU_Float2T<T_out>(a, seed);
}

// each thread for one element
__global__ static void CU_embed_forw_v0(floatX* out, const int* tokens, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int N   = B * T * C;
    if (idx >= N) {
        return;
    }

    int bt = idx / C, b = bt / T, t = bt % T, c = idx % C;
    int ix               = tokens[b * T + t];
    floatX* out_btc      = out + b * T * C + t * C + c;
    const floatX* wte_ix = wte + ix * C + c;
    if (wpe == nullptr) {
        *out_btc = *wte_ix;
    } else {
        const floatX* wpe_tc = wpe + t * C + c;
#if defined(USE_FP8_BASELINE)
#else
        *out_btc = *wte_ix + *wpe_tc;
#endif
    }
}

// each thread for one token(C elements)
__global__ static void CU_embed_forw_(floatX* out, const int* tokens, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int N   = B * T;
    if (idx >= N) {
        return;
    }
    int ix = tokens[idx], pos = idx % T;
    const floatX* wpe_tc = wpe + pos * C;
    for (int c = 0; c < C; c++) {
        floatX* out_btc      = out + idx * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        if (wpe == nullptr) {
            *out_btc = *wte_ix;
        } else {
#if defined(USE_FP8_BASELINE)
#else
            *out_btc = *wte_ix + wpe_tc[c];
#endif
        }
    }
}

template <class Typ>
__global__ void CU_embed_ternary_forw_(Typ* out, const int* tokens, floatGama* gama, const char* wte, const Typ* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= B * T) {
        return;
    }

    int ix = tokens[idx], pos = idx % T;
    const Typ* wpe_tc = wpe + pos * C;
    float average     = gama[ix];
    Typ ta = (Typ)(average), tb = (Typ)(-average), t0 = (Typ)(0), a;
    for (int c = 0; c < C; c += 8) {
        Typ* out_btc = out + idx * C + c;
        // const floatX* wte_ix = wte + ix * C + c;
        unsigned char tbyte = wte[(ix * C + c) / 8], bit;
        for (int bpos = 0; bpos < 8; bpos++, out_btc++) {
            bit = BYTE_bit(tbyte, bpos);  //(tbyte >> (7-bpos)) & 0x1;
            a   = bit ? ta : t0;          // binary quant after Implicit RELU
            if (wpe == nullptr) {
                *out_btc = a;
            } else {
#if defined(USE_FP8_BASELINE)
#else
                *out_btc = a + wpe_tc[c];
#endif
            }
        }
    }
}

__global__ static void CU_embed_forw_tc(floatX* out, const int* tokens, const floatX* wte, int T, int C, int ldW, bool isTrans = false) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int N   = T * C;
    if (idx >= N) {
        return;
    }

    int t = idx % T, c = idx % C, ix = tokens[t];
    size_t pos_1 = t * C + c, pos_2 = ix * C + c;
    // floatX* out_btc = out + t * C + c;
    // const floatX *wte_ix = wte + ix * C + c;
    // *out_btc = *wte_ix;
    if (isTrans) {
        // pos_1 = t*C+c, pos_2 = c*ldW+t;
        pos_1 = c * T + t, pos_2 = c * ldW + t;
    }
    out[pos_1] = wte[pos_2];  // why result is not deterministic?
}

// no duplicate
__global__ static void CU_embed_back_(floatX* dwte, const int* tokens, const floatX* dout, int T, int C, int ldW, bool isTrans = false) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int N   = T * C;
    if (idx >= N) {
        return;
    }

    int t = idx % T, c = idx % C, ix = tokens[t];
    size_t pos_1 = t * C + c, pos_2 = ix * C + c;
    // floatX out_btc = dout[t * C + c]*scale;
    // floatX* wte_ix = dwte + ix * C + c;
    // *wte_ix += out_btc;
    if (isTrans) {
        // pos_1 = t*C+c, pos_2 = c*ldW+t;
        pos_1 = c * T + t, pos_2 = c * ldW + t;
    }
#if defined(USE_FP8_BASELINE)
#else
    dwte[pos_2] += dout[pos_1];
#endif
}

template <int BLOCK_SIZE = 256>
__global__ static void wte_backward_kernel(floatX* dwte, const int4* bucket_info, const int* workload_indices, const floatX* dout, const int* inp,
                                           unsigned int seed, int B, int T, int C) {
    // In order to be deterministic, we preprocess the inputs on the cpu into "buckets"
    // Each bucket corresponds to (WARP_SIZE * X128::size) channels for a single vocabulary token
    // Each thread handles X128::size channels, e.g. 256 per warp for BF16
    // Each block handles (BLOCK_SIZE / WARP_SIZE) elements in a single bucket in parallel
    // If a bucket has less than 8 elements, some warps will return immediately
    // If a bucket has more than 8 elements, we will loop over all of them
    // The buckets are sorted on the CPU so the largest buckets start 1st
    int bucket     = blockIdx.x;
    int warp_id    = threadIdx.x / WARP_SIZE;
    int lane_id    = threadIdx.x % WARP_SIZE;
    int c_per_warp = WARP_SIZE * X128::size;

    int bucket_start_idx = bucket_info[bucket].x;
    int bucket_size      = bucket_info[bucket].y;
    int bucket_ix        = bucket_info[bucket].z;
    int c                = bucket_info[bucket].w * c_per_warp + (lane_id * X128::size);

    // Each thread handles "X128::size" channels, so at fp8, each warp would handle 512 channels
    // If C is not a multiple of this (e.g. 768), some buckets/c_groups cannot use the entire warp
    if (c >= C) {
        return;
    }
    // Exit early if this is a small bucket and this warp doesn't have any items to process
    if (warp_id >= bucket_size) {
        return;
    }

    float accum[X128::size] = {0.0f};
    __shared__ float accum_shared[X128::size * BLOCK_SIZE];

    for (int item = warp_id; item < bucket_size; item += BLOCK_SIZE / WARP_SIZE) {
        int bt = workload_indices[bucket_start_idx + item];

        const floatX* dout_btc = dout + bt * C + c;
        X128 packed_inp1       = load128cs(dout_btc);
        for (int k = 0; k < packed_inp1.size; k++) {
            accum[k] += (float)packed_inp1[k];
        }
    }

    if (warp_id != 0) {
        // we accumulate into warp 0, so only the other warps need to write to shared memory
        for (int k = 0; k < X128::size; k++) {
            accum_shared[threadIdx.x + k * BLOCK_SIZE] = accum[k];
        }
        return;  // only warp 0 is needed after writing to shared memory
    }

    // Read dwte for warp 0 even if other warps are not finished yet to maximise latency tolerance
    floatX* dwte_ix    = dwte + bucket_ix * C + c;
    X128 packed_in_out = load128(dwte_ix);

    // note: threads which have returned are considered synchronised by CUDA so no risk of deadlock
    __syncthreads();

    // Accumulate into warp 0's registers by reading the values of the other warps in shared memory
    for (int i = threadIdx.x + WARP_SIZE; i < min(BLOCK_SIZE, bucket_size * WARP_SIZE); i += WARP_SIZE) {
        for (int k = 0; k < X128::size; k++) {
            accum[k] += accum_shared[i + k * BLOCK_SIZE];
        }
    }

    // Add the result to dwte and write back to global memory (read-modify-write)
    for (unsigned int k = 0; k < X128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        // stochastic_rounding(accum[k] + (float)packed_in_out[k], &packed_in_out[k], seed + bucket * WARP_SIZE + threadIdx.x + k);
        packed_in_out[k] = CU_Float2T<floatX>(accum[k] + (float)packed_in_out[k], seed + bucket * WARP_SIZE + threadIdx.x + k);
    }
    store128(dwte_ix, packed_in_out);
}

__global__ static void wpe_backward_kernel(floatX* dwpe, const floatX* dout, const int* inp, int B, int T, int C, unsigned int seed) {
    // Each thread handles X128::size "channel positions", e.g. 256 per warp for BF16
    // For gpt2-124M BF16, C=768 and T=1024, so 3 warps per channel and 3072 warps in total
    // For each "channel position" we sum the gradients for every batch at that C/T element
    // This way each dwte element is only updated once, and the kernel is fully deterministic!
    // The previous kernel was not deterministic, as batches were aggregated with atomicAdd
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * X128::size;
    if (idx >= T * C) {
        return;
    }

    // if C is not a multiple of WARP_SIZE*X128::size, it's OK for some warps to handle multiple t
    int t                   = idx / C;
    int c                   = idx % C;
    float accum[X128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        X128 packed_dout = load128cs(dout + (b * T * C) + (t * C) + c);  // will never be read again
        for (int k = 0; k < X128::size; k++) {
            accum[k] += (float)packed_dout[k];
        }
    }

    floatX* dwpe_tc  = dwpe + (t * C) + c;
    X128 packed_dwpe = load128(dwpe_tc);
    for (unsigned int k = 0; k < X128::size; k++) {
        // We use stochastic rounding to go from FP32 to BF16
        // The seed is deterministic and unique for each parameter to guarantee we have determinism AND
        // to avoid **potential** issues with positionX int SquirrelNoise5 argument overflowing which is UB
        // and that somehow messing the quality of random numbers
        // stochastic_rounding(accum[k] + (float)packed_dwpe[k], &packed_dwpe[k], seed + idx + k);
        packed_dwpe[k] = CU_Float2T<floatX>(accum[k] + (float)packed_dwpe[k], seed + idx + k);
    }
    store128(dwpe_tc, packed_dwpe);
}

// ----------------------------------------------------------------------------
// kernel launchers

inline void encoder_forward(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C, cudaStream_t stream) {
    const int block_size = 256;
    const int N          = B * T * C;
    const int grid_size  = CEIL_DIV(N, (int)(block_size * X128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// Fully deterministic (see comments in wte_backward_kernel and wpe_backward_kernel for more details)
inline void encoder_backward(floatX* dwte, floatX* dwpe, floatX* scratch,                // gpu outputs & scratch
                             int* workload_indices, int4* bucket_info,                   // cpu scratch buffers
                             const floatX* dout, const int* inp, const int* inputs_cpu,  // cpu/gpu inputs
                             int B, int T, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N          = T * C / X128::size;
    const int grid_size  = CEIL_DIV(N, block_size);
    if (dwpe != nullptr)
        wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, dout, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());
    //  dwte,const int* inp,const floatX* dout, int B, int T, int C,unsigned int seed
    // embed_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwte, inp, dout, B, T, C,0x0);    return;

    // check the GPU scratch buffer is large enough to hold the bucket info and workload indices
    // todo - this is trivially true given hardcoded scratch buffer size here, is this useful?
    int num_c_groups = CEIL_DIV(C, X128::size * WARP_SIZE);
    assert(B * T * num_c_groups * (sizeof(int4) + sizeof(int)) <= B * T * 3 * C * sizeof(floatX));

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group << 32ULL) + ((uint64_t)inputs_cpu[bt] << 42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }

    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),  // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    int num_buckets    = buckets.size();
    int bucket_index   = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index;                                       // bucket start
        bucket_info[bucket_index].y = bucket.second.size();                                 // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1);  // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1);  // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL << 31ULL) - 1ULL));
        }
        bucket_index++;
    }

    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info     = (int4*)scratch;
    int* d_workload_indices = (int*)(scratch + B * T * num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, stream));

    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, dout, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}

inline void encoder_backward_1(floatX* dwte, floatX* dwpe, const floatX* deltaIn, const int* inp, int B, int T, int C, floatX* scratch, int num_buckets,
                               unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // Launch wpe kernel first (so it runs on the GPU in parallel with the CPU pre-processing for wte)
    const int block_size = 256;
    const int N          = T * C / X128::size;
    const int grid_size  = CEIL_DIV(N, block_size);
    if (dwpe != nullptr)
        wpe_backward_kernel<<<grid_size, block_size, 0, stream>>>(dwpe, deltaIn, inp, B, T, C, seed);
    cudaCheck(cudaGetLastError());

    int num_c_groups        = CEIL_DIV(C, (WARP_SIZE * X128::size));
    int4* d_bucket_info     = (int4*)scratch;
    int* d_workload_indices = (int*)(scratch + B * T * num_c_groups * sizeof(int4));
    // Launch wte kernel
    // todo - profile block sizes on more content (depends on number of buckets and on GPU?)
    wte_backward_kernel<256><<<num_buckets, 256, 0, stream>>>(dwte, d_bucket_info, d_workload_indices, deltaIn, inp, seed, B, T, C);
    cudaCheck(cudaGetLastError());
}
