/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Quantizer is much more subtle & complex than most people think. It refects the essence of our world, just like quantum mechanics
 *
 *  \brief Some quant kernels
 *  \author Yingshi Chen
 */

#include <math_constants.h>

#include <cmath>

#include "../../Tensor/GTensor.hpp"
#include "../../Tensor/GeQuant.hpp"
#include "../cuda_common.h"
#include "../g_float.hpp"
#include "operator.cuh"
#include "utils.cuh"

// CUDA kernel for assigning points to clusters
template <typename T>
__global__ void CU_KMeans_asign(const T* data, const floatGama* centroids, int* assignments, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float min_dist = FLT_MAX, a = CU_T2Float(data+idx);
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; cluster++) {
        float diff = fabs(a - (float)(centroids[cluster]));
        if (diff < min_dist) {
            min_dist     = diff;
            best_cluster = cluster;
        }
    }

    assignments[idx] = best_cluster;
}

template <typename T>
__global__ void CU_KMeans_update(const T* data, floatGama* centroids, const int* assignments, int* cluster_counts, int n, int k) {
    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k)
        return;

    centroids[cluster] = 0.0f, cluster_counts[cluster] = 0;
    for (int i = 0; i < n; i++) {
        if (assignments[i] == cluster) {
            atomicAdd(&centroids[cluster], (floatGama)CU_T2Float(data+i));
            atomicAdd(&cluster_counts[cluster], 1);
        }
    }
    // Compute new centroid (average)
    int count = cluster_counts[cluster];
    if (count > 0) {
        centroids[cluster] /= count;
    }
}

// row-scaling  2 thrshold
template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_ternary_2thrshold(floatGama* gama, T* mat, int M, int N, int update) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid, ldJ = blockDim.x;
    float ta = 1.0, tb = (-1.0), t0 = 0.0;
    for (int j = tid; j < M; j += ldJ) {
        float sum_1 = 0.0f, sum_2 = 0.0f, a;
        int n_1 = 0, n_2 = 0;
        T* x0 = mat + j * N;
        for (int k = 0; k < N; k++) {
            a = CU_T2Float(x0 + k);
            if (a > 0) {
                sum_1 += a;
                n_1++;
            } else {
                sum_2 -= a;
                n_2++;
            }
        }

        if (update == QUANT_ALG::W_SCALE) {
            // gama[j] = average;
            ta = n_1 == 0 ? t0 : (sum_1 / n_1), tb = n_2 == 0 ? t0 : (-sum_2 / n_2);
        } else {
            gama[j] = 1.0f;
        }
        T xa = (T)ta, xb = (T)tb;
        for (int k = 0; k < N; k++) {
            a = CU_T2Float(x0 + k);
            if (a > ta / 2)
                x0[k] = xa;
            else if (a < tb / 2)
                x0[k] = xb;
            else {
                x0[k] = k % 2 == 0 ? xa : xb;
            }
            // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
        }
    }
    __syncthreads();
}

/*
template <class T>
__device__ inline void CU_ternary_row(floatGama* gama, T* row, int N, int update) {
    T ta = (T)1.0, tb = (T)(-1.0), t0 = (T)(0.0);
    float sum = 0.0f, a, average = 0.0f;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        sum += fabs(a);
    }
    average = (sum / (N)) + 1.0e-5;

    if (update == QUANT_ALG::W_SCALE) {
        // gama[idx] = average;
        ta = (T)(average), tb = (T)(-average);
    } else {
        // gama[idx] = 1.0f;
    }
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(row + k);
        if (a > average / 2)
            row[k] = ta;
        else if (a < -average / 2)
            row[k] = tb;
        else {
            row[k] = k % 2 == 0 ? ta : tb;
        }
        // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
    }
}*/
// row-scaling  & online update
template <class T>
__global__ void CU_ternary_online(T* mat, int M, int N, int seed) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid;
    if (idrow >= M)
        return;

    float sum = 0.0f, a, average = 0.0f;
    T* x0 = mat + idrow * N;
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(x0 + k);
        sum += fabs(a);
    }
    average     = sum / N;
    float thrsh = average / 2;
    T ta = CU_Float2T<T>(average, seed), tb = CU_Float2T<T>(-average, seed);
    for (int k = 0; k < N; k++) {
        a = CU_T2Float(x0 + k);
        if (a > thrsh)
            x0[k] = ta;
        else if (a < -thrsh)
            x0[k] = tb;
        else {
            x0[k] = k % 2 == 0 ? ta : tb;
        }
        // x0[k] = a > average / 2 ? ta : a < -average / 2 ? tb : t0;
    }

    // __syncthreads();
}

// row-scaling  1 thrshold
template <class T>
__global__ void CU_X2ternary_(floatGama* gama, T* mat0, char* terns, int M, int N, int bpe, bool isOverwrite) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard
    CU_X2ternary_row(gama + idrow, mat0 + idrow * N, terns + (idrow * N) / 8, N, isOverwrite);

    // __syncthreads();
}

// row-scaling  1 thrshold
template <class T>
__global__ void CU_ternary2X_(floatGama* gama, const char* terns, T* mat0, int M, int N, int seed) {
    int tid = threadIdx.x, idrow = blockIdx.x * blockDim.x + tid, bit = 0;
    if (idrow >= M)
        return;  // guard

    float average = gama[idrow];
    T* x0         = mat0 + idrow * N;
    if (average == 0) {
        memset(x0, 0x0, sizeof(T) * N);
        return;
    }
    T ta = (T)(average), tb = (T)(-average), t0 = (T)(0);
    // T ta = CU_Float2T<T>(average, seed), tb = CU_Float2T<T>(-average, seed);
    const char* tern = terns + (idrow * N) / 8;
    for (int k = 0; k < N; k += 8, tern++) {
        unsigned char tbyte = *tern;  // terns[(idrow * N + k) / 8];
#pragma unroll
        for (int bpos = 0; bpos < 8; bpos++, x0++) {
            // int idx = idrow * N + k + bpos;
            // if (idx == 0) {
            //     int debug = 0;
            // }
            bit = BYTE_bit(tbyte, bpos);  //(tbyte >> (7-bpos)) & 0x1;
            *x0 = bit ? ta : t0;          // binary quant after Implicit RELU
            // *x0 = bit ? ta : tb;
            // *x0 = bit ? (bpos%2==1 ? ta : tb) : t0;      // would explode
        }
    }

    // __syncthreads();
}

#define WARPS_PER_BLOCK 32
#define EMB_DIM 128

template <typename T, typename Tproj>
__global__ static void CU_QJL_key(T* key_states, uint8_t* key_quant, uint8_t* key_outlier_quant, const uint8_t* outlier_indices, const Tproj* rand_prj,
                                  T* outlier_norms, int batch_size, int head_size, int n_size, int group_size, int sketch_dim, int outlier_sketch_dim,
                                  int emb_dim, int outlier_counts) {
    size_t bhn        = blockIdx.x;
    size_t threadLane = threadIdx.x;
    size_t wIdx       = threadIdx.y;
    size_t gIdx       = blockIdx.y * WARP_SIZE;
    size_t pIdx       = blockIdx.z * WARPS_PER_BLOCK + wIdx;

    int hash_dim         = sketch_dim / 8;
    int outlier_hash_dim = outlier_sketch_dim / 8;

    int base_index_key_quant     = (bhn * group_size * hash_dim) + ((gIdx + threadLane) * hash_dim);
    int base_index_outlier_quant = (bhn * group_size * outlier_hash_dim) + ((gIdx + threadLane) * outlier_hash_dim);

    int base_index_outlier_indices = bhn * outlier_counts;
    const uint8_t* outlier_ind     = outlier_indices + base_index_outlier_indices;

    int base_index_key = (bhn * group_size * emb_dim) + (gIdx * emb_dim);
    T* key             = key_states + base_index_key;

    int base_index_rand_prj = (pIdx * emb_dim);
    const Tproj* sketch     = rand_prj + base_index_rand_prj;

    int base_index_outlier_norm = (bhn * group_size) + gIdx;
    T* key_outlier_norm         = outlier_norms + base_index_outlier_norm;

    __shared__ uint8_t shared_mask[EMB_DIM];
    size_t tIdx = wIdx * WARP_SIZE + threadLane;
#pragma unroll
    for (size_t tile_idx{tIdx}; tile_idx < EMB_DIM; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_mask[tile_idx] = 0;
    }
    __syncthreads();
    if (tIdx < outlier_counts) {
        size_t otlr_idx       = outlier_ind[tIdx];
        shared_mask[otlr_idx] = 1;
    }
    __syncthreads();

    __shared__ float shared_keys[EMB_DIM][WARP_SIZE];
#pragma unroll
    for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
#pragma unroll
        for (size_t chnl_tile{threadLane}; chnl_tile < EMB_DIM; chnl_tile += WARP_SIZE) {
            shared_keys[chnl_tile][grp_tile] = (float)(key[grp_tile * EMB_DIM + chnl_tile]);
        }
    }
    __syncthreads();

    float sketched_keys     = 0.0;
    float sketched_outliers = 0.0;
#pragma unroll
    for (size_t chnl_idx{0}; chnl_idx < EMB_DIM; chnl_idx++) {
        float key_proj_prod = (float)(sketch[chnl_idx]) * shared_keys[chnl_idx][threadLane];
        if (shared_mask[chnl_idx] == 0) {
            sketched_keys += key_proj_prod;
        } else {
            sketched_outliers += key_proj_prod;
        }
    }

    __shared__ float shared_outlier_norms[WARP_SIZE];
    if (blockIdx.z == 0) {
        if (wIdx == 0) {
            shared_outlier_norms[threadLane] = 0.0;
        }
        __syncthreads();

#pragma unroll
        for (size_t chnl_idx{wIdx}; chnl_idx < EMB_DIM; chnl_idx += WARPS_PER_BLOCK) {
            if (shared_mask[chnl_idx] != 0) {
                atomicAdd(&shared_outlier_norms[threadLane], pow(shared_keys[chnl_idx][threadLane], 2));
            }
        }
    }
    __syncthreads();

    __shared__ uint8_t shared_key_quant[WARP_SIZE][WARPS_PER_BLOCK];
    __shared__ uint8_t shared_key_outlier_quant[WARP_SIZE][WARPS_PER_BLOCK];
    shared_key_quant[threadLane][wIdx]         = (sketched_keys > 0 ? (1 << (wIdx % 8)) : 0);
    shared_key_outlier_quant[threadLane][wIdx] = (sketched_outliers > 0 ? (1 << (wIdx % 8)) : 0);
    __syncthreads();

    if (gIdx + threadLane >= group_size)
        return;

    if ((wIdx % 8) == 0) {
        uint8_t hashed_key = 0;
#pragma unroll
        for (int shift = 0; shift < 8; shift++) {
            hashed_key += shared_key_quant[threadLane][wIdx + shift];
        }
        key_quant[base_index_key_quant + pIdx / 8] = hashed_key;

        if (pIdx >= outlier_sketch_dim)
            return;

        uint8_t hashed_outlier = 0;
#pragma unroll
        for (int shift = 0; shift < 8; shift++) {
            hashed_outlier += shared_key_outlier_quant[threadLane][wIdx + shift];
        }
        key_outlier_quant[base_index_outlier_quant + pIdx / 8] = hashed_outlier;
    } else if ((wIdx == 1) && (blockIdx.z == 0)) {
        key_outlier_norm[threadLane] = (T)(sqrtf(shared_outlier_norms[threadLane]));
    }
    return;
}

template <typename T, typename Tproj>
__global__ void calc_score_kernel(T* query_states, const uint8_t* key_quant, const uint8_t* key_outlier_quant, T* key_norm, T* key_outlier_norm,
                                  const uint8_t* outlier_indices, const float* query_sketch, const Tproj* rand_prj, float* scores, int batch_size,
                                  int head_size, int n_size, int group_size, int sketch_dim, int outlier_sketch_dim, int emb_dim, int outlier_counts) {
    size_t bh         = blockIdx.x;
    size_t n          = blockIdx.y;
    size_t threadLane = threadIdx.x;
    size_t wIdx       = threadIdx.y;
    size_t gIdx       = blockIdx.z * WARP_SIZE;

    int hash_dim         = sketch_dim / 8;
    int outlier_hash_dim = outlier_sketch_dim / 8;

    int base_index_outlier_indices = (bh * n_size * outlier_counts) + (n * outlier_counts);
    const uint8_t* outlier_ind     = outlier_indices + base_index_outlier_indices;

    int base_index_query_sketch = (bh * sketch_dim);
    const float* q_sketch       = query_sketch + base_index_query_sketch;

    int base_index_key_quant = (bh * n_size * group_size * hash_dim) + (n * group_size * hash_dim) + (gIdx * hash_dim);
    const uint8_t* k_quant   = key_quant + base_index_key_quant;

    int base_index_outlier_quant = (bh * n_size * group_size * outlier_hash_dim) + (n * group_size * outlier_hash_dim) + (gIdx * outlier_hash_dim);
    const uint8_t* outlier_quant = key_outlier_quant + base_index_outlier_quant;

    int base_index_key_norm = (bh * n_size * group_size) + (n * group_size) + gIdx;
    const T* k_norm         = key_norm + base_index_key_norm;
    const T* outlier_norm   = key_outlier_norm + base_index_key_norm;

    int base_index_query_states = (bh * emb_dim);
    const T* query              = query_states + base_index_query_states;

    // load query states into shared memory
    __shared__ float shared_query[EMB_DIM];
    size_t tIdx = wIdx * WARP_SIZE + threadLane;
    for (size_t tile_idx{tIdx}; tile_idx < emb_dim; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_query[tile_idx] = convert_to_float<T>(query[tile_idx]);
    }
    // load outlier indices into shared buffer
    __shared__ uint8_t shared_outlier_ind[WARP_SIZE];
    for (size_t tile_idx{tIdx}; tile_idx < outlier_counts; tile_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
        shared_outlier_ind[tile_idx] = outlier_ind[tile_idx];
    }
    // allocate shared memory to inner products of quantized keys or outliers with query_sketch
    __shared__ float shared_innprod[WARP_SIZE];
    __shared__ float shared_outlier_innprod[WARP_SIZE];
    if (wIdx == 0) {
        shared_innprod[threadLane]         = 0.0;
        shared_outlier_innprod[threadLane] = 0.0;
    }
    __syncthreads();

    // reserve shared memory for a block of query sketch and query outlier sketch
    __shared__ float shared_q_sketch[WARP_SIZE][8];
    __shared__ float shared_q_outliers_sketch[WARP_SIZE][8];
    for (size_t chnl_tile{0}; chnl_tile < sketch_dim; chnl_tile += (8 * WARP_SIZE)) {
        // load a block of query sketch and compute query outlier sketch
        for (size_t q_idx{tIdx}; q_idx < (8 * WARP_SIZE); q_idx += (WARP_SIZE * WARPS_PER_BLOCK)) {
            shared_q_sketch[q_idx / 8][q_idx % 8]          = 0.0;
            shared_q_outliers_sketch[q_idx / 8][q_idx % 8] = 0.0;
            if (chnl_tile + q_idx < sketch_dim) {
                shared_q_sketch[q_idx / 8][q_idx % 8] = q_sketch[chnl_tile + q_idx];
                for (size_t i{0}; i < outlier_counts; i++) {
                    int otlr_idx = shared_outlier_ind[i];
                    shared_q_outliers_sketch[q_idx / 8][q_idx % 8] +=
                        shared_query[otlr_idx] *
                        convert_to_float<Tproj>(rand_prj[(otlr_idx * sketch_dim) + chnl_tile + q_idx]);  // convert_to_float(const_query[bh][otlr_idx])
                }
            }
        }

        for (size_t grp_tile{wIdx}; grp_tile < WARP_SIZE; grp_tile += WARPS_PER_BLOCK) {
            // load key quant and outlier quant
            uint8_t key_quant_buffer     = k_quant[grp_tile * hash_dim + chnl_tile / 8 + threadLane];
            uint8_t outlier_quant_buffer = 0;
            if (chnl_tile + 8 * threadLane < outlier_sketch_dim) {
                outlier_quant_buffer = outlier_quant[grp_tile * outlier_hash_dim + chnl_tile / 8 + threadLane];
            }
            __syncthreads();

            float k_inner_prod       = 0.0;
            float outlier_inner_prod = 0.0;
            for (int shift = 0; shift < 8; shift++) {
                float q_sketch_val = shared_q_sketch[threadLane][shift] - shared_q_outliers_sketch[threadLane][shift];
                k_inner_prod += (((key_quant_buffer >> shift) & 1) ? q_sketch_val : -q_sketch_val);
                if (chnl_tile + 8 * threadLane < outlier_sketch_dim) {
                    float q_otlr_sketch_val = shared_q_outliers_sketch[threadLane][shift];
                    outlier_inner_prod += (((outlier_quant_buffer >> shift) & 1) ? q_otlr_sketch_val : -q_otlr_sketch_val);
                }
            }
            __syncthreads();

            k_inner_prod       = warpReduceSum(k_inner_prod);
            outlier_inner_prod = warpReduceSum(outlier_inner_prod);
            __syncthreads();
            if (threadLane == 0) {
                shared_innprod[grp_tile] += k_inner_prod;
                shared_outlier_innprod[grp_tile] += outlier_inner_prod;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    if (gIdx + threadLane >= group_size)
        return;
    if (wIdx == 0) {
        float scl       = sqrtf(M_PI_2) / static_cast<float>(sketch_dim);
        float scl_otlr  = sqrtf(M_PI_2) / static_cast<float>(outlier_sketch_dim);
        float norm_otlr = convert_to_float<T>(outlier_norm[threadLane]);
        float norm_k    = sqrtf(pow(convert_to_float<T>(k_norm[threadLane]), 2) - pow(norm_otlr, 2));
        float score     = scl * norm_k * shared_innprod[threadLane] + scl_otlr * norm_otlr * shared_outlier_innprod[threadLane];
        scores[(bh * n_size * group_size) + (n * group_size) + gIdx + threadLane] = score;
    }
}

template <typename T, typename Tproj>
float Q_JL<T, Tproj>::Update(shared_ptr<GTensor> hTensor, int flag) {
    T* norms       = TO<T>(outlier_norms);
    uint8_t *quant = TO<uint8_t>(key_quant), *outlier_quant = TO<uint8_t>(key_outlier_quant);
    const uint8_t* outlier_indices;
    int outlier_sketch_dim = nOutlier;
    int sketch_dim         = hProj->shape[0];
    dim3 dimBlocks;   //(batch * head * n, blocksPerGroup, numProjBlocks);
    dim3 dimThreads;  //(WARP_SIZE, WARPS_PER_BLOCK, 1);
    /*
        norms = key_states.norm(dim=-2)
        outlier_count_general = 8
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        self.outlier_indices = outlier_indices.to(torch.uint8).contiguous()
    */
    // CU_QJL_key<<<dimBlocks, dimThreads>>>((T*)(tensor->data), quant, outlier_quant, outlier_indices, JL, norms, batch_size, head_size, n_size,
    //                            group_size, sketch_dim, outlier_sketch_dim, emb_dim, nOutlier);

    return 0.0;
}
template class Q_JL<float, float>;
template class Q_JL<bf16, bf16>;

template <typename T>
float Q_Cluster<T>::Update(shared_ptr<GTensor> hTensor, int flag) {
    size_t nEle = hTensor->size(), dGrid = CEIL_DIV(nEle, CU_T4B_MIDDLE);
    void* quant = hTensor->BeforeQuant(this);
    T* mat0     = TO<T>(hTensor);
    int *assign = (int*)quant, *cluster_counts = nullptr;
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1];
    floatGama* centroids = hTensor->gama_T();

    for (int iter = 0; iter < this->nMostLoop; iter++) {  // Save current centroids for convergence check
        // cudaMemcpy(previous_centroids, d_centroids, k * dim * sizeof(float), cudaMemcpyDeviceToHost);
        CU_KMeans_asign<<<dGrid, CU_T4B_MIDDLE>>>(mat0, centroids, assign, nCol, nCluster);
        cudaDeviceSynchronize();
        CU_KMeans_update<<<dGrid, CU_T4B_MIDDLE>>>(mat0, centroids, assign, cluster_counts, nCol, nCluster);
        cudaDeviceSynchronize();
        // Check convergence (simplified)
        // if (iter > 0 && has_converged(previous_centroids, 1e-5f)) {
        //     break;
        // }
    }
    return this->imbalance;
}
template class Q_Cluster<bf16>;
template class Q_Cluster<float>;

template <typename T>
float Q_SinkNormal<T>::Update(shared_ptr<GTensor> hTensor, int flag) {
    size_t nEle = hTensor->size(), dGrid = CEIL_DIV(nEle, CU_T4B_MIDDLE);
    void* quant = hTensor->BeforeQuant(this);
    int nRow = hTensor->shape[0], nCol = hTensor->shape[1];
    floatGama *rowStdDev = hTensor->gama_T(), *colStdDev = rowStdDev + nRow;

    T* mat0 = TO<T>(hTensor);
    for (int loop = 0; loop < this->nMostLoop; loop++) {
        CU_RowStdDev<<<CEIL_DIV(nRow, CU_T4B_MIDDLE), CU_T4B_MIDDLE>>>(mat0, rowStdDev, nRow, nCol);
        CU_ColStdDev<<<CEIL_DIV(nCol, CU_T4B_MIDDLE), CU_T4B_MIDDLE>>>(mat0, colStdDev, nRow, nCol);
        // CU_DualScale<<<CEIL_DIV(M*N,CU_T4B_MIDDLE),CU_T4B_MIDDLE>>>(mat0, rowStdDev,colStdDev, M, N);
        if (this->imbalance > 1.0)
            break;
    }
    // CU_X2ternary_<<<CEIL_DIV(M*N,CU_T4B_MIDDLE),CU_T4B_MIDDLE>>>(mat0, rowStdDev,colStdDev, M, N);

    hTensor->AfterQuant(this, typNUMBER::F8E5M2, nullptr);
    return this->imbalance;
}
template class Q_SinkNormal<bf16>;
template class Q_SinkNormal<float>;

template __global__ void CU_ternary_online<bf16>(bf16* mat, int M, int N, int seed = 0x0);
template __global__ void CU_ternary2X_<bf16>(floatGama* gama, const char* terns, bf16* mat0, int M, int N, int seed = 0x0);
template __global__ void CU_X2ternary_<bf16>(floatGama* gama, bf16* mat0, char* terns, int M, int N, int bpe, bool isOverwrite = false);
