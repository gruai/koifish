/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief bitonic inplace sorting and ranking algorithms.  from 
 * 		1. https://github.com/nickjillings/bitonic-sort/blob/master/BitonicSortCUDA.cu
 * 		2. https://github.com/teddykoker/torchsort/blob/main/torchsort/isotonic_cuda.cu
 * 		3. https://linebender.org/wiki/gpu/sorting/
 * 
 *  \author Yingshi Chen
 */
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <stdint.h>

template<typename T>
__device__ void _bitonicStep1_(T* smem, int tid, int tpp, int d) {
    int m     = tid / (d >> 1);
    int tib   = tid - m * (d >> 1);
    int addr1 = d * m + tib;
    int addr2 = (m + 1) * d - tib - 1;

    T A     = smem[addr1];
    T B     = smem[addr2];
    smem[addr1] = max(A, B);
    smem[addr2] = min(A, B);
}

template<typename T>
__device__ void _bitonicStep2_(T* smem, int tid, int tpp, int d) {
    int m     = tid / (d >> 1);
    int tib   = tid - m * (d >> 1);
    int addr1 = d * m + tib;
    int addr2 = addr1 + (d >> 1);

    T A     = smem[addr1];
    T B     = smem[addr2];
    smem[addr1] = max(A, B);
    smem[addr2] = min(A, B);
}

template<typename T>
__global__ void bitonicSortKernel128_fp32(T* mem) {
    // Operating on 64 samples
    int bid = blockIdx.x;                                              // Block UID
    int tpp = threadIdx.x;                                             // Thread position in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                   // Thread global UID
    __shared__ T smem[256];                                        // Two blocks worth of shared memory
    smem[tpp]              = mem[blockDim.x * (2 * bid) + tpp];        // Coalesced memory load
    smem[tpp + blockDim.x] = mem[blockDim.x * ((2 * bid) + 1) + tpp];  // Coalesced memory load
    int blocks             = 8;
    for (int blockNum = 1; blockNum <= blocks; blockNum++) {
        int d = 1 << blockNum;
        _bitonicStep1_(smem, tpp, tpp, d);
        __syncthreads();
        d = d >> 1;
        while (d >= 2) {
            _bitonicStep2_(smem, tpp, tpp, d);
            __syncthreads();
            d = d >> 1;
        }
    }

    mem[blockDim.x * (2 * bid) + tpp]       = smem[tpp];
    mem[blockDim.x * ((2 * bid) + 1) + tpp] = smem[tpp + blockDim.x];
}

template<typename T>
__global__ void bitonicSortKernelXBlock1_fp32(T* mem, int blockNum) {
    int bid = blockIdx.x;                             // Block UID
    int tpp = threadIdx.x;                            // Thread position in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread global UID
    int d   = 1 << blockNum;
    _bitonicStep1_(mem, tid, tpp, d);
}
template<typename T>
__global__ void bitonicSortKernelXBlock2_fp32(T* mem, int blockNum, int d) {
    int bid = blockIdx.x;                             // Block UID
    int tpp = threadIdx.x;                            // Thread position in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread global UID
    _bitonicStep2_(mem, tid, tpp, d);
}


template<typename T>
cudaError_t CU_BitonicSort(T* dev_mem, int N) {
    cudaError_t cudaStatus;
    // Launch a kernel on the GPU with one thread for each element.
    int numBlocks = log2((float)N);
    bitonicSortKernel128_fp32<<<N / 256, 128>>>(dev_mem);
    for (int b = 9; b <= numBlocks; b++) {
        int d = 1 << b;
        bitonicSortKernelXBlock1_fp32<<<N / 512, 256>>>(dev_mem, b);
        d = d >> 1;
        while (d >= 2) {
            bitonicSortKernelXBlock2_fp32<<<N / 512, 256>>>(dev_mem, b, d);
            d = d >> 1;
        }
    }
    // bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
Error:

    return cudaStatus;
}

// BitonicRank for float
__device__ void _bitonic_rank_Step1_fp32(float* smem, unsigned int* sindex, int tid, int tpp, int d) {
    int m     = tid / (d >> 1);
    int tib   = tid - m * (d >> 1);
    int addr1 = d * m + tib;
    int addr2 = (m + 1) * d - tib - 1;

    float A             = smem[addr1];
    float B             = smem[addr2];
    unsigned int Ai     = sindex[addr1];
    unsigned int Bi     = sindex[addr2];
    unsigned int _addr1 = addr1;
    unsigned int _addr2 = addr2;
    if (A > B) {
        _addr1 = addr2;
        _addr2 = addr1;
    }
    smem[_addr1]   = A;
    smem[_addr2]   = B;
    sindex[_addr1] = Ai;
    sindex[_addr2] = Bi;
}

__device__ void _bitonic_rank_Step2_fp32(float* smem, unsigned int* sindex, int tid, int tpp, int d) {
    int m     = tid / (d >> 1);
    int tib   = tid - m * (d >> 1);
    int addr1 = d * m + tib;
    int addr2 = addr1 + (d >> 1);

    float A             = smem[addr1];
    float B             = smem[addr2];
    unsigned int Ai     = sindex[addr1];
    unsigned int Bi     = sindex[addr2];
    unsigned int _addr1 = addr1;
    unsigned int _addr2 = addr2;
    if (A > B) {
        _addr1 = addr2;
        _addr2 = addr1;
    }
    smem[_addr1]   = A;
    smem[_addr2]   = B;
    sindex[_addr1] = Ai;
    sindex[_addr2] = Bi;
}

__global__ void bitonicSortRankKernel128_fp32(float* mem, unsigned int* index) {
    // Operating on 64 samples
    int bid = blockIdx.x;                                                // Block UID
    int tpp = threadIdx.x;                                               // Thread position in block
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;                     // Thread global UID
    __shared__ float smem[256];                                          // Two blocks worth of shared memory
    __shared__ unsigned int sindex[256];                                 // Place the index as local
    smem[tpp]                = mem[blockDim.x * (2 * bid) + tpp];        // Coalesced memory load
    smem[tpp + blockDim.x]   = mem[blockDim.x * ((2 * bid) + 1) + tpp];  // Coalesced memory load
    sindex[tpp]              = blockDim.x * (2 * bid) + tpp;
    sindex[tpp + blockDim.x] = blockDim.x * ((2 * bid) + 1) + tpp;
    int blocks               = 8;
    for (int blockNum = 1; blockNum <= blocks; blockNum++) {
        int d = 1 << blockNum;
        _bitonic_rank_Step1_fp32(smem, sindex, tpp, tpp, d);
        __syncthreads();
        d = d >> 1;
        while (d >= 2) {
            _bitonic_rank_Step2_fp32(smem, sindex, tpp, tpp, d);
            __syncthreads();
            d = d >> 1;
        }
    }

    index[blockDim.x * (2 * bid) + tpp]       = sindex[tpp];
    index[blockDim.x * ((2 * bid) + 1) + tpp] = sindex[tpp + blockDim.x];

    mem[blockDim.x * (2 * bid) + tpp]       = smem[tpp];
    mem[blockDim.x * ((2 * bid) + 1) + tpp] = smem[tpp + blockDim.x];
}

__global__ void bitonicSortRankKernelXBlock1_fp32(float* mem, unsigned int* index, int blockNum) {
    // int bid = blockIdx.x;                             // Block UID
    int tpp = threadIdx.x;                            // Thread position in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread global UID
    int d   = 1 << blockNum;
    _bitonic_rank_Step1_fp32(mem, index, tid, tpp, d);
}
__global__ void bitonicSortRankKernelXBlock2_fp32(float* mem, unsigned int* index, int blockNum, int d) {
    int bid = blockIdx.x;                             // Block UID
    int tpp = threadIdx.x;                            // Thread position in block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Thread global UID
    _bitonic_rank_Step2_fp32(mem, index, tid, tpp, d);
}

cudaError_t BitonicSortCUDARank(float* mem, unsigned int* index, int N) {
    cudaError_t cudaStatus;
    float* dev_mem;
    unsigned int* dev_index;
    int numBlocks;

    // Allocate GPU buffers for vector
    cudaStatus = cudaMalloc((void**)&dev_mem, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_index, N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_mem, mem, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    numBlocks = log2((float)N);

    bitonicSortRankKernel128_fp32<<<N / 256, 128>>>(dev_mem, dev_index);
    for (int b = 9; b <= numBlocks; b++) {
        int d = 1 << b;
        bitonicSortRankKernelXBlock1_fp32<<<N / 512, 256>>>(dev_mem, dev_index, b);
        d = d >> 1;
        while (d >= 2) {
            bitonicSortRankKernelXBlock2_fp32<<<N / 512, 256>>>(dev_mem, dev_index, b, d);
            d = d >> 1;
        }
    }

    // bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(index, dev_index, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_mem);
    cudaFree(dev_index);

    return cudaStatus;
}

cudaError_t BitonicSortCUDARankZero(float* dev_mem, unsigned int* dev_index, int N) {
    cudaError_t cudaStatus;
    float* dev_mem_copy;
    int numBlocks;

    // Allocate GPU buffers for vector
    cudaStatus = cudaMalloc((void**)&dev_mem_copy, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from device memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_mem_copy, dev_mem, N * sizeof(float), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    numBlocks = log2((float)N);

    bitonicSortRankKernel128_fp32<<<N / 256, 128>>>(dev_mem_copy, dev_index);
    for (int b = 9; b <= numBlocks; b++) {
        int d = 1 << b;
        bitonicSortRankKernelXBlock1_fp32<<<N / 512, 256>>>(dev_mem_copy, dev_index, b);
        d = d >> 1;
        while (d >= 2) {
            bitonicSortRankKernelXBlock2_fp32<<<N / 512, 256>>>(dev_mem_copy, dev_index, b, d);
            d = d >> 1;
        }
    }

    // bitonicSortKernelTestDbg <<< N / 256, 128 >>> (dev_mem);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "bitonicSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
Error:
    cudaFree(dev_mem_copy);

    return cudaStatus;
}

    /*
		int threads = 256, blocks = (n_vocab + threads - 1) / threads;
		for (int k = 2; k <= n_vocab*2; k <<= 1) {     // why fail?
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<floatLogits><<<blocks, threads>>>((floatLogits*)_logits, j, k,n_vocab);
            cudaDeviceSynchronize();
        }
    }   */

//  Best when array size is power of 2
template <typename T>
__global__ void bitonicSortKernel(T *arr, int j, int k, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;    
    if (i >= n) return;    
    int ij = i ^ j;
    
    // Only compare if ij is within bounds and ij > i to avoid duplicate swaps
    if (ij < n && ij > i) {
        bool shouldSwap = false;        
        if ((i & k) == 0) {            // Ascending order
            shouldSwap = (arr[i] > arr[ij]);
        } else {            // Descending order  
            shouldSwap = (arr[i] < arr[ij]);
        }
        
        if (shouldSwap) {
            T temp = arr[i];
            arr[i] = arr[ij];
            arr[ij] = temp;
        }
    }
}

template <typename T>
__global__ static void mergeSortKernel(T* input, T* output, int n, int size) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * size * 2;

    if (start >= n)
        return;

    int middle = min(start + size, n);
    int end    = min(start + size * 2, n);

    int i = start, j = middle, k = start;

    while (i < middle && j < end) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }

    while (i < middle) output[k++] = input[i++];
    while (j < end) output[k++] = input[j++];
}
