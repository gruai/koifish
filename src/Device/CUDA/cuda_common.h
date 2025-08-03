/*
Common utilities for CUDA code.
*/
#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <type_traits>  // std::bool_constant

#include "../../CLI_params.hpp"
#include "../../Utils/GST_log.hpp"
#include "../../Utils/GST_util.hpp"
#include "../../g_float.hpp"
#include "../../g_stddef.hpp"

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
extern const size_t cublaslt_workspace_size;
extern void *cublaslt_workspace;
extern cublasComputeType_t cublas_compute;
extern cublasLtHandle_t cublaslt_handle;
extern cublasHandle_t cublas_handle;
// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void inline cublasCheck(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) \
    { cublasCheck((status), __FILE__, __LINE__); }
// ----------------------------------------------------------------------------
// Global defines and settings

// Device properties of the CUDA device used in this process
// defined as extern here because the individual kernels wish to use it
// but it is actually created and instantiated in the main program file
extern cudaDeviceProp deviceProp;
// #define THREAD_TILE_M 16U
// #define THREAD_TILE_N 16U
#define THREAD_TILE_M 8U
#define THREAD_TILE_N 8U
// #define THREAD_TILE_M 4U
// #define THREAD_TILE_N 4U
// WarpSize is not a compile time constant, Defining here like this possibly allows the compiler to optimize better
#define WARP_SIZE 32U

// Thread number of each block  - If each thread requires big private memory, then using less threads per block helps but its not infinite so should be
// soft-limited to a minimum like 32 or 64 depending on algorithm. But maximum is hard-limited to 1024 threads per block.
#define CU_T4B_SMALL 256U
#define CU_T4B_BIG 1024U
// try to make sure that 2 blocks fit on A100/H100 to maximise latency tolerance
// this needs to be defines rather than queried to be used for __launch_bounds__
#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// convenience macro for calculating grid/block dimensions for kernels
// #define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// short-cuts for compile-time boolean values that can be used as function arguments
constexpr std::bool_constant<true> True;
constexpr std::bool_constant<true> False;

// ----------------------------------------------------------------------------
// Error checking

// CUDA error checking
inline void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        _INFO("[CUDA ERROR] at file %s:%d:\n\"%s\" (%s code=%d)\n", file, line, cudaGetErrorString(error), cudaGetErrorName(error), error);
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))


inline void cudaCheckLast(const char* const file, const int line){
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // std::exit(EXIT_FAILURE);
    }
}
#define CHECK_LAST_CUDA_ERROR() cudaCheckLast(__FILE__, __LINE__)

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}


// like cudaFree, but checks for errors _and_ resets the pointer.
template <class T>
inline void cudaFreeCheck(T **ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        _INFO("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}
#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

// ----------------------------------------------------------------------------
// CUDA Precision settings and defines

// ----------------------------------------------------------------------------
// Load and store with streaming cache hints
// Older nvcc does not provide __ldcs and __stcs for bfloat16, despite these
// actually just being unsigned shorts. We need to be careful here to only define
// our own versions if none already exist, otherwise the compiler will complain.
// If not, you easily get "no viable overload" (for sm52) and "function already exists" (sm_80)

#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
__device__ floatX __ldcs(const floatX *address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short *>(address));
    return __nv_bfloat16_raw{bf};
}

__device__ void __stcs(floatX *address, floatX value) { __stcs(reinterpret_cast<unsigned short *>(address), ((__nv_bfloat16_raw)value).x); }
#elif defined(ENABLE_FP8)
__device__ inline floatX __ldcs(const floatX *address) {
    assert(0);
    return (floatX)(0.0);
}
__device__ inline void __stcs(floatX *address, floatX value) { assert(0); }
#endif

// ----------------------------------------------------------------------------
// Profiler utils

class NvtxRange {
   public:
    NvtxRange(const char *s) { nvtxRangePush(s); }
    NvtxRange(const std::string &base_str, int number) {
        std::string range_string = base_str + " " + std::to_string(number);
        nvtxRangePush(range_string.c_str());
    }
    ~NvtxRange() { nvtxRangePop(); }
};
#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// ----------------------------------------------------------------------------
// Utilities to Read & Write between CUDA memory <-> files

// copy num_bytes from device pointer src into file dest, using double buffering running on the given stream.
inline void device_to_file(FILE *dest, void *src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    char *buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2 * buffer_size));
    // split allocation in two
    void *read_buffer  = buffer_space;
    void *write_buffer = buffer_space + buffer_size;

    // prime the read buffer; first copy means we have to wait
    char *gpu_read_ptr = (char *)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    size_t rest_bytes        = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount;

    std::swap(read_buffer, write_buffer);
    // now the main loop; as long as there are bytes left
    while (rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        // while this is going on, transfer the write buffer to disk
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        cudaCheck(cudaStreamSynchronize(stream));  // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // make sure to write the last remaining write buffer
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);
    cudaCheck(cudaFreeHost(buffer_space));
}

// copy num_bytes from file src into device pointer dest, using double buffering running on the given stream.
inline void file_to_device(void *dest, FILE *src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // allocate pinned buffer for faster, async transfer
    // from the docs
    // (https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__HIGHLEVEL_ge439496de696b166ba457dab5dd4f356.html) WC memory is
    // a good option for buffers that will be written by the CPU and read by the device via mapped pinned memory or host->device transfers.
    char *buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2 * buffer_size, cudaHostAllocWriteCombined));
    // split allocation in two
    void *read_buffer  = buffer_space;
    void *write_buffer = buffer_space + buffer_size;

    // prime the read buffer;
    char *gpu_write_ptr = (char *)dest;
    size_t copy_amount  = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src);

    size_t rest_bytes        = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    std::swap(read_buffer, write_buffer);

    // now the main loop; as long as there are bytes left
    while (rest_bytes > 0) {
        // initiate next read
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        // while this is going on, read from disk
        freadCheck(read_buffer, 1, copy_amount, src);
        cudaCheck(cudaStreamSynchronize(stream));  // wait for both buffers to be ready.

        std::swap(read_buffer, write_buffer);
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // copy the last remaining write buffer to gpu
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaFreeHost(buffer_space));
}

// tied to enum PrecisionMode, in a future refactor make them the same
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

typedef struct {
    float TF_32;     // tensor-core performance 32 bit
    float BF_16_32;  // bf16 with 32 bit accumulate
    float FP_16_32;  // fp16 with 32 bit accumulate
    float FP_16_16;  // fp16 with 16 bit accumulate
    float FP_8_32;   // and so on
    float FP_8_16;
    float CLOCK;  // clock frequency from the spec sheet
    float CORES;  // #TCs from the spec sheet
} PerfData;

// basic default data from the nvidia whitepapers
static const PerfData VOLTA             = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
static const PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
static const PerfData AMPERE_CONSUMER   = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
static const PerfData HOPPER            = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
static const PerfData ADA               = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

typedef struct {
    const char *name;
    const PerfData *perf_data;
    float new_cores;
    float new_mhz;
} GPUEntry;

// the overrides for each specific GPU
static GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112, 1777},
    {"NVIDIA RTX A2000 ADA", &ADA, 88, 2130},
    {"NVIDIA RTX A4000 ADA", &ADA, 192, 2175},
    {"NVIDIA RTX A4500 ADA", &ADA, 224, 2580},
    {"NVIDIA RTX A5000 ADA", &ADA, 400, 2550},
    {"NVIDIA RTX A5880 ADA", &ADA, 440, 2460},
    {"NVIDIA RTX A6000 ADA", &ADA, 568, 2505},
    {"NVIDIA GeForce RTX 4090", &ADA, 512, 2520},
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320, 2550},
    {"NVIDIA GeForce RTX 4080", &ADA, 304, 2505},
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264, 2610},
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240, 2610},
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136, 2535},
    {"NVIDIA GeForce RTX 4060", &ADA, 96, 2460},
    {"NVIDIA H100 PCIe", &HOPPER, 456, 1620},
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528, 1830},  // HBM3 = SXM5
};

inline float get_flops_promised(const char *device, int precision_mode) {
    /*
    This function is used to estimate the Model Flops Utilization (MFU)
    basically we have to figure out how many flops the GPU can do per second.
    Note that this is not a simple endeavor and may well go wrong! The details are tricky.
    The returned value is in units of 1e12.

    For the non-top models, actual performance numbers aren't that easy to find, e.g.,
    here https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756, does "Theoretical Performance"
    seems to be without tensor cores.

    So, instead we use that all these cards just use the same types of tensor cores in different
    numbers and at different frequencies. Then we just need to look up these two easily accesible
    numbers for all the other GPUs.
    linear scaling seems to work: comparing spec sheet and calculation:
    4080: 304TCs, 2505 GHz; 97.5TFlops = 165.2/512*304 /2520 * 2505

    Original numbers for the top GPUS are from.
    https://resources.nvidia.com/en-us-tensor-core
    https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    */

    // validate the precision mode as one of the three possible values
    if (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        return -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    int num_gpu_entries = sizeof(gpu_db) / sizeof(gpu_db[0]);
    for (int i = 0; i < num_gpu_entries; i++) {
        if (strcmp(gpu_db[i].name, device) == 0) {
            const PerfData *perf_data = gpu_db[i].perf_data;

            // look up the default flops value for the given precision mode
            float value = -1.0f;
            if (precision_mode == MFUH_PRECISION_BF16) {
                value = perf_data->BF_16_32;
            }
            if (precision_mode == MFUH_PRECISION_FP32) {
                value = perf_data->TF_32;
            }
            if (precision_mode == MFUH_PRECISION_FP16) {
                value = perf_data->FP_16_32;
            }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            if (value < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                return -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            float new_cores = gpu_db[i].new_cores;
            float new_mhz   = gpu_db[i].new_mhz;
            float adjusted  = value * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            return adjusted;
        }
    }

    return -1.0f;  // ¯\_(ツ)_/¯
}

extern cudaStream_t main_stream;
extern int g_dump_level;
template <typename T>
void inline PrintTensor(const char *title, const T *src, bool isDevice, int n1, int n2, int n3 = 1, int n4 = 1, int flag = 0x0) {
    if (g_dump_level > 0 && flag >= 0)
        return;
    T *host_dat  = (T *)src;
    size_t nElem = (size_t)(n1)*n2 * n3 * n4;
    if (nElem == 0)
        return;
    assert(src != nullptr);
    if (isDevice) {
        // SYNC_DEVICE();
        host_dat = (T *)malloc(sizeof(T) * nElem);
        cudaCheck(cudaMemcpy(host_dat, src, nElem * sizeof(T), cudaMemcpyDeviceToHost));
    }

    PrintT(title, host_dat, n1, n2, n3, n4, flag);
    if (isDevice) {
        free(host_dat);
    }
}

inline size_t get_compute_capability() {
    int current_device;
    cudaCheck(cudaGetDevice(&current_device));
    struct cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, current_device));
    return prop.major * 10 + prop.minor;
}

inline bool is_ampere_arch() {
    auto cc = get_compute_capability();
    return (80 <= cc) && (cc < 89);
}

inline bool is_ada_arch() {
    auto cc = get_compute_capability();
    return (cc == 89);
}

inline bool is_hopper_arch() {
    auto cc = get_compute_capability();
    return (90 <= cc) && (cc < 100);
}

inline bool is_blackwell_arch() {
    auto cc = get_compute_capability();
    return (100 <= cc);
}

// inline bool
// is_arch_supported_by_cudnn() {
//     if (cudnnGetVersion() < 8600 && (is_hopper_arch() || is_ada_arch())) {
//         return false;
//     }
//     return true;
// }

inline bool check_device_arch_newer_than(std::string const &arch) {
    size_t arch_major = 6;
    size_t arch_minor = 0;
    if (arch == "blackwell") {
        arch_major = 10;
    }
    if (arch == "hopper") {
        arch_major = 9;
    }
    if (arch == "ampere") {
        arch_major = 8;
    }
    if (arch == "turing") {
        arch_major = 7;
        arch_minor = 5;
    }
    if (arch == "volta") {
        arch_major = 7;
    }
    if (arch == "pascal") {
        arch_major = 6;
    }

    auto queried_version = arch_major * 10 + arch_minor;
    if (get_compute_capability() >= queried_version) {
        return true;
    }
    return false;
}

static half cpu_float2half_rn(float f) {
    void *f_ptr = &f;
    unsigned x  = *((int *)f_ptr);
    unsigned u  = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    __half_raw hr;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        hr.x = 0x7fffU;
        // Add an indirection to get around type aliasing check
        void *hr_ptr = &hr;
        return *reinterpret_cast<half *>(hr_ptr);
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        hr.x = static_cast<unsigned short>(sign | 0x7c00U);
        // Add an indirection to get around type aliasing check
        void *hr_ptr = &hr;
        return *reinterpret_cast<half *>(hr_ptr);
    }
    if (u < 0x33000001) {
        hr.x = static_cast<unsigned short>(sign | 0x0000U);
        // Add an indirection to get around type aliasing check
        void *hr_ptr = &hr;
        return *reinterpret_cast<half *>(hr_ptr);
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift    = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb    = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    hr.x = static_cast<unsigned short>((sign | (exponent << 10) | mantissa));

    // Add an indirection to get around type aliasing check
    void *hr_ptr = &hr;
    return *reinterpret_cast<half *>(hr_ptr);
}

static float cpu_half2float(half h) {
    // Add an indirection to get around type aliasing check
    void *h_ptr   = &h;
    __half_raw hr = *reinterpret_cast<__half_raw *>(h_ptr);

    unsigned sign     = ((hr.x >> 15) & 1);
    unsigned exponent = ((hr.x >> 10) & 0x1f);
    unsigned mantissa = ((hr.x & 0x3ff) << 13);

    if (exponent == 0x1f) { /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) { /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1; /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    // Add an indirection to get around type aliasing check
    void *temp_ptr = &temp;
    float *res_ptr = reinterpret_cast<float *>(temp_ptr);
    return *res_ptr;
}

// Generate uniform numbers [0,1)
static void initImage(float *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10f;  // 2^-32
    }
}

static void initImage(half *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed         = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
static void initImage(int8_t *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate random integers [0, 50] to avoid uint8 overflow
static void initImage(uint8_t *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 50]
        image[index] = (uint8_t)(50 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
static void initImage(int32_t *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
static void initImage(int64_t *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int64_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate booleans
static void initImage(bool *image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        int64_t val = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32

        // val is 0 or 1
        image[index] = (val == 1);
    }
}

template <typename T_ELEM>
struct Surface {
    T_ELEM *devPtr  = NULL;
    T_ELEM *hostPtr = NULL;
    int64_t n_elems = 0;

   protected:
    explicit Surface() {}

   public:
    explicit Surface(int64_t n_elems, [[maybe_unused]] bool hasRef) : n_elems(n_elems) {
        cudaCheck(cudaMalloc((void **)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
        hostPtr = (T_ELEM *)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        initImage(hostPtr, n_elems);
        cudaCheck(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t n_elems, [[maybe_unused]] bool hasRef, bool isInterleaved) {
        (void)isInterleaved;
        cudaCheck(cudaMalloc((void **)&(devPtr), (n_elems) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM *)calloc(n_elems, sizeof(hostPtr[0]));
        initImage(hostPtr, n_elems);
        uint32_t *temp = (uint32_t *)hostPtr;
        for (auto i = 0; i < n_elems; i = i + 2) {
            temp[i + 1] = 1u;
        }

        cudaCheck(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t size, [[maybe_unused]] bool hasRef, T_ELEM fillValue) : n_elems(size) {
        cudaCheck(cudaMalloc((void **)&(devPtr), (size) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM *)calloc(size, sizeof(hostPtr[0]));
        for (int i = 0; i < size; i++) {
            hostPtr[i] = fillValue;
        }
        cudaCheck(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
    }

    Surface(const Surface &other) : n_elems(other.n_elems) {
        cudaCheck(cudaMalloc((void **)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
        hostPtr = (T_ELEM *)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        std::copy(other.hostPtr, other.hostPtr + n_elems, hostPtr);
        cudaCheck(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());
    }

    Surface(Surface &&other) noexcept : Surface() { swap(*this, other); }

    Surface &operator=(Surface other) {
        swap(*this, other);
        return *this;
    }

    friend void swap(Surface &first, Surface &second) {
        std::swap(first.n_elems, second.n_elems);
        std::swap(first.hostPtr, second.hostPtr);
        std::swap(first.devPtr, second.devPtr);
    }

    ~Surface() {
        if (devPtr) {
            cudaFree(devPtr);
            devPtr = nullptr;
        }
        if (hostPtr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
    }
};

void inline D20(void *dev,size_t szData,int flag=0x0){
    cudaCheck(cudaMemset(dev, 0, szData));
}

bool D2H(void *dev,void *host,size_t szData,int flag=0x0);
bool H2D(void *dev,void *host,size_t szData,int flag=0x0);

// copy value of one elemetn from device
template<typename T>
bool D2e(void *dev,T& host,int flag=0x0){
    return D2H(dev,&host,sizeof(T),flag);
}

typedef int (*fnPOS)(int r, int c, int M, int N);
// Column major to be compatible with cuBlas
#define CR2POS(r, c, M, N) ((c) * (M) + (r))
__device__ inline int fnCR2POS(int r, int c, int M, int N) {
// #ifndef NDEBUG
//     assert(r>=0 && r<M && c>=0 && c<N);
// #endif
    return c * M + r;
}

// Row major to be compatible with cuBlas
#define RC2POS(r, c, M, N) ((r) * (N) + (c))
__device__ inline int fnRC2POS(int r, int c, int M, int N) {
// #ifndef NDEBUG
//     assert(r>=0 && r<M && c>=0 && c<N);
// #endif
    return r * N + c;
}
