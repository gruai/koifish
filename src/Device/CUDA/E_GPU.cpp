/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief CUDA&CUDNN
 *  \author Yingshi Chen
 */

#include "../EDevice.hpp"
#include "../Utils/GST_MemBuffer.hpp"
#include "../Utils/os/GST_os.hpp"
#include "./cuda_common.h"
#include "./gpu_report.hpp"

// Static member definitions
std::atomic<int> CudaDriverManager::allocation_count{0};
std::mutex CudaDriverManager::shutdown_mutex;
bool CudaDriverManager::driver_initialized = false;
bool CudaDriverManager::shutdown_initiated = false;

void SUM::GPU_TIME(double& a, const double b, int flag) {
    return;
#ifndef NDEBUG
    // cudaCheck(cudaStreamSynchronize(main_stream));
    SYNC_STREAM();  // Ensure all GPU work is complete, slow, only for debug version
#endif
    a += GST_us() - b;
}

int EDGE_DEVICES::GPU_::MAX_COUNT = 16;  //  16
std::vector<EDGE_DEVICES::GPU_> EDGE_DEVICES::GPU_::cudaGetAllDevice(int flag) {
    std::vector<GPU_> devices;
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    {
        int major_version     = 0;
        size_t version_length = 0;
        if (rocblas_get_version_string_size(&version_length) == rocblas_status_success) {
            std::vector<char> version(version_length + 1, '\0');
            if (rocblas_get_version_string(version.data(), version.size()) == rocblas_status_success) {
                version.resize(::strlen(version.data()));
                int parsed_value = 0;
                if (std::from_chars(version.data(), version.data() + version.size(), parsed_value).ec == std::errc()) {
                    major_version = parsed_value;
                }
            }
        }
        if (major_version < 4) {
            _LOG_DEBUG(_CUDA_NAME " calling rocblas_initialize as a workaround for a rocBLAS bug\n");
            rocblas_initialize();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
#endif
    int device_count = 0;
    cudaError_t err  = cudaGetDeviceCount(&device_count);  // CUDA functions do not throw exceptions, why?
    if (err != cudaSuccess) {
        _ERROR("%s: failed to initialize CUDA: %s\n", __func__, cudaGetErrorString(err));
        return devices;
    }
    // assert(device_count <= GPU_DEVICE::MAX_COUNT);

#ifdef _CUDA_FORCE_MMQ
    _INFO("%s: _CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    _INFO("%s: _CUDA_FORCE_MMQ:    no\n", __func__);
#endif  // _CUDA_FORCE_MMQ
#ifdef _CUDA_FORCE_CUBLAS
    _INFO("%s: _CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    _INFO("%s: _CUDA_FORCE_CUBLAS: no\n", __func__);
#endif  // _CUDA_FORCE_CUBLAS
    _INFO("%s: found %d CUDA devices:\n", __func__, device_count);
    devices.resize(device_count);
    for (int id = 0; id < device_count; ++id) {
        int device_vmm = 0;
#if defined(_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id         = id;
            CU_CHECK(cuMemGetAllocationGranularity(&devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif  // defined(_USE_VMM)
        devices[id].vmm = !!device_vmm;
        cudaDeviceProp prop;
        cudaCheck(cudaGetDeviceProperties(&prop, id));
        devices[id].total_vram += prop.totalGlobalMem;
        devices[id].nsm       = prop.multiProcessorCount;
        devices[id].smpb      = prop.sharedMemPerBlock;
        devices[id].warp_size = prop.warpSize;
#if defined(_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        devices[id].smpbo = prop.sharedMemPerBlock;

        devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((devices[id].cc & 0xff00) == 0x0) {
            _LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n", id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                devices[id].cc = _CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                devices[id].cc += prop.minor * 0x10;
            }
        }
        _INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d\n", id, prop.name, prop.gcnArchName, devices[id].cc & 0xffff, device_vmm ? "yes" : "no",
              prop.warpSize);
#elif defined(_USE_MUSA)
        // TODO: refine the .cc to reflect MUSA's actual CC capabilities
        devices[id].smpbo = prop.sharedMemPerBlockOptin;
        devices[id].cc    = 100 * prop.major + 10 * prop.minor;
        _INFO("  Device %d: %s, compute capability %d.%d, VMM: %s\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no");
#else
        devices[id].smpbo = prop.sharedMemPerBlockOptin;
        devices[id].cc    = 100 * prop.major + 10 * prop.minor;
        _INFO("  Device %d: %s(%.6gM), compute capability %d.%d, VMM: %s\n", id, prop.name, devices[id].total_vram / 1.0e6, prop.major, prop.minor,
              device_vmm ? "yes" : "no");
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        namespace gr = strata::program::gpu_report; 
        gr::Card c;     //  v0.1 strata::print_startup_gpu_report
        c.index    = id;
        c.free_mib = (uint64_t)(free_b >> 20), c.total_mib = (uint64_t)(total_b >> 20);
        c.name = prop.name;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        char bus[64] = {};
        if (cudaDeviceGetPCIBusId(bus, (int)sizeof bus, id) == cudaSuccess)
            c.processes_known = gr::query_other_processes(bus, c.processes);
        else
            (void)cudaGetLastError();
#endif
        _INFO(" GPU memory right now (best effort):\n%s\n", gr::format_gpu_memory_report({c}, curPID()).c_str());
#endif  // defined(_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    }

    return devices;
}

/**
 * v0.1 strata::print_startup_gpu_report
 */
static void print_startup_gpu_report(const std::vector<int>& devices) {
    namespace gr = strata::program::gpu_report;
    std::vector<gr::Card> cards;
    int count = 0, prev = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        (void)cudaGetLastError();
        return;
    }
    if (cudaGetDevice(&prev) != cudaSuccess) {
        (void)cudaGetLastError();
        prev = 0;
    }
    std::vector<int> seen;
    for (const int d : devices) {
        if (d < 0 || d >= count || std::find(seen.begin(), seen.end(), d) != seen.end())
            continue;
        seen.push_back(d);
        if (cudaSetDevice(d) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }  // no context fits there either
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        gr::Card c;
        c.index     = d;
        c.free_mib  = (uint64_t)(free_b >> 20);
        c.total_mib = (uint64_t)(total_b >> 20);
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess)
            c.name = prop.name;
        else
            (void)cudaGetLastError();
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
        char bus[64] = {};
        if (cudaDeviceGetPCIBusId(bus, (int)sizeof bus, d) == cudaSuccess)
            c.processes_known = gr::query_other_processes(bus, c.processes);
        else
            (void)cudaGetLastError();
#endif
        cards.push_back(std::move(c));
    }
    if (prev >= 0 && prev < count)
        (void)cudaSetDevice(prev);
    (void)cudaGetLastError();
    if (cards.empty())
        return;

    std::fprintf(stderr, "strata generate: GPU memory right now (best effort):\n%s\n", gr::format_gpu_memory_report(cards, curPID()).c_str());
}

bool InitCUDNN();
inline int convert_SM_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
                                       {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
                                       {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    _INFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor, nGpuArchCoresPerSM[index - 1].cores);
    return nGpuArchCoresPerSM[index - 1].cores;
}
bool launch_blocking() {
#if defined(USE_HIP) || defined(HIP_GFX906)
    return false;
#else
    const char* v = std::getenv("CUDA_LAUNCH_BLOCKING");
    bool bRet     = v != nullptr && std::atoi(v) != 0;
    if (!bRet) {
        _WARN("CUDA_LAUNCH_BLOCKING=1: the verify windows cannot run with blocking launches\n");
    }

    return bRet;
#endif
}
bool InitCUDA(const CLI_params& hparams, EDGE_DEVICES* hDevice, int flag) {
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    launch_blocking();

    int local_device_idx = 0;  // override_enable_tf32 = 1;
    cudaError_t err      = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[InitCUDA] failed at cudaSetDevice! ERR=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaCheck(cudaGetDeviceProperties(&deviceProp, local_device_idx));
    // int cuda_num_SMs = deviceProp.multiProcessorCount;
    // int cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    // int cuda_arch_major = deviceProp.major;
    // int cuda_arch_minor = deviceProp.minor;

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    // nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cublasCheck(cublasCreate(&cublas_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    // bool enable_tf32 = PARAMS_TYPE == typNUMBER::F32 && deviceProp.major >= 8 && override_enable_tf32;
    // cublas_compute   = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
#ifdef __USE_CUDNN__
    InitCUDNN();
#endif
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    int nCorePP = convert_SM_to_cores(deviceProp.major, deviceProp.minor);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    _INFO("\t君子不器 - \"%s\" \n", deviceProp.name);
    _INFO("\tCUDA driver version: %d.%d, %s runtime version: %d.%d %s\n", driver_version / 1000, (driver_version % 100) / 10, COLOR_ORANGE,
          runtime_version / 1000, (runtime_version % 100) / 10, COLOR_RESET);
    _INFO("\tCUDA capability major/minor version number: %d.%d. ECC=%d\n", deviceProp.major, deviceProp.minor, deviceProp.ECCEnabled);
    _INFO("\t\n");
    _INFO("\t%d multiprocessors, %d CUDA cores/MP, %d CUDA cores\n", deviceProp.multiProcessorCount, nCorePP, nCorePP * deviceProp.multiProcessorCount);

    _INFO("\tGPU max clock rate: %.0f MHz (%0.2f GHz)\n", static_cast<double>(deviceProp.clockRate) * 1e-3, static_cast<double>(deviceProp.clockRate) * 1e-6);
    _INFO("\tPeak bandwidth %.1f GByte/s.\tMemory clock rate: %.0f MHz (%0.2f GHz)\n",
          (double)deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 2 / 1e6, static_cast<double>(deviceProp.memoryClockRate) * 1e-3,
          static_cast<double>(deviceProp.memoryClockRate) * 1e-6);
    _INFO("\tMemory bus width: %d-bit\n", deviceProp.memoryBusWidth);
    _INFO("\tGlobal memory: %.0f MBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalGlobalMem) / 1048576, deviceProp.totalGlobalMem);
    _INFO("\tConstant memory: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalConstMem) / 1024, deviceProp.totalConstMem);
    _INFO("\tShared memory per block: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerBlock) / 1024, deviceProp.sharedMemPerBlock);
    _INFO("\tShared memory per multiprocessor: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerMultiprocessor) / 1024,
          deviceProp.sharedMemPerMultiprocessor);
    _INFO("\tL2 cache size: %.0f KBytes (%d Bytes)\n", static_cast<double>(deviceProp.l2CacheSize) / 1024, deviceProp.l2CacheSize);
    _INFO("\tTotal number of registers available per block: %d\n", deviceProp.regsPerBlock);
    _INFO("\tWarp size: %d, Max number of threads per block: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock);
    _INFO("\tMax number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    _INFO("\tMax dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
    _INFO("\tMax dimension size of a grid size (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    _INFO("+-----------------------+----------------------------------------------------+\n");

    fflush(stdout);
    // Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state
    // available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors.
    hDevice->nCore = deviceProp.multiProcessorCount;

    return true;
}

//  Destroy @EDGE_DEVICES::ClearGPU
bool EDGE_DEVICES::InitGPU(const CLI_params& hparams, int flag) {
    string sTp = hparams.KV({"train", "device"}, "");
    gpus       = EDGE_DEVICES::GPU_::cudaGetAllDevice(flag);
    mostRAM    = 0;
    for (auto gpu : gpus) mostRAM += gpu.total_vram;
#ifdef __USE_CUDA__
    if (gpus.size() == 0)
        return false;

    if (!InitCUDA(hparams, this, flag))
        return false;
#endif
    // K_EXIT(KOIFISH_EXIT_DEBUG);  // Only for testing gurad
    return true;
}

bool EDGE_DEVICES::ClearGPU(int flag) { return true; }

void DestroyCUDNN();

// Global cleanup function
static bool g_cuda_cleaned = false;
void CUDA_cleanup() {
    if (g_cuda_cleaned)
        return;
    // CHECK_LAST_CUDA_ERROR();

    if (cublas_handle != nullptr)
        cublasCheck(cublasDestroy(cublas_handle));

    if (cublaslt_handle != nullptr)
        cublasCheck(cublasLtDestroy(cublaslt_handle));
#ifdef __USE_CUDNN__
    DestroyCUDNN();
#endif
    if (main_stream != nullptr) {
        cudaError_t error = cudaStreamDestroy(main_stream);
        if (error != cudaSuccess) {
            _ERROR("[CUDA ERROR] @CUDA_cleanup_ error=\"%s\" (%s code=%d)\n", cudaGetErrorString(error), cudaGetErrorName(error), error);
            exit(KOIFISH_CUDA_CHECK);
        }
    }

    cudaCheck(cudaDeviceReset());
    g_cuda_cleaned = true;
}

void mem_mark(const char* where) {
    static const char* const env = std::getenv("STRATA_TRACE");
    if (env == nullptr)
        return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long)(free_b >> 20), where);
    // STRATA_TRACE=2: also wait for the device here and name the first step after which it reports an error.  A
    // "sticky" failure (#2013: unspecified launch failure surfacing in a later, unrelated call) is then pinned to
    // the step that ran the faulting work.  Only a diagnostic: it adds a synchronisation per mark.
    if (env[0] == '2') {
        const cudaError_t sync_e = cudaDeviceSynchronize();
        const cudaError_t last_e = cudaGetLastError();
        std::fprintf(stderr, "strata trace: device %s after %s (sync: %s, last error: %s)\n", sync_e == cudaSuccess && last_e == cudaSuccess ? "ok" : "ERROR",
                     where, cudaGetErrorString(sync_e), cudaGetErrorString(last_e));
    }
}
