/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief CUDA&CUDNN
 *  \author Yingshi Chen
 */

#include "../EDevice.hpp"
#ifdef __USE_GGML__
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "ggml-sycl.h"
#endif

int EDGE_DEVICES::GPU_::MAX_COUNT = 16;  //  16
std::vector<EDGE_DEVICES::GPU_> EDGE_DEVICES::GPU_::cudaGetDevice(int flag) {
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
#endif  // defined(_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
    }

    return devices;
}

bool InitCUDA(const CLI_params &hparams, EDGE_DEVICES *hDevice, int flag);

//  Destroy @EDGE_DEVICES::ClearGPU
bool EDGE_DEVICES::InitGPU(const CLI_params &hparams, int flag) {
    string sTp = hparams.KV({"train", "device"}, "");
    gpus       = EDGE_DEVICES::GPU_::cudaGetDevice(flag);
    mostRAM    = 0;
    for (auto gpu : gpus) mostRAM += gpu.total_vram;
#ifdef __USE_CUDA__
    if (gpus.size() == 0)
        return false;

    if (!InitCUDA(hparams, this, flag))
        return false;
#endif
    return true;
}

/*
void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    _PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < _NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        _ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < _NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        _ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    _PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    _PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        _PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            _ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                _PRINT_DEBUG(" %u", c);
            }
        }
        _PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                _LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}
*/