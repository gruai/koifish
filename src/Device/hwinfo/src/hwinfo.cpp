#include <hwinfo/hwinfo.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../pch/gss_internal.h"
#include "../../system/system.h"
#include "hwinfo/utils/PCIMapper.h"

#if defined(GGML_USE_CUDA)
#include "../../cuda/ggml-cuda.h"
void GRUS_Get_GPU_Info() { int nDevice = ggml_backend_cuda_get_device_count(); }
#else
void GRUS_Get_GPU_Info() {
    // assert(0);
}
#endif

/**
 * 1.   Logical cores != physical cores because of Intel's hyperthreading and AMD's Simultaneous Multi Threading (SMT).
 */
extern "C" void GRUS_Get_SystemInfo() {
    GRUS_Get_GPU_Info();

#ifdef _LIMIT_GSS_NO_LAPACK
    G_SYS_INFO.LAPACK_NB = 16;
#else
    G_SYS_INFO.LAPACK_NB = LAPACK_ILAENV(&ispec, name, UPLO, &N, &i, &i, &i);  // get LAPACK--dependent blocksize
#endif
    G_SYS_INFO.LAPACK_NB = MAX(16, G_SYS_INFO.LAPACK_NB);
    ASSERT(G_SYS_INFO.LAPACK_NB > 0);

    INT_63 dump = (INT_63)(GRUS_CONTROL[GRUS_DUMP]);
    bool isDump = BIT_IS(dump, GRUS_DUMP_ALL);
    isDump      = false;
    // std::cout << std::endl << "Hardware Report:" << std::endl << std::endl;
    if (isDump)
        std::cout << "----------------------------------- CPU -----------------------------------" << std::endl;
    auto cpus             = hwinfo::getAllCPUs();
    G_SYS_INFO.nCores     = 0;
    G_SYS_INFO.nLogiCores = 0;
    for (const auto &cpu : cpus) {
        if (isDump) {
            std::cout << "Socket " << cpu.id() << ":\n";
            std::cout << std::left << std::setw(20) << " vendor:";
            std::cout << cpu.vendor() << std::endl;
            std::cout << std::left << std::setw(20) << " model:";
            std::cout << cpu.modelName() << std::endl;
            std::cout << std::left << std::setw(20) << " physical cores:";
            std::cout << cpu.numPhysicalCores() << std::endl;
        }
        G_SYS_INFO.nCores += cpu.numPhysicalCores();
        G_SYS_INFO.nLogiCores += cpu.numLogicalCores();
        if (isDump) {
            std::cout << std::left << std::setw(20) << " logical cores:";
            std::cout << cpu.numLogicalCores() << std::endl;
            std::cout << std::left << std::setw(20) << " max frequency:";
            std::cout << cpu.maxClockSpeed_MHz() << std::endl;
            std::cout << std::left << std::setw(20) << " regular frequency:";
            std::cout << cpu.regularClockSpeed_MHz() << std::endl;
            std::cout << std::left << std::setw(20) << " cache size (L1, L2, L3): ";
            std::cout << cpu.L1CacheSize_Bytes() << ", " << cpu.L2CacheSize_Bytes() << ", " << cpu.L3CacheSize_Bytes() << std::endl;
            auto threads_utility = cpu.threadsUtilisation();
            auto threads_speed   = cpu.currentClockSpeed_MHz();
            for (int thread_id = 0; thread_id < threads_utility.size(); ++thread_id) {
                std::cout << std::left << std::setw(20) << "   Thread " + std::to_string(thread_id) + ": ";
                std::cout << threads_speed[thread_id] << " MHz (" << threads_utility[thread_id] * 100 << "%)" << std::endl;
            }
        }
        // std::cout << cpu.currentTemperature_Celsius() << std::endl;
    }
    if (!isDump) {
        return;
    }
    hwinfo::OS os;
    std::cout << "----------------------------------- OS ------------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "Operating System:";
    std::cout << os.name() << std::endl;
    std::cout << std::left << std::setw(20) << "version:";
    std::cout << os.version() << std::endl;
    std::cout << std::left << std::setw(20) << "kernel:";
    std::cout << os.kernel() << std::endl;
    std::cout << std::left << std::setw(20) << "architecture:";
    std::cout << (os.is32bit() ? "32 bit" : "64 bit") << std::endl;
    std::cout << std::left << std::setw(20) << "endianess:";
    std::cout << (os.isLittleEndian() ? "little endian" : "big endian") << std::endl;

    auto gpus = hwinfo::getAllGPUs();
    std::cout << "----------------------------------- GPU -----------------------------------" << std::endl;
    for (auto &gpu : gpus) {
        std::cout << "GPU " << gpu.id() << ":\n";
        std::cout << std::left << std::setw(20) << "  vendor:";
        std::cout << gpu.vendor() << std::endl;
        std::cout << std::left << std::setw(20) << "  model:";
        std::cout << gpu.name() << std::endl;
        std::cout << std::left << std::setw(20) << "  driverVersion:";
        std::cout << gpu.driverVersion() << std::endl;
        std::cout << std::left << std::setw(20) << "  memory [MiB]:";
        std::cout << static_cast<double>(gpu.memory_Bytes()) / 1024.0 / 1024.0 << std::endl;
        std::cout << std::left << std::setw(20) << "  frequency:";
        std::cout << gpu.frequency_MHz() << std::endl;
        std::cout << std::left << std::setw(20) << "  cores:";
        std::cout << gpu.num_cores() << std::endl;
        std::cout << std::left << std::setw(20) << "  vendor_id:";
        std::cout << gpu.vendor_id() << std::endl;
        std::cout << std::left << std::setw(20) << "  device_id:";
        std::cout << gpu.device_id() << std::endl;
    }

    hwinfo::Memory memory;
    std::cout << "----------------------------------- RAM -----------------------------------" << std::endl;
    std::cout << std::left << std::setw(20) << "size [MiB]:";
    std::cout << memory.total_Bytes() / 1024 / 1024 << std::endl;
    std::cout << std::left << std::setw(20) << "free [MiB]:";
    std::cout << memory.free_Bytes() / 1024 / 1024 << std::endl;
    std::cout << std::left << std::setw(20) << "available [MiB]:";
    std::cout << memory.available_Bytes() / 1024 / 1024 << std::endl;
    for (auto &module : memory.modules()) {
        std::cout << "RAM " << module.id << ":\n";
        std::cout << std::left << std::setw(20) << "  vendor:";
        std::cout << module.vendor << std::endl;
        std::cout << std::left << std::setw(20) << "  model:";
        std::cout << module.model << std::endl;
        std::cout << std::left << std::setw(20) << "  name:";
        std::cout << module.name << std::endl;
        std::cout << std::left << std::setw(20) << "  serial-number:";
        std::cout << module.serial_number << std::endl;
        std::cout << std::left << std::setw(20) << "  Frequency [MHz]:";
        std::cout << module.frequency_Hz / 1000 / 1000 << std::endl;
    }
    /*
        hwinfo::MainBoard main_board;
        std::cout << "------------------------------- Main Board --------------------------------" << std::endl;
        std::cout << std::left << std::setw(20) << "vendor:";
        std::cout << main_board.vendor() << std::endl;
        std::cout << std::left << std::setw(20) << "name:";
        std::cout << main_board.name() << std::endl;
        std::cout << std::left << std::setw(20) << "version:";
        std::cout << main_board.version() << std::endl;
        std::cout << std::left << std::setw(20) << "serial-number:";
        std::cout << main_board.serialNumber() << std::endl;

        std::vector<hwinfo::Battery> batteries = hwinfo::getAllBatteries();
        std::cout << "------------------------------- Batteries ---------------------------------" << std::endl;
        if (!batteries.empty()) {
            int battery_counter = 0;
            for (auto& battery : batteries) {
            std::cout << "Battery " << battery_counter++ << ":" << std::endl;
            std::cout << std::left << std::setw(20) << "  vendor:";
            std::cout << battery.vendor() << std::endl;
            std::cout << std::left << std::setw(20) << "  model:";
            std::cout << battery.model() << std::endl;
            std::cout << std::left << std::setw(20) << "  serial-number:";
            std::cout << battery.serialNumber() << std::endl;
            std::cout << std::left << std::setw(20) << "  charging:";
            std::cout << (battery.charging() ? "yes" : "no") << std::endl;
            std::cout << std::left << std::setw(20) << "  capacity:";
            std::cout << battery.capacity() << std::endl;
            }
            std::cout << "---------------------------------------------------------------------------" << std::endl;
        } else {
            std::cout << "No Batteries installed or detected" << std::endl;
        }*/
    /*
        std::vector<hwinfo::Disk> disks = hwinfo::getAllDisks();
        std::cout << "--------------------------------- Disks -----------------------------------" << std::endl;
        if (!disks.empty()) {
            int disk_counter = 0;
            for (const auto& disk : disks) {
            std::cout << "Disk " << disk_counter++ << ":" << std::endl;
            std::cout << std::left << std::setw(20) << "  vendor:";
            std::cout << disk.vendor() << std::endl;
            std::cout << std::left << std::setw(20) << "  model:";
            std::cout << disk.model() << std::endl;
            std::cout << std::left << std::setw(20) << "  serial-number:";
            std::cout << disk.serialNumber() << std::endl;
            std::cout << std::left << std::setw(20) << "  size:";
            std::cout << disk.size_Bytes() << std::endl;
            }
            std::cout << "---------------------------------------------------------------------------" << std::endl;
        } else {
            std::cout << "No Disks installed or detected" << std::endl;
        }*/

    return;
}

/*		lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         48 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  16
  On-line CPU(s) list:   0-15
Vendor ID:               AuthenticAMD
  Model name:            AMD Ryzen 9 5900HX with Radeon Graphics
    CPU family:          25
    Model:               80
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           1
    Stepping:            0
    Frequency boost:     enabled
    CPU max MHz:         3300.0000
    CPU min MHz:         1200.0000
    BogoMIPS:            6587.54
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse ss
                         e2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid
                          extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt ae
                         s xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowpr
                         efetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat
                         _l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid
                          cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_oc
                         cup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv
                         svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic
                         v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smc
                         a fsrm
Virtualization features:
  Virtualization:        AMD-V
Caches (sum of all):
  L1d:                   256 KiB (8 instances)
  L1i:                   256 KiB (8 instances)
  L2:                    4 MiB (8 instances)
  L3:                    16 MiB (1 instance)
NUMA:
  NUMA node(s):          1
  NUMA node0 CPU(s):     0-15
Vulnerabilities:
  Gather data sampling:  Not affected
  Itlb multihit:         Not affected
  L1tf:                  Not affected
  Mds:                   Not affected
  Meltdown:              Not affected
  Mmio stale data:       Not affected
  Retbleed:              Not affected
  Spec rstack overflow:  Mitigation; safe RET, no microcode
  Spec store bypass:     Mitigation; Speculative Store Bypass disabled via prctl and seccomp
  Spectre v1:            Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:            Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP always-on, RSB filling, PBRSB-eIBRS No
                         t affected
  Srbds:                 Not affected
  Tsx async abort:       Not affecte
  */

/*
    Logical cores != physical cores because of Intel's hyperthreading and AMD's Simultaneous Multi Threading (SMT).

int GRUS_Get_NumCores() {
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    //https://stackoverflow.com/questions/150355/programmatically-find-the-number-of-cores-on-a-machine
    int n1 = sysinfo.dwNumberOfProcessors;
    DWORD length = 0;
    const BOOL result_first = GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    assert(GetLastError() == ERROR_INSUFFICIENT_BUFFER);

    std::unique_ptr< std::byte[] > buffer(new std::byte[length]);
    const PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            reinterpret_cast< PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX >(buffer.get());

    const BOOL result_second = GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length);
    assert(result_second != FALSE);

    size_t nb_physical_cores = 0;
    size_t offset = 0;
    do {
        const PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX current_info =
            reinterpret_cast< PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX >(buffer.get() + offset);
        offset += current_info->Size;
        ++nb_physical_cores;
    } while (offset < length);

    int n2 = nb_physical_cores;
    return n2;
#elif MACOS
    int nm[2];
    size_t len = 4;
    uint32_t count;

    nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
    sysctl(nm, 2, &count, &len, NULL, 0);

    if(count < 1) {
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        if(count < 1) { count = 1; }
    }
    return count;
#else
    return sysconf(_SC_NPROCESSORS_ONLN);
#endif
}*/
