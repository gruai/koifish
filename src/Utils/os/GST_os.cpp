/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some functions related to Operation System
 *
 *  \brief
 *  \author Yingshi Chen
 */
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <dxgi1_4.h>
#include <windows.h>

#include <cstring>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif
#include "../GST_log.hpp"
#include "../GST_util.hpp"
#include "GST_os.hpp"

uint32_t curPID() {
#if defined(_WIN32)
    const uint32_t self = (uint32_t)GetCurrentProcessId();
#else
    const uint32_t self = (uint32_t)getpid();
#endif
    return self;
}

/**
 *  v0.1    20250918
 */
void GRUAI_KOIFISH_VERSION(char* str, int flag = 0x0) {
    char sName[80] = "\0";
    int i, nFrame = 68, off;  //, nLen = (int)strlen(Koifish-v0.4)
    std::string sDat = DATE(100);
    sprintf(sName, " %s %s-v%s %s (%s by gcc %s ENV=\"%s\")", COLOR_ORANGE, K_APP_NAME, K_APP_VERSION, COLOR_RESET, sDat.c_str(), __VERSION__, K_CONDA_ENV);
    int pad = (nFrame - strlen(sName)) / 2 - 1;
    assert(pad >= 0);

    sprintf(str, "%s*%*s%s%*s*\n%s", "********************************************************************\n", pad, "", sName,
            (int)(nFrame - strlen(sName) - pad - 2), "",
            "*  SPDX-FileCopyrightText: Yingshi Chen                  *\n"
            "*  SPDX-License-Identifier: MIT                                    *\n"
            "*  MAIL: gsp.cys@gmail.com                                         *\n"
            "********************************************************************\n");

    // print_build_info();

    /*
        // "\tCompiler-specific macros:%d"
    #ifdef __GNUC__
        std::cout << "GCC/G++ Version: " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
    #endif
    #ifdef _MSC_VER
        std::cout << "MSVC Version: " << _MSC_VER << "\n";
    #endif
    #ifdef __clang__
        std::cout << "Clang Version: " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__ << "\n";
    #endif*/
    return;
}

#if (defined _WINDOWS) || (defined WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    char str_version[1000];
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            GRUAI_KOIFISH_VERSION(str_version);
            _INFO("%s", str_version);
            break;
        case DLL_THREAD_ATTACH:
            break;
        default:
            break;
    }

    return TRUE;
}
#else
// https://stackoverflow.com/questions/22763945/dll-main-on-windows-vs-attribute-constructor-entry-points-on-linux
__attribute__((constructor)) void dllLoad() {
    char str_version[1000];
    GRUAI_KOIFISH_VERSION(str_version, 0x0);
    _INFO("%s", str_version);
    _INFO("\n");
}

__attribute__((destructor)) void dllUnload() {}
#endif

// STRATA_PCIE_BALANCE=1 (opt-in): each verify layer's PCIe count from measured costs (Verifier::set_pcie_balance)
// instead of the probe's fixed share above - the CPU's speed sets the right share as much as the link's.  IQ2_XS,
// one AVX2 pool worker (emulated on an RTX 5090 + 9950X3D): 16.8-17.2 -> 12.8-13.0 ms a round; the 9950X3D's own
// 16 workers: 12.2 either way.  One GPU, no batch slots, no --pcie-frac.
/*auto pcie_balance = [&]() {
    const char* e = std::getenv("STRATA_PCIE_BALANCE");
    return !pcie_given && native_pack && !multi_gpu && o.batch <= 0 && e != nullptr && std::strcmp(e, "1") == 0;
};*/

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) {
        r.note = "nothing to lock";
        return r;
    }
    HANDLE self   = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin  = (SIZE_T)512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T)bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note =
            "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long)(new_min >> 20)) + " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base        = (uint8_t*)p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T)n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long)(off >> 20)) + " of " + std::to_string((unsigned long long)(bytes >> 20)) +
                     " MiB (error " + std::to_string(GetLastError()) + ")";
            r.ok   = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok   = true;
    r.note = "locked " + std::to_string((unsigned long long)(bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    uint64_t total = 0;
    for (const auto& g : regions) total += g.second;
    if (total == 0) {
        r.note = "nothing to lock";
        return r;
    }
    HANDLE self   = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // same growth as lock_resident, but for the sum of the regions and with ONE margin
    const SIZE_T margin  = (SIZE_T)512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T)total + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note =
            "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long)(new_min >> 20)) + " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    for (const auto& g : regions) {
        uint8_t* base = (uint8_t*)g.first;
        for (uint64_t off = 0; off < g.second; off += chunk) {
            const uint64_t n = g.second - off < chunk ? g.second - off : chunk;
            if (!VirtualLock(base + off, (SIZE_T)n)) {
                r.note = "VirtualLock stopped after " + std::to_string((unsigned long long)(r.locked_bytes >> 20)) + " of " +
                         std::to_string((unsigned long long)(total >> 20)) + " MiB (error " + std::to_string(GetLastError()) + ")";
                r.ok   = r.locked_bytes > 0;
                return r;
            }
            r.locked_bytes += n;
        }
    }
    r.ok   = true;
    r.note = "locked " + std::to_string((unsigned long long)(total >> 20)) + " MiB in " + std::to_string(regions.size()) +
             " regions via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0)
        return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk) VirtualUnlock((uint8_t*)p + off, (SIZE_T)(bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) {
        why = "dxgi.dll not found";
        return false;
    }
    using CreateFactory    = HRESULT(WINAPI*)(REFIID, void**);
    const auto create      = (CreateFactory)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**)&factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why     = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr)
            break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage  = info.CurrentUsage;
                ok     = budget > 0;
                why    = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr)
                a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) {
        r.note = "nothing to lock";
        return r;
    }
    if (mlock(p, bytes) != 0) {
        r.note = "mlock failed (raise ulimit -l)";
        return r;
    }
    r.ok           = true;
    r.locked_bytes = bytes;
    r.note         = "mlock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    for (const auto& g : regions) {
        if (g.first == nullptr || g.second == 0)
            continue;
        if (mlock(g.first, g.second) != 0) {
            r.note = "mlock failed (raise ulimit -l)";
            r.ok   = r.locked_bytes > 0;
            return r;
        }
        r.locked_bytes += g.second;
    }
    r.ok   = r.locked_bytes > 0;
    r.note = r.ok ? "mlock of " + std::to_string(regions.size()) + " regions" : "nothing to lock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0)
        munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why            = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t)pages * (uint64_t)page : 0;
}

namespace {
constexpr uint64_t kAdviseStep = 128ull << 10;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled())
        return;
    const long ps      = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t)ps : 4096, end = (uintptr_t)p + bytes;
    for (uintptr_t a = (uintptr_t)p & ~(pg - 1); a < end; a += kAdviseStep)
        (void)madvise((void*)a, (size_t)std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}
/**
 * Issuing a prefetch hint lets the kernel overlap disk I/O with CPU work, hiding latency. For workloads with a known, sequential access pattern (e.g., scanning a large columnar file, streaming a checkpoint, reading a model file before mmap-less
 * I/O),  Without it, each  read() blocks cold on the first page of each region.
 * The default readahead window on Linux is 128 KiB for a block device
 */
void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled())
        return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
#if defined(__APPLE__)  // macOS: F_RDADVISE is the read-ahead hint
    {
        radvisory ra{(off_t)(offset + at), (int)std::min(kAdviseStep, bytes - at)};
        (void)fcntl(fd, F_RDADVISE, &ra);
    }
#else
        (void)posix_fadvise(fd, (off_t)(offset + at), (off_t)std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
#endif
}
#endif

ProcIo proc_io_sample() {
    ProcIo r;
#if defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        unsigned long long v;
        while (std::fscanf(f, "%63[^:]: %llu\n", key, &v) == 2)
            if (std::strcmp(key, "read_bytes") == 0) {
                r.read_bytes = v;
                r.valid      = true;
            }
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n]         = 0;
        std::fclose(f);
        if (const char* p = std::strrchr(buf, ')')) {  // fields after the command: state ppid ... minflt cminflt majflt
            unsigned long long minflt = 0, cminflt = 0, majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %llu %llu %llu", &minflt, &cminflt, &majflt) == 3)
                r.major_faults = majflt;
        }
    }
#endif
    return r;
}