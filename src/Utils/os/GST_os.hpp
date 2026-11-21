/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Utility related to OS(Platform) 
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>  //for shared_ptr
#include <string>

#if defined(_WIN32)
#include <dbghelp.h>
#include <io.h>
#include <psapi.h>
#include <tlhelp32.h>
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#if !defined(__APPLE__)  // major() / minor(): <sys/types.h> on macOS
#include <sys/sysmacros.h>
#endif
#endif
#include <map>

//  GetCurrentProcessId
uint32_t curPID();

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Pin-by-need: lock several disjoint regions (the holes between the registered ranges of an arena).  The Windows
/// working-set minimum is raised ONCE for the total (lock_resident per region would add its margin every time);
/// stops at the first refusal and reports the bytes locked so far.  Undo each region with unlock_resident.
LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// #243, Windows: the GPU's shared (non-local) memory budget and this process's use of it, from DXGI
/// (IDXGIAdapter3::QueryVideoMemoryInfo, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL) for the adapter whose LUID is the
/// 8 bytes at `luid` (cudaDeviceProp::luid).  Page-locked host memory the GPU maps is charged there.  False (and
/// `why` says so) when the query is not possible - always elsewhere than Windows.
bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why);

/// The machine's physical RAM in bytes (0 when unknown).
uint64_t total_physical_memory();

/// #357/#577: whether the OS file cache could keep the `read_bytes` the expert files are read for, beside
/// `arena_bytes` of RAM held by the engine's own copy of the experts and `margin` for everything else, with `avail`
/// bytes of RAM available.  The file tier passes only the expert bytes it really reads from the files (the experts
/// outside its resident RAM copy) and the RAM that copy really holds - not every shard's bytes and the requested
/// budget, which on a 96 GB PC (#577) made the file tier read unbuffered when the cache could keep its reads.
/// #1194: the file tier's reads are a skewed set (the GPU cache and the RAM copy took the hottest experts; the same
/// few of the rest come back token after token), so the cache does not have to hold all `read_bytes` to serve most of
/// them.  With `hot_fraction` < 1 it is enough that the room holds that share of them, but never less than
/// `floor_bytes` (below it the cache holds nothing worth having) and never more than every read.  The defaults ask for
/// every byte, as the Windows rule does.  Measured on Linux (#1194, a Tesla P100 box under 20 to 32 GB cgroups): through
/// the cache beat the unbuffered reads at every room tried, down to 7% of the read bytes (decode +7% to +118%, a third to
/// a sixth of the drive traffic).
inline bool file_cache_keeps(uint64_t avail, uint64_t arena_bytes, uint64_t read_bytes,
                             uint64_t margin = 4ull << 30, double hot_fraction = 1.0, uint64_t floor_bytes = 0) {
    const uint64_t room = avail > arena_bytes + margin ? avail - arena_bytes - margin : 0;
    uint64_t need = (uint64_t) ((double) read_bytes * hot_fraction);
    if (need < floor_bytes) need = floor_bytes;
    if (need > read_bytes) need = read_bytes;
    return room >= need;
}

/// #3237: with the room rule saying the file cache could keep the experts that come back, but holding only
/// `held_fraction` (< 0.5) of the read set: whether the unbuffered reads win, from a start-up timing of the drive both
/// ways (`direct_bps`: O_DIRECT preads on the I/O threads, `cached_bps`: first touches of a mapping's pages).
/// On a PCIe 5 NVMe (8.2 GB/s direct against 4.9 through the cache, the cache holding 13%) the cached path read prompts at
/// half the speed and decoded 3-4x slower than unbuffered reads (#3237).  On a 1.7-2.5 GB/s NVMe (ext4, notes-021-iD) the
/// cache decoded 1.1-1.6x faster at 10-60% held while direct reads read prompts 1.1-1.4x faster, so a timing ratio alone
/// (2.0x there, 1.66x on the reporter's) cannot separate the two: a miss costs 1/bandwidth, and only on a fast drive is
/// a miss cheap enough that the cache's faults and read-ahead cost more than the hits save.  Hence an absolute floor: the
/// direct path must run at 4 GB/s or more.  Then a cache that holds under 30% needs the direct path 1.3x faster; up to
/// half, 1.5x.
inline bool file_tier_prefers_direct(double held_fraction, double direct_bps, double cached_bps) {
    if (!(direct_bps > 0.0) || !(cached_bps > 0.0)) return false;
    if (held_fraction >= 0.5 || direct_bps < 4.0e9) return false;
    const double ratio = direct_bps / cached_bps;
    return held_fraction < 0.3 ? ratio >= 1.3 : ratio >= 1.5;
}

/// Whether `advise_willneed` asks the OS for anything: not on Windows, nor with STRATA_READ_AHEAD=0.
bool read_ahead_enabled();
/// Asks the OS to start reading [p, p + bytes) of a file mapping, without waiting.  Linux reads at most one
/// readahead window per request, so the range is asked for in 128 KiB steps.
void advise_willneed(const void* p, uint64_t bytes);
/// Issuing a prefetch hint lets the kernel overlap disk I/O with CPU work. For [offset, offset + bytes) of an open file.
void advise_willneed(int fd, uint64_t offset, uint64_t bytes);

/// What the OS says this process read from storage: `read_bytes` (/proc/self/io: bytes fetched from the block layer on
/// its behalf, page faults on a mapping included, page-cache hits not) and `major_faults` (/proc/self/stat).  Linux
/// only; `valid` is false elsewhere.  Take one at the start of a request and one at its end.
struct ProcIo {
    bool valid = false;
    uint64_t read_bytes = 0, major_faults = 0;
};
ProcIo proc_io_sample();

#if !defined(_WIN32)
/*
    A rotational disk is a spinning hard disk drive (HDD) — one with physical platters that rotate on a spindle. Its opposite is a non-rotational disk, i.e.,
   flash-based storage (SSD, NVMe, eMMC, USB flash) which has no moving parts. The Linux kernel exposes this distinction through a file:
   /sys/block/<device>/queue/rotational 1 → rotational (HDD) 0 → non-rotational (SSD / NVMe / flash)
*/
inline int on_rotational_disk(const std::string& path, std::string& dev) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0)
        return -1;
    char sys[64];
    std::snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", (unsigned)major(st.st_dev), (unsigned)minor(st.st_dev));
    std::error_code ec;
    std::filesystem::path p = std::filesystem::canonical(sys, ec);
    for (int up = 0; !ec && up < 2; ++up, p = p.parent_path()) {  // a partition has its disk's queue
        std::ifstream q(p / "queue" / "rotational");
        int r = -1;
        if (q >> r) {
            dev = p.filename().string();
            return r;
        }
    }
    return -1;
}

// #605: the engine's threads in uninterruptible sleep (state D - almost always waiting for the disk), and where
// (wchan): 16 threads in blk_io_schedule is a disk that cannot keep up, not a deadlock in the engine
inline void disk_wait_threads(std::FILE* f) {
    std::map<std::string, int> where;
    int waiting = 0, threads = 0;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/task", ec)) {
        ++threads;
        std::ifstream st(e.path() / "stat");
        std::string line;
        std::getline(st, line);
        const size_t rp = line.rfind(')');
        if (rp == std::string::npos || rp + 2 >= line.size() || line[rp + 2] != 'D')
            continue;
        ++waiting;
        std::ifstream w(e.path() / "wchan");
        std::string wc;
        std::getline(w, wc);
        ++where[wc.empty() || wc == "0" ? "?" : wc];
    }
    if (threads == 0)
        return;
    std::string list;
    for (const auto& [name, n] : where) list += (list.empty() ? " (" : ", ") + name + " x" + std::to_string(n);
    if (!list.empty())
        list += ")";
    std::fprintf(f, "  threads waiting on the disk (state D): %d of %d%s%s\n", waiting, threads, list.c_str(),
                 waiting > 0 ? " - the engine is waiting for the drive: a rotational or failing disk with --ple-io "
                               "direct (#605: --ple-io ram), or a drive too slow for the reads asked of it"
                             : "");
}
#endif

inline bool under_wddm() {
#ifdef _WIN32
    return true;  //  Windows Display Driver Model
#else
    //  /dev/dxg is the DirectX Graphics kernel-mode driver device node exposed by WSL2's GPU virtualization (dxgkrnl). It lets Linux user-space (inside WSL2)
    //  talk to the Windows GPU driver via the /dev/dxg char device — i.e., you're on Linux but under the Windows WDDM driver stack through WSLg / GPU-PV.
    static const bool dxg = std::filesystem::exists("/dev/dxg");
    return dxg;
#endif
}

extern inline void create_dir_if_not_exists(const char* dir) {
    if (dir == NULL) {
        return;
    }
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

template <typename T>
T ENV2VAL(const std::string& var, T default_value = 0, int flag = 0) {
    T a  = default_value;
    if (const char* s = std::getenv(var.c_str())) {
        try {
            a = std::stoi(s);
        } catch (const std::exception&) {
            
        }
    }
    return a;
}

