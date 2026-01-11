/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Utility functions
 *  \author Yingshi Chen
 */
#include "GST_util.hpp"

#include <sys/resource.h>

#include <cstdio>
#include <filesystem>  // C++17

#include "../g_float.hpp"
#include "GST_log.hpp"
#include "GST_obj.hpp"
#include "GST_os.hpp"

int GST_util::dump = 10;

int GST_util::verify = 0;

void _TIME_INFO(const string& info, double fmillis, int flag) {
    _INFO("%s", info.c_str());
    if (fmillis < 1000.0f) {
        _INFO("%.1fms", (float)fmillis);
        return;
    }
    if (fmillis < 60 * 1000.0f) {  // 60sec
        _INFO("%.2fs", fmillis / 1000);
        return;
    }
    const int64_t one_sec = 1000, one_min = one_sec * 60, one_hour = one_min * 60, one_day = one_hour * 24;

    int64_t millis  = (int64_t)fmillis;
    int64_t days    = millis / one_day;
    int64_t hours   = (millis - days * one_day) / one_hour;
    int64_t minutes = (millis - days * one_day - hours * one_hour) / one_min;
    int64_t seconds = (millis - days * one_day - hours * one_hour - minutes * one_min) / one_sec;

    // to print int64_t either cast to (long long int) or use macro PRId64 from <inttypes.h>
    if (days > 0) {
        _INFO("%lldd ", (long long int)days);
    }
    if (hours == 0 && minutes == 0) {
        _INFO("%02ld", seconds);
    } else
        _INFO("%02lld:%02lld:%02lld", (long long int)hours, (long long int)minutes, (long long int)seconds);
}

int SUM::nUpdateParam    = 0;
int SUM::nMostMemItem    = 6;
int SUM::nMinTensorAlloc = 100 * 1024 * 1024;
double SUM::tX = 0.0, SUM::tX1 = 0.0, SUM::tRemater = 0.0, SUM::tPreLogits = 0.0;
size_t SUM::szUpload = 0;
double SUM::tQKV = 0.0, SUM::tFFN = 0.0, SUM::tUpload = 0.0, SUM::tData = 0.0;
double SUM::tLoadData = 0.0, SUM::tLoadParam = 0.0, SUM::tEval_1 = 0.0, SUM::tEval_0 = 0.0;
int SUM::nInitParam = 0, SUM::nSaveParam = 0, SUM::nLoadParam = 0, SUM::nDogLeg = 0;
int SUM::nzLoadParam = 0, SUM::nzSaveParam = 0;
std::vector<MEM_USAGE> SUM::mems;

int SUM::nQuantTensor   = 0;
size_t SUM::szQuantBits = 0;
double SUM::tQuant = 0, SUM::tF8Ex = 0;
string SUM::sQuantInfo = "";

size_t MEM_USAGE::szA = 0, MEM_USAGE::szW = 0, MEM_USAGE::szG = 0, MEM_USAGE::szMoment = 0, MEM_USAGE::szTemp = 0, MEM_USAGE::szOther = 0;
MEM_USAGE::MEM_USAGE(size_t sz_, string d_, void* hData_, int flag) : sz(sz_), desc(d_), hData(hData_) {
    char last = desc[desc.length() - 1];
    switch (last) {
        case 'a':
            if (desc.substr(0, 3) == "tmp") {
                type = TYPE::TEMP, szTemp += sz;
            } else {
                type = TYPE::ACTIVATION, szA += sz;
            }

            break;
        case 'w':
            type = TYPE::WEIGHT, szW += sz;
            break;
        case 'g':
            type = TYPE::GRAD, szG += sz;
            break;
        case 'm':
            type = TYPE::MOMENT, szMoment += sz;
            break;
        case 't':
            type = TYPE::TEMP, szTemp += sz;
            break;
        default:
            type = TYPE::OTHER, szOther += sz;
            ;
            break;
    }
}
void SUM::Reset(string typ, int flag) {
    if (typ == "time") {
        tX = 0.0, tX1 = 0.0, tRemater = 0.0;
        tQKV = 0.0;
        tFFN = 0.0;
    }
    if (typ == "memory") {
        SUM::szUpload = 0;
        _WARN("All MEM_USAGE(%ld) infos are cleard!", mems.size());
        mems.clear();
    }
}

/*  CUDA version
__device__ static void coopstage(uint64_t* stats, int stage) {
    __shared__ uint64_t lastt;
    if (stats && blockIdx.x == 0 && threadIdx.x == 0) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (stage >= 0) {
            stats[stage] += t - lastt;
        }
        lastt = t;
    }
}
*/
void SUM::TimeInfo(int type, int flag) {
    if (type <= 0)
        return;
    // _TIME_INFO(" R=",SUM::tRemater);
    _TIME_INFO("(data=", tData);
    _TIME_INFO(" QKV=", tQKV);
    _TIME_INFO(" FFN=", tFFN);
    if (tX1 > 0)
        _TIME_INFO(" X=", tX1);
    _INFO(")");
}

std::string SUM::CPU_Info(int flag) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    float mCPU = usage.ru_maxrss / 1000.0, mFreeCPU = 0.0;

    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    uint64_t freeMemoryKB = 0;

    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") != std::string::npos) {
            sscanf(line.c_str(), "MemAvailable: %lu kB", &freeMemoryKB);
            break;
        }
    }
    // return freeMemoryKB * 1024; // Convert KB to bytes
    char buf[1024];
    sprintf(buf, "mCPU=%.6gM(free=%.6gM)", mCPU, freeMemoryKB / 1000.0);
    return buf;
}

bool SUM::FreeMem(void* hObj, int flag) {
    int id = 0;
    for (auto mem : mems) {
        if (mem.hData == hObj) {
            mems.erase(mems.begin() + id);
            return true;
        }
        id++;
    }
    assert(0 && "SUM::FreeMem failed");
    return false;
}
/**
 * 1. cudaMemGetInfo reports:
        Memory allocated by your program (via cudaMalloc, cudaMemcpy, etc.).
        Additional memory reserved by the CUDA runtime (e.g., for kernels, libraries like cuBLAS).
        Example: If your app allocates 500MB, cudaMemGetInfo might report 600MB due to overhead.
    2. nvidia-smi reports:
        Actual physical memory in use by all processes (including other CUDA apps, graphics compositors, etc.).
        Excludes reserved but unused memory (e.g., fragmentation, CUDA context pools)

*/
void SUM::MemoryInfo(int type, int flag) {
    size_t sz0, sz1;
    cudaError_t err = cudaMemGetInfo(&sz0, &sz1);
    std::sort(mems.begin(), mems.end(),  // ugly because we don't have a typedef for the std::pair
              [](const MEM_USAGE& a, const MEM_USAGE& b) { return a.sz > b.sz; });
    size_t szNow = 0, i = 0, szFree = 0;
    double mUsed = (sz1 - sz0) / 1.0e6;
    for (auto mem : mems) {
        szNow += mem.sz;
    }
    szFree = MEM_USAGE::szA + MEM_USAGE::szW + MEM_USAGE::szG + MEM_USAGE::szMoment + MEM_USAGE::szTemp + MEM_USAGE::szOther - szNow;
    _INFO(
        "[MEMORY] Current usage statistics:  %ld mem-blocks sum=%.3gG(%.3gG) \n\tactivation=%.5gM weight=%.5gM grad=%.5gM moments=%.5gM temp=%.5gM "
        "other=%.5gM\n",
        mems.size(), szNow / 1.0e9, szFree / 1.0e9, MEM_USAGE::szA / 1.0e6, MEM_USAGE::szW / 1.0e6, MEM_USAGE::szG / 1.0e6, MEM_USAGE::szMoment / 1.0e6,
        MEM_USAGE::szTemp / 1.0e6, MEM_USAGE::szOther / 1.0e6);
    _INFO("\tcurBrach=%.5gM mUsed==%.5gM\n", (MEM_USAGE::szA + MEM_USAGE::szW + MEM_USAGE::szG + MEM_USAGE::szMoment) / 1.0e6, mUsed);
    int nDump = type == KOIFISH_MISS_MEMBLOCK ? mems.size() : type == KOIFISH_OUTOF_GPUMEMORY ? 32 : nMostMemItem;
    if (nDump > 0) {  // decsend by memory size
        size_t szTotal = 0;
        for (auto mem : mems) {
            szTotal += mem.sz;
            _INFO("\t%ld\t%6gM  @%s \t%.3gG\n", i++, mem.sz / 1.0e6, mem.desc.c_str(), szTotal / 1.0e9);
            if (i > nDump)
                break;
        }
        _INFO("\n");
    }
}

#ifdef _GST_MATLAB_
#pragma message("\t\tMATLAB:	R2012b \r\n")
#include "mat.h"
#include "matrix.h"
#pragma comment(lib, "G:\\MATLAB\\R2011a\\extern\\lib\\win32\\microsoft\\libmat.lib")
#pragma comment(lib, "G:\\MATLAB\\R2011a\\extern\\lib\\win32\\microsoft\\libmx.lib")

int GST_util::LoadDoubleMAT(const char* sPath, const char* sVar, int* nRow, int* nCol, double** val, int flag) {
    MATFile* pMat;
    mxArray *problem, *A;
    mxClassID cls;
    double *Ax, *Az;
    int nz, *Ap, *Ai;

    pMat = matOpen(sPath, "r");
    if (pMat == NULL) {
        printf("Error reopening file %s\n", sPath);
        return -101;
    }
    problem = matGetVariable(pMat, sVar);
    if (problem == NULL) {
        printf("Error reading existing matrix Problem\n");
        return -102;
    }
    cls   = mxGetClassID(problem);
    *nRow = mxGetM(problem);
    *nCol = mxGetN(problem);
    if (*nRow <= 0 || *nCol <= 0) {
        printf("Error get element A\n");
        return -103;
    }
    Ax   = mxGetPr(problem);
    Az   = mxGetPi(problem);
    nz   = (*nRow) * (*nCol);
    *val = new double[nz];
    memcpy(*val, Ax, sizeof(double) * nz);
    mxDestroyArray(problem);
    if (matClose(pMat) != 0) {
        printf("Error closing file %s\n", sPath);
        return -104;
    }

    return 0x0;
}
#endif

//	std::filesystem::exists.
/*extern inline FILE *fexist_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr,
                "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to "
                "dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}*/

bool VERIFY_DIR_EXIST(const std::string& path, bool isCreate) {
    if (path.empty())
        return false;

    bool isExist = false;
    try {
        std::filesystem::path filePath = path;
        std::string dir_path           = filePath.parent_path();
        isExist                        = std::filesystem::exists(dir_path);
        if (!isExist && isCreate) {
            bool created = std::filesystem::create_directories(dir_path);
            if (created) {
                std::cout << "Directory created: " << dir_path << std::endl;
            } else {
                std::cout << "Directory already exists or failed: " << dir_path << std::endl;
            }
        }
        isExist = std::filesystem::exists(dir_path);
        if (!isExist) {
            _WARN("%s is not exist!", path.c_str());
        }
        return isExist;
    } catch (const std::filesystem::filesystem_error& e) {
        _ERROR("\"%s\" is not exist! ERR=%s\n", path.c_str(), e.what());
        return isExist;
    }
}

/**
 *  v0.1    20250918
 */
const char* GRUAI_KOIFISH_APP_NAME = "Koifish-v0.2";
void GRUAI_KOIFISH_VERSION(char* str, int flag = 0x0) {
    char sName[80] = "\0";
    int i, nLen = (int)strlen(GRUAI_KOIFISH_APP_NAME), nFrame = 68, off;
    std::string sDat = DATE(100);
    sprintf(sName, " %s %s %s (%s by gcc %s)", COLOR_ORANGE, GRUAI_KOIFISH_APP_NAME, COLOR_RESET, sDat.c_str(), __VERSION__);
    int pad = (nFrame - strlen(sName)) / 2 - 1;
    assert(pad >= 0);

    sprintf(str, "%s*%*s%s%*s*\n%s", "********************************************************************\n", pad, "", sName,
            (int)(nFrame - strlen(sName) - pad - 2), "",
            "*  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen                  *\n"
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

void GG_log_callback_default(DUMP_LEVEL level, const char* text, void* user_data) {
    (void)level;
    (void)user_data;
    fputs(text, stderr);
    fflush(stderr);
}

static char log_buffer[KOIFISH_MOST_LOG], coloredMsg[KOIFISH_MOST_LOG + 100];
void GG_log_internal_v(DUMP_LEVEL level, const char* format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    GG_log_head(log_buffer);
    //  If>MOST_LOG Output is ​​truncated but valid​​.
    int len = vsnprintf(log_buffer, KOIFISH_MOST_LOG, format, args);
    va_end(args_copy);

    const char* color_0 = "";
    coloredMsg[0]       = '\0';
    switch (level) {
        case DUMP_ERROR:
            snprintf(coloredMsg, sizeof(coloredMsg), "%s error: %s%s", COLOR_RED, log_buffer, COLOR_RESET);
            fflush(stdout);
            break;
        case DUMP_WARN:
            snprintf(coloredMsg, sizeof(coloredMsg), "%s warn: %s%s", COLOR_MAGENTA, log_buffer, COLOR_RESET);
            fflush(stdout);
            break;
        case DUMP_WARN0:
            snprintf(coloredMsg, sizeof(coloredMsg), "%s %s%s", COLOR_MAGENTA, log_buffer, COLOR_RESET);
            fflush(stdout);
            break;
        default:
            snprintf(coloredMsg, sizeof(coloredMsg), "%s", log_buffer);
            break;
    }
    // if(strcmp(coloredMsg,"\n")==0){
    //     int debug = 0;
    // }

    fputs(coloredMsg, stderr);
    // fputs(log_buffer, stderr);
    fflush(stderr);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            // strip newline
            buffer[len - 1] = '\0';
        }
    }
}

std::string FILE2STR(const std::string fPath, int flag) {
    std::string info;
    std::ifstream file(fPath);
    if (!file.is_open()) {
        _ERROR("<<<<<<< FILE2STR failed @%s\n", fPath.c_str());
        return "";
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    info = buffer.str();
    file.close();

    return info;
}

bool STR2FILE(const std::string fPath, const std::string& info, std::ofstream::openmode mode, int flag) {
    std::ofstream file(fPath, mode);
    if (!file.is_open()) {
        _ERROR("STR2FILE failed @%s", fPath.c_str());
        return "";
    }
    file << info << std::endl;

    return true;
}

std::string FILE_EXT(const std::string& path) {
    std::filesystem::path filePath = path;
    std::string sExt               = filePath.extension().string();
    if (sExt.size() > 1)
        sExt = sExt.substr(1);  // substr(1) to remove the dot
    return sExt;
}

std::vector<std::string> FilesOfDir(const std::string& path, const std::vector<std::string>& keys, int flag) {
    std::vector<std::string> files;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cout << "failed to open directory" << std::endl;
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        // Skip . and .. directory entries
        if (filename == "." || filename == "..") {
            continue;
        }
        // if(!keys.empty() && !G_Has_(filename,keys))
        //     continue;
        string sExt = FILE_EXT(filename);
        if (!keys.empty() && !G_Has_(sExt, keys))
            continue;
        files.push_back(path + "/" + filename);
    }
    closedir(dir);
    if (files.empty()) {
        std::cout << "no files found" << std::endl;
        return files;
    }
    std::sort(files.begin(), files.end());

    return files;
}

void PrintQ4(const char* title, const hBITARR src, int n1, int n2, int n3, int n4, int flag) {
    size_t nElem = (size_t)(n1)*n2 * n3 * n4, i, nz = 0, nEach = 2;
    if (nElem == 0)
        return;
    assert(src != nullptr);
    // if(strlen(title)>0) _INFO("%s\n", title);
    float sum = 0.0, a1 = 16, a0 = 0;
    BIT_8 hi, lo;
    double len = 0.0, sum2 = 0.0;
    for (i = 0; i < nElem; i += 2) {
        BIT_8 byte = src[i / 2];
        hi         = (byte >> 4) & 0x0F;
        lo         = byte & 0x0F;
        if (i < nEach || i >= nElem - nEach || fabs(i - nElem / 2) <= nEach)
            _INFO("%d %d ", hi, lo);
        if (i == nEach || i == nElem - nEach - 1)
            _INFO("...");
        sum += fabs(hi + lo);
        sum2 += hi * hi + lo * lo;
        // if (a == 0)
        //     nz++;
        // a1 = std::max(a1, a);
        // a0 = std::min(a0, a);
    }
    assert(!isnan(sum2) && !isinf(sum2));
    len = sqrt(sum2 / nElem);
    //  printf output is only displayed if the kernel finishes successfully,  cudaDeviceSynchronize()
    _INFO("\t\"%s\" |avg|=%g(%ld) avg_len=%g sum2=%g [%f,%f] nz=%.3g\n", title, sum / nElem, nElem, len, sum2, a0, a1, nz * 1.0 / nElem);
    fflush(stdout);
}