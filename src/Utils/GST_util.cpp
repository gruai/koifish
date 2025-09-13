/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Utility functions
 *  \author Yingshi Chen
 */

#include "GST_util.hpp"

#include <filesystem>  // C++17

#include "../g_float.hpp"
#include "GST_log.hpp"
#include "GST_os.hpp"

int GST_util::dump = 10;

int GST_util::verify = 0;

void _TIME_INFO(const string &info, double fmillis, int flag) {
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

double SUM::tX = 0.0, SUM::tX1 = 0.0, SUM::tRemater = 0.0;
size_t SUM::szUpload = 0;
double SUM::tQKV = 0.0, SUM::tFFN = 0.0, SUM::tUpload = 0.0;
double SUM::tLoadData = 0.0, SUM::tLoadParam = 0.0, SUM::tEval_1 = 0.0, SUM::tEval_0 = 0.0;
int SUM::nInitParam = 0, SUM::nSaveParam = 0, SUM::nLoadParam = 0, SUM::nDogLeg = 0;
std::vector<MEM_USAGE> SUM::mems;

size_t MEM_USAGE::szA = 0, MEM_USAGE::szW = 0, MEM_USAGE::szG = 0, MEM_USAGE::szMoment = 0, MEM_USAGE::szTemp = 0, MEM_USAGE::szOther = 0;
MEM_USAGE::MEM_USAGE(size_t sz_, string d_, void *hData_, int flag) : sz(sz_), desc(d_),hData(hData_) {
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
            type = TYPE::OTHER, szA += sz;
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
        mems.clear();
    }
}
void SUM::TimeInfo(int flag) {
    // _TIME_INFO(" R=",SUM::tRemater);
    _TIME_INFO(" QKV=", tQKV);
    _TIME_INFO(" FFN=", tFFN);
    if (tX1 > 0)
        _TIME_INFO(" X=", tX1);
}

bool SUM::FreeMem(void *hObj,int flag){
    int id=0;
    for(auto mem:mems){
        if(mem.hData == hObj){
            mems.erase(mems.begin() + id);
            return true;
        }
        id++;
    }
    assert(0 && "SUM::FreeMem failed");
    return false;
}
void SUM::MemoryInfo(int type, int flag) {
    std::sort(mems.begin(), mems.end(),  // ugly because we don't have a typedef for the std::pair
              [](const MEM_USAGE &a, const MEM_USAGE &b) { return a.sz > b.sz; });
    size_t szTotal = 0, i = 0;
    for (auto mem : mems) {
        szTotal += mem.sz;
    }
    _INFO("%ld mem-blocks \tTotal=%.3gG activation=%.5gM weight=%.5gM grad=%.5gM moments=%.5gM temp=%.5gM other=%.5gM\n", mems.size(), szTotal / 1.0e9,
          MEM_USAGE::szA / 1.0e6, MEM_USAGE::szW / 1.0e6, MEM_USAGE::szG / 1.0e6, MEM_USAGE::szMoment / 1.0e6, MEM_USAGE::szTemp / 1.0e6,
          MEM_USAGE::szOther / 1.0e6);
    if (1) {  // decsend by memory size
        szTotal = 0;
        for (auto mem : mems) {
            szTotal += mem.sz;
            _INFO("\t%ld\t%6gM  @%s \t%.3gG\n", i++, mem.sz / 1.0e6, mem.desc.c_str(), szTotal / 1.0e9);
            if (i > 100)
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

int GST_util::LoadDoubleMAT(const char *sPath, const char *sVar, int *nRow, int *nCol, double **val, int flag) {
    MATFile *pMat;
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

bool VERIFY_DIR_EXIST(const std::string &path, bool isCreate) {
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
        return isExist;
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return isExist;
    }
}

std::string FILE_EXT(const std::string &path) {
    std::filesystem::path filePath = path;
    std::string sExt               = filePath.extension().string();
    if (sExt.size() > 1)
        sExt = sExt.substr(1);  // substr(1) to remove the dot
    return sExt;
}

const char *GRUAI_KOIFISH_APP_NAME = "Koifish-alpha";
void GRUAI_KOIFISH_VERSION(char *str, int flag = 0x0) {
    char sName[80] = "\0";
    int i, nLen = (int)strlen(GRUAI_KOIFISH_APP_NAME), nFrame = 68, off;
    std::string sDat = DATE(100);
    sprintf(sName, "%s (%s by gcc %s)", GRUAI_KOIFISH_APP_NAME, sDat.c_str(), __VERSION__);
    int pad = (nFrame - strlen(sName)) / 2 - 1;
    assert(pad >= 0);
    // for (i = 0; i < nFrame; i++) sName[i] = i == 0 || i == nFrame - 1 ? '*' : ' ';
    // sName[nFrame]     = '\n';
    // sName[nFrame + 1] = '\0';
    // off               = (nFrame - 2 - nLen) / 2 + 1;
    // for (i = 0; i < nLen; i++) sName[i + off] = GRUAI_KOIFISH_APP_NAME[i];

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