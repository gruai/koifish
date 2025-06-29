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

int GST_util::dump = 10;

int GST_util::verify = 0;

void _TIME_INFO(const string &info, double fmillis, int flag) {
    _INFO("%s", info.c_str());
    if (fmillis < 1000.0f) {
        _INFO("%.1fms", (float)fmillis);
        return;
    }
    if (fmillis < 60 * 1000.0f) {  // 60sec
        _INFO("%.2f", fmillis / 1000);
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
double SUM::tQKV = 0.0, SUM::tFFN = 0.0, SUM::tUpload = 0.0;
void SUM::Reset(string typ, int flag) {
    if (typ == "time") {
        tX = 0.0, tX1 = 0.0, tRemater = 0.0;
        tQKV = 0.0;
        tFFN = 0.0;
    }
}
void SUM::TimeInfo(int flag) {
    // _TIME_INFO(" R=",SUM::tRemater);
    _TIME_INFO(" QKV=", tQKV);
    _TIME_INFO(" FFN=", tFFN);
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
extern inline FILE *fexist_check(const char *path, const char *mode, const char *file, int line) {
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
}

std::string FILE_EXT(const std::string &path) {
    std::filesystem::path filePath = path;
    std::string sExt               = filePath.extension().string();
    if (sExt.size() > 1)
        sExt = sExt.substr(1);  // substr(1) to remove the dot
    return sExt;
}
