/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Log is much complex than printf
 *  \author Yingshi Chen
 */

#pragma once

// ANSI Color Codes
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"  // Often used as "orange"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"
#define COLOR_WHITE "\033[37m"
#define COLOR_RESET "\033[0m"
// Bright/Vivid variants (often supported)
#define COLOR_ORANGE "\033[93m"  // Bright Yellow (common "orange")
#define COLOR_BOLD_RED "\x1b[1;31m"

#include <cstdarg>
#include <iostream>

#include "../g_float.hpp"
#define die(msg)                           \
    do {                                   \
        fputs("error: " msg "\n", stderr); \
        exit(1);                           \
    } while (0)
#define die_fmt(fmt, ...)                                 \
    do {                                                  \
        fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); \
        exit(1);                                          \
    } while (0)

/*
    10 levels of dumps, 0-9. 0 is a full dump,The lower the number the more dump.
*/
extern int g_dump_level, g_dump_each, g_dump_sigfigs;
enum DUMP_LEVEL {
    DUMP_NONE      = 0,
    DUMP_DEBUG     = 1,
    DUMP_INFO      = 2,
    DUMP_WARN      = 3,
    DUMP_WARN0     = 30,
    DUMP_ERROR     = 4,
    DUMP_EXCEPTION = 5,
    DUMP_CONT      = 6,  // continue previous log
};
inline bool NOT_DUMP(int t = 0) {
    if (g_dump_level <= t)
        return false;
    return true;
}
inline bool DUMP(int t = 0) { return !NOT_DUMP(t); }

void _LOG(DUMP_LEVEL level, const char* format, ...);

#define _INFO(...) _LOG(DUMP_INFO, __VA_ARGS__)
#define _WARN(...) _LOG(DUMP_WARN, __VA_ARGS__)
#define _WARN0(...) _LOG(DUMP_WARN0, __VA_ARGS__)
#define _ERROR(...) _LOG(DUMP_ERROR, __VA_ARGS__)
// #define _ERROR_exit(...) {_LOG(DUMP_ERROR, __VA_ARGS__);    exit(-1);   }
#define _EXCEPTION(...) _LOG(DUMP_ERROR, __VA_ARGS__)
#define _INFO_IF(...)                     \
    {                                     \
        if (DUMP())                       \
            _LOG(DUMP_INFO, __VA_ARGS__); \
    }

void _TIME_INFO(const std::string& info, double fmillis, int flag = 0x0);

template <typename T>
void inline PrintT(const char* title, const T* src, int n1, int n2, int n3 = 1, int n4 = 1, int flag = 0x0) {
    if (g_dump_level > 0 && flag >= 0)
        return;
    const T* cur = src;
    size_t nElem = (size_t)(n1)*n2 * n3 * n4, i, nz = 0, nEach = g_dump_each;
    if (nElem == 0)
        return;
    assert(cur != nullptr);
    char format[20];
    snprintf(format, sizeof(format), "%%.%dg ", g_dump_sigfigs);
    typNUMBER tpData = TYPE_<T>();
    _INFO("%s<%s>\t", title, K_FLOATS[tpData].name.c_str());
    float a1 = -FLT_MAX, a0 = FLT_MAX, a;
    double sum = 0.0, len = 0.0, sum2 = 0.0;
    if (g_dump_each > 0) {
    }
    for (i = 0; i < nElem; i++) {
        a = T2Float<T>(cur, i);
        assert(!isnan(a) && !isinf(a));
        if (g_dump_each > 0) {
            if (i < nEach || i >= nElem - nEach || fabs(i - nElem / 2) <= nEach) {
                _INFO(format, a);  //  "%.15g "
            }

            if (i == nEach || i == nElem - nEach - 1)
                _INFO("...");
        }
        sum += fabs(a);
        sum2 += a * a;
        if (a == 0)
            nz++;
        a1 = std::max(a1, a);
        a0 = std::min(a0, a);
    }

    assert(!isnan(sum2) && !isinf(sum2));
    len = sqrt(sum2 / nElem);
    //  printf output is only displayed if the kernel finishes successfully,  cudaDeviceSynchronize()
    _INFO(" |avg|=%g(%ld) [%.3e,%.3e] avg_len=%g sum2=%g nz=%.3g\n", sum / nElem, nElem, a0, a1, len, sum2, nz * 1.0 / nElem);
    fflush(stdout);
}
