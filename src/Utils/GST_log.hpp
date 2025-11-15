/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
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
extern int g_dump_level;
enum DUMP_LEVEL {
    DUMP_NONE  = 0,
    DUMP_DEBUG = 1,
    DUMP_INFO  = 2,
    DUMP_WARN  = 3,
    DUMP_WARN0  = 30,
    DUMP_ERROR = 4,
    DUMP_CONT  = 5,  // continue previous log
};
inline bool NOT_DUMP(int t = 0) {
    if (g_dump_level <= t)
        return false;
    return true;
}
inline bool DUMP(int t = 0) { return !NOT_DUMP(t); }

#define MAX_LOG_BUFFER

void GG_log_callback_default(DUMP_LEVEL level, const char* text, void* user_data);

inline void GG_log_head(char* buff, const char* tag = "KOIFISH", int flag = 0x0) {
    const char* file = (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) : __FILE__);
    // sprintf(buff, "[%s %s %d:%ld %s:%d %s] ", tag, curr_time(), get_pid(), get_tid(),
    //             file, __LINE__,__FUNCTION__ );
}
void GG_log_internal_v(DUMP_LEVEL level, const char* format, va_list args);

inline void GG_log_internal(DUMP_LEVEL level, const char* format, ...) {
    va_list args;
    va_start(args, format);
    GG_log_internal_v(level, format, args);
    va_end(args);
}

#define _INFO(...) GG_log_internal(DUMP_INFO, __VA_ARGS__)
#define _WARN(...) GG_log_internal(DUMP_WARN, __VA_ARGS__)
#define _WARN0(...) GG_log_internal(DUMP_WARN0, __VA_ARGS__)
#define _ERROR(...) GG_log_internal(DUMP_ERROR, __VA_ARGS__)
#define _INFO_IF(...)                                \
    {                                                \
        if (DUMP())                                  \
            GG_log_internal(DUMP_INFO, __VA_ARGS__); \
    }

void _TIME_INFO(const std::string& info, double fmillis, int flag = 0x0);

template <typename T>
void inline PrintT(const char* title, const T* src, int n1, int n2, int n3 = 1, int n4 = 1, int flag = 0x0) {
    if (g_dump_level > 0 && flag >= 0)
        return;
    const T* cur = src;
    size_t nElem = (size_t)(n1)*n2 * n3 * n4, i, nz = 0, nEach = 2;
    if (nElem == 0)
        return;
    assert(cur != nullptr);
    // if(strlen(title)>0) _INFO("%s\n", title);
    float sum = 0.0, a1 = -FLT_MAX, a0 = FLT_MAX, a;
    double len = 0.0, sum2 = 0.0;
    for (i = 0; i < nElem; i++) {
        a = T2Float<T>(cur, i);
        assert(!isnan(a) && !isinf(a));
        if (i < nEach || i >= nElem - nEach || fabs(i - nElem / 2) <= nEach)
            _INFO("%g ", a);
        if (i == nEach || i == nElem - nEach -1)
            _INFO("...");
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
    _INFO("\t\"%s\" |avg|=%g(%ld) avg_len=%g sum2=%g [%f,%f] nz=%.3g\n", title, sum / nElem, nElem, len, sum2, a0, a1, nz * 1.0 / nElem);
    fflush(stdout);
}
