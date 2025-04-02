/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Log is much complex than printf
 *  \author Yingshi Chen
 */
#include <iostream>
#include <cstdarg> 
#pragma once

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

/*
    10 levels of dumps, 0-9. 0 is a full dump,The lower the number the more dump.
*/  
extern int g_dump_level;
enum DUMP_LEVEL {
    DUMP_NONE  = 0,
    DUMP_DEBUG = 1,
    DUMP_INFO  = 2,
    DUMP_WARN  = 3,
    DUMP_ERROR = 4,
    DUMP_CONT  = 5, // continue previous log
};
inline bool NOT_DUMP(int t=0) {
    if(g_dump_level<=t)    return false;
    return true;
}
inline bool DUMP(int t=0) {
    return !NOT_DUMP(t);
}

#define MAX_LOG_BUFFER 


inline void GG_log_callback_default(DUMP_LEVEL level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

inline void GG_log_internal_v(DUMP_LEVEL level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[256];
    int len = vsnprintf(buffer, 256, format, args);
    if (len < 256) {
        GG_log_callback_default(level, buffer, nullptr);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        GG_log_callback_default(level, buffer2, nullptr);
        delete[] buffer2;
    }
    va_end(args_copy);
}

inline void GG_log_internal(DUMP_LEVEL level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    GG_log_internal_v(level, format, args);
    va_end(args);
}

#define _INFO(...)  GG_log_internal(DUMP_INFO , __VA_ARGS__)
#define _WARN(...)  GG_log_internal(DUMP_WARN , __VA_ARGS__)
#define _ERROR(...) GG_log_internal(DUMP_ERROR, __VA_ARGS__)
#define _INFO_IF(...)   {   if(DUMP())  GG_log_internal(DUMP_INFO , __VA_ARGS__);}