/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Log is much complex than printf
 *  \author Yingshi Chen
 */

 #pragma once

#include <iostream>
#include <cstdarg> 

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

void _TIME_INFO(const std::string&info,double fmillis,int flag=0x0);

template <typename T>
void inline PrintT(const char* title,const T *src, int n1,int n2,int n3=1,int n4=1,int flag=0x0){
    if( g_dump_level>0 && flag>=0 ) return;
    const T *cur=src; 
    size_t nElem=(size_t)(n1)*n2*n3*n4,i,nz=0,nEach=2;
	if(nElem==0)	return;
    assert(cur!=nullptr);  
    // if(strlen(title)>0) _INFO("%s\n", title);
    float a1=-FLT_MAX,a0=FLT_MAX,a;    
    double sum = 0.0,len=0.0,sum2=0.0;
    for (i = 0; i < nElem; i++,cur++) {
        // a = float(*cur);	
        a = T2Float<T>(cur);        assert(!isnan(a) && !isinf(a));
        if(i<nEach || i>nElem-nEach || fabs(i-nElem/2)<=nEach)
            _INFO("%g ",a);
        if(i==nEach || i==nElem-nEach)    _INFO("...");
        sum += fabs(a);     sum2+=a*a;
        if(a==0)      nz++;
        a1 = std::max(a1,a);      a0 = std::min(a0,a);
    }
    assert(!isnan(sum2) && !isinf(sum2));
    len = sqrt(sum2/nElem);
    //  printf output is only displayed if the kernel finishes successfully,  cudaDeviceSynchronize()
    _INFO("\t\"%s\" |avg|=%g(%ld) avg_len=%g sum2=%g [%f,%f] nz=%.3g\n",title,sum/nElem,nElem,len,sum2,a0,a1,nz*1.0/nElem);
}
