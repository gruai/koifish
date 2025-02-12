/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */
#pragma once

#include "../kGPT/llmc/cuda_common.h"
#include <typeinfo>
#include <float.h>

extern cudaStream_t main_stream;
extern cudaDeviceProp deviceProp;

template <typename T>
inline float T2Float(T* a0)   {
    float a;        
    if(typeid(T)==typeid(half)){
        a = __half2float(*(half*)a0);
    }else if(typeid(T)==typeid(nv_bfloat16)) {
        a = __bfloat162float(*(nv_bfloat16*)a0);
    }else if(typeid(T)==typeid(float)) {
        a = *a0;
    }else if(typeid(T)==typeid(int)) {
        a = *a0;
    }else{
        assert(0);
    }
    return a;
}

extern int g_dump_level;
template <typename T>
void inline PrintTensor(const char* title,T *src, bool isDevice,int n1,int n2,int n3=1,int n4=1,int flag=0x0){
    if( g_dump_level>0 ) return;

    T *dst=(T*)src; 
    size_t nElem=n1*n2*n3*n4,sz = nElem*sizeof(T),i,nz=0,nEach=3;
    if(isDevice){
        dst = new T[nElem];
        cudaCheck(cudaMemcpyAsync(dst,src, sz, cudaMemcpyDeviceToHost, main_stream));
    }
    T *cur = dst;    
    // if(strlen(title)>0) printf("%s\n", title);
    float a1=-FLT_MAX,a0=FLT_MAX,a;    
    double sum = 0.0,len=0.0,sum2=0.0;
    for (i = 0; i < nElem; i++,cur++) {
        a = T2Float<T>(cur);
        if(i<nEach || i>nElem-nEach || fabs(i-nElem/2)<=nEach)
            printf("%g ",a);
        if(i==nEach || i==nElem-nEach)    printf("...");
        sum += fabs(a);     sum2+=a*a;
        if(a==0)      nz++;
        a1 = std::max(a1,a);      a0 = std::min(a0,a);
    }
    len = sqrt(sum2/nElem);
    printf("\t\"%s\" avg=%g(%ld) avg_len=%g sum2=%g [%f,%f]\n",title,sum/nElem,nElem,len,sum2,a0,a1);
    
    if(isDevice){
        delete[] dst;
    }
}