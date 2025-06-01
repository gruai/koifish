/*
cuBLAS related utils
*/
#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ----------------------------------------------------------------------------
// cuBLAS globals for workspace, handle, settings

// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
extern const size_t cublaslt_workspace_size;
extern void* cublaslt_workspace;
extern cublasComputeType_t cublas_compute;
extern cublasLtHandle_t cublaslt_handle;
extern cublasHandle_t cublas_handle;
// ----------------------------------------------------------------------------
// Error checking

// cuBLAS error checking
void inline cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif // CUBLAS_COMMON_H