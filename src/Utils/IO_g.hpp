#pragma once

#include "../SpMV/Matrix.hpp"

int GBIN_Read( char *sBinPath,int* nrow,int* ncol,int* nnz,int** colptr,int** rowind,double** g_val,double**rhs,int flag );