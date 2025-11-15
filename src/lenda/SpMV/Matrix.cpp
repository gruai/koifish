/**
 *  SPDX-FileCopyrightText: 2013-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "Matrix.hpp"

#include <memory>

#include "../util/BLAS_t.hpp"

template <>
hGVEC MatCCS<D>::Transform(hGVEC &vecY, const hGVEC &vecX, int flag) {
    G_DATATYPE dt = vecX->DataType();
    if (dt == FLOAT_D)
        Ax<double>(vecY, vecX, flag);
    else if (dt == FLOAT_Z)
        Ax<COMPLEXd>(vecY, vecX, flag);
    else
        throw runtime_error("MatCCS::Transform");
    return vecY;
}

template <>
hGMAT &MatCCS<D>::Transform(hGMAT &mV, const hGMAT &mU, int flag) {  // V=AU
    G_DATATYPE dt = mU->DataType();
    if (dt == FLOAT_D)
        AU<double>(mV, mU, flag);
    else if (dt == FLOAT_Z)
        AU<COMPLEXd>(mV, mU, flag);
    else
        throw runtime_error("MatCCS::Transform");
    return mV;
}

/*
     the tridiagonal matrix derived from standard central difference of the 1-d convection diffusion operator - u" + rho*u'
     on the interval [0, 1] with zero Dirichlet boundary condition.
*/
void TRI_1_Z(int dim, int **colptr, int **rowind, Z **val, Z **rhs) {
    int nz = dim * 3 - 2, i, pos = 0;
    if (rhs != nullptr)
        *rhs = nullptr;
    *rowind = static_cast<int *>(operator new[](sizeof(int) * nz));
    *val    = static_cast<Z *>(operator new[](sizeof(Z) * nz));
    *colptr = static_cast<int *>(operator new[](sizeof(int) * (dim + 1)));
    Z rho = Z(10, 0), h = 1.0 / Z(dim + 1, 0.0), h2 = h * h, s = rho / 2.0, sigma = 0.0;
    Z s1 = -1.0 / h2 - s / h, s2 = 2.0 / h2 - sigma, s3 = -1.0 / h2 + s / h;
    *colptr[0] = 0;
    for (i = 0; i < dim; i++) {
        if (i > 0) {
            (*rowind)[pos] = i - 1;
            (*val)[pos++]  = s3;
        }
        (*rowind)[pos] = i;
        (*val)[pos++]  = s2;
        if (i < dim - 1) {
            (*rowind)[pos] = i + 1;
            (*val)[pos++]  = s1;
        }
        (*colptr)[i + 1] = pos;
    }
    assert(pos == nz);
}