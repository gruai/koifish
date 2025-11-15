#pragma once

#include <cassert>
#include <complex>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../../Utils/BLAS_t.hpp"
#include "../../g_stddef.hpp"
#include "../SpMV/GVMAT.h"
#include "../SpMV/Matrix.hpp"
// #include "lapacke.h"
using namespace std;

int PerSVD(char* filename, int m, int n, int k, int l, int p, float* Mf, float* mU, int ldU, float* sigma, float* mVt, int ldV, int flag = 0x0);
/**
 * SVD: M = U*SIGMA*V'
 * https://www.netlib.org/lapack/lapack-3.1.1/html/dgesvd.f.html
 * where SIGMA is an M-by-N matrix which is zero except for its*  min(m,n) diagonal elements,
 * U is an M-by-M orthogonal matrix, and V is an N-by-N orthogonal matrix.
 * The singular value decomposition is not unique! For example, if A=U*S*V^T then(-U, S, -V) is also an SVD decomposition. *
 */
template <typename T>
class LoSVD : public Matrix<T> {
   protected:
    int rankMode, rank = -1, nHeavy = -1;
    double TOL = 0, trace_0 = 0, traceHeavy = 0;
    char JOBU = 'A', JOBVT = 'A';
    T* mU     = nullptr;  //	(LDU,M)  contains the M-by-M orthogonal matrix U;
    T* sigma  = nullptr;  //	min(M,N)  sorted so that S(i) >= S(i+1).
    T* mVt    = nullptr;  // (LDVT,N) contains the N-by-N orthogonal matrix VT;
    T *approx = nullptr, *work = nullptr;
    int ldU = -1, ldV = -1;

    //[5120x5120] nHeavy=4(0/0) SIGMA={12.7609,...,25.0786} OFF=0.825211(|A|=60.464),tX=0.214
    int SVD_(int flag = 0x0) {  // int m, int n, int k, int l, int p, mat **U, mat **S, mat **V)
        int m = this->nRow, n = this->nCol, l = nHeavy, iRet = -1, i, p;
        size_t nzU = m * nHeavy, nzV = ldV * nHeavy;
#ifdef _USE_OPENBLAS_
        iRet = PerSVD(NULL, m, n, nHeavy, nHeavy + nHeavy / 2, 2, this->val, mU, ldU, sigma, mVt, ldV);
#endif
        switch (this->tpOut) {
            case typNUMBER::F8E5M2:
                float_to_fp8e5m2(nzU, (float*)mU, (f8e5*)mU);
                float_to_fp8e5m2(nzV, (float*)mVt, (f8e5*)mVt);
                break;
            default:
                break;
        }
        assert(isValidF(nzU, mU));
        assert(isValidF(nzV, mVt));
        return iRet;
    }

    float rHeavy() {
        assert(sigma[nHeavy - 1] > sigma[0]);
        float r = sigma[nHeavy - 1] / sigma[0];
        return r;
    }
    //[5120x5120] nHeavy=4(65.4239/1664.51) SIGMA={25.08,...,12.8479} OFF=0.824088(|A|=60.464),tX=1007.9
    int SVD_0(int flag = 0x0) {
#ifdef _USE_OPENBLAS_
        T one_ = 1.0, fuyi_ = -1.0, zero_ = 0.0;
        int m = this->nRow, n = this->nCol, iRet = -1, i;
        T* A = new T[m * n];  // the contents of A are destroyed!!!
        memcpy(A, this->val, sizeof(T) * m * n);
        // iRet = LAPACKE_sgesvd( LAPACK_COL_MAJOR, JOBU, JOBVT, m, n, A, n, sigma, mU, ldU, mVt, ldV, work );
        // too slow!!!  912sec for 5120x5120
        iRet = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, JOBU, JOBVT, m, n, A, n, sigma, mU, ldU, mVt, ldV, work);
        if (iRet != 0)
            return iRet;
        for (int j = 0; j < rank; j++) {
            trace_0 += fabs(sigma[j]);
        }
        double delta = this->val[0];
        for (int j = 0; j < rank; j++) {  // only for debug
            delta -= mU[j] * sigma[j] * mVt[j * ldV];
        }
        // nHeavy = rank;	//only for debug
        if (nHeavy > 0) {
            for (int j = 0; j < nHeavy; j++) {
                traceHeavy += fabs(sigma[j]);
            }
        } else {
            nHeavy = rank;
            if (TOL > 0) {
                for (i = 0; i < rank; i++) {
                    if (fabs(sigma[i]) < TOL) {
                        nHeavy = i;
                        break;
                    }
                }
            }
        }

        if (nHeavy < rank) {
            T *rU = new T[m * nHeavy], *rV = new T[nHeavy * ldV];
            for (i = 0; i < m; i++) {
                COPY(nHeavy, mU + i * ldU, rU + i * nHeavy);
            }
            ldU = nHeavy;
            memcpy(rV, mVt, sizeof(T) * ldV * nHeavy);
            FREE_a(mU);
            FREE_a(mVt);
            mU  = rU;
            mVt = rV;
            Approx();
        }
        return iRet;
#else
        return -1;
#endif
    }
    void _dump(int type, int flag = 0x0) {
#ifdef _USE_OPENBLAS_
        T* A = this->val;
        Approx(flag);
        int len    = this->nRow * this->nCol;
        double res = 0, nrmA = NRM2(len, A), delta = A[0];
        for (int i = 0; i < len; i++) {
            delta = A[i] - approx[i];
            res += delta * delta;
        }
        res = sqrt(res) / nrmA;
        // assert(res<1.0e-5*nrmA);

        GST_util::print("\tSVD@\"%s\" [%dx%d] nHeavy=%d(%g/%g) SIGMA={%g,...,%g} OFF=%g(|A|=%g),tX=%g\r\n", this->name.c_str(), this->nRow, this->nCol, nHeavy,
                        traceHeavy, trace_0, sigma[0], sigma[nHeavy - 1], res, nrmA, this->perf.tX);
#endif
    }

    int QB_(int flag = 0x0) { return -1; }

   public:
    LoSVD(const std::string nam_, T* A, int m, int n, int k, float tol_, typNUMBER tpN_ = typNUMBER::F32, int flag = 0x0)
        : Matrix<T>(m, n, A), nHeavy(k), TOL(tol_) {
        this->name  = nam_;
        this->tpOut = tpN_;
        if (k <= 0) {
            rankMode = 0;
            k        = min(m, n);
        } else {
            rankMode = 1;
        }
        int ld = 2 * max(3 * min(m, n) + max(m, n), 5 * min(m, n));
        work   = new T[ld];
        rank   = min(m, n);
        ldU    = m;
        ldV    = n;
        if (flag == 0x100) {
            mU  = new T[m * ldU];  //	(LDU,M)  contains the M-by-M orthogonal matrix U;
            mVt = new T[n * ldV];  //	(LDVT,N) contains the N-by-N orthogonal matrix VT;
        } else {
            ldU = nHeavy;
            ldV = n;
            mU  = new T[m * nHeavy];
            mVt = new T[nHeavy * ldV];
        }
        sigma = new T[rank];
    }
    virtual ~LoSVD() {
        FREE_a(mU);
        FREE_a(sigma);
        FREE_a(mVt);
        FREE_a(work);
        FREE_a(approx);
    }

    void US(T* US_, int flag = 0x0) {
        int m = this->nRow, lda = this->nCol, n = this->nCol, k = nHeavy;
        T one_ = 1.0;
        GEMD(charN, m, k, &one_, mU, ldU, sigma, US_, k);
    }
    T* U() { return mU; }
    T* V() { return mVt; }
    T* S() { return sigma; }
    T* Approx(int flag = 0x0) {
#ifdef _USE_OPENBLAS_
        if (approx == nullptr) {
            int j, m = this->nRow, lda = this->nCol, n = this->nCol, k = nHeavy;
            T *US = new float[m * k], *USV = new float[m * n], *A = this->val, one_ = 1.0, fuyi_ = -1.0, zero_ = 0.0;

            GEMD(charN, m, k, &one_, mU, ldU, sigma, US, k);
            // for(j=0;j<n;j++)	{	//only for debug
            // 	delta -= US[j]*mVt[j*ldV];
            // }
            // GEMM( charN,charN,i1-i,nRow,1,&fuyi_,pB,ldW,pA,ldW,&one_,pC,ldW );
            GEMM(charN, charN, n, m, k, &one_, mVt, ldV, US, k, &zero_, USV, n);
            approx = USV;
            delete[] US;
        } else {
        }

        return approx;
#else
        return nullptr;
#endif
    }

    bool Build(int flag = 0x0) override {
        GST_TIC(t_0);
        int m = this->nRow, n = this->nCol, iRet = -1;
        iRet = SVD_(flag);  // or QB_
        if (iRet != 0x0) {
            return false;
        }
        this->perf.tX += GST_TOC(t_0);
        _dump(0);

        return true;
    }

    friend class SLP;
};
typedef shared_ptr<LoSVD<float>> hLSVD_f;
