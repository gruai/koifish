/**
 *  SPDX-FileCopyrightText: 2013-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <float.h>

#include <cassert>
#include <complex>
#include <memory>
#include <typeinfo>
#include <vector>
#include <fstream>

#include "../g_stddef.hpp"
#include "Pattern.h"
using namespace std;

typedef std::complex<double> COMPLEXd;
typedef std::complex<float> COMPLEXf;

typedef std::complex<double> Z;
typedef std::complex<float> C;
typedef float S;
typedef double D;

enum G_DATATYPE {
    DATA_UNKNOWN,
    FLOAT_S = 0x1,
    FLOAT_D = 0x2,
    FLOAT_C = 0x4,
    FLOAT_Z = 0x8,
};
enum G_PATTERN {
    PTN_UNKNOWN,
    PTN_DENSE = 0x1,
    PTN_CCS   = 0x2,
};

class GeMAT;
typedef shared_ptr<GeMAT> hGMAT;
class GeVEC;
typedef shared_ptr<GeVEC> hGVEC;


/*
 */
class GeMAT {
    GeMAT(const GeMAT &);
    GeMAT &operator=(const GeMAT &);

   protected:
    int nRow, nCol, flag;
    std::string name;
    // double *minimum,*maximum;	//range
    GeMAT() : nRow(0), nCol(0), flag(0), type(M_UNKNOWN) { ; }
    GeMAT(int m, int n, int flag_ = 0x0) : nRow(m), nCol(n), flag(flag_), type(M_UNKNOWN) { ; }
    virtual ~GeMAT() {
        // delete[] minimum;		delete[] maximum;
    }
    virtual hGMAT AM(hGMAT mV, const hGMAT mU, int flag = 0) { throw runtime_error("GeMAT::AM is ..."); }
    template <typename T>
    static G_DATATYPE _dataType() {
        return typeid(T) == typeid(double)     ? FLOAT_D
               : typeid(T) == typeid(float)    ? FLOAT_S
               : typeid(T) == typeid(COMPLEXd) ? FLOAT_Z
               : typeid(T) == typeid(COMPLEXf) ? FLOAT_C
                                               : DATA_UNKNOWN;
    }
    // performance
    struct PERF {
        float tX       = 0.0;
        int runs       = 0;
        int64_t cycles = 0;
    };
    PERF perf;

   public:
    enum TYPE {
        M_UNKNOWN   = -1,
        M_GENERAL   = 0x0,
        M_SYMMETRY  = 0x1,
        M_DIAGONAL  = 0x02,
        M_TRIDIAG   = 0x03,
        M_UNIT      = 0x10,
        M_ZERO      = 0x20,
        M_UNITARY   = 0x100,
        M_HERMITIAN = 0x1000,
    };
    TYPE type;
    enum BIT_FLAG {
        MAT_TRANS  = 0x100,
        DATA_ZERO  = 0x10000,
        DATA_REFER = 0x20000,
    };
    virtual int RowNum() const { return 0; }
    virtual int ColNum() const { return 0; }
    virtual int Count() const { return RowNum() * ColNum(); }
    SHAPE Shape(int flag = 0x0) {
        SHAPE shape = {RowNum(), RowNum()};
        return shape;
    }
    virtual void *Data() const { return nullptr; }

    virtual void *Column() const { return nullptr; }
    virtual G_DATATYPE DataType() { return DATA_UNKNOWN; }
    virtual G_PATTERN PatternType() { return PTN_UNKNOWN; }

    virtual hGMAT Sub(int r_0, int c_0, int nR, int nC) { return nullptr; }

    virtual void Copy(const hGMAT &mB) {};

    virtual double Nrm2(int flag = 0) { throw runtime_error("GeMAT::Nrm2 is vitual!!!"); }
    virtual void Scald(double d, int flag = 0) { throw runtime_error("GeMAT::Scald is vitual!!!"); }
    // Build Operation
    virtual bool Build(int flag = 0x0) { throw runtime_error("GeMAT::Build is vitual!!!"); }
    // Quantization
    virtual void Range(double *minimum, double *maximum, int start_dim, int flag = 0x0) {}
    // Spectral operation
    virtual bool SimilarTrans(hGMAT &mB, int flag = 0) { throw runtime_error("GeMAT::SimilarTrans is vitual!!!"); }
    // Forward operation
    virtual hGVEC Transform(hGVEC &vecY, const hGVEC &vecX, int flag = 0) { throw runtime_error("GeMAT::Transform is vitual!!!"); }
    virtual hGMAT &Transform(hGMAT &mV, const hGMAT &mU, int flag = 0) { throw runtime_error("GeMAT::Base method"); }
    //	virtual	hGMAT Mul(hGMAT mV, const hGMAT mU, int flag = 0);
    // More utility
    friend std::ostream &operator<<(std::ostream &os, GeMAT *hmat) {
        hmat->dump(os);
        return os;
    }
    friend std::ostream &operator<<(std::ostream &os, const hGMAT &mat) {
        mat->dump(os);
        return os;
    }
    virtual void dump(std::ostream &os) { throw runtime_error("GeMAT::dump unimplemented!!!"); }
    virtual void dump(const std::string &path_, const std::string &sX = nullptr, int flag = 0x0) {
        try {
            ios_base::openmode mode = BIT_TEST(flag, 0x100) ? ios_base::out | ios_base::app : ios_base::out;
            std::string sPath(path_);
            if (!sX.empty())
                sPath += sX;
            std::ofstream file(sPath, mode);
            if (file.is_open()) {
                file << __func__ << ":\tN=" << RowNum() << "," << ColNum() << "\n";
                dump(file);
                file.close();
            }
        } catch (...) {
            throw std::ios_base::failure(__func__);
        }
    }
};

template <typename T>
T *TO(const hGMAT hA) {
    if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf)) {
        //		G_DATATYPE dt=GeMAT::_dataType<T>();
        //		assert( dt==hA->DataType() );
        return (T *)(hA->Data());
    } else {
        return (T *)(hA.get());
    }
    throw runtime_error("TO");
}

template <typename T>
T *TO(const hGMAT hA, int r, int c) {  // hGMAT�Ĵ洢Ϊ������
    //		size_t tid = typeid(T).hash_code( );
    if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf)) {
        T *data  = TO<T>(hA);
        int nRow = hA->RowNum(), nCol = hA->ColNum();
        if (r < 0 || r >= nRow || c < 0 || c >= nCol)
            throw range_error("TO(const hGMAT hA,int r,int c )");
        return data + c * nRow + r;
    } else {
        throw runtime_error("TO");
    }
    throw runtime_error("TO");
}

template <typename T>
T *TO(const hGMAT hA, int c) {  // hGMAT�Ĵ洢Ϊ������
    //		size_t tid = typeid(T).hash_code( );
    if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf)) {
        T *data  = TO<T>(hA);
        int nCol = hA->ColNum(), nRow = hA->RowNum();
        if (c < 0 || c >= nCol)
            throw range_error("TO(const hGMAT hA,int r,int c )");
        return data + c * nRow;
    } else {
        throw runtime_error("TO");
    }
    throw runtime_error("TO");
}

#define ToZ TO<Z>
#define ToD TO<D>
#define ToS TO<S>
#define ToC TO<C>
#define ToT TO<T>

class GeVEC : public GeMAT {
   public:
    virtual int ColNum() { return 1; }
};

class Filter : public GeMAT {
   private:
    hGMAT hBase;

   protected:
    hGMAT hFilt;

   public:
    Filter(const hGMAT &hA) : hBase(hA) { ; }
    virtual int RowNum() const { return hBase->RowNum(); }
    virtual int ColNum() const { return hBase->ColNum(); }

    virtual double Nrm2(int flag = 0) { return DBL_MAX; }
    //	virtual hGVEC Transform(hGVEC &vec1, const hGVEC &vec0, int flag = 0)	{ return vec1;  }
    //	virtual hGMAT& Transform(hGMAT &M1, const hGMAT &M0, int flag = 0)		{ return M1; }
    virtual hGVEC Transform(hGVEC &vec1, const hGVEC &vec0, int flag = 0) {
        assert(hFilt != nullptr);
        hFilt->Transform(vec1, vec0);
        return vec1;
    }
    virtual hGMAT &Transform(hGMAT &M1, const hGMAT &M0, int flag = 0) {
        assert(hFilt != nullptr);
        hFilt->Transform(M1, M0);
        return M1;
    }
    virtual hGVEC TransSpectral(const hGVEC &vLenda, bool inverse, int flag = 0x0) { return vLenda; }
};
typedef shared_ptr<Filter> hFILTER;

void Unitary(hGVEC &mV, int flag = 0);

// void DOT(void *dot, const hGVEC vY, const hGVEC vX, int flag = 0);

/*	V=AU;		return V */
// hGMAT Mul(hGMAT mV, const hGMAT mA, const hGMAT mU, int flag = 0);
// hGVEC Mul(hGVEC mV, const hGMAT mA, const hGVEC mU, int flag = 0);
void inline Swap(hGMAT &vX, hGMAT &vY) { vX.swap(vY); }
void inline Swap(hGVEC &vX, hGVEC &vY) { vX.swap(vY); }
