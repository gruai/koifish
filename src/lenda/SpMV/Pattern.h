/**
 *  SPDX-FileCopyrightText: 2013-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <cassert>

#define SMAT_CONTROL_ITEM 32
/*
    ѹ���и�ʽ�洢��ϡ�����
    ע�⣺
        ֻ֧�ַ���
*/
class CCS_pattern {
   private:
    CCS_pattern &operator=(const CCS_pattern &);

   protected:
    //	double control[SMAT_CONTROL_ITEM];
    int type;
    //	int *permut;

    void Create(int dim, int *ptr, int *ind, int m_type = 0x0);

   public:
    int nCol, *ptr, *ind;
    int *ptr_e;  // for MKL call

    int nNZ() const { return ptr == nullptr ? 0 : ptr[nCol]; }
    virtual int nzLU() { return 0; }

    CCS_pattern() : nCol(0), type(0), ptr(nullptr), ptr_e(nullptr), ind(nullptr) { ; }
    // CCS_pattern( const CCS_pattern &)	{;}
    CCS_pattern(const CCS_pattern &ptn, int r_0, int c_0, int nR, int nC);
    CCS_pattern(const CCS_pattern &ptn) { Create(ptn.nCol, ptn.ptr, ptn.ind, ptn.type); }
    CCS_pattern(int dim, int *ptr, int *ind, int m_type = 0x0) { Create(dim, ptr, ind, m_type); }
    virtual ~CCS_pattern(void) { Clear(); }

    //	void Create( const CCS_pattern &ptn,int flag,int r_0=-1,int c_0=-1,int nR=-1,int nC=-1 );
    void Clear();

    int Order(int *, int flag);
    int Compress(int *comp, int *, int alg, int flag);
    virtual bool isValid() { return nCol > 0 && ptr != nullptr && ind != nullptr; }

    //	template<typename T>  friend class Matrix;
};

/*
template<typename T>
class CCS_I : public CCS_mat<typename T>	{
public:
    CCS_I( int d,int *ptr_,int *ind_,T *val_,int m_type ):CCS_mat( d,NULL,NULL,m_type){
    }
    int Decomposition( int type,int flag )	{		return 0;	}
    //	x=inv(A)*y
    virtual int SOLVE( T *x,T *y )	{
        for( int i = 0; i < dim; i++ )	{
            x[i]=y[i];
        }
        return 0x0;
    }

};*/
