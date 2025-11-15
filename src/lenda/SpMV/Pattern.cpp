/**
 *  SPDX-FileCopyrightText: 2013-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "Pattern.h"

#include <string.h>

#include <memory>
#include <stdexcept>

CCS_pattern::CCS_pattern(const CCS_pattern &ptn, int r_0, int c_0, int nR, int nC) {
    if (r_0 < 0 || c_0 < 0 || r_0 + nR >= ptn.nCol || r_0 + nC >= ptn.nCol)
        throw std::out_of_range("CCS_pattern( const CCS_pattern &ptn,int r_0,int c_0,int nR,int nC ");
    int nzMax = 0, i, j, r, pos = 0, *ptr0 = ptn.ptr, *ind0 = ptn.ind;
    nzMax  = ptr0[c_0 + nC] - ptr0[c_0];
    nCol   = nC;
    type   = ptn.type;
    ptr    = new int[nCol + 1];
    ind    = new int[nzMax];
    ptr[0] = 0;
    for (i = c_0; i < c_0 + nC; i++) {
        for (j = ptr0[i]; j < ptr0[i + 1]; j++) {
            r = ind0[j];
            if (r < r_0 || r >= r_0 + nR)
                continue;
            ind[pos++] = r - r_0;
        }
        ptr[i - c_0 + 1] = pos;
    }
    assert(pos <= nzMax);
    return;
}
/*
 */
void CCS_pattern::Create(int d, int *ap, int *ai, int t) {
    try {
        ptr = nullptr, ptr_e = nullptr, ind = nullptr;
        nCol = -1, type = 0x0;
        assert(d > 0 && "CCS_pattern::CCS_pattern");
        //	int i;
        nCol = d;
        type = t;
        if (ap != nullptr) {
            int nnz = ap[d];
            assert(nnz > 0 && ai != nullptr && "CCS_pattern:	nnz>0");
            ptr = new int[nCol + 1];
            ind = new int[nnz];
            memcpy(ptr, ap, sizeof(int) * (nCol + 1));
            memcpy(ind, ai, sizeof(int) * (nnz));
            /*		ptr_e = new int[nCol];
                    for( int i = 0; i < nCol; i++ )	{
                        ptr_e[i] = ptr[i+1]-1;
                    }*/
        }
    } catch (...) {
        //	type=P_BAD;
        //	Clear( );
    }
}
/*
void CCS_pattern::Create( const CCS_pattern &ptn,int flag,int r_0=-1,int c_0=-1,int nR=-1,int nC=-1 )	{
    nCol = d;
    type = t;
    if( ap!=nullptr )	{
        int nnz = ap[d];					assert( nnz>0 && ai!=nullptr && "CCS_pattern:	nnz>0" );
        ptr = new int[nCol+1];				ind = new int[nnz];
        memcpy_s( ptr,sizeof(int)*(nCol+1),ap,sizeof(int)*(nCol+1) );
        memcpy_s( ind,sizeof(int)*(nnz),ai,sizeof(int)*(nnz) );
    }
}*/

/*
 */
void CCS_pattern::Clear() {
    try {
        //	if( n2n!=nullptr )
        //	{	delete[] n2n;			n2n=nullptr;	}
        //	if( permut!=nullptr )
        //	{	delete[] permut;		permut=nullptr;	}
        if (ptr != nullptr) {
            delete[] ptr;
            ptr = nullptr;
        }
        if (ptr_e != nullptr) {
            delete[] ptr_e;
            ptr_e = nullptr;
        }
        if (ind != nullptr) {
            delete[] ind;
            ind = nullptr;
        }
    } catch (...) {
        //	type=P_BAD;
    }
}

/*
 */
int CCS_pattern::Order(int *I_temp, int flag) { return -1; }

/*
    v0.1	cys
        5/28/2012
*/
int CCS_pattern::Compress(int *comp, int *cluster, int alg, int flag) {
    int i, j, k, no, same = 0, nComp = 0, nCls = 0, nzMax = 0, nz = 0, nPass = 0, no_c = 0;
    int *stmp = new int[nCol], stp = 0, *sum = new int[nCol];
    bool isSame;

    for (i = 0; i < nCol; i++) {
        stmp[i] = stp;
        comp[i] = -1;
        sum[i]  = 0;
        for (j = ptr[i]; j < ptr[i + 1]; j++) {
            no = ind[j];
            sum[i] += no;
        }
    }
    float similar = 0.0, r_1, r_2;

    for (i = 0; i < nCol; i++) {
        if (comp[i] != -1)
            continue;
        nzMax += ptr[i + 1] - ptr[i];
        cluster[no_c++] = i;
        comp[i]         = nCls++;
        stp++;
        r_1 = (float)(ptr[i + 1] - ptr[i]);
        if (r_1 == 0.0)
            continue;
        for (j = ptr[i]; j < ptr[i + 1]; j++) {
            no       = ind[j];
            stmp[no] = stp;
        }

        for (j = ptr[i]; j < ptr[i + 1]; j++) {
            no = ind[j];
            if (no == i || comp[no] != -1)
                continue;
            r_2 = (float)(ptr[no + 1] - ptr[no]);
            if (r_2 == 0.0)
                continue;
            /*		if( sum[i]!=sum[no] )
                        continue;
                    if( r_1!=r_2 )
                        continue;
                    isSame=true;
                    for( k = ptr[no]; k < ptr[no+1]; k++ )	{
                        if( stmp[ind[k]]!=stp )
                        {	isSame=false;		break;			}
                    }*/
            same = 0;
            for (k = ptr[no]; k < ptr[no + 1]; k++) {
                if (stmp[ind[k]] == stp) {
                    same++;
                }
            }
            similar = ((same) / r_1 + (same) / r_2) / 2.0f;
            assert(similar <= 1.0);
            isSame = similar > 0.95;
            //	isSame = similar ==1.0;
            if (isSame) {
                cluster[no_c++] = no;
                comp[no]        = comp[i];
                nComp++;
            } else {
                nPass++;
            }
        }
    };
    assert(no_c == nCol);
    assert(nComp + nCls == nCol);
    delete[] stmp;
    delete[] sum;
    return nCls;
}