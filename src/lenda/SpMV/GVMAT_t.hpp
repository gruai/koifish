#pragma once

#include <cassert>
#include <complex>
#include <memory>
#include <typeinfo>
#include <math.h>
#include <string.h>
#include "GVMAT.h"
#include "../../Utils/BLAS_t.hpp"
using namespace std;

template<typename T>
double M_Dis(hGVEC vR,hGVEC vY, hGVEC vX, T &u, int flag=0)	{	//R=min|Y-uX|
	int dim = vX->RowNum();
	T *tX = (T *)TO<T>(vX), *tR = (T *)TO<T>(vR), *tY = (T *)TO<T>(vY);
	COPY( dim,tY,tR );					//r=y;
	DOT(u,dim, tX, tR);					//u=x'*r
	AXPY(dim, -u, tX,tR );				//r-=u*x
	double dis = NRM2(dim,tR);
	return dis;
}

/*
template<typename T>
double OFF_Cols(double *offs,hGMAT hA,hGMAT hB, int flag=0)	{	//����ÿ�е�|A(:,i)-B(:,i)|
	int i,j,nCol=hA->ColNum(),ldY=hA->RowNum( );
	assert( hB->ColNum()==nCol && ldY=hB->RowNum( ));
	double *a,*b,c;
	for( i = 0; i < batch; i++ )	{
		a = TO<double>(hA,i);	b=TO<double>(hB,i);
		c = 0.0;
		for( j = 0; j<ldY; j++)		c+=(a[j]-b[j])*(a[j]-b[j]);
		offs[j] = sqrt(c);
	}
	return dis;
}*/

template<typename T>
double AX_uX( hGMAT hA,int n,T &u,T *x, int flag=0 )	{		//R=Ax-u*X,	return |vR|
	hGVEC	vR=make_shared<Vector<T>>(n),vX=make_shared<Vector<T>>(n,x,GeMAT::DATA_REFER);		
	double dis = AX_uX(vR,hA,u,vX,flag);
	return dis;
}

template<typename T>
double AX_uX( hGVEC vR,hGMAT hA,T &u,hGVEC vX, int flag=0 )	{		//R=Ax-u*X,	return |vR|
	int dim = vX->RowNum();
	hA->Transform(vR, vX);			//vR=Ax
	T *tX = (T *)TO<T>(vX), *tR = (T *)TO<T>(vR);
	AXPY(dim, -u, tX,tR );				//|Ax-u*x|
	double dis = NRM2(dim,tR);
	return dis;
}

template<typename T>
void Set(hGMAT hA, GeMAT::TYPE type,T val=0 )	{	//A=val
	T *tA = (T *)TO<T>(hA);
	int i,nRow,nCol,nz = hA->Count( );
	switch( type )	{
	case GeMAT::M_ZERO:
		memset( tA,0x0,sizeof(T)*nz );
		break;
	case GeMAT::M_UNIT:
		nRow=hA->RowNum(),nCol=hA->ColNum();
		assert( nRow==nCol && nz==nRow*nCol );
		memset( tA,0x0,sizeof(T)*nz );
		for( i = 0; i < nRow; i++ )		tA[i*nRow+i]=T(1.0);
		break;
	case GeMAT::M_GENERAL:
		for( i = 0; i < nz; i++ )		tA[i]=val;
		break;
	default:
		assert(0);
		break;
	}	
}


template<typename T>
void Inc(hGMAT hY,const hGMAT hX, T alpha=1 )	{	//Y+=aX
	T *tY = (T *)TO<T>(hY),*tX = (T *)TO<T>(hX);
	int nz = hY->Count( );
	if( alpha==1)	{
		for( int i = 0; i < nz; i++ )		tY[i]+=tX[i];
	} else	{
		for( int i = 0; i < nz; i++ )		tY[i]+=alpha*tX[i];
	}
}

template<typename T>
void Rand(hGMAT hA, int seed=3, int flag=0)	{
	T *tA = (T *)TO<T>(hA);
	int dim = hA->Count();
	assert( dim>0 );
	LARNV(dim, tA,seed );
}

template<typename T>
void SIGMODx( hGMAT &hZ,hGMAT &hA, int flag=0)	{
	T *tA = (T *)TO<T>(hA),*tZ=(T *)TO<T>(hZ);
	int nnz = hA->Count(),i;
	for( i = 0; i < nnz; i++ )	{
		tA[i] = 1.0/(1.0+exp(-tA[i]));
		tZ[i]= tA[i]*(1.0-tA[i]);
	}
}

template<typename T>
void HADAM( hGMAT &hZ,hGMAT &hX,hGMAT &hY, int flag=0)	{	//Z=X��Y
	T *tZ=(T *)TO<T>(hZ),*tX = (T *)TO<T>(hX),*tY = (T *)TO<T>(hY);
	int nnz = hZ->Count(),i;
	for( i = 0; i < nnz; i++ )	{
		tZ[i] = tX[i]*tY[i];
	}
}

enum WEIGHT_TYPE{
	ABS_MAX,SIFT_CLOSEST,SIFT_INV_CLOSEST
};
template<typename T>
void Weight( int m,T sift,hGMAT hA,double *ww,WEIGHT_TYPE type )	{
	int i,nnz = hA->Count();
	assert( m<=nnz && ww!=nullptr );
	T *tA = (T *)TO<T>(hA),s;
	for( i = 0; i < m; i++ )	{	
		switch( type )	{
		case ABS_MAX:
			ww[i] = abs(tA[i]);
			break;
		case SIFT_CLOSEST:
			ww[i] = abs(tA[i]-sift);
			ww[i] = ww[i]==0.0 ? DBL_MAX : 1.0/ww[i];
			break;
		default:
			assert( false);
			break;
		}
	}
}

template<typename T>
double Weight( T a,T sift,WEIGHT_TYPE type )	{
	T s;
	double wa=DBL_MAX;
	switch( type )	{
	case ABS_MAX:
		wa = abs(a);
		break;
	case SIFT_CLOSEST:
		wa = abs(a-sift);
		wa = wa==0.0 ? DBL_MAX : 1.0/wa;
		break;
	default:
		assert( false);
		break;
	}
	return wa;
}