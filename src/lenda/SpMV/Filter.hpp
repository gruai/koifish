#pragma once

#include <cassert>
#include <complex>
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include "GVMAT.h"
#include "Matrix.hpp"
#include "../util/BLAS_t.hpp"
#include "..\LU\GeLU.hpp"

using namespace std;

//shift-invert
template<typename T>
class FiltSI : public Filter	{
	T shift;
//	hGMAT hFilt;
	MatCCS<T> *hCCS;
public:
	FiltSI( const hGMAT &hA0,T sift=0.0, int flag=0x0 ) : Filter(hA0),hCCS(nullptr)	{
		if( CCS_d *hD=dynamic_cast<CCS_d *>(hA0.get( ) ) )	{
			hCCS =new MatCCS<T>( *hD );
		} else if( CCS_z *hD=dynamic_cast<CCS_z *>(hA0.get( ) ) )	{
			hCCS =new MatCCS<T>( *hD );
		}		
		if( hCCS==nullptr )
			throw invalid_argument( "FiltSI::const hGMAT &hA0" );
		shift = sift;
		hCCS->Shift( shift );
		hFilt = make_shared<LU_gss<T>>(hCCS,0x0);
		Dump( );
	}

	virtual T GetShift( )	{	return shift; }

	virtual hGVEC TransSpectral( const hGVEC &vLenda,bool inverse ,int flag=0x0 )	{
		int i,n=vLenda->RowNum();
		T *lenda = TO<T>(vLenda);
		if( inverse )	{
			for( i = 0; i < n; i ++ )	{
				lenda[i] = 1.0/lenda[i]+shift;
			}
		}
		return vLenda;
	}

	void Dump( )	{;}
};

template<> void FiltSI<D>::Dump( )	{
	printf( "Filter_SI<Z> shift=%.7g\n",shift );
}
template<> void FiltSI<Z>::Dump( )	{
	printf( "Filter_SI<Z> shift=<%.7g,%.7g>\n",shift.real(),shift.imag( ) );
}

//rational krylov
template<typename T>
class FiltRK : public Filter	{
	T shift;	
	MatCCS<T> *hCCS;
public:
	FiltRK( const hGMAT &hA0,T sift=0.0, int flag=0x0 ) : Filter(hA0)	{
		if( CCS_d *hD=dynamic_cast<CCS_d *>(hA0.get( ) ) )	{
			hCCS =new MatCCS<T>( *hD );
		} else		{

		}
		if( hCCS==nullptr )
			throw invalid_argument( "FiltRK::const hGMAT &hA0" );
		shift = sift;
		hFilt = make_shared<LU_gss<T>>(hCCS,0x0);
		Dump( );
	}

	virtual T GetShift( )	{	return shift; }

	virtual hGVEC TransSpectral( const hGVEC &vLenda,bool inverse ,int flag=0x0 )	{
		int i,n=vLenda->RowNum();
		T *lenda = TO<T>(vLenda);
		if( inverse )	{
			for( i = 0; i < n; i ++ )	{
				lenda[i] = 1.0/lenda[i]+shift;
			}
		}
		return vLenda;
	}

	void Dump( )	{;}
};

template<> void FiltRK<D>::Dump( )	{
	printf( "Filter_SI<Z> shift=%.7g\n",shift );
}
template<> void FiltRK<Z>::Dump( )	{
	printf( "Filter_SI<Z> shift=<%.7g,%.7g>\n",shift.real(),shift.imag( ) );
}


//FEAST
template<typename T>
class FiltFEAST : public Filter	{
	const static int nE=8;	//Ne-point Gauss-Legendre quadrature
	const static double x[8],w[8];
	T lenda_0,lenda_1;
	hGMAT hFilt[nE];
public:
	FiltFEAST( const hGMAT &hA0,T l_0,T l_1, int flag=0x0 ) : Filter(hA0)	{
		int i;
		double xita;
		COMPLEXd ze,zi(0,1.0);
		CCS_d *hCCS = TO<CCS_d>( hA0 );
		for( i = 0; i < nE; i++ )	{
			xita = -M_PI/2*(x[i]-1);		//θe = −(π/2)(xe − 1)
			ze = (lenda_0+lenda_1)/2.0+(lenda_0-lenda_1)/2.0*exp(zi*xita);						//Ze = (λmax + λmin)/2 + r exp(ıθe),
			hFilt[nE]=nullptr;			//(Ze − A)
		}
	}

	virtual hGMAT& Transform(hGMAT &M1, const hGMAT &M0, int flag = 0)		{ 
	//	hFilt->Transform( M1,M0 );  
		return M1;
	}

	virtual hGVEC TransSpectral( const hGVEC &vLenda,bool inverse ,int flag=0x0 )	{
		int i,n=vLenda->RowNum();
		T *lenda = TO<T>(vLenda);
		if( inverse )	{
			for( i = 0; i < n; i ++ )	{
				lenda[i] = 1.0/lenda[i];
			}
		}
		return vLenda;
	}

};

template<typename T>
const double FiltFEAST<T>::x[8] = {0.183434642495649,-0.183434642495649,0.525532409916328,-0.525532409916328,0.796666477413626,-0.796666477413626,0.960289856497536,-0.960289856497536};
template<typename T>
const double FiltFEAST<T>::w[8] = {0.362683783378361,0.362683783378361,0.313706645877887,0.313706645877887,0.222381034453374,0.222381034453374,0.101228536290376,0.101228536290376};
