#pragma once

#include <stddef.h>
#include <algorithm>
#include <vector>
#include "../SpMV/Matrix.hpp"
#include "../SpMV/GVMAT_t.hpp"
#include "../SpMV/GeQuant.hpp"
#include "../util/BLAS_t.hpp"
#include "../util/GST_util.hpp"
#ifdef _GE_LU_
#include "..\lu\GeLU.hpp"
#endif

// Optimal Brain Quanization for linear WX
template<typename T>
class OBS_WX 	{
private:
	OBS_WX(const OBS_WX&);
	OBS_WX& operator=(const OBS_WX&);
protected:
	shared_ptr<Matrix<T>> hX,hW,hQ,hERR;
	shared_ptr<Matrix<T>> HInv;
	shared_ptr<Quantizer<T>> hQuant = nullptr;
	hFILTER hFilt;
	hGVEC vLenda;
	
	double tole,nrmA;
	int *i_temp,nFlag,blas_block=128;
	
	hGVEC TransSpectral( const hGVEC &vLenda,bool inverse ,int flag=0x0 )	{
		int i,n=vLenda->RowNum();
		T *lenda = TO<T>(vLenda);
		if( inverse )	{
			// for( i = 0; i < n; i ++ )	{
			// 	lenda[i] = 1.0/lenda[i]+shift;
			// 	ritzs[i]->SetRitz( lenda[i].real(),lenda[i].imag() );
			// }
		}
		return vLenda;
	}

public:	
	WEIGHT_TYPE wwType;
	double RadiConfi;

	int H_dim,nEV,nConv,nRestart,nLastConv,bits=16;
	// static double tExpand,tRestart,tOP,tX,reortho;
	// static int nIter,nOPx,starting;

	OBS_WX( hGMAT H_,hGMAT W_,hGMAT Q_,int bits_,int flag=0x0 ) : bits(bits_)	{	
		H_dim = H_->RowNum();		assert(H_dim==H_->ColNum());
		assert(W_->ColNum()==H_dim);
		HInv = std::dynamic_pointer_cast<Matrix<T>>(H_);	
		assert(HInv!=nullptr);
		hW = std::dynamic_pointer_cast<Matrix<T>>(W_);		assert(hW!=nullptr);
		hQ =  std::dynamic_pointer_cast<Matrix<T>>(Q_);		assert(hW!=nullptr);	//std::make_shared<Matrix<T>>(hW);
		hQ->Copy(hW);
		hERR = std::make_shared<Matrix<T>>(hW);
		tole=1.0e-12;		wwType = ABS_MAX;	
		
		hQuant = std::make_shared<Quantizer<T>>(4,true,false);
  		hQuant->Init(hW);
	}
	virtual ~OBS_WX( ){}

	double GetTole(  )	{	return tole;	}
	double SetTole( double tol,int flag );
/*	Eigen( const MATA &mA,int nev,int flag ) : matA(mA),nEV(nev),nFlag(flag)	{		
	}*/	

	bool isConverge( )	{	return false;	}

	virtual int Run( int flag )				{	
		GST_TIC(tic);
		T one_=1.0,fuyi_=-1.0,zero_=0.0;
		int iRet = 0x0,i0,i1,blk,i,nRow=hW->RowNum(),ldW=hW->ColNum();
		double loss=0,los,tX=0;
		for(i0=0;i0<H_dim;i0+=blas_block)	{
			i1 = min(i0+blas_block,H_dim);	
			blk = i1-i0;
			for(i=i0;i<i1;i++)	{
				T d = HInv->Diagnol(i);				
				los = hQuant->Update(nRow,hW->Column(i),hQ->Column(i),0x0,d,hERR->Column(i),ldW,0x0);	
				T *pA=hERR->Column(i),*pB=HInv->Row(i)+i,*pC=hW->Column(i);				
				if(1)
					//GEMM( charN,charN,dim-i,dim,1,&fuyi_,pB,dim,pA,dim,&one_,pC,dim );	
					GEMM( charN,charN,i1-i,nRow,1,&fuyi_,pB,ldW,pA,ldW,&one_,pC,ldW );		
				else{
					for(int r=0;r<nRow;r++)		{
						for(int c=0;c<i1-i;c++)	{
							pC[r*ldW+c] += fuyi_*pA[r*ldW]*pB[c];
						}
					}					
				}				
				// ((GeMAT*)(hW.get()))->dump("./log/OBS_W_",G_STR(i),0x0);		//only for debug
				loss += los/2;	
			}				
			if(i1<H_dim)	{
				GST_TIC(t0);
				T *pA=hERR->Column(i0),*pB=HInv->Row(i0)+i1,*pC=hW->Column(i1);
				GEMM( charN,charN,H_dim-i1,nRow,blk,&fuyi_,pB,ldW,pA,ldW,&one_,pC,ldW );	//
				tX+=GST_TOC(t0);
			}
		}
		if(GST_util::dump>0){				//11856.407
			// ((GeMAT*)(hQ.get()))->dump("./log/OBS_Q.dat","",0x0);
			// ((GeMAT*)(hW.get()))->dump("./log/OBS_W.dat","",0x0);
		}printf("+ OBS_WX::%s loss=%.6g N=%d/%d T=%.3g(%.3g)\n",__func__,loss,H_dim,blas_block,GST_TOC(tic),tX);
		
		return iRet;
	}

	virtual int Contract( int flag )		{	assert(false);	return 0;	}
	virtual void Dump( int type,int flag )	{;}	
};
typedef shared_ptr<OBS_WX<float>> hQWX_f;

class Quant_Factory{
public:
	enum MODEL {
		OPT_BRAIN,	//Optimal Brain
	};
	template<typename T>
	static shared_ptr<OBS_WX<T>> CreateQuanter(Quant_Factory::MODEL type,hGMAT H_,hGMAT W_,hGMAT Q_,int bits_,int flag)	{
		shared_ptr<OBS_WX<T>> quant = nullptr;
		switch(type){
		default:
			quant = std::make_shared<OBS_WX<T>>(H_,W_,Q_,bits_,flag);
			break;
		}
		assert(quant!=nullptr);
		return quant;
	}
};
