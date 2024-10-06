#include <memory>
#include <iostream>
#include <algorithm>
#include <tchar.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "../SpMV/GVMAT_t.hpp"
#include "../util/GST_util.hpp"
#include "../util/GST_bmp.hpp"

Dataset::Dataset( int n,int ld,int tp,double *x ) : nMost(n),ldX(ld),type(tp),tag(nullptr),X(x)	{
	tag = new int[nMost]( );	
	if( !BIT_TEST(type,TAG_ZERO) )
		for( int i = 0; i < nMost; i++ )	tag[i]=-1;
	if( x==nullptr )	{
		X = new double[nMost*ldX];
	} else		{
		X = x;
	}
}
void Dataset::Clear( ){	
	delete[] tag;		operator delete[] (X); 
	tag=nullptr;		X=nullptr;		
	ldX=0;				nMost=0;
}
		
int Dataset::Load( const wstring sPath,int flag )	{
	int ret=-1;
	FILE *fp=_tfopen( sPath.c_str( ),_T("rb") );
	if( fp==NULL )
		goto _FIO_EXIT_;
	if( fread( &nMost,sizeof(int),1,fp )!=1 )
	{	ret=-10;		goto _FIO_EXIT_;	}
	if( fread( &ldX,sizeof(int),1,fp )!=1 )
	{	ret=-11;		goto _FIO_EXIT_;	}
	tag = (int*)operator new[]( sizeof(int)*nMost );
	if( fread( tag,sizeof(int),nMost,fp )!=nMost )
	{	ret=-12;		goto _FIO_EXIT_;	}
	X = (double*)operator new[]( sizeof(double)*nMost*ldX );
	if( fread( X,sizeof(double),nMost*ldX,fp )!=nMost*ldX )
	{	ret=-13;		goto _FIO_EXIT_;	}
_FIO_EXIT_:
	if( fp != NULL && fclose( fp )!= 0 )
	{	ret=-7;			}
	if( ret==0 )	{
	}else	{
		ret = fp!=NULL ? ferror( fp ) : ret;
//		G_PRINTF( _T("\t!!!Failed to save %s. err=%d"),sLibPath,ret );
	}
	return ret;
}
int Dataset::Save( const wstring sPath,int flag )	{
	int ret=-1,dim=nSample( );
	FILE *fp=_tfopen( sPath.c_str( ),_T("wb") );
	if( fp==NULL )
		goto _FIO_EXIT_;
	
	if( fwrite( &dim,sizeof(int),1,fp )!=1 )
	{	ret=-10;		goto _FIO_EXIT_;	}
	if( fwrite( &ldX,sizeof(int),1,fp )!=1 )
	{	ret=-11;		goto _FIO_EXIT_;	}
	if( fwrite( tag,sizeof(int),dim,fp )!=dim )
	{	ret=-12;		goto _FIO_EXIT_;	}
	if( fwrite( X,sizeof(double),dim*ldX,fp )!=dim*ldX )
	{	ret=-13;		goto _FIO_EXIT_;	}

_FIO_EXIT_:
	if( fp != NULL && fclose( fp )!= 0 )
	{	ret=-7;			}
	if( ret==0 )	{
	}else	{
		ret = fp!=NULL ? ferror( fp ) : ret;
//		G_PRINTF( _T("\t!!!Failed to save %s. err=%d"),sLibPath,ret );
	}
	return ret;
}

double Err_Tag( int lenT,double *acti,int tag,int& nOK,int flag )	{
	double a,a_1=0,err=0.0;		
	int no_1=-1;
	for( int i = 0; i < lenT; i++ )	{
		a = i==tag ? acti[i]-1.0 : acti[i];
		if( acti[i]>a_1 )	{
			a_1 = acti[i];		no_1=i;
		}
		err += a*a;
	}
	if( no_1==tag )
		nOK++;
	return err;
}
double Err_Auto( int lenT,double *in,double *out,int& nOK,int flag )	{
	double a,a_1=0,err=0.0;		
	for( int i = 0; i < lenT; i++ )	{
		a = in[i]-out[i];		
		err += a*a;			a_1+=in[i]*in[i];
	}
	if( err<1.0e-5*a_1 )
		nOK++;
	return err;
}

int Dataset::Shrink( int nTo,int flag )	{
	int nFrom=nSample( ),grid=nFrom/nTo,i,pos;
	if( grid<=1 )
		return nFrom;
	for( i = 0; i < nTo; i++)	{
		pos=i*grid;
		tag[i]=tag[pos];
		memcpy( X+ldX*i,X+ldX*pos,ldX*sizeof(double) );
	}
	for( i=nTo; i<nFrom; i++ )		tag[i]=-1;
	return nTo;
}
void Dataset::Copy( int i,const Dataset &hS,int no,int nSample,int flag )	{
	assert( i>=0 && i<nMost && no>=0 && no<hS.nSample( ) );
	assert( ldX==hS.ldX );
	memcpy( X+ldX*i,hS.X+ldX*no,ldX*sizeof(double)*nSample );
	memcpy( tag+i,hS.tag+no,sizeof(int)*nSample );
//	tag[i]=hS.tag[no];
	return;
}

void Dataset::_mean( T *x,int flag )	{
	int i;
	T a0=FLT_MAX,a1=-FLT_MAX,mean=0.0;
	for( i = 0; i < ldX; i++ )	{
		a0=min(a0,x[i]);		a1=max(a1,x[i]);
		mean+=x[i];
	}
	mean /= ldX;
	for( i = 0; i < ldX; i++ )	x[i]-=mean;
}
void Dataset::_normal( int nz,T *x,int type,int flag )	{
	int i;
	T a0=FLT_MAX,a1=-FLT_MAX;
	double mean=0.0,devia=0.0,a;
	for( i = 0; i < nz; i++ )	{
		a0=min(a0,x[i]);		a1=max(a1,x[i]);
		mean+=x[i];
	}
	mean /= nz;
	for( i = 0; i < nz; i++ )	{
		x[i]-=mean;		devia+=x[i]*x[i];
	}
	devia = 3.0*sqrt(devia/nz);
	for( i = 0; i < nz; i++ )	{
		x[i]=min( devia,x[i] );		x[i]=max( -devia,x[i] );
		a = x[i]/devia;
		x[i] = (a+1.0)*0.4+0.1;
		assert( x[i]>=0.1 && x[i]<=0.9 );
	}
}
void Dataset::Preprocess( int alg,int flag )	{
	int i,nS=nSample( ),width=sqrt(ldX*1.0),height=ldX/width;
	double *x;
	for( i = 0; i < nS; i++ )	{
		x = X+ldX*i;
	//	_mean(x,flag );
	}
	_normal( ldX*nS,X,0x0,0x0 );
	return;
}
int Dataset::ToBmp( int epoch,int _x,int flag )	{
#ifdef _GST_BITMAP_
	int n=nSample(),width=sqrt(ldX*1.0),height=ldX/width;
	if( n<=0 )
		return -1;
//	n=100;
	if( BIT_TEST(type,GST_bitmap::COLOR) )	{
		int nPixel=ldX/3;
		width=sqrt(nPixel*1.0),height=nPixel/width;
		return GST_bitmap::save_doublebmp_n( n,epoch,width,height,X,n,type );
	}else
		return GST_bitmap::save_doublebmp_n( n,epoch,width,height,X,n,type );
#else
	return -2;
#endif
}

#ifdef _GST_MATLAB_
hDATASET Dataset::ImportMat( const char *sPath,const char* sVar,int flag )	{
	hDATASET hDat=nullptr;
	int nRow,nCol,i;
	double *val;
	int iRet=GST_util::LoadDoubleMAT( sPath,sVar,&nRow,&nCol,&val,0x0 );
	if( iRet==0x0 )	{
		hDat = new Dataset( nCol,nRow,GST_bitmap::COLOR|Dataset::TAG_ZERO,val );
//		for( i = 0; i < hDat->nMost; i++ )	hDat->tag[i]=0;
		hDat->ToBmp(-1);//_dumpSample( data,0x0 );
		hDat->Save( _T("F:\\deep learning\\dataset\\\stl_ZCA.dat"),0x0 );
	}
	return hDat;
}
#endif