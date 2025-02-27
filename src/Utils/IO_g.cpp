#include <memory>
#include "BLAS_t.hpp"

int GBIN_Read( char *sBinPath,int* nrow,int* ncol,int* nnz,int** colptr,int** rowind,double** g_val,double**rhs,int flag )	{ 
	FILE* stream;
	int x[8]={0,0,0,0,0,0,0,0};			//reserved for future use
	int *ptr,*ind,ret=0;
	double *val;

	if( (stream=fopen( sBinPath,"rb" )) == nullptr )	{
		ret=-1;	goto _FIO_EXIT_;
	}
	if( fread( nrow,sizeof(int),1,stream )!= 1 )
	{	ret=-2;	goto _FIO_EXIT_;	}
	if( fread( ncol,sizeof(int),1,stream )!= 1 )
	{	ret=-2;	goto _FIO_EXIT_;	}
	if( fread( nnz,sizeof(int),1,stream )!= 1 )
	{	ret=-2;	goto _FIO_EXIT_;	}
	if( fread( x,sizeof(int),8,stream )!= 8 )
	{	ret=-3;	goto _FIO_EXIT_;	}

	ptr=(int*)malloc( sizeof(int)*(*ncol+1) );
	ind=(int*)malloc( sizeof(int)*(*nnz) );
	val=(double*)malloc( sizeof(double)*(*nnz) );
	if( fread( ptr,sizeof(int),(*ncol)+1,stream )!= (*ncol)+1 )
	{	ret=-4;	goto _FIO_EXIT_;	}
	if( fread( ind,sizeof(int),(*nnz),stream )!= (*nnz) )
	{	ret=-5;	goto _FIO_EXIT_;	}
	if( fread( val,sizeof(double),(*nnz),stream )!= (*nnz) )
	{	ret=-6;	goto _FIO_EXIT_;	}

_FIO_EXIT_:
	if( stream != 0x0 && fclose( stream )!= 0 )
	{	ret=-7;			}
	if( ret==0 )	{
//		printf( "\tLOAD %s:nEntry=%d\r\n",sBinPath,(*nnz) );
		*colptr=ptr;		*rowind=ind;
		*g_val=val;
		if( rhs!=nullptr )
			*rhs=0x0;
	}else	{
		ret = stream!=0x0 ? ferror( stream ) : ret;
		printf( "\r\n\t!!!Failed to read %s. err=%d\r\n",sBinPath,ret );
	}

	return ret;
}
