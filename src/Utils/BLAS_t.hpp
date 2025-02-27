#pragma once
/**
 *  Copyright 2010-2024 by Grusoft 
 * 
 *  \brief C++ template of BLAS & BLAS-like functions
 *  \author Yingshi Chen
 */

#include "../lenda/SpMV/GVMAT.h"

typedef int G_INT_64;

#ifdef _USE_OPENBLAS_
 	#include "f77blas.h"
	//	#include "cblas.h"
	// #include "lapacke.h"	would fail,so strange!
	#define LAPACK_ROW_MAJOR               101
	#define LAPACK_COL_MAJOR               102
	typedef int lapack_int;
	#define lapack_complex_double std::complex<double>

	extern "C"	{
	lapack_int LAPACKE_sgesvd( int matrix_layout, char jobu, char jobvt,lapack_int m, lapack_int n, float* a, lapack_int lda,
							float* s, float* u, lapack_int ldu, float* vt,lapack_int ldvt, float* superb );
	// void LAPACK_dlarnv_(    lapack_int const* idist, lapack_int* iseed, lapack_int const* n,double* X );
	lapack_int LAPACKE_dlarnv( lapack_int idist, lapack_int* iseed, lapack_int n,double* x );
	// void LAPACK_zlarnv(    lapack_int const* idist, lapack_int* iseed, lapack_int const* n,    lapack_complex_double* X );

	}

	typedef double BLAS_Z;

	#define DNRM2	dnrm2_
	#define DSCAL 	dscal_
	#define DSWAP 	dswap_
	#define DDOT	ddot_
	#define DCOPY 	dcopy_
	#define DTRSV 	dtrsv_
	#define DGEMM 	dgemm_
	#define SGEMM 	sgemm_
	#define DGEMV 	dgemv_
	#define DTRSM 	dtrsm_
	#define DAXPY 	daxpy_
	#define DSYRK	dsyrk_
	#define DPOTRF 	dpotrf_
	#define DLARNV 	LAPACKE_dlarnv

	#define SNRM2	snrm2_
	#define SCOPY 	scopy_
	#define SGER	sger_
	#define SAXPY 	saxpy_

	#define ZSCAL 	zscal_	
	#define ZDSCAL	zdscal_
	#define DZNRM2	dznrm2_
	#define ZDOTC	zdotc_
	#define ZSWAP 	zswap_
	#define ZCOPY 	zcopy_
	#define ZTRSV 	ztrsv_
	#define ZGEMM 	zgemm_
	#define ZHERK 	zherk_
	#define ZGEMV 	zgemv_
	#define ZTRSM 	ztrsm_
	#define ZAXPY 	zaxpy_
	#define ZSYRK	zsyrk_
	#define ZPOTRF 	zpotrf_
	#define ZLARNV 	LAPACK_zlarnv

	#define CSCAL 	cscal_
	#define CSWAP 	cswap_
	#define CCOPY 	ccopy_
	#define CTRSV 	ctrsv_
	#define CGEMM 	cgemm_
	#define CHERK 	cherk_
	#define CGEMV 	cgemv_
	#define CTRSM 	ctrsm_
	#define CAXPY 	caxpy_
	#define CSYRK	csyrk_
	#define CPOTRF 	 cpotrf_

	// void dpotrf_(  char* uplo,  G_INT_64* n, double* a, G_INT_64* lda, G_INT_64* info );	
#else
#endif

const char charA='A',charN='N',charT='T',charC='C',charE='E',charV='V',charB='B',charL='L',charR='R',charU='U',charS='S';

template <typename T> bool IS_NAN(  T&a )		{	return std::isnan(a);		}
template<> bool IS_NAN<COMPLEXd>(  COMPLEXd &a );
template<> bool IS_NAN<COMPLEXf>(  COMPLEXf &a );

/*
template <typename T> void a2Txt(  T& val,string& txt );
template <> void a2Txt(  double& val,string& txt );
template <> void a2Txt(  COMPLEXd& val,string& txt );*/

//a*a'= |a|*|a|
template <typename T> double TxTc(  T&a )		{	return a*a;	}
template<> double TxTc<COMPLEXd>(  COMPLEXd &a );
template<> double TxTc<COMPLEXf>(  COMPLEXf &a );

template <typename T> double NRM2( int dim,T *X )		{	assert(0);		return 0.0;	}
template<> double NRM2<double>( int dim,double *X );
template<> double NRM2<float>( int dim,float *X );
template<> double NRM2<COMPLEXd>( int dim,COMPLEXd *X );

template <typename T> void SCALd(  int dim, double alpha,T *X )		{	assert(0);		return;	}
template<> void SCALd<float>(  int dim, double alpha,float *X );
template<> void SCALd<double>(  int dim, double alpha,double *X );
template<> void SCALd<COMPLEXd>(  int dim, double alpha,COMPLEXd *X );
template<> void SCALd<COMPLEXf>(  int dim, double alpha,COMPLEXf *X );

template <typename T> void SCAL(  int dim, T alpha,T *X )		{	assert(0);		return;	}
template<> void SCAL<COMPLEXd>(  int dim, COMPLEXd alpha,COMPLEXd *X );

template <typename T> double NORMAL(  int dim,T *X )		{	
	double nrm=NRM2(dim,X);		assert( nrm!=0.0);	SCALd( dim,1.0/nrm,X );	
	return nrm;
}

template <typename T> void DOT( T &dot, int dim, T *X, T *Y )		{	assert(0);		return;	}
template<> void DOT<double>( double &dot, int dim, double *X, double *Y );
template<> void DOT<float>( float &dot, int dim, float *X, float *Y );
template<> void DOT<COMPLEXd>( COMPLEXd &dot, int dim, COMPLEXd *X, COMPLEXd *Y );

template <typename T> void COPY(  int dim, T *X,T *Y )		{	assert(0);		return;	}
template<> void COPY<double>(  int dim, double *X,double *Y );
template<> void COPY<float>(  int dim, float *X,float *Y );
template<> void COPY<COMPLEXd>(  int dim, COMPLEXd *X,COMPLEXd *Y );

//�ܺ�ʱ��!!! in-place transposition/copying of matrices
template <typename T> void IMAT_T(  char ordering, size_t rows, size_t cols,T * AB)	{	throw "IMATCOPY is ...";		}
template <> void IMAT_T<COMPLEXf>(  char ordering, size_t rows, size_t cols,COMPLEXf * AB);

template <typename T> void SWAP(  int dim,T *X,int,T *Y,int )		{	assert(0);		return;	}
template<> void SWAP<COMPLEXf>(  int dim,COMPLEXf *X,int incx,COMPLEXf *Y,int incy );
//Y := a*X + Y
template <typename T> void AXPY(  int dim, T alpha, T *X,T *Y )		{	assert(0);		return;	}
template<> void AXPY<float>(  int dim, float alpha, float *X,float *Y );
template<> void AXPY<double>(  int dim, double alpha, double *X,double *Y );
template<> void AXPY<COMPLEXd>(  int dim, COMPLEXd alpha, COMPLEXd *X,COMPLEXd *Y );
template<> void AXPY<COMPLEXf>(  int dim, COMPLEXf alpha, COMPLEXf *X,COMPLEXf *Y );

//Y := a*X + b*Y
template <typename T> void AXPBY(  int dim, T alpha, T *X, T beta,T *Y )		{	assert(0);		return;	}
template<> void AXPBY<float>(  int dim, float alpha, float *X, float beta,float *Y );
template<> void AXPBY<double>(  int dim, double alpha, double *X, double beta,double *Y );

template <typename T> void GER_11(  int M, int N,  T alpha,  T *x,  T *y, T *a,  int lda )		{	assert(0);		return;	}
template<> void GER_11<double>(  int M, int N,  double alpha,  double *x,  double *y, double *a,  int lda );

//�μ�GVMAT_t.cpp::Set<T>
template <typename T>  void vSET(  int dim,T *a,T b=0.0 ){	for( int i=0;i<dim; i++)	a[i]=b;	}

template <typename T>  double vCOS(  int dim, T *a, T *b,int flag=0x0 ){	assert(0);		return 0;	}
template <>  double vCOS<double>(  int dim, double *a, double *b,int flag );

template <typename T>  void vMUL(  int dim,T *a,T *b,T*y ){	assert(0);		return;	}
template <>  void vMUL<double>(  int dim,double *a,double *b,double *y );
template <>  void vMUL<float>(  int dim,float *a,float *b,float *y );

template <typename T>  void vEXP(  int dim,T *Z ){	assert(0);		return;	}
template <>  void vEXP<double>(  int dim,double *Z );
template <>  void vEXP<float>(  int dim,float *Z );

template <typename T>  void vSIGMOD(  int dim,T *Z ){	assert(0);		return;	}
template <>  void vSIGMOD<double>(  int dim,double *Z );
template <>  void vSIGMOD<float>(  int dim,float *Z );

template <typename T>  void vSOFTMAX(  int dim,int ld,T *Z ){	assert(0);		return;	}
template <>  void vSOFTMAX<float>(  int dim,int ld,float *Z );
template <>  void vSOFTMAX<double>(  int dim,int ld,double *Z );
template <typename T>  void vSOFTMAX_trunc(  int dim,int ld,T *Z,float thrsh ){	assert(0);		return;	}
template <>  void vSOFTMAX_trunc<double>(  int dim,int ld,double *Z,float thrsh );

template <typename T> void GEMV( char transa, int M, int N,  T alpha,  T *A,  int lda,  T *X,  int incx,  T beta, T *Y,  int incy )	{	assert(0);		return;	}
template<> void GEMV<double>( char transa, int M, int N,  double alpha,  double *A,  int lda,  double *X,  int incx,  double beta, double *Y,  int incy );
template<> void GEMV<C>( char transa, int M, int N,  C alpha,  C *A,  int lda,  C *X,  int incx,  C beta, C *Y,  int incy );
template<> void GEMV<Z>( char transa, int M, int N,  Z alpha,  Z *A,  int lda,  Z *X,  int incx,  Z beta, Z *Y,  int incy );

template <typename T> void GEAV( char transa, int M, int N,  T *A,  int lda,  T *X, T *Y )	{	assert(0);		return;	}
template <> void GEAV<double>( char transa, int M, int N,  double *A,  int lda,  double *X, double *Y );
template<> void GEAV<Z>( char transa, int M, int N,  Z *A,  int lda,  Z *X, Z *Y );
template<> void GEAV<C>( char transa, int M, int N,  C *A,  int lda,  C *X, C *Y );

//A += alpha*x*y'	rank-1 update
template<typename T> void GER( int m, int n, T *alpha, T *vX, int incx, T *vY, int incy, T *beta, T *A, int lda );	
template<>
void GER<float>( int m, int n, float *alpha, float *vX, int ldx, float *vY, int ldy, float *beta, float *A, int lda );

template<typename T> void AB2C( char transa, char transb, int m, int n, int k,T *a, int lda, T *b, int ldb,T *c, int ldc ){
	T one(1.0),zero(0.0);
	GEMM( transa,transb,m,n,k,&one,a,lda,b,ldb,&zero,c,ldc );
}

template<typename T> void COO_MM( int m, int n, int k, T *alpha, T *a, int *rowA,int*colA, int nnz,T *b, int ldb, T *beta, T *c, int ldc );
template<> void COO_MM<double>( int m, int n, int k, double *alpha, double *a, int *rowA,int*colA, int nnz,double *b, int ldb, double *beta, double *c, int ldc );
template<> void COO_MM<float>( int m, int n, int k, float *alpha, float *a, int *rowA,int*colA, int nnz,float *b, int ldb, float *beta, float *c, int ldc );

template<typename T> void C_GEMM( char transa, char transb, int m, int n, int k,  T *alpha, T *a, int lda, T *b, int ldb,  T *beta, T *c, int ldc )
{	assert(0);		return;		}	
template<> 
void C_GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k,  COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb,  COMPLEXf *beta, COMPLEXf *c, int ldc );

//matrix-diagnol multiplication, NO BLAS!
template<typename T> void GEMD( char transa, int m, int n, T *alpha, const T *a, int lda, const T *d, T *c, int ldc ){
	const T *src=a;
	T *dst=c;	
	assert(lda>=n && ldc>=n);
	for(int i =0;i<m;i++,src+=lda,dst+=ldc){
		for(int j =0;j<n;j++){
			dst[j] = (*alpha)*src[j]*d[j];
		}
	}		
}	


template<typename T> void GEMM( char transa, char transb, int m, int n, int k,  T *alpha, T *a, int lda, T *b, int ldb,  T *beta, T *c, int ldc )
{	assert(0);		return;		}	
template<> 
void GEMM<COMPLEXd>( char transa, char transb, int m, int n, int k,  COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb,  COMPLEXd *beta, COMPLEXd *c, int ldc );
template<> 
void GEMM<COMPLEXf>( char transa, char transb, int m, int n, int k,  COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb,  COMPLEXf *beta, COMPLEXf *c, int ldc );
template<>
void GEMM<double>( char transa, char transb, int m, int n, int k,  double *alpha, double *a, int lda, double *b, int ldb,  double *beta, double *c, int ldc );
template<>
void GEMM<float>( char transa, char transb, int m, int n, int k,  float *alpha, float *a, int lda, float *b, int ldb,  float *beta, float *c, int ldc );

template<typename T> void HERK_s( char uplo, char trans, int m, int k, T *a, int lda, T *c, int ldc,int flag );	
template<> 
void HERK_s<COMPLEXd>( char uplo, char trans, int m, int k, COMPLEXd *a, int lda, COMPLEXd *c, int ldc, int flag );
template<> 
void HERK_s<COMPLEXf>( char uplo, char trans, int m, int k, COMPLEXf *a, int lda,  COMPLEXf *c, int ldc,int flag );

template<typename T> void TRSV( char uplo,char transa, char diag, int m, T *a, int lda, T *b, int inc_ );	
template<> 
void TRSV<COMPLEXd>( char uplo,char transa, char diag, int m, COMPLEXd *a, int lda, COMPLEXd *b, int inc_ );
template<> 
void TRSV<COMPLEXf>( char uplo,char transa, char diag, int m, COMPLEXf *a, int lda, COMPLEXf *b, int inc_ );

template<>
void TRSV<double>( char uplo,char transa, char diag, int m, double *a, int lda, double *b, int inc_ );

template<typename T> void TRSM( char side,char uplo,char transa, char diag, int m, int n,  T *alpha, T *a, int lda, T *b, int ldb );	
template<> 
void TRSM<COMPLEXd>( char side,char uplo,char transa, char diag, int m, int n,  COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb );
template<> 
void TRSM<COMPLEXf>( char side,char uplo,char transa, char diag, int m, int n,  COMPLEXf *alpha, COMPLEXf *a, int lda, COMPLEXf *b, int ldb );
template<>
void TRSM<double>( char side,char uplo,char transa, char diag, int m, int n,  double *alpha, double *a, int lda, double *b, int ldb );

template<typename T> int GELS(char trans, int m, int n,int nrhs, T* a, int lda,T* b, int ldb );
template<> int GELS<double>(char trans, int m, int n,int nrhs, double* a, int lda,double* b, int ldb );
template<> int GELS<COMPLEXf>(char trans, int m, int n,int nrhs, COMPLEXf* a, int lda,COMPLEXf* b, int ldb );
template<> int GELS<COMPLEXd>(char trans, int m, int n,int nrhs, COMPLEXd* a, int lda,COMPLEXd* b, int ldb );

template<typename T> void GELSQ(char trans, int m, int n,int nrhs, T* a, int lda,T* b, int ldb,T*Q );
template<> void GELSQ<double>(char trans, int m, int n,int nrhs, double* a, int lda,double* b, int ldb,double *Q );

template <typename T> void LARNV_seed(  int dim,T *X,int dist,int seed )		{	assert(0);		return;	}
template<> void LARNV_seed<double>(  int dim,double *X,int dist,int seed );
template<> void LARNV_seed<float>(  int dim,float *X,int dist,int seed );

template <typename T> void LARNV(  int dim,T *X,int dist=3 )		{	assert(0);		return;	}
template<> void LARNV<double>(  int dim,double *X,int dist );
template<> void LARNV<float>(  int dim,float *X,int dist );
template<> void LARNV<COMPLEXd>(  int dim,COMPLEXd *X,int dist ); 
template<> void LARNV<COMPLEXf>(  int dim,COMPLEXf *X,int dist ); 

//normal distributed random numbers
template <typename T> void GaussRNV(  int dim,T *X,T mean,T sigma,int flag )		{	assert(0);		return;	}
template<> void GaussRNV<float>(  int dim,float *X,float mean,float sigma,int flag );
template<> void GaussRNV<double>(  int dim,double *X,double mean,double sigma,int flag );
template<> void GaussRNV<COMPLEXd>(  int dim,COMPLEXd *X,COMPLEXd mean,COMPLEXd sigma,int flag );

template  <typename T> bool isReal( T val )	{	return true; }
template <> bool isReal<COMPLEXd>( COMPLEXd val )	;

//LU factorization of A[dim:dim] A is in row-major
template  <typename T> int GETRF_r( int dim,T*a,int lda,int *ipiv,int flag )			{	assert(0);		return -1;	}
template<> int GETRF_r( int dim,COMPLEXf*a,int lda,int *ipiv,int flag );
template<> int GETRF_r( int dim,COMPLEXd*a,int lda,int *ipiv,int flag );
template  <typename T> int GETRS_r( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int GETRS_r( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );
template<> int GETRS_r( int dim,COMPLEXd*a,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag );

//LU factorization of A[dim:dim] A is in column-major
template  <typename T> int GETRF( int dim,T*a,int lda,int *ipiv,int flag )			{	assert(0);		return -1;	}
template<> int GETRF( int dim,COMPLEXf*a,int lda,int *ipiv,int flag );
template  <typename T> int GETRS( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int GETRS( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );

template  <typename T> int SYTRF_r( int dim,T*a,int lda,int *ipiv,int nRhs,T*b,int ldb,int flag )			{	assert(0);		return -1;	}
template<> int SYTRF_r( int dim,COMPLEXf*a,int lda,int *ipiv,int nRhs,COMPLEXf*b,int ldb,int flag );
template<> int SYTRF_r( int dim,COMPLEXd*a,int lda,int *ipiv,int nRhs,COMPLEXd*b,int ldb,int flag );

template  <typename T> bool isUnitary( int n,  const hGMAT &mU,T *tau );
template<> bool isUnitary<COMPLEXd>( int n,  const hGMAT &mU,COMPLEXd *tau );
template  <typename T> bool isIMat( int n,  const hGMAT &mU );
template<> bool isIMat<COMPLEXd>( int n,  const hGMAT &mU);
template  <typename T> bool isTriangular( int n,  const hGMAT &mU,bool isUp );
template<> bool isTriangular<COMPLEXd>( int n,  const hGMAT &mU,bool isUp );
template  <typename T> bool isHMat( int n,  const hGMAT &mU,T *tau ){	return false;	}

template <typename T>  void GEQR(int m,int n,hGMAT &mB,  const hGMAT &mA,T info, int flag=0){	assert(0);		return;	}
template<>  void GEQR<COMPLEXd>(int m,int n,hGMAT &mB,  const hGMAT &mA,COMPLEXd info, int flag);
template <typename T> void SchurDecomp( int rank,hGMAT &mA,hGMAT &hU,hGVEC &vW,T info,int flag=0 ){	assert(0);		return;	}
template<> void SchurDecomp<COMPLEXd>( int rank,hGMAT &mA,hGMAT &hU,hGVEC &vW,COMPLEXd info,int flag );
/**/

//AR: the upper triangle(trapezoidal) is overwritten by R.
template <typename T> int GEQR_p(int m, int n, T *AR, int ldA,T *Q,int *jpvt, int flag = 0x0);
template<> int GEQR_p<COMPLEXd>(int m, int n, COMPLEXd *A, int ldA,COMPLEXd *R,int *jpvt, int flag );
template<> int GEQR_p<COMPLEXf>(int m, int n, COMPLEXf *A, int ldA,COMPLEXf *R,int *jpvt, int flag );
//Interpolative Decomposition --  Y=Y(:, J)X		return the numeric rank
//expressing Y as linear combination of selected columns of Y		epsi-compression tolerance


template <typename T>  void GESVD(  int m, int n,T *a,T *sval,T *u,T *v,int lda=0,int ldu=0,int ldv=0 );
template<> void GESVD<double>(  int m, int n,double *a,double *sval,double *u,double *v,int lda,int ldu,int ldv );
template<> void GESVD<float>(  int m, int n,float *a,float *sval,float *u,float *v,int lda,int ldu,int ldv );
template<> void GESVD<COMPLEXf>(  int m, int n,COMPLEXf *a,COMPLEXf *sval,COMPLEXf *u,COMPLEXf *v,int lda,int ldu,int ldv );

//covariance/correlation
template <typename T> void GECOV( int dim,int n,T *X,double* cov,double *cor,int flag );
template<> void GECOV<float>( int dim,int n,float *X,double* cov,double *cor,int flag );

//inversion
template <typename T> int GEINV( int dim,T *X,int flag );
template<> int GEINV<float>( int dim,float *X,int flag );
template<> int GEINV<double>( int dim,double *X,int flag );

template <typename T> int POTRF(  char uplo,int dim,T *X,int ldX )	{		assert(0);	return -1;		}
template<> int POTRF<COMPLEXd>(  char uplo,int dim,COMPLEXd *X,int ldX );
template<> int POTRF<COMPLEXf>(  char uplo,int dim,COMPLEXf *X,int ldX );

//Cholesky factorization of a symmetric(Hermitian) positive-definite matrix.
template <typename T> int POTR_FS( int dim,T *X,int ldX,int nRhs,T *B,int ldB,int flag );
template<> int POTR_FS<float>( int dim,float *X,int ldX,int nRhs,float *B,int ldB,int flag );
template<> int POTR_FS<double>( int dim,double *X,int ldX,int nRhs,double *B,int ldB,int flag );
template<> int POTR_FS<COMPLEXd>( int dim,COMPLEXd *X,int ldX,int nRhs,COMPLEXd *B,int ldB,int flag );
template<> int POTR_FS<COMPLEXf>( int dim,COMPLEXf *X,int ldX,int nRhs,COMPLEXf *B,int ldB,int flag );

//A��ı�!!! Overwritten by the factors L and U from the factorization of A =P*L*U; the unit diagonal elements of L are not stored.
template <typename T> int GESV(int dim, T *A, int nRhs, T *B,T *X, int flag);
//A��ı�!!! Overwritten by the factors L and U from the factorization of A =P*L*U; the unit diagonal elements of L are not stored.
template<> int GESV<double>(int dim, double *A, int nRhs, double *B, double *X,int flag);
template<> int GESV<COMPLEXd>(int dim, COMPLEXd *A, int nRhs, COMPLEXd *B, COMPLEXd *X,int flag);

//inversion by SVD
template <typename T> void GEINV_SVD( int dim,T *X,double thrsh,int flag );
template<> void GEINV_SVD<float>( int dim,float *X,double thrsh,int flag );
template<> void GEINV_SVD<double>( int dim,double *X,double thrsh,int flag );

//sums over all outer products
template <typename T> void GESOP( int dim,int n,T *X,double* sop,int flag );
template<> void GESOP<float>( int dim,int n,float *X,double* sop,int flag );

template <typename T> 
void OrthoDeflate( int k,T *y,T *Q,int flag=0x0 )	{	//sorensen�Ĺ�ʽ(Lemma 4.1)���Ȼ��BUG���������˳Ծ���������˼��
	MEM_CLEAR( Q,sizeof(T)*k*k );
	COPY( k,y,Q );
	int i,j;
	T *q,gama,dot=0,c;
	double tao_0=abs(y[0]),sigma=tao_0*tao_0,a,tao;
	a = NRM2(k,y);		assert( a==1.0 );
	for( i = 1; i < k; i++ )	{
		q = Q+i*k;
		a = abs(y[i]);		sigma+=a*a;
		tao=sqrt(sigma);
		if( tao_0!=0 )	{
			gama = y[i]/tao/tao_0;		c=-T(gama.real(),-gama.imag());
			AXPY( i,c,y,q );
			q[i] = tao_0/tao;
		} else		{
			q[i-1] = 1.0;
		}
		tao_0 = tao;
		if( 1 )	{
			dot=0;
			for( j = 0; j <= i; j++ )	{	
				c= T(q[j].real(),-q[j].imag())*y[j];
				dot += c;
			}
			assert( abs(dot)<1.0e-15 );
		}
	}
}