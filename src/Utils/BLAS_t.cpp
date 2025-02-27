/**
 *  Copyright 2010-2024 by Grusoft 
 * 
 *  \brief C++ template of BLAS & BLAS-like functions
 *  \author Yingshi Chen
 */

#include <memory>
#include <algorithm>
#include "BLAS_t.hpp"
#include "GST_util.hpp"

#ifdef _USE_OPENBLAS_
	static COMPLEXd Z1_mkl={1.0,0.0},Z0_mkl={0.0,0.0};
#elif defined _USE_MKL_
	#include "mkl_blas.h"	
	#include "mkl_lapack.h"
	#include "mkl_lapacke.h"
	static MKL_Complex16 Z1_mkl={1.0,0.0},Z0_mkl={0.0,0.0};
#endif

static double D1_=1.0,D0_=0.0;
static int inc_1=1;

template <> bool isReal<COMPLEXd>( COMPLEXd val )	{	return val.imag()==0.0; }

static char trans[]={'N','T','C'};
//template <typename T> T NRM2( int dim,T *X )
template<> double NRM2<double>( int dim,double *X )	{
	return DNRM2( &dim,X, &inc_1);
}
template<> double NRM2<float>( int dim,float *X ){
	return SNRM2( &dim,X, &inc_1);
}
template<> double NRM2<COMPLEXd>( int dim,COMPLEXd *X )	{
	return DZNRM2( &dim,(BLAS_Z*)X, &inc_1);
}
//template <typename T> void SCAL( int dim,double alpha,T *X )		
template<> void SCALd<double>( int dim,double alpha,double *X )	{
	DSCAL( &dim,&alpha, X, &inc_1);
}
template<> void SCALd<COMPLEXd>( int dim,double alpha,COMPLEXd *X )	{
	ZDSCAL( &dim,&alpha, (BLAS_Z*)X, &inc_1);
}

template<> void SCAL<COMPLEXd>( int dim,COMPLEXd alpha,COMPLEXd *X )	{
	ZSCAL( &dim,(BLAS_Z*)&alpha, (BLAS_Z*)X, &inc_1);
}

//template <typename T> T DOT( int dim,const T *X,const T *Y )	
template <>
void DOT<double>( double& dot,int dim,double *X,double *Y )	{
	dot = DDOT( &dim, X, &inc_1, Y, &inc_1);
}
template <>
void DOT<COMPLEXd>( COMPLEXd& dot,int dim,COMPLEXd *X,COMPLEXd *Y )	{
	dot = ZDOTC(&(dim), (BLAS_Z *)X, &inc_1, (BLAS_Z *)Y, &inc_1);
}

template<> void COPY<double>( int dim,double *X,double *Y )	
{	DCOPY( &dim,X,&inc_1,Y,&inc_1 );}
template<> void COPY<float>( int dim,float *X,float *Y )	
{	SCOPY( &dim,X,&inc_1,Y,&inc_1 );}
template<> void COPY<COMPLEXd>( int dim,COMPLEXd *X,COMPLEXd *Y )	
{	ZCOPY( &dim,(BLAS_Z *)X,&inc_1,(BLAS_Z *)Y,&inc_1 );}


//Y := a*X + Y
template <>
void AXPY<double>( int dim,double alpha,double *X,double *Y )	{
	DAXPY( &dim,&alpha, X, &inc_1, Y, &inc_1);
//	for( int k = 0; k < dim; k++ )	Y[k]+=alpha*X[k];
}
template<> void AXPY<float>(  int dim, float alpha, float *X,float *Y ){
	SAXPY( &dim,&alpha, X, &inc_1, Y, &inc_1);
}
template<> void AXPY<COMPLEXd>( int dim,COMPLEXd alpha,COMPLEXd *X,COMPLEXd *Y )	{
	ZAXPY( &dim,(BLAS_Z *)(void *)(&alpha), (BLAS_Z *)(void *)X, &inc_1, (BLAS_Z *)(void *)Y, &inc_1);
}

template<> 
void GER_11<double>( int M,int N, double alpha, double *x, double *y, double *A, int lda )	{
//	A[0] += x[0]*y[0];
//	DGER( &M, &N, &alpha, x, &incx, y, &incy, A, &lda );
	int i,j;
	double *pa,x_i;
	for( i = 0; i < M; i++ ) {
		pa = A+i*lda;		x_i = x[i];
		for( j = 0; j < N; j++ ) {
			pa[j] += x_i*y[j];
		}
	}
}

template<>
void GER<float>( int m, int n, float *alpha, float *vX, int incx, float *vY, int incy, float *beta, float *A, int lda ){
	SGER(&m, &n, alpha, vX, &incx, vY, &incy, A, &lda);
}

template<> 
void GEMV<double>( char transa,int M,int N, double alpha, double *A, int lda, double *X, int incx, double beta, double *Y, int incy )	{
	DGEMV(&transa,&M,&N,&alpha,A,&lda,X,&incx,&beta,Y,&incy);
}
template<> 
void GEMV<Z>( char transa,int M,int N, Z alpha, Z *A, int lda, Z *X, int incx, Z beta, Z *Y, int incy )	{
	ZGEMV(&transa,&M,&N,(BLAS_Z*)(&alpha),(BLAS_Z*)A,&lda,(BLAS_Z*)X,&incx,(BLAS_Z*)(&beta),(BLAS_Z*)Y,&incy);
}

template<> 
void GEAV<Z>( char transa,int M,int N, Z *A, int lda, Z *X, Z *Y  )	{	//Y=AX
	ZGEMV(&transa,&M,&N,(BLAS_Z*)&Z1_mkl,(BLAS_Z*)A,&lda,(BLAS_Z*)X,&inc_1,(BLAS_Z*)&Z0_mkl,(BLAS_Z*)Y,&inc_1);
}

template<> 
void GEMM<COMPLEXd>( char transa, char transb, int m, int n, int k, COMPLEXd *alpha, COMPLEXd *a, int lda, COMPLEXd *b, int ldb, COMPLEXd *beta, COMPLEXd *c, int ldc )	{
	ZGEMM(&transa, &transb, &m, &n, &k, (BLAS_Z*)alpha, (BLAS_Z*)a, &lda, (BLAS_Z*)b, &ldb, (BLAS_Z*)beta, (BLAS_Z*)c, &ldc);
}
template<> 
void GEMM<double>( char transa, char transb, int m, int n, int k, double *alpha, double *a, int lda, double *b, int ldb, double *beta, double *c, int ldc )	{
	DGEMM(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}
template<> 
void GEMM<float>( char transa, char transb, int m, int n, int k, float *alpha, float *a, int lda, float *b, int ldb, float *beta, float *c, int ldc )	{
	SGEMM(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
}

/*
template<> void GEAX<COMPLEXd>(Vector<COMPLEXd> vY,const CCS_d &mA ,const Vector<COMPLEXd> &vX)	{		//Y=AX
	double alpha=1.0,beta=0.0;
	char transa='N',matdescra[6]="\0";
	mkl_zcscmv (transa, &nRow,  &nCol,&alpha, matdescra,val,ind,pattern.ptr,pattern.ptr_e,x,beta,y);
}

template<> void GEMM(Matrix<COMPLEXd> Y,const CCS_d &mA ,const Matrix<COMPLEXd> &X)	{		//V=AU
	mkl_zcscmm ( 'N', MKL_INT *m, MKL_INT *n, MKL_INT *k, BLAS_Z
*alpha, char *matdescra, BLAS_Z *val, MKL_INT *indx, MKL_INT *pntrb, MKL_INT
*pntre, BLAS_Z *b, MKL_INT *ldb, BLAS_Z *beta, BLAS_Z *c, MKL_INT
*ldc);
	return;
}*/

template<> void LARNV<double>( int dim,double *X,int seed ){
//idist <1: uniform (0,1)	2: uniform (-1,1)	3: normal (0,1)>
//iseed: the array elements must be between 0 and 4095, and iseed(4) must be odd.
	int iseed[4] = { 1, 3, 5, 1 }, idist = 3;
	for( int i = 0; i < 3; i++ )	iseed[i]=(dim*i+1)%4095;
	DLARNV( idist,iseed,dim,X );		//DLARNV( &idist,iseed,&dim,X );
#ifdef _DEBUG
//	double a = NRM2( dim,X );
//	a /= dim;
#endif
}/*
template<> void LARNV<COMPLEXd>( int dim,COMPLEXd *X,int seed ){
//2: real and imaginary parts each uniform (-1,1)
//4: uniformly distributed on the disc abs(z) < 1		5: uniformly distributed on the circle abs(z) = 1
	int iseed[4] = { 1, seed, 5, 7 }, idist = 2;	
	ZLARNV( &idist,iseed,&dim,(BLAS_Z*)X );
}*/

template<> bool isUnitary<COMPLEXd>( int n, const hGMAT &mU,COMPLEXd *tau )	{
	if( GST_util::verify<GST_util::VERIRY_UNITARY_MAT )
		return true;
	int m=mU->RowNum( );
	COMPLEXd *tU=TO<COMPLEXd>(mU);
	ZGEMM(trans+2, trans, &n, &n, &m, (BLAS_Z*)&Z1_mkl, (BLAS_Z*)tU, &m, (BLAS_Z*)tU, &m, (BLAS_Z*)&Z0_mkl, (BLAS_Z*)tau, &n);
	int i,j;
	double a,thrsh=1.0e-10,rel,img;
	for( i = 0; i < n; i++ )	{
	for( j = 0; j < n; j++ )	{
		rel = tau[i*n+j].real(),		img = tau[i*n+j].imag();
		a = i==j ?1.0- sqrt(rel*rel+img*img) : sqrt(rel*rel+img*img);
		if( fabs(a)<thrsh )	{
		}
		else		{
			return false;
		}
	}
	}
	return true;
}
template<> bool isIMat<COMPLEXd>( int nCol, const hGMAT &mU)	{
	if( GST_util::verify<GST_util::VERIRY_I_MAT )
		return true;

	int i,j,m=mU->RowNum( );
	COMPLEXd *tU=TO<COMPLEXd>(mU),one(1.0);
	double off,thrsh=1.0e-10;
	
	for( i = 0; i < nCol; i++ )	{
	for( j = 0; j < m; j++ )	{
		off = ( i==j ) ? abs( tU[i*m+j]-one) : abs( tU[i*m+j]);
		if( off<thrsh )	{
		}
		else		{
			printf( "isIMat<%d> failed at (%d,%d,%g)\r\n",nCol,j,i,off );
			return false;
		}
	}
	}
	return true;
}

template<> bool isTriangular<COMPLEXd>( int nCol, const hGMAT &mU,bool isUp )	{
	if( GST_util::verify<GST_util::VERIRY_TRI_MAT )
		return true;
	int i,j,m=mU->RowNum( );
	assert(m>=nCol);
	COMPLEXd *tU=TO<COMPLEXd>(mU);
	double a,thrsh=1.0e-10;
	bool isTri=true;
	
	for( i = 0; i < nCol; i++ )	{
	for( j = 0; j < nCol; j++ )	{
		a = abs( tU[i*m+j]);
		if( isUp )	{
			if( j>i && a>thrsh )
				isTri = false;
		}else	{
			if( j<i && a>thrsh )
				isTri = false;
		}
		if( !isTri )	{
			printf( "isTriangular<%d> failed at (%d,%d,%g)\r\n",nCol,j,i,a );
			return false;
		}
	}
	}
	return true;
}

template <>  
void GEQR<COMPLEXd>( int m,int n,hGMAT &mB, const hGMAT &mA,COMPLEXd sift, int flag){	//A=QR
/*	assert( m<=mA->RowNum( ) && n<=mA->ColNum( ) && m<=mB->RowNum( ) && n<=mB->ColNum( ) );
	assert( mA->Count()==mB->Count() );
	int ldA=mA->RowNum(),ldB=mB->RowNum(),i;
	COMPLEXd *tQ=TO<COMPLEXd>(mB),*tA=TO<COMPLEXd>(mA);
	BLAS_Z *tau=new BLAS_Z[n*n];
	memcpy( tQ,tA,sizeof(COMPLEXd)*mA->Count() );
	if( sift!=0.0 )	{
		for( i = min(m,n)-1; i >= 0 ; i-- )
			tQ[i*ldB+i] -= sift;
	}
	if( m<ldB )	{
		for( int i = 0; i < n; i++ )	{
			for( int j = m; j < ldB; j++ )	{
				tQ[i*ldB+j]=0;
			}
		}
	}
	LAPACKE_zgeqrf( LAPACK_COL_MAJOR,m,n, (BLAS_Z*)tQ, ldB, tau );
	LAPACKE_zungqr( LAPACK_COL_MAJOR, m, n, n, (BLAS_Z*)tQ, ldB, tau );
	if( 1 )	{	//verify
		assert( isUnitary( n,mB,(COMPLEXd *)tau ) );	
	}
	delete[] tau;*/
	assert(0);
}

/*
	A=UTU'
	ע�� 
		����ʱA���滻ΪT!!!
*/
template<>  void SchurDecomp<COMPLEXd>( int rank,hGMAT &mA,hGMAT &mU,hGVEC &vW,COMPLEXd info,int flag )	{
/*	int n=rank,lda=mA->RowNum( ),sdim,i,no=-1,ldu=mU->RowNum();
	COMPLEXd *ta=TO<COMPLEXd>(mA),*tu=TO<COMPLEXd>(mU),*w=TO<COMPLEXd>(vW);
	no = LAPACKE_zgees(LAPACK_COL_MAJOR,charV,charN, nullptr,n, (BLAS_Z*)ta, lda, &sdim,(BLAS_Z*)w,(BLAS_Z*)tu, ldu );
	double a,aMax=0.0;
	for( i = 0; i < n; i++ )	{
		a = abs(w[i]);
		if( a>aMax )	
		{	aMax=a;		no=i;	}
	}
	if( no!=0 )	{
		LAPACKE_ztrexc( LAPACK_COL_MAJOR, charV, n,(BLAS_Z*)ta,lda, (BLAS_Z*)tu,ldu,no+1,1 );
		swap( w[no],w[0] );
	}*/
	assert(0);
}


//N is Number of observations 
template<> 
void GECOV<float>( int dim,int n,float *X,double* dcov,double *dcor,int flag ){
#ifdef _USE_MKL_    
	VSLSSTaskPtr task;
	float *cov=new float[dim*dim],*cor=new float[dim*dim];
    float *mean=new float[dim],*variation=new float[dim],*min_esti=new float[dim],*max_esti=new float[dim];
    float *x=nullptr,*raw2=nullptr,*raw3=nullptr,*raw4=nullptr,*cen2=nullptr,*cen3=nullptr,*cen4=nullptr;
    int i,j,errcode,errnums = 0;
    unsigned MKL_INT64 estimate = 0;
    /***** Initializing parameters for Summary Statistics task *****/
    MKL_INT x_storage   = VSL_SS_MATRIX_STORAGE_COLS;		//the first p-dimensional observation of the vector �� comes first,
    //MKL_INT x_storage   = VSL_SS_MATRIX_STORAGE_ROWS;		//n data points for the vector component ��1 come first
    MKL_INT cov_storage = VSL_SS_MATRIX_STORAGE_FULL;
    MKL_INT cor_storage = VSL_SS_MATRIX_STORAGE_FULL;

    for(i = 0; i < dim; i++)    {
        min_esti[i] = X[i];        max_esti[i] = X[i];
    }

    /***** Create Summary Statistics task *****/
    errcode = vslsSSNewTask( &task, &dim, &n, &x_storage, (float*)X, 0, 0 );    CheckVslError(errcode);
    /***** Edit task parameters for min and max computation *****/
    errcode = vslsSSEditTask( task, VSL_SS_ED_MIN, min_esti );		CheckVslError(errcode);
    errcode = vslsSSEditTask( task, VSL_SS_ED_MAX, max_esti );		CheckVslError(errcode);

    errcode = vslsSSEditMoments( task, mean, raw2, raw3, raw4,cen2, cen3, cen4 );
    CheckVslError(errcode);
    //errcode = vslsSSEditTask( task, VSL_SS_ED_VARIATION, variation );    CheckVslError(errcode);
    /***** Initialization of the task parameters using FULL_STORAGE   for covariance/correlation matrices *****/
    errcode = vslsSSEditCovCor( task, mean, (float*)cov, &cov_storage,(float*)cor, &cor_storage );	CheckVslError(errcode);

    /***** Minimum and maximum are included in the vector of estimates
           to compute *****/
    estimate |= VSL_SS_MIN | VSL_SS_MAX | VSL_SS_MEAN ;
	//estimate |= VSL_SS_VARIATION;
    estimate |=  VSL_SS_COV ;

    /***** Compute the estimates using FAST method *****/
    errcode = vslsSSCompute( task, estimate, VSL_SS_METHOD_FAST );
    CheckVslError(errcode);

	for( i=dim*dim-1;i>=0;i-- ){
		dcov[i]=cov[i];		
		if( dcor!=nullptr )		dcor[i]=cor[i];
	}

    /* Comparison of observations with min and max estimates */
    for(x=X,i=0; i < dim; i++,x+=n)    {
        for(j = 0; j < n; j++)        {
            if(x[j] < min_esti[i]) errnums++;
            if(x[j] > max_esti[i]) errnums++;
        }
    }
    /***** Delete Summary Statistics task *****/
    errcode = vslSSDeleteTask( &task );
    CheckVslError(errcode);

    MKL_Free_Buffers();
#endif
    return ;
}

template<> void GESOP<float>( int dim,int n,float *X,double* dSop,int flag ){
	int i,inc=1;
	float one=1,*vec,*sop=new float[dim*dim]();
	for( i=0;i<n;i++ ){
		vec=X+dim*i;
		GER(dim, dim,&one, vec, inc, vec, inc, &one,sop, dim);
	}
	for( i=dim*dim-1;i>=0;i-- )dSop[i]=sop[i];
	delete[] sop;
}


template <>  void GESVD<float>(  int m, int n,float *a,float *sval,float *u,float *v,int lda,int ldu,int ldv )	{
	float *superb=new float[min(m,n)];
	if( lda==0 )	lda=m;
	if( ldu==0 )	ldu=m;
	if( ldv==0 )	ldv=n;
	int iRet = LAPACKE_sgesvd(LAPACK_COL_MAJOR, charA, charA, m,n,a,lda,sval,u,ldu,v,ldv,superb );
	delete[] superb;
	if( iRet!=0 )	throw  ("GESVD failed!!!");;
}

