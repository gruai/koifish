#pragma once

#include <stdio.h>
#include <stddef.h>
#include <assert.h>
#include <fstream>
#include <stdarg.h>
#include <memory>		//for shared_ptr
#include <string>
#include <vector>
#include <typeinfo>
#include <algorithm>
#include <complex>
#include <limits.h>
#include <cstring>
#include <stdio.h>  
#include <unordered_map>
#include <map>
#include <math.h>
#include <float.h>

using namespace std;

#define UNUSED(x) (void)(x)

#ifdef WIN32
	#include <time.h>
	#define GST_NOW( )		(clock( ))
	#define GST_TIC(tick)	clock_t tick=clock( );
	#define GST_TOC(tick)	((clock()-(tick))*1.0f/CLOCKS_PER_SEC)
#else
	#include <chrono>
	#include <thread>

	typedef std::chrono::high_resolution_clock Clock;
	#define GST_NOW( )	(Clock::now( ))
	#define GST_TIC(tick)	auto tick = Clock::now( );
	#define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now( )-(tick)).count( ))/1000.0)
#endif

/*
	Dataset��Matrix�������ƣ�Ҳ��������κϲ���		 10/19/2014		cys
*/
struct Dataset{
	enum{	TAG_ZERO=0x10 };
	typedef double T;
	void _mean( T *x,int flag );
	void _normal( int nz,T *x,int type,int flag );

	int nMost,ldX,*tag,type;
	T *X;

	Dataset( int n,int ld,int tp=0x0,double *x=nullptr );
	~Dataset( )	{	Clear( );/*operator delete[] (tag);		operator delete[] (X);*/ }
	void Clear( );

	void Preprocess( int alg,int flag )	;

	int Shrink( int nTo,int flag ) 	;
	int nSample( ) const	{	
		if( nMost<=0 )	return 0;
		int i;	
		for(i=0; i < nMost && tag[i]>=0; i++);	
		return i;	
	}

	double *Sample( int k ) const	{	assert(k>=0&&k<nMost);	return X+ldX*k;	}
	void Copy( int i,const Dataset &hS,int no,int nSample=1,int flag=0x0 );		
//	double Cost( NeuralNet*hNN,int flag );
#ifdef _GST_MATLAB_
	static Dataset* ImportMat( const char *sPath,const char* sVar,int flag );
#endif

	int Load( const std::wstring sPath,int flag );
	int Save( const std::wstring sPath,int flag );
	int ToBmp( int epoch,int _x=0,int flag=0x0 );
};
typedef Dataset* hDATASET;	
double Err_Tag( int lenT,double *acti,int tag,int& nOK,int flag );
double Err_Auto( int lenT,double *in,double *out,int& nOK,int flag );

template<typename T>
std::string G_STR(const T& x){
	std::stringstream ss;
	ss<<x;				
	return ss.str( );
}

class GST_util{
public:
	static int dump;
	static int verify;
	static double tX;

	enum{
		VERIRY_KRYLOV_FORM=1,
		VERIRY_UNITARY_MAT=2,VERIRY_I_MAT=2,VERIRY_TRI_MAT=2,
	};
	static void print( const char *format,... )	{	
		if( dump==0 )
			return;
		char buffer[1000];
		va_list args;
		va_start( args, format );
	//	vfprintf( stderr,format,args );
		vsnprintf( buffer,1000,format,args );
		printf( "%s",buffer );
		va_end(args);
	}
#ifdef _GST_MATLAB_
	static int LoadDoubleMAT( const char *sPath,const char* sVar,int *nRow,int *nCol,double **val,int flag );
#endif
	//	friend std::ostream& operator<<( std::ostream& os,const hGMAT &mat )	{	mat->dump(os);		return os; }
	template<typename T>
	static void Output( char *sPath,int m,int n,T *dat,int flag=0x0 )	{
		std::ofstream file;
		file.open( sPath, std::ios::out );	
		if( file.is_open( ) )	{
			//???
			// hGMAT hMat= make_shared<Matrix<Z>>(m,n,dat,GeMAT::DATA_REFER );
			// file<< hMat.get() << endl;		
			file.close( );
		}
	}
};