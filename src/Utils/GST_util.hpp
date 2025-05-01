/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief GRUS SPARSE TEMPLATE	- Utility
 *  \author Yingshi Chen
 */

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
#include <time.h>
using namespace std;

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(WIN32)
	
	#define GST_NOW( )		(clock( ))
	#define GST_TIC(tick)	clock_t tick=clock( );
	#define GST_TOC(tick)	((clock()-(tick))*1.0f/CLOCKS_PER_SEC)

	static int64_t timer_freq, timer_start;
	inline void GST_time_init(void) {
		LARGE_INTEGER t;
		QueryPerformanceFrequency(&t);
		timer_freq = t.QuadPart;

		// The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
		// and the uptime is high enough.
		// We subtract the program start time to reduce the likelihood of that happening.
		QueryPerformanceCounter(&t);
		timer_start = t.QuadPart;
	}
	inline int64_t GST_ms(void) {
		LARGE_INTEGER t;
		QueryPerformanceCounter(&t);
		return ((t.QuadPart-timer_start) * 1000) / timer_freq;
	}
	inline int64_t GST_us(void) {
		LARGE_INTEGER t;
		QueryPerformanceCounter(&t);
		return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
	}
#else
	#include <chrono>
	#include <thread>

	typedef std::chrono::high_resolution_clock Clock;
	#define GST_NOW( )	(Clock::now( ))
	#define GST_TIC(tick)	auto tick = Clock::now( );
	// #define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now( )-(tick)).count( ))/1000.0)
	#define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now( )-(tick)).count( ))/1000000.0)

	inline void GST_time_init(void) {}

	//	A millisecond is a unit of time in the International System of Units equal to one thousandth of a second or 1000 microseconds
	inline double GST_ms(void) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return (int64_t)ts.tv_sec*1000.0 + (int64_t)ts.tv_nsec/1000000.0;
	}

	// A microsecond is equal to 1000 nanoseconds or 1⁄1,000 of a millisecond
	inline double GST_us(void) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return (int64_t)ts.tv_sec*1000000.0 + (int64_t)ts.tv_nsec/1000.0;
	}
#endif

#define TIMING_ms(a,sum) \
    do { \
        double t0=GST_ms(); \
        a; \
        sum += GST_ms()-t0; \
    } while (0)

/*
	Dataset 		 10/19/2014		cys
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

std::string FILE_EXT(const std::string&path);

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