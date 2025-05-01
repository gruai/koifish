#pragma once

#include <cassert>
#include <complex>
#include <memory>
//#define __STDC_WANT_LIB_EXT1__ 1
#include <string.h>
#include "GVMAT.h"
#include "Pattern.h"
#include "../../Utils/BLAS_t.hpp"
#include "../../Utils/GST_util.hpp"
#include "../../g_float.hpp"

using namespace std;

template<typename T>
class Vector : public GeVEC {
private:
	typedef Vector<T> _Mybase;
	T *data;
public:
//	int len;
//	typedef shared_ptr<Vector> handle;
//	typedef const shared_ptr<Vector> const_handle;
/*	Vector( int len,... )	{
		assert(len>=1);
		nRow=len;		flag=0x0;		
		size_t szd=(sizeof(T)*nRow);
		data=static_cast<T *>( operator new[](szd) );	
		va_list arg_ptr;
		va_start( arg_ptr,len );
		for( int i=0; i < len; i++ ){
			data[i]=va_arg(arg_ptr,T);
		}
		va_end(arg_ptr);
	}*/

	virtual void dump( std::ostream& os )					{ for( int i = 0; i < nRow; i++ )	os<<data[i]; }

	Vector( int len,T *v=nullptr,int flg=0x0 )	{
		assert(len>0 );
		nRow=len;		flag=flg;
		if( BIT_TEST( flag,DATA_REFER) )	{
			data = v;
		} else		{
			size_t szd=(sizeof(T)*nRow);
			data=static_cast<T *>( operator new[](szd) );
			if( v!=nullptr )	{
				//memcpy_s( data,szd,v,szd );
				memcpy( data,v,szd );
			}		else		{
				memset( data,0x0,szd );
			}
			if(  BIT_TEST( flag,DATA_ZERO) )
				memset( data,0x0,szd );
		}
		assert( data!=nullptr );
	}
	~Vector( )	{
		if( BIT_TEST( flag,DATA_REFER) )	{
		} else		{
			operator delete[] (data);
		}
	}

	virtual int RowNum() const override{ return nRow; }
	virtual int ColNum() const override{ return 1; }
	virtual void* Data() const override  {	return data;		}
	virtual G_DATATYPE DataType() override	{ 		return	GeMAT::_dataType<T>(); 	}

	void Set( double *v,int lv )		{
		for( int i = 0; i < lv; i++ )	data[i]=v[i];
	}
	void Set( T a,int lv )		{		for( int i = 0; i < lv; i++ )	data[i]=a;	}

	double Nrm2(int flag = 0)	override	 {		return 	NRM2( nRow,data );		}

	void Scald( double alpha,int flag )	override		{ SCALd( nRow,alpha,data ); }
};

typedef Vector<double> VEC_d;
typedef Vector<COMPLEXd> VEC_z;

template<typename T>class MatCCS;

/*
	1 ֻ��AX����
*/
template<typename T>
class GeAX :public GeMAT {
public:
	typedef void (*AX_FUNC)(T *vIn,T *vOut) ;
private:
	AX_FUNC hAX;
	T shift;
protected:
public:
	//typedef void (*AX_FUNC)(T *vIn,T *vOut) ;

	GeAX( ) : GeMAT(),hAX(nullptr)	{;	}
	GeAX( int dim,AX_FUNC ax,T sft=T(0) ) : hAX(ax),shift(sft)
	{	nRow=dim,			nCol=dim;	}

	void SetShift( T s )	{	shift=s;	}
	
	virtual hGVEC Transform(hGVEC &vecY, const hGVEC &vecX, int flag = 0)	override {		//y=Ax
		int m=vecY->RowNum(),n=vecX->RowNum(),ldA=RowNum();
		assert( m==RowNum() && n==ColNum() );
		T *X=TO<T>(vecX),*Y=TO<T>(vecY);
		hAX( X,Y );
		return vecY;
	}

	virtual hGMAT& Transform(hGMAT &mV, const hGMAT &mU, int flag = 0) override	{ //V=AU 
		int m=mV->RowNum(),n=mU->RowNum(),nCol=mU->ColNum( ),i;
		assert( m==RowNum() && n==ColNum() && nCol<=mV->ColNum( ) );
		T *tv=TO<T>(mV),*tu=TO<T>(mU);
		for( i = 0; i < nCol; i++ )	{
			T *X=tu+i*n,*Y=tv+i*m;
			hAX( X,Y );
		}
		return mV;
	}

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

	virtual double Nrm2(int flag=0)					{ return DBL_MAX; }
	virtual int RowNum() const override{ return nRow; }
	virtual int ColNum() const override{ return nCol; }
	virtual G_DATATYPE DataType() override	{ 		return	GeMAT::_dataType<T>(); 	}
};
typedef GeAX<double> GeAX_d;
typedef GeAX<COMPLEXd> GeAX_z;

/*
	1 �����ȱʡ�洢��ʽΪ������

*/
template<typename T>
class Matrix :public GeMAT {
private:
	typedef Matrix<T> _Mybase;
	typedef shared_ptr<Matrix<T>> _MyPTR;
protected:
	T *val;

	void create( int nR,int nC,TYPE tp=M_GENERAL,T* v0=nullptr,size_t szV=0,int flg=0x0 )	{
		nRow=nR,		nCol=nC,		type=tp;
		if( val!=nullptr)	operator delete[](val);
		nRow=nR;		nCol=nC;			flag=flg;
		type=tp;		val=nullptr;
		if( BIT_TEST( flag,DATA_REFER) )	{
			val = v0;		
		} else		{
			if( szV<=0 )	
				szV=nR*nC;
			val=static_cast<T *>( operator new[](szV*sizeof(T)) );
			if( v0!=nullptr )	{
				//memcpy_s( val,szV*sizeof(T),v0,szV*sizeof(T) );
				memcpy( val,v0,szV*sizeof(T) );
			}else if( BIT_TEST( flag,DATA_ZERO) )	{
				memset( val,0x0,szV*sizeof(T) );
			}
		}
		// GST_util::print("+ %s nRow=%d nCol=%d val=[%g-%g]\n",__func__,nRow,nCol,val[0],val[nRow*nCol-1]);
		assert( val!=nullptr );
	}
public:
//	typedef typename shared_ptr<_Mybase> handle;
//	typedef const shared_ptr<_Mybase> const_handle;	

	typNUMBER  tpOut = typNUMBER::F16;		//default is float
	Matrix( ) : GeMAT(),val(nullptr)	{;	}
	Matrix( int nR,int nC,int flg=0x0,TYPE tp=M_GENERAL ) : val(nullptr)
	{	create( nR,nC,tp,nullptr,0,flg );	}
	Matrix( int nR,int nC,T *v,int flg=DATA_REFER,TYPE tp=M_GENERAL ) : val(nullptr)
	{	create( nR,nC,tp,v,0,flg );	}
	Matrix( const hGMAT &mS ) : val(nullptr)		
	{	create( mS->RowNum(),mS->ColNum(),mS->type );	}	
	Matrix( const shared_ptr<Matrix<T>> mS,int flag=0x0 ) : val(nullptr)		
	{	create( mS->RowNum(),mS->ColNum(),mS->type,mS->val,0x0,flag );	}	

	virtual void Copy(const shared_ptr<Matrix<T>> mS,int flag=0x0)	
	{	memcpy(	val,mS->val,sizeof(T)*nRow*nCol);	}

	
	virtual bool SimilarTrans(hGMAT &mQ, int flag = 0) override	{ //similar transforma
#ifdef _USE_OPENBLAS_
		if( nRow!=nCol || nRow!=mQ->RowNum( ) )
			return false;
		if( !BIT_TEST( mQ->type,M_UNITARY) )	
			return false;
		T *Q=TO<T>(mQ),*temp=new T[nRow*nRow],one(1.0),zero(0.0);
		GEMM( charN,charN,nRow,nRow,nRow,&one,Q,nRow,val,nRow,&zero,temp,nRow);		//AQ
		GEMM( charC,charN,nRow,nRow,nRow,&one,Q,nRow,temp,nRow,&zero,val,nRow);		//Q'A
		delete[] temp;
#endif
		return true;
	}

	virtual hGVEC Transform(hGVEC &vecY, const hGVEC &vecX, int flag = 0)	override {		//y=Ax
		int m=vecY->RowNum(),n=vecX->RowNum(),ldA=RowNum();
		assert( m<=RowNum() && n<=ColNum() );
		char trans=charN;
		T *X=TO<T>(vecX),*Y=TO<T>(vecY),*A=val,one=1,zero=0;
		GEAV<T>( trans,m,n, A, ldA, X, Y);
		return vecY;
	}

	virtual hGMAT& Transform(hGMAT &mV, const hGMAT &mU, int flag = 0) override	{ //V=AU || V=A'U
#ifdef _USE_OPENBLAS_
		int m=mV->RowNum(),n=mV->ColNum(),k=mU->RowNum(),ldA=RowNum();
		bool isTrans=BIT_TEST( flag,MAT_TRANS );
		char charA=charN;
		if( isTrans )	{
			assert( m==ColNum( ) && k==RowNum() && n==mU->ColNum() );
			charA = charT;
		} else		{
			assert( m==RowNum() && k==ColNum( ) && n==mU->ColNum() );
		}
		T *tV=TO<T>(mV),*tU=TO<T>(mU),*tA=val,one=1,zero=0;
		GEMM( charA,charN,m,n,k,&one,tA,ldA,tU,k,&zero,tV,m);		
#endif
		return mV;
	}

	
	virtual int RowNum() const override		{ return nRow; }
	virtual int ColNum() const override		{ return nCol; }
	virtual void* Data() const override 	{ return val; }
	virtual T Diagnol(int no) const  		{ assert(nRow==nCol && no<nCol);	return val[no*nCol+no]; }
	virtual T* Row(int no) 		{ assert(no>=0 && no<nRow);	return val+no*nCol; }
	virtual T* Column(int no) 	{ assert(no>=0 && no<nCol);	return val+no; 		}

	virtual void Range(double *minimum,double *maximum,int start_dim,int flag=0x0){
		assert(start_dim==1);
		int i,j;
		T *p = val;
		double a_0=DBL_MAX,a_1=-DBL_MAX;
		for(i=0;i<nRow;i++)		{
			for(j=0;j<nCol;j++,p++)		{
				double a = (double)(*p);
				minimum[i] = min(minimum[i],a);
				maximum[i] = max(maximum[i],a);
			}
			a_1 = max(maximum[i],a_1);
			a_0 = min(minimum[i],a_0);			
		}
	}
/*
	//0-|arr|		1-|arr.real|
template<typename T>
void MIN_MAX( size_t len,T *arr,double&a_0,double&a_1,int flag=0x0 ){
	size_t i;
	double a;
	if( flag==1 ){
		for( a_0=DBL_MAX,a_1=-DBL_MAX,i=0;i<len;i++ ){
			COMPLEXd z=arr[i];
			a = z.real( );
			a_0=MIN(a_0,a);		a_1=MAX(a_1,a);
		}
	}else{
		for( a_0=DBL_MAX,a_1=-DBL_MAX,i=0;i<len;i++ ){
			a = ABS_1(arr[i]);
			a_0=MIN(a_0,a);		a_1=MAX(a_1,a);
		}
	}
}*/

	virtual double Quanti_col(shared_ptr<Quantizer<T>> hQuant,int no,int type,T d,_MyPTR hERR ,int flag=0x0)	{
		T *scale=hQuant->scale,*zero=hQuant->zero,maxq=hQuant->maxq,loss=0;
		// hQuant->Update<T>(type,nRow,val+no,q,0x0);
		if( maxq < 0){
//return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
			return loss;
		}
		int ld=nCol,len=nRow;		//row-major
		double a=0,w;
		T *col=val+no,*q=val+no,*err=hERR->val+no;
		for(int i=0;i<len;i++,col+=ld){
			if(no==2 && i==3)	//only for debug
				no = 2;
			w = *col;
			a = round((*col)/scale[i]+zero[i]);	//10., 13., 10.,
			if(a<0)	a= 0;
			if(a>maxq)	a=maxq;
			*q = scale[i]*(a-zero[i]);		//0.0618,  0.1926,  0.0604, -0.0526,  0.0000,  0.1382, -0.1075, -0.0445,
			w -= *q;
			*err = w/d;				//err1 = (w - q) / d
			loss += w*w/d/d;		//Losses1[:, i] = (w - q) ** 2 / d ** 2
			q += ld;		err += ld;
		}
		//GST_util::print();
		return loss;	
	}
	virtual G_DATATYPE DataType() override	{ 		return	GeMAT::_dataType<T>(); 	}
//	virtual typename Vector<T>::handle operator*( const Vector<T> &vec ) const	{		throw std::runtime_error( "operator*" );	}

	virtual ~Matrix(void){
		if( BIT_TEST( flag,DATA_REFER) )	{
		} else		{
			if( val!=nullptr)	operator delete[](val);	//delete[] val;
		}
	}

	virtual void Shift( T sift )				{
		for( int i = 0 ; i < nCol; i++ )
			val[i*nRow+i] -= sift;
	}
	bool isSquare( )	const	{	return nRow==nCol;	}

	virtual void dump( std::ostream& os )					{ 
		if(0)		{	//row major
			for( int i = 0; i < nRow; i++ )	{	
				for( int j = 0; j < nCol; j++ )	{	//column major
					os<<val[i*nCol+j]<<" "; 
					//if(j>8)	{os<<"...D="<<val[i*nCol+i]<<"..."<<val[i*nCol+nCol-1];	break;}
				}	
				os<<endl;
			}
		}	else	{		//column major
			for( int i = 0; i < nCol; i++ )	{	
				os<<"Column "<<i<<endl;
				for( int j = 0; j < nRow; j++ )	{	//column major
					os<<val[j*nRow+i]<<" "; 
					// if(j>8)	{	os<<"...D="<<val[j*nRow+j]<<"..."<<val[j*nRow+nRow-1];	break; 	}
				}	
				os<<endl;
			}			
		}
	}
};
typedef Matrix<double> MAT_d;
typedef Matrix<COMPLEXd> MAT_z;

void TRI_1_Z( int dim,int** colptr,int** rowind,Z **val,Z**rhs );

template<typename T>
class MatCCS : public Matrix<T>	{
	typedef MatCCS<T> _Mybase;
protected:

	virtual	hGMAT AM(hGMAT mV, const hGMAT mU, int flag = 0)	{ throw exception(); }
public:
//	typedef shared_ptr<_Mybase> handle;
//	typedef const shared_ptr<_Mybase> const_handle;	

	static hGMAT Produce( int dim,GeMAT::TYPE type,int flag )	{
		hGMAT hmat=nullptr;
		int *t_ptr=nullptr,*t_ind=nullptr;
		T *t_val=nullptr;
		switch( type )	{
		case GeMAT::M_TRIDIAG:
			TRI_1_Z( dim,&t_ptr,&t_ind,&t_val,nullptr );
			hmat = make_shared<MatCCS<T>>(dim,t_ptr,t_ind,t_val,GeMAT::M_GENERAL);
			operator delete[](t_ptr);		operator delete[](t_ind);		operator delete[](t_val);
		break;
		default:
			assert(false);
		break;
		}
		return hmat;
	}

	CCS_pattern pattern;
	template<typename Ta0>
	MatCCS( MatCCS<Ta0> &mA0 )	: pattern( mA0.pattern )	{
		int i,nnz=mA0.nNZ( );
		Ta0 *ta0=(Ta0*)(mA0.Data());	//(Ta0 *)(mA0);
		create( mA0.RowNum(),mA0.ColNum(),mA0.type,nullptr,nnz );
		for( i = 0; i < nnz; i++ )		
			this->val[i] = ta0[i];
	}

	MatCCS( int d,int *p,int *i,T *v,GeMAT::TYPE m_type=GeMAT::M_GENERAL ) : pattern( d,p,i,0) {
		create( d,d,m_type,v,p[d]);
	}
	hGMAT Sub(int r_0, int c_0, int nR, int nC)	{ 
		int nRow = nR, nCol = nC, nnz = 0, i, j, r, pos = 0, *ptr = pattern.ptr, *ind = pattern.ind,nzMost = ptr[c_0 + nC] - ptr[c_0];
		int *ptr_1 = new int[nC+1], *ind_1 = new int[nzMost];
		T *val_1 = new T[nzMost];
		ptr_1[0] = 0;
		for( i=c_0; i<c_0+nC; i++ )	{
			for( j = ptr[i]; j<ptr[i+1]; j++ )	{
				r = ind[j];
				if( r<r_0 || r>=r_0+nR )
					continue;
				ind_1[pos] = r-r_0;
				val_1[pos++] = this->val[j];
			}
			ptr_1[i-c_0+1] = pos;
		}
		assert(pos <= nzMost);
		return make_shared<MatCCS<T>>(nC, ptr_1, ind_1, val_1);
	}

	int nNZ( )	{	return pattern.nNZ( );	}

	virtual void Shift( T sift )	override			{
		int i,j;
		for( i = 0 ; i < this->nCol; i++ )	{
			for( j = pattern.ptr[i]; j < pattern.ptr[i+1]; j++)	{
				if( pattern.ind[j]==i )
					this->val[j] -= sift;
			}
		}
	}

	template <typename Tv>
	hGVEC Ax(hGVEC vecY, const hGVEC vecX, int flag = 0) {
		Tv *x = TO<Tv>(vecX);
		Tv *y = TO<Tv>(vecY);
		int i, j, *ptr = pattern.ptr, *ind = pattern.ind, r;
		bool isSym =  BIT_TEST( this->type,GeMAT::M_HERMITIAN );
		memset( y,0x0,sizeof(Tv)*this->nRow );
	/*	if( 1 )	{
			double alpha=1.0,beta=0.0;
			char transa='N',matdescra[6]="\0";
			mkl_zcscmv (transa, &nRow,  &nCol,&alpha, matdescra,val,ind,pattern.ptr,pattern.ptr_e,x,beta,y);
		} else	*/	{
			for (i = 0; i < this->nCol; i++)	{
				Tv a = x[i];
				for (j = ptr[i]; j < ptr[i + 1]; j++)	{
					r = ind[j];				assert(r >= 0 && r<this->nRow);
					y[r] += this->val[j] * a;
					if( isSym && r!=i )	{
						y[i] += this->val[j] * x[r];
					}
				}
			}

		}
		return vecY;
	}

	hGVEC Transform(hGVEC &vecY, const hGVEC &vecX, int flag = 0)	override {
		Ax<T>( vecY,vecX,flag );
		return vecY;
	}

	template<typename Tm>
	void AU( hGMAT &mV,const hGMAT &mU,int flag = 0 )	const	{	//V=AU
		int i,j,k,*ptr=pattern.ptr,*ind=pattern.ind,r,M=mU->RowNum(),N=mU->ColNum();
		Tm *tv=TO<Tm>(mV),*v,*tu=TO<Tm>(mU),*u;
		bool isSym =  BIT_TEST( this->type,GeMAT::M_HERMITIAN );
		memset( tv,0x0,sizeof(Tm)*M*N );
		for( i = 0; i < this->nCol; i++ )	{				
			for( j = ptr[i]; j < ptr[i+1]; j++ )	{
				r = ind[j];
				v=tv+r;			u=tu+i;
				T a = this->val[j];
				for( k = 0; k < N; k++)
					v[k*M] += a*u[k*M];
				if( isSym && r!=i )	{
					v=tv+i;		u=tu+r;	
					for( k = 0; k < N; k++)
						v[k*M] += a*u[k*M];
				}
			}
		}
		return ;
	}

	hGMAT& Transform(hGMAT &mV, const hGMAT &mU, int flag = 0)	override {		//V=AU
		AU<T>( mV,mU,flag );
		return mV;
	}

	double Nrm2(int flag = 0)	override	 {		return 	NRM2( nNZ( ),this->val );		}
};
typedef MatCCS<double> CCS_d;
typedef MatCCS<COMPLEXd> CCS_z;



template<typename T>
class MatDiagonal : public Matrix<T>	{
public:
	MatDiagonal( int d,T *v,int m_type ) : Matrix<T>( d,d,nullptr,GeMAT::M_DIAGONAL|m_type){
		this->val = new T[d*d]();
		int i;
		for( i = 0;	i < d*d; i++ )	{	this->val[i]=1.0e-10;	}
		for( int i = 0; i < d; i++ )	{
			this->val[i*d+i]=v[i];
			//memcpy( val+i*d+i,v+i,sizeof(T) );
		}
	}
	int Decomposition( int type,int flag )	{		return 0;	}
	//	x=inv(A)*y
	virtual int SOLVE( T *x,T *y )	{	
		for( int i = 0; i < this->nRow; i++ )	{
		//	assert(val[i]!=0.0);	
		//	x[i]=y[i]/val[i];
		}
		return 0x0;
	}
};

