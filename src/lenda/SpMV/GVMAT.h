#pragma once

/*
	
*/
#include <cassert>
#include <complex>
#include <memory>
#include <vector>
#include <typeinfo>
#include <float.h>
#include "Pattern.h"
#include "GeQuant.hpp"
using namespace std;

typedef std::complex<double> COMPLEXd;
typedef std::complex<float>  COMPLEXf;

typedef std::complex<double>	Z;
typedef std::complex<float>		C;
typedef float					S;
typedef double					D;

//λ
#define BIT_SET( val,flag ) ((val) |= (flag))	
#define BIT_RESET( val,flag ) ((val) &= (~(flag)) ) 
#define BIT_TEST( val,flag ) (((val)&(flag))==(flag))
#define BIT_IS( val,flag ) (((val)&(flag))!=0)

#define MEM_CLEAR(mem,size)			memset( (mem),(0x0),(size) )

enum G_DATATYPE{
	DATA_UNKNOWN,
	FLOAT_S = 0x1, FLOAT_D = 0x2, FLOAT_C = 0x4, FLOAT_Z = 0x8,
};
enum G_PATTERN{
	PTN_UNKNOWN,
	PTN_DENSE = 0x1, PTN_CCS = 0x2, 
};

class GeMAT;
typedef shared_ptr<GeMAT> hGMAT;
class GeVEC;
typedef shared_ptr<GeVEC> hGVEC;

typedef std::vector<int> SHAPE;

/*
*/
class GeMAT{
	GeMAT(const GeMAT&);
	GeMAT& operator=(const GeMAT&);
protected:
	int nRow,nCol,flag;
	//double *minimum,*maximum;	//range
	GeMAT( ) :  nRow(0),nCol(0),flag(0),type(M_UNKNOWN) {;}
	GeMAT( int m,int n,int flag_=0x0) :  nRow(m),nCol(n),flag(flag_),type(M_UNKNOWN) {;}
	virtual ~GeMAT( ){
		//delete[] minimum;		delete[] maximum;
	}
	virtual	hGMAT AM(hGMAT mV, const hGMAT mU, int flag = 0)	{ throw runtime_error("GeMAT::AM is ..."); }
	template<typename T>
	static G_DATATYPE _dataType() { 
		return	typeid(T)==typeid(double) ? FLOAT_D : typeid(T)==typeid(float) ? FLOAT_S :
				typeid(T) == typeid(COMPLEXd) ? FLOAT_Z :  typeid(T) == typeid(COMPLEXf) ? FLOAT_C: DATA_UNKNOWN; 
	}
 // performance
	struct PERF{
		float 	tX = 0.0;
		int		runs = 0;
		int64_t cycles = 0;
	};
	PERF perf;
public:
	enum TYPE{
		M_UNKNOWN=-1,M_GENERAL=0x0,
		M_SYMMETRY=0x1,M_DIAGONAL=0x02,M_TRIDIAG=0x03,
		M_UNIT=0x10,M_ZERO=0x20,
		M_UNITARY=0x100,
		M_HERMITIAN=0x1000,
	};
	TYPE type;
	enum BIT_FLAG{
		MAT_TRANS=0x100,
		DATA_ZERO=0x10000,
		DATA_REFER=0x20000,
	};
	virtual int RowNum() const	{ return 0; }
	virtual int ColNum() const	{ return 0; }
	virtual int Count() const	{ return RowNum()*ColNum(); }
	SHAPE Shape(int flag=0x0)	{
		SHAPE shape={RowNum(),RowNum()};
		return shape;
	}
	virtual void* Data() const	{ return nullptr; }
	
	virtual void* Column( )	const {	return nullptr;	}
	virtual G_DATATYPE DataType()	{ return DATA_UNKNOWN; }
	virtual G_PATTERN PatternType() { return PTN_UNKNOWN; }

	virtual hGMAT Sub(int r_0, int c_0, int nR, int nC){ return nullptr; }

	virtual void Copy( const hGMAT &mB )	{};

	virtual double Nrm2(int flag=0)					{ throw runtime_error("GeMAT::Nrm2 is vitual!!!"); }
	virtual void Scald( double d,int flag=0 )			{ throw runtime_error("GeMAT::Scald is vitual!!!"); }
//Build Operation
	virtual bool Build(int flag=0x0)	{	throw runtime_error("GeMAT::Build is vitual!!!");	}
//Quantization
	virtual void Range(double *minimum,double *maximum,int start_dim,int flag=0x0)	{}
//Spectral operation
	virtual bool SimilarTrans(hGMAT &mB, int flag = 0) 	{ throw runtime_error("GeMAT::SimilarTrans is vitual!!!"); }
//Forward operation
	virtual hGVEC Transform(hGVEC &vecY, const hGVEC &vecX, int flag = 0)	{ throw runtime_error("GeMAT::Transform is vitual!!!"); }
	virtual hGMAT& Transform(hGMAT &mV, const hGMAT &mU, int flag = 0)	{ throw runtime_error("GeMAT::Base method"); }
//	virtual	hGMAT Mul(hGMAT mV, const hGMAT mU, int flag = 0);
// More utility
	friend std::ostream& operator<<( std::ostream& os,GeMAT* hmat )	{	hmat->dump( os );		return os; }
	friend std::ostream& operator<<( std::ostream& os,const hGMAT &mat )	{	mat->dump(os);		return os; }
	virtual void dump( std::ostream& os )					{ throw runtime_error("GeMAT::dump unimplemented!!!"); }
	virtual void dump( const std::string& path_,const std::string& sX=nullptr,int flag=0x0 ){
	try{
		ios_base::openmode mode = BIT_TEST(flag,0x100) ? ios_base::out|ios_base::app : ios_base::out;
		std::string sPath(path_);
		if(!sX.empty())
			sPath+=sX;
		std::ofstream file( sPath,mode  );
		if( file.is_open( ) )		{
			file<<__func__<<":\tN="<<RowNum()<<","<<ColNum()<<"\n";
			dump(file);		file.close( );		
		}
	}catch(...){
		throw  std::ios_base::failure(__func__);
	}
	}
};

template<typename T>
T* TO(const hGMAT hA) 	{
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf))	{
//		G_DATATYPE dt=GeMAT::_dataType<T>();
//		assert( dt==hA->DataType() );
		return (T*)(hA->Data());
	}	else {
		return (T*)(hA.get());
	}
	throw runtime_error("TO"); 
}


template<typename T>
T* TO(const hGMAT hA,int r,int c ) 	{	//hGMAT�Ĵ洢Ϊ������
//		size_t tid = typeid(T).hash_code( );
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf))	{
		T* data = TO<T>(hA);
		int nRow=hA->RowNum(),nCol=hA->ColNum( );
		if( r<0 || r>=nRow || c<0 || c>=nCol )
			throw range_error( "TO(const hGMAT hA,int r,int c )");
		return data+c*nRow+r;
	}	else {
		throw runtime_error("TO"); 
	}
	throw runtime_error("TO"); 
}

template<typename T>
T* TO(const hGMAT hA,int c ) 	{	//hGMAT�Ĵ洢Ϊ������
//		size_t tid = typeid(T).hash_code( );
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(COMPLEXd) || typeid(T) == typeid(COMPLEXf))	{
		T* data = TO<T>(hA);
		int nCol=hA->ColNum( ),nRow=hA->RowNum( );
		if( c<0 || c>=nCol )
			throw range_error( "TO(const hGMAT hA,int r,int c )");
		return data+c*nRow;
	}	else {
		throw runtime_error("TO"); 
	}
	throw runtime_error("TO"); 
}

#define ToZ TO<Z>
#define ToD TO<D>
#define ToS TO<S>
#define ToC TO<C>
#define ToT TO<T>

class GeVEC : public GeMAT {
public:
	virtual int ColNum()	{ return 1; }
};

class Filter :public GeMAT {
private:
	hGMAT hBase;
protected:
	hGMAT hFilt;

public:
	Filter( const hGMAT &hA ) : hBase(hA) {;}
	virtual int RowNum() const	{ return hBase->RowNum(); }
	virtual int ColNum() const	{ return hBase->ColNum(); }

	virtual double Nrm2(int flag=0)					{ return DBL_MAX; }
//	virtual hGVEC Transform(hGVEC &vec1, const hGVEC &vec0, int flag = 0)	{ return vec1;  }
//	virtual hGMAT& Transform(hGMAT &M1, const hGMAT &M0, int flag = 0)		{ return M1; }
	virtual hGVEC Transform(hGVEC &vec1, const hGVEC &vec0, int flag = 0)	{ 
		assert( hFilt!=nullptr );
		hFilt->Transform( vec1,vec0 );  
		return vec1;
	}
	virtual hGMAT& Transform(hGMAT &M1, const hGMAT &M0, int flag = 0)		{ 
		assert( hFilt!=nullptr );
		hFilt->Transform( M1,M0 );  
		return M1;
	}
	virtual hGVEC TransSpectral( const hGVEC &vLenda,bool inverse,int flag=0x0 )		{	return vLenda;	}
};
typedef shared_ptr<Filter> hFILTER;

template<typename T>
struct Quantizer{
	// TODO: len(shape)=4,3,2
	SHAPE shape;
	int bits = -1;
	double maxq,maxshrink,mse,norm,grid;
	T *scale=nullptr,*zero=nullptr;
	bool trits,perchannel=false,sym=false;

	virtual void Flattern(	)	{
	}

	Quantizer(int bits_, bool perchannel_=false, bool sym_=true, bool mse_=false, double norm_=2.4, int grid_=100, 
		double maxshrink_=.8,bool trits_=false) : bits(bits_),trits(trits_),perchannel(perchannel_),sym(sym_)	{
		maxq = pow(2.0,bits) - 1;
		if(trits)
            maxq = -1;
	}

	//onlys support shape==2
	void Init(hGMAT x, bool weight=false){
		shape=x->Shape();			assert(shape.size()==2);
		int i,ld=shape[0];
		double *xmax = new double[ld](), *xmin = new double[ld]();
		x->Range(xmin,xmax,perchannel ? 1 : 0);		
		scale = new T[ld];	zero = new T[ld];		
		
		if(maxq<0){
			for(i=0;i<ld;i++){
				scale[i] = xmin[i];          zero[i] = xmax[i];
			}			
		}else{
			for(i=0;i<ld;i++)	{
				scale[i] = (xmax[i] - xmin[i]) / maxq;
				//On CPU tensors rounds half-to-even and on GPU tensor rounds away-from-zero !
				zero[i] = round(-xmin[i]/scale[i]);	// torch.round(-xmin / self.scale)
			}
			double T_zero = (maxq + 1) / 2;
			if(sym){
				for(i=0;i<ld;i++)	{
					zero[i] = T_zero;			//torch.full_like(self.scale, (self.maxq + 1) / 2)
				}				
			}								
		}
		if(GST_util::dump>0){
			GST_util::print("+ %s x=[%g-%g] scale=[%g,%g] zero=[%g,%g]\n",__func__,xmin[0],xmax[0],scale[0],scale[ld-1],zero[0],zero[ld-1]);
			if(ld<16){
				for(i=0;i<ld;i++)	printf("%g ",scale[i]);		printf("\n");
				for(i=0;i<ld;i++)	printf("%g ",zero[i]);		printf("\n");
			}			
		}

		delete[] xmax;		delete[] xmin;
	}
	
	virtual double Update(int len,T*col,T*q,int type,T d,T* err,int ld,int flag=0x0)	{
		double loss=0;
		if( maxq < 0){
//return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
			return loss;
		}
		double a=0,w;
		// T *col=val+no,*q=val+no,*err=hERR->val+no;
		for(int i=0;i<len;i++,col+=ld){
			if(i==3)	//only for debug
				i = 3;
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
	virtual ~Quantizer( ){
		delete[] scale;		delete[] zero;
	}
};


void Unitary( hGVEC &mV,int flag=0 );

//void DOT(void *dot, const hGVEC vY, const hGVEC vX, int flag = 0);

/*	V=AU;		return V */
//hGMAT Mul(hGMAT mV, const hGMAT mA, const hGMAT mU, int flag = 0);
//hGVEC Mul(hGVEC mV, const hGMAT mA, const hGVEC mU, int flag = 0);
void inline Swap(hGMAT &vX, hGMAT &vY)	{ vX.swap(vY); }
void inline Swap(hGVEC &vX, hGVEC &vY)	{ vX.swap(vY); }

