#pragma once
#include <memory> //for shared_ptr
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
#include <random>
#include <math.h>
#include <float.h>
#include <assert.h>
using namespace std;

#define BIT_SET( val,flag ) ((val) |= (flag))	
#define BIT_RESET( val,flag ) ((val) &= (~(flag)) ) 
#define BIT_TEST( val,flag ) (((val)&(flag))==(flag))
#define BIT_IS( val,flag ) (((val)&(flag))!=0)

#define MEM_CLEAR(mem,size)			memset( (mem),(0x0),(size) )

#ifdef WIN32
	#define G_INT_64 __int64
	//typedef __int64 INT_63;	
	#if !defined (_G_INT64_) 
		typedef int INT_63;
	#else
		typedef __int64 INT_63;
	#endif
#else
	#if !defined (_G_INT64_) 
		#define INT_63 int
	#else
		#define INT_63 long long
	#endif
	// #define G_INT_64  long long
#endif

//Prefer a struct when you can. It may involve some overhead, but is definitely easier for maintenance.
/*
    64-bit ID + double weight
*/
struct N64w {
    INT_63 id=-1;
    double w=0;
    N64w(){}
    N64w(INT_63 id_, double w_) : id(id_),w(w_){
    }
};

template<typename T>
T G_S2T_(const string& s,T init){
    std::stringstream ss(s);
    T x=init;
    ss>>x;
    return x;
}

template<typename T>
void G_S2TTT_(const string& str_, vector<T>&nums, const char* seps=" ,:;{}()\t=", int flag = 0x0) {
    nums. clear();
    string str = str_;
    char *token = strtok((char*) str.c_str(), seps);
    while (token != NULL) {
        if( typeid(T)==typeid(int) ){
            int nInt;
            if( sscanf( token,"%d",&nInt)==1 ) {
                nums.push_back(nInt);
            }
        }else if( typeid(T)==typeid(double) || typeid(T)==typeid(float) ){
            double a;
            if( sscanf( token,"%lf",&a)==1 ) {
                nums.push_back(a);
            }
        }else {
            throw "Str2Numbers is ...";
        }
        token = strtok(NULL, seps);
    }
}

template <typename T>
void G_SomeXat_( vector<int>& pos, const vector<T>& someX, const vector<T>& allX, int flag=0x0 ){
    for( T legend : someX ){
        for( int i=0;i<allX.size();i++) {
            if( legend!=allX[i] )
                continue;
            //arrV4X.push_back( make _ pair(allX[i],i) );
            pos.push_back( i );
        }
    }
}

inline int CLAMP(const int v, const int min, const int max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

inline size_t HASH_combine(size_t h1, size_t h2) {
    return h1 ^ (h2 << 1);
}

template <typename T>
int G_ClosesTick( const T& x, const vector<T>&ticks, int flag=0x0 ){
    int i,nTick=ticks. size( ), tick=-1;
    double dis=0, dis_0=DBL_MAX;
    for( i=0; i<nTick; i++ ){
        dis = abs(ticks[i]-x);
        if( dis<dis_0 ){
            dis_0=dis; tick=i;
        }
    }
    return tick;
}

/*
http://stackoverflow.com/questions/15735035/why-dont-vectors-need-to-be-manually-freed
vector's destructor simply destroys every contained element, then deallocates its internal array.
But Destroying a plain pointer doesn't destroy what it points to!!!
*/
template<typename T>
void FREE_hVECT( vector<T*>& vecs ){
    for( T *t : vecs)
        delete t;
    vecs. clear( );
}

template<typename T>
T *NEW_( size_t len,T a0 ){
    T *arr=new T[len];
    for( int i=0;i<len;i++ ) arr[i]=a0;
    return arr;
}

//delete[] array
template<typename T>
void FREE_a( T* &ptr ){
    if( ptr!=nullptr ) {
        delete[] ptr;
        ptr=nullptr;
    }
}

#ifdef HOPSCOTCH_MAP_LIB
    template<typename K, typename V>
    //V *HT _FIND( unordered_map<K,V*>&HT,K key){
    V *HT_FIND( tsl::hopscotch _ map<K,V*>δHT,K key){
    if( HT. find(key)==HT. end( ) )
    return nullptr;
    else
    return HT[key];
    }
#else
    template<typename K, typename V>
    V *HT_FIND( unordered_map<K,V*>&HT,K key){
        if( HT. find(key)==HT. end( ) )
            return nullptr;
        else
            return HT[key];
    }
#endif

template <typename T> 
bool isInRange(const T* inp,size_t nz,T t0,T t1){    
    for(size_t i=0;i<nz;i++,inp++){
        if(*inp<t0 || *inp>t1)
            return false;
    }
    return true;
}

// double a = fabs(arr[i]) !!!
template<typename T, typename tpFF>
void G_minmax(T *arr,tpFF dim, double& a_0, double& a_1,tpFF&nz){
    nz = 0;
    a_0 = DBL_MAX; a_1=0;
    assert(arr!=nullptr);
    for(tpFF i=0;i<dim;i++){
        if(arr[i]!=0) {
            double a = fabs(arr[i]);
            a_0=std::min(a_0,a), a_1=std::max(a_1,a);
            nz++;
        }
    }
}


template<typename T, typename W>
struct COMPACT_GRAPH    {
    typedef W WEIGHT;
    T nNode=0,nEdge=0;
    T *ptr=nullptr,*adj=nullptr;
    int *_tag=nullptr; //the tag of each vertex(column)
    W *_wNode=nullptr,*_wEdge=nullptr;

    bool isWeighted() { return _wNode!=nullptr || _wEdge!=nullptr; }

    ~COMPACT_GRAPH() {
        FREE_a(ptr); FREE_a(adj);
        FREE_a(_tag);
        FREE_a(_wNode); FREE_a(_wEdge);
    }
};
typedef std::shared_ptr<COMPACT_GRAPH<int, float>> GRAPH_32;
typedef std::shared_ptr<COMPACT_GRAPH<INT_63, float>> GRAPH_64;
template<typename T, typename W>
struct WeightID {
    typedef W WEIGHT;
    T nMost=-1,N=-1;
    T *no=nullptr;
    W *weights=nullptr;

    WeightID(T nM_,T *n_,W *w_) : nMost(nM_), no(n_), weights(w_) {
    }

    inline W operator[](const T pos) const {
        return weights[pos];
    }

    void Set(T id,T n,W a){
        assert(id>=0 && id<nMost);
        no[id]=n; weights[id] = a;
    }

    void Update(T id,T n,W a){
        assert(id>=0 && id<N);
        no[id]+=n; weights[id] += a;
    }
    
    void MergeOnW(T id,W a, int flag=0x0) {
        if(weights[id]==0){
        no[N++] = id;
        }
        weights[id] += a;
    }
};
struct COMPACT_I {
    INT_63 dim=0,nSplit=0,nMostSplit=0;
    INT_63*ptr=nullptr,*no=nullptr,*type=nullptr;
    COMPACT_I() {}
    COMPACT_I(INT_63 _dim,INT_63 _nMostS) : dim(_dim),nMostSplit(_nMostS){Init(_dim,_nMostS);
    }
    virtual void Init(INT_63 _dim,INT_63 _nMostS,INT_63 flag=0x0) {
        dim=_dim; nMostSplit=_nMostS;
        nSplit=0;
        no=new INT_63[dim]; type=new INT_63[dim]();
        ptr=new INT_63[nMostSplit+1]();
        ptr[0] = 0;
        if(flag==1){
            nSplit = nMostSplit;
        }
    }
    virtual ~COMPACT_I(){
        FREE_a(ptr);
        FREE_a(no); FREE_a(type);
    }
    virtual void OnNZ(INT_63 *nz){
        ptr[0] = 0;
        for(INT_63 i=0;i<nSplit;i++){
            ptr[i+1] = ptr[i]+nz[i];
            nz[i] = ptr[i];
        }
    }
    INT_63 N(INT_63 i) { assert(i>=0 && i<dim); return ptr[i+1]-ptr[i]; }
    const INT_63 *NOs(INT_63 i) { assert(i>=0 && i<dim); return no+ptr[i]; }
    virtual bool isValid();
    virtual void Add(INT_63 start,INT_63 end,INT_63 flag=0x0);
    virtual void Add(INT_63 N,INT_63*id, bool isAppend,INT_63 flag=0x0);
    virtual void ToFile(const std::string& sPath, int flag);
    virtual void Dump(int flag=0x0);
};
bool VerifyFile( const std::string&path, bool isCreate=false );
bool VerifyDirectory( const std::string&path, bool isDelAll=false );



template<class T>
T base_name(T const & path, T const & delims = "/\\")
{
  return path.substr(path.find_last_of(delims) + 1);
}
template<class T>
T remove_extension(T const & filename)
{
  typename T::size_type const p(filename.find_last_of('.'));
  return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
}

template<class T>
const char* CSTR(const T&obj,int flag=0x0)      {
    string suffix, prefix;  
    string info = obj.__repr__(suffix,prefix,flag);
    return info.c_str();
}
template<class T>
const char* CSTR(const shared_ptr<T> obj,int flag=0x0)      {
    return CSTR(*obj,flag);
}

#define HACK_07092024   assert(0)

#define CHILD_0909_WIKIS
#define CHILD_1218_GRAD //


#define CHILD_1012_CACHE true