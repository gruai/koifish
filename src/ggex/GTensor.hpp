/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */
#pragma once
#include <cassert>
#include <cstdarg>
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h> 
#include <fstream> 
#include <regex> 
using namespace std;
#include <stdio.h>
#include <string.h>
#include "ggml.h"

#define GG_V12

class GTensor;
class huTensor;
class Fish;
class EDGE_DEVICES;
typedef shared_ptr<GTensor> hGTensor;
typedef std::vector<int> SHAPE;

#ifdef _TENSOR_G_
    #include "../Device/CUDA/cuda_common.h"
#else
    typedef float floatX;
#endif

#define BIT_SET( val,flag ) ((val) |= (flag))	
#define BIT_RESET( val,flag ) ((val) &= (~(flag)) ) 
#define BIT_TEST( val,flag ) (((val)&(flag))==(flag))
#define BIT_IS( val,flag ) (((val)&(flag))!=0)

static char buffer[GGML_MAX_NAME];
//set name of a tensor if its name is "\0" & its grad
int gTN(hGTensor ,const char *format,...);
//clear then set name of a tensor & its grad
int gTN0(hGTensor cur,const char *format,... );

enum INIT_WEIGHT    {
    W_SKIP=0X0,
    FIX_1,
    RANDOM,
    COPY_WIKI,
    COPY_SWARM_HEAD,
    SERIALIZE
};
/**
 *  Edge of Operation(TASK) GRAPH
 */
struct GENSOR_OP{
    string sX;
    int level=-1,ID=-1,dad,c_id;
    int op=-1;
    hGTensor _t;

    GENSOR_OP(hGTensor t,int flag=0x0):_t(t){;}
    static std::shared_ptr<GENSOR_OP> Inst(hGTensor t,int flag=0x0) {
        return std::make_shared<GENSOR_OP>(t,flag);
    }

    string __repr__(string& suffix,string& prefix,int flag=0x0) const {  
        char buf[512]="\0";
        if(dad==-1){
            sprintf(buf+strlen(buf),"ROOT"); 
        }else
            sprintf(buf+strlen(buf),"[%d %d.%d l=%d]",ID,dad,c_id,level); 
        return buf;
    }

    static bool comp(GENSOR_OP& a, GENSOR_OP& b) {
        return a.ID < b.ID;
    }
};
typedef shared_ptr<GENSOR_OP> hGOP;

/**
 * 1.   Support dynamic change shape & type!
 */
class GTensor   {
private:
    struct ggml_tensor  *gg=nullptr;
protected:
    std::shared_ptr<EDGE_DEVICES> hDevice = nullptr;   
    size_t szData=0;
    int recompute=1;
    //  support dynamic change shape&type!
    virtual bool ReShape(SHAPE shape_,typNUMBER tpD_,int flag=0x0);
    virtual hGTensor _Multiply(const hGTensor& other) { assert(0);  return nullptr;    }
public:
    static const int MAX_NAME=64;
    // static int B,T,C;       //shortcut parameter of LLM models
    static hGTensor bt4c,delta,scratch_output,scratch_ff1;
    static void* buff;      //  temporary shared memory 
    float residual_scale=1.0,wnorm=0,gnorm=0;   // some tricks
    float rLARS(float s0,float T_lars,int flag);
    size_t offset = 0x0;
    SHAPE shape; 
    SHAPE x_shape;     //  1.padded for high performance or menory alignment(x_shape<=shape)
    // shape=>x_shape
    virtual void* DataPad(void* src0,int flag=0x0);     

    static typNUMBER tpFloatX;
    typNUMBER  type;
    INIT_WEIGHT tpInit=INIT_WEIGHT::RANDOM;
    enum BIT_FLAG {
        F_INPUT=0x1,F_OUTPUT=0x2,F_PARAM=0x4,F_LOSS=0x8, 
        F_NOALLOC=0x100,F_GPU=0x200,

        F_TOX=0x10000,  F_PADDED=0x20000
    };    
        
    static hGTensor NEW_(struct ggml_tensor*gg,int flag=0x0) {
        hGTensor hT = std::make_shared<GTensor>(gg,flag);
        return hT;
    }
    GTensor()   {}
    GTensor(SHAPE shape_,typNUMBER tpD_,bool isAlloc=true,int flag=0x0);
    GTensor(struct ggml_tensor*gg_,int flag=0x0) : gg(gg_)      {     assert(0);    }
    virtual bool CopyGG(struct ggml_tensor*gg_,int flag=0x0)    {     assert(0);    return false;   }

    virtual ~GTensor();
    virtual bool Alloc(int tpInit=0,int flag=0x0);
    virtual bool InitParam(int tpInit,int flag=0x0)           {     assert(0);    return false;   }
    virtual bool Free() {   return true;    }
    template<typename T> 
    void PrintX(const string& title, int typ, int flag){
        bool isDevice = true;
        PrintTensor<T>(title.c_str(),(T *)data, isDevice,ne[0],ne[1],ne[2],ne[3],flag);
    }
    virtual bool Dump(int type,const string&title="",int flag=0x0)  const;
//operations
    hGTensor operator*(const hGTensor& other) {
        return _Multiply(other);
    }

    hGTensor Relu();
    hGTensor Silu();
    hGTensor Norm(float epsilon,int flag=0x0);

//operations
    virtual bool OverWrite(struct ggml_tensor*gg_,bool isSrc=true,int flag=0x0);
    virtual bool OverWrite(hGTensor,bool isSrc=true,int flag=0x0);      
    virtual hGTensor GetRow(hGTensor, hGTensor token,hGTensor pos,int flag=0x0);
    virtual hGTensor Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward=true,int flag=0x0)   {   assert(0);  return nullptr;}//  Loss
    virtual hGTensor CrossEntropy( const hGTensor b,int flag=0x0 );    

    int64_t ne[GGML_MAX_DIMS]; // number of elements
    //stride in bytes:nb[0] = ggml_type_size(type);nb[i] = nb[i-1] * ne[i-1]
    size_t  nb[GGML_MAX_DIMS]; 
    enum ggml_op op;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags=0x0;

    bool isParam()  {   return BIT_TEST(flags,F_PARAM);  }
    bool isGPU()  {   return BIT_TEST(flags,F_GPU);  }
    
    vector<hGOP> src;
    // virtual void AddSrc(const hGOP t,int type,int flag=0x0);
    virtual void AddSrc(const vector<hGTensor>& ts,int flag=0x0);
    // struct ggml_tensor * view_src=nullptr;
    // size_t               view_offs=0;

    void * data=nullptr;
    void * grad=nullptr; 
    virtual bool SerialGP(void *yD,void *yG,size_t szY,bool isToY,int flag=0x0)   {   assert(0);  return false;   }

    char name[MAX_NAME] = "\0";
    void * extra; // extra things e.g. for ggml-cuda.cu
    //  return ggml_tensor
    struct ggml_tensor*GG();
    struct ggml_context*CTX()   {   return nullptr;  }
    //  byte per element, may be fraction!!!
    virtual double bpe();       
    virtual size_t size(int typ=0)  const;
    virtual int dims()    const         {   
        for (int i = GGML_MAX_DIMS - 1; i >= 1; --i) {
            if (ne[i] > 1) {    return i + 1;   }
        }
        return 1;         
    }
    virtual size_t nByte( )  const      {   return szData;  /*ggml_nbytes(gg);*/       }
    virtual void SetName(const string&name,int flag=0x0)   {   ggml_set_name(gg,name.c_str()); }
    //Returns the value of this tensor(with one element!)
    virtual bool isEmpty()  const                       {   
        // if(size()>0)    {   assert(B>0 && T>0 && C>0); }
        return size()==0;    
    }
    virtual size_t ld(int no){
        assert(no>=0&&no<4);        return nb[no]/bpe();
    }
    virtual bool isSameShape(const hGTensor b) const    {   return szData==b->szData;    }
    virtual void Zero( )       {  Set(0.0);   }  
    virtual void Set(float a,int flag=0x0);
    template<typename T>
    void Set(int i0, int i1, int i2, int i3, T value){
        void * val   = (char *) data + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3];
        switch (type) {
        case typNUMBER::I8:             {
                ((int8_t *)(val))[0] = value;
            } break;
        case typNUMBER::I16:            {
                ((int16_t *)(val))[0] = value;
            } break;
        case typNUMBER::I32:            {
                ((int32_t *)(val))[0] = value;
            } break;
        case typNUMBER::F16:            {
                GGML_ABORT("fatal error");
                // ((ggml_fp16_t *)(val))[0] = GGML_FP32_TO_FP16(value);
            } break;
        case typNUMBER::BF16:            
            {   GGML_ABORT("fatal error");
                // ((ggml_bf16_t *)(val))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case typNUMBER::F32:            {
                ((float *)(val))[0] = value;
            } break;
        default:            {
                GGML_ABORT("fatal error");
            }
        }
    }

    virtual void SetFlag(int64_t flag)   {  
        flags |= (int32_t)flag;     if(gg!=nullptr) gg->flags |= (int32_t)flag;    
    }
    virtual float Get(int i,int flag=0x0)   const;
    virtual float Item()    const{
        assert(ggml_is_scalar(gg));  
        assert(size()==1);  return Get(0);
    }   

    virtual int SerialJSON(const std::string& name, const JSON& val, void* bytes_ptr, size_t bytes_size,int flag=0x0);
    friend class huTensor;
    friend class OPT_Adam;
};

template <typename T> 
T* TO(hGTensor t) { 
    assert(t!=nullptr && t->data!=nullptr);
    BIT_SET(t->flags,GTensor::F_TOX);
    return (T*)(t->data); 
}

inline hGTensor operator+( const hGTensor &a,const hGTensor &b ) 	{		
    auto cur = ggml_add(a->CTX(), a->GG(), b->GG() );    
    return GTensor::NEW_(cur);
}
inline hGTensor operator+=( const hGTensor &a,const hGTensor &b ) 	{		
    auto cur = ggml_add(a->CTX(), a->GG(), b->GG() );    
    return GTensor::NEW_(cur);
}


#ifdef _TENSOR_G_
    typedef hGTensor hGensor;
    inline  struct ggml_tensor* G(hGensor T) {   assert(T!=nullptr);     return T->GG(); }
    inline hGensor NEW_(struct ggml_tensor*gg,int flag=0x0) {
            return GTensor::NEW_(gg,flag);
    };    
    inline void ZERO_(hGensor T)       {  T->Zero();   }
    inline size_t tELEM(hGensor T)    { return T->size();   }
    inline size_t tBYTE(hGensor T)    { return T->nByte();  }
    inline int tDIM(hGensor T)    { return T->dims();  }
    inline float tGET(hGensor T,int i)          { return T->Get(i);  }
    inline void tSET(hGensor T,float a)         { T->Set(a);  }
    inline void tFLAG(hGensor T,int64_t flag)   { T->SetFlag(flag);  }
    double tNormOf(const std::vector<hGTensor>& tensors,int flag);
    double tNormOf(const hGTensor tensor,int flag=0x0);
#else
    int gTN(struct ggml_tensor *cur,const char *format,... );
    int gTN0(struct ggml_tensor *cur,const char *format,... );

    typedef struct ggml_tensor* hGensor;
    inline  struct ggml_tensor* G(hGensor T) {   return (struct ggml_tensor  *)(T); }

    inline hGensor NEW_(hGensor gg)     {  return gg;   }
    inline void ZERO_(hGensor gg)       {  ggml_set_zero(gg);   }
    inline size_t tELEM(hGensor gg)    { return ggml_nelements(gg);  }
    inline size_t tBYTE(hGensor gg)    { return ggml_nbytes(gg);  }
    inline int tDIM(hGensor gg)         { return ggml_n_dims(gg);  }
    inline float tGET(hGensor gg,int i)   { return ggml_get_f32_1d(gg, i);  }
    inline void tSET(hGensor gg,float a)         { ggml_set_f32(gg, a);  }
    inline void tSET_nd(hGensor gg,int i0, int i1, int i2, int i3, int32_t value)         { ggml_set_i32_nd(gg, i0, i1, i2, i3, value);  }
    inline void tFLAG(hGensor gg,int64_t a)      { gg->flags |= (int32_t)a;  }
    hGensor tSCAL(struct ggml_context * _ctx,struct ggml_tensor  * a,float s,int flag=0x0);
    hGensor Permute(struct ggml_context * ctx_,struct ggml_tensor* cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4,bool isCont=true);
    template <typename T>
    inline float T2Float(T* a0)   {        float a=*a0;        return a;      }
#endif
inline floatX *ToX(hGensor t) { 
    assert(t!=nullptr);
    BIT_SET(t->flags,GTensor::F_TOX);
    return (floatX*)(t->data); 
}
inline floatX *ToX0(hGensor t) { 
    if(t==nullptr)  
        return nullptr;
    return ToX(t); 
}
inline floatX *ToG(hGTensor t) { 
    assert(t!=nullptr);
    assert(t!=nullptr && t->grad!=nullptr);
    return (floatX*)(t->grad); 
}
inline floatX *ToG0(hGTensor t) { 
    if(t==nullptr)  
        return nullptr;    
    return ToG(t); 
}
hGensor TENSO(void* ctx0,typNUMBER typ,SHAPE,int flag=0x0,const string&name="" );
hGensor tRAND(hGensor  tensor, struct random_normal_distribution * rnd);

/**
 *  tensor stored in hybrid memory of(CPU/GPU/DISK...)
 */
class huTensor : public GTensor   {
protected:    
    hGTensor _Multiply(const hGTensor& other); 
public:
    huTensor(const string&name_,SHAPE shape,typNUMBER tpD_,bool isParam,int flag=0x0);    
    virtual ~huTensor();

    bool Alloc(int tpInit=0,int flag=0x0)    override;
    bool InitParam(int tpInit,int flag=0x0)    override;
    bool CopyGG(struct ggml_tensor*gg_,int flag=0x0) override;
    bool Free() override;
    void Set(float a,int flag=0x0)  override
    {   ; }
    bool SerialGP(void *yD,void *yG,size_t szY,bool isToY,int flag=0x0)   override;
    bool OverWrite(hGTensor,bool isSrc=true,int flag=0x0) override;
    hGTensor CrossEntropy( const hGTensor b,int flag=0x0 )  override;
    hGTensor GetRow(hGTensor, hGTensor token,hGTensor pos,int flag)   override;
    hGTensor Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward=true,int flag=0x0)   override;
    // bool Dump(int type,int flag=0x0)  const override;
};

struct GENSOR_INFO{
    string sX;
    int level=-1,ID=-1,dad,c_id;
    bool isAdam = true;
    // first momentfor,second moment,past function values of Optimizer
    // hGensor gm=nullptr,gv=nullptr,gpf=nullptr; // 
    float *gm=nullptr,*gv=nullptr,*gpf=nullptr;
    
    GENSOR_INFO(){;}
    GENSOR_INFO(int id_,int l,int d,int c) : ID(id_),level(l),dad(d),c_id(c)  {
        string suffix,prefix;
        sX = __repr__(suffix,prefix);
    }
    string __repr__(string& suffix,string& prefix,int flag=0x0) const {  
        char buf[512]="\0";
        if(dad==-1){
            sprintf(buf+strlen(buf),"ROOT"); 
        }else
            sprintf(buf+strlen(buf),"[%d %d.%d l=%d]",ID,dad,c_id,level); 
        return buf;
    }

    static bool comp(GENSOR_INFO& a, GENSOR_INFO& b) {
        return a.ID < b.ID;
    }
};
struct GENSORS{
    // name_ and gg_tensor
    std::map<std::string, hGensor> nag;
    std::map<hGensor, GENSOR_INFO> infos;
    virtual bool has(hGensor gensor){
        assert(nag.size()==infos.size());
        bool b1 = nag.find(gensor->name) != nag.end(),b2=infos.find(gensor) != infos.end();
        assert(b1==b2);
        return b2;
    }
 
    void Insert(hGensor gensor,const GENSOR_INFO&gi,int flag=0x0){
        auto key = gensor->name;
        // assert(strlen(key)>0);
        assert(nag.find(key) == nag.end());
        nag[key] = gensor;

        assert(infos.find(gensor) == infos.end());
        infos[gensor] = gi;    
        infos[gensor].sX = gensor->name;
    }

    void Insert(const std::map<std::string, hGensor>& src){
        nag.insert(src.begin(), src.end());
    }
    size_t size()   {   return nag.size();  }
    virtual hGensor Get(const string&name, int flag = 0x0);
    virtual void Clear() {   
        nag.clear();    
        infos.clear();
    }

    virtual void TopoOrder()    {
        // sort(gimap.begin(), gimap.end(), comp);
    }
};

