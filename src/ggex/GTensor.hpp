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

class GTensor;
class cuTensor;
class Fish;
class EDGE_DEVICES;
typedef shared_ptr<GTensor> hGTensor;
typedef std::vector<int> SHAPE;

#ifdef _TENSOR_CUD_
    #include "../kGPT/llmc/cuda_common.h"
    #include "../Device/cutils.cuh"
    // #if defined(ENABLE_FP32)
    //     typedef float floatX;
    //     #define PRECISION_MODE PRECISION_FP32
    //     // use fp16 (note: this may require gradient scaler, currently not implemented!)
    // #elif defined(ENABLE_FP16)
    //     typedef half floatX;
    //     #define PRECISION_MODE PRECISION_FP16
    // #else // Default to bfloat16
    //     typedef __nv_bfloat16 floatX;
    //     #define PRECISION_MODE PRECISION_BF16
    // #endif
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

class GTensor   {
private:
    struct ggml_context *_ctx=nullptr;
    struct ggml_tensor  *gg=nullptr;
protected:
    std::shared_ptr<EDGE_DEVICES> hDevice = nullptr;
    struct ggml_backend_buffer * buffer;
    size_t szData=0;
    int recompute=1;
    virtual hGTensor _Multiply(const hGTensor& other) { assert(0);  return nullptr;    }
public:
    static int B,T,C;       //shortcut parameter of LLM models
    static hGTensor scratch_bt4c,scratch_btc,scratch_output;
    float residual_scale=1.0;   // some tricks
    size_t offset = 0x0;
    SHAPE shape; 
    SHAPE x_shape;     //  1.padded_shape  for high performance or menory alignment
    typedef enum ggml_type tpDATA;
    static tpDATA tpFloatX;
    tpDATA  type;
    int tpInit=2;
    enum BIT_FLAG {
        F_INPUT=0x1,F_OUTPUT=0x2,F_PARAM=0x4,F_LOSS=0x8, 
        F_NOALLOC=0X100,
        F_TOX=0x10000,
    };    
        
    static hGTensor NEW_(struct ggml_tensor*gg,int flag=0x0) {
        hGTensor hT = std::make_shared<GTensor>(gg,flag);
        return hT;
    }
    GTensor()   {}
    GTensor(SHAPE shape_,tpDATA tpD_,bool isAlloc=true,int flag=0x0);
    GTensor(struct ggml_tensor*gg_,int flag=0x0) : gg(gg_)      {     assert(0);    }
    virtual bool CopyGG(struct ggml_tensor*gg_,int flag=0x0)    {     assert(0);    return false;   }

    virtual ~GTensor();
    virtual bool Alloc(int tpInit=0,int flag=0x0);
    virtual bool InitParam(int tpInit,int flag=0x0)           {     assert(0);    return false;   }
    virtual bool Free() {   return true;    }

    virtual bool Dump(int type,int flag=0x0)  const;
//operations
    hGTensor operator*(const hGTensor& other) {
        return _Multiply(other);
    }

    hGTensor Relu() {  auto cur=ggml_relu(_ctx, gg);  return NEW_(cur);  }
    hGTensor Silu() {  auto cur=ggml_silu(_ctx, gg);  return NEW_(cur);  }
    hGTensor Norm(float epsilon,int flag=0x0) {  auto cur=ggml_silu(_ctx, gg);  return NEW_(cur);  }

//operations
    virtual bool OverWrite(struct ggml_tensor*gg_,bool isSrc=true,int flag=0x0);
    virtual bool OverWrite(hGTensor,bool isSrc=true,int flag=0x0);
      
    virtual hGTensor GetRow(hGTensor, hGTensor token,hGTensor pos,int flag=0x0);
    virtual hGTensor Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward=true,int flag=0x0)   {   assert(0);  return nullptr;}
    // virtual hGTensor QKV(hGTensor hOut,hGTensor hQKV,hGTensor hATTN,hGTensor w,hGTensor b,int NH,hGTensor proj_w,hGTensor proj_b,int flag)          {   assert(0);  return nullptr;}
    // virtual hGTensor FFN(hGTensor hOut,hGTensor hUp,hGTensor wUP,hGTensor bUp,hGTensor fch,hGTensor wDown,hGTensor bDown,int gelu_fusion,int flag)  {   assert(0);  return nullptr;}
    // virtual hGTensor ResiNormal(hGTensor hOut,hGTensor hNormed,hGTensor _mean,hGTensor _rstd,hGTensor hInp1,hGTensor hInp2,hGTensor w,hGTensor b,int flag)                        {   assert(0);  return nullptr;}                       
    // virtual float FusedLoss(float *hostLoss,float dLoss,hGTensor hLoss,hGTensor hTarget,hGTensor tX, hGTensor w,int V,bool isForward, int flag)  {   assert(0);  return 0;}
//  Loss
    virtual hGTensor CrossEntropy( const hGTensor b,int flag=0x0 )   	{
        auto cur = ggml_cross_entropy_loss(_ctx,gg, b->GG() );   
        // ggml_cross_entropy_loss_1(_ctx, cur, target_probs); 
        return GTensor::NEW_(cur);
    }
    

    int64_t ne[GGML_MAX_DIMS]; // number of elements
    size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                // nb[0] = ggml_type_size(type)
                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                // nb[i] = nb[i-1] * ne[i-1]
    // compute data
    enum ggml_op op;
    // op params - allocated as int32_t for alignment
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags=0x0;
    bool isParam()  {   return BIT_TEST(flags,F_PARAM);  }
    
    // struct ggml_tensor * grad;
    // struct ggml_tensor * src[GGML_MAX_SRC];
    vector<hGTensor> src;
    virtual void AddSrc(const hGTensor t)           {   assert(t!=nullptr); src.push_back(t);   }
    virtual void AddSrc(const vector<hGTensor>& ts,int flag=0x0);
    // source tensor and offset for views
    struct ggml_tensor * view_src=nullptr;
    size_t               view_offs=0;
    void * data=nullptr;
    void * grad=nullptr; 
    virtual bool SerialGP(void *param,void *g,bool isSerial,int flag=0x0)   {   assert(0);  return false;   }

    char name[GGML_MAX_NAME];
    void * extra; // extra things e.g. for ggml-cuda.cu
    //  return ggml_tensor
    struct ggml_tensor*GG();
    struct ggml_context*CTX()   {   return _ctx;  }
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
    virtual bool isSameShape(const hGTensor b) const    {   return szData==b->szData;    }
    virtual void Zero( )       {  Set(0.0);   }  
    virtual void Set(float a,int flag=0x0);
    template<typename T>
    void Set(int i0, int i1, int i2, int i3, T value){
        void * val   = (char *) data + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3];
        switch (type) {
        case GGML_TYPE_I8:             {
                ((int8_t *)(val))[0] = value;
            } break;
        case GGML_TYPE_I16:            {
                ((int16_t *)(val))[0] = value;
            } break;
        case GGML_TYPE_I32:            {
                ((int32_t *)(val))[0] = value;
            } break;
        case GGML_TYPE_F16:            {
                GGML_ABORT("fatal error");
                // ((ggml_fp16_t *)(val))[0] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:            
            {   GGML_ABORT("fatal error");
                // ((ggml_bf16_t *)(val))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:            {
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
    virtual float Get(int i,int flag=0x0)
    {   return ggml_get_f32_1d(gg, i); }
    virtual float Item(){
        assert(ggml_is_scalar(gg));  
        assert(size()==1);  return ggml_get_f32_1d(gg, 0);
    }   
    friend class cuTensor;
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


#ifdef _TENSOR_CUD_
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
    BIT_SET(t->flags,GTensor::F_TOX);
    return (floatX*)(t->data); 
}
inline floatX *ToG(hGTensor t) { 
    assert(t!=nullptr && t->grad!=nullptr);
    return (floatX*)(t->grad); 
}
hGensor TENSO(void* ctx0,int typ,SHAPE,int flag=0x0,const string&name="" );
hGensor tRAND(hGensor  tensor, struct random_normal_distribution * rnd);

class cuTensor : public GTensor   {
protected:    
    // PrecisionMode precision;
    hGTensor _Multiply(const hGTensor& other); 
public:
    cuTensor(const string&name_,SHAPE shape,tpDATA tpD_,bool isParam,int flag=0x0);    
    virtual ~cuTensor();

    bool Alloc(int tpInit=0,int flag=0x0)    override;
    bool InitParam(int tpInit,int flag=0x0)    override;
    bool CopyGG(struct ggml_tensor*gg_,int flag=0x0) override;
    bool Free() override;
    void Set(float a,int flag=0x0)  override
    {   ; }
    bool SerialGP(void *param,void *g,bool isSerial,int flag=0x0)   override;
    bool OverWrite(hGTensor,bool isSrc=true,int flag=0x0) override;
    hGTensor CrossEntropy( const hGTensor b,int flag=0x0 )  override;
    hGTensor GetRow(hGTensor, hGTensor token,hGTensor pos,int flag)   override;
    hGTensor Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward=true,int flag=0x0)   override;
    // hGTensor QKV(hGTensor hOut,hGTensor hQKV,hGTensor hATTN,hGTensor w,hGTensor b,int NH,hGTensor proj_w,hGTensor proj_b,int flag)    override;
    // hGTensor FFN(hGTensor hOut,hGTensor hUp,hGTensor wUP,hGTensor bUp,hGTensor fch,hGTensor wDown,hGTensor bDown,int gelu_fusion,int flag)  override;
    // hGTensor ResiNormal(hGTensor hOut,hGTensor hNormed,hGTensor _mean,hGTensor _rstd,hGTensor hInp1,hGTensor hInp2,hGTensor w,hGTensor b,int flag) override;                 
    // float FusedLoss(float dloss,hGTensor hLoss,hGTensor hTarget,hGTensor tX, hGTensor w,int V,bool isForward, int flag)  override;
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
    virtual hGensor Get(const string&name, int flag = 0x0)    {        
        if(flag==0x100){    //  .weight=>.w
            for(auto ng:nag){
                if(strstr(name.c_str(),ng.first.c_str())!= NULL){
                    return ng.second;
                }
            }
            return nullptr;
        }else{
            if(nag.find(name) == nag.end()){
                assert(0);  return nullptr;
            }
            return nag[name];
        }
        
    } 
    virtual void Clear() {   
        nag.clear();    
        infos.clear();
    }

    virtual void TopoOrder()    {
        // sort(gimap.begin(), gimap.end(), comp);
    }
};

int cuLiteTest(size_t B,size_t T,size_t C,int stage=0,int flag=0);
int FUSE_ResiNormal(hGTensor hOut,hGTensor hInp1,hGTensor hInp2,hGTensor hNormed,hGTensor N_mean,hGTensor N_rstd,hGTensor w,hGTensor b,int flag);
int FUSE_QKV(hGTensor hOut,hGTensor hIn,hGTensor hQKV,hGTensor hATTN,hGTensor w,hGTensor b,int NH,hGTensor proj_w,hGTensor proj_b,int flag);
int FUSE_FFN(hGTensor hOut,hGTensor hIn,hGTensor hLatent,hGTensor wUp,hGTensor bUp,hGTensor hGelu,hGTensor wDown,hGTensor bDown,int gelu_fusion,int flag);

