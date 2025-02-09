
/**
 *  Copyright 2023-2025 by Grusoft 
 *
 *  \brief Collection of neurons
 *  \author Yingshi Chen
 */

#pragma once

#include <cassert>
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
#include <regex>
#include <stack>
using namespace std;

#include "../ggex/GTensor.hpp"
#include "DataLoader.hpp"

static const string sWeight=".weight",sBias=".bias";        //".w"
static const string sNorm="_norm";            //".norm"

class Fish;
struct NeLayer;

struct NP_
{ // Paramer of GeNeuron
    std::string type, title;
    SHAPE shape;
    NP_(const std::string &t, const std::string &n, SHAPE s) : type(t), title(n), shape(s) {}
};

class GeNeuron  {
    // GeNeuron(const GeNeuron&)=default;       for model.layers.resize(n_layer)@llama.cpp
    GeNeuron &operator=(const GeNeuron &) = default;

protected: 
    int n_batch,n_ctx,n_embd,n_embd_head,n_head;
    Fish *hFish = nullptr;
    COMPRESSIVE_SENSING compression = SKIP;
    SHAPE shape;
    int level=-1,ID=-1,dad,c_id;    //topo info
    vector<double> jvals; 


    string _repr_1( string& suffix,string& prefix,string info,int flag=0x0);

    // std::vector<shared_ptr<GeNeuron>> brothers;
public:
    enum BIT_FLAG {
        F_BIAS=0x10000, 
    }; 

    static shared_ptr<GeNeuron> MakeInstance(Fish *hG_,struct ggml_context *ctx_build,const string& guid,JSON::const_iterator jit,int flag=0x0);
    hGensor w = nullptr, b = nullptr, out = nullptr;
    bool isBias = true, isResidual = true;
    
    NeLayer *hLay = nullptr;
    string sT = "";     //  short-cut infomation
    std::string name = "N",type_info="";
    GeNeuron() {}    
    GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag);
    virtual ~GeNeuron() { ; }

    virtual bool isValid();
    virtual bool isForward();
    // 1. Set C/T/H/...
    virtual void Init(Fish *hG_, int flag=0x0);
    virtual bool Empty() { return shape.size() == 0; }
    virtual size_t nElem()  { return 0x0; }

    virtual hGensor Interact(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    virtual hGensor Backward(void *user_ctx,hGensor cur,int flag=0x0);
    virtual hGensor Forward2(struct ggml_context *ctx_build,hGensor,hGensor,int flag=0x0)   {   assert(0);      return nullptr;     }
    virtual hGensor BeforeForward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    virtual hGensor AfterForward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    virtual bool Build(int flag)   {assert(0);     return false;}
    // Init & build with more option
    virtual void BuildX(const std::string &key_, const SHAPE &shape, Fish *hG_, int flag);
    virtual string __repr__( string& suffix,string& prefix,int flag=0x0);
    virtual bool isGang()   {   return false;    }       

friend class Fish;
friend class NLP_AutoRegressive;
};
typedef shared_ptr<GeNeuron> hNeuron;

// Collection of neurons, only special operation!
struct Ganglia : public GeNeuron    { 
    std::vector<hNeuron> ns;
    Ganglia(Fish *hG_,const string& guid,std::vector<hNeuron>& ns_,int flag);
    bool isValid()  override    {   return ns.size()>0; }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    bool isGang()   override    {   return true;    }  
    hGensor Interact(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)    override   {   return cur; } 
};

//a lookup table instead of matrix*vector(as in SLP)
struct Embed : public GeNeuron    { 
    bool isAddPos = false;
    int padded_nCls=-1;        //for cuda version
    Embed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Interact(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

class ROPE : public GeNeuron    { 
protected:
    int n_rot=0;
    hGensor KQ_pos;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
public:    
    ROPE() {}
    ROPE(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag=0x0)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

struct Relu : public GeNeuron    { 
    Relu()  {;}
    Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Interact(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    bool isValid()  override    {   return true;    }
};

struct Drop : public GeNeuron    { 
    Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Interact(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    bool isValid()  override    {   return true;    }
};


// single layer perceptron
struct SLP : public GeNeuron    {
    hGensor u = nullptr, s = nullptr, v = nullptr;
    SLP( ) {}
    SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);

    bool Empty() override   { return w==nullptr; }
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context *ctx0, hGensor cur, int flag = 0x0)     override;
    // only for deprecated function"UpdateGensor"
    hGensor UpdateGensor(int flag=0x0);
    size_t nElem()  override;  
    hGTensor operator<<(hGTensor a);
};
struct LayerNormal : public GeNeuron    {  
    bool isAffineTrans = true;       // Learnable affine transform parameters 
    //  always float
    hGensor mean=nullptr, rstd=nullptr;
    hGensor out=nullptr;
    LayerNormal() {}
    //Deprecated
    LayerNormal(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    size_t nElem()  override;    
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    // hGTensor operator>>(hGTensor & a){  return a;   }
    hGTensor operator<<(hGTensor a);
};
struct MOE : public GeNeuron  {
    bool isSiLU = false;
    
    MOE() {}
    MOE(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward2(struct ggml_context * ctx0,hGensor cur,hGensor ,int flag=0x0)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

//  reversible residual network (RevNet)
class RevNet : public GeNeuron{

};

class SelfAttention : public GeNeuron  {
protected:
    int tpNormal=1,n_ff=0;
    bool isLinear = false;

    //markov transition matrix from KQ
    enum TRANSITION_MODE{
        SOFT_MAX=0,
        SIGMOID=1,  
        RELU2=2,
        RELU_=3,
        LINEAR=4,
    };

    // 1 linear(No softmax!) 2 sim(q,k)>=0
    TRANSITION_MODE tpTrans=SOFT_MAX;
    enum LINEAR_MODE{   
        L_OFF,
        ELU=0x1,    NORM_1=0x2,
    };
    LINEAR_MODE tpLinear=L_OFF;
    enum SPARSE_MODE{   
        S_OFF,
        POOLING=0x1,
        LINFORMER=0x2,
    };

    bool isAttOnBC = false;     //  // Nearly same. If true,attenion on all tokens, memory would explode!
    bool isRope = true;
    hGensor attn_k=nullptr,attn_q=nullptr;
    // int n_rot=-1;
    // hGensor W_rope(struct ggml_context *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,const string&shortcut,int flag=0x0);
    hGensor MyAttention(struct ggml_context * ctx_,hGensor inpL,int flag);
    hGensor vXattn(struct ggml_context *ctx, hGensor v,hGensor attn,int flag);
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
public:
    bool use_cache = false;
    bool isLast = false;
    float f_max_alibi_bias;
    int n_head_kv,n_embd_gqa,n_tokens;
    hGensor KQ_pos=nullptr,KQ_mask=nullptr;
    LayerNormal norm;
#ifdef _TENSOR_CUD_
    hGensor attn=nullptr,trans=nullptr;
#endif
    SLP Q, K, V;
    ROPE rope;
    // union{
    //     SLP brown;
    // };
    MOE moe;
    SLP proj_cat;       //   concatenate the heads and combine them with a final weight matrix.
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;

    int FUSE_cuda(hGTensor inpL,int flag);
};

/*
    Gated SelfAttention
*/    
struct GatedAttention : public SelfAttention  {    
protected:
    int attn_mode = 1;
    SLP down,upU,upV;
public:
    GatedAttention() {}
    GatedAttention(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

struct cuAttention : public SelfAttention  {    
protected:
    int attn_mode = 1;
    SLP down,upU,upV;
public:
    cuAttention() {}
    cuAttention(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

// cuda version by K
struct BROWN_attn : public SelfAttention  {    
protected:
    bool Transfer_1 = false;
    int n_rot=-1;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    

public:    
    BROWN_attn() {}
    BROWN_attn(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    

    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

struct FFN : public GeNeuron  {
    LayerNormal norm;
    SLP up,down,gate;
    Relu relu;
    // hGensor pre_gelu = nullptr;
    int gelu_fusion = 0;
    FFN() {}
    FFN(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;

    int FUSE_cuda(hGTensor inpL,int flag);
};



struct FFN_MOE : public FFN{
    MOE Moe;
};

struct OutCLS : public GeNeuron  {
    bool isSymProj = true;
    LayerNormal norm;
    SLP proj;
    hGTensor target=nullptr,preLogits=nullptr;
    hSampLoader hLoader=nullptr;
    int nCls = 0,dB = 1,nzLoss = 0;
    int padded_nCls; // padded to e.g. %128==0, 
    float mean_loss=0, rLoss=1.0,*hostLoss=nullptr;
    OutCLS() {}
    OutCLS(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual ~OutCLS()   {
        if(hostLoss!=nullptr)
            delete[] hostLoss;
    }
    bool Build(int flag)   override;
    hGensor Interact(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;    
    int FUSE_cuda(hGTensor inpL,hGTensor token_embed,int flag);
};

struct NeLayer
{ // Neural Layer with many neurons
    int id = -1;
    bool isLast = false;
    std::string name;
    std::vector<hNeuron> neurons;

    void Add(hNeuron hN, int flag = 0x0)
    {
        neurons.push_back(hN);
        hN->hLay = this; //???  
    }
    NeLayer(int id_) : name("NeLayer"),id(id_) {}
    NeLayer(const std::string &n_, int flag = 0x0) : name(n_) {}
    virtual ~NeLayer() {}
    // NeLayer* N_(const std::string& t,const std::string& n,SHAPE s)   {
    //     return this;
    // }
    virtual string __repr__( string& suffix,string& prefix,int flag=0x0)   {    return "";  }
};
typedef shared_ptr<NeLayer> hLayer;

hGTensor  operator>>(hGTensor t, const LayerNormal& norm);
hGTensor  operator>>(hGTensor t, const SLP& slp);
hGTensor  operator>>(hGTensor t, const Relu& relu);

