
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT 
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

class Fish;
struct NeLayer;
struct LayerNormal;
struct NP_
{ // Paramer of GeNeuron
    std::string type, title;
    SHAPE shape;
    NP_(const std::string &t, const std::string &n, SHAPE s) : type(t), title(n), shape(s) {}
};

/**
 * Each Neuron has
 * 1. inp,out,w
 * 2. b     (if isBias is true)
 * 3. delta (optional)
 * 4. activation
 * 
 * Fusion
 * 1. gelu_fusion
 */

class GeNeuron  {
    // GeNeuron(const GeNeuron&)=default;       for model.layers.resize(n_layer)@llama.cpp
    GeNeuron &operator=(const GeNeuron &) = default;

protected: 
    int B,T,C;  //n_batch,n_ctx,n_embd,
    int n_embd_head,n_head;
    int gelu_fusion = 0;
    Fish *hFish = nullptr;
    COMPRESSIVE_SENSING compression = SKIP;
    SHAPE shape;
    int level=-1,ID=-1,dad,c_id;    //topo info
    vector<double> jvals;   
    string _repr_1( string& suffix,string& prefix,string info,int flag=0x0);

    // std::vector<shared_ptr<GeNeuron>> brothers;
public:
    enum BIT_FLAG {
        F_BIAS=0x10000, F_DELTA=0x20000
    }; 

    static shared_ptr<GeNeuron> MakeInstance(Fish *hG_,void *ctx_build,const string& guid,JSON::const_iterator jit,int flag=0x0);

    hGensor w = nullptr, b = nullptr, out = nullptr;
    hGensor inp = nullptr;          //  may change! maybe nullptr!
    hGensor delta = nullptr,tmpDelta = nullptr;        //  error tensor for each layer(may share memory!)
    bool isBias = true, isResidual = true;
    
    NeLayer *hLay = nullptr;
    string sT = "";     //  short-cut infomation
    std::string name = "N",type_info="";
    GeNeuron() {}    
    GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag);
    virtual ~GeNeuron() { ; }
    virtual std::vector<hGensor> Gensors(bool isNoRef=true) {
        std::vector<hGensor> gensors={  w,b };     
        if(!isNoRef)   {
            std::vector<hGensor> refs={  inp,out,delta };     
            gensors.insert(gensors.end(), refs.begin(), refs.end());
        }
        return gensors;
    }
    virtual bool isValid();
    virtual bool isOnlyInfer();
    virtual bool isForward();
    // 1. Set C/T/H/...
    virtual void Init(Fish *hG_, int flag=0x0);
    virtual bool Empty()    const { return shape.size() == 0; }
    virtual size_t nElem()  { return 0x0; }
    //  无知觉明
    virtual hGensor Ming(void *ctx_build,hGensor cur,int flag=0x0);
    virtual hGensor Backward(void *user_ctx,hGensor cur,int flag=0x0);
    virtual hGensor Forward2(void *ctx_build,hGensor,hGensor,int flag=0x0)   {   assert(0);      return nullptr;     }
    virtual hGensor BeforeForward(void *ctx_build,hGensor cur,int flag=0x0);
    virtual hGensor AfterForward(void *ctx_build,hGensor cur,int flag=0x0);
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
    hGensor Ming(void *ctx_build,hGensor cur,int flag=0x0)    override   {   return cur; } 
};

class ROPE : public GeNeuron    { 
protected:
    int n_rot=0,n_ctx_orig=0;
    hGensor KQ_pos;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
public:    
    ROPE() {}
    ROPE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Ming(void * ctx0,hGensor cur,int flag=0x0)    override;
    int FUSE_cuda(hGTensor QKV,bool isFX=true,int flag=0x0);
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

struct Relu : public GeNeuron    { 
    Relu()  {;}
    Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Ming(void *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    bool isValid()  override    {   return true;    }
};

struct Drop : public GeNeuron    { 
    Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Ming(void *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    bool isValid()  override    {   return true;    }
};


// single layer perceptron
struct SLP : public GeNeuron    {
    hGensor u = nullptr, s = nullptr, v = nullptr;
    SLP( ) {}
    SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);

    bool Empty() const  override   { return w==nullptr; }
    bool Build(int flag)   override;
    hGensor Ming(void *ctx0, hGensor cur, int flag = 0x0)     override;
    // only for deprecated function"UpdateGensor"
    hGensor UpdateGensor(int flag=0x0);
    size_t nElem()  override;  
    // hGTensor operator<<(hGTensor a);
    /*  Forward or remate in Back
        1.  rhs = SLP(lhs)  or rhs = W*lhs+b
        2.  gelu = GELU(rhs)
    */
    int Forw(hGTensor rhs,hGTensor lhs,hGTensor gelu=nullptr,int flag=0x0);
    int Back(hGTensor delta,hGTensor inp,hGTensor deltaIn,hGTensor gelu=nullptr,float* dbias_buffer=nullptr,int flag=0x0);
    int FUSE_cuda_block(hGTensor rhs,hGTensor lhs,hGTensor gelu=nullptr,bool isForw=true,int flag=0x0);
};

struct LayerNormal : public GeNeuron    {  
    bool isAffineTrans = true;       // Learnable affine transform parameters 
    bool isRMS = true;               // Root Mean Square Layer Normalization
    //  always float
    hGensor mean=nullptr, rstd=nullptr;
    hGensor out=nullptr;
    LayerNormal() {}    
    LayerNormal(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    hGTensor FUSE_cuda(hGTensor inpL,float* scratch=nullptr,hGTensor deltaIn=nullptr,int flag=0x0);
    size_t nElem()  override;    
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    // hGTensor operator>>(hGTensor & a){  return a;   }
    // hGTensor operator<<(hGTensor a);
};

struct MOE : public GeNeuron  {
    bool isSiLU = false;
    
    MOE() {}
    MOE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward2(void * ctx0,hGensor cur,hGensor ,int flag=0x0)    override;
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
    bool isPreNormal = false;      //  Pre /Post Normalization
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
    bool remater_qkv = false;
    bool isAttOnBC = false;     //  // Nearly same. If true,attenion on all tokens, memory would explode!
    bool isRope = true;
    hGensor attn_k=nullptr,attn_q=nullptr;
    // int n_rot=-1;
    // hGensor W_rope(void *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,const string&shortcut,int flag=0x0);
    hGensor MyAttention(void * ctx_,hGensor inpL,int flag);
    hGensor vXattn(void *ctx, hGensor v,hGensor attn,int flag);
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
public:
    bool use_cache = false;
    bool isLast = false;
    float f_max_alibi_bias;
    int n_head_kv,n_embd_gqa,n_tokens,C_qkv=-1;
    hGensor bqkv=nullptr;       //  biases for qkv (qwen)	
    hGensor KQ_pos=nullptr,KQ_mask=nullptr;
    LayerNormal norm,*fuseNorm=nullptr;
#ifdef _TENSOR_G_
    hGensor attn=nullptr,trans=nullptr;
    hGensor deltaCat=nullptr;       //floatX* dl_btc=nullptr;
#endif
    SLP Q, K, V;
    ROPE rope;
    MOE moe;
    SLP proj_cat;       //   concatenate the heads and combine them with a final weight matrix.
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;

    hGTensor FUSE_cuda(hGTensor inpL,hGTensor residual,hGTensor deltaIn,float* scratchF,int flag);
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
    GatedAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};

struct cuAttention : public SelfAttention  {    
protected:
    int attn_mode = 1;
    SLP down,upU,upV;
public:
    cuAttention() {}
    cuAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
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
    BROWN_attn(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    

    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
};


// Variational Encoder/Decoer
class VarCoder : public GeNeuron   {
protected:
    int nTop=-1,nBottom=-1;
    bool isResi = false;
    hGensor resi = nullptr;
    int tpNorm=-2; 
    SLP up,down,gate;
    Relu relu;    
    
    bool Build(int flag)   override;
public:
    LayerNormal norm;
    // hGensor encode=nullptr,decode=nullptr,norm=nullptr;
    VarCoder()  {}
    VarCoder(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    VarCoder(Fish *hG_,std::vector<int>&dims,int level,bool isR = false,bool isSym=true,int tpN=2,int flag=0x0);
    virtual hGensor ENC(const hGensor x0);
    virtual hGensor DEC(hGensor x);
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
    std::vector<hGensor> Gensors(bool isNoRef=true) override  {
        std::vector<hGensor> gensors;
        GPLUS(gensors,up.Gensors());    
        GPLUS(gensors,down.Gensors());       GPLUS(gensors,gate.Gensors());
        return gensors;
    }

friend class TokenEmbed;
friend class MAEC;
};
typedef shared_ptr<VarCoder> hVarCoder;

struct FFN : public VarCoder  {
    LayerNormal *fuseNorm=nullptr;;
    
    // hGensor pre_gelu = nullptr;
    int latent;
#ifdef _TENSOR_G_    
    floatX *input_1=nullptr,*scratchX=nullptr;//,*residual=nullptr;
    // SelfAttention *lastQKV=nullptr;
    bool remater_ffn = false;
#endif

    FFN() {}
    FFN(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;

    hGTensor FUSE_cuda(hGTensor inpL,floatX *scratch,int flag);
};

//  multi-level auto encoder
struct MAEC: public GeNeuron    { 
    int nIn=-1,nOut=-1;
    vector<hVarCoder> codes; 
    LayerNormal normE,normD;
    bool reserve_x = false;
    vector<hGensor> resi_x;

    MAEC(Fish *hG_, const std::string &key_, int flag);
    virtual hGensor ENC(hGensor x,int flag=0x0);
    virtual hGensor DEC(hGensor x,bool isForw,int flag=0x0);
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    bool Empty( )   const override    {   return codes.size()>0;   }
};
typedef shared_ptr<MAEC> hMAEC;

/*
    Each token is embed to latent vector
*/
struct TokenEmbed : public GeNeuron    {      
    hBATCH hBatch = nullptr;

    int *workload_indices=nullptr,nVocab=-1,latent;
    int4 *bucket_info=nullptr;
    bool isAddPos = false;
    int padded_nCls=-1; //*hostInput=nullptr,inp_pos=0;          

    virtual bool SetMAEC(hMAEC maec,int flag=0x0);
    hMAEC maec = nullptr;

    TokenEmbed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual ~TokenEmbed();
    virtual bool InitBucket(size_t num_c_groups,int flag=0x0);
    // virtual int InitMAC(int flag=0x0); 
    virtual hGensor Ming(void *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;
    virtual int FUSE_cuda(hGTensor hIn,floatX *scratch,LayerNormal*neuron_x,unsigned int flag);
};
    
struct FFN_MOE : public FFN{
    MOE Moe;
};

struct OutCLS : public GeNeuron  {    
    hMAEC maec;
    LayerNormal norm;
    SLP proj;
    hGTensor target=nullptr,preLogits=nullptr;
    hSampLoader hLoader=nullptr;
    int nCls = 0,dB = 1,nzLoss = 0,latent=0;
    int padded_nCls; // padded to e.g. %128==0, 
    float mean_loss=0, rLoss=1.0,*hostLoss=nullptr;
    OutCLS() {}
    OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual ~OutCLS()   {
        if(hostLoss!=nullptr)
            delete[] hostLoss;
    }
    bool Build(int flag)   override;
    hGensor Ming(void * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
    string __repr__( string& suffix,string& prefix,int flag=0x0)    override;    
    // Backward: return lnf->out;       Forward: return preLogits or loss?
    hGTensor FUSE_cuda(hGTensor inpL,hGTensor token_embed,int flag);
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
hGTensor  operator>>(hGTensor t, const GeNeuron* neuron);

