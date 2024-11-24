
/**
 *  Copyright 2023-2024 by Grusoft
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

#include "../ggex/GG_util.hpp"

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
    Fish *hOrg = nullptr;
    COMPRESSIVE_SENSING compression = SKIP;
    SHAPE shape;
    int level=-1,ID=-1,dad,c_id;    //topo info
    vector<double> jvals; 
    virtual hGensor BeforeForward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    virtual hGensor AfterForward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    // std::vector<shared_ptr<GeNeuron>> brothers;
public:
    static shared_ptr<GeNeuron> MakeInstance(Fish *hG_,struct ggml_context *ctx_build,const string& guid,JSON::const_iterator jit,int flag=0x0);
    hGensor w = nullptr, b = nullptr;
    bool isBias = true, isResidual = true;
    
    NeLayer *hLay = nullptr;
    string sT = "";     //  short-cut infomation
    std::string name = "N",type_info="";
    GeNeuron() {}    
    GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag);
    virtual ~GeNeuron() { ; }

    virtual bool isValid();
    virtual void Init(Fish *hG_, int flag=0x0) {    hOrg=hG_;   }
    virtual bool Empty() { return shape.size() == 0; }
    virtual size_t nElem()  { return 0x0; }

    virtual hGensor Forward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
    virtual bool Build(int flag)   {assert(0);     return false;}
    // Init & build with more option
    virtual void BuildX(const std::string &key_, const SHAPE &shape, Fish *hG_, int flag);

friend class Fish;
};
typedef shared_ptr<GeNeuron> hNeuron;

// Collection of neurons, only special operation!
struct Ganglia : public GeNeuron    { 
    std::vector<hNeuron> ns;
    Ganglia(Fish *hG_,const string& guid,std::vector<hNeuron>& ns_,int flag);
    bool isValid()  override{   return ns.size()>0; }

    // virtual hGensor Forward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0);
};

//a lookup table instead of matrix*vector(as in SLP)
struct Embed : public GeNeuron    { 
    bool isAddPos = false;

    Embed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Forward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
};

struct Relu : public GeNeuron    { 
    Relu()  {;}
    Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Forward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
    bool Build(int flag)   override;
    bool isValid()  override    {   return true;    }
};

struct Drop : public GeNeuron    { 
    Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag);
    virtual hGensor Forward(struct ggml_context *ctx_build,hGensor cur,int flag=0x0)  override;
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
    hGensor Forward(struct ggml_context *ctx0, hGensor cur, int flag = 0x0)     override;
    // only for deprecated function"UpdateGensor"
    hGensor UpdateGensor(int flag=0x0);
    size_t nElem()  override;  
};
struct LayerNormal : public GeNeuron    {  
    bool isAffineTrans = true;       // Learnable affine transform parameters 
    LayerNormal() {}
    //Deprecated
    LayerNormal(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    size_t nElem()  override;      
};

struct SelfAttention : public GeNeuron  {
    bool use_cache = false;
    bool isLast = false,isAttOnBC = false;
    float f_max_alibi_bias;
    int n_batch,n_ctx,n_embd,n_embd_head,n_head,n_head_kv,n_embd_gqa,n_tokens;
    hGensor KQ_pos=nullptr,KQ_mask=nullptr;
    LayerNormal norm;
    SLP Q, K, V;
    SLP proj_cat;       //   concatenate the heads and combine them with a final weight matrix.
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
};
/*
    SelfAttention with Rotary Position Embedding
*/
struct QKV_rope : public SelfAttention  {    
protected:
    int n_rot=-1;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    hGensor W_rope(struct ggml_context *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,const string&shortcut,int flag=0x0);
    hGensor vXkq(struct ggml_context *ctx, hGensor v,hGensor kq);
public:
    QKV_rope() {}
    QKV_rope(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;    

    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
};

struct FFN : public GeNeuron  {
    LayerNormal norm;
    SLP up,down;
    Relu relu;
    FFN() {}
    FFN(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
};

struct MOE : public GeNeuron  {
};

struct FFN_MOE : public FFN{
    MOE Moe;
};

struct OutCLS : public GeNeuron  {
    LayerNormal norm;
    SLP proj;
    int nCls = 0;
    OutCLS() {}
    OutCLS(Fish *ctx, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag)   override;
    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag)    override;
    bool isValid()  override    {   return true;    }
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