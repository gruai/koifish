
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
    // std::vector<shared_ptr<GeNeuron>> brothers;
public:
    static shared_ptr<GeNeuron> MakeInstance(Fish *hG_,struct ggml_context *ctx_compute,const string& guid,const string& typ,const JSON& j,int flag=0x0);
    hGensor w = nullptr, b = nullptr;
    bool isBias = true;
    
    NeLayer *hLay = nullptr;
    string sT = "";     //  short-cut infomation
    std::string name = "N";
    GeNeuron() {}    
    GeNeuron(const std::string &key_,SHAPE shape_, Fish *hG_, int flag);
    virtual ~GeNeuron() { ; }

    virtual bool isValid()  {   return true;    }
    virtual void Init(Fish *hG_, int flag=0x0) {    hOrg=hG_;   }
    virtual bool Empty() { return shape.size() == 0; }
    virtual size_t nElem()  { return 0x0; }

    virtual hGensor Forward(struct ggml_context *ctx_compute,hGensor cur,int flag=0x0);
};
typedef shared_ptr<GeNeuron> hNeuron;

// Collection of neurons, only special operation!
struct Ganglia : public GeNeuron    { 
    std::vector<hNeuron> ns;
    Ganglia(struct ggml_context *ctx_compute,const string& guid,std::vector<hNeuron>& ns_,int flag) : ns(ns_)  {}

    // virtual hGensor Forward(struct ggml_context *ctx_compute,hGensor cur,int flag=0x0);
};

//a lookup table instead of matrix*vector(as in SLP)
struct Embed : public GeNeuron    { 
};

struct Relu : public GeNeuron    { 
};

struct Drop : public GeNeuron    { 
};


// single layer perceptron
struct SLP : public GeNeuron    { 
    
    hGensor u = nullptr, s = nullptr, v = nullptr;
    SLP( ) {}
    SLP(Fish *hG_, const std::string &key_, const SHAPE &shape_,  int flag) : GeNeuron(key_,shape_, hG_, flag)    {
        // compression = hOrg->params.compression;
        // Build(key_, shape_, flag);
    }

    // void Init(hGensor w_, hGensor b_, const SHAPE &shape_, int flag = 0x0);
    void Build(const std::string &key_, const SHAPE &shape, int flag);
    hGensor Forward(struct ggml_context *ctx0, hGensor cur, int flag = 0x0);
    
    size_t nElem()  override;  
};
struct LayerNormal : public GeNeuron    {    
    LayerNormal() {}
    LayerNormal(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag)    {
        Build(key_, shape, flag);
    }
    void Build(const std::string &key_, const SHAPE &shape, int flag);
    hGensor Forward(struct ggml_context * ctx0,hGensor cur,int flag);
    size_t nElem()  override;      
};

struct SelfAttention : public GeNeuron
{
    SLP q, k, v;
    SLP proj;
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag);
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