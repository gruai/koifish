/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <cassert>
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h> 
using namespace std;
#include "../ggex/GG_util.hpp"
// #include "../lenda/util/GST_util.hpp"

class Fish;
class TGraph;   
class Optimizer;
typedef shared_ptr<TGraph> hTGraph;

class TGraph : public std::enable_shared_from_this<TGraph> {
    TGraph(const TGraph&);
	TGraph& operator=(const TGraph&);

    
protected:    
    enum ORDER {
        LEFT_TO_RIGHT = 0,        RIGHT_TO_LEFT,        COUNT
    };
    Fish *hFish = nullptr;
    // Optimizer *hOPT = nullptr;
    string name;
    bool isOnlySymbol = true, isBackward =false;
    double tX=0,tCompute=0.0,tPlan=0.0;
    std::vector<uint8_t> work_buffer;
    //lite hash_set
    std::set<hGensor> gset;

    int size=0,nForwN=0,nForwL=0,nNeedGrad=0,nInput=0; // n_nodes=0,n_leafs=0;+
    static int curLayer;       // no of layer in LLM/Brain structure
    hGensor*nodes=nullptr,*grads=nullptr,*leafs=nullptr;
    std::vector<hGensor> topo_nodes;    //nodes in Topological order
    std::vector<hGensor> sinks;      //  sinks of tensor flow graph
    
    ORDER order=LEFT_TO_RIGHT;
    // performance
    int     perf_runs=0;
    int64_t perf_cycles=0,perf_time_us=0;
    void * ctx=nullptr;
    size_t ctx_size = 0;

    void Clear()    {
        sinks.clear();
        // if (visited_hash_table.size>0)
        //     memset(visited_hash_table.keys, 0, visited_hash_table.size * sizeof(hGensor));
    }
#ifdef __USE_GGML__
    struct ggml_cgraph * cgraph=nullptr;        //only for debug
    struct ggml_cgraph * raw() {
#ifdef _TENSOR_G_
    return nullptr;
#endif
        assert(cgraph!=nullptr);
        return cgraph;
    }
#endif

public:
    TGraph()    {}
    TGraph(Fish *hF_,const string&nam_,void *ctx_,bool isGrad,int flag=0x0);
    TGraph(void *ctx_, size_t size_, bool grad_,bool isOnlySymbol_,int flag=0x0) :
        size(size_),ctx(ctx_),isOnlySymbol(isOnlySymbol_)  {
        assert(0);  //Deprecated        
    }
    virtual ~TGraph()   {   Clear();    }
    virtual int has(const string&name,int flag=0x0);
    virtual bool isValid( );
    
    size_t Prepare4Train(void *ctx_,GD_METHOD, int flag=0x0);
    
    // ORDER Order() {   return cgraph->order;   }
    bool empty();
    virtual string __repr__( string& suffix,string& prefix,hGensor root=nullptr,int flag=0x0);
    virtual string __repr__( hGensor root=nullptr,int flag=0x0){
        string suffix="", prefix="\t"; 
        return __repr__(suffix,prefix,root,flag);
    }

    virtual size_t Size(int flag=0x0)      {   return ctx_size;  }

    virtual bool TopoOrder(int flag=0x0); 

    int compute_on_plan( struct ggml_cplan* cplan,int flag=0x0);    

    virtual struct ggml_cgraph * BuildBackward(void * ctx_build,std::shared_ptr<TGraph> gf, bool accumulate=false,int flag=0x0);
    // Push new added node to last position
    void PushBack(hGensor node,int flag=0x0);    
    bool isSink(hGensor node,int flag=0x0); 
    virtual void Traverse(int flag=0x0);

    friend class Fish;
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class EDGE_DEVICES;
    friend class GeNeuron;
};
typedef std::shared_ptr<TGraph> hTGraph;