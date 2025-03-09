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

    struct ggml_cgraph * cgraph=nullptr;        //only for debug
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
    //lite ggml_hash_set
    std::set<hGensor> gset;
    //from GGML
    int size=0,nForwN=0,nForwL=0,nNeedGrad=0,nInput=0; // n_nodes=0,n_leafs=0;+
    hGensor*nodes=nullptr,*grads=nullptr,*leafs=nullptr;
    std::vector<hGensor> topo_nodes;    //nodes in Topological order
    std::vector<hGensor> sinks;      //  sinks of tensor flow graph
    // struct ggml_hash_set visited_hash_table = { 0, nullptr };   
    ORDER order=LEFT_TO_RIGHT;
    // performance
    int     perf_runs=0;
    int64_t perf_cycles=0,perf_time_us=0;
    struct ggml_context * ctx=nullptr;
    size_t ctx_size = 0;

    void Clear()    {
        sinks.clear();
        // if (visited_hash_table.size>0)
        //     memset(visited_hash_table.keys, 0, visited_hash_table.size * sizeof(hGensor));
    }
/*#ifndef GG_V12
    size_t hash_insert(const struct ggml_hash_set& hash_set, hGensor key) {
        size_t i = ggml_hash_find(&hash_set, key);
        assert(i != GGML_HASHSET_FULL);
        if (hash_set.keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;
        }
        // insert
        assert(hash_set.keys[i] == NULL);
        hash_set.keys[i] = key;
        return i;
    }
#endif*/

    struct ggml_cgraph * raw() {
#ifdef _TENSOR_G_
    return nullptr;
#endif
        assert(cgraph!=nullptr);
        return cgraph;
    }

public:
    TGraph()    {}
    // TGraph(Fish *hF_,const string&nam_,struct ggml_cgraph *c,bool isB=false,int flag=0x0) : 
    //     hFish(hF_),name(nam_),cgraph(c),isBackward(isB) {
    // }

    TGraph(Fish *hF_,const string&nam_,struct ggml_context *ctx_,bool isGrad,int flag=0x0);

    TGraph(struct ggml_context *ctx_, size_t size_, bool grad_,bool isOnlySymbol_,int flag=0x0) :
        size(size_),ctx(ctx_),isOnlySymbol(isOnlySymbol_)  {
        assert(0);  //Deprecated
        /*_INFO("=== %s ===\n",__func__);
        const size_t obj_size = ggml_graph_nbytes(size, grads);
        struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
        cgraph = (struct ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);
        hGensor* data_start = (hGensor*) (cgraph + 1);

        size_t hash_size = ggml_hash_size(size * 2);
        nodes = data_start;
        leafs = nodes + size;
        hGensor* hash_keys_ptr = leafs + size;
        grads = grad_ ? hash_keys_ptr + hash_size : NULL;
        // check that we allocated the correct amount of memory
        assert(obj_size == (size_t) (
            (grads ? (char *)(grads + size) : (char *)(hash_keys_ptr + hash_size)) - (char *)cgraph));

        memset(hash_keys_ptr, 0, hash_size * sizeof(hGensor));
        // CYS_0826
        // visited_hash_table = { hash_size, hash_keys_ptr };   
        // *cgraph = (struct ggml_cgraph) {
        //     size,0,0,nodes,grads,leafs,{ hash_size, hash_keys_ptr },
        //     LEFT_TO_RIGHT,0,0,0,
        // };     */
    }
    virtual ~TGraph()   {   Clear();    }
    virtual int has(const string&name,int flag=0x0);
    virtual bool isValid( );
    
    size_t Prepare4Train(struct ggml_context *ctx_,GD_METHOD, int flag=0x0);
    
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

    /*void compute_helper(int32_t n_threads,int flag=0x0){
        _INFO("%s ...",__func__);
        // ggml_graph_compute_helper(work_buffer, cgraph, n_threads);
        GST_TIC(t0);
        struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads,NULL);
        if (plan.work_size > 0) {
            work_buffer.resize(plan.work_size);
            plan.work_data = work_buffer.data();
        }
        tPlan = GST_TOC(t0);       

        compute_on_plan(&plan);
        _INFO("%s:  nT=%d  symbol=%d T = %.3g(plan=%.3g)sec\n", __func__,n_threads,isOnlySymbol, GST_TOC(t0),tPlan);
    }*/

    virtual struct ggml_cgraph * BuildBackward(struct ggml_context * ctx_build,std::shared_ptr<TGraph> gf, bool accumulate=false,int flag=0x0);
    // Push new added node to last position
    void PushBack(hGensor node,int flag=0x0);
    
    bool isSink(hGensor node,int flag=0x0);   

    virtual void Traverse(int flag=0x0);
    /* Deprecated
    void print( ) {
        int64_t perf_total_per_op_us[GGML_OP_COUNT] = {0};

        _INFO("=== GRAPH ===\n");

        _INFO("n_nodes = %d\n", cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            hGensor node = nodes[i];
        }

        _INFO("n_leafs = %d\n", cgraph->n_leafs);
        for (int i = 0; i < cgraph->n_leafs; i++) {
            hGensor node = leafs[i];
            // _INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",i,node->ne[0], node->ne[1],ggml_op_name(node->op), ggml_get_name(node));        
        }
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (perf_total_per_op_us[i] == 0) {
                continue;
            }
            //_INFO("perf_total_per_op_us[%16s] = %7.3f ms\n", ggml_op_name(i), (double) perf_total_per_op_us[i] / 1000.0);
        }
        _INFO("========================================\n");
    }*/

    friend class Fish;
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class EDGE_DEVICES;
};
typedef std::shared_ptr<TGraph> hTGraph;