/**
 *  Copyright 2023-2024 by Grusoft 
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
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h> 
using namespace std;
// #include "../ggml/ggml.h"
// #include "../ggml/ggml-impl.h"
// #include "../ggml/ggml-alloc.h"
// #include "../ggml/ggml-backend.h"
// #include "../ggex/common-ggml.h"
#include "../ggex/GG_util.hpp"
#include "../lenda/util/GST_util.hpp"


class TGraph;   
typedef shared_ptr<TGraph> hTGraph;

class TGraph : public std::enable_shared_from_this<TGraph> {
    TGraph(const TGraph&);
	TGraph& operator=(const TGraph&);

    struct ggml_cgraph * cgraph=nullptr;        //only for debug
protected:
    bool isOnlySymbol = true;
    double tX=0,tCompute=0.0,tPlan=0.0;
    std::vector<uint8_t> work_buffer;
    //from GGML
    int size=0,n_nodes=0,n_leafs=0;
    hGensor*nodes=nullptr,*grads=nullptr,*leafs=nullptr;
    struct ggml_hash_set visited_hash_table = { 0, nullptr };   
    enum ggml_cgraph_eval_order order=GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;
    // performance
    int     perf_runs=0;
    int64_t perf_cycles=0,perf_time_us=0;
    struct ggml_context * ctx=nullptr;
    size_t ctx_size = 0;

    void clear()    {
        n_leafs = 0;
        n_nodes = 0;
        if (visited_hash_table.size>0)
            memset(visited_hash_table.keys, 0, visited_hash_table.size * sizeof(hGensor));
    }

    size_t hash_insert(struct ggml_hash_set& hash_set, hGensor key) {
        size_t i = ggml_hash_find(hash_set, key);
        GGML_ASSERT(i != GGML_HASHTABLE_FULL);
        if (hash_set.keys[i] == key) {
            return GGML_HASHTABLE_ALREADY_EXISTS;
        }
        // insert
        GGML_ASSERT(hash_set.keys[i] == NULL);
        hash_set.keys[i] = key;
        return i;
    }

    void visit_parents(hGensor node,int flag=0x0) {
        if (node->grad == NULL) {
            // this usually happens when we generate intermediate nodes from constants in the backward pass
            // it can also happen during forward pass, if the user performs computations with constants
            if (node->op != GGML_OP_NONE) {
                //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
            }
        }

        // check if already visited
        if (hash_insert(visited_hash_table, node) == GGML_HASHTABLE_ALREADY_EXISTS) {
            return;
        }

        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const int k =
                (order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
                (order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
                /* unknown order, just fall back to using i*/ i;
            if (node->src[k]) {
                visit_parents(node->src[k]);
            }
        }

        if (node->op == GGML_OP_NONE && node->grad == NULL) {
            // reached a leaf node, not part of the gradient graph (e.g. a constant)
            GGML_ASSERT(n_leafs < size);

            if (strlen(node->name) == 0) {
                ggml_format_name(node, "leaf_%d", n_leafs);
            }

            leafs[n_leafs] = node;
            n_leafs++;
        } else {
            GGML_ASSERT(n_nodes < size);

            if (strlen(node->name) == 0) {
                ggml_format_name(node, "node_%d", n_nodes);
            }

            nodes[n_nodes] = node;
            if (grads) {
                grads[n_nodes] = node->grad;
            }
            n_nodes++;
        }
    }

public:
    TGraph()    {}
    TGraph(struct ggml_context *ctx_,int flag=0x0) :ctx(ctx_)   {
    }

    TGraph(struct ggml_context *ctx_, size_t size_, bool grad_,bool isOnlySymbol_,int flag=0x0) :
        size(size_),ctx(ctx_),isOnlySymbol(isOnlySymbol_)  {
        GGML_PRINT("=== %s ===\n",__func__);
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
        visited_hash_table = { hash_size, hash_keys_ptr };   
        *cgraph = (struct ggml_cgraph) {
            size,0,0,nodes,grads,leafs,{ hash_size, hash_keys_ptr },
            GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,0,0,0,
        };     

        //allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    }

    virtual ~TGraph()   {   clear();    }
    
    virtual size_t Size(int flag=0x0)      {   return ctx_size;  }

    hGensor get_tensor(const char * name,int flag=0x0) {
        for (int i = 0; i < n_leafs; i++) {
            hGensor leaf = leafs[i];
            if (strcmp(leaf->name, name) == 0) {
                return leaf;
            }
        }
        for (int i = 0; i < n_nodes; i++) {
            hGensor node = nodes[i];
            if (strcmp(node->name, name) == 0) {
                return node;
            }
        }
        assert(0);
        return NULL;
    }

    

    int compute_on_plan( struct ggml_cplan* cplan,int flag=0x0);

    void compute_helper(int32_t n_threads,int flag=0x0){
        GGML_PRINT_DEBUG("%s ...",__func__);
        // ggml_graph_compute_helper(work_buffer, cgraph, n_threads);
        GST_TIC(t0);
        struct ggml_cplan plan = ggml_graph_plan(cgraph, n_threads);
        if (plan.work_size > 0) {
            work_buffer.resize(plan.work_size);
            plan.work_data = work_buffer.data();
        }
        tPlan = GST_TOC(t0);       

        compute_on_plan(&plan);
        GGML_PRINT_DEBUG("%s:  nT=%d  symbol=%d T = %.3g(plan=%.3g)sec\n", __func__,n_threads,isOnlySymbol, GST_TOC(t0),tPlan);
    }

    void build_forward(hGensor tensor, bool expand,int flag=0x0)    {
        const int n0 = n_nodes;
        // UNUSED(n0);
        if(1){
            visit_parents(tensor);
            cgraph->n_nodes = n_nodes;
            cgraph->n_leafs = n_leafs;            
        }else{
            ggml_visit_parents_x(cgraph, tensor,0);
            n_nodes = cgraph->n_nodes;
            n_leafs = cgraph->n_leafs;
        }
        const int n_new = n_nodes - n0;
        GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

        if (n_new > 0) {
            // the last added node should always be starting point
            GGML_ASSERT(nodes[n_nodes - 1] == tensor);
        }        
    }

    void disconnect_node(ggml_tensor * t) {
        t->op = GGML_OP_NONE;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t->src[i] = NULL;
        }
    }

    virtual void Traverse(int flag=0x0);

    void print( ) {
        int64_t perf_total_per_op_us[GGML_OP_COUNT] = {0};

        GGML_PRINT("=== GRAPH ===\n");

        GGML_PRINT("n_nodes = %d\n", n_nodes);
        for (int i = 0; i < n_nodes; i++) {
            hGensor node = nodes[i];

            perf_total_per_op_us[node->op] += MAX(1, node->perf_time_us);

            GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64 "] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
                    i,
                    node->ne[0], node->ne[1], node->ne[2],
                    ggml_op_name(node->op), (node->flags & GGML_TENSOR_FLAG_PARAM) ? "x" : node->grad ? "g" : " ", node->perf_runs,
                    (double) node->perf_cycles  / (double) ggml_cycles_per_ms(),
                    (double) node->perf_cycles  / (double) ggml_cycles_per_ms() / (double) node->perf_runs,
                    (double) node->perf_time_us / 1000.0,
                    (double) node->perf_time_us / 1000.0 / node->perf_runs);
        }

        GGML_PRINT("n_leafs = %d\n", n_leafs);
        for (int i = 0; i < n_leafs; i++) {
            hGensor node = leafs[i];
            GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                    i,node->ne[0], node->ne[1],ggml_op_name(node->op), ggml_get_name(node));        }
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (perf_total_per_op_us[i] == 0) {
                continue;
            }
            //GGML_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", ggml_op_name(i), (double) perf_total_per_op_us[i] / 1000.0);
        }
        GGML_PRINT("========================================\n");
    }

    struct ggml_cgraph * CGraph() {
        assert(cgraph!=nullptr);
        return cgraph;
    }

    friend class Ganglia;
};
