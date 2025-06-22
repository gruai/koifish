/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Edge devices & resource limited scheduling
 *  \author Yingshi Chen
 */

#include "EDevice.hpp"
#include "../Manifold/Neuron.hpp"
#include "../Manifold/Fish.hpp"
#include "../Manifold/TGraph.hpp"

size_t EDGE_DEVICES::AfterBuild(hTGraph hTG,void *ctx,int flag)    {
    INIT_WEIGHT tpInitWeight = hTG->hFish->tpInitWeight;
    if(hRLS!=nullptr){
        for(auto tensor : hTG->gset){
            assert(hRLS->tensors.find(tensor)!=hRLS->tensors.end());
        }
        // hRLS->Prepare();
        
    }   else/**/    {
        for(auto tensor : hTG->gset){
            if(tpInitWeight==SERIALIZE)        
                tensor->tpInit = tpInitWeight;
            
            tensor->Alloc( );
        }  
    }

    return sz;
}  

bool RLS_BP::Planning(int flag) {
    T_fore=10,T_back=10;
    int t = 0;
    double costs = 0;
    for(auto node : nodes){
        if( costs + node->cost>budget )  {
            T_fore = t;  break;      
        }
        node->begin = 0;
        costs += node->cost;             
        t++;
    }

    costs = 0;
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)    {
        Node *node = *it;
        if( costs + node->cost>budget )  {
            T_back = t;  break;      
        }
        node->begin = 0;
        costs += node->cost;             
        t++;
    }

    Verify();
    return true;
};

bool RLS_BP::Verify(int flag)   {
    int t = 0;
    for(auto node : nodes){     //validate
        t++;
    }
    return true;
}

void GeNeuron::OnRemater(RLS_BP *schedule,int typ,int flag){
    switch(typ){
    // case OFF_LOAD:
    //     for(auto v : vRemater)
    //         ;
    //     break;
    // case REMATER:
    //     for(auto v : vRemater)
    //     ;
    //     break;
    default:
        assert(0);
    }
}

RLSchedule::tpSTATUS RLS_BP::GetTensorStatus(int step,hGTensor tensor,int flag){
    assert(tensors.find(tensor)!=tensors.end());
    return tensors[tensor];
}
RLSchedule::tpSTATUS RLS_BP::SetTensorStatus(int step,hGTensor tensor,tpSTATUS sta,int flag){
    // int iter = hFish->hOPT->GetIter();
    assert(tensors.find(tensor)!=tensors.end());
    tensor->last_stp = step;
    tensors[tensor] = sta;
    return tensors[tensor];
}

RLSchedule::tpSTATUS RLS_BP::GetStatus(int t,void *hObj,int flag){
    tpSTATUS status = PASS;
    GeNeuron *hNeuron = (GeNeuron *)(hObj);
    int nN = nodes.size();      
    assert(t<2*nN);
    Node *hNode = nullptr;
    if(t<nN) {
        if(t<T_fore)
            ;//hNeuron->OnRemater(this);
    }else{
            // t<nN ? nodes[t] : nodes[2*nN-t];
    }    
    assert(hNode->hOBJ == hObj);

    return status;
}

EDGE_DEVICES::EDGE_DEVICES(const CLI_params&config, int flag){
    #ifdef _TENSOR_G_
        InitGPU(config,flag);
        hRLS = std::make_shared<RLS_BP>(this,config,flag);
        return;
    #else
        assert(back_data==nullptr);
        assert(workers.size()==0);
        const size_t dev_count = 1; //ggml_backend_dev_count();
        _INFO("%s: %zu devices\n\n",__func__, dev_count);
        int n_ok=0,nT0=std::thread::hardware_concurrency(),nT=config.nThread();
        string sTp = config.KV({"train","device"},"");
        ggml_backend_t backend = nullptr;
        size_t free, total;    
        for (size_t i = 0; i < dev_count; ++i) {        
            assert(0);
    #ifdef GG_V12  
            /*auto dev = ggml_backend_dev_get(i);
            devs.push_back(dev);
            ggml_backend_dev_memory(dev, &free, &total);
            printf("[EDGE_DEVICE]_%d %s:%s  memory: %zu MB (%zu MB free)\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev)
                ,total / 1024 / 1024, free / 1024 / 1024);   
            
            ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);*/
    #else
            backend = ggml_backend_cpu_init();
    #endif
            assert(backend != NULL);
            /*if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nT);
                auto buft = ggml_backend_cpu_buffer_type();
                bufts.push_back(buft);
            } else {
                bufts.push_back(ggml_backend_get_default_buffer_type(backend));
            }
            workers.push_back(backend);*/
        }
        
    #ifdef GG_V12    
        for (auto backend : workers) {
            // Put the backend to be tested in front so that it's prioritized:
            std::vector<ggml_backend_t> backends_modded = {backend};
            backends_modded.insert(backends_modded.end(), workers.begin(), workers.end());
    
            sched0 = ggml_backend_sched_new(
                backends_modded.data(), nullptr, backends_modded.size(), GGML_DEFAULT_GRAPH_SIZE/*2048*/, false);
            break;
            // std::pair<int, int> result = test_backend(backend_sched, backends[i]);
            // ggml_backend_sched_free(backend_sched);
        }    
    #else
        sched0 = ggml_backend_sched_new(workers.data(), bufts.data(), bufts.size(), LLAMA_TRAIN_MAX_NODES, false);
    #endif
        int i,nBack = ggml_backend_sched_get_n_backends(sched0);
        for (int i = 0; i < nBack; i++) {
            auto back = ggml_backend_sched_get_backend(sched0, i);
            _INFO("");
        }
    
        /*
            static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
             {
                ggml_backend_cpu_buffer_type_get_name,
                 ggml_backend_cpu_buffer_type_alloc_buffer,
                 ggml_backend_cpu_buffer_type_get_alignment,
                NULL, // defaults to SIZE_MAX
                 NULL, // defaults to tBYTE
                ggml_backend_cpu_buffer_type_is_host,
            },
             NULL,
        };
        */
    #endif
    }
    
    EDGE_DEVICES::~EDGE_DEVICES(){
        ClearGPU(0x0);
    #ifdef __USE_GGML__
       for (auto backend : workers) {
            ggml_backend_free(backend); 
       }
       if(sched0!=nullptr)
            ggml_backend_sched_free(sched0);
        if(alloc_tmp!=nullptr)
            ggml_gallocr_free(alloc_tmp);
    #endif
    }
    
    /*
        llm_build_cb cb = [&](hGensor  cur, const char * name, int il) 
        why "norm"      ???
    */
    int EDGE_DEVICES::SetBackend(hGensor cur0,int flag)    {
        int il = 0, no=0,pick=-1;    
        // if (strcmp(cur->name, "norm") != 0) // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        //     return -1;
        auto cur = G(cur0);
    #ifdef __USE_GGML__
        for (auto * backend : workers) {
            bool isBuft = false/*ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft)*/;
            bool isOP = ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur);
            if (  isOP /*&& isBuft*/ ) {
                ggml_backend_sched_set_tensor_backend(GetSched(), cur, backend);
                pick = no;
                break;
            }
            no++;
        }
    #endif
        return pick;    
    }
    
    #ifdef __USE_GGML__
    #include "ggml-impl.h"
    /*
        const int node_backend_id = tensor_backend_id(node); =1 for "norm???"
    */
    bool EDGE_DEVICES::SplitSched(hTGraph hTG,int flag)  {    
        assert(hTG!=nullptr);
        if(workers.size()==1)   //no need to split
            return true;
    
        int n0=0;
        for(auto node : hTG->sinks){
            int no = SetBackend(node);
            if(no==0)   n0++;
        }    
        auto cgraph = hTG->raw();   //    assert(cgraph!=nullptr);
        int nNode = cgraph->n_nodes;
        auto sched = GetSched();
        if (!ggml_backend_sched_reserve(sched, cgraph)) {
            _ERROR("%s: failed to allocate compute buffers\n", __func__);
            // llama_free(ctx);
            return false;
        }
    
        for (size_t i = 0; i < workers.size(); i++) {
            ggml_backend_t backend = workers[i];
            ggml_backend_buffer_type_t buft = bufts[i];
            size_t size = 0x0;  //ggml_backend_sched_get_buffer_size(sched, backend);
            if (size > 1) {
                _INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),size / 1024.0 / 1024.0);
            }
        }
    
        // note: the number of splits during measure is higher than during inference due to the kv shift
        int n_splits = ggml_backend_sched_get_n_splits(sched);
        _INFO("%s: graph nNode=%d nSplits=%d\n", __func__, nNode, n_splits );
        assert(n_splits>=2);
        return true;
    }
    #endif
    
    int EDGE_DEVICES::GridDim(size_t nEle,int typ,int flag){
        int nActivePC = 1;		//	for __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR) 
        // cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nActivePC, kernel_output<__nv_fp8_e5m2, AT>, dBLOCK, smemPB));        
        return nCore*nActivePC;
    }
    
    int EDGE_DEVICES::SetThread(int nThread,int flag)   {
        assert(0);
        /*int nSet = 0;
        for(auto worker : workers){
            if (ggml_backend_is_cpu(worker))   {
                ggml_backend_cpu_set_n_threads(worker, nThread);
                nSet ++;
                //ggml_backend_cpu_set_threadpool(hEDS->cpu, threadpool);
            // ggml_backend_cpu_set_abort_callback(hEDS->cpu, lctx.abort_callback, lctx.abort_callback_data);
            }
        }
    
    #ifdef GGML_USE_BLAS
        if (lctx.backend_blas != nullptr) {
            ggml_backend_blas_set_n_threads(lctx.backend_blas, n_threads);
        }
    #endif*/
        return 0x0;
    }
    
    string EDGE_DEVICES::__repr__( string& suffix,string& prefix,int flag)  {
        return "";
        char buf[5012]="\0";
        const char*tab=prefix.c_str();
        if(isOnlyCPU()){
            // assert(workers.size()==1);
            sprintf(buf+strlen(buf),"OnlyCPU"); 
        }else{
            /*for (auto * backend : workers) {
                if (ggml_backend_is_cpu(backend)) {  
                    sprintf(buf+strlen(buf),"CPU,"); 
                }else{
                    sprintf(buf+strlen(buf),"GPU,");
                }
            }  */      
        }
    
        if(flag>0)
            _INFO("%s",buf); 
        return buf;  
    }
    
    bool EDGE_DEVICES::AllocGraph(hTGraph graph,int flag)    {
        bool bRet = false;    
        
        return bRet;
    }
    
    bool EDGE_DEVICES::Reserve(hTGraph graph,int flag){
        return false;
    }
     