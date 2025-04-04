/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief 
 *  \author Yingshi Chen
 */
#pragma once
#include "../../ggex/GG_util.hpp"
#ifdef __USE_GGML__
    #include "ggml-backend.h"
    #include "ggml-cpu.h"
#endif
class TGraph;
typedef shared_ptr<TGraph> hTGraph;

class EDGE_DEVICES{
public:
    // Fish *hFish = nullptr;
    int nCore = 1;  //  cores of CPU,GPU(Streaming Multiprocessors)
    size_t sz = 0x0;
#ifdef __USE_GGML__
    ggml_backend_buffer_t back_data = NULL; 
    // ggml_backend_t cpu = nullptr;
    std::vector<ggml_backend_t> workers;
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_gallocr_t alloc_tmp = nullptr;
    ggml_backend_sched_t sched0 = nullptr;

    virtual ggml_backend_sched_t GetSched(int flag=0x0){
        assert(sched0!=nullptr);
        return sched0;
    }
    virtual bool SplitSched(hTGraph ,int flag=0x0);
#endif

#ifdef _TENSOR_G_
    virtual bool InitGPU(const CLI_params&hparams,int flag=0x0);
#endif   
    
    EDGE_DEVICES(const CLI_params&hparams, int flag=0x0);
    virtual ~EDGE_DEVICES();
    EDGE_DEVICES(EDGE_DEVICES const&)    = delete;
    void operator=(EDGE_DEVICES const&)  = delete;
    static shared_ptr<EDGE_DEVICES> GetInstance(const CLI_params&hparams, int flag=0x0);

    virtual size_t Alloc(hTGraph graph,void *ctx,int flag=0x0);
    // reserve the buffers to avoid reallocations
    virtual bool Reserve(hTGraph graph,int flag=0x0);
    virtual bool AllocGraph(hTGraph graph,int flag=0x0);
    // bool Build(struct ggml_cgraph *cgraph,bool isPlan,int flag=0x0);
    virtual string __repr__( string& suffix,string& prefix,int flag=0x0);

    bool isOnlyCPU()    {
#ifdef __USE_GGML__
        for(auto worker : workers){
            if (!ggml_backend_is_cpu(worker))
                return false;
        }
        return true;    /*!=nullptr && workers.size()==1;*/
#else
        return false;
#endif
    }

    int SetThread(int nThread,int flag=0x0);       
    virtual int SetBackend(hGensor cur,int flag=0x0);

};
typedef shared_ptr<EDGE_DEVICES>hEDevices;

void SYNC_DEVICE(int flag=0x0);