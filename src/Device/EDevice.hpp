/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Edge devices & resource limited scheduling
 *  \author Yingshi Chen
 */
#pragma once
#include "../CLI_params.hpp"
#include "../Tensor/GTensor.hpp"
#include "../Utils/GST_obj.hpp"

class RLS_BP;
class TGraph;
typedef shared_ptr<TGraph> hTGraph;

class EDGE_DEVICES {
   public:
    struct GPU_ {
        int cc;                  // compute capability
        int nsm;                 // number of streaming multiprocessors
        size_t smpb;             // max. shared memory per block
        size_t smpbo;            // max. shared memory per block (with opt-in)
        bool vmm;                // virtual memory support
        size_t vmm_granularity;  // granularity of virtual memory
        size_t total_vram = 0;
        int warp_size;  // Number of threads in a dispatch

        int maxThreadsPerBlock;  //   Maximum number of threads per block 4090:(1536)
        int maxThreadsDim[3];    //   Maximum size of each dimension of a block, 4090:    (1024, 1024, 64)
        int maxGridSize[3];      //  4090: (2147483647, 65535, 65535)

        static int MAX_COUNT;  //  16
        static std::vector<GPU_> cudaGetDevice(int flag = 0x0);
    };
    std::vector<GPU_> gpus;
    // Fish *hFish = nullptr;
    int nCore = 1;  //  cores of CPU,GPU(Streaming Multiprocessors)
    size_t sz = 0x0, mostRAM = 0;
    shared_ptr<RLS_BP> hRLS = nullptr;

    virtual bool InitGPU(const CLI_params& hparams, int flag = 0x0);
    virtual bool ClearGPU(int flag = 0x0);

    EDGE_DEVICES();
    EDGE_DEVICES(const CLI_params& hparams, int flag = 0x0);
    virtual ~EDGE_DEVICES();
    EDGE_DEVICES(EDGE_DEVICES const&)   = delete;
    void operator=(EDGE_DEVICES const&) = delete;
    static shared_ptr<EDGE_DEVICES> GetInstance(const CLI_params& hparams, int flag = 0x0);

    virtual size_t AfterBuild(hTGraph graph, void* ctx, int flag = 0x0);
    // reserve the buffers to avoid reallocations
    virtual bool Reserve(hTGraph graph, int flag = 0x0);
    virtual bool AllocGraph(hTGraph graph, int flag = 0x0);
    // bool Build(struct ggml_cgraph *cgraph,bool isPlan,int flag=0x0);
    virtual string __repr__(string& suffix, string& prefix, int flag = 0x0);

    bool isOnlyCPU() {
#ifdef __USE_GGML__
        for (auto worker : workers) {
            if (!ggml_backend_is_cpu(worker))
                return false;
        }
        return true; /*!=nullptr && workers.size()==1;*/
#else
        return false;
#endif
    }
    int GridDim(size_t nEle, int typ = 0x0, int flag = 0x0);
    int SetThread(int nThread, int flag = 0x0);
    virtual int SetBackend(hGensor cur, int flag = 0x0);
    template <typename T>
    T* GetScheduler() {
        T* hS = dynamic_cast<T*>(hRLS.get());
        assert(hS != nullptr);
        return hS;
    }
};
typedef shared_ptr<EDGE_DEVICES> hEDevices;

bool SYNC_DEVICE(const std::string& sX = "", int flag = 0x0);