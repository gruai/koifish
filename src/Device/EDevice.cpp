/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Edge devices & resource limited scheduling
 *  \author Yingshi Chen
 */

#include "EDevice.hpp"

#include "../Manifold/Fish.hpp"
#include "../Manifold/Neuron.hpp"
#include "../Manifold/TGraph.hpp"

std::string SUM::GPU_Info(int flag) {
    size_t szFree, szTotal;
    char buf[1024];
    cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
    sprintf(buf, "mGPU=%.6gM(free=%.6gM)", (szTotal - szFree) / 1.0e6, szFree / 1.0e6);
    return buf;
}

size_t EDGE_DEVICES::AfterBuild(hTGraph hTG, void* ctx, int flag) {
    INIT_WEIGHT tpInitWeight = hTG->hFish->config.model.tpInitWeight;
    if (hRLS != nullptr) {
        std::vector<hGensor> tInMaps;
        for (auto gt : hRLS->tMaps) {
            tInMaps.push_back(gt.first);
        }
        // assert(hRLS->tMaps.size()>=hTG->gset.size());
        for (auto tensor : hTG->gset) {
            if (hRLS->tMaps.find(tensor) == hRLS->tMaps.end()) {
                //  CHECK_SAME_TENSORS???
                Gensors2File(TO_VECTOR(hTG->gset), "~/gset_1.info");
                Gensors2File(tInMaps, "~/gset_2.info");
                exit(KOIFISH_INVALID_GSET);
            }
        }
    } else /**/ {
        for (auto tensor : hTG->gset) {
            if (tpInitWeight == SERIALIZE)
                tensor->tpInit = tpInitWeight;

            tensor->Alloc();
        }
    }

    return sz;
}

bool RLS_BP::Planning(int flag) {
    T_fore = 10, T_back = 10;
    int t        = 0;
    double costs = 0;
    for (auto node : curTasks()) {
        if (costs + node->cost > budget) {
            T_fore = t;
            break;
        }
        node->begin = 0;
        costs += node->cost;
        t++;
    }

    costs = 0;
    for (auto it = curTasks().rbegin(); it != curTasks().rend(); ++it) {
        TaskNode* node = *it;
        if (costs + node->cost > budget) {
            T_back = t;
            break;
        }
        node->begin = 0;
        costs += node->cost;
        t++;
    }

    Verify();
    return true;
};

bool RLS_BP::Verify(int flag) {
    int t = 0;
    for (auto node : curTasks()) {  // validate
        t++;
    }
    return true;
}

void GeNeuron::OnRemater(RLS_BP* schedule, int typ, int flag) {
    switch (typ) {
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

TASK_STATUS RLS_BP::GetTensorStatus(int step, hGTensor tensor, int flag) {
    assert(tMaps.find(tensor) != tMaps.end());
    return tMaps[tensor];
}
TASK_STATUS RLS_BP::SetTensorStatus(int step, hGTensor tensor, TASK_STATUS sta, int flag) {
    // int iter = hFish->hOPT->GetIter();
    assert(tMaps.find(tensor) != tMaps.end());
    tensor->last_stp = step;
    tMaps[tensor]    = sta;
    return tMaps[tensor];
}

TASK_STATUS RLS_BP::GetStatus(int t, void* hObj, int flag) {
    TASK_STATUS status = PASS;
    GeNeuron* hNEURON  = (GeNeuron*)(hObj);
    int nN             = curTasks().size();
    assert(t < 2 * nN);
    TaskNode* hNode = nullptr;
    if (t < nN) {
        if (t < T_fore)
            ;  // hNEURON->OnRemater(this);
    } else {
        // t<nN ? curTasks[t] : curTasks[2*nN-t];
    }
    assert(hNode->hOBJ == hObj);

    return status;
}

EDGE_DEVICES::EDGE_DEVICES(const CLI_params& config, int flag) {
    InitGPU(config, flag);
    hRLS = std::make_shared<RLS_BP>(this, config, flag);
    return;
}

EDGE_DEVICES::~EDGE_DEVICES() { ClearGPU(0x0); }

/*
    llm_build_cb cb = [&](hGensor  cur, const char * name, int il)
    why "norm"      ???
*/
int EDGE_DEVICES::SetBackend(hGensor cur0, int flag) {
    int il = 0, no = 0, pick = -1;
    // if (strcmp(cur->name, "norm") != 0) // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
    //     return -1;
    // auto cur = G(cur0);
    return pick;
}

int EDGE_DEVICES::GridDim(size_t nEle, int typ, int flag) {
    int nActivePC = 1;  //	for __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
    // cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nActivePC, kernel_output<__nv_fp8_e5m2, AT>, dBLOCK, smemPB));
    return nCore * nActivePC;
}

int EDGE_DEVICES::SetThread(int nThread, int flag) {
    assert(0);
    return 0x0;
}

string EDGE_DEVICES::__repr__(string& suffix, string& prefix, int flag) {
    return "";
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    if (isOnlyCPU()) {
        // assert(workers.size()==1);
        sprintf(buf + strlen(buf), "OnlyCPU");
    } else {
    }

    if (flag > 0)
        _INFO("%s", buf);
    return buf;
}

bool EDGE_DEVICES::AllocGraph(hTGraph graph, int flag) {
    bool bRet = false;

    return bRet;
}

bool EDGE_DEVICES::Reserve(hTGraph graph, int flag) { return false; }
