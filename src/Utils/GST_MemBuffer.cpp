/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  v0.1        need many refactor
 *
 *  \brief GRUSOFT TEMPLATE	- Memory/Buffer using the RAII (Resource Acquisition Is Initialization) principle
 *  \author Yingshi Chen
 */

#include "GST_MemBuffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>

#include "../Tensor/GTensor.hpp"
#include "GST_log.hpp"
#include "GST_obj.hpp"

hTensorBuffer gBUFF = nullptr;

GST_TensorBuffer::~GST_TensorBuffer() {
    Clear();

    records.Clear();
    /* Free the main buffer
    if (buffer_ != nullptr) {
        cudaFree(buffer_);
        buffer_ = nullptr;
    }*/
}
/**
 * 1. cudaMemGetInfo reports:
        Memory allocated by your program (via cudaMalloc, cudaMemcpy, etc.).
        Additional memory reserved by the CUDA runtime (e.g., for kernels, libraries like cuBLAS).
        Example: If your app allocates 500MB, cudaMemGetInfo might report 600MB due to overhead.
    2. nvidia-smi reports:
        Actual physical memory in use by all processes (including other CUDA apps, graphics compositors, etc.).
        Excludes reserved but unused memory (e.g., fragmentation, CUDA context pools)
*/
void GST_TensorBuffer::Dump(int type, int flag) const {
    size_t sz0, sz1;
    cudaError_t err = cudaMemGetInfo(&sz0, &sz1);
    std::vector<Allocation> mems;

    std::sort(mems.begin(), mems.end(),  // ugly because we don't have a typedef for the std::pair
              [](const Allocation& a, const Allocation& b) { return a.sz > b.sz; });
    size_t szNow = 0, i = 0, szFree = 0;
    double mUsed = (sz1 - sz0) / 1.0e6;
    for (auto mem : mems) {
        szNow += mem.sz;
    }
    size_t szA = szSum[ACTIVATION], szW = szSum[WEIGHT], szG = szSum[GRAD], szMoment = szSum[MOMENT], szTemp = szSum[TEMP], szOther = szSum[OTHER];
    szFree = szA + szW + szG + szMoment + szTemp + szOther - szNow;
    _INFO(
        "[MEMORY] Current usage statistics:  %ld mem-blocks sum=%.3gG(%.3gG) \n\tactivation=%.5gM weight=%.5gM grad=%.5gM moments=%.5gM temp=%.5gM "
        "other=%.5gM\n",
        mems.size(), szNow / 1.0e9, szFree / 1.0e9, szA / 1.0e6, szW / 1.0e6, szG / 1.0e6, szMoment / 1.0e6, szTemp / 1.0e6, szOther / 1.0e6);
    _INFO("\tcurBrach=%.5gM mUsed==%.5gM\n", (szA + szW + szG + szMoment) / 1.0e6, mUsed);
    int nDump = type == KOIFISH_MISS_MEMBLOCK ? mems.size() : type == KOIFISH_OUTOF_GPUMEMORY ? 32 : nMostMemItem;
    if (nDump > 0) {  // decsend by memory size
        size_t szTotal = 0;
        for (auto mem : mems) {
            szTotal += mem.sz;
            _INFO("\t%ld\t%6gM  @%s \t%.3gG\n", i++, mem.sz / 1.0e6, mem.desc.c_str(), szTotal / 1.0e9);
            if (i > nDump)
                break;
        }
        _INFO("\n");
    }
}

GST_TensorBuffer::Allocation::Allocation(size_t sz_, string d_, void* hData_, int flag) : sz(sz_), desc(d_), hData(hData_) {
    assert(!desc.empty());
    if (G_Has_(d_, {"ROPE.sincos", "tmpGateDelta"})) {
        DEBUG_HERE;
    }
    char last = desc[desc.length() - 1];
    switch (last) {
        case 'a':
            if (desc.substr(0, 3) == "tmp") {
                type = TYPE::TEMP;
            } else {
                type = TYPE::ACTIVATION;
            }
            break;
        case 'w':
            type = TYPE::WEIGHT;
            break;
        case 'g':
            type = TYPE::GRAD;
            break;
        case 'm':
            type = TYPE::MOMENT;
            break;
        case 't':
            type = TYPE::TEMP;
            break;
        default:
            type = TYPE::OTHER;
            break;
    }
}

void GlobalMemoryInfo(int type, int flag) {}

void* GST_TensorBuffer::Register(size_t sz_, string d_, void* hData_, int flag) {
    try {
        Allocation alloc(sz_, d_, hData_, flag);
        records.insert(d_, alloc);
        szSum[alloc.type] += alloc.sz;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}
shared_ptr<GTensor> GST_TensorBuffer::RegisterTensor(Fish* hFish, const string& name_, SHAPE shape, typNUMBER tpD_, int flag) {
    hGTensor tensor = GT(hFish, tpD_, shape);
    zTensors.push_back(tensor.get());
    return tensor;
}

/**/
bool GST_TensorBuffer::Free(void* hObj, int flag) {
    for (size_t id = 0; id < records.size(); id++) {
        Allocation alloc;
        records.at(id, &alloc);
        if (alloc.hData == hObj) {
            records.erase(alloc.desc);
            return true;
        }
    }
    _WARN("[Free] failed to find record of %p! records=%ld.\n", hObj, records.size());
    // assert(0 && "GST_TensorBuffer::FreeMem failed");
    return false;
}
/*
 */
bool GST_TensorBuffer::Clear(int flag) {
    try {
        if (bt4c == nullptr)  // hack
            return true;

        // if (cublaslt_workspace != nullptr)
        //     cudaCheck(cudaFree(cublaslt_workspace));
        // if (GTensor::stat_info != nullptr)
        //     cudaCheck(cudaFree(GTensor::stat_info));
        // if (GTensor::cudnn_workspace != nullptr)
        //     cudaCheck(cudaFree(GTensor::cudnn_workspace));

        bt4c = nullptr, delta = nullptr, tmpDelta = nullptr, outL = nullptr, scratch = nullptr, tmpFF1 = nullptr, tmpW = nullptr, tmpGW = nullptr,
        residual   = nullptr;
        tmpTernary = nullptr;
        gate_delta = nullptr;
        return true;
    } catch (const std::exception& e) {
        _WARN("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char* info) {
        _WARN("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _WARN("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}
