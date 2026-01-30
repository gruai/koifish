/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Memory/Buffer
 *  \author Yingshi Chen
 */

#pragma once

#include "../g_def_x.hpp"
#include "../g_float.hpp"
#include "GST_obj.hpp"

using namespace std;

class GTensor;
class Fish;

class GST_TensorBuffer {
   protected:
    enum TYPE { ACTIVATION, WEIGHT, MOMENT, GRAD, TEMP, OTHER, COUNT };
    // size_t szA = 0, szW = 0, szG = 0, szMoment = 0, szTemp = 0, szOther = 0;
    size_t szSum[COUNT] = {};

    struct Allocation {
        TYPE type;
        void* hData = nullptr;
        size_t sz;
        string desc;
        Allocation() {}
        Allocation(size_t sz_, string d_, void* hData = nullptr, int flag = 0x0);
    };

    int nMostMemItem       = 1024;
    Fish* hFish            = nullptr;
    void* buffer_          = nullptr;
    size_t total_size_     = 0x0;
    size_t current_offset_ = 0x0;
    GST_Dict<Allocation> records;
    std::vector<GTensor*> zTensors;

   public:
    shared_ptr<GTensor> bt4c = nullptr, delta = nullptr, gate_delta = nullptr, tmpDelta = nullptr, scratch = nullptr, tmpFF1 = nullptr, tmpW = nullptr,
                        tmpGW = nullptr, tmpQout = nullptr, tmpKout = nullptr, residual = nullptr, tmpTernary = nullptr, outL = nullptr;

    GST_TensorBuffer(Fish* hFish, int flag = 0x0);
    // GST_TensorBuffer(size_t size) : total_size_(size), current_offset_(0) {
    //     cudaError_t err = cudaMalloc(&buffer_, total_size_);
    //     if (err != cudaSuccess) {
    //         throw std::runtime_error("Failed to allocate GPU buffer: " + std::string(cudaGetErrorString(err)));
    //     }
    // }

    virtual ~GST_TensorBuffer();

    // Prevent copying
    GST_TensorBuffer(const GST_TensorBuffer&)            = delete;
    GST_TensorBuffer& operator=(const GST_TensorBuffer&) = delete;

    bool Prepare(int flag = 0x0);

    shared_ptr<GTensor> RegisterTensor(Fish* hFish, const string& name_, SHAPE shape, typNUMBER tpD_, int flag = 0x0);
    void* Register(size_t sz_, string d_, void* hData_, int flag = 0x0);
    bool Free(void* hObj, int flag = 0x0);

    virtual bool Clear(int flag = 0x0);

    void Dump(int type, int flag = 0x0) const;

    size_t get_used_memory() const { return current_offset_; }
    size_t get_total_memory() const { return total_size_; }
    size_t get_available_memory() const { return total_size_ - current_offset_; }
};
typedef shared_ptr<GST_TensorBuffer> hTensorBuffer;

void GlobalMemoryInfo(int type, int flag);

extern hTensorBuffer gBUFF;