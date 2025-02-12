/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief CUDA&CUDNN
 *  \author Yingshi Chen
 */

#include "../ggex/GTensor.hpp"
#include "./EDevice.hpp"
#include "../kGPT/llmc/cuda_common.h"
// #include "../kGPT/llmc/cuda_utils.cuh"

#include "ggml-cuda.h"
#include "ggml-sycl.h"
#include "ggml-alloc.h"








bool InitCUDNN(const CLI_params&hparams,int flag);
bool EDGE_DEVICES::InitGPU(const CLI_params&hparams,int flag){
    string sTp = hparams.KV({"train","device"},"");
#ifdef __USE_CUDA__
    if(!InitCUDNN(hparams,flag))
        return false;

    /*int nGPU = ggml_backend_cuda_get_device_count();     //  ggml_cuda_init: found 1 CUDA devices:    
    for (int device = 0; device < nGPU; ++device) {
        ggml_backend_t backend = ggml_backend_cuda_init(device);
        if (backend == nullptr) {
            _ERROR("%s: failed to initialize CUDA%d backend\n", __func__, device);
        }
        if(sTp=="onlycpu")
            continue;
        workers.push_back(backend);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend));
        // char *guid = (char*)(backend->guid);
        _INFO("Fish::%s init CUDA backend @%p\n", __func__, backend);
    }*/
    
#endif
    return true;
}



//  https://stackoverflow.com/questions/16468440/how-to-split-class-definition-between-multiple-cpp-and-cu-files
hGensor TENSO(void* ctx0,int typ,SHAPE shape,int flag,const string&name ) {
    auto type = (GTensor::tpDATA)(typ);
    hGensor hT = std::make_shared<cuTensor>(name,shape,type,true,flag);
    return hT;    
}
