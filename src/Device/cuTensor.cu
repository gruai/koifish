#include "../CLI_params.hpp"
#include "../ggex/GTensor.hpp"
#include "cutils.cuh"
#include "../kGPT/llmc/rand.h"
#include "../kGPT/llmc/global_norm.cuh"

cuTensor::cuTensor(const string&name_,SHAPE shape,tpDATA tpD_,bool isX,int flag) : GTensor(shape,tpD_,false,flag){
    size_t nEle=size();
    // hFish->InitGensor(nullptr,name,attn,false);
    if(!name_.empty())
        snprintf(name, sizeof(name), "%s",name_.c_str());
    else
        name[0]='\0';
    // if (isParam )        {
    //     SetFlag(GTensor::F_PARAM);
    // }
}        

static size_t szMaloc = 0;
bool cuTensor::Alloc(int tpX,int flag){
    if(BIT_TEST(flags,F_NOALLOC))   // For example: operator fusing, memory reuse,rematerialization
        return true;

    assert(szData>0);
    cudaError_t error = cudaMalloc((void**)&data, szData);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", __FILE__, __LINE__, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }    
    cudaCheck(cudaMemset(data, 0, szData));
    size_t sz = szData;
    if(isParam()){
        InitParam(tpX,flag);
        cudaError_t error = cudaMalloc((void**)&grad, szData);      sz+=szData;
        if (error != cudaSuccess) {
            printf("[CUDA ERROR] at file %s:%d:\n%s\n", __FILE__, __LINE__, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }   
    }
    szMaloc += sz;
    if(sz>=100*1.0e6)
        printf("\tcudaMalloc=%gM(%gG)@%s\n",sz*1.0f/1.0e6,szMaloc*1.0/1.0e9,name);
    return true;
}
bool cuTensor::Free() {
try{
    if(data!=nullptr)       
    {    cudaFreeCheck(&data);      data=nullptr;   }
    if(grad!=nullptr)       
    {    cudaFreeCheck(&grad);      grad=nullptr;   }
}catch(...){
    assert(0);
}
    return true;
}

bool cuTensor::InitParam(int tpX,int flag){
    size_t nElem0 = size(),i;
    size_t nInit = size(1),nB = ggml_type_sizef(type);
    
    if(tpInit>0){
        mt19937_state init_rng;            
        floatX* tmp = new floatX[nInit];
        switch(tpInit){
        case 1:
            for(i=0;i<nInit;i++)        tmp[i]=1; 
            break;
        default:
            // manual_seed(&init_rng, 42);     //cys   only for debug
            float *tmp32 = new float[nInit];
            normal_(tmp32, nInit, 0.0f, 0.02f*residual_scale, &init_rng);
            for(i=0;i<nInit;i++)        tmp[i]=tmp32[i];      //ony for debug    
            delete[] tmp32;        
            break;
        }
        cudaCheck(cudaMemcpy(data, tmp, nInit*nB, cudaMemcpyHostToDevice));
        delete[] tmp;
    } 
         
    return true;
}

/*
   Only for gguf-serialize
*/
bool cuTensor::CopyGG(struct ggml_tensor*gg_,int flag) {
    int i=0;
    assert(gg == nullptr );
    bool isAlloc = data!=nullptr;
    
    if(!isAlloc){    
        memcpy(name,gg_->name,sizeof(char)*GGML_MAX_NAME);
        for(i=0;i<GGML_MAX_DIMS;i++)  {
            shape.push_back(gg_->ne[i]);
            nb[i] = gg_->nb[i];
        }
        type = gg_->type;
        Alloc( );
        // flags = gg_->flags;     //bug in ggml: don't support flag serialization        
        size_t nB = ggml_type_sizef(type);     // ggml_row_size  ???
        szData = size()*nB;          
    }else{
        for(i=0;i<shape.size();i++)  {
            assert(shape[i]==gg_->ne[i]);
            assert(nb[i] == gg_->nb[i]);
        }
    }
    size_t sz   = ggml_nbytes(gg_);  
    assert(sz==szData);  

#ifdef _TENSOR_CUD_
   bool toDevice = SerialGP(gg_->data,nullptr,false,0x0);
   assert(toDevice);
#endif
    return true;
}

hGTensor cuTensor::CrossEntropy( const hGTensor b,int flag ){
    return b;
}

bool cuTensor::SerialGP(void *hostD,void *hostG,bool isToHost,int flag)   {
try{
    if(isToHost){
        cudaCheck(cudaMemcpy(hostD,data, szData, cudaMemcpyDeviceToHost));
        if(hostG!=nullptr){
            assert(grad!=nullptr);
            cudaCheck(cudaMemcpy(hostG,grad, szData, cudaMemcpyDeviceToHost));
        }
    }else{
        cudaCheck(cudaMemcpy(data, hostD, szData, cudaMemcpyHostToDevice));
        if(hostG!=nullptr){
            assert(grad!=nullptr);
            cudaCheck(cudaMemcpy(grad, hostG, szData, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(grad, hostG, szData, cudaMemcpyHostToDevice));
        }
    }
    return true;
}catch(...){
    return false;
}    
    
}

bool cuTensor::OverWrite(hGTensor hGT,bool isSrc,int flag) {    
    size_t nEle = size();
    assert(isSameShape(hGT) && szData>0);   
    if(isSrc) {
        cuTensor *src = dynamic_cast<cuTensor *>(hGT.get());
        if(src==nullptr)    //  Host => Device
            cudaCheck(cudaMemcpy(data, hGT->data, szData, cudaMemcpyHostToDevice));
        else{
            assert(0);
        }
    }else{                  //  Device => Device
        assert(0);
    }
    
    return true;
}


template<class T>
__global__ inline void _norm2_kernel(float* out, const T* data,size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // need to try grid stride looping for more perf later
    if (idx < n) {
        out += data[idx]*data[idx];
    }
}


double tNormOf(const std::vector<hGTensor>& tensors,int flag){
    const int block_size = 512;
    float* grad_norm_squared,a;
    grad_norm_squared = (float*)(GTensor::scratch_bt4c->data);
    double norm = 0.0f;
    int num_slices[2] = {1, 1},zero_stage=1,max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    size_t nz=0;
    bool is_first_pass = true;  //i==0    
    for(auto tensor:tensors){
        //ShardInfo shard ={0, tensor->size()};
        size_t nEle = tensor->size();       nz+=nEle;
        assert(tensor->grad!=nullptr);
        floatX* val = (floatX*)(tensor->grad);        
        // _norm2_kernel<<<dim3(grid_size, 1), block_size, 0, main_stream>>>(grad_norm_squared, val, nEle, nEle);
        global_norm_squared(grad_norm_squared, val, nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);
        is_first_pass = false;
        // PrintTensor<floatX>("tNormOf",val,true,nEle,1);
        // break;
    }
    global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    cudaCheck(cudaMemcpy(&a, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));

    norm = sqrt(a);
    a = sqrt(a/nz);
    return norm;
}