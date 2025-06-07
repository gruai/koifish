#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "./kernel/Operator.cuh"
#include "./kernel/utils.cuh"
#include "../../ggex/GTensor.hpp"
#include "../../Utils/GST_log.hpp" 
#include "../../Utils/GST_rander.hpp"
#include "../../Manifold/Fish.hpp"
// const int block_512 = 512;
huTensor::huTensor(Fish *fish,const string&name_,const SHAPE shape,typNUMBER tpD_,bool isAlloc,int flag) : GTensor(fish,shape,tpD_,false,flag){
    size_t nEle=size();
    if(DEBUG.T_cpu==1){
        flags |= BIT_FLAG::F_HOSTALLOC;
    }else
        flags |= BIT_FLAG::F_GPU;
    // hFish->InitGensor(nullptr,name,attn,false);
    if(!name_.empty())
        snprintf(name, sizeof(name), "%s",name_.c_str());
    else
        name[0]='\0';
    
    if(isAlloc){
        Alloc(0x0,flag);
    }
}        

size_t GTensor::szMaloc = 0;
size_t huTensor::mostMemory(int typ)  const {
    if(BIT_TEST(flags,F_NOALLOC))
        return 0x0;
    // if(BIT_TEST(flags,F_HOSTALLOC))
    //     return 0x0;
    if(hRef!=nullptr){
        return 0x0;
    }
    size_t most = nByte();
    if(isParam() && hFish->isTrain())   {
        most += nByte();                    // grad
        most += sizeof(float)*size()*2;     // gm,gv is float array
    }
    return most;
}
/*
    cudaHostAlloc is a function used to allocate pinned (page-locked) host memory, which can improve data transfer performance between the host (CPU) and device (GPU). Pinned memory allows for faster transfers because it bypasses the operating system's virtual memory system. 
*/
size_t huTensor::Alloc_1(void **dst,bool isZero,size_t sz0,int flag){
    assert(*dst==nullptr);

    bool hostAlloc = BIT_TEST(flags,F_HOSTALLOC);
    cudaError_t error = cudaSuccess;
    size_t szAlloc = sz0==0 ? szData : sz0;     assert(szAlloc>0);
    error = hostAlloc ? cudaHostAlloc(dst, szAlloc,0) : cudaMalloc(dst, szAlloc);    //8420
    // strange behavior of callo
    //data = calloc(szAlloc,1);  sAlloc = "Alloc_c/cu";   //8386
    if (error != cudaSuccess) {
        printf("[CUDA Alloc] failed @%s, ERR=%s!\n", name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }    
    if(isZero)
        cudaCheck(cudaMemset(*dst, 0, szAlloc));
    szMaloc += szAlloc;
    return szAlloc;
}
size_t huTensor::Free_1(void **obj,const string&info){    
    assert(*obj!=nullptr);
    // _INFO("\t%s%s freed@%p(%.3gM)!",name,info.c_str(),*obj,(szData)/1.0e6);   
    if(BIT_TEST(flags,F_HOSTALLOC) )
        cudaFreeHost(*obj); 
    else {
        // cudaFreeCheck(obj);   
        cudaError_t error = cudaFree(*obj);
        if (error != cudaSuccess) {
            // _INFO("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
            _INFO("[CUDA] free failed @\"%s\"! err=%s.\n", name, cudaGetErrorString(error));
            // exit(EXIT_FAILURE);
        }
        *obj = nullptr;
    }
        
    *obj=nullptr;   szMaloc -= szData;  
    
    return szMaloc;
}

bool huTensor::InitParam(int tpX){
    size_t nElem0 = size(),i;
    size_t nInit = size(1),nB = BPE(type);
    bool isTmp = true;
    if(tpInit>0 && tpInit!=SERIALIZE){
        // _INFO("[InitParam]\t%ld-%ld@%s\n",size(),nInit,name);
        mt19937_state init_rng;            
        floatX* tmp = new floatX[nInit];
        switch(tpInit){
        case FIX_1:
            for(i=0;i<nInit;i++)        tmp[i]=1; 
            break;
        default:
#ifdef NDEBUG
            {   CU_normal<floatX>(nInit,(floatX*)data,0.02f*residual_scale);     isTmp = false;  }
#else
                // manual_seed(&init_rng, 42);     //cys   only for debug
                float *tmp32 = new float[nInit];
                assert(nInit<INT_MAX);
                normal_(tmp32, nInit, 0.0f, 0.02f*residual_scale, &init_rng);
                for(i=0;i<nInit;i++)        tmp[i]=tmp32[i];      //ony for debug    
                delete[] tmp32;   
#endif     
            break;
        }
        if(isTmp){
            cudaCheck(cudaMemcpy(data, tmp, nInit*nB, cudaMemcpyHostToDevice));  
        }  
        delete[] tmp;          
        // Print(name,0,-1);
    } 
         
    return true;
}

/*
   Only for gguf-serialize
*/
bool huTensor::CopyGG(struct ggml_tensor*gg_,int flag) {
#ifdef __USE_GGML__
    int i=0;
    assert(gg == nullptr );
    bool isAlloc = data!=nullptr;
    void *src = gg_->data;
    if(!isAlloc){    
        memcpy(name,gg_->name,sizeof(char)*GGML_MAX_NAME);
        for(i=0;i<GGML_MAX_DIMS;i++)  {
            shape.push_back(gg_->ne[i]);
            nb[i] = gg_->nb[i];
        }
        type = (typNUMBER)gg_->type;
        Alloc( );
        // flags = gg_->flags;     //bug in ggml: don't support flag serialization        
        double fnB = BPE(type);     // ggml_row_size  ???
        szData = size()*fnB;          
    }else{
        for(i=0;i<shape.size();i++)  {
            if(BIT_TEST(flags,F_PADDED))
                assert(shape[i]>=gg_->ne[i]);
            else
                assert(shape[i]==gg_->ne[i]);
            if(type==(typNUMBER)gg_->type)
                assert(nb[i] == gg_->nb[i]);
        }
    }
    size_t szSrc = ggml_nbytes(gg_);  
    if(type==(typNUMBER)gg_->type) {
        if(szSrc!=szData){ 
            if(BIT_TEST(flags,F_PADDED)){
                assert(strcmp(name,"token_embd.weight")==0 && szSrc<=szData);
            }else{
                assert(0);
            }            
        }
    };  

#ifdef _TENSOR_G_
   bool toDevice = SerialGP(src,nullptr,szSrc,false,0x0);
   assert(toDevice);
#endif
#endif
    // if(src!=data)       delete[] src;
    return true;
}

//  From:   https://stackoverflow.com/questions/57948643/whats-a-good-way-to-zero-out-cudamallocd-data
/*__global__ void clear_scratch_space_kernel(int * data, int blocks, int threads) {
    // BOZO: change the code to just error out if we're any of the border cases below
    const int idx = blockIdx.x * threads + threadIdx.x;
    long size = sizeof(int) * COUNT;
    long size_of_typical_chunk = round_up(size / (blocks * threads), GPU_CACHE_LINE_SIZE_IN_BYTES);
    // Due to truncation, the threads at the end won't have anything to do.  This is a little sloppy but costs us
    // hardly anything in performance, so we do the simpler thing.

    long this_threads_offset = idx * size_of_typical_chunk;
    if (this_threads_offset > SIZE_OF_DATA) {
        return;
    }

    long size_of_this_threads_chunk;
    if (this_threads_offset + size_of_typical_chunk >= SIZE_OF_DATA) {
        // We are the last thread, so we do a partial write
        size_of_this_threads_chunk = SIZE_OF_DATA - this_threads_offset;
    } else {
        size_of_this_threads_chunk = size_of_typical_chunk;
    }
    void * starting_address = reinterpret_cast<void *>(reinterpret_cast<char *>(data) + this_threads_offset);
    memset((void *) starting_address, 0, size_of_this_threads_chunk);
}
__global__ void clear_scratch_space_with_coalesced_writes_kernel(int * data, int blocks, int threads) {
    if (COUNT % (blocks * threads) != 0) {
        printf("Adjust the SIZE_OF_DATA so it's divisible by the number of (blocks * threads)\n");
    }
    const long count_of_ints_in_each_blocks_chunk = COUNT / blocks;

    int block = blockIdx.x;
    int thread = threadIdx.x;

    const long rounds_needed = count_of_ints_in_each_blocks_chunk / threads;

    const long this_blocks_starting_offset = block * count_of_ints_in_each_blocks_chunk;

    //printf("Clearing %li ints starting at offset %li\n", count_of_ints_in_each_blocks_chunk, this_blocks_starting_offset);

    int * this_threads_base_pointer = &data[this_blocks_starting_offset + thread];
    for (int round = 0; round < rounds_needed; ++round) {
        *this_threads_base_pointer = 0;
        this_threads_base_pointer += threads;
    }
}
void set_gpu_data_to_ones(int * data_on_gpu) {
    cudaMemset(data_on_gpu, 1, SIZE_OF_DATA);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
void check_gpu_data_is_zeroes(int * data_on_gpu, char * data_on_cpu) {
    cudaMemcpy(data_on_cpu, data_on_gpu, SIZE_OF_DATA, cudaMemcpyDeviceToHost);
    for (long i = 0; i < SIZE_OF_DATA; ++i) {
        if (data_on_cpu[i] != 0) {
            printf("Failed to zero-out byte offset %i in the data\n", i);
        }
    }
}*/

void huTensor::Zero()   {
    assert(data!=nullptr);
    //  https://stackoverflow.com/questions/57948643/whats-a-good-way-to-zero-out-cudamallocd-data
    cudaCheck(cudaMemset(data, 0, szData));
    if(grad!=nullptr){
        ZeroGrad();
    }
}
void huTensor::ZeroGrad()   {
    assert(grad!=nullptr);
    cudaCheck(cudaMemset(grad, 0, szData));
    // cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
}
bool cuClearGrad(std::vector<hGTensor> tensors,int flag){
    for(auto tensor:tensors){
        if(tensor->isRefer())
            continue;
        cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
    }
    
    return true;
}

bool D2H(void *dev,void *host,size_t szData,int flag=0x0){
try{
    assert(host!=nullptr && dev!=nullptr);        
    cudaCheck(cudaMemcpy(host,dev, szData, cudaMemcpyDeviceToHost));    
    return true;
}catch(...){
    return false;
}        
}

bool huTensor::SerialData(const string&info,void *host,bool isToHost,int flag) {
try{
    if(host==nullptr){
        assert(host_data!=nullptr);
        host = host_data;
    }
    if(isToHost){        
        //cudaCheck(cudaMemcpyAsync(host,data, szData, cudaMemcpyDeviceToHost));
         cudaCheck(cudaMemcpy(host,data, szData, cudaMemcpyDeviceToHost));
    }else{
        //cudaCheck(cudaMemcpyAsync(data, host,szData, cudaMemcpyHostToDevice));        
         cudaCheck(cudaMemcpy(data, host,szData, cudaMemcpyHostToDevice));   
    }
    if(flag<0){    
        char buf[1024];
        sprintf(buf,"%s:%s@%s",info.c_str(),isToHost?"SAVE":"LOAD",name);
        Print(buf,0,-1);
    }
    
    return true;
}catch(...){
    return false;
}    
}
//  this <=> Y
bool huTensor::SerialGP(void *yD,void *yG,size_t szY,bool isToY,int flag)   {
try{
    if(isToY){
        assert(szY>=szData);
        cudaCheck(cudaMemcpy(yD,data, szY, cudaMemcpyDeviceToHost));
        if(yG!=nullptr){
            assert(grad!=nullptr);
            cudaCheck(cudaMemcpy(yG,grad, szY, cudaMemcpyDeviceToHost));
        }
    }else{
        assert(szY<=szData);
        cudaCheck(cudaMemcpy(data, yD, szY, cudaMemcpyHostToDevice));
        if(yG!=nullptr){
            assert(grad!=nullptr);
            cudaCheck(cudaMemcpy(grad, yG, szY, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(grad, yG, szY, cudaMemcpyHostToDevice));
        }
    }
    return true;
}catch(...){
    return false;
}    
    
}

bool huTensor::OverWrite(hGTensor hGT,bool isSrc,int flag) {    
    size_t nEle = size();
    assert(isSameShape(hGT) && szData>0);   
    if(isSrc) {
        huTensor *src = dynamic_cast<huTensor *>(hGT.get());
        if(src==nullptr)    //  hGT => this
            cudaCheck(cudaMemcpy(data, hGT->data, szData, cudaMemcpyHostToDevice));
        else{
            cudaCheck(cudaMemcpy(data, hGT->data, szData, cudaMemcpyDeviceToDevice));
        }
    }else{                  //  this => hGT
        assert(0);
    }
    
    return true;
}


hGTensor huTensor::CrossEntropy( const hGTensor b,int flag ){
    return b;
}

// Helper function determines the maximum number of block sums
inline int get_max_num_block_sums(int* num_slices_all, int numel) {
    // NOTE: this needs to be kept in sync with `global_norm_squared` below.
    const int block_size = 512;
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
    assert(grid_size > 0);
    int max_num_block_sums = 0;
    for (int i = 0; i < numel; i++) {
        int num_slices = num_slices_all[i];
        const int gx = CEIL_DIV(grid_size, num_slices);
        const int gy = num_slices;
        max_num_block_sums = max(max_num_block_sums, gx * gy);
    }

    return max_num_block_sums;
}
template<class T>
__device__ inline float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i];
    }
    // block-level reduce
    return blockReduce<warpReduceSum>(accumulator);
}
template<class T>
__global__ static void global_norm_squared_kernel(float* out, const T* data, size_t count, ptrdiff_t stride) {
    float block_sum = global_norm_squared_for_range(data + blockIdx.y * stride, count);
    // each block accumulates its partial sum to out[out_index]
    // we want to avoid using atomic add here so we combine this kernel with another kernel call
    // that sums up the partial block sums
    if(threadIdx.x == 0) {
        size_t out_index = blockIdx.y * gridDim.x + blockIdx.x;
        out[out_index] = out[out_index] + block_sum;
    }
}
template<typename T>
inline float global_norm_squared(float* norm2, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
    constexpr int block_size = 512; //256 may be better for shared memory of CU_x2_
    // launch just enough blocks to fill the grid. deliberately no DIV_CEIL.
    // having one block less than possible is a tiny performance hit, having
    // one block too many is catastrophic, since it only can start once all the other
    // blocks finish. anyway, I think cuda_threads_per_SM should be a multiple of 512
    // on all gpus, so the division really is going to be exact.
    auto now = GST_us();
    float a = 0, b = 0;
    if(1)   {
        const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size;
        assert(grid_size > 0);      // gives a better error than letting the call below fail
        const int gx = CEIL_DIV(grid_size, num_slices),gy = num_slices;
        assert(gx * gy < 1024);  // we want to later accumulate the block sums in a single block        
        if (reset) {
            cudaCheck(cudaMemsetAsync(norm2, 0, max_num_block_sums * sizeof(float), stream));
        }
        global_norm_squared_kernel<<<dim3(gx, gy), block_size, 0, stream>>>(norm2, values, count, stride);    
        cudaCheck(cudaGetLastError());
        global_sum_deterministic(norm2, norm2, max_num_block_sums, main_stream);
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
    }else{
        cudaCheck(cudaMemsetAsync(norm2, 0, sizeof(float), stream));
        CU_x2_<T,block_size><<<CEIL_DIV(count,block_size), block_size, 0, main_stream>>>(norm2, values, count);
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));        
    }
    SUM::tX1 += GST_us()-now;
    return a;
}

double tNormOf(const std::vector<hGTensor>& tensors,int flag){    
    float* grad_norm_squared,a,a_pre=0.0;
    grad_norm_squared = (float*)(GTensor::bt4c->data);
    double norm = 0.0f;
    int num_slices[2] = {1, 1},max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    size_t nz=0;
    bool is_first_pass = true;  //i==0    
    for(auto tensor:tensors){     
        assert(0);      // Deprecated   
        /*//ShardInfo shard ={0, tensor->size()};
        size_t nEle = tensor->size();       nz+=nEle;
        assert(tensor->grad!=nullptr);
        floatX* val = (floatX*)(tensor->grad);   
        global_norm_squared(grad_norm_squared, val, nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);
        if(DEBUG.check_tensor_norm){
            cudaCheck(cudaMemcpy(&a, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
            assert(a>=a_pre); 
            tensor->gnorm = sqrt(a-a_pre);           a_pre = a;            
        }        
        is_first_pass = false;*/
        // PrintTensor<floatX>("tNormOf",val,true,nEle,1);
        // break;
    }
    global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    cudaCheck(cudaMemcpy(&a, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));

    norm = sqrt(a);
    a = sqrt(a/nz);
    return norm;
}

//  TODO: Fuse to sgdv_update
double tNormOf(const hGTensor tensor,int flag){
    float a,*norm2 = (float*)(GTensor::bt4c->data);
    int num_slices[2] = {1, 1},max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    size_t nz=0;
    bool is_first_pass = true;
        //ShardInfo shard ={0, tensor->size()};
    size_t nEle = tensor->size();       nz+=nEle;
    assert(tensor->grad!=nullptr);
    
    if(tensor->grad!=nullptr)     {
        int block_size=1024,grid_size = deviceProp.maxThreadsPerMultiProcessor*deviceProp.multiProcessorCount/block_size;        
        a = global_norm_squared(norm2, (floatX*)(tensor->grad), nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);            
        // global_sum_deterministic(norm2, norm2, max_num_block_sums, main_stream);
        // cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        tensor->gnorm = sqrt(a);
        a = sqrt(a/nz);
    }
    if(tensor->data!=nullptr)     {
        a = global_norm_squared(norm2, (floatX*)(tensor->data), nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);            
        // global_sum_deterministic(norm2, norm2, max_num_block_sums, main_stream);
        // cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        tensor->wnorm = sqrt(a);
    }
    
    return tensor->gnorm;
}

bool GTensor::ToTernary(int flag)  { 
    float *xxx = TO<float>(GTensor::scratch);
    Print("Before",0,-1);
    size_t count = size();
    switch(type){
        case typNUMBER::T_SIGN:
            break;
        case typNUMBER::F16:
            CU_ternary_<<<CEIL_DIV(count,CU_T4B_SMALL), CU_T4B_SMALL, 0, main_stream>>>(xxx,(__nv_bfloat16*)data, count);
            break;
        case typNUMBER::F8E5M2:
            CU_ternary_<<<CEIL_DIV(count,CU_T4B_SMALL), CU_T4B_SMALL, 0, main_stream>>>(xxx,(__nv_fp8_e5m2*)data, count);
            break;
        default:
            break;
    }
    D2H(xxx,info,sizeof(info));
    Print(name,0,-1);
    // type = typNUMBER::T_SIGN;
    return true;
}

hGTensor huTensor::GetRow(hGTensor hOut,hGTensor token,hGTensor pos,int flag)   {

    return hOut;    
}

void huTensor::Print(const string& title, int x, int flag,size_t nEle)   const {
    bool isDevice = !isAtHost();     
    switch(type){
    case typNUMBER::F8E5M2:
    //    PrintTensor<__nv_fp8_e5m2>(title.c_str(),(__nv_fp8_e5m2 *)data, isDevice,ne[0],ne[1],ne[2],ne[3],flag);
       PrintTensor<f8e5m2_t>(title.c_str(),(f8e5m2_t *)data, isDevice,ne[0],ne[1],ne[2],ne[3],flag);
       break;
    default:
       GTensor::Print(title,x,flag,nEle);
       break;
    } 
}
 
huTensor::~huTensor()  {
    Free();

}

/*
float RAW_backward_1{
if(config.Fuse_Normal==0)
        lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    else
        lnf = &(lastFFN->norm); 

floatX* dresidual = ToX(GTensor::delta),*scratchX = ToX(cls->preLogits),*dl_bt4c = ToX(GTensor::bt4c);   
        floatX* gb = lnf->b==nullptr ? nullptr : ToG(lnf->b);
        cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));
        PrintTensor<floatX>("back of P",ToX(GTensor::bt4c),true,B,T,C);
        // backward the final layernorm
        SelfAttention *QKV=fish->GetNeuron<SelfAttention>("SelfAttention",L-1),*preQKV=nullptr;
        FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1),*preFFN=nullptr;  
        floatX* residual = ToX(ffn->out);   //acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        if(config.Fuse_Normal==0){
            layernorm_backward(dresidual, ToG(lnf->w), gb, (float*)scratchX, ToX(GTensor::bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
            PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
        }
        // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
        // scratch for backward computations
        floatX* dl_btc = ToX(ffn->out); //residual;
        for (int l = L-1; l >= 0; l--) {
            NvtxRange layer_range("Layer", l);
            QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
            ffn = fish->GetNeuron<FFN>("FFN",l);        preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1); 
            residual = l == 0 ? ToX(embed->out) : ToX(preFFN->out);   //acts.residual3 + (l-1) * B * T * C;
            ffn->residual=dl_btc;     ffn->lastQKV=QKV;    
            QKV->dl_btc=dl_btc;   
            if(config.Fuse_Normal==0){
                LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
                ffn->FUSE_cuda(QKV->out,scratchX, hNorm, 0x0);    
                QKV->FUSE_cuda(QKV->norm.out,residual,&(ffn->norm),(float*)scratchX,0x0);
            }else{
                LayerNormal *hNorm = l>0 ? &(fish->GetNeuron<SelfAttention>("FFN",l-1)->norm) : lnf;
                ffn->FUSE_cuda(QKV->out,scratchX, &(ffn->norm), 0x0);    
                QKV->FUSE_cuda(QKV->norm.out,residual,&(QKV->norm),(float*)scratchX,0x0);
            }         
        }
        if(config.Fuse_Normal==1){
            lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
            layernorm_backward(dresidual, ToG(lnf->w), ToG(lnf->b), (float*)scratchX, ToX(GTensor::bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
            PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
        }
        int *input = TO<int>(fish->Input());
        if (bucket_info == NULL) {      //grads_memory
            // NvtxRange rng("InitGrads");
            size_t num_c_groups = CEIL_DIV(C, (WARP_SIZE * x128::size));
            assert((size_t)(B * T) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
            workload_indices = (int*)mallocCheck(sizeof(int) * B * T * num_c_groups);
            bucket_info = (int4*)mallocCheck(sizeof(int4) * B * T * num_c_groups);
        }
        encoder_backward(ToG(embed->w), ToG(embed->b), scratchX, workload_indices, bucket_info,dresidual, input, hostInput, B, T, C, random_u32(&rng_state), main_stream);
}
*/