#include "./llm_c/rand.h"
#include "./llm_c/global_norm.cuh"
#include "../../ggex/GTensor.hpp"

cuTensor::cuTensor(const string&name_,SHAPE shape,typNUMBER tpD_,bool isX,int flag) : GTensor(shape,tpD_,false,flag){
    size_t nEle=size();
    // hFish->InitGensor(nullptr,name,attn,false);
    if(!name_.empty())
        snprintf(name, sizeof(name), "%s",name_.c_str());
    else
        name[0]='\0';
}        

static size_t szMaloc = 0;
bool cuTensor::Alloc(int tpX,int flag){
    if(BIT_TEST(flags,F_NOALLOC))   // For example: operator fusing, memory reuse,rematerialization
        return true;

    assert(szData>0);
    cudaError_t error = cudaMalloc((void**)&data, szData);
    if (error != cudaSuccess) {
        printf("[CUDA Alloc] failed @%s, ERR=%s!\n", name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }    
    cudaCheck(cudaMemset(data, 0, szData));
    size_t sz = szData;
    if(isParam()){
        InitParam(tpX,flag);
        cudaError_t error = cudaMalloc((void**)&grad, szData);      sz+=szData;
        if (error != cudaSuccess) {
            printf("[CUDA Alloc] failed @%s, ERR=%s\n", name, cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }   
    }
    szMaloc += sz;
    if(sz>=100*1.0e6)
        printf("\tcudaMalloc=%gM(%gG)@%s shape=[%ld,%ld,%ld,%ld]\n",sz*1.0f/1.0e6,szMaloc*1.0/1.0e9,name,ne[0],ne[1],ne[2],ne[3]);
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
    size_t nInit = size(1),nB = BPE(type);
    
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
        type = (typNUMBER)gg_->type;
        Alloc( );
        // flags = gg_->flags;     //bug in ggml: don't support flag serialization        
        size_t nB = BPE(type);     // ggml_row_size  ???
        szData = size()*nB;          
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
    size_t sz = ggml_nbytes(gg_);  
    if(type==(typNUMBER)gg_->type) {
        if(sz!=szData){ //bug
            assert(strcmp(name,"token_embd.weight")==0);
            return true;
        }
    };  

#ifdef _TENSOR_G_
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

double tNormOf(const hGTensor tensor,int flag){
    const int block_size = 512;
    float a,*norm2 = (float*)(GTensor::scratch_bt4c->data);
    int num_slices[2] = {1, 1},zero_stage=1,max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    size_t nz=0;
    bool is_first_pass = true;
        //ShardInfo shard ={0, tensor->size()};
    size_t nEle = tensor->size();       nz+=nEle;
    assert(tensor->grad!=nullptr);
     
    if(tensor->grad!=nullptr)     {
        global_norm_squared(norm2, (floatX*)(tensor->grad), nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);            
        global_sum_deterministic(norm2, norm2, max_num_block_sums, main_stream);
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        tensor->gnorm = sqrt(a);
        a = sqrt(a/nz);
    }
    if(tensor->data!=nullptr)     {
        global_norm_squared(norm2, (floatX*)(tensor->data), nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);            
        global_sum_deterministic(norm2, norm2, max_num_block_sums, main_stream);
        cudaCheck(cudaMemcpy(&a, norm2, sizeof(float), cudaMemcpyDeviceToHost));
        tensor->wnorm = sqrt(a);
    }
    
    return tensor->gnorm;
}

hGTensor cuTensor::GetRow(hGTensor hOut,hGTensor token,hGTensor pos,int flag)   {
    /*floatX *out=(floatX*)(hOut->data),*wte=(floatX*)(data),*wpe=pos==nullptr?nullptr : (floatX*)(pos->data);
    // int nCls = shape[1],i;
    const int* inp=(int*)(token->data);
    // assert(isInRange(inp,token->size(),0,nCls));

    encoder_forward(out, inp, wte, wpe, B, T, C, main_stream);
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());*/

    // PrintTensor<floatX>("wte",params.wte,true,Vp,C);        PrintTensor<floatX>("wpe",params.wpe,true,T,C);
    // PrintTensor<int>("inputs",model->inputs,true,B,T);      PrintTensor<floatX>("GetRow",ToX(embed->out),true,B,T,C);
    return hOut;    
}

/*
float RAW_backward_1{
if(config.Fuse_Normal==0)
        lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    else
        lnf = &(lastFFN->norm); 

floatX* dresidual = ToX(GTensor::scratch_btc),*scratchX = ToX(cls->preLogits),*dl_bt4c = ToX(GTensor::scratch_bt4c);   
        floatX* gb = lnf->b==nullptr ? nullptr : ToG(lnf->b);
        cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));
        PrintTensor<floatX>("back of P",ToX(GTensor::scratch_bt4c),true,B,T,C);
        // backward the final layernorm
        SelfAttention *QKV=fish->GetNeuron<SelfAttention>("SelfAttention",L-1),*preQKV=nullptr;
        FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1),*preFFN=nullptr;  
        floatX* residual = ToX(ffn->out);   //acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        if(config.Fuse_Normal==0){
            layernorm_backward(dresidual, ToG(lnf->w), gb, (float*)scratchX, ToX(GTensor::scratch_bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
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
            layernorm_backward(dresidual, ToG(lnf->w), ToG(lnf->b), (float*)scratchX, ToX(GTensor::scratch_bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
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