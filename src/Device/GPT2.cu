/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 * 
 *  \brief Optimizer
 *  \author Yingshi Chen
 */

#include "../CLI_params.hpp"
#include "../ggex/GTensor.hpp"
#include "../g_stddef.hpp" 
#include "../kGPT/llmc/sampler.h"
// #include "cutils.cuh"
// #include "../kGPT/llmc/cublas_common.h"
// 
#include "../kGPT/llmc/layernorm.cuh"
#include "../kGPT/llmc/encoder.cuh"
// #include "../kGPT/llmc/fused_classifier.cuh"
// #include "../kGPT/llmc/global_norm.cuh"
#include "../kGPT/llmc/adamw.cuh"
#define ENABLE_CUDNN

#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "../kGPT/llmc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "../kGPT/llmc/attention.cuh"
#endif
#include "../Manifold/Neuron.hpp"
#include "../Manifold/Fish.hpp"
// #include "../kGPT/llmc/mfu.h"
#define NOMINMAX
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
extern int tpFuseCu;
extern cudaStream_t main_stream;

enum class DType : uint8_t {
    FP32, FP16, BF16
};

size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}
DType dtype_of(float* f)        { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f)        { return DType::FP16; }

__global__ void __launch_bounds__(1024) test_print_kernel(half *__restrict__ arr){
    // printf("test kernel\n");
    if (((int)blockIdx.x == 0) && ((int)threadIdx.x == 0))    {
        // arr[0] = ((half)(2));
        __syncthreads();
        printf("%f ", __half2float(arr[0]));

    }
}
static int V=50257,Vp=50304,recompute=1;
static unsigned long long rng_state=0;
float* accumulated_mean_loss=nullptr;
bool cuClear(std::vector<hGTensor> tensors,int flag){
    for(auto tensor:tensors){
        cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
    }
    
    return true;
}

typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;
static int micro_step=0,*workload_indices=nullptr;
static int4 *bucket_info=nullptr;
float RAW_backward(Fish *fish,const int* hostInput, int grad_accum_steps,bool isOnlyEvaluate,int flag) {    
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    size_t B=GTensor::B, T=GTensor::T;    
    auto hparams = fish->hparams;
    const size_t L = hparams.nLayer(),NH = hparams.n_head(),C = hparams.n_embd;    
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if (micro_step == 0) {
        cudaCheck(cudaMemsetAsync(TO<float>(cls->out), 0,B*T* sizeof(float), main_stream));
    }

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);    
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    cls->FUSE_cuda(lnf->out,embed->w,0x0);
    
    if(isOnlyEvaluate){

    }else{    // backward pass: go in the reverse order of the forward pass, and call backward() functions
        // reset residual stream gradients (put here to work with gradient accumulation)
        floatX* dresidual = ToX(GTensor::scratch_btc),*scratchX = ToX(cls->preLogits),*dl_bt4c = ToX(GTensor::scratch_bt4c);   
        cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));
        PrintTensor<floatX>("back of P",ToX(GTensor::scratch_bt4c),true,B,T,C);
        // backward the final layernorm
        SelfAttention *QKV=fish->GetNeuron<SelfAttention>("SelfAttention",L-1),*preQKV=nullptr;
        FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1),*preFFN=nullptr;  
        floatX* residual = ToX(ffn->out);   //acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        layernorm_backward(dresidual, ToG(lnf->w), ToG(lnf->b), (float*)scratchX, ToX(GTensor::scratch_bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
        PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
        // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
        // scratch for backward computations
        floatX* dl_btc = ToX(ffn->out); //residual;

        // now backward all the layers
        for (int l = L-1; l >= 0; l--) {
            NvtxRange layer_range("Layer", l);
            QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
            ffn = fish->GetNeuron<FFN>("FFN",l);        preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1); 
            residual = l == 0 ? ToX(embed->out) : ToX(preFFN->out);   //acts.residual3 + (l-1) * B * T * C;
            // floatX* dl_bt4c = ToX(GTensor::scratch_bt4c),*l_residual2 = ToX(QKV->out);   
            LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
            ffn->dl_btc=dl_btc;     ffn->lastQKV=QKV;       
            ffn->FUSE_cuda(QKV->out,scratchX, nullptr,hNorm, 0x0);            
            QKV->dl_btc=dl_btc;     
            QKV->FUSE_cuda(QKV->norm.out,residual,ffn->norm,(float*)scratchX,0x0);   
        }
        int *input = TO<int>(fish->Input());
        if (bucket_info == NULL) {      //grads_memory
            // NvtxRange rng("InitGrads");
            size_t num_c_groups = CEIL_DIV(C, (WARP_SIZE * x128::size));
            assert((size_t)(GTensor::B * GTensor::T) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
            workload_indices = (int*)mallocCheck(sizeof(int) * GTensor::B * GTensor::T * num_c_groups);
            bucket_info = (int4*)mallocCheck(sizeof(int4) * GTensor::B * GTensor::T * num_c_groups);
        }
        encoder_backward(ToG(embed->w), ToG(embed->b), scratchX, workload_indices, bucket_info,dresidual, input, hostInput, B, T, C, random_u32(&rng_state), main_stream);
        // g_dump_level = 0;                 
        // PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
        // g_dump_level = 1;
        // PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
    }

    // Aggregate all gradients that are not part of the transformer blocks5
    if(tpFuseCu==0 && last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(accumulated_mean_loss, TO<float>(cls->out), B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(accumulated_mean_loss, accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&cls->mean_loss, accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        cls->mean_loss /= B*T*grad_accum_steps;
    }
    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        // cls->mean_loss /= B*T*grad_accum_steps;
        micro_step = 0;
    } else {
        cls->mean_loss = -1.f; // no loss available yet
        micro_step++;
    }
    
    return cls->mean_loss;
}   

void RAW_forward(Fish *fish,int flag) {
    // if(model==nullptr){
    //     model = _init_model(fish,flag);
    // }
    NVTX_RANGE_FN();
    auto hparams = fish->hparams;
    size_t B=GTensor::B, T=GTensor::T;    
    const size_t L = hparams.nLayer(),NH = hparams.n_head(),C = hparams.n_embd;       
    // forward pass
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);
    embed->Interact(nullptr,fish->Input());     //acts.encoded=ToX(embed->out);
    // encoder_forward(acts.encoded, fish->Input(), params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]
    SelfAttention *QKV0 = fish->GetNeuron<SelfAttention>("SelfAttention",0);
    QKV0->norm.Interact(nullptr,embed->out,0x0);      // first layernorm isn't fused
    SelfAttention *QKV=nullptr,*lastQKV=nullptr;
    FFN *ffn=nullptr,*lastFFN=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);
        floatX* residual = l == 0 ? ToX(embed->out) : ToX(lastFFN->out); //floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;  
        floatX* l_ln2 = (recompute < 2) ? ToX(ffn->norm.out) : nullptr;      
        QKV->FUSE_cuda(QKV->norm.out,residual,ffn->norm,nullptr,0x0);   
        LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->FUSE_cuda(QKV->out,l_ln2, ToX(QKV->out),hNorm, 0x0);
        lastFFN = ffn;  lastQKV = QKV;
    }    
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if(tpFuseCu==1)
        cls->FUSE_cuda(lnf->out,embed->w,flag);
    else
        ;//cls->preLogits = lnf->out*embed->w;   //matmul_forward_cublaslt(ToX(cls->preLogits), ToX(lnf->out), ToX(embed->w), NULL, B, T, C, Vp, main_stream);
    PrintTensor<floatX>("output",ToX(cls->preLogits),true,B,T,C);
    cudaCheck(cudaDeviceSynchronize());
}


static void *grads_memory=nullptr;
static float *m_memory=nullptr,*v_memory=nullptr,*master_weights=nullptr;
int RAW_update(std::vector<hGTensor>& tensors,ADAM_params_ adam,float learning_rate,float& grad_norm,int iter,int alg,int flag) {
    grad_norm = flag==0x10002 ? 1.0e6 : tNormOf(tensors,0x0);
    float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
    float beta1=adam.beta1, beta2=adam.beta2, eps=adam.eps;
    float weight_decay=adam.decay*adam.alpha;
    NVTX_RANGE_FN();
    size_t shard_num_parameters = adam.n_parameters,np=0;
    int num_slices = 1;
    // lazily allocate m,v memory and master weights (usually on the first iteration)
    if (m_memory == NULL) {
        NvtxRange rng("InitOpt");
        printf("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
        printf("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
        cudaCheck(cudaMalloc((void**)&m_memory, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&v_memory, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(m_memory, 0, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(v_memory, 0, shard_num_parameters * sizeof(float)));
    }

    bool init_master_weights = false;
    if (alg == 1 && master_weights == NULL) {
        printf("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        cudaCheck(cudaMalloc((void**)&master_weights, shard_num_parameters * sizeof(float)));
        init_master_weights = true;
    }

    // for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {        // generate a unique seed for each tensor
    for(auto tensor:tensors)    {
        unsigned int seed = random_u32(&rng_state);
        const char*name = tensor->name;
        ShardInfo shard = {0,tensor->size()};
        float wd = weight_decay;            // we only want to weight decay the 2D tensors and leave all 1D tensors alone
        if(tensor->shape.size()==1)
            wd = 0;
        // ptrdiff_t local_offset_full=0,local_offset_partial=tensor->offset;
        floatX* param_ptr = ToX(tensor);    //(floatX*)params_memory + local_offset_full;
        floatX* grad_ptr = ToG(tensor);     //(floatX*)grads_memory + local_offset_full;
        
        ptrdiff_t opt_state_offset = np; //multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;        
        np += tensor->size();
        float* m_ptr = m_memory + opt_state_offset;
        float* v_ptr = v_memory + opt_state_offset;
        float* master_ptr = NULL;   
        if (master_weights != NULL) { master_ptr = master_weights + opt_state_offset; }        
        if(init_master_weights) {
            size_t grid_size = CEIL_DIV(shard.size, 512);
            copy_and_cast_kernel<<<dim3(grid_size, num_slices), 512, 0, main_stream>>>(master_ptr, param_ptr, shard.size,shard.size, shard.size);
            cudaCheck(cudaGetLastError());
            // cudaCheck(cudaDeviceSynchronize());
        }
          
        if(flag!=0x10001){  //some debug
            adamw_update(param_ptr, master_ptr, grad_ptr,m_ptr, v_ptr,
                        shard.size, shard.size, shard.size, shard.size, num_slices,      //num_parameters,ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices,
                        learning_rate,beta1, beta2, iter, eps, wd, grad_scale, seed, main_stream);
        }
        cudaCheck(cudaGetLastError());
    }
    assert(np==adam.n_parameters);
    cudaCheck(cudaDeviceSynchronize());
    return 0x0;
}

