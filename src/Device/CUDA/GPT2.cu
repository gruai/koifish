/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 * 
 *  \brief Optimizer
 *  \author Yingshi Chen
 */

#include "./sampler.h"
#include "./layernorm.cuh"
#include "./encoder.cuh"
#include "./cuda_utils.cuh"
#include "./adamw.cuh"
#define ENABLE_CUDNN

#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "./cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "./attention.cuh"
#endif
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Optimizer.hpp"
// #include "./mfu.h"
#define NOMINMAX
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
extern int tpFuseCu;
extern cudaStream_t main_stream;

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
static int recompute=1;     //  int V=50257,Vp=50304,
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

template <typename Tp, typename Tg>
__device__ void sgd_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    float grad = grad_scale * (float)grads_memory[idx];    
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    float param = old_param - ( learning_rate * grad  + weight_decay * old_param);
    // stochastic_rounding(param, &params_memory[idx], seed);
    params_memory[idx] = param;
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
__global__ void sgd_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    sgd_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,   
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp, typename Tg>
__device__ void sgdv_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    float grad = grad_scale * (float)grads_memory[idx];
    float v = v_memory[idx];
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;    
    v /= beta2_correction;  // v_hat    
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    float param = old_param - (learning_rate * (grad / (sqrtf(v) + eps) + weight_decay * old_param));
    // stochastic_rounding(param, &params_memory[idx], seed);
    params_memory[idx] = param;
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg>
__global__ void sgdv_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    sgdv_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,                 
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp, typename Tg>
void adamw_core(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    if(m_memory==nullptr){
        if(v_memory==nullptr){
            sgd_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                num_parameters, w_stride, g_stride, s_stride,learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,grad_scale, seed);
        }else{
            sgdv_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                v_memory, num_parameters, w_stride, g_stride, s_stride,learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,grad_scale, seed);
        }
    }else   {
        adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                         m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                         learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                         grad_scale, seed);
    }
    cudaCheck(cudaGetLastError());
}

static int micro_step=0,*workload_indices=nullptr;
static int4 *bucket_info=nullptr;
float RAW_backward(Fish *fish,const int* hostInput, int grad_accum_steps,bool isOnlyEvaluate,int flag) {    
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    size_t B=GTensor::B, T=GTensor::T;    
    auto config = fish->config;
    const size_t L = config.nLayer(),NH = config.n_head(),C = config.n_embd;    
    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if (micro_step == 0) {
        cudaCheck(cudaMemsetAsync(TO<float>(cls->out), 0,B*T* sizeof(float), main_stream));
    }

    NvtxRange classifier_and_loss_range("classifier_and_loss");
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);    
    LayerNormal* lnf = nullptr;
    FFN *lastFFN=fish->GetNeuron<FFN>("FFN",L-1);
    if(config.Fuse_Normal==0)
        lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    else
        lnf = &(lastFFN->norm); 
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
        if(config.Fuse_Normal==0){
            layernorm_backward(dresidual, ToG(lnf->w), ToG(lnf->b), (float*)scratchX, ToX(GTensor::scratch_bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
            PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
        }
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
    NVTX_RANGE_FN();
    auto config = fish->config;
    size_t B=GTensor::B, T=GTensor::T;    
    const size_t L = config.nLayer(),NH = config.n_head(),C = config.n_embd;       
    hGensor out_= nullptr;
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);
    embed->Interact(nullptr,fish->Input());     //acts.encoded=ToX(embed->out);
    // encoder_forward(acts.encoded, fish->Input(), params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]
    SelfAttention *QKV0 = fish->GetNeuron<SelfAttention>("SelfAttention",0);
    if(config.Fuse_Normal==0){
        QKV0->norm.Interact(nullptr,embed->out,0x0);
    }else
        lnf->Interact(nullptr,embed->out,0x0);      //
    SelfAttention *QKV=nullptr,*lastQKV=nullptr;
    FFN *ffn=nullptr,*lastFFN=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);            //ffn->out = GTensor::scratch_btc;
        if(config.Fuse_Normal==0){
            floatX* residual = l == 0 ? ToX(embed->out) : ToX(lastFFN->out);     
            QKV->FUSE_cuda(QKV->norm.out,residual,&(ffn->norm),nullptr,0x0);   
            LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
            ffn->FUSE_cuda(QKV->out,nullptr, hNorm, 0x0);
            out_ = hNorm->out;
        }else{
            floatX* residual = l == 0 ? ToX(embed->out) : ToX(lastFFN->out);    
            LayerNormal *hNorm = l==0 ? lnf : &(lastFFN->norm); 
            QKV->FUSE_cuda(hNorm->out,residual,&(QKV->norm),nullptr,0x0);   
            ffn->FUSE_cuda(QKV->out,nullptr, &(ffn->norm), 0x0);
            out_ = ffn->norm.out;
        }        
        // ffn->norm.out->PrintX<floatX>("inp1",0,-1);        
        lastFFN = ffn;  lastQKV = QKV;
    }    
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if(tpFuseCu==1)
        cls->FUSE_cuda(out_,embed->w,flag); //lnf->out,
    else
        ;//cls->preLogits = lnf->out*embed->w;   //matmul_forward_cublaslt(ToX(cls->preLogits), ToX(lnf->out), ToX(embed->w), NULL, B, T, C, Vp, main_stream);
    PrintTensor<floatX>("output",ToX(cls->preLogits),true,B,T,C);
    // ffn->norm.out->PrintX<floatX>("inp1",0,-1); 
    cudaCheck(cudaDeviceSynchronize());
}

static void *grads_memory=nullptr;
static float *m_memory=nullptr,*v_memory=nullptr,*master_weights=nullptr;

void Optimizer::ClearCUDA(int flag){

}
void Optimizer::InitCUDA(int flag){
    ADAM_params_ adam = TrainParams().adam;
    GD_METHOD tpCurGD = tpGD;

    int num_slices = 1;
    if (m_memory == NULL) {
        NvtxRange rng("InitOpt");
        
        if(tpGD!=SGD){
            _INFO("Optimizer \t cudaMalloc=%zu MiB for v\n", (adam.n_parameters * sizeof(float)) >> 20);
            cudaCheck(cudaMalloc((void**)&v_memory, adam.n_parameters * sizeof(float)));
            cudaCheck(cudaMemset(v_memory, 0, adam.n_parameters * sizeof(float)));            
        }  
    }
    if (TrainParams().opt_alloc_weight==1 && master_weights == NULL) {
        _INFO("Optimizer \t cudaMalloc=%zu MiB for parametrs(FP32)\n", (adam.n_parameters * sizeof(float)) >> 20);
        cudaCheck(cudaMalloc((void**)&master_weights, adam.n_parameters * sizeof(float)));
    }
    size_t off=0;
    for(auto tensor:opt_ps) {
        size_t nP=tensor->size(),grid_size = CEIL_DIV(nP, 512);
        auto& im = _fish->GetGensorInfo(tensor);
        if(tpGD==SGD_HYBRID){
            tpCurGD = im.isAdam ? ADAMw : SGD;
        }
        if(tpCurGD==ADAMw){
            // _INFO("Optimizer allocating %zu MiB for m\n", (adam.n_parameters * sizeof(float)) >> 20);        
            cudaCheck(cudaMalloc((void**)&(im.gm), tensor->size() * sizeof(float)));        
            cudaCheck(cudaMemset(im.gm, 0, tensor->size() * sizeof(float)));
        }
        if(master_weights!=nullptr){
            copy_and_cast_kernel<<<dim3(grid_size, num_slices), 512, 0, main_stream>>>(master_weights+off, ToX(tensor), nP,nP, nP);
            cudaCheck(cudaGetLastError());
        }
        off += nP;
    }

    
}

int UpdateTensorParam_cuda(hGTensor tensor,size_t np,Optimizer *hOPT,float& grad_norm,int flag){
    CLI_params config = hOPT->_fish->config;
    ADAM_params_ adam = hOPT->TrainParams().adam;
    auto& im = hOPT->_fish->GetGensorInfo(tensor);
    GD_METHOD tpCurGD = hOPT->tpGD;
    if(hOPT->tpGD==SGD_HYBRID){
        tpCurGD = im.isAdam ? ADAMw : SGD;
    }
    float learning_rate=hOPT->LearningRate(),beta1=adam.beta1, beta2=adam.beta2, eps=adam.eps,weight_decay=adam.decay*adam.alpha;
    int num_slices = 1,iter=hOPT->GetITER();
    unsigned int seed = random_u32(&rng_state);
    const char*name = tensor->name;
    ShardInfo shard = {0,tensor->size()};
    float wd = weight_decay;            // we only want to weight decay the 2D tensors and leave all 1D tensors alone
    if(tensor->shape.size()==1)
        wd = 0;
    
    floatX *param_ptr = ToX(tensor),*grad_ptr = ToG(tensor);
    ptrdiff_t opt_state_offset = np; //multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;        
    float* m_ptr = im.gm;   //m_memory==nullptr? nullptr : m_memory + opt_state_offset;
    float* v_ptr = v_memory==nullptr? nullptr : v_memory + opt_state_offset;
    float* master_ptr = NULL;   // why this would slow down converge?
    if (master_weights != NULL && im.isAdam) { 
        master_ptr = master_weights + opt_state_offset;    
    }        

    if(adam.clip_alg!=0 || config.lars_ratio>0){
        grad_norm = tNormOf(tensor,0x0);        //gnorm_1+=grad_norm*grad_norm;
    }        
    float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
    if( config.lars_ratio>0  ){
        grad_scale = tensor->rLARS(grad_scale,config.lars_ratio,0x0);
    }
    if(flag!=0x10001){  //some debug
        adamw_core(param_ptr, master_ptr, grad_ptr,m_ptr, v_ptr,
                    shard.size, shard.size, shard.size, shard.size, num_slices,      //num_parameters,ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices,
                    learning_rate,beta1, beta2, iter, eps, wd, grad_scale, seed, main_stream);
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

int RAW_update(std::vector<hGTensor>& tensors,Optimizer *hOPT,float& grad_norm,int alg,int flag) {
    CLI_params config = hOPT->_fish->config;
    ADAM_params_ adam = hOPT->TrainParams().adam;
    if(adam.clip_alg==0)
        grad_norm = flag==0x10002 ? 1.0e6 : tNormOf(tensors,0x0);
    double gnorm_0=grad_norm,gnorm_1=0;
    float learning_rate = hOPT->LearningRate();
    float beta1=adam.beta1, beta2=adam.beta2, eps=adam.eps,weight_decay=adam.decay*adam.alpha;
    NVTX_RANGE_FN();
    size_t shard_num_parameters = adam.n_parameters,np=0;
    int num_slices = 1,iter=hOPT->GetITER();
    
    // for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {        // generate a unique seed for each tensor
    for(auto tensor:tensors)    {
        if(alg==0){
            UpdateTensorParam_cuda(tensor,np,hOPT,grad_norm,flag);
        }else{
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
            
            float* m_ptr = m_memory + opt_state_offset;
            float* v_ptr = v_memory + opt_state_offset;
            float* master_ptr = NULL;   
            if (master_weights != NULL) { master_ptr = master_weights + opt_state_offset; }        
            // if(init_master_weights) {
            //     size_t grid_size = CEIL_DIV(shard.size, 512);
            //     copy_and_cast_kernel<<<dim3(grid_size, num_slices), 512, 0, main_stream>>>(master_ptr, param_ptr, shard.size,shard.size, shard.size);
            //     cudaCheck(cudaGetLastError());
            // }
            if(adam.clip_alg!=0 || config.lars_ratio>0){
                grad_norm = tNormOf(tensor,0x0);        gnorm_1+=grad_norm*grad_norm;
            }      
            float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
            // if( config.lars_ratio>0 && tensor->shape.size()>1){
            //     grad_scale = tensor->rLARS(config.lars_ratio,0x0);
            // }
                
            if(flag!=0x10001){  //some debug
                adamw_core(param_ptr, master_ptr, grad_ptr,m_ptr, v_ptr,
                            shard.size, shard.size, shard.size, shard.size, num_slices,      //num_parameters,ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices,
                            learning_rate,beta1, beta2, iter, eps, wd, grad_scale, seed, main_stream);
            }
            cudaCheck(cudaGetLastError());
        }
        np += tensor->size();
    }
    // assert(fabs(gnorm_1-gnorm_0*gnorm_0)<1.0e-6*gnorm_1);       // verify
    assert(np==adam.n_parameters);
    cudaCheck(cudaDeviceSynchronize());
    return 0x0;
}

