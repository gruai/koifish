/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 * 
 *  \brief Transformer in cuda kernel
 *  \author Yingshi Chen
 */
#include "./Operator.cuh"
#include "./llm_c/sampler.h"
#include "./llm_c/layernorm.cuh"
#include "./llm_c/encoder.cuh"
#include "./llm_c/cuda_utils.cuh"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"

extern int tpFuseCu;
extern cudaStream_t main_stream;

__global__ void __launch_bounds__(1024) test_print_kernel(half *__restrict__ arr){
    // printf("test kernel\n");
    if (((int)blockIdx.x == 0) && ((int)threadIdx.x == 0))    {
        // arr[0] = ((half)(2));
        __syncthreads();
        printf("%f ", __half2float(arr[0]));

    }
}

unsigned long long rng_state=0;
float* accumulated_mean_loss=nullptr;

static int micro_step=0;    //*workload_indices=nullptr;
// static int4 *bucket_info=nullptr;
float RAW_backward(Fish *fish,const int* iX, int grad_accum_steps,bool isOnlyEvaluate,int flag) {    
    assert(!isOnlyEvaluate);
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    auto config = fish->config;
    int B,T,C,L = config.nLayer(),NH = config.n_head();      
    fish->GetBTC(B,T,C);    
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    hGensor cur=nullptr,delta=cls->delta;      
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed",0); 
    if (micro_step == 0) {// on the first micro-step zero the gradients, as we're about to += accumulate into them;     
        cudaCheck(cudaMemsetAsync(TO<float>(cls->out), 0,B*T* sizeof(float), main_stream));
    }
    
    // floatX *scratchX=(floatX *)GTensor::buff;     
    GTensor::delta->Zero();         //cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));    //???
    PrintTensor<floatX>("back of P",ToX(GTensor::bt4c),true,B,T,C);
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1);  
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);    
    delta = cls->FUSE_cuda(nullptr,0x0);    //some operation fused in forward pass
    lnf->FUSE_cuda(delta,0x0);  //

    hGensor lastDelta = ffn->out;    
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        SelfAttention *QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);        //preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1);         
        // GeNeuron *last = l == 0 ? embed : (GeNeuron *)(fish->GetNeuron<FFN>("FFN",l-1));    //residual = l == 0 ? ToX(embed->out) : ToX(preFFN->out);  
        // ffn->delta = lastDelta;     ffn->tmpDelta = lastDelta;        
        // QKV->deltaCat=ffn->delta;   
        // LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->FUSE_cuda(nullptr, 0x0);   
        QKV->FUSE_cuda(nullptr,0x0);   //QKV->norm.out,last->out
    }
    embed->OnEmbed(nullptr,0x0);    //,random_u32(&rng_state)
    
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
/*
void RAW_forward(Fish *fish,int flag) {
    NVTX_RANGE_FN();
    auto config = fish->config;
    int B,T,C,tpFuseNormal=config.Fuse_Normal;      
    fish->GetBTC(B,T,C);    
    const size_t L = config.nLayer(),NH = config.n_head();       
    hGensor cur=nullptr,residual=nullptr;
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed",0);
    cur = embed->Ming(nullptr,fish->Input());     
    residual = cur; //embed->out;
    SelfAttention *QKV0 = fish->GetNeuron<SelfAttention>("SelfAttention",0),*QKV=nullptr;

    if(tpFuseNormal==1){
        cur=QKV0->norm.FUSE_cuda(cur);   
    } 

    FFN *ffn=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);            //ffn->out = GTensor::delta;
        LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->fuseNorm = tpFuseNormal==1?hNorm:nullptr;       
        QKV->fuseNorm =  tpFuseNormal==1?&(ffn->norm):nullptr;
        cur = QKV->FUSE_cuda(cur,residual,nullptr,nullptr,flag);        
        cur = ffn->FUSE_cuda(cur,nullptr, 0x0);  
        residual = ffn->out;
    }    
    if(tpFuseNormal==0){
        cur = lnf->FUSE_cuda(ffn->out); 
    }
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if(tpFuseCu==1)
        cls->FUSE_cuda(cur,flag); //embed->w,
    else
        assert(0);//cls->preLogits = lnf->out*embed->w;   //matmul_forward_cublaslt(ToX(cls->preLogits), ToX(lnf->out), ToX(embed->w), NULL, B, T, C, Vp, main_stream);
    PrintTensor<floatX>("output",ToX(cls->preLogits),true,B,T,C);
    // ffn->norm.out->PrintX<floatX>("inp1",0,-1); 
    cudaCheck(cudaDeviceSynchronize());
}*/

void LAMA3_forward(Fish *fish,int flag)    {
    NVTX_RANGE_FN();
    auto config = fish->config;
    int B,T,C,tpFuseNormal=config.Fuse_Normal;      
    fish->GetBTC(B,T,C);    
    const size_t L = config.nLayer(),NH = config.n_head();       
    hGensor cur=nullptr,residual=nullptr;
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed",0);
    SelfAttention *QKV0 = fish->GetNeuron<SelfAttention>("SelfAttention",0),*QKV=nullptr;
    // forward pass
    // ParameterTensors params = model->params; // for brevity
    // ActivationTensors acts = model->acts;
    // float *residual;
    // // The freq_cos and freq_sin (cis-values) are same for every layer in the model. So, loading them for one time only
    // // TODO: Extract them from activations (if needed)
    // float *freq_cos = acts.freq_cos;
    // float *freq_sin = acts.freq_sin;
    cur = embed->OnEmbed(fish->Input(),0x0);      residual = cur;     
    // encoder_forward(acts.encoded, model->inputs, params.wte, B, T, C);       // encoding goes into residual[0]
    // precompute_freqs_cis(freq_cos, freq_sin, ((C / NH) / 2), T, rope_theta); // rope_theta = 10000.0

    for (int l = 0; l < L; l++)    {

        /*residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // get the pointers of the weights for this layer
        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_qkvb = params.qkvb + l * 3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcb = params.fcb + l * 4 * C;
        float *l_fcw_g = params.fcw_g + l * 4 * C * C; // (L, 4*C, C) Added for gate Mechanism of SwiGLU Activation
        float *l_fcb_g = params.fcb_g + l * 4 * C;     // (L, 4*C)
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float *l_ln1 = acts.ln1 + l * B * T * C;
        // float *l_ln1_mean = acts.ln1_mean + l * B * T;
        // float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkvr = acts.qkvr + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        // float *l_ln2_mean = acts.ln2_mean + l * B * T;
        // float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_glu = acts.fch_glu + l * B * T * 4 * C;
        float *l_fch_swiglu = acts.fch_swiglu + l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float *scratch = acts.output;

        // now do the forward pass
        rmsnorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward_gqa(l_atty, l_qkvr, l_att, scratch, freq_cos, freq_cos, B, T, C, NH, num_kv_heads); // Added  acts.freq_cos, acts.freq_sin for q and k - RoPE
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        rmsnorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);         // xW
        matmul_forward(l_fch_glu, l_ln2, l_fcw_g, l_fcb_g, B, T, C, 4 * C); // xV
        swiglu_forward(l_fch_swiglu, l_fch, l_fch_glu, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_swiglu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);*/
    }

    /* residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
    rmsnorm_forward(acts.lnf, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    also forward the cross-entropy loss function if we have the targets
    if (targets != NULL)    {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++)        {
            mean_loss += model->cpu_losses[i];
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    }    else    {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }*/
}


    /*if (model->params_memory == NULL)    {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int num_kv_heads = model->config.num_kv_heads;
    float rope_theta = model->config.rope_theta;*/

    // validate inputs, all indices must be in the range [0, V)
    /*for (int i = 0; i < B * T; i++)    {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL)        {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }
    // allocate space for all the activations if needed (done here, lazily)
    if (model->acts_memory == NULL)    {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void **)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void **)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void **)&model->cpu_losses, B * T * sizeof(float)));
    }    else    {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len)        {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }
    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL)    {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }*/



