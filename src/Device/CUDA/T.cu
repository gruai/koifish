/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Some trial/testing cuda kernels
 *  \author Yingshi Chen
 */
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "../../Manifold/gLLM.hpp"
#include "./kernel/Operator.cuh"

extern int tpFuseCu;
extern cudaStream_t main_stream;

// static int coopsms;
// static __constant__ CoopLayer<> cooplayers[MAX_LAYERS];

unsigned long long rng_state = 0;
float* accumulated_mean_loss = nullptr;

static int micro_step = 0;  //*workload_indices=nullptr;
// static int4 *bucket_info=nullptr;
float RAW_backward(Fish* fish, const int* iX, int grad_accum_steps, bool isOnlyEvaluate, int flag) {
    assert(!isOnlyEvaluate);
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    auto config    = fish->config;
    int B, T, C, L = config.nLayer(), NH = config.n_head();
    fish->GetBTC(B, T, C);
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS", 0);
    hGensor cur = nullptr, delta = cls->delta;
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    if (micro_step == 0) {  // on the first micro-step zero the gradients, as we're about to += accumulate into them;
        cudaCheck(cudaMemsetAsync(TO<float>(cls->out), 0, B * T * sizeof(float), main_stream));
    }

    // floatX *scratchX=(floatX *)GTensor::buff;
    GTensor::delta->Zero();  // cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));    //???
    PrintTensor<floatX>("back of P", ToX(GTensor::bt4c), true, B, T, C);
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    FFN* ffn         = fish->GetNeuron<FFN>("FFN", L - 1);
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal", 0);
    delta            = cls->cuTrain(nullptr, 0x0);  // some operation fused in forward pass
    lnf->cuTrain(delta, 0x0);                       //

    hGensor lastDelta = ffn->out;
    for (int l = L - 1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        SelfAttention* QKV = fish->GetNeuron<SelfAttention>("SelfAttention", l);
        ffn                = fish->GetNeuron<FFN>("FFN", l);  // preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1);
        // GeNeuron *last = l == 0 ? embed : (GeNeuron *)(fish->GetNeuron<FFN>("FFN",l-1));    //residual = l == 0 ? ToX(embed->out) : ToX(preFFN->out);
        // ffn->delta = lastDelta;     ffn->tmpDelta = lastDelta;
        // LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->cuTrain(nullptr, 0x0);
        QKV->cuTrain(nullptr, 0x0);  // QKV->norm.out,last->out
    }
    embed->OnEmbed(nullptr, 0x0);  //,random_u32(&rng_state)

    // Aggregate all gradients that are not part of the transformer blocks5
    if (tpFuseCu == 0 && last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(accumulated_mean_loss, TO<float>(cls->out), B * T, main_stream);
// reduce loss across GPUs to a single, final float across all microsteps and GPUs
#if MULTI_GPU
        ncclCheck(ncclAllReduce(accumulated_mean_loss, accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
#endif
        cudaCheck(cudaMemcpyAsync(&cls->mean_loss, accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        cls->mean_loss /= B * T * grad_accum_steps;
    }
    cudaCheck(cudaDeviceSynchronize());
    if (last_step) {
        // cls->mean_loss /= B*T*grad_accum_steps;
        micro_step = 0;
    } else {
        cls->mean_loss = -1.f;  // no loss available yet
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
        cur=QKV0->norm.cuTrain(cur);
    }

    FFN *ffn=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);            //ffn->out = GTensor::delta;
        LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->fuseNorm = tpFuseNormal==1?hNorm:nullptr;
        QKV->fuseNorm =  tpFuseNormal==1?&(ffn->norm):nullptr;
        cur = QKV->cuTrain(cur,residual,nullptr,nullptr,flag);
        cur = ffn->cuTrain(cur,nullptr, 0x0);
        residual = ffn->out;
    }
    if(tpFuseNormal==0){
        cur = lnf->cuTrain(ffn->out);
    }
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if(tpFuseCu==1)
        cls->cuTrain(cur,flag); //embed->w,
    else
        assert(0);//cls->preLogits = lnf->out*embed->w;
    PrintTensor<floatX>("output",ToX(cls->preLogits),true,B,T,C);
    // ffn->norm.out->PrintX<floatX>("inp1",0,-1);

}*/
