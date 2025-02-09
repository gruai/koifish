#include "../CLI_params.hpp"
#include "../ggex/GTensor.hpp"
#include "../g_stddef.hpp" 
#include "../kGPT/llmc/sampler.h"
#include "cutils.cuh"
#include "../kGPT/llmc/cublas_common.h"
#include "../kGPT/llmc/matmul.cuh"
#include "../kGPT/llmc/layernorm.cuh"
#include "../kGPT/llmc/encoder.cuh"
#include "../kGPT/llmc/fused_classifier.cuh"
#include "../kGPT/llmc/global_norm.cuh"
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
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
extern cudaStream_t main_stream;
void* cublaslt_workspace = NULL;
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
cublasLtHandle_t cublaslt_handle;

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

typedef struct {
    floatX* encoded; // (B, T, C)
    floatX* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    floatX* atty; // (L, B, T, C)
    // cuDNN saves only some statistics information
#ifdef ENABLE_CUDNN
    float* att;  // (L, B, NH, T)
#else
    floatX* att; // (L, B, NH, T, T)
#endif

    floatX* residual2; // (L, B, T, C)
    floatX* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    floatX* fch; // (L, B, T, 4*C)
    floatX* fch_gelu; // (L, B, T, 4*C)
    floatX* residual3; // (L, B, T, C)
    floatX* lnf; // (B, T, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T), will be accumulated in micro-steps
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    floatX* qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    floatX* output;
    // some additional scratch buffers
    // floatX* scratch_bt4c;   // (B, T, 4*C)
    // floatX* scratch_btc;    // (B, T, C)
} ActivationTensors;
typedef struct {
    floatX* wte; // (V, C)
    floatX* wpe; // (maxT, C)
    floatX* ln1w; // (L, C)             Forawd of layer_normal
    floatX* ln1b; // (L, C)
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C)
    floatX* attprojb; // (L, C)
    floatX* ln2w; // (L, C)             Backward of layer_normal
    floatX* ln2b; // (L, C)
    floatX* fcw; // (L, 4*C, C)
    floatX* fcb; // (L, 4*C)
    floatX* fcprojw; // (L, C, 4*C)
    floatX* fcprojb; // (L, C)
    floatX* lnfw; // (C)
    floatX* lnfb; // (C)
} ParameterTensors;
struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};
#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};
#define NUM_ACTIVATION_TENSORS 21
constexpr const int NUM_PARAMETER_TENSORS = 16;

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;


typedef struct {
    GPT2Config config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    float* master_weights;     // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss=nullptr; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    int use_master_weights = 1; // keep master weights copy in float for optim update? 0|1
    int gelu_fusion; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute; // recompute gelu | layernorm forward during model backward? 0|1|2
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
} GPT2;
GPT2 *model=nullptr;

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte         50304*768
    param_sizes[1] = maxT * C; // wpe       1024*768
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb

    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

GPT2 *_init_model(Fish *fish,int flag){
    GPT2 *gpt = new GPT2();
    gpt->config.vocab_size = 50257;
    gpt->config.padded_vocab_size = 50304; // padded to 128 for CUDA kernel efficiency
    gpt->config.num_layers = 12;
    gpt->config.num_layers = fish->hparams.nLayer();
    gpt->config.channels = 768;
    gpt->config.num_heads = 12;
    gpt->config.max_seq_len = 1024;
    gpt->recompute = 1;
    fill_in_parameter_sizes(gpt->param_elements, gpt->param_sizeof, gpt->config);
    gpt->num_parameters = 0;
    gpt->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        gpt->num_parameters += gpt->param_elements[i];
        gpt->num_parameters_bytes += gpt->param_elements[i] * gpt->param_sizeof[i];
    }
    
    // create memory for model parameters on the device
    //malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += gpt->param_elements[i] * gpt->param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);
    gpt->params.wte=ToX(embed->w),  gpt->params.wpe=ToX(embed->b);       gpt->acts.encoded=ToX(embed->out);
    gpt->grads.wte=ToG(embed->w);   gpt->grads.wpe=ToG(embed->b);
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    gpt->acts.lnf=ToX(lnf->out);    gpt->acts.lnf_mean=TO<float>(lnf->mean); gpt->acts.lnf_rstd=TO<float>(lnf->rstd);
    gpt->params.lnfw=ToX(lnf->w),   gpt->params.lnfb=ToX(lnf->b); 
    gpt->grads.lnfw=ToG(lnf->w),    gpt->grads.lnfb=ToG(lnf->b);

    
    for(int l = 0; l < gpt->config.num_layers; l++) {
        SelfAttention* qkv = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        FFN* ffn = fish->GetNeuron<FFN>("FFN",l);
        if(l==0){
            gpt->acts.ln1=ToX(qkv->norm.out);       gpt->acts.ln1_mean=TO<float>(qkv->norm.mean), gpt->acts.ln1_rstd=TO<float>(qkv->norm.rstd);
            gpt->params.ln1w=ToX(qkv->norm.w);     gpt->params.ln1b=ToX(qkv->norm.b);
        }        
    }
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    gpt->acts.output = ToX(cls->preLogits);    
    gpt->acts.losses = TO<float>(cls->out);
    // gpt->acts.scratch_bt4c = ToX(GTensor::scratch_bt4c);    gpt->acts.scratch_btc = ToX(GTensor::scratch_btc);
    //gpt->params.wte=ToX(cls->proj.w); same as embed->w, maybe BUG!!!    
        
    /*void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += gpt->param_elements[i] * gpt->param_sizeof[i];
    }
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }
    */

    gpt->inputs = TO<int>(fish->Input());   
    gpt->targets =  TO<int>(fish->Target());   
    PrintTensor<int>("target",gpt->targets,true,GTensor::B,GTensor::T);
    gpt->batch_size = GTensor::B;        gpt->seq_len = GTensor::T;
    cudaCheck(cudaMalloc(((void**)&(gpt->accumulated_mean_loss)), sizeof(float)));
    gpt->cpu_losses = new float[GTensor::B * GTensor::T];
    return gpt;
}

void RAW_forward(Fish *fish,int flag) {
    if(model==nullptr){
        model = _init_model(fish,flag);
    }
    NVTX_RANGE_FN();
    size_t B=GTensor::B, T=GTensor::T;    
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;    
    /*
    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    // validate inputs, all indices must be in the range [0, V)
    // we can do this while the copies are already underway
    tokenCheck(inputs, B*T, V);*/
    
    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C, main_stream); // encoding goes into residual[0]
    PrintTensor<floatX>("wte",params.wte,true,Vp,C);PrintTensor<floatX>("wpe",params.wpe,true,T,C);
    PrintTensor<int>("inputs",model->inputs,true,B,T);      PrintTensor<floatX>("GetRow",acts.encoded,true,B,T,C);
    // first layernorm isn't fused
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, T, C, main_stream);
    // PrintTensor<floatX>("ln1W",params.ln1w,true,C,1);       PrintTensor<floatX>("ln1B",params.ln1b,true,C,1);
    PrintTensor<floatX>("ln1",acts.ln1,true,B,T,C);
    SelfAttention *QKV=nullptr,*lastQKV=nullptr;
    FFN *ffn=nullptr,*lastFFN=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);
        floatX* residual = l == 0 ? acts.encoded : ToX(lastFFN->out); //floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        // get the pointers of the weights for this layer
        floatX* l_qkvw = ToX(QKV->Q.w);     // floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = ToX(QKV->Q.b);     // floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = ToX(QKV->proj_cat.w); // floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = ToX(QKV->proj_cat.b);// floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = ToX(ffn->norm.w);// floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = ToX(ffn->norm.b);// floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = ToX(ffn->up.w);     // floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = ToX(ffn->up.b);     // floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = ToX(ffn->down.w);// floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = ToX(ffn->down.b);// floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? ToX(QKV->norm.out) : nullptr;   // floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
        floatX* l_qkvr = ToX(QKV->Q.out);       // floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
        floatX* l_atty = ToX(QKV->attn);        // floatX* l_atty = acts.atty + l * B * T * C;
        floatX* l_residual2 = ToX(QKV->out);  // floatX* l_residual2 = acts.residual2 + l * B * T * C;
        floatX* l_ln2 = (model->recompute < 2) ? ToX(ffn->norm.out) : nullptr; // floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
        float* l_ln2_mean = TO<float>(ffn->norm.mean);      // float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = TO<float>(ffn->norm.rstd);      // float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        floatX* l_fch = ToX(ffn->up.out);   // floatX* l_fch = acts.fch + l * B * T * 4*C;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        // floatX* l_fch_gelu = ToX(ffn->relu.out);       //floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
        floatX* l_residual3 = ToX(ffn->out);      // floatX* l_residual3 = acts.residual3 + l * B * T * C;
        floatX* scratch = ToX(GTensor::scratch_output);         //(floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.
        floatX* l_fch_gelu = ToX(GTensor::scratch_output);
        // now do the forward pass
        #ifdef ENABLE_CUDNN
        float* l_att = TO<float>(QKV->trans); //(float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        // PrintTensor<floatX>("l_qkvw",l_qkvw,true,3*C,C);       PrintTensor<floatX>("l_qkvb",l_qkvb,true,3*C,1);
        // PrintTensor<floatX>("l_qkvr",l_qkvr,true,B,T,3*C);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, T, NH, C, main_stream);
        #else
        floatX* l_att = ToX(QKV->trans);  //floatX* l_att = acts.att + l * B * NH * T * T;
        if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
            cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        }
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH, main_stream);
        #endif
        PrintTensor<floatX>("l_atty",l_atty,true,B,T,C);
        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, T, C, C, main_stream);
        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*T, C, main_stream);
        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, T, C, 4*C, main_stream, l_fch, model->gelu_fusion);
        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C, main_stream);
        PrintTensor<floatX>("ffn",scratch,true,B,T,C);
        // OK, fusion across blocks.
        if(l+1 != L) {
            SelfAttention *ln1 = fish->GetNeuron<SelfAttention>("SelfAttention",l+1);
            floatX* l_ln1 = ToX(ln1->norm.out);     // floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * T * C : acts.lnf;
            float* l_ln1_mean = TO<float>(ln1->norm.mean);    // float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * T;
            float* l_ln1_rstd = TO<float>(ln1->norm.rstd);      // float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * T;
            const floatX* l_ln1w = ToX(ln1->norm.w); // const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = ToX(ln1->norm.b); // const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,B * T, C, main_stream);
            PrintTensor<floatX>("residual3",l_residual3,true,B,T,C);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,params.lnfw, params.lnfb,B * T, C, main_stream);
        }
        lastFFN = ffn;  lastQKV = QKV;
    }
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    if(tpFuseCu==1)
        cls->FUSE_cuda(lnf->out,embed->w,flag);
    else
        matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    PrintTensor<floatX>("output",acts.output,true,B,T,C);
    cudaCheck(cudaDeviceSynchronize());
}

typedef struct {
    ptrdiff_t offset;
    size_t size;
} ShardInfo;
ShardInfo gpt2_get_tensor_at_layer(const GPT2 *model, int layer_id, int param_tensor_id) {
    // first offset our way to the parameter tensor start
    ptrdiff_t offset = 0;
    for (int i = 0; i < param_tensor_id; i++) {
        offset += (ptrdiff_t)model->param_elements[i];
    }
    size_t size = model->param_elements[param_tensor_id] ;
    // if we are in the transformer block, we need to additionally offset by the layer id
    if(2 <= param_tensor_id && param_tensor_id <= 13) {
        size /= model->config.num_layers;
        offset += (ptrdiff_t)(layer_id * size);
    }
    return {offset, size};
}

bool cuClear(std::vector<hGTensor> tensors,int flag){
    for(auto tensor:tensors){
        cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
    }
    
    return true;
}
static int micro_step=0;
float RAW_backward(Fish *fish,const int* hostInput, int grad_accum_steps,bool isOnlyEvaluate,int flag) {    
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        NvtxRange rng("InitGrads");
        // allocate buffers for weight gradients
        // printf("allocating %d MiB for parameter gradients\n", (int)round(model->num_parameters * sizeof(floatX) / (1024 * 1024)));
        // model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);
        // initialise cpu scratch buffers for encoder backward
        size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
        assert((size_t)(model->batch_size * model->seq_len) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
        model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * model->seq_len * num_c_groups);
        model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * model->seq_len * num_c_groups);
    }

    // on the first micro-step zero the gradients, as we're about to += accumulate into them
    if (micro_step == 0) {
        // there are currently two state vars during the gradient accumulation inner loop:
        // 1) the losses accumulate += into acts.losses, reset here
        // 2) the gradients accumulate += into grads_memory, reset here
        cudaCheck(cudaMemsetAsync(model->acts.losses, 0, model->batch_size * model->seq_len * sizeof(float), main_stream));
        // cudaCheck(cudaMemsetAsync(model->grads_memory, 0, model->num_parameters * sizeof(floatX), main_stream));
    }

    // convenience shortcuts, size_t instead of int so that pointer arithmetics don't overflow
    const size_t B = model->batch_size;
    const size_t T = model->seq_len;
    const size_t V = model->config.vocab_size;
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;

    // accumulate the losses inside acts.losses, and kick off the backward pass inside the fused classifier
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    Embed* embed = fish->GetNeuron<Embed>("Embed",0);
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    if(tpFuseCu==1){
        model->mean_loss = cls->mean_loss;
        cls->FUSE_cuda(lnf->out,embed->w,0x0);
    }else{
        const float dloss = 1.0f / (float)(B * T * grad_accum_steps); // results in the uniform average loss over all elements
        fused_classifier(acts.output, acts.losses, dloss, model->targets, B, T, V, Vp, True, main_stream);        
        PrintTensor<int>("target",model->targets,true,B,T);
        PrintTensor<floatX>("grad of output",acts.output,true,B,T,C);    
        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // this was done in the fused classifier kernel as last step of forward pass
        // technically that is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        // next: backward the classifier matmul      matmul_forward_cublaslt(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
        matmul_backward(ToX(GTensor::scratch_bt4c), grads.wte, NULL, acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp, main_stream);
    }

    if(isOnlyEvaluate){

    }else{    // backward pass: go in the reverse order of the forward pass, and call backward() functions
        // reset residual stream gradients (put here to work with gradient accumulation)
        floatX* dresidual = ToX(GTensor::scratch_btc),*scratchX = (floatX*)acts.output; 
        cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));
        float*  scratchF = (float*)acts.output;
        PrintTensor<floatX>("back of P",ToX(GTensor::scratch_bt4c),true,B,T,C);
        // backward the final layernorm
        SelfAttention *QKV=fish->GetNeuron<SelfAttention>("SelfAttention",L-1),*preQKV=nullptr;
        FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1),*preFFN=nullptr;  
        floatX* residual = ToX(ffn->out);   //acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
        layernorm_backward(dresidual, grads.lnfw, grads.lnfb, scratchF, ToX(GTensor::scratch_bt4c), residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, main_stream);
        PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
        // from this point on, we no longer need the values stored in the last residual, so we can reuse that memory as generic
        // scratch for backward computations
        floatX* dl_btc = residual;

        // now backward all the layers
        for (int l = L-1; l >= 0; l--) {
            NvtxRange layer_range("Layer", l);
            QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
            ffn = fish->GetNeuron<FFN>("FFN",l);        preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1); 
            residual = l == 0 ? acts.encoded : ToX(preFFN->out);   //acts.residual3 + (l-1) * B * T * C;

            // get the pointers of the weights for this layer
            floatX* l_ln1w = ToX(QKV->norm.w);          //floatX* l_ln1w = params.ln1w + l * C;
            floatX* l_ln1b = ToX(QKV->norm.b);          //floatX* l_ln1b = params.ln1b + l * C;
            floatX* l_qkvw = ToX(QKV->Q.w);             //floatX* l_qkvw = params.qkvw + l * 3*C * C;
            floatX* l_attprojw = ToX(QKV->proj_cat.w);  // floatX* l_attprojw = params.attprojw + l * C * C;
            floatX* l_ln2w = ToX(ffn->norm.w);          //floatX* l_ln2w = params.ln2w + l * C;
            floatX* l_ln2b = ToX(ffn->norm.b);          //floatX* l_ln2b = params.ln2b + l * C;
            floatX* l_fcw = ToX(ffn->up.w);             // floatX* l_fcw = params.fcw + l * 4*C * C;
            floatX* l_fcprojw = ToX(ffn->down.w);       // floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
            // get the pointers of the gradients of the weights for this layer
            floatX* dl_ln1w = ToG(QKV->norm.w);         //floatX* dl_ln1w = grads.ln1w + l * C;
            floatX* dl_ln1b = ToG(QKV->norm.b);         //floatX* dl_ln1b = grads.ln1b + l * C;
            floatX* dl_qkvw = ToG(QKV->Q.w);            //floatX* dl_qkvw = grads.qkvw + l * 3*C * C;
            floatX* dl_qkvb = ToG(QKV->Q.b);         //floatX* dl_qkvb = grads.qkvb + l * 3*C;
            floatX* dl_attprojw = ToG(QKV->proj_cat.w);  //floatX* dl_attprojw = grads.attprojw + l * C * C;
            floatX* dl_attprojb = ToG(QKV->proj_cat.b);  //floatX* dl_attprojb = grads.attprojb + l * C;
            floatX* dl_ln2w = ToG(ffn->norm.w);          //floatX* dl_ln2w = grads.ln2w + l * C;
            floatX* dl_ln2b = ToG(ffn->norm.b);          //floatX* dl_ln2b = grads.ln2b + l * C;
            floatX* dl_fcw = ToG(ffn->up.w);             //floatX* dl_fcw = grads.fcw + l * 4*C * C;
            floatX* dl_fcb = ToG(ffn->up.b);             //ffloatX* dl_fcb = grads.fcb + l * 4*C;
            floatX* dl_fcprojw = ToG(ffn->down.w);       //floatX* dl_fcprojw = grads.fcprojw + l * C * 4*C;
            floatX* dl_fcprojb = ToG(ffn->down.b);       //floatX* dl_fcprojb = grads.fcprojb + l * C;
            // get the pointers of the activations for this layer
            floatX* l_ln1 = ToX(QKV->norm.out);                 //floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * T * C : acts.lnf;
            float* l_ln1_mean = TO<float>(QKV->norm.mean);      //float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = TO<float>(QKV->norm.rstd);      //acts.ln1_rstd + l * B * T;
            floatX* l_qkvr = ToX(QKV->Q.out);                   //floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
            floatX* l_atty = ToX(QKV->attn);                  //floatX* l_atty = acts.atty + l * B * T * C;
            floatX* l_residual2 = ToX(QKV->out);                //floatX* l_residual2 = acts.residual2 + l * B * T * C;
            floatX* l_ln2 = (model->recompute < 2) ? ToX(ffn->norm.out) : nullptr;  //floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * T * C : acts.lnf;
            float* l_ln2_mean = TO<float>(ffn->norm.mean);    //float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = TO<float>(ffn->norm.rstd);    //float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            //floatX* fch; // (L, B, T, 4*C)    floatX* fch_gelu; // (L, B, T, 4*C)
            floatX* l_fch_pre_gelu = ToX(ffn->up.out);            //floatX* l_fch_pre_gelu = acts.fch + l * B * T * 4*C;
            // floatX* l_fch_gelu = ToX(ffn->relu.out);                //floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * T * 4*C : acts.fch_gelu;
            float* l_att = TO<float>(QKV->trans); //float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
            floatX* scratch = ToX(GTensor::scratch_output);         //(floatX*)acts.output; // used for non-cudnn attention, fcproj, attproj, etc.
            floatX* l_fch_gelu = ToX(GTensor::scratch_output);
            // get the pointers of the gradients of the activations for this layer
            // notice that there is no l *, because we just have a single copy, and keep
            // re-using this memory in every Transformer block as we calculate backward pass
            floatX* dl_bt4c = ToX(GTensor::scratch_bt4c);   
            
            // start the backward pass for this layer
            if(model->recompute >= 1) {            // recompute >= 1 means we recompute gelu. in this case,
                // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
                //  gelu = (floatX)(0.5f * pre * (1.0f + tanhf(GELU_SCALING_FACTOR * (pre + 0.044715f *pre*pre*pre))));
                PrintTensor<floatX>("pre gelu",l_fch_pre_gelu,true,B,T,4*C);
                gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);     
                PrintTensor<floatX>("back of gelu",l_fch_gelu,true,B,T,4*C);
            }
            matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);
            PrintTensor<floatX>("back of ffn1",dl_bt4c,true,B,T,4*C);
            if(model->recompute >= 2) {
                // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
                layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
            }
            matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
            // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
            layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);
            matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream);
            PrintTensor<floatX>("back of ffn0",dl_btc,true,B,T,C);
            #ifdef ENABLE_CUDNN
            //float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
            attention_backward_cudnn(dl_bt4c, dl_btc, l_qkvr, l_atty, (float*)l_att, B, T, NH, C, main_stream);
            #else
            floatX* l_att = acts.att + l * B * NH * T * T;
            // we need B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
            floatX* buffer_a = l_atty;
            floatX* buffer_b = l_fch_pre_gelu;        // this is B x T x 4C, so even larger than what we need
            attention_backward(dl_bt4c, buffer_b, scratchX, buffer_a, dl_btc, l_qkvr, l_att, B, T, C, NH, main_stream);
            #endif
            if(model->recompute >= 2) {
                layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
            }
            PrintTensor<floatX>("back of attn",dl_bt4c,true,B,T,C);
            // QKV parameter gradients
            matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, scratchF, B, T, C, 3 * C, main_stream);
            // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, scratchF, dl_btc, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, main_stream);

            // Accumulate gradients from this layer in a background stream.
            if(last_step) {
                floatX* const pointers[] = {
                    dl_ln1w, dl_ln1b,
                    dl_qkvw, dl_qkvb,
                    dl_attprojw, dl_attprojb,
                    dl_ln2w, dl_ln2b,
                    dl_fcw, dl_fcb,
                    dl_fcprojw, dl_fcprojb
                };
                const size_t nelem[] = {
                    C, C,
                    3 * C * C, 3 * C,
                    C * C, C,
                    C, C,
                    4 * C * C, 4 * C,
                    C * 4 * C, C
                };
                // multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);     // no multi-GPU, just exit.
            }
        }
        encoder_backward(grads.wte, grads.wpe, scratchX, model->workload_indices, model->bucket_info,
                        dresidual, model->inputs, hostInput, B, T, C, random_u32(&model->rng_state), main_stream);
        // g_dump_level = 0;                 
        PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
        g_dump_level = 1;
        PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
    }

    // Aggregate all gradients that are not part of the transformer blocks5
    if(tpFuseCu==0 && last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(model->accumulated_mean_loss, acts.losses, B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(model->accumulated_mean_loss, model->accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&model->mean_loss, model->accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        model->mean_loss /= B*T*grad_accum_steps;
        // reduce the gradients for non-transformer block parameters
        // floatX* const pointers[] = {grads.wte, grads.wpe, grads.lnfw, grads.lnfb};
        // const size_t nelem[] = {Vp * C, T * C, C, C};
        // multi_gpu_async_reduce_gradient(pointers, nelem, &multi_gpu_config, main_stream);
    }

    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        // model->mean_loss /= B*T*grad_accum_steps;
        micro_step = 0;
    } else {
        model->mean_loss = -1.f; // no loss available yet
        micro_step++;
    }
    
    return model->mean_loss;
}   

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
int RAW_update(std::vector<hGTensor>& tensors,ADAM_params_ adam,float learning_rate,float& grad_norm,int iter,int flag) {
    grad_norm = flag==0x10002 ? 1.0e6 : tNormOf(tensors,0x0);
    float grad_scale = (grad_norm > adam.gclip) ? adam.gclip / grad_norm : 1.0f;
    float beta1=adam.beta1, beta2=adam.beta2, eps=adam.eps;
    float weight_decay=adam.decay*adam.alpha;
    NVTX_RANGE_FN();
    size_t shard_num_parameters = adam.n_parameters,np=0;
    int num_slices = 1;
    // lazily allocate m,v memory and master weights (usually on the first iteration)
    if (model->m_memory == NULL) {
        NvtxRange rng("InitOpt");
        printf("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
        printf("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
        cudaCheck(cudaMalloc((void**)&model->m_memory, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->v_memory, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, shard_num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, shard_num_parameters * sizeof(float)));
    }

    bool init_master_weights = false;
    if (model->use_master_weights == 1 && model->master_weights == NULL) {
        printf("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        cudaCheck(cudaMalloc((void**)&model->master_weights, shard_num_parameters * sizeof(float)));
        init_master_weights = true;
    }

    // for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {        // generate a unique seed for each tensor
    for(auto tensor:tensors)    {
        unsigned int seed = random_u32(&model->rng_state);
        const char*name = tensor->name;
        // size_t szT = tensor->nByte();
        ShardInfo shard = {0,tensor->size()};
        /*int num_slices = model->config.num_layers;
        if((i < 2 || i > 13)) {
            num_slices = 1;
        }
        ShardInfo off_size = gpt2_get_tensor_at_layer(model, 0, i);   //  {offset, size};
        ShardInfo shard = {0, off_size.size};    //multi_gpu_get_shard_offset(tensor.size, multi_gpu_config, 1);
        ptrdiff_t local_offset_full = off_size.offset + shard.offset;
        ptrdiff_t local_offset_partial = off_size.offset; // / multi_gpu_config->num_processes;
        
        // we only want to weight decay the 2D tensors and leave all 1D tensors alone
        // in particular this also decays the embedding weights, but this is ok:
        // - the token embeddings are weight shared and participate in the final projection to logits
        // - the position embeddings actively participate at every forward/backward pass
        float wd = (i == 0 || i == 1 || i == 4 || i == 6 || i == 10 || i == 12) ? weight_decay : 0.0f;*/
        float wd = weight_decay;
        if(tensor->shape.size()==1)
            wd = 0;
        // ptrdiff_t local_offset_full=0,local_offset_partial=tensor->offset;
        floatX* param_ptr = ToX(tensor);    //(floatX*)model->params_memory + local_offset_full;
        floatX* grad_ptr = ToG(tensor);     //(floatX*)model->grads_memory + local_offset_full;

        ptrdiff_t opt_state_offset = np; //multi_gpu_config->zero_stage < 1 ?  local_offset_full : local_offset_partial;        
        float* m_ptr = model->m_memory + opt_state_offset;
        float* v_ptr = model->v_memory + opt_state_offset;
        float* master_ptr = NULL;   
        if (model->master_weights != NULL) { master_ptr = model->master_weights + opt_state_offset; }
        np += tensor->size();
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

