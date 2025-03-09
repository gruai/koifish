
// #include "../ggex/GG_util.hpp"       //ugly  "__builtin_ia32_ldtilecfg" is undefined
#include "./cuda_common.h"
#include "./cublas_common.h"
#include "./llm_c/matmul.cuh"
#include "./llm_c/layernorm.cuh"
#include "./llm_c/encoder.cuh"
#include "./llm_c/fused_classifier.cuh"
#include "./kernel/rope.cuh"
// #include "./TE/fused_attn/fused_attn_fp8.cu"
#include "../../Manifold/Neuron.hpp"
// #include "./mfu.h"
#define NOMINMAX
#include <cudnn_frontend.h>

#undef ENABLE_CUDNN
namespace fe = cudnn_frontend;
// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
    // static_assert(false, "cuDNN is not supported in FP32 mode.")
    // use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
    #define ENABLE_CUDNN
    #define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
    #define ENABLE_CUDNN
    #define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif
#ifdef ENABLE_CUDNN
    // defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
    #include "./llm_c/cudnn_att.h"
#else
    // defines: attention_forward, attention_backward
    #include "./llm_c/attention.cuh"
#endif

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
cublasLtHandle_t cublaslt_handle;
void* cublaslt_workspace = NULL;
cudaStream_t main_stream=nullptr;
cudaDeviceProp deviceProp;
static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;
static cudaEvent_t cuStart, cuEnd;

#define fuMM matmul_forward_cublaslt

static void cuDNNCheck(cudnnStatus_t error, const char *file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

static void checkCudnnFE(const fe::error_object& e, const char *file, int line) {
    if(!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    Stats_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID
};

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int,int,int,int, int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;

// Loosely based on cuDNN frontend samples functions and massively simplified
auto lookup_cache_or_build_graph_fwd(int B,int H,int T,int HS, int is_inference_only) {

    static cache_type_fwd user_maintained_cache_fwd;

    auto key = std::make_tuple(B, H, T, HS, is_inference_only);

    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(ENABLE_BF16)
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_uid(Q_UID)
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_uid(K_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_uid(V_UID)
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_uid(Attn_scale_UID)
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1})
                               .set_uid(Stats_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    // Reallocate the workspace if the required size is greater than the current workspace
    // In H100 this may be around 16B
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_fwd.insert({key, graph});

    return graph;
}

auto lookup_cache_or_build_graph_bwd(int B, int NH, int T, int HS) {
    static cache_type_bwd user_maintained_cache_bwd;

    auto key = std::make_tuple(B, NH, T, HS);

    auto it = user_maintained_cache_bwd.find(key);
    if (it != user_maintained_cache_bwd.end()) {
        return it->second;
    }

    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(ENABLE_BF16)
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif
    // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}));
    auto O = graph->tensor(fe::graph::Tensor_attributes().set_name("O")
                            .set_dim({B, NH, T, HS})
                            .set_uid(O_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO")
                            .set_dim({B, NH, T, HS})
                            .set_uid(dO_UID)
                            .set_stride({NH * HS * T, HS, NH * HS, 1}));

    auto stats = graph->tensor(fe::graph::Tensor_attributes().set_name("stats")
                            .set_dim({B, NH, T, 1})
                            .set_uid(Stats_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_uid(Attn_scale_UID)
                            .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(true) // 1.5+ needs this for determinism
#endif
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride({3 * NH * HS * T, HS, 3 * NH * HS, 1}).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    user_maintained_cache_bwd.insert({key, graph});
    return graph;
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<int64_t , void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {Attn_scale_UID, &attn_scale_cpu}, {O_UID, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[Stats_UID] = stats;
    }

    // Execute graph
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();
    int HS = C / NH; // number of features per head

    // Get graph and tensors from cache (or generate it on first use)
    auto graph = lookup_cache_or_build_graph_bwd(B, NH, T, HS);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = qkvr;
    void* devPtrK = (qkvr + NH * HS);
    void* devPtrV = (qkvr + 2 * NH * HS);
    void* devPtrO = o;
    void* devPtrdO = dout;
    void* devPtrStats = stats;
    float attn_scale_cpu = 1.0 / sqrtf(HS);

    void* devPtrdQ = dqkvr;
    void* devPtrdK = (dqkvr + NH * HS);
    void* devPtrdV = (dqkvr + 2 * NH * HS);

    // Build variant pack that links each tensor to its data pointer
    std::unordered_map<int64_t, void*> variant_pack = {
        {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, devPtrdO}, {Stats_UID, devPtrStats},
        {dQ_UID, devPtrdQ}, {dK_UID, devPtrdK}, {dV_UID, devPtrdV},
        {Attn_scale_UID, &attn_scale_cpu}};

    // Execute graph
    cuDNNCheck(cudnnSetStream(cudnn_handle, stream));
    checkCudnnFE(graph->execute(cudnn_handle, variant_pack, cudnn_workspace));
    cudaCheck(cudaGetLastError());
}

void create_cudnn() {
    cuDNNCheck(cudnnCreate(&cudnn_handle));
}

void destroy_cudnn() {
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}

bool InitCUDNN(const CLI_params&hparams,int flag){
    //  cudaDriverGetVersion
    //  cudaRuntimeGetVersion
    int local_device_idx = 0, override_enable_tf32 = 1;
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[InitCUDNN] failed at cudaSetDevice! ERR=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaCheck(cudaGetDeviceProperties(&deviceProp, local_device_idx));
    if (1) {
        printf("[System]\n");
        printf("Device %d: %s\n", local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = FLOAT_TYPE == typNUMBER::F32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    create_cudnn();
/*
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log dir        | %-50s |\n", output_log_dir == NULL ? "NULL" : output_log_dir);
    printf("| checkpoint_every      | %-50d |\n", checkpoint_every);
    printf("| resume                | %-50d |\n", resume);
    printf("| micro batch size B    | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| total batch size      | %-50d |\n", total_batch_size);
    printf("| LR scheduler          | %-50s |\n", lr_scheduler_type);
    printf("| learning rate (LR)    | %-50e |\n", learning_rate);
    printf("| warmup iterations     | %-50d |\n", warmup_iterations);
    printf("| final LR fraction     | %-50e |\n", final_learning_rate_frac);
    printf("| weight decay          | %-50e |\n", weight_decay);
    printf("| skip update lossz     | %-50f |\n", skip_update_lossz);
    printf("| skip update gradz     | %-50f |\n", skip_update_gradz);
    printf("| max_steps             | %-50d |\n", max_steps);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("| overfit_single_batch  | %-50d |\n", overfit_single_batch);
    printf("| use_master_weights    | %-50s |\n", use_master_weights ? "enabled" : "disabled");
    printf("| gelu_fusion           | %-50d |\n", gelu_fusion);
    printf("| recompute             | %-50d |\n", recompute);*/
    printf("+-----------------------+----------------------------------------------------+\n");
    const char* precision_str = (FLOAT_TYPE == typNUMBER::F32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (FLOAT_TYPE == typNUMBER::F16 ? "FP16" : "BF16");
    printf("| device                | %-50s |\n", deviceProp.name);
    // printf("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, FLOAT_TYPE));
    printf("| precision             | %-50s |\n", precision_str);
    printf("+-----------------------+----------------------------------------------------+\n");

    cudaCheck(cudaEventCreate(&cuStart));
    cudaCheck(cudaEventCreate(&cuEnd));
    cudaCheck(cudaProfilerStart());
    return true;
}

hGTensor cuTensor::_Multiply(const hGTensor& b) {
    cuTensor *cuB=dynamic_cast<cuTensor *>(b.get());
    assert(cuB!=nullptr);
    return nullptr;
}

int Embed::FUSE_cuda(hGTensor tokens,floatX *scratchX,LayerNormal*neuron_x,unsigned int seed){
try{
    int OC=w->ne[1],nCls = shape[1];    
    assert(C==w->ne[0]);
    const int* inp=(int*)(tokens->data);
    // assert(isInRange(inp,token->size(),0,nCls));
    if(isForward()){ 
        //   cur = w->GetRow(out,tokens,b);       
        encoder_forward(ToX(out), inp, ToX(w), ToX0(b), B, T, C, main_stream);
        // PrintTensor<floatX>("wte",params.wte,true,Vp,C);        PrintTensor<floatX>("wpe",params.wpe,true,T,C);
        // PrintTensor<int>("inputs",model->inputs,true,B,T);      PrintTensor<floatX>("GetRow",ToX(embed->out),true,B,T,C);
    }else{        
        floatX* dresidual = ToX(GTensor::scratch_btc);
        encoder_backward(ToG(w), ToG0(b), scratchX, workload_indices, bucket_info,dresidual, inp, hostInput, B, T, C, seed, main_stream);
        // g_dump_level = 0;                 
    // PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
    // g_dump_level = 1;
    // PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
    }
    return 0x0;
}catch(...){
    assert(0);
    return -1;
}
}

int SLP::FUSE_cuda(hGTensor rhs_,hGTensor lhs_,hGTensor gelu,bool isForw,int flag){
try{
    floatX *rhs=ToX(rhs_),*inp=ToX(lhs_);
    int OC=w->ne[1],IC=w->ne[0];
    // assert(C==w->ne[0]);
    assert(rhs_->ne[0]*rhs_->ne[1]*rhs_->ne[2]>=B*T*OC);        //  ne of scatch
    floatX *pre_gelu = ToX0(gelu);
    float* dbias_buffer=nullptr;
    if(isForw){ //Forward or remate in Back
        // matmul_forward_cublaslt(rhs, inp, ToX(w), ToX0(b), B, T, C, OC, main_stream,pre_gelu,gelu_fusion);
        if (gelu_fusion < 1 && pre_gelu) {
            matmul_cublaslt(pre_gelu, ToX(w), inp, ToX0(b), OC, B*T, IC, main_stream, true, false, 0, 0, 0, 0, false, NULL, false);
            gelu_forward(rhs, pre_gelu, B*T*OC, main_stream);
        } else {
            matmul_cublaslt(rhs, ToX(w), inp, ToX0(b), OC, B*T, IC, main_stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
        }
        // PrintTensor<floatX>("l_qkvw",l_qkvw,true,3*C,C);       PrintTensor<floatX>("l_qkvb",l_qkvb,true,3*C,1);
        // PrintTensor<floatX>("l_qkvr",l_qkvr,true,B,T,3*C);
    }else{
/*
      floatX* dinp, floatX* dweight, floatX* dbias,floatX* dout, floatX* inp, floatX* weight,float* dbias_buffer,
*/
       matmul_backward(rhs, ToG(w), ToG0(b), ToG(rhs_), inp, ToX(w), dbias_buffer, B, T, IC, OC, main_stream); 
    }
    return 0x0;
}catch(...){
    assert(0);
    return -1;
}
}
int SLP::FUSE_cuda_block(hGTensor rhs,hGTensor lhs,hGTensor gelu,bool isForw,int flag){
    return 0x0;
}

/*
    QKV = (B, T, 3, NH, HS) 
*/
int ROPE::FUSE_cuda(hGTensor QKV,bool isFX,int flag){
    if(Empty()){
        return -1;
    }
    int NH=n_head;
    floatX *q=nullptr,*k=nullptr;
    const int64_t ne00 = QKV->ne[0]/3,ne01 = QKV->ne[1],ne02 = QKV->ne[2]; // num heads    
    const int64_t nr = ne01*ne02*QKV->ne[3];  //ggml_nrows(src0);    
    // void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) 
    /*const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);*/   
    floatX* devPtrQ = ToX(QKV),*devPtrK = devPtrQ + C,*devPtrV = devPtrQ + 2 * C;

    const size_t s01 = QKV->ld(1);      //src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = QKV->ld(2);      //src0->nb[2] / ggml_type_size(src0->type);
    const int n_dims     = n_rot;
    float freq_base=10000.0,freq_scale=1,ext_factor=0,attn_factor=1,beta_fast=32,beta_slow=1;
    const int32_t * pos = nullptr;  //(const int32_t *) src1_d;
    const float * freq_factors = nullptr;
    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
    floatX* dst = ToX(out);
    if(isForward()){
        rope_neox_cuda<true>(
            ToX(QKV),dst,  ne00, ne01, s01, s02, n_dims, nr, pos, 
                freq_scale,freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
    }else{
        rope_neox_cuda<false>(
            ToX(QKV),dst,  ne00, ne01, s01, s02, n_dims, nr, pos, 
                freq_scale,freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
    }
    /*if (QKV->type == GGML_TYPE_F32) {
        rope_neox_cuda<forward>(
            (const float *) src0_d, ToX(QKV), ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
            freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
    } else if (src0->type == GGML_TYPE_F16) {
        rope_neox_cuda<forward>(
            (const half *) src0_d, (half *) dst_d, ne00, ne01, s01, s02, n_dims, nr, pos, freq_scale,
            freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
    } else {
        assert(0 && "fatal error@ROPE::FUSE_cuda");
    }   */ 

    return 0x0;
}

//
hGTensor SelfAttention::FUSE_cuda(hGTensor inpL,floatX* residual,float* scratchF,int flag){    
    int NH=n_head;
    floatX *qkvr=ToX(Q.out);//,*atty=ToX(attn);    
    float *l_att = TO<float>(trans); //(float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
    if(isForward()){    //  data=ToX(QKV->norm.out)
        hGTensor QKV=remater_qkv?GTensor::scratch_ff1:Q.out;
        if(fuseNorm==nullptr){
            inpL=norm.FUSE_cuda(inpL);       
        }        
 
#ifdef ENABLE_CUDNN
        Q.FUSE_cuda(QKV,inpL);  
        rope.FUSE_cuda(QKV);        
        attention_forward_cudnn(ToX(attn), l_att, ToX(QKV), B, T, NH, C, main_stream);
#else
        // if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
        //     cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        // }
        hGTensor scrath = GTensor::scratch_bt4c;        //only forward
        Q.FUSE_cuda(scrath,inpL);          // fuMM(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
        rope.FUSE_cuda(scrath); 
        attention_forward(ToX(attn), ToX(QKV), ToX(trans), ToX(scrath), B, T, C, NH, main_stream);
#endif
        PrintTensor<floatX>("l_atty",ToX(attn),true,B,T,C);
        floatX *pw=ToX(proj_cat.w), *pb=ToX0(proj_cat.b);
        floatX* scratch = ToX(GTensor::scratch_output),*ouput=(floatX *)out->data;
        proj_cat.FUSE_cuda(GTensor::scratch_output,attn);   //fuMM(scratch, ToX(attn), pw, pb, B, T, C, C, main_stream);       
                
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        residual_forward(ouput, residual, scratch, B*T*C, main_stream);
        if(fuseNorm!=nullptr){
            float *mean=TO<float>(fuseNorm->mean),*rstd=TO<float>(fuseNorm->rstd);
            layernorm_forward(ToX(fuseNorm->out), mean, rstd, ouput,ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, 1, C, main_stream);
        }           
    }else{
        floatX* dl_bt4c = ToX(GTensor::scratch_bt4c),*dresidual = ToX(GTensor::scratch_btc),
            *gQb = Q.b==nullptr?nullptr:ToG(Q.b),*gNb=norm.b==nullptr?nullptr:ToG(norm.b);  
        if(remater_qkv)  {   
            qkvr=ToX(GTensor::scratch_ff1);
            //  scratch_ff1 = inpL*Q.w+Q.b
            Q.FUSE_cuda(GTensor::scratch_ff1,inpL); // fuMM(qkvr, data, weight, bias, B, T, C, 3*C, main_stream);
        }
#ifdef ENABLE_CUDNN
        attention_backward_cudnn(dl_bt4c, dl_btc, qkvr, ToX(attn), l_att, B, T, NH, C, main_stream);
#else
        assert(0);
#endif
        PrintTensor<floatX>("back of attn",dl_bt4c,true,B,T,C);
        // if(model->recompute >= 2) {
        //     layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        // }
        // Q.FUSE_cuda()
        matmul_backward(dl_btc, ToG(Q.w), gQb, dl_bt4c, ToX(norm.out), ToX(Q.w), scratchF, B, T, C, 3 * C, main_stream);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        layernorm_backward(dresidual, ToG(norm.w), gNb, scratchF, dl_btc, residual, ToX(norm.w), TO<float>(norm.mean), TO<float>(norm.rstd), B, T, C, main_stream);
    }
    return out;
}

//  hIn = QKV->out
hGTensor FFN::FUSE_cuda(hGTensor hIn,floatX *scratch,int flag){
    floatX *ff2=ToX(down.out),*ff1=ToX(up.out);
    // float *mean=TO<float>(fuseNorm->mean),*rstd=TO<float>(fuseNorm->rstd);floatX   *normed=ToX(fuseNorm->out);
    hGTensor tGelu=GTensor::scratch_output;
    bool isBias = up.b!=nullptr;  
    
    if(isForward()){  
        if(fuseNorm==nullptr){
            norm.FUSE_cuda(hIn);       
        }
        floatX * inp1_ = ToX(norm.out);         
        if(remater_ffn)  {
            input_1 = inp1_;
            ff1=ToX(GTensor::scratch_ff1);              
        } 
        assert(ff1!=nullptr);       // ff1=gelu_forward(out, l_fch_gelu, B*T*OC, stream);
        floatX *scratch = ToX(GTensor::scratch_btc);    
        up.FUSE_cuda(tGelu,norm.out,remater_ffn?GTensor::scratch_ff1:up.out);        // fuMM(l_fch_gelu,inp1_, (floatX*)up.w->data,ToX0(up.b), B, T, C, latent, main_stream, ff1, gelu_fusion);
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,1,-1);  
        down.FUSE_cuda(GTensor::scratch_btc,tGelu);        // fuMM(scratch, l_fch_gelu, (floatX*)down.w->data, ToX0(down.b), B, T, latent, C, main_stream);   //???
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);
        PrintTensor<floatX>("ffn",scratch,true,B,T,C);

        // fused_residual_forward5(ToX(out), normed,mean,rstd, ToX(hIn), scratch, ToX(fuseNorm->w), xb, B*T, C, main_stream);
        residual_forward(ToX(out), ToX(hIn), scratch, B*T*C, main_stream);
        if(fuseNorm!=nullptr){
            return fuseNorm->FUSE_cuda(out);   
            // layernorm_forward(ToX(fuseNorm->out), TO<float>(fuseNorm->mean),TO<float>(fuseNorm->rstd), ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, 1, C, main_stream);
            // return fuseNorm->out;
        }
        
        // PrintTensor<floatX>("inp1",ToX(norm.out),true,B,T,C,1,-1);
        out->PrintX<floatX>("residual3",0,0);
    }else{
        floatX *dl_bt4c = ToX(GTensor::scratch_bt4c),*dresidual = ToX(GTensor::scratch_btc)
            ,*gPb=lastQKV->proj_cat.b==nullptr?nullptr:ToG(lastQKV->proj_cat.b),*gNb=norm.b==nullptr?nullptr:ToG(norm.b); 
        float*  scratchF = (float*) scratch;   // not the same inp1 of forward !!!
        if(input_1!=nullptr){
            input_1 =  ToX(norm.out);
            ff1=ToX(GTensor::scratch_ff1);  
            up.FUSE_cuda(tGelu,norm.out,GTensor::scratch_ff1);            
            // fuMM(l_fch_gelu,input_1, (floatX*)up.w->data, ToX0(up.b), B, T, C, latent, main_stream, ff1, gelu_fusion);
            // norm.out->PrintX<floatX>("inp1",0,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,-1);  
        }else
            gelu_forward(ToX(tGelu), ff1, B*T*latent, main_stream);  
        assert(ff1!=nullptr);   
        matmul_backward(dl_bt4c, ToG(down.w), ToG0(down.b), dresidual, ToX(tGelu), ToX(down.w), scratchF, B, T, latent, C, main_stream, ff1, gelu_fusion);
        PrintTensor<floatX>("back of ffn1",dl_bt4c,true,B,T,latent);
        
        matmul_backward(residual, ToG(up.w), ToG0(up.b), dl_bt4c, ToX(norm.out), ToX(up.w), scratchF, B, T, C, latent, main_stream);
        // // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        layernorm_backward(dresidual, ToG(norm.w), gNb, scratchF, residual, ToX(hIn), ToX(norm.w), TO<float>(norm.mean), TO<float>(norm.rstd), B, T, C, main_stream);
        matmul_backward(residual, ToG(lastQKV->proj_cat.w), gPb, dresidual, ToX(lastQKV->attn), ToX(lastQKV->proj_cat.w), scratchF, B, T, C, C, main_stream);
        PrintTensor<floatX>("back of ffn0",residual,true,B,T,C);
    }
    
    return out;
}

/*
    layernorm_forward(floatX* out, float* mean, float* rstd, floatX* inp, const floatX* weight, const floatX* bias,         int B, int T, int C, cudaStream_t stream)
    layernorm_backwar(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,const floatX* dout, const floatX* inp, const floatX* weight, const float* mean, const float* rstd,          int B, int T, int C, cudaStream_t stream)
*/
hGTensor cuTensor::Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward,int flag) {
    assert(!hOut->isEmpty());
    int B=hOut->ne[0],T=hOut->ne[1],C=w->ne[0];
    // assert(b!=nullptr);     
    floatX *weight=(floatX*)(w->data),*bias=ToX0(b);    //b==nullptr?nullptr:(floatX*)(b->data);    
    floatX *out=(floatX*)(hOut->data); // (B, T, C)
    if(isForward)
        layernorm_forward(out, (float*)_mean->data, (float*)_rstd->data, (floatX *)data,weight,bias, B, T, C, main_stream);
    else{
        layernorm_backward(nullptr, (floatX*)(w->grad), ToG0(b), nullptr, nullptr,nullptr, weight, 
            (float*)_mean->data, (float*)_rstd->data, B, T, C, main_stream);
    }
    
    return hOut;
}

hGTensor LayerNormal::FUSE_cuda(hGTensor inpL,float* scratch,int flag) {
    floatX *inp = ToX(inpL);
    if(isForward()){    //cur = cur->Normal(out,mean,rstd,w,b);   
        layernorm_forward(ToX(out), TO<float>(mean),  TO<float>(rstd), inp,ToX(w),ToX0(b), B, T, C, main_stream);
    }        
    else{   //  layernorm_backward(dresidual, ToG(lnf->w), gb, (float*)scratchX, ToX(GTensor::scratch_bt4c), residual, ToX(lnf->w), TO<float>(lnf->mean), TO<float>(lnf->rstd), B, T, C, main_stream);
        const floatX* dout=ToX(GTensor::scratch_bt4c);
        floatX* dresidual = ToX(GTensor::scratch_btc);
        layernorm_backward(dresidual, ToG(w), ToG0(b), scratch, dout,inp, ToX(w), TO<float>(mean),  TO<float>(rstd), B, T, C, main_stream);
        PrintTensor<floatX>("back of normal",dresidual,true,B,T,C);
    }
    return out;
}

//void fused_classifier(Type* logits, float* cuLoss,const float dloss, const int* targets,int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
//float cuTensor::FusedLoss(float dloss,hGTensor hLoss,hGTensor hTarget,hGTensor hLastLayer, hGTensor w,int V,bool isForward,int flag){
hGTensor OutCLS::FUSE_cuda(hGTensor inpL,hGTensor token_embed,int flag)   {
    int V=nCls,Vp=padded_nCls, gelu_fusion=1;
    assert(proj.b==nullptr);
    mean_loss = 0.0f;
    const int *targets=(int*)(target->data);
    float* cuLoss = (float*)out->data;   
    floatX* errLogits = ToX(preLogits),*z0=ToX(inpL),*w=nullptr,*gw=nullptr,*pre_gelu=nullptr;  
    floatX* errOut = ToX(GTensor::scratch_bt4c);   //B * T * 4 * C
    if(proj.w==nullptr){ //  isEmbedWeightTying
        w=ToX0(token_embed);         gw=ToG0(token_embed);
    }else{
        w=ToX(proj.w);         gw=ToG(proj.w);
    }
    if(isForward()){        
        // cudaCheck(cudaDeviceSynchronize());         
        // cudaCheck(cudaMemset(cuLoss, 0, B*T*sizeof(float)));
        cudaCheck(cudaMemset(cuLoss, 0, B*T*sizeof(float)));
        assert( target->isSameShape(out) );
        constexpr std::bool_constant<true> cuFalse;    
        for(size_t i=0;i<B;i+=dB){
            size_t off=i*T*Vp,n1=i*T,nZ=i*T*C;
            off=0;      //reduce memory
            // proj.FUSE_cuda_block(preLogits,inpL);
            fuMM(errLogits+off, z0+nZ, w, NULL, dB, T, C, Vp, main_stream);  //[32,1024,50304]=[32,1024,768]*[768,50304]
            fused_classifier(errLogits+off, cuLoss+n1, rLoss, targets+n1, dB, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
            if(gw!=nullptr && errOut!=nullptr){
                matmul_cublaslt(errOut+nZ, w, errLogits+off, NULL, C, dB*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);   
                matmul_cublaslt(gw, z0+nZ, errLogits+off, NULL /*dbias*/, C, Vp, dB*T, main_stream, false, true, 0, 0, 0, 0,true /* accumulate */, NULL, true);                
            }                         
        }
        // fused_classifier(errLogits, cuLoss, rLoss, targets, B, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
        cudaCheck(cudaMemcpy(hostLoss, cuLoss, B * T * sizeof(float), cudaMemcpyDeviceToHost));                 
        cudaCheck(cudaDeviceSynchronize());
        /*if(flag==0x1001 && gw!=nullptr && errOut!=nullptr){            //matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);      //accumulate=true  
            matmul_cublaslt(errOut, w, errLogits, NULL, C, B*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);
            if (gelu_fusion < 2 && pre_gelu) {
                gelu_backward_inplace(errOut, pre_gelu, B*T*C, main_stream);
            }
            matmul_cublaslt(gw, z0, errLogits, NULL , C, Vp, B*T, main_stream, false, true, 0, 0, 0, 0,true , NULL, true);
        }*/
            
        for (int i = 0; i < B*T; i++) {
            mean_loss += hostLoss[i];
        }   
        mean_loss /= B*T;
    }else{        
        // matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);
        return inpL;
    }
    cudaCheck(cudaGetLastError());
    return preLogits;
}

// #define ENABLE_CUDNN

cuTensor::~cuTensor()  {
    Free();

}

