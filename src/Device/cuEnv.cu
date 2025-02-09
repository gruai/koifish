#include "../CLI_params.hpp"
#include "../ggex/GTensor.hpp"
#include "../g_stddef.hpp" 
// #include "../ggex/GG_util.hpp"       //ugly  "__builtin_ia32_ldtilecfg" is undefined
#include "../kGPT/llmc/cuda_common.h"
#include "../kGPT/llmc/cublas_common.h"
#include "../kGPT/llmc/matmul.cuh"
#include "../kGPT/llmc/layernorm.cuh"
#include "../kGPT/llmc/encoder.cuh"
#include "../kGPT/llmc/fused_classifier.cuh"
#include "../Manifold/Neuron.hpp"
// #include "../kGPT/llmc/mfu.h"
#define NOMINMAX
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
// Specific configurations based on the enabled precision
#if defined(ENABLE_FP32)
    static_assert(false, "cuDNN is not supported in FP32 mode.")
    // use fp16 (note: this may require gradient scaler, currently not implemented!)
#elif defined(ENABLE_FP16)
    #define CUDNN_16BIT fe::DataType_t::HALF
#else // Default to bfloat16
    #define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

cudaStream_t main_stream=nullptr;
cudaDeviceProp deviceProp;
static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;
static cudaEvent_t cuStart, cuEnd;

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
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

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
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

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
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
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
    const char* precision_str = (PRECISION_MODE == PRECISION_FP32)
                              ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32")
                              : (PRECISION_MODE == PRECISION_FP16 ? "FP16" : "BF16");
    printf("| device                | %-50s |\n", deviceProp.name);
    // printf("| peak TFlops           | %-50.1f |\n", get_flops_promised(deviceProp.name, PRECISION_MODE));
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
    // floatX* out = nullptr;
    // matmul_forward_cublaslt((floatX*)out, (floatX*)data, (floatX*)b->data, NULL, B, T, C, Vp, main_stream);
    return nullptr;
}

/*
    1. residual_forward(residual, inp1, inp2, N*C, stream);
    2. layernorm_forward(normed, mean, rstd, residual, weight, bias, N, 1, C, stream);
*/
int FUSE_ResiNormal(hGTensor hOut,hGTensor hInp1,hGTensor hInp2,hGTensor hNormed,hGTensor N_mean,hGTensor N_rstd,hGTensor w,hGTensor b,int flag){
    assert(b!=nullptr);     //I would fix this bug
    assert(hOut->isSameShape(hInp1) && hOut->isSameShape(hInp2));
    int B=GTensor::B,T=GTensor::T,C=GTensor::C;
    floatX *inp1=(floatX*)hInp1->data,*inp2=(floatX*)hInp2->data,*normed=(floatX*)hNormed->data,*residual=(floatX*)hOut->data;
    assert(N_mean->type==GGML_TYPE_F32 && N_rstd->type==GGML_TYPE_F32);
    float *mean=(float*)N_mean->data,*rstd=(float*)N_rstd->data;
    if(flag==1)
        fused_residual_forward5((floatX*)hOut->data, normed, mean,rstd, inp1, inp2, (floatX*)w->data, (floatX*)b->data, B*T, C, main_stream);
    else{
        residual_forward(residual, inp1, inp2, B*T*C, main_stream);
        layernorm_forward(normed, mean, rstd, residual, (floatX*)w->data, (floatX*)b->data, B*T, 1, C, main_stream);        
    }

    return 0x0;
}

int FUSE_QKV(hGTensor hOut,hGTensor hIn,hGTensor hQKV,hGTensor hATTN,hGTensor w,hGTensor b,int NH,hGTensor proj_w,hGTensor proj_b,int flag) {
    // float* l_att = (float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
    //floatX*out,floatX* inp, floatX* weight, floatX* bias,int B, int T, int C, int OC, cudaStream_t stream,floatX* pre_gelu=NULL, int gelu_fusion=1
    int B=GTensor::B,T=GTensor::T,C=GTensor::C;
    floatX *weight=(floatX*)w->data, *bias=b==nullptr?nullptr:(floatX*)b->data;
    floatX *out=(floatX*)hOut->data,*qkv=(floatX*)hQKV->data,*attn=(floatX*)hATTN->data;
    floatX *data=(floatX*)hIn->data;
    float *stats=nullptr;
    matmul_forward_cublaslt(qkv, (floatX *)(data), weight, bias, B, T, C, 3*C, main_stream);
    //  floatX* out,  float* stats, floatX* inp,  int B, int T, int NH, int C, cudaStream_t stream
    attention_forward_cudnn(attn, stats, qkv, B, T, NH, C, main_stream);

    floatX *pw=(floatX*)proj_w->data, *pb=proj_b==nullptr?nullptr:(floatX*)proj_b->data;
    matmul_forward_cublaslt(out, attn, pw, pb, B, T, C, C, main_stream);
    
    return 0x0;
}  

int SelfAttention::FUSE_cuda(hGTensor inpL,int flag){
    int B=GTensor::B,T=GTensor::T,C=GTensor::C,NH=n_head;
    floatX *weight=ToX(Q.w), *bias=Q.b==nullptr?nullptr:ToX(Q.b),*qkv=ToX(Q.out);
    floatX *inp1=ToX(proj_cat.out),*atty=ToX(attn),*normed=ToX(norm.out),*data=ToX(inpL),*inp2=data,*resi=(floatX *)out->data;
    float *mean=TO<float>(norm.mean),*rstd=TO<float>(norm.rstd);
    float *stats=nullptr;    
    if(isForward()){      
        // FUSE_QKV(proj_cat.out,inpL,Q.out,attn,Q.w,Q.b,n_head,proj_cat.w,proj_cat.b,0x0 );        cur=proj_cat.out;
        matmul_forward_cublaslt(qkv, (floatX *)(data), weight, bias, B, T, C, 3*C, main_stream);
        attention_forward_cudnn(atty, stats, qkv, B, T, NH, C, main_stream);
        floatX *pw=ToX(proj_cat.w), *pb=proj_cat.b==nullptr?nullptr:ToX(proj_cat.b);
        matmul_forward_cublaslt(inp1, atty, pw, pb, B, T, C, C, main_stream);        
        // iRet = FUSE_ResiNormal(out,down.out,lastResi,norm.out,norm.mean,norm.rstd,norm.w,norm.b,0x0);   cur = norm.out;
        if(flag==1)
            fused_residual_forward5(resi, normed,mean,rstd, inp1, inp2, ToX(norm.w), ToX(norm.b), B*T, C, main_stream);
        else{
            residual_forward(resi, inp1, inp2, B*T*C, main_stream);
            layernorm_forward(normed,mean,rstd, inp1, ToX(norm.w), ToX(norm.b), B*T, 1, C, main_stream);        
        }
    }else{
    }
    return 0x0;
}

int FFN::FUSE_cuda(hGTensor hIn,int flag){
    int B=GTensor::B,T=GTensor::T,C=GTensor::C;
    floatX *inp1=ToX(down.out),*x=ToX(up.out),*normed=ToX(norm.out);
    float *mean=TO<float>(norm.mean),*rstd=TO<float>(norm.rstd);
    floatX *data=(floatX*)hIn->data,*inp2=data,*resi=(floatX *)out->data;
    bool isBias = up.b!=nullptr;  assert(isBias);
    if(isForward()){      
        // iRet = FUSE_FFN(down.out,cur,up.out,up.w,up.b,relu.out,down.w,down.b,gelu_fusion,0);            cur = down.out;    
        matmul_forward_cublaslt(x, (floatX*)data, (floatX*)up.w->data, (floatX*)up.b->data, B, T, C, 4*C, main_stream, (floatX*)relu.out->data, gelu_fusion);
        matmul_forward_cublaslt(inp1, x, (floatX*)down.w->data, (floatX*)down.b->data, B, T, 4*C, C, main_stream);        
        // iRet = FUSE_ResiNormal(out,down.out,lastResi,norm.out,norm.mean,norm.rstd,norm.w,norm.b,0x0);   cur = norm.out;
        if(flag==1)
            fused_residual_forward5(resi, normed,mean,rstd, inp1, inp2, ToX(norm.w), ToX(norm.b), B*T, C, main_stream);
        else{
            residual_forward(resi, inp1, inp2, B*T*C, main_stream);
            layernorm_forward(normed,mean,rstd, inp1, ToX(norm.w), ToX(norm.b), B*T, 1, C, main_stream);        
        }
    }else{
        // floatX* dl_bt4c = (floatX*)model->acts.scratch_bt4c;

        // // start the backward pass for this layer
        // if(model->recompute >= 1) {
        //     // recompute >= 1 means we recompute gelu. in this case,
        //     // l_fch_gelu is just a buffer, so re-compute the gelu from l_fch here
        //     gelu_forward(l_fch_gelu, l_fch_pre_gelu, B*T*4*C, main_stream);
        // }
        // matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_gelu, l_fcprojw, scratchF, B, T, 4*C, C, main_stream, l_fch_pre_gelu, model->gelu_fusion);
        // if(model->recompute >= 2) {
        //     // same as gelu above, l_ln1 and l_ln2 are just buffers if recompute >= 2, recompute them here on demand
        //     layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, main_stream);
        // }
        // matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, scratchF, B, T, C, 4 * C, main_stream);
        // // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        // layernorm_backward(dresidual, dl_ln2w, dl_ln2b, scratchF, dl_btc, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, main_stream);
        // matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, scratchF, B, T, C, C, main_stream)
    }
    
    return 0x0;
}

int FUSE_FFN(hGTensor hOut,hGTensor hIn,hGTensor hLatent,hGTensor wUp,hGTensor bUp,hGTensor hGelu,hGTensor wDown,hGTensor bDown,int gelu_fusion,int flag) {
    int B=GTensor::B,T=GTensor::T,C=GTensor::C;
    floatX *out=(floatX*)hOut->data,*x=(floatX*)hLatent->data;
    floatX *data=(floatX*)hIn->data;
    bool isBias = bUp!=nullptr;  assert(isBias);
    matmul_forward_cublaslt(x, (floatX*)data, (floatX*)wUp->data, (floatX*)bUp->data, B, T, C, 4*C, main_stream, (floatX*)hGelu->data, gelu_fusion);
    matmul_forward_cublaslt(out, x, (floatX*)wDown->data, (floatX*)bDown->data, B, T, 4*C, C, main_stream);
    return 0x0;
}

hGTensor cuTensor::GetRow(hGTensor hOut,hGTensor token,hGTensor pos,int flag)   {
    floatX *out=(floatX*)(hOut->data),*wte=(floatX*)(data),*wpe=pos==nullptr?nullptr : (floatX*)(pos->data);
    // int nCls = shape[1],i;
    const int* inp=(int*)(token->data);
    // assert(isInRange(inp,token->size(),0,nCls));

    encoder_forward(out, inp, wte, wpe, B, T, C, main_stream);
    return hOut;
}

/*
    layernorm_forward(floatX* out, float* mean, float* rstd, floatX* inp, const floatX* weight, const floatX* bias,         int B, int T, int C, cudaStream_t stream)
    layernorm_backwar(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,const floatX* dout, const floatX* inp, const floatX* weight, const float* mean, const float* rstd,          int B, int T, int C, cudaStream_t stream)
*/
hGTensor cuTensor::Normal(hGTensor hOut,hGTensor _mean,hGTensor _rstd,hGTensor w,hGTensor b,bool isForward,int flag) {
    assert(!hOut->isEmpty());
    
    assert(b!=nullptr);     //I would fix this bug
    floatX *weight=(floatX*)(w->data),*bias=b==nullptr?nullptr:(floatX*)(b->data);    
    floatX *out=(floatX*)(hOut->data); // (B, T, C)
    if(isForward)
        layernorm_forward(out, (float*)_mean->data, (float*)_rstd->data, (floatX *)data,weight,bias, B, T, C, main_stream);
    else{
        layernorm_backward(nullptr, (floatX*)(w->grad), (floatX*)(b->grad), nullptr, nullptr,nullptr, weight, 
            (float*)_mean->data, (float*)_rstd->data, B, T, C, main_stream);
    }
    
    return hOut;
}

//void fused_classifier(Type* logits, float* cuLoss,const float dloss, const int* targets,int B, int T, int V, int P, std::bool_constant<WriteDLogits> write_dlogits, cudaStream_t stream) {
//float cuTensor::FusedLoss(float dloss,hGTensor hLoss,hGTensor hTarget,hGTensor hLastLayer, hGTensor w,int V,bool isForward,int flag){
int OutCLS::FUSE_cuda(hGTensor inpL,hGTensor token_embed,int flag)   {
    int B=GTensor::B,T=GTensor::T,C=GTensor::C,V=nCls,Vp=padded_nCls, gelu_fusion=1;
    assert(proj.b==nullptr);
    mean_loss = 0.0f;
    const int *targets=(int*)(target->data);
    float* cuLoss = (float*)out->data;   
    floatX* errLogits = ToX(preLogits),*z0=ToX(inpL),*w=nullptr,*gw=nullptr,*pre_gelu=nullptr;  
    floatX* errOut = ToX(GTensor::scratch_bt4c);   //B * T * 4 * C
    if(isSymProj){
        assert(proj.w==nullptr && token_embed!=nullptr);
        w=ToX(token_embed);         gw=ToG(token_embed);
    }else{
        assert(proj.w!=nullptr);
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
            matmul_forward_cublaslt(errLogits+off, z0+nZ, w, NULL, dB, T, C, Vp, main_stream);  //[32,1024,50304]=[32,1024,768]*[768,50304]
            fused_classifier(errLogits+off, cuLoss+n1, rLoss, targets+n1, dB, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
            if(flag!=0x1001 && gw!=nullptr && errOut!=nullptr){
                matmul_cublaslt(errOut+nZ, w, errLogits+off, NULL, C, dB*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);   
                matmul_cublaslt(gw, z0+nZ, errLogits+off, NULL /*dbias*/, C, Vp, dB*T, main_stream, false, true, 0, 0, 0, 0,true /* accumulate */, NULL, true);                
            }                         
        }
        // fused_classifier(errLogits, cuLoss, rLoss, targets, B, T, V, Vp, cuFalse, main_stream);        //target=[32,1024]
        cudaCheck(cudaMemcpy(hostLoss, cuLoss, B * T * sizeof(float), cudaMemcpyDeviceToHost));                 
        cudaCheck(cudaDeviceSynchronize());
        if(flag==0x1001 && gw!=nullptr && errOut!=nullptr){            //matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);      //accumulate=true  
            matmul_cublaslt(errOut, w, errLogits, NULL, C, B*T, Vp, main_stream, false, false, 0, 0, 0, 0, false,gelu_fusion >= 2 ? pre_gelu : NULL, true);
            if (gelu_fusion < 2 && pre_gelu) {
                gelu_backward_inplace(errOut, pre_gelu, B*T*C, main_stream);
            }
            matmul_cublaslt(gw, z0, errLogits, NULL /*dbias*/, C, Vp, B*T, main_stream, false, true, 0, 0, 0, 0,true /* accumulate */, NULL, true);
        }
            
        for (int i = 0; i < B*T; i++) {
            mean_loss += hostLoss[i];
        }   
        mean_loss /= B*T;
    }else{        
        // matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

// #define ENABLE_CUDNN

cuTensor::~cuTensor()  {
    Free();

}

/*int cuLiteTest(size_t B,size_t T,size_t C,int stage,int flag){
    int l=0,recompute=1;
    TensorSpec tensors[NUM_ACTIVATION_TENSORS];
    ActivationTensors acts;
    size_t Vp=50304,L=12,NH=12,maxT=1024;
    size_t nzP=0,param_elements[NUM_PARAMETER_TENSORS],param_sizeof[NUM_PARAMETER_TENSORS];
    param_elements[0] = Vp * C; // wte         50304*768
    param_elements[1] = maxT * C; // wpe       1024*768
    param_elements[2] = L * C; // ln1w
    param_elements[3] = L * C; // ln1b
    param_elements[4] = L * (3 * C) * C; // qkvw
    param_elements[5] = L * (3 * C); // qkvb
    param_elements[6] = L * C * C; // attprojw
    param_elements[7] = L * C; // attprojb
    param_elements[8] = L * C; // ln2w
    param_elements[9] = L * C; // ln2b
    param_elements[10] = L * (4 * C) * C; // fcw
    param_elements[11] = L * (4 * C); // fcb
    param_elements[12] = L * C * (4 * C); // fcprojw
    param_elements[13] = L * C; // fcprojb
    param_elements[14] = C; // lnfw
    param_elements[15] = C; // lnfb
    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
    ParameterTensors params,grads;
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        nzP += param_elements[i];
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }
    // malloc all parameters all at once on the device
    void* params_memory;
    printf("[cuTest] mem=%.5g(G)\tnP=%ld\n",num_parameters_bytes/1.0e9,num_parameters_bytes);
    cudaCheck(cudaMalloc((void**)&params_memory, nzP));
    if(stage==1)
        return -1;
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params.wte, &params.wpe, &params.ln1w, &params.ln1b, &params.qkvw, &params.qkvb,
        &params.attprojw, &params.attprojb, &params.ln2w, &params.ln2b, &params.fcw, &params.fcb,
        &params.fcprojw, &params.fcprojb, &params.lnfw, &params.lnfb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }

    tensors[0] = TENSOR_SPEC(acts.encoded, B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(acts.ln1,  (recompute < 2) ? L * B * T * C : 0);
    tensors[2] = TENSOR_SPEC(acts.ln1_mean, L * B * T);
    tensors[3] = TENSOR_SPEC(acts.ln1_rstd, L * B * T);
    tensors[4] = TENSOR_SPEC(acts.atty, L * B * T * C);
#ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(acts.att, L * B * NH * T);
#else
    tensors[5] = TENSOR_SPEC(acts.att, L * B * NH * T * T);
#endif
    tensors[6] = TENSOR_SPEC(acts.residual2, L * B * T * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(acts.ln2, (recompute < 2) ? L * B * T * C : 0);
    tensors[8] = TENSOR_SPEC(acts.ln2_mean, L * B * T);
    tensors[9] = TENSOR_SPEC(acts.ln2_rstd, L * B * T);
    tensors[10] = TENSOR_SPEC(acts.fch, L * B * T * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(acts.fch_gelu, (recompute < 1) ? L * B * T * 4*C : B * T * 4*C);
    tensors[12] = TENSOR_SPEC(acts.residual3, L * B * T * C);
    tensors[13] = TENSOR_SPEC(acts.lnf, B * T * C);
    tensors[14] = TENSOR_SPEC(acts.lnf_mean, B * T);
    tensors[15] = TENSOR_SPEC(acts.lnf_rstd, B * T);
    tensors[16] = TENSOR_SPEC(acts.losses, B * T);
    tensors[17] = TENSOR_SPEC(acts.qkvr, L * B * T * 3*C);
    tensors[18] = TENSOR_SPEC(acts.output, B * T * max(3*C, max(NH*T, Vp)));
    tensors[19] = TENSOR_SPEC(acts.scratch_bt4c, B * T * 4 * C);
    tensors[20] = TENSOR_SPEC(acts.scratch_btc, B * T * C);

    // acts_memory = malloc_and_point_activations(acts_specs);
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }
    // printf0("allocating %d MiB for activations\n", (int)round(bytes / (1024 * 1024)));
    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));
    cudaCheck(cudaMemset(acts_memory, 0, bytes));
    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }

    floatX* l_qkvr = acts.qkvr + l * B * T * 3*C;
    floatX* l_ln1 = acts.ln1 + l * B * T * C;
    floatX* l_qkvw = params.qkvw + l * 3*C * C;
    floatX* l_qkvb = params.qkvb + l * 3*C;
    matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C, main_stream);
    return 0x0;
}*/ 