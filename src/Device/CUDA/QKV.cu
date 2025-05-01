#include "./cuda_common.h"
#include "./cublas_common.h"
#include "./llm_c/matmul.cuh"
#include "./llm_c/layernorm.cuh"
#include "./llm_c/encoder.cuh"
#include "./llm_c/fused_classifier.cuh"
#include "./kernel/rope.cuh"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
#include "./EDevice.hpp"
#define NOMINMAX

// #undef ENABLE_CUDNN
#ifdef ENABLE_CUDNN
    #include "cudnn_frontend.h"
    #include "./llm_c/cudnn_att.h"
    namespace fe = cudnn_frontend;
    #if defined(ENABLE_FP32)
        // static_assert(false, "cuDNN is not supported in FP32 mode.")
        // use fp16 (note: this may require gradient scaler, currently not implemented!)
    #elif defined(ENABLE_FP16)
        #define CUDNN_16BIT fe::DataType_t::HALF
    #else // Default to bfloat16
        #define CUDNN_16BIT fe::DataType_t::BFLOAT16    
    #endif
#else
    // defines: attention_forward, attention_backward
    #include "./llm_c/attention.cuh"
#endif

static cudaEvent_t cuStart, cuEnd;
#define fuMM matmul_forward_cublaslt

#ifdef ENABLE_CUDNN
static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;

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
        cudnn_workspace_size = graph->get_workspace_size();     //1008599040
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
#endif


bool InitCUDA(const CLI_params&hparams,EDGE_DEVICES *hDevice,int flag){
    //  cudaDriverGetVersion
    //  cudaRuntimeGetVersion
    int local_device_idx = 0, override_enable_tf32 = 1;
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[InitCUDA] failed at cudaSetDevice! ERR=%s\n", cudaGetErrorString(err));
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
#ifdef ENABLE_CUDNN
    create_cudnn();
#endif
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
    // Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors.
    hDevice->nCore = deviceProp.multiProcessorCount;
    cudaCheck(cudaEventCreate(&cuStart));
    cudaCheck(cudaEventCreate(&cuEnd));
    cudaCheck(cudaProfilerStart());
    return true;
}

/*
    QKV = (B, T, 3, NH, HS) 
*/
int ROPE::FUSE_cuda(hGTensor QKV,bool isFX,int flag){
    if(Empty()){
        return -1;
    }
    int NH=n_head;    
    // hFish->GetBTC(B,T,C);
    floatX* devQ = ToX(QKV),*devK = devQ + C;    //*devPtrV = devPtrQ + 2 * C;

    const size_t s01 = QKV->ld(1);      //src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = QKV->ld(2);      //src0->nb[2] / ggml_type_size(src0->type);
    const int n_dims     = n_rot;
    float freq_base=10000.0,freq_scale=1,ext_factor=0,attn_factor=1,beta_fast=32,beta_slow=1;
    const int32_t * pos = nullptr;  //(const int32_t *) src1_d;
    const float * freq_factors = nullptr;
    rope_corr_dims corr_dims;
    // ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
    floatX* dst = ToX(out);
    if(isForward()){
        grid_size = CEIL_DIV(B*T*C/2, block_size);
        // encoder_forward(ToX(cur), samps, ToX(wSrc), nullptr, 1, T, C, main_stream);
        CU_rope_<<<grid_size, block_size, 0, main_stream>>>(devQ,devQ, q_dim, head_dim, theta, n_rot,B,T,C);   
        CU_rope_<<<grid_size, block_size, 0, main_stream>>>(devK,devK, kv_dim, head_dim, theta, n_rot,B,T,C);   
    }else{
        
    }

    return 0x0;
}

//
hGTensor SelfAttention::FUSE_cuda(hGTensor inpL,hGTensor residual,hGTensor deltaIn,float* scratchF,int flag){    
    int NH=n_head;
    floatX *qkvr=ToX(Q.out);    
    float *l_att = TO<float>(trans); //(float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
    if(isForward()){    //  data=ToX(QKV->norm.out)
        hGTensor QKV=remater_qkv?GTensor::tmpFF1:Q.out;
        if(fuseNorm==nullptr){
            inpL=norm.FUSE_cuda(inpL);       
        }        
 
#ifdef ENABLE_CUDNN
        Q.Forw(QKV,inpL);  
        rope.FUSE_cuda(QKV);        
        attention_forward_cudnn(ToX(attn), l_att, ToX(QKV), B, T, NH, C_qkv, main_stream);
#else
        // if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
        //     cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        // }
        hGTensor scrath = GTensor::bt4c;        //only forward
        Q.Forw(QKV,inpL);          
        rope.FUSE_cuda(scrath); 
        attention_forward(ToX(attn), qkvr, ToX(trans), ToX(QKV), B, T, C, NH, main_stream);  //  l_atty, l_qkvr, l_att, scratch
#endif
        PrintTensor<floatX>("l_atty",ToX(attn),true,B,T,C);
        // floatX *pw=ToX(proj_cat.w), *pb=ToX0(proj_cat.b),*scratch = ToX(GTensor::scratch); //*ouput=(floatX *)out->data;

        proj_cat.Forw(GTensor::scratch,attn);   //fuMM(scratch, ToX(attn), pw, pb, B, T, C, C, main_stream);       
                
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        residual_forward(ToX(out), ToX(residual), ToX(GTensor::scratch), B*T*C, main_stream);
        if(fuseNorm!=nullptr){
            float *mean=TO<float>(fuseNorm->mean),*rstd=TO<float>(fuseNorm->rstd);
            layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, 1, C, main_stream);
        }           
    }else{
        assert(delta!=nullptr);
        proj_cat.Back(deltaCat,attn,deltaIn,nullptr,scratchF);

        hGensor delta_attn = GTensor::bt4c; //* dl_bt4c = ToX(GTensor::bt4c); 
        if(remater_qkv)  {   
            qkvr=ToX(GTensor::tmpFF1);
            //  tmpFF1 = inpL*Q.w+Q.b
            Q.Forw(GTensor::tmpFF1,inpL); // fuMM(qkvr, data, weight, bias, B, T, C, 3*C, main_stream);
        }
#ifdef ENABLE_CUDNN
        attention_backward_cudnn(ToX(delta_attn), ToX(deltaCat), qkvr, ToX(attn), l_att, B, T, NH, C_qkv, main_stream);
#else
        assert(0);
#endif
        PrintTensor<floatX>("back of attn",ToX(delta_attn),true,B,T,C);
        // if(model->recompute >= 2) {
        //     layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, main_stream);
        // }
        // Q.FUSE_cuda()
        matmul_backward(ToX(deltaCat), ToG(Q.w), ToG0(Q.b), ToX(delta_attn), ToX(norm.out), ToX(Q.w), scratchF, B, T, C_qkv, 3 * C_qkv, main_stream);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        float *_mean = norm.mean==nullptr?nullptr : TO<float>(norm.mean);
        norm.FUSE_cuda(residual,scratchF,deltaCat);
        // layernorm_backward(ToX(delta), ToG(norm.w), gNb, scratchF, ToX(tmpDelta), ToX(residual), ToX(norm.w), _mean, TO<float>(norm.rstd), B, T, C, main_stream);
    }
    return out;
}