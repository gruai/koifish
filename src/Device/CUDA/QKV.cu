// #include <catch2/catch_test_macros.hpp>
#include "./cuda_common.h"
#include "./kernel/gemm.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/embed.cuh"
#include "./kernel/rope.cuh"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
// #include "./EDevice.hpp"
#define NOMINMAX

// #undef ENABLE_CUDNN
#ifdef ENABLE_CUDNN
    #include "cudnn_frontend.h"    
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

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::map<std::tuple<int,int,int,int, int>, std::shared_ptr<fe::graph::Graph>>;
using cache_type_bwd = std::map<std::tuple<int,int,int,int>, std::shared_ptr<fe::graph::Graph>>;
static std::map<std::tuple<int,int,int,int, int>, std::shared_ptr<fe::graph::Graph>> 
    cudnn_graph_fwd,cudnn_graph_bwd;
    
std::shared_ptr<fe::graph::Graph>
cudnn_sdpa_forward_graph(int64_t const b,
                          int64_t const h_q,
                          int64_t const h_k,
                          int64_t const h_v,
                          int64_t const s_q,        //  seq length of q & kv
                          int64_t const s_kv,
                          int64_t const d_qk,       // latent dim of each head
                          int64_t const d_v,
                          float const attn_scale  = 1.0f,
                          bool const is_inference = false,
                          bool const causal_mask  = false,
                          bool const alibi_mask   = false,
                          bool const padding_mask = false,
                          bool has_attn_bias      = false) {
    // Create a graph and set common global properties.
    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
    graph->set_io_data_type(CUDNN_16BIT).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif
    // graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, h_q, s_q, d_qk})
                               .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, h_k, s_kv, d_qk})
                               .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, h_v, s_kv, d_v})
                               .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_is_inference(is_inference)
                            .set_alibi_mask(alibi_mask)
                            .set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    if (has_attn_bias) {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_uid(BIAS_UID)
                                      .set_dim({b, 1, s_q, s_kv})
                                      .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);
    }

    if (padding_mask) {
        auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_q")
                                       .set_uid(SEQ_LEN_Q_UID)
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * d_v, d_v, b * h_q * d_v, 1}).set_uid(O_UID);

    if (is_inference) {
        assert(Stats == nullptr);
    } else {
        // Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_uid(STATS_UID);
    }

    return graph;
}

std::shared_ptr<fe::graph::Graph>
cudnn_sdpa_backward_graph(int64_t const b,
                           int64_t const h_q,
                           int64_t const h_k,
                           int64_t const h_v,
                           int64_t const s_q,
                           int64_t const s_kv,
                           int64_t const d_qk,
                           int64_t const d_v,
                           float const attn_scale                   = 1.0f,
                           [[maybe_unused]] bool const is_inference = false,
                           bool const causal_mask                   = false,
                           bool const alibi_mask                    = false,
                           bool const padding_mask                  = false,
                           bool has_attn_bias                       = false) {
    // Create a graph and set common global properties
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // Define input tensors Q, K, V
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_uid(Q_UID)
                               .set_dim({b, h_q, s_q, d_qk})
                               .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_uid(K_UID)
                               .set_dim({b, h_k, s_kv, d_qk})
                               .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_uid(V_UID)
                               .set_dim({b, h_v, s_kv, d_v})
                               .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

    // Define output tensor O
    auto O = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("O")
                               .set_uid(O_UID)
                               .set_dim({b, h_q, s_q, d_v})
                               .set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    // Define gradient tensor dO
    auto dO = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("dO")
                                .set_uid(dO_UID)
                                .set_dim({b, h_q, s_q, d_v})
                                .set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    // Define stats tensor
    auto stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Stats")
                                   .set_uid(STATS_UID)
                                   .set_dim({b, h_q, s_q, 1})
                                   .set_stride({h_q * s_q, s_q, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));

    // Set SDPA backward options
    auto sdpa_options = fe::graph::SDPA_backward_attributes()
                            .set_name("flash_attention_backward")
                            .set_alibi_mask(alibi_mask)
                            .set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_causal_mask(true);
        // sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);
    }

    // If attention bias is provided, set it
    if (has_attn_bias) {
        auto bias = graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_uid(BIAS_UID)
                                      .set_dim({b, 1, s_q, s_kv})
                                      .set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);

        auto dbias = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("dbias")
                                       .set_uid(DBIAS_UID)
                                       .set_dim({1, h_q, s_q, s_kv})
                                       .set_stride({s_q * s_kv * h_q, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_dbias(dbias);
    }

    // If padding mask is enabled, set sequence lengths
    if (padding_mask) {
        auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_q")
                                       .set_uid(SEQ_LEN_Q_UID)
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

    // Compute SDPA backward and get gradients dQ, dK, dV
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_options);

    // Set output tensors dQ, dK, dV
    dQ->set_output(true)
        .set_uid(dQ_UID)
        .set_dim({b, h_q, s_q, d_qk})
        .set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
    dK->set_output(true)
        .set_uid(dK_UID)
        .set_dim({b, h_k, s_kv, d_qk})
        .set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
    dV->set_output(true)
        .set_uid(dV_UID)
        .set_dim({b, h_v, s_kv, d_v})
        .set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1});

    return graph;
}


auto cudnn_qkv_graph(int B,int NH,int T,int HS, float attn_scale, int ld, int is_inference_only,bool isSepQKV=false,bool isBack=false) {
    auto key = std::make_tuple(B, NH, T, HS, is_inference_only);
    if(isBack){
        auto it = cudnn_graph_bwd.find(key);
        if (it != cudnn_graph_bwd.end()) {
            return it->second;
        }
    }else{
        auto it = cudnn_graph_fwd.find(key);
        if (it != cudnn_graph_fwd.end()) {
            return it->second;
        }
    }

    std::shared_ptr<fe::graph::Graph> graph = nullptr;
    bool causal_mask=true,padding_mask=false,alibi_mask=false,has_attn_bias=false;
    if(isSepQKV)    {
        int64_t h_q=NH,h_k=NH,h_v=NH,b=B;
        int64_t s_q        = T;  // q tensor  seq length
        int64_t s_kv       = T;  // k and v tensor  seq length
        int64_t d_qk       = HS;   // hidden dim
        int64_t d_v        = HS;   // hidden dim
        graph = cudnn_sdpa_forward_graph(B,h_q, h_k,h_v, s_q,s_kv, d_qk, d_v,
                                           attn_scale,
                                           is_inference_only,   causal_mask,alibi_mask,padding_mask,has_attn_bias);
    }else   {
        graph = std::make_shared<fe::graph::Graph>();
#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
    graph->set_io_data_type(CUDNN_16BIT).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif
    std::vector<int64_t> stride = {ld*T,HS,ld,1};   //{3 * NH * HS * T,  HS, 3 * NH * HS, 1};
    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                               .set_dim({B, NH, T, HS})
                               .set_uid(Q_UID)
                               .set_stride(stride));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                               .set_dim({B, NH, T, HS})
                               .set_uid(K_UID)
                               .set_stride({stride}));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                               .set_dim({B, NH, T, HS})
                               .set_uid(V_UID)
                               .set_stride({stride}));

    // auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
    //                            .set_dim({1, 1, 1, 1})
    //                            .set_stride({1, 1, 1, 1})
    //                            .set_uid(Attn_scale_UID)
    //                            .set_is_pass_by_value(true)
    //                            .set_data_type(fe::DataType_t::FLOAT));
    
    
    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);
    
    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, NH, T, HS}).set_stride({NH * HS * T, HS, NH * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, NH, T, 1})
                               .set_stride({NH * T, T, 1, 1})
                               .set_uid(STATS_UID);
    }
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
    if(isBack){
        cudnn_graph_bwd.insert({key, graph});
    }else{
        cudnn_graph_fwd.insert({key, graph});
    }

    return graph;
}

auto cudnn_qkv_bwd(int B, int NH, int T, int HS, float attn_scale, int ld, bool isSepQKV=false) {
    // static cache_type_bwd cudnn_graph_bwd;
    auto key = std::make_tuple(B, NH, T, HS,0x0);
    auto it = cudnn_graph_bwd.find(key);
    if (it != cudnn_graph_bwd.end()) {
        return it->second;
    }

    std::shared_ptr<fe::graph::Graph> graph = nullptr;
    bool causal_mask=true,padding_mask=false,alibi_mask=false,has_attn_bias=false;
    if(isSepQKV)    {
        int64_t h_q=NH,h_k=NH,h_v=NH,b=B;
        int64_t s_q        = T;  // q tensor  seq length
        int64_t s_kv       = T;  // k and v tensor  seq length
        int64_t d_qk       = HS;   // hidden dim
        int64_t d_v        = HS;   // hidden dim
        graph = cudnn_sdpa_backward_graph(B,h_q, h_k,h_v, s_q,s_kv, d_qk, d_v,
                                           attn_scale,
                                           false,   causal_mask,alibi_mask,padding_mask,has_attn_bias);
    }else   {
        graph = std::make_shared<fe::graph::Graph>();
#if defined(ENABLE_BF16) || defined(ENABLE_FP16)
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif
    std::vector<int64_t> stride = {ld*T,HS,ld,1};   // (B, N, 3, NH, HS)
    // must come from inp (which means we also need to convert THAT to FP16)
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q")
                            .set_dim({B, NH, T, HS})
                            .set_uid(Q_UID)
                            .set_stride(stride));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K")
                            .set_dim({B, NH, T, HS})
                            .set_uid(K_UID)
                            .set_stride(stride));
    auto V = graph->tensor(fe::graph::Tensor_attributes().set_name("V")
                            .set_dim({B, NH, T, HS})
                            .set_uid(V_UID)
                            .set_stride(stride));
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
                            .set_uid(STATS_UID)
                            .set_stride({NH * T, T, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
    // auto attn_scale = graph->tensor(fe::graph::Tensor_attributes().set_name("attn_scale")
    //                         .set_dim({1, 1, 1, 1})
    //                         .set_stride({1, 1, 1, 1})
    //                         .set_is_pass_by_value(true)
    //                         .set_uid(Attn_scale_UID)
    //                         .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                            .set_deterministic_algorithm(true) // 1.5+ needs this for determinism
#endif
                            .set_causal_mask(true)
                            .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, NH, T, HS}).set_stride(stride).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, NH, T, HS}).set_stride(stride).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, NH, T, HS}).set_stride(stride).set_uid(dV_UID);
    }
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
        cudnn_workspace_size = graph->get_workspace_size();     //1008599040  2.1G for GPT2_LARGE
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    cudnn_graph_bwd.insert({key, graph});
    return graph;
}

//
bool SelfAttention::FUSE_cudnn(floatX* dqkvr,floatX* dout,int flag){
    assert(cudnn_handle!=nullptr);
    NVTX_RANGE_FN();
    int NH = n_head,HS=C/n_head,stride=3*NH*HS;
    float attn_scale = 1.0 / sqrtf(HS*1.0),*stats = TO<float>(transition);
    void* devPtrO = attn->data; //out;
    cuDNNCheck(cudnnSetStream(cudnn_handle, main_stream));
    // if(var_packs.empty()){}
    var_packs = {
            {Q_UID, devPtrQ}, {K_UID, devPtrK}, {V_UID, devPtrV}, {O_UID, devPtrO}, {dO_UID, dout},
            {dQ_UID, devDeltaQ}, {dK_UID, devDeltaK}, {dV_UID, devDeltaV}, {STATS_UID, stats}
        };
    
    if(isSeparateQKV){
        stride=NH*HS;
    }else{
        
    }
    if(isForward()){ 
        bool is_inference_only = false;    
        auto graph = cudnn_qkv_graph(B, NH, T, HS,attn_scale, stride, is_inference_only);
        rope.FUSE_cuda(ToX(tmpQKV)); 
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, cudnn_workspace));
    }else{  //Backward      dout=>(dQ,dK,dV)
        assert(devDeltaQ==dqkvr);
        auto graph = cudnn_qkv_bwd(B, NH, T, HS, attn_scale, stride);
        Q.w->Print("Q.w0",0,dump_flag);        // float a = tNormOf(Q.w);
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, cudnn_workspace));      
        Q.w->Print("Q.w1",0,dump_flag);       // a = tNormOf(Q.w);  
    }
    cudaCheck(cudaGetLastError());
    return true;
}

struct CudnnHandleDeleter {
    void
    operator()(cudnnHandle_t* handle) const {
        if (handle) {
            cuDNNCheck(cudnnDestroy(*handle));
            delete handle;
        }
    }
};
bool InitCUDNN() {    
    bool padding_mask  = (cudnnGetVersion() >= 8903);
    bool alibi_mask    = (cudnnGetVersion() >= 8904);
    bool has_attn_bias = (cudnnGetVersion() >= 8903);
    if (cudnnGetVersion() < 8903) {
        _ERROR("Test requires cudnn 8.9.3 or above");
        return false;
    }    
    // auto handle = std::make_unique<cudnnHandle_t>();
    // cuDNNCheck(cudnnCreate(handle.get()));
    // return std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter>(handle.release(), CudnnHandleDeleter());
    cuDNNCheck(cudnnCreate(&cudnn_handle));
    return true;
}

void destroy_cudnn() {
    if (cudnn_workspace != NULL) { cudaCheck(cudaFree(cudnn_workspace)); }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}
#endif

inline int convert_SM_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                                       {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
                                       {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
                                       {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    _INFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor,nGpuArchCoresPerSM[index - 1].cores);
    return nGpuArchCoresPerSM[index - 1].cores;
}

bool InitCUDA(const CLI_params&hparams,EDGE_DEVICES *hDevice,int flag){
    int local_device_idx = 0, override_enable_tf32 = 1;
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("[InitCUDA] failed at cudaSetDevice! ERR=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaCheck(cudaGetDeviceProperties(&deviceProp, local_device_idx));
    // int cuda_num_SMs = deviceProp.multiProcessorCount;
    // int cuda_threads_per_SM = deviceProp.maxThreadsPerMultiProcessor;
    // int cuda_arch_major = deviceProp.major;
    // int cuda_arch_minor = deviceProp.minor;

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cublasCheck(cublasCreate(&cublas_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = PARAMS_TYPE == typNUMBER::F32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
#ifdef ENABLE_CUDNN
    InitCUDNN();
#endif
    int precision_mode = MFUH_PRECISION_BF16;
    const char* precision_str = 
            PARAMS_TYPE == typNUMBER::F32 ? (cublas_compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : "FP32") : 
            PARAMS_TYPE == typNUMBER::F16 ? "FP16" : 
            PARAMS_TYPE == typNUMBER::BF16 ? "BF16" : 
            PARAMS_TYPE == typNUMBER::F8E5M2 ? "F8E5M2" : "X";
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);      cudaRuntimeGetVersion(&runtime_version);
    int nCorePP = convert_SM_to_cores(deviceProp.major, deviceProp.minor);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    // _INFO("| device                | %-50s |\n", deviceProp.name);
    // _INFO("| peak flops(BF16) T    | %-40.1f %10d cores |\n", get_flops_promised(deviceProp.name, precision_mode),deviceProp.multiProcessorCount);
    // _INFO("| precision of weights  | %-50s |\n", precision_str);
    // _INFO("| peak bandwidth GB/s   | %-50.1f |\n", (double)deviceProp.memoryClockRate*(deviceProp.memoryBusWidth / 8) * 2 / 1e6);
    // _INFO("+-----------------------+----------------------------------------------------+\n");
    _INFO("\t君子不器 - \"%s\" \n", deviceProp.name);
    _INFO("\tCUDA driver version / runtime version: %d.%d / %d.%d\n", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
    _INFO("\tCUDA capability major/minor version number: %d.%d. ECC=%d\n", deviceProp.major, deviceProp.minor,deviceProp.ECCEnabled);    
    _INFO("\t\n");
    _INFO("\t%d multiprocessors, %d CUDA cores/MP, %d CUDA cores\n", deviceProp.multiProcessorCount,nCorePP,nCorePP*deviceProp.multiProcessorCount);

    _INFO("\tGPU max clock rate: %.0f MHz (%0.2f GHz)\n", static_cast<double>(deviceProp.clockRate) * 1e-3,
         static_cast<double>(deviceProp.clockRate) * 1e-6);
    _INFO("\tPeak bandwidth %.1f GByte/s.\tMemory clock rate: %.0f MHz (%0.2f GHz)\n", (double)deviceProp.memoryClockRate*(deviceProp.memoryBusWidth / 8) * 2 / 1e6,
            static_cast<double>(deviceProp.memoryClockRate) * 1e-3,static_cast<double>(deviceProp.memoryClockRate) * 1e-6);
    _INFO("\tMemory bus width: %d-bit\n", deviceProp.memoryBusWidth);
    _INFO("\tGlobal memory: %.0f MBytes (%zu Bytes)\n",
         static_cast<double>(deviceProp.totalGlobalMem) / 1048576, deviceProp.totalGlobalMem);
    _INFO("\tConstant memory: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalConstMem) / 1024,
         deviceProp.totalConstMem);
    _INFO("\tShared memory per block: %.0f KBytes (%zu Bytes)\n",
         static_cast<double>(deviceProp.sharedMemPerBlock) / 1024, deviceProp.sharedMemPerBlock);
    _INFO("\tShared memory per multiprocessor: %.0f KBytes (%zu Bytes)\n",
         static_cast<double>(deviceProp.sharedMemPerMultiprocessor) / 1024, deviceProp.sharedMemPerMultiprocessor);
    _INFO("\tL2 cache size: %.0f KBytes (%d Bytes)\n", static_cast<double>(deviceProp.l2CacheSize) / 1024,
         deviceProp.l2CacheSize);
    _INFO("\tTotal number of registers available per block: %d\n", deviceProp.regsPerBlock);
    _INFO("\tWarp size: %d, Max number of threads per block: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock);
    _INFO("\tMax number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    _INFO("\tMax dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0],
         deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    _INFO("\tMax dimension size of a grid size (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    fflush(stdout);
    // Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors.
    hDevice->nCore = deviceProp.multiProcessorCount;
    cudaCheck(cudaEventCreate(&cuStart));
    cudaCheck(cudaEventCreate(&cuEnd));
    //  https://github.com/jonasmr/microprofile
    cudaCheck(cudaProfilerStart());
    return true;
}

//  cudaStreamSynchronize(stream)/cudaEventSynchronize(event) maybe better
void SYNC_DEVICE(int flag)   {
#ifdef __USE_CUDA__
    if(main_stream!=nullptr)
        cudaCheck(cudaDeviceSynchronize());
#endif
} 

bool EDGE_DEVICES::ClearGPU(int flag){    
    cudaCheck(cudaStreamDestroy(main_stream));
    cublasCheck(cublasDestroy(cublas_handle));

    cudaCheck(cudaFree(cublaslt_workspace));    
    cublasCheck(cublasLtDestroy(cublaslt_handle));
#ifdef ENABLE_CUDNN
    destroy_cudnn();
#endif
    return true;
}
/*
    QKV = (B, T, 3, NH, HS) 
*/
int ROPE::FUSE_cuda(floatX* inp,bool isFX,int flag){
    if(Empty()){
        return -1;
    }
    int NH=n_head,NH_kv=NH;    
    hFish->GetBTC(B,T,C);    
    // const int n_dims     = n_rot;
    // float freq_base=10000.0,freq_scale=1,ext_factor=0,attn_factor=1,beta_fast=32,beta_slow=1;
    // rope_corr_dims corr_dims;
    // ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);
    grid_size = CEIL_DIV(B*T*C/2, block_size);  
    float *fcos = TO<float>(hCos),*fsin = TO<float>(hSin);
    int threads = head_dim / 2;
    if(isForward()){
        floatX* q = inp,*k = q + B*T*C;           
        dim3 blocks_q(B, T, NH_kv),blocks_k(B, T, NH);        
        
        apply_rope_forward_q1<<<blocks_q, threads>>>(q, fcos, fsin, B, T, NH_kv, head_dim);
        apply_rope_forward_k1<<<blocks_k, threads>>>(k, fcos, fsin, B, T, NH, head_dim);
        SYNC_DEVICE();
        // CU_rope_<<<grid_size, block_size, 0, main_stream>>>(q,q, q_dim, head_dim, theta, n_rot,B,T,C);   
        // CU_rope_<<<grid_size, block_size, 0, main_stream>>>(k,k, kv_dim, head_dim, theta, n_rot,B,T,C);   
    }else{
        floatX* delta_q = inp,*delta_k = delta_q + B*T*C;
        dim3 blocks(B, T, NH); 
        apply_rope_backward_kernel1<<<blocks, threads>>>(delta_q, delta_k, fcos, fsin, B, T, NH_kv, NH, head_dim);
        SYNC_DEVICE();
        // CU_rope_back<<<grid_size, block_size, 0, main_stream>>>(delta_q,delta_q, q_dim, head_dim, theta, n_rot,B,T,C);   
        // CU_rope_back<<<grid_size, block_size, 0, main_stream>>>(delta_k,delta_k, kv_dim, head_dim, theta, n_rot,B,T,C);  
    }

    return 0x0;
}

/*
    Forward:    cur = FUSE_cuda(cur,residual,flag);   
    Backward:   QKV->FUSE_cuda(last->out,QKV->norm.out,0x0);
*/
hGTensor SelfAttention::FUSE_cuda(hGTensor inpL,int flag){    
    floatX *qkvr=ToX(tmpQKV);    // Q.out/K.out/V.out
    float *l_att = TO<float>(transition); //(float*)acts.att + l * B * NH * T; // cuDNN needs a smaller FP32 tensor
    if(isForward()){    //  data=ToX(QKV->norm.out)
        inp = OnInput(inpL);            //         inp->Print("inp",0x0,dump_flag); 
        GTensor::residual=inp;      //GTensor::residual->OverWrite(inp);          
        hGTensor inpQ = inpL;      
        if(fuseNorm==nullptr){
            inpQ=norm.FUSE_cuda(inpL);          inpQ->Print("qkvn.in",0x0,dump_flag);  
            // norm.w->Print("qkvn.w",0x0,dump_flag);          norm.b->Print("qkvn.b",0x0,dump_flag);  
            // norm.mean->Print("qkvn.mean",0x0,dump_flag);       norm.rstd->Print("qkvn.rstd",0x0,dump_flag);  
        }   
        if(isSeparateQKV)     {  
            Q.out->data=devPtrQ;    K.out->data=devPtrK;                V.out->data=devPtrV;    
            Q.Forw(Q.out,inpQ);     K.Forw(K.out,inpQ);                 V.Forw(V.out,inpQ);  
            // Q.out->Print("Q.out",0x0,dump_flag);    K.out->Print("K.out",0x0,dump_flag);    V.out->Print("V.out",0x0,dump_flag); 
        }else{
            Q.Forw(tmpQKV,inpQ);
            // Q.w->Print("Q.w",0x0,dump_flag);     Q.b->Print("Q.b",0x0,dump_flag);    
        }                 
        // tmpQKV->Print("QKV",0x0,dump_flag); 
#ifdef ENABLE_CUDNN
        FUSE_cudnn(nullptr,nullptr,flag);
        // rope.FUSE_cuda(ToX(tmpQKV));       
#else
        // if (T != model->seq_len) { // unused parts of attention buffer must be zeroed (T-dependent)
        //     cudaCheck(cudaMemset(l_att, 0, B * NH * T * T * sizeof(floatX)));
        // }
        hGTensor scrath = GTensor::bt4c;        //only forward
        Q.Forw(QKV,inpL);          
        rope.FUSE_cuda(scrath); 
        attention_forward(ToX(attn), qkvr, ToX(transition), ToX(QKV), B, T, C, NH, main_stream);  //  l_atty, l_qkvr, l_att, scratch
#endif
        attn->Print("l_atty",0x0,dump_flag);  
        proj_cat.Forw(GTensor::scratch,attn);   //fuMM(scratch, ToX(attn), pw, pb, B, T, C, C, main_stream);       
        // GTensor::scratch->Print("proj_cat",0x0,dump_flag);        
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        if(!hFish->isRemater()){
            residual_forward(ToX(out), ToX(GTensor::residual), ToX(GTensor::scratch), B*T*C, main_stream);      
            if(fuseNorm!=nullptr){
                float *mean=TO<float>(fuseNorm->mean),*rstd=TO<float>(fuseNorm->rstd);
                layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, 1, C, main_stream);
            }      
            out->Print("out",0x0,dump_flag);      
        }else{
            
        }
        return out;
    }else{      //  Backward
        float *scratchF=(float *)GTensor::buff;        
         assert(inpL==GTensor::delta);
        delta->Print("delta",0x0,dump_flag);     
        proj_cat.Back(GTensor::tmpDelta,attn,GTensor::delta,nullptr);

        hGensor delta_qkv = GTensor::bt4c; 
        if(remater_qkv)  {   
            hGTensor norm_out = norm.out;
            if(isSeparateQKV)     {
                Q.out->data=devPtrQ;    K.out->data=devPtrK;                V.out->data=devPtrV;    
                Q.Forw(Q.out,norm_out);     K.Forw(K.out,norm_out);         V.Forw(V.out,norm_out);  
            }else{
                Q.Forw(tmpQKV,norm_out);       //Q.Forw(GTensor::tmpFF1,norm_out)
            }  
        }
        GTensor::tmpDelta->Print("delta_cat",0x0,dump_flag);
#ifdef ENABLE_CUDNN
        FUSE_cudnn(ToX(delta_qkv), ToX(GTensor::tmpDelta),flag);
#else
        assert(0);
#endif
        delta_qkv->Print("delta_qkv",0x0,dump_flag);     //PrintTensor<floatX>("back of attn",ToX(delta_attn),true,B,T,C);        
        // rope.FUSE_cuda(ToX(delta_attn)); 
        // Q.Back()
        if(isSeparateQKV)     {
            // Q.Back(GTensor::tmpDelta,norm.out,Q.tmpDelta);     
            // K.Back(GTensor::tmpDelta,norm.out,K.tmpDelta);         
            // V.Back(GTensor::tmpDelta,norm.out,V.tmpDelta); 
            matmul_backward(ToX(GTensor::tmpDelta), ToG(Q.w), ToG0(Q.b), (floatX*)devDeltaQ, ToX(norm.out), ToX(Q.w), scratchF, B, T, C_qkv, C_qkv, main_stream,false,NULL,false);
            GTensor::tmpDelta->Print("delta_0",0,dump_flag);
            matmul_backward(ToX(GTensor::tmpDelta), ToG(K.w), ToG0(K.b), (floatX*)devDeltaK, ToX(norm.out), ToX(K.w), scratchF, B, T, C_qkv, C_qkv, main_stream,false,NULL,true);
            GTensor::tmpDelta->Print("delta_1",0,dump_flag);
            matmul_backward(ToX(GTensor::tmpDelta), ToG(V.w), ToG0(V.b), (floatX*)devDeltaV, ToX(norm.out), ToX(V.w), scratchF, B, T, C_qkv, C_qkv, main_stream,false,NULL,true);
            GTensor::tmpDelta->Print("delta_2",0,dump_flag);
        }else{
            matmul_backward(ToX(GTensor::tmpDelta), ToG(Q.w), ToG0(Q.b), ToX(delta_qkv), ToX(norm.out), ToX(Q.w), scratchF, B, T, C_qkv, 3 * C_qkv, main_stream);
            GTensor::tmpDelta->Print("delta_3",0,dump_flag);
        }
        
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        norm.FUSE_cuda(GTensor::tmpDelta);      // would backpropagatioin to GTensor::delta 
        GTensor::delta->Print("back of QKV",0,dump_flag);
        return GTensor::delta;   
    }
    return nullptr;
}


/*
auto cudnn_qkv_graph_v1(int B,int NH,int T,int HS,int ld, int is_inference,bool isBack=false) {
    int64_t h_q=NH,h_k=NH,h_v=NH,b=B;
    int64_t s_q        = T;  // q tensor  seq length
    int64_t s_kv       = T;  // k and v tensor  seq length
    int64_t d_qk       = HS;   // hidden dim
    int64_t d_v        = HS;   // hidden dim
    float attn_scale   = 0.123f;
    bool causal_mask   = true;
    bool padding_mask  = (cudnnGetVersion() >= 8903),alibi_mask    = (cudnnGetVersion() >= 8904),has_attn_bias = (cudnnGetVersion() >= 8903);

    auto graph = cudnn_sdpa_forward_graph(B,
                                           h_q, h_k,h_v,
                                           s_q,s_kv,
                                           d_qk, d_v,
                                           attn_scale,
                                           is_inference,
                                           causal_mask,
                                           alibi_mask,
                                           padding_mask,
                                           has_attn_bias);

    graph->build(cudnn_handle, {fe::HeurMode_t::A});
    assert(graph.is_good());
    //// Build variant pack
    Surface<half> q_tensor(b * h_q * s_q * d_qk, false);
    Surface<half> k_tensor(b * h_k * d_qk * s_kv, false);
    Surface<half> v_tensor(b * h_v * d_v * s_kv, false);
    Surface<half> o_tensor(b * s_q * h_q * d_qk, false);

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
        {Q_UID, q_tensor.devPtr}, {K_UID, k_tensor.devPtr}, {V_UID, v_tensor.devPtr}, {O_UID, o_tensor.devPtr}};

    Surface<half> bias_tensor(b * 1 * s_q * s_kv, false);
    if (has_attn_bias) {
        variant_pack[BIAS_UID] = bias_tensor.devPtr;
    }

    Surface<int32_t> devActualSeqlenQ(b, false);
    Surface<int32_t> devActualSeqlenKV(b, false);
    if (padding_mask) {
        std::vector<int32_t> hostActualSeqlenQ(b, 20);
        std::vector<int32_t> hostActualSeqlenKV(b, 20);

        cudaCheck(cudaMemcpy(devActualSeqlenQ.devPtr,
                              hostActualSeqlenQ.data(),
                              sizeof(hostActualSeqlenQ[0]) * b,
                              cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(devActualSeqlenKV.devPtr,
                              hostActualSeqlenKV.data(),
                              sizeof(hostActualSeqlenKV[0]) * b,
                              cudaMemcpyHostToDevice));
        cudaCheck(cudaDeviceSynchronize());

        variant_pack[SEQ_LEN_Q_UID]  = devActualSeqlenQ.devPtr;
        variant_pack[SEQ_LEN_KV_UID] = devActualSeqlenKV.devPtr;
    }

    Surface<float> statsTensor(b * h_q * s_q * 1, false);
    if (is_inference == false) {
        variant_pack[STATS_UID] = statsTensor.devPtr;
    }

    int64_t workspace_size = 0;
    graph->get_workspace_size(workspace_size).is_good();
    Surface<int8_t> workspace(workspace_size, false);

    NEED_(graph->execute(cudnn_handle, variant_pack, workspace.devPtr).is_good());

    cudaCheck(cudaDeviceSynchronize());
}*/