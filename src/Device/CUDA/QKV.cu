/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some codes are from llm.c
 *
 *  \brief cuda kernel of QKV self attention
 *  \author Yingshi Chen
 */

// #include <catch2/catch_test_macros.hpp>
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "./cuda_common.h"
#include "./kernel/gelu.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"

#define NOMINMAX

// #undef ENABLE_CUDNN
#ifdef ENABLE_CUDNN
#include "cudnn_frontend.h"
namespace fe = cudnn_frontend;
#if defined(USE_FP16_BASELINE)
#define CUDNN_16BIT fe::DataType_t::HALF
#else  // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif
#else
// defines: attention_forward, attention_backward
#endif

static cudaEvent_t cuStart, cuEnd;

#ifdef ENABLE_CUDNN
static cudnnHandle_t cudnn_handle;

static void cuDNNCheck(cudnnStatus_t error, const char* file, int line) {
    if (error != CUDNN_STATUS_SUCCESS) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, cudnnGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cuDNNCheck(err) (cuDNNCheck(err, __FILE__, __LINE__))

static void checkCudnnFE(const fe::error_object& e, const char* file, int line) {
    if (!e.is_good()) {
        printf("[CUDNN ERROR] at file %s:%d:\n%s\n", file, line, e.err_msg.c_str());
        exit(EXIT_FAILURE);
    }
}
#define checkCudnnFE(err) checkCudnnFE(err, __FILE__, __LINE__)

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast

using cache_cudnn_graph = std::map<QKV_KEY6, std::shared_ptr<fe::graph::Graph>>;
static cache_cudnn_graph cudnn_graph_fwd, cudnn_graph_bwd;

std::shared_ptr<fe::graph::Graph> cudnn_sdpa_forward_graph(int64_t const b, int64_t const h_q, int64_t const h_k, int64_t const h_v,
                                                           int64_t const s_q,  //  seq length of q & kv
                                                           int64_t const s_kv,
                                                           int64_t const d_qk,  // latent dim of each head
                                                           int64_t const d_v, float const attn_scale = 1.0f, bool const is_inference = false,
                                                           bool const causal_mask = false, bool const alibi_mask = false, bool const padding_mask = false,
                                                           bool has_attn_bias = false) {
    // Create a graph and set common global properties.
    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(USE_BF16_BASELINE) || defined(USE_FP16_BASELINE)
    graph->set_io_data_type(CUDNN_16BIT).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif
    // graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID).set_dim({b, h_q, s_q, d_qk}).set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

    auto K = graph->tensor(
        fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID).set_dim({b, h_k, s_kv, d_qk}).set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

    auto V = graph->tensor(
        fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID).set_dim({b, h_v, s_kv, d_v}).set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

    auto sdpa_options =
        fe::graph::SDPA_attributes().set_name("flash_attention").set_is_inference(is_inference).set_alibi_mask(alibi_mask).set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);
    }

    if (has_attn_bias) {
        auto bias = graph->tensor(
            fe::graph::Tensor_attributes().set_name("bias").set_uid(BIAS_UID).set_dim({b, 1, s_q, s_kv}).set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
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

std::shared_ptr<fe::graph::Graph> cudnn_sdpa_backward_graph(int64_t const b, int64_t const h_q, int64_t const h_k, int64_t const h_v, int64_t const s_q,
                                                            int64_t const s_kv, int64_t const d_qk, int64_t const d_v, float const attn_scale = 1.0f,
                                                            [[maybe_unused]] bool const is_inference = false, bool const causal_mask = false,
                                                            bool const alibi_mask = false, bool const padding_mask = false, bool has_attn_bias = false) {
    // Create a graph and set common global properties
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    // Define input tensors Q, K, V
    auto Q = graph->tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID).set_dim({b, h_q, s_q, d_qk}).set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1}));

    auto K = graph->tensor(
        fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID).set_dim({b, h_k, s_kv, d_qk}).set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1}));

    auto V = graph->tensor(
        fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID).set_dim({b, h_v, s_kv, d_v}).set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1}));

    // Define output tensor O
    auto O =
        graph->tensor(fe::graph::Tensor_attributes().set_name("O").set_uid(O_UID).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    // Define gradient tensor dO
    auto dO = graph->tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_uid(dO_UID).set_dim({b, h_q, s_q, d_v}).set_stride({h_q * s_q * d_v, s_q * d_v, d_v, 1}));

    // Define stats tensor
    auto stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Stats")
                                   .set_uid(STATS_UID)
                                   .set_dim({b, h_q, s_q, 1})
                                   .set_stride({h_q * s_q, s_q, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));

    // Set SDPA backward options
    auto sdpa_options = fe::graph::SDPA_backward_attributes().set_name("flash_attention_backward").set_alibi_mask(alibi_mask).set_attn_scale(attn_scale);

    if (causal_mask) {
        sdpa_options.set_causal_mask(true);
        // sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT).set_diagonal_band_right_bound(0);
    }

    // If attention bias is provided, set it
    if (has_attn_bias) {
        auto bias = graph->tensor(
            fe::graph::Tensor_attributes().set_name("bias").set_uid(BIAS_UID).set_dim({b, 1, s_q, s_kv}).set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
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
    dQ->set_output(true).set_uid(dQ_UID).set_dim({b, h_q, s_q, d_qk}).set_stride({h_q * s_q * d_qk, s_q * d_qk, d_qk, 1});
    dK->set_output(true).set_uid(dK_UID).set_dim({b, h_k, s_kv, d_qk}).set_stride({h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1});
    dV->set_output(true).set_uid(dV_UID).set_dim({b, h_v, s_kv, d_v}).set_stride({h_v * s_kv * d_v, s_kv * d_v, d_v, 1});

    return graph;
}

static bool is_inference_only = false;
size_t cudnn_qkv_forw(int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn, int flag) {
    auto key = std::make_tuple(B, Hq, Hkv, T, HS, (int)is_inference_only);
    assert(cudnn_graph_fwd.find(key) == cudnn_graph_fwd.end());
    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(USE_BF16_BASELINE) || defined(USE_FP16_BASELINE)
    graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
#else
    assert(0);
#endif

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    int H                       = Hq + 2 * Hkv;
    std::vector<int64_t> stride = {Hq * HS * T, HS, Hq * HS, 1}, strideKV = {Hkv * HS * T, HS, Hkv * HS, 1};
    if (qkv4dnn == QKV_PACK::QKVQKV) {
        stride = {H * HS * T, HS, H * HS, 1}, strideKV = stride;
    }

    auto Q          = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim({B, Hq, T, HS}).set_uid(Q_UID).set_stride(stride));
    auto K          = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim({B, Hkv, T, HS}).set_uid(K_UID).set_stride(strideKV));
    auto V          = graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_dim({B, Hkv, T, HS}).set_uid(V_UID).set_stride(strideKV));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("attn_scale")
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

    // Output is (B, T, Hq, HS) BF16/FP16 and stats for backward pass is (B, Hq, T) FP32
    O->set_output(true).set_dim({B, Hq, T, HS}).set_stride({Hq * HS * T, HS, Hq * HS, 1}).set_uid(O_UID);

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({B, Hq, T, 1}).set_stride({Hq * T, T, 1, 1}).set_uid(STATS_UID);
    }

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));
    size_t need = graph->get_workspace_size();
    assert(need == 0x0);
    // if (need > cudnn_workspace_size) {
    //     if (cudnn_workspace_size > 0) {
    //         cudaCheck(cudaFree(cudnn_workspace));
    //     }
    //     cudnn_workspace_size = graph->get_workspace_size()*10;
    //     cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    // }
    cudnn_graph_fwd.insert({key, graph});
    return 0x0;
}

size_t cudnn_qkv_back(int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn) {
    QKV_KEY6 key = std::make_tuple(B, Hq, Hkv, T, HS, 0x0);
    assert(cudnn_graph_bwd.find(key) == cudnn_graph_bwd.end());
    // auto it = cudnn_graph_bwd.find(key);
    // if (it != cudnn_graph_bwd.end()) {
    //     return it->second;
    // }

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);

    std::vector<int64_t> stride = {Hq * HS * T, HS, Hq * HS, 1}, strideKV = {Hkv * HS * T, HS, Hkv * HS, 1};
    if (qkv4dnn == QKV_PACK::QKVQKV) {
        assert(0);
        // stride ={H * HS * T,  HS, H * HS, 1}, strideKV = stride;
    }
    auto Q = graph->tensor(fe::graph::Tensor_attributes().set_name("Q").set_dim({B, Hq, T, HS}).set_uid(Q_UID).set_stride(stride));
    auto K = graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_dim({B, Hkv, T, HS}).set_uid(K_UID).set_stride(strideKV));
    auto V = graph->tensor(
        fe::graph::Tensor_attributes().set_name("V").set_dim({B, Hkv, T, HS}).set_uid(V_UID).set_stride(strideKV));  //  {H * HS * T, HS, H * HS, 1}
    auto O  = graph->tensor(fe::graph::Tensor_attributes().set_name("O").set_dim({B, Hq, T, HS}).set_uid(O_UID).set_stride({stride}));
    auto dO = graph->tensor(fe::graph::Tensor_attributes().set_name("dO").set_dim({B, Hq, T, HS}).set_uid(dO_UID).set_stride({stride}));

    auto stats                 = graph->tensor(fe::graph::Tensor_attributes()
                                                   .set_name("stats")
                                                   .set_dim({B, Hq, T, 1})
                                                   .set_uid(STATS_UID)
                                                   .set_stride({Hq * T, T, 1, 1})
                                                   .set_data_type(fe::DataType_t::FLOAT));
    auto attn_scale            = graph->tensor(fe::graph::Tensor_attributes()
                                                   .set_name("attn_scale")
                                                   .set_dim({1, 1, 1, 1})
                                                   .set_stride({1, 1, 1, 1})
                                                   .set_is_pass_by_value(true)
                                                   .set_uid(Attn_scale_UID)
                                                   .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                     .set_name("flash_attention_backward")
#if CUDNN_FRONTEND_MAJOR_VERSION > 1 || CUDNN_FRONTEND_MINOR_VERSION >= 5
                                     .set_deterministic_algorithm(true)  // 1.5+ needs this for determinism
#endif
                                     .set_causal_mask(true)
                                     .set_attn_scale(attn_scale);

    // Create the graph operation and get the output tensors back
    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, stats, sdpa_backward_options);

    dQ->set_output(true).set_dim({B, Hq, T, HS}).set_stride({stride}).set_uid(dQ_UID);
    dK->set_output(true).set_dim({B, Hkv, T, HS}).set_stride(strideKV).set_uid(dK_UID);
    dV->set_output(true).set_dim({B, Hkv, T, HS}).set_stride(strideKV).set_uid(dV_UID);

    checkCudnnFE(graph->validate());

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    checkCudnnFE(graph->build_operation_graph(cudnn_handle));
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    checkCudnnFE(graph->check_support(cudnn_handle));
    checkCudnnFE(graph->build_plans(cudnn_handle));

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > GTensor::cudnn_workspace_size) {
        if (GTensor::cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(GTensor::cudnn_workspace));
        }
        GTensor::cudnn_workspace_size = graph->get_workspace_size();  // 1008599040  2.1G for GPT2_LARGE
        cudaCheck(cudaMalloc(&GTensor::cudnn_workspace, GTensor::cudnn_workspace_size));
    }

    cudnn_graph_bwd.insert({key, graph});
    return GTensor::cudnn_workspace_size;
}

//
bool SelfAttention::FUSE_cudnn(floatX* dqkvr, floatX* dout, int flag) {
    assert(cudnn_handle != nullptr);
    NVTX_RANGE_FN();
    int HS           = head_dim;
    float* stats     = TO<float>(transition);  //  (B, Hq, T)
    float attn_scale = 1.0 / sqrtf(HS * 1.0f);
    // void* devPtrO    = attn->data;  // out;
    cuDNNCheck(cudnnSetStream(cudnn_handle, main_stream));

    UpdateQKVPack();
    //   devQ = Q.out->data, devK = K.out->data, devV = V.out->data;
    var_packs = {
        {Q_UID, devQ},       {K_UID, devK},       {V_UID, devV},       {O_UID, attn->data}, {dO_UID, dout},
        {dQ_UID, devDeltaQ}, {dK_UID, devDeltaK}, {dV_UID, devDeltaV}, {STATS_UID, stats},  {Attn_scale_UID, &attn_scale},
    };

    QKV_KEY6 key = std::make_tuple(B, n_head, n_head_kv, T, HS, 0x0);
    if (isForward()) {
        assert(cudnn_graph_fwd.find(key) != cudnn_graph_fwd.end());
        auto graph = cudnn_graph_fwd[key];  // cudnn_qkv_forw(B, n_head, n_head_kv, T, HS, is_inference_only);
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, GTensor::cudnn_workspace));
        attn->Print("[qk]v", 0x0, dump_flag);
    } else {  // Backward      dout=>(dQ,dK,dV)
        assert(devDeltaQ == dqkvr);
        assert(cudnn_graph_bwd.find(key) != cudnn_graph_bwd.end());
        auto graph = cudnn_graph_bwd[key];  // cudnn_qkv_back(B, n_head, n_head_kv, T, HS, 0x0);
        // Q.w->Print("Q.w0", 0, dump_flag);  // float a = tNormOf(Q.w);
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, GTensor::cudnn_workspace));
        // Q.w->Print("Q.w1", 0, dump_flag);  // a = tNormOf(Q.w);
    }
    cudaCheck(cudaGetLastError());
    return true;
}

struct CudnnHandleDeleter {
    void operator()(cudnnHandle_t* handle) const {
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

void DestroyCUDNN() {
    // if (GTensor::cudnn_workspace != NULL) {
    //     cudaCheck(cudaFree(GTensor::cudnn_workspace));
    // }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}
#endif

inline int convert_SM_to_cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128},
                                       {0x60, 64},  {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},
                                       {0x86, 128}, {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    _INFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d cores/SM", major, minor, nGpuArchCoresPerSM[index - 1].cores);
    return nGpuArchCoresPerSM[index - 1].cores;
}

bool InitCUDA(const CLI_params& hparams, EDGE_DEVICES* hDevice, int flag) {
    int local_device_idx = 0;  // override_enable_tf32 = 1;
    cudaError_t err      = cudaSetDevice(0);
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
    // bool enable_tf32 = PARAMS_TYPE == typNUMBER::F32 && deviceProp.major >= 8 && override_enable_tf32;
    // cublas_compute   = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
#ifdef ENABLE_CUDNN
    InitCUDNN();
#endif
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    int nCorePP = convert_SM_to_cores(deviceProp.major, deviceProp.minor);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    // _INFO("| device                | %-50s |\n", deviceProp.name);
    // _INFO("| peak flops(BF16) T    | %-40.1f %10d cores |\n", get_flops_promised(deviceProp.name, precision_mode),deviceProp.multiProcessorCount);
    // _INFO("| precision of weights  | %-50s |\n", precision_str);
    // _INFO("| peak bandwidth GB/s   | %-50.1f |\n", (double)deviceProp.memoryClockRate*(deviceProp.memoryBusWidth / 8) * 2 / 1e6);
    // _INFO("+-----------------------+----------------------------------------------------+\n");
    _INFO("\t君子不器 - \"%s\" \n", deviceProp.name);
    _INFO("\tCUDA driver version: %d.%d, %s runtime version: %d.%d %s\n", driver_version / 1000, (driver_version % 100) / 10, COLOR_ORANGE,
          runtime_version / 1000, (runtime_version % 100) / 10, COLOR_RESET);
    _INFO("\tCUDA capability major/minor version number: %d.%d. ECC=%d\n", deviceProp.major, deviceProp.minor, deviceProp.ECCEnabled);
    _INFO("\t\n");
    _INFO("\t%d multiprocessors, %d CUDA cores/MP, %d CUDA cores\n", deviceProp.multiProcessorCount, nCorePP, nCorePP * deviceProp.multiProcessorCount);

    _INFO("\tGPU max clock rate: %.0f MHz (%0.2f GHz)\n", static_cast<double>(deviceProp.clockRate) * 1e-3, static_cast<double>(deviceProp.clockRate) * 1e-6);
    _INFO("\tPeak bandwidth %.1f GByte/s.\tMemory clock rate: %.0f MHz (%0.2f GHz)\n",
          (double)deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) * 2 / 1e6, static_cast<double>(deviceProp.memoryClockRate) * 1e-3,
          static_cast<double>(deviceProp.memoryClockRate) * 1e-6);
    _INFO("\tMemory bus width: %d-bit\n", deviceProp.memoryBusWidth);
    _INFO("\tGlobal memory: %.0f MBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalGlobalMem) / 1048576, deviceProp.totalGlobalMem);
    _INFO("\tConstant memory: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.totalConstMem) / 1024, deviceProp.totalConstMem);
    _INFO("\tShared memory per block: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerBlock) / 1024, deviceProp.sharedMemPerBlock);
    _INFO("\tShared memory per multiprocessor: %.0f KBytes (%zu Bytes)\n", static_cast<double>(deviceProp.sharedMemPerMultiprocessor) / 1024,
          deviceProp.sharedMemPerMultiprocessor);
    _INFO("\tL2 cache size: %.0f KBytes (%d Bytes)\n", static_cast<double>(deviceProp.l2CacheSize) / 1024, deviceProp.l2CacheSize);
    _INFO("\tTotal number of registers available per block: %d\n", deviceProp.regsPerBlock);
    _INFO("\tWarp size: %d, Max number of threads per block: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock);
    _INFO("\tMax number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);

    _INFO("\tMax dimension size of a thread block (x,y,z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
    _INFO("\tMax dimension size of a grid size (x,y,z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    _INFO("+-----------------------+----------------------------------------------------+\n");
    fflush(stdout);
    // Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the cores of CPUs. That is, SMs both execute computations and store state
    // available for computation in registers, with associated caches. Compared to CPU cores, GPU SMs are simple, weak processors.
    hDevice->nCore = deviceProp.multiProcessorCount;
    cudaCheck(cudaEventCreate(&cuStart));
    cudaCheck(cudaEventCreate(&cuEnd));
    //  https://github.com/jonasmr/microprofile
    cudaCheck(cudaProfilerStart());
    return true;
}

/*
      cudaStreamSynchronize(stream)/cudaEventSynchronize(event) maybe better

      1. cudaEventSynchronize
        cudaEventSynchronizeis a CUDA function that blocks the host (CPU) until a CUDA event is completed. It's crucial for proper timing, synchronization, and
   debugging in CUDA applications.
      2. cudaEventRecord
        cudaEventRecordis a CUDA function that records a CUDA event at a specific point in a CUDA stream. It's used for timing, synchronization, and
   establishing dependencies between operations.
*/
bool SYNC_DEVICE(const std::string& sX, int flag) {
#ifdef __USE_CUDA__
    if (main_stream != nullptr) {
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            _INFO("_SYNC_DEVICE_ err=\"%s\" (%s code=%d)\t%s\n", cudaGetErrorString(error), cudaGetErrorName(error), error, sX.c_str());
            if (flag == 1)
                return false;
            assert(0 && "_SYNC_DEVICE_");
            exit(KOIFISH_EXIT_SYNC_DEVICE);
        }
        return true;
    }
#endif
    return true;
}
void SYNC_STREAM(int flag) {
#ifdef __USE_CUDA__
    if (main_stream != nullptr)
        cudaCheck(cudaStreamSynchronize(main_stream));
#endif
}

hGTensor SelfAttention::cuInfer(hGTensor inpL, int flag) {
    assert(hCache != nullptr && isSeparateQKV);
    hBATCH hBatch = hFish->GetCurBatch(true);
    int pos       = hBatch->tok_pos;
    if (pos > 0)
        _devQKV(pos);  // update k,v
    // // hCache:   (layer, seq_len, kv_dim)
    floatX *key_cache = (floatX*)hCache->Get(KVCache::KV_KEY, layid - 1, 0), *val_cache = (floatX*)hCache->Get(KVCache::KV_VAL, layid - 1, 0);
    // K.out->data = key_cache + (size_t)pos * kv_dim, V.out->data = val_cache + (size_t)pos * kv_dim;
    hGensor tmpQKV = gBUFF->tmpFF1;  
    floatX* qkvr = ToX(tmpQKV);  // Q.out/K.out/V.out
    int nToken = nBatchToken(), seq_len = hFish->config.n_ctx(), nEmbed = hFish->config.nEmbed();
    inp = OnInput(inpL);  //  may remater by hIn->SerialData
    inp->Print("inp", 0x0, dump_flag);
    gBUFF->residual = inp;
    hGTensor inpQ   = inpL;
    if (fuseNorm == nullptr) {
        inpQ = norm.cuFlow(inpL);
        inpQ->Print("qkv.in", 0x0, dump_flag, nToken * nEmbed);
        // norm.w->Print("qkvn.w",0x0,dump_flag);          norm.b->Print("qkvn.b",0x0,dump_flag);
        // norm.mean->Print("qkvn.mean",0x0,dump_flag);       norm.rstd->Print("qkvn.rstd",0x0,dump_flag);
    }
    if (isSeparateQKV) {
        Q.Forw(Q.out, inpQ), K.Forw(K.out, inpQ), V.Forw(V.out, inpQ);
    } else {
        Q.Forw(tmpQKV, inpQ);
        // Q.w->Print("Q.w",0x0,dump_flag);     Q.b->Print("Q.b",0x0,dump_flag);
    }
    // tmpQKV->Print("QKV",0x0,dump_flag);
    Q.out->Print("Q.out", 0x0, dump_flag, nToken * q_dim), K.out->Print("K.out", 0x0, dump_flag, nToken * kv_dim),
        V.out->Print("V.out", 0x0, dump_flag, nToken * kv_dim);

    if (rope != nullptr) {
        rope->cuInfer(this, rope_seed, pos);
    }
    if (DEBUG.T_kvcache_quant == 1) {
        // size_t smemPB = head_dim * sizeof(float);
        // CU_X2ternary_multihead<floatX><<<n_head_kv, head_dim, smemPB, main_stream>>>(nullptr, ToX(K.out), (char*)key_cache, n_head_kv, head_dim, true);
    }
    if (1) {
        int qk_threads_per_block = std::min(1024, pos + 1);

        attention_qk_kernel<<<n_head, qk_threads_per_block>>>(ToX(attn), ToX(Q.out), key_cache, pos, seq_len, n_head, n_head_kv, head_dim);
        // // 6.2: softmax
        CU_softmax_multihead<<<n_head, 1>>>(ToX(attn), pos, seq_len);
        attn->Print("attn", 0x0, dump_flag, pos + 1);
        // // 6.3: aggregate V values
        attention_v_kernel<<<n_head, head_dim>>>(ToX(Q.out), ToX(attn), val_cache, pos, seq_len, n_head, n_head_kv, head_dim);
        Q.out->Print("att_out", 0x0, dump_flag, nToken * q_dim);
    } else {
        FUSE_cudnn(nullptr, nullptr, flag);
        attn->Print("l_atty", 0x0, dump_flag);
    }
    if (0)  // ShareLayerOut: inpL/out is same data
        ;//CU_mv_(ToX(inpL), ToX(proj_cat.w), ToX(Q.out), C, q_dim, 1.0f, 1.0f);
    else {  //       From cuFlow
        proj_cat.Forw(gBUFF->scratch, Q.out);
        gBUFF->scratch->Print("proj_cat", 0x0, dump_flag, nToken * nEmbed);
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        if (!hFish->isRemater()) {
            gBUFF->residual->Print("residual", 0x0, dump_flag, nToken * nEmbed);
            residual_forward(ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), nToken * nEmbed, main_stream);
            assert(fuseNorm == nullptr && "Try fuse normal later...");
            {
                // float *mean = TO<float>(fuseNorm->mean), *rstd = TO<float>(fuseNorm->rstd);
                // layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out), ToX(fuseNorm->w), ToX0(fuseNorm->b), nToken, 1, C, main_stream);
            }

        } else {
        }
    }
    out->Print("qkv.out", 0x0, 0);
    return out;
};

/*
    Forward:    cur = cuFlow(cur,residual,flag);
    Backward:   QKV->cuFlow(last->out,QKV->norm.out,0x0);
*/
hGTensor SelfAttention::cuFlow(hGTensor inpL, int flag) {
    // NVTX_RANGE_FN();
    hGensor tmpQKV = gBUFF->tmpFF1;  
    floatX* qkvr = ToX(tmpQKV);  // Q.out/K.out/V.out
    int nEmbed = hFish->config.nEmbed();
    // bool isAlternate = true;                   // layer%2==1;layer>1;
    if (isForward()) {  //  data=ToX(QKV->norm.out)
        NvtxRange range(name.c_str(), 0);
        inp             = OnInput(inpL);  //         inp->Print("inp",0x0,dump_flag);
        gBUFF->residual = inp;            // gBUFF->residual->OverWrite(inp);
        hGTensor inpQ   = inpL;
        if (fuseNorm == nullptr) {
            inpQ = norm.cuFlow(inpL);
            inpQ->Print("qkvn.in", 0x0, dump_flag);
            // norm.w->Print("qkvn.w",0x0,dump_flag);         if(norm.b!=nullptr)  norm.b->Print("qkvn.b",0x0,dump_flag);
            // norm.mean->Print("qkvn.mean",0x0,dump_flag);       norm.rstd->Print("qkvn.rstd",0x0,dump_flag);
        }
        if (isSeparateQKV) {
            Q.Forw(Q.out, inpQ);
            K.Forw(K.out, inpQ);
            V.Forw(V.out, inpQ);
        } else {
            Q.Forw(tmpQKV, inpQ);
            // Q.w->Print("Q.w",0x0,dump_flag);     Q.b->Print("Q.b",0x0,dump_flag);
        }
        // Q.out->Print("Q.out", 0x0, dump_flag, C);  // K.out->Print("K.out", 0x0, dump_flag), V.out->Print("V.out", 0x0, dump_flag);
        INSPECT_THIS;
        if (rope != nullptr) {
            rope->cuFlow(this, rope_seed);
        }
        // GTensor::tZ->Print(GTensor::tZ->name, 0, dump_flag);
#ifdef ENABLE_CUDNN
        FUSE_cudnn(nullptr, nullptr, flag);
#else
        attention_forward(ToX(attn), qkvr, ToX(transition), ToX(QKV), B, T, C, NH, main_stream);  //  l_atty, l_qkvr, l_att, scratch
#endif
        // GTensor::tZ->Print(GTensor::tZ->name, 0, dump_flag);

        proj_cat.Forw(gBUFF->scratch, attn);  // fuMM(scratch, ToX(attn), pw, pb, B, T, C, C, main_stream);
        // gBUFF->scratch->Print("proj_cat",0x0,dump_flag);
        // fused_residual_forward5(ouput, normed,mean,rstd, residual, scratch, ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T, C, main_stream);
        if (!hFish->isRemater()) {
            gBUFF->scratch->Print("proj_out", 0x0, dump_flag, B * T * nEmbed);
            gBUFF->residual->Print("residual", 0x0, dump_flag, B * T * nEmbed);
            residual_forward(ToX(out), ToX(gBUFF->residual), ToX(gBUFF->scratch), B * T * nEmbed, main_stream);
            assert(fuseNorm == nullptr && "Try fuse normal later...");
            // if (fuseNorm != nullptr) {
            //     float *mean = TO<float>(fuseNorm->mean), *rstd = TO<float>(fuseNorm->rstd);
            //     layernorm_forward(ToX(fuseNorm->out), mean, rstd, ToX(out), ToX(fuseNorm->w), ToX0(fuseNorm->b), B * T, 1, C, main_stream);
            // }
        } else {
        }
        SYNC_DEVICE();
        return out;
    } else {  //  Backward
        NvtxRange range(name.c_str(), 1);
        INSPECT_THIS;
        // Q.w->Print("Qw", 1, dump_flag);
        assert(inpL == gBUFF->delta);
        delta->Print("delta", 0x0, dump_flag);
        proj_cat.Back(gBUFF->tmpDelta, attn, gBUFF->delta, nullptr);

        hGensor delta_qkv = gBUFF->bt4c;
        if (hFish->config.scheduling.strategy != MEM_STRATEGY::MEM_SWAP_GUOKE) {  //    remater_qkv
            hGTensor norm_out = norm.out;
            if (isSeparateQKV) {
                Q.Forw(Q.out, norm_out), K.Forw(K.out, norm_out), V.Forw(V.out, norm_out);
                if (gBUFF->tmpQout != nullptr) {
                    gBUFF->tmpQout->OverWrite(Q.out);
                    gBUFF->tmpKout->OverWrite(K.out);
                }
            } else {
                Q.Forw(tmpQKV, norm_out);
            }
            if (rope != nullptr)
                rope->cuFlow(this, rope_seed, true, F_REMATER);
        }
        gBUFF->tmpDelta->Print("delta_cat", 0x0, dump_flag);
#ifdef ENABLE_CUDNN
        FUSE_cudnn(ToX(delta_qkv), ToX(gBUFF->tmpDelta), flag);
#else
        assert(0);
#endif
        // delta_qkv->Print("delta_qkv", 0x0, dump_flag, C);
        deltaQ->Print("deltaQ", 0x0, dump_flag, q_dim), deltaK->Print("deltaK", 0x0, dump_flag, kv_dim), deltaV->Print("deltaV", 0x0, dump_flag, kv_dim);
        if (rope != nullptr) {
            if (gBUFF->tmpQout != nullptr) {  //
                Q.out->OverWrite(gBUFF->tmpQout), K.out->OverWrite(gBUFF->tmpKout);
            } else if (rope->hnQ != nullptr) {
                Q.Forw(Q.out, norm.out), K.Forw(K.out, norm.out);
            }
            rope->cuFlow(this, rope_seed);
            // Q.out->Print("Q.rope",0x0,dump_flag);    K.out->Print("K.rope",0x0,dump_flag);
        }
        float* scratchF = (float*)GTensor::buff;
        if (isSeparateQKV) {
            // Q.w->Print("Qw", 0, dump_flag);  Q.b->Print("Qb", 0, dump_flag);   norm.out->Print("norm.out", 0, dump_flag);
            Q.Back(gBUFF->tmpDelta, norm.out, deltaQ, nullptr);
            // matmul_backward(ToX(gBUFF->tmpDelta), ToG(Q.w), ToG0(Q.b), (floatX*)devDeltaQ, ToX(norm.out), Q.w->GetDataX(), scratchF, B, T, Q.nOut, Q.nIn,
            //                 main_stream, false, NULL, false);
            // gBUFF->tmpDelta->Print("delta_0", 0, dump_flag);
            // K.Back(gBUFF->tmpDelta, norm.out, deltaK, nullptr, true);
            matmul_backward(ToX(gBUFF->tmpDelta), ToG(K.w), ToG0(K.b), (floatX*)devDeltaK, ToX(norm.out), K.w->GetDataX(), scratchF, B, T, K.nIn, K.nOut,
                            main_stream, false, NULL, true);
            // gBUFF->tmpDelta->Print("delta_1", 0, dump_flag);
            // K.w->Print("Kw", 1, dump_flag);
            // V.Back(gBUFF->tmpDelta, norm.out, deltaV, nullptr, true);
            matmul_backward(ToX(gBUFF->tmpDelta), ToG(V.w), ToG0(V.b), (floatX*)devDeltaV, ToX(norm.out), V.w->GetDataX(), scratchF, B, T, V.nIn, V.nOut,
                            main_stream, false, NULL, true);
            // gBUFF->tmpDelta->Print("delta_2", 0, dump_flag);
            // V.w->Print("Vw", 1, dump_flag);
        } else {
            matmul_backward(ToX(gBUFF->tmpDelta), ToG(Q.w), ToG0(Q.b), ToX(delta_qkv), ToX(norm.out), ToX(Q.w), scratchF, B, T, C_qkv, 3 * C_qkv, main_stream);
            gBUFF->tmpDelta->Print("delta_3", 0, dump_flag);
        }

        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        gBUFF->tmpDelta->Print("DLN1", 0, dump_flag);
        norm.cuFlow(gBUFF->tmpDelta);  // would backpropagatioin to gBUFF->delta
        gBUFF->delta->Print("back of QKV", 0, dump_flag);
        SYNC_DEVICE();
        return gBUFF->delta;
    }
    return nullptr;
}
