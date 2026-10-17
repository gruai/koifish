/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  All cu functions that call cudnn
 *  \brief
 *  \author Yingshi Chen
 */

#include "../../Manifold/Neuron.hpp"
#include "../../Shoal/Fish.hpp"
#include "../Utils/GST_obj.hpp"
#include "./cuda_common.h"
#include "./kernel/operator.cuh"
// #define __USE_CUDNN__
#ifdef __USE_CUDNN__
#include "cudnn_frontend.h"
namespace fe = cudnn_frontend;
#if defined(USE_FP16_BASELINE)
#define CUDNN_16BIT fe::DataType_t::HALF
#else  // Default to bfloat16
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#endif

// B, Hq, Hkv, T, head_dim, x
using QKV_KEY6 = std::tuple<int, int, int, int, int, int, int>;
enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID,
    STATS_UID,
    BIAS_UID,
    DBIAS_UID,
    SEQ_LEN_Q_UID,
    SEQ_LEN_KV_UID,
};

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

static bool is_inference_only = false;
size_t cudnn_qkv_forw(Fish* hFish, int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn, bool isCausal = false, bool padding_mask = false, int flag = 0x0) {
    auto key = std::make_tuple(B, Hq, Hkv, T, HS, (int)is_inference_only, (int)isCausal);
    if (cudnn_graph_fwd.find(key) != cudnn_graph_fwd.end())
        return 0x0;
    auto graph = std::make_shared<fe::graph::Graph>();
#if defined(USE_BF16_BASELINE) || defined(USE_FP16_BASELINE)
    graph->set_io_data_type(fe::DataType_t::BFLOAT16).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
#else
    graph->set_io_data_type(fe::DataType_t::FP8_E5M2).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
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
    sdpa_options.set_causal_mask(isCausal);  // hFish->config.model.isCausalMask
    // if (has_attn_bias) {
    //     auto bias = graph->tensor(
    //         fe::graph::Tensor_attributes().set_name("bias").set_uid(BIAS_UID).set_dim({b, 1, s_q, s_kv}).set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
    //     sdpa_options.set_bias(bias);
    // }

    if (padding_mask) {
        auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_q")
                                        .set_uid(SEQ_LEN_Q_UID)
                                        .set_dim({B, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({B, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }
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
    // if (need > workspace_size) {
    //     if (workspace_size > 0) {
    //         cudaCheck(cudaFree(cudnn_workspace));
    //     }
    //     workspace_size = graph->get_workspace_size()*10;
    //     cudaCheck(cudaMalloc(&cudnn_workspace, workspace_size));
    // }
    cudnn_graph_fwd.insert({key, graph});
    _INFO("[cudnn_qkv_] fore@[%d,%d,%d,%d,%d,%d,causal=%d]\n", B, Hq, Hkv, T, HS, (int)is_inference_only, (int)isCausal);
    return 0x0;
}

size_t cudnn_qkv_back(Fish* hFish, int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn, bool isCausal = false, bool padding_mask = false) {
    QKV_KEY6 key = std::make_tuple(B, Hq, Hkv, T, HS, 0x0, (int)isCausal);
    if (cudnn_graph_bwd.find(key) != cudnn_graph_bwd.end())
        return 0x0;
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
                                     .set_causal_mask(isCausal)  // hFish->config.model.isCausalMask
                                     .set_attn_scale(attn_scale);

    /*if (has_attn_bias) {
        // auto bias = graph->tensor(
        //     fe::graph::Tensor_attributes().set_name("bias").set_uid(BIAS_UID).set_dim({B, 1, s_q, s_kv}).set_stride({s_q * s_kv, s_q * s_kv, s_kv, 1}));
        // sdpa_options.set_bias(bias);
        // auto dbias = graph->tensor(fe::graph::Tensor_attributes()
        //                                .set_name("dbias")
        //                                .set_uid(DBIAS_UID)
        //                                .set_dim({1, h_q, s_q, s_kv})
        //                                .set_stride({s_q * s_kv * h_q, s_q * s_kv, s_kv, 1}));
        // sdpa_backward_options.set_dbias(dbias);
    }*/

    if (padding_mask) {
        auto seq_q  = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_q")
                                        .set_uid(SEQ_LEN_Q_UID)
                                        .set_dim({B, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        auto seq_kv = graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_kv")
                                        .set_uid(SEQ_LEN_KV_UID)
                                        .set_dim({B, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
        sdpa_backward_options.set_padding_mask(padding_mask).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
    }

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
    if (graph->get_workspace_size() > GTensor::workspace_size) {
        if (GTensor::workspace_size > 0) {
            cudaCheck(cudaFree(GTensor::qkv_workspace));
        }
        GTensor::workspace_size = graph->get_workspace_size();  // 1008599040  2.1G for GPT2_LARGE
        // cudaCheck(cudaMalloc(&GTensor::qkv_workspace, GTensor::workspace_size));
    }

    cudnn_graph_bwd.insert({key, graph});
    _INFO("[cudnn_qkv_] back@[%d,%d,%d,%d,%d,%d,causal=%d]\n", B, Hq, Hkv, T, HS, (int)is_inference_only, (int)isCausal);
    return GTensor::workspace_size;
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
    // if (GTensor::qkv_workspace != NULL) {
    //     cudaCheck(cudaFree(GTensor::qkv_workspace));
    // }
    cuDNNCheck(cudnnDestroy(cudnn_handle));
}

#elif defined __USE_TILELANG__
#include "../tile_kernel/tl_kernels.hpp"
#else
// defines: attention_forward, attention_backward
#endif

bool SelfAttention::FUSE_qkv(floatX* dqkvr, floatX* dout, int flag) {
    NVTX_RANGE_FN();
    float* devLse    = TO<float>(lse);  //  (B, Hq, T)
    float attn_scale = 1.0 / sqrtf(head_dim * 1.0f);
    // int group        = n_head / n_head_kv;

    UpdateQKVPack();

#ifdef __USE_CUDNN__
    assert(cudnn_handle != nullptr);
    cuDNNCheck(cudnnSetStream(cudnn_handle, main_stream));
    var_packs = {
        {Q_UID, devQ},       {K_UID, devK},       {V_UID, devV},       {O_UID, qk_v->data}, {dO_UID, dout},
        {dQ_UID, devDeltaQ}, {dK_UID, devDeltaK}, {dV_UID, devDeltaV}, {STATS_UID, devLse}, {Attn_scale_UID, &attn_scale},
    };
    if (isVarLen) {
        var_packs[SEQ_LEN_Q_UID]  = devQlen;
        var_packs[SEQ_LEN_KV_UID] = devKVlen;
    }
#elif defined __USE_TILELANG__
    dump_flag = -1;
#endif
    QKV_KEY6 key = std::make_tuple(B, n_head, n_head_kv, T, head_dim, 0x0, causalMode);
    if (isForward()) {
#ifdef __USE_CUDNN__
        assert(cudnn_graph_fwd.find(key) != cudnn_graph_fwd.end());
        auto graph = cudnn_graph_fwd[key];
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, GTensor::qkv_workspace));
#elif defined __USE_TILELANG__
        //  K,  Output,  Q,  V, float* lse
        flash_fwd_T64_32_S36864_bfloat16<<<dim3(T / 64, n_head, B), dim3(128, 1, 1), 36864, main_stream>>>((floatX*)devK, ToX(qk_v), (floatX*)devQ,
                                                                                                           (floatX*)devV, devLse);
#else
        assert(0);
#endif
        qk_v->Print("[qk]v", 0x0, dump_flag);
    } else {  // Backward      dout=>(dQ,dK,dV)
#ifdef __USE_CUDNN__
        assert(devDeltaQ == dqkvr);
        assert(cudnn_graph_bwd.find(key) != cudnn_graph_bwd.end());
        auto graph = cudnn_graph_bwd[key];  // cudnn_qkv_back(B, n_head, n_head_kv, T, HS, 0x0);
        // Q.w->Print("Q.w0", 0, dump_flag);  // float a = tNormOf(Q.w);
        checkCudnnFE(graph->execute(cudnn_handle, var_packs, GTensor::qkv_workspace));
#elif defined __USE_TILELANG__
        assert(devDeltaQ == dqkvr);
        float *deltaOdO = (float*)GTensor::qkv_workspace, *deltaQ32 = deltaOdO + B * n_head * T;
        float *dK_group = (float*)(deltaQ32 + B * n_head * head_dim * T), *dV_group = dK_group + B * n_head_kv * head_dim * T * group;
        //  float* Delta, __nv_bfloat16* O, __nv_bfloat16* dO
        flash_bwd_preprocess_T32_32_S0_bfloat16<<<dim3(n_head, T / 32, B), dim3(128, 1, 1), 0, main_stream>>>(deltaOdO, ToX(qk_v), dout);
        PrintTensor<float>("deltaOdO", deltaOdO, true, B, n_head, T, 1, -1);
        PrintTensor<float>("deltaQ32", deltaQ32, true, B, n_head, T, head_dim, -1);
        // CUBIN_1();
        flash_bwd_T64_16_S49152_bfloat16<<<dim3(n_head, T / 64, B), dim3(128, 1, 1), 49152, main_stream>>>(
            deltaOdO, (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, deltaQ32, dV_group, devLse
            //    deltaOdO, (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, dV_group, devLse
            //     // (floatX*)devK, (floatX*)devQ, (floatX*)devV, dK_group, dout, dV_group, devLse
        );
        cudaStreamSynchronize(main_stream);
        CHECK_LAST_CUDA_ERROR();
        cudaDeviceSynchronize();
        PrintTensor<float>("deltaQ32", deltaQ32, true, B, n_head, T, head_dim, -1);
        PrintTensor<float>("dK_group", dK_group, true, B, n_head_kv, T, head_dim, -1);
        PrintTensor<float>("dV_group", dV_group, true, B, n_head_kv, T, head_dim, -1);

        //    float* dQ, dQ_out
        flash_bwd_postprocess_T64_64_S0_bfloat16<<<dim3(T / 64, n_head, B), dim3(128, 1, 1), 0, main_stream>>>(deltaQ32, dqkvr);

#else
        assert(0);
#endif
        // Q.w->Print("Q.w1", 0, dump_flag);  // a = tNormOf(Q.w);
        // dqkvr->Print("delta_qkv", 0x0, dump_flag, C);
        deltaQ->Print("deltaQ", 0x0, dump_flag, q_dim), deltaK->Print("deltaK", 0x0, dump_flag, kv_dim), deltaV->Print("deltaV", 0x0, dump_flag, kv_dim);
    }
    cudaCheck(cudaGetLastError());
    return true;
}

// size_t cudnn_qkv_forw(Fish*, int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn, bool has_attn_bias = false, bool padding_mask = false, int flag = 0x0);
// size_t cudnn_qkv_back(Fish*, int B, int Hq, int Hkv, int T, int HS, QKV_PACK qkv4dnn, bool has_attn_bias = false, bool padding_mask = false);
// int test_FA2();
bool Fish::AllocBuffer(int flag) {
    try {
#ifdef __USE_TVM__
        test_FA2();
#endif
        int B = config.n_batch(), T = curContextLen(LIMIT), NH = config.n_head(), head_dim = config.head_dim();
        int group = NH / config.n_head_kv();
        if (isLocalInfer) {
            if (config.phase == P_CHAT_1) {  // P_EVAL no need cache!
                // hCache = std::make_shared<KVCache>(this);
            }
        }
        if (isModel({MD_QWEN, MD_AR_QWEN})) {
        } else {
            hCache = std::make_shared<KVCache>(this);
        }

        bool isCausal = config.model.isCausalMask, padding_mask = phase == P_SFT;
        if (phase != P_CHAT_1) {
#ifdef __USE_CUDNN__
            cudnn_qkv_forw(this, B, NH, config.n_head_kv(), T, config.head_dim(), config.model.qkv4dnn, isCausal, padding_mask);
            if (!isAtPhase({P_EVAL_, P_CHAT_N})) {
                size_t alloc = cudnn_qkv_back(this, B, NH, config.n_head_kv(), T, config.head_dim(), config.model.qkv4dnn, isCausal, padding_mask);
                _INFO("\tcudnn_qkv_back = %.5gM\n", alloc / 1.0e6);
            }
            if (isModel({MD_AR_QWEN})) {
                cudnn_qkv_forw(this, B, NH, config.n_head_kv(), T, config.head_dim(), config.model.qkv4dnn, !isCausal, padding_mask);
                if (!isAtPhase({P_EVAL_, P_CHAT_N})) {
                    size_t alloc = cudnn_qkv_back(this, B, NH, config.n_head_kv(), T, config.head_dim(), config.model.qkv4dnn, !isCausal, padding_mask);
                    _INFO("\tcudnn_qkv_back = %.5gM\n", alloc / 1.0e6);
                }
            }
#endif
            size_t mostQKV          = B * T * NH * head_dim + B * NH * T +                // dQ[[batch, seq_len, heads, dim_qk]] + Delta[batch, heads, seq_len]
                                      B * T * config.n_head_kv() * head_dim * group * 4;  // [groups, batch, seq_len, head_kv, dim_v]
            GTensor::workspace_size = std::max(GTensor::workspace_size, mostQKV * 4);     // 134742016
            GTensor::workspace_size = std::max(GTensor::workspace_size, (size_t)32 * 1024 * 1024);
            assert(GTensor::workspace_size >= 32 * 1024 * 1024);
            cudaCheck(cudaMalloc(&GTensor::qkv_workspace, GTensor::workspace_size));
        } else {
            GTensor::workspace_size = std::max(GTensor::workspace_size, (size_t)32 * 1024 * 1024);
            cudaCheck(cudaMalloc(&GTensor::qkv_workspace, GTensor::workspace_size));
        }

        return true;
    } catch (const std::exception& e) {
        _INFO("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char* info) {
        _INFO("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}