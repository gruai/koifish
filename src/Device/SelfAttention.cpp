#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
#ifdef __USE_TVM__
#ifndef DMLC_ALWAYS_INLINE
#define DMLC_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

#ifndef TVM_ALWAYS_INLINE
#define TVM_ALWAYS_INLINE inline __attribute__((always_inline))
#endif
#include <dlpack/dlpack.h>
// #include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
// #include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
// #include <tvm/runtime/ndarray.h>

#include "../Utils/GST_obj.hpp"
#include "../Tensor/GTensor.hpp"
#include "CUDA/cuda_common.h"

class FlashAttentionWrapper {
   private:
    void* handle;

    // 函数指针类型定义
    using ForwardFunc = void (*)(float* output,       // [batch_size, seq_len, num_heads, head_dim]
                                 const float* query,  // [batch_size, seq_len, num_heads, head_dim]
                                 const float* key,    // [batch_size, seq_len, num_heads, head_dim]
                                 const float* value,  // [batch_size, seq_len, num_heads, head_dim]
                                 int batch_size, int seq_len, int num_heads, int head_dim, float dropout_prob, bool is_causal, cudaStream_t stream);

    using BackwardFunc = void (*)(float* dq,          // 梯度 [batch_size, seq_len, num_heads, head_dim]
                                  float* dk,          // 梯度
                                  float* dv,          // 梯度
                                  const float* dout,  // 输出梯度
                                  const float* query, const float* key, const float* value,
                                  const float* output,  // 前向输出
                                  int batch_size, int seq_len, int num_heads, int head_dim, float dropout_prob, bool is_causal, cudaStream_t stream);

    ForwardFunc forward_func;
    BackwardFunc backward_func;

   public:
    FlashAttentionWrapper() : handle(nullptr), forward_func(nullptr), backward_func(nullptr) {}
    ~FlashAttentionWrapper() {
        if (handle) {
            dlclose(handle);
        }
    }

    //  pip install flash-attn --no-build-isolation
    // find $(python -c "import site; print(site.getsitepackages()[0])") -name "*flash_attn*.so" 2>/dev/null
    //  /home/vipuser/miniconda3/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so

    bool load_library(const std::string& so_path) {
        handle = dlopen(so_path.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Failed to load library: " << dlerror() << std::endl;
            return false;
        }

        forward_func = (ForwardFunc)dlsym(handle, "flash_attention_forward");
        if (!forward_func) {
            std::cerr << "Failed to load forward function: " << dlerror() << std::endl;
            dlclose(handle);
            handle = nullptr;
            return false;
        }

        // 加载 backward 函数
        backward_func = (BackwardFunc)dlsym(handle, "flash_attention_backward");
        if (!backward_func) {
            std::cerr << "Failed to load backward function: " << dlerror() << std::endl;
            dlclose(handle);
            handle = nullptr;
            return false;
        }

        std::cout << "Successfully loaded FlashAttention library" << std::endl;
        return true;
    }

    void forward(float* output, const float* query, const float* key, const float* value, int batch_size, int seq_len, int num_heads, int head_dim,
                 float dropout_prob = 0.0f, bool is_causal = true, cudaStream_t stream = 0) {
        if (!forward_func) {
            throw std::runtime_error("Forward function not loaded");
        }

        forward_func(output, query, key, value, batch_size, seq_len, num_heads, head_dim, dropout_prob, is_causal, stream);
    }

    void backward(float* dq, float* dk, float* dv, const float* dout, const float* query, const float* key, const float* value, const float* output,
                  int batch_size, int seq_len, int num_heads, int head_dim, float dropout_prob = 0.0f, bool is_causal = true, cudaStream_t stream = 0) {}
};

int test_FA2() {
    try {
        auto mod = tvm::ffi::Module::LoadFromFile("/root/tilelang/build/lib/libtilelang.so");
        string sName = "main";  //"default_function";
        auto f   = mod->GetFunction(sName.c_str());
        if (f == nullptr) {
            _ERROR( "[TILL] function=\"%s\" not found!", sName.c_str());
           K_EXIT_NOW(KOIFISH_TILL_FUNCTION);
        }
        // DLTensor a_tensor, b_tensor, c_tensor;
        // f(&a_tensor, &b_tensor, &c_tensor);
        return 0x0;
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

    cudaSetDevice(0);

    int batch_size     = 2;
    int seq_len        = 1024;
    int num_heads      = 16;
    int head_dim       = 64;
    size_t tensor_size = batch_size * seq_len * num_heads * head_dim;
    string so_path     = "/home/vipuser/miniconda3/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so";
    FlashAttentionWrapper flash_attn;
    if (!flash_attn.load_library(so_path)) {
        std::cerr << "Failed to load FlashAttention library" << std::endl;
        return 1;
    }

    float *d_query, *d_key, *d_value, *d_output;
    float *d_dout, *d_dq, *d_dk, *d_dv;

    cudaMalloc(&d_query, tensor_size * sizeof(float));
    cudaMalloc(&d_key, tensor_size * sizeof(float));
    cudaMalloc(&d_value, tensor_size * sizeof(float));
    cudaMalloc(&d_output, tensor_size * sizeof(float));
    cudaMalloc(&d_dout, tensor_size * sizeof(float));
    cudaMalloc(&d_dq, tensor_size * sizeof(float));
    cudaMalloc(&d_dk, tensor_size * sizeof(float));
    cudaMalloc(&d_dv, tensor_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    try {
        std::cout << "Running forward pass..." << std::endl;
        flash_attn.forward(d_output, d_query, d_key, d_value, batch_size, seq_len, num_heads, head_dim,
                           0.1f,  // dropout_prob
                           true,  // is_causal
                           stream);

        std::cout << "Running backward pass..." << std::endl;
        flash_attn.backward(d_dq, d_dk, d_dv, d_dout, d_query, d_key, d_value, d_output, batch_size, seq_len, num_heads, head_dim,
                            0.1f,  // dropout_prob
                            true,  // is_causal
                            stream);

        cudaStreamSynchronize(stream);
        std::cout << "FlashAttention executed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    cudaStreamDestroy(stream);
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_dout);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);

    return 0;
}
#endif

#ifdef __USE_FLASH_ATTENTION__
#include "./flash_attn/src/flash.h"
#include "./flash_attn/src/static_switch.h"
using namespace flash;
void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q, const size_t seqlen_k, const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
                      const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const hGTensor q, const hGTensor k, const hGTensor v, hGTensor out, void* cu_seqlens_q_d, void* cu_seqlens_k_d, void* seqused_k,
                      void* p_d, void* softmax_lse_d, float p_dropout, float softmax_scale, int window_size_left, int window_size_right, const float softcap,
                      bool seqlenq_ngroups_swapped = false, const bool unpadded_lse = false) {
    // Reset the parameters
    params = {};
    /*
        params.is_bf16 = q.dtype() == torch::kBFloat16;

        // Set the pointers and strides.
        params.q_ptr = q.data_ptr();
        params.k_ptr = k.data_ptr();
        params.v_ptr = v.data_ptr();
        // All stride are in elements, not bytes.
        params.q_row_stride = q.stride(-3);
        params.k_row_stride = k.stride(-3);
        params.v_row_stride = v.stride(-3);
        params.q_head_stride = q.stride(-2);
        params.k_head_stride = k.stride(-2);
        params.v_head_stride = v.stride(-2);
        params.o_ptr = out.data_ptr();
        params.o_row_stride = out.stride(-3);
        params.o_head_stride = out.stride(-2);

        if (cu_seqlens_q_d == nullptr) {
            params.q_batch_stride = q.stride(0);
            params.k_batch_stride = k.stride(0);
            params.v_batch_stride = v.stride(0);
            params.o_batch_stride = out.stride(0);
            if (seqlenq_ngroups_swapped) {
                 params.q_batch_stride *= seqlen_q;
                 params.o_batch_stride *= seqlen_q;
            }
        }

        params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
        params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
        params.seqused_k = static_cast<int *>(seqused_k);

        // P = softmax(QK^T)
        params.p_ptr = p_d;

        // Softmax sum
        params.softmax_lse_ptr = softmax_lse_d;

        // Set the dimensions.
        params.b = b;
        params.h = h;
        params.h_k = h_k;
        params.h_h_k_ratio = h / h_k;
        params.seqlen_q = seqlen_q;
        params.seqlen_k = seqlen_k;
        params.seqlen_q_rounded = seqlen_q_rounded;
        params.seqlen_k_rounded = seqlen_k_rounded;
        params.d = d;
        params.d_rounded = d_rounded;

        // Set the different scale values.
        #ifdef FLASHATTENTION_DISABLE_SOFTCAP
            TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
        #endif
        if (softcap > 0.0) {
            params.softcap = softmax_scale / softcap;
            params.scale_softmax = softcap;
            params.scale_softmax_log2 = softcap * M_LOG2E;
        } else{
            // Remove potential NaN
            params.softcap = 0.0;
            params.scale_softmax = softmax_scale;
            params.scale_softmax_log2 = softmax_scale * M_LOG2E;
        }

        // Set this to probability of keeping an element to simplify things.
        params.p_dropout = 1.f - p_dropout;
        // Convert p from float to int so we don't have to convert the random uint to float to compare.
        // [Minor] We want to round down since when we do the comparison we use <= instead of <
        // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
        // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
        params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
        params.rp_dropout = 1.f / params.p_dropout;
        params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
        TORCH_CHECK(p_dropout < 1.f);
        #ifdef FLASHATTENTION_DISABLE_DROPOUT
            TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
        #endif

        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        params.is_causal = window_size_left < 0 && window_size_right == 0;

        if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
        if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
        params.window_size_left = window_size_left;
        params.window_size_right = window_size_right;

        #ifdef FLASHATTENTION_DISABLE_LOCAL
            TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
                "This flash attention build does not support local attention.");
        #endif

        params.is_seqlens_k_cumulative = true;

        #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
            TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
        #endif

        params.unpadded_lse = unpadded_lse;
        params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;*/
}

std::tuple<hGTensor, hGTensor> set_params_splitkv(Flash_fwd_params& params, const int batch_size, const int num_heads, const int head_size,
                                                  const int max_seqlen_k, const int max_seqlen_q, const int head_size_rounded, const float p_dropout,
                                                  const int num_splits, const int num_sm) {  //, struct c10::TensorOptions opts

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n      = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits      = num_splits;
    hGTensor softmax_lse_accum;
    hGTensor out_accum;

    /*if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }*/

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}

int mha_fwd(hGTensor q,                              // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
            const hGTensor k,                        // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
            const hGTensor v,                        // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
            hGTensor out,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
            std::optional<hGTensor>& alibi_slopes_,  // num_heads or batch_size x num_heads
            const float p_dropout, const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, const float softcap,
            const bool return_softmax, std::optional<hGTensor> gen_) {
    // Otherwise the kernel will be launched from cuda:0 device
    /*at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    //TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    //TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    //TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    //TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    //CHECK_DEVICE(q); //CHECK_DEVICE(k); //CHECK_DEVICE(v);

    //TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    //TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    //TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");*/

    const auto sizes      = q->shape;
    const int batch_size  = sizes[0];
    int seqlen_q          = sizes[1];
    int num_heads         = sizes[2];
    const int head_size   = sizes[3];
    const int seqlen_k    = k->shape[1];
    const int num_heads_k = k->shape[2];
    // TORCH_CHECK(batch_size > 0, "batch size must be positive");
    // TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    // TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    // TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (window_size_left >= seqlen_k) {
        window_size_left = -1;
    }
    if (window_size_right >= seqlen_k) {
        window_size_right = -1;
    }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
        is_causal = false;
    }
    if (is_causal) {
        window_size_right = 0;
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f &&
                                        head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        // q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q  = ngroups;
        num_heads = num_heads_k;
    }

    // CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    // CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    // CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    auto round_multiple         = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_q_rounded  = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded  = round_multiple(seqlen_k, 128);

    // auto opts = q.options();
    // auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    hGTensor p, softmax_lse;
    // Only return softmax if there's dropout to reduce compilation time
    // if (return_softmax) {
    //     //TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    //     p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    // }
    // else {
    //     p = torch::empty({ 0 }, opts);
    // }

    Flash_fwd_params params;
    set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded, q, k, v,
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr, return_softmax ? p->data : nullptr,
                     softmax_lse->data,  //.data_ptr(),
                     p_dropout, softmax_scale, window_size_left, window_size_right, softcap);

    // Keep references to these tensors to extend their lifetime
    hGTensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k, seqlen_q, head_size_rounded, p_dropout,
                                                                /*num_splits*/ 0, 168);  // get_num_sm(get_current_device())

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    uint64_t rng_state = 20260215;
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = &rng_state;

    /*if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);*/
    run_mha_fwd(params, stream);
    return 0x0;
    /*if (seqlen_k > 0) {
        // auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};*/
}
#endif