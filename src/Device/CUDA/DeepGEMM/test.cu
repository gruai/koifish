#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <iostream>

#include "./fp8_gemm.cuh"
#include "cutlass/cutlass.h"

using namespace deep_gemm;
/*
    https://github.com/deepseek-ai/DeepGEMM

    Do a normal GEMM with FP8 inputs and BF16 output, with 1x128 LHS scaling and 128x128 RHS scaling.
    LHS, RHS, RHS scaling factors, and output tensors must be in contiguous format.
    RHS and RHS scaling factors are required to be transposed.
    The LHS scaling tensor requires TMA-aligned transposed format, if your input does not match the requirement,
        this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
             the second element is an FP32 128x128 scaling tensor for RHS of shape `[⌈n / 128⌉, ⌈k / 128⌉]`.
        out: the BF16 output tensor of shape `[m, n]`, representing the result.
*/
void DS_gemm(const uint32_t N0, const uint32_t K0, const uint32_t BLOCK_M0, const uint32_t BLOCK_N0, const uint32_t NUM_STAGES,
             const uint32_t NUM_TMA_MULTICAST, int flag) {
    int num_sms;
    uint32_t smem_size;
    // Templated args from Python JIT call
    constexpr auto N = 1024, K = 768, BLOCK_M = 768, BLOCK_N = 768, kNumStages = 768, kNumTMAMulticast = 768;

    // Make a templated GEMM
    using GemmType = Gemm<N, K, BLOCK_M, BLOCK_N, 128, 1, kNumStages, kNumTMAMulticast, GemmType::Normal>;

    // Launch kernel
    auto tma_a_desc        = GemmType::make_2d_tma_a_desc(lhs, m);
    auto tma_b_desc        = GemmType::make_2d_tma_b_desc(rhs);
    auto tma_scales_a_desc = GemmType::make_2d_tma_scales_a_desc(lhs_scales, m);
    auto tma_d_desc        = GemmType::make_2d_tma_d_desc(out, m);
    GemmType::run(out, rhs_scales, nullptr, m, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc, stream, num_sms, smem_size);
}

extern "C" void launch(void* __raw_lhs, void* __raw_rhs, void* __raw_scale, void* __raw_out, bool enable_double_streams, void* __raw_stream,
                       int& __return_code) {
    // Cast raw types (if needed)
    auto lhs    = reinterpret_cast<__nv_fp8_e4m3*>(__raw_lhs);
    auto rhs    = reinterpret_cast<__nv_fp8_e4m3*>(__raw_rhs);
    auto scale  = reinterpret_cast<float*>(__raw_scale);
    auto out    = reinterpret_cast<__nv_bfloat16*>(__raw_out);
    auto stream = reinterpret_cast<cudaStream_t>(__raw_stream);

    std::cout << reinterpret_cast<uint64_t>(lhs) << std::endl;
    std::cout << reinterpret_cast<uint64_t>(rhs) << std::endl;
    std::cout << reinterpret_cast<uint64_t>(scale) << std::endl;
    std::cout << reinterpret_cast<uint64_t>(out) << std::endl;
    std::cout << enable_double_streams << std::endl;
    std::cout << reinterpret_cast<uint64_t>(stream) << std::endl;
}
