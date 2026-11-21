/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Utility functions
 *  \author Yingshi Chen
 */
#include <cstdio>
#include <filesystem>  // C++17

#include "../g_float.hpp"
#include "./os/GST_os.hpp"
#include "GST_log.hpp"
#include "GST_obj.hpp"
#include "GST_util.hpp"

#define GGML_COMMON_DECL_CPP
#include "../../third_party/strata/build/ggml-tq-overlay/ggml-common-tq.h"
#include "../../third_party/strata/sycl/third_party/ggml/ggml-common.h"
#ifndef STRATA_VERIFY_MAX_T
#define STRATA_VERIFY_MAX_T 8
#endif
constexpr int MAX_NCOLS = STRATA_VERIFY_MAX_T > 8 ? STRATA_VERIFY_MAX_T : 8;

constexpr int QK            = 256;
constexpr int Q8K           = 32;
constexpr int QI            = 32;
constexpr int VDR           = 2;
constexpr int WARPS         = 4;
constexpr int WARP          = 32;
constexpr int QUANT_THREADS = 256;

constexpr int Q8P_PERSIST_GRID   = 640;
constexpr int Q8P_PERSIST_MAX_IN = 4096;
// `load` makes one cudaMalloc per projection matrix, and the driver backs each call with 2 MiB granules
// (measured on the RTX 5060: 300 buffers of 6.73 MiB asked 2019 MiB and cost 2400).  So a stage's real
// footprint is the granule-rounded sum, not the payload sum.  This rule - a granule for every matrix of a
// MiB or more, the payload for the rest - prices the iq3_s pack's 300 matrices at 2255 MiB where they
// really take 2238 (+0.8%).  It over-prices, which is the safe side.  (Pricing every matrix at a granule is
// far worse, 2378 MiB: the driver packs the sub-MiB matrices into the slack of the rounded ones.)
constexpr uint64_t kAllocGranule = 2ull << 20;
constexpr uint64_t kSmallAlloc   = 1ull << 20;
uint64_t native_alloc_bytes(uint64_t bytes) { return bytes < kSmallAlloc ? bytes : (bytes + kAllocGranule - 1) / kAllocGranule * kAllocGranule; }

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 && offsetof(Q3KBlock, scales) == 96 &&
              offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 && offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 && offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 && offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 && offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 && offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14 ||
           ggml_type == 20 || ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 || ggml_type == 21 ||
           ggml_type == 22 || ggml_type == 29;
};

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS)
        throw std::invalid_argument("native MMVQ requires 1 <= ncols <= MAX_NCOLS");
}

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16:
            return (size_t)(n / 256) * sizeof(block_iq2_xxs);
        case 17:
            return (size_t)(n / 256) * sizeof(block_iq2_xs);
        case 18:
            return (size_t)(n / 256) * sizeof(block_iq3_xxs);
        case 20:
            return (size_t)(n / 32) * sizeof(block_iq4_nl);
        case 21:
            return (size_t)(n / 256) * sizeof(block_iq3_s);
        case 22:
            return (size_t)(n / 256) * sizeof(block_iq2_s);
        case 29:
            return (size_t)(n / 256) * sizeof(block_iq1_m);
        case 23:
            return (size_t)(n / 256) * sizeof(block_iq4_xs);
        case 11:
            return (size_t)(n / 256) * sizeof(block_q3_K);
        case 42:
            return (size_t)(n / 64) * sizeof(block_q2_0);
        case 12:
            return (size_t)(n / 256) * sizeof(block_q4_K);
        case 13:
            return (size_t)(n / 256) * sizeof(block_q5_K);
        case 14:
            return (size_t)(n / 256) * sizeof(block_q6_K);
        case 7:
            return (size_t)(n / 32) * sizeof(block_q5_1);
        case 6:
            return (size_t)(n / 32) * sizeof(block_q5_0);
        case 2:
            return (size_t)(n / 32) * sizeof(block_q4_0);
        case 3:
            return (size_t)(n / 32) * sizeof(block_q4_1);
        case 40:
            return (size_t)(n / 64) * sizeof(block_nvfp4);
        case 8:
            return (size_t)(n / 32) * sizeof(block_q8_0);
        case 30:
            return (size_t)n * 2;  // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        case 144:
            return (size_t)(n / 128) * sizeof(block_tq2_t);  // APR trellis types
        case 145:
            return (size_t)(n / 128) * sizeof(block_tqk6);
        case 146:
            return (size_t)(n / 128) * sizeof(block_tqk7);
        case 147:
            return (size_t)(n / 128) * sizeof(block_tqp5);
        case 148:
            return (size_t)(n / 128) * sizeof(block_tqp6);
        case 149:
            return (size_t)(n / 128) * sizeof(block_tqp7);
        default:
            return 0;
    }
}

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

bool q8_sm60_on() {
#if defined(__HIPCC__)
    return false;  // a CUDA kernel for one NVIDIA chip: HIP builds never compile it
#else
    static const bool on = [] {
        const char* v  = std::getenv("STRATA_Q8_SM60");
        const bool yes = v && v[0] == '1' && v[1] == '\0';
        if (yes)
            std::fprintf(stderr, "strata q8 sm60: Q8_0 decode GEMVs on the Pascal kernel (STRATA_Q8_SM60=1)\n");
        return yes;
    }();
    return on;
#endif
}
bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled || q8_sm60_on();
}
bool q8_packed_shape(int n_in, int n_out) {  // the shapes native_q8_0_packed_kernel was measured on
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_in <= Q8P_PERSIST_MAX_IN && n_out >= 2048;
}
bool native_q8_0_packed_eligible(int n_in, int n_out) {
    if (q8_sm60_on())
        return n_in >= 128 && n_in % 32 == 0 && n_out > 0;  // the Pascal kernel takes any of them
    return q8_packed_shape(n_in, n_out);
}
void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks      = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs                = static_cast<int8_t*>(out);
    auto* d                 = reinterpret_cast<half*>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(half));
    }
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
        case 2:
            block_elems = 32;
            block_bytes = 18;
            break;
        case 6:
            block_elems = 32;
            block_bytes = 22;
            break;
        case 7:
            block_elems = 32;
            block_bytes = 24;
            break;
        case 8:
            block_elems = 32;
            block_bytes = 34;
            break;
        case 20:
            block_elems = 32;
            block_bytes = 18;
            break;
        case 11:
            block_elems = 256;
            block_bytes = 110;
            break;
        case 12:
            block_elems = 256;
            block_bytes = 144;
            break;
        case 13:
            block_elems = 256;
            block_bytes = 176;
            break;
        case 14:
            block_elems = 256;
            block_bytes = 210;
            break;
        case 23:
            block_elems = 256;
            block_bytes = 136;
            break;
        case 42:
            block_elems = 64;
            block_bytes = 18;
            break;
        case 16:
        case 17:
        case 18:
        case 21:
        case 22:
        case 29:
            block_elems = 256;
            block_bytes = (int)iq_row_bytes(ggml_type, 256);
            break;
        default:
            throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0)
        throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}