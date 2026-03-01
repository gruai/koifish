/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief  Some functions on float number
 *  \author Yingshi Chen
 */

#include <assert.h>
#include <math.h>

#include <cfloat>
#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#endif
// #include "f16cintrin.h"
#include "../CLI_params.hpp"
#include "../g_def_x.hpp"
#include "../g_float.hpp"
#include "../g_float_cpu.hpp"

typNUMBER tpNumOf(const std::string& dtype_str) {
    std::string sType = dtype_str;
    std::transform(sType.begin(), sType.end(), sType.begin(), ::toupper);
    typNUMBER type = typNUMBER::T_OTHER;
    for (auto k_float : K_FLOATS) {
        if (sType == k_float.second.name) {
            type = k_float.first;
            return type;
        }
        for (auto alias : k_float.second.alias) {
            if (sType == alias) {
                type = k_float.first;
                return type;
            }
        }
    }
    /*
    if (sType == "F32") {
        type = typNUMBER::F32;
    } else if (sType == "F16") {
        type = typNUMBER::F16;
    } else if (sType == "BF16") {
        type = typNUMBER::BF16;
    } else if (sType == "F8_E5M2") {
        type = typNUMBER::F8E5M2;
    } else if (sType == "FP8") {
        type = typNUMBER::F8E5M2;
    } else if (sType == "F8_E4M3") {
        type = typNUMBER::F8E4M3;
    } else if (sType == "I32") {
        type = typNUMBER::I32;
    } else if (sType == "I16") {
        type = typNUMBER::I16;
    } else if (sType == "I8") {
        type = typNUMBER::I8;
    } else if (sType == "U8") {
        type = typNUMBER::I8;
    } else if (sType == "TERNARY") {
        type = typNUMBER::T_SIGN;
    } else if (sType == "BINARY") {
        type = typNUMBER::T_BINARY;
    } else*/
    if (type == typNUMBER::T_OTHER) {
        std::string sErr = "Invalid typNumber@" + sType;
        assert(0 && sErr.c_str());
    }
    return type;
}

size_t FLOAT_META::nByte(size_t nElem) const {
    assert(nElem * bis % 8 == 0);
    return nElem * bis / 8;
}

//
double BitPE(typNUMBER type) {
    // int no = (int)type;
    // assert(no<sizeof(K_FLOATS)/sizeof(FLOAT_META));
    if (type == typNUMBER::T_BINARY_TILE) {
        return 0.125;
    }
    double bit = K_FLOATS[type].bis;
    return bit;
}

#if defined(__AVX2__) && defined(__F16C__)
float half_to_float(__gcc_fp16 x) { return _cvtsh_ss(x); }
__gcc_fp16 float_to_half(float x) { return _cvtss_sh(x, 0); }
#else
float half_to_float(__gcc_fp16 x) {
    assert(false && "float16 not supported on this platform");
    return 0.0f;
}
__gcc_fp16 float_to_half(float x) {
    assert(false && "float16 not supported on this platform");
    return 0;
}
#endif

float dotprod_fp16(void* w, int n, int i, float* x) {
    __gcc_fp16* r = (__gcc_fp16*)w + i * n;
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % 16 == 0);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int j = 0; j < n; j += 16) {
        __m256i rw  = _mm256_loadu_si256((__m256i*)&r[j]);
        __m128i rlo = _mm256_castsi256_si128(rw);
        __m128i rhi = _mm256_extractf128_si256(rw, 1);
        __m256 x0   = _mm256_loadu_ps(&x[j]);
        __m256 x1   = _mm256_loadu_ps(&x[j + 8]);
        acc0        = _mm256_add_ps(_mm256_mul_ps(x0, _mm256_cvtph_ps(rlo)), acc0);
        acc1        = _mm256_add_ps(_mm256_mul_ps(x1, _mm256_cvtph_ps(rhi)), acc1);
    }
    __m256 acc8 = _mm256_add_ps(acc0, acc1);
    __m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
    __m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
    return _mm_cvtss_f32(accf);
#else
    float val = 0.0f;
#pragma omp simd reduction(+ : val) simdlen(32)
    for (int j = 0; j < n; j++) {
        val += (float)(r[j]) * x[j];
    }
    return val;
#endif
}

float dotprod_fp8(void* w, int n, int i, float* x) {
    char* r = (char*)w + i * n;
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % 16 == 0);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int j = 0; j < n; j += 16) {
        __m128i rw  = _mm_loadu_si128((__m128i*)&r[j]);
        __m128i rlo = _mm_unpacklo_epi8(_mm_setzero_si128(), rw);
        __m128i rhi = _mm_unpackhi_epi8(_mm_setzero_si128(), rw);
        __m256 x0   = _mm256_loadu_ps(&x[j]);
        __m256 x1   = _mm256_loadu_ps(&x[j + 8]);
        acc0        = _mm256_add_ps(_mm256_mul_ps(x0, _mm256_cvtph_ps(rlo)), acc0);
        acc1        = _mm256_add_ps(_mm256_mul_ps(x1, _mm256_cvtph_ps(rhi)), acc1);
    }
    __m256 acc8 = _mm256_add_ps(acc0, acc1);
    __m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
    __m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
    return _mm_cvtss_f32(accf);
#else
    float val = 0.0f;
#pragma omp simd reduction(+ : val) simdlen(32)
    for (int j = 0; j < n; j++) {
        float a = fp82half(r[j]);
        val += a * x[j];
    }
    return val;
#endif
}

float dotprod_gf4(void* w, int n, int i, float* x) {
    uint32_t* r = (uint32_t*)w + i * n / 8;
#if defined(__AVX2__) && defined(__F16C__)
    assert(n % 32 == 0);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int j = 0; j < n; j += 32) {
        __m128i wg           = _mm_loadu_si128((__m128i*)&r[j / 8]);
        const __m128i wgfm   = _mm_setr_epi8(-1, 0, -1, 4, -1, 8, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1);
        __m128 wgf           = _mm_cvtph_ps(_mm_shuffle_epi8(wg, wgfm));  // note: scale 1/-4.f is baked into wgtab below
        __m256 x0            = _mm256_loadu_ps(&x[j]);
        __m256 x1            = _mm256_loadu_ps(&x[j + 8]);
        __m256 x2            = _mm256_loadu_ps(&x[j + 16]);
        __m256 x3            = _mm256_loadu_ps(&x[j + 24]);
        __m256i wgp          = _mm256_broadcastsi128_si256(wg);
        __m256 wgfp          = _mm256_castsi256_ps(_mm256_broadcastsi128_si256(_mm_castps_si128(wgf)));
        const __m256i wgbits = _mm256_setr_epi32(8, 11, 14, 17, 20, 23, 26, 29);
        const __m256 wgtab   = _mm256_setr_ps(-4 / -4.f, -3 / -4.f, -2 / -4.f, -1 / -4.f, 0 / -4.f, 1 / -4.f, 2 / -4.f, 3 / -4.f);
        __m256 w0            = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0x00), wgbits));
        __m256 w1            = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0x55), wgbits));
        __m256 w2            = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0xaa), wgbits));
        __m256 w3            = _mm256_permutevar8x32_ps(wgtab, _mm256_srlv_epi32(_mm256_shuffle_epi32(wgp, 0xff), wgbits));
        acc0                 = _mm256_add_ps(_mm256_mul_ps(w0, _mm256_mul_ps(x0, _mm256_shuffle_ps(wgfp, wgfp, 0x00))), acc0);
        acc1                 = _mm256_add_ps(_mm256_mul_ps(w1, _mm256_mul_ps(x1, _mm256_shuffle_ps(wgfp, wgfp, 0x55))), acc1);
        acc0                 = _mm256_add_ps(_mm256_mul_ps(w2, _mm256_mul_ps(x2, _mm256_shuffle_ps(wgfp, wgfp, 0xaa))), acc0);
        acc1                 = _mm256_add_ps(_mm256_mul_ps(w3, _mm256_mul_ps(x3, _mm256_shuffle_ps(wgfp, wgfp, 0xff))), acc1);
    }
    __m256 acc8 = _mm256_add_ps(acc0, acc1);
    __m128 acc4 = _mm_add_ps(_mm256_castps256_ps128(acc8), _mm256_extractf128_ps(acc8, 1));
    __m128 accf = _mm_dp_ps(acc4, _mm_set1_ps(1.0f), 0xf1);
    return _mm_cvtss_f32(accf);
#else
    float val = 0.0f;
    for (int j = 0; j < n; j += 8) {
        uint32_t wg = r[j / 8];
        for (int k = 0; k < 8; ++k) {
            val += gf4_ff(wg, k) * x[j + k];
        }
    }
    return val;
#endif
}

int cdiv(int a, int b) { return (a + b - 1) / b; }

void S_matmul(float* xout, float* x, float* w, int n, int d, const int* block_size, float* scale) {
    // W (d,n) @ x (n,) -> xout (d,)
    static float one        = 1.0f;
    int dummy_block_size[2] = {d, n};
    if (scale == nullptr) {
        scale      = &one;
        block_size = dummy_block_size;
    }
    int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
    for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
        int ii;
#pragma omp parallel for private(ii)
        for (ii = 0; ii < block_size[0]; ii++) {
            int i = scale_i * block_size[0] + ii;
            if (i >= d) {
                continue;
            }
            float val = 0.0f;
            for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
                float scale_val = scale[scale_i * scale_num_cols + scale_j];
                for (int jj = 0; jj < block_size[1]; jj++) {
                    int j = scale_j * block_size[1] + jj;
                    if (j >= n) {
                        break;
                    }
                    val += (w[i * n + j] * x[j]) * scale_val;
                }
            }
            xout[i] = val;
        }
    }
}

// Scale_matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
void S_matmul(float* xout, float* x, __gcc_fp16* w, int n, int d, const int* block_size, float* scale) {
#if defined(__AVX2__) && defined(__F16C__)
    // W (d,n) @ x (n,) -> xout (d,)
    assert(n % 16 == 0);
    assert(scale == nullptr || block_size[1] % 16 == 0);
    static float one        = 1.0f;
    int dummy_block_size[2] = {d, n};
    if (scale == nullptr) {
        scale      = &one;
        block_size = dummy_block_size;
    }
    int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
    for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
        int ii;
#pragma omp parallel for private(ii)
        for (ii = 0; ii < block_size[0]; ii++) {
            int i = scale_i * block_size[0] + ii;
            if (i >= d) {
                continue;
            }
            // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
            __m256 sumlo = _mm256_setzero_ps();
            __m256 sumhi = _mm256_setzero_ps();
            for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
                // Broadcast scale_val to all elements of a vector
                float scale_val  = scale[scale_i * scale_num_cols + scale_j];
                __m256 scale_vec = _mm256_set1_ps(scale_val);
                for (int jj = 0; jj < block_size[1]; jj += 16) {
                    int j = scale_j * block_size[1] + jj;
                    if (j >= n) {
                        break;
                    }

                    // Extract the next set of 16 float16 weights from `w` and store them
                    // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
                    __m256i wvec     = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
                    __m128i wveclo   = _mm256_extractf128_si256(wvec, 0);
                    __m128i wvechi   = _mm256_extractf128_si256(wvec, 1);
                    __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
                    __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);

                    // Scale the weight vectors
                    wveclo_ps = _mm256_mul_ps(wveclo_ps, scale_vec);
                    wvechi_ps = _mm256_mul_ps(wvechi_ps, scale_vec);

                    // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
                    __m256 xveclo = _mm256_loadu_ps(&x[j]);
                    __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);

                    // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
                    sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
                    sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
                }
            }
            // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
            __m256 sum8 = _mm256_add_ps(sumlo, sumhi);  // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
            __m128 sum4 = _mm_add_ps(                   // sum4[0:4] = sum8[0:4] + sum8[4:8]
                _mm256_extractf128_ps(sum8, 0), _mm256_extractf128_ps(sum8, 1));
            __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1);  // sum1[0] = dot(sum4, [1,1,1,1])
            xout[i]     = _mm_cvtss_f32(sum1);
        }
    }
#else
    assert(false && "float16 not supported on this platform");
#endif
}

float dotprod_fp32(void* w, int n, int row, float* x) {
    float* U = (float*)w;
    float *r = (float*)w + row * n, val = 0.0f;
#pragma omp simd reduction(+ : val) simdlen(32)
    for (int j = 0; j < n; j++) {
        val += (r[j]) * x[j];
    }
    assert(isValidF(&val));
    return val;
}

/*
   W (nOut,nIn) @ x (nIn) -> xout (nOut)    by dotprod_t
*/
void D_matvec(float* xout, float* x, void* w, float* b, int nIn, int nOut, dotprod_t dotprod) {
    //  void matmul(float* xout, float* x, void* w, float* b, int n, int d, dotprod_t dotprod)
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < nOut; i++) {
        float val = dotprod(w, nIn, i, x);
        if (b) {
            val += b[i];
        }
        xout[i] = val;
    }
}

void D_matmul_sparse(float* xout, float* x, void* w, float* b, int n, int d, int* hot, dotprod_t dotprod) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0;
        if (hot[i] == 1)
            val = dotprod(w, n, i, x);

        if (b) {
            val += b[i];
        }
        xout[i] = val;
    }
}

/*
    T_hot=0.2 80   ppl=14.5=>30.3 drop too much!
    T_hot=0.2 800  ppl=4.96=>8.2
    T_hot=0.4 800  ppl=4.96=>5.66
*/
void D_matmul_sparse_2(float* xout, float* x, void* w, float* b, int n, int d, int nHot, int* hot, float* dTemp, dotprod_t dotprod) {
    int i, c;
    float* x_sub = dTemp;
    char *w_sub = (char*)(dTemp + sizeof(float) * n), *w0 = (char*)w;
    size_t ld = sizeof(char) * d, nW = 0;
#pragma omp parallel for private(i, c)
    for (i = 0; i < nHot; i++) {
        int no   = hot[i];
        x_sub[i] = x[no];
        // row major -  memcpy(w_sub+ld*nHot, w+ld*i, ld);
        // column major
        char *dst = w_sub + i, *src = w0 + no;
        for (c = 0; c < d; c++) {
            dst[c * nHot] = src[c * n];
        }
    }
    // assert(no==nHot);

    D_matvec(xout, x_sub, w_sub, b, nHot, d, dotprod);
    // matmul(xout, x, w, b, nHot, d, dotprod);
}

// Scale_matmul supporting float8e5m2 weights via AVX2 and F16C extensions, which (1)
// allows vectorized conversion from f8e5m2 to float16 and (2) conversion from
// float16 to float32 values before calculations.
void S_matmul(float* xout, float* x, f8e5* w, int n, int d, const int* block_size, float* scale) {
#if defined(__AVX2__) && defined(__F16C__)
    // W (d,n) @ x (n,) -> xout (d,)
    assert(n % 16 == 0);
    assert(scale == nullptr || block_size[1] % 16 == 0);
    static float one        = 1.0f;
    int dummy_block_size[2] = {d, n};
    if (scale == nullptr) {
        scale      = &one;
        block_size = dummy_block_size;
    }
    int scale_num_cols = (n + block_size[1] - 1) / block_size[1];
    for (int scale_i = 0; scale_i < cdiv(d, block_size[0]); scale_i++) {
        int ii;
#pragma omp parallel for private(ii)
        for (ii = 0; ii < block_size[0]; ii++) {
            int i = scale_i * block_size[0] + ii;
            if (i >= d) {
                continue;
            }
            // Vectorized dot product of w[i][:] and x[:] where w is a packed float8e5m2 array.
            __m256 sumlo = _mm256_setzero_ps();
            __m256 sumhi = _mm256_setzero_ps();
            for (int scale_j = 0; scale_j < cdiv(n, block_size[1]); scale_j++) {
                // Broadcast scale_val to all elements of a vector
                float scale_val  = scale[scale_i * scale_num_cols + scale_j];
                __m256 scale_vec = _mm256_set1_ps(scale_val);
                for (int jj = 0; jj < block_size[1]; jj += 16) {
                    int j = scale_j * block_size[1] + jj;
                    if (j >= n) {
                        break;
                    }

                    // Extract the next set of 16 float8e5m2 weights from `w` and store them
                    // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
                    __m128i wvec = _mm_loadu_si128((__m128i*)&w[i * n + j]);
                    // Take each half of `wvec` which consists of 8 float8e5m2 weights and
                    // pad each 8-bit float8e5m2 value with 8 zeros in the mantissa (least significant bits),
                    // converting to 8 float16 values.
                    __m128i wveclo = _mm_unpacklo_epi8(_mm_setzero_si128(), wvec);
                    __m128i wvechi = _mm_unpackhi_epi8(_mm_setzero_si128(), wvec);
                    // Widen each 8xf16 vector to 8xf32.
                    __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
                    __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);

                    // Scale the weight vectors
                    wveclo_ps = _mm256_mul_ps(wveclo_ps, scale_vec);
                    wvechi_ps = _mm256_mul_ps(wvechi_ps, scale_vec);

                    // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
                    __m256 xveclo = _mm256_loadu_ps(&x[j]);
                    __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
                    // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
                    sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
                    sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
                }
            }
            // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
            __m256 sum8 = _mm256_add_ps(sumlo, sumhi);  // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
            __m128 sum4 = _mm_add_ps(                   // sum4[0:4] = sum8[0:4] + sum8[4:8]
                _mm256_extractf128_ps(sum8, 0), _mm256_extractf128_ps(sum8, 1));
            __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1);  // sum1[0] = dot(sum4, [1,1,1,1])
            xout[i]     = _mm_cvtss_f32(sum1);
        }
    }
#else
    assert(false && "float8e5m2 not supported on this platform");
#endif
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float* o, float* x, int size) {
    float score_max = -FLT_MAX;
    for (int i = 0; i < size; ++i) {
        if (x[i] > score_max) {
            score_max = x[i];
        }
    }
    float score_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        o[i] = expf(x[i] - score_max);
        score_sum += o[i];
    }
    for (int i = 0; i < size; ++i) {
        o[i] /= score_sum;
    }
}

inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
/*
static void moe_gate(
  float* moe_weights,
  float* moegate_bias,
  int* active_experts,
  float* x,
  int n_routed_experts,
  int n_active_routed,
  bool norm_topk_prob,
  float routed_scaling_factor,
  ScoringFunc scoring_func,
  TopKMethod topk_method,
  int n_group,
  int topk_group
) {
  // Set moe_weights[:n_active_routed] to the weights of the top K experts.
  // Set active_experts[:n_active_routed] to the indices of the top K experts.
  if (scoring_func == ScoringFunc::SOFTMAX) {
    softmax(x, x, n_routed_experts);
  } else if (scoring_func == ScoringFunc::SIGMOID) {
    for (int i = 0; i < n_routed_experts; i++) {
      x[i] = sigmoid(x[i]);
    }
  }

  if (moegate_bias) {
    for (int i = 0; i < n_routed_experts; ++i) {
      x[i] += moegate_bias[i];
    }
  }

  // top k
  float wsum = 0.0f;
  if (topk_method == TopKMethod::GREEDY) {
    assert(n_routed_experts <= 256);
    std::array<uint8_t, 32> mask{};
    for (int k = 0; k < n_active_routed; ++k) {
      int best = -1;
      for (int j = 0; j < n_routed_experts; ++j) {
        int mask_i = j / 8;
        int mask_r = j % 8;
        if ((mask[mask_i] & (1ull << mask_r)) == 0 && (best == -1 || x[j] > x[best])) {
          best = j;
        }
      }

      active_experts[k] = best;
      wsum += x[active_experts[k]];
      int best_mask_i = best / 8;
      int best_mask_r = best % 8;
      mask[best_mask_i] |= 1ull << best_mask_r;
    }
  } else if (topk_method == TopKMethod::GROUP_LIMITED_GREEDY) {
    int group_size = n_routed_experts / n_group;

    // First pass: select topk_group within each group
    std::array<uint8_t, 32> mask{};

    for (int g = 0; g < n_group; g++) {
      // Select topk_group items from this group
      for (int k = 0; k < topk_group; k++) {
        int best = -1;
        for (int j = g*group_size; j < (g+1)*group_size; j++) {
          int mask_i = j / 8;
          int mask_r = j % 8;
          if ((mask[mask_i] & (1u << mask_r)) == 0 && x[j] > x[best]) {
            best = j;
          }
        }
        int best_mask_i = best / 8;
        int best_mask_r = best % 8;
        mask[best_mask_i] |= 1u << best_mask_r;
      }
    }
    // Flip mask so that now we only look at the topk_group items in each group
    for (int i = 0; i < 32; i++) {
      mask[i] = ~mask[i];
    }

    // Second pass: select top n_active_routed overall
    for (int k = 0; k < n_active_routed; ++k) {
      int best = -1;
      for (int j = 0; j < n_routed_experts; ++j) {
        int mask_i = j / 8;
        int mask_r = j % 8;
        if ((mask[mask_i] & (1ull << mask_r)) == 0 && (best == -1 || x[j] > x[best])) {
          best = j;
        }
      }

      active_experts[k] = best;
      wsum += x[active_experts[k]];
      int best_mask_i = best / 8;
      int best_mask_r = best % 8;
      mask[best_mask_i] |= 1ull << best_mask_r;
    }
  } else if (topk_method == TopKMethod::NOAUX_TC) {
    assert(false && "TODO: implement noaux_tc");
  }

  if (!norm_topk_prob) {
    wsum = 1.0;
  }
  for (int k = 0; k < n_active_routed; ++k) {
    moe_weights[k] = x[active_experts[k]] / wsum * routed_scaling_factor;
  }
}*/
float rmsnorm(float* o, float* x, float* weight, int size, float eps, bool ln) {
    // calculate mean
    float mean = 0.0f;

    if (ln) {
        for (int j = 0; j < size; j++) {
            mean += x[j];
        }
        mean /= size;
    }

    // calculate sum of squared deltas
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += (x[j] - mean) * (x[j] - mean);
    }

    float var = ss / size;

    // normalize and scale
    float scale = 1.0f / sqrtf(var + eps);
    for (int j = 0; j < size; j++) {
        o[j] = (x[j] - mean) * scale * weight[j];
    }
    return scale;
}
float rmsnorm(float* o, float* x, float* weight, int size, float eps) {
    float rms = 0.0f;
    for (int i = 0; i < size; ++i) {
        rms += x[i] * x[i];
    }
    rms         = sqrtf(rms / size + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < size; ++i) {
        o[i] = x[i] * scale * weight[i];
    }
    return scale;
}

[[maybe_unused]] static void layernorm(float* o, float* x, float* weight, float* bias, int size, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += x[i];
    }
    mean /= size;
    float var = 0.0f;
    for (int i = 0; i < size; ++i) {
        var += (x[i] - mean) * (x[i] - mean);
    }
    var /= size;
    float scale = 1.0f / sqrtf(var + eps);
    if (bias) {
        for (int i = 0; i < size; ++i) {
            o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
        }
    } else {
        for (int i = 0; i < size; ++i) {
            o[i] = (x[i] - mean) * scale * weight[i];
        }
    }
}

void rope(float* buf, float* vec, int d, int head_dim, int pos, float theta) {
    // For some reason, DeepSeek-V2 was trained using rope output
    // layout transposed compared to the input. This means we need a buffer
    // to hold intermediate results.
    assert(d % 2 == 0);
    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        float v0           = vec[i];
        float v1           = vec[i + 1];
        buf[i / 2]         = v0 * fcr - v1 * fci;
        buf[i / 2 + d / 2] = v0 * fci + v1 * fcr;
    }
    for (int i = 0; i < d; i++) {
        vec[i] = buf[i];
    }
}

static void rope_v3(float* vec, int d, int head_dim, int pos, float theta) {
    int rotary_dim = head_dim;

    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        float v0   = vec[i];
        float v1   = vec[i + 1];
        vec[i]     = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        float v0   = vec[i];
        float v1   = vec[i + 1];
        vec[i]     = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void rope(float* buf, __gcc_fp16* vec, int d, int head_dim, int pos, float theta) {
    // For some reason, DeepSeek-V2 was trained using rope output
    // layout transposed compared to the input. This means we need a buffer
    // to hold intermediate results.
    assert(d % 2 == 0);
    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        float v0           = half_to_float(vec[i]);
        float v1           = half_to_float(vec[i + 1]);
        buf[i / 2]         = v0 * fcr - v1 * fci;
        buf[i / 2 + d / 2] = v0 * fci + v1 * fcr;
    }
    for (int i = 0; i < d; i++) {
        vec[i] = float_to_half(buf[i]);
    }
}

static void rope_v3(__gcc_fp16* vec, int d, int head_dim, int pos, float theta) {
    int rotary_dim = head_dim;

    for (int i = 0; i < d; i += 2) {
        int j_head = i % head_dim;
        float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
        float val  = pos * freq;
        float fcr  = cosf(val);
        float fci  = sinf(val);

        float v0   = half_to_float(vec[i]);
        float v1   = half_to_float(vec[i + 1]);
        vec[i]     = float_to_half(v0 * fcr - v1 * fci);
        vec[i + 1] = float_to_half(v0 * fci + v1 * fcr);
    }
}

// Compute next value in a sequence for a single causal self-attention head.
void attn(float* xout,     // (n_kv_heads * v_head_dim,) - output vector
          float* atth,     // (kv_len,) - scratch space to hold attention scores of the sequence
          float* qh,       // (head_dim,) - query vector for this head
          __gcc_fp16* kh,  // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
          __gcc_fp16* vh,  // (kv_len, n_kv_heads, v_head_dim) - buffer containing value vectors of the sequence for all KV heads
          int head_dim,    // size of the "key-space"
          int v_head_dim,  // size of the "value-space"
          int n_kv_heads,  // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
          int kv_len       // number of tokens of the sequence we will attend over
) {
    int k_stride = n_kv_heads * head_dim;  // stride per token in this k head
    // calculate attention scores as dot products of q and k
    for (int t = 0; t < kv_len; ++t) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += qh[i] * half_to_float(kh[t * k_stride + i]);
        }
        score /= sqrtf(head_dim);
        atth[t] = score;
    }

    // softmax the scores to get attention weights over [0..kv_len)
    softmax(atth, atth, kv_len);

    int v_stride = n_kv_heads * v_head_dim;  // stride per token in this v head
    // mix values with attention weights
    for (int i = 0; i < v_head_dim; ++i) {
        float vi = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            vi += atth[t] * half_to_float(vh[t * v_stride + i]);
        }
        xout[i] = vi;
    }
}

void mha_cpu(float* xout,     // (n_heads, head_dim)
             float* att,      // (n_heads, max_seq_len)
             __gcc_fp16* kb,  // (max_seq_len, n_kv_heads, head_dim)
             __gcc_fp16* vb,  // (max_seq_len, n_kv_heads, head_dim)
             float* q,        // (n_heads, head_dim)
             int head_dim, int v_head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads) {
    // Multihead attention. Iterate over all heads.
    int q_per_kv_head = n_heads / n_kv_heads;  // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        int k_head_offset = (h / q_per_kv_head) * head_dim;
        int v_head_offset = (h / q_per_kv_head) * v_head_dim;
        __gcc_fp16* kh    = kb + k_head_offset;
        __gcc_fp16* vh    = vb + v_head_offset;
        attn(xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, kh, vh, head_dim, v_head_dim, n_kv_heads, kv_len);
    }
}

void matmul_unscaled(float* xout, float* x, float* w, int n, int d) { S_matmul(xout, x, w, n, d, nullptr, nullptr); }
void matmul_unscaled(float* xout, float* x, __gcc_fp16* w, int n, int d) { S_matmul(xout, x, w, n, d, nullptr, nullptr); }
void matmul_unscaled(float* xout, float* x, f8e5* w, int n, int d) { S_matmul(xout, x, w, n, d, nullptr, nullptr); }
