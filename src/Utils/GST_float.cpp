#include <assert.h>
#include <cfloat>
#include <math.h>
#include "immintrin.h"
#include "f16cintrin.h"
#include "../g_float.hpp"
/*
#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(f16_t x) {
  return _cvtsh_ss(x);
}
inline f16_t float_to_half(float x) {
  return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
  assert(false && "float16 not supported on this platform");
  return 0.0f;
}
inline f16_t float_to_half(float x) {
  assert(false && "float16 not supported on this platform");
  return 0;
}
#endif

inline float float8e5m2_to_float(f8e5m2_t x) {
  f16_t val = 0;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&val, &x, sizeof(f8e5m2_t));
#else
  memcpy((char*)&val + sizeof(f8e5m2_t), &x, sizeof(f8e5m2_t));
#endif
  return half_to_float(val);
}
[[maybe_unused]] inline f8e5m2_t float_to_float8e5m2(float x) {
  f16_t val = float_to_half(x);
  f8e5m2_t out;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  memcpy(&out, (char*)&val, sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#else
  memcpy(&out, (char*)&val + sizeof(f8e5m2_t), sizeof(f8e5m2_t)); // TODO: round instead of truncate?
#endif
  return out;
}*/

int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

void matmul(float* xout, float* x, float* w, int n, int d, const int* block_size, float* scale) {
  // W (d,n) @ x (n,) -> xout (d,)
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
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

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
void matmul(float* xout, float* x, f16_t* w, int n, int d, const int* block_size, float* scale) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  assert(scale == nullptr || block_size[1] % 16 == 0);
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
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
        float scale_val = scale[scale_i * scale_num_cols + scale_j];
        __m256 scale_vec = _mm256_set1_ps(scale_val);
        for (int jj = 0; jj < block_size[1]; jj+=16) {
          int j = scale_j * block_size[1] + jj;
          if (j >= n) {
            break;
          }
          
          // Extract the next set of 16 float16 weights from `w` and store them
          // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
          __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
          __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
          __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
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
      __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
      __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
        _mm256_extractf128_ps(sum8, 0), 
        _mm256_extractf128_ps(sum8, 1)
      );
      __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
      xout[i] = _mm_cvtss_f32(sum1);
    }
  }
#else
  assert(false && "float16 not supported on this platform");
#endif
}

// matmul supporting float8e5m2 weights via AVX2 and F16C extensions, which (1) 
// allows vectorized conversion from f8e5m2 to float16 and (2) conversion from 
// float16 to float32 values before calculations.
void matmul(float* xout, float* x, f8e5m2_t* w, int n, int d, const int* block_size, float* scale) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  assert(scale == nullptr || block_size[1] % 16 == 0);
  static float one = 1.0f;
  int dummy_block_size[2] = {d, n};
  if (scale == nullptr) {
    scale = &one;
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
        float scale_val = scale[scale_i * scale_num_cols + scale_j];
        __m256 scale_vec = _mm256_set1_ps(scale_val);
        for (int jj = 0; jj < block_size[1]; jj+=16) {
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
      __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
      __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
        _mm256_extractf128_ps(sum8, 0), 
        _mm256_extractf128_ps(sum8, 1)
      );
      __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
      xout[i] = _mm_cvtss_f32(sum1);
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

inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}
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

static void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
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

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
  return x < -v ? -v : (x > v ? v : x);
}

static void rope(float* buf, float* vec, int d, int head_dim, int pos, float theta) {
  // For some reason, DeepSeek-V2 was trained using rope output 
  // layout transposed compared to the input. This means we need a buffer
  // to hold intermediate results.
  assert(d % 2 == 0);
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    buf[i/2] = v0 * fcr - v1 * fci;
    buf[i/2 + d/2] = v0 * fci + v1 * fcr;
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
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

static void rope(float* buf, f16_t* vec, int d, int head_dim, int pos, float theta) {
  // For some reason, DeepSeek-V2 was trained using rope output
  // layout transposed compared to the input. This means we need a buffer
  // to hold intermediate results.
  assert(d % 2 == 0);
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = 1.0f / powf(theta, (float)j_head / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = half_to_float(vec[i]);
    float v1 = half_to_float(vec[i + 1]);
    buf[i/2] = v0 * fcr - v1 * fci;
    buf[i/2 + d/2] = v0 * fci + v1 * fcr;
  }
  for (int i = 0; i < d; i++) {
    vec[i] = float_to_half(buf[i]);
  }
}

static void rope_v3(f16_t* vec, int d, int head_dim, int pos, float theta) {
  int rotary_dim = head_dim;

  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = half_to_float(vec[i]);
    float v1 = half_to_float(vec[i + 1]);
    vec[i] = float_to_half(v0 * fcr - v1 * fci);
    vec[i + 1] = float_to_half(v0 * fci + v1 * fcr);
  }
}


// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (n_kv_heads * v_head_dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, v_head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int v_head_dim, // size of the "value-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int k_stride = n_kv_heads * head_dim; // stride per token in this k head
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

  int v_stride = n_kv_heads * v_head_dim; // stride per token in this v head
  // mix values with attention weights
  for (int i = 0; i < v_head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(vh[t * v_stride + i]);
    }
    xout[i] = vi;
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int v_head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int k_head_offset = (h / q_per_kv_head) * head_dim;
    int v_head_offset = (h / q_per_kv_head) * v_head_dim;
    f16_t* kh = kb + k_head_offset;
    f16_t* vh = vb + v_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, v_head_dim, n_kv_heads, kv_len
    );
  }
}

void matmul_unscaled(float* xout, float* x, float* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr);
}
void matmul_unscaled(float* xout, float* x, f16_t* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr);
}
void matmul_unscaled(float* xout, float* x, f8e5m2_t* w, int n, int d) {
  matmul(xout, x, w, n, d, nullptr, nullptr);
}

