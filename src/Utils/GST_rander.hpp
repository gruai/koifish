/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Lite Random Generator(no need std::mt19937)
 *  \author Yingshi Chen
 */

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../g_float.hpp"
// #include "pcg_oneil/pcg_basic.h"

#define rotl(r, n) (((r) << (n)) | ((r) >> ((8 * sizeof(r)) - (n))))

/*
    http://www.drdobbs.com/tools/fast-high-quality-parallel-random-number/229625477?pgno=2
*/

class GRander {
    unsigned int x = 123456789;  //
    uint64_t xx, yy, zz;
    uint64_t RandRersResrResdra();

    inline int RandInt16() {
        x = RandRersResrResdra();
        return static_cast<int>((x >> 16) & 0x7FFF);
    }

    inline float NextFloat() { return static_cast<float>(RandInt16()) / (32768.0f); }

   protected:
    // pcg32_random_t rng_neil;
    std::random_device device;
    uint32_t seed;

   public:
    GRander() { Init(20070514); }
    GRander(uint32_t seed_) { Init(seed_); }

    virtual void Init(uint32_t seed_) {
        seed = seed_;

        unsigned n;
        xx = 914489ULL;
        yy = 8675416ULL;
        zz = 439754684ULL;
        for (n = ((seed >> 22) & 0x3ff) + 20; n > 0; n--) {
            xx = rotl(xx, 8) - rotl(xx, 29);
        }
        for (n = ((seed >> 11) & 0x7ff) + 20; n > 0; n--) {
            yy = rotl(yy, 21) - yy;
            yy = rotl(yy, 20);
        }
        for (n = ((seed) & 0x7ff) + 20; n > 0; n--) {
            zz = rotl(zz, 42) - zz;
            zz = rotl(zz, 14) + zz;
        }

        x = seed;
    }
    inline int RandInt32() {
        x = RandRersResrResdra();
        return static_cast<int>(x & 0x7FFFFFFF);
    }
    inline uint32_t RandU32() {
        x          = RandRersResrResdra();
        uint32_t i = static_cast<uint32_t>(x & 0x7FFFFFFF);
        return i;
    };
    inline double Uniform_(double a0, double a1) {
        int cur  = RandInt32();
        double a = cur * 1.0 / 0x7FFFFFFF;
        assert(a >= -1.0 && a <= 1.0);
        double b = a0 + (a1 - a0) * a;  // (a + 1) / 2.0;
        return b;
    }
    // random float32 in [0,1)
    inline virtual float NextFloat_01(int flag = 0x0) {
        int cur = RandU32();
        float a = cur * 1.0 / 0x7FFFFFFF;
        assert(a >= 0.0 && a < 1.0);
        return a;
    }
    inline bool NextCoin(float thrsh = 0.5) {
        int cur = RandU32();
        float a = cur * 1.0 / 0x7FFFFFFF;
        assert(a >= 0.0 && a < 1.0);
        return a < thrsh;
    }

    inline void RandFloat(int N, std::vector<float>& arr, int flag = 0x0) {
        arr.clear();
        for (int i = 0; i < N; ++i) {
            arr.push_back(NextFloat_01(flag));
        }
    }

    inline void RandMask_MN(int M, int N, std::vector<float>& T_m, int* mask32, int flag = 0x0) {
        assert(T_m.size() == M);
        std::vector<float> probs;
        int* mask = mask32;
        for (int i = 0; i < M; i++) {
            float thrsh = T_m[i];
            RandFloat(N, probs);
            for (int i = 0; i < N; i++, mask++) {
                *mask = probs[i] < thrsh;
            }
        }
    }

    //  nn.Embedding(vocab_size, n_embd)
    virtual void Embedding_nn(int nVocab, int nEmbed, std::vector<float>& embeds, int flag = 0x0);

    /* Another kSampleInN
        vector<int> IDs( nVocab );
        std::iota( IDs.begin(),IDs.end(), 0 );      //  Fills the range [first, last) with ++value.
        std::mt19937 g(seed);       //std::random_device rd;  g(rd());
        std::shuffle(IDs.begin(), IDs.end(),g);    //std::random_shuffle(IDs.begin(), IDs.end());
        for(int i=0;i<nSample;i++){
            samps[i] = IDs[i];
            assert(samps[i]>=0 && samps[i]<nVocab);
        }    */

    /*
        K sample in N	(K<=N)
        v0.1
            3/2/2019
    */
    inline std::vector<int> kSampleInN(int K, int N, bool isOrder = true, int flag = 0x0) {
        std::vector<int> ret;
        ret.reserve(K);
        if (K > N || K <= 0) {
            return ret;
        } else if (K == N) {
            for (int i = 0; i < N; ++i) {
                ret.push_back(i);
            }
        } else if (K > 1 && K > (N / std::log2(K))) {
            for (int i = 0; i < N; ++i) {
                double prob = (K - ret.size()) / static_cast<double>(N - i);
                if (NextFloat() < prob) {
                    ret.push_back(i);
                }
            }
        } else {
            std::set<int> sample_set;
            while (static_cast<int>(sample_set.size()) < K) {
                int next = RandInt32() % N;
                if (sample_set.count(next) == 0) {
                    sample_set.insert(next);
                }
            }
            for (auto iter = sample_set.begin(); iter != sample_set.end(); ++iter) {
                ret.push_back(*iter);
            }
        }
        return ret;
    }

    inline int kSampleInN(int* root_set, int K, int N, bool isOrder = true, int flag = 0x0) {
        std::vector<int> sampls = kSampleInN(K, N, isOrder, flag);
        K                       = sampls.size();
        for (int i = 0; i < K; i++) {
            root_set[i] = sampls[i];
        }
        return K;
    }
};
typedef std::shared_ptr<GRander> hRANDER;

#include <math.h>

#define MERSENNE_STATE_M 397u
#define MERSENNE_STATE_N 624u

#define LMASK 0x7ffffffful
#define UMASK 0x80000000ul

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

typedef struct {
    unsigned long long seed_;
    int left_;
    unsigned int next_;
    unsigned int state_[MERSENNE_STATE_N];
    unsigned int MATRIX_A[2];
} MT19937_torch;

inline void manual_seed(MT19937_torch* state, unsigned int seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0]   = seed & 0xffffffff;
    for (unsigned int j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->left_ = 1;
    state->next_ = 0;
}

inline void next_state(MT19937_torch* state) {
    state->left_ = MERSENNE_STATE_N;
    state->next_ = 0;
    unsigned int y, j;
    for (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y                = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    for (; j < MERSENNE_STATE_N - 1; j++) {
        y                = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    y                                   = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

inline unsigned int randint32(MT19937_torch* state) {
    if (!state)
        return 0;
    if (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df)
        manual_seed(state, 5489);  // auto-initialize
    if (--state->left_ <= 0) {
        next_state(state);
    }
    unsigned int y = state->state_[state->next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >> 18;
    return y;
}

inline unsigned long long randint64(MT19937_torch* state) { return (((unsigned long long)(randint32(state)) << 32) | randint32(state)); }

inline float randfloat32(MT19937_torch* state) { return (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24)); }

inline double randfloat64(MT19937_torch* state) { return (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53)); }

inline void uniform_(float* data, unsigned int numel, float from, float to, MT19937_torch* state) {
    for (unsigned int t = 0; t < numel; t++) {
        data[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
template <typename T>
inline void normal_fill_16(T* data, float mean, float std) {
#define EPSILONE 1e-12f
    for (unsigned int t = 0; t < 8; t++) {
        float u1     = 1 - T2Float(data + t);
        float u2     = T2Float(data + t + 8);
        float radius = sqrtf(-2 * logf(u1 + EPSILONE));
        float theta  = (float)(2.0 * M_PI * u2);
        data[t]      = (radius * cosf(theta) * std + mean);
        data[t + 8]  = (radius * sinf(theta) * std + mean);
    }
}

template <typename T>
inline void normal_fill(T* data, unsigned int numel, float mean, float std, MT19937_torch* state) {
    assert(numel > 0);
    for (unsigned int t = 0; t < numel; t++) {
        data[t] = randfloat32(state);
    }
    for (unsigned int i = 0; i < numel - 15; i += 16) {
        normal_fill_16(data + i, mean, std);
    }
    if (numel % 16 != 0) {
        // recompute the last 16 values
        data = data + numel - 16;
        for (unsigned int i = 0; i < 16; i++) {
            data[i] = randfloat32(state);
        }
        normal_fill_16(data, mean, std);
    }
}

template <typename T>
inline void normal_19937(T* data, unsigned int numel, float mean, float std, MT19937_torch* state) {
#define EPSILONE 1e-12f
    if (numel >= 16) {
        normal_fill(data, numel, mean, std, state);
    } else {
        double next_double_normal_sample  = 0.0;  // make compiler warning happy, won't be used
        int has_next_double_normal_sample = 0;
        for (unsigned int t = 0; t < numel; t++) {
            if (has_next_double_normal_sample) {
                data[t]                       = (float)(next_double_normal_sample * std + mean);
                has_next_double_normal_sample = 0;
                continue;
            }
            // for numel < 16 we draw a double (float64)
            float u1                      = (float)randfloat64(state);
            float u2                      = (float)randfloat64(state);
            float radius                  = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            float theta                   = (float)(2.0 * M_PI * u1);
            next_double_normal_sample     = radius * sinf(theta);
            has_next_double_normal_sample = 1;
            float a                       = (radius * cosf(theta) * std + mean);
            data[t]                       = Float2T<T>(&a);
        }
    }
}

inline void init_identity_permutation(int* data, int numel) {
    for (int i = 0; i < numel; i++) {
        data[i] = i;
    }
}

inline void random_permutation(int* data, int numel, MT19937_torch* state) {
    for (int i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        int j = randint32(state) % (i + 1);
        // swap i <-> j
        int tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

/*
    SCPQ(a Sudden Confession under Persistent Questioning):
        When discussion solutions with AI tools(doubao,copilot), oftern encounters SCPQ: the AI rambels in circles, offering many specious, harf-backed solutions.
        Suddenly(may after several miniutes or even hours). it aknowledgeing its own errors.
    SCPQ is a clear sign that current (transformer based)-AI has no full human intelligence. 
    Know what one does not know(然乎然,不然乎不然) is one of humanity's most vital forms of intelligence.     

    Philox backend is a typical SCPQ codes fromf copilot. Copilot recommend Philox at first, but the generated sequnce is not match Torch. After one hours' work, copilot says "On current PyTorch releases, torch.rand() on CPU does not use
        Philox as its RNG backend. "
*/
struct GRanderTorch : public GRander {
    uint64_t seed64;
    uint64_t offset;
    MT19937_torch mt_cpu;
    bool isCPU = true;

    static constexpr uint32_t PHILOX_M0 = 0xD2511F53;
    static constexpr uint32_t PHILOX_M1 = 0xCD9E8D57;
    static constexpr uint32_t PHILOX_W0 = 0x9E3779B9;
    static constexpr uint32_t PHILOX_W1 = 0xBB67AE85;

    GRanderTorch(uint64_t user_seed, uint64_t offset_ = 0) : offset(offset_) {
        if (isCPU) {
            manual_seed(&mt_cpu, user_seed);
        } else {
            uint64_t z = user_seed + 0x9E3779B97f4A7C15ULL;
            z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z          = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            seed64     = z ^ (z >> 31);
        }
    }

    static inline uint32_t mulhilo(uint32_t a, uint32_t b, uint32_t& hi) {
        uint64_t p = (uint64_t)a * b;
        hi         = p >> 32;
        return (uint32_t)p;
    }

    std::array<uint32_t, 4> philox_round(std::array<uint32_t, 4> ctr, std::array<uint32_t, 2> key) {
        uint32_t hi0, hi1;
        uint32_t lo0 = mulhilo(PHILOX_M0, ctr[0], hi0);
        uint32_t lo1 = mulhilo(PHILOX_M1, ctr[2], hi1);

        return {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
    }

    std::array<uint32_t, 4> philox_generate(uint64_t seed64, uint64_t offset) {
        std::array<uint32_t, 4> ctr = {(uint32_t)offset, (uint32_t)(offset >> 32), 0, 0};  //(uint32_t)seed64, (uint32_t)(seed64 >> 32)
        std::array<uint32_t, 2> key = {(uint32_t)seed64, (uint32_t)(seed64 >> 32)};
        for (int i = 0; i < 10; i++) {
            ctr = philox_round(ctr, key);
            key[0] += PHILOX_W0;
            key[1] += PHILOX_W1;
        }
        return ctr;
    }

    float NextFloat_01(int flag = 0x0) override;
};