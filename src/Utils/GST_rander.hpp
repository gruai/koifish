/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief GRUS SPARSE TEMPLATE	- Lite Random Generator(no need std::mt19937)
 *  \author Yingshi Chen
 */

#pragma once
#include <string>
#include <random>
#include <stdint.h>
#include <vector>
#include <set>
#include <assert.h>
#include <stdint.h>
//#include "pcg_oneil/pcg_basic.h"

#define rotl(r,n) (((r)<<(n)) | ((r)>>((8*sizeof(r))-(n))))

/*
	http://www.drdobbs.com/tools/fast-high-quality-parallel-random-number/229625477?pgno=2
*/
namespace Grusoft{
	class GRander {
		unsigned int x = 123456789;		//
		uint64_t xx, yy, zz;
		uint64_t RandRersResrResdra();

		inline int RandInt16()		{
			x = RandRersResrResdra();
			return static_cast<int>((x >> 16) & 0x7FFF);
		}

		inline float NextFloat()	{
			return static_cast<float>(RandInt16()) / (32768.0f);
		}

	protected:
		//pcg32_random_t rng_neil;
		std::random_device device;
		uint32_t seed;

	public:
		GRander() {
			// std::random_device rd;
			// auto genrator = std::mt19937(rd());
			// std::uniform_int_distribution<int> distribution(0, x);
			// x = distribution(genrator);

			//pcg32_srandom_r(&rng_neil, 42u, 54u);
		}
		GRander(uint32_t seed_) { Init(seed_);	 }

		virtual void Init(uint32_t seed_) {
			seed = seed_;	 

			unsigned n;
			xx = 914489ULL;
			yy = 8675416ULL;
			zz = 439754684ULL;
			for (n = ((seed >> 22) & 0x3ff) + 20; n>0; n--) { xx = rotl(xx, 8) - rotl(xx, 29); }
			for (n = ((seed >> 11) & 0x7ff) + 20; n>0; n--) { yy = rotl(yy, 21) - yy;  yy = rotl(yy, 20); }
			for (n = ((seed) & 0x7ff) + 20; n>0; n--) { zz = rotl(zz, 42) - zz;  zz = rotl(zz, 14) + zz; }
			
			x = seed;
		}
		inline int RandInt32() {
			x = RandRersResrResdra();
			return static_cast<int>(x & 0x7FFFFFFF);
		}
		inline  int RandU32()		{
			x = RandRersResrResdra();
			int i = static_cast<int>(x & 0x7FFFFFFF);
			assert(i>=0);
			return i;
		};
		inline double Uniform_(double a0,double a1) {
			int cur = RandInt32();
			double a = cur*1.0 / 0x7FFFFFFF;
			assert(a>=-1.0 && a <=1.0);
			double b = a0 + (a1 - a0)*a;	// (a + 1) / 2.0;
			return b;
		}
		// random float32 in [0,1)
		inline float NextFloat_01()	{
			int cur = RandU32();
			float a = cur*1.0 / 0x7FFFFFFF;
			assert(a>=0.0 && a <1.0);
			return a;	
		}
		inline bool NextCoin(float thrsh=0.5)	{
			int cur = RandU32();
			float a = cur*1.0 / 0x7FFFFFFF;
			assert(a>=0.0 && a <1.0);
			return a<thrsh;	
		}


		/*virtual size_t operator()(size_t n)	{
			std::uniform_int_distribution<size_t> d(0, n ? n - 1 : 0);
			return d(g);
		}*/

		/*
			K sample in N	(K<=N)
			v0.1	
				3/2/2019
		*/
		inline std::vector<int> kSampleInN(int K,int N, bool isOrder=true,int flag=0x0) {
			std::vector<int> ret;
			ret.reserve(K);
			if (K > N || K <= 0) {
				return ret;
			}	else if (K == N) {
				for (int i = 0; i < N; ++i) {
					ret.push_back(i);
				}
			}
			else if (K > 1 && K > (N / std::log2(K))) {
				for (int i = 0; i < N; ++i) {
					double prob = (K - ret.size()) / static_cast<double>(N - i);
					if (NextFloat() < prob) {
						ret.push_back(i);
					}
				}
			}
			else {
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

		inline int kSampleInN(int *root_set, int K, int N, bool isOrder = true, int flag = 0x0) {
			std::vector<int> sampls = kSampleInN(K, N, isOrder, flag);
			K = sampls.size();
			for (int i = 0; i < K; i++) {
				root_set[i] = sampls[i];
			}
			return K;
		}
	};

	/*
	//standard normal distribution
	class DIST_Normal : public GRander	{
		std::normal_distribution<> d;
	public:
		DIST_Normal(int seed) : GRander(seed) {		}

		template<typename T>
		T gen(){
			double a = d(g);
			return T(a);
		}
	};

	//standard normal distribution in [min-max]
	class DIST_RangeN : public GRander	{
		std::normal_distribution<> d;
		double rMin, rMax;
	public:
		DIST_RangeN(int seed, double a0, double a1);
		double gen();
	};*/
}

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
} mt19937_torch;
typedef mt19937_torch mt19937_state;

inline void manual_seed(mt19937_torch* state, unsigned int seed) {
    state->MATRIX_A[0] = 0x0u;
    state->MATRIX_A[1] = 0x9908b0df;
    state->state_[0] = seed & 0xffffffff;
    for (unsigned int j = 1; j < MERSENNE_STATE_N; j++) {
        state->state_[j] = 1812433253 * (state->state_[j - 1] ^ (state->state_[j - 1] >> 30)) + j;
        state->state_[j] &= 0xffffffff;
    }
    state->left_ = 1;
    state->next_ = 0;
}

inline void next_state(mt19937_torch* state) {
    state->left_ = MERSENNE_STATE_N;
    state->next_ = 0;
    unsigned int y, j;
    for (j = 0; j < MERSENNE_STATE_N - MERSENNE_STATE_M; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    for (; j < MERSENNE_STATE_N - 1; j++) {
        y = (state->state_[j] & UMASK) | (state->state_[j + 1] & LMASK);
        state->state_[j] = state->state_[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
    }
    y = (state->state_[MERSENNE_STATE_N - 1] & UMASK) | (state->state_[0] & LMASK);
    state->state_[MERSENNE_STATE_N - 1] = state->state_[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state->MATRIX_A[y & 0x1];
}

inline unsigned int randint32(mt19937_torch* state) {
    if (!state) return 0;
    if (state->MATRIX_A[0] != 0 || state->MATRIX_A[1] != 0x9908b0df) manual_seed(state, 5489); // auto-initialize
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

inline unsigned long long randint64(mt19937_torch* state) {
    return (((unsigned long long)(randint32(state)) << 32) | randint32(state));
}

inline float randfloat32(mt19937_torch* state) {
    return (randint32(state) & ((1ull << 24) - 1)) * (1.0f / (1ull << 24));
}

inline double randfloat64(mt19937_torch* state) {
    return (randint64(state) & ((1ull << 53) - 1)) * (1.0 / (1ull << 53));
}

inline void uniform_(float* data, unsigned int numel, float from, float to, mt19937_torch* state) {
    for (unsigned int t = 0; t < numel; t++) {
        data[t] = randfloat32(state) * (to - from) + from;
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
inline void normal_fill_16(float* data, float mean, float std) {
    #define EPSILONE 1e-12f
    for (unsigned int t = 0; t < 8; t++) {
        float u1 = 1 - data[t];
        float u2 = data[t + 8];
        float radius = sqrtf(-2 * logf(u1 + EPSILONE));
        float theta = (float) (2.0 * M_PI * u2);
        data[t] = (radius * cosf(theta) * std + mean);
        data[t + 8] = (radius * sinf(theta) * std + mean);
    }
}

inline void normal_fill(float* data, unsigned int numel, float mean, float std, mt19937_torch* state) {
    assert(numel>0);
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

inline void normal_(float* data, unsigned int numel, float mean, float std, mt19937_torch* state) {
    #define EPSILONE 1e-12f
    if (numel >= 16) {
        normal_fill(data, numel, mean, std, state);
    }
    else {
        double next_double_normal_sample = 0.0; // make compiler warning happy, won't be used
        int has_next_double_normal_sample = 0;
        for (unsigned int  t = 0; t < numel; t++) {
            if (has_next_double_normal_sample) {
                data[t] = (float)(next_double_normal_sample * std + mean);
                has_next_double_normal_sample = 0;
                continue;
            }
            // for numel < 16 we draw a double (float64)
            float u1 = (float) randfloat64(state);
            float u2 = (float) randfloat64(state);
            float radius = sqrtf(-2 * logf(1 - u2 + EPSILONE));
            float theta = (float) (2.0 * M_PI * u1);
            next_double_normal_sample = radius * sinf(theta);
            has_next_double_normal_sample = 1;
            data[t] = (radius * cosf(theta) * std + mean);
        }
    }
}

inline void init_identity_permutation(int *data, int numel) {
    for (int i = 0; i < numel; i++) {
        data[i] = i;
    }
}

inline void random_permutation(int* data, int numel, mt19937_torch* state) {
    for (int i = numel - 1; i > 0; i--) {
        // pick an index j in [0, i] with equal probability
        int j = randint32(state) % (i + 1);
        // swap i <-> j
        int tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}


// struct random_normal_distribution {
//     std::mt19937 gen;
//     std::normal_distribution<float> rd;
//     float min;
//     float max;
// };
// struct random_normal_distribution * init_random_normal_distribution(    int seed, float mean, float std, float min, float max);