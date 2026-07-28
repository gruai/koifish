/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Random Generator
 *  \author Yingshi Chen
 */

#include "GST_rander.hpp"

extern "C" uint64_t xoroshiro_next(void);

uint64_t GRander::RandRersResrResdra() {  // Combined period = 2^116.23
    int alg = 2;
    switch (alg) {
        case 0:
            break;  // return pcg32_random_r(&rng_neil);		//32-bit unsigned int   -  period:      2^64
        case 1:
            return 0;  // xoroshiro_next();
        default:
            xx = rotl(xx, 8) - rotl(xx, 29);  // RERS,   period = 4758085248529 (prime)
            yy = rotl(yy, 21) - yy;
            yy = rotl(yy, 20);  // RESR,   period = 3841428396121 (prime)
            zz = rotl(zz, 42) - zz;
            zz = zz + rotl(zz, 14);  // RESDRA, period = 5345004409 (prime)
            return xx ^ yy ^ zz;
    }
    return 0;
}

/*

*/
float GRanderTorch::NextFloat_01(int flag) {
    float rf = -1.0f;
    if (isCPU) {
        rf = randfloat32(&mt_cpu);
        // double v = randfloat64(&mt_cpu);
    } else {
        auto out = philox_generate(seed64, offset);

        offset += 4;                                         // PyTorch increments offset by 4 per rand()
        uint32_t mantissa = out[0] >> 9;                     // PyTorch float conversion: top 23 bits → [0,1)
        rf                = mantissa * (1.0f / 8388608.0f);  // 2^-23
    }
    assert(rf >= 0.0f && rf < 1.0f);
    return rf;
}

void GRander::Embedding_nn(int nVocab, int nEmbed, std::vector<float>& embeds, int flag) {
    float bound = 1.0f / std::sqrt((float)nEmbed);
    for (int i = 0; i < nVocab; ++i) {
        for (int j = 0; j < nEmbed; ++j) {
            float u = NextFloat_01(flag);         // uniform [0,1)
            float w = (u * 2.0f - 1.0f) * bound;  // uniform [-bound, bound]
            embeds.push_back(w);
        }
    }
}

    /*
    DIST_RangeN::DIST_RangeN(int seed, double a0, double a1) :
        GRander(seed), rMin(a0), rMax(a1)  {
        std::normal_distribution<> d1((rMax+rMin)/2,(rMax-rMin)/6);
        d=d1;
    }

    double DIST_RangeN::gen(){
        double a;
        do{
            a = d(g);
        } while (a<rMin || a>rMax);
        return (a);
    }*/