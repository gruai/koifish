/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief GRUS SPARSE TEMPLATE	- Random Generator
 *  \author Yingshi Chen
 */

#include "GST_rander.hpp"
using namespace Grusoft;

extern "C" uint64_t xoroshiro_next(void);

uint64_t GRander::RandRersResrResdra() {  // Combined period = 2^116.23
	int alg = 2;
	switch (alg) {
	case 0:
		break;	//return pcg32_random_r(&rng_neil);		//32-bit unsigned int   -  period:      2^64
	case 1:
		return 0;	//xoroshiro_next();
	default:
		xx = rotl(xx, 8) - rotl(xx, 29);                 //RERS,   period = 4758085248529 (prime)
		yy = rotl(yy, 21) - yy;  yy = rotl(yy, 20);      //RESR,   period = 3841428396121 (prime)
		zz = rotl(zz, 42) - zz;  zz = zz + rotl(zz, 14); //RESDRA, period = 5345004409 (prime)
		return xx ^ yy ^ zz;
	}
	return 0;
}

/*
Deprecated

static unsigned int random_u32(uint64_t *state){
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0,1)
float random_f32(uint64_t *state)   { 
    return (random_u32(state) >> 8) / 16777216.0f;
}
struct random_uniform_distribution {
    std::mt19937 gen;
    std::uniform_real_distribution<float> rd;
};

struct random_normal_distribution * init_random_normal_distribution(    int seed, float mean, float std, float min, float max) {
    struct random_normal_distribution * rnd = (struct random_normal_distribution *) malloc(sizeof(struct random_normal_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
    return rnd;
}

struct random_uniform_distribution * init_random_uniform_distribution(int seed, float min, float max) {
    struct random_uniform_distribution * rnd = (struct random_uniform_distribution *) malloc(sizeof(struct random_uniform_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::uniform_real_distribution<float>{min, max};
    return rnd;
}

void free_random_normal_distribution (struct random_normal_distribution  * rnd) {
    free(rnd);
}

void free_random_uniform_distribution(struct random_uniform_distribution * rnd) {
    free(rnd);
}

float frand() {
    return (float)rand()/((float)(RAND_MAX) + 1.0f);
}

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}	


*/

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