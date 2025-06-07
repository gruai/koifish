/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 * 
 *  \brief Some trial/testing cuda kernels
 *  \author Yingshi Chen
 */
#include "./kernel/Operator.cuh"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/gLLM.hpp"
#include "../../Manifold/Optimizer.hpp"


extern cudaStream_t main_stream;



/*
template <typename T, typename KVT>
struct CoopArgs {
	CoopArgs(Fish* hFish,int kvlen_,int kvpos_,int pos_,int flag=0) :
		kv_len(kvlen_),kv_pos(kvpos_),pos(pos_)	{
		size_t szMost = hFish->MostMemSize();
		bw = PROF_TOKEN(szMost);
		auto config = hFish->config;
		dim = config.nEmbed();
		n_layers = config.nLayer();
		hidden_dim = config.n_ff();
		n_heads = config.n_head();					n_kv_heads = config.n_head_kv();
		head_dim = config.n_embd_head();
		seq_len = config.model.seq_len; 
		// rope_theta = config.model.rope_theta;		
		rotary_dim = config.model.rotary_dim;		

		n_experts = config.model.n_experts;			
		n_experts_ac = config.model.n_experts_ac;		assert(n_experts_ac==0);
		q_dim = head_dim*n_heads, kv_dim = head_dim*n_kv_heads,att_dim=n_heads*config.model.seq_len*2;
		hb_dim = hidden_dim;	
		assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);
		he_dim = n_experts_ac * hidden_dim;			
		// if (getenv("CUDA_INJECTION64_PATH")) {
		// 	coopperf = (uint64_t*)cuda_devicealloc(sizeof(uint64_t) * 16);
		// 	CUDA_CHECK(cudaMemset(coopperf, 0, sizeof(uint64_t) * 16));
		// }
		assert(tX!=nullptr && tX->nByte()>=(dim+hb_dim+q_dim+att_dim)*sizeof(float));
		x = TO<float>(tX);		//dim
		hb = x+dim;
		q = hb+hb_dim;			//	(float*)cuda_devicealloc(q_dim * sizeof(float));
		att = q+q_dim;		//	(float*)cuda_devicealloc(config->n_heads * config->seq_len * 2 * sizeof(float));

		hKVCache hCache = hFish->GetOptimizer()->hCache;
		key_cache = (KVT*)hCache->Get(KVCache::KV_KEY);	
		val_cache = (KVT*)hCache->Get(KVCache::KV_VAL);	

		norm_eps = config.model.norm_eps;
		theta_log2 = log2(config.model.rope_theta);
		qkv_clip = config.model.clip_qkv;
		assert(n_layers>0 && dim>0 && hidden_dim>0 && head_dim>0 && n_heads>0 && n_kv_heads>0);
		norm_ln = config.model.isNormalBias;
		weight_dbits = BitPE(config.model.tpWeight);
	}
	hGTensor tX = GTensor::scratch;
	uint64_t bw;
	uint64_t* perfstats = nullptr;		//	"CUDA_INJECTION64_PATH"
	float* x = nullptr,* hb = nullptr,* he = nullptr,* q = nullptr,* att = nullptr;
	KVT* key_cache = nullptr;
	KVT* val_cache = nullptr;    
	int n_layers,dim,hidden_dim,head_dim,n_heads,n_kv_heads,weight_dbits;
	int n_experts,n_experts_ac,seq_len,rotary_dim,q_dim=-1,hb_dim=-1,he_dim=-1,kv_dim=-1,att_dim=-1;
	bool norm_ln = false, act_gelu = false;;	
	int kv_len,kv_pos,pos;
	float norm_eps,theta_log2,qkv_clip;
};*/

// static int coopsms;
// static __constant__ CoopLayer<> cooplayers[MAX_LAYERS];



