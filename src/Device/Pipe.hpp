/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  PIPE:   transfer data between device & host
 *  For example:    In CUDA, there is a clear distinction between host and device code, and you cannot directly call host functions from device code.
        calling a __host__ function from a __global__ function  is not allowed  !
 * 
 *  \brief PIPE - transfer data between device & host
 *  \author Yingshi Chen
 */
#include "../Manifold/Fish.hpp"
#include "../Manifold/Optimizer.hpp"

#define PROF_TOKEN(bytes) ((0xCDAFull << 48) | (bytes))

//  For cudaLaunchCooperativeKernel
template <typename T=void>
struct CoopLayer {
    float* rms_att_weight=nullptr;
    T* wq=nullptr, *wk=nullptr, *wv=nullptr, *wo=nullptr;
    float* bqkv=nullptr,*rms_ffn_weight=nullptr;
    T* moegate=nullptr, *w1=nullptr, *w2=nullptr, *w3=nullptr;
};
#define CoopLayer_MAX 128
#define KV_SINKS 2          // attention sinks for rolling buffer
template <typename T, typename KVT>
struct TRANSFORMER_PIPE : public MODEL_CARD {
    CoopLayer<void> *cLayers=nullptr;
    hFISH hFish = nullptr;
    hGensor out_weight = nullptr;
	float *x=nullptr,*xb=nullptr,*xb2=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*att=nullptr,*exp=nullptr;
	float *hb=nullptr,*hb2=nullptr,*he=nullptr;

	TRANSFORMER_PIPE(hFISH hFish_,int pos_,int flag=0) :
        hFish(hFish_)	{
		size_t szMost = hFish->MostMemSize();
		bw = PROF_TOKEN(szMost);
		auto config = hFish->config;
		vocab_size = config.model.vocab_size;		assert(vocab_size>0);
		dim = config.nEmbed();
		n_layers = config.nLayer();
		hidden_dim = config.n_ff();
		n_heads = config.n_head();					n_kv_heads = config.n_head_kv();
		head_dim = config.n_embd_head();
		seq_len = config.model.seq_len; 
		rope_theta = config.model.rope_theta;		
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
		size_t nzTmp = 0;
		x = TO<float>(tX);
		if(DEBUG.T_cpu){
			xb = x+dim;		xb2 = xb+dim;		hb = xb2+dim;		hb2 = hb+hidden_dim;
			q = hb2+hidden_dim;		k = q+q_dim;	v = k+kv_dim;
			att = v+kv_dim;		exp = att+n_heads *seq_len;
			nzTmp = exp-x + n_experts + (n_experts_ac ? n_experts_ac : 1) * 2;
		}else{			
			hb = x+dim;		q = hb+hb_dim;		att = q+q_dim;			
			nzTmp = att+att_dim-x;		
		}
		assert(tX->nByte()>=nzTmp*sizeof(float));		tX->Zero();	

		hKVCache hCache = hFish->GetOptimizer()->hCache;
		key_cache = (KVT*)hCache->Get(KVCache::KV_KEY);	
		val_cache = (KVT*)hCache->Get(KVCache::KV_VAL);	

		norm_eps = config.model.norm_eps;
		theta_log2 = log2(config.model.rope_theta);
		qkv_clip = config.model.clip_qkv;
		assert(n_layers>0 && dim>0 && hidden_dim>0 && head_dim>0 && n_heads>0 && n_kv_heads>0);
		norm_ln = config.model.isNormalBias;
		weight_dbits = BitPE(config.model.tpWeight);

        InitCLayer(flag);        
		UpdatePos(pos_);
	}
	virtual void UpdatePos(int pos_)	{
		// following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
		pos = pos_;		assert(pos>=0);
		int kv_sink = pos >= seq_len ? KV_SINKS : 0;
        kv_pos = kv_sink + (pos - kv_sink) % (seq_len - kv_sink);
        kv_len = pos >= seq_len ? seq_len : pos + 1;
	}

    virtual ~TRANSFORMER_PIPE() {
        //	FREE_a(cLayers);
		delete[ ] cLayers;
    }

	hGTensor tX = GTensor::scratch;
	uint64_t bw;
	uint64_t* perfstats = nullptr;		//	"CUDA_INJECTION64_PATH"
	
	KVT* key_cache = nullptr;
	KVT* val_cache = nullptr;    
	int n_layers,dim,hidden_dim,head_dim,n_heads,n_kv_heads,weight_dbits;
	int n_experts,n_experts_ac,seq_len,rotary_dim,q_dim=-1,hb_dim=-1,he_dim=-1,kv_dim=-1,att_dim=-1;
	bool norm_ln = false, act_gelu = false;	
	bool norm_par = false;     // use parallel MLP/attention by omitting intermediate normalization
	int kv_len,kv_pos,pos;
	float norm_eps,theta_log2,qkv_clip;

    virtual void InitCLayer(int flag=0x0){    
        out_weight = hFish->GetGensor("model.output.weight", 0);
				
        /*TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed",0); 
		if ( out_weight == nullptr) {
			out_weight = embed->w;		// tied weights
			
		}	*/
        
        cLayers = new CoopLayer<void>[CoopLayer_MAX];	
        for (int l = 0; l < n_layers; ++l) {			
			SelfAttention *QKV = hFish->GetNeuron<SelfAttention>("SelfAttention",l); 
			cLayers[l].rms_att_weight = TO<float>(QKV->norm.w);	//weights->rms_att_weight[l];
			cLayers[l].wq = ToX(QKV->Q.w);
			cLayers[l].wk = ToX(QKV->K.w);
			cLayers[l].wv = ToX(QKV->V.w);
			cLayers[l].wo = ToX(QKV->proj_cat.w);		//weights->wo[l];
			cLayers[l].bqkv = TO<float>(QKV->bqkv);	//weights->bqkv[l];
			FFN *ffn=hFish->GetNeuron<FFN>("FFN",l); 
			cLayers[l].rms_ffn_weight = TO<float>(ffn->norm.w);	//weights->rms_ffn_weight[l];
			cLayers[l].moegate = nullptr;	//weights->moegate[l];
			cLayers[l].w1 = TO(ffn->PGensors(),"w1");		//weights->w1[l];
			cLayers[l].w2 = TO(ffn->PGensors(),"w2");		//weights->w2[l];
			cLayers[l].w3 = TO(ffn->PGensors(),"w3");		//weights->w3[l];
		}
    }
};
