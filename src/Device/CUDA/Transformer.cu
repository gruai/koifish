/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *  
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 * 
 *  \brief Transformer in cuda kernel
 *  \author Yingshi Chen
 */
#include "./kernel/Operator.cuh"
// #include "./llm_c/sampler.h"
#include "./kernel/layernorm.cuh"
#include "./kernel/embed.cuh"
#include "./kernel/utils.cuh"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Fish.hpp"
#include "../../Device/Pipe.hpp"

extern int tpFuseCu;
extern cudaStream_t main_stream;

__global__ void __launch_bounds__(1024) test_print_kernel(half *__restrict__ arr){
    // printf("test kernel\n");
    if (((int)blockIdx.x == 0) && ((int)threadIdx.x == 0))    {
        // arr[0] = ((half)(2));
        __syncthreads();
        printf("%f ", __half2float(arr[0]));

    }
}

unsigned long long rng_state=0;
float* accumulated_mean_loss=nullptr;

static int micro_step=0;    //*workload_indices=nullptr;
// static int4 *bucket_info=nullptr;
float RAW_backward(Fish *fish,const int* iX, int grad_accum_steps,bool isOnlyEvaluate,int flag) {    
    assert(!isOnlyEvaluate);
    NVTX_RANGE_FN();
    bool last_step = micro_step == grad_accum_steps - 1;
    auto config = fish->config;
    int B,T,C,L = config.nLayer(),NH = config.n_head();      
    fish->GetBTC(B,T,C);    
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    hGensor cur=nullptr,delta=cls->delta;      
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed",0); 
    if (micro_step == 0) {// on the first micro-step zero the gradients, as we're about to += accumulate into them;     
        cudaCheck(cudaMemsetAsync(TO<float>(cls->out), 0,B*T* sizeof(float), main_stream));
    }
    
    // floatX *scratchX=(floatX *)GTensor::buff;     
    GTensor::delta->Zero();         //cudaCheck(cudaMemset(dresidual, 0, B * T * C * sizeof(floatX)));    //???
    PrintTensor<floatX>("back of P",ToX(GTensor::bt4c),true,B,T,C);
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    FFN *ffn=fish->GetNeuron<FFN>("FFN",L-1);  
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);    
    delta = cls->FUSE_cuda(nullptr,0x0);    //some operation fused in forward pass
    lnf->FUSE_cuda(delta,0x0);  //

    hGensor lastDelta = ffn->out;    
    for (int l = L-1; l >= 0; l--) {
        NvtxRange layer_range("Layer", l);
        SelfAttention *QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);        //preFFN = l==0 ? nullptr : fish->GetNeuron<FFN>("FFN",l-1);         
        // GeNeuron *last = l == 0 ? embed : (GeNeuron *)(fish->GetNeuron<FFN>("FFN",l-1));    //residual = l == 0 ? ToX(embed->out) : ToX(preFFN->out);  
        // ffn->delta = lastDelta;     ffn->tmpDelta = lastDelta;   
        // LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->FUSE_cuda(nullptr, 0x0);   
        QKV->FUSE_cuda(nullptr,0x0);   //QKV->norm.out,last->out
    }
    embed->OnEmbed(nullptr,0x0);    //,random_u32(&rng_state)
    
    // Aggregate all gradients that are not part of the transformer blocks5
    if(tpFuseCu==0 && last_step) {
        // reduce all the losses within the current GPU (across all microsteps)
        global_sum_deterministic(accumulated_mean_loss, TO<float>(cls->out), B*T, main_stream);
        // reduce loss across GPUs to a single, final float across all microsteps and GPUs
        #if MULTI_GPU
        ncclCheck(ncclAllReduce(accumulated_mean_loss, accumulated_mean_loss, sizeof(float), ncclFloat, ncclAvg, multi_gpu_config.nccl_comm, main_stream));
        #endif
        cudaCheck(cudaMemcpyAsync(&cls->mean_loss, accumulated_mean_loss, sizeof(float), cudaMemcpyDeviceToHost, main_stream));
        cls->mean_loss /= B*T*grad_accum_steps;
    }
    cudaCheck(cudaDeviceSynchronize());
    if(last_step) {
        // cls->mean_loss /= B*T*grad_accum_steps;
        micro_step = 0;
    } else {
        cls->mean_loss = -1.f; // no loss available yet
        micro_step++;
    }
    
    return cls->mean_loss;
}   
/*
void RAW_forward(Fish *fish,int flag) {
    NVTX_RANGE_FN();
    auto config = fish->config;
    int B,T,C,tpFuseNormal=config.Fuse_Normal;      
    fish->GetBTC(B,T,C);    
    const size_t L = config.nLayer(),NH = config.n_head();       
    hGensor cur=nullptr,residual=nullptr;
    LayerNormal* lnf = fish->GetNeuron<LayerNormal>("LayerNormal",0);
    TokenEmbed* embed = fish->GetNeuron<TokenEmbed>("TokenEmbed",0);
    cur = embed->Ming(nullptr,fish->Input());     
    residual = cur; //embed->out;
    SelfAttention *QKV0 = fish->GetNeuron<SelfAttention>("SelfAttention",0),*QKV=nullptr;

    if(tpFuseNormal==1){
        cur=QKV0->norm.FUSE_cuda(cur);   
    } 

    FFN *ffn=nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV = fish->GetNeuron<SelfAttention>("SelfAttention",l);
        ffn = fish->GetNeuron<FFN>("FFN",l);            //ffn->out = GTensor::delta;
        LayerNormal *hNorm = l+1 != L ? &(fish->GetNeuron<SelfAttention>("SelfAttention",l+1)->norm) : lnf;
        ffn->fuseNorm = tpFuseNormal==1?hNorm:nullptr;       
        QKV->fuseNorm =  tpFuseNormal==1?&(ffn->norm):nullptr;
        cur = QKV->FUSE_cuda(cur,residual,nullptr,nullptr,flag);        
        cur = ffn->FUSE_cuda(cur,nullptr, 0x0);  
        residual = ffn->out;
    }    
    if(tpFuseNormal==0){
        cur = lnf->FUSE_cuda(ffn->out); 
    }
    OutCLS* cls = fish->GetNeuron<OutCLS>("OutCLS",0);
    if(tpFuseCu==1)
        cls->FUSE_cuda(cur,flag); //embed->w,
    else
        assert(0);//cls->preLogits = lnf->out*embed->w;   
    PrintTensor<floatX>("output",ToX(cls->preLogits),true,B,T,C);
    // ffn->norm.out->PrintX<floatX>("inp1",0,-1); 

}*/

#define MAX_LAYERS 128
#define MAX_EXPERTS 64
__device__ static void syncgrid() {
	volatile unsigned int* barrier = &cooperative_groups::details::get_grid_workspace()->barrier;

	if (threadIdx.x == 0) {
		unsigned int nb = 1;
		if (blockIdx.x == 0) {
			nb = 0x80000000 - (gridDim.x - 1);
		}

		unsigned int old_arrive;
		asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(old_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier), "r"(nb) : "memory");

		unsigned int current_arrive;
		do {
			asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(current_arrive) : _CG_ASM_PTR_CONSTRAINT(barrier) : "memory");
		} while (((old_arrive ^ current_arrive) & 0x80000000) == 0);
	}

	__syncthreads();
}

__device__ static void coopstage(uint64_t* stats, int stage) {
	__shared__ uint64_t lastt;
	if (stats && blockIdx.x == 0 && threadIdx.x == 0) {
		uint64_t t;
		asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
		if (stage >= 0) {
			stats[stage] += t - lastt;
		}
		lastt = t;
	}
}

/*
	T(__nv_fp8_e5m2)	type of weight
	KVT(__half)			type of KV cache
	AT float			type of activation
*/
template <typename T, typename KVT, typename AT>
__global__ __launch_bounds__(1024, 1) static void T_forward_qkv(const __grid_constant__ TRANSFORMER_PIPE<T, KVT> args) {
	extern __shared__ char smem[];
	__shared__ float rmsscale;

	AT* xs = (AT*)smem;
	int dim = args.dim,l=args.layNo;
	int hidden_dim = args.hidden_dim;
	int head_dim = args.head_dim;
	int kv_mul = args.n_heads / args.n_kv_heads;
	int q_dim = args.head_dim * args.n_heads;
	int kv_dim = args.head_dim * args.n_kv_heads;
	const int IK = 4; // K consecutive warps per block, groups of K are interleaved across SMs for better work distribution
	int io = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
	int ib = (gridDim.x * blockDim.x) / warpSize;
	coopstage(args.perfstats, -1); // init timing

	static __device__ int badsoftmax = 0;

	//for (int l = 0; l < args.n_layers; ++l) {
		// const CoopLayer<T>* L = (const CoopLayer<T>*)&cooplayers[l];
		const CoopLayer<T>* L = (const CoopLayer<T>*)(args.cLayers+l);		//	args.cLayers+l
		assert(L!=nullptr);
		if (blockIdx.x == 0 && threadIdx.x < warpSize) {
			badsoftmax = 0;
		}

		// pre-attention CU_rmsnorm (into shared memory)
		rmsscale = CU_rmsnorm(xs, args.x, L->rms_att_weight, dim, args.norm_eps, args.norm_ln);	//57.1846123
		size_t loff = (size_t)l * args.seq_len * kv_dim; // kv cache layer offset for convenience
		KVT* keyb = args.key_cache + loff;
		KVT* valb = args.val_cache + loff;
		
		// qkv matmul + RoPE encoding + update KV cache
		for (int j = io * 2; j < q_dim + kv_dim * 2; j += ib * 2) {
			T* w = j < q_dim ? L->wq : (j < q_dim + kv_dim ? L->wk : L->wv);
			int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);
			if(k==0){
				k = 0;
			}
			float v0 = matmul_warppar(xs, w, k + 0, dim) * rmsscale;
			float v1 = matmul_warppar(xs, w, k + 1, dim) * rmsscale;

			if (L->bqkv) {
				v0 += L->bqkv[j + 0];
				v1 += L->bqkv[j + 1];
			}

			v0 = min(max(v0, -args.qkv_clip), args.qkv_clip);
			v1 = min(max(v1, -args.qkv_clip), args.qkv_clip);

			if (threadIdx.x % warpSize == 0) {
				int j_head = j % head_dim;
				float freq = j_head >= args.rotary_dim ? 0.f : exp2f(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);
				float fcr, fci;
				sincosf(args.pos * freq, &fci, &fcr);

				if (j < q_dim) {
					args.q[k + 0] = v0 * fcr - v1 * fci;
					args.q[k + 1] = v0 * fci + v1 * fcr;
				} else if (j < q_dim + kv_dim) {
					// note: k layout is transposed / tiled to improve attn_score performance
					int off = args.kv_pos * 16 + args.seq_len * (k / 16) * 16 + (k % 16);
					keyb[off + 0] = KVT(v0 * fcr - v1 * fci);
					keyb[off + 1] = KVT(v0 * fci + v1 * fcr);
				} else {
					// note: v layout is transposed (we store all positions for a given head contiguously) to improve attn_mix performance
					valb[args.kv_pos + args.seq_len * (k + 0)] = KVT(v0);
					valb[args.kv_pos + args.seq_len * (k + 1)] = KVT(v1);
				}
			}
		}

		__syncthreads(); 
		syncgrid();
		coopstage(args.perfstats, 0);		

		// attention score
		int kv_lent = (args.kv_len + 7) / 8;

		for (int j = io; j < kv_lent * args.n_heads; j += ib) {
			int h = j % args.n_heads;
			int kvh = h / kv_mul;
			int t = (j / args.n_heads) * 8 + (threadIdx.x % warpSize) / 4;

			unsigned active = __ballot_sync(0xffffffff, t < args.kv_len);

			if (t < args.kv_len) {
				float* qh = args.q + h * head_dim;
				KVT* kh = keyb + kvh * head_dim * args.seq_len;
				float* atth = args.att + h * args.seq_len * 2;

				float score = attn_score(kh, qh, head_dim, args.seq_len, t, 4 * (threadIdx.x % 4));

				// reduce score across threads in warp; every 4 threads are processing the same output score
				score += __shfl_xor_sync(active, score, 2);
				score += __shfl_xor_sync(active, score, 1);
				score /= sqrtf(head_dim);

				atth[t] = expf(score);
				atth[t + args.seq_len] = score;

				// to reduce latency we prefer computing softmax without the numeric stabilization, which is safe if all inputs are small
				if (fabsf(score) > 40) {
					badsoftmax = 1;
				}
			}
		}

		syncgrid();
		coopstage(args.perfstats, 1);		

		if (badsoftmax) {			// attention softmax
			if (blockIdx.x < args.n_heads) {
				int h = blockIdx.x;
				float* atth = args.att + h * args.seq_len * 2;
				CU_softmax_v0(atth, atth + args.seq_len, args.kv_len);
			}
			syncgrid();
			coopstage(args.perfstats, 2);
		}

		// attention mix
		for (int j = io; j < q_dim; j += ib) {
			int h = j / head_dim;
			int kvh = h / kv_mul;
			int j_head = j % head_dim;

			float* atth = args.att + h * args.seq_len * 2;
			KVT* vh = valb + kvh * head_dim * args.seq_len;
			KVT* val = vh + j_head * args.seq_len;

			float res = attn_warpdot(val, atth, args.kv_len);

			if (threadIdx.x % warpSize == 0) {
				args.q[j] = res;
			}
		}

		syncgrid();
		coopstage(args.perfstats, 3);		// args.att

		// attention output
		for (int j = io; j < dim; j += ib) {
			float val = matmul_warppar(args.q, L->wo, j, q_dim);

			if (threadIdx.x % warpSize == 0) {
				args.x[j] += val;
			}
		}		
		__syncthreads(); // TODO: unclear why this is needed for determinism
		syncgrid();
		coopstage(args.perfstats, 4);		
}

template <typename T, typename KVT, typename AT>
__global__ __launch_bounds__(1024, 1) static void T_forward_ffn(const __grid_constant__ TRANSFORMER_PIPE<T, KVT> args) {
	extern __shared__ char smem[];
	__shared__ float rmsscale;
	__shared__ float moe_weights[32];
	__shared__ int moe_experts[32];

	AT* xs = (AT*)smem;
	int dim = args.dim,l=args.layNo;
	int hidden_dim = args.hidden_dim;
	int head_dim = args.head_dim;
	int kv_mul = args.n_heads / args.n_kv_heads;
	int q_dim = args.head_dim * args.n_heads;
	int kv_dim = args.head_dim * args.n_kv_heads;
	const int IK = 4; // K consecutive warps per block, groups of K are interleaved across SMs for better work distribution
	int io = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
	int ib = (gridDim.x * blockDim.x) / warpSize;
	// dummy moe weights for non-moe models; will be overwritten by moe gate
	moe_weights[0] = 1.f;
	moe_experts[0] = 0;

	coopstage(args.perfstats, -1); // init timing

	static __device__ int badsoftmax = 0;
		const CoopLayer<T>* L = (const CoopLayer<T>*)(args.cLayers+l);	
		// post-attention CU_rmsnorm (into shared memory)
		if (L->rms_ffn_weight) {
			rmsscale = CU_rmsnorm(xs, args.x, L->rms_ffn_weight, dim, args.norm_eps, args.norm_ln);
		}
		// moegate
		if (args.n_experts) {
			__shared__ float exp[32];
			int j = threadIdx.x / warpSize;
			if (j < args.n_experts) {
				float val = matmul_warppar(xs, L->moegate, j, dim) * rmsscale;
				exp[j] = val;
			}
			__syncthreads();
			if (threadIdx.x < warpSize) {
				moe_gate_warp(moe_weights, moe_experts, exp, args.n_experts, args.n_experts_ac);
			}
			__syncthreads();
		}

		// F.cu_silu(self.w1(x)) * self.w3(x)
		for (int j = io; j < hidden_dim * args.n_experts_ac; j += ib) {
			int je = (j % hidden_dim) + moe_experts[j / hidden_dim] * hidden_dim;
			float v1 = matmul_warppar(xs, L->w1, je, dim) * rmsscale;
			float v3 = matmul_warppar(xs, L->w3, je, dim) * rmsscale;

			float val = (args.act_gelu ? cu_gelu(v1) : cu_silu(v1)) * v3;

			if (threadIdx.x % warpSize == 0) {
				args.hb[j] = val;
			}
		}
		syncgrid();
		coopstage(args.perfstats, 5);		// args.hb

		// self.w2(...) + pre-CU_rmsnorm residual
		for (int j = io; j < dim * args.n_experts_ac; j += ib) {
			int je = (j % dim) + moe_experts[j / dim] * dim;
			float val = matmul_warppar(args.hb + (j / dim) * hidden_dim, L->w2, je, hidden_dim);

			if (threadIdx.x % warpSize == 0) {
				atomicAdd(&args.x[j % dim], val * moe_weights[j / dim]);
			}
		}

		__syncthreads(); // TODO: unclear why this is needed for determinism
		syncgrid();
		coopstage(args.perfstats, 6);
	//}
}

typedef __nv_fp8_e5m2 tpEMBED;
template <> inline float T2Float<__nv_fp8_e5m2>(const __nv_fp8_e5m2* a0)   {
    float a = float(*a0);	
	//a = fp8_to_float(*a0);
    assert(!isnan(a) && !isinf(a));
    return a;
}

float* T_generate_cuda(hFISH hFish, bool isOnlyUpdateKV,bool isCPU,unsigned flags){	//int token, int pos, 	
    assert(hFish!=nullptr);

	TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed",0); 
	int token=embed->hBatch->CurToken(), pos = embed->hBatch->pos++;		
	TRANSFORMER_PIPE<uint32_t, __nv_fp8_e5m2> args(hFish,pos);	//	kv_len,kv_pos,
	
	hGensor out_weight = args.out_weight;	
	int dim = args.dim,vocab_size=embed->nVocab;
	// copy the token embedding into x	
	tpEMBED *token_embedding_table = TO<tpEMBED>(embed->w);
	kernel_embed<<<dim / 32, 32, 0, main_stream>>>(args.x, token_embedding_table, token, dim);
	embed->w->Print("wte",0,0);			PrintTensor<float>("GetRow",args.x,true,dim,1);		
	PrintTensor<__nv_fp8_e5m2>("wq",(__nv_fp8_e5m2*)(args.cLayers[0].wq),true,dim,dim);		
	PrintTensor<__nv_fp8_e5m2>("wk",(__nv_fp8_e5m2*)(args.cLayers[0].wk),true,args.kv_dim,dim);		
	PrintTensor<__nv_fp8_e5m2>("wv",(__nv_fp8_e5m2*)(args.cLayers[0].wv),true,args.kv_dim,dim);		
	// rotate sink tokens forward to keep pace with non-sink tokens
	// if (kv_sink > 0) {
	// 	kernel_rotate_sink<<<dim3(kv_sink * kv_dim / 64, p->n_layers), 32, 0, stream>>>(
	// 	    PROF_TOKEN(kv_sink * kv_dim * sizeof(KVT)), kv_dim, (KVT*)s->key_cache, p->head_dim, kv_sink, log2(p->rope_theta), p->seq_len, p->rotary_dim);
	// }
	OutCLS* cls = hFish->GetNeuron<OutCLS>("OutCLS",0);
	LayerNormal* lnf = hFish->GetNeuron<LayerNormal>("LayerNormal",0);    
    float *logits = TO<float>(cls->preLogits),*rms_final_weight = TO<float>(lnf->w); // (dim,);	
    void* argsp = &args;
	int coopsms = hFish->curDevice()->nCore;
	// PrintTensor<float>("x_0",args.x,true,dim,1);	uint32_t,__nv_fp8_e5m2,float
	double now = GST_ms();
	hGensor cur=GTensor::outL,residual=nullptr;
    // auto err = cudaLaunchCooperativeKernel((void*)T_forward_qkv<__nv_fp8_e5m2, __half, float>,coopsms, 1024, &argsp, args.dim * sizeof(float), main_stream);
    // cudaCheck( err );
	for (int l = 0; l < args.n_layers; ++l) {
		args.layNo = l;
		SelfAttention *QKV = hFish->GetNeuron<SelfAttention>("SelfAttention",l);
		// QKV->FUSE_cuda();
		auto err = cudaLaunchCooperativeKernel((void*)T_forward_qkv<__nv_fp8_e5m2, __half, float>,coopsms, 1024, &argsp, args.dim * sizeof(float), main_stream);
		cudaCheck( err );
		if(0){
			FFN *ffn = hFish->GetNeuron<FFN>("FFN",l);   	
			ffn->FUSE_cuda(cur,0x0);
		}else{
			err = cudaLaunchCooperativeKernel((void*)T_forward_ffn<__nv_fp8_e5m2, __half, float>,coopsms, 1024, &argsp, args.dim * sizeof(float), main_stream);
			cudaCheck( err );
		}
	}
    if (isOnlyUpdateKV) {
		return NULL;
	}
	PrintTensor<float>("q",args.q,true,args.q_dim,1);	
	PrintTensor<float>("att",args.att,true,dim,1);		//why it's inf ???
	PrintTensor<float>("x_1",args.x,true,dim,1);	
	PrintTensor<float>("hb",args.hb,true,args.hb_dim,1);		
//-0.170935 0.218924 0.0977959 ...0.0455288 0.0808399 -0.204381 0.0195478 ...-0.138804 -0.0699319 	"x" avg=0.111772(1536) avg_len=0.18982 sum2=55.3446 [-0.958953,2.267083]
	
	int output_blk = 32 * 32, output_par = 1;
	cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&output_par, kernel_output<__nv_fp8_e5m2, float>, output_blk, dim * sizeof(float)));
	uint64_t pTok = PROF_TOKEN(vocab_size * dim * args.weight_dbits / 8);
	// // classifier into logits		
	kernel_output<__nv_fp8_e5m2, float><<<coopsms * output_par, output_blk, dim * sizeof(float), main_stream>>>(
	    pTok, logits, args.x, TO<__nv_fp8_e5m2>(out_weight), rms_final_weight, dim, vocab_size, args.norm_eps, args.norm_ln);	
	cudaCheck(cudaStreamSynchronize(main_stream));
	cudaCheck(cudaGetLastError()); // check for kernel launch errors; they might fail with OOM due to lazy kernel compilation
	SUM::tX1 += GST_ms()-now;
	// cls->preLogits->Print("logits",0,-1,dim);
	cls->preLogits->SerialData("",nullptr,true);
	return logits;
}