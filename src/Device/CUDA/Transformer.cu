/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 *
 *  \brief Transformer in cuda kernel
 *  \author Yingshi Chen
 */
#include "../../Device/Pipe.hpp"
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "./kernel/Operator.cuh"
#include "./kernel/embed.cuh"
#include "./kernel/gemm.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/utils.cuh"

extern cudaStream_t main_stream;

// __global__ void __launch_bounds__(1024) test_print_kernel(half *__restrict__ arr){
//     // printf("test kernel\n");
//     if (((int)blockIdx.x == 0) && ((int)threadIdx.x == 0))    {
//         // arr[0] = ((half)(2));
//         __syncthreads();
//         printf("%f ", __half2float(arr[0]));

//     }
// }

#define MAX_LAYERS 128
#define MAX_EXPERTS 64

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
__global__ __launch_bounds__(1024, 1) static void T_forward_qkv(const __grid_constant__ KERNEL_PIPE<AT, KVT> args) {
    extern __shared__ char smem[];
    __shared__ float rmsscale;

    AT* xs  = (AT*)smem;
    int dim = args.dim, l = args.layNo;
    int hidden_dim = args.hidden_dim;
    int head_dim   = args.head_dim;
    int kv_mul     = args.n_heads / args.n_kv_heads;
    int q_dim      = args.head_dim * args.n_heads;
    int kv_dim     = args.head_dim * args.n_kv_heads;
    const int IK   = 4;  // K consecutive warps per block, groups of K are interleaved across SMs for better work distribution
    int io         = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
    int ib         = (gridDim.x * blockDim.x) / warpSize;
    coopstage(args.perfstats, -1);  // init timing

    static __device__ int badsoftmax = 0;

    // for (int l = 0; l < args.n_layers; ++l) {
    //  const CoopLayer<T>* L = (const CoopLayer<T>*)&cooplayers[l];
    const CoopLayer<T>* L = (const CoopLayer<T>*)(args.cLayers + l);  //	args.cLayers+l
    assert(L != nullptr);
    if (blockIdx.x == 0 && threadIdx.x < warpSize) {
        badsoftmax = 0;
    }

    // pre-attention CU_rmsnorm (into shared memory)
    rmsscale    = CU_rmsnorm(xs, args.x, L->rms_att_weight, dim, args.norm_eps, args.norm_ln);  // 57.1846123
    size_t loff = (size_t)l * args.seq_len * kv_dim;                                            // kv cache layer offset for convenience
    KVT* keyb   = args.key_cache + loff;
    KVT* valb   = args.val_cache + loff;

    // qkv matmul + RoPE encoding + update KV cache
    for (int j = io * 2; j < q_dim + kv_dim * 2; j += ib * 2) {
        T* w  = j < q_dim ? L->wq : (j < q_dim + kv_dim ? L->wk : L->wv);
        int k = j < q_dim ? j : (j < q_dim + kv_dim ? j - q_dim : j - q_dim - kv_dim);
        if (k == 0) {
            k = 0;
        }
        float v0 = matmul_warppar(xs, w, k + 0, dim) * rmsscale;  //	2.92934465
        float v1 = matmul_warppar(xs, w, k + 1, dim) * rmsscale;  //	2.08009243

        if (L->bqkv) {
            v0 += L->bqkv[j + 0];  // 3.53125
            v1 += L->bqkv[j + 1];
        }

        v0 = min(max(v0, -args.qkv_clip), args.qkv_clip);
        v1 = min(max(v1, -args.qkv_clip), args.qkv_clip);

        if (threadIdx.x % warpSize == 0) {
            int j_head = j % head_dim;
            float freq = j_head >= args.rotary_dim ? 0.f : exp2f(-args.theta_log2 * (float)j_head / (float)args.rotary_dim);
            float fcr0, fci0;
            sincosf(args.pos * freq, &fci0, &fcr0);
            float fcr = fcr0, fci = fci0;
            if (j < q_dim) {
                args.q[k + 0] = v0 * fcr - v1 * fci;
                args.q[k + 1] = v0 * fci + v1 * fcr;
            } else if (j < q_dim + kv_dim) {
                // note: k layout is transposed / tiled to improve attn_score performance
                int off       = args.kv_pos * 16 + args.seq_len * (k / 16) * 16 + (k % 16);
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
    SYNC_GRID();
    coopstage(args.perfstats, 0);

    // attention score
    int kv_lent = (args.kv_len + 7) / 8;

    for (int j = io; j < kv_lent * args.n_heads; j += ib) {
        int h   = j % args.n_heads;
        int kvh = h / kv_mul;
        int t   = (j / args.n_heads) * 8 + (threadIdx.x % warpSize) / 4;

        unsigned active = __ballot_sync(0xffffffff, t < args.kv_len);

        if (t < args.kv_len) {
            AT* qh   = args.q + h * head_dim;
            KVT* kh  = keyb + kvh * head_dim * args.seq_len;
            AT* atth = args.att + h * args.seq_len * 2;

            float score = attn_score(kh, qh, head_dim, args.seq_len, t, 4 * (threadIdx.x % 4));

            // reduce score across threads in warp; every 4 threads are processing the same output score
            score += __shfl_xor_sync(active, score, 2);
            score += __shfl_xor_sync(active, score, 1);
            score /= sqrtf(head_dim);

            atth[t]                = expf(score);
            atth[t + args.seq_len] = score;

            // to reduce latency we prefer computing softmax without the numeric stabilization, which is safe if all inputs are small
            if (fabsf(score) > 40) {
                badsoftmax = 1;
            }
        }
    }

    SYNC_GRID();
    coopstage(args.perfstats, 1);

    if (badsoftmax) {  // attention softmax
        if (blockIdx.x < args.n_heads) {
            int h    = blockIdx.x;
            AT* atth = args.att + h * args.seq_len * 2;
            CU_softmax_v0(atth, atth + args.seq_len, args.kv_len);
        }
        SYNC_GRID();
        coopstage(args.perfstats, 2);
    }

    // attention mix
    for (int j = io; j < q_dim; j += ib) {
        int h      = j / head_dim;
        int kvh    = h / kv_mul;
        int j_head = j % head_dim;

        AT* atth = args.att + h * args.seq_len * 2;
        KVT* vh  = valb + kvh * head_dim * args.seq_len;
        KVT* val = vh + j_head * args.seq_len;

        float res = attn_warpdot(val, atth, args.kv_len);

        if (threadIdx.x % warpSize == 0) {
            args.q[j] = res;
        }
    }

    SYNC_GRID();
    coopstage(args.perfstats, 3);  // args.att

    // attention output
    for (int j = io; j < dim; j += ib) {
        float val = matmul_warppar(args.q, L->wo, j, q_dim);

        if (threadIdx.x % warpSize == 0) {
            args.x[j] += val;
        }
    }
    __syncthreads();  // TODO: unclear why this is needed for determinism
    SYNC_GRID();
    coopstage(args.perfstats, 4);
}

template <typename T>
__device__ inline void CU_dot16x4_(float* out, T* x, __nv_fp8_e5m2* w, int i, int n) {
    int tid = threadIdx.x, idx = blockIdx.x * blockDim.x + tid;
    if (idx * 4 >= n) {
        return;
    }
    float val     = 0.0f;
    int NUM_WARPS = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    assert(NUM_WARPS <= WARP_SIZE);
    // for(int j = lane * 4; j < n; j += warpSize * 4) {
    float4 ww = fp8x4_e5m2_ff((__nv_fp8x4_e5m2*)(w + i * n + idx * 4));
    float4 xx = {x[idx * 4], x[idx * 4 + 1], x[idx * 4 + 2], x[idx * 4 + 3]};  //*(float4*)&x[idx];
    val += ww.x * xx.x;
    val += ww.y * xx.y;
    val += ww.z * xx.z;
    val += ww.w * xx.w;
    //}
    // *out = warpreduce_sum(val);
    float block_sum = blockReduce<warpReduceSum>(val, true);
    if (tid == 0)
        atomicAdd(out, block_sum);
}

//	T=_nv_fp8_e5m2, KVT=__half, AT=AT
template <typename T, typename KVT, typename AT>
__global__ __launch_bounds__(1024, 1) static void T_forward_ffn(const __grid_constant__ KERNEL_PIPE<AT, KVT> args) {  // KERNEL_PIPE<AT, KVT> args
    extern __shared__ char smem[];
    __shared__ float rmsscale, moe_weights[32];
    __shared__ int moe_experts[32];

    AT* xs  = (AT*)smem;
    int dim = args.dim, hidden_dim = args.hidden_dim;
    const CoopLayer<T>* L = (const CoopLayer<T>*)(args.cLayers + args.layNo);
    const int IK          = 4;  // K consecutive warps per block, groups of K are interleaved across SMs for better work distribution
    int io                = blockIdx.x * IK + (threadIdx.x / warpSize % IK) + gridDim.x * IK * (threadIdx.x / warpSize / IK);
    // blockIdx.x + threadIdx.x / warpSize + gridDim.x * (threadIdx.x / warpSize )
    int ib = (gridDim.x * blockDim.x) / warpSize;
    // dummy moe weights for non-moe models; will be overwritten by moe gate
    moe_weights[0] = 1.f;
    moe_experts[0] = 0;

    coopstage(args.perfstats, -1);  // init timing

    if (L->rms_ffn_weight) {
        rmsscale = CU_rmsnorm(xs, args.x, L->rms_ffn_weight, dim, args.norm_eps, args.norm_ln);  // 5.11613894
    }
    // __syncthreads(); 	SYNC_GRID();		return;
    // moegate
    if (args.n_experts) {
        __shared__ float exp[32];
        int j = threadIdx.x / warpSize;
        if (j < args.n_experts) {
            float val = (float)matmul_warppar(xs, L->moegate, j, dim) * rmsscale;
            exp[j]    = val;
        }
        __syncthreads();
        if (threadIdx.x < warpSize) {
            moe_gate_warp(moe_weights, moe_experts, exp, args.n_experts, args.n_experts_ac);
        }
        __syncthreads();
    }

    // F.CU_silu(self.w1(x)) * self.w3(x)
    for (int j = io; j < hidden_dim * args.n_experts_ac; j += ib) {
        int je = (j % hidden_dim) + moe_experts[j / hidden_dim] * hidden_dim;
        // float v1 = matmul_warppar(xs, L->w1, je, dim) ;
        float v1 = L->gama_1 == nullptr ? matmul_warppar(xs, L->w1, je, dim) : matmul_warppar(xs, L->w1, je, dim) * L->gama_1[j];
        // CU_dot16x4_(&v1, xs, L->w1, je, dim);			SYNC_GRID();
        float v3  = matmul_warppar(xs, L->w3, je, dim);
        float val = (args.act_gelu ? CU_gelu(v1 * rmsscale) : CU_silu(v1 * rmsscale)) * v3 * rmsscale;

        if (threadIdx.x % warpSize == 0) {
            args.hb[j] = (AT)val;
        }
    }
    SYNC_GRID();
    coopstage(args.perfstats, 5);  // args.hb

    // self.w2(...) + pre-CU_rmsnorm residual
    for (int j = io; j < dim * args.n_experts_ac; j += ib) {
        int je    = (j % dim) + moe_experts[j / dim] * dim;
        float val = matmul_warppar(args.hb + (j / dim) * hidden_dim, L->w2, je, hidden_dim);

        if (threadIdx.x % warpSize == 0) {
            atomicAdd(&args.x[j % dim], val * moe_weights[j / dim]);
        }
    }

    __syncthreads();  // TODO: unclear why this is needed for determinism
    SYNC_GRID();
    coopstage(args.perfstats, 6);
}
typedef __nv_fp8_e5m2 tpEMBED;

// classifier into logits		T=__nv_fp8_e5m2, AT=AT
template <typename T, typename AT>
__global__ static void kernel_output(uint64_t pTok, float* xout, AT* x, T* w, float* rms_weight, int n, int d, float norm_eps, bool norm_ln) {
    extern __shared__ char smem[];
    AT* xs         = (AT*)smem;
    float rmsscale = CU_rmsnorm(xs, x, rms_weight, n, norm_eps, norm_ln);  //	x0=0.626402617	0.286915213
    int wid        = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int NUM_WARPS  = (gridDim.x * blockDim.x) / warpSize;
    for (int j = wid; j < d; j += NUM_WARPS) {
        float val = matmul_warppar(xs, w, j, n);
        // CU_dot16x4_(&val, xs, w, j, n);
        // instead of writing one value per block, we transpose the values and write all results from first warp
        val = blocktranspose(val * rmsscale, 0.f);

        if (threadIdx.x < blockDim.x / warpSize) {
            xout[j + threadIdx.x] = val;
        }
    }
}

template <typename AT>
float* T_generate_cuda(hFISH hFish, bool isOnlyUpdateKV, int id, unsigned flags) {  // int token, int pos,
    assert(hFish != nullptr);
    if (id == 0) {
        _INFO("\t %s: |T_weight|=%d |T_kv|=%d |T_activity|=%d \n", __func__, sizeof(__nv_fp8_e5m2), sizeof(__half), sizeof(AT));
    }
    TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    int token = embed->hBatch->CurToken(), pos = embed->hBatch->pos++;
    KERNEL_PIPE<AT, __half> args(hFish, pos);  //	kv_len,kv_pos,

    int nActivePC = 1;  //	for __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
    // cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nActivePC, kernel_output<__nv_fp8_e5m2, AT>, dBLOCK, smemPB));
    int nCore = hFish->curDevice()->nCore, smemPB = args.dim * sizeof(float);
    int dim = args.dim, vocab_size = embed->nVocab, dBLOCK = 1024, dGRID = nCore * nActivePC;
    // copy the token embedding into x
    tpEMBED* token_embedding_table = TO<tpEMBED>(embed->w);
    kernel_embed<<<dim / 32, 32, 0, main_stream>>>(args.x, token_embedding_table, token, dim);
    embed->w->Print("wte", 0, 0);
    PrintTensor<AT>("GetRow", args.x, true, dim, 1);

    // rotate sink tokens forward to keep pace with non-sink tokens
    // if (kv_sink > 0) {
    // 	kernel_rotate_sink<<<dim3(kv_sink * kv_dim / 64, p->n_layers), 32, 0, stream>>>(
    // 	    PROF_TOKEN(kv_sink * kv_dim * sizeof(KVT)), kv_dim, (KVT*)s->key_cache, p->head_dim, kv_sink, log2(p->rope_theta), p->seq_len, p->rotary_dim);
    // }
    OutCLS* cls             = hFish->GetNeuron<OutCLS>("OutCLS", 0);
    LayerNormal* lnf        = hFish->GetNeuron<LayerNormal>("LayerNormal", 0);
    float* logits           = TO<float>(cls->preLogits);
    float* rms_final_weight = TO<float>(lnf->w);  // (dim,);
    void* argsp             = &args;

    // PrintTensor<float>("x_0",args.x,true,dim,1);	uint32_t,__nv_fp8_e5m2,float

    hGensor cur = GTensor::outL, residual = nullptr;
    // auto err = cudaLaunchCooperativeKernel((void*)T_forward_qkv<__nv_fp8_e5m2, __half, float>,dGRID, dBLOCK, &argsp, smemPB, main_stream);
    // cudaCheck( err );
    for (int l = 0; l < args.n_layers; ++l) {
        args.InitLayer(l);
        SelfAttention* QKV                = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        const CoopLayer<__nv_fp8_e5m2>* L = (const CoopLayer<__nv_fp8_e5m2>*)(args.cLayers + l);  //	args.cLayers+l
        PrintTensor<float>("rms_att_weight", L->rms_att_weight, true, dim, 1);
        // QKV->cuTrain();

        auto err = cudaLaunchCooperativeKernel((void*)T_forward_qkv<__nv_fp8_e5m2, __half, AT>, dGRID, dBLOCK, &argsp, smemPB, main_stream);
        PrintTensor<AT>("hb", args.hb, true, args.hb_dim, 1);
        PrintTensor<AT>("q", args.q, true, args.q_dim, 1);
        PrintTensor<AT>("att", args.att, true, dim, 1);  // why it's inf ???
        PrintTensor<AT>("x_qkv", args.x, true, dim, 1);
        cudaCheck(err);
        cudaCheck(cudaStreamSynchronize(main_stream));
        // exit(-13);
        double now = GST_ms();
        FFN* ffn   = hFish->GetNeuron<FFN>("FFN", l);
        if (0) {             //	-0.010254 -0.238807 ...0.491514 0.0411605 0.318263 ...0.088502 	"ffn_inp" |avg|=0.115405(1536) avg_len=0.195593 sum2=58.7623
            ffn->OnDebug();  //[-1.496535,1.585582] nz=0
            ffn->cuTrain(cur, 0x0);
        } else {
            void* kernelArgs[] = {argsp};
            // err = cudaLaunchCooperativeKernel((void*)T_forward_ffn<__nv_fp8_e5m2, __half, AT>,dGRID, dBLOCK, &argsp, smemPB, main_stream);
            err = cudaLaunchCooperativeKernel((void*)T_forward_ffn<__nv_fp8_e5m2, __half, AT>, dGRID, dBLOCK, kernelArgs, smemPB, main_stream);
            // T_forward_ffn<__nv_fp8_e5m2, __half, AT><<<dGRID, dBLOCK, smemPB, main_stream>>>(&args);	//why its slower than cudaLaunchCooperativeKernel
            cudaCheck(err);
        }
        cudaCheck(cudaStreamSynchronize(main_stream));
        // SUM::tX1 += GST_ms() - now;
        PrintTensor<AT>("x_ffn", args.x, true, dim, 1);
        // args.AfterLayer(l);
    }
    if (isOnlyUpdateKV) {
        return NULL;
    }
    //-0.170935 0.218924 0.0977959 ...0.0455288 0.0808399 -0.204381 0.0195478 ...-0.138804 -0.0699319 	"x" avg=0.111772(1536) avg_len=0.18982 sum2=55.3446
    //[-0.958953,2.267083]

    hGensor out_weight = args.out_weight;
    uint64_t pTok      = PROF_TOKEN(vocab_size * dim * args.weight_dbits / 8);
    // template <typename T, typename AT> (uint64_t, float* xout, AT* x, T* w, float* rms_weight, int n, int d, float norm_eps, bool norm_ln)
    kernel_output<__nv_fp8_e5m2, AT><<<dGRID, dBLOCK, smemPB, main_stream>>>(pTok, logits, args.x, TO<__nv_fp8_e5m2>(out_weight), rms_final_weight, dim,
                                                                             vocab_size, args.norm_eps, args.norm_ln);
    cudaCheck(cudaStreamSynchronize(main_stream));
    cudaCheck(cudaGetLastError());  // check for kernel launch errors; they might fail with OOM due to lazy kernel compilation

    // cls->preLogits->Print("logits",0,-1,dim);
    cls->preLogits->SerialData("", nullptr, true);
    return logits;
}

float* T_generate_(hFISH hFish, int id, typNUMBER tpActivity, unsigned flags) {  // int token, int pos,
    float* logits = nullptr;

    if (DEBUG.T_cpu == 1) {
        // T_generate_cpu(SharedThis(), false, flag);
    } else {
        switch (tpActivity) {
            case typNUMBER::F32:
                logits = T_generate_cuda<float>(hFish, false, id, flags);
                break;
            case typNUMBER::BF16:
                logits = T_generate_cuda<__nv_bfloat16>(hFish, false, id, flags);
                break;
            // case typNUMBER::F8E5M2:
            // 	logits = T_generate_cuda<__nv_fp8_e5m2>(hFish,false,id,flags);
            // 	break;
            default:
                assert(0);
        }
    }
    return logits;
}