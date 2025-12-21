/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some idea is from https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu
 *
 *  \brief A large kernel to generate tokens
 *  \author Yingshi Chen
 */
#include "../../Device/Pipe.hpp"
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "./kernel/embed.cuh"
#include "./kernel/gelu.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"
#include "./kernel/sort_rank.cuh"
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

/*
    T(__nv_fp8_e5m2)	type of weight
    KVT(__half)			type of KV cache
    AT float			type of activation
*/
template <typename T, typename KVT, typename AT>
__global__ __launch_bounds__(1024, 1) static void T_forward_qkv(const __grid_constant__ KERNEL_PIPE<AT, KVT, T> args) {
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
    KVT* keyb   = (KVT*)args.hCache->Get(KVCache::KV_KEY) + loff;                               // args.key_cache
    KVT* valb   = (KVT*)args.hCache->Get(KVCache::KV_VAL) + loff;                               // args.val_cache

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
    float block_sum = blockReduce_v0<warpReduceSum>(val, true);
    if (tid == 0)
        atomicAdd(out, block_sum);
}

//	T=_nv_fp8_e5m2, KVT=__half, AT=AT
template <typename T, typename KVT, typename AT>
__global__ __launch_bounds__(1024, 1) static void T_forward_ffn(const __grid_constant__ KERNEL_PIPE<AT, KVT, T> args) {  // KERNEL_PIPE<AT, KVT> args
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
        int je     = (j % hidden_dim) + moe_experts[j / hidden_dim] * hidden_dim;
        float gama = L->gama_1 == nullptr ? 1.0f : (float)(L->gama_1[j]);
        float v1   = matmul_warppar(xs, L->w1, je, dim) * gama;
        // float v1 = matmul_warppar(xs, L->w1, je, dim) ;
        // CU_dot16x4_(&v1, xs, L->w1, je, dim);			SYNC_GRID();
        float v3  = matmul_warppar(xs, L->w3, je, dim);
        float val = (args.act_gelu ? CU_gelu(v1 * rmsscale) : CU_silu(v1 * rmsscale)) * v3 * rmsscale;

        if (threadIdx.x % warpSize == 0) {
            args.hb[j] = (AT)val;
        }
    }
    SYNC_GRID();
    coopstage(args.perfstats, 5);  // args.hb

    // self.w2(...) + pre-CU_rmsnorm resi
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

/*  may have high performance than T_generate_cuda
template <typename AT>
float* T_generate_Cooperative(hFISH hFish, bool isOnlyUpdateKV, int id, unsigned flags) {  // int token, int pos,
    assert(hFish != nullptr);
    if (id == 0) {
        _INFO("\t %s: |T_weight|=%d |T_kv|=%d |T_activity|=%d \n", __func__, sizeof(__nv_fp8_e5m2), sizeof(__half), sizeof(AT));
    }
    TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    int token = embed->hBatch->CurToken(), pos = embed->hBatch->pos++;
    QWEN3_PIPE args(hFish, pos);
    // QWEN_CALM_PIPE args(hFish, pos);  //	kv_len,kv_pos,

    int nActivePC = 1;  //	for __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
    // cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nActivePC, kernel_output<__nv_fp8_e5m2, AT>, dBLOCK, smemPB));
    int nCore = hFish->curDevice()->nCore, smemPB = args.dim * sizeof(float);
    int dim = args.dim, vocab_size = embed->nVocab, dBLOCK = 1024, dGRID = nCore * nActivePC;
    // copy the token embedding into x
    tpEMBED* token_embedding_table = TO<tpEMBED>(embed->w);
    CU_embed_forw_1<<<dim / 32, 32, 0, main_stream>>>(args.x, token_embedding_table, token, dim);
    // embed->w->Print("wte", 0, 0);
    PrintTensor<AT>("token_embed", args.x, true, dim, 1);

    // rotate sink tokens forward to keep pace with non-sink tokens
    // if (kv_sink > 0) {
    // 	kernel_rotate_sink<<<dim3(kv_sink * kv_dim / 64, p->n_layers), 32, 0, stream>>>(
    // 	    PROF_TOKEN(kv_sink * kv_dim * sizeof(KVT)), kv_dim, (KVT*)args.key_cache, p->head_dim, kv_sink, log2(p->rope_theta), p->seq_len, p->rotary_dim);
    // }
    OutCLS* cls             = hFish->GetNeuron<OutCLS>("OutCLS", 0);
    LayerNormal* lnf        = hFish->GetNeuron<LayerNormal>("LayerNormal", 0);
    float* logits           = TO<float>(cls->preLogits);
    float* rms_final_weight = TO<float>(lnf->w);  // (dim,);
    void* argsp             = &args;

    // PrintTensor<float>("x_0",args.x,true,dim,1);	uint32_t,__nv_fp8_e5m2,float

    hGensor cur = GTensor::outL, residual = nullptr;
    for (int l = 0; l < args.n_layers; ++l) {
        args.InitLayer(l);
        SelfAttention* QKV                = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        const CoopLayer<QWEN3_PIPE::tpWeight>* L = (const CoopLayer<QWEN3_PIPE::tpWeight>*)(args.cLayers + l);  //	args.cLayers+l
        PrintTensor<QWEN3_PIPE::tpWeight>("rms_att_weight", L->rms_att_weight, true, dim, 1);
        // QKV->cuFlow();

        cudaError_t err = cudaLaunchCooperativeKernel((void*)T_forward_qkv<__nv_fp8_e5m2, __half, AT>, dGRID, dBLOCK, &argsp, smemPB, main_stream);
        PrintTensor<AT>("hb", args.hb, true, args.hb_dim, 1);
        PrintTensor<AT>("q", args.q, true, args.q_dim, 1);
        PrintTensor<AT>("att", args.att, true, dim, 1);  // why it's inf ???
        PrintTensor<AT>("x_qkv", args.x, true, dim, 1);
        cudaCheck(err);
        SYNC_DEVICE();
        // exit(-13);
        double now = GST_ms();
        FFN* ffn   = hFish->GetNeuron<FFN>("FFN", l);
        if (0) {             //	-0.010254 -0.238807 ...0.491514 0.0411605 0.318263 ...0.088502 	"ffn_inp" |avg|=0.115405(1536) avg_len=0.195593 sum2=58.7623
            ffn->OnDebug();  //[-1.496535,1.585582] nz=0
            ffn->cuFlow(cur, 0x0);
        } else {
            void* kernelArgs[] = {argsp};
            // err = cudaLaunchCooperativeKernel((void*)T_forward_ffn<__nv_fp8_e5m2, __half, AT>,dGRID, dBLOCK, &argsp, smemPB, main_stream);
            err = cudaLaunchCooperativeKernel((void*)T_forward_ffn<__nv_fp8_e5m2, __half, AT>, dGRID, dBLOCK, kernelArgs, smemPB, main_stream);
            // T_forward_ffn<__nv_fp8_e5m2, __half, AT><<<dGRID, dBLOCK, smemPB, main_stream>>>(&args);	//why its slower than cudaLaunchCooperativeKernel
            cudaCheck(err);
        }
        SYNC_DEVICE();
        // SUM::tX1 += GST_ms() - now;
        PrintTensor<AT>("x_ffn", args.x, true, dim, 1);
        // args.AfterLayer(l);
    }
    if (isOnlyUpdateKV) {
        return NULL;
    }
    hGensor out_weight = args.out_weight;
    uint64_t pTok      = PROF_TOKEN(vocab_size * dim * args.weight_dbits / 8);
    // template <typename T, typename AT> (uint64_t, float* xout, AT* x, T* w, float* rms_weight, int n, int d, float norm_eps, bool norm_ln)
    kernel_output<__nv_fp8_e5m2, AT><<<dGRID, dBLOCK, smemPB, main_stream>>>(pTok, logits, args.x, TO<__nv_fp8_e5m2>(out_weight), rms_final_weight, dim,
                                                                             vocab_size, args.norm_eps, args.norm_ln);
    SYNC_DEVICE();
    cudaCheck(cudaGetLastError());  // check for kernel launch errors; they might fail with OOM due to lazy kernel compilation

    // cls->preLogits->Print("logits",0,-1,dim);
    cls->preLogits->SerialData("", nullptr, true);
    return logits;
}*/

// fp32_out should not share same address of bf16_in
__global__ void convert_bf16_to_fp32_kernel(__nv_bfloat16* bf16_in, float* fp32_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        fp32_out[i] = __bfloat162float(bf16_in[i]);
    }
}

bool LogitsInfo::Init(int n_vocab, bool isCPU_, hGensor hClsLogits, int flag) {
    isCPU = isCPU_;
    dim   = n_vocab;

    // assert(cls->preLogits->host_data == nullptr);
    index = new int[n_vocab];
    for (int i = 0; i < n_vocab; i++) {
        index[i] = i;
    }
    if (isCPU) {
        logits = new floatLogits[n_vocab];

        hClsLogits->host_data = logits;
    } else {
        logits = TO<floatLogits>(hClsLogits);

        int* host_index = index;
        cudaCheck(cudaMalloc(&index, n_vocab * sizeof(int)));
        H2D(index, host_index, n_vocab * sizeof(int));
        delete[] host_index;

        cudaCheck(cudaMalloc(&index_sorted, n_vocab * sizeof(int)));
        cudaCheck(cudaMalloc(&logits_sorted, n_vocab * sizeof(floatLogits)));
    }
    return true;
}

__global__ void CU_init_i(int* vec, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vec[i] = i;
    }
}

void LogitsInfo::SortPair(int nPick, int flag) {
    if (nPick <= 0)
        nPick = dim;
    if (isCPU) {
        // qsort(probindex, n_cands, sizeof(ProbIndex), compare_prob_desc);
        assert(nPick <= dim);
        for (int i = 0; i < nPick; i++) {
            for (int j = 1; j < nPick; j++) {
                if (logits[i] < logits[j]) {
                    Swap(i, j);
                }
            }
        }
    } else {
        // CU_RadixSorter sorter(_logits,n_vocab);
        if (d_temp == nullptr) {
            cub::DeviceRadixSort::SortPairs(d_temp, szTemp, logits, logits_sorted, index, index_sorted, nPick);
            cudaCheck(cudaMalloc(&d_temp, szTemp));  //
        }
        CU_init_i<<<CEIL_DIV(nPick, CU_T4B_SMALL), CU_T4B_SMALL>>>(index, nPick);
        // cub::DeviceRadixSort::SortKeys(d_temp, szTemp, logits, logits, nPick);
        //  In-place operations are not supported. There must be no overlap between any of the provided ranges!!!
        // cudaMemcpy(index_out, index, sizeof(int) * nPick, cudaMemcpyDeviceToDevice);
        // cudaMemcpy(logits_out, logits, sizeof(floatLogits) * nPick, cudaMemcpyDeviceToDevice);
        cub::DeviceRadixSort::SortPairs(d_temp, szTemp, logits, logits_sorted, index, index_sorted, nPick);
        PrintTensor<floatLogits>("sort_logits", logits_sorted, true, nPick, 1, 1, 1, 0);
        PrintTensor<int>("sort_index", index_sorted, true, nPick, 1, 1, 1, 0);
    }
}

template <typename T>
__global__ void CU_sample(T* prelogitst, int* index, int dim, float coin, int flag) {
    return;
}

bool GeneratOnPrompt::VerifyLogits(int flag) {
    // PrintTensor<floatLogits>("_logits", logits, true, tokenizer->nVocab(), 1, 1, 1, -1);
    return true;
}

TOKEN_ID GeneratOnPrompt::Sample(int idx, bool is_resampling) {
    if (samp_params.isSampleCPU)
        return Sample_cpu(idx, is_resampling);

    // TOKEN_ID id = 0;
    int n_vocab = fish_1 == nullptr ? wiki0->n_vocab : fish_1->nClass();
    assert(n_vocab == gpuLogits.dim);
    PrintTensor<floatLogits>("_logits", gpuLogits.logits, true, n_vocab, 1, 1, 1, 0);
    gpuLogits.SortPair(-1);

    // float coin  = 0.1;  // random_f32(&rng_state);
    // CU_sample<<<1, 1>>>(gpuLogits.logits_sorted, gpuLogits.index_sorted, n_cands, coin, 0x0);
    // H2D(&id, gpuLogits.index_sorted, sizeof(int));
    //  return id
    size_t off = gpuLogits.dim - nCanTopK;
    D2H(gpuLogits.index_sorted + off, cpuLogits.index, sizeof(int) * nCanTopK);
    D2H(gpuLogits.logits_sorted + off, cpuLogits.logits, sizeof(floatLogits) * nCanTopK);
    return Sample_cpu(idx, true);
}

template <>
hGTensor QWEN3_PIPE::tX = nullptr;

template <typename AT>
floatLogits* T_generate_cuda(hFISH hFish, bool isOnlyUpdateKV, MODEL_CARD* hPipe, unsigned flags) {  // int token, int pos,
    constexpr int HEAD_DIM = 128;
    int szBuffer           = hFish->config.chat_sampler.szBuffer;
    int seq_len            = hFish->config.chat_sampler.seq_len;
    int N_HEADS = hFish->config.n_head(), N_KV_HEADS = hFish->config.n_head_kv();
    float rope_theta = hFish->config.model.rope_theta;
    assert(hFish != nullptr && hPipe != nullptr);

    TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    int token = embed->hBatch->CurToken(), pos = embed->hBatch->tok_pos;
    QWEN3_PIPE* hQwen = dynamic_cast<QWEN3_PIPE*>(hPipe);
    // QWEN3_PIPE qwen_pipe(hFish, pos);
    // QWEN3_PIPE *hQwen = &qwen_pipe;
    // hQwen->UpdatePos(0x0);
    int nCore = hFish->curDevice()->nCore, dim = hQwen->dim;

    // tpEMBED* token_embedding_table = TO<tpEMBED>(embed->w);
    if (1) {
        embed->cuInfer(hQwen->inpL, 0x0);
    } else {
        switch (embed->w->type) {
            case typNUMBER::F8E5M2:
                CU_embed_forw_1<<<dim / 32, 32, 0, main_stream>>>(hQwen->x, TO<f8e5>(embed->w), token, dim, 0);
                break;
            default:
                CU_embed_forw_1<<<dim / 32, 32, 0, main_stream>>>(hQwen->x, TO<bf16>(embed->w), token, dim, 0);
                break;
        }
        embed->w->Print("wte", 0, 0), PrintTensor<AT>("token_embed", hQwen->x, true, dim, 1, 1, 1, 0);
    }

    OutCLS* cls         = hFish->GetNeuron<OutCLS>("OutCLS", 0);
    floatLogits* logits = TO<floatLogits>(cls->preLogits);
    LayerNormal* lnf    = hFish->GetNeuron<LayerNormal>("LayerNormal", 0);
    // PrintTensor<float>("x_0",hQwen->x,true,dim,1);	uint32_t,__nv_fp8_e5m2,float
    // printf("\tpos=%d\n", pos);
    for (int l = 0; l < hQwen->n_layers; ++l) {
        double now = GST_us();
        // bf16 *layer_key_cache = hQwen->key_cache + (size_t)l * seq_len * KV_DIM, *layer_value_cache = hQwen->val_cache + (size_t)l * seq_len * KV_DIM;
        bf16 *layer_key_cache = (bf16*)hQwen->hCache->Get(KVCache::KV_KEY, l, 0), *layer_value_cache = (bf16*)hQwen->hCache->Get(KVCache::KV_VAL, l, 0);
        bf16* k_cache_pos = layer_key_cache + (size_t)pos * hQwen->kv_dim;
        bf16* v_cache_pos = layer_value_cache + (size_t)pos * hQwen->kv_dim;
        // if (pos == 1) {
        //     g_dump_level = 0;
        //     DEBUG_HERE;
        // }
        hQwen->InitLayer(l);
        const CoopLayer<QWEN3_PIPE::tpWeight>* L = (const CoopLayer<QWEN3_PIPE::tpWeight>*)(hQwen->cLayers + l);  //	hQwen->cLayers+l
        SelfAttention* QKV                       = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        if (DEBUG.verInferQKV > 0) {  // flags == 0
            // QKV->OnDebug();
            QKV->cuInfer(hQwen->inpL, 0x0);
        } else {
            // PrintTensor<QWEN3_PIPE::tpWeight>("rms_att_weight", L->rms_att_weight, true, dim, 1);
            CU_rms_v2(hQwen->xb, hQwen->x, L->rms_att_weight, hQwen->dim);
            // PrintTensor<QWEN3_PIPE::tpActivation>("rms_0", hQwen->x, true, dim, 1);
            PrintTensor<QWEN3_PIPE::tpActivation>("rms_xb", hQwen->xb, true, dim, 1);

            CU_mv_(hQwen->q, L->wq, hQwen->xb, hQwen->q_dim, hQwen->dim);
            // PrintTensor<QWEN3_PIPE::tpWeight>("wq", L->wq, true, hQwen->q_dim, hQwen->dim);
            PrintTensor<QWEN3_PIPE::tpActivation>("q", hQwen->q, true, hQwen->q_dim, 1);
            CU_mv_(k_cache_pos, L->wk, hQwen->xb, hQwen->kv_dim, hQwen->dim);
            PrintTensor<QWEN3_PIPE::tpActivation>("k", k_cache_pos, true, hQwen->kv_dim, 1);
            CU_mv_(v_cache_pos, L->wv, hQwen->xb, hQwen->kv_dim, hQwen->dim);
            PrintTensor<QWEN3_PIPE::tpActivation>("v", v_cache_pos, true, hQwen->kv_dim, 1);

            // qk_norm_fused_gpu(hQwen->q, k_cache_pos, L->wq_norm, L->wk_norm);
            constexpr int QK_NORM_THREADS_PER_BLOCK = 64;
            if (1) {
                CU_rmsnorm_multihead<<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(hQwen->q, L->wq_norm, N_HEADS, HEAD_DIM);
                PrintTensor<QWEN3_PIPE::tpActivation>("q.norm", hQwen->q, true, hQwen->q_dim, 1);
                CU_rmsnorm_multihead<<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(k_cache_pos, L->wk_norm, N_KV_HEADS, HEAD_DIM);
                PrintTensor<QWEN3_PIPE::tpActivation>("k.norm", k_cache_pos, true, hQwen->kv_dim, 1);
            } else {
                fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM>
                    <<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(hQwen->q, L->wq_norm, N_HEADS, 1.0f / HEAD_DIM);
                fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM>
                    <<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(k_cache_pos, L->wk_norm, N_KV_HEADS, 1.0f / HEAD_DIM);
            }

            // rope_gpu_naive(hQwen->q, k_cache_pos, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, rope_theta);
            CU_rope2_forward<<<dim3(N_HEADS, 1, 1), dim3(HEAD_DIM / 2, 1, 1)>>>(hQwen->q, k_cache_pos, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, rope_theta);
            PrintTensor<QWEN3_PIPE::tpActivation>("q.rope", hQwen->q, true, hQwen->q_dim, 1),
                PrintTensor<QWEN3_PIPE::tpActivation>("k.rope", k_cache_pos, true, hQwen->kv_dim, 1);

            // 6. MHA (QK^T V)
            // 6.1: calculate QK scores
            int qk_threads_per_block = std::min(1024, pos + 1);
            if (DEBUG.T_cuQK == 0)
                attention_qk_kernel<<<N_HEADS, qk_threads_per_block>>>(hQwen->att, hQwen->q, layer_key_cache, pos, seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
            else
                attention_qk_kernel_v2<<<dim3(N_HEADS, pos + 1, 1), HEAD_DIM>>>(hQwen->att, hQwen->q, layer_key_cache, pos, seq_len, N_HEADS, N_KV_HEADS,
                                                                                HEAD_DIM);
            // 6.2: softmax
            CU_softmax_multihead<<<N_HEADS, 1>>>(hQwen->att, pos, seq_len);
            PrintTensor<float>("attn", hQwen->att, true, pos + 1, 1);
            // 6.3: aggregate V values
            attention_v_kernel<<<N_HEADS, HEAD_DIM>>>(hQwen->q, hQwen->att, layer_value_cache, pos, seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM);
            // PrintTensor<AT>("hb", hQwen->hb, true, hQwen->hb_dim, 1);
            PrintTensor<AT>("att_out", hQwen->q, true, hQwen->q_dim, 1);
            // PrintTensor<float>("att", hQwen->att, true, dim, 1);  // why it's inf ???
            // 7. final attention output projection and resi connection (fused)
            CU_mv_(hQwen->x, L->wo, hQwen->q, hQwen->dim, hQwen->q_dim, 1.0f, 1.0f);
            PrintTensor<QWEN3_PIPE::tpActivation>("att_x", hQwen->x, true, hQwen->dim, 1);
        }
        cudaCheck(cudaGetLastError());
        hQwen->inpL->Print("x_qkv", 0x0, 0);
        // if (pos == 1) {            K_EXIT(-13);        }
        SUM::GPU_TIME(SUM::tQKV, now);  // += GST_us() - now;
        now      = GST_us();
        FFN* ffn = hFish->GetNeuron<FFN>("FFN", l);
        if (DEBUG.verInferFFN > 0) {
            // ffn->OnDebug();
            ffn->cuInfer(hQwen->inpL, 0x0);
        } else {
            CU_rms_v2(hQwen->xb, hQwen->x, L->rms_ffn_weight, hQwen->dim);
            // 9. FFN projections (Gate and Up)
            // output of w1 matmul is hQwen->hb. output of w3 matmul is hQwen->hb2.
            CU_mv_(hQwen->hb, L->w3, hQwen->xb, hQwen->hidden_dim,
                   hQwen->dim);  // matmul_cublas(cublas_handle, hQwen->hb, L->w3, hQwen->xb, hQwen->hidden_dim, hQwen->dim);
            CU_mv_(hQwen->hb2, L->w1, hQwen->xb, hQwen->hidden_dim,
                   hQwen->dim);  // matmul_cublas(cublas_handle, hQwen->hb2, L->w1, hQwen->xb, hQwen->hidden_dim, hQwen->dim);
            // 9. SwiGLU
            // in-place operation on hQwen->hb, using hQwen->hb2 as the gate.
            swiglu_forward(hQwen->hb, hQwen->hb, hQwen->hb2, hQwen->hidden_dim);
            // 10. final FFN Down Projection matmul and resi connection (fused)
            CU_mv_(hQwen->x, L->w2, hQwen->hb, hQwen->dim, hQwen->hidden_dim, 1.0f,
                   1.0f);  // matmul_cublas(cublas_handle, hQwen->x, L->w2, hQwen->hb, hQwen->dim, hQwen->hidden_dim, 1.0f, 1.0f);
            cudaCheck(cudaGetLastError());
        }
        SUM::GPU_TIME(SUM::tFFN, now);  // SUM::tFFN += GST_us() - now;
        PrintTensor<AT>("x_ffn", hQwen->x, true, dim, 1, 1, 1, 0);
        // hQwen->AfterLayer(l);
        if (l > DEBUG.T_generate_most_layer)
            exit(KOIFISH_EXIT_DEBUG);
    }
    if (isOnlyUpdateKV) {
        return NULL;
    }

    // 11. final RMSNorm
    // in-place operation on hQwen->x
    floatX* rms_final_weight = TO<floatX>(lnf->w);  // (dim,);
    CU_rms_v2(hQwen->x, hQwen->x, rms_final_weight, hQwen->dim);
    // 12. classifier Matmul
    if (1) {
        cls->cuInfer(hQwen->inpL, 0x0);
    } else {
        CU_rms_v2(hQwen->x, hQwen->x, rms_final_weight, hQwen->dim);
        hGensor out_weight = hQwen->out_weight;
        CU_mv_(hQwen->xlogit, ToX(out_weight), hQwen->x, hQwen->vocab_size, hQwen->dim);
        // int grid_size = (hQwen->vocab_size + CU_T4B_SMALL - 1) / CU_T4B_SMALL;
        // only for using floatLogits = float;
        // convert_bf16_to_fp32_kernel<<<grid_size, CU_T4B_SMALL>>>(hQwen->xlogit, logits, hQwen->vocab_size);
        cls->preLogits->SerialData("", nullptr, true);
    }
    // cls->preLogits->Print("logits",0,-1,hQwen->vocab_size);

    return logits;
}

floatLogits* T_generate_(hFISH hFish, MODEL_CARD* hPipe, typNUMBER tpActivity, int flags) {  // int token, int pos,
    floatLogits* logits = nullptr;

    if (DEBUG.T_cpu == 1) {
        // T_generate_cpu(SharedThis(), false, flag);
    } else {
        switch (tpActivity) {
            case typNUMBER::F32:
                // logits = T_generate_cuda<float>(hFish, false, hPipe, flags);
                break;
            case typNUMBER::BF16:
                logits = T_generate_cuda<__nv_bfloat16>(hFish, false, hPipe, flags);
                break;
            // case typNUMBER::F8E5M2:
            // 	logits = T_generate_cuda<__nv_fp8_e5m2>(hFish,false,hPipe,flags);
            // 	break;
            default:
                assert(0);
        }
    }
    return logits;
}