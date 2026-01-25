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

__global__ void CU_init_i(int* vec, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        vec[i] = i;
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
        AT *layer_key_cache = (AT*)hQwen->hCache->Get(KVCache::KV_KEY, l, 0), *layer_value_cache = (AT*)hQwen->hCache->Get(KVCache::KV_VAL, l, 0);
        AT* k_cache_pos = layer_key_cache + (size_t)pos * hQwen->kv_dim;
        AT* v_cache_pos = layer_value_cache + (size_t)pos * hQwen->kv_dim;
        // if (pos == 1) {
        //     DEBUG_HERE;
        // }
        hQwen->InitLayer(l);
        const CoopLayer<QWEN3_PIPE::tpWeight>* L = (const CoopLayer<QWEN3_PIPE::tpWeight>*)(hQwen->cLayers + l);  //	hQwen->cLayers+l
        SelfAttention* QKV                       = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        if (DEBUG.verInferQKV > 0) {  
            INSPECT inspect(QKV);
            QKV->cuInfer(hQwen->inpL, 0x0);
        } else {
            // PrintTensor<QWEN3_PIPE::tpWeight>("rms_att_weight", L->rms_att_weight, true, dim, 1);
            CU_rms_infer(hQwen->xb, hQwen->x, L->rms_att_weight, hQwen->dim);
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
            if (0) {
                // CU_rms_infer(ToX(out), ToX(inpDelta), weight, C);
            } else {
#if defined(USE_FP8_BASELINE)
#else
                CU_rmsnorm_multihead<<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(hQwen->q, L->wq_norm, N_HEADS, HEAD_DIM);
                PrintTensor<QWEN3_PIPE::tpActivation>("q.norm", hQwen->q, true, hQwen->q_dim, 1);
                CU_rmsnorm_multihead<<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>(k_cache_pos, L->wk_norm, N_KV_HEADS, HEAD_DIM);
                PrintTensor<QWEN3_PIPE::tpActivation>("k.norm", k_cache_pos, true, hQwen->kv_dim, 1);
#endif
            }

            // rope_gpu_naive(hQwen->q, k_cache_pos, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, rope_theta);
            CU_rope2_v0<<<dim3(1, 1, N_HEADS), dim3(HEAD_DIM / 2, 1, 1)>>>(hQwen->q, k_cache_pos, pos, N_HEADS, N_KV_HEADS, HEAD_DIM, rope_theta, 42);
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
        SUM::GPU_TIME(SUM::tQKV_forw, now);  // += GST_us() - now;
        now      = GST_us();
        FFN* ffn = hFish->GetNeuron<FFN>("FFN", l);
        if (DEBUG.verInferFFN > 0) {
            INSPECT inspect(ffn);
            ffn->cuInfer(hQwen->inpL, 0x0);
        } else {
            CU_rms_infer(hQwen->xb, hQwen->x, L->rms_ffn_weight, hQwen->dim);
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
    CU_rms_infer(hQwen->x, hQwen->x, rms_final_weight, hQwen->dim);     // last layer normal
    // 12. classifier Matmul
    if (1) {
        INSPECT inspect(cls);
        cls->cuInfer(hQwen->inpL, 0x0);
    } else {
        CU_rms_infer(hQwen->x, hQwen->x, rms_final_weight, hQwen->dim);
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