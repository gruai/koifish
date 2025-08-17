
// #include "../ggex/GG_util.hpp"       //ugly  "__builtin_ia32_ldtilecfg" is undefined
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "./cuda_common.h"
#include "./kernel/Operator.cuh"
#include "./kernel/embed.cuh"
#include "./kernel/fused_classifier.cuh"
#include "./kernel/gelu.cuh"
#include "./kernel/layernorm.cuh"
#define NOMINMAX

cublasComputeType_t cublas_compute   = CUBLAS_COMPUTE_32F;
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
cublasLtHandle_t cublaslt_handle;
cublasHandle_t cublas_handle;
void *cublaslt_workspace = NULL;
cudaStream_t main_stream = nullptr;
cudaDeviceProp deviceProp;

hGTensor huTensor::_Multiply(const hGTensor &b) {
    huTensor *cuB = dynamic_cast<huTensor *>(b.get());
    assert(cuB != nullptr);
    return nullptr;
}

bool TokenEmbed::UpdateBucket(int type, int flag) {
    num_c_groups = CEIL_DIV(C, (WARP_SIZE * x128::size));
    if (bucket_info != NULL)
        return false;

    assert((size_t)(B * T) * num_c_groups < (1ULL << 31ULL));  // todo - maybe an issue for llama3-400B(?)
    workload_indices = new int[B * T * num_c_groups];
    bucket_info      = new int4[B * T * num_c_groups];
    return true;
}
void TokenEmbed::WorkloadOnBucker(int *inputs_cpu, int flag) {
    // if(num_buckets>0)
    //     return;

    // Step 1: Sort inputs into buckets
    int total_items = 0;
    std::unordered_map<uint64_t, std::vector<uint64_t>> buckets;
    for (uint64_t bt = 0; bt < B * T; bt++) {
        for (uint64_t c_group = 0; c_group < num_c_groups; c_group++) {
            // todo - passing c_group/inputs_cpu[bt] in data to avoid a second hash lookup is a bit hacky
            uint64_t data = bt + (c_group << 32ULL) + ((uint64_t)inputs_cpu[bt] << 42ULL);
            buckets[c_group + num_c_groups * inputs_cpu[bt]].push_back(data);
            total_items++;
        }
    }
    // Step 2: Sort buckets by size in descending order
    // this is so the largest buckets are processed first by the GPU
    // otherwise, if they started late, they would still be running with the rest of the GPU idle
    std::vector<std::pair<uint64_t, std::vector<uint64_t>>> sortedBuckets(buckets.begin(), buckets.end());
    std::sort(sortedBuckets.begin(), sortedBuckets.end(),  // ugly because we don't have a typedef for the std::pair
              [](const std::pair<uint64_t, std::vector<uint64_t>> &a, const std::pair<uint64_t, std::vector<uint64_t>> &b) {
                  return a.second.size() > b.second.size();
              });

    num_buckets        = buckets.size();
    int bucket_index   = 0;
    int workload_index = 0;
    for (const auto &bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index;                                       // bucket start
        bucket_info[bucket_index].y = bucket.second.size();                                 // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1);  // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1);  // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL << 31ULL) - 1ULL));
        }
        bucket_index++;
    }

    floatX *scratch = (floatX *)GTensor::buff;
    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4 *d_bucket_info     = (int4 *)scratch;
    int *d_workload_indices = (int *)(scratch + B * T * num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, main_stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, main_stream));
}

hGTensor TokenEmbed::OnEmbed(hGensor inpL, int seed) {
    try {
        int OC = w->ne[1], Vp = padded_nCls;
        hGTensor cur = out, curW = w;
        if (isForward()) {
            inp = inpL;
            // grid_size = CEIL_DIV(B * T * C, block_size);
            // CU_embed_forw_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(cur), TO<int>(inp), ToX(curW), ToX0(b), B, T, C);
            if (w->type == typNUMBER::T_BINARY_3) {
                CU_embed_ternary_forw_<floatX>
                    <<<CEIL_DIV(B * T, block_size), block_size, 0, main_stream>>>(ToX(cur), TO<int>(inp), curW->gama_T(), TO<char>(curW), ToX0(b), B, T, C);
            } else
                CU_embed_forw_<<<CEIL_DIV(B * T, block_size), block_size, 0, main_stream>>>(ToX(cur), TO<int>(inp), ToX(curW), ToX0(b), B, T, C);
            w->Print("wte", 0, 0);  // ToX(w),true,Vp,C
            if (b != nullptr)
                PrintTensor<floatX>("wpe", ToX0(b), true, T, C);
            // PrintTensor<int>("inputs",tokens,true,B,T);            PrintTensor<floatX>("GetRow",ToX(cur),true,B,T,C);
            if (maec != nullptr) {
                maec->ENC(cur);
            }
        } else {
            UpdateBucket(0x0);
            WorkloadOnBucker(hBatch->host, 0x0);
            floatX *scratchX = (floatX *)GTensor::buff;
            hGTensor delta = GTensor::delta, cur = delta;
            if (maec != nullptr) {
                cur = maec->ENC(cur);
            }
            // encoder_backward_1(ToG(w), ToG0(b), ToX(cur), tokens, B, T, C, seed, main_stream);
            encoder_backward(ToG(w), ToG0(b), scratchX, workload_indices, bucket_info, ToX(cur), TO<int>(inp), hBatch->host, B, T, C, seed, main_stream);
            // lnW.cuTrain();   lnf->cuTrain(ToG(w),(float*)scratchX,delta);
            // PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
            // PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
        }
        return cur;
    } catch (...) {
        assert(0);
        return nullptr;
    }
}

//  seed - use stochastic rounding to go from FP32 to BF16
hGTensor TokenEmbed::SubW(hGTensor hSamp, bool isForw, hGTensor wOut, int flag) {
    try {
        int nSamp = hSamp->size(), *samps = TO<int>(hSamp), nLayer = hFish->config.nLayer();
        int OC = w->ne[1], Vp = padded_nCls, seed = 42, T = nSamp, B = 1;
        grid_size    = CEIL_DIV(B * T * C, block_size);
        hGTensor cur = wOut, wSrc = flag == 0 ? w : wInv;

        if (isForw) {
            // encoder_forward(ToX(cur), samps, ToX(wSrc), nullptr, 1, T, C, main_stream);
            CU_embed_forw_tc<<<grid_size, block_size, 0, main_stream>>>(ToX(cur), samps, ToX(wSrc), T, C, Vp, flag == 1);
            cur->Print("subW", 0, 0);
        } else {
            CU_embed_back_<<<grid_size, block_size, 0, main_stream>>>(ToG(wSrc), samps, ToX(cur), T, C, Vp, flag == 1);
            // encoder_backward(ToG(wSrc), nullptr, scratchX, workload_indices, bucket_info, ToX(cur), samps, hBatch->host, 1, T, C, seed, main_stream);
        }
        return cur;
    } catch (...) {
        assert(0);
        return nullptr;
    }
}

// wrapper of CU_abc_ & cublasGemmEx & more ...
void CU_mm_(floatX *d, hGTensor gensor, const floatX *b, const floatX *bias, int m, int n, int k, cudaStream_t stream, int transA, int transB, float beta,
            floatX *pre_gelu, bool backward) {
    const float alpha     = 1.0f;  //, beta = accumulate ? 1.0f : 0.0f;
    cublasOperation_t opA = (transA) ? CUBLAS_OP_T : CUBLAS_OP_N, opB = (transB) ? CUBLAS_OP_T : CUBLAS_OP_N;
    if (bias != nullptr || pre_gelu != nullptr) {  //  bias != nullptr || pre_gelu != nullptr
        floatX *wX = gensor->GetDataX();
        CU_mm_blas(d, wX, b, bias, m, n, k, main_stream, transA, transB, 1.0, beta, pre_gelu, backward);
        return;
    }
    bool isBlas = true;
    int lda = transA ? k : m, ldb = transB ? n : k;
    if (!transB && DEBUG.T_GEMM >= 0) {  //  !transA && !transB && DEBUG.T_GEMM >= 0
        // Back of delta: [768,50304] x [50304,8192] => [768,8192]
        CU_abc(d, gensor, b, bias, m, n, k, stream, transA, transB, beta, pre_gelu, backward);
        isBlas = false;
    }
    if (isBlas) {
        floatX *wX = gensor->GetDataX();
        // [50304,768] x [768,8192] => [50304,8192]         or(transA) [768,50304]' x [768,8192] => [50304,8192]
        cublasGemmEx(cublas_handle, opA, opB, m, n, k, &alpha, wX, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &beta, d, CUDA_R_16BF, m, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT);  //  CUBLAS_GEMM_DEFAULT_TENSOR_OP[DEPRECATED]
    }

    return;
}

int SLP::Forw(hGTensor rhs_0, hGTensor lhs_, hGTensor toGelu, int flag) {
    try {
        floatX *rhs = ToX(rhs_0), *to_gelu = ToX0(toGelu);
        int OC = nOut, IC = nIn;
        assert(gelu_fusion == 0);
        assert(rhs_0->size() >= B * T * OC);
        size_t dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
        inp          = lhs_;
        bool transAW = true;
        // if(isTransW)
        //     transAW = false;
        if (compression == SAMPLE && subw != nullptr) {
            subw->SubW(hSamps, true, GTensor::tmpW, samp_type);
            // wX = ToX(GTensor::tmpW);
            // GTensor::tmpW->Print("subW",0,-1);
        }

        rhs = to_gelu ? to_gelu : rhs;
        CU_mm_(rhs, w, ToX(lhs_), ToX0(b), OC, B * T, IC, main_stream, true);
        // CU_mm_blas(rhs, wX, ToX(lhs_), ToX0(b), OC, B * T,  IC, main_stream, true);
        // if(G_Has_(name,{"ffn_down", "ffn_up"}))
        //     lhs_->Print(name,0,-1);
        if (to_gelu) {
            gelu_forward(ToX(rhs_0), to_gelu, B * T * OC, main_stream);
            // swiglu_forward(rhs, to_gelu, B*T*OC, main_stream);
        }
        if (compression == SAMPLE) {
            // rhs_0->Print(rhs_0->name,0,-1);
        }

        return 0x0;
    } catch (...) {
        assert(0);
        return -1;
    }
}

floatX *GTensor::GetDataX(int flag, const string &sX) {
    size_t dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
    // int seed = 42;
    floatX *wX = (floatX *)(data);
    if (hRef != nullptr) {
        int debug = 0x0;
    }

    switch (type) {
        case typNUMBER::T_SIGN:
            assert(0);
            break;
        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3:
            if (DEBUG.T_ternary == 1) {          // each weight(floatX) is {-1,0,1}, no need to extract
            } else if (DEBUG.T_ternary == -1) {  // each weight {-1,0,1}
                return nullptr;
            } else {  // extract each weight {-1,0,1} to floatX
                wX = ToX(GTensor::tmpTernary);
                CU_ternary2X_<floatX><<<CEIL_DIV(ne[0], dT4B), dT4B, 0, main_stream>>>(gama_T(), (char *)(data), wX, ne[0], ne[1], 1);
                SYNC_DEVICE();
                if (flag == -1)
                    GTensor::tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1);
            }

            // PrintTensor<floatX>("wX", wX, true, ne[0], ne[1], ne[2], ne[3], -1);
            break;
        case typNUMBER::T_BINARY_TILE: {
            dim3 dBlock(THREAD_TILE_M * THREAD_TILE_N), dGrid(CEIL_DIV(ne[0], THREAD_TILE_M), CEIL_DIV(ne[1], THREAD_TILE_N));
            assert(ne[0] % THREAD_TILE_M == 0 && ne[1] % THREAD_TILE_N == 0);
            wX              = ToX(GTensor::tmpTernary);
            floatGama *gam_ = gama_T();  //
            CU_Tile2X_<floatX><<<dGrid, dBlock, smemPB, main_stream>>>(wX, gam_, 0.0, ne[0], ne[1], seed);
            // SYNC_DEVICE();
            if (flag == -1)
                GTensor::tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1);
        } break;
        default:
            break;
    }

    return wX;
}

int SLP::Back(hGTensor delta, hGTensor inp, hGTensor deltaIn, hGTensor to_gelu, int flag) {
    try {
        // w->isUpdateParam = false;
        floatX *wX = w->GetDataX(), *gW = ToG(w);
        float *dbias_buffer = (float *)GTensor::buff;
        int OC = nOut, IC = nIn;
        assert(delta != nullptr);
        deltaIn->Print("delta_in", 0, flag);
        if (compression == SAMPLE && subw != nullptr) {  // remater to get wX
            subw->SubW(hSamps, true, GTensor::tmpW, samp_type);
            wX = ToX(GTensor::tmpW);  // assert(nSample==OC || nSample==IC);
            gW = ToX(GTensor::tmpGW);
            cudaCheck(cudaMemsetAsync(gW, 0, GTensor::tmpGW->nByte(), main_stream));
        }
        matmul_backward(ToX(delta), gW, ToG0(b), ToX(deltaIn), ToX(inp), wX, dbias_buffer, B, T, IC, OC, main_stream, isTransW, ToX0(to_gelu));
        if (compression == SAMPLE && subw != nullptr) {
            subw->SubW(hSamps, false, GTensor::tmpGW, samp_type);
        }
        // if(w->grad_ref!=nullptr)
        w->Dogleg(-1);

        return 0x0;
    } catch (...) {
        assert(0);
        return -1;
    }
}
int SLP::FUSE_cuda_block(hGTensor rhs, hGTensor lhs, hGTensor gelu, bool isForw, int flag) { return 0x0; }

//  hIn = QKV->out
hGTensor FFN::cuTrain(hGTensor hIn, int flag) {
    hGTensor tGelu = GTensor::scratch, up_out = remater_ffn ? GTensor::tmpFF1 : up.out;
    bool isBias = up.b != nullptr;

    if (isForward()) {
        inp = OnInput(hIn);
        inp->Print("ffn.in", 0, dump_flag, C);
        GTensor::residual = inp;  // GTensor::residual->OverWrite(inp);          //
        if (fuseNorm == nullptr) {
            norm.cuTrain(inp);
        }
        norm.out->Print("ffn.norm", 0, dump_flag, C);
        if (!gate.Empty()) {
            gate.Forw(tGelu, norm.out, up_out);
        }
        // up.dump_flag = dump_flag;
        up.w->Print("ffn.up.w", 0, dump_flag);
        up.Forw(tGelu, norm.out, up_out);
        tGelu->Print("ffn.up+gelu", 0, dump_flag, C);
        hGTensor down_out = GTensor::delta;  // remater_ffn ? GTensor::tmpFF1 : down.out;
        // PrintTensor<floatX>("ffn.norm",ToX(norm.out),true,B,T,C,1,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,1,-1);
        down.Forw(down_out, tGelu, nullptr, isSymmetric);
        down_out->Print("ffn.down", 0, dump_flag, B * T * C);  // PrintTensor<floatX>("ffn",scratch,true,B,T,C);
        if (!hFish->isRemater()) {
            residual_forward(ToX(out), ToX(GTensor::residual), ToX(down_out), B * T * C, main_stream);
            if (fuseNorm != nullptr) {
                return fuseNorm->cuTrain(out);
                // layernorm_forward(ToX(fuseNorm->out), TO<float>(fuseNorm->mean),TO<float>(fuseNorm->rstd), ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T,
                // 1, C, main_stream); return fuseNorm->out;
            }
            out->Print("ffn.out", 0, dump_flag);
            return out;
        } else {
            // inp->Print("ffn.in",0,-1);
        }
    } else {
        SelfAttention *lastQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", layer - 1);
        dump_flag              = 0;
        assert(hIn == GTensor::delta);
        norm.out->Print("ffn.norm", 0, dump_flag);
        if (remater_ffn) {
            up.Forw(tGelu, norm.out, up_out);
        } else {  // gelu just inplace operation on ff1, maybe could share memory!
            gelu_forward(ToX(tGelu), ToX(up_out), up_out->size(), main_stream);
        }
        up_out->Print("ffn.up", 0, dump_flag);
        tGelu->Print("ffn.up+gelu", 0, dump_flag, up_out->size());  // GTensor::delta->Print("ffn.down.delta",0,dump_flag);

        down.Back(GTensor::bt4c, tGelu, GTensor::delta, up_out);
        GTensor::delta->Print("ffn.up.delta", 0, dump_flag);
        // hGensor tmpDelta = GTensor::FromBuffer();
        up.Back(GTensor::tmpDelta, norm.out, GTensor::bt4c, nullptr);
        // // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        // norm.cuTrain(residual,scratchF,tmpDelta);
        float *_mean = norm.mean == nullptr ? nullptr : TO<float>(norm.mean);
        layernorm_backward(ToX(GTensor::delta), ToG(norm.w), ToG0(norm.b), (float *)GTensor::buff, ToX(GTensor::tmpDelta), ToX(inp), ToX(norm.w), _mean,
                           TO<float>(norm.rstd), B, T, C, main_stream);
        if (dump_flag == -1) {  //    only for debug
            GTensor::tmpDelta->Print("deltaIn", 0, dump_flag);
            inp->Print("ffn.norm.inp", 0, dump_flag);
            norm.w->Print("ffn.norm.w", 1, dump_flag);
            GTensor::delta->Print("back of ffn0", 0, dump_flag);
            lastQKV->Q.w->Print("Qw1", 1, dump_flag);  //  0x7ffe5bc00000
            dump_flag = 0;
        }
        return GTensor::delta;
    }
    return nullptr;
}

/*
    layernorm_forward(floatX* out, float* mean, float* rstd, floatX* inp, const floatX* weight, const floatX* bias,         int B, int T, int C, cudaStream_t
   stream) layernorm_backwar(floatX* dinp, floatX* dweight, floatX* dbias, float* scratch,const floatX* dout, const floatX* inp, const floatX* weight, const
   float* mean, const float* rstd,          int B, int T, int C, cudaStream_t stream)
*/
hGTensor huTensor::Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward, int flag) {
    assert(!hOut->isEmpty());
    int B = hOut->ne[0], T = hOut->ne[1], C = w->ne[0];
    // assert(b!=nullptr);
    floatX *weight = (floatX *)(w->data), *bias = ToX0(b);  // b==nullptr?nullptr:(floatX*)(b->data);
    floatX *out = (floatX *)(hOut->data);                   // (B, T, C)
    if (isForward)
        layernorm_forward(out, (float *)_mean->data, (float *)_rstd->data, (floatX *)data, weight, bias, B, T, C, main_stream);
    else {
        layernorm_backward(nullptr, (floatX *)(w->grad), ToG0(b), nullptr, nullptr, nullptr, weight, (float *)_mean->data, (float *)_rstd->data, B, T, C,
                           main_stream);
    }

    return hOut;
}

hGTensor LayerNormal::cuTrain(hGTensor inpDelta, int flag) {  //,hGTensor deltaIn
    NVTX_RANGE_FN();
    float *_mean = mean == nullptr ? nullptr : TO<float>(mean), *_rstd = TO<float>(rstd);
    if (isForward()) {
        inp = inpDelta;
        // layernorm_forward(ToX(out), _mean,  TO<float>(rstd), ToX(inp),ToX(w),ToX0(b), B, T, C, main_stream);
        floatX *weight = ToX(w), *bias = ToX0(b), *in = ToX(inp);
        const int block_size = 256, N = B * T;
        int block_y         = block_size / WARP_SIZE;
        const int grid_size = CEIL_DIV(N, block_y);
        size_t smem         = (2 + block_y) * C * sizeof(floatX);
        // in order to use more than 48 KiB of smem, need to call cudaFuncSetAttribute
        // this may fail, in which case we fall back to the smem free implementation.
        if (mean == nullptr) {  // RMS
            // auto status = cudaFuncSetAttribute(CU_rms_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            // cudaCheck(cudaGetLastError());            assert(status == cudaSuccess);
            // scale = CU_rmsnorm<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(TOBF(out), TOBF(inp), weight, N*C, 1.0e-5, false);
            // _rstd[0] = scale;
            CU_rms_forward<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _rstd, in, weight, N, C);

        } else {
            auto status = cudaFuncSetAttribute(layernorm_forward_kernel6, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            cudaCheck(cudaGetLastError());
            if (status == cudaSuccess) {
                layernorm_forward_kernel6<<<grid_size, dim3(WARP_SIZE, block_y), smem, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, N, C);
            } else {
                // fall back to the version without shared memory
                const int grid_size_fb = CEIL_DIV(N, (block_size / WARP_SIZE));
                layernorm_forward_kernel3<<<grid_size_fb, block_size, 0, main_stream>>>(ToX(out), _mean, _rstd, in, weight, bias, N, C);
            }
        }
        cudaCheck(cudaGetLastError());
        return out;
    } else {
        hGTensor deltaIn = inpDelta;
        assert(deltaIn != nullptr);
        float *scratch = (float *)GTensor::buff;
        deltaIn->Print("LN.delta.in", 0, 0);
        layernorm_backward(ToX(delta), ToG(w), ToG0(b), scratch, ToX(deltaIn), ToX(inp), ToX(w), _mean, _rstd, B, T, C, main_stream);
        delta->Print("back of normal", 0, 0);
        return delta;
    }
}

hGTensor OutSimilarity::cuTrain(hGTensor inp, int flag) { return nullptr; }

/*
    Each block for one token
    todo - 1.   fuse CU_mm_ &replace MM with dot-function
*/
__global__ static void CU_classifier_(floatX *logits_BT, float *losses, floatX *probs, const float dloss, const int *targets, int B, int T, int V, int P,
                                      float *metric, bool WriteDLogits = true) {
    // int64_t idx = gridDim.x - (blockIdx.x + 1);
    int idx = blockIdx.x, tid = threadIdx.x, token = targets[idx];
    bool WriteProbs = probs != nullptr;
    floatX *logits  = logits_BT + (size_t)(idx)*P;

    // SoftmaxParams sp = CU_prepare_softmax(logits, V);    //
    float thread_maxval = -INFINITY, thread_sumval = 0.0f, sum = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        float v       = (float)logits[i];
        thread_maxval = fmaxf(thread_maxval, v);
    }
    float block_maxval = blockReduce_v0<warpReduceMax>(thread_maxval, false, -INFINITY);
    for (int i = tid; i < V; i += blockDim.x) {
        float v = (float)logits[i];
        thread_sumval += expf(v - block_maxval);
    }
    float partition = blockReduce_v0<warpReduceSum>(thread_sumval);  //  canonical partition function
                                                                     // calculate the probability needed for the loss and update (single-threaded)
    __shared__ float pAt;                                            // prob of target token
    if (threadIdx.x == 0) {
        pAt = expf((float)logits[token] - block_maxval) / partition;
        losses[idx] -= logf(pAt);
        // metric[METRIC::LOSS] += losses[idx], metric[METRIC::PPL] += -losses[idx];
    }
    __syncthreads();
    if (WriteDLogits) {
        for (int i = tid; i < V; i += blockDim.x) {
            float v = (float)logits[i], prob = expf(v - block_maxval) / partition;
            // prob            = i == token ? pAt : (1.0 - pAt) / (V - 1);  //  from cys
            float indicator = (i == token) ? 1.0f : 0.0f;
            float dlogit    = (prob - indicator) * dloss;
            logits[i]       = (floatX)dlogit;
        }
    }
    __syncthreads();
}

hGTensor OutCLS::cuTrain(hGTensor inp_, int flag) {
    int V = nCls, Vp = padded_nCls, gelu_fusion = 1, i;
    assert(proj.b == nullptr);
    mean_loss          = 0.0f;
    const int *targets = (int *)(target->data);

    float *cuLoss = (float *)out->data;
    hGTensor cur = preLogits, w = proj.w;
    float alpha4g = 1.0, beta4g = 1.0, logprob = 0;  //  rLoss = 1.0f / (B * T)
    if (isForward()) {
        inp = inp_;
        if (maec != nullptr) {
            inp = maec->DEC(inp, true);
            C   = inp->ne[2];
        }
        floatX *z0 = ToX(inp), *to_gelu = nullptr;  //* errLogits = ToX(preLogits),
        cudaCheck(cudaMemset(cuLoss, 0, B * T * sizeof(float)));
        assert(target->isSameShape(out));
        bool isBack = ToG0(w) != nullptr && delta != nullptr && !hFish->isAtPhase(LIFE_PHASE::P_EVAL_), write_dlogits = isBack;
        // target->Print("oucls.target", 1, -1), isBack = false, write_dlogits = false;  //   Debug_PPL
        w->Print("oucls.proj.w", 1, dump_flag);
        for (i = 0; i < B; i += dB) {
            size_t off = i * T * Vp, n1 = i * T, nZ = i * T * C;
            off = 0;  // reduce memory
            // PrintTensor<floatX>("OutCLS.proj.w", w->GetDataX(), true, w->ne[0], w->ne[1], w->ne[2], w->ne[3], -1);
            // [50304,768] x [768,8192] => [50304,8192]
            CU_mm_(ToX(cur) + off, w, z0 + nZ, NULL, Vp, dB * T, C, main_stream, true, false, false);
            // SYNC_DEVICE();
            if (DEBUG.T_classifier_ver == 1) {
                CU_classifier_<<<dB * T, 1024, 0, main_stream>>>(ToX(cur) + off, cuLoss + n1, nullptr, rLoss, targets + n1, dB, T, V, Vp, dev_metric,
                                                                 write_dlogits);
            } else
                fused_classifier_kernel5<<<dB * T, 1024, 0, main_stream>>>(ToX(cur) + off, cuLoss + n1, nullptr, rLoss, targets + n1, dB, T, V, Vp,
                                                                           write_dlogits);
            if (isBack) {  //  back of delta & grad
                CU_mm_(ToX(delta) + nZ, w, ToX(cur) + off, NULL, C, dB * T, Vp, main_stream, 0, 0, 0, gelu_fusion >= 2 ? to_gelu : NULL, true);
                CU_mm_blas(ToG(w), z0 + nZ, ToX(cur) + off, NULL, C, Vp, dB * T, main_stream, false, true, alpha4g, beta4g /* accumulate */, NULL, true);
            }
        }
        // fused_classifier(errLogits, cuLoss, rLoss, targets, B, T, V, Vp, write_dlogits, main_stream);        //target=[32,1024]
        cudaMemcpy(hostLoss, cuLoss, B * T * sizeof(float), cudaMemcpyDeviceToHost);
        if (!SYNC_DEVICE("OutCLS", 1)) {
            assert(0);
            exit(KOIFISH_EXIT_OUT_CLS);
        }
        for (logprob = 0, i = 0; i < B * T; i++) {
            assert(!std::isnan(hostLoss[i]));
            mean_loss += hostLoss[i];
            logprob += -hostLoss[i];
        }
        mean_loss /= B * T;
        float ppl = exp(-logprob / (B * T));    //  just exp(mean_loss)
        hLoader->iiLoss.Add(mean_loss);
        hLoader->iiPPL.Add(ppl);
    } else {
        // matmul_backward(errOut, gw, NULL, errLogits, z0, w, NULL, B, T, C, Vp, main_stream);
        if (maec != nullptr) {
            cur = maec->DEC(delta, false);
            return cur;
        } else
            return delta;
    }
    cudaCheck(cudaGetLastError());
    return preLogits;
}
