
//        //ugly  "__builtin_ia32_ldtilecfg" is undefined
#include "../../Manifold/Fish.hpp"
#include "../../Manifold/Neuron.hpp"
#include "../../Manifold/Optimizer.hpp"
#include "./cuda_common.h"
#include "./kernel/embed.cuh"
#include "./kernel/fused_classifier.cuh"
#include "./kernel/gelu.cuh"
#include "./kernel/layernorm.cuh"
#include "./kernel/operator.cuh"
#define NOMINMAX

cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F;
// Hardcoding workspace to 32MiB but only Hopper needs 32 (for others 4 is OK)
const size_t cublaslt_workspace_size = 32 * 1024 * 1024;
cublasLtHandle_t cublaslt_handle     = nullptr;
cublasHandle_t cublas_handle         = nullptr;
void* cublaslt_workspace             = nullptr;
cudaStream_t main_stream             = nullptr;
cudaDeviceProp deviceProp;

hGTensor huTensor::_Multiply(const hGTensor& b) {
    huTensor* cuB = dynamic_cast<huTensor*>(b.get());
    assert(cuB != nullptr);
    return nullptr;
}

bool TokenEmbed::UpdateBucket(int type, int flag) {
    num_c_groups = CEIL_DIV(latent, (WARP_SIZE * X128::size));
    assert(num_c_groups > 0);
    if (bucket_info != NULL)
        return false;

    assert((size_t)(B * T) * num_c_groups < (1ULL << 31ULL));  // todo - maybe an issue for llama3-400B(?)
    workload_indices = new int[B * T * num_c_groups];
    bucket_info      = new int4[B * T * num_c_groups];
    return true;
}
void TokenEmbed::WorkloadOnBucker(int* inputs_cpu, int flag) {
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
              [](const std::pair<uint64_t, std::vector<uint64_t>>& a, const std::pair<uint64_t, std::vector<uint64_t>>& b) {
                  return a.second.size() > b.second.size();
              });

    num_buckets        = buckets.size();
    int bucket_index   = 0;
    int workload_index = 0;
    for (const auto& bucket : sortedBuckets) {
        bucket_info[bucket_index].x = workload_index;                                       // bucket start
        bucket_info[bucket_index].y = bucket.second.size();                                 // bucket size
        bucket_info[bucket_index].z = (bucket.second[0] >> 42ULL) & ((1ULL << 20ULL) - 1);  // bucket ix
        bucket_info[bucket_index].w = (bucket.second[0] >> 32ULL) & ((1ULL << 10ULL) - 1);  // bucket c

        for (uint64_t idx : bucket.second) {
            workload_indices[workload_index++] = (int)(idx & ((1ULL << 31ULL) - 1ULL));
        }
        bucket_index++;
    }

    floatX* scratch = (floatX*)GTensor::buff;
    // Step 3: Copy data from host to device (async until the last one to avoid synchronising CPU/GPU twice)
    // todo - could use CUDA events (even without streams) to avoid CPU/GPU synchronisation completely
    int4* d_bucket_info     = (int4*)scratch;
    int* d_workload_indices = (int*)(scratch + B * T * num_c_groups * sizeof(int4));
    cudaCheck(cudaMemcpyAsync(d_bucket_info, bucket_info, num_buckets * sizeof(int4), cudaMemcpyHostToDevice, main_stream));
    cudaCheck(cudaMemcpyAsync(d_workload_indices, workload_indices, total_items * sizeof(int), cudaMemcpyHostToDevice, main_stream));
}

//
hGTensor TokenEmbed::cuInfer(hGTensor hOut, int flag) {
    int token = hBatch->CurToken(), pos = hBatch->tok_pos, dim = latent, nRow = w->shape[0];
    hQUANT hQuant = w->GetDynamicQuant();
    assert(dim % 32 == 0);
    switch (w->type) {
        case typNUMBER::Q4:
            if (hQuant->params.isNormalFloat) {
                CU_embed_forw_nf4<<<dim / 32, 16, 0, main_stream>>>(w->gama_T(), hBITARR(w->data), ToX(hOut), token, nRow, dim, 0, 42);  // 151644
            } else
                CU_embed_forw_q4<<<dim / 32, 16, 0, main_stream>>>(w->gama_T(), hBITARR(w->data), ToX(hOut), token, nRow, dim, 0, 42);
            break;
        case typNUMBER::Q3:
            assert(0);
            // if(hQuant->params.isNormalFloat){
            //     CU_embed_forw_nf3<<<dim / 32, 16, 0, main_stream>>>(w->gama_T(), hBITARR(w->data), ToX(hOut), token, nRow, dim, 0, 42); //151644
            // }else
            //     CU_embed_forw_q3<<<dim / 32, 16, 0, main_stream>>>(w->gama_T(), hBITARR(w->data), ToX(hOut), token, nRow, dim, 0, 42);
            break;
        case typNUMBER::F8E5M2:
            CU_embed_forw_1<<<dim / 32, 32, 0, main_stream>>>(ToX(hOut), TO<f8e5>(w), token, dim, 42);
            break;
        case typNUMBER::BF16:
            CU_embed_forw_1<<<dim / 32, 32, 0, main_stream>>>(ToX(hOut), TO<bf16>(w), token, dim, 42);
            break;
        default:
            assert(0);
            break;
    }
    // w->Print("wte", 0, -1);
    // hOut->Print("token_embed", 0, -1);
    return hOut;
}

hGTensor TokenEmbed::OnEmbed(hGensor inpL, int seed) {
    try {
        int OC = w->ne[1], Vp = padded_nCls, C = hFish->config.nEmbed();
        int nToken   = nBatchToken();
        hGTensor cur = out, curW = w;
        if (isForward()) {
            if (hFish->isAtPhase(LIFE_PHASE::P_GENERATE)) {
                int token = hBatch->CurToken();
                CU_embed_forw_1<<<C / 32, 32, 0, main_stream>>>(ToX(cur), ToX(curW), token, C, 0);
            } else {
                inp = inpL;
                inp->Print("token_input", 0, 0);
                if (w->type == typNUMBER::T_BINARY_3) {
                    curW->Print("curW", 0, 0);
                    CU_embed_ternary_forw_<floatX>
                        <<<CEIL_DIV(B * T, block_size), block_size, 0, main_stream>>>(ToX(cur), TO<int>(inp), curW->gama_T(), TO<char>(curW), ToX0(b), B, T, C);
                } else
                    CU_embed_forw_<<<CEIL_DIV(B * T, block_size), block_size, 0, main_stream>>>(ToX(cur), TO<int>(inp), ToX(curW), ToX0(b), B, T, C);
                // w->Print("wte", 0, 0);
                if (b != nullptr)
                    PrintTensor<floatX>("wpe", ToX0(b), true, T, C);
            }
            // PrintTensor<int>("inputs",tokens,true,B,T);
            cur->Print("token_embed", 0, 0, nToken * C);
            if (maec != nullptr) {
                maec->ENC(cur);
            }
        } else {
            UpdateBucket(0x0);
            WorkloadOnBucker(hBatch->host, 0x0);
            floatX* scratchX = (floatX*)GTensor::buff;
            hGTensor delta = gBUFF->delta, cur = delta;
            if (maec != nullptr) {
                cur = maec->ENC(cur);
            }
            // encoder_backward_1(ToG(w), ToG0(b), ToX(cur), tokens, B, T, C, seed, main_stream);
            encoder_backward(ToG(w), ToG0(b), scratchX, workload_indices, bucket_info, ToX(cur), TO<int>(inp), hBatch->host, B, T, C, seed, main_stream);
            // lnW.cuFlow();   lnf->cuFlow(ToG(w),(float*)scratchX,delta);
            // PrintTensor<floatX>("grad of wte",grads.wte,true,Vp,C);         PrintTensor<float>("losses",acts.losses,true,B,T);
            // PrintTensor<floatX>("grad of wpe",grads.wpe,true,T,C);
        }
        return cur;
    } catch (...) {
        bool isF = isForward();
        assert(0);
        return nullptr;
    }
}

//  seed - use stochastic rounding to go from FP32 to BF16
hGTensor TokenEmbed::SubW(hGTensor hSamp, bool isForw, hGTensor wOut, int flag) {
    try {
        int nSamp = hSamp->size(), *samps = TO<int>(hSamp), nLayer = hFish->config.nLayer();
        int OC = w->ne[1], Vp = padded_nCls, T = nSamp, B = 1;  //  , seed = 42
        grid_size    = CEIL_DIV(B * T * latent, block_size);
        hGTensor cur = wOut, wSrc = flag == 0 ? w : wInv;

        if (isForw) {
            // encoder_forward(ToX(cur), samps, ToX(wSrc), nullptr, 1, T, C, main_stream);
            CU_embed_forw_tc<<<grid_size, block_size, 0, main_stream>>>(ToX(cur), samps, ToX(wSrc), T, latent, Vp, flag == 1);
            cur->Print("subW", 0, 0);
        } else {
            CU_embed_back_<<<grid_size, block_size, 0, main_stream>>>(ToG(wSrc), samps, ToX(cur), T, latent, Vp, flag == 1);
            // encoder_backward(ToG(wSrc), nullptr, scratchX, workload_indices, bucket_info, ToX(cur), samps, hBatch->host, 1, T, C, seed, main_stream);
        }
        return cur;
    } catch (...) {
        assert(0);
        return nullptr;
    }
}

//  W'=b(:,rank)*a(rank,:)  => rhs = b*(a*lhs)
int HIERARCH_LoRA::Forw(floatX* rhs, floatX* lhs, int BT, int flag) {
    int ldA = a->ne[0], ldB = b->ne[1], dump_flag = -1;
    // b->Print(b->name, 1, dump_flag);
    CU_mm_((floatX*)Ax, a, lhs, nullptr, rank, BT, ldA, main_stream, 1, 0, 0.0f);
    CU_mm_(rhs, b, (floatX*)Ax, nullptr, ldB, BT, rank, main_stream, 1, 0, beta_F);
    // b->Print(b->name, 1, dump_flag);
    return 0x0;
}

int SLP::Forw(hGTensor rhs_0, hGTensor lhs_, hGTensor toGelu, Relu* hRelu, int flag) {
    NVTX_RANGE_FN();
    try {
        floatX *rhs = ToX(rhs_0), *to_gelu = ToX0(toGelu);
        int OC = nOut, IC = nIn, nToken = nBatchToken();
        assert(gelu_fusion == 0);
        assert(rhs_0->size() >= nToken * OC);
        size_t dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
        inp          = lhs_;
        bool transAW = true;
        // if(isTransW)
        //     transAW = false;
        switch (compression) {
            case SAMPLE:
                assert(subw != nullptr);
                subw->SubW(hSamps, true, gBUFF->tmpW, samp_type);
                // wX = ToX(gBUFF->tmpW);
                // gBUFF->tmpW->Print("subW",0,-1);
                break;
            case LORA:
                break;
            default:
                break;
        }

        rhs  = to_gelu ? to_gelu : rhs;
        tRhs = to_gelu ? toGelu : rhs_0;
        assert(rhs != nullptr);
        switch (compression) {
            case SAMPLE:
                CU_mm_(rhs, w, ToX(lhs_), ToX0(b), OC, nToken, IC, main_stream, true);
                break;
            case LORA:  //  rhs += a*b*lhs_
                if (tpLORA != LORA_ADAPT_W::AB)
                    CU_mm_(rhs, w, ToX(lhs_), ToX0(b), OC, nToken, IC, main_stream, true);
                // rhs_0->Print(rhs_0->name, 0, -1);   //w->Print(w->name, 0, -1),
                if (tpLORA != LORA_ADAPT_W::W0) {
                    for (auto lora : wLORAs) {
                        lora->Forw(rhs, ToX(lhs_), nToken, flag);
                    }
                }
                // rhs_0->Print(rhs_0->name,0,-1);
                break;
            default:
                CU_mm_(rhs, w, ToX(lhs_), ToX0(b), OC, nToken, IC, main_stream, true);
                break;
        }
        // if (hRelu != nullptr) {
        //     // w->Print("ffn.w", 0, dump_flag);
        //     toGelu->Print("ffn.up", 0, dump_flag, OC);
        //     hRelu->Forw(rhs_0, toGelu);
        // }
        return 0x0;
    } catch (...) {
        assert(0);
        return -1;
    }
}

//  Forward: rhs = b*(a*inp)
int HIERARCH_LoRA::Back(hGTensor delta, hGTensor inp, hGTensor deltaIn, int flag) {
    if (!isBack)
        return 0x0;

    int ldA = a->ne[0], ldB = b->ne[1], dump_flag = 0;
    delta->Print("delta_0", 0, dump_flag);
    float* dbias_buffer = (float*)GTensor::buff;
    bool isTransW       = False;
    //  delta->Print(delta->name, 0, dump_flag);
    a->Print(a->name, 0, dump_flag), b->Print(b->name, 0, dump_flag);
    // assert(inp->isSameShape({B, T, ldA}) && deltaIn->isSameShape({B, T, ldB}));
    //  Forward: rhs = b*Ax
    matmul_backward((floatX*)Adelta, ToG(b), nullptr, ToX(deltaIn), (floatX*)Ax, ToX(b), dbias_buffer, B, T, rank, ldB, main_stream, isTransW);
    b->Print(b->name, 1, dump_flag);
    inp->Print("inp", 0, dump_flag), PrintTensor<floatX>("A_delta", (floatX*)Adelta, true, B, T, rank, 1, dump_flag);
    //  Forward: Ax = a*Inp     [64,8192] x [8192,768]=>[64,768]
    matmul_backward(ToX(delta), ToG(a), nullptr, (floatX*)Adelta, ToX(inp), ToX(a), dbias_buffer, B, T, ldA, rank, main_stream, isTransW, nullptr,
                    isAccumuDelta);
    a->Print(a->name, 0, dump_flag), a->Print(a->name, 1, dump_flag);
    delta->Print("delta_1", 0, dump_flag);
    return 0x0;
}

int SLP::Back(hGTensor delta, hGTensor inp, hGTensor deltaIn, hGTensor to_gelu, bool isAccumuDelta, int flag) {
    try {
        size_t szBuf = 0x0;
        w->BeforeBackward(szBuf, true);
        floatX *wX = w->GetDataX(), *gW = ToG(w);
        float* dbias_buffer = (float*)((char*)GTensor::buff + szBuf);

        int OC = nOut, IC = nIn;
        assert(delta != nullptr);
        // assert(inp->isSameShape({B, T, IC}) && deltaIn->isSameShape({B, T, OC}));
        // deltaIn->Print("delta_in", 0, flag);
        switch (compression) {
            case SAMPLE:  // remater to get wX
                subw->SubW(hSamps, true, gBUFF->tmpW, samp_type);
                wX = ToX(gBUFF->tmpW);  // assert(nSample==OC || nSample==IC);
                gW = ToX(gBUFF->tmpGW);
                cudaCheck(cudaMemsetAsync(gW, 0, gBUFF->tmpGW->nByte(), main_stream));
                break;
            default:
                break;
        }

        switch (compression) {
            case SAMPLE:
                matmul_backward(ToX(delta), gW, ToG0(b), ToX(deltaIn), ToX(inp), wX, dbias_buffer, B, T, IC, OC, main_stream, isTransW, ToX0(to_gelu),
                                isAccumuDelta);
                subw->SubW(hSamps, false, gBUFF->tmpGW, samp_type);
                break;
            case LORA:
                if (tpLORA != LORA_ADAPT_W::AB && tpLORA != LORA_ADAPT_W::refW_AB)
                    matmul_backward(ToX(delta), gW, ToG0(b), ToX(deltaIn), ToX(inp), wX, dbias_buffer, B, T, IC, OC, main_stream, isTransW, ToX0(to_gelu),
                                    isAccumuDelta);
                if (tpLORA != LORA_ADAPT_W::W0)
                    for (auto lora : wLORAs) {
                        lora->Back(delta, inp, deltaIn, flag);
                    }
                break;
            default:
                matmul_backward(ToX(delta), gW, ToG0(b), ToX(deltaIn), ToX(inp), wX, dbias_buffer, B, T, IC, OC, main_stream, isTransW, ToX0(to_gelu),
                                isAccumuDelta);
                break;
        }

        if (flag != 0x100 && !isAccumuDelta)
            w->Dogleg(-1);

        return 0x0;
    } catch (...) {
        assert(0);
        return -1;
    }
}

bool SLP::PrepareMemory(bool isBack, int flag) {
    /*assert(isBack);
    int m = w->ne[0], n = w->ne[1];
    if (BIT_TEST(w->flags, GTensor::F_TMP_GRAD)) {
        size_t off = 0;
        w->grad    = (floatX *)GTensor::buff, off += w->szGrad;
        w->BeforeBackward();
        gW           = w->grad;
        dbias_buffer = (float *)(GTensor::buff + off), off += sizeof(float) * m;
        assert(off < GTensor::buff_len);
    } else {
        //
        gW           = ToG(w);
        dbias_buffer = (float *)GTensor::buff;
    }*/

    return true;
}

int Q_nThreadOfBlock(int N, int bit, int nT0 = CU_T4B_BIG) {
    if (bit >= 8)
        return nT0;
    int nT = nT0;
    if ((8 % bit == 0)) {   // bit=4,2,1
        int npb = 8 / bit;  //  number of quants per byte(8bit)
        while (!(N % nT == 0 && (N / nT) % npb == 0)) {
            nT /= 2;
        }
    } else {  // bit=3, 3*8=24bit
        while (!(N % nT == 0 && (N / nT) % 8 == 0)) {
            nT /= 2;
        }
    }
    assert(nT > 1);
    return nT;
}

floatX* GTensor::GetDataX(int flag, const string& sX) {
    size_t dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
    size_t nEle = size();
    floatX* wX  = (floatX*)(data);
    int nRow = ne[0], nCol = ne[1];
    if (hRef != nullptr) {
        nRow = hRef->ne[0], nCol = hRef->ne[1];
        DEBUG_HERE;
    }

    switch (type) {
        case typNUMBER::T_SIGN:
            assert(0);
            break;
        case typNUMBER::Q4: {
            // dT4B = CU_T4B_MIDDLE;
            wX = ToX(gBUFF->tmpTernary);
            assert(gBUFF->tmpTernary->size() >= nEle);
            dT4B = Q_nThreadOfBlock(nCol, 4);
            if (hQuant->isRTN()) {
                if (hQuant->params.isNormalFloat) {
                    CU_Q42X_NF4<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
                } else
                    CU_Q42X_RTN<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            } else
                CU_Q42X_<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            // gBUFF->tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1, nRow*nCol);
        } break;
        case typNUMBER::Q3: {
            wX = ToX(gBUFF->tmpTernary);
            assert(gBUFF->tmpTernary->size() >= nEle);
            dT4B = Q_nThreadOfBlock(nCol, 3);  // 1 byre for 4 quant
            if (hQuant->isRTN()) {
                if (hQuant->params.isNormalFloat) {
                    CU_Q32X_NF3<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
                } else
                    CU_Q32X_RTN<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            } else
                CU_Q32X_<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            // gBUFF->tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1, nRow*nCol);
        } break;
        case typNUMBER::Q2: {
            wX = ToX(gBUFF->tmpTernary);
            assert(gBUFF->tmpTernary->size() >= nEle);
            dT4B = Q_nThreadOfBlock(nCol, 2);  // 1 byre for 4 quant
            if (hQuant->isRTN()) {
                CU_Q22X_RTN<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            } else
                CU_Q22X_<floatX><<<nRow, dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, disq.rc_normal, 42);
            // gBUFF->tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1, nRow*nCol);
        } break;

        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3:
            if (DEBUG.T_ternary == 1) {          // each weight(floatX) is {-1,0,1}, no need to extract
            } else if (DEBUG.T_ternary == -1) {  // each weight {-1,0,1}
                return nullptr;
            } else {  // extract each weight {-1,0,1} to floatX
                wX = ToX(gBUFF->tmpTernary);
                CU_ternary2X_<floatX><<<CEIL_DIV(nRow, dT4B), dT4B, 0, main_stream>>>(gama_T(), hBITARR(data), wX, nRow, nCol, 1);
                SYNC_DEVICE();
                if (flag == -1)
                    gBUFF->tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1);
            }

            // PrintTensor<floatX>("wX", wX, true, nRow, nCol, ne[2], ne[3], -1);
            break;
        case typNUMBER::T_BINARY_TILE: {
            dim3 dBlock(THREAD_TILE_M * THREAD_TILE_N), dGrid(CEIL_DIV(nRow, THREAD_TILE_M), CEIL_DIV(nCol, THREAD_TILE_N));
            assert(nRow % THREAD_TILE_M == 0 && nCol % THREAD_TILE_N == 0);
            wX              = ToX(gBUFF->tmpTernary);
            floatGama* gam_ = gama_T();  //
            CU_Tile2X_<floatX><<<dGrid, dBlock, smemPB, main_stream>>>(wX, gam_, 0.0, nRow, nCol, seed);
            // SYNC_DEVICE();
            if (flag == -1)
                gBUFF->tmpTernary->Print(sX.empty() ? name : sX, 0x0, -1);
        } break;
        case typNUMBER::F8E5M2: {
            wX = ToX(gBUFF->tmpTernary);
            assert(gBUFF->tmpTernary->size() >= nEle);
            CU_F82Float<floatX><<<CEIL_DIV(nEle, CU_T4B_MIDDLE), CU_T4B_MIDDLE>>>((f8e5*)data, wX, nEle, 0, 0);
            break;
        }
        default:
            break;
    }

    return wX;
}

int SLP::FUSE_cuda_block(hGTensor rhs, hGTensor lhs, hGTensor gelu, bool isForw, int flag) { return 0x0; }

hGTensor FFN::cuInfer(hGTensor hIn, int flag) {
    hGTensor tGelu = gBUFF->scratch, up_out = remater_ffn ? gBUFF->tmpFF1 : up.out;
    bool isBias = up.b != nullptr;
    int nToken = nBatchToken(), nEmbed = hFish->config.nEmbed(), C = nEmbed;
    inp = OnInput(hIn);
    // inp->Print("ffn.in", 0, dump_flag, C);
    gBUFF->residual = inp;  // gBUFF->residual->OverWrite(inp);          //
    if (fuseNorm == nullptr) {
        norm.cuFlow(inp);
    }
    norm.out->Print("ffn.norm", 0, dump_flag, C);
    if (!gate.Empty()) {
        gate.Forw(tGelu, norm.out, nullptr);
        // gate.w->Print("ffn.gate.w", 0, dump_flag);
        tGelu->Print("ffn.gate", 0, dump_flag, nToken * latent);
    }
    // up.dump_flag = dump_flag;
    // up.w->Print("ffn.up.w", 0, dump_flag);
    up.Forw(tGelu, norm.out, up_out, &relu);
    relu.Forw(tGelu, up_out);
    tGelu->Print("ffn.up+gelu", 0, dump_flag, latent);
    hGTensor down_out = gBUFF->delta;
    // PrintTensor<floatX>("ffn.norm",ToX(norm.out),true,B,T,C,1,-1);          PrintTensor<floatX>("ff1",ff1,true,B,T,latent,1,-1);
    down.Forw(down_out, tGelu, nullptr, nullptr, isSymmetric);
    down_out->Print("ffn.down", 0, dump_flag, nToken * C);  // PrintTensor<floatX>("ffn",scratch,true,B,T,C);
    if (!hFish->isRemater()) {
        TASKA_1p1<floatX> task_11(nToken * C, main_stream);
        T1p1(CU_add3<floatX>, task_11, ToX(out), ToX(gBUFF->residual), ToX(down_out), 0x0);
        // residual_forward(ToX(out), ToX(gBUFF->residual), ToX(down_out), nToken * C, main_stream);
        if (fuseNorm != nullptr) {
            return fuseNorm->cuFlow(out);
            // layernorm_forward(ToX(fuseNorm->out), TO<float>(fuseNorm->mean),TO<float>(fuseNorm->rstd), ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T,
            // 1, C, main_stream); return fuseNorm->out;
        }
        out->Print("ffn.out", 0, 0, nToken * C);
        return out;
    } else {
        // inp->Print("ffn.in",0,-1);
    }
    return nullptr;
}

template <typename T>
void CU_set(const char* title, T* dst, int n1, int n2, int n3 = 1, int n4 = 1, int flag = 0x0) {
    size_t nElem = n1 * n2 * n3 * n4, i, nz = 0, nEach = 3;
    if (nElem == 0)
        return;
    T *src = new T[nElem], *cur = src;
    char info[1028] = "\0";
    sprintf(info + strlen(info), ">>>> %s\t", title);
    float a1 = -FLT_MAX, a0 = FLT_MAX, a;
    double sum = 0.0, len = 0.0, sum2 = 0.0;
    for (i = 0; i < nElem; i++, cur++) {
        a    = (i % 2 + 1) * 1.0e-6f;
        *cur = a;
        if (i < nEach || i >= nElem - nEach || fabs(i - nElem / 2) <= nEach)
            sprintf(info + strlen(info), "%g ", a);
        if (i == nEach || i == nElem - nEach)
            sprintf(info + strlen(info), "...");
        sum += fabs(a);
        sum2 += a * a;
        if (a == 0)
            nz++;
        a1 = std::max(a1, a);
        a0 = std::min(a0, a);
    }
    len = sqrt(sum2 / nElem);
    sprintf(info + strlen(info), " |avg|=%g(%ld) avg_len=%g sum2=%g [%f,%f] >>>>\n", sum / nElem, nElem, len, sum2, a0, a1);
    fputs(info, stderr);
    fflush(stderr);

    cudaCheck(cudaMemcpyAsync(dst, src, nElem * sizeof(T), cudaMemcpyHostToDevice));
    delete[] src;
}

//  hIn = QKV->out
hGTensor FFN::cuFlow(hGTensor hIn, int flag) {
    hGTensor tGelu = gBUFF->scratch, up_out = remater_ffn ? gBUFF->tmpFF1 : up.out;
    bool isBias = up.b != nullptr;
    int C       = hFish->config.nEmbed();
    if (isForward()) {
        inp = OnInput(hIn);
        // inp->Print("ffn.in", 0, dump_flag, B * T * C);
        gBUFF->residual = inp;  // gBUFF->residual->OverWrite(inp);          //
        if (fuseNorm == nullptr) {
            norm.w->Print("ffn.norm.w", 0, dump_flag, C);
            norm.cuFlow(inp);
            norm.out->Print("ffn.norm", 0, dump_flag, B * T * C);
        }
        if (!gate.Empty()) {
            gate.Forw(tGelu, norm.out, nullptr);
        }
        // up.w->Print("ffn.up.w", 0, dump_flag);
        up.Forw(up_out, norm.out);  // up.Forw(tGelu, norm.out, up_out, &relu);
        up_out->Print("ffn.up", 0, dump_flag, latent);
        relu.Forw(tGelu, up_out);

        hGTensor down_out = gBUFF->delta;

        down.Forw(down_out, tGelu, nullptr, nullptr, isSymmetric);
        down_out->Print("ffn.down", 0, dump_flag, B * T * C);  // PrintTensor<floatX>("ffn",scratch,true,B,T,C);
        // GTensor::tZ->Print(GTensor::tZ->name, 0, dump_flag);
        if (!hFish->isRemater()) {
            TASKA_1p1<floatX> task_11(B * T * C, main_stream);
            T1p1(CU_add3<floatX>, task_11, ToX(out), ToX(gBUFF->residual), ToX(down_out), 0x0);
            // residual_forward(ToX(out), ToX(gBUFF->residual), ToX(down_out), B * T * C, main_stream);
            if (fuseNorm != nullptr) {
                return fuseNorm->cuFlow(out);
                // layernorm_forward(ToX(fuseNorm->out), TO<float>(fuseNorm->mean),TO<float>(fuseNorm->rstd), ToX(out),ToX(fuseNorm->w), ToX0(fuseNorm->b), B*T,
                // 1, C, main_stream); return fuseNorm->out;
            }
            // out->Print("ffn.out", 0, dump_flag);
            return out;
        } else {
            // inp->Print("ffn.in",0,-1);
        }
    } else {
        SelfAttention* lastQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", layid - 1);
        assert(hIn == gBUFF->delta);
        // norm.out->Print("ffn.norm", 0, dump_flag);
        if (remater_ffn) {
            up.Forw(up_out, norm.out);  // up.Forw(tGelu, norm.out, up_out, &relu);
            if (!gate.Empty()) {
                gate.Forw(tGelu, norm.out, nullptr);
            }
            relu.Forw(tGelu, up_out);
        }
        INSPECT inspect(this);
        gBUFF->delta->Print("ffn.down.delta", 0, dump_flag);
        down.Back(gBUFF->bt4c, tGelu, gBUFF->delta, up_out);
        // CU_set<nv_bfloat16>("dSwigLU", ToX(gBUFF->bt4c), B, T, latent, 1, dump_flag);
        gBUFF->bt4c->Print("dSwigLU", 0, dump_flag, B * T * latent);
        // up_out->Print("ffn.up", 0, dump_flag, B * T * latent);
        assert(!relu.Empty());
        if (!gate.Empty()) {  // ugly code, need refactor!
            gate.Forw(tGelu, norm.out, nullptr);
            // tGelu->Print("swig.gate", 0, -1, latent);
        }
        relu.Back(gBUFF->bt4c, up_out);
        // hGensor tmpDelta = GTensor::FromBuffer();
        up.Back(gBUFF->tmpDelta, norm.out, gBUFF->bt4c, nullptr);
        if (!gate.Empty()) {
            gate.Back(gBUFF->tmpDelta, norm.out, gate.delta, nullptr, true);
        }
        gBUFF->tmpDelta->Print("norm.delta", 0, dump_flag);
        // // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        // norm.cuFlow(residual,scratchF,tmpDelta);
        float* _mean = norm.mean == nullptr ? nullptr : TO<float>(norm.mean);
        layernorm_backward(ToX(gBUFF->delta), ToG(norm.w), ToG0(norm.b), (float*)GTensor::buff, ToX(gBUFF->tmpDelta), ToX(inp), ToX(norm.w), _mean,
                           TO<float>(norm.rstd), B, T, C, main_stream);
        if (dump_flag == -1) {  //    only for debug
            gBUFF->tmpDelta->Print("deltaIn", 0, dump_flag);
            inp->Print("ffn.norm.inp", 0, dump_flag);
            norm.w->Print("ffn.norm.w", 1, dump_flag);
            gBUFF->delta->Print("back of ffn0", 0, dump_flag);
            lastQKV->Q.w->Print("Qw1", 1, dump_flag);  //  0x7ffe5bc00000
            dump_flag = 0;
        }
        // CU_set<nv_bfloat16>("delta", ToX(gBUFF->delta), B, T, C, 1, dump_flag);
        return gBUFF->delta;
    }
    return nullptr;
}

hGTensor huTensor::Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward, int flag) {
    assert(0);  //  need refactor
    /*assert(!hOut->isEmpty());
    int B = hOut->ne[0], T = hOut->ne[1], C = w->ne[0];
    // assert(b!=nullptr);
    floatX *weight = (floatX*)(w->data), *bias = ToX0(b);  // b==nullptr?nullptr:(floatX*)(b->data);
    floatX* out = (floatX*)(hOut->data);                   // (B, T, C)
    if (isForward)
        layernorm_forward(out, (float*)_mean->data, (float*)_rstd->data, (floatX*)data, weight, bias, B, T, C, main_stream);
    else {
        layernorm_backward(nullptr, (floatX*)(w->grad), ToG0(b), nullptr, nullptr, nullptr, weight, (float*)_mean->data, (float*)_rstd->data, B, T, C,
                           main_stream);
    }*/

    return hOut;
}

hGTensor OutSimilarity::cuFlow(hGTensor inp, int flag) { return nullptr; }

/*
    Each block for one token
    todo - 1.   fuse CU_mm_ &replace MM with dot-function
*/
__global__ static void CU_classifier_(floatX* logits_BT, float* losses, floatX* probs, const float dloss, const int* targets, int B, int T, int V, int P,
                                      float* metric, bool WriteDLogits = true) {
    // int64_t idx = gridDim.x - (blockIdx.x + 1);
    int idx = blockIdx.x, tid = threadIdx.x, token = targets[idx];
    bool WriteProbs = probs != nullptr;
    floatX* logits  = logits_BT + (size_t)(idx)*P;

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

hGTensor OutCLS::cuInfer(hGTensor inp_, int flag) {
    double now = GST_us();
    assert(norm.Empty());
    inp_->Print("cls.inp_", 0, dump_flag);

    proj.Forw(preLogits, inp_);
    SUM::GPU_TIME(SUM::tPreLogits, now);
    preLogits->Print("logits", 0, dump_flag, nCls);

    if (hFish->config.chat_sampler.isSampleCPU)
        preLogits->SerialData("", nullptr, true);
    else {
        ;  // preLogits->Print("preLogits",0, -1);
    }

    return preLogits;
}

hGTensor OutCLS::cuFlow(hGTensor inp_, int flag) {
    INSPECT inspect(this);
    int V = nCls, Vp = padded_nCls, gelu_fusion = 1, i, C = hFish->config.nEmbed();
    assert(proj.b == nullptr);
    mean_loss          = 0.0f;
    const int* targets = (int*)(target->data);
    float* cuLoss      = (float*)out->data;
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
        target->Print("oucls.target", 0, dump_flag);
        // w->Print("oucls.proj.w", 1, dump_flag);
        for (i = 0; i < B; i += dB) {
            size_t off = i * T * Vp, n1 = i * T, nZ = i * T * C;
            off = 0;  // reduce memory
            // PrintTensor<floatX>("OutCLS.proj.w", w->GetDataX(), true, w->ne[0], w->ne[1], w->ne[2], w->ne[3], -1);
            // [50304,768] x [768,8192] => [50304,8192]
            hGensor subZ = inp->Partial("partialZ", nZ, {dB, T, C}), subDelta = delta->Partial("partialDeltaZ", nZ, {dB, T, C});
            proj.Forw(cur, subZ);
            // cur->Print("preLogist",0, dump_flag);
            // SYNC_DEVICE();
            if (DEBUG.T_classifier_ver == 1) {
                CU_classifier_<<<dB * T, 1024, 0, main_stream>>>(ToX(cur) + off, cuLoss + n1, nullptr, rLoss, targets + n1, dB, T, V, Vp, dev_metric,
                                                                 write_dlogits);
            } else
                fused_classifier_kernel5<<<dB * T, 1024, 0, main_stream>>>(ToX(cur) + off, cuLoss + n1, nullptr, rLoss, targets + n1, dB, T, V, Vp,
                                                                           write_dlogits);
            if (isBack) {
                PrintTensor<float>("loss", cuLoss + n1, true, dB, T, 1, 1, dump_flag);
                // cur->Print("oucls.delta", 0, dump_flag);
                proj.Back(subDelta, subZ, cur, nullptr, false, 0x100);  // hGTensor delta, hGTensor inp, hGTensor deltaIn
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
        float ppl = exp(-logprob / (B * T));  //  just exp(mean_loss)
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

int Relu::Forw(hGTensor out, hGTensor inp, int flag) {
    size_t nz            = SHAPE2NZ(shape);
    const int block_size = 128;
    const int grid_size = CEIL_DIV(nz, block_size), C = hFish->config.n_ff();
    assert(B * T * C == nz);
    hGTensor gate = nullptr;
    switch (fAct) {
        case SWIG:
            assert(slp_gate != nullptr && slp_gate->tRhs != nullptr);
            gate = slp_gate->tRhs;
            gate->Print("swig.gate", 0, dump_flag, C);
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                assert(C % X128::size == 0);
                assert((nz) % (block_size * X128::size) == 0);
                const int num_blocks = CEIL_DIV(nz, (int)(block_size * X128::size));
                assert(gate != nullptr);
                // CU_swiglu_v1<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), C);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            }
            out->Print("ffn.swig", 0, dump_flag, nz);
            break;
        case GELU:
            gelu_forward(ToX(out), ToX(inp), nz, main_stream);
            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

template <typename T>
__global__ static void CU_swiglu_back_v0(T* delta_in_out, T* delta_gate, const T* gate, const T* inp, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xiW   = CU_T2Float(gate + idx);
        float xiV   = CU_T2Float(inp + idx);
        float delta = delta_in_out[idx];
        // if(idx==0)    // only for debug
        // {    nout("nout<%d>: gate=%g ffn.up=%g delta=%g\n", idx, xiW,xiV,delta);    }
        float sigW      = 1.0f / (1.0f + expf(-xiW));
        delta_gate[idx] = delta * xiV * sigW * (1 + xiW * (1.0f - sigW));

        delta_in_out[idx] = delta * xiW * sigW;  //  delta * swish_out[i];
    }
}

//  delta is both delta_in & delta_out
int Relu::Back(hGTensor delta_in_out, hGTensor pre_gelu, int flag) {
    size_t nz            = SHAPE2NZ(shape);
    const int block_size = 128, C = hFish->config.n_ff();
    assert(B * T * C == nz);
    const int grid_size = CEIL_DIV(nz, block_size);
    hGTensor gate       = nullptr;
    switch (fAct) {
        case SWIG:
            assert(slp_gate != nullptr);
            gate = slp_gate->tRhs;
            pre_gelu->Print("ffn.up", 0, dump_flag, nz);
            gate->Print("swig.gate", 0, dump_flag, C);  //-1.890625
            // inp->Print("swig.inp", 0, dump_flag, C);
            if (version == 0) {  // CU_swiglu_v0(ToX(out), ToX(out), ToX(inp), nz, main_stream);
                CU_swiglu_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(out), ToX(gate), ToX(inp), nz);
            } else {
                assert(C % X128::size == 0);
                assert(nz % (block_size * X128::size) == 0);
                const int num_blocks = CEIL_DIV(nz, (int)(block_size * X128::size));
                assert(gate != nullptr && slp_gate != nullptr);
                CU_swiglu_back_v0<<<grid_size, block_size, 0, main_stream>>>(ToX(delta_in_out), ToX(slp_gate->delta), ToX(gate), ToX(pre_gelu), nz);
            }
            delta_in_out->Print("dUp", 0, dump_flag, C);
            slp_gate->delta->Print("dGate", 0, dump_flag, C);
            break;
        case GELU:
            //  gelu_backward_inplace fused @matmul_backward
            gelu_backward_inplace(ToX(delta_in_out), ToX(pre_gelu), nz, main_stream);

            break;
        default:
            assert(0);
            break;
    }
    cudaCheck(cudaGetLastError());
    return 0x0;
}

void GeNeuron::SetInp4Back(hGensor inp_, int flag) {
    inp = inp_;
    if (dev_window != nullptr) {
        size_t szCopy = std::min(inp->nByte(), (size_t)CU_DEV_WINDOW);
        D2D(dev_window->data, inp->data, CU_DEV_WINDOW);
    }
}

template <typename T>
__global__ void CU_memcmp(const T* a, const T* b, size_t n, int flag = 0x0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n && a[idx] != b[idx]) {
        assert(0);
        // atomicAdd(nMiss, 1);
    }
}
bool GeNeuron::VerifyInp4Back(hGensor inp_, int flag) {
    if (dev_window != nullptr) {
        // int nMiss = 0;
        CU_memcmp<BIT_8><<<1, CU_DEV_WINDOW>>>((hBITARR)(dev_window->data), (hBITARR)(inp->data), CU_DEV_WINDOW);
        // assert(nMiss == 0);
    }

    return true;
}