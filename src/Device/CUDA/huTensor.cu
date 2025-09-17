
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Functions of huTensor
 *  \author Yingshi Che
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../../Manifold/Fish.hpp"
#include "../../Tensor/GTensor.hpp"
#include "../../Utils/GST_log.hpp"
#include "../../Utils/GST_rander.hpp"
#include "../../g_float.hpp"
#include "./kernel/Operator.cuh"
#include "./kernel/utils.cuh"
static Grusoft::GRander randParam;
huTensor::huTensor(Fish* fish, const string& name_, const SHAPE shape, typNUMBER tpD_, bool isAlloc, int flag) : GTensor(fish, shape, tpD_, false, flag) {
    size_t nEle       = size();
    MEM_STRATEGY stra = fish->config.scheduling.strategy;
    if (stra == MEM_STRATEGY::PRE_ALLOC_HOST_MAP) {
        flags |= BIT_FLAG::F_HOSTALLOC;
    } else
        flags |= BIT_FLAG::F_GPU;
    // hFish->InitGensor(nullptr,name,attn,false);
    if (!name_.empty())
        snprintf(name, sizeof(name), "%s", name_.c_str());
    else
        name[0] = '\0';
    param_seed = randParam.RandU32();
    if (isAlloc) {
        Alloc(0x0, flag);
    }
}

hGTensor huTensor::Partial(const string& name_, size_t nOff, SHAPE shape, int flag) {
    hGTensor sub = GT(hFish, type, shape);
    if (!name_.empty())
        snprintf(sub->name, sizeof(name), "%s", name_.c_str());
    else
        sub->name[0] = '\0';
    assert(nOff + sub->size() <= size());
    assert(BitPE(type) == 8 || BitPE(type) == 16);
    int nB     = (int)(BitPE(type) / 8);
    sub->data  = (char*)data + nOff * nB;
    sub->grad  = grad + nOff;
    sub->flags = flags;
    BIT_SET(sub->flags, F_ONLYREF);
    return sub;
}

size_t GTensor::szGlobalMaloc = 0;
size_t huTensor::mostMemory(int typ) const {
    if (BIT_TEST(flags, F_NOALLOC))
        return 0x0;
    // if(BIT_TEST(flags,F_HOSTALLOC))
    //     return 0x0;
    if (hRef != nullptr) {
        return 0x0;
    }
    size_t most = nByte();
    if (isParam() && hFish->isTrain()) {
        most += nByte();                     // grad
        most += sizeof(float) * size() * 2;  // gm,gv is float array
    }
    if (isParam()) {
        most += sizeof(float) * ne[0];
    }
    return most;
}
/*
    cudaHostAlloc is a function used to allocate pinned (page-locked) host memory, which can improve data transfer performance between the host (CPU) and device
   (GPU). Pinned memory allows for faster transfers because it bypasses the operating system's virtual memory system.
*/
size_t huTensor::Alloc_1(void** dst, bool isZero, string desc, size_t sz0, int flag) {
    assert(*dst == nullptr);

    bool hostAlloc    = BIT_TEST(flags, F_HOSTALLOC);
    cudaError_t error = cudaSuccess;
    size_t szAlloc    = sz0 == 0 ? szData : sz0;
    assert(szAlloc > 0);
    if (!hostAlloc) {
        size_t szFree, szTotal;
        cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
        if (szAlloc > szFree) {
            _ERROR("[CUDA Alloc] Outof GPU Memory @%s!  Free=%gM < Need=%gM.\n", name, szFree / 1.0e6, szAlloc / 1.0e6);
            exit(KOIFISH_OUTOF_GPUMEMORY);
        }
    }
    error = hostAlloc ? cudaHostAlloc(dst, szAlloc, cudaHostAllocMapped) : cudaMalloc(dst, szAlloc);  // 8420
    // strange behavior of callo
    // data = calloc(szAlloc,1);  sAlloc = "Alloc_c/cu";   //8386
    if (error != cudaSuccess) {
        _INFO("[CUDA Alloc] failed @%s, ERR=%s!\n", name, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    if (isZero)
        cudaCheck(cudaMemset(*dst, 0, szAlloc));
    szGlobalMaloc += szAlloc;
    SUM::mems.push_back(MEM_USAGE(szAlloc, desc, *dst));
    return szAlloc;
}
size_t huTensor::Free_1(void** obj, const string& info) {
    if (BIT_TEST(flags, F_ONLYREF))
        return 0x0;

    assert(*obj != nullptr);
    // _INFO("\t%s%s freed@%p(%.3gM)!",name,info.c_str(),*obj,(szData)/1.0e6);
    if (BIT_TEST(flags, F_HOSTALLOC))
        cudaFreeHost(*obj);
    else {
        cudaError_t error = cudaFree(*obj);
        if (error != cudaSuccess) {
            cudaError_t const last_err{cudaGetLastError()};
            _INFO("[CUDA] free failed @\"%s\"! err=%s(%s).\n", name, cudaGetErrorString(error), cudaGetErrorString(last_err));
            // exit(EXIT_FAILURE);
        }
    }
    SUM::FreeMem(*obj);
    *obj = nullptr;
    szGlobalMaloc -= szData;

    return szGlobalMaloc;
}

static mt19937_state rngOfParams;
bool huTensor::InitParam(int tpX) {
    size_t nElem0 = size(), i;
    size_t nInit  = size(1);
    // bool isTmp          = true;
    int iter = hFish->GetCurIter();
    SUM::nInitParam++;  // may skip(bias is always init to 0)
    if (tpInit > 0 && tpInit != SERIALIZE) {
        if (strcmp(name, "model.out.weight_b") == 0) {  //  model.blk.34.ffn_down.weight
            int debug = 0;                              // Print(name, 1, -1);
        }
        // _INFO("[InitParam@%d]\t%ld-%ld@%s\n",iter,size(),nInit,name);
        if (BIT_TEST(flags, F_LORA_B)) {
            return true;
        }
        floatX* tmp = nullptr;  // new floatX[nInit];
        switch (tpInit) {
            case FIX_1:
                tmp = new floatX[nInit];
                for (i = 0; i < nInit; i++) tmp[i] = (floatX)(1.0);
                break;
            default:
                if (BIT_TEST(flags, F_TERNARY)) {
                    // int ldT = 8 / BitPE(type);
                    // assert(ldT == 8 || ldT == 64);
                    size_t dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
                    floatX* paramX = ToX(GTensor::tmpTernary);
                    cudaMalloc(&paramX, nInit * sizeof(floatX));
                    CU_disti_normal<floatX>(nInit, (floatX*)paramX, 0.02f * residual_scale, param_seed);
                    ToTernary(paramX);
                    cudaFree(paramX);
                    // Print(name,0,-1);
                } else
                    CU_disti_normal<floatX>(nInit, (floatX*)data, 0.02f * residual_scale, param_seed);

                break;
        }
        if (tmp != nullptr) {
            H2D(data, tmp, nInit * BPE(type));  // cudaCheck(cudaMemcpy(data, tmp, nInit * nB, cudaMemcpyHostToDevice));
            delete[] tmp;
        }

        // Print(name,0,-1);
    } else {
        if (tpInit == SERIALIZE) {  //  ???
            if (host_data != nullptr) {
                SerialGP(host_data, nullptr, szData, false);
            }
        }
    }
    if (DUMP())
        Print(name, 0, -1);  // dump some value
    return true;
}

/*
   Only for gguf-serialize
*/
bool huTensor::CopyGG(struct ggml_tensor* gg_, int flag) {
#ifdef __USE_GGML__
    int i = 0;
    assert(gg == nullptr);
    bool isAlloc = data != nullptr;
    void* src    = gg_->data;
    if (!isAlloc) {
        memcpy(name, gg_->name, sizeof(char) * GGML_MAX_NAME);
        for (i = 0; i < GGML_MAX_DIMS; i++) {
            shape.push_back(gg_->ne[i]);
            nb[i] = gg_->nb[i];
        }
        type = (typNUMBER)gg_->type;
        Alloc();
        // flags = gg_->flags;     //bug in ggml: don't support flag serialization
        double fnB = BPE(type);  // ggml_row_size  ???
        szData     = size() * fnB;
    } else {
        for (i = 0; i < shape.size(); i++) {
            if (BIT_TEST(flags, F_PADDED))
                assert(shape[i] >= gg_->ne[i]);
            else
                assert(shape[i] == gg_->ne[i]);
            if (type == (typNUMBER)gg_->type)
                assert(nb[i] == gg_->nb[i]);
        }
    }
    size_t szSrc = ggml_nbytes(gg_);
    if (type == (typNUMBER)gg_->type) {
        if (szSrc != szData) {
            if (BIT_TEST(flags, F_PADDED)) {
                assert(strcmp(name, "token_embd.weight") == 0 && szSrc <= szData);
            } else {
                assert(0);
            }
        }
    };

#ifdef _TENSOR_G_
    bool toDevice = SerialGP(src, nullptr, szSrc, false, 0x0);
    assert(toDevice);
#endif
#endif
    // if(src!=data)       delete[] src;
    return true;
}

//  From:   https://stackoverflow.com/questions/57948643/whats-a-good-way-to-zero-out-cudamallocd-data
/*__global__ void clear_scratch_space_kernel(int * data, int blocks, int threads) {
    // BOZO: change the code to just error out if we're any of the border cases below
    const int idx = blockIdx.x * threads + threadIdx.x;
    long size = sizeof(int) * COUNT;
    long size_of_typical_chunk = round_up(size / (blocks * threads), GPU_CACHE_LINE_SIZE_IN_BYTES);
    // Due to truncation, the threads at the end won't have anything to do.  This is a little sloppy but costs us
    // hardly anything in performance, so we do the simpler thing.

    long this_threads_offset = idx * size_of_typical_chunk;
    if (this_threads_offset > SIZE_OF_DATA) {
        return;
    }

    long size_of_this_threads_chunk;
    if (this_threads_offset + size_of_typical_chunk >= SIZE_OF_DATA) {
        // We are the last thread, so we do a partial write
        size_of_this_threads_chunk = SIZE_OF_DATA - this_threads_offset;
    } else {
        size_of_this_threads_chunk = size_of_typical_chunk;
    }
    void * starting_address = reinterpret_cast<void *>(reinterpret_cast<char *>(data) + this_threads_offset);
    memset((void *) starting_address, 0, size_of_this_threads_chunk);
}
__global__ void clear_scratch_space_with_coalesced_writes_kernel(int * data, int blocks, int threads) {
    if (COUNT % (blocks * threads) != 0) {
        _INFO("Adjust the SIZE_OF_DATA so it's divisible by the number of (blocks * threads)\n");
    }
    const long count_of_ints_in_each_blocks_chunk = COUNT / blocks;

    int block = blockIdx.x;
    int thread = threadIdx.x;

    const long rounds_needed = count_of_ints_in_each_blocks_chunk / threads;

    const long this_blocks_starting_offset = block * count_of_ints_in_each_blocks_chunk;

    //_INFO("Clearing %li ints starting at offset %li\n", count_of_ints_in_each_blocks_chunk, this_blocks_starting_offset);

    int * this_threads_base_pointer = &data[this_blocks_starting_offset + thread];
    for (int round = 0; round < rounds_needed; ++round) {
        *this_threads_base_pointer = 0;
        this_threads_base_pointer += threads;
    }
}
void set_gpu_data_to_ones(int * data_on_gpu) {
    cudaMemset(data_on_gpu, 1, SIZE_OF_DATA);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}
void check_gpu_data_is_zeroes(int * data_on_gpu, char * data_on_cpu) {
    cudaMemcpy(data_on_cpu, data_on_gpu, SIZE_OF_DATA, cudaMemcpyDeviceToHost);
    for (long i = 0; i < SIZE_OF_DATA; ++i) {
        if (data_on_cpu[i] != 0) {
            _INFO("Failed to zero-out byte offset %i in the data\n", i);
        }
    }
}*/

void huTensor::Zero() {
    assert(data != nullptr);
    //  https://stackoverflow.com/questions/57948643/whats-a-good-way-to-zero-out-cudamallocd-data
    cudaCheck(cudaMemset(data, 0, szData));
    if (grad != nullptr) {
        ZeroGrad();
    }
}
void huTensor::ZeroGrad() {
    assert(grad != nullptr);
    cudaCheck(cudaMemset(grad, 0, szData));
    // cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
}
bool cuClearGrad(std::vector<hGTensor> tensors, int flag) {
    for (auto tensor : tensors) {
        if (tensor->isRefer())
            continue;
        cudaCheck(cudaMemsetAsync(ToG(tensor), 0, tensor->nByte(), main_stream));
    }

    return true;
}

bool D2H(void* dev, void* host, size_t szData, int flag) {
    try {
        assert(host != nullptr && dev != nullptr);
        cudaCheck(cudaMemcpy(host, dev, szData, cudaMemcpyDeviceToHost));
        return true;
    } catch (...) {
        return false;
    }
}
bool H2D(void* dev, void* host, size_t szData, int flag) {
    try {
        assert(host != nullptr && dev != nullptr);
        cudaCheck(cudaMemcpy(dev, host, szData, cudaMemcpyHostToDevice));
        return true;
    } catch (...) {
        return false;
    }
}

bool huTensor::SerialData(const string& info, void* host, bool isToHost, int flag) {
    try {
        if (host == nullptr) {
            assert(host_data != nullptr);
            host = host_data;
        }
        if (isToHost) {
            // cudaCheck(cudaMemcpyAsync(host,data, szData, cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(host, data, szData, cudaMemcpyDeviceToHost));
        } else {
            // cudaCheck(cudaMemcpyAsync(data, host,szData, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(data, host, szData, cudaMemcpyHostToDevice));
        }
        if (flag < 0) {
            char buf[1024];
            sprintf(buf, "%s:%s@%s", info.c_str(), isToHost ? "SAVE" : "LOAD", name);
            Print(buf, 0, -1);
        }

        return true;
    } catch (...) {
        return false;
    }
}

//  如果CUDA支持统一内存（Unified Memory） 或GPUDirect RDMA，可以直接映射 GPU 内存到文件：
static bool isGPUDirectMMap = true;
//
bool GTensor::Serial_MMAP(bool isSave, bool isReset, int flag) {
    try {
        assert(isParam() && isGPUDirectMMap);
        bool bRet     = true;
        int dumpFlag  = 0;
        char* tmpData = (char*)host_data;  // new char[szData];
        assert(tmpData != nullptr);
        floatX* x = (floatX*)(tmpData);

        if (isSave) {
            assert(data != nullptr && gm != nullptr);
            // cudaCheck(cudaMemcpyAsync(host,data, szData, cudaMemcpyDeviceToHost));
            Print("mmap_save", 0, dumpFlag);
            cudaCheck(cudaMemcpy(tmpData, data, szData, cudaMemcpyDeviceToHost));
            if (tmpData != host_data) {
                memcpy(host_data, tmpData, szData), msync(host_data, szData, MS_SYNC);
            }

            cudaCheck(cudaMemcpy(tmpData + szData, gm, szM + szV, cudaMemcpyDeviceToHost));
            if (hRef != nullptr && isReset) {
                data = nullptr, gm = nullptr, gv = nullptr;
            }
            SUM::nSaveParam++;
        } else {
            if (hRef != nullptr) {  // huTensor::Alloc
                ShareMemory(hRef, 0x100);
            }
            assert(data != nullptr);
            SUM::szUpload += szData;
            if (tmpData != host_data) {
                memcpy(tmpData, host_data, szData), msync(host_data, szData, MS_SYNC);
            }
            // cudaCheck(cudaMemcpyAsync(data, host,szData, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(data, tmpData, szData, cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(gm, tmpData + szData, szM + szV, cudaMemcpyHostToDevice));
            Print("mmap_load", 0, dumpFlag);
            Print("mmap_load", 3, dumpFlag), Print("mmap_load", 2, dumpFlag);
            SUM::nLoadParam++;
        }
        if (tmpData != host_data)
            delete[] tmpData;
        return bRet;
    } catch (...) {
        return false;
    }
}

bool GTensor::Serial_MMAP_x(void* xdata, bool isSave, int flag) {
    try {
        assert(isParam() && isGPUDirectMMap);
        bool bRet     = true;
        int dumpFlag  = 0;
        char* tmpData = (char*)host_data;  // new char[szData];
        assert(tmpData != nullptr);
        assert(xdata != nullptr);
        if (isSave) {
            // cudaCheck(cudaMemcpyAsync(host,data, szData, cudaMemcpyDeviceToHost));
            Print("mmap_save_x", 0, dumpFlag);
            cudaCheck(cudaMemcpy(tmpData, xdata, szData, cudaMemcpyDeviceToHost));
            // SUM::nSaveParam++;
        } else {
            SUM::szUpload += szData;
            cudaCheck(cudaMemcpy(xdata, tmpData, szData, cudaMemcpyHostToDevice));
            // SUM::nLoadParam++;
        }
        return bRet;
    } catch (...) {
        return false;
    }
}

//  this <=> Y
bool huTensor::SerialGP(void* yD, void* yG, size_t szY, bool isToY, int flag) {
    try {
        if (isToY) {
            assert(szY >= szData);
            assert(data != nullptr);
            cudaCheck(cudaMemcpy(yD, data, szY, cudaMemcpyDeviceToHost));
            if (yG != nullptr) {
                assert(grad != nullptr);
                cudaCheck(cudaMemcpy(yG, grad, szY, cudaMemcpyDeviceToHost));
            }
        } else {
            assert(szY <= szData);
            SUM::szUpload += szY;
            cudaCheck(cudaMemcpy(data, yD, szY, cudaMemcpyHostToDevice));
            if (yG != nullptr) {
                assert(grad != nullptr);
                SUM::szUpload += szY;
                cudaCheck(cudaMemcpy(grad, yG, szY, cudaMemcpyHostToDevice));
                // cudaCheck(cudaMemcpy(grad, yG, szY, cudaMemcpyHostToDevice));
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool huTensor::OverWrite(hGTensor hGT, bool isSrc, int flag) {
    size_t nEle = size();
    assert(hGT->type == type);
    assert(isSameShape(hGT) && szData > 0);
    if (isSrc) {
        huTensor* src = dynamic_cast<huTensor*>(hGT.get());
        if (src == nullptr)  //  hGT => this
            cudaCheck(cudaMemcpy(data, hGT->data, szData, cudaMemcpyHostToDevice));
        else {
            cudaCheck(cudaMemcpy(data, hGT->data, szData, cudaMemcpyDeviceToDevice));
        }
    } else {  //  this => hGT
        assert(0);
    }

    return true;
}

hGTensor huTensor::CrossEntropy(const hGTensor b, int flag) { return b; }

double tNormsOf(const std::vector<hGTensor>& tensors, int flag) {
    float *grad_norm_squared, a, a_pre = 0.0;
    grad_norm_squared = (float*)(GTensor::bt4c->data);
    double norm       = 0.0f;
    int num_slices[2] = {1, 1}, max_num_block_sums = get_max_num_block_sums(num_slices, 2);
    size_t nz          = 0;
    bool is_first_pass = true;  // i==0
    for (auto tensor : tensors) {
        assert(0);  // Deprecated
        /*//ShardInfo shard ={0, tensor->size()};
        size_t nEle = tensor->size();       nz+=nEle;
        assert(tensor->grad!=nullptr);
        floatX* val = (floatX*)(tensor->grad);
        global_norm_squared(grad_norm_squared, val, nEle, 0, 1,max_num_block_sums, is_first_pass, main_stream);
        if(DEBUG.check_tensor_norm){
            cudaCheck(cudaMemcpy(&a, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));
            assert(a>=a_pre);
            tensor->gnorm = sqrt(a-a_pre);           a_pre = a;
        }
        is_first_pass = false;*/
        // PrintTensor<floatX>("tNormsOf",val,true,nEle,1);
        // break;
    }
    global_sum_deterministic(grad_norm_squared, grad_norm_squared, max_num_block_sums, main_stream);
    cudaCheck(cudaMemcpy(&a, grad_norm_squared, sizeof(float), cudaMemcpyDeviceToHost));

    norm = sqrt(a);
    a    = sqrt(a / nz);
    return norm;
}

static float* devBlockSum2 = nullptr;
double GTensor::Length(int type, int flag) {
    bool is_first_pass = true;
    size_t nEle        = size();
    floatX* src        = type == 1 ? (floatX*)grad : (floatX*)data;
    assert(src != nullptr);
    assert(sizeof(floatGrad) == sizeof(floatX));

    // float a = global_norm_squared((floatX*)(src), nEle, 0, 1, is_first_pass, main_stream); //0.00190092938
    float a = 0.0;

    constexpr int block_size = 512, num_slices = 1;
    auto now             = GST_us();
    const int dMaxThread = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount, grid_size = dMaxThread / block_size;
    if (devBlockSum2 == nullptr) {
        cudaMalloc(&devBlockSum2, sizeof(float) * grid_size * 2);
    }
    if (DEBUG.algCuX2 == 0) {   // too complex
        assert(grid_size > 0);  // gives a better error than letting the call below fail
        const int gx = CEIL_DIV(grid_size, num_slices), gy = num_slices;
        assert(gx * gy < 1024);  // we want to later accumulate the block sums in a single block
        if (strcmp(name, "model.blk.11.ffn_up.weight") == 0) {
            // Print(name, 1, -1);
            int debug = 0x0;
        }
        // cudaCheck(cudaMemsetAsync(norm2, 0, grid_size * sizeof(float), main_stream));
        CU_X2_partial<<<grid_size, block_size, 0, main_stream>>>(devBlockSum2, src, nEle);
        cudaCheck(cudaGetLastError());
        global_sum_deterministic(devBlockSum2, devBlockSum2, grid_size, main_stream);
        cudaCheck(cudaMemcpy(&a, devBlockSum2, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        size_t smemPB = 1024 * sizeof(float);
        int dT4B = 512, dGRID = dMaxThread / dT4B;
        dGRID = 512;
        assert(dGRID < 1024);  //  blockReduce_v0<warpReduceSum>
        cudaCheck(cudaMemset(devBlockSum2, 0, sizeof(float) * dGRID));
        CU_x2_<floatX><<<dGRID, dT4B, smemPB, main_stream>>>(devBlockSum2, src, nEle);  //  0.00190092938
        cudaCheck(cudaMemcpy(&a, devBlockSum2, sizeof(float), cudaMemcpyDeviceToHost));
        cudaStreamSynchronize(main_stream);
    }
    // SUM::tX1 += GST_us()-now;
    if (type == 1) {
        gnorm = sqrt(a);
        // if (fabs(gnorm) < 1.0e-10) {  //  0.0435996503
        //     Print(name, 1, -1);
        //     _INFO("\tZero |g|=%g@%s!", gnorm, name);
        // }
        if (isnan(gnorm)) {
            Print(name, 1, -1);
            _INFO("!!! NAN |g|@%s !!!\n", name), exit(KOIFISH_GRAD_EXPLODE);
        }
    } else {
        wnorm = sqrt(a);
    }
    a = sqrt(a / nEle);

    return gnorm;
}

hGTensor huTensor::GetRow(hGTensor hOut, hGTensor token, hGTensor pos, int flag) { return hOut; }

huTensor::~huTensor() { Free(); }

template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_ternary_v0(float* out, T* x0, int N) {  // block version
    int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    int idx = blockIdx.x * NUM_THREADS + tid, blockSize = blockDim.x;
    // if(idx >= N) { return; }

    __shared__ float average;
    // __shared__ T Ta,Tb;
    float sum = 0.0f, a;
    for (int j = tid; j < N; j += blockSize) {
        a = CU_T2Float(x0 + j);
        sum += fabs(a);  //	6.5
                         // sum += (float)(x0[j]);				//	6.3
    }
    float block_sum = blockReduce_v0<warpReduceSum>(sum, true);
    // if (tid == 0) atomicAdd(out, block_sum);
    // SYNC_GRID();
    if (tid == 0) {
        *out    = block_sum;
        average = (*out / (N)) + 1.0e-5;
        out[1]  = average;
        //  average = average/2;
    }
    for (int j = tid; j < N; j += blockSize) {
        a     = CU_T2Float(x0 + j);
        x0[j] = a > average / 2 ? (T)1.0 : a < -average / 2 ? (T)(-1.0) : (T)(0.0);
    }
    __syncthreads();
}

template <class T, int NUM_THREADS = CU_T4B_SMALL>
__global__ static void CU_binary_(float* out, T* x0, size_t N) {  // block version
    int tid = threadIdx.x, warp = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    int idx = blockIdx.x * NUM_THREADS + tid, blockSize = blockDim.x;
    // if(idx >= N) { return; }

    __shared__ float average;
    // __shared__ T Ta,Tb;
    float sum = 0.0f, a;
    for (int j = tid; j < N; j += blockSize) {
        a = CU_T2Float(x0 + j);  //	6.5
        sum += (float)(a);       //	6.3
    }
    float block_sum = blockReduce_v0<warpReduceSum>(sum, true);
    // if (tid == 0) atomicAdd(out, block_sum);
    // SYNC_GRID();
    if (tid == 0) {
        *out    = block_sum;
        average = (*out / (N)) + 1.0e-5;
        out[1]  = average;
        //  average = average/2;
    }
    for (int j = tid; j < N; j += blockSize) {
        a     = CU_T2Float(x0 + j);
        x0[j] = a - average > 0 ? (T)1.0 : (T)(-1.0);
    }
    __syncthreads();
}

int _CheckX2Tile_(floatX* A, floatGama* hgama, int M, int N, int flag) {
    int r, c, nz = 0;
    floatGama* gama = hgama;
    float off, off_0 = FLT_MAX, off_1 = -FLT_MAX, a, average;
    for (int r0 = 0; r0 < M; r0 += THREAD_TILE_M) {
        for (int c0 = 0; c0 < N; c0 += THREAD_TILE_N, gama++) {
            float tile_sum = 0;
            for (r = r0; r < r0 + THREAD_TILE_M; r++) {
                for (c = c0; c < c0 + THREAD_TILE_N; c++) {
                    a = (float)(A[r * N + c]);
                    tile_sum += a;
                    average += fabs(a), nz++;
                }
            }
            tile_sum /= THREAD_TILE_M * THREAD_TILE_N;
            a   = *gama;
            off = fabs(tile_sum - a);                           //  0.294229507
            if (off > max(fabs(tile_sum), fabs(a)) * 1.0e-2) {  // 0.0158233643
                assert(0);
                break;
            }
            off_0 = std::min(off_0, off), off_1 = std::max(off_1, off);
        }
    }
    average /= (M * N);
    if (sizeof(floatGama) == 4)
        assert(off_1 < average * 1.0e-6);
    else
        assert(off_1 < average * 1.0e-2);
    return 0x0;
}

bool huTensor::Mutation(int flag) {
    if (!isWMAT())
        return false;
    float w0 = wnorm;
    if (0) {
        Print(name, 0, -1);
        Length(0);  //  type == 1 ? (floatX*)grad : (floatX*)data;
        assert(fabs(w0 - wnorm) < 1.0e-5 * wnorm);
    }

    int nParam = size(), dT4B = 512, M = ne[0], N = ne[1], mGRID = CEIL_DIV(M, dT4B), pGRID = CEIL_DIV(nParam, dT4B), nRander = M;
    curandState* d_states;
    cudaCheck(cudaMalloc(&d_states, nRander * sizeof(curandState)));
    int seed = 42;  // rander.RandU32();
    CU_initrand<<<CEIL_DIV(nRander, 256), 256>>>(d_states, seed, nRander);

    float T_scale = wnorm == 0 ? 1.0 : sqrt(wnorm * wnorm / M / N), T_mutation = 1.0e-5;  //  1.0e-4 would explode
    CU_mutation_<<<mGRID, dT4B, 0, main_stream>>>(d_states, T_mutation, T_scale * 0.01, (floatX*)(data), (floatX*)nullptr, nParam, N);
    cudaCheck(cudaFree(d_states));
    return true;
}

bool huTensor::ToTernary(floatX* paramX, int flag) {
    if (!BIT_TEST(flags, GTensor::F_TERNARY))
        return false;
    size_t count = size(), dT4B = CU_T4B_SMALL, smemPB = 1024 * sizeof(float);
    bool isOverwrite = false;
    assert(this->isParam() && paramX != nullptr);
    assert(isWMAT());  // only for 2D weight
    // auto dGRID         = 1;  // hFish->curDevice()->GridDim(count);
    // void* kernelArgs[] = {(void*)&gama_T, (void*)&data, (void*)&ne[0], (void*)&ne[1], (void*)&tpQuant};
    if (type == typNUMBER::T_BINARY_TILE) {
        dim3 dBlock(THREAD_TILE_M * THREAD_TILE_N), dGrid(CEIL_DIV(ne[0], THREAD_TILE_M), CEIL_DIV(ne[1], THREAD_TILE_N));
        assert(ne[0] % THREAD_TILE_M == 0 && ne[1] % THREAD_TILE_N == 0);
        CU_X2Tile_<floatX><<<dGrid, dBlock, smemPB, main_stream>>>(paramX, gama_T(), 0.0, ne[0], ne[1], isOverwrite, 0, 0);
    } else
        CU_X2ternary_<floatX><<<CEIL_DIV(ne[0], dT4B), dT4B, smemPB, main_stream>>>(gama_T(), paramX, (char*)data, ne[0], ne[1], 1, true);
    // tNormOf();
    PrintTensor<floatX>("BitOfX0", paramX, true, ne[0], ne[1], 1, 1, 0);
    // GTensor::tmpTernary->Print("BitOfX0", 0, -1, count);
    // D2H(gama_T(), info, sizeof(info));   // floatGama->float

    // GTensor::tmpTernary->Print("X2T", 0, -1, count);
    if (1) {  //  only for debug
        if (type == typNUMBER::T_BINARY_TILE) {
            // GetDataX();
            floatX* hx       = new floatX[size() * 2];
            floatGama* hgama = (floatGama*)(hx + size());
            D2H(gama_T(), hgama, size() * sizeof(floatGama) / THREAD_TILE_M / THREAD_TILE_N);
            D2H(paramX, hx, size() * sizeof(floatX));
            _CheckX2Tile_(hx, hgama, ne[0], ne[1], 0x0);
            delete[] hx;
        } else {
            assert(GTensor::tmpFF1->size() >= count);
            CU_ternary2X_<floatX><<<CEIL_DIV(ne[0], dT4B), dT4B, smemPB, main_stream>>>(gama_T(), (char*)data, ToX(GTensor::tmpFF1), ne[0], ne[1]);
            double off = OFF_(paramX, ToX(GTensor::tmpFF1), count);
            GTensor::tmpFF1->Print("BitOfX", 0, 0, count);
            // assert(off <= 1.0e-5);
        }
    }

    if (tpQuant == QUANT_ALG::W_NOSCALE) {
        assert(info[0] == 1.0);
    }
    // Print(name,0,-1);
    if (DEBUG.T_ternary == 1) {
        cudaCheck(cudaMemcpy(data, paramX, sizeof(floatX) * count, cudaMemcpyDeviceToDevice));
    }
    return true;
}