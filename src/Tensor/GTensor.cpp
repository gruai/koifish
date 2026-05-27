
/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Che
 */
#include "GTensor.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../Manifold/Fish.hpp"
#include "../PackedQ.hpp"
#include "GeQuant.hpp"

GTensor* GTensor::tZ      = nullptr;
float* GTensor::stat_info = nullptr;
void *GTensor::buff = nullptr, *GTensor::host_buff = nullptr, *GTensor::qkv_workspace = nullptr;
size_t GTensor::buff_len = 0, GTensor::workspace_size = 0;

float GTensor::rLARS(float s0, float T_lars, int flag) {
    if (shape.size() <= 1)
        return s0;

    float eps         = 1.0e-8;
    float trust_ratio = wnorm / (gnorm + eps);
    trust_ratio       = std::min(trust_ratio, T_lars);
    float r           = trust_ratio;
    return r;
}

GTensor::GTensor(Fish* hFis_, SHAPE shape_, typNUMBER tpD_, bool isX, int flag) : hFish(hFis_), flags(flag) { ReShape(shape_, tpD_, flag); }

hGTensor GT(SHAPE shape_, void* src, typNUMBER tpD_, int flag) {
    hGTensor t = std::make_shared<GTensor>(nullptr, shape_, tpD_, true, 0x0);
    t->Alloc();
    assert(0);  // memcpy is dangerous
    memcpy(t->data, src, t->nByte());
    assert(!t->isEmpty());
    return t;
}

hGTensor GT(Fish* hFish, typNUMBER type, SHAPE shape, int flag, const string& name) {
    hGensor hT = std::make_shared<huTensor>(hFish, name, shape, type, false, flag);
    return hT;
}

/*
    A weight matrix is a linear operator between RMS-normed vector spaces.
    Let y = Wx, then |y|/|x| ~ ||W|| the spectral norm(largest singular value)
    报元（ 酉矩阵）守一（1-norm vector）
    1. fan-out,fan-in are the dimensions of the weight matrix
    2. No Embedding layer that takes one-hot inputs.
*/
bool GTensor::isWMAT(int flag) const {
    if (!BIT_TEST(flags, F_WMATRIX) || !BIT_TEST(flags, F_PARAM))
        return false;
    assert(ne[0] > 1 && ne[1] > 1 && ne[2] == 1 && ne[3] == 1);
    return true;
}

bool GTensor::isSameShape(SHAPE sp, int flag) const {
    assert(sp.size() <= 4);
    size_t nz_0 = 1, i = 0;
    for (auto s : sp) {
        if (s != ne[i++])
            return false;
        nz_0 *= s;
    }
    if (nz_0 != size())
        return false;
    return true;
}
bool GTensor::isSameShape(const hGTensor b) const {
    // different layer may use differnt Q4->Q3->Q2
    bool isMatch = isSameShape(b->shape);
    return isMatch;
    /*
        if (szData != b->szData) {
            if (isMatch) {
                DEBUG_HERE;
            }
            return false;
        }
        assert(isMatch);*/
    return true;
}

size_t GTensor::Offset(size_t nEle, int flag) const {
    size_t off  = nEle;
    double nBit = BitPE(type);
    switch (type) {
        case typNUMBER::T_SIGN:
        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3:
        case typNUMBER::T_BINARY_TILE:
            assert(0);
            break;
        default:
            off *= (nBit / 8);
            break;
    }
    return off;
}

size_t GTensor::Offset(int i0, int i1, int i2, int i3, int flag) const {
    size_t off = i0 + i1 * ne[0] + i2 * ne[0] * ne[1] + i3 * ne[0] * ne[1] * ne[2];
    //  off = i0 * nb[0] + i1 * nb[1] + i2 * nb[2] + i3 * nb[3];
    double nBit = BitPE(type);
    switch (type) {
        case typNUMBER::T_SIGN:
        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3:
        case typNUMBER::T_BINARY_TILE:
            assert(0);
            break;
        default:
            off *= (nBit / 8);
            break;
    }

    // if (nBit >= 8.0) {
    //     nb[0] = (size_t)(nBit / 8.0);
    //     nb[1] = nb[0] * ne[0];
    //     assert(nb[0] * 8 == nBit);
    // } else {  // For bit-representation of ld
    //     assert(ne[0] % 8 == 0);
    //     nb[0] = 1, nb[1] = nb[0] * (ne[0] / 8);
    // }
    // // nb[1] = nb[0] * (ne[0] / szBlk);
    // for (int i = 2; i < N_DIMS; i++) {
    //     nb[i] = nb[i - 1] * ne[i - 1];
    // }
    assert(off < szData);
    return off;
}

//  support dynamic change shape&type!  return false if not change anything
bool GTensor::ReShape(SHAPE shape_, typNUMBER tpD_, int flag) {
    if (type == tpD_ && shape == shape_)
        return false;

    if (shape != shape_) {
        shape = shape_;
        int i = 0;
        for (auto n : shape) {
            ne[i++] = n;
            assert(n > 0 && "Invalid ne");
        }
        for (i = shape.size(); i < N_DIMS; i++) ne[i] = 1;
    }
    type = tpD_;

    double nBit = BitPE(type), a = size() / 8.0 * nBit;
    szData = (size_t)(a);
    assert(szData * 1.0 == a);
    bool isRelloc = !BIT_TEST(flag, F_OP_NO_REALLOC);
    if (data != nullptr && isRelloc) {
        Free();
        INIT_WEIGHT tpOldInit = tpInit;
        tpInit                = W_SKIP;  // just pass the InitParam in the Alloc function
        Alloc();
        tpInit = tpOldInit;
    }
    // _INFO();
    return true;
}

void* GTensor::BeforeQuant(GeQuant* hQuant, int flag0) {
    assert(size() <= hQuant->nzMost);
    bool bRet = ReShape(shape, hQuant->bit2typ(), flag0);
    //      hTensor->Alloc_1(&data_fp8, true, nam_, szData);
    return nullptr;
}

// TODO - Shrink memory. But CUDA does not​ provide a direct way to shrink an existing allocation!
bool GTensor::AfterQuant(GeQuant* hQuant, typNUMBER type_quant, void* data_quant, int flag) {
    typNUMBER oldType = type;
    void* data_old    = data;
    bool bRet         = ReShape(shape, type_quant, F_OP_NO_REALLOC);
    assert(bRet);
    if (hQuant != nullptr && !hQuant->isGPU) {
        H2D(data, data_quant, szData);
    } else {
        Free_1(&data_old);
        data = data_quant;
        Print("F8Ex", 0, 0);
        for (auto t : refered) {
            assert(t->type == oldType);
            t->type = type;
        }
    }

    return true;
}

void GTensor::SetRefer(hGTensor hR, int flag) {
    if (strcmp(name, "model.blk.6.attn.wq.weight") == 0) {
        int flag = 0x0;
    }
    assert(hRef == nullptr);  //  only ref once
    hRef = hR;
    hR->refered.push_back(this);
    type = hRef->type;
    if (hFish->config.dumpSwitch.tensor_ref > 0)
        _INFO("\t%s =====> %s\n", name, hR->name);
}
/*  Deprecated
bool GTensor::SetTernary(typNUMBER tpT_, int flag) {
    if (type == tpT_)
        return true;
    BIT_SET(flags, GTensor::F_TERNARY);
    // tpQuant = W_SCALE;
    if (DEBUG.T_ternary == 1) {  //  only for debug
        type = tpT_;
        return true;
    }
    for (auto t : refered) {
        t->type = tpT_;
    }
    return ReShape(shape, tpT_);
}*/

float GTensor::Get(int i, int flag) const {
    assert(0);
    return 0.f;
}

int GTensor::GetDynamicEmbed(int flag) const {
    int n0 = hFish->config.nEmbed();
    return n0;
}
hQUANT GTensor::GetDynamicQuant(int flag) const {
    if (hQuant == nullptr)
        return nullptr;
    if (size() > hQuant->nzMost)
        return nullptr;
    // if(!G_Has_(name,{"model.layers.1.mlp.down_proj.weight"})){
    //     return nullptr;
    // }
    if (flag == 0x100) {  // only for some test
        assert(ginfo != nullptr);
        int lay = ginfo->hNeron->layid - 1;
        if (lay == 0)  //      lay <= 1 || lay>10
            return nullptr;
    }
    return hQuant;
}

GTensor::~GTensor() {
    if (!BIT_TEST(flags, F_GPU)) {
        if (!BIT_TEST(flags, F_MMAP)) {
            FREE_a(data);
            FREE_a(grad);
        }
    }
    if (BIT_TEST(flags, F_MMAP)) {
    } else {
        if (BIT_TEST(flags, F_GPU)) {
            /* No need to free host_data, since
            1. host_data is just ref of MMF "host = src" @GTensor::LoadParam_
            */
        } else
            FREE_a(host_data);
    }
}

bool GTensor::isAtHost() const {
    bool isGPU  = BIT_TEST(flags, F_GPU);
    bool isHost = BIT_TEST(flags, F_HOSTALLOC) || BIT_TEST(flags, F_MMAP);
    return isHost;
}

void GTensor::AddSrc(const vector<hGTensor>& ts, int flag) {
    for (auto t : ts) {
        if (t == nullptr)
            continue;
        hGOP hop = std::make_shared<GENSOR_OP>(t);
        src.push_back(hop);
    }
}

void GTensor::Set(float a, int flag) {
    assert(!isEmpty());
    if (a == 0) {
        memset(data, 0x0, szData);
    } else {
        assert(0);
    }
}
#ifdef __USE_GGML__
bool GTensor::OverWrite(struct ggml_tensor* gg_, bool isSrc, int flag) {
    assert(size() == ggml_nelements(gg_));
    assert(type == (typNUMBER)gg_->type);
    if (isSrc) {
        memcpy(data, gg_->data, szData);
    } else {
        memcpy(gg_->data, data, szData);
    }

    return true;
}
#endif

bool GTensor::OverWrite(hGTensor hGT, bool isSrc, int flag) {
    /*huTensor *src = dynamic_cast<huTensor *>(hGT.get());
    size_t nEle = size();
    assert(isSameShape(hGT));
    if(src!=nullptr){
       // cudaCheck(cudaMemcpy(data, src->data, szData, cudaMemcpyHostToDevice));
       return true;
    }*/
    assert(0);
    return false;
}

bool GTensor::ShareMemory(hGTensor src, int flag) {
    assert(src != nullptr && src->data != nullptr);
    // assert(src->nByte()>=nByte());   // may fail
    data = src->data;

    if (src->grad != nullptr) {
        grad = src->grad;
        gm   = src->gm;
        gv   = src->gv;
    }
    return true;
}

hGTensor GTensor::CrossEntropy(const hGTensor b, int flag) {
    // auto cur = ggml_cross_entropy_loss(nullptr,(struct ggml_tensor *)gg, b->GG() );      // ggml_cross_entropy_loss_1(_ctx, cur, target_probs);
    // return GTensor::NEW_(cur);
    return nullptr;
}

hGensor GENSOR_TOPU::Get(MODEL_ARCH arch, const string& name, int flag) {
    if (flag == 0x100) {  //  .weight=>.w
        for (auto ng : nag) {
            if (strstr(name.c_str(), ng.first.c_str()) != NULL) {
                return ng.second;
            }
        }
        return nullptr;
    } else {
        string key  = name;
        bool isMiss = nag.find(name) == nag.end();
        if (isMiss) {  //@NN2NAME
            size_t pos = 0;
            /*if (arch == MODEL_ARCH::NLP_QWEN2) {    //some hack for mismatch of name
                std::map<std::string, std::string> S2S={
                    {"input_layernorm","self_attn.norm"}
                };
                for(auto ss : S2S){
                    if( (pos = key.find(ss.first)) != std::string::npos) {
                        key.replace(pos, ss.first.length(), ss.second);
                        isMiss = nag.find(key) == nag.end();
                        break;
                    }
                }
            }*/
        }
        if (isMiss) {
            for (auto ng : nag) {
                _INFO("\t%s,", ng.first.c_str());
            }
            _ERROR("Failed to get tensor=%s nGensor=%d", name.c_str(), nag.size());
            return nullptr;
        }
        return nag[key];
    }  //  model.layers.0.input_layernorm.weight
}
void ToChebyshev(int N, float* rows, int flag = 0x0);

/*
floatGama* gamaOf(floatGama* gama_0, GTensor::GAMA_TYPE type, int nRow, int nCol, int flag) {
    switch (type) {
        case GTensor::R_SCALE:
            return gama_0;
        case GTensor::C_SCALE:
            return gama_0 + nRow;
        case GTensor::ZERO:
            return gama_0 + nRow + nCol;
        case GTensor::STEP:
            return gama_0 + nRow + nCol + nRow;
        case GTensor::LUT:
            return gama_0 + nRow + nCol;
        case GTensor::OFF: {
            int off = flag;
            return gama_0 + nRow + nCol + off;
        }
        case GTensor::BACKUP: {
            int off = flag;
            return gama_0 + nRow + nCol + off;
        }
        default:
            assert(0);
            break;
    }
    return nullptr;
}*/

/*
    return scaling/LUT of quant weight
        1. RC_SCALE is the scaling of each row/column(is not same as nGroup/lGroup!)
        2. gama_0 may be some temp memory
*/
floatGama* GTensor::gama_T(GAMA_TYPE gama_type, floatGama* gama_0) const {
    if (hRef != nullptr)
        return hRef->gama_T();

    bool isUserGama = gama_0 != nullptr;
    int nRow = ne[0], nCol = ne[1], nGroup = 0, lGroup = 0;
    if (hQuant != nullptr) {
        SHAPE spGroup = hQuant->GroupShapeOfT(this);
        nGroup = spGroup[0], lGroup = spGroup[1];
        // if (szGama == 0) {
        //     assert(hQuant->params.type == AWQ);
        //     return nullptr;
        // }
    }
    if (!isUserGama) {  // gama_0 == nullptr
        assert(data != nullptr);
        if (szGama == 0)
            return nullptr;
        gama_0 = reinterpret_cast<floatGama*>((hBITARR)data + szData);
    }
    if (gama_type == GAMA)
        return gama_0;

    floatGama* gama_1 = gama_0 + nRow + nCol;
    if (!isUserGama) {
        assert(gama_1 - gama_0 <= szGama);
    }
    switch (gama_type) {
        case GTensor::R_SCALE:
            return gama_0;
        case GTensor::AVERAGE:
            return gama_0;
        case GTensor::C_SCALE:
            return gama_0 + nRow;
        case GTensor::ZERO:
            return gama_1;
        case GTensor::STEP:
            return gama_1 + nGroup;
        case GTensor::LUT:
            return gama_1;
        // case GTensor::OFF: {
        //     int off = flag;
        //     return gama_0 + nRow + nCol + off;
        // }
        // case GTensor::BACKUP: {
        //     int off = flag;
        //     return gama_0 + nRow + nCol + off;
        // }
        default:
            assert(0);
            break;
    }
    // floatGama* gama_ = gamaOf(gama_0, type, ne[0], ne[1]);
    return gama_0;
}

/**
 * LittleEndian!!! [5, 10, 9, 11, 12, 7, 9, 7]=>内存顺序（从左到右,低到高）：0x5A, 0x9B, 0xC7, 0x97 => uint32=0x97C79B5A(以小端解释32位整数)
 */
void PrintQ_128(const char* title, const BIT_128* src, int nPer128, int qBias, size_t n0, int flag) {
    bool isLittleEndian = BIT_TEST(flag, FLOAT_META::ENDIAN_LITTLE);
    assert(isLittleEndian);
    int qq[128];
    size_t nElem = (size_t)(n0) / nPer128, i, nz = 0, nEach = 2;
    if (nElem == 0)
        return;
    assert(src != nullptr);
    // if(strlen(title)>0) _INFO("%s\n", title);
    float sum = 0.0, a1 = 16, a0 = 0;
    double len = 0.0, sum2 = 0.0;

    for (i = 0; i < nElem; i++) {
        bool isDump = (i < nEach || i >= nElem - nEach || fabs(i - nElem / 2) <= nEach);
        if (isDump)
            _INFO("%#X=", src[i]);
        if (nPer128 == 32)
            UNPACK_128to4_UNSIGNED_(src + i, qq);
        if (nPer128 == 64)
            UNPACK_128to2_UNSIGNED_(src + i, qq);
        if (nPer128 == 128)
            UNPACK_128to1_UNSIGNED_(src + i, qq);

        //    assert(0 && "Invalid nPer128 of PrintQ_128");

        for (int j = 0; j < nPer128; j++) {
            int a = qq[j] - qBias;
            if (j < 8 && isDump)
                _INFO("%d ", a);

            sum += fabs(a);
            sum2 += a * a;
        }
        if (i == nEach || i == nElem - nEach - 1)
            _INFO("...");
    }
    assert(!isnan(sum2) && !isinf(sum2));

    nElem *= nPer128;
    len = sqrt(sum2 / nElem);
    //  printf output is only displayed if the kernel finishes successfully,  cudaDeviceSynchronize()
    _INFO("\t\"%s\" |avg|=%g(%ld) avg_len=%g sum2=%g [%f,%f] nz=%.3g\n", title, sum / nElem, nElem, len, sum2, a0, a1, nz * 1.0 / nElem);
    fflush(stdout);
}

void GTensor::Print(const string& title0, int x, int flag, size_t nEle) const {
    if (g_dump_level > 0 && flag >= 0)
        return;
    assert(nEle >= 0);
    bool isDevice = !isAtHost();
    void *src = x == 3 ? gv : x == 2 ? gm : x == 1 ? grad : data, *hData = nullptr;
    typNUMBER tpSrc = x == 0 ? type : typNUMBER::BF16;
    string suffix   = x == 3 ? "GV_" : x == 2 ? "GM_" : x == 1 ? "GRAD_" : "";
    if (x == 4) {
        assert(host_data != nullptr);
        src      = host_data;
        tpSrc    = type;
        isDevice = false;
    }
    if (src == nullptr) {
        _INFO("Failed to print! %s of \"%s\" is nullptr!", x == 3 ? "gv" : x == 2 ? "gm" : x == 1 ? "grad" : "data", name);
        return;
    }
    size_t szHost = nEle > 0 ? (nEle * BitPE(tpSrc)) / 8 : szData + szGama;
    if (isDevice) {  // so many crash when print
        // SYNC_DEVICE();
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            _INFO("_SYNC_DEVICE_ err=\"%s\" (%s code=%d)\t%s\n", cudaGetErrorString(error), cudaGetErrorName(error), error, "");
            assert(0 && "_SYNC_DEVICE_@ GTensor::Print");
            exit(KOIFISH_EXIT_PRINT);
        }
        hData = new char[szHost];
        D2H(src, hData, szHost);
        src = hData;
    }

    string title = suffix + title0;
    // if (x == 1)
    //     title = "GRAD_" + title;
    int64_t sp[N_DIMS] = {ne[0], ne[1], ne[2], ne[3]};
    if (nEle > 0 && nEle != ne[0] * ne[1] * ne[2] * ne[3]) {
        sp[0] = nEle, sp[1] = 1;
        sp[2] = 1;
        sp[3] = 1;
    }
    if (nEle == 0)
        nEle = size();

    floatGama *rNormal = nullptr, *cNormal = nullptr;
    if (disq.rc_normal > 0) {
        rNormal = gama_T(R_SCALE), cNormal = gama_T(C_SCALE);
    }
    switch (tpSrc) {
        case typNUMBER::T_BINARY_3:
        case typNUMBER::T_BINARY_TILE:
            assert(0);
            break;
        case typNUMBER::Q4: {
            // floatX* wX = GetDataX();
            // PrintTensor<floatX>(title.c_str(), wX, true, sp[0], sp[1], sp[2], sp[3], flag);
            PrintQ_128(szGama == 0 ? name : "_Q4", (BIT_128*)src, 32, 0, nEle, flag | FLOAT_META::ENDIAN_LITTLE);
            hQuant->PrintGama(this);
        } break;
        case typNUMBER::BOOL1:
        case typNUMBER::T_BINARY:
        case typNUMBER::T_SIGN: {
            title += tpSrc == typNUMBER::T_SIGN ? "_Q2" : "_Q1";
            PrintQ_128(title.c_str(), (BIT_128*)src, tpSrc == typNUMBER::T_SIGN ? 64 : 128, 1, nEle, flag | FLOAT_META::ENDIAN_LITTLE);
            hQuant->PrintGama(this);
            // floatX* wX = GetDataX();
            // PrintTensor<floatX>(name, wX, true, sp[0], sp[1], sp[2], sp[3], flag);
        } break;
        case typNUMBER::Q3:
            // PrintQ3("_Q3", (hBITARR)src, ne[0], ne[1], ne[2], ne[3], flag);
            if (disq.rc_normal > 0) {
                PrintTensor<floatGama>("_rNormal", rNormal, true, ne[0], 1, 1, 1, flag);
                PrintTensor<floatGama>("_cNormal", cNormal, true, ne[1], 1, 1, 1, flag);
            }
            // PrintTensor<floatGama>("_gamaQ", gamaQ, true, ne[0], 2, 1, 1, flag);
            break;
        case typNUMBER::Q2:
            // PrintQ2("_Q2", (hBITARR)src, ne[0], ne[1], ne[2], ne[3], flag);
            if (disq.rc_normal > 0) {
                PrintTensor<floatGama>("_rNormal", rNormal, true, ne[0], 1, 1, 1, flag);
                PrintTensor<floatGama>("_cNormal", cNormal, true, ne[1], 1, 1, 1, flag);
            }
            // PrintTensor<floatGama>("_gamaQ", gamaQ, true, ne[0], 2, 1, 1, flag);
            break;
        case typNUMBER::F8E5M2:
            //    PrintTensor<__nv_fp8_e5m2>(title.c_str(),(__nv_fp8_e5m2 *)data, isDevice,ne[0],ne[1],ne[2],ne[3],flag);
            PrintTensor<f8e5>(title.c_str(), (f8e5*)src, false, ne[0], ne[1], ne[2], ne[3], flag);
            break;
        case typNUMBER::F16:
            PrintTensor<half>(title.c_str(), (half*)src, false, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        case typNUMBER::BF16:
            PrintTensor<bf16>(title.c_str(), (__nv_bfloat16*)src, false, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        case typNUMBER::F32:
            PrintTensor<float>(title.c_str(), (float*)src, false, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        case typNUMBER::I32:
            PrintTensor<int>(title.c_str(), (int*)src, false, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        default:
            assert(0);
            break;
    }
    if (hData != nullptr)
        delete[] (char*)hData;
}

std::string GTensor::Alias(int flag) {
    string alias, alias_0 = name;
    std::vector<size_t> dots;
    for (size_t i = 0; i < alias_0.size(); ++i) {
        if (alias_0[i] == '.') {
            dots.push_back(i);
        }
    }
    int start = dots.size() >= 3 ? dots[1] + 1 : 0, end = dots.size() > 1 ? dots[dots.size() - 1] : alias_0.size();
    alias = alias_0.substr(start, end - start);
    return alias;
}

// CPU version is Dreprecated!
bool GTensor::DumpX(int tpDump, const string& title, int flag) const {
    if (strcmp(name, "model.layers.0.self_attn.wqkv.bias") == 0) {
        DEBUG_HERE;
    }
    size_t nz = 0, nElems = size(), i = 0, n = 10;
    float *fdata = (float*)data, a1 = -FLT_MAX, a0 = FLT_MAX;
    const char* A = "d";
    if (flags & GTensor::F_PARAM) {
        A = "P";
    }
    if (BIT_TEST(flags, F_GPU))
        n = 0;
    switch (tpDump) {
        case 100:
            _INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n", i, ne[0], ne[1], "", name);  //
            break;
        default:
            double sum = 0.0;
            /*if (fdata != nullptr && !BIT_TEST(flags, F_GPU)) {    //  CPU version is Dreprecated!
                for (i = 0; i < nElems; i++) {
                    sum += fdata[i];
                    if (fdata[i] == 0)
                        nz++;
                    a1 = std::max(a1, fdata[i]);
                    a0 = std::min(a0, fdata[i]);
                }
            }*/
            if (szUse == 0x0) {
                _INFO("\t%s %-36s %-4s", title.c_str(), name, A);
                _WARN0(" NO ALLOC ");
                _INFO(" \t[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %s] ", ne[0], ne[1], ne[2], ne[3], cNameOf(type));
            } else
                _INFO("\t%s %-36s %-4s szAlloc=%6gM(gama=%.3g,M=%.3g,V=%.3g)\t[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %s] ", title.c_str(), name, A,
                      szUse / 1.0e6, szGama / 1.0e6, szM / 1.0e6, szV / 1.0e6, ne[0], ne[1], ne[2], ne[3], cNameOf(type));
            if (disq.err > 0)
                _INFO("eQ=%.3g ", disq.err);
            _INFO("\n");
            if (DEBUG.dump_TensorDetail)
                Print(name, 0, -1);
            /*if (n > 0 && a1 != -FLT_MAX) {
                _INFO("\nsum=%g data=[%f : %f] rZ=%.3g%%\n\t", sum, a0, a1, nz * 100.0 / nElems);
                for (int i = 0; i < std::min((size_t)(ne[0] * ne[1]), n); i++) {
                    _INFO("%.5f ", fdata[i]);
                    if (i != 0 && i % ne[0] == 0) {
                        // printf("\n");
                    }
                }
                printf("...");
                for (int i = 0; i < std::min((size_t)(ne[0] * ne[1]), n); i++) {
                    printf("%.5f ", fdata[nElems - n + i]);
                    if ((nElems - n + i) % ne[0] == 0) {
                        // printf("\n");
                    }
                }
                printf("}\n");
            }*/

            if (fdata != data && fdata != nullptr) {
                delete[] fdata;
            }
    }
    return true;
}

void _T_repr_(hGensor t, const char* tab, char* buf, const GENSOR_INFO& info) {
    if (t == nullptr)
        return;
    const char* A = "d";
    if (t->flags & GTensor::F_PARAM) {
        A = "P";
    } else {
        if (t->grad != nullptr) {
            A = "G";
        }
    }

    auto ne   = t->ne;
    size_t n0 = strlen(buf), n1;                      // char buf[64*1024]="\0";
    string suf, pref, sX = info.__repr__(suf, pref);  //    info.sX;
    sprintf(buf + strlen(buf), "%s %s %s %s \tdata=%p grad=>%p\t[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " ] \n", tab, sX.c_str(), A, t->name, t->data,
            t->grad, ne[0], ne[1], ne[2], ne[3]);  // cNameOf(t->type)
    n1 = strlen(buf);
}

bool GTensor::Alloc(int tpInit, int flag) {
    assert(szData > 0);
    data = new char[szData];
    if (isParam()) {
        // if(hFish!=nullptr && hFish->isTrain())
        grad = new floatGrad[size()];
    }
    mem_status = 1;
    return true;
}

bool GTensor::Activate(int tpInit, int flag) {
    assert(0 && "Not implemented...");
    return false;
}

// only called@GeQuant::LowBit_worker
bool GTensor::InitShadoW(const void* srcData, bool isGPU, DISTILLATION_CARD& distill, int flag) {
    if (distill.isKeepShadoW()) {
        assert(shadoW == nullptr);
        string desc = name;
        Alloc_1(&shadoW, true, desc + ".w0", szData);
        if (isGPU) {
            assert(0);
        } else
            H2D(shadoW, srcData, szData);
    } else {
        shadoW = ToX(gBUFF->tmpTernary);
    }
    return true;
}

/*in many case, params are not update, even data is not allocated!
bool GTensor::isUpdateParam(int iter, int flag) const {
    if (data == nullptr)
        return false;
    return true;
}*/

bool huTensor::BeforeBackward(size_t& off, int flag) {
    off = 0x0;
    if (hRef != nullptr)
        return true;

    assert(szGrad > 0);
    int m = ne[0], n = ne[1];
    if (BIT_TEST(flags, GTensor::F_TMP_GRAD)) {
        grad = (floatX*)GTensor::buff, off += szGrad;
        // BeforeBackward();
        D20(grad, szGrad);
        assert(off < GTensor::buff_len);
    } else {
        // gW           = ToG(w);
        // dbias_buffer = (float *)GTensor::buff;
    }

    return true;
}

bool GTensor::InitGamaParam(int flag) {
    // Print(name, 0, -1);

    gama_param = nullptr;
    assert(hQuant != nullptr);
    assert(isParam());
    SHAPE sp = hQuant->GroupShapeOfT(this);
    int nG = sp[0], lG = sp[1], nQuant = hQuant->params.default_bits;

    floatGama *zero = gama_T(GTensor::ZERO), *step = gama_T(GTensor::STEP);
    std::ptrdiff_t off = hBITARR(zero) - hBITARR(data);
    string name_1      = string(name) + "_gama<" + std::to_string(nQuant) + ">";
    typNUMBER tpGama   = TYPE_<floatGama>();
    gama_param         = Partial(name_1, static_cast<size_t>(off), {sp[0], 2}, tpGama, flag);
    gama_param->grad   = grad;
    gama_param->gm     = gm;
    gama_param->gv     = gv;

    assert(needUpdateParam);
    gama_param->needUpdateParam = true;
    needUpdateParam             = false;

    size_t nEle = gama_param->size();
    assert(nEle == nG * 2 && nEle <= size());
    assert(sizeof(floatX) * nEle <= szM && "Failed to share gm,gv & grad");  // share gm,gv & grad
    gama_param->szM = sizeof(floatX) * nEle;
    assert(gama_param->szM <= szM);
    gama_param->szV = sizeof(floatX) * nEle;
    assert(gama_param->szV <= szV);
    gama_param->szGrad = sizeof(floatX) * nEle;

    assert(gama_param->hQuant == nullptr);
    gama_param->hQuant = hQuant;
    BIT_SET(gama_param->flags, F_GAMA);

    assert(gama_param->hRef == nullptr);
    gama_param->hRef = shared_from_this();
    // if(G_Has_(name,{"model.layer.self_attn.o_proj"})){

    // gama_param->Print(gama_param->name, 3, -1, nEle);
    // }
    return true;
}

/**
 * Train
 *      1. AfterBuild->Prepare->InitGUOKE->ManegeMemory-> Alloc each of Neuron::PickGensors
 *
 * Evaluate/Chat
 *      1. AfterBuild->SafeTensors_Serialize->LoadParam_
 *      2.
 */
bool huTensor::Activate(int iter, int flagInit) { return Alloc(iter, flagInit); }

bool huTensor::Alloc(int iter, int flagInit) {
    assert(strlen(name) > 0);
    if (G_Has_(name, {"model.layers.27.mlp.down_proj.weight"})) {  // model.layers.0.mlp.down_proj.weight qzeros scales model.layers.5.self_attn.v_proj.weight
        DEBUG_HERE;                                                //
    }
    size_t sz0 = szGlobalMaloc;
    if (BIT_TEST(flags, F_NOALLOC))  // For example: operator fusing, memory reuse,rematerialization
        return true;
    if (BIT_TEST(flags, F_MMAP))  // For example: operator fusing, memory reuse,rematerialization
        return true;
    if (hRef != nullptr) {  // Activation or Parameters
        // if (DUMP(0))
        if (BIT_TEST(flags, GTensor::F_RELOAD)) {
            if (hFish->config.dumpSwitch.tensor_ref > 0)
                _INFO("\t%s =====> %s\n", name, hRef->name);
        } else {
            ShareMemory(hRef);  //  grad => src->grad;
            return true;
        }
    }

    assert(szData > 0 || type == typNUMBER::T_BINARY_TILE);
    bool hostAlloc = BIT_TEST(flags, F_HOSTALLOC);
    bool isTrain   = hFish != nullptr && hFish->isTrain();
    if (BIT_TEST(flags, F_HOSTDATA) && host_data == nullptr) {
        host_data = new char[szData];
    }
    bool allocData = data == nullptr;
    if (isBackGama) {
        assert(szM > 0 && szV > 0 && szGrad > 0);
    } else {
        szGrad = sizeof(floatGrad) * size();
    }

    string desc = name;
    if (allocData) {
        if (type == typNUMBER::T_BINARY_TILE) {
            size_t nTile = (size_t)(CEIL_DIV(ne[0], THREAD_TILE_M) * CEIL_DIV(ne[1], THREAD_TILE_N));
            szGama       = sizeof(floatGama) * nTile;
            // Alloc_1((void **)(&gama_T), false, sizeof(float) * nTile);
            szM /= THREAD_TILE_M * THREAD_TILE_N;
            szV = szM;
        } else if (type == typNUMBER::Q4 || type == typNUMBER::Q3 || type == typNUMBER::Q2) {
            // szGama is set @GeQuant::RTN_x, ...
            if (!isTrain) {  // only used at infer stage
                szM = 0, szV = 0;
            }
        } /*else if (BIT_TEST(flags, F_TERNARY)) {
            szGama = sizeof(floatGama) * ne[0];
            // Alloc_1((void **)(&gama_T), false, sizeof(float) * ne[0]);
        }*/
        string suffix = isParam() ? ".w" : ".a";               //  weight or activation
        Alloc_1(&data, true, desc + suffix, szData + szGama);  //  1048576+98560
        // if(hQuant!=nullptr)
        //     hQuant->ExTensor(this);

        raw_data = data;
    }

    if (isParam()) {
        if (allocData) {
            InitParam(flagInit);
            BIT_SET(flags, GTensor::F_RESIDENT);  //  guoke ???
        }

        if (isTrain) {
            if (grad == nullptr) {
                if (isBackGama) {
                    SHAPE sp  = hQuant->GroupShapeOfT(this);  //  @InitGamaParam_
                    int nGama = sp[0] * 2;
                    assert(nGama > 0 && nGama < size());
                    szM = sizeof(floatMV) * nGama, szV = sizeof(floatMV) * nGama;
                } else {
                    szM = sizeof(floatMV) * size(), szV = sizeof(floatMV) * size();
                }
                if (BIT_TEST(flags, F_TMP_GRAD)) {  //
                    grad = nullptr;
                } else {
                    Alloc_1((void**)(&grad), true, desc + ".g", szGrad);  // sgd_kernel would zero grad!
                }
                string method = hFish->config.Get({"train", "optimizatioin", "method"}, string("adamw"), false);
                if (method == "adamw") {
                    Alloc_1(&gm, true, desc + ".m", szM + szV), gv = (char*)gm + szM;
                } else if (method == "lion") {
                    Alloc_1(&gm, true, desc + ".m", szM), szV = 0;
                } else if (method == "muon") {
                    if (hFish->config.common.muon.isAdamW(this)) {
                        Alloc_1(&gm, true, desc + ".m", szM + szV), gv = (char*)gm + szM;
                    } else {
                        assert(szM > 0 && "The size of gm should > 0");
                        Alloc_1(&gm, true, desc + ".m", szM), szV = 0;
                    }
                    // Alloc_1(&gm, true, desc+".m", szMV);
                } else if (method == "adams") {  // why converge so slow for 1445M?
                    Alloc_1(&gm, true, desc + ".m", szM), szV = 0;
                } else {
                    Alloc_1(&gv, true, desc + ".m", szM), szV = 0;
                }
                assert(gm != nullptr && "gm is nullptr@huTensor::Alloc");
            }
        } else {  // infer or generate stage
            szV = 0, szGrad = 0, szM = 0;
        }
    } else {
        szM = 0, szV = 0;
    }
    szUse = szGlobalMaloc - sz0;
    // assert(szGlobalMaloc - sz0 <= mostMemory());
    if (iter <= 1000 /*&& */) {
        string sA = hostAlloc ? "HostAlloc" : "cudaMalloc";
        if (hFish->isRemater()) {
            sA = "Remater";
        }
        if (ne[0] == 262144 || ne[1] == 151936) {
            int isDebug = 0;
        }
        if (szGlobalMaloc - sz0 >= SUM::nMinTensorAlloc || type == typNUMBER::T_SIGN) {  // 100 * 1.0e6
            if (G_Has_(name, {"self_attn.q_proj.weight"})) {                             // model.layers.0.mlp.down_proj.weight qzeros scales
                DEBUG_HERE;                                                              //
            }
            _INFO("\t %s=%gM@%s type=%s shape=[%ld,%ld,%ld,%ld]%s alloc_sum=%gG\n", sA.c_str(), (szGlobalMaloc - sz0) * 1.0f / 1.0e6, name, cNameOf(type),
                  ne[0], ne[1], ne[2], ne[3], grad != nullptr ? "x2" : "", szGlobalMaloc * 1.0 / 1.0e9);
        }
    }
    mem_status = 1;
    return true;
}
bool huTensor::Free(bool isPassResident) {
    try {
        if (G_Has_(name, {"tmpDelta2", "tmpOutL"})) {
            DEBUG_HERE;
        }
        if (isParam()) {
            // GeNeuron::ManageMemory would call free many times
        }

        if (isRefer()) {
            if (BIT_TEST(flags, GTensor::F_RELOAD)) {
                DEBUG_HERE;
            } else
                return true;
        }
        bool isPass = isPassResident && BIT_TEST(flags, GTensor::F_RESIDENT);
        size_t sz0  = szGlobalMaloc;
        if (data != nullptr) {
            if (isPass) {
                int pass = 0;
            } else {
                Free_1(&data);
                // if (gama_T != nullptr)     Free_1((void **)(&gama_T));
                // _INFO("\t%s (-%.3gM)\n",name,(sz0-szGlobalMaloc)/1.0e6);
            }
        } else {
            assert(grad == nullptr);
            return true;
        }
        if (!isPass) {  //! BIT_TEST(flags, GTensor::F_RESIDENT)
            if (grad != nullptr)
                Free_1((void**)(&grad), "_grad");
            if (gm != nullptr)
                Free_1((void**)(&gm), "_m");
        }

        // _INFO("\t%s freed(%.3gM)!",name,(sz0-szGlobalMaloc)/1.0e6);
    } catch (...) {
        assert(0);
    }
    return true;
}

bool Gensors2File(std::vector<hGensor> gensors, const std::string& path, int flag) {
    /*FILE* logFile = freopen(path.c_str(), "w", stderr);
    if (!logFile) {
        perror("Failed to redirect stderr");
        return false;
    }*/
    _INFO("\n>>>> %ld Gensors to File@%s", gensors.size(), path.c_str());
    for (auto t : gensors) {
        t->DumpX(0x0);
    }

    // If you want to restore it back to the console, you typically need ​​low-level file descriptor redirection (dup2)​​, which is ​​POSIX-specific
    // (Linux/macOS)​​. Not available in pure standard C++.
    return true;
}

hGTensor huTensor::Partial(const string& name_, size_t szOff, const SHAPE shape, typNUMBER tyP, int flag) {
    if (tyP == typNUMBER::T_OTHER)
        tyP = type;
    hGTensor sub = GT(hFish, tyP, shape);
    if (!name_.empty())
        snprintf(sub->name, sizeof(name), "%s", name_.c_str());
    else
        sub->name[0] = '\0';
    size_t sz1 = sub->nByte(), sz2 = sizeof(floatGama) * (ne[0] + ne[1]), szAll = nByte() + szGama;
    if (szOff + sz1 > szAll) {
        _ERROR("Out of size! szOff=%ld sub=%ld most=%ld. x=%ld", szOff, sz1, szAll, szOff + sz1 - szAll);
        K_EXIT(KOIFISH_TENSOR_PARTIAL);
    }
    // assert(BitPE(type) == 8 || BitPE(type) == 16);
    // int nB    = (int)(BitPE(type) / 8);
    sub->data = (hBITARR)data + szOff;  // nOff * nB;
    if (grad == nullptr) {
        sub->grad = nullptr;
    } else
        sub->grad = grad + szOff;
    sub->flags = flags;
    BIT_SET(sub->flags, F_ONLYREF);
    return sub;
}

/*
1. RAII (Resource Acquisition Is Initialization)
2. Throw exceptions from constructors​ if resource acquisition fails

class SafeCUDAMemory {
private:
    void* ptr;

public:
    SafeCUDAMemory(size_t size) : ptr(nullptr) {
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc failed");
        }
    }

    ~SafeCUDAMemory() {
        if (ptr != nullptr) {
            // Note: Don't throw from destructor!
            cudaFree(ptr);
        }
    }

    // Prevent copying
    SafeCUDAMemory(const SafeCUDAMemory&) = delete;
    SafeCUDAMemory& operator=(const SafeCUDAMemory&) = delete;

    // Allow moving
    SafeCUDAMemory(SafeCUDAMemory&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    void* get() const { return ptr; }
};
*/