
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
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
#include "GeQuant.hpp"

GTensor* GTensor::tZ   = nullptr;
hGTensor GTensor::outL = nullptr, GTensor::delta = nullptr, GTensor::gate_delta = nullptr, GTensor::tmpDelta = nullptr;
float* GTensor::stat_info = nullptr;
hGTensor GTensor::bt4c = nullptr, GTensor::scratch = nullptr, GTensor::tmpW = nullptr, GTensor::tmpGW = nullptr, GTensor::tmpFF1 = nullptr,
         GTensor::tmpTernary = nullptr, GTensor::tmpQout = nullptr, GTensor::tmpKout = nullptr, GTensor::residual = nullptr;
void *GTensor::buff = nullptr, *GTensor::host_buff = nullptr, *GTensor::cudnn_workspace = nullptr;
size_t GTensor::buff_len = 0, GTensor::cudnn_workspace_size = 0;

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
        Alloc();
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
    if (hQuant != nullptr && hQuant->isCPU) {
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

bool GTensor::SetTernary(typNUMBER tpT_, int flag) {
    if (type == tpT_)
        return true;
    BIT_SET(flags, GTensor::F_TERNARY);
    tpQuant = W_SCALE;
    if (DEBUG.T_ternary == 1) {  //  only for debug
        type = tpT_;
        return true;
    }
    for (auto t : refered) {
        t->type = tpT_;
    }
    return ReShape(shape, tpT_);
}

float GTensor::Get(int i, int flag) const {
    assert(0);
    return 0.f;
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
/*
   Only for gguf-serialize
*/
struct ggml_tensor* GTensor::GG() {
#ifdef __USE_GGML__
    ggml_tensor* hgg = (ggml_tensor*)gg;
    if (hgg == nullptr) {
        hgg = new ggml_tensor();

        *hgg = (struct ggml_tensor){
            // @ggml_new_tensor_impl
            /*.type         =*/type,
            /*.backend      =*/GGML_BACKEND_TYPE_CPU,
            /*.buffer       =*/NULL,
            /*.ne           =*/{ne[0], ne[1], ne[2], ne[3]},
            /*.nb           =*/{nb[0], nb[1], nb[2], nb[3]},
            /*.op           =*/GGML_OP_NONE,
            /*.op_params    =*/{0},
            /*.flags        =*/flags,
            /*.grad         =*/NULL,
            /*.src          =*/{NULL},
            /*.view_src     =*/view_src,
            /*.view_offs    =*/view_offs,
            /*.data         =*/data,
            /*.name         =*/{0},
            /*.extra        =*/NULL,
            ///*.padding      =*/ { 0 },
        };

        hgg->data = new char[szData];
        memcpy(hgg->name, name, sizeof(char) * GGML_MAX_NAME);
    }
    size_t sz = ggml_nbytes(hgg);  // 154389504
    assert(sz == szData);
#ifdef _TENSOR_G_
    bool toHost = SerialGP(hgg->data, nullptr, true, 0x0);
    assert(toHost);
#endif
    assert(isParam());
    gg = hgg;
    return hgg;
#else
    return nullptr;
#endif
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
        FREE_a(host_data);
    }
}

bool GTensor::isAtHost() const {
    bool isGPU  = BIT_TEST(flags, F_GPU);
    bool isHost = BIT_TEST(flags, F_HOSTALLOC) || BIT_TEST(flags, F_MMAP);
    return isHost;
}
// void GTensor::AddSrc(const hGOP t,int type,int flag)           {
//    assert(t!=nullptr); src.push_back(t);
// }
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
        /*if (isMiss) {   @NN2NAME
            size_t pos = 0;
            if (arch == MODEL_ARCH::NLP_QWEN2_) {    //some hack for mismatch of name
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
            }
        }*/
        if (isMiss) {
            _ERROR("Failed to get tensor=%s nGensor=%d\n", name.c_str(), nag.size());
            return nullptr;
        }
        return nag[key];
    }  //  model.layers.0.input_layernorm.weight
}
void ToChebyshev(int N, float* rows, int flag = 0x0);
/*
    parse_tensor
    device_to_file   using double buffering running on the given stream.
*/
int GTensor::SerialJSON(const std::string& name_, const JSON& val, void* bytes_ptr, size_t bytes_size, int flag) {
    if (G_Has_(name, {"model.embed_tokens.weight"})) {  //  "tokenizer.tokens"    "model.embed_tokens.weight"
        DEBUG_HERE;
    }
    if (strcmp(name, name_.c_str()) != 0) {
        strcpy(name, name_.c_str());
    }
    std::string dtype_str = val.value("dtype", "");
    typNUMBER tpMMP       = tpNumOf(dtype_str);
    if (tpMMP == typNUMBER::I32 && hQuant != nullptr) {
        tpMMP = hQuant->bit2typ();
    }
    SHAPE spJ;  // JSON2SHAPE()
    size_t numel = 1;
    if (val.at("shape").size() > 4) {
        std::cerr << "shape exceeds 4 dimensions" << std::endl;
    }
    for (size_t i = 0; i < val.at("shape").size() && i < 4; i++) {
        if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
            std::cerr << "bad shape" << std::endl;
            return -2;
        }
        int n = val.at("shape")[i].get<int>();
        spJ.push_back(n);  // shape[i] =
        numel *= spJ[i];
    }
    // ReShape(spJ, tpMMP);
    if (SHAPE2NZ(shape) != SHAPE2NZ(spJ)) {  // quant
        // assert(hQuant!=nullptr);
    }
    ReShape(shape, tpMMP);

    if (val.at("data_offsets").size() != 2) {
        return -3;
    }
    size_t offset_start = static_cast<size_t>(val.at("data_offsets")[0]);  // 1544148992
    size_t offset_end   = static_cast<size_t>(val.at("data_offsets")[1]);  // 1545276932
    if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
        std::cerr << "bad offsets" << std::endl;
        return -1;
    }
    size_t szSrc = offset_end - offset_start;
    // validate the shape matches the size
    if (szData != szSrc && szData * 3 != szSrc) {
        _WARN("GTensor::SerialJSON failed! size mismach[%ld!=%ld] @%s", szData * 3, szSrc, name);
        // std::cerr << "bad size" << std::endl;
        return -1;
    }

    void* src = (char*)bytes_ptr + offset_start;

    // if(G_Has_(name,{"layers.27.mlp.weight"})){   //only for debug    815288320
    //    float *rows = new float[ne[0]];
    //    T2Float_arr(ne[0],(f8e5*)src,rows);
    //    ToChebyshev(ne[0],rows);
    //    delete[] rows;
    //    // PrintTensor<f8e5>(name,(f8e5*)src,ne[0],ne[1]);
    // }
    if (BIT_TEST(flag, F_NOALLOC)) {  //  "tokenizer.tokens","tokenizer.scores"
        data = src;                   // ((char*)(src))[szSrc-1]    (char*)bytes_ptr + offset_end-1
        BIT_SET(flags, F_MMAP);
    } else {
        if (data != nullptr) {
            SerialGP(src, nullptr, szSrc, false);
        } else {
            host_data = src;
            if (DEBUG.T_cpu == 1) {
                data = src;
                BIT_SET(flags, F_MMAP);
            }
            if (!hFish->isTrain()) {  // otherwize, mmap file is free & host_data is invalid
                tpInit = W_SKIP;
                Alloc(-1, flag);
                Serial_Quant_MMAP(false, false);
                host_data = nullptr;  // mmap file would release
            } else {
                G_NORM_STAT<bf16>(size(), (bf16*)host_data, disq.sum_2, disq.sum_1, disq.nrm_1);
            }
            // if (G_Has_(name, hFish->config.quant.filter_WeightF8Ex)) {  // model.embed_tokens.weight    only for debug
            //     ToF8Ex(0x0);
            // }
        }
    }
    if (strlen(name) > 0 && flag > 0)
        DumpX(0);
    return 0;
}

void GTensor::Print(const string& title0, int x, int flag, size_t nEle) const {
    if (g_dump_level > 0 && flag >= 0)
        return;
    assert(nEle >= 0);
    bool isDevice = !isAtHost();
    void *src = x == 3 ? gv : x == 2 ? gm : x == 1 ? grad : data, *hData = nullptr;
    string suffix = x == 3 ? "GV_" : x == 2 ? "GM_" : x == 1 ? "GRAD_" : "";
    if (x == 4) {
        assert(host_data != nullptr);
        src      = host_data;
        isDevice = false;
    }
    if (src == nullptr) {
        _INFO("Failed to print! %s of \"%s\" is nullptr!", x == 3 ? "gv" : x == 2 ? "gm" : x == 1 ? "grad" : "data", name);
        return;
    }
    size_t szHost = nEle > 0 ? (nEle * BitPE(type)) / 8 : szData + szGama;
    if (isDevice) {
        SYNC_DEVICE();
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
    floatGama *rNormal = (floatGama*)((char*)data + szData), *cNormal = rNormal + shape[0];
    floatGama* gamaQ = cNormal + shape[1];
    switch (type) {
        case typNUMBER::T_BINARY:
        case typNUMBER::T_BINARY_3:
        case typNUMBER::T_BINARY_TILE:
            assert(0);
            break;
        case typNUMBER::Q4:
            PrintQ4("_Q4", (hBITARR)src, ne[0], ne[1], ne[2], ne[3], flag);
            if (disq.rc_normal > 0) {
                PrintTensor<floatGama>("_rNormal", rNormal, true, ne[0], 1, 1, 1, flag);
                PrintTensor<floatGama>("_cNormal", cNormal, true, ne[1], 1, 1, 1, flag);
            }
            PrintTensor<floatGama>("_gamaQ", gamaQ, true, ne[0], 2, 1, 1, flag);
            break;
        case typNUMBER::Q3:
            // PrintQ4("_Q3", (hBITARR)src, ne[0], ne[1], ne[2], ne[3], flag);
            if (disq.rc_normal > 0) {
                PrintTensor<floatGama>("_rNormal", rNormal, true, ne[0], 1, 1, 1, flag);
                PrintTensor<floatGama>("_cNormal", cNormal, true, ne[1], 1, 1, 1, flag);
            }
            PrintTensor<floatGama>("_gamaQ", gamaQ, true, ne[0], 2, 1, 1, flag);
            break;
        case typNUMBER::Q2:
            // PrintQ2("_Q2", (hBITARR)src, ne[0], ne[1], ne[2], ne[3], flag);
            if (disq.rc_normal > 0) {
                PrintTensor<floatGama>("_rNormal", rNormal, true, ne[0], 1, 1, 1, flag);
                PrintTensor<floatGama>("_cNormal", cNormal, true, ne[1], 1, 1, 1, flag);
            }
            PrintTensor<floatGama>("_gamaQ", gamaQ, true, ne[0], 2, 1, 1, flag);
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
            _INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n", i, ne[0], ne[1], "", name);  //  ggml_op_name(op),
            break;
        default:
            if (type != typNUMBER::F32 && n > 0) {
                fdata = new float[nElems];
                if (type == typNUMBER::F16) {
                    /*ggml_fp16_t *src_ = (ggml_fp16_t *)(data);
                    for (int i = 0; i < nElems; i++) {
                        fdata[i] = src_[i];
                    }*/
                } else {  // need dequant
                    fdata = nullptr;
                    n     = 0;
                }
            }
            double sum = 0.0;
            if (fdata != nullptr && !BIT_TEST(flags, F_GPU)) {
                for (i = 0; i < nElems; i++) {
                    sum += fdata[i];
                    if (fdata[i] == 0)
                        nz++;
                    a1 = std::max(a1, fdata[i]);
                    a0 = std::min(a0, fdata[i]);
                }
            }
            if (szUse == 0x0) {
                _INFO("\t%s %-36s %-4s", title.c_str(), name, A);
                _WARN0(" NO ALLOC ");
                _INFO(" \t[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %s] \n", ne[0], ne[1], ne[2], ne[3], cNameOf(type));
            } else
                _INFO("\t%s %-36s %-4s szAlloc=%6gM\t[%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %s] ", title.c_str(), name, A, szUse / 1.0e6, ne[0],
                      ne[1], ne[2], ne[3], cNameOf(type));
            if (disq.err > 0)
                _INFO("eQ=%.3g ", disq.err);
            _INFO("\n");
            if (n > 0 && a1 != -FLT_MAX) {
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
            }

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

/*in many case, params are not update, even data is not allocated!
bool GTensor::isUpdateParam(int iter, int flag) const {
    if (data == nullptr)
        return false;
    return true;
}*/

// return scaling/LUT of quant weight
floatGama* GTensor::gama_T(GAMA_TYPE type, int row) {
    if (hRef != nullptr)
        return hRef->gama_T();
    assert(data != nullptr);
    floatGama* gama_0 = reinterpret_cast<floatGama*>((hBITARR)data + szData);
    int nRow = ne[0], nCol = ne[1];
    switch (type) {
        case R_SCALE:
            return gama_0;
        case C_SCALE:
            return gama_0 + nRow;
        case ZERO:
            return gama_0 + nRow + nCol;
        case STEP:
            return gama_0 + nRow + nCol + nRow;
        case LUT:
            return gama_0 + nRow + nCol;
        default:
            assert(0);
            break;
    }

    return gama_0;
}

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

/**
 * Train
 *      1. AfterBuild->Prepare->InitGUOKE->ManegeMemory-> Alloc each of Neuron::PickGensors
 *
 * Evaluate/Chat
 *      1. AfterBuild->HF_Serialize->SerialJSON
 *      2.
 */
bool huTensor::Alloc(int iter, int flagInit) {
    if (G_Has_(name, {"preLogits"})) {  // model.layers.0.mlp.down_proj.weight
        DEBUG_HERE;                            //
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
    szM = sizeof(floatMV) * size(), szV = sizeof(floatMV) * size();
    szGrad = sizeof(floatGrad) * size(), szGama = 0x0;
    string desc = name;
    if (allocData) {
        if (type == typNUMBER::T_BINARY_TILE) {
            size_t nTile = (size_t)(CEIL_DIV(ne[0], THREAD_TILE_M) * CEIL_DIV(ne[1], THREAD_TILE_N));
            szGama       = sizeof(floatGama) * nTile;
            // Alloc_1((void **)(&gama_T), false, sizeof(float) * nTile);
            szM /= THREAD_TILE_M * THREAD_TILE_N;
            szV = szM;
        } else if (type == typNUMBER::Q4 || type == typNUMBER::Q3 || type == typNUMBER::Q2) {
            int bits = BitPE(type), nQuant = 1 << bits;
            szGama = sizeof(floatGama) * (shape[0] * nQuant + shape[0] + shape[1]);  // 36/1024=0.035
            szM = 0, szV = 0;                                                        // only used at infer stage
        } else if (BIT_TEST(flags, F_TERNARY)) {
            szGama = sizeof(floatGama) * ne[0];
            // Alloc_1((void **)(&gama_T), false, sizeof(float) * ne[0]);
        }
        string suffix = isParam() ? ".w" : ".a";  //  weight or activation
        Alloc_1(&data, true, desc + suffix, szData + szGama);
        if (hQuant != nullptr) {
            // SHAPE spQ, spS;
            // qScale = this->Partial(".qzeros", szData, spQ);
            // qZero  = this->Partial(".qzeros", szData + qScale->nByte(), spS);
        }

        raw_data = data;
    }

    if (isParam()) {
        if (allocData) {
            InitParam(flagInit);
            BIT_SET(flags, GTensor::F_RESIDENT);  //  guoke ???
        }

        if (grad == nullptr && isTrain) {
            if (BIT_TEST(flags, F_TMP_GRAD)) {  // grad_ref != nullptrgrad = ToX(grad_ref);
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
                    Alloc_1(&gm, true, desc + ".m", szM), szV = 0;
                }
                // Alloc_1(&gm, true, desc+".m", szMV);
            } else if (method == "adams") {  // why converge so slow for 1445M?
                                             /*if(isStrMatch(name, {"embd","output","norm"})){
                                                 Alloc_1(&gm, true, desc+".m", szMV * 2), gv = (char *)gm + szMV;
                                             }else*/
                Alloc_1(&gm, true, desc + ".m", szM), szV = 0;
            } else {
                Alloc_1(&gv, true, desc + ".m", szM), szV = 0;
            }
            assert(gm != nullptr && "gm is nullptr@huTensor::Alloc");
        } else {
            szV = 0, szGrad = 0, szM = 0;
        }
    } else {
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

bool GTensor::FreeBuffer(int flag) {
    try {
        bt4c = nullptr, delta = nullptr, tmpDelta = nullptr, outL = nullptr, scratch = nullptr, tmpFF1 = nullptr, tmpW = nullptr, tmpGW = nullptr,
        residual   = nullptr;
        tmpTernary = nullptr;
        return true;
    } catch (const std::exception& e) {
        _WARN("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char* info) {
        _WARN("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _WARN("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}