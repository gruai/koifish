
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Che
 */
#include "GTensor.hpp"

#include "../Manifold/Fish.hpp"
#include "../ggex/GG_util.hpp"

hGTensor GTensor::outL = nullptr, GTensor::delta = nullptr, GTensor::tmpDelta = nullptr;
;
hGTensor GTensor::bt4c = nullptr, GTensor::scratch = nullptr, GTensor::tmpFF1 = nullptr, GTensor::tmpW = nullptr, GTensor::tmpGW = nullptr,
         GTensor::residual = nullptr;
void *GTensor::buff = nullptr, *GTensor::host_buff = nullptr;

float GTensor::rLARS(float s0, float T_lars, int flag) {
    if (shape.size() <= 1)
        return s0;

    float eps         = 1.0e-8;
    float trust_ratio = wnorm / (gnorm + eps);
    trust_ratio       = std::min(trust_ratio, T_lars);
    float r           = trust_ratio;
    return r;
}

GTensor::GTensor(Fish *hFis_, SHAPE shape_, typNUMBER tpD_, bool isX, int flag) : hFish(hFis_), flags(flag) { ReShape(shape_, tpD_, flag); }

hGTensor GT(SHAPE shape_, void *src, typNUMBER tpD_, int flag) {
    hGTensor t = std::make_shared<GTensor>(nullptr, shape_, tpD_, true, 0x0);
    t->Alloc();
    assert(0);  // memcpy is dangerous
    memcpy(t->data, src, t->nByte());
    assert(!t->isEmpty());
    return t;
}

hGTensor GT(Fish *hFish, typNUMBER type, SHAPE shape, int flag, const string &name) {
    hGensor hT = std::make_shared<huTensor>(hFish, name, shape, type, false, flag);
    return hT;
}

//  https://stackoverflow.com/questions/16468440/how-to-split-class-definition-between-multiple-cpp-and-cu-files
// hGensor TENSO(void* ctx0,typNUMBER typ,SHAPE shape,int flag,const string&name ) {
//    auto type = (typNUMBER)(typ);
//    hGensor hT = std::make_shared<huTensor>((Fish*)ctx0,name,shape,type,false,flag);
//    return hT;
// }

bool GTensor::ReShape(SHAPE shape_, typNUMBER tpD_, int falg) {
    if (type == tpD_ && shape == shape_)
        return true;

    shape = shape_;
    type  = tpD_;
    int i = 0;
    for (auto n : shape) {
        ne[i++] = n;
        assert(n > 0 && "");
    }
    for (i = shape.size(); i < N_DIMS; i++) ne[i] = 1;

    // from ggml_new_tensor
    size_t szBlk = NPBlck(type);
    nb[0]        = BPE(type);
    assert(szBlk == 1);
    nb[1] = nb[0] * (ne[0] / szBlk);
    for (int i = 2; i < N_DIMS; i++) {
        nb[i] = nb[i - 1] * ne[i - 1];
    }

    double nB = BPE(type);
    assert(nB >= 1.0);
    szData = size() * nB;

    if (data != nullptr) {
        Free();
        Alloc();
    }
    // _INFO();
    return true;
}

float GTensor::Get(int i, int flag) const {
    assert(0);
    return 0.f;
    // return ggml_get_f32_1d(gg, i);
}
/*
   Only for gguf-serialize
*/
struct ggml_tensor *GTensor::GG() {
#ifdef __USE_GGML__
    ggml_tensor *hgg = (ggml_tensor *)gg;
    if (hgg == nullptr) {
        hgg = new ggml_tensor();
#ifdef GG_V12
#else
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
#endif
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

    FREE_a(host_data);
}

bool GTensor::isAtHost() const {
    bool isGPU  = BIT_TEST(flags, F_GPU);
    bool isHost = BIT_TEST(flags, F_HOSTALLOC) || BIT_TEST(flags, F_MMAP);
    return isHost;
}
// void GTensor::AddSrc(const hGOP t,int type,int flag)           {
//    assert(t!=nullptr); src.push_back(t);
// }
void GTensor::AddSrc(const vector<hGTensor> &ts, int flag) {
    for (auto t : ts) {
        if (t == nullptr)
            continue;
        hGOP hop = std::make_shared<GENSOR_OP>(t);
        // AddSrc(hop,0x0);
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
    // ggml_set_f32(gg, 1.0f);
}
bool GTensor::OverWrite(struct ggml_tensor *gg_, bool isSrc, int flag) {
#ifdef __USE_GGML__
    assert(size() == ggml_nelements(gg_));
    assert(type == (typNUMBER)gg_->type);
    if (isSrc) {
        memcpy(data, gg_->data, szData);
    } else {
        memcpy(gg_->data, data, szData);
    }
#endif
    return true;
}

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

bool GTensor::ShareWeight(hGTensor src, int flag) {
    assert(src != nullptr && src->data != nullptr);
    data = src->data;

    if (src->grad != nullptr) {
        grad = src->grad;
        gm   = src->gm;
        gv   = src->gv;
    }
    return true;
}

hGTensor GTensor::Relu() {
    // auto cur=ggml_relu(nullptr, (struct ggml_tensor *)gg);  return NEW_(cur);
    return nullptr;
}
hGTensor GTensor::Silu() {
    // auto cur=ggml_silu(nullptr, (struct ggml_tensor *)gg);  return NEW_(cur);
    return nullptr;
}
hGTensor GTensor::Norm(float epsilon, int flag) { return nullptr; }

hGTensor GTensor::CrossEntropy(const hGTensor b, int flag) {
    // auto cur = ggml_cross_entropy_loss(nullptr,(struct ggml_tensor *)gg, b->GG() );      // ggml_cross_entropy_loss_1(_ctx, cur, target_probs);
    // return GTensor::NEW_(cur);
    return nullptr;
}

hGTensor GTensor::GetRow(hGTensor, hGTensor tokens, hGTensor pos, int flag) {
    assert(0);  // GGML VERSION
    // assert(ne[1]==shape[0]);
    // struct ggml_tensor *cur = ggml_get_rows(_ctx, gg, tokens->gg);       gTN(cur, name);
    // if(pos!=nullptr)        {
    //    cur = ggml_add(_ctx, cur, pos->gg);
    // }
    // return GTensor::NEW_(cur);
    return nullptr;
}

hGensor GENSORS::Get(const string &name, int flag) {
    if (flag == 0x100) {  //  .weight=>.w
        for (auto ng : nag) {
            if (strstr(name.c_str(), ng.first.c_str()) != NULL) {
                return ng.second;
            }
        }
        return nullptr;
    } else {
        if (nag.find(name) == nag.end()) {
            _ERROR("Failed to get tensor=%s nGensor=%d\n", name.c_str(), nag.size());
            return nullptr;
        }
        return nag[name];
    }
}
void ToChebyshev(int N, float *rows, int flag = 0x0);
/*
    parse_tensor
    device_to_file   using double buffering running on the given stream.
*/
int GTensor::SerialJSON(const std::string &name_, const JSON &val, void *bytes_ptr, size_t bytes_size, int flag) {
    // if(name=="tokenizer.tokens"){
    //    std::cerr << name << std::endl;
    // }
    if (strcmp(name, name_.c_str()) != 0) {
        strcpy(name, name_.c_str());
    }
    std::string dtype_str = val.value("dtype", "");
    SHAPE spJ;
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
    ReShape(spJ, tpNumOf(dtype_str));

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
    if (szData != szSrc) {
        std::cerr << "bad size" << std::endl;
        return -1;
    }

    void *src = (char *)bytes_ptr + offset_start;
    // if(G_Has_(name,{"layers.27.mlp.w1.weight"})){   //only for debug    815288320
    //    float *rows = new float[ne[0]];
    //    T2Float_arr(ne[0],(f8e5m2_t*)src,rows);
    //    ToChebyshev(ne[0],rows);
    //    delete[] rows;
    //    // PrintTensor<f8e5m2_t>(name,(f8e5m2_t*)src,ne[0],ne[1]);
    // }
    if (BIT_TEST(flag, F_NOALLOC)) {
        data = src;  // ((char*)(src))[szSrc-1]    (char*)bytes_ptr + offset_end-1
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
        }
    }
    if (strlen(name) > 0 && flag > 0)
        Dump(0);
    return 0;
}

void GTensor::Print(const string &title0, int x, int flag, size_t nEle) const {
    assert(nEle >= 0);
    bool isDevice = true;

    void *src    = x == 1 ? grad : data;
    string title = title0;
    if (x == 1)
        title = "GRAD_" + title;
    int64_t sp[N_DIMS] = {ne[0], ne[1], ne[2], ne[3]};
    if (nEle > 0 && nEle != ne[0] * ne[1] * ne[2] * ne[3]) {
        sp[0] = nEle, sp[1] = 1;
        sp[2] = 1;
        sp[3] = 1;
    }
    switch (type) {
        case typNUMBER::F16:
            PrintTensor<half>(title.c_str(), (half *)src, isDevice, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        case typNUMBER::BF16:
            PrintTensor<__nv_bfloat16>(title.c_str(), (__nv_bfloat16 *)src, isDevice, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        case typNUMBER::F32:
            PrintTensor<float>(title.c_str(), (float *)src, isDevice, sp[0], sp[1], sp[2], sp[3], flag);
            break;
        default:
            assert(0);
            break;
    }
}

bool GTensor::Dump(int tpDump, const string &title, int flag) const {
    size_t nz = 0, nElems = size(), i = 0, n = 10;
    float *fdata = (float *)data, a1 = -FLT_MAX, a0 = FLT_MAX;
    const char *A = "d";
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
            printf("\t%s %s %s \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n", title.c_str(), name, A, ne[0], ne[1], ne[2], ne[3],
                   cNameOf(type));
            if (n > 0 && a1 != -FLT_MAX) {
                printf("\nsum=%g data=[%f : %f] rZ=%.3g%%\n\t", sum, a0, a1, nz * 100.0 / nElems);
                for (int i = 0; i < std::min((size_t)(ne[0] * ne[1]), n); i++) {
                    printf("%.5f ", fdata[i]);
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

void _T_repr_(hGensor t, const char *tab, char *buf, const GENSOR_INFO &info) {
    if (t == nullptr)
        return;
    const char *A = "d";
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
    sprintf(buf + strlen(buf), "%s %s %s %s \tdata=%p grad=>%p\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n", tab, sX.c_str(), A, t->name,
            t->data, t->grad, ne[0], ne[1], ne[2], ne[3], cNameOf(t->type));
    n1 = strlen(buf);
}

bool GTensor::Alloc(int tpInit, int flag) {
    assert(szData > 0);
    data = new char[szData];
    if (isParam()) {
        // if(hFish!=nullptr && hFish->isTrain())
        grad = new char[szData];
    }
    return true;
}

bool huTensor::Alloc(int iter, int flagInit) {
    if (strcmp(name, "model.blk.0.attn.wq.weight") == 0 ||
        strcmp(name, "model.embed.weight") == 0) {  //  model.inp_embd.weight       model.out.weight model.embed.weight
        int debug = 0x0;
    }

    size_t sz0 = szMaloc;
    if (BIT_TEST(flags, F_NOALLOC))  // For example: operator fusing, memory reuse,rematerialization
        return true;
    if (BIT_TEST(flags, F_MMAP))  // For example: operator fusing, memory reuse,rematerialization
        return true;
    if (isParam()) {
        // if(tpInit == SERIALIZE)
        //    return true;
    }
    if (hRef != nullptr) {
        ShareWeight(hRef);  //  grad => src->grad;
        if (DUMP(0))
            _INFO("\t%s =====> %s\n", name, hRef->name);
        return true;
    }

    assert(szData > 0);
    bool hostAlloc = BIT_TEST(flags, F_HOSTALLOC);
    bool isTrain   = hFish != nullptr && hFish->isTrain();
    if (BIT_TEST(flags, F_HOSTDATA) && host_data == nullptr) {
        host_data = new char[szData];
    }
    bool allocData = data == nullptr;
    if (allocData) {
        Alloc_1(&data, true);
        // _INFO("\t%s (+%.3gM)\n",name,(szMaloc-sz0)/1.0e6);
    }
    if (isParam() && isTrain) {
        if (allocData) {
            InitParam(flagInit);
            if (1)  // DEBUG.isParamResident
                BIT_SET(flags, GTensor::F_RESIDENT);
        }
        size_t szMV = sizeof(float) * size();
        if (grad == nullptr) {
            Alloc_1(&grad, true, szData + szMV * 2);  // sgd_kernel would zero grad!
            string method = hFish->config.Get({"train", "optimizatioin", "method"}, string("adamw"), false);
            if (method == "adamw") {
                gm = (char*)grad + szData, gv = (char*)gm + szMV;
            } else if (method == "lion") {
                gm = (char*)grad + szData, gv = nullptr;
            } else {
                gm = nullptr, gv = (char*)grad + szMV;
            }
        }
    } else {
    }
    // if(isParam())      {
    //    Alloc_1((void**)(&gama_T),false,sizeof(float)*ne[0]);
    //    float *tmp = new float[ne[0]];
    //    for(int i=0;i<ne[0];i++)   tmp[i]=1;
    //    H2D(gama_T,tmp,sizeof(float)*ne[0]);
    //    delete[] tmp;
    // }
    assert(szMaloc - sz0 <= mostMemory());
    if (iter <= 1 /*&& */) {
        string sA = hostAlloc ? "HostAlloc" : "cudaMalloc";
        if (hFish->isRemater()) {
            sA = "Remater";
        }
        if (ne[0] == 262144 || ne[1] == 151936) {
            int isDebug = 0;
        }
        if (szMaloc - sz0 >= 100 * 1.0e6 || type == typNUMBER::T_SIGN)
            printf("\t %s=%gM@%s type=%s shape=[%ld,%ld,%ld,%ld]%s sum=%gG\n", sA.c_str(), (szMaloc - sz0) * 1.0f / 1.0e6, name, cNameOf(type), ne[0], ne[1],
                   ne[2], ne[3], grad != nullptr ? "x2" : "", szMaloc * 1.0 / 1.0e9);
    }

    return true;
}
bool huTensor::Free(bool isPassResident) {
    try {
        if (isRefer())
            return true;

        size_t sz0 = szMaloc;
        if (data != nullptr) {
            if (isPassResident && BIT_TEST(flags, GTensor::F_RESIDENT)) {
                int pass = 0;
            } else {
                Free_1(&data);
                // _INFO("\t%s (-%.3gM)\n",name,(sz0-szMaloc)/1.0e6);
            }
        } else {
            assert(grad == nullptr);
            return true;
        }
        if (!BIT_TEST(flags, GTensor::F_RESIDENT) && grad != nullptr) {
            Free_1(&grad, "_grad");
        }
        if (gama_T != nullptr)
            Free_1((void **)(&gama_T));
        // _INFO("\t%s freed(%.3gM)!",name,(sz0-szMaloc)/1.0e6);
    } catch (...) {
        assert(0);
    }
    return true;
}

// inline hGensor To4D(struct ggml_context * ctx_build,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4){
//     cur = ggml_reshape_4d(ctx_build, cur, n1, n2,n3,n4);
//     return cur;
// }
// inline hGensor Permute(struct ggml_context * ctx_,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4,bool isCont=true)    {
//     hGensor q = ggml_permute(ctx_, cur, n1,n2,n3,n4);
//     gTN0(q,"%s.#",cur->name);
//     if(isCont)    {
//         q = ggml_cont(ctx_,q);
//         gTN(q,"%s.#c",cur->name);
//     }
//     return q;
// }