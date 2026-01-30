/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */
#pragma once
#include <float.h>
#include <inttypes.h>
#include <stdio.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <thread>
#include <typeinfo>
#include <vector>
using namespace std;
#include <stdio.h>
#include <string.h>

#include "../g_def_x.hpp"

#define GG_V12

class GTensor;
class huTensor;
class Fish;
class SparseNeuron;
class EDGE_DEVICES;
typedef shared_ptr<GTensor> hGTensor;
class GeQuant;
class GeNeuron;
struct GENSOR_INFO;
struct GENSOR_TOPU;

bool Gensors2File(std::vector<hGTensor> gset, const std::string& path, int flag = 0x0);

#include "../Device/CUDA/cuda_common.h"

// set name of a tensor if its name is "\0" & its grad
int gTN(hGTensor, const char* format, ...);
// clear then set name of a tensor & its grad
int gTN0(hGTensor cur, const char* format, ...);

enum DATA_PLACE { VOID, SYMBOLIC, REFER, MEMORY, DEV_MEM, MMAP, DISK, CLOUD, FREE_DEV };

/**
 *  Edge of Operation(TASK) GRAPH
 */
struct GENSOR_OP {
    string sX;
    int level = -1, ID = -1, dad, c_id;
    int op = -1;
    hGTensor _t;

    GENSOR_OP(hGTensor t, int flag = 0x0) : _t(t) { ; }
    static std::shared_ptr<GENSOR_OP> Inst(hGTensor t, int flag = 0x0) { return std::make_shared<GENSOR_OP>(t, flag); }

    string __repr__(string& suffix, string& prefix, int flag = 0x0) const {
        char buf[512] = "\0";
        if (dad == -1) {
            sprintf(buf + strlen(buf), "ROOT");
        } else
            sprintf(buf + strlen(buf), "[%d %d.%d l=%d]", ID, dad, c_id, level);
        return buf;
    }

    static bool comp(GENSOR_OP& a, GENSOR_OP& b) { return a.ID < b.ID; }
};
typedef shared_ptr<GENSOR_OP> hGOP;

#include <cstdlib>
#include <memory>

template <typename T>
struct VirtualAllocator {
    using value_type = T;
    T* allocate(size_t n) { return static_cast<T*>(std::malloc(n * sizeof(T))); }
    void deallocate(T* p, size_t n) { std::free(p); }

    VirtualAllocator() = default;
    template <typename U>
    VirtualAllocator(const VirtualAllocator<U>&) {}
};

template <typename T>
struct GPUAllocator : public VirtualAllocator<T> {};

/*
    distri info of a tensor, transfer data between device & host
    1. related to quantization,    @/home/cys/rnd/lic/src/GBDT/data_fold/Distribution.hpp
    2. sigma = rstd
*/
struct Distri_PIPE {
    int B_ = 0, T_ = 0, C_ = 0;  // For activation, it's number of batch,token(int each batch), channel(of each token)
    double mean = 0, sum_1 = 0.0, sum_2 = 0.0, nrm_1 = 0.0;
    double abs_max = 0;  //  norm_0
    float sigma_   = 0;

    std::vector<float> codebook;  // at most 256 bins
    enum ASYMMETRY {
        NO,
        ZERO_OFF,  //  Zero may not map exactly to any quantized value
        PN_SCALE,  //  use both Positive,Negative scale
    };
    ASYMMETRY asymmetry = ASYMMETRY::NO;
    // (zero,scale) or scaleP/N(Positive,Negative scale)
    union {
        float zero = 0.f;
        float scaleN;
    };
    union {
        float scale = 1.f;
        float scaleP;
    };
    float vmin = FLT_MAX, vmax = -FLT_MAX, e_0 = FLT_MAX, e_1 = -FLT_MAX;
    float maxP, minN;  //  max Positive & min Negative
    float imbalance = 0.0;
    int rc_normal   = 0;
    double err      = 0.0;  // error of quantizer
    string info     = "";   // more info of quantizer

    Distri_PIPE(ASYMMETRY a = ASYMMETRY::NO) : asymmetry(a) {}
    Distri_PIPE(float f1, float f2, int flag = 0x0) : scaleN(f1), scaleP(f2) {}

    template <typename T>
    void Next(T a0) {
        float a = T2Float(&a0);
        vmax = std::max(vmax, a), vmin = std::min(vmin, a);
        abs_max = (double)std::max(fabs(vmax), fabs(vmin));
        nrm_1 += fabs(a);
        sum_1 += a;
        sum_2 += a * a;
    }
    virtual void Prepare(int nQuant, int flag = 0x0);
    virtual BIT_8 X2NormalF(int bits, const float weight, int flag = 0x0);
    virtual float Normal01(const float weight, int flag = 0x0);
};

/**
 * 1.   Support dynamic online change shape & type!
 * 2.   Support quantization
 * 2.   Support BIT representation
 * 3.   Row-major order (first index varies most slowly and last index varies most quickly)
 * 4.   May not contiguous in memory! All tensor operations have to take the stride into account and not assume that the tensor is contiguous in memory!
 */
class GTensor : public std::enable_shared_from_this<GTensor> {
   private:
    void* raw = nullptr;

   protected:
    Fish* hFish = nullptr;
    // include neuron,hquant,...
    GENSOR_INFO* ginfo = nullptr;

    std::vector<GTensor*> refered;
    std::vector<hGTensor> fuyous;
    std::shared_ptr<EDGE_DEVICES> hDevice = nullptr;
    size_t szData = 0, szGama = 0, szUse = 0, szGrad = 0, szM = 0, szV = 0;
    int last_iter = -1;
    //  support dynamic change shape&type!  return false if not change anything
    virtual bool ReShape(SHAPE shape_, typNUMBER tpD_, int flag = 0x0);
    virtual hGTensor _Multiply(const hGTensor& other) {
        assert(0);
        return nullptr;
    }
    int mem_status = -1;
    uint32_t seed = 88888888, param_seed = 0x0;
    enum REF_TYPE {
        NO,
        ALL,
        PARAM,
    };
    REF_TYPE tpRef = REF_TYPE::NO;
    hGTensor hRef  = nullptr;
    void* raw_data = nullptr;

    Distri_PIPE disq;  // distri info related to quantization

    // @GetDynamicQuant
    shared_ptr<GeQuant> hQuant = nullptr;
    hGTensor qZero = nullptr, qScale = nullptr;  // quantization params, may have different meaning/name in different model

    virtual size_t Alloc_1(void** dst, bool isZero, string desc, size_t sz = 0x0, int flag = 0x0) { return 0x0; };
    virtual size_t Free_1(void** obj, const string& info = "") { return 0x0; };

   public:
    static const int MAX_NAME = 64;
    static const int N_DIMS   = 4;

    static size_t szGlobalMaloc;
    static GTensor* tZ;

    // static bool FreeBuffer(int flag = 0x0);
    //  temporary shared memory 1) buff sz>=8*nCTX*nToken(from preLogits)
    static void *buff, *host_buff, *cudnn_workspace;
    // float stat_info[1024] in GPU
    static float* stat_info;
    static size_t buff_len, cudnn_workspace_size;
    float residual_scale = 1.0, wnorm = 0, gnorm = 0;  // some tricks
    float rLARS(float s0, float T_lars, int flag);
    size_t offset = 0x0;
    SHAPE shape;
    SHAPE x_shape;  //  1.padded for high performance or menory alignment(x_shape<=shape)
    // shape=>x_shape
    vector<hGOP> src;
    virtual void AddSrc(const vector<hGTensor>& ts, int flag = 0x0);

    void* host_data = nullptr;  // somtimes, we need data both in device&host
    void* data      = nullptr;
    // a serial of LORA for weight

    //
    enum GAMA_TYPE {
        R_SCALE,
        C_SCALE,
        ZERO,  // zero of RTN
        STEP,  // step of RTN
        LUT,   // values of codebook
    };
    // return scaling/LUT of quant weight
    floatGama* gama_T(GAMA_TYPE type = R_SCALE, int row = 0);
    // virtual bool isUpdateParam(int iter = -1, int flag = 0x0) const;  // in many case, params are not update, even data is not allocated!
    bool needUpdateParam = false;
    int tile_r1 = 0, tile_c1 = 0;  //  tile_r0 = 0,tile_c0 = 0,
    floatGrad* grad   = nullptr;   //
    hGTensor grad_ref = nullptr;
    void *gm = nullptr, *gv = nullptr;  // first moment, second moment of grad

    float info[8];  // Some info of some operations

    virtual void* DataPad(void* src0, int flag = 0x0);

    typNUMBER type;
    INIT_WEIGHT tpInit = INIT_WEIGHT::RANDOM;
    enum BIT_FLAG {
        F_INPUT     = 0x1,
        F_OUTPUT    = 0x2,
        F_PARAM     = 0x4,
        F_LOSS      = 0x8,
        F_WMATRIX   = 0x10,  // A weight matrix is a linear operator on RMS-normed vector spaces.
        F_NOALLOC   = 0x100,
        F_GPU       = 0x200,
        F_HOSTALLOC = 0x400,
        F_MMAP      = 0x800,  // data is part of a memory-mapped file
        F_RESIDENT  = 0x1000,
        F_HOSTDATA  = 0x2000,  // always alloc host_data(may also alloc device data)
        F_RELOAD    = 0x4000,
        F_TOX       = 0x10000,
        F_PADDED    = 0x20000,
        F_ONLYREF   = 0x40000,  // Partial/Sub tensor
        F_TMP_GRAD  = 0x80000,

        F_TERNARY = 0x100000,
        F_LORA_A  = 0x200000,
        F_LORA_B  = 0x400000,

        F_DEBUG = 0x10000000
    };

    enum BIT_OP_FLAG {
        F_OP_NO_REALLOC = 0x10000,
    };

    QUANT_ALG tpQuant = W_SCALE;

    static size_t MostOverhead() { return sizeof(GTensor) * 2; }
    GTensor() {}
    GTensor(Fish* hFish, SHAPE shape_, typNUMBER tpD_, bool isAlloc = true, int flag = 0x0);

    virtual hGTensor Partial(const string& name, size_t offset, SHAPE shape, int flag = 0x0) {
        assert(0);
        return nullptr;
    }
    virtual ~GTensor();

    virtual bool Alloc(int tpInit = 0, int flag = 0x0);
    virtual bool InitParam(int tpInit) {
        assert(0);
        return false;
    }
    // Init to zero in default
    virtual bool BeforeBackward(size_t& szBuf, int flag = 0x0) {
        assert(0);
        return false;
    }
    virtual bool Free(bool isPassResident = false) { return true; }
    //   x==3 ? gv : x==2 ? gm : x == 1 ? grad : data
    virtual void Print(const string& title, int typ, int flag, size_t nEle = 0) const;
    virtual bool DumpX(int type, const string& title = "", int flag = 0x0) const;
    // operations
    hGTensor operator*(const hGTensor& other) { return _Multiply(other); }

    double Length(int tp, int flag = 0x0);  //  return ||x||

    // operations
    virtual bool ShareMemory(hGTensor, int flag = 0x0);
    virtual bool OverWrite(hGTensor, bool isSrc = true, int flag = 0x0);

    virtual hGTensor Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward = true, int flag = 0x0) {
        assert(0);
        return nullptr;
    }  //  Loss
    virtual hGTensor CrossEntropy(const hGTensor b, int flag = 0x0);
    //  ternary {-1, 0, 1}
    virtual bool ToTernary(floatX*, int flag = 0x0) { throw "ToTernary is ...."; }
    virtual void* BeforeQuant(GeQuant* hQuant, int flag = 0x0);
    virtual bool AfterQuant(GeQuant* hQuant, typNUMBER type_quant, void* data_quant, int flag = 0x0);
    virtual float ToF8Ex(int type, int flag = 0x0) { throw "ToF8Ex is ...."; }
    virtual bool Mutation(int flag = 0x0) { throw "Mutation is ...."; }

    // row-major order. ne contains the number of elements in each dimension & nb is the number of bytes ("nb", a.k.a. stride).
    int64_t ne[N_DIMS];
    int32_t flags = 0x0, last_stp = -1;

    bool isParam() const { return BIT_TEST(flags, F_PARAM); }
    bool isAtHost() const;
    bool isRefer(int tp = 0x0) const { return hRef != nullptr; }
    hGTensor GetRefer() { return hRef; }
    virtual void SetRefer(hGTensor hR, int flag = 0x0);
    virtual bool SetTernary(typNUMBER typ, int flag = 0x0);
    //  load mapping-memory would do Quant !
    virtual bool Serial_Quant_MMAP(bool isSave, bool isX = true, int flag = 0x0);
    virtual bool Serial_MMAP_x(void* dest, bool isSave, int flag = 0x0);
    virtual bool SerialGP(void* yD, void* yG, size_t szY, bool isToY, int flag = 0x0) {
        assert(0);
        return false;
    }
    virtual bool SerialData(const string& info, void* host, bool isToHost, int flag = 0x0) {
        assert(0);
        return false;
    }
    template <typename T>
    T* GetHostData(int flag = 0x0, const string& sX = "") {
        return nullptr;
    }
    virtual floatX* GetDataX(int flag = 0x0, const string& sX = "");

    // may different wit hQuant
    shared_ptr<GeQuant> GetDynamicQuant(int flag = 0x0) const;

    // Some optimization function
    virtual int Dogleg(int flag = 0x0);

    char name[MAX_NAME] = "\0";
    std::string Alias(int flag=0x0);
    size_t hash = 0x0;    //  std::hash<std::string>
    void* extra = nullptr;  // extra things 

    // struct ggml_tensor* GG();
    struct ggml_context* CTX() { return nullptr; }
    virtual size_t size(int typ = 0) const;
    virtual size_t mostMemory(int typ = 0) const { return nByte(); }
    virtual int dims() const {
        for (int i = N_DIMS - 1; i >= 1; --i) {
            if (ne[i] > 1) {
                return i + 1;
            }
        }
        return 1;
    }
    virtual size_t nByte() const { return szData; }
    //   A weight matrix is a linear operator on RMS-normed vector spaces.
    virtual bool isWMAT(int flag = 0x0) const;
    //  The offset of (i0,i1,i2,i3) in byte
    virtual size_t Offset(int i0, int i1, int i2, int i3, int flag = 0x0) const;
    //  The offset of N-th elemen in byte
    virtual size_t Offset(size_t N, int flag = 0x0) const;

    virtual bool isEmpty() const {
        // if(size()>0)    {   assert(B>0 && T>0 && C>0); }
        return size() == 0;
    }
    virtual bool isSameShape(SHAPE shape, int flag = 0x0) const;
    virtual bool isSameShape(const hGTensor b) const;
    virtual void Zero() { Set(0.0); }
    virtual void ZeroGrad() { assert(0); }
    virtual void Set(float a, int flag = 0x0);
    template <typename T>
    void Set(int i0, int i1, int i2, int i3, T value) {
        void* val = (char*)data + Offset(i0, i1, i2, i3);
        switch (type) {
            case typNUMBER::I8: {
                ((int8_t*)(val))[0] = value;
            } break;
            case typNUMBER::I16: {
                ((int16_t*)(val))[0] = value;
            } break;
            case typNUMBER::I32: {
                ((int32_t*)(val))[0] = value;
            } break;
            case typNUMBER::F16: {
                assert(0 && "fatal error");
            } break;
            case typNUMBER::BF16: {
                assert(0 && "fatal error");
            } break;
            case typNUMBER::F32: {
                ((float*)(val))[0] = value;
            } break;
            default: {
                assert(0 && "fatal error");
            }
        }
    } /**/

    virtual void SetFlag(int64_t flag) { flags |= (int32_t)flag; }
    virtual float Get(int i, int flag = 0x0) const;
    // Returns the value of this tensor(with one element!)
    virtual float Item() const {
        assert(size() == 1);
        return Get(0);
    }

    virtual int SerialJSON(const std::string& name, const JSON& val, void* bytes_ptr, size_t bytes_size, int flag = 0x0);
    friend class GeNeuron;
    friend class GeQuant;
    friend class huTensor;
    friend class OPT_Adam;
    friend class Fuyou;
    friend class SLP;
    friend class Fish;
    friend class GENSOR_TOPU;
};

template <typename T>
T* TO(hGTensor t) {
    assert(t != nullptr);
    if (t->isRefer()) {
        t = t->GetRefer();
    }
    assert(t->data != nullptr);
    // assert(t->type==TYPE_<T>())
    BIT_SET(t->flags, GTensor::F_TOX);
    return (T*)(t->data);
}
#define TOBF TO<__nv_bfloat16>

template <typename T = floatX>
T* TO(const std::vector<hGTensor>& gensors, const std::string& key, int flag = 0x0) {
    assert(gensors.size() > 0);
    for (auto gensor : gensors) {
        if (gensor == nullptr)
            continue;
        if (strstr(gensor->name, key.c_str()) != NULL) {
            return TO<T>(gensor);
        }
    }
    assert(0);
    return nullptr;
}

inline hGTensor operator+(const hGTensor& a, const hGTensor& b) {
    return nullptr;
}
inline hGTensor operator+=(const hGTensor& a, const hGTensor& b) {
    return nullptr;
}

typedef hGTensor hGensor;  // some trick
// inline struct ggml_tensor* G(hGensor T) {
//     assert(T != nullptr);
//     return T->GG();
// }

inline void ZERO_(hGensor T) { T->Zero(); }
inline size_t tELEM(hGensor T) { return T == nullptr ? 0 : T->size(); }
inline size_t tBYTE(hGensor T) { return T == nullptr ? 0 : T->nByte(); }
inline int tDIM(hGensor T) { return T == nullptr ? 0 : T->dims(); }
inline float tGET(hGensor T, int i) { return T->Get(i); }
inline void tSET(hGensor T, float a) { T->Set(a); }
inline void tFLAG(hGensor T, int64_t flag) { T->SetFlag(flag); }
double tNormsOf(const std::vector<hGTensor>& tensors, int flag);
// double tNormOf(const hGTensor tensor, int flag = 0x0);

inline floatX* ToX(hGensor t) {
    assert(t != nullptr);
    BIT_SET(t->flags, GTensor::F_TOX);
    return (floatX*)(t->data);
}
inline floatX* ToX0(hGensor t) {
    if (t == nullptr)
        return nullptr;
    return ToX(t);
}
inline floatX* ToG(hGTensor t) {
    assert(t != nullptr);
    assert(t != nullptr && t->grad != nullptr);
    assert(typeid(floatGrad) == typeid(floatX));
    return (floatX*)(t->grad);
}
inline floatX* ToG0(hGTensor t) {
    if (t == nullptr || t->grad == nullptr)
        return nullptr;
    return ToG(t);
}
// only create tensor
// hGensor TENSO(void* ctx0,typNUMBER typ,SHAPE,int flag=0x0,const string&name="" );

// Generate GTensor
hGTensor GT(Fish* hFish, typNUMBER typ, SHAPE, int flag = 0x0, const string& name = "");
//  Create GTensor & alloc & copy data
hGTensor GT(SHAPE shape_, void* data, typNUMBER tpD_, int flag = 0x0);

hGensor tRAND(hGensor tensor, struct random_normal_distribution* rnd);

/**
 *  tensor stored in hybrid memory of(CPU/GPU/DISK...)
 */
class huTensor : public GTensor {
   protected:
    hGTensor _Multiply(const hGTensor& other);
    size_t Alloc_1(void** dst, bool isZero, string desc, size_t sz = 0x0, int flag = 0x0) override;
    size_t Free_1(void** obj, const string& info = "") override;

   public:
    huTensor(Fish* hFish, const string& name_, const SHAPE shape, typNUMBER tpD_, bool isAlloc, int flag = 0x0);
    virtual ~huTensor();
    hGTensor Partial(const string&, size_t offset, const SHAPE shape, int flag = 0x0) override;

    bool Alloc(int tpInit = 0, int flag = 0x0) override;
    size_t mostMemory(int typ = 0) const override;
    bool InitParam(int tpInit) override;
    bool BeforeBackward(size_t& szBuf, int flag = 0x0) override;

    bool Free(bool isPassResident = false) override;
    void Zero() override;
    void ZeroGrad() override;
    void Set(float a, int flag = 0x0) override { assert(0); }
    bool SerialGP(void* yD, void* yG, size_t szY, bool isToY, int flag = 0x0) override;
    bool SerialData(const string& info, void* host, bool isToHost, int flag = 0x0) override;
    bool OverWrite(hGTensor, bool isSrc = true, int flag = 0x0) override;
    hGTensor CrossEntropy(const hGTensor b, int flag = 0x0) override;

    hGTensor Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward = true, int flag = 0x0) override;
    // void Print(const string &title, int typ, int flag, size_t nEle = 0) const override;

    bool ToTernary(floatX* tmp, int flag = 0x0) override;
    float ToF8Ex(int type, int flag = 0x0) override;
    bool Mutation(int flag = 0x0) override;
    friend class HIERARCH_LoRA;
};

struct GENSOR_INFO {
    string sX;
    int level = -1, ID = -1, dad = -1, c_id = -1;
    bool isAdam                 = true;
    shared_ptr<GeNeuron> hNeron = nullptr;

    // first moment, second moment,past function values of Optimizer
    float *gm = nullptr, *gv = nullptr, *gpf = nullptr;

    GENSOR_INFO() { ; }
    GENSOR_INFO(int id_, int l, int d, int c) : ID(id_), level(l), dad(d), c_id(c) {
        string suffix, prefix;
        sX = __repr__(suffix, prefix);
    }
    string __repr__(string& suffix, string& prefix, int flag = 0x0) const {
        char buf[512] = "\0";
        if (dad == -1) {
            sprintf(buf + strlen(buf), "ROOT");
        } else
            sprintf(buf + strlen(buf), "[%d %d.%d l=%d]", ID, dad, c_id, level);
        return buf;
    }

    static bool comp(GENSOR_INFO& a, GENSOR_INFO& b) { return a.ID < b.ID; }
};

// gensors & its topu relation
struct GENSOR_TOPU {
    // name_ => tensor
    std::map<std::string, hGensor> nag;
    //  gensor => info
    std::map<hGensor, GENSOR_INFO> infos;

    virtual bool has(hGensor gensor);
    void Insert(hGensor gensor, const GENSOR_INFO& gi, int flag = 0x0);

    void Insert(const std::map<std::string, hGensor>& src) { nag.insert(src.begin(), src.end()); }
    size_t size() { return nag.size(); }
    virtual hGensor Get(MODEL_ARCH arch, const string& name, int flag = 0x0);
    virtual void Clear() {
        nag.clear();
        infos.clear();
    }

    virtual void TopoOrder() {
        // sort(gimap.begin(), gimap.end(), comp);
    }
};

void assert_shape_1d(hGensor tensor, int64_t ne0);
void assert_shape_2d(hGensor tensor, int64_t ne0, int64_t ne1);
void assert_shape_3d(hGensor tensor, int64_t ne0, int64_t ne1, int64_t ne2);
void assert_shape_4d(hGensor tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

inline bool CHECK_SHAPE(const SHAPE& shape) {
    bool isValid = shape.size() > 0;
    for (auto s : shape) {
        if (s < 0) {
            isValid = false;
            break;
        }
        if (s > 1024 * 1024) {
            isValid = false;
            break;
        }
    }
    assert(isValid);
    return isValid;
}

int CHECK_SAME_TENSORS(const string& desc, const std::vector<hGensor>& arrA, const std::vector<hGensor>& arrB, int flag = 0x0);

void Gensor2float_(const hGensor w, float* A, int flag = 0x0);
inline float* Gensor2float(struct ggml_context* ctx0, const hGensor w, int flag = 0x0) {
    size_t ne00 = tELEM(w), nbyte = tBYTE(w);
    void* data_0 = w->data;
    float* A     = new float[ne00];
    Gensor2float_(w, A, flag);
    return A;
}

void _T_repr_(hGensor t, const char* tab, char* buf, int typ = 0x0);
void _T_repr_(hGensor t, const char* tab, char* buf, const GENSOR_INFO& info);

template <typename T>
double P_softmax(int idx, T* logits, int size);