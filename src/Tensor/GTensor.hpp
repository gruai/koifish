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
#include <threads.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <typeinfo>
#include <vector>
using namespace std;
#include <stdio.h>
#include <string.h>

#ifdef __USE_GGML__
#include "ggml.h"
#endif

#define GG_V12

class GTensor;
class huTensor;
class Fish;
class EDGE_DEVICES;
typedef shared_ptr<GTensor> hGTensor;
typedef std::vector<int> SHAPE;

#ifdef _TENSOR_G_
#include "../Device/CUDA/cuda_common.h"
#else
typedef float floatX;
#endif

// #define BIT_SET(val, flag) ((val) |= (flag))
// #define BIT_RESET(val, flag) ((val) &= (~(flag)))
// #define BIT_TEST(val, flag) (((val) & (flag)) == (flag))
// #define BIT_IS(val, flag) (((val) & (flag)) != 0)

// set name of a tensor if its name is "\0" & its grad
int gTN(hGTensor, const char *format, ...);
// clear then set name of a tensor & its grad
int gTN0(hGTensor cur, const char *format, ...);

enum DATA_PLACE { VOID, SYMBOLIC, REFER, MEMORY, DEV_MEM, MMAP, DISK, CLOUD, FREE_DEV };

enum INIT_WEIGHT { W_SKIP = 0X0, FIX_1, RANDOM, COPY_WIKI, COPY_SWARM_HEAD, SERIALIZE };
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

    string __repr__(string &suffix, string &prefix, int flag = 0x0) const {
        char buf[512] = "\0";
        if (dad == -1) {
            sprintf(buf + strlen(buf), "ROOT");
        } else
            sprintf(buf + strlen(buf), "[%d %d.%d l=%d]", ID, dad, c_id, level);
        return buf;
    }

    static bool comp(GENSOR_OP &a, GENSOR_OP &b) { return a.ID < b.ID; }
};
typedef shared_ptr<GENSOR_OP> hGOP;

enum QUANT_ALG {
    // Ternary
    // Value of weights
    W_SCALE = 0X100,
    W_NOSCALE  //
};

/**
 * 1.   Support dynamic change shape & type!
 * 2.   Row-major order (first index varies most slowly and last index varies most quickly)
 * 3.   May not contiguous in memory! All tensor operations have to take the stride into account and not assume that the tensor is contiguous in memory!
 */
class GTensor {
   private:
    void *gg = nullptr;

   protected:
    Fish *hFish   = nullptr;
    hGTensor hRef = nullptr;
    std::vector<GTensor *> refered;
    std::shared_ptr<EDGE_DEVICES> hDevice = nullptr;
    size_t szData                         = 0;
    int recompute                         = 1;

    //  support dynamic change shape&type!
    virtual bool ReShape(SHAPE shape_, typNUMBER tpD_, int flag = 0x0);
    virtual hGTensor _Multiply(const hGTensor &other) {
        assert(0);
        return nullptr;
    }

   public:
    static const int MAX_NAME = 64;
    static const int N_DIMS   = 4;

    static size_t szMaloc;
    static hGTensor bt4c, delta, tmpDelta, outL, scratch, tmpFF1, tmpW, tmpGW, residual;
    static bool AllocBuffer(Fish *hFish, int flag = 0x0);
    static bool FreeBuffer(int flag = 0x0);
    static void *buff, *host_buff;                     //  temporary shared memory
    float residual_scale = 1.0, wnorm = 0, gnorm = 0;  // some tricks
    float rLARS(float s0, float T_lars, int flag);
    size_t offset = 0x0;
    SHAPE shape;
    SHAPE x_shape;  //  1.padded for high performance or menory alignment(x_shape<=shape)
    // shape=>x_shape
    virtual void *DataPad(void *src0, int flag = 0x0);

    // static typNUMBER tpFloatX,tpPreLogits;
    typNUMBER type;
    INIT_WEIGHT tpInit = INIT_WEIGHT::RANDOM;
    enum BIT_FLAG {
        F_INPUT     = 0x1,
        F_OUTPUT    = 0x2,
        F_PARAM     = 0x4,
        F_LOSS      = 0x8,
        F_NOALLOC   = 0x100,
        F_GPU       = 0x200,
        F_HOSTALLOC = 0x400,
        F_MMAP      = 0x800,
        F_RESIDENT  = 0x1000,
        F_HOSTDATA  = 0x2000,  // always alloc host_data(may also alloc device data)
        F_RELOAD    = 0x4000,
        F_TOX       = 0x10000,
        F_PADDED    = 0x20000,

        F_TERNARY = 0x100000,
        F_DEBUG   = 0x10000000

    };

    QUANT_ALG tpQuant = W_SCALE;

    static size_t MostOverhead() { return sizeof(GTensor) * 2; }
    GTensor() {}
    GTensor(Fish *hFish, SHAPE shape_, typNUMBER tpD_, bool isAlloc = true, int flag = 0x0);
    GTensor(struct ggml_tensor *gg_, int flag = 0x0) : gg(gg_) { assert(0); }
    virtual bool CopyGG(struct ggml_tensor *gg_, int flag = 0x0) {
        assert(0);
        return false;
    }

    virtual ~GTensor();
    virtual bool Alloc(int tpInit = 0, int flag = 0x0);
    virtual bool InitParam(int tpInit) {
        assert(0);
        return false;
    }
    virtual bool Free(bool isPassResident = false) { return true; }
    virtual void Print(const string &title, int typ, int flag, size_t nEle = 0) const;
    virtual bool Dump(int type, const string &title = "", int flag = 0x0) const;
    // operations
    hGTensor operator*(const hGTensor &other) { return _Multiply(other); }

    hGTensor Relu();
    hGTensor Silu();
    hGTensor Norm(float epsilon, int flag = 0x0);

    // operations
    virtual bool OverWrite(struct ggml_tensor *gg_, bool isSrc = true, int flag = 0x0);
    virtual bool ShareMemory(hGTensor, int flag = 0x0);
    virtual bool OverWrite(hGTensor, bool isSrc = true, int flag = 0x0);

    virtual hGTensor GetRow(hGTensor, hGTensor token, hGTensor pos, int flag = 0x0);
    virtual hGTensor Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward = true, int flag = 0x0) {
        assert(0);
        return nullptr;
    }  //  Loss
    virtual hGTensor CrossEntropy(const hGTensor b, int flag = 0x0);
    //  ternary {-1, 0, 1}
    virtual bool ToTernary(int flag = 0x0) { throw "ToTernary is ...."; }
    virtual bool ToQuant(int flag = 0x0) { throw "ToQuant is ...."; }

    // row-major order. ne contains the number of elements in each dimension & nb is the number of bytes ("nb", a.k.a. stride).
    int64_t ne[N_DIMS];
    // stride in bytes:  nb[0]=type_size(type);    nb[i]=nb[i-1]*ne[i-1]
    size_t nb[N_DIMS];
    int32_t flags = 0x0, last_stp = -1;

    bool isParam() const { return BIT_TEST(flags, F_PARAM); }
    bool isAtHost() const;
    bool isRefer(int type = 0x0) const { return hRef != nullptr; }
    hGTensor GetRefer() { return hRef; }
    virtual void SetRefer(hGTensor hR, int flag = 0x0) {
        hRef = hR;
        hR->refered.push_back(this);
        _INFO("\t%s =====> %s\n", name, hR->name);
    }

    vector<hGOP> src;
    // virtual void AddSrc(const hGOP t,int type,int flag=0x0);
    virtual void AddSrc(const vector<hGTensor> &ts, int flag = 0x0);
    // struct ggml_tensor * view_src=nullptr;
    // size_t               view_offs=0;
    void *host_data = nullptr;  // somtimes, we need data both in device&host
    float *gama_T   = nullptr;  // scaling coefficient of 1-bit weight
    void *data = nullptr, *grad = nullptr;
    void *gm = nullptr, *gv = nullptr;  // first moment, second moment of grad
    float info[8];                      // Some info of some operations
    virtual bool SerialGP(void *yD, void *yG, size_t szY, bool isToY, int flag = 0x0) {
        assert(0);
        return false;
    }
    virtual bool SerialData(const string &info, void *host, bool isToHost, int flag = 0x0) {
        assert(0);
        return false;
    }

    char name[MAX_NAME] = "\0";
    void *extra;  // extra things e.g. for ggml-cuda.cu
    //  return ggml_tensor
    struct ggml_tensor *GG();
    struct ggml_context *CTX() { return nullptr; }
    //  byte per element, may be fraction!!!
    virtual double bpe();
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

    virtual bool isEmpty() const {
        // if(size()>0)    {   assert(B>0 && T>0 && C>0); }
        return size() == 0;
    }
    virtual size_t ld(int no) {
        assert(no >= 0 && no < 4);
        return nb[no] / bpe();
    }
    virtual bool isSameShape(const hGTensor b) const { return szData == b->szData; }
    virtual void Zero() { Set(0.0); }
    virtual void ZeroGrad() { assert(0); }
    virtual void Set(float a, int flag = 0x0);
    template <typename T>
    void Set(int i0, int i1, int i2, int i3, T value) {
        void *val = (char *)data + i0 * nb[0] + i1 * nb[1] + i2 * nb[2] + i3 * nb[3];
        switch (type) {
            case typNUMBER::I8: {
                ((int8_t *)(val))[0] = value;
            } break;
            case typNUMBER::I16: {
                ((int16_t *)(val))[0] = value;
            } break;
            case typNUMBER::I32: {
                ((int32_t *)(val))[0] = value;
            } break;
            case typNUMBER::F16: {
                assert(0 && "fatal error");
                // ((ggml_fp16_t *)(val))[0] = GGML_FP32_TO_FP16(value);
            } break;
            case typNUMBER::BF16: {
                assert(0 && "fatal error");
                // ((ggml_bf16_t *)(val))[0] = GGML_FP32_TO_BF16(value);
            } break;
            case typNUMBER::F32: {
                ((float *)(val))[0] = value;
            } break;
            default: {
                assert(0 && "fatal error");
            }
        }
    }

    virtual void SetFlag(int64_t flag) { flags |= (int32_t)flag; }
    virtual float Get(int i, int flag = 0x0) const;
    // Returns the value of this tensor(with one element!)
    virtual float Item() const {
        assert(size() == 1);
        return Get(0);
    }

    virtual int SerialJSON(const std::string &name, const JSON &val, void *bytes_ptr, size_t bytes_size, int flag = 0x0);
    friend class GeNeuron;
    friend class huTensor;
    friend class OPT_Adam;
};

template <typename T>
T *TO(hGTensor t) {
    assert(t != nullptr);
    if (t->isRefer()) {
        t = t->GetRefer();
    }
    assert(t->data != nullptr);
    // assert(t->type==TYPE_<T>())
    BIT_SET(t->flags, GTensor::F_TOX);
    return (T *)(t->data);
}
#define TOBF TO<__nv_bfloat16>

template <typename T = floatX>
T *TO(const std::vector<hGTensor> &gensors, const std::string &key, int flag = 0x0) {
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

inline hGTensor operator+(const hGTensor &a, const hGTensor &b) {
    // auto cur = ggml_add(a->CTX(), a->GG(), b->GG() );
    // return GTensor::NEW_(cur);
    return nullptr;
}
inline hGTensor operator+=(const hGTensor &a, const hGTensor &b) {
    // auto cur = ggml_add(a->CTX(), a->GG(), b->GG() );
    // return GTensor::NEW_(cur);
    return nullptr;
}

#ifdef _TENSOR_G_
typedef hGTensor hGensor;
inline struct ggml_tensor *G(hGensor T) {
    assert(T != nullptr);
    return T->GG();
}
// inline hGensor NEW_(struct ggml_tensor*gg,int flag=0x0) {
//         return GTensor::NEW_(gg,flag);
// };
inline void ZERO_(hGensor T) { T->Zero(); }
inline size_t tELEM(hGensor T) { return T == nullptr ? 0 : T->size(); }
inline size_t tBYTE(hGensor T) { return T == nullptr ? 0 : T->nByte(); }
inline int tDIM(hGensor T) { return T == nullptr ? 0 : T->dims(); }
inline float tGET(hGensor T, int i) { return T->Get(i); }
inline void tSET(hGensor T, float a) { T->Set(a); }
inline void tFLAG(hGensor T, int64_t flag) { T->SetFlag(flag); }
double tNormOf(const std::vector<hGTensor> &tensors, int flag);
double tNormOf(const hGTensor tensor, int flag = 0x0);
#else
int gTN(struct ggml_tensor *cur, const char *format, ...);
int gTN0(struct ggml_tensor *cur, const char *format, ...);

typedef struct ggml_tensor *hGensor;
inline struct ggml_tensor *G(hGensor T) { return (struct ggml_tensor *)(T); }

inline hGensor NEW_(hGensor gg) { return gg; }
inline void ZERO_(hGensor gg) { ggml_set_zero(gg); }
inline size_t tELEM(hGensor gg) { return ggml_nelements(gg); }
inline size_t tBYTE(hGensor gg) { return ggml_nbytes(gg); }
inline int tDIM(hGensor gg) { return ggml_n_dims(gg); }
inline float tGET(hGensor gg, int i) { return ggml_get_f32_1d(gg, i); }
inline void tSET(hGensor gg, float a) { ggml_set_f32(gg, a); }
inline void tSET_nd(hGensor gg, int i0, int i1, int i2, int i3, int32_t value) { ggml_set_i32_nd(gg, i0, i1, i2, i3, value); }
inline void tFLAG(hGensor gg, int64_t a) { gg->flags |= (int32_t)a; }
hGensor tSCAL(struct ggml_context *_ctx, struct ggml_tensor *a, float s, int flag = 0x0);
hGensor Permute(struct ggml_context *ctx_, struct ggml_tensor *cur, int64_t n1, int64_t n2, int64_t n3, int64_t n4, bool isCont = true);
template <typename T>
inline float T2Float(T *a0) {
    float a = *a0;
    return a;
}
#endif
inline floatX *ToX(hGensor t) {
    assert(t != nullptr);
    BIT_SET(t->flags, GTensor::F_TOX);
    return (floatX *)(t->data);
}
inline floatX *ToX0(hGensor t) {
    if (t == nullptr)
        return nullptr;
    return ToX(t);
}
inline floatX *ToG(hGTensor t) {
    assert(t != nullptr);
    assert(t != nullptr && t->grad != nullptr);
    return (floatX *)(t->grad);
}
inline floatX *ToG0(hGTensor t) {
    if (t == nullptr)
        return nullptr;
    return ToG(t);
}
// only create tensor
// hGensor TENSO(void* ctx0,typNUMBER typ,SHAPE,int flag=0x0,const string&name="" );

// Generate GTensor
hGTensor GT(Fish *hFish, typNUMBER typ, SHAPE, int flag = 0x0, const string &name = "");
//  Create GTensor & alloc & copy data
hGTensor GT(SHAPE shape_, void *data, typNUMBER tpD_, int flag = 0x0);

hGensor tRAND(hGensor tensor, struct random_normal_distribution *rnd);

/**
 *  tensor stored in hybrid memory of(CPU/GPU/DISK...)
 */
class huTensor : public GTensor {
   protected:
    hGTensor _Multiply(const hGTensor &other);
    size_t Alloc_1(void **dst, bool isZero, size_t sz = 0x0, int flag = 0x0);
    size_t Free_1(void **obj, const string &info = "");

   public:
    huTensor(Fish *hFish, const string &name_, const SHAPE shape, typNUMBER tpD_, bool isAlloc, int flag = 0x0);
    virtual ~huTensor();

    bool Alloc(int tpInit = 0, int flag = 0x0) override;
    size_t mostMemory(int typ = 0) const override;
    bool InitParam(int tpInit) override;
    bool CopyGG(struct ggml_tensor *gg_, int flag = 0x0) override;
    bool Free(bool isPassResident = false) override;
    void Zero() override;
    void ZeroGrad() override;
    void Set(float a, int flag = 0x0) override { assert(0); }
    bool SerialGP(void *yD, void *yG, size_t szY, bool isToY, int flag = 0x0) override;
    bool SerialData(const string &info, void *host, bool isToHost, int flag = 0x0) override;
    bool OverWrite(hGTensor, bool isSrc = true, int flag = 0x0) override;
    hGTensor CrossEntropy(const hGTensor b, int flag = 0x0) override;
    hGTensor GetRow(hGTensor, hGTensor token, hGTensor pos, int flag) override;
    hGTensor Normal(hGTensor hOut, hGTensor _mean, hGTensor _rstd, hGTensor w, hGTensor b, bool isForward = true, int flag = 0x0) override;
    void Print(const string &title, int typ, int flag, size_t nEle = 0) const override;

    bool ToTernary(int flag = 0x0) override;
};

struct GENSOR_INFO {
    string sX;
    int level = -1, ID = -1, dad = -1, c_id = -1;
    bool isAdam = true;
    // first moment, second moment,past function values of Optimizer
    // hGensor gm=nullptr,gv=nullptr,gpf=nullptr; //
    float *gm = nullptr, *gv = nullptr, *gpf = nullptr;

    GENSOR_INFO() { ; }
    GENSOR_INFO(int id_, int l, int d, int c) : ID(id_), level(l), dad(d), c_id(c) {
        string suffix, prefix;
        sX = __repr__(suffix, prefix);
    }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) const {
        char buf[512] = "\0";
        if (dad == -1) {
            sprintf(buf + strlen(buf), "ROOT");
        } else
            sprintf(buf + strlen(buf), "[%d %d.%d l=%d]", ID, dad, c_id, level);
        return buf;
    }

    static bool comp(GENSOR_INFO &a, GENSOR_INFO &b) { return a.ID < b.ID; }
};
struct GENSORS {
    // name_ and gg_tensor
    std::map<std::string, hGensor> nag;
    std::map<hGensor, GENSOR_INFO> infos;
    virtual bool has(hGensor gensor) {
        assert(nag.size() == infos.size());
        bool b1 = nag.find(gensor->name) != nag.end(), b2 = infos.find(gensor) != infos.end();
        assert(b1 == b2);
        return b2;
    }

    void Insert(hGensor gensor, const GENSOR_INFO &gi, int flag = 0x0) {
        auto key = gensor->name;
        // assert(strlen(key)>0);
        assert(nag.find(key) == nag.end());
        nag[key] = gensor;

        assert(infos.find(gensor) == infos.end());
        infos[gensor]    = gi;
        infos[gensor].sX = gensor->name;
    }

    void Insert(const std::map<std::string, hGensor> &src) { nag.insert(src.begin(), src.end()); }
    size_t size() { return nag.size(); }
    virtual hGensor Get(const string &name, int flag = 0x0);
    virtual void Clear() {
        nag.clear();
        infos.clear();
    }

    virtual void TopoOrder() {
        // sort(gimap.begin(), gimap.end(), comp);
    }
};
