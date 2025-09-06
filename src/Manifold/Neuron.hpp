
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Collection of neurons
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
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <stack>
#include <typeinfo>
#include <vector>
using namespace std;

#include "../TokenSet/DataLoader.hpp"
#include "../Utils/GST_util.hpp"

class Fish;
struct NeLayer;
struct LayerNormal;
class RLS_BP;
class SparseNeuron;
/**
 * Each Neuron has
 * 1. inp,out,w
 * 2. b     (if isBias is true)
 * 3. delta (optional)
 * 4. activation
 *
 * Fusion
 * 1. gelu_fusion
 *
 * Rematerization
 * 1.   vRemater
 */

/*
   Hierarchical lora representation of a sparse matrix
     1. LORA - LOW-RANK ADAPTATION    w=b(:,rank)*a(rank,:)
*/
struct HIERARCH_LoRA {
    static string sNeurons;

    float beta_F       = 1.0f;
    bool isAccumuDelta = true;

    Fish *hFish            = nullptr;
    SparseNeuron *spNeuron = nullptr;
    bool isBack            = true;
    int B, T, rankMode, rank = -1, nHeavy = -1;
    hGTensor wBase = nullptr;
    void *Adelta = nullptr, *Ax = nullptr;
    hGTensor a = nullptr, b = nullptr;
    HIERARCH_LoRA(SparseNeuron *hF, hGTensor w_, int r_, int flag = 0x0);
    virtual ~HIERARCH_LoRA();

    virtual void UpdateAdapt(int flag = 0x0);

    virtual int Forw(floatX *rhs, floatX *lhs, int BT, int flag = 0x0);
    virtual int Back(hGTensor delta, hGTensor inp, hGTensor deltaIn, int flag = 0x0);
};
typedef std::shared_ptr<HIERARCH_LoRA> H_LORA;
typedef std::vector<H_LORA> arrLORA;

class GeNeuron {
    // GeNeuron(const GeNeuron&)=default;
    GeNeuron &operator=(const GeNeuron &) = default;

   protected:
    struct STATISTIC {
        double time = 0.0, tFore = 0.0, tBack = 0.0;
        double mem = 0.0;

        virtual void Reset() {
            time = 0.0, tFore = 0.0, tBack = 0.0;
            mem = 0.0;
        }
    };

    STATISTIC stat;
    int block_size = 256, grid_size = 0;  // for cuda kernel function

    int n_embd_head, n_head;
    int gelu_fusion = 0, dump_flag = 0;
    Fish *hFish = nullptr;
    SHAPE shape;
    COMPRESSIVE_SENSING compression = SKIP;
    typNUMBER tpWeight = typNUMBER::BF16, tpActivation = typNUMBER::BF16, tpGradient = typNUMBER::BF16;
    bool isPassBack = false;
    int level = -1, ID = -1, dad, c_id;  // topo info
    int layer = -1;                      // no of layer in LLM/Brain structure
    int branch = 0, hierarch = 0;
    int xxx   = 0;
    vector<double> jvals;
    // vector<hGTensor> vRemater;      //support rematerization
    string _repr_1(string &suffix, string &prefix, string info, int flag = 0x0);
    size_t dev_most_mem = 0x0, host_most_mem = 0x0;  // may at different place
    void *host_inp = nullptr;                        // backup of input in device memory
    // 天地本逆旅, 你我皆过客(Guoke)
    GeNeuron *hGuoke = nullptr;
    std::set<hGensor> tReloads;
    // std::vector<shared_ptr<GeNeuron>> brothers;

   public:
    enum BIT_FLAG { F_BIAS = 0x10000, F_DELTA = 0x20000, F_GRADREF = 0x40000, F_HOTPICK = 0x100000 };

    DATA_PLACE place = DATA_PLACE::VOID;

    static shared_ptr<GeNeuron> MakeInstance(Fish *hG_, void *ctx_build, const string &guid, JSON::const_iterator jit, int flag = 0x0);
    int B, T, C;  // from n_batch,n_ctx,n_embd

    // w is the 道 of neuron.   道者,千变万化之动(wLORAs)
    hGensor w = nullptr;
    ;
    arrLORA wLORAs;  // LORA of w
    LORA_ADAPT_W tpLORA = LORA_ADAPT_W::W0;

    hGensor b = nullptr, out = nullptr;

    hGensor inp   = nullptr;  //  may change! maybe nullptr!
    hGensor delta = nullptr;  //  backward-error tensor at each layer(may share memory!)
    bool isBias = true, isResidual = true, isSparse = false, isTransW = false;
    bool isShortcut = false;

    NeLayer *hLay    = nullptr;
    string sT        = "";  //  short-cut infomation
    std::string name = "N", type_info = "";
    GeNeuron() {}
    GeNeuron(const std::string &key_, JSON::const_iterator jit, Fish *hG_, int flag);
    virtual ~GeNeuron();
    //  Gensors with physical memory
    virtual std::vector<hGensor> PhysicalGensors(bool isNoRef = true, int flag = 0x0) { return {}; }
    // Pick gensors(child,partial,vitual,ref,lora,...)
    virtual std::vector<hGensor> PickGensors(bool isLORA = true, int flag = 0x0);
    virtual hGensor GetGensor(const std::string &key, int flag = 0x0);
    virtual int SetGuoke(GeNeuron *hGuoke_, bool isRefParam, int flag = 0x0);
    virtual void SetDType(typNUMBER tpW, typNUMBER tpA, typNUMBER tpG) { tpWeight = tpW, tpActivation = tpA, tpGradient = tpG; }

    virtual bool isValid();
    virtual bool isOnlyInfer();
    virtual bool isForward();
    // 1. Set C/T/H/...
    virtual void Init(Fish *hG_, int flag = 0x0);
    virtual bool Empty() const { return shape.size() == 0; }
    virtual size_t nElem() { return 0x0; }
    // memory management at different place
    virtual void ManageMemory(DATA_PLACE target, int typ = 0x0, int flag = 0x0);
    virtual hGensor OnInput(hGensor hIn, int flag = 0x0);

    virtual bool BeforeForward(int iter, int op = 0x0, int flag = 0x0) { return true; }

    //  无知觉明
    virtual hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0);
    virtual hGensor BeforeMing(RLS_BP *hRLS, hGensor cur, int flag = 0x0);
    virtual hGensor AfterMing(RLS_BP *hRLS, hGensor cur, int flag = 0x0);

    virtual hGensor Backward(void *user_ctx, hGensor cur, int flag = 0x0);
    virtual hGensor Forward2(void *ctx_build, hGensor, hGensor, int flag = 0x0) {
        assert(0);
        return nullptr;
    }

    virtual bool Build(int flag) {
        assert(0);
        return false;
    }
    // Init & build with more option
    virtual void BuildX(const std::string &key_, const SHAPE &shape, Fish *hG_, int flag);
    virtual bool InitCompression(COMPRESSIVE_SENSING type, LORA_ADAPT_W tpLora, int flag = 0x0) { return false; }

    virtual void OnDebug(const std::string &info = "", int typ = 0x0, int flag = 0x0);
    virtual void OnRemater(RLS_BP *schedule, int typ, int flag = 0x0);

    virtual string __repr__(string &suffix, string &prefix, int flag = 0x0);
    virtual bool isGang() { return false; }

    virtual void SetRefer(const GeNeuron *src, bool isBias = false, int flag = 0x0);
    virtual std::vector<GeNeuron *> SubNeurons(int flag = 0x0) { return {}; }
    virtual bool OnData(hGTensor X, hGTensor Y, int *hot, int flag = 0x0) { return false; }
    virtual bool Sparsing(int flag = 0x0) { return false; }
    friend class Fish;
    friend class NLP_AutoRegressive;
    friend class RLS_BP;
    friend class HIERARCH_LoRA;
    friend class Fuyou;
};

class HotPicker;
class TokenEmbed;
template <typename T>
class LoSVD;

// Pick some neurons from context sparcity
class CS_Picker {
   protected:
    bool isMerge = false;
    int dim = -1, nLastHot = -1;
    float T_hot = 0.2, T_zero = 1.0e-3;

   public:
    int *hot     = nullptr;
    float *dTemp = nullptr;
    static double tPick;  //  Picker should much fast than dot!
    CS_Picker() {}
    CS_Picker(shared_ptr<Fish> hFish, int flag = 0x0);
    //  uses the first layer’s attention output to predict the sparsity pattern for the entire model
    virtual ~CS_Picker() {
        FREE_a(hot);
        FREE_a(dTemp);
    }

    int Update(int level, float *hb, int flag = 0x0);
};
typedef shared_ptr<CS_Picker> hCSPicker;

/*
How get sparse index
    1. Learing by GBDT/NN/
    2. Subsampling
    3. Low-rank approximation(SVD)
*/
class SparseNeuron : public GeNeuron {
   protected:
    int method      = 0;
    hGTensor hSamps = nullptr;
    int samp_type = 0, samp_0 = 0, samp_1 = 0;
    float rSampNorm = 1.0f;
    // subw = w[samples]
    TokenEmbed *subw = nullptr;

    shared_ptr<HotPicker> hPicker = nullptr;

    // hGensor u = nullptr, s = nullptr, v = nullptr;
    shared_ptr<LoSVD<float>> hSVD = nullptr;

    virtual bool InitSVD(int flag = 0x0);
    virtual bool InitLoRA(LORA_ADAPT_W tpLora, int flag = 0x0);

   public:
    SparseNeuron() {}
    SparseNeuron(const std::string &key_, JSON::const_iterator jit, Fish *hG_, int flag);
    virtual ~SparseNeuron() {}

    bool OnData(hGTensor X, hGTensor Y, int *hot, int flag = 0x0) override;
    virtual bool GetHotIndex(int nPoint, floatI *data, int *hot, int flag = 0x0);
    bool Sparsing(int flag = 0x0) override;
    bool InitCompression(COMPRESSIVE_SENSING type, LORA_ADAPT_W tpLora, int flag = 0x0) override;
    virtual void SetGanglia(const SparseNeuron *gang, int flag = 0x0) {
        compression = gang->compression;
        layer       = gang->layer;
        hSamps      = gang->hSamps;
    }
    virtual void SetEmbed(TokenEmbed *embd_, int type, int flag = 0x0);
    virtual void UpdateSamps(int seed, int flag = 0x0);
};
typedef shared_ptr<GeNeuron> hNeuron;

// Collection of neurons, only special operation!
struct Ganglia : public SparseNeuron {
    std::vector<hNeuron> ns;
    Ganglia(Fish *hG_, const string &guid, std::vector<hNeuron> &ns_, int flag);
    bool isValid() override { return ns.size() > 0; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    bool isGang() override { return true; }
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override { return cur; }
};

class ROPE : public SparseNeuron {
   protected:
    void *devQ = nullptr, *devK = nullptr, *devDeltaQ = nullptr, *devDeltaK = nullptr;
    int n_rot = 0, n_ctx_orig = 0, head_dim = 0, q_dim = 0, kv_dim = 0, r_dim = 0;
    hGensor KQ_pos, hSin = nullptr, hCos = nullptr;
    float f_norm_rms_eps, freq_base, freq_scale, theta;
    MODEL_CARD::tpROPE alg;
    bool isInterleave;  // Interleave the even and odd encodings
    Grusoft::GRander rRounding;

   public:
    ROPE() {}
    ROPE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    int cuTrain(LayerNormal *normQ, LayerNormal *normK, uint32_t seed, bool isFX = true, int flag = 0x0);
    bool isValid() override { return true; }
    bool Empty() const override;
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    friend class SelfAttention;
};
typedef shared_ptr<ROPE> hRope;
struct Relu : public SparseNeuron {
    Relu() { ; }
    Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    bool Build(int flag) override;
    bool isValid() override { return true; }
};

struct Drop : public SparseNeuron {
    Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    bool Build(int flag) override;
    bool isValid() override { return true; }
};

// single layer perceptron
struct SLP : public SparseNeuron {
    SLP() {}
    SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    // The channel/neuron number of input&output
    int nIn = -1, nOut = -1;
    bool Empty() const override { return w == nullptr; }
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    // only for deprecated function"UpdateGensor"
    hGensor UpdateGensor(int flag = 0x0);
    size_t nElem() override;
    virtual int OnMultiscale(SLP *src, int flag = 0x0);

    // hGTensor operator<<(hGTensor a);
    /*  Forward or remate in Back
        1.  rhs = SLP(lhs)  or rhs = W*lhs+b
        2.  rhs = GELU(W*lhs+b);    to_gelu=W*lhs+b
    */
    int Forw(hGTensor rhs, hGTensor lhs, hGTensor to_gelu = nullptr, int flag = 0x0);
    // CPU version
    int Forw(float *rhs, float *lhs, int flag = 0x0);
#ifndef ENABLE_FP32
    int Forw(floatX *rhs, floatX *lhs, int flag = 0x0) {
        assert(0);
        return 0x0;
    }
#endif
    /*  Backward
        inp & to_gelu is defined in forward: inp=GELU(to_gelu)
    */
    int Back(hGTensor delta, hGTensor inp, hGTensor deltaIn, hGTensor to_gelu = nullptr, int flag = 0x0);
    int FUSE_cuda_block(hGTensor rhs, hGTensor lhs, hGTensor gelu = nullptr, bool isForw = true, int flag = 0x0);
};

struct LayerNormal : public SparseNeuron {
    bool isAffineTrans = true;  // Learnable affine transform parameters
    bool isRMS         = true;  // Root Mean Square Layer Normalization
    int nHead          = 0;
    //  always float
    hGensor mean = nullptr, rstd = nullptr;
    float scale = 0.0;
    LayerNormal() {}
    LayerNormal(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag) override;
    hGTensor cuTrain(hGTensor inpL, int flag = 0x0);
    size_t nElem() override;
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    // hGTensor operator>>(hGTensor & a){  return a;   }
    // hGTensor operator<<(hGTensor a);
};

struct LayerSoftmax : public SparseNeuron {
    LayerSoftmax() {}
    LayerSoftmax(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
};

struct MOE : public SparseNeuron {
    bool isSiLU = false;

    MOE() {}
    MOE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    hGensor Forward2(void *ctx0, hGensor cur, hGensor, int flag = 0x0) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
};

//  reversible residual network (RevNet)
class RevNet : public SparseNeuron {};

enum UIDs {
    Q_UID,
    K_UID,
    V_UID,
    Attn_scale_UID,
    O_UID,
    dO_UID,
    dQ_UID,
    dK_UID,
    dV_UID,
    STATS_UID,
    BIAS_UID,
    DBIAS_UID,
    SEQ_LEN_Q_UID,
    SEQ_LEN_KV_UID,
};
class SelfAttention : public SparseNeuron {
   protected:
    int tpNormal = 1, n_ff = 0;
    bool isLinear    = false;
    bool isPreNormal = false;  //  Pre /Post Normalization
    bool isQKNormal  = false;
    LayerNormal normQ, normK;  //  Only w vector to save memory
    bool isSeparateQKV = false;
    bool isBqkv        = false;  // to align with some model

    //  tensor format={'SBhd', 'BShd', 'thd'}, default = 'BShd',   t=B*S
    void *devQ = nullptr, *devK = nullptr, *devV = nullptr, *devDeltaQ = nullptr, *devDeltaK = nullptr, *devDeltaV = nullptr;
    // markov transition matrix from KQ
    enum TRANSITION_MODE {
        SOFT_MAX = 0,
        SIGMOID  = 1,
        RELU2    = 2,
        RELU_    = 3,
        LINEAR   = 4,
    };
#ifdef ENABLE_CUDNN
    virtual bool FUSE_cudnn(floatX *dqkvr, floatX *dout, int flag = 0x0);
#endif
    // 1 linear(No softmax!) 2 sim(q,k)>=0
    TRANSITION_MODE tpTrans = SOFT_MAX;
    enum LINEAR_MODE {
        L_OFF,
        ELU    = 0x1,
        NORM_1 = 0x2,
    };
    LINEAR_MODE tpLinear = L_OFF;
    enum SPARSE_MODE {
        S_OFF,
        POOLING   = 0x1,
        LINFORMER = 0x2,
    };
    bool remater_qkv = false;
    bool isAttOnBC   = false;  //  // Nearly same. If true,attenion on all tokens, memory would explode!
    int Rope_version = 0;
    hGensor attn_k = nullptr, attn_q = nullptr, tmpQKV = nullptr;
    // int n_rot=-1;
    // hGensor W_rope(void *ctx ,hGensor cur,hGensor w,hGensor KQ_pos,SHAPE shape,const string&shortcut,int flag=0x0);
    hGensor MyAttention(RLS_BP *ctx_, hGensor inpL, int flag);
    hGensor vXattn(void *ctx, hGensor v, hGensor attn, int flag);
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    std::unordered_map<int64_t, void *> var_packs;
    uint32_t rope_seed = 666888;

   public:
    bool use_cache = false;
    bool isLast    = false;
    float f_max_alibi_bias;
    int n_head_kv, n_embd_gqa, n_tokens, C_qkv = -1;
    hGensor bqkv   = nullptr;  //  biases for qkv (qwen)
    hGensor KQ_pos = nullptr, KQ_mask = nullptr;
    LayerNormal norm, *fuseNorm       = nullptr;

    hGensor attn = nullptr, transition = nullptr;

    SLP Q, K, V;
    static hRope rope;
    MOE moe;
    SLP proj_cat;  //   concatenate the heads and combine them with a final weight matrix.
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    // std::vector<hGensor> PickGensors(bool isLORA = true, int flag = 0x0) override;
    std::vector<GeNeuron *> SubNeurons(int flag = 0x0) override;
    int SetGuoke(GeNeuron *hGuoke_, bool isRefParam, int flag = 0x0) override;
    bool BeforeForward(int iter, int lay = 0x0, int flag = 0x0) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;

    hGTensor cuTrain(hGTensor inpL, int flag);
};

/*
    Gated SelfAttention
*/
struct GatedAttention : public SelfAttention {
   protected:
    int attn_mode = 1;
    SLP down, upU, upV;

   public:
    GatedAttention() {}
    GatedAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
};

struct cuAttention : public SelfAttention {
   protected:
    int attn_mode = 1;
    SLP down, upU, upV;

   public:
    cuAttention() {}
    cuAttention(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
};

// cuda version by K
struct BROWN_attn : public SelfAttention {
   protected:
    bool Transfer_1 = false;
    int n_rot       = -1;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;

   public:
    BROWN_attn() {}
    BROWN_attn(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    bool Build(int flag) override;

    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
};

// Variational Encoder/Decoer
class VarCoder : public SparseNeuron {
   protected:
    int nTop = -1, nBottom = -1;
    bool isResi      = false;
    bool isSymmetric = false;
    hGensor resi     = nullptr;
    int tpNorm       = -2;
    SLP up, down, gate;
    Relu relu;

    bool Build(int flag) override;

   public:
    LayerNormal norm;
    // hGensor encode=nullptr,decode=nullptr,norm=nullptr;
    VarCoder() {}
    VarCoder(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    VarCoder(Fish *hG_, std::vector<int> &dims, int level, bool isR = false, bool isSym = true, int tpN = 2, int flag = 0x0);
    virtual hGensor ENC(const hGensor x0);
    virtual hGensor DEC(hGensor x);
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    // std::vector<hGensor> PickGensors(bool isNoRef = true, int flag = 0x0) override;
    friend class TokenEmbed;
    friend class MAEC;
    friend class OutCLS;
};
typedef shared_ptr<VarCoder> hVarCoder;

struct FFN : public VarCoder {
    LayerNormal *fuseNorm = nullptr;
    bool isShareParam     = false;
    // hGensor pre_gelu = nullptr;
    int latent;
    // SelfAttention *lastQKV=nullptr;
    bool remater_ffn = false;

    static FFN *first;
    FFN() {}
    FFN(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual ~FFN() {}
    bool Build(int flag) override;
    std::vector<GeNeuron *> SubNeurons(int flag = 0x0) override;
    // std::vector<hGensor> PickGensors(bool isNoRef = true, int flag = 0x0) override;
    int SetGuoke(GeNeuron *hGuoke_, bool isRefParam, int flag = 0x0) override;

    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;

    hGTensor cuTrain(hGTensor inpL, int flag);
    int CPU_v0(void *ctx, int layer, int flag = 0x0);
};

//  multi-level auto encoder
struct MAEC : public SparseNeuron {
    int nIn = -1, nOut = -1;
    vector<hVarCoder> codes;
    LayerNormal normE, normD;
    bool reserve_x = false;
    vector<hGensor> resi_x;

    MAEC(Fish *hG_, const std::string &key_, int flag);
    virtual hGensor ENC(hGensor x, int flag = 0x0);
    virtual hGensor DEC(hGensor x, bool isForw, int flag = 0x0);
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    bool Empty() const override { return codes.size() > 0; }
};
typedef shared_ptr<MAEC> hMAEC;

/*
    Each token is embed to latent vector
*/
struct TokenEmbed : public SparseNeuron {
    hBATCH hBatch = nullptr;
    LayerNormal lnW, lnWInv;
    int *workload_indices = nullptr, nVocab = -1, latent, *hostID = nullptr, num_c_groups = -1, num_buckets = -1;
    int4 *bucket_info = nullptr;
    bool isAddPos     = false;
    int padded_nCls   = -1;  //*hostInput=nullptr,inp_pos=0;
    // the inverse of w: [nEmbed]=>[nToken]; w is row-major; wInv is column-major
    hGensor wInv = nullptr;
    virtual bool SetMAEC(hMAEC maec, int flag = 0x0);
    hMAEC maec = nullptr;
    Grusoft::GRander rRounding;
    
    TokenEmbed() {}
    TokenEmbed(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual ~TokenEmbed();
    virtual bool UpdateBucket(int type, int flag = 0x0);
    virtual void WorkloadOnBucker(int *inputs_cpu, int flag);
    // virtual int InitMAC(int flag=0x0);
    virtual hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag = 0x0) override;
    bool Build(int flag) override;
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    virtual hGTensor OnEmbed(hGensor inpL, int seed);
    virtual hGTensor SubW(hGTensor hSamp, bool isForw, hGTensor subw, int flag = 0x0);
};

struct FFN_MOE : public FFN {
    MOE Moe;
};

struct OutCLS : public SparseNeuron {
    hMAEC maec;
    LayerNormal norm;
    SLP proj;
    TokenEmbed *hEmbed = nullptr;
    // host version of target is SampLoader::hostTargetProbs
    hGTensor target = nullptr, preLogits = nullptr;
    //  device=>host    floatX=>float
    float *fLogits(int flag = 0x0);
    float metric[METRIC_MOST], *dev_metric = nullptr;
    hSampLoader hLoader = nullptr;
    int nCls = 0, dB = 1, nzLoss = 0, latent = 0;
    int padded_nCls;  // padded to e.g. %128==0,
    float mean_loss = 0, rLoss = 1.0, *hostLoss = nullptr;
    OutCLS() {}
    OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    virtual ~OutCLS() {
        if (hostLoss != nullptr)
            delete[] hostLoss;
    }
    bool Build(int flag) override;
    hGensor Ming(RLS_BP *hRLS, hGensor cur, int flag) override;
    bool isValid() override { return true; }
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;
    // Backward: return lnf->out;       Forward: return preLogits or loss?
    virtual hGTensor cuTrain(hGTensor inpL, int flag);
};

struct OutSimilarity : public OutCLS {
    OutSimilarity(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    hGTensor cuTrain(hGTensor inpL, int flag) override;
};

struct OutEntropy : public OutCLS {
    OutEntropy(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag);
    // hGTensor cuTrain(hGTensor inpL,int flag)  override;
};

struct NeLayer {  // Neural Layer with many neurons
    Fish *hFish = nullptr;
    int id      = -1;
    bool isLast = false;
    std::string name;
    std::vector<hNeuron> neurons;

    void Add(hNeuron hN, int flag = 0x0) {
        neurons.push_back(hN);
        hN->hLay = this;  //???
    }
    NeLayer(int id_) : name("NeLayer"), id(id_) {}
    NeLayer(const std::string &n_, int flag = 0x0) : name(n_) {}
    virtual ~NeLayer() {}
    // NeLayer* N_(const std::string& t,const std::string& n,SHAPE s)   {
    //     return this;
    // }
    virtual string __repr__(string &suffix, string &prefix, int flag = 0x0) { return ""; }
};
typedef shared_ptr<NeLayer> hLayer;

hGTensor operator>>(hGTensor t, const LayerNormal &norm);
hGTensor operator>>(hGTensor t, const SLP &slp);
hGTensor operator>>(hGTensor t, const Relu &relu);
hGTensor operator>>(hGTensor t, const GeNeuron *neuron);
hGTensor operator>>(hGTensor t, const SparseNeuron *neuron);
