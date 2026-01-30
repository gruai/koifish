/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  PIPE:   transfer data between device & host
 *  For example:    In CUDA, there is a clear distinction between host and device code, and you cannot directly call host functions from device code.
        calling a __host__ function from a __global__ function  is not allowed  !
 *
 *  \brief PIPE - transfer data between device & host
 *  \author Yingshi Chen
 */
#include <memory>
#include <type_traits>

#include "../Manifold/Fish.hpp"
#include "../Manifold/Optimizer.hpp"

struct PIPE_Optimizer {
    Optimizer* hOPT = nullptr;
    GTensor* tensor = nullptr;
    string name;
    float* tmp        = nullptr;
    floatGama* gama_T = nullptr;
    // use double to reduce Non-Determinism in CUDA Sums!
    double* arrNorm       = nullptr;
    size_t num_parameters = 0;
    ptrdiff_t w_stride, g_stride, s_stride;

    float learning_rate;
    int iter, tile_r1, tile_c1;
    int64_t ne[4] = {0};
    int ldT       = 64;
    float weight_decay, grad_scale, grad_norm;
    unsigned int seed;
    uint64_t flags;
    float T_spike             = 50;
    bool isBitParam           = false;
    bool isStochasticRounding = true;
    QUANT_ALG tpQuant;

    virtual void Update(GTensor* tensor_, float wd, float _grad_scale, unsigned int _seed, int flag = 0x0) {}
    virtual void CU_core(cudaStream_t stream, int flag = 0x0) {}
};
typedef std::shared_ptr<PIPE_Optimizer> hPipeOpt;

template <typename Tp, typename Tmv>
// struct PIPE_Adamw : public MODEL_CARD {
struct PIPE_Adamw : public PIPE_Optimizer {
    Tp *params, *grads0, *paramX = nullptr;
    Tmv* gm = nullptr;
    Tmv* gv = nullptr;
    float beta1, beta2, beta1_correction, beta2_correction, eps;

    PIPE_Adamw(Optimizer* hOPT_, int _flags, float _learning_rate, float _beta1, float _beta2, float _eps, float _weight_decay) {
        hOPT = hOPT_;
        // num_parameters = _num_parameters, w_stride = _w_stride, g_stride = _g_stride, s_stride = _s_stride;
        flags = _flags, learning_rate = _learning_rate, beta1 = _beta1, beta2 = _beta2, eps = _eps, weight_decay = _weight_decay;
        //  grad_scale = _grad_scale, grad_norm = _grad_norm, seed = _seed;
        // lr_0 = learning_rate;
    }

    void Update(GTensor* tensor_, float wd, float _grad_scale, unsigned int _seed, int flag = 0x0) override {
        tensor = tensor_;
        assert(tensor != nullptr);
        num_parameters = tensor->size();
        w_stride = num_parameters, g_stride = num_parameters, s_stride = num_parameters;
        name       = tensor->name;
        flags      = tensor->flags;
        grad_scale = _grad_scale, grad_norm = tensor->gnorm, seed = _seed;
        weight_decay  = wd;
        learning_rate = hOPT->LearningRate();
        iter          = hOPT->GetITER();
        arrNorm       = (double*)GTensor::stat_info;  //  sizeof(float)*5120
        assert(arrNorm != nullptr);

        params = (Tp*)(tensor->data), grads0 = (Tp*)(tensor->grad);
        gm = (Tmv*)tensor->gm, gv = (Tmv*)tensor->gv;
        // tile_r0 = tensor->tile_r0,tile_c0= tensor->tile_c0;
        tile_r1 = tensor->tile_r1, tile_c1 = tensor->tile_c1;
        memcpy(ne, tensor->ne, sizeof(ne));
        isBitParam = BIT_TEST(tensor->flags, GTensor::F_TERNARY);
        if (isBitParam) {
            assert(tensor->isWMAT());  // only for 2D weight
            learning_rate *= 3;  //  1-bit models often exhibit greater training stability compared to their full-precision counterparts, allowing for more
                                 //  aggressive initial learning steps.
            paramX = ToX(gBUFF->tmpTernary);
            gama_T = tensor->gama_T();
        }
        tpQuant = tensor->tpQuant;
    }

    void CU_core(cudaStream_t stream, int flag = 0x0) override;
};

template <typename Tp, typename Tmv>
struct PIPE_Muon : public PIPE_Adamw<Tp, Tmv> {
    float mui = 1.0, eps_muon = 1e-7;
    float a = 3.4445, b = -4.7750, c = 2.0315;
    int most_iter = 5, ldAB = 0, dimA = -1;
    bool isTrans = false;
    MUON_params_ muon;                                                                          //  A[ldAB:ldAB]
    Tmv *mG = nullptr, *A = nullptr, *B = nullptr, *BX = nullptr, *X = nullptr, *Xt = nullptr;  //  mG - Momentum matrix
    PIPE_Muon(Optimizer* hOPT_, int _flags, float _learning_rate, float _beta1, float _beta2, float _eps, float _weight_decay)
        : PIPE_Adamw<Tp, Tmv>(hOPT_, _flags, _learning_rate, _beta1, _beta2, _eps, _weight_decay) {
        muon           = this->hOPT->TrainParams().muon;
        ldAB           = muon.ldAB;
        this->eps_muon = muon.eps;
        this->mui      = muon.mui;
    }

    void Update(GTensor* tensor_, float wd, float _grad_scale, unsigned int _seed, int flag = 0x0) override;

    void CU_core(cudaStream_t stream, int flag = 0x0) override;
};

#define PROF_TOKEN(bytes) ((0xCDAFull << 48) | (bytes))

//  For cudaLaunchCooperativeKernel
template <typename T = void>
struct CoopLayer {
    T *rms_att_weight = nullptr, *bqkv = nullptr, *rms_ffn_weight = nullptr;
    T *wq = nullptr, *wk = nullptr, *wq_norm = nullptr, *wk_norm = nullptr, *wv = nullptr, *wo = nullptr;
    T *moegate = nullptr, *w1 = nullptr, *w2 = nullptr, *w3 = nullptr;
    floatGama *gama_1 = nullptr, *gama_2 = nullptr, *gama_3 = nullptr;
};
#define CoopLayer_MAX 128
#define KV_SINKS 2  // attention sinks for rolling buffer

/*
    So strange that no way to call host-function in a CUDA kernel! So have to use this pipe to transfer some data to kernel
    T - type of activation
    KVT
    Tw - type of weight
*/
template <typename T, typename KVT, typename Tw>
struct KERNEL_PIPE : public MODEL_CARD {
    using tpActivation = T;
    using tpKV         = T;
    using tpWeight     = Tw;

    CoopLayer<void>* cLayers = nullptr;
    int layNo                = -1;
    hFISH hFish              = nullptr;
    hGensor out_weight       = nullptr;
    //  inpL = gBUFF->outL->Partial("inpL", 0, {dim, 1, 1})
    hGensor inpL = nullptr;
    float* att   = nullptr;  // buffer for scores/attention values (N_HEADS, seq_len)
    // T *att = nullptr;     nearly same as float*att !
    T *x = nullptr, *xb = nullptr, *xb2 = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *exp = nullptr;
    T *hb = nullptr, *hb2 = nullptr, *he = nullptr;
    T* xlogit = nullptr;

    KERNEL_PIPE(hFISH hFish_, int pos_, int flag = 0) : hFish(hFish_) {
        size_t szMost = hFish->MostMemSize();
        bw            = PROF_TOKEN(szMost);
        auto config   = hFish->config;
        vocab_size    = config.model.vocab_size;
        assert(vocab_size > 0);
        dim        = config.nEmbed();
        n_layers   = config.nLayer();
        hidden_dim = config.n_ff();
        n_heads    = config.n_head();
        n_kv_heads = config.n_head_kv();
        head_dim   = config.head_dim();
        seq_len    = config.n_ctx();  // model.seq_len;
        rope_theta = config.model.rope_theta;
        rotary_dim = config.model.rotary_dim;
        q_dim = config.Q_dim(), kv_dim = config.KV_dim(), att_dim = n_heads * seq_len * 2;

        n_experts    = config.model.n_experts;
        n_experts_ac = config.model.n_experts_ac;
        assert(n_experts_ac == 0);
        n_experts_ac = max(n_experts_ac, 1);  //	???

        hb_dim = hidden_dim;
        assert(dim % 32 == 0 && kv_dim % 32 == 0 && hidden_dim % 32 == 0);
        he_dim = n_experts_ac * hidden_dim;
        // if (getenv("CUDA_INJECTION64_PATH")) {
        // 	coopperf = (uint64_t*)cuda_devicealloc(sizeof(uint64_t) * 16);
        // 	CUDA_CHECK(cudaMemset(coopperf, 0, sizeof(uint64_t) * 16));
        // }
        size_t nzTmp  = 0;
        typNUMBER tpD = typNUMBER::BF16;  // TYPE_<T>();
        if (tX == nullptr) {
            //  gBUFF->outL->ReShape({dim * 16 + vocab_size}, tpD);
            tX = gBUFF->outL;
        }
        x = TO<T>(tX);

        inpL = tX->Partial("inpL", 0, {dim, 1, 1});  //  ->Partial("partialZ", nZ, {dB, T, C}), subDelta = delta->Partial("partialDeltaZ", nZ, {dB, T, C});
        if (DEBUG.T_cpu) {
            xb = x + dim, xb2 = xb + dim;
            hb = xb2 + dim, hb2 = hb + hidden_dim;
            q     = hb2 + hidden_dim;
            k     = q + q_dim;
            v     = k + kv_dim;
            att   = (float*)(v + kv_dim);
            exp   = (T*)(att + n_heads * seq_len);
            nzTmp = exp - x + n_experts + (n_experts_ac ? n_experts_ac : 1) * 2;
        } else {
            xb = x + dim, xb2 = xb + dim;
            hb = xb2 + dim, hb2 = hb + hb_dim;
            q   = hb2 + hb_dim;
            att = (float*)(q + q_dim);  //  hard-code
            // xlogit = (T*)(att + dim);
            nzTmp = (char*)(att + n_heads * seq_len) - (char*)x;
        }
        assert(tX->nByte() >= nzTmp);
        tX->Zero();

        hCache = hFish->curCache();
        // key_cache       = (KVT*)hCache->Get(KVCache::KV_KEY);
        // val_cache       = (KVT*)hCache->Get(KVCache::KV_VAL);

        norm_eps   = config.model.norm_eps;
        theta_log2 = log2(config.model.rope_theta);
        qkv_clip   = config.model.clip_qkv;
        assert(n_layers > 0 && dim > 0 && hidden_dim > 0 && head_dim > 0 && n_heads > 0 && n_kv_heads > 0);
        norm_ln      = config.model.isNormalBias;
        weight_dbits = BitPE(config.model.tpWeight);

        // InitCLayer(flag);
        out_weight = hFish->GetGensor("model.output.weight", 0);
        cLayers    = new CoopLayer<void>[CoopLayer_MAX];

        // UpdatePos(pos_);
    }
    virtual void UpdatePos(int pos_) {
        // following "attention sinks" from StreamingLLM we keep the first few tokens in the KV cache as is
        pos = pos_;
        assert(pos >= 0);
        int kv_sink = pos >= seq_len ? KV_SINKS : 0;
        kv_pos      = kv_sink + (pos - kv_sink) % (seq_len - kv_sink);
        kv_len      = pos >= seq_len ? seq_len : pos + 1;

        tX->Zero();
    }

    virtual ~KERNEL_PIPE() { delete[] cLayers; }

    static hGTensor tX;  // gBUFF->outL;
    uint64_t bw;
    uint64_t* perfstats = nullptr;  //	"CUDA_INJECTION64_PATH"
    hKVCache hCache     = nullptr;
    // KVT* key_cache = nullptr;
    // KVT* val_cache = nullptr;
    //  dim=config.nEmbed();
    int dim, hidden_dim, head_dim, q_dim = -1, hb_dim = -1, he_dim = -1, kv_dim = -1, att_dim = -1, rotary_dim;
    int n_layers, n_heads, n_kv_heads, weight_dbits, n_experts, n_experts_ac, seq_len;
    bool norm_ln = false, act_gelu = false;
    bool norm_par = false;  // use parallel MLP/attention by omitting intermediate normalization
    int kv_len, kv_pos, pos;
    float norm_eps, theta_log2, qkv_clip;

    virtual void InitLayer(int l, int flag = 0x0) {
        RLS_BP* hRLS = hFish->GetScheduler<RLS_BP>();
        assert(hRLS != nullptr);
        assert(l >= 0 && l < n_layers);
        layNo              = l;
        SelfAttention* QKV = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        // QKV->BeforeMing(hRLS, nullptr);
        cLayers[l].rms_att_weight = TO<float>(QKV->norm.w);  // weights->rms_att_weight[l];
        cLayers[l].wq = ToX(QKV->Q.w), cLayers[l].wk = ToX(QKV->K.w), cLayers[l].wv = ToX(QKV->V.w);
        cLayers[l].wq_norm = ToX(QKV->normQ.w), cLayers[l].wk_norm = ToX(QKV->normK.w);
        cLayers[l].wo   = ToX(QKV->proj_cat.w);
        cLayers[l].bqkv = ToX0(QKV->bqkv);
        FFN* ffn        = hFish->GetNeuron<FFN>("FFN", l);
        // ffn->BeforeMing(hRLS, nullptr);
        cLayers[l].rms_ffn_weight = TO<float>(ffn->norm.w);  // weights->rms_ffn_weight[l];
        cLayers[l].moegate        = nullptr;                 // weights->moegate[l];
        hGensor w1 = ffn->GetGensor("", FFN_UP, ".weight"), w2 = ffn->GetGensor("", FFN_DOWN, ".weight"), w3 = ffn->GetGensor("", FFN_GATE, ".weight");
        cLayers[l].w1 = w1->data, cLayers[l].gama_1 = w1->gama_T();
        cLayers[l].w2 = w2->data, cLayers[l].gama_2 = w2->gama_T();
        cLayers[l].w3 = w3->data, cLayers[l].gama_3 = w3->gama_T();
        if (cLayers[l].bqkv != nullptr)
            PrintTensor<Tw>("bqkv", (Tw*)(cLayers[l].bqkv), true, dim, 1);
        PrintTensor<Tw>("wq", (Tw*)(cLayers[l].wq), true, q_dim, dim);
        PrintTensor<Tw>("wk", (Tw*)(cLayers[l].wk), true, kv_dim, dim);
        PrintTensor<Tw>("wv", (Tw*)(cLayers[l].wv), true, kv_dim, dim);
    }

    virtual void AfterLayer(int l, int flag = 0x0) {
        RLS_BP* hRLS = hFish->GetScheduler<RLS_BP>();
        assert(hRLS != nullptr);
        assert(layNo == l);
        SelfAttention* QKV = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
        QKV->AfterMing(hRLS, nullptr);
        FFN* ffn = hFish->GetNeuron<FFN>("FFN", l);
        ffn->AfterMing(hRLS, nullptr);
    }
};

// typedef KERNEL_PIPE<bf16, bf16, bf16> QWEN3_PIPE;
typedef KERNEL_PIPE<floatX, floatX, floatX> QWEN3_PIPE;
typedef KERNEL_PIPE<float, bf16, __nv_fp8_e5m2> QWEN_CALM_PIPE;