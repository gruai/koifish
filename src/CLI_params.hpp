/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once
#include <cassert>

#include "./Utils/GST_log.hpp"
#include "./Utils/json.hpp"
#include "./g_float.hpp"

/**
 *  All paramters defined here
 */
struct CLI_params;

/*
    生者一过客，
        (activations)
    逝者一归尘；
        (sparse)
    天地一逆旅，
        (back propagation)
    知止一至人
        (local minima)
*/
enum LIFE_PHASE {
    // Pre-training
    P_TRAIN,
    P_SFT,  //  supervised fine-tuning
    P_DPO,  //  direct preference optimization

    // evaluate
    P_EVAL_,
    // Inference     fish->isLocalInfer = true
    P_PREFILL,
    P_GENERATE,
    P_CHAT = P_GENERATE
};

enum CHAT_MODE {
    YABA,
    CHAT_SMOKE,
    CHATML_ASSIST,
    CHATML_THINK,
};

enum METRIC { M_VOID, CLS_LOSS, PPL, METRIC_MOST };

enum COMPRESSIVE_SENSING {
    SKIP,
    SVD,
    SVD_a,
    GBTQ,
    GBDT,
    LORA,
    EoRA,    // Eigenspace Low-Rank Approximation
    SAMPLE,  // random sub-sampling
};

enum NORMAL_MODE {
    NO_NORMAL,
    SINKHORN,
    ROW_01,
};

/**
 *
 */
enum ACTIVATION_FUNC { SIGMOID, RELU2, RELU_, LINEAR, GELU, GLU, SWIG };

/**
 *
 */
enum FFN_TYPE {
    SWIGLU = 0,
    VANILLA,
    ONLY_LNormal,
    ONLY_RMSNormal,
    VAR_0,
    VAR_LAST,  // last layer with gaussian noise
    SMOE,      // Sparsely-Gated Mixture-of-Experts Layer
    GATE_CYS,
};

enum GD_METHOD {
    ADAMw = 0x0,
    SGD,
    SGD_v,
    SGD_blk_v,
    SGD_HYBRID,
    LION,
    ADAM_MINI,  //  https://arxiv.org/pdf/2406.16793
    ADAM_S,     //  https://arxiv.org/pdf/2505.16363
    ADAM_GALORE,
    // ADAM_spike,
    MUON,
    // ADAMw_cuda,
};
static std::string GD_NAME[] = {"ADAMw", "SGD", "SGD_v", "SGD_blk_v", "SGD_HYBRID", "LION", "ADAM_MINI", "ADAM_S", "ADAM_GALORE", "MUON"};

enum MODEL_ARCH {
    _X_,
    NLP_GPT2,
    NLP_GPT2_char,
    NLP_LLAMA,
    NLP_MISTRAL,
    NLP_MAMBA,

    NLP_QWEN2,
    NLP_QWEN3,
    NLP_DEEPSEEK,

    NLP_GUPPY,

    NLP_MOE,  //???
              //////
    SCORE_,
    SAM_
};

enum MEM_STRATEGY {
    PRE_ALLOC_GPU,
    PRE_ALLOC_HOST_MAP,
    MEM_SWAP,        //  deprecated
    MEM_SWAP_GUOKE,  //  deprecated
};
static std::map<MEM_STRATEGY, std::string> MEM_STRATEGY_desc = {
    {PRE_ALLOC_GPU, "PRE_ALLOC_GPU"},
    {PRE_ALLOC_HOST_MAP, "PRE_ALLOC_HOST_MAP"},
    {MEM_SWAP, "SWAP"},
    {MEM_SWAP_GUOKE, "SWAP_GUOKE"},
};
// parameters of scheduling
struct SKDU_params {
    MEM_STRATEGY strategy = PRE_ALLOC_GPU;

    bool isUpdateParamV0() const;
    bool canSave(int iter, int flag = 0x0) const;
    void Dump(int typ) const;
};

struct Fuyou_params {
    enum ENSEMBLE {
        AGGREGATION,
        FUYOU_BEST,
        RANDOM_1,
        MULTI_SCALE,
    };
    ENSEMBLE ensemble                      = FUYOU_BEST;
    std::vector<std::string> filter_reload = {"ffn", "attn.wq", "attn.wk", "attn.wv", "attn.wo"};
    // Only current fuyou's param in GPU, others in MMAP
    bool paramIsGuoke = false;
    enum ROLE {
        F_COMMON,
        F_HEAD,
        F_FOLLOWER,
    };
    enum ALGORITHM {
        NO_EVOL,
        PARTICLE_SWARM,
        GENE_MIX,
        GENE_MUTATION,
        PARTICLE_GENETIC,
        KEPLER,
        GREY_WOLF,
        PANG_PI,
        SALP,
    };
    std::map<ALGORITHM, std::string> Algo2Name = {
        {NO_EVOL, ""}, {PARTICLE_SWARM, "pso"}, {GENE_MIX, "mix"}, {GENE_MUTATION, "mutation"}, {PARTICLE_GENETIC, "pso_ga"},
    };
    ALGORITHM algorithm = NO_EVOL;

    // layer in branch
    int nLayerInBranch = -1, LIB_0 = -1, LIB_1 = 0, LIB_iter_switch = 100;
    int nBranch = -1;
    int nWarmup(int flag = 0x0);
    // int tpParamResident = 0;  //  0-      1-
    bool isON() { return nBranch > 0; }
    bool Init(CLI_params* hConfig, const JSON& jConfig, int flag = 0x0);
    bool InitSection(int nLayer, int nLS, int nSwitch = 100, int flag = 0x0);
    bool isFirst(int layer, int flag = 0x0);
    float alpha     = 0.9;
    float cognitive = 0, social = 2;              // exploration and exploitation
    float T_crossover = 0.6, T_mutation = 0.001;  //  mutation:   [0.001–0.1]

    void Dump(int typ) const;
};

enum tpNEURON4NAME {
    ATTN_PRE_NORMAL,
    ATTN_Q_NORM,
    ATTN_K_NORM,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_OUT,
    FFN_PRE_NORMAL,
    FFN_UP,
    FFN_RELU,
    FFN_DOWN,
    FFN_GATE,
    LN_RSTD
};

enum tpROPE {
    ROPE_NONE = -1,
    ROPE_NORM = 0,
    ROPE_NEOX = 2,
    ROPE_GLM  = 4,
};

enum INIT_WEIGHT { W_SKIP = 0X0, FIX_1, RANDOM, GAUSSIAN_NORMAL, COPY_WIKI, COPY_SWARM_HEAD, SERIALIZE };
enum QKV_PACK {
    QQKKVV,
    QKVQKV,
    Q_K_V,  // separate three tensor
};
/**
 * should have config.json,tokenizer.json & tokenizer_config.json
 * generation_config.json
 * model.safetensors.index.json
 */
class MODEL_CARD {
   protected:
    /*
        void* wq[MAX_LAYERS]; // (n_heads * head_dim, dim)
        void* wk[MAX_LAYERS]; // (n_kv_heads * head_dim, dim)
        void* wv[MAX_LAYERS]; // (n_kv_heads * head_dim, dim)
        void* wo[MAX_LAYERS]; // (dim, n_heads * head_dim)
        // weights for ffn
        void* w1[MAX_LAYERS]; // (n_experts?, ff, dim)
        void* w2[MAX_LAYERS]; // (n_experts?, dim, ff)
        void* w3[MAX_LAYERS]; // (n_experts?, ff, dim)
        // biases for qkv (qwen)
        float* bqkv[MAX_LAYERS]; // ((n_heads + n_kv_heads * 2) * head_dim)
        // moe gate weights (mixtral)
        void* moegate[MAX_LAYERS]; // (n_experts, dim)
    */
    struct LAY_PARAM {
        uint32_t _head, _kv_head, _head_dim;
        uint32_t ff;
        LAY_PARAM(uint32_t h, uint32_t k, uint32_t hd, uint32_t f) : _head(h), _kv_head(k), _head_dim(hd), ff(f) { assert(h > 0 && k > 0 && f > 0 && hd > 0); }
        virtual void SetHead(int nH) {
            assert(nH > 0 && nH < 1024 * 1024);
            _head    = nH;
            _kv_head = nH;
        }
        virtual void SetFF(int nF) {
            assert(nF > 0 && nF < 1024 * 1024);
            ff = nF;
        }

        uint32_t n_head() const { return _head; }
        uint32_t n_head_dim() const { return _head_dim; }
        uint32_t n_head_kv() const { return _kv_head; }
        uint32_t n_ff() const { return ff; }

        /**
         * The GQA model efficiently breaks the query into n_heads, and the key and value are divided into n_kv_heads groups,
         * enabling multiple key-value heads to share the same query.   more groups (closer to MHA) result in higher quality but slower performance, whereas
         * fewer groups (near to MQA) boost speed at the risk of sacrificing quality.
         */
        uint32_t n_gqa() const {
            assert(_head >= _kv_head && _head % _kv_head == 0);
            return _head / _kv_head;
        }
    };

    std::vector<LAY_PARAM> layerps;

   public:
    static std::string sWeight, sBias, sLayer, sEmbed, sInvEmbed, sQzeros, sQscale;
    bool enable_thinking = false;
    bool isSparse() { return sparse.method != 0; }
    struct Sparsing {
        int method = 0;  //  1-GBDT
        std::string model_path;
    };
    Sparsing sparse;
    // MODEL_ENSEMBLE ensemble = Fuyou_params::FUYOU_BEST;
    ACTIVATION_FUNC fActFFN = SWIG, fActSLP = SWIG;
    INIT_WEIGHT tpInitWeight = INIT_WEIGHT::RANDOM;

    std::string sCardPath = "", sTokenJsonPath = "", sTokenBinPath = "";
    std::string sArch, torch_dtype, transformers_version, model_type;
    std::string act_type, norm_type;
    typNUMBER tpWeight = typNUMBER::BF16, tpGradient = typNUMBER::BF16, tpActivation = typNUMBER::BF16;
    // typNUMBER tpPreLogits = typNUMBER::F32;tpEmbed = typNUMBER::BF16,
    dotprod_t fDotW;
    JSON jModelParam, jSafetensorsIndex;  //
    int vocab_size = -1;
    // 1. in some model, no bos_token_id!(GPT-2/GPT-3,unsloth/Qwen3-4B-Base,...)
    int bos_token_id, eos_token_id;
    // Instruct model always has there templates! Base Model may has no template
    std::string prompt_template, system_prompt_template;
    bool isInstructModel() {
        // system_prompt_template.empty()
        return !prompt_template.empty();
    }
    int preLogits_dB     = 2;  //
    bool isNormalBias    = true;
    bool isSLPBias       = true;
    bool isQKVBias       = true;
    bool isPaddedCls     = false;
    bool isFFNShareParam = false;

    // dim(=head_dim*n_heads)
    int dim = -1, hidden_dim = -1, n_layers = -1, hidden_size = -1, intermediate_size = -1;
    int max_pos_embeddings = -1, num_attention_heads = -1, num_key_value_heads = -1;
    std::vector<int> token_embeds;
    std::vector<int> qkv_embeds;  // try multi-level embed of QKV

    //  ****
    bool isFFNWeightTying   = true;
    
    /*
    1. If true, reduces the memory,  often results in better and faster outcomes ???
    2. According to findings from OLMo the weight tying is beneficial for smaller models like 1B but for larger ones starting from 7B it starts to hurt the
    performance - instability in loss curves. I don't know why it is not discussed in their paper, but one of the researchers is talking about it in TWIML AI
    podcast around 16:50: https://youtu.be/mwS9zPCv_dY?t=1010
    3. output softmax wants embeddings to be very large so their inner products will produce very different values.
        input embeddings want a much smaller range so they can have stable dynamics throughout training.
        All the "old" code bases had this scalar (usually sqrt(d)) but the llama arch dropped this when they started untying
    4. Use Weight Tying Only When the Distributional Hypothesis Holds   ???
    */
    bool isEmbedWeightTying = false;
    std::vector<std::string> skip_st;

    bool isQKNormal    = false;
    bool isSeparateQKV = false;
    QKV_PACK qkv4dnn   = QKV_PACK::QKVQKV;  // the  fromat of input var_PACK for cudnn GRAPH
    bool isBqkv        = false;
    float clip_qkv     = FLT_MAX;  // Clipping Q/K/V.  to prevent numerical instability
    size_t nTotalSize  = 0x0;
    std::map<std::string, void*> st_map;  // map of st in jSafetensorsIndex
    //  Experts
    int n_experts = 0, n_experts_ac = 0;
    //  eps
    float norm_eps = 1e-5f, norm_rms_eps = 1e-5f;
    //  rope
    float rope_freq_base = 10000.0f, rope_freq_scale = 1.0f, rope_theta = 0;

    tpROPE rope_type = ROPE_NORM;
    int rotary_dim   = 0;
    // int seq_len      = 0;

    MODEL_CARD();
    // virtual bool OnJsonCALM(CLI_params* hConfig, const std::string& path, const JSON& meta, int flag = 0X0);
    // more param from HF's "model_card"
    virtual bool InitHugFace(CLI_params* hConfig, const JSON& jConfig, const std::string& sCardPath, int flag = 0x0);
    virtual bool InitChatTemplate(CLI_params* hConfig, int flag = 0x0);
    bool isLoadCard() { return !sCardPath.empty(); }
    void Dump(int typ);

    friend class CLI_params;
};

enum EVICTION_MODE {
    NO_EVICTION,
    H2O,  //  Heavy Hitter Oracle
};

/**
 * Quantizer is much complex & hard than most people think. It refects the essence of our world, just like quantum mechanics
 */
enum QUANT_MODE {
    NO_QUANT,
    
    RTN,    //  Round-to-Nearest
    RTN_ZS, //  Same as RTN, some vendors use explicit zero/scale tensors
    RTNf,   // nf4, nf3
    MINI,   // Minimise impurity
    F8Ex,

    // AWQ,     // awq is the quant method(find 1% salient/outlier weights by activataion), and the qdata format is still RTN

    KV_JL,     //  Johnson-Lindenstrauss (JL) transform
    KV_AQUA,   //  https://arxiv.org/pdf/2501.19392
    KV_SQUAT,  //  Subspace-orthogonal KV cache quantization   https://arxiv.org/pdf/2503.24358
    KV_PolarQuant
};

// Deprecated
enum QUANT_ALG {
    // Ternary
    // Value of weights
    W_SCALE = 0X100,
    W_NOSCALE  //
};
struct QUANT_CARD {
    int TransA = 1;
    int default_bits = 4;
    int nPassLayer   = 0;  // fist layer is hard to quant
    float T_errQ     = 0.3;
    int T_group = 512, T_group_batch = 8;
    SHAPE spMost;
    std::string sX = "";

    bool isVendorQuant = false;
    bool isNormalFloat = true;  //  each bin under a normal distribution N(0,1) contains equal probability mass
    bool isSymmetric   = false;
    bool isZeroPoint   = false;
    typNUMBER tpZero, tpScale, tpQWeight;
    
    NORMAL_MODE norm = SINKHORN;
    enum DYNAMIC_MODE { NO_DYNAMIC };
    DYNAMIC_MODE dynamic = NO_DYNAMIC;
    // dynamic & adpative, only set type at runtime
    QUANT_MODE type = NO_QUANT;

    QUANT_CARD() {}
    // virtual void InitFromVendor(const JSON& jVendor, int flag = 0x0);
    virtual void Init4Neuron(const std::string& name, const JSON& jQuant, int flag = 0);

    virtual bool isPass() const { return type == NO_QUANT; }
    // std::vector<std::string> filter_KVcache;

    // Koifish needs quant params for each neuron. 
    static JSON Vendor2JSONx(const JSON& jX, int flag = 0x0);
    virtual bool isValid() const;
    void Dump(int typ);
    std::size_t Hash(const QUANT_CARD& params, const std::type_info& ti, const std::string& desc) const;
};

struct ADAM_params_ {
    size_t n_parameters;
    int decay_min_ndim;
    float alpha, min_alpha, decay = 0, beta1, beta2;
    int clip_alg = 0;  // if 1: adaptive local clip
    float gclip;
    float eps      = 1e-8f;  // epsilon for numerical stability
    float eps_loss = 1e-5f;  // epsilon for convergence test

    void Dump(int typ);
};

struct MUON_params_ {
    size_t n_parameters;
    enum Orthogonalization {
        NewtonSchulz,
        Chebyshev,  //  https://github.com/GrishKate/accelerating_orthogonalization
        Gluon,      //  https://arxiv.org/pdf/2505.13416

    };
    Orthogonalization tpOrthogonal = NewtonSchulz;
    MUON_params_();
    bool isNesterov  = true;
    bool isTransDown = true;
    bool isAdamW(void* hUserData, int flag = 0x0);
    float lr_scale = 50.f;  // 100.f 50.f?
    //  torch:  self←self+λ⋅(b−self)          sAtB(a, b, λ):  a+λ*(b-a)
    float mui      = 0.95;
    float eps      = 1e-7f;  // epsilon for numerical stability
    float eps_loss = 1e-5f;  // epsilon for convergence test
    int ldAB       = 0;
    //  0:No decay 1: equal decay  2:  high decay due to high LR!
    int tpDecay = 0;
    void Dump(int typ);
};

struct TRAIN_CARD {
    int dump_every = 1;
    int gpt_every  = -1;  // eval_every=-1,

    int seed     = -1;
    int n_epochs = -1;
    bool Empty() { return n_epochs < 0; }

    int n_ctx = -1, n_batch = -1, n_threads = -1, n_gradient_accumulation = -1, n_gpu_layers = -1;

    bool custom_n_ctx;
    bool isSuperBatch = true;
    bool use_flash;
    bool use_checkpointing;

    std::string sample_start;
    bool include_sample_start;
    bool escape;
    bool overlapping_samples;
    bool fill_with_next_samples;
    bool separate_with_eos;
    bool separate_with_bos;
    bool sample_random_offsets;

    bool force_reshuffle;

    float rSubSample = 1;
    int nMostIter = -1, nEpochIter = -1;
    int warmup, lr_restart         = 0;
    // int   cos_decay_steps;
    // float cos_decay_restart;
    // float cos_decay_min;
    // bool  enable_restart;

    int opt_past;
    float opt_delta;
    int opt_max_no_improvement;
    int opt_stochastic_rounding = 0;
    int opt_alloc_weight        = 0;

    int remater_ffn    = 0;
    std::string method = "muon";
    ADAM_params_ adam;
    MUON_params_ muon;
    float residual_scale = 1.0f;
    float LearningRate() const { return adam.alpha; }
    size_t nTokenInBatch() const { return n_batch * n_ctx; }
    TRAIN_CARD();
    bool Init(CLI_params* hConfig, const JSON& jConfig, int flag = 0x0);
};

struct CHAT_SAMPLER {
    enum METHOD { TEMPERATURE, Top_K, Top_P, Min_P, BEAM };
    METHOD method  = METHOD::TEMPERATURE;
    CHAT_MODE mode = CHAT_MODE::YABA;

    float temperature = 0.6f;
    float top_p       = 0.95f;
    int top_k         = 20;
    bool ignore_eos   = false;  // ignore EOS token when generating text
    bool isSampleCPU  = false;

    int repeat_last_n        = 64;
    float repeat_penalty     = 1.00f;
    bool interactive         = false;
    int32_t interactive_port = -1;

    std::string prompt     = "";
    std::string token_test = "";
    // Define the length of batch input,   different with n_ctx_train, n_ctrx_origin !!!
    int seq_len  = 8192;
    int szBuffer = 32768;
};

struct DUMP_SWITCH {
    int tensor_ref             = 0;
    int train_time             = 0;
    int tensor_load            = 0;
    int nn_structure           = 1;
    std::string train_csv_path = "";
};

struct DEUG_SWITCH {
    float fLongTail          = -1;
    int SelfAttention_noraml = 1;
    bool NO_loss             = false;
    bool check_tensor_norm   = false;

    int test_quant = 0;

    int dict_latent_dim    = -1;
    int graph_dump         = 0;  //  10 levels of dumps, 0-9. 0 is a full dump_,The lower the number the more dump_.
    int train_hyperparams  = 0;
    int train_datas        = 0;
    int back_graph_version = 0;
    int verCuda            = 0;
    int verInferQKV        = 1;
    int verInferFFN        = 1;
    int verCuX2            = 0;
    int verShuffleSamp     = 0;
    int verSampJump        = 0;
    int verGenerate        = 0;

    int T_ternary        = 0;
    int T_classifier_ver = 0;
    int T_cpu            = 0;
    int T_GEMM           = -1;
    int T_fuyou          = 1;

    int T_cuQK                = 0;
    int T_generate_most_layer = 10000000;
    int T_kvcache_quant       = 0;
    int N_mostiter            = -1;

    int cmd_p1 = 0, cmd_p2 = 0, cmd_p3 = 0;  // some commandline parameter for debug
    int x1 = 0;
    std::string x_str;
    std::vector<std::string> prompts;

    float Time_most = 60;
    void Dump(int typ);
};
extern DEUG_SWITCH DEBUG;

enum LORA_ADAPT_W { W0, AB, W_AB, refW_AB, refW_AB_ffn };

/*
    1. Genenal checkpoint(Koifsh support load/save multiple checkpoints/model files)
    2. Has at least one checkpoint (with all parameters & its moments)
*/
struct CheckPoint_Params {
    std::string jKey = "";  //  unique id of checkpoint

    enum TYPE {
        STATE,  //  Has all parameters & its moments
        BEST,   //  Only has parameters of best fuyou
        FULL,   //  Has parameters of all fuyou

    };
    enum FORMAT {
        HF,  //   HugFace transformers compatible,
        KOIFISH,
    };

    int curEpoch = -1, curIter = -1, curFuyou = -1;
    std::vector<std::string> fuyou_filter_reload;

    std::map<std::string, uint32_t> seeds;
    // More variables of current state
    std::map<std::string, double> variabls;

    TYPE type     = BEST;
    FORMAT format = KOIFISH;
    int iter      = -1;
    void* hAllST  = nullptr;
    CheckPoint_Params() {}
    CheckPoint_Params(const JSON& jConfig, const std::string& key, bool isSave, int flag = 0x0);
    // CheckPoint_Params(const std::string& tp, const std::string& p, int x, bool in = false);
    // bool isIn = false;
    // std::string in, out;
    // std::string model_out, model_base;
    std::string sDir, sModelPath, sX;
    int save_every = -1;
    std::string FullPath(bool isSave, int flag = 0x0);
    bool empty() { return sDir.empty(); }
    bool needSave(int it, int flag = 0x0) {
        if (save_every > 0 && it % save_every == 0) {
            return true;
        }
        return false;
    }

    virtual void Init(int flag = 0x0);
    virtual bool SerialSnap(JSON& jSnapshot, bool isSave, int flag = 0x0);
};
static std::map<CheckPoint_Params::TYPE, std::string> CKP_ext = {
    {CheckPoint_Params::STATE, "ckp"},
    {CheckPoint_Params::BEST, "fish"},
    {CheckPoint_Params::FULL, "fish"},
};
static std::map<CheckPoint_Params::TYPE, std::string> CKP_desc = {
    {CheckPoint_Params::STATE, "state"},
    {CheckPoint_Params::BEST, "best"},
    {CheckPoint_Params::FULL, "full"},
};

struct CLI_params {
    LIFE_PHASE phase = LIFE_PHASE::P_TRAIN;

    TRAIN_CARD common;
    MODEL_CARD model;

    std::vector<CheckPoint_Params> ckp_in, ckp_out;
    CheckPoint_Params state;
    void InitAllStates(int flag);

    DUMP_SWITCH dumpSwitch;

    struct DataTypes {
        std::vector<std::string> arrTernary;
        std::vector<std::string> arrTile;
    };
    DataTypes datatypes;

    SKDU_params scheduling;
    Fuyou_params fuyou;
    LORA_ADAPT_W tpLORA = LORA_ADAPT_W::W0;

    std::string eval_metric                  = "";
    std::vector<std::string> filter_tmp_grad = {"ffn", "attn.wq", "attn.wk", "attn.wv", "attn.wo"};

    // Always false,     GGML don't support back of FLASH_ATTEN !
    bool isFlashAtten() {
        common.use_flash = false;
        return common.use_flash;
    }

    uint32_t nThread() const;
    /**
     * 1. latent of TokenEmbed
     * 2.
     */
    uint32_t nEmbed(int flag = 0x0) const;
    uint32_t nLayer() const {
        if (nLayerX > 0)
            return nLayerX;
        assert(n_layer_train > 0);
        return n_layer_train;
    }

    uint32_t n_ctx() const;  // number of tokens in each sample
    uint32_t n_ctx_orig() const { return n_ctx_orig_yarn != 0 ? n_ctx_orig_yarn : n_ctx_train; }
    void SetNCTX(int _nctx) {
        assert(_nctx > 0 && _nctx < 1024 * 1024);
        common.n_ctx = _nctx;
    }
    uint32_t max_seq_len = 0, n_ctx_orig_yarn = 0, n_ctx_train = 0;
    bool isLongRope(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        const auto n_ctx_pre_seq = n_ctx() / max_seq_len;
        bool isLong              = n_ctx_pre_seq > n_ctx_orig_yarn;
        return isLong;
    }

    uint32_t n_batch() const { return common.n_batch; }  // number of samps in each batch
    uint32_t nGradAccumulate() const { return common.n_gradient_accumulation; }
    size_t nTokenInBatch() const { return common.n_batch * common.n_ctx; }
    size_t nTokensPerGrad() const { return nTokenInBatch() * common.n_gradient_accumulation; }

    bool isShareLayerOut() const;
    std::string jsPath = "";
    JSON jConfig;
    nlohmann::ordered_json jModel, jBackBone;
    JSON jQuant, jVendorQuant;

    MODEL_ARCH ModelArch();
    virtual void OnArch();
    virtual bool isValid(int flag = 0x0);
    virtual bool JModel2Params(int flag);
    virtual void OnMostToken(size_t nMost, int flag = 0x0);
    std::string exec_name = "", test = "", compute_graph = "";
    std::vector<std::string> fn_model_base;
    //  std::string fn_vocab_model;
    std::string model_title = "";
    // std::string fp_train_data;   serial_path
    std::string train = "";  //"scratch"

    bool isOnlyGPT = false;

    CHAT_SAMPLER chat_sampler;
    CHAT_MODE ChatMode() { return chat_sampler.mode; }

    bool passLoadToken    = false;
    bool only_write_model = false;
    bool ffn_use_gate     = false;
    uint32_t n_swarm      = 1;
    // uint32_t n_outputs = 1;
    int n_embd_head_k = -1, n_embd_head_v = -1;  // nEmbed() = -1,

    int n_layer_train = -1, nLayerX = -1, nFFX = -1;
    int Fuse_Normal = 0;

    int nabla = 1;  // cys
    // std::string sigma = "";
    std::string vae           = "";
    std::string prompt        = "";
    std::string dict_vae_dims = "", dict_dialect = "", dict_logits = "";

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t wkv_head_size          = 0;

    // MOE
    uint32_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms  = false;

    template <typename T>
    bool is(const std::vector<std::string>& keys, const T& t) {
        T v0;
        bool isDump = !NOT_DUMP(0);
        T val       = jKV(jConfig, keys, v0, isDump);
        // return jKV_is(jConfig,keys,target);
        return val == t;
    }
    bool is(const std::vector<std::string>& keys, const char* t) { return is(keys, std::string(t)); }

    template <typename T>
    T Get(const std::vector<std::string>& keys, const T& t, bool isCLI = true) const {
        T val = jKV(jConfig, keys, t, isCLI);
        return val;
    }
    std::string KV(const std::vector<std::string>& keys, const std::string& t = "", bool isCLI = true) const {
        std::string val = Get(keys, t);
        return val;
    }

    // It is also not easy to change keys in a std::map
    template <typename T>
    T Set(const std::vector<std::string>& keys, const T& t) {
        //  https://github.com/nlohmann/json/issues/1723
        assert(0);
        T v0;
        T val = jKV(jConfig, keys, v0, false);
        return val;
    }
    // 1. step of eval   2.
    float step                      = 1.0f;
    float f_clamp_kqv               = 0.0f;
    float f_max_alibi_bias          = 0.0f;
    float f_logit_scale             = 0.0f;
    float f_attn_logit_softcapping  = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    // parameters of wiki
    std::string wiki_actor = "", wiki_logits = "";

    // bool only_infer = false;

    /*enum TUNE_ALG {
        OFF = 0,
        LORA,
        LORA_SVD,
        LORA_AB,
        LORA_Q,
        // VARIATIONAL,
    };
    enum TUNE_ALG tune = OFF;
    std::string sTune(int flag = 0x0) {
        std::string tune_desc[] = {
            "", "_AB", "_SVD", "_SVD_AB", "_VARIATIONAL",
        };
        return tune_desc[tune];
    }*/

    // parameters of datasets
    float rSplit = 0.1;

    std::string tpBatchSample;
    std::string tpWiki = "";
    float lars_ratio = 0.0f, ZMUV_ratio = 0.0;

    int32_t lora_r = 0, lora_alpha = 0;

    uint32_t FOMULA_n_ff(int n_mult) {
        const uint32_t n_ff = ((2 * (4 * nEmbed()) / 3 + n_mult - 1) / n_mult) * n_mult;
        return n_ff;
    }
    void SetHead(uint32_t nH) {
        for (int il = 0; il < model.layerps.size(); il++) {
            model.layerps[il].SetHead(nH);
        }
    }
    uint32_t n_head(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        return model.layerps[il].n_head();
    }
    uint32_t n_head_kv(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        return model.layerps[il].n_head_kv();
    }

    uint32_t n_ff(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        if (nFFX <= 0)
            return model.layerps[il].n_ff();
        else
            return nFFX;
    }
    uint32_t head_dim(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        int h_dim = model.layerps[il].n_head_dim();
        assert(h_dim > 0 && h_dim < 10240);
        return h_dim;
    }

    uint32_t Q_dim(int il = 0) const { return head_dim(il) * n_head(il); }
    uint32_t KV_dim(int il = 0) const { return head_dim(il) * n_head_kv(il); }

    uint32_t n_gqa(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        return model.layerps[il].n_gqa();
        /*const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);
        if (n_head_kv == 0) {
            return 0;
        }
        return n_head/n_head_kv;*/
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const {  // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const {  // dimension of value embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);
        return n_embd_head_v * n_head_kv;
    }

    uint32_t n_embd_k_s() const {  // dimension of the rolling state embeddings
        // corresponds to Mamba's conv_states size or RWKV's token_shift states size
        if (wkv_head_size != 0) {
            // for RWKV models
            return 2 * nEmbed();
        } else {
            // TODO: maybe support other convolution strides than 1
            // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
            return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
        }
    }

    uint32_t n_embd_v_s() const {  // dimension of the recurrent state embeddings
        if (wkv_head_size != 0) {
            // corresponds to RWKV's wkv_states size
            return nEmbed() * wkv_head_size;
        } else {
            // corresponds to Mamba's ssm_states size
            return ssm_d_state * ssm_d_inner;
        }
    }

    bool operator!=(const CLI_params& other) const;

    void Dump(int flag = 0x0);

    bool parse(int argc, char** argv);
    virtual bool InitJConfig(int flag = 0x0);
    virtual bool ToJConfig(int flag = 0x0);
    virtual bool InitChekcpoints(int argc, char** argv, const std::string& ckp_queue, int flag = 0x0);
    virtual JSON ToJSON(int type, int flag = 0x0);
    std::string GetDataPath(const std::string type, int flag = 0x0);
};