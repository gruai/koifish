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

//  生者一过客，逝者一归尘；天地一逆旅，只待骑鹤人
enum LIFE_PHASE {
    // Pre-training
    P_TRAIN,
    P_SFT,  //  supervised fine-tuning
    P_DPO,  //  direct preference optimization

    // evaluate
    P_EVAL_,
    // Inference     fish->isLocalInfer = true
    P_PREFILL,
    P_GENERATE
};

enum COMPRESSIVE_SENSING {
    SKIP,
    SVD,
    SVD_a,
    GBTQ,
    GBDT,
    EoRA,    // Eigenspace Low-Rank Approximation
    SAMPLE,  // random sub-sampling
};

enum MODEL_ARCH {
    _X_,
    NLP_GPT2,
    NLP_GPT2_char,
    NLP_LLAMA,
    NLP_MISTRAL,
    NLP_MAMBA,

    NLP_QWEN2,
    NLP_DEEPSEEK,

    NLP_GUPPY,

    NLP_MOE,  //???
              //////
    SCORE_,
    SAM_
};

/**
 * should have config.json,tokenizer.json & tokenizer_config.json
 * generation_config.json
 * model.safetensors.index.json
 */
class MODEL_CARD {
   protected:
    struct LAY_PARAM {
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
        uint32_t head, kv_head;
        uint32_t ff;
        LAY_PARAM(uint32_t h, uint32_t k, uint32_t f) : head(h), kv_head(k), ff(f) { assert(h > 0 && k > 0 && f > 0); }

        uint32_t n_head() const { return head; }
        virtual void SetHead(int nH) {
            assert(nH > 0 && nH < 1024 * 1024);
            head    = nH;
            kv_head = nH;
        }
        virtual void SetFF(int nF) {
            assert(nF > 0 && nF < 1024 * 1024);
            ff = nF;
        }
        uint32_t n_head_kv() const { return kv_head; }
        uint32_t n_ff() const { return ff; }

        /**
         * The GQA model efficiently breaks the query into n_heads, and the key and value are divided into n_kv_heads groups,
         * enabling multiple key-value heads to share the same query.   more groups (closer to MHA) result in higher quality but slower performance, whereas
         * fewer groups (near to MQA) boost speed at the risk of sacrificing quality.
         */
        uint32_t n_gqa() const {
            assert(head >= kv_head && head % kv_head == 0);
            return head / kv_head;
        }
        uint32_t n_embd_head(int _embd) const {
            assert(_embd > 0 && _embd % head == 0 && _embd >= head);
            return _embd / head;
        }

        uint32_t n_embd_gqa(int _embd) const {
            int gqa = n_gqa();
            assert(_embd % gqa == 0 && _embd >= gqa);
            return _embd / gqa;
        }
    };

    std::vector<LAY_PARAM> layerps;

   public:
    static std::string sWeight, sBias, sNorm, sLayer, sAttnOut;

    bool isSparse() { return sparse.method != 0; }
    struct Sparsing {
        int method = 0;  //  1-GBDT
        std::string model_path;
    };
    Sparsing sparse;

    std::string sCardPath = "", sTokenPath = "";
    std::string sArch, torch_dtype, transformers_version, model_type;
    std::string act_type, norm_type;
    typNUMBER tpWeight = typNUMBER::BF16, tpEmbed = typNUMBER::BF16, tpPreLogits = typNUMBER::F32, tpGradient = typNUMBER::BF16, tpActivation = typNUMBER::BF16;

    dotprod_t fDotW;
    JSON jModelParam;  //
    int vocab_size       = -1, bos_token_id, eos_token_id;
    int preLogits_dB     = 2;  // epsilon for convergence test
    bool isNormalBias    = true;
    bool isSLPBias       = true;
    bool isPaddedCls     = false;
    bool isFFNShareParam = false;

    // dim(=head_dim*n_heads)
    int dim = -1, hidden_dim = -1, n_layers = -1, n_heads = -1, n_kv_heads = -1, head_dim = 1;
    //  ****
    bool isFFNWeightTying   = true;
    bool isEmbedWeightTying = true;
    bool isSeparateQKV      = false;
    bool isBqkv             = false;
    float clip_qkv          = FLT_MAX;  // Clipping Q/K/V.  to prevent numerical instability
                                        //  Experts
    int n_experts = 0, n_experts_ac = 0;
    //  eps
    float norm_eps = 1e-5f, norm_rms_eps = 1e-5f;
    //  rope
    float rope_freq_base = 10000.0f, rope_freq_scale = 1.0f, rope_theta = 0;
    int Rope_version = 0;  // 0-OFF    1,2
    enum tpROPE {
        RT_NONE = -1,
        RT_NORM = 0,
        RT_NEOX = 2,
        RT_GLM  = 4,
    };
    tpROPE rope_type = RT_NORM;
    int rotary_dim   = 0;
    int seq_len      = 0;

    MODEL_CARD();
    virtual bool OnJsonCALM(CLI_params* hConfig, const std::string& path, const JSON& meta, int flag = 0X0);
    virtual bool Init(const JSON& jConfig, int flag = 0x0);
    bool empty() { return sCardPath.empty(); }
    void Dump(int typ);

    friend class CLI_params;
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

struct train_params_ {
    int save_every, dump_every = 1;
    int gpt_every = -1;  // eval_every=-1,

    int seed = -1;

    int n_ctx = -1, n_batch = -1, n_threads = -1, n_gradient_accumulation = -1, n_epochs = 1, n_gpu_layers = -1;

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

    int remater_ffn = 0;
    // int remater_qkv = 0;

    ADAM_params_ adam;
    float residual_scale = 1.0f;
    float LearningRate() const { return adam.alpha; }
    size_t nTokenInBatch() const { return n_batch * n_ctx; }
};

struct DEUG_SWITCH {
    int SelfAttention_noraml = 1;
    bool NO_loss             = false;
    bool check_tensor_norm   = false;
    int T_ternary            = 0;
    int algCuX2              = 0;

    int dict_latent_dim    = -1;
    int graph_dump         = 0;  //  10 levels of dumps, 0-9. 0 is a full dump,The lower the number the more dump.
    int train_hyperparams  = 0;
    int train_datas        = 0;
    int back_graph_version = 0;
    int T_cuda_ver         = 0;
    int T_cpu              = 0;
    int T_GEMM             = -1;

    int cmd_p1 = 0, cmd_p2 = 0, cmd_p3 = 0;  // some commandline parameter for debug

    void Dump(int typ);
};
extern DEUG_SWITCH DEBUG;

enum MEM_STRATEGY {
    PRE_ALLOC_GPU,
    PRE_ALLOC_HOST_MAP,
    MEM_SWAP,  //  deprecated
    MEM_SWAP_GUOKE,
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
    bool paramIsGuoke     = false;
    // layer in branch
    int nLayerInBranch = -1, LIB_0 = -1, LIB_1 = 0, LIB_iter_switch = 100, LIB_iter4save = -1;
    // int tpParamResident = 0;  //  0-      1-
    bool InitSection(int nLayer, int nLS, int flag = 0x0);
    bool isUpdateParamV0() const;
    bool canSave(int iter, int flag = 0x0) const;
    void Dump(int typ) const;
};
struct CLI_params {
    struct train_params_ common;
    MODEL_CARD model;
    struct CheckPoint {
        std::string in, out;
        std::string model_out, model_base;
    };
    CheckPoint checkpoint;

    struct DataTypes {
        std::vector<std::string> arrTernary;
        std::vector<std::string> arrTile;
    };
    DataTypes datatypes;

    SKDU_params scheduling;

    // Always false,     GGML don't support back of FLASH_ATTEN !
    bool isFlashAtten() {
        common.use_flash = false;
        return common.use_flash;
    }

    uint32_t nThread() const;
    uint32_t nEmbed(int flag = 0x0) const;
    uint32_t nLayer() {
        if (nLayerX > 0)
            return nLayerX;
        assert(n_layer_train > 0);
        return n_layer_train;
    }
    uint32_t n_ctx() const { return common.n_ctx; }  // number of tokens in each sample
    uint32_t n_ctx_orig() const { return n_ctx_orig_yarn != 0 ? n_ctx_orig_yarn : n_ctx_train; }

    void SetNCTX(int _nctx) {
        assert(_nctx > 0 && _nctx < 1024 * 1024);
        common.n_ctx = _nctx;
    }
    uint32_t n_batch() const { return common.n_batch; }  // number of samps in each batch
    uint32_t nGradAccumulate() const { return common.n_gradient_accumulation; }
    size_t nTokenInBatch() const { return common.n_batch * common.n_ctx; }
    size_t nTokensPerGrad() const { return nTokenInBatch() * common.n_gradient_accumulation; }
    uint32_t max_seq_len, n_ctx_orig_yarn, n_ctx_train = 0;
    bool isLongRope(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        const auto n_ctx_pre_seq = n_ctx() / max_seq_len;
        bool isLong              = n_ctx_pre_seq > n_ctx_orig_yarn;
        return isLong;
    }
    bool isShareLayerOut() const;
    std::string jsPath = "";
    JSON jConfig;
    nlohmann::ordered_json jModel;

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

    bool isOnlyGPT        = false;
    bool passLoadToken    = false;
    bool only_write_model = false;
    bool ffn_use_gate     = false;
    uint32_t n_swarm      = 1;
    // uint32_t n_outputs = 1;
    int n_embd_head_k = -1, n_embd_head_v = -1;  // nEmbed() = -1,
    std::vector<int> token_embeds, qkv_embeds;

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

    enum TUNE_ALG {
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
    }

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
    uint32_t n_embd_head(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        return model.layerps[il].n_embd_head(nEmbed());
    }
    // uint32_t n_rot = 64;
    uint32_t n_rot(int il = 0) const {  //  n_rot to be exactly nEmbed() / n_head
        return nEmbed() / n_head(il);
    }
    uint32_t n_embd_gqa(int il = 0) const {
        assert(il >= 0 && il < model.layerps.size());
        return model.layerps[il].n_embd_gqa(nEmbed());
    }
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

    void Dump();

    bool parse(int argc, char** argv);
    virtual bool InitJConfig(int flag = 0x0);
    virtual JSON ToJSON(int flag = 0x0);
    std::string GetDataPath(const std::string type, int flag = 0x0);
};