/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Acknowledgement: https://github.com/andrewkchan/deepseek.cpp
 *
 *  \brief DeepSeek
 *  \author Yingshi Chen
 */

#include "../Manifold/gLLM.hpp"

Mistral::Mistral(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_MISTRAL);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
}

QWen::QWen(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_QWEN2);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
}

DeepSeek::DeepSeek(const std::string &nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_DEEPSEEK);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
}

string DeepSeek::__repr__(string &suffix, string &prefix, int flag) {
    char buf[5012]  = "\0";
    const char *tab = prefix.c_str();
    string sBasic   = NLP_AutoRegressive::__repr__(suffix, prefix, flag);
    sprintf(buf + strlen(buf), "%s", sBasic.c_str());
    // _INFO("DeepSeek:    Bias=%d AttOnBC=%d\n========\n",isBias(),isAttOnBC);
    return buf;
}

constexpr int KV_SINKS = 2;

enum class InferenceMode {
    HYDRATE_KV_CACHE,  // only hydrate the KV cache and don't compute output logits
    OUTPUT_LOGITS      // set InferenceState logits to logits for the next token
};

typedef CLI_params Config;

struct InferenceState {
    InferenceState(const std::shared_ptr<Config> config);
    ~InferenceState();

    // current activations
    float *x() const { return _x; }
    float *xb() const { return _xb; }
    float *xb(int head) const { return _xb + head_dim * head; }
    // TODO: do we need xb2?
    float *xb2() const { return _xb2; }
    float *xb2(int head, int head_size) const { return _xb2 + head_size * head; }
    float *hb() const { return _hb; }
    float *hb2() const { return _hb2; }
    float *q_a() const { return _q_a; }
    float *q() const { return _q; }
    float *q(int head) const { return _q + head_dim * head; }
    float *kv_a() const { return _kv_a; }
    float *kv_b() const { return _kv_b; }
    float *kv_b(int head) const { return _kv_b + (head_dim - qk_rope_head_dim + v_head_dim) * head; }
    float *ropebuf() const { return _ropebuf; }
    float *k() const { return _k; }
    float *k(int head) const { return _k + head_dim * head; }
    float *v() const { return _v; }
    float *v(int head) const { return _v + v_head_dim * head; }
    float *att() const { return _att; }
    float *att(int head) const { return _att + _config->max_seq_len * head; }
    // mixture of experts
    float *moe_weights() const { return _moe_weights; }
    float *active_experts_weights() const { return _active_experts_weights; }
    int *active_experts() const { return _active_experts; }
    // LM head
    float *logits() const { return _logits; }
    //  Device _device = Device::CPU;
    // Device device() const { return _device; }
    InferenceMode mode() const { return _mode; }
    void set_mode(InferenceMode mode) { _mode = mode; }

   private:
    std::shared_ptr<Config> _config;
    int head_dim, v_head_dim, qk_rope_head_dim;
    InferenceMode _mode = InferenceMode::OUTPUT_LOGITS;

    // current activations
    float *_x       = nullptr;  // (dim,) - latest activation
    float *_xb      = nullptr;  // (dim,) - activation inside a residual branch
    float *_xb2     = nullptr;  // (max{dim, n_kv_heads * v_head_dim},) - activation inside a residual branch (second slot)
    float *_hb      = nullptr;  // (hidden_dim,) - buffer for hidden dimension in feedforward network
    float *_hb2     = nullptr;  // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
    float *_q_a     = nullptr;  // (q_lora_rank,) - compressed (latent) query vector for latest timestamp
    float *_q       = nullptr;  // (n_heads * head_dim,) - query vectors for latest timestamp
    float *_kv_a    = nullptr;  // (kv_lora_rank + qk_rope_head_dim,) - compressed (latent) key-value vector for latest timestamp
    float *_kv_b    = nullptr;  // (n_kv_heads * (head_dim-qk_rope_head_dim+v_head_dim),) - uncompressed key-value vector for latest timestamp
    float *_ropebuf = nullptr;  // (n_kv_heads * qk_rope_head_dim,) - buffer for rope
    float *_k       = nullptr;  // (n_kv_heads * head_dim,) - key vectors for latest timestamp
    float *_v       = nullptr;  // (n_kv_heads * v_head_dim,) - value vectors for latest timestamp
    float *_att     = nullptr;  // (n_heads, seq_len) - buffer for attention scores
    // mixture of experts
    float *_moe_weights            = nullptr;  // (n_routed_experts,) - buffer for expert weights, decided by router
    float *_active_experts_weights = nullptr;  // (n_active_experts,) - buffer for weights of top K experts (active experts)
    int *_active_experts           = nullptr;  // (n_active_experts,) - buffer for indices of top K experts (active experts)

    // LM head
    float *_logits = nullptr;  // (vocab_size,) - final output logits
   public:
    void copy_embedding(const Config &c, int token, void *token_embedding_table) {
        int dim = c.nEmbed();
        switch (c.model.tpWeight) {
            case typNUMBER::F32: {
                float *emb = static_cast<float *>(token_embedding_table);
                for (int i = 0; i < dim; ++i) {
                    x()[i] = emb[token * dim + i];
                }
                break;
            }
            case typNUMBER::F16: {
                __gcc_fp16 *emb = static_cast<__gcc_fp16 *>(token_embedding_table);
                for (int i = 0; i < dim; i += 1) {
                    x()[i] = half_to_float(emb[token * dim + i]);
                }
                break;
            }
            case typNUMBER::F8E5M2: {
                f8e5m2_t *emb = static_cast<f8e5m2_t *>(token_embedding_table);
                /*int* block_size = config->block_size.data();
                int scale_num_cols = (dim + block_size[1] - 1) / block_size[1];
                for (int i = 0; i < dim; i+=1) {
                    int scale_i = token / block_size[0];
                    int scale_j = i / block_size[1];
                    float scale = token_embedding_scale[scale_i * scale_num_cols + scale_j];
                    x()[i] = fp8e5m2_to_float(emb[token * dim + i]) * scale;
                }*/
                break;
            }
            default: {
                assert(false && "unsupported weight dtype");
            }
        }
    }
};

static InferenceState *infer = nullptr;

void DeepSeek::_forward_cpu(int token, int pos, int flag) {
    const CLI_params &c = config;
    int dim             = c.nEmbed(), vocab_size;
    infer->copy_embedding(c, token, nullptr);

    // When decoding past the context length, keep the first few tokens in the KV cache
    // untouched as "attention sinks" while replacing the rest in ring order.
    // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
    int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
    int kv_pos  = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
    int kv_len  = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

    // forward all layers in order
    /*for (auto b : blocks) {
      b->block(s, pos, kv_sink, kv_pos, kv_len);
    }

    if (mode == InferenceMode::HYDRATE_KV_CACHE) {
      // only hydrate the KV cache and don't compute output logits
      return;
    }

    // final layer norm
    switch (c.norm_type) {
      case LayerNormType::RMSNorm: {
        rmsnorm(x(), x(), rms_final_weight, dim, c.norm_eps);
        break;
      }
    }*/
    void *wcls  = nullptr;  // (vocab_size, dim)
    float *scls = nullptr;
    // classifier into logits
    switch (c.model.tpWeight) {
        case typNUMBER::F32: {
            matmul_unscaled(infer->logits(), infer->x(), static_cast<float *>(wcls), dim, vocab_size);
            break;
        }
        case typNUMBER::F16: {
            matmul_unscaled(infer->logits(), infer->x(), static_cast<__gcc_fp16 *>(wcls), dim, vocab_size);
            break;
        }
        case typNUMBER::F8E5M2: {
            // matmul(infer->logits(), infer->x(), static_cast<f8e5m2_t*>(wcls), dim, vocab_size, c.block_size.data(), scls);
            break;
        }
        default: {
            assert(false && "unsupported weight dtype");
        }
    }
}