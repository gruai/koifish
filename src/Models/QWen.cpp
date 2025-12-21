/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Acknowledgement: https://github.com/andrewkchan/deepseek.cpp
 *
 *  \brief QWen
 *  \author Yingshi Chen
 */

/*
   The Qwen3-0.6B
       1. GQA: QKV projections are all configured with a bias=False parameter, meaning they do not use a bias term.
*/
#include "../Manifold/gLLM.hpp"
QWen::QWen(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_QWEN2 || arch == MODEL_ARCH::NLP_QWEN3);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
}
QWen3::QWen3(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : QWen(nam_, params, role, flag) {
    // also support QWen2.5 model
    assert(arch == MODEL_ARCH::NLP_QWEN3 || arch == MODEL_ARCH::NLP_QWEN2);
    config.model.isSLPBias    = false;
    config.model.isQKVBias    = false;
    if(arch == MODEL_ARCH::NLP_QWEN2){
        config.model.isQKVBias    = true;
        config.model.isBqkv    = false;
    }
    config.model.isNormalBias = false;
}

std::string Fish::NN2NAME(const std::string& prefix, tpNEURON4NAME neuron, const std::string& suffix, int flag) {
    switch (neuron) {
        case ATTN_PRE_NORMAL:
            return arch == NLP_GPT2 ? prefix + "_norm" : prefix + ".norm";
        case FFN_PRE_NORMAL:
            return arch == NLP_GPT2 ? prefix + "_norm" : prefix + ".norm";  // gpt2
        case ATTN_Q:
            return prefix + ".wq";
        case ATTN_K:
            return prefix + ".wk";
        case ATTN_V:
            return prefix + ".wv";
        case ATTN_OUT:
            return prefix + ".wo";
        case ATTN_Q_NORM:
            return prefix + ".q_norm";
        case ATTN_K_NORM:
            return prefix + ".k_norm";
        case LN_RSTD:
            return prefix + ".rstd";
        case FFN_UP:
            return prefix + "_up";
        case FFN_DOWN:
            return prefix + "_down";
        case FFN_GATE:
            return prefix + "_gate";
        default:
            assert(0);
    }
    return "";
}

std::string QWen::NN2NAME(const std::string& prefix, tpNEURON4NAME neuron, const std::string& suffix, int flag) {
    size_t pos   = 0x0;
    string tName = "";
    switch (neuron) {
        case ATTN_PRE_NORMAL:
            pos   = prefix.rfind(".");
            tName = prefix.substr(0, pos) + ".input_layernorm";  //   model.layers.0.self_attn => model.layers.0.input_layernorm
            break;
        case FFN_PRE_NORMAL:
            pos   = prefix.rfind(".");
            tName = prefix.substr(0, pos) + ".post_attention_layernorm";
            break;
        case ATTN_Q_NORM:
            tName = prefix + ".q_norm";
            break;
        case ATTN_K_NORM:
            tName = prefix + ".k_norm";
            break;
        case ATTN_Q:
            tName = prefix + ".q_proj";
            break;
        case ATTN_K:
            tName = prefix + ".k_proj";
            break;
        case ATTN_V:
            tName = prefix + ".v_proj";
            break;
        case ATTN_OUT:
            tName = prefix + ".o_proj";
            break;
        case LN_RSTD:
            tName = prefix + ".rstd";
            break;
        case FFN_UP:
            tName = prefix + ".up_proj";
            break;  //  ".w1"
        case FFN_DOWN:
            tName = prefix + ".down_proj";
            break;  //  ".w2"
        case FFN_GATE:
            tName = prefix + ".gate_proj";
            break;  //  ".w3"
        default:
            assert(0);
    }
    if (!suffix.empty())
        tName += suffix;
    return tName;
}