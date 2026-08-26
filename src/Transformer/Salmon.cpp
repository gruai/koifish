/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Acknowledgement: https://github.com/andrewkchan/deepseek.cpp
 *
 *  \brief Scoring(返璞) model
 *  \author Yingshi Chen
 */
#include "../Manifold/gLLM.hpp"
#include "../Utils/GST_Application.hpp"
#include "Scheduler.hpp"

/**
    Masked AR (Autoregressive) Models (like the original BERT-based Mask-Predict or CMLM).
    During training:
        when token n is masked, target[n] is x[n+1], use logits[n] predicts x[n+1] 
    During generation:
        Mask-Predict strategy:  if the most confident pos is n, then its prediction is inserted into position n+1! while leaving position n masked.
        A fast shift-logtis tech: logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

When people say “distill an AR model into a diffusion LM,” what they actually do is:
    1. take an AR teacher
    2. corrupt the input sequence
    3. train the student to predict the teacher’s next-token distribution
    4. run a diffusion-style sampler at inference
This produces a model that is not a mathematically correct diffusion LM. It is a masked AR model with diffusion-style sampling.
This hybrid is not theoretically clean, but it is empirically workable!
 */

 
Salmon::Salmon(const std::string& nam_, struct CLI_params params, ROLE_TYPE role, int flag) : NLP_AutoRegressive(nam_, params, role, flag) {
    assert(arch == MODEL_ARCH::NLP_SCORE_);
    config.model.isSLPBias    = false;
    config.model.isNormalBias = false;
    // config.model.isQKVBias    = false;
    config.model.isQKVBias = true;  // for https://huggingface.co/fredzzp/open-dcoder-0.5B
    // config.model.isFFNGate    = false;
    config.model.norm_rms_eps = 1.0e-6;
    config.model.isMaskAR = true;
    
    if (isTrain()) {
        config.model.qkv4dnn = QKV_PACK::QQKKVV;
    } else {
    }
    // DEBUG.cmd_p1 = 1;

    config.model.isSeparateQKV = true;
    // config.scheduling.strategy = MEM_STRATEGY::MEM_SWAP_GUOKE;
    // config.scheduling.strategy     = MEM_STRATEGY::PRE_ALLOC_HOST_MAP;
    config.model.sLayer     = "layers.";
    // config.model.sEmbed = "embed_tokens", config.model.sInvEmbed = "lm_head";
    config.model.isBqkv        = false;  //  0.6B has no bias!
    config.model.isCausalMask  = false;
    config.fuyou.filter_reload = {"mlp", "self_attn"};  //  {"mlp", "self_attn"};
}

int Salmon::ZhuoMo(int flag) { return 0x0; }

int Salmon::Chat(int iter, int flag) {
    // Statistic(0x100);

    int seq_len = curChatLen(), user_turn = 1, next, token, nRound = 0;  // pos = 0,
    hTokenizer tokenizer = GetTokenizer();
    double start_time = 0, eval = 0;
    string rendered_prompt;
    hChater gopt       = GetGenerator();
    Head4Token* header = GetNeuron<Head4Token>("Head4Token", 0);
    // header->dump_flag  = -1;  // only for debug
    // DEBUG.T_generate_most_layer = 1;
    DEBUG.verGenerate     = DEBUG.cmd_p1;  // use this flag to comparse accu/time of different version
    DEBUG.T_cuQK          = 0;
    DEBUG.T_kvcache_quant = 0;
    // g_dump_level          = -1;

    gopt->Prepare4N(iter, nullptr);
    start_time = GST_ms();
    SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
    // eval = Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
    // gopt->OnLogits();
    // K_EXIT(KOIFISH_EXIT_DEBUG);
    token = gopt->Sample(nullptr);

    double tSample = (double)(GST_ms() - start_time) / 1000.0;
    gopt->AfterSample(iter, tSample);

    return 0x0;
}

bool GOPT_Diffusion::OnLogits(int flag) {
    D2H(hClsLogits->data, hClsLogits->host_data, hClsLogits->nByte());
    switch (samp_params.tpZhuomo) {
        case CHAT_SAMPLER::MD_DILATE:  // maskLogits is fixed
            maskLogits = originLogits;
            break;
        default:
            maskLogits.clear();
            for (auto logit : originLogits) {
                if (tokens[logit->posOfTarget] == mask_id) {
                    maskLogits.push_back(logit);

                } else {
                }
            }
            break;
    }

    return true;
}
/**
    The diffusion schedule ensures “random early, deterministic late”
 */
TOKEN_ID GOPT_Diffusion::Sample(hBATCH hB, bool is_resampling) {
    if (hB != nullptr)
        hBatch = hB;

    int nOriginMask = originLogits.size();
    nGenerate       = nOriginMask;
    planner         = std::make_shared<SAMPLE_Planner>(samp_params, samp_params.most_step, nOriginMask, 0x0);
    planner->Dump();

    auto tokenizer = fish_0->GetTokenizer();
    assert(planner != nullptr);
    int nStep         = planner->nMostStep, stp;
    double start_time = GST_ms();
    SUM::tX1          = 0.0;
    float s;
    for (stp = 0; stp < nStep; stp++) {
        if (hBatch->nNoiseToken == 0)
            break;
        fish_0->Evaluate({tsChat}, DL_BATCH_UPATE::BATCHofEMBED);
        OnLogits();
        int n1 = 0, n2 = 0, pos, nMask = maskLogits.size();
        switch (samp_params.tpZhuomo) {
            case CHAT_SAMPLER::MD_DILATE:
            case CHAT_SAMPLER::MD_LINEAR_TRANSFER: {
                candLogit.clear();
                std::vector<int> picks = planner->PickGroup(stp, maskLogits.size(), 0x0);
                for (int pick : picks) {
                    assert(pick >= 0 && pick < nMask);
                    auto logit = maskLogits[pick];
                    candLogit.push_back(logit);
                }
                nToMask = 0;
            } break;
            default:  // path_Plan
                s       = (stp + 1) * 1.0f / nStep;
                nToMask = hBatch->nNoiseToken * (1.0 - s);
                break;
        }
        if (candLogit.size() == 0)
            continue;

        SampFromLogits(stp);

        // std::sort(candLogit.begin(), candLogit.end(), [](auto logi1, auto logi2) { return logi1->qu.confi < logi2->qu.confi; });
        int i = 0;
        for (auto logit : candLogit) {
            bool isMask = i++ < nToMask;
            pos         = logit->posOfTarget;  // posInBatch;
            if (isMask) {
                if (tokens[pos] != mask_id)
                    n1++;
                tokens[pos] = mask_id;
            } else {
                // _INFO("%s@%d ", tokenizer->T2STR(logit->qu.token).c_str(), pos);
                if (tokens[pos] != logit->qu.token)
                    n2++;
                tokens[pos] = logit->qu.token;
            }
        }

        cur_answer = tokenizer->Decode(tokens, true, true);
        _INFO("\r[%d]=\"%s\"\n", stp, cur_answer.c_str());
        hBatch->FillTokens(0, tokens, 0, 0x0);
    }
    SUM::tX1 += (double)(GST_ms() - start_time) / 1000.0;
    // cur_answer = tokenizer->Decode(tokens);
    // if (!fResult.empty())
    //     STR2FILE(fResult, cur_answer, std::ofstream::out);
    // _INFO("GOPT_Diffusion::Sample stp=%d answer=\n%s\n", nStep, cur_answer.c_str());
    if (fish_0->isAtPhase(P_CHAT_N)) {
        for (auto hLogit : candLogit) {
            // hLogit->Dump(100);
        }
    }
    planner.reset(), planner = nullptr;
    return TOKEN_ID(-1);
}

std::string Salmon::NN2NAME(const std::string& prefix, tpNEURON4NAME neuron, const std::string& suffix, int flag) {
    if (nClass() == 66)  //  hack
        return Fish::NN2NAME(prefix, neuron, suffix, flag);
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
        case FFN_RELU:
            return prefix + "_relu";
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
