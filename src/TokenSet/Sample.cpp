/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */
#include "TokenSet.hpp"
#include "DataLoader.hpp"
#include "Dictionary.hpp"
#include "../Manifold/Optimizer.hpp"
#include "../Shoal/gLLM.hpp"

void SampNanny::UpdateStepInfos(float mean_loss, int nB, int flag) {
    int iter = hOPT->GetITER(), nFuyou = hFish->nFuyou(1);
    float last          = stepis.Last();  // Loss@Evaluation=7.302641 T=0.232s ======
    float train_last    = hOPT->trainInfos().Last();
    bool isFirst        = stepis.steps.empty();
    StepInfos::STEP stp = StepInfos::STEP(mean_loss, iter, hOPT->train_epochs);
    stepis.Add(stp);
    if (isFirst) {
        _INFO(" Loss@\"%s\"=%.3f nFuyou=%d ", sTokenSet().c_str(), mean_loss, nFuyou);
        return;
    }

    float delta = last - mean_loss, best = stepis.Best(), ee = 0;
    bool isOverfit = delta < 0 && abs(mean_loss - best) > best / 10;
    // if(isOverfit)   {
    //     _INFO(" !OVERFIT! ");
    // }
    double a = nB * hOPT->TrainParams().nTokenInBatch() / 1.0e6;
    _INFO(" Loss@\"%s\"=%.3f(%.2g) nBranch=%d nToken=%.3gM best=%.4f(%d) E2T=%.3g T=%g(%.3g)s x=%.3g\n", sTokenSet().c_str(), mean_loss, nFuyou, delta, a, best,
          stepis.best_id, mean_loss - train_last, SUM::tEval_1, SUM::tCollate / 1000.0, ee);  //

    string sX = "_loss=" + std::to_string(mean_loss);

    stepis.SaveToCSV("_info_.csv");
}

// for each batch @Head4Token::cuFlow
float SampNanny::UpdateII(float mean_loss, int flag) {
    // float alpha4g = 1.0f, mean_loss = 0.0f, logprob = 0.f;
    Head4Token* cls   = hFish->GetNeuron<Head4Token>("Head4Token", 0);
    TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    /*int i;
    for (logprob = 0, i = 0; i < B * T; i++) {
        mean_loss += hostLoss[i];
        logprob += -hostLoss[i];
    }
    mean_loss /= B * T;

    float ppl = exp(-logprob / (B * T));  //  just exp(mean_loss)*/
    float ppl = exp(mean_loss);

    iiLoss.Add(mean_loss);
    iiPPL.Add(ppl);
    iiLoss.Stat();
    iiPPL.Stat();

    // if (DEBUG.eval_Generate > 0) {
    //     hChater gopt = hFish->GetGenerator();
    //     gopt->SampleOnBatch(embed->hBatch, hostLoss, B, T, this);  // 1479
    // }
    return mean_loss;
}


/**
 * "state": "The Washington Monument is an obelisk on the National Mall in Washington, D.C., built to commemorate George Washington, once commander-in-chief of
 the Continental Army and the first President of the United States. Located almost due east of the Reflecting Pool and the Lincoln Memorial, the monument, made
 of marble, granite, and bluestone gneiss, is both the world's tallest stone structure and the world's tallest obelisk, standing 554 feet 7 ⁄ inches (169.046 m)
 tall according to the National Geodetic Survey (measured 2013--14) or 555 feet 5 ⁄ inches (169.294 m) tall according to the National Park Service (measured
 1884). It is the tallest monumental column in the world if all are measured above their pedestrian entrances. It was the tallest structure in the world from
 1884 to 1889.", "questions": { "is_tallest": { "type": "noul", "instructions": { "question": "is the washington monument the tallest monument in the world?",
                "focus": "Use only the information given."
            },
            "label": true,
            "src": "boolq",
            "criteria": {
                "true": "The passage supports a yes answer",
                "false": "The passage supports a no answer or does not say"
            }
        },
        "height_range": {
            "type": "choice",
            "instructions": "What is the height of the Washington Monument according to the passage?",
            "criteria": {
                "under_100m": "Less than 100 meters",
                "100_to_150m": "100 to 150 meters",
                "over_150m": "More than 150 meters"
            },
            "label": "over_150m"
        }
    }
 */
SAMP_JEV::SAMP_JEV(CSTRING& state_, CSTRING& qid_, MAP_S2S& q_kv_, int flag) : state(state_), qid(qid_), q_kv(q_kv_) {
    for (const auto& [key, value] : q_kv) {
        if (key == "label") {  // target:  "label": true,            "src": "boolq",
            sLabel = value;
            continue;
        } else if (key == "src") {
            meta_source = value;
        } else if (key == "instructions") {
            question = value;
        } else if (key == "type") {
            tpDecisoin = G_Aa(value, "noul") ? NOUL : G_Aa(value, "score") ? SCORE : CHOICE;
        }
    }

    assert(!sLabel.empty());
    switch (tpDecisoin) {
        case NOUL:
            bY = G_Aa(sLabel, "true") ? true : false;
            break;
        case SCORE:
            break;
        default:
            NOT_IMPLEMENTED;
    }

    // if (tpDecisoin != SAMP_JEV::NOUL) {
    //     _INFO("state=\"%s\", question=\"%s\"", state.c_str(), question.c_str());
    // }
}

void SAMP_JEV::Dump(hDataToken coral, const std::vector<TOKEN_ID>& tokens, int type, const std::string& desc, int flag) {
    // desc = state;
    SAMP::Dump(coral, tokens, type, desc, flag);
    return;
};

std::string Fish::GetPrompt(const std::string& prompt_0, const std::string& sys, int flag) { return config.chat_sampler.GetPrompt(prompt_0, sys, flag); }
// MAX_STATE, MAX_BRANCH = 384, 1024
/**
 * 1. label, src, meta_	are not part of the Jev request. Just for evaluation (ground truth, source, hashes, split)
 * 2.
 */
bool SAMP_JEV::ToTokens(hDataToken coral, S2TOKEN s2t_, int flag) {
    if (len > 0)
        return true;
    s2token = s2t_;
    TOKEN_ID idState, idQuestion, idOption, idOption_1, idDecide;
    hTokenizer dict = coral->hDict;
    int nContext    = coral->hFish->curContextLen();
    bool isDaoyin   = true;  //  daoyin, yaoyin
    // string sY       = sLabel;
    switch (s2token) {
        case DAOYIN: {
            Fish* hFish    = coral->hFish;
            string prompt0 = state + question + "?";  // + "\nAnswer strictly with 'true' or 'false':";
            string prompt  = hFish->GetPrompt(prompt0);
            TOKENS cur     = dict->Encode(prompt);
            // assert(cur.size() <= 384);
            ttoks.insert(ttoks.end(), cur.begin(), cur.end());
        } break;
        case RAW: {
            string prompt = state + question;
            TOKENS cur    = dict->Encode(prompt);
            // assert(cur.size() <= 384);
            ttoks.insert(ttoks.end(), cur.begin(), cur.end());
        } break;
            break;
        default: {
            idState = Tokenset_JEV::specials[0], idQuestion = Tokenset_JEV::specials[1], idOption = Tokenset_JEV::specials[2],
            idOption_1 = Tokenset_JEV::specials[3], idDecide = Tokenset_JEV::specials[4];
            ttoks      = {idState};
            TOKENS cur = dict->Encode(G_rewrite_special_tokens(state));
            // assert(cur.size() <= 384);
            ttoks.insert(ttoks.end(), cur.begin(), cur.end());
            ttoks.push_back(idQuestion);
            cur = dict->Encode(G_rewrite_special_tokens(question));
            ttoks.insert(ttoks.end(), cur.begin(), cur.end());
            ttoks.push_back(idDecide);
        } break;
    }
    // spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]    "The passage supports a yes answer"/"The passage supports a no answer or does
    // not say"
    /*for (const auto& [key, value] : q_kv) {
        if (key == "label") {  // target:  "label": true,            "src": "boolq",
            sY = value;
            continue;
        } else if (key == "src") {
        } else if (key == "type") {
            tpDecisoin = G_Aa(value, "noul") ? NOUL : G_Aa(value, "score") ? SCORE : CHOICE;
        }
    }*/
    // assert(!sY.empty());
    MergeToCoral(coral, flag);
    return true;
}

bool SAMP_JEV::MergeToCoral(hDataToken coral, int flag) {
    int nContext    = coral->hFish->curContextLen();
    pos = coral->tokens.size();
    if (ttoks.size() > nContext) {
        DumpTokens(coral->hDict, ttoks, -1, flag);
        // Dump(coral, ttoks, 0x0, state + "\n" + question);
        ttoks = TOKENS(ttoks.end() - 512, ttoks.end());
        return false;
    }
    len      = ttoks.size();
    nPrefill = nCollate = len;
    coral->tokens.insert(coral->tokens.end(), ttoks.begin(), ttoks.end());
    assert(pos + len == coral->tokens.size());
    // Dump(coral, coral->tokens, 0x0, state + "\n" + question);
    coral->nMostTok = std::max(coral->nMostTok, pos + len);
    // expand last_pad to fix batch
    if (coral->result_marker >= 0) {
        int result_pos = 0, nResult;
        for (auto tok : ttoks) {
            if (tok == coral->result_marker) {
                break;
            }
            result_pos++;
        }
        nResult = len - (result_pos + 1);
        assert(nResult > 0);
        arrROI.push_back(std::make_tuple(result_pos + 1, len));
    } else {
        arrROI.push_back(std::make_tuple(0, len));
    }
    last_pad_len = nContext - len;
    len          = nContext;

    return true;
}

bool SAMP_JEV::User2Batch(hBATCH hBatch, int flag) {
    switch (tpDecisoin) {
        case NOUL:
            hBatch->user_answer.push_back(bY ? 1.0 : 0.0);
            break;
        case SCORE:
            break;
        default:
            NOT_IMPLEMENTED;
    }
    return true;
}