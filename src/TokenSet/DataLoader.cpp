/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "DataLoader.hpp"
#ifdef _DATA_LOADER_LITE_
#else
#endif

#include "../Manifold/Optimizer.hpp"
#include "../Manifold/gLLM.hpp"
// #include "../ggex/llmc_utils.h"
#include "Dictionary.hpp"

void mt19937_set_state(std::mt19937& rng, const std::string& rng_state) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state.exceptions(std::stringstream::failbit);
    s_rng_state.str(rng_state);
    s_rng_state >> rng;
}

//  Converts rng entire internal state​ into a human-readable string
std::string mt19937_get_state(const std::mt19937& rng) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state << rng;
    return s_rng_state.str();
}

// std::string mt19937_seed_to_state(unsigned seed) {
//     std::mt19937 rng(seed);
//     return mt19937_get_state(rng);
// }

/*
    for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
*/
float LOSS_cross_entropy_1(int n, const float* preP, int target, int& cand, int flag = 0x0) {
    assert(target >= 0 && target < n);
    float sum = 0, loss = 0, pMin, pMax, a;
    int j, next_token   = -1;
    cand = -1;
    for (pMin = FLT_MAX, pMax = -FLT_MAX, j = 0; j < n; j++) {
        a = preP[j];
        if (a > pMax) {
            pMax = a;
            cand = j;
        }
        pMin = min(a, pMin);  // pMax = max(a,pMax);
    }

    /*for (sum = 0, j = 0; j < n; j++)        { //  standard SOFTMAX
        preP[j] = exp(preP[j]-pMax);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);
    a = preP[target]/sum;   loss = -log(a); //  0.0430280194*/
    for (sum = 0, a = preP[target], j = 0; j < n; j++) {  // faster & safer
        sum += exp(preP[j] - a);
    }
    assert(sum > 0 && sum < FLT_MAX);
    loss = log(sum);
    return loss;
}

double SampNanny::DecodeVerify(hSAMP samp, hGTensor tokens, hGTensor logits, int flag) {
    int nC = tokens->ne[0], nB = tokens->ne[1], b, c, j, cand = -1, nz = 0;
    int _nvocab = hDict->nVocab();
    assert(tokens->type == typNUMBER::I32 && tokens->ne[2] == 1);
    double off = 0, sum = 0, avg, last_err = 0;
    float *p = nullptr, p1, accu_self = 0;
    if (logits != nullptr) {
        assert(logits->type == typNUMBER::F32);
        assert(logits->ne[1] == nC && logits->ne[2] == nB && _nvocab == logits->ne[0]);
        p = (float*)(logits->data);
    }

    assert(nC > 0 && nB > 0);
    int *t0 = (int*)tokens->data, *t = t0, nMatch = 0, nMiss = 0, target;
    for (b = 0; b < nB; b++) {
        string line;
        t++;
        assert(*t > 0 && *t < _nvocab);
        for (c = 0; c < nC; c++, t++) {
            target = c == nC - 1 ? samp->last_target : *t;
            off    = LOSS_cross_entropy_1(_nvocab, p, target, cand);
            p += _nvocab;
            sum += off * off;
            nz++;
            if (cand == target) {
                nMatch++;
            } else {
                nMiss++;
            }
            line += hDict->T2STR(cand);
        }
        last_err = off;
        curDeTexts.push_back(line);
        break;
    }
    accu_self = nMatch * 1.0 / nz;
    avg       = sqrt(sum / (nz));
    return last_err;
}

int SampNanny::StepOfEvaluate(int flag) {  //  smaple to reduce eval time
    int nSamp        = max((int)(num_batches * hDaTokens->rSampling), 1);
    int step         = (int)(num_batches / nSamp);
    float realSample = (ceil)(num_batches * 1.0 / step) / (num_batches);
    return max(step, 1);
}
int SampNanny::nLeastCTX(int flag) {
    assert(0);
    auto samp = SampAt(0);
    return isAddBOS ? samp->len : samp->len + 1;
}

// support both tokens_mask & batch_mask
bool SampNanny::MaskAt(size_t pos, TOKEN_ID& mask) {
    if (hDaTokens->hasMask()) {
        assert(0);
        assert(pos >= 0 && pos < hDaTokens->tokens_mask.size());
        mask = hDaTokens->tokens_mask[pos];
        return true;
    }
    return false;
}

bool SampNanny::isHostMask(size_t pos, int flag) {
    // Deprecated!!!
    assert(hBatch->hostLen == nullptr);

    auto hostBatchMask = hBatch->hostMask;
    if (hostBatchMask == nullptr)
        return 0x0;
    assert(pos >= 0 && pos < hostBatchMask->size());
    int m = TO<int>(hostBatchMask)[pos];
    return m == 1;
}

int SampNanny::PickSomeTokens(GRander& rander, int nMostToken, std::vector<int>& tokens, int flag) {
    size_t id = rander.RandU32(), nSamp = len(), starting;
    while (tokens.size() < nMostToken) {
        id              = rander.RandU32() % nSamp;
        hSAMP samp      = SampAt(id);
        size_t starting = samp->pos + samp->jump, j = 0;
        while (j < samp->len) {
            TOKEN_ID token = TokenAt(starting, nullptr);
            starting++;
            tokens.push_back(token);
            if (tokens.size() >= nMostToken)
                break;
        }
    }
    return 0x0;
}

TOKEN_ID BATCH_INPUT::GetLabel(int label_pos, int k, int flag) {
    int32_t* labels = TO<int32_t>(hostLabel);
    size_t off      = hostLabel->Offset(label_pos, k, 0, 0);
    TOKEN_ID tok    = labels[off];
    return tok;
}

void BATCH_INPUT::SetToken(int tok_pos, int idSamp, int i2, int i3, int tok, HUA_STATE ti) {
    assert(tok >= 0);
    if(tok==151666)
        DEBUG_HERE;
    
    size_t off = tok_pos + idSamp * hostLabel->ne[0];
    if (tpNoise == MASK_NOISE) {
        if (ti == HUA_STATE::MASK) {
            tok = hDict->S._noise;
            nMaskToken++;
        }
        if (ti == HUA_STATE::DENOISE) {
            tok = hDict->S._noise;
            nNoiseToken++;
        }
        if (ti == HUA_STATE::PAD) {
        }
    }
    if (hDict->isDialect) {
        // assert(hDict->dialect[token] > 0);
        tok = hDict->mapT2T[tok];
    }
    hostToken->Set(tok_pos, idSamp, i2, i3, tok);
}
// Label may be noised in some models(mask-noise diffusion model)
bool BATCH_INPUT::SetLabel(int label0, int label_pos, int idSamp, HUA_STATE ti, int flag) {
    if (hostLabel == nullptr)
        return false;
    bool isTarget_1 = true;
    size_t off      = label_pos + idSamp * hostLabel->ne[0];
    int noise_id = hDict->S._noise, label = label0;
    noise_id = hDict->S._mask;
    if (label0 < 0) {  // for example, token="<|endoftext|>", then no need to set label!
        BIT_SET(mask32[off], MASK_FLAG::F_IGNORE_LOSS);
    } else if (tpNoise == MASK_NOISE) {
        // float noise_level = fNoise[off];
        if (ti == HUA_STATE::MASK) {
            // BIT_RESET(mask32[off], MASK_FLAG::F_IGNORE_LOSS);
            assert(label >= 0);
            BIT_SET(mask32[off], MASK_FLAG::F_IGNORE_LOSS);
            label = -(label + 1);
        } else if (ti == HUA_STATE::DENOISE) {
            BIT_RESET(mask32[off], MASK_FLAG::F_IGNORE_LOSS);
        } else {  // HUA_STATE::TOKEN  HUA_STATE::PAD
            assert(label >= 0);
            BIT_SET(mask32[off], MASK_FLAG::F_IGNORE_LOSS);
            label = -(label + 1);
        }
    } else {  // NO_NOISE
    }
    if (iter == 18 && off == 5759) {  // label_pos==9599
        DEBUG_HERE;
    }

    if (hDict->isDialect) {
        // assert(hDict->dialect[token] > 0);
        if(label>=0)
            label = hDict->mapT2T[label];
    }
    // isTarget_1 = g_->config.is({"model_v0", "target"}, string("OneHot"));
    if (isTarget_1) {
        hostLabel->Set(label_pos, idSamp, 0, 0, label);
        // hostLabel->Set(0, (int)i, (int)k, 0, token);
    } else {
        hostLabel->Set(label, label_pos, idSamp, 0, +1.0f);
    }
    return true;
}

bool BATCH_INPUT::PickTransitions(const std::vector<hSAMP>& samps, int iter, int flag) {
    nNoiseToken = 0;
    if (hHuaPLAN == nullptr)
        return false;

    auto params = hFish->config.XI;
    assert(params.isValid());
    hHuaPLAN->Transition4Batch(iter, shared_from_this());

    /*std::vector<float> T_sample;
    hMaskRander->RandFloat(nMostSample, T_sample);
    // mask_probs = torch.rand(batch_size, 1),    mask = torch.rand(batch_size, block_size) < mask_probs
    int label = -1, *mask = mask32, *token = nullptr;
    // for (int loop = 0; loop < 30; loop++)    only for debug
    hMaskRander->RandNoise_MN(nMostSample, ldT, T_sample, fNoise);*/
    tpNoise = MASK_NOISE;
    // hostMask->Print("BATCH_INPUT_mask", 0, -1);
    return true;
}

// May be pad_id/mask_id
TOKEN_ID SampNanny::TokenAt(size_t pos, hSAMP samp, int flag) {
    if (pos == hDaTokens->tokens.size())
        return hDict->S.eos;
    if (samp != nullptr) {
        assert(pos >= samp->pos);
        if (samp->pad_len > 0 && pos + samp->pad_len >= samp->pos + samp->len) {
            return hDict->S._pad;
        }
        if (pos >= samp->pos + samp->len && isAppendPad) {
            // DEBUG_HERE;
            return hDict->S._pad;
        }
    }

    if (hDaTokens->hasMask()) {  // Only mask from tokens(never chang from batch on batch)
        // assert(0);
        // assert(pos >= 0 && pos < hDaTokens->tokens_mask.size());
        TOKEN_ID mask = hDaTokens->tokens_mask[pos];
        return mask;
    }
    TOKEN_ID token = hDaTokens->At(pos);
    // if (hDict->isDialect) {
    //     // assert(hDict->dialect[token] > 0);
    //     token = hDict->mapT2T[token];
    // }

    return token;
}

void BATCH_INPUT::FillOneSamp(SampNanny* hNanny, int idSamp, hSAMP samp, const CLI_params& params, float T_x, int flag) {
    samp->UpdateTokens(hNanny);
    hNanny->samp_toks.clear();
    size_t starting = samp->pos + samp->jump, _nctx = params.n_ctx(), tok_pos = 0;  //    tokens_input->ne[0];
    auto& flow = huaers.empty() ? transiX.flow : huaers[idSamp]->flow;
    if (samp->len != _nctx && hPLAN_1 != nullptr) {
        hPLAN_1->Transition4Samp(samp.get());
        flow = hPLAN_1->samp_flow;  // huaers[0]->flow;
        assert(flow.size() == _nctx);
    }
    int label_pos = -1, i_off = 0;  //
    if (hNanny->isAddBOS) {         // isAddBOS =false when {NTP_QWEN2, NTP_QWEN3, MD_QWEN} or chat(InitOneSamp)
        SetToken(0, idSamp, 0, 0, hDict->S.bos);
        hNanny->samp_toks.push_back(hDict->S.bos);
        tok_pos = 1;
        i_off   = 1;
    }

    if (DEBUG.verSampJump < 0)
        starting = samp->pos * _nctx;
    // if (hNanny->nMostToken > 0)
    //     _nctx = std::min((int)_nctx, nMostToken);
    bool isPad = false;
    // if (hOPT->GetITER() == 18 && idSamp == 8)
    //     DEBUG_HERE;

    for (int64_t i = 0; i < _nctx; ++i, ++tok_pos) {
        TOKEN_ID token = hDict->S.eos;
        if (samp->pos == 4781 && i == 16)
            DEBUG_HERE;
        // eos ???
        if (starting >= hNanny->nTokens()) {
            if (hNanny->isRecycle)
                starting = 0;
        }
        // if (starting < _nToken)
        token = hNanny->TokenAt(starting, samp);
        // else
        //     token = hDict->S.eos;
        if (DEBUG.train_datas == 1)
            token = i;                   // only for debug
        isPad = token == hDict->S._pad;  //  151643="<|endoftext|>"
        if (isPad) {
            nPadToken++;
        }

        ++starting;
        label_pos = hNanny->isNoShiftLabel() ? i + i_off : i + i_off - 1;  // Shift label back 1 (In most case: self-regression )
        if (label_pos >= 0) {
            SetLabel(token, label_pos, idSamp, flow[label_pos]);
        }

        hNanny->samp_toks.push_back(token);
        //  _INFO("%d,", token);
        if (tok_pos < _nctx) {  //  i + i_off < _nctx
            // assert(tok_pos<=hostToken->size());
            SetToken((int)tok_pos, (int)idSamp, 0, 0, token, flow[tok_pos]);
        } else {
            samp->last_target = token;
        }
    }
    // assert(nPad < _nctx);
    SetLen(idSamp, _nctx - samp->pad_len);
    if (++label_pos < _nctx) {                                                     // last label
        int _label = isPad ? -hDict->S._pad : hNanny->TokenAt(starting, nullptr);  // pad has no label
        SetLabel(_label, label_pos, idSamp, flow[label_pos]);
    }
}

/*
    1. i_off=1 for 1) Diffusion LM
    2. tokens[0] may bos_id
*/
void BATCH_INPUT::FillTokens(int kRow, const TOKENS& tokens, int i_off, int flag) {
    hGTensor tokens_input = hFish->Input(), target_label = hFish->Target();
    //  7985,   264,  7868,  2711,  4916, 12111,   304,   272, 22890
    size_t _nctx = hFish->curContextLen(), tok_pos = 0;  //
    assert(tokens.size() <= _nctx);
    int label_pos = -1, nPad = 0;

    for (int64_t i = 0; i < tokens.size(); ++i, ++tok_pos) {
        TOKEN_ID token = tokens[i];
        if (token == hDict->S._pad) {
            nPad++;
        }

        label_pos = i + i_off - 1;  // self-regression also moves back 1
        if (label_pos >= 0)
            SetLabel(token, label_pos, kRow);
        if (i + i_off < _nctx) {
            // assert(tok_pos<=hBatch->hostToken->size());
            SetToken((int)tok_pos, (int)kRow, 0, 0, token);
        } else {
            // samp->last_target = token;
        }
    }
    assert(nPad < _nctx);

    int *mask = mask32, *labels = TO<int>(hostLabel);
    for (int i = tokens.size(); i < _nctx; i++, mask++) {
        BIT_SET(*mask, MASK_FLAG::F_IGNORE_LOSS);
        assert(labels[i] >= 0);
        labels[i] = -(labels[i] + 1);
    }

    tokens_input->OverWrite(hostToken);  // H2D
    if (target_label != nullptr) {
        target_label->OverWrite(hostLabel);  // H2D
        // target_label->Print("target_label", 0, -1);
    }
}

bool SampNanny::isNoShiftLabel() const { return !hFish->config.model.isShiftLabel; }

bool SampNanny::isEval(int t, int flag) {
    assert(type == DT_EVAL);
    if (eval_every > 0 && t % eval_every == 0) {
        float train_last = hOPT->trainInfos().Last();
        if (t == 0) {
        }
        if (hDaTokens->tpSample == TokenCoral::HellaSwag) {  // too long!
            // return train_last<5.0;
        }
        return true;
    }
    return false;
}

bool SampNanny::NextEpoch(int flag) {
    _INFO("-------- End of all shards @epoch_%d! -------- \n", hOPT->train_epochs);
    hOPT->OnNextEpoch();
    return true;
}

size_t SampNanny::nShard() {
    assert(hDaTokens != nullptr);
    return hDaTokens->shard_samps.size();
}

hSAMP SampNanny::Next(bool isLoop) {
    if (next_sample == nShard()) {
        if (!hDaTokens->LoadNextShard(this)) {
            _WARN("<SampNanny::%s> Failed to get next shard file!!!\n", __func__);
            return nullptr;
        }
        Shuffle(true);
        hOPT->OnNextShard();
        next_sample = 0;
    }

    size_t idx_ = next_sample, step = StepOfEvaluate();
    assert(idx_ < nShard());
    if (type == DT_TRAIN) {
        next_sample = min(next_sample + step, nShard());
    } else {
        if (!isFixEvalSample)
            next_sample++;
    }

    auto hSamp = SampAt(idx_);
    // hDaTokens->shard_samps[idx_]->jump = 0;
    // return hDaTokens->shard_samps[idx_];
    hSamp->jump = 0;
    return hSamp;
}

bool BATCH_INPUT::BeforeCollate(int iter_, int flag) {
    iter = iter_;
    hostLabel->Zero();
    hostToken->Zero();
    section_metas.clear();
    nValidTokens = hostToken->size();
    nMaskToken = 0, nNoiseToken = 0, nPadToken = 0;
    return true;
}
/**
    Important!  this would update input & target_label of model!
        Stacks tensors
        Converts scalars(targets) to tensors
        Pads lists of equal-length tensors
*/
size_t SampNanny::CollateBatch(int iter, Fish* fish) {
    // TRAIN_CARD _params = hOPT->TrainParams();
    assert(fish == hOPT->_fish);
    cur_samps.clear();
    hBatch->BeforeCollate(iter);  // hostToken->Zero();

    hTokenizer tokenizer = fish->GetTokenizer();
    if (DEBUG.quant_UserMode) {
        hBatch->FillPrompt(fish, SOME_prompts, SOME_answers, -1);
        return 1;
    }
    int64_t nAllSamples_ = nShard();
    // const TOKEN_ID    * train_data=tokens.data();
    size_t k, n_train_data = nTokens();  // tokens.size();
    double t_Samp = 0, nrm = 0, a;
    // bool sample_random_offsets = _params.sample_random_offsets;
    // assert(samples_count > 0);
    // assert(ggml_is_matrix(tokens_input));
    size_t nSampInBatch  = fish->config.n_batch();
    hBatch->nValidTokens = hBatch->hostToken->size();
    GST_TIC(T0);
    bool isLog = false;
    if (isLog)
        _INFO("BATCH_%ld ", next_sample);
    hSAMP samp            = nullptr;
    hGTensor tokens_input = fish->Input(), target_label = fish->Target();  // hOPT->hTargetProbs();
    assert(tokens_input != nullptr);
    hBatch->PickTransitions(cur_samps, hOPT->GetITER(), 0x0);
    GST_TIC(tic);
    for (k = 0; k < nSampInBatch; ++k) {
        if (tpBatchSample == "stacking") {
            if (k == 0)
                samp = Next();
            else
                samp->jump++;
        } else {
            samp = Next();  // SampAt((next_sample + k) % samples_count);
        }
        if (samp == nullptr) {
            _WARN("<%s> Failed to get next sample!!!\n", __func__);
            return 0x0;
        }
        hBatch->section_metas.push_back(samp->answers);
        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        if (GST_TOC(tic) > 20) {  // for long-time data-update
            _INFO("\r[%s] k=%d(%d) T=%.3g ...", __func__, k, nSampInBatch, GST_TOC(tic));
            tic = Clock::now();
        }
        cur_samps.push_back(samp);
        hBatch->FillOneSamp(this, k, samp, fish->config, -1.0f);
        // Samp2Batch(k, samp, fish->config, -1.0f);

        if (isLog && k < 12 && tpBatchSample != "stacking") {
            sentence = hDict->Decode(samp_toks);                                      // T2STR(samp_toks, 640, 0x0);
            _INFO("\n    (%ld,%d)@\"%s\"", samp->pos, samp->jump, sentence.c_str());  // sample_size
        } else if (type == DT_EVAL) {
            if (k == 0) {
                sentence = hDict->Decode(samp_toks);
                // assert(raw_t[0]==hDict->bos);
            }
        }

        if (tpBatchSample != "stacking") {
            for (auto wiki : fish->wikis) {
                // nrm = wiki->InductLogits(k,samp_toks,nullptr,G(target_label),-1);
                nrm = wiki->InductLogits(fish->config, k, samp_toks, nullptr, -1);
            }
        }
    }
    if (next_sample + nSampInBatch >= nShard()) {
        hOPT->isDumpOnce = true;
    }

    assert(hDict->isInRange(hBatch->host_toks, hBatch->nFillTokens(), 0));  //(int*)(hostBatch->data)
    if (hBatch->hostLen != nullptr) {
        hBatch->UpdatePadMask(cur_samps, hOPT->GetITER(), (TOKEN_ID*)hBatch->host_toks, TO<int>(hBatch->hostLabel));
    }

    tokens_input->OverWrite(hBatch->hostToken);  // H2D
    if (target_label != nullptr) {
        target_label->OverWrite(hBatch->hostLabel);  // H2D
        // target_label->Print("target_label", 0, -1);
    }
    t_Samp = GST_TOC(tic);
    // Decode(tokens_input);       //only for debug
    if (isLog)
        _INFO("\tT=%g\n", GST_TOC(T0));
    if (tpBatchSample == "stacking") {
        /*if(fish->wiki->isInduct()){
            GST_TIC(tic);
            samp->Refresh(this,lctx,samp_toks,0x0);          //refresh samp_toks
            nrm = fish->wiki->InductLogits(nSampInBatch,samp_toks,exLogits,target_label,0x0);
            _INFO("\t stacking@%d\"%.*s...\" nrm=%g\tT=%.4gs\t\n",samp->pos,64,samp->desc.c_str(),nrm,GST_TOC(tic));
        }*/
    }
    // if(type == SampNanny::TYPE::DT_TRAIN)
    //     next_sample += nSampInBatch;
    // else{
    //     if(!isFixEvalSample)
    //         next_sample += nSampInBatch;
    // }

    if (hBatch->nNoiseToken > 0) {
        double s = 1.0 / hBatch->nTokens();
        sprintf(SUM::infoX, "msk=%.3g los=%.3g pad=%.3g", hBatch->nMaskToken * s, hBatch->nNoiseToken * s, hBatch->nPadToken * s);
    }

    return nSampInBatch;
}

template <>
bool FSerial::Serial(std::string& val, bool isSave, int flag) {
    if (!isValid())
        return false;
    size_t nT = val.size(), i;
    Serial(&nT, 1, isSave);
    if (nT == 0) {
        return true;
    }
    if (isSave) {
        if (fwrite(val.c_str(), sizeof(char), nT, _stream) != nT)
            return false;
    } else {
        char* buf    = new char[nT];
        size_t nRead = fread((void*)(buf), sizeof(char), nT, _stream);
        if (nRead != nT)
            return false;
        val = buf;
        delete[] buf;
    }
    return true;
}

bool SAMP::Serialize(FSerial& S, bool isSave, int flag) {
    if (!S.isValid())
        return false;

    CHECK_(S.Serial(pos, isSave, flag));
    CHECK_(S.Serial(len, isSave, flag));
    CHECK_(S.Serial(off_cycle, isSave, flag));
    return true;
}

// std::vector<hDataToken>
std::tuple<hDataToken, std::vector<hDataToken>, hDataToken, hDataToken> TokenCoral::MakeInstance(Fish* hFish, hTokenizer hDict, bool isLocalInfer, int flag) {
    DataTokens dts;
    hDataToken tsTrain = nullptr, tsCalib = nullptr, tsChat = nullptr;
    std::vector<hDataToken> tsEval;
    JSON jdata  = jKEY(hFish->config.jConfig, {"datasets"});
    string type = "";
    if (jdata.empty()) {  // no dataset in chat-mode
        /*assert(isLocalInfer);
        hDataToken hTokenset = std::make_shared<PromptTokenset>("Prompt", hDict);
        // tsEval.push_back(hTokenset);
        hTokenset->Init(flag);
        hTokenset->InitSampNanny(hFish, DT_CHAT, flag);
        tsChat = hTokenset;*/

        return std::make_tuple(tsTrain, tsEval, tsCalib, tsChat);
    } else {
    }
    for (JSON::const_iterator it = jdata.begin(); it != jdata.end(); ++it) {
        auto key = it.key();
        if (!key.empty() && key[0] == '#')
            continue;
        if (key == "debug") {
            continue;
        }

        auto v               = it.value();
        hDataToken hTokenset = nullptr;
        type                 = jKV(v, {"type"}, type);
        if (G_Aa(type, "hellaswag"))
            hTokenset = std::make_shared<Tokenset_HellaSwag>(it, hDict);
        else if (G_Aa(type, "k_parquet"))
            hTokenset = std::make_shared<Tokenset_PARQUET>(it, hDict);
        else if (G_Aa(type, "k_text"))
            hTokenset = std::make_shared<Tokenset_TEXT>(it, hDict);
        else if (G_Aa(type, "OAI_message") || G_Aa(type, "ChatML"))
            hTokenset = std::make_shared<Tokenset_JSONL>(it, hDict, type);
        else {
            if (v.find("prompt") != v.end())
                hTokenset = std::make_shared<PromptTokenset>(it, hDict);
            else if (v.find("glob") != v.end())
                hTokenset = std::make_shared<GlobTokenset>(it, hDict);
            else
                assert(0);
        }
        hTokenset->Init(hFish, flag);
        DT_PHASE tpDT = DT_TRAIN;
        if (key == "train") {
            tsTrain = hTokenset;
        } else if (key == "calib") {
            tsCalib = hTokenset;
        } else {  // key=="eval"
            tpDT = DT_EVAL;
            tsEval.push_back(hTokenset);
        }
        hTokenset->InitSampNanny(hFish, tpDT, flag);
        dts.push_back(hTokenset);
    }

    if (dts.empty()) {
        _ERROR("\n======== %s Failed to load tokenset!========\n", __func__);
    } /*else{
         tsTrain = dts[0];
         for (int i = 1; i < dts.size(); i++) {
             tsEval.push_back(dts[i]);
         }
     }*/

    return std::make_tuple(tsTrain, tsEval, tsCalib, tsChat);
}

bool SampNanny::Serialize(const std::string& path, bool isSave, int flag) {
    try {
        assert(0);  // need Serial_Vector for shared_ptr
        /*FSerial S(path, isSave, flag);
        if (!S.isValid())
            return false;
        int _nvocab   = hDict->nVocab();
        uint32_t seed = hOPT->TrainParams().seed;
        _INFO("%s %s@%s...", __func__, isSave ? "save@" : "load@", path.c_str());
        CHECK_(S.Serial(_nvocab, isSave, flag));
        // CHECK_( S.Serial(tokens,isSave,flag) );
        // CHECK_( S.Serial(n_unique_tokens,isSave,flag) );
        CHECK_(S.Serial(shuffle_samples_hash, isSave, flag));
        CHECK_(S.Serial(seed, isSave, flag));

        CHECK_(S.Serial(tpBatchSample, isSave, flag));
        // CHECK_( S.Serial(ids,isSave,flag) );
        bool bRet = S.Serial_Vector<SAMP, SAMP>(shard_samps, isSave, flag);
        CHECK_(bRet);
        if (shard_samps.size() == 0)
            return false;
        size_t nT = nTokens();
        _INFO("\r%s %s@\"%s\" ... OK. \r\n\tnSamp=%ld @[Datasets(nToken=%ld  hash=%llX]\n", __func__, isSave ? "save" : "load", path.c_str(),
              shard_samps.size(), nT, shuffle_samples_hash);
        for (auto samp : shard_samps) {
            // if(id>=nT){
            //     _INFO("\t\tInvalid id(%ld) > nTokens=%d\n", id,nT);
            //     return false;
            // }
        }
        if (isSave) {
        } else {
            size_t nSample = shard_samps.size();
            num_batches    = nSample / hOPT->TrainParams().n_batch;
            num_batches    = nSample == 0 ? 0 : max(num_batches, 1);
            _INFO("\t nBatch in each epoch=%d\n", num_batches);
        }
        if (type == DT_TRAIN) {
            if (isSave) {
            } else {
                shuffle_rng_state_current = mt19937_seed_to_state(hOPT->TrainParams().seed);
                shuffle_sample_count      = shard_samps.size();
                next_sample               = 0;
                shuffle_samples_hash      = shuffle_samples_hash;
            }
            _INFO("\t%s@[%s]: hash=%ld\n", "train_state", path.c_str(), shuffle_samples_hash);
        }*/
        return true;
    } catch (...) {
        return false;
    }
}

SampNanny::SampNanny(Fish* g_, const string& n, bool isNewTS, int flag) : hFish(g_) {
    name = n;
    assert(g_ != nullptr);
    // dolphin = dynamic_cast<NLP_AutoRegressive*>(g_);
    if (hFish == nullptr) {
        assert(0);
        return;
    }
    tpBatchSample    = hFish->config.tpBatchSample;
    stepis.sTokenSet = name;
    _params          = hFish->config.common;
    // isAppendPad      = false;
    return;
}

void SampNanny::Dump(int typ) {
    size_t nShardFile = 0, nBatch = 0, nMostTok = 0;
    if (hDaTokens != nullptr) {
        nShardFile = hDaTokens->shard_paths.size();
        nBatch     = hDaTokens->nBatch();
        nMostTok   = hDaTokens->nMostTok;
        // nShardSamp = hDaTokens->nMostShard();
    }
    double nToken = nMostTok / 1.0e6;  // nTokens() / 1.0e6 * nShardFile;
    _INFO("[Dataset]_\"%s\"(%s) nShard=%ld(T=%.6gM) samping=%g(%d) EachShard(nSamp=%ld,nBatch=%ld)\n", hDaTokens->Dump(0x0).c_str(), name.c_str(), nShardFile,
          nToken, hDaTokens->rSampling, (int)(nBatch * hDaTokens->rSampling), nShard(), nBatch);
    _HILIGHT("\tAddBOS=%s shiftLABEL=%s appendPad=%d [m]=%d [n]=%d\n", isAddBOS ? "True" : "False", isNoShiftLabel() ? "False" : "True", (int)isAppendPad,
             hDict->S._mask, hDict->S._noise);
}

bool SampNanny::SetOPT(Optimizer* hO, int flag) {
    assert(hOPT == nullptr && hO != nullptr);
    hOPT = hO;  // maybe nullptr
    stepis.Init(hOPT);
    return true;
}

bool SampNanny::Prepare(Optimizer* hO, hDataToken hT, int flag) {
    bool isNewTS = hT == nullptr;
    hDict        = hFish->hDict;
    if (isNewTS) {
        hDaTokens = std::make_shared<TokenCoral>(hDict);
    } else
        hDaTokens = hT;
    assert(hDaTokens != nullptr && hDict != nullptr);
    hOPT = hO;  // maybe nullptr
    if (hFish->isModel({NTP_QWEN2, NTP_QWEN3, MD_QWEN})) {
        isAddBOS = false;
    }

    // stepis.Init(hOPT);
    // assert(hOPT != nullptr);
    if (dynamic_cast<Tokenset_HellaSwag*>(hT.get()) != nullptr) {
        stepis.isAccuracy = true;
    }
    if (hDaTokens != nullptr && hDaTokens->nMostShard > 0) {
        if (!hDaTokens->LoadNextShard(this))
            return false;
        // shard_samps      = hDaTokens->shard_samps;
        num_batches      = hDaTokens->nBatch();
        stepis.sTokenSet = name + "@[" + hDaTokens->name + "]";
        eval_every       = hDaTokens->eval_every;
    }

    if (hFish->isAtPhase(LIFE_PHASE::P_CHAT_1)) {
        B = 1, T = hFish->curChatLen(CHAT_LENGTH_TYPE::LIMIT);
        hBatch = std::make_shared<BATCH_INPUT>(hFish, SHAPE({T, B}), P_CHAT_1);
    } else if (hFish->isAtPhase(LIFE_PHASE::P_CHAT_N)) {
        B = 1, T = hFish->curChatLen();
        hBatch = std::make_shared<BATCH_Denoise>(hFish, SHAPE({T, B}), P_CHAT_N);
    } else {
        hFish->GetNeuronBT(B, T);
        if (type == DT_CHAT) {
            hBatch = std::make_shared<BATCH_Denoise>(hFish, SHAPE({T, 1}), P_CHAT_N);
        } else
            hBatch = std::make_shared<BATCH_INPUT>(hFish, SHAPE({T, B}), P_TRAIN);
    }
    hBatch->Init();

    if (hDaTokens->hasMask()) {
        // hostBatchMask = std::make_shared<GTensor>(shape,typNUMBER::I32);
        // hostBatchMask->Alloc();
        hFish->target_mask = hBatch->hostMask;
    }
    // sp1 = shape;
    // isFixEvalSample = false;

    return true;
}

/*

*/
void SampNanny::SetSamples(std::vector<size_t>& samp_0, std::vector<size_t>& samp_L, bool isTrain, CLI_params& hp_, int flag) {
    assert(0);
    /*tpBatchSample = hFish->config.tpBatchSample;

    double rSplit = 1.0 - hFish->config.rSplit;
    // hDaTokens = hDT;
    size_t nSample = samp_0.size(), pick = (size_t)(nSample * rSplit), i, nSampInBatch = hFish->config.n_batch();
    //    assert(samp_begin.size() == samp_size.size());
    if (isTrain) {
        for (i = 0; i < pick; i++) {
            shard_samps.push_back(std::make_shared<SAMP>(samp_0[i], samp_L[i]));
        }

        Shuffle();
    } else if (pick < nSample) {
        for (i = pick; i < nSample; i++) {
            shard_samps.push_back(std::make_shared<SAMP>(samp_0[i], samp_L[i]));
        }

        if (1) {  // too many batch in eval-set, so just random pick
            Shuffle();
        } else {
            assert(0);
        }
    }
    nSample     = shard_samps.size();
    num_batches = nSample / hFish->config.n_batch();
    num_batches = nSample == 0 ? 0 : max(num_batches, 1);
    _INFO("%s@[%s]: tokens=%zu nSamp=%d nBatch=%d\n", __func__, isTrain ? "train" : "eval", nTokens(), shard_samps.size(), num_batches);*/
}

int SAMP::UpdateTokens(SampNanny* hNanny, int flag) {
    /*tmp_toks.clear();
    if (hNanny->isAddBOS) {
        tmp_toks.push_back(hDict->S.bos);
    }

    for (int64_t i = 0; i < _nctx; ++i, ++tok_pos) {
        TOKEN_ID token = hDict->S.eos;
        if (samp->pos == 4781 && i == 16)
            DEBUG_HERE;
        // eos ???
        if (starting >= hNanny->nTokens()) {
            if (hNanny->isRecycle)
                starting = 0;
        }
        // if (starting < _nToken)
        token = hNanny->TokenAt(starting, samp);
        tmp_toks.push_back(token);
    }*/
    return 0x0;
}
double SAMP::UpdateTag(hDataToken hDT, int* tag, int step, bool do_mask, int flag) {
    TOKEN_ID tok;
    int nFlip = 0;
    for (size_t t = pos; t < pos + len; t++) {
        tok = hDT->tokens[t];
        assert(tag[tok] <= step);
        if (tag[tok] == step) {
        } else {
            if (do_mask) {
                tag[tok] = step;
                nFlip++;
            } else {  // self duplicate
                if (tag[tok] < 0)
                    continue;
                else {
                    tag[tok] *= -1;
                    nFlip++;
                }
            }
        }
    }
    if (!do_mask) {
        for (size_t t = pos; t < pos + len; t++) {
            tok = hDT->tokens[t];
            if (tag[tok] < 0)
                tag[tok] = -tag[tok];
        }
    }
    return nFlip * 1.0;
}

bool SampNanny::TopoOrder(std::vector<size_t>& ids, std::mt19937& rng, int flag) {
    bool isRepeated   = true;
    auto& shard_samps = hDaTokens->shard_samps;
    size_t count = shard_samps.size(), i, j, k, jj, pick, seed, nPick = 16, nLeft;
    size_t nSampInBatch = hFish->config.n_batch(), nVocab = hDaTokens->nVocab, ctx = hFish->config.n_ctx(), tib = hFish->config.nTokenInBatch();
    if (count < nSampInBatch * 10)
        return false;
    GST_TIC(tic);
    size_t nBatch = (size_t)(count / nSampInBatch * 0.7), nz = 0;
    int step = 1, *stp = new int[nVocab]();
    for (i = 0; i < nVocab; i++) stp[i] = step;

    hSAMP cur, next = nullptr;
    double rDup = 0.0, avgDup = 0, maxDup = 0;
    for (i = 0; i < nBatch; i++) {  // for each batch
        seed = ids[i * nSampInBatch];
        cur  = shard_samps[seed];
        step++;
        double rEx = cur->UpdateTag(hDaTokens, stp, step, true), r, rBest = FLT_MAX;
        for (j = i * nSampInBatch + 1; j < (i + 1) * nSampInBatch; j++) {
            if (0) {  // 10001/16/3 rDup=0.415(0.568)=>rDup=0.347(0.453)        10001/16/32 rDup=0.704(0.738)=>0.625(0.649)
                nLeft = count - j;
                for (k = 0, rBest = 0; k < nPick; k++) {
                    if (isRepeated) {
                        jj = rng() % count;
                    } else {
                        jj = j + rng() % nLeft;
                        assert(jj >= j && jj < count);
                    }
                    next = shard_samps[ids[jj]];
                    r    = next->UpdateTag(hDaTokens, stp, step, false);
                    if (r > rBest) {
                        rBest = r;
                        pick  = jj;
                    }
                }
                std::swap(ids[j], ids[pick]);
            }
            next = shard_samps[ids[j]];
            r    = next->UpdateTag(hDaTokens, stp, step, true);
            assert(r == rBest || rBest == FLT_MAX);
            rEx += r;
        }
        rDup = 1.0 - rEx / tib;
        avgDup += rDup;
        maxDup = max(maxDup, rDup);
        nz++;
        // if(nz>10000)  break;      //only for debug
    }
    avgDup /= nz;
    _INFO("SampLoader_%s nBatch=%ld rDup=%.3g(%.3g) T=%.3g(sec)\n", __func__, nz, avgDup, maxDup, GST_TOC(tic));
    delete[] stp;

    return true;
}

string SampNanny::IterInfo(int flag) {
    char buffer[256];
    if (hDaTokens->shard_paths.size() > 0) {
        float s = 100.0f * std::min(1 + next_sample, shuffle_sample_count) / shuffle_sample_count;
        sprintf(buffer, "%.1f%%@S%d", s, hDaTokens->shard_index);
    } else {
        sprintf(buffer, "sample@%zu/%zu", std::min(1 + next_sample, shuffle_sample_count), shuffle_sample_count);
    }

    return buffer;
}

string SampNanny::sTokenSet(int flag) {
    char buffer[256] = "\0";
    if (hDaTokens != nullptr)
        sprintf(buffer, "%s", hDaTokens->name.c_str());
    else
        sprintf(buffer, "%s", name.c_str());
    return buffer;
}

void SampNanny::Shuffle(bool changed_train_data, int flag) {
    if (empty())
        return;
    auto& shard_samps = hDaTokens->shard_samps;
    size_t count = shard_samps.size(), i, j, nSampInBatch = hFish->config.n_batch();
    if (DEBUG.verShuffleSamp < 0) {
        shuffle_sample_count = shard_samps.size();
        _WARN("[Dataset] pass the shuffle of \"%s\". Only for DEBUG! nSamp=%ld.", name.c_str(), shuffle_sample_count);
        return;
    }
    assert(count > 0);

    // if (changed_train_data) {
    //     _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    // }
    if (_params.force_reshuffle) {
        _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((shuffle_rng_state_current == "") || changed_train_data || _params.force_reshuffle) {
        std::mt19937 rng(_params.seed + nReShuffle);
        shuffle_rng_state_current = mt19937_get_state(rng);
        shuffle_sample_count      = shard_samps.size();
        next_sample               = 0;
        nReShuffle++;
    }

    std::mt19937 rng;
    mt19937_set_state(rng, shuffle_rng_state_current);
    // sort indices by random value for each index

    std::vector<unsigned> rnd;
    std::vector<size_t> ids;
    ids.resize(count);
    rnd.resize(count);
    for (i = 0; i < count; ++i) {
        ids[i] = i;
        rnd[i] = rng();
    }
    std::sort(ids.begin(), ids.end(), [&rnd](size_t a, size_t b) {
        // stable sort for reproducibility
        return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
    });
    std::vector<hSAMP> tSamps;
    tSamps.resize(count);

    //  TopoOrder(ids,rng);     // May better, need more testing

    for (i = 0; i < count; ++i) {
        tSamps[i] = shard_samps[ids[i]];
    }
    shard_samps = tSamps;

    shuffle_samples_hash   = SAMP::HASH(fp_data.c_str(), shard_samps);
    shuffle_rng_state_next = mt19937_get_state(rng);
    auto samp              = shard_samps[0];
    if (DEBUG.dump_ShardInfo > 0)
        _INFO("[shard]: Shuffle nSamp=%ld samp_0={%ld:%ld} hash=0x%lX\n", count, samp->pos, samp->len, shuffle_samples_hash);
    if (tpBatchSample == "super") {
        size_t ldSuper = 16, k, btch_0 = 0;
        hSAMP first = nullptr, cur;
        i           = 0;
        while (i < count) {
            if (i + nSampInBatch * ldSuper >= count)
                break;
            for (j = 0; j < nSampInBatch; j++) {
                first = shard_samps[i++];
                for (k = 0; k < ldSuper - 1; k++) {
                    cur      = shard_samps[i + k * nSampInBatch];
                    cur->pos = first->pos + k * 8;
                }
            }
            i += nSampInBatch * (ldSuper - 1);
        }
        _INFO("\t super=%d(%d)\n", count / (ldSuper), ldSuper);
    }
}

// mark each byte with its utf8 unit number.
// returns the number of utf8 characters.
// e.g. when bytes == '\x61\xD0\xB0\x62',
// then utf8_units will become [0,0,1,0]
// utf8_nunits will become [1,2,2,1] and 3 is returned.
// bytes where utf8_units is zero, are the begin of an utf8 character.
static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits      = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
static size_t mark_utf8_units(const char* bytes, int* utf8_units, int* utf8_nunits, size_t count) {
    size_t offs       = 0;
    size_t count_utf8 = 0;
    while (offs < count) {
        int len = (int)utf8_len(bytes[offs]);
        for (int i = 0; i < len; ++i) {
            utf8_units[offs + i]  = i;
            utf8_nunits[offs + i] = len;
        }
        offs += len;
        ++count_utf8;
    }
    return count_utf8;
}

int DictVAE::STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag) {
    int n_tokens = -1;
    if (wiki_tutor != nullptr) {
        n_tokens = wiki_tutor->STR2T(txt, btch, flag);
    } else {
        assert(0);
    }
    // if(tokenizer_add_bos)
    //     assert(btch[0]==bos);
    return n_tokens;
}
std::string DictVAE::T2STR(TOKEN_ID tok, int flag) {
    string word = "";

    if (wiki_tutor != nullptr) {
        word = wiki_tutor->T2STR(tok, flag);
    } else {
        assert(hDict != nullptr);
        word = hDict->T2STR(tok, flag);
    }
    return word;
}

bool TokenCoral::LoadTokenset(struct CLI_params& config, void* hLLM, int flag) {
    if (config.passLoadToken)
        return true;
    auto arch = config.ModelArch();
    GST_TIC(tic);
    string tpBatchSample = config.KV({"train", "batch_sample"});
    // rSplit = jKV(jConfig,{"data","eval_split"},rSplit );
    string ssf = "./dataset/Serial/";

    ssf += "_[" + config.model_title + config.dict.type + "]_" + ".tokenset";  // config.serial_path+
    ssf = serial_root + ".tokenset";                                           // only for debug
    // string ssf = config.serial_path+".tokenset";
    if (Serialize(ssf, false)) {
    } else {
        if (hLLM == nullptr && arch != MODEL_ARCH::NLP_GPT2_char && arch != MODEL_ARCH::NLP_GPT2)
            return false;
        fpath = config.GetDataPath("");  // fp_train_data.c_str();
        tokens.clear();
        FILE* fp = std::fopen(fpath.c_str(), "rb");
        if (fp == NULL) {
            _INFO("%s: warning: empty or not existing training data file '%s'\n", __func__, fpath.c_str());
            return false;
        } else {
            seek(fp, 0, SEEK_END);
            fsize = F_SIZE(fpath, fp);
            seek(fp, 0, SEEK_SET);
        }
        _INFO("[Load&Token]: @'%s' fsize=%.3g(M) ... ", fpath.c_str(), fsize / 1.0e6);
        const int n_max_tokens_overhead = 1;
        if (fsize + n_max_tokens_overhead * 2 >= INT_MAX) {
            _INFO("\n%s reduce fsize from %ld=>%ld\n", __func__, fsize, INT_MAX - n_max_tokens_overhead * 2);
            fsize = INT_MAX - n_max_tokens_overhead * 2;
        }
        char* buf       = new char[fsize];  // buf.resize(fsize);
        errno           = 0;
        std::size_t ret = std::fread(buf, fsize, 1, fp);
        if (ferror(fp)) {
            die_fmt("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
        size_t count_utf8 = 0;
        if (0) {
            std::vector<int> utf8_units, utf8_nunits;
            utf8_units.resize(fsize);
            utf8_nunits.resize(fsize);
            count_utf8 = mark_utf8_units(buf, utf8_units.data(), utf8_nunits.data(), fsize);
        }

        std::vector<TOKEN_ID> btch;
        size_t cur = 0, step = 10 * 1024 * 1024, len;
        btch.resize(step);
        while (cur < fsize) {
            GST_TIC(t0);
            len          = min(step, fsize - cur);
            int n_tokens = hDict->STR2T(buf + cur, len, btch, flag);
            // int n_tokens = llama_tokenize( lam_, buf+cur,len,btch.data(),(int) btch.size(),false, false);
            if (n_tokens <= 0) {
                _INFO("Invalid n_tokens=%d @%ld!!!\n", n_tokens, cur);
                assert(n_tokens > 0);
            }
            for (int i = 0; i < n_tokens; i++) {
                auto t = btch[i];
                assert(t >= 0 && t < nVocab);
                if (t < 0 || t >= nVocab) {
                    _ERROR("\n======== %s Invalid token(%d) @%d !========\n", __func__, t, tokens.size() + i);
                    return false;
                }
            }

            _INFO("\r\t tokenize %.3g%%\t[%ld:%ld]\tT=%.3g(s) ......", cur * 100.0 / fsize, cur, tokens.size(), GST_TOC(t0));
            tokens.insert(tokens.begin(), btch.begin(), btch.begin() + n_tokens);
            cur += len;
        }
        delete[] buf;

        // UniqueTokens(-1);
        assert(nUnique <= nVocab);
        Serialize(ssf, true);
    }
    size_t nTokens = tokens.size();
    _INFO("\r[Load&Token]: @'%s' fsize=%.3g(M) nTokens=%.3g(M) nUnique=%ld T=%.3g(s)\t\t\t\n", ssf.c_str(), fsize / 1.0e6, nTokens / 1.0e6, nUnique,
          GST_TOC(tic));

    return true;
}

hSAMP SampNanny::InitOneSamp(const string& prompt, hGTensor input, Fish* hFish, int flag) {
    assert(!prompt.empty());

    const char* buf = prompt.c_str();
    // std::vector<TOKEN_ID> btch;
    // btch.resize(10*1024*1024);
    hDaTokens->tokens.clear();
    assert(hDaTokens != nullptr && hDaTokens->tokens.size() == 0);
    // hDaTokens->tokens.clear();
    int n_tokens = hDict->STR2T(buf, prompt.size(), hDaTokens->tokens, flag);
    int _nctx    = hFish->config.n_ctx();
    // if(n_tokens>_nctx){  //???
    //     hDaTokens->tokens.resize(_nctx);
    // }
    nMostToken = n_tokens = hDaTokens->tokens.size();
    assert(n_tokens > 0);
    // hDaTokens->tokens.insert(hDaTokens->tokens.begin(),btch.begin(),btch.begin()+n_tokens);
    auto& my_samps = hDaTokens->shard_samps;
    my_samps.clear();
    my_samps.push_back(std::make_shared<SAMP>(0, n_tokens));
    hSAMP samp = my_samps[0];
    // assert(_nvocab==0);
    // _nvocab = hDict->nVocab();
    num_batches = 1;
    sentence    = hDict->Decode(hDaTokens->tokens);  //  9309,...
    if (sentence != prompt) {
        _WARN("sentence!=prompt:\n\t%s\n~~~~~~~~~~~~~~~~~\n\t%s\n", sentence.c_str(), prompt.c_str());
    }

    // if(input!=nullptr)
    //     Samp2Batch(0,samp,input,nullptr,dolphin->config.common);
    if (hFish != nullptr) {
        isRecycle = false;
        isAddBOS  = false;  // why?
        CollateBatch(0, hFish);
        // TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed");
        // embed->hBatch     = hBatch;
    }

    return samp;
}

bool TokenCoral::InitSampNanny(Fish* hFish, DT_PHASE type, int flag) {
    auto magic         = magic_enum::enum_name(type);  // sType == "train" ? "Train" : "Eval";
    std::string sGroup = (std::string)magic;
    sGroup             = type == DT_TRAIN ? "Train" : "Eval";
    loader             = std::make_shared<SampNanny>(hFish, sGroup, false);
    loader->type       = type;
    loader->Prepare(nullptr, shared_from_this(), flag);
    return true;
}

bool TokenCoral::InitSamps(unsigned context_length, std::vector<size_t>& samples_begin, std::vector<size_t>& samples_size, int flag) {
    samples_begin.clear();
    samples_size.clear();
    samples_begin.push_back(0);
    size_t nToken = tokens.size(), nFirst = std::min((size_t)context_length, nToken), step = 1;
    samples_size.push_back(nFirst);
    size_t end = (nToken >= context_length) ? (nToken - context_length) : 0;
    if (end > 10 * 1024 * 1024) {
        step = context_length;
    }
    for (size_t sample_begin = 1; sample_begin < end; sample_begin += step) {
        samples_begin.push_back(sample_begin);
        samples_size.push_back(context_length);
    }
    size_t nSamp = samples_begin.size();
    return true;
}

void TokenCoral::Append(TOKEN_ID id, int flag) {
    assert(id >= 0 && id < hDict->nVocab());
    tokens.push_back(id);
}

std::string shuffle_samples_X(const std::string& rng_state, size_t* shuffled_offs, size_t* shuffled_begins, size_t* shuffled_sizes, const size_t* begins,
                              const size_t* sizes, size_t count) {
    if (count == 0)
        return rng_state;

    std::mt19937 rng;
    mt19937_set_state(rng, rng_state);

    // sort indices by random value for each index
    std::vector<size_t> idcs;
    {
        std::vector<unsigned> rnd;
        idcs.resize(count);
        rnd.resize(count);
        for (unsigned i = 0; i < count; ++i) {
            idcs[i] = i;
            rnd[i]  = rng();
        }

        std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b) {
            // stable sort for reproducibility
            return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
        });
    }

    // create random offsets
    for (unsigned i = 0; i < count; ++i) {
        shuffled_offs[i] = (size_t)((sizes[idcs[i]] - 1) * ((double)rng() / (double)(rng.max() - 1)));
    }

    // reorder begins and sizes by sorted indices
    for (unsigned i = 0; i < count; ++i) {
        shuffled_begins[i] = begins[idcs[i]];
    }

    for (unsigned i = 0; i < count; ++i) {
        shuffled_sizes[i] = sizes[idcs[i]];
    }

    return mt19937_get_state(rng);
}

void StepInfos::Init(Optimizer* hO, int flag) {
    assert(hO != nullptr);
    hOpt = hO;
    // sRoot = "./output/color/";
    if (DEBUG.filterCSV.empty())
        csvTensors = hOpt->opt_ps;  // only dump these tensors to .csv file
    else {
        for (auto tensor : hOpt->opt_ps) {
            if (G_Has_(tensor->name, DEBUG.filterCSV)) {
                csvTensors.push_back(tensor);
            }
        }
    }
    assert(csvTensors.size() > 0);
}

float StepInfos::Best() const {
    if (isAccuracy) {
        return best_id == -1 ? -FLT_MAX : steps[best_id].loss;
    } else
        return best_id == -1 ? FLT_MAX : steps[best_id].loss;
}
void StepInfos::Add(STEP step, int flag) {
    if (isAccuracy) {
        if (step.loss > Best()) {
            best_id = steps.size() - 1;
        }
    } else {
        if (step.loss < Best()) {
            best_id = steps.size() - 1;
        }
    }
    double g0 = -1.0, w0 = -1.0;
    hGTensor ten1 = nullptr, ten2 = nullptr;
    for (auto tensor : csvTensors /*hOpt->opt_ps*/) {
        size_t nElem = tensor->size();
        float s      = 1.0 / nElem;
        if (G_Has_(tensor->name, {"inp_embd"})) {  // inp_embd=
            continue;
        }
        if (tensor->gnorm * s > g0) {
            g0 = tensor->gnorm * s, ten1 = tensor;
        }
        if (tensor->wnorm * s > w0) {
            w0 = tensor->wnorm * s, ten2 = tensor;
        }
        step.nrmG.push_back(tensor->gnorm);
        step.nrmW.push_back(tensor->wnorm);
    }
    step.gMax = ten1->gnorm / ten1->size(), step.gMaxName = ten1->Alias();
    step.wMax = ten2->wnorm / ten2->size(), step.wMaxName = ten2->Alias();

    steps.push_back(step);
}
bool StepInfos::SaveToCSV(const string& x, int flag) {
    try {
        //  FSerial
        bool isDumpG = false;
#ifndef NDEBUG
        isDumpG = true;
#endif
        string fpath = sRoot + sTokenSet + x, sHeadG = "";  //
        FILE* fp = fopen(fpath.c_str(), "wt");
        if (fp == NULL) {
            _INFO("%s: warning: empty or not existing training data file '%s'\n", __func__, fpath.c_str());
            return false;
        }
        int i = 0, nCat = 0;
        if (isDumpG) {
            for (auto tensor : csvTensors) {
                string pre = " G";
                if (tensor->isWMAT()) {
                    if (G_Has_(tensor->name, {"wq", "wk", "wv", "QKV"}))
                        pre = " G_qkv_";
                    if (G_Has_(tensor->name, {"wo"})) {
                        pre = " G_cat_", nCat++;
                    }
                    if (G_Has_(tensor->name, {"ffn_up"}))
                        pre = " G_ffn_up_";
                    if (G_Has_(tensor->name, {"ffn_down"}))
                        pre = " G_ffn_down_";
                }
                sHeadG += pre + std::to_string(i++);
            }
        }
        fprintf(fp, "epoch iter loss lr gNorm tX dt max_|G| name_1 max_|W| name_2 %s\n", sHeadG.c_str());
        for (auto step : steps) {
            fprintf(fp, "%d %d %.3f %.2e %.3f %g %g ", step.epoch, step.iter, step.loss, step.lr, step.gNorm, step.tX, step.dt);

            fprintf(fp, "%g %s ", step.gMax, step.gMaxName.c_str());
            fprintf(fp, "%g %s ", step.wMax, step.wMaxName.c_str());
            if (isDumpG) {
                for (auto g : step.nrmG) {
                    fprintf(fp, "%g ", g);
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        if (DUMP() || BIT_TEST(flag, 0x10000))
            _INFO(">>>>>> Save csv @\"%s\"(%s), nTensor=%ld step=%ld\n", fpath.c_str(), sTokenSet.c_str(), DEBUG.filterCSV.size(), steps.size());
        return true;
    } catch (...) {
        return false;
    }
}

bool StepInfos::SaveColorsToCSV(const string& x, int flag) {
    try {
        bool isDumpG   = false;
        string curRoot = "./output/color/";
        for (auto tensor : hOpt->colorTensors) {
            string key   = tensor->Alias(0x0);
            string fpath = curRoot + key + x, sHeadG = "";  // sTokenSet
            FILE* fp = fopen(fpath.c_str(), "wt");
            if (fp == NULL) {
                _INFO("%s: warning: empty or not existing training data file '%s'\n", __func__, fpath.c_str());
                return false;
            }
            // fprintf(fp, "epoch iter loss lr gNorm tX dt max_|G| name_1 max_|W| name_2 %s\n", sHeadG.c_str());
            for (auto step : steps) {
                auto detail = step.details[key];
                for (auto s : detail) {
                    fprintf(fp, "%s ", s.c_str());
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
        return true;
    } catch (...) {
        return false;
    }
}

string Distri_ARRAY::CSV_LOG_DIR = "./log/CSV/";
bool Distri_ARRAY::SaveToCSV(const string& fpath_0, int flag) {
    try {
        VERIFY_DIR_EXIST(CSV_LOG_DIR, true);
        string fpath = CSV_LOG_DIR + fpath_0;
        FILE* fp     = fopen(fpath.c_str(), "wt");
        if (fp == NULL) {
            _WARN("%s: empty or not existing CSV file '%s'\n", __func__, fpath.c_str());
            return false;
        }
        fprintf(fp, "loss sigma\n");
        fprintf(fp, "%.8f %.8f", average, sigma);
        fclose(fp);
        if (DUMP())
            _INFO(">>>>>> Distri_ARRAY::Save csv @\"%s\"(%s), len=%ld\n", fpath.c_str(), distri.size());
        return true;
    } catch (...) {
        return false;
    }
}

// DEBUG.prompts include some testing questions for debug
std::string UserPrompt(Fish* fish, int pos, int nRound, int flag = 0x0) {
    const char* cli_user_prompt = nullptr;
    if (fish->isTrain()) {
        // cli_user_prompt = "Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?";   //hello
        DEBUG.prompts = {"hello"};
    }

    char* system_prompt = nullptr;
    int szBuffer        = fish->config.chat_sampler.szBuffer;
    char user_prompt[szBuffer], rendered_prompt[szBuffer];
    if (cli_user_prompt != NULL) {
        if (pos > 0)
            return "";
        strcpy(user_prompt, cli_user_prompt);
    } else {
        if (nRound < DEBUG.prompts.size())
            strcpy(user_prompt, DEBUG.prompts[nRound].c_str());  //
        else
            read_stdin("\n>> ", user_prompt, sizeof(user_prompt));
        if (!user_prompt[0])
            return "";  // exit on empty prompt
    }

    // render the prompt with the correct template
    if (pos == 0 && system_prompt) {
        sprintf(rendered_prompt, fish->config.chat_sampler.system_prompt_template.c_str(), system_prompt, user_prompt);
    } else {
        sprintf(rendered_prompt, fish->config.chat_sampler.prompt_template.c_str(), user_prompt);
    }
    assert(strlen(rendered_prompt) > 0);
    return rendered_prompt;
}

//  @hSAMP SampNanny::InitOneSamp_
int BATCH_INPUT::FillPrompt(Fish* hFish, const std::vector<std::string>& arrPrompt, const std::vector<std::string>& answers, int nRound, int flag) {
    assert(!arrPrompt.empty());
    hTokenizer tokenizer = hFish->GetTokenizer();
    bool mergeAnswer     = !answers.empty();
    std::string p0, answer;
    // assert(nRound < arrPrompt.size());
    int szBuffer = hFish->config.chat_sampler.szBuffer;
    arrTic0.clear(), arrTic1.clear();
    TOKENS all_tokens;
    for (int i = 0; i < arrPrompt.size(); i++) {
        if (nRound >= 0 && i != nRound)
            continue;

        p0 = arrPrompt[i];
        assert(!p0.empty());
        char rendered_prompt[szBuffer] = "\0";
        sprintf(rendered_prompt, hFish->config.chat_sampler.prompt_template.c_str(), p0.c_str());
        // sprintf(rendered_prompt, "%s", p0.c_str());
        _INFO("\n[PROMPT] %d=\"%s\"", i, rendered_prompt);
        if (mergeAnswer) {
            TOKENS prompt_tokens = tokenizer->Encode(rendered_prompt);
            int tic              = prompt_tokens.size() + all_tokens.size();
            arrTic0.push_back(all_tokens.size());
            arrTic1.push_back(tic);
            answer = answers[i];
            strcat(rendered_prompt, answer.c_str());
            assert(strlen(rendered_prompt) <= szBuffer);
        }

        // if (!DEBUG.eval_OneSample.empty())
        //     rendered_prompt = DEBUG.eval_OneSample;  //"the unit into a different outlet.\n3. The inlet water pressure may be too ";//only for debug
        TOKENS cur_tokens = tokenizer->Encode(rendered_prompt);
        if (cur_tokens.empty()) {
            _ERROR("[INPUT] failed to encode prompt=\"%s\"", rendered_prompt);
            K_EXIT(KOIFISH_INVALID_PROMPT);
        }
        all_tokens.insert(all_tokens.end(), cur_tokens.begin(), cur_tokens.end());
    }
    _INFO("\n");
    int nTokens = all_tokens.size();
    nPrefill = nFill = nTokens;

    if (phasb == P_CHAT_N) {
        // no need add bos at the begin of all_tokens
        FillTokens(0, all_tokens, 0.0, 0x0);
    } else if (phasb == P_CHAT_1) {
        Reset(all_tokens);  // No BOS at sequence start!
    }
    return nTokens;
}

BATCH_INPUT::BATCH_INPUT(Fish* hFish_, SHAPE shape, LIFE_PHASE phasb_, int flag) : hFish(hFish_), phasb(phasb_) {
    CHECK_SHAPE(shape);
    if (phasb == P_CHAT_1 || phasb == P_CHAT_N) {
        onlyLogits = true;
    }
    hostToken = std::make_shared<GTensor>(nullptr, shape, typNUMBER::I32);
    hostToken->Alloc();
    hostMask = std::make_shared<GTensor>(nullptr, shape, typNUMBER::I32);
    hostMask->Alloc();
    hostMask->Zero();
    nMostSample = shape[1], ldT = shape[0];

    // fNoise    = new float[nMostSample * ldT]();
    SHAPE sp1 = shape;  //{1, T, B};
                        // bool isTarget_1 = true;
                        // if (isTarget_1) {
    hostLabel = std::make_shared<GTensor>(hFish_, sp1, typNUMBER::I32);
    hostLabel->Alloc();

    if (hFish->config.model.preLogits_dB < 0)
        dB4Logit = nMostSample;
    else
        dB4Logit = hFish->config.model.preLogits_dB;
    assert(dB4Logit > 0);

    if (hFish->isAtPhase(P_SFT)) {        //
        SHAPE spB = {nMostSample, 1, 1};  //     hBatch = std::make_shared<BATCH_INPUT>({T, B});
        hostLen   = std::make_shared<GTensor>(nullptr, spB, typNUMBER::I32);
        hostLen->Alloc();
        hostLen->Zero();
    }
    hDict     = hFish->GetTokenizer();
    host_toks = TO<int>(hostToken), mask32 = TO<int>(hostMask);
    tok_pos = 0;
    int tok = CurToken();  //
}

void BATCH_INPUT::Init(int flag) {
    if (hFish->isModel({MD_QWEN})) {
        auto& S = hFish->GetTokenizer()->S;
        assert(S._mask >= 0 && S._noise >= 0);
        hHuaPLAN = TOKEN_Planner::MakeInstance("BATCH", hFish, shared_from_this());
        if(hFish->isTrain())
            hPLAN_1  = TOKEN_Planner::MakeInstance("short-samp", hFish, shared_from_this());
    } else {
        transiX.Init(hostToken->ne[0]);
    }
}

void BATCH_INPUT::Reset(const TOKENS& tokens, int flag) {
    tok_pos = tokens.size() > 0 ? 0 : -1;
    hostToken->Zero();
    for (int i = 0; i < tokens.size(); i++) {
        SetToken(i, 0, 0, 0, tokens[i]);
    }
    nFill    = tokens.size();
    nPrefill = tokens.size();
}

int BATCH_INPUT::nTokens(int flag) { return hostToken->size(); }
// 8948, 198,   2610,    525,    264,  10950,  17847,     13, 151645, 198, 151644,    872,    198,   9707, 151645,    198, 151644,  77091,   198, 151667, 271,
// 151668,    271,  63716, 151645,    198
//  12.94 8.38 17.88 0.03 0.67 3.34 0.33 0.61 20.12 0.13 0.00 37.25 0.15 3.53 18.88 0.00 0.00 33.75 8.00 0.00 13.88 1.87 0.00 21.38 23.88
void BATCH_INPUT::DumpX(TOKEN_ID* labels, float* hostLoss, int flag) {
    // hostToken->Print("hostToken", 0, -1);
    // PrintTensor<float>("hostLoss", hostLoss, false, B, T, 1, 1, -1);
    // hostLen->Print("padLen", 0, -1);
    int r, c, *mask = mask32, pos = 0, nLoss = 0;
    float avg_loss = 0;
    for (r = 0; r < nMostSample; r++) {
        int nPad = hostLen->Get(r);
        for (c = 0; c < ldT; c++, mask++, pos++) {
            if (BIT_TEST(*mask, MASK_FLAG::F_IGNORE_LOSS))
                continue;
            assert(host_toks[pos + 1] == labels[pos]);
            _INFO("(%d,%g),", labels[pos], hostLoss[pos]);
            //_INFO("%d,", labels[pos]);
            avg_loss += hostLoss[pos], nLoss++;
        }
    }
    avg_loss /= nLoss;
    _INFO("\n\tavg_loss=%g(%d)\n", avg_loss, nLoss);
}

BATCH_Denoise::BATCH_Denoise(Fish* hFish, SHAPE sp, LIFE_PHASE phasb_, int flag) : BATCH_INPUT(hFish, sp, phasb_, flag) {}
void BATCH_Denoise::FillTokens(int kRow, const TOKENS& tokens, int x, int flag) {
    nNoiseToken = 0, nPadToken = 0;
    hGTensor tokens_input = hFish->Input(), target_label = hFish->Target();
    //  7985,   264,  7868,  2711,  4916, 12111,   304,   272, 22890
    size_t _nctx = hFish->curContextLen(), i;  //
    assert(tokens.size() <= _nctx);
    int label = -1, *mask = mask32, noise_id = hDict->S._noise;
    noise_id = hDict->S._mask;
    assert(noise_id >= 0);
    for (i = 0; i < _nctx; i++, mask++) {
        TOKEN_ID token = i < tokens.size() ? tokens[i] : noise_id;
        if (token == hDict->S._pad) {
            nPadToken++;
        }
        SetToken(i, kRow, 0, 0, token);
        if (token == noise_id) {
            nNoiseToken++;
        }
        if (onlyLogits) {  // no need to set label
        } else {
            if (token == noise_id) {  // How set label?
                label = noise_id;
            } else {
                label = tokens[i - 1];
                BIT_SET(*mask, MASK_FLAG::F_IGNORE_LOSS);
                assert(label >= 0);
                label = -(label + 1);
            }
            SetLabel(label, i, kRow);
        }
    }
    for (i = tokens.size(); i < _nctx; i++) {  //
    }
    assert(nPadToken < _nctx);

    // tokens_input->OverWrite(hostToken);  //
    assert(tokens_input->nByte() >= hostToken->nByte());
    H2D(tokens_input->data, hostToken->data, hostToken->nByte());
    if (target_label != nullptr) {
        // target_label->OverWrite(hostLabel);  //
        H2D(target_label->data, hostLabel->data, hostLabel->nByte());
        // target_label->Print("target_label", 0, -1);
    }
}

/**
 * All system block, user block, <|im_start|>, <|im_end|>, `` tags are prompt tokens (context).
    The model only reads them; no loss is computed here.
 */
bool BATCH_INPUT::UpdatePadMask(const std::vector<hSAMP>& samps, int iter, TOKEN_ID* tokens, int* labels, int flag) {
    bool isSft = hFish->isAtPhase(P_SFT);
    // if(!isSft)
    //     return false;
    assert(gBUFF->Qlen != nullptr);
    nValidTokens = 0;
    gBUFF->Qlen->OverWrite(hostLen);
    gBUFF->KVlen->OverWrite(hostLen);
    // if (iter == 2)
    //     DEBUG_HERE;
    bool multi_turn = false;
    string answer = "", a, all_answer = "";
    int r, c, nMostToken = hostToken->size(), *mask = nullptr, *samp_label = nullptr, nIgnore = 0, nPad = 0, posFine = -1;
    TOKEN_ID* samp_token = nullptr;
    for (r = 0; r < nMostSample; r++) {
        int nTokenLen = hostLen->Get(r), posA = -1, posThink = -1;
        posFine    = -1;
        mask       = mask32 + r * ldT;
        samp_label = labels + r * ldT;
        samp_token = tokens + r * ldT;
        // if (iter == 2 && r==0) {
        //     DumpTokens(hDict, TOKENS(line, line + ldT), -1);
        // }

        assert(nTokenLen >= 0 && nTokenLen < ldT);
        bool isAnswer = false;
        answer        = "";
        for (c = 0; c < ldT; c++) {
            if (c >= nTokenLen) {
                BIT_SET(mask[c], MASK_FLAG::F_PAD);
            } else
                BIT_RESET(mask[c], MASK_FLAG::F_PAD);
            BIT_SET(mask[c], MASK_FLAG::F_IGNORE_LOSS);
        }

        // size_t pos = 0;
        // ChatML_samp chatml(pos, multi_turn, multi_turn ? hDict->S._pad : hDict->id_im_end);
        // Tokens2Samp_Chatml(hDict, TOKENS(samp_token, samp_token + ldT), pos, chatml, multi_turn, flag);
        // const TOKENS_SECTION& section = chatml.answers;  // section_metas[r];
        // if (section != section_metas[r]) {                                                       //
        //     samps[r]->Dump(hDict, TOKENS(samp_token, samp_token + ldT), 0x100, "Failed @samp");  //("Failed @samp(%s)", samp->desc.c_str());
        //     assert(0);
        // }
        const TOKENS_SECTION& section = section_metas[r];
        assert(section.size() > 0);
        for (auto [a, b] : section) {  // 139,165
            for (c = a; c < b; c++) {  // it's labels, not original tokens
                assert(samp_token[c] == samp_label[c - 1]);
                answer += hDict->Decode({(TOKEN_ID)samp_token[c]});
                BIT_RESET(mask[c - 1], MASK_FLAG::F_IGNORE_LOSS);
                nValidTokens++;
            }
        }

        all_answer += answer + "\n";
    }
    // _INFO("all_answer=%s",all_answer.c_str());
    if (devMask == nullptr)
        devMask = std::make_shared<huTensor>(hFish, "MaskOfBatch", hostMask->shape, typNUMBER::I32, true, GTensor::F_DEBUG);
    devMask->OverWrite(hostMask);
    if (nValidTokens < nMostToken) {
        mask   = mask32;
        answer = "";
        for (int i = 0; i < nMostToken; i++, mask++) {
            if (BIT_TEST(*mask, MASK_FLAG::F_IGNORE_LOSS)) {
                if (labels[i] >= 0) {
                    labels[i] = -(labels[i] + 1);
                }
                nIgnore++;
            } else {
                a = hDict->Decode({(TOKEN_ID)labels[i]});
                answer += a;
            }
            if (BIT_TEST(*mask, MASK_FLAG::F_PAD)) {
                nPad++;
            }
        }
    }
    // assert(all_answer == answer + "\n");
    assert(nValidTokens > 0);
    // _INFO("[SFT_BATCH] nValidToken=%d(%d) nIgNore=%d nPad=%d answer=%s", nValidTokens, posFine, nIgnore, nPad, all_answer.c_str());
    return true;
}