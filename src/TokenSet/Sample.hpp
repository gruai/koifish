/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  CORALs provide enviroments for fish to survice, reproduce & evolve.
 *  Tokenset manager. Key component of speed & accuracy
 *
 *  \brief Tokenset from files/hugging/...
 *  \author Yingshi Chen
 */
#pragma once
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

class TokenCoral;
class BATCH_INPUT;

struct SAMP {
    size_t pos = 0, len = 0;   //  range is [pos,pos+len)
    size_t off_cycle     = 0;  // more random
    int jump             = 0;
    int last_pad_len     = 0;  // the length of last pad section <= BATCH_INPUT::nPadToken
    int nPrefill         = 0;
    int nCollate         = 0;
    int prefix_blank     = 0;
    TOKEN_ID last_target = (TOKEN_ID)(-1);
    std::vector<TOKEN_ID> ttoks;  // tmp_tokens - may change in training iters
    std::string desc, sLabel;

    void* target = nullptr;  // for Tokenset_HellaSwag
    TOKENS_SECTION arrROI;   // region of interest:    like 1). answers for chatml-samp 2). tokens after result_marker
    size_t nzROI(int no, int flag = 0x0) {
        assert(no >= 0 && no < arrROI.size());
        auto [begin, end] = arrROI[0];
        return end - begin;
    }
    SAMP() {}
    SAMP(size_t p, size_t l, int nPad = 0);
    virtual ~SAMP() {}

    enum S2TOKEN { RAW, DAOYIN, STATE_QUESTION };
    S2TOKEN s2token = DAOYIN;
    virtual bool ToTokens(std::shared_ptr<TokenCoral> coral, S2TOKEN s2t, int flag = 0x0) {
        assert(len > 0);
        return true;
    }
    bool Serialize(FSerial& S, bool isSave, int flag);
    virtual void Dump(std::shared_ptr<TokenCoral> coral, const std::vector<TOKEN_ID>& tokens, int type, const std::string& desc = "", int flag = 0x0);
    virtual double UpdateTag(std::shared_ptr<TokenCoral> hDT, int* tag, int step, bool flip, int flag = 0x0);

    static size_t HASH(const char* fn, const std::vector<std::shared_ptr<SAMP>>& samps) {
        std::hash<std::string> h_string;
        std::hash<unsigned long long> h_ull;
        size_t h = h_string(std::string(fn)), sample_count = samps.size();
        h = HASH_combine(h, h_ull((unsigned long long)sample_count));
        for (auto samp : samps) {
            h = HASH_combine(h, h_ull((unsigned long long)samp->pos));
            h = HASH_combine(h, h_ull((unsigned long long)samp->len));
        }
        return h;
    }
    virtual bool User2Batch(shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) { return true; }
};
typedef std::shared_ptr<SAMP> hSAMP;

/**
 * "messages": [
    {"role": "system", "content": "You are a knowledgeable assistant."},
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "The sky appears blue because molecules in the air scatter shorter blue wavelengths of sunlight more than longer red
 wavelengths."}
  ]
 */
struct SAMP_CHATML : public SAMP {};  //  struct ChatML_samp

// With json structure, store much more meta info
struct SAMP_JEV : public SAMP {
    SAMP_JEV(CSTRING& state, CSTRING& qid, MAP_S2S& q_kv, int flag = 0x0);
    //  a Choice maps onto branches, a Score onto a threshold, a Noul onto an if
    enum tpDECISION {
        NOUL,  //  "type": "noul",
        CHOICE,
        SCORE
    };
    tpDECISION tpDecisoin = NOUL;
    union {
        bool bY;
        float fY;
        int iY;
    };

    std::string state, qid, question;  //
    MAP_S2S q_kv;
    // bool state_is_object = false;
    // MAP_S2S questions_raw;  // key -> raw json
    // meta info
    int64_t meta_row = -1;
    std::string meta_source, meta_id, meta_split;

    bool MergeToCoral(std::shared_ptr<TokenCoral> coral, int flag = 0x0);
    bool ToTokens(std::shared_ptr<TokenCoral> coral, S2TOKEN s2t, int flag = 0x0) override;
    void Dump(std::shared_ptr<TokenCoral> coral, const std::vector<TOKEN_ID>& tokens, int type, const std::string& desc = "", int flag = 0x0) override;
    bool User2Batch(shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};