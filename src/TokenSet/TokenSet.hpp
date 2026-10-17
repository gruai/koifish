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

#include <assert.h>
#include <glob.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <filesystem>
#include <memory>
#include <random>
#include <vector>

#include "../CLI_params.hpp"
#include "../Manifold/Serialize.hpp"
#include "../Utils/GST_obj.hpp"
#include "../Utils/GST_rander.hpp"
#include "../g_def_x.hpp"

struct DictVAE;
class GTokenizer;
typedef std::shared_ptr<GTokenizer> hTokenizer;
class Fish;
class TokenCoral;
class Head4Token;
class BATCH_INPUT;
typedef std::shared_ptr<TokenCoral> hDataToken;
class SampNanny;
typedef shared_ptr<SampNanny> hSampNanny;
namespace Grusoft {
class FeatsOnFold;
};

// the type of tokenset/dataset
enum DT_PHASE { DT_TRAIN = 1, DT_EVAL, DT_CHAT, DT_PREDICT, DT_MERGE };

//
enum CORAL_FTYPE { CORAL_FILE, CORAL_BIN, CORAL_TEXT, CORAL_PARQUET, CORAL_GLOB };

struct SAMP {
    size_t pos = 0, len = 0;   //  range is [pos,pos+len)
    size_t off_cycle     = 0;  // more random
    int jump             = 0;
    int pad_len          = 0;  // the length of last pad section
    int prefix_blank     = 0;
    TOKEN_ID last_target = (TOKEN_ID)(-1);
    std::vector<TOKEN_ID> ttoks;  // tmp_tokens - may change in training iters
    std::string desc;

    void* target = nullptr;  // for Tokenset_HellaSwag
    TOKENS_SECTION arrROI;   // region of interest:    like answers for chatml-samp

    SAMP() {}
    SAMP(size_t p, size_t l, int nPad = 0);
    virtual ~SAMP() {}

    virtual bool ToTokens(hDataToken coral, int flag = 0x0) {
        assert(len > 0);
        return true;
    }
    bool Serialize(FSerial& S, bool isSave, int flag);
    virtual void Dump(hDataToken coral, const std::vector<TOKEN_ID>& tokens, int type, const std::string& desc = "", int flag = 0x0);
    virtual double UpdateTag(hDataToken hDT, int* tag, int step, bool flip, int flag = 0x0);

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

    std::string state, qid, sLabel, question;  //
    MAP_S2S q_kv;
    // bool state_is_object = false;
    // MAP_S2S questions_raw;  // key -> raw json
    // meta info
    int64_t meta_row = -1;
    std::string meta_source, meta_id, meta_split;

    bool ToTokens(hDataToken coral, int flag = 0x0) override;
    void Dump(hDataToken coral, const std::vector<TOKEN_ID>& tokens, int type, const std::string& desc = "", int flag = 0x0) override;
    bool User2Batch(shared_ptr<BATCH_INPUT> hBatch, int flag = 0x0) override;
};

/*
    v0.2    09/02/2025
        Support multiple shard_samps for multiple hSample
*/
class TokenCoral : public std::enable_shared_from_this<TokenCoral> {
   public:
    enum SAMPLE_TYPE {
        RANDOM_GENERATE,
        HellaSwag,
    };
    enum BIT_FLAG {
        ONLY_COUNT_TOKENS = 0x100,  // NO NEED GEN SAMPLES
        PREPARE4COLLATE   = 0x200
    };

   protected:
    CORAL_FTYPE fType    = CORAL_FILE;
    hSampNanny loader    = nullptr;  //  Each tokenset has a unique samp_loader
    hRANDER hPickRander  = nullptr;
    SAMPLE_TYPE tpSample = RANDOM_GENERATE;
    std::vector<string> shard_paths;
    int nMostShard = -1;

    int shard_index = 0;
    int eval_every  = -1;
    int bpToken     = 4;  // bytes_per_token
    // float rStepOfEval = 0;
    float rSampling = 1.0;
    // bool isNextEpoch = false;
    string name;
    string serial_root;

    Fish* hFish      = nullptr;
    hTokenizer hDict = nullptr;

    // TabularData is mainly for decision models on features
    std::shared_ptr<Grusoft::FeatsOnFold> hTabularData = nullptr;

    std::string fpath;
    size_t fsize = 0, nUnique = 0, nVocab = 0, nDialect = 0, nMostTok = 0;
    std::vector<hSAMP> shard_samps;
    size_t nBatch(int flag = 0x0);
    // char *buffer = nullptr;

    void seek(FILE* fp, size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        assert(ret == 0);  // same
    }

   public:
    static std::tuple<hDataToken, std::vector<hDataToken>, hDataToken, hDataToken> MakeInstance(Fish* hFish, hTokenizer, bool isLocalInfer, int flag);

    int N_mostSamp = -1;
    std::vector<TOKEN_ID> tokens, tokens_mask;
    TokenCoral(hTokenizer hDict);
    virtual ~TokenCoral();
    virtual bool Init(Fish* hFish, int flag = 0x0) { return true; }
    virtual bool InitSampNanny(Fish* hFish, DT_PHASE type, int flag = 0x0);
    // virtual hSampNanny GetSampNanny(int flag = 0x0) { return loader; }
    virtual std::string Name(int type = 0x0) { return name; }
    
    bool hasMask() { return tokens_mask.size() > 0; }

    TOKEN_ID At(size_t pos);

    bool Serialize(const std::string& path, bool isSave, int flag = 0x0);
    virtual bool LoadNextShard(SampNanny* hLoader, int flag = 0x0) { return true; }
    virtual bool LoadTokenset(struct CLI_params& config, void* hLLM, int flag = 0x0);
    virtual void Append(TOKEN_ID id, int flag = 0x0);

    bool InitSamps(unsigned context_length, std::vector<size_t>& samples_begin, std::vector<size_t>& samples_size, int flag = 0x0);
    virtual std::string Dump(int type, int flag = 0x0);

    friend class NLP_AutoRegressive;
    friend class Fish;
    friend class Optimizer;
    friend class SampNanny;
    friend class GeneratOnPrompt;
    friend class SAMP_JEV;
    friend class SAMP;
};
typedef std::vector<hDataToken> DataTokens;

class PromptTokenset : public TokenCoral {
   protected:
    string sPrompt;

   public:
    PromptTokenset(const JSON& jit, hTokenizer hDict, int flag = 0x0);
    PromptTokenset(const string& prompt, hTokenizer hDict, int flag = 0x0);
};
class GlobTokenset : public TokenCoral {
   protected:
    string glob_pattern;
    FILE* fpShard  = nullptr;
    bool isShuffle = false;
    bool isCache   = false;
    virtual bool fp2Tokens(int flag = 0x0);
    virtual bool Shard2Sample(int id, int flag = 0x0);
    virtual bool GetShardInfo(int id, int flag = 0x0) { return false; }
    virtual size_t OnShardFile(int id, bool load = false, int flag = 0x0);
    bool LoadNextShard(SampNanny* hLoader, int flag = 0x0) override;
    size_t total_batch_size;  // total across all processes
    // size_t local_batch_offset;  // inner-sample offset for this process
    size_t longest_example_bytes;
    int header_bytes;
    // int B = -1, T = -1;  // header size in bytes
    size_t szFile, nShardSamples = 0, nShardToks = 0;
    virtual size_t nMostToken(int flag = 0x0) { return nShardToks; }

   public:
    GlobTokenset(const JSON& jit, hTokenizer hDict, int flag = 0x0);
    bool Init(Fish* hFish, int flag = 0x0) override;
};

class Tokenset_HellaSwag : public GlobTokenset {
   protected:
    int nMostCompletion = 4;
    bool Shard2Sample(int id, int flag = 0x0) override;

   public:
    struct QUESTION {
        int b0, b1;
        int label;

        QUESTION(int l, int _b0, int _b1, int flag = 0x0) : label(l), b0(_b0), b1(_b1) {}
    };
    // typedef std::shared_ptr<QUESTION> hQuestion;
    typedef QUESTION* hQuestion;
    std::vector<hQuestion> questions;
    std::vector<hQuestion> quesInBatch;

    Tokenset_HellaSwag(const JSON& jit, hTokenizer hDict, int flag = 0x0);
    virtual ~Tokenset_HellaSwag() {
        for (auto q : questions) delete q;
        questions.clear();
    }
    // double LossOnResult(hSampNanny hLoader, Head4Token* cls, int flag = 0x0) override;
};

/*
    Use .safetensors file to store tokenset: Each sequence/samp is internally consistent!
*/
class Tokenset_KST : public GlobTokenset {
   protected:
    bool Shard2Sample(int id, int flag = 0x0) override;
    // bool GetShardInfo(int id, int flag = 0x0) override;

   public:
    Tokenset_KST(const JSON& jit, hTokenizer hDict, int flag = 0x0);
};

/*
    Use .parquet file to store tokenset: Each sequence/samp is internally consistent!
*/
class Tokenset_PARQUET : public GlobTokenset {
   protected:
    int nCol, nRow;     // for current parquet file
    std::string nameI;  // name of column
    std::vector<string> arrText, arrHeader;
    bool Shard2Sample(int id, int flag = 0x0) override;
    bool GetShardInfo(int id, int flag = 0x0) override;
    std::filesystem::path pathToken;
    virtual bool LoadArrow(const std::filesystem::path&, int flag = 0x0) { NOT_IMPLEMENTED; }

   public:
    Tokenset_PARQUET(const JSON& jit, hTokenizer hDict, int flag = 0x0);
};

class Tokenset_KCODER : public Tokenset_PARQUET {
   protected:
    bool LoadArrow(const std::filesystem::path& path, int flag = 0x0) override;

   public:
    Tokenset_KCODER(const JSON& jit, hTokenizer hDict, int flag = 0x0);
};

/**
 *  A question answering dataset for yes/no questions containing 15942 examples.  Each example is a triplet of (question, passage, answer)
 *      train=9427	validatio=3270
 */
class Tokenset_BoolQ : public Tokenset_PARQUET {
   protected:
    int nPosi = 0, nNega = 0;
    bool LoadArrow(const std::filesystem::path& path, int flag = 0x0) override;

   public:
    Tokenset_BoolQ(const JSON& jit, hTokenizer hDict, int flag = 0x0);
};

/*
    Only one text, random pick sample from this text
*/
class Tokenset_TEXT : public GlobTokenset {
   protected:
    int nCol, nRow;  // for current parquet file
    std::vector<string> arrText;
    bool Shard2Sample(int id, int flag = 0x0) override;
    bool GetShardInfo(int id, int flag = 0x0) override;
    size_t nMostToken(int flag = 0x0) override;

    std::filesystem::path pathToken;

   public:
    Tokenset_TEXT(const JSON& jit, hTokenizer hDict, int flag = 0x0);

    std::string Dump(int type, int flag = 0x0) override;
};

class Tokenset_JSONL : public GlobTokenset {
   protected:
    //  ChatML​ is a tokenization-friendly text formatthat encodes chat messages into a single string
    std::string toChatML(JSON& jMsg, int flag = 0x0);
    string format;
    bool isJsonTxt       = false;
    bool enable_thinking = false;
    bool multi_turn      = false;
    std::vector<string> messages;
    virtual bool GetShardInfo_txt(int id, int flag = 0x0);

    bool Shard2Sample(int id, int flag = 0x0) override;
    bool GetShardInfo(int id, int flag = 0x0) override;
    // void DumpSamp(const std::string& msg, const std::vector<TOKEN_ID>& tokens, int flag = 0x0);

   public:
    Tokenset_JSONL(const JSON& jit, hTokenizer hDict, const string& format, int flag = 0x0);
    virtual ~Tokenset_JSONL() {}
};

class Tokenset_JEV : public Tokenset_JSONL {
   protected:
    bool isAddNonePair = false;
    // bool Shard2Sample(int id, int flag = 0x0) override;
    bool GetShardInfo(int id, int flag = 0x0) override;
    bool Shard2Sample(int id, int flag = 0x0) override;
    // void DumpSamp(const std::string& msg, const std::vector<TOKEN_ID>& tokens, int flag = 0x0);

   public:
    static std::vector<TOKEN_ID> specials;

    Tokenset_JEV(const JSON& jit, hTokenizer hDict, const string& format, int flag = 0x0);
    virtual ~Tokenset_JEV() {}

    std::string Dump(int type, int flag = 0x0) override;
};
int InitSpecialTokensOfDecision(hTokenizer hDict, int flag = 0x0);

class DTS_GPT2 : public TokenCoral {
   public:
    DTS_GPT2(hTokenizer hDict) : TokenCoral(hDict) {}
};