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
#include "Sample.hpp"

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
enum DT_PHASE { DT_X = 0, DT_TRAIN, DT_EVAL, DT_CHAT, DT_PREDICT, DT_MERGE };

//
enum CORAL_FTYPE { CORAL_FILE, CORAL_BIN, CORAL_TEXT, CORAL_PARQUET, CORAL_GLOB };

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
    DT_PHASE tpDT        = DT_X;
    hSampNanny loader    = nullptr;  //  Each tokenset has a unique samp_loader
    hRANDER hPickRander  = nullptr;
    SAMPLE_TYPE tpSample = RANDOM_GENERATE;
    std::vector<string> shard_paths;
    int nMostShard  = -1;
    int nMostSamp   = -1;
    int shard_index = 0;
    int eval_every  = -1;  // Actually: nanny->eval_every = eval_every;
    int bpToken     = 4;   // bytes_per_token
    // float rStepOfEval = 0;
    int result_marker = -1;  // Tokens after this marker are results(only sum the loss on results). Some marker strings are "=,等于,结果是",

    float rSampling      = 1.0;
    double rSplitOfTrain = 0.5;
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
    TokenCoral(CSTRING& name_, hDataToken hBase, int flag = 0x0);
    virtual ~TokenCoral();
    std::tuple<hDataToken, hDataToken> Split(int type = 0x0, int flag = 0x0);
    virtual bool Init(Fish* hFish, int flag = 0x0);
    virtual bool InitSampNanny(Fish* hFish, DT_PHASE type, int flag = 0x0);
    // virtual hSampNanny GetSampNanny(int flag = 0x0) { return loader; }
    virtual std::string Name(int type = 0x0) { return name; }
    virtual hSampNanny GetNanny() {
        assert(loader != nullptr);
        return loader;
    }

    bool hasMask() { return tokens_mask.size() > 0; }
    virtual bool SetSamps(const std::vector<hSAMP>&, int flag = 0x0);
    TOKEN_ID At(size_t pos);

    bool Serialize(const std::string& path, bool isSave, int flag = 0x0);
    virtual bool LoadNextShard(SampNanny* hLoader, int flag = 0x0) { return true; }
    virtual bool LoadTokenset(struct CLI_params& config, void* hLLM, int flag = 0x0);
    virtual void Append(TOKEN_ID id, int flag = 0x0);

    bool InitSamps(unsigned context_length, std::vector<size_t>& samples_begin, std::vector<size_t>& samples_size, int flag = 0x0);
    virtual std::string Dump(int type, int flag = 0x0);

    friend class NLP_AutoRegressive;
    friend class Fish;
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

class SyntheticCoral : public TokenCoral {
   protected:
    // bool Shard2Sample(int id, int flag = 0x0) override;
    // bool GetShardInfo(int id, int flag = 0x0) override;
    int tpProblem = 1;
    std::filesystem::path pathTxt;
    int nBASE = 97;
    std::string sOperation;
    std::shared_ptr<SAMP> S_math(int i, int j, MAP_S2S& q_kv, int flag = 0x0);
    //  given two elements 𝑎 and 𝑏 from a group 𝐺,the network must predict the result of the operation 𝑎 ∘ 𝑏, where ∘ is a generic operation
    std::shared_ptr<SAMP> S_permutegroup(int i, int j, MAP_S2S& q_kv, int flag = 0x0);
    std::shared_ptr<SAMP> S_Polynomial(int i, int j, MAP_S2S& q_kv, int flag = 0x0);
    std::shared_ptr<SAMP> S_parity(int i, int j, MAP_S2S& q_kv, int flag = 0x0);    //    Sparse Parities. the binary classification tasks of 𝑛-length binary strings

   public:
    SyntheticCoral(const JSON& jit, hTokenizer hDict, int flag = 0x0);
    bool Init(Fish* hFish, int flag = 0x0) override;
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