/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  concise dictionary on VAE
 *
 *  \brief GTokenizer & Dictionary
 *  \author Yingshi Chen
 */

#pragma once

#include "../CLI_params.hpp"
#include "../Utils/GST_hashmap.hpp"
#include "../Utils/GST_obj.hpp"
#include "../g_float.hpp"
using namespace std;

typedef std::vector<TOKEN_ID> TOKENS;

struct LiteralSet {
    struct Lit {
        std::string s;
        TOKEN_ID id;
    };
    std::array<std::vector<Lit>, 256> by_first;  // longest literal first within a bucket
    bool empty = true;
    void Build(std::vector<std::pair<std::string, TOKEN_ID>> lits);
    bool Find(std::string_view text, size_t from, size_t& start, size_t& len, TOKEN_ID& id) const;
};
struct TOKEN_Special {
    int _pad  = -1;
    int _mask = -1;
    // [noise]/[riddle] indicator in training & infer of diffusion lm. Although many AR llm model has a special token for mask, like ""mask_token": "<M>" in
    // tokenizer_config.json.
    int _noise = -1;

    int sep = -1, cls = -1, assist = -1;
    // 1. in some model, no bos_token!(GPT-2/GPT-3,unsloth/Qwen3-4B-Base,...)
    int bos = -1;
    //  In many LLMs (including GPT‑2, Qwen, and others):eos == pad == <|endoftext|>, this requires correct attention masks!
    int eos = -1;

    // mainly for sft_training
    int im_start = -1, im_end = -1;
    int think_open = -1, think_close = -1;  //"<think>")"</think>"
    int eot     = -1;                       //  End of Tool / End of Text block
    int unk     = -1;                       //<unk>
    int newline = -1, newline2 = -1;        //\n\n    Qwen3(198,271)

    int equal_token = -1, op_token = -1;  // a*b=c

    std::vector<int*> AllTokens(int flag = 0x0) {
        auto fields = {&_pad, &_mask, &_noise, &sep, &cls, &assist, &bos, &eos, &im_start, &im_end, &think_open, &think_close, &eot, &unk, &newline, &newline2};
        return fields;
    }

    string sep_token = "[SEP]";
    // The token used for padding, for example when batching sequences of different lengths.
    string pad_token = "[PAD]";
    // The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is
    // the first token of the sequence when built with special tokens.
    string cls_token = "[CLS]";
    // The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will
    // try to predict.
    string mask_token = "[MASK]";
    string unk_token  = "<unk>";  //  unknown word, may be null!(null means the tokenizer is designed to never encounter an unknown token​ in practice.)

    std::vector<std::string> control_tokens_;
    std::unordered_map<std::string, TOKEN_ID> mapS2T;

    // 原生LLaMA对中文的支持很弱，一个汉字往往被切分成多个token，因此需要对其进行中文词表扩展。思路通常是在中文语料库上训练一个中文tokenizer模型，然后将中文tokenizer与LLaMA原生tokenizer进行合并，最终得到一个扩展后的tokenizer模型。国内Chinese-LLaMA-Alpaca开源项目详细说明了词表扩展。
    LiteralSet always_, special_;
    size_t max_special_len_ = 0;
    virtual bool Init(CSTRING& jType, const MAP_VOCAB& vocab, int flag = 0x0);

    bool isValid(int flag = 0x0) const;
    bool Has(const int tok, int flag) const;
    std::string Dump(int type, int flag = 0X0) const;
};

//

class Fish;

using ChatMessage  = std::pair<std::string, std::string>;
using ChatMessages = std::vector<ChatMessage>;

/*
 */
class GTokenizer {
   protected:
    CLI_params config;
    string sTokenizerClass = "";
    /* The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering.*/
    string sep_token = "[SEP]";
    // The token used for padding, for example when batching sequences of different lengths.
    string pad_token = "[PAD]";
    // The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification).
    string cls_token = "[CLS]";
    // The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will
    // try to predict.
    string mask_token = "[MASK]";
    string unk_token  = "<unk>";  //  unknown word, may be null!(null means the tokenizer is designed to never encounter an unknown token​ in practice.)

    TOKEN_ID whitespace_token = -1;  // token for " ";
    int byte_fallback         = -1;  // BPE has byte fallback option to convert unk character to utf-8 bytes

    MAP_VOCAB vocab;

    virtual int Lookup(const std::string& word, int flag = 0x0);

    // JSON jTokenizer, jVocab, jAddedTokens;  // from "tokenizer.json"
    JSON jTokenConfig;  // from "tokenizer_config.json"

    int max_input_chars_per_word = 0, nAddedToken = 0;

    std::string name = "no_vocab";  //"no_vocab","llama","bert","gpt2"

    std::vector<const char*> merges;
    // bool isIignoreMerges = false;
    // std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    float* scores  = nullptr;
    int* toktypes  = nullptr;
    bool isVirtual = false;

    // Dialect support
    int UpdateUniqueTokens(const TOKENS& tokens, int flag = 0x0);
    MAP_T2T mapT2T;
    MAP_T2T invT2T;

    // start index of the byte fallback range (256 tokens). -1 if none.
    int byte_fallback_start = -1;

    // "tokenizer.json" in Huggingface's model card
    virtual bool InitFromFiles(Fish* dolphin, int flag = 0x0);
    // virtual bool LoadBin(Fish* dolphin, int flag = 0x0) { return false; }   //Deprecated
    virtual bool ReserveVocab(int nReserve, int flag = 0x0);
    // virtual bool InitFrom(Fish* dolphin, hGTensor tokens, hGTensor scores, int flag = 0x0);
    // convenience array containing the decodings for the fixed 256 byte fallbacks '{0x00}\0', '{0x01}\0', ..., '{0xFF}\0'.
    // TODO: use constexpr?
    std::string byte_pieces[256];

   public:
    static const int MAX_TOKEN_LENGTH = 512;
    static const int MAX_TEMPLATE     = 1024;

    TOKEN_Special S;  // Special tokens 1.from "tokenizer.json" &"tokenizer_config.json"
    bool isDialect = false;
    virtual TOKENS Translate(const TOKENS& ids, int flag = 0x0) const;

    enum class PreKind { Qwen35, DeepSeek3 };
    PreKind tpPreKind = PreKind::Qwen35;

    enum BIT_FLAG {
        F_JVOCAB = 0x10000,

        F_DIALECT_TOKENS = 0x10000,
    };
    GTokenizer() {}
    GTokenizer(Fish* dolphin, int flag = 0x0);
    virtual ~GTokenizer() {
        FREE_a(scores);
        FREE_a(toktypes);
    }
    virtual bool empty(int flag = 0x0) const { return vocab.empty(); }
    virtual int nVocab(int flag = 0x0) const;

    virtual bool isValid(bool allowEmpty = false, int flag = 0x0) const;
    virtual bool isInRange(const int* inp, size_t nz, int flag);
    virtual bool isSpecialTok(const int tok, int flag = 0x0) const;

    virtual std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false);
    virtual std::vector<TOKEN_ID> Encode(const std::wstring& text, bool encode_bos = false, bool encode_eos = false);
    virtual std::string Decode(const TOKENS& ids, bool skip_pad = true, bool skip_special_tokens = false) const;

    virtual int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) {
        btch.clear();
        string line(txt, txt_len);
        btch = Encode(line);
        return btch.size();
    }

    // may return -1, only if txt in tokens-table, would return correct token_id
    virtual int STR2T(const std::string& txt) const { return -1; }

    virtual std::string T2STR(TOKEN_ID tok, int flag = 0x0) const { return Decode({tok}); }
    virtual std::string T2STR(const int* arrT, int nTok, int flag = 0x0) const {
        std::vector<TOKEN_ID> toks(nTok);
        std::copy(arrT, arrT + nTok, toks.begin());
        return Decode(toks);
    }

    virtual bool DoSomeTest(int flag = 0x0);
    virtual bool CheckSpecialTokens(bool isAllowNone, int flag = 0x0);

    virtual std::string Dump(int type, int flag = 0X0) const;

    std::string decode_one(int prev_token, int token) const;
    // std::string encoding_to_debug_string(const std::vector<TOKEN_ID>& encoding) const;
    friend class BATCH_INPUT;
    friend class TokenCoral;
    friend class Tokenset_HellaSwag;
    friend class Tokenset_PARQUET;
    friend class Tokenset_TEXT;
    friend class Tokenset_JSONL;
    friend class GlobTokenset;
    friend class SampNanny;
    friend class Fish;
    friend class NLP_AutoRegressive;
};
typedef std::shared_ptr<GTokenizer> hTokenizer;

//  compatible with HuggingFace tokenizer.json
class HF_Tokenizer : public GTokenizer {
   protected:
    bool InitVocab(CSTRING& path, int flag = 0x0);
    bool InitFromFiles(Fish* dolphin, int flag = 0x0) override;

   public:
    HF_Tokenizer();
    HF_Tokenizer(Fish*, int flag = 0x0);
    ~HF_Tokenizer();

    // Disable copying (PIMPL unique_ptr constraint)
    HF_Tokenizer(const HF_Tokenizer&)            = delete;
    HF_Tokenizer& operator=(const HF_Tokenizer&) = delete;

    // --- Core API ---
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode_i(const std::vector<int>& ids, bool skip_special_tokens = true, int flag = 0x0) const;

    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override {
        auto ids = encode(text, encode_bos);
        return TOKENS(ids.begin(), ids.end());
    }

    std::string Decode(const TOKENS& tokens, bool skip_pad = true, bool skip_special_tokens = false) const override {
        assert(!tokens.empty());
        std::vector<int> ids(tokens.begin(), tokens.end());
        return decode_i(ids, skip_special_tokens || skip_pad);  // pad is also special tokens
    }
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) const override;

    int STR2T(const std::string& txt) const override { return token_to_id(txt); }

    // --- Helpers ---
    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;
    // int nVocab(int flag = 0x0) const override;

    // Special Token Accessors
    int pad_token_id() const;
    int bos_token_id() const;
    int eos_token_id() const;
    int unk_token_id() const;

    // --- Chat Template ---
    void set_chat_template(const std::string& template_str);

    std::string apply_chat_template(const ChatMessages& messages, bool add_generation_prompt = true) const;

    std::string apply_chat_template(const std::string& json_str, bool add_generation_prompt = true) const;

    // --- Loading ---
    bool load_from_json_str(const std::string& json_content);

    // --- Configuration ---
    void set_clean_up_tokenization_spaces(bool clean);

   private:
    struct Impl;  // Forward declaration
    std::unique_ptr<Impl> impl_;
};

class GTokenizer_GPT2 : public GTokenizer {
   protected:
   public:
    GTokenizer_GPT2(Fish*, int flag = 0x0);
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) const override;
};

class GTokenizer_CHARset : public GTokenizer {
   protected:
    std::vector<char> charset;
    std::map<char, TOKEN_ID> mapC2T;

   public:
    GTokenizer_CHARset(Fish* nlp_, const std::vector<char>& charset, int flag = 0x0);
    // void LoadVocab(const char* fn_model_base, int flag) override { ; }
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) const override;
    bool isValid(bool allowEmpty, int flag) const override { return true; }
    std::string Decode(const TOKENS& ids, bool skip_pad = true, bool skip_special_tokens = false) const override;
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override;
};

/**
 *
 */
class GTokenizer_QWEN3 : public HF_Tokenizer {
   protected:
    bool InitFromFiles(Fish* dolphin, int flag = 0x0) override;

   public:
    GTokenizer_QWEN3(Fish*, int flag = 0x0);
    // bool LoadBin(Fish* dolphin, int flag = 0x0) override;   //Deprecated
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override;
};

class GTokenizer_STRATA : public HF_Tokenizer {
   protected:
    bool InitFromFiles(Fish* dolphin, int flag = 0x0) override;

   public:
    GTokenizer_STRATA(Fish*, int flag = 0x0);
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override;
};

// Deprecated!
class CDict_GPT2 : public GTokenizer {
   protected:
    // uint32_t vocab_size;
    char** token_table = nullptr;
    int eot_token;  // <|endoftext|> token id

    // bool LoadTokenizer(const char *filename,int flag=0x0)   override;
   public:
    CDict_GPT2(Fish* nlp_, int flag = 0x0);
    virtual ~CDict_GPT2() {
        /*if (token_table != nullptr) {
            for (uint32_t i = 0; i < hDict->nVocab(); i++) {
                free(token_table[i]);
            }
            free(token_table);
        }*/
    }
    int InitMAEC(void* ctx, const std::vector<int>& dims_, int flag = 0x0);
    std::string T2STR(TOKEN_ID tok, int flag = 0x0);
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
};

/*
    Just map characters to int, only for debug!
*/
class CDict_CHAR : public GTokenizer {
   public:
    CDict_CHAR(Fish* nlp_, int flag = 0x0);
    // virtual void LoadVocab(const char* fn_model_base, int flag);
    int InitMAEC(void* ctx, const std::vector<int>& dims_, int flag = 0x0);
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
    std::string T2STR(TOKEN_ID tok, int flag = 0x0);
};

void DumpTokens(hTokenizer hDict, const TOKENS& tokens, int nPad, int flag = 0x0);
bool Tokens2Samp_Chatml(hTokenizer hDict, const TOKENS& tokens, size_t& pos, ChatML_samp& meta, bool multi_turn, int flag = 0x0);

// namespace tokenizer {

class AutoTokenizer {
   public:
    static std::shared_ptr<HF_Tokenizer> from_pretrained(const std::string& path);
};

// } // namespace tokenizer
