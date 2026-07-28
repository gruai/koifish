/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  concise dictionary on VAE
 *
 *  \brief GTokenizer & Dictionary
 *  \author Yingshi Chen
 */

#pragma once
#include <unordered_map>

#include "../Manifold/Fish.hpp"
#include "../Manifold/VAE.hpp"

typedef std::vector<TOKEN_ID> TOKENS;
typedef std::unordered_map<std::string, int> VOCAB_MAP;

struct TOKEN_Special {
    int pad  = -1;
    int mask = -1;
    // [noise]/[riddle] indicator in training & infer of diffusion lm. Although many AR llm model has a special token for mask, like ""mask_token": "<M>" in
    // tokenizer_config.json.
    int noise = -1;

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

    std::unordered_map<int, string> literals;

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
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.*/
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

    int byte_fallback = -1;  // BPE has byte fallback option to convert unk character to utf-8 bytes

    // Deprecated!  vector where the index is the token id and the value is the token string
    // std::vector<std::string> vocab;

    std::unordered_map<std::string, int> vocab;
    // std::unordered_map<int, std::string> id_to_token_;

    virtual int Lookup(const std::string& word, int flag = 0x0);
    // trie mapping token strings to token ids
    // TokenTrie vocab_trie;
    JSON jTokenizer, jVocab;  // from "tokenizer.json"
    JSON jTokenConfig;        // from "tokenizer_config.json"

    size_t max_input_chars_per_word = 0;

    std::string name = "no_vocab";  //"no_vocab","llama","bert","gpt2"

    // 原生LLaMA对中文的支持很弱，一个汉子往往被切分成多个token，因此需要对其进行中文词表扩展。思路通常是在中文语料库上训练一个中文tokenizer模型，然后将中文tokenizer与LLaMA原生tokenizer进行合并，最终得到一个扩展后的tokenizer模型。国内Chinese-LLaMA-Alpaca开源项目详细说明了词表扩展。
    std::vector<const char*> merges;
    bool isIignoreMerges = false;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    float* scores = nullptr;
    int* toktypes = nullptr;
    // Dialect support
    bool isDialect = false;
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
    // special_tokens support
    std::vector<std::string> special_tokens;

    // start index of the byte fallback range (256 tokens). -1 if none.
    int byte_fallback_start = -1;

    // "tokenizer.json" in Huggingface's model card
    virtual bool InitHF(Fish* dolphin, int flag = 0x0);
    virtual bool LoadBin(Fish* dolphin, int flag = 0x0) { return false; }
    virtual bool ReserveVocab(int nReserve, int flag = 0x0);
    // virtual bool InitFrom(Fish* dolphin, hGTensor tokens, hGTensor scores, int flag = 0x0);
    // convenience array containing the decodings for the fixed 256 byte fallbacks '{0x00}\0', '{0x01}\0', ..., '{0xFF}\0'.
    // TODO: use constexpr?
    std::string byte_pieces[256];

   public:
    static const int MAX_TOKEN_LENGTH = 512;
    static const int MAX_TEMPLATE     = 1024;

    TOKEN_Special S;  // Special tokens 1.from "tokenizer.json" &"tokenizer_config.json"

    enum BIT_FLAG {

        F_JVOCAB = 0x10000,
    };
    GTokenizer() {}
    GTokenizer(Fish* lama_, int flag = 0x0);
    virtual ~GTokenizer() {
        FREE_a(scores);
        FREE_a(toktypes);
    }
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
    virtual std::string T2STR(const std::vector<TOKEN_ID>& toks, int flag = 0x0) const { return Decode(toks); }
    virtual std::string T2STR(const int* arrT, int nTok, int flag = 0x0) const {
        std::vector<TOKEN_ID> toks(nTok);
        std::copy(arrT, arrT + nTok, toks.begin());
        return Decode(toks);
    }

    virtual bool DoSomeTest(int flag = 0x0);
    virtual bool CheckSpecialTokens(bool isAllowNone, int flag = 0x0);

    std::string decode_one(int prev_token, int token) const;
    // std::string encoding_to_debug_string(const std::vector<TOKEN_ID>& encoding) const;

    friend class DataTokenSet;
    friend class Tokenset_HellaSwag;
    friend class Tokenset_JSONL;
    friend class GlobTokenset;
    friend class SampLoader;
    friend class Fish;
    friend class NLP_AutoRegressive;
};
typedef std::shared_ptr<GTokenizer> hTokenizer;

//  compatible with HuggingFace tokenizer.json
class HF_Tokenizer : public GTokenizer {
   protected:
    bool InitHF(Fish* dolphin, int flag = 0x0) override;

   public:
    HF_Tokenizer();
    HF_Tokenizer(Fish*, int flag = 0x0);
    ~HF_Tokenizer();

    // Disable copying (PIMPL unique_ptr constraint)
    HF_Tokenizer(const HF_Tokenizer&)            = delete;
    HF_Tokenizer& operator=(const HF_Tokenizer&) = delete;

    // --- Core API ---
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override {
        auto ids = encode(text, encode_bos);
        return TOKENS(ids.begin(), ids.end());
    }

    std::string Decode(const TOKENS& tokens, bool skip_pad = true, bool skip_special_tokens = false) const override {
        std::vector<int> ids(tokens.begin(), tokens.end());
        return decode(ids, skip_special_tokens);
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

   public:
    GTokenizer_CHARset(Fish* nlp_, const std::vector<char>& charset, int flag = 0x0);
    // void LoadVocab(const char* fn_model_base, int flag) override { ; }
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) const override;
    bool isValid(bool allowEmpty, int flag) const override { return true; }
};

class GTokenizer_Heap : public GTokenizer {
   protected:
    struct TokenIndex {
        const char* str = nullptr;
        int id          = -1;
    };
    struct TokenIndex* sorted_vocab = nullptr;

    static int compare_tokens(const void* a, const void* b) { return strcmp(((struct TokenIndex*)a)->str, ((struct TokenIndex*)b)->str); }

    int sLookup(const char* str, int flag = 0x0);
    int Lookup(const std::string& word, int flag = 0x0) override;
    struct Merge {
        int lpos, lid;
        int rpos, rid;
        int resid;
        float score;
    };

    void heap_swap(struct Merge* heap, int i, int j) {
        struct Merge tmp = heap[i];
        heap[i]          = heap[j];
        heap[j]          = tmp;
    }

    void heap_insert(struct Merge* heap, int n_heap, struct Merge merge) {
        // insert a new element at the end (breaks heap invariant)
        heap[n_heap] = merge;
        n_heap++;

        // bubble up the new element to its correct position
        int i = n_heap - 1;
        while (i > 0 && heap[i].score > heap[(i - 1) / 2].score) {
            heap_swap(heap, i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }

    void heap_poptop(struct Merge* heap, int n_heap) {
        // move the last element to the top (breaks heap invariant)
        n_heap--;
        heap[0] = heap[n_heap];

        // bubble down the new top element to its correct position
        int i = 0;
        while (i * 2 + 1 < n_heap) {
            // find the largest child
            int j = i * 2 + 1;
            if (j + 1 < n_heap && heap[j + 1].score > heap[j].score) {
                j++;
            }
            // if the largest child is smaller than the parent, we're done
            if (heap[j].score <= heap[i].score) {
                break;
            }
            // otherwise, swap the parent and child
            heap_swap(heap, i, j);
            i = j;
        }
    }
    int merge_tokens_tryadd(struct Merge* heap, int n_heap, int lpos, int lid, int rpos, int rid);
    int merge_tokens(std::vector<TOKEN_ID>& tokens, int flag = 0x0);

    virtual bool Prepare(int flag = 0x0);

   public:
    GTokenizer_Heap(Fish*, int flag = 0x0);
    virtual ~GTokenizer_Heap() { FREE_a(sorted_vocab); }
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override;
};

/**
 *
 */
class GTokenizer_QWEN3 : public HF_Tokenizer {
   protected:
   public:
    GTokenizer_QWEN3(Fish*, int flag = 0x0);
    bool LoadBin(Fish* dolphin, int flag = 0x0) override;
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos = false, bool encode_eos = false) override;
};

// Deprecated!
struct DictVAE : public VariationaAE {
    enum OUTPUT_OP {
        ONLY_LOAD = 0x0,  // lr=0.001 much more oscillation than 0.0001
        RND_GRAD,         // lr=0.001
        LOAD_GRAD,
        LOAD_GRAD_norm,
    };
    OUTPUT_OP opOut = RND_GRAD;  // LOAD_GRAD_norm;   ONLY_LOAD
    LayerNormal _norm;
    SLP _output;
    hGTensor tok_embeddings = nullptr;  // norm=nullptr,output=nullptr;

    bool init_ok   = false;
    bool isSVD     = false;
    hGTensor out_u = nullptr, out_v = nullptr, out_d = nullptr;
    int lo_rank      = 128;
    hWIKI wiki_tutor = nullptr;

    virtual int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0);
    virtual std::string T2STR(TOKEN_ID tok, int flag = 0x0);
    virtual std::string T2STR(const std::vector<TOKEN_ID>& toks, int flag = 0x0) {
        string line = "";
        for (auto t : toks) {
            if (t == hDict->S.eos)
                break;
            line += T2STR(t, flag);
        }
        return line;
    }

    hTokenizer hDict      = nullptr;
    bool isLoadTokenEmbed = false;
    Fish* dolphin         = nullptr;
    int nToken = 0, lama_embed = 0, latent_dim = 256, nLevel = 0;

    DictVAE(Fish* lama_, int flag = 0x0);
    virtual ~DictVAE() {
        // FREE_a(scores);      FREE_a(toktypes);
    }
    //  n_vocab,scores,toktypes,special_,tokens
    // virtual void LoadVocab_v0(const char*fn_model_base,int flag);
    virtual void LoadVocab(const char* fn_model_base, int flag) { assert(0); }

    // virtual bool LoadTokenizer(const char *filename,int flag=0x0)   {   assert(0);  }

    virtual void InitVAE(int flag = 0x0);

    virtual void Update(struct random_normal_distribution* rnd, int flag = 0x0) {
        if (nLevel > 0) {
            Update_1(rnd, flag);
        } else {
            Update_0(rnd, flag);
        }
    }
    virtual hGTensor Embed2Output(void* ctx, hGTensor t33, int flag = 0x0);
    virtual void Update_0(struct random_normal_distribution* rnd, int flag = 0x0);
    void Update_1(struct random_normal_distribution* rnd, int flag = 0x0);
    void CreateEmbeddings(int flag);

    string __repr__(string& suffix, string& prefix, int flag = 0x0) override;
};
typedef std::shared_ptr<DictVAE> hCDICT;

class CDict_LLAMA : public DictVAE {
   public:
    CDict_LLAMA(Fish* nlp_, int flag = 0x0);
};

// Deprecated!
class CDict_GPT2 : public DictVAE {
   protected:
    // uint32_t vocab_size;
    char** token_table = nullptr;
    int eot_token;  // <|endoftext|> token id

    // bool LoadTokenizer(const char *filename,int flag=0x0)   override;
   public:
    CDict_GPT2(Fish* nlp_, int flag = 0x0);
    virtual ~CDict_GPT2() {
        if (token_table != nullptr) {
            for (uint32_t i = 0; i < hDict->nVocab(); i++) {
                free(token_table[i]);
            }
            free(token_table);
        }
    }
    int InitMAEC(void* ctx, const std::vector<int>& dims_, int flag = 0x0) override;
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) override;
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
};

/*
    Just map characters to int, only for debug!
*/
class CDict_CHAR : public DictVAE {
   public:
    CDict_CHAR(Fish* nlp_, int flag = 0x0);
    void LoadVocab(const char* fn_model_base, int flag) override;
    int InitMAEC(void* ctx, const std::vector<int>& dims_, int flag = 0x0) override;
    int STR2T(const char* txt, int txt_len, std::vector<TOKEN_ID>& btch, int flag = 0x0) override;
    std::string T2STR(TOKEN_ID tok, int flag = 0x0) override;
};

void DumpTokens(hTokenizer hDict, const TOKENS& tokens, int nPad, int flag = 0x0);
bool Tokens2Samp_Chatml(hTokenizer hDict, const TOKENS& tokens, size_t& pos, ChatML_samp& meta, bool multi_turn, int flag = 0x0);

// namespace tokenizer {

class AutoTokenizer {
   public:
    static std::shared_ptr<HF_Tokenizer> from_pretrained(const std::string& path);
};

// } // namespace tokenizer

/**
 struct TokenTrie;

struct TokenTrie {
    std::unordered_map<char, std::shared_ptr<TokenTrie>> children;
    // If non-negative, then this represents the ID of the token formed by the path from the root to this node.
    int token_id = -1;
};

class TrieNode {
   public:
    std::unordered_map<wchar_t, std::unique_ptr<TrieNode>> children;
    bool is_end;
    std::wstring delimiter;

    TrieNode() : is_end(false) {}
};

class Splitter {
   private:
    std::unique_ptr<TrieNode> root;

    void insert(const std::wstring& str) {
        TrieNode* current = root.get();
        for (wchar_t ch : str) {
            if (!current->children[ch]) {
                current->children[ch] = std::make_unique<TrieNode>();
            }
            current = current->children[ch].get();
        }
        current->is_end    = true;
        current->delimiter = str;
    }

   public:
    Splitter(const std::vector<std::wstring>& delimiters) {
        root = std::make_unique<TrieNode>();
        for (const auto& delimiter : delimiters) {
            insert(delimiter);
        }
    }

    std::vector<std::wstring> split(const std::wstring& input) {
        std::vector<std::wstring> result;
        size_t start = 0;

        while (start < input.length()) {
            // Try to find the next delimiter starting from current position
            size_t best_match_length = 0;
            std::wstring matched_delimiter;

            // Check for possible delimiter match starting at current position
            TrieNode* current = root.get();
            size_t pos        = start;

            while (pos < input.length() && current->children.count(input[pos])) {
                current = current->children[input[pos]].get();
                pos++;
                if (current->is_end) {
                    best_match_length = pos - start;
                    matched_delimiter = current->delimiter;
                }
            }

            if (best_match_length > 0) {
                // Add substring before delimiter if it exists
                if (start < start + best_match_length) {
                    result.push_back(input.substr(start, best_match_length));
                }
                start += best_match_length;
            } else {
                // No delimiter found at current position
                size_t next_pos = start + 1;
                bool found_next = false;

                // Find next possible delimiter start
                while (next_pos < input.length()) {
                    if (root->children.count(input[next_pos])) {
                        found_next = true;
                        break;
                    }
                    next_pos++;
                }

                // Add the substring up to next possible delimiter or end
                result.push_back(input.substr(start, (found_next ? next_pos - start : std::wstring::npos)));
                start = next_pos;
            }
        }

        return result;
    }
};*/