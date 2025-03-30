/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT 
 *  
 *  concise dictionary on VAE
 * 
 *  \brief GTokenizer & Dictionary
 *  \author Yingshi Chen
 */

#pragma once
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Fish.hpp"   
#include "../Manifold/VAE.hpp" 

class Fish;
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
        current->is_end = true;
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
            size_t pos = start;
            
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
};

/*
    BPE(Byte-Pair Encoding)

    Smart implementation of https://github.com/andrewkchan/yalm/blob/main/src/tokenizer.h 

    A tokenizer vocab consists of a concatenated tensor with the key "tokenizer.tokens" in the .yalm file.
    Shown as a list of strings:
    ```
    "tokenizer.tokens": [
    "<unk>",        // 0
    "<s>",          // 1
    "</s>",         // 2
    "<0x00>",       // 3--------------+
    "<0x01>",       // 4              |  Byte
    "<0x02>",       // 5              |  Fallback 
    ...                               |  Tokens
    "<0xFE>",       // 257            |
    "<0xFF>",       // 258------------+
    "▁▁",           // 259
    "▁▁▁▁",         // 260
    "▁t",           // 261
    "in",           // 262
    "er",           // 263
    ...
    ]
    ```
    In tensor form, it looks like a UTF-8 encoded byte array:
    ```
    <unk>\0<s>\0</s>\0<0x00>\0<0x01>\0<0x02>\0...\0<0xFE>\0<0xFF>\0▁▁\0▁▁▁▁\0▁t\0in\0er\0...
    ```
    Important token IDs are included in the metadata of the .yalm file:
    ```
    "bos_token_id": "1",
    "eos_token_id": "2",
    ```
*/
typedef std::vector<TOKEN_ID> TOKENS;
class GTokenizer {
protected:    
    CLI_params config;
    /* The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.*/
    string sep_token ="[SEP]";         
    // The token used for padding, for example when batching sequences of different lengths.
    string pad_token ="[PAD]";
    // The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
    string cls_token="[CLS]";
    // The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.   
    string mask_token = "[MASK]";
    string unk_token = "<unk>";      //  unknown word

    int byte_fallback = -1;    //BPE has byte fallback option to convert unk character to utf-8 bytes

    // vector where the index is the token id and the value is the token string
    std::vector<std::string> vocab;
    // trie mapping token strings to token ids
    TokenTrie vocab_trie;
    JSON jTokenizer,jVocab;
    size_t max_input_chars_per_word=-1;
    
    bool tokenizer_add_bos = false;

    std::string name = "no_vocab";         //"no_vocab","llama","bert","gpt2"

//原生LLaMA对中文的支持很弱，一个汉子往往被切分成多个token，因此需要对其进行中文词表扩展。思路通常是在中文语料库上训练一个中文tokenizer模型，然后将中文tokenizer与LLaMA原生tokenizer进行合并，最终得到一个扩展后的tokenizer模型。国内Chinese-LLaMA-Alpaca开源项目详细说明了词表扩展。
    std::vector<const char*> merges;
    bool isIignoreMerges = false;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    float *scores=nullptr;
    int *toktypes=nullptr;
// Dialect support 
    bool isDialect = false;  
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
// special_tokens support
    std::vector<std::string> special_tokens;
    // std::unordered_map<token, id> special_tokens_cache;
 
    // start index of the byte fallback range (256 tokens). -1 if none.
    int byte_fallback_start = -1;

    // "tokenizer.json" in Huggingface's model card
    bool LoadHFJson(const string& path,int flag=0x0);
    virtual void InitTrier(int flag=0x0);
    virtual bool InitHF(Fish *dolphin,int flag=0x0);
    virtual bool InitFrom(Fish *dolphin,hGTensor tokens,hGTensor scores,int flag=0x0);
    // convenience array containing the decodings for the fixed 256 byte fallbacks '{0x00}\0', '{0x01}\0', ..., '{0xFF}\0'.
    // TODO: use constexpr?
    std::string byte_pieces[256];
public:
    int sep_id=-1,pad_id=-1,cls_id=-1,mask_id=-1;
    int bos_id = -1,eos_id = -1,eot_id = -1;

    enum BIT_FLAG {        

        F_JVOCAB=0x10000,  
    };  
    GTokenizer()    {}
    GTokenizer(Fish *lama_,int flag=0x0);
    virtual ~GTokenizer()  {
        FREE_a(scores);      FREE_a(toktypes);
    }
    virtual int nVocab(int flag=0x0)    const;
    virtual bool isValid(int flag=0x0)  const;
    virtual bool isInRange(const int* inp,size_t nz,int flag);

    virtual std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos=false) const;
    virtual std::vector<TOKEN_ID> Encode(const std::wstring& text, bool encode_bos=false) const;
    virtual std::string Decode(const TOKENS& ids, bool skip_special_tokens=false);
    
    virtual int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0){
        string line(txt,txt_len);
        btch = Encode(line);
        return btch.size();
    }
    virtual std::string T2STR(TOKEN_ID tok,int flag=0x0 ){
        return Decode({tok});
    }
    virtual std::string T2STR(const std::vector<TOKEN_ID>&toks,int flag=0x0 ) {  
        return Decode(toks);
    } 

    std::string decode_one(int prev_token, int token) const;
    std::string encoding_to_debug_string(const std::vector<TOKEN_ID>& encoding) const;

    /*
    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
        assert(token_left.find(' ') == std::string::npos);
        assert(token_left.find('\n') == std::string::npos);
        assert(token_right.find(' ') == std::string::npos);
        assert(token_right.find('\n') == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }    */

friend class DataTokenSet;  friend class Tokenset_HellaSwag; friend class GlobTokenset;
friend class SampLoader;
friend class Fish;
friend class NLP_AutoRegressive;
};
typedef std::shared_ptr<GTokenizer> hTokenizer;

/*
    subword tokenization algorithm
    from  https://github.com/Sorrow321/huggingface_tokenizer_cpp
*/
class WordPieceTokenizer : public GTokenizer   {
private:


protected:
    // std::string trim( std::string const& original )
    // {
    //     std::string::const_iterator right = std::find_if(original.rbegin(), original.rend(), IsNotSpace()).base();
    //     std::string::const_iterator left = std::find_if(original.begin(), right, IsNotSpace() );
    //     return std::string( left, right );
    // }
    wstring wunk;
    vector<wstring> wspecial;
    std::vector<std::wstring> split(const std::wstring& input)  const {
        std::wstringstream stream(input);
        std::vector<std::wstring> words;
        std::wstring word;
        while (stream >> word) {
            words.push_back(word);
        }
        return words;
    }
    std::string decode_one(int prev_token, int token) const;
public:
    WordPieceTokenizer(Fish *lama_,int flag=0x0);
    WordPieceTokenizer(const string& config_path);

    int get_word_index(const wstring& word) const;
    std::vector<TOKEN_ID> Encode(const std::string& text, bool encode_bos=false) const  override;
    vector<TOKEN_ID> Encode(const wstring& input_text, bool split_specials=false) const override;
    // std::string Decode(const TOKENS& ids, bool skip_special_tokens=false)   override;   

    vector<wstring> wordpiece_tokenize(const wstring& input_text) const   {
        vector<wstring> tokens = split(input_text);
        vector<wstring> output_tokens;
        for(size_t i = 0; i < tokens.size(); i++) {
            auto& tok = tokens[i];
            if(tok.length() > max_input_chars_per_word) {
                output_tokens.push_back(wunk);
                continue;
            }

            bool is_bad = false;
            size_t start = 0;
            vector<wstring> sub_tokens;

            while(start < tok.length()) {
                size_t end = tok.length();
                wstring cur_substr;
                while(start < end) {
                    wstring substr = tok.substr(start, end-start);
                    if(start > 0) {
                        substr = L"##" + substr;
                    }
                    size_t idx = get_word_index(substr);
                    if(idx != -1) {
                        cur_substr = substr;
                        break;
                    }
                    end--;
                }

                if(cur_substr.empty()) {
                    is_bad = true;
                    break;
                }
                sub_tokens.push_back(cur_substr);
                start = end;
            }

            if(is_bad) {
                output_tokens.push_back(wunk);
            }else{
                output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
            }
        }
        return output_tokens;
    }

    vector<size_t> convert_tokens_to_ids(const vector<wstring>& input_seq)  const  {
        vector<size_t> output_ids;
        for(size_t i = 0; i < input_seq.size(); i++) {
            output_ids.push_back(get_word_index(input_seq[i]));
        }
        return output_ids;
    }

};

struct DictVAE : public VariationaAE    {    
    enum OUTPUT_OP {
        ONLY_LOAD=0x0,          //lr=0.001 much more oscillation than 0.0001
        RND_GRAD,               //lr=0.001
        LOAD_GRAD,
        LOAD_GRAD_norm,
    };
    OUTPUT_OP opOut=RND_GRAD;     //LOAD_GRAD_norm;   ONLY_LOAD
    LayerNormal _norm;
    SLP _output;
    hGensor tok_embeddings=nullptr; //norm=nullptr,output=nullptr;
    // bool tokenizer_add_bos = false;
    bool init_ok = false;
    bool isSVD = false;
    hGensor out_u=nullptr,out_v=nullptr,out_d=nullptr;
    int lo_rank = 128;
    hWIKI wiki_tutor=nullptr;    

    virtual int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0);
    virtual std::string T2STR(TOKEN_ID tok,int flag=0x0 );  
    virtual std::string T2STR(const std::vector<TOKEN_ID>&toks,int flag=0x0 ) {  
        string line="";
        // for(auto t:toks){
        //     if(t==eos)
        //         break;
        //     line += T2STR(t,flag);
        // }
            
        return line;
    } 

    hTokenizer hDict = nullptr;
    bool isLoadTokenEmbed = false;
    Fish *dolphin = nullptr;
    int nToken=0,lama_embed=0,latent_dim=256,nLevel=0;

    DictVAE(Fish *lama_,int flag=0x0);
    virtual ~DictVAE()  {
        // FREE_a(scores);      FREE_a(toktypes);
    }
    //  n_vocab,scores,toktypes,special_,tokens
    // virtual void LoadVocab_v0(const char*fn_model_base,int flag);
    virtual void LoadVocab(const char*fn_model_base,int flag)   {   assert(0);  }
    virtual bool isValid(int flag=0x0)  {   
        
        return true; 
    }
    // virtual bool LoadTokenizer(const char *filename,int flag=0x0)   {   assert(0);  }
    
    
    virtual void InitVAE(int flag=0x0);

    virtual void Update(struct random_normal_distribution * rnd,int flag=0x0)   {
        if(nLevel>0){
            Update_1(rnd,flag);
        }else{
            Update_0(rnd,flag);
        }
    }
    virtual hGensor Embed2Output(struct ggml_context * ctx,hGensor t33,int flag=0x0);
    virtual void Update_0(struct random_normal_distribution * rnd,int flag=0x0);
    void Update_1(struct random_normal_distribution * rnd,int flag=0x0);  
    void CreateEmbeddings(int flag);

    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};
typedef std::shared_ptr<DictVAE> hCDICT;


class CDict_LLAMA : public DictVAE{
public:
    CDict_LLAMA(Fish *nlp_,int flag=0x0);
};
class CDict_GPT2 : public DictVAE{
protected:
    // uint32_t vocab_size;
    char **token_table=nullptr;    
    int eot_token; // <|endoftext|> token id
    
    // bool LoadTokenizer(const char *filename,int flag=0x0)   override;
public:
    CDict_GPT2(Fish *nlp_,int flag=0x0);
    virtual ~CDict_GPT2()   {
        if(token_table!=nullptr){
            for (uint32_t i = 0; i < hDict->nVocab(); i++) {
                free(token_table[i]);
            }
            free(token_table);            
        }

    }
    int InitMAEC(struct ggml_context *ctx,const std::vector<int>& dims_,int flag=0x0) override;
    std::string T2STR(TOKEN_ID tok,int flag=0x0 ) override;   
    int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0)    override;
};

/*
    Just map characters to int, only for debug!
*/
class CDict_CHAR : public DictVAE{
public:
    CDict_CHAR(Fish *nlp_,int flag=0x0);
    void LoadVocab(const char*fn_model_base,int flag)   override;
    int InitMAEC(struct ggml_context *ctx,const std::vector<int>& dims_,int flag=0x0) override;
    int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0)    override;
    std::string T2STR(TOKEN_ID tok,int flag=0x0 ) override;   
};