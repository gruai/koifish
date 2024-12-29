/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  concise dictionary on VAE
 * 
 *  \brief NLP_AutoRegressive Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#pragma once
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Fish.hpp"   
#include "../Manifold/VAE.hpp" 

struct NLP_AutoRegressive;
struct ConsiceDict : public VariationaAE    {    
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
    bool tokenizer_add_bos = false;
    bool init_ok = false;
    bool isSVD = false;
    hGensor out_u=nullptr,out_v=nullptr,out_d=nullptr;
    int lo_rank = 128;
    hWIKI wiki_tutor=nullptr;

    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;        
    struct token_data {
        token text;     float score;        ttype type;
    };   

    bool isLoadTokenEmbed = false;
    NLP_AutoRegressive *dolphin = nullptr;
    int nToken=0,lama_embed=0,latent_dim=256,nLevel=0;
    // int nUniqueToken = 0;
    
    std::string tokenizer_name;         //"no_vocab","llama","bert","gpt2"
    // vocabulary
    std::vector<const char*> vocab;
    //原生LLaMA对中文的支持很弱，一个汉子往往被切分成多个token，因此需要对其进行中文词表扩展。思路通常是在中文语料库上训练一个中文tokenizer模型，然后将中文tokenizer与LLaMA原生tokenizer进行合并，最终得到一个扩展后的tokenizer模型。国内Chinese-LLaMA-Alpaca开源项目详细说明了词表扩展。
    std::vector<const char*> merges;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    float * scores=nullptr;
    int * toktypes=nullptr;
    bool isDialect = true;
    int tVocab()    {   
        if(!isDialect)
            return n_vocab;
        else{
            assert(!mapT2T.empty());
            return mapT2T.size();
        }               
    }   

    virtual int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0);
    virtual std::string T2STR(TOKEN_ID tok,int flag=0x0 );  
    virtual std::string T2STR(const std::vector<TOKEN_ID>&toks,int flag=0x0 ) {  
        string line="";
        for(auto t:toks){
            if(t==eos)
                break;
            line += T2STR(t,flag);
        }
            
        return line;
    }   
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;

    int token_idx,n_vocab=-1,score_idx,toktype_idx,merges_keyidx,n_merges;
    id special_bos_id = 1,special_eos_id = 2,special_unk_id = 0,special_sep_id = -1,special_pad_id = -1,special_cls_id  = -1,special_mask_id = -1;

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;
    std::vector<std::string> special_tokens;
    std::unordered_map<token, id> special_tokens_cache;

    int32_t bos = 1,eos = 2;   
    
    int special_add_bos = -1; // -1 unknown, 1 add, 0 don't add.
    int special_add_eos = -1; // -1 unknown, 1 add, 0 don't add.
    id linefeed_id       = 13;
    id special_prefix_id = -1;
    id special_suffix_id = -1;
    id special_middle_id = -1;
    id special_eot_id    = -1; // TODO: move above after "eos_id", and here add "file separator" token

    bool add_space_prefix = true;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
        GGML_ASSERT(token_left.find(' ') == std::string::npos);
        GGML_ASSERT(token_left.find('\n') == std::string::npos);
        GGML_ASSERT(token_right.find(' ') == std::string::npos);
        GGML_ASSERT(token_right.find('\n') == std::string::npos);

        auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }    

    ConsiceDict(NLP_AutoRegressive *lama_,int flag=0x0);
    virtual ~ConsiceDict()  {
        FREE_a(scores);      FREE_a(toktypes);
    }
    //  n_vocab,scores,toktypes,special_,tokens
    virtual void LoadVocab_v0(const char*fn_model_base,int flag);
    virtual void LoadVocab(const char*fn_model_base,int flag)   {   assert(0);  }
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
typedef std::shared_ptr<ConsiceDict> hCDICT;

class CDict_GPT2 : public ConsiceDict{
protected:
    // uint32_t vocab_size;
    char **token_table;    
    int eot_token; // <|endoftext|> token id
    
    // bool LoadTokenizer(const char *filename,int flag=0x0)   override;
public:
    CDict_GPT2(NLP_AutoRegressive *nlp_,int flag=0x0);
    virtual ~CDict_GPT2()   {
        for (uint32_t i = 0; i < n_vocab; i++) {
            free(token_table[i]);
        }
        free(token_table);
    }
    int InitMAEC(struct ggml_context *ctx,const std::vector<int>& dims_,int flag=0x0) override;
    std::string T2STR(TOKEN_ID tok,int flag=0x0 ) override;   
    int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0)    override;
};

/*
    Just map characters to int, only for debug!
*/
class CDict_CHAR : public ConsiceDict{
public:
    CDict_CHAR(NLP_AutoRegressive *nlp_,int flag=0x0);
    void LoadVocab(const char*fn_model_base,int flag)   override;
    int InitMAEC(struct ggml_context *ctx,const std::vector<int>& dims_,int flag=0x0) override;
    int STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0)    override;
    std::string T2STR(TOKEN_ID tok,int flag=0x0 ) override;   
};