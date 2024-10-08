/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  concise dictionary on VAE
 * 
 *  \brief LLaMeta Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#pragma once
#include "../ggex/GG_util.hpp"   
#include "../Manifold/Fish.hpp"   
#include "../Manifold/VAE.hpp" 

struct LLaMeta;
struct ConsiceDict : public VariationaAE    {    
    enum OUTPUT_OP {
        ONLY_LOAD=0x0,          //lr=0.001 much more oscillation than 0.0001
        RND_GRAD,               //lr=0.001
        LOAD_GRAD,
        LOAD_GRAD_norm,
    };
    OUTPUT_OP opOut=RND_GRAD;     //LOAD_GRAD_norm;   ONLY_LOAD
    hGensor tok_embeddings=nullptr,norm=nullptr,output=nullptr;
    bool isSVD = false;
    hGensor out_u=nullptr,out_v=nullptr,out_d=nullptr;
    int lo_rank = 128;

    using id    = int32_t;
    using token = std::string;
    using ttype = llama_token_type;        
    struct token_data {
        token text;     float score;        ttype type;
    };   

    bool isLoadTokenEmbed = false;
    LLaMeta *hLM = nullptr;
    int nToken=0,lama_embed=0,latent_dim=256,nLevel=0;
    int nUniqueToken = 0;
    
    std::string tokenizer_name;         //"no_vocab","llama","bert","gpt2"
    std::vector<const char*> tokens;
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
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;

    int token_idx,n_vocab=-1,score_idx,toktype_idx,merges_keyidx,n_merges;
    id special_bos_id = 1,special_eos_id = 2,special_unk_id = 0,special_sep_id = -1,special_pad_id = -1,special_cls_id  = -1,special_mask_id = -1;

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;
    std::unordered_map<token, id> special_tokens_cache;

    int32_t bos,eos;   
    
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

    ConsiceDict(LLaMeta *lama_,int flag=0x0);
    virtual ~ConsiceDict()  {
        FREE_a(scores);      FREE_a(toktypes);
    }
    void LoadVocab(const char*fn_model_base,int flag);
    void LoadVocab_v1(const char*fn_model_base,struct CLI_params& params,llama_model & model,int flag);

    virtual void InitVAE(int flag=0x0);

   

    /*  
        tok_embeddings = llama_get_model_tensor(lama, TN(LLM_TENSOR_TOKEN_EMBD));      nParams+=ggml_nelements(tok_embeddings);
        norm           = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT_NORM));     nParams+=ggml_nelements(norm);
        output         = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT));          nParams+=ggml_nelements(output);
        // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32); 
        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);
    */  
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
    void CreateEmbeddings(struct random_normal_distribution * rnd,int flag);

    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};
typedef std::shared_ptr<ConsiceDict> hCDICT;