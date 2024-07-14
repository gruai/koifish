/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General LLama model  
 * 
 *  \brief LLaMeta Model(https://llama.meta.com/)
 *  \author Yingshi Chen
 */

#include "gLAMA.hpp"
#include "../LLAMA/unicode.h"

// gguf constants (sync with gguf.py)
static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";

static const char * LLM_KV_GENERAL_NAME                = "general.name";
static const char * LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture";
static const char * LLM_KV_GENERAL_FILE_TYPE           = "general.file_type";
static const char * LLM_KV_VOCAB_SIZE                  = "%s.vocab_size";
static const char * LLM_KV_CONTEXT_LENGTH              = "%s.context_length";
static const char * LLM_KV_EMBEDDING_LENGTH            = "%s.embedding_length";

static const char * LLM_KV_BLOCK_COUNT                 = "%s.block_count";
static const char * LLM_KV_FEED_FORWARD_LENGTH         = "%s.feed_forward_length";
static const char * LLM_KV_ATTENTION_HEAD_COUNT        = "%s.attention.head_count";
static const char * LLM_KV_ATTENTION_LAYERNORM_RMS_EPS = "%s.attention.layer_norm_rms_epsilon";
static const char * LLM_KV_ROPE_DIMENSION_COUNT        = "%s.rope.dimension_count";
static const char * LLM_KV_ROPE_FREQ_BASE              = "%s.rope.freq_base"; // TODO load in llama.cpp
static const char * LLM_KV_ROPE_SCALE_LINEAR           = "%s.rope.scale_linear";
static const char * LLM_KV_ATTENTION_HEAD_COUNT_KV     = "%s.attention.head_count_kv";

static const char * LLM_KV_TOKENIZER_MODEL             = "tokenizer.ggml.model";
static const char * LLM_KV_TOKENIZER_LIST              = "tokenizer.ggml.tokens";
static const char * LLM_KV_TOKENIZER_TOKEN_TYPE        = "tokenizer.ggml.token_type";
static const char * LLM_KV_TOKENIZER_SCORES            = "tokenizer.ggml.scores";
static const char * LLM_KV_TOKENIZER_MERGES            = "tokenizer.ggml.merges";
static const char * LLM_KV_TOKENIZER_BOS_ID            = "tokenizer.ggml.bos_token_id";
static const char * LLM_KV_TOKENIZER_EOS_ID            = "tokenizer.ggml.eos_token_id";
static const char * LLM_KV_TOKENIZER_UNK_ID            = "tokenizer.ggml.unknown_token_id";
static const char * LLM_KV_TOKENIZER_SEP_ID            = "tokenizer.ggml.seperator_token_id";
static const char * LLM_KV_TOKENIZER_PAD_ID            = "tokenizer.ggml.padding_token_id";

    // { LLM_KV_DICT_VAE_LAYERS,               "dict.vae.layers"       },
    // { LLM_KV_DICT_LATENT_DIM,                  "%s.dict_latent_dim"},
static const char * LLM_KV_DICT_VAE_LAYERS             = "dict.vae.layers";
static const char * LLM_KV_DICT_LATENT_DIM             = "%s.dict_latent_dim";

static const char * arch = "gruai";     //llm_arch_from_string
static char keybuf[512];
const char * kv(const char * key)   {    
    snprintf(keybuf, 512, key, arch);
    return keybuf;
};

void Ganglia::save_train_(struct save_train_model * data, struct train_state * train) { 
    int64_t iter = train->opt->iter;
    _INFO("%s: iter_%ld\n", __func__, iter);
    string sBaseName = get_train_filename(data->fn_model_out.c_str(), data->pattern_fn_it.c_str(), data->fn_latest.c_str(), -1  );
    if (strlen(data->fn_checkpoint_out.c_str()) > 0) {
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model, train);
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_model_base, data->model, train);
    }
    if (strlen(data->fn_model_out.c_str()) > 0) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        arch = "gruai";                 //llm_arch_from_string
        string sOut = "g_" + sBaseName; 
        save_gguf(sOut.c_str(),0x0);
    }
    if(1){  //only for debug
        arch = "llama";
        string sOut = "l_" + sBaseName;     //hack        
        save_gguf(sOut.c_str(),0x0);
    }

    return;
}

void _T_repr_(hGensor t,const char*tab,char *buf,int flag=0x0){
    if(t==nullptr)      return;
    const char* A = "d";
    if(t->grad!=nullptr){
        A = "P";
    }
    auto ne=t->ne;
    sprintf(buf+strlen(buf),"%s%s '%s' %.3lf(M)\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab, 
        A,t->name,ggml_nelements(t)/1.0e6,ne[0], ne[1], ne[2], ne[3], ggml_type_name(t->type)); 
}

string LLaMeta::__repr__( string& suffix,string& prefix,int flag)         {
    // Ganglia::__repr__(suffix,prefix,flag);
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s(%s):nParams = %ld(%.6gM)",tab,
        "LLaMeta",nParams,nParams/1.0e6);
    sprintf(buf+strlen(buf),"\n%s  tensors=%ld gf=(%d %d)  gb=(%d %d) ",tab, tensors.size(),gf->n_nodes,gf->n_leafs,gb->n_nodes,gb->n_leafs);
    string s="\n",p=prefix+"\t";
    sprintf(buf+strlen(buf),"%s",hDict->__repr__(s,p,0x0).c_str());
    int nLayer = layers.size();
    if(nLayer>0)    {
        auto layer = layers[0];
        sprintf(buf+strlen(buf),"%s  [%s] x %d\n",tab,layer->name.c_str(),nLayer);
        sprintf(buf+strlen(buf),"%s",layer->__repr__(s,p,0x0).c_str());     
        sprintf(buf+strlen(buf),"%s  ......\n",tab);
        sprintf(buf+strlen(buf),"%s",layers[layers.size()-1]->__repr__(s,p,0x0).c_str());    
    }
    _T_repr_(target_probs,"  target_probs=",buf);  
    if(wiki!=nullptr && wiki->logits!=nullptr)
        _T_repr_(wiki->logits,"  wiki_logits=",buf);    
    _T_repr_(loss,"  loss=",buf);   

    sprintf(buf+strlen(buf),"%s",suffix.c_str()); 
    _INFO("%s",buf); 
    return buf;
}

string MutliCoder::__repr__( string& suffix,string& prefix,int flag)   {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s resi=%d tpNorm=%d\n",prefix.c_str(),isResi,tpNorm);
    _T_repr_(encode,tab,buf);   
    _T_repr_(decode,tab,buf);   
    _T_repr_(norm,tab,buf);   
    _T_repr_(resi,tab,buf);   
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}

string ConsiceDict::__repr__( string& suffix,string& prefix,int flag)     {
    char buf[5012]="\0";
    const char* _ops[]= {
        "ONLY_LOAD","RND_GRAD","LOAD_GRAD,","LOAD_GRAD_norm",
    };
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s(%s):resi=%d tpNorm=%d opOut=\"%s\" nLevel=%d\n",prefix.c_str(),
        "ConsiceDict",(int)(reserve_x),tpNorm,_ops[opOut],nLevel);
    
    _T_repr_(tok_embeddings,tab,buf);   
    _T_repr_(norm,tab,buf);   
    _T_repr_(output,tab,buf);   
    if(nLevel>0){
        // sprintf(buf+strlen(buf),"%s\tdims=",tab);
        
        string s="\n",p=prefix+"\t";
        auto vae = MAEC[0];
        sprintf(buf+strlen(buf),"%s  [%s] x %d\tdims=",tab,vae->Name().c_str(),MAEC.size());
        for(auto dim : dims)           {
            sprintf(buf+strlen(buf),"%d ",dim);
        }
        sprintf(buf+strlen(buf),"%s",vae->__repr__(s,p,0x0).c_str());  
    }
    sprintf(buf+strlen(buf),"\n");

    sprintf(buf+strlen(buf),"%s",suffix.c_str());
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}


ConsiceDict::ConsiceDict(LLaMeta *lama_,int flag) : VariationaAE(),hLM(lama_)   {
    assert(hLM->isValid());
    hparams = hLM->hparams;
    reserve_x = true;
    isSymmetric = false;
    lama_embed = hparams.n_embd;

    if(hLM->hparams.nabla==3){
         dims = {hparams.n_embd, 256};
        // dims = {hparams.n_embd, 1024, 256};
        //dims = {hparams.n_embd,1024,256,64};       //little difference with {hparams.n_embd,1024,256,128}
        nLevel = dims.size()-1;   
        latent_dim = dims[nLevel];
    }   else if(hLM->hparams.nabla>3)
        assert(0);         
    _INFO("%s symmetric=%d resi=%d tpNorm=%d opOut=%d nLevel=%d dims= ",__func__,(int)(isSymmetric),(int)(reserve_x),tpNorm,opOut,nLevel);
    for(auto dim : dims)           {
        _INFO("%d ",dim);
    }
    _INFO("\n");
}

void ConsiceDict::InitVAE(int flag)  {
    assert(nLevel>0);   
    if(nLevel>=1){
        isLoadTokenEmbed = true;
        InitMAEC(hLM->ctx,dims);
        // hMultiCoder hCoder = std::make_shared<MutliCoder>(hLM->ctx, hparams.n_embd, latent_dim);
        // MAEC.push_back(hCoder);
        // encoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, hparams.n_embd, latent_dim);     
        // decoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, latent_dim, hparams.n_embd); 
    }    
    hLM->hparams.n_embd = latent_dim;        
}

/**
 * Only for compare & debug
 * 
 * llm_load_vocab

void ConsiceDict::LoadVocab_v1(const char*fn_model_base,struct CLI_params& params,llama_model & model,int flag) {
    llama_model_loader ml(fn_model_base, true,false,nullptr);
    // model.hparams.vocab_only = params.vocab_only;
    assert(model.arch==LLM_ARCH_GRUAI);
    

    auto & vocab = model.vocab;
    bpe_ranks = vocab.bpe_ranks;
}*/

void ConsiceDict::CreateEmbeddings(struct random_normal_distribution * rnd,int flag){
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab,last_dim=dims[dims.size()-1];
    auto lama = hLM->GetRawModel( );  
    auto ctx = hLM->ctx;    
    
    if(isLoadTokenEmbed) {
        const uint32_t n1 = isSymmetric ? n_embd : last_dim;
        if(opOut==RND_GRAD){
            norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n1);
            output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n1, n_vocab);  
        }else if(opOut==LOAD_GRAD_norm){
            output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n1, n_vocab);  
        }
        return;
    }

    tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);  
}

void ConsiceDict::Update_0(struct random_normal_distribution * rnd,int flag){
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab;
    auto lama = hLM->GetRawModel( );  
    if(isLoadTokenEmbed) {
        bool isParam = false;
        // get tensors from llama_model (possibly mmapped)
        tok_embeddings = llama_get_model_tensor(lama, TN(LLM_TENSOR_TOKEN_EMBD));      
        if(isParam) nParams+=ggml_nelements(tok_embeddings);
        norm           = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT_NORM));     
        if(isParam) nParams+=ggml_nelements(norm);
        output         = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT));          
        if(isParam) nParams+=ggml_nelements(output);
    }   else   {
        auto ctx = hLM->ctx;
        /*tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
        norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab); */ 

        hLM->InitGensor(ctx,tok_embeddings, TN(LLM_TENSOR_TOKEN_EMBD), rnd);
        hLM->InitGensor(ctx,norm,           TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        hLM->InitGensor(ctx,output,         TN(LLM_TENSOR_OUTPUT), rnd);
    }
    // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);              
    }else{

    }      
}

void ConsiceDict::Update_1(struct random_normal_distribution * rnd,int flag) {
    const uint32_t n_embd  = hparams.n_embd,n_vocab = hparams.n_vocab;

    bool isParam = false;
    // get tensors from llama_model (possibly mmapped)
    auto lmodel = hLM->GetRawModel( );  
    tok_embeddings = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_TOKEN_EMBD) );        //TN(LLM_TENSOR_TOKEN_EMBD)
    if(isParam) hLM->nParams+=ggml_nelements(tok_embeddings);
    switch(opOut){
    case ONLY_LOAD:
        norm           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );       //  
        // if(isParam) hLM->nParams+=ggml_nelements(norm);
        output         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  );            //
        // if(isParam) hLM->nParams+=ggml_nelements(output);
        break;
    case LOAD_GRAD_norm:    //bug@Optimizer::ggml_train
        norm           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );
        assert(norm->type==GGML_TYPE_F32);
        ggml_set_param(hLM->ctx, norm);         hLM->nParams += ggml_nelements(norm);           hLM->tensors[norm->name] = norm;
        hLM->InitGensor(hLM->ctx,output,         TN(LLM_TENSOR_OUTPUT), rnd);
        break;
    case LOAD_GRAD:     //bug!!!
        norm           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );
        if(norm->type!=GGML_TYPE_F32)   Gensor2float(hLM->ctx,norm);
        ggml_set_param(hLM->ctx, norm);         hLM->nParams += ggml_nelements(norm);           hLM->tensors[norm->name] = norm;
        output         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  ); 
        if(output->type!=GGML_TYPE_F32)   {
            output->data = Gensor2float(hLM->ctx,output);       output->type = GGML_TYPE_F32;
        }
        ggml_set_param(hLM->ctx, output);     hLM->nParams += ggml_nelements(output);           hLM->tensors[output->name] = output;
        break;
    case RND_GRAD:
        hLM->InitGensor(hLM->ctx,norm,           TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        hLM->InitGensor(hLM->ctx,output,         TN(LLM_TENSOR_OUTPUT), rnd);
        break;

    default:
        assert(0);
    }
    
    assert(tok_embeddings!=nullptr && norm!=nullptr && output!=nullptr);

    // ggml_tensor_dequant(ctx_compute,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, hparams.n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, hparams.n_vocab);              
    }
    int i = 0;
    for(auto map : MAEC){
        std::string name = TN(LLM_DICT_DOWN, i);    //"dict.0.down.weight"
        hLM->InitGensor(hLM->ctx, map->encode,    TN(LLM_DICT_DOWN, i),     rnd); 
        if(map->decode!=nullptr)
            hLM->InitGensor(hLM->ctx, map->decode,    TN(LLM_DICT_UP, i),       rnd);    
        i++;            
    }
    //ggml_set_param(hLM->ctx, norm);              hLM->nParams+=ggml_nelements(norm);
    //output is Q6k would fail @float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i)
    //ggml_set_param(hLM->ctx, output);            hLM->nParams+=ggml_nelements(output);

    hLM->tensors[ggml_get_name(tok_embeddings)] = tok_embeddings;
    hLM->tensors[ggml_get_name(norm)] = norm;
    hLM->tensors[ggml_get_name(output)] = output;  
    assert(tensors.size()==0);          
}

void ConsiceDict::LoadVocab(const char*fn_model_base,int flag)     {
    string word;
    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;   //LLAMA_FTYPE_ALL_F32;
    struct gguf_init_params params = {        false,NULL,    };
    struct gguf_context * vctx = gguf_init_from_file(fn_model_base, params);

    token_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_LIST));
    if (token_idx == -1) {
        die("cannot find tokenizer vocab in model file");
    }
    n_vocab = gguf_get_arr_n(vctx, token_idx);
    int nTT = gguf_get_arr_n(vctx, token_idx);          assert(n_vocab==nTT);
    score_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_SCORES));
    if (score_idx == -1) {
        die("cannot find tokenizer scores in model file");
    }
    scores = new float[nTT];
    memcpy(scores,gguf_get_arr_data(vctx, score_idx),sizeof(float)*nTT);    

    toktype_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
    if (toktype_idx == -1) {
        die("cannot find token type list in GGUF file");
    }
    assert( nTT == gguf_get_arr_n(vctx, toktype_idx));
    toktypes = new int[nTT];
    memcpy(toktypes,gguf_get_arr_data(vctx, toktype_idx),sizeof(int)*nTT);    
    GGUF_GET_KEY(vctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));
    if (tokenizer_name == "llama") {
        // default special tokens
        special_bos_id = 1;
        special_eos_id = 2;
        special_unk_id = 0;
        special_sep_id = -1;
        special_pad_id = -1;
    } else if (tokenizer_name == "gpt2") {
        // read and copy bpe merges
        merges_keyidx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_MERGES));
        if (merges_keyidx == -1) {
            die("cannot find tokenizer merges in model file");
        }
        n_merges = gguf_get_arr_n(vctx, merges_keyidx);
        // std::vector<const char*> merges;
        merges.resize(n_merges);
        for (int i = 0; i < n_merges; i++) {
            merges[i] = strdup(gguf_get_arr_str(vctx, merges_keyidx, i));
            word = merges[i];
            GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);
            std::string first,second;
            const size_t pos = word.find(' ', 1);
            if (pos != std::string::npos) {
                first  = word.substr(0, pos);
                second = word.substr(pos + 1);
            }
            bpe_ranks.emplace(std::make_pair(first, second), i);
        }
        word = merges[0];
        // gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_MERGES), merges.data(), n_merges);
        // default special tokens
        special_bos_id = 11;        special_eos_id = 11;        special_unk_id = -1;
        special_sep_id = -1;        special_pad_id = -1;
    } else {
        fprintf(stderr, "%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
        fprintf(stderr, "%s: using default tokenizer: 'llama'", __func__);
    }

    // std::vector<const char*> tokens;
    tokens.resize(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
        tokens[i] = strdup(gguf_get_arr_str(vctx, token_idx, i));
    }
    // gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), tokens.data(), n_vocab);
    GGUF_GET_KEY(vctx, special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
    GGUF_GET_KEY(vctx, special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
    GGUF_GET_KEY(vctx, special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
    GGUF_GET_KEY(vctx, special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
    GGUF_GET_KEY(vctx, special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

    gguf_free(vctx);
}

static void save_checkpoint_file(const char * filename, const char * fn_model_base, struct llama_model * model, struct train_state * train) {
    _INFO("%s: saving to %s\n", __func__, filename);
    struct gguf_context * fctx = gguf_init_empty();

    // save_checkpoint_gguf(fctx, fn_model_base, model, train);
    gguf_set_val_str(fctx, LLM_KV_TRAINING_TYPE, LLM_KV_TRAINING_TYPE_TRAIN_MODEL);
    // save_llama_model_gguf(fctx, fn_model_base, model);
    save_train_state_gguf(fctx, train);

    // write file
    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

// void save_llama_model_file(const char * filename, const char * fn_model_base, struct llama_model * model) {
//     _INFO("%s: saving to %s\n", __func__, filename);
//     struct gguf_context * fctx = gguf_init_empty();

//     save_llama_model_gguf(fctx, fn_model_base, model);

//     // write file
//     const bool only_meta = false;
//     gguf_write_to_file(fctx, filename, only_meta);
//     gguf_free(fctx);
// }

/*
    // optionally save the session on first sample (for faster prompt loading next time)
    if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
        need_to_save_session = false;
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

        LOG("saved session to %s\n", path_session.c_str());
    }
*/
string LLaMeta::lama_layer::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    _T_repr_(attention_norm,tab,buf);
    _T_repr_(wq,tab,buf);        _T_repr_(wk,tab,buf);       _T_repr_(wv,tab,buf);   _T_repr_(wo,tab,buf);   
    _T_repr_(ffn_norm,tab,buf);     _T_repr_(ffn_gate,tab,buf);     _T_repr_(ffn_down,tab,buf);     _T_repr_(ffn_up,tab,buf);  
    _T_repr_(eps,tab,buf);
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}

//n_embd_head, n_head_kv
hGensor  LLaMeta::build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor  KQ_pos,/*hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down,*/ int flag) {
    auto train_params = hparams.common;
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;  
    const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    const int n_past = 0, n_head_kv=hparams.n_head_kv,n_embd_head = hparams.n_embd_head();
    hGensor wq = UpdateGensor (layer->wq->name);                     
    hGensor wk = layer->wk==nullptr ? nullptr : UpdateGensor (layer->wk->name);
    hGensor wv = layer->wv==nullptr ? nullptr : UpdateGensor (layer->wv->name);
    hGensor wo = UpdateGensor (layer->wo->name);
    hGensor attention_norm = UpdateGensor (layer->attention_norm->name);    
    hGensor ffn_norm = layer->ffn_norm==nullptr ? nullptr : UpdateGensor (layer->ffn_norm->name); 
    hGensor ffn_up = nullptr,ffn_gate=nullptr,ffn_down=nullptr;
    if(layer->ffn_up!=nullptr){                
        ffn_up = UpdateGensor (layer->ffn_up->name);
        ffn_gate = UpdateGensor (layer->ffn_gate->name);
        ffn_down = UpdateGensor (layer->ffn_down->name);                
    }  
    /*  LORA
        hGensor wq = n_rank_wq ==0 ? nullptr : UpdateGensor (layer->wq->name);                     
        hGensor wk = n_rank_wk ==0 ? nullptr : UpdateGensor (layer->wk->name);
        hGensor wv = n_rank_wv ==0 ? nullptr : UpdateGensor (layer->wv->name);
        hGensor wo = UpdateGensor (layer->wo->name);
        hGensor a_norm = UpdateGensor (layer->attention_norm->name);   
        hGensor ffn_norm = UpdateGensor (layer->ffn_norm->name);
        hGensor ffn_up = UpdateGensor (layer->ffn_up->name);
        hGensor ffn_gate = UpdateGensor (layer->ffn_gate->name);
        hGensor ffn_down = UpdateGensor (layer->ffn_down->name);
    */

    //  rms_norm:   Root Mean Square Layer Normalization
    hGensor  t02 = ggml_rms_norm     (ctx_compute, cur, f_norm_rms_eps);                    set_name(t02, "t02");     assert_shape_2d(t02, n_embd, N*n_batch);
    hGensor  t03 = ggml_repeat       (ctx_compute, attention_norm, t02);              set_name(t03, "t03");     assert_shape_2d(t03, n_embd, N*n_batch);
    hGensor  t04 = ggml_mul          (ctx_compute, t03, t02);                               set_name(t04, "t04");     assert_shape_2d(t04, n_embd, N*n_batch);
    // QKV_Motion qkv(wq,wk,wv,n_embd,n_head,  N, n_batch,n_rot, n_ctx,n_head_kv,f_norm_rms_eps,rope_freq_base,rope_freq_scale);
    hBrownMotion hBrown = CreateBrownMotion(wq, wk, wv);                              
    hGensor t16 = hBrown->Build(ctx_compute , t04,  KQ_pos,  train_params.use_flash);        
    hGensor  t17 = ggml_permute      (ctx_compute, t16, 0, 2, 1, 3);                        set_name(t17, "t17");     assert_shape_4d(t17, n_embd/n_head, n_head, N, n_batch);
    hGensor  t18 = ggml_cont         (ctx_compute, t17);                                    set_name(t18, "t18");     assert_shape_4d(t18, n_embd/n_head, n_head, N, n_batch);
    hGensor  t19 = ggml_reshape_2d   (ctx_compute, t18, n_embd, N*n_batch);                 set_name(t19, "t19");     assert_shape_2d(t19, n_embd, N*n_batch);
    hGensor  t20 = ggml_mul_mat      (ctx_compute, wo, t19);                          set_name(t20, "t20");     assert_shape_2d(t20, n_embd, N*n_batch);
    hGensor  t21 = ggml_add          (ctx_compute, t20, cur);                               set_name(t21, "t21");     assert_shape_2d(t21, n_embd, N*n_batch);
    hGensor  ffn = nullptr;
    switch(tpFFN)   {
    case VAR_LAST:
    case SWIGLU:    {
        hGensor  t22 = ggml_rms_norm     (ctx_compute, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
        ffn = t22;
        if(ffn_norm!=nullptr)       {
            hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            hGensor  t24 = ggml_mul          (ctx_compute, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch); 
            ffn = t24;                 
        }
          
        if(ffn_up!=nullptr){
            // hGensor  t22 = ggml_rms_norm     (ctx_compute, t21, f_norm_rms_eps);                    set_name(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
            // hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            // hGensor  t24 = ggml_mul          (ctx_compute, t23, t22);                               set_name(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
            hGensor  t24 = ffn;
            hGensor  t25 = ggml_mul_mat      (ctx_compute, ffn_up, t24);                      set_name(t25, "t25");     assert_shape_2d(t25, n_ff, N*n_batch);
            hGensor  t26 = ggml_mul_mat      (ctx_compute, ffn_gate, t24);                    set_name(t26, "t26");     assert_shape_2d(t26, n_ff, N*n_batch);
            hGensor  t27 = ggml_silu         (ctx_compute, t26);                                    set_name(t27, "t27");     assert_shape_2d(t27, n_ff, N*n_batch);
            hGensor  t28 = ggml_mul          (ctx_compute, t27, t25);                               set_name(t28, "t28");     assert_shape_2d(t28, n_ff, N*n_batch);
            hGensor  t29 = ggml_mul_mat      (ctx_compute, ffn_down, t28);                    set_name(t29, "t29");     assert_shape_2d(t29, n_embd, N*n_batch);
            hGensor  t30 = ggml_add          (ctx_compute, t29, t21);                               set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
            ffn = t30;
        }
        if(layer->eps!=nullptr){
            // hGensor  t300 = ffn!=nullptr ? ggml_rms_norm(ctx_compute, ffn, f_norm_rms_eps) : ggml_rms_norm(ctx_compute, t21, f_norm_rms_eps);
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_compute, layer->eps, 0.001);
            ffn = ggml_add          (ctx_compute, ffn,noise);     
        }else{
        }
        return ffn;   
    }
    case VAR_0:
    case ONLY_LNormal:
    case ONLY_RMSNormal:    {
        assert(ffn_up==nullptr);
        hGensor  t22 = tpFFN==ONLY_LNormal ? ggml_norm(ctx_compute, t21, f_norm_rms_eps) :
            ggml_rms_norm(ctx_compute, t21, f_norm_rms_eps);
        set_name(t22, "t22");               assert_shape_2d(t22, n_embd, N*n_batch);   
        if(tpFFN==VAR_0)     {
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_compute, layer->eps, 0.001);
            ffn = ggml_add          (ctx_compute, t22,noise);     
            // ffn = t22;        
        }else{
            hGensor  t23 = ggml_repeat       (ctx_compute, ffn_norm, t22);                    set_name(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            ffn = ggml_mul          (ctx_compute, t23, t22);
        }
        return ffn;
    }
    default:
        TO_DO;
        break;
    }
    if(ffn_up==nullptr)   {        
        
        /*trick v0
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = eps * std + mu        
        //  
        // hGensor  tMu = ggml_mul          (ctx_compute, layer->w_mu, t21);
        // hGensor  tVar = ggml_mul          (ctx_compute, layer->w_var, t21);*/
        //  trick v1
                              
            // set_name(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);
        return ffn;
    }else{
        
    }                 
}

void VariationaAE::save_gguf(struct gguf_context *fctx, int flag)   {
    if(MAEC.size()==0)
        return;
    int nLay = MAEC.size()+1;       assert(nLay>=2);
    gguf_set_arr_data(fctx, kv(LLM_KV_DICT_VAE_LAYERS), GGUF_TYPE_INT32, dims.data(), nLay);  
    for(auto coder:MAEC){
        gguf_add_tensor(fctx, coder->encode);
        if(coder->decode!=nullptr)
            gguf_add_tensor(fctx, coder->decode);
    }   
}

void LLaMeta::save_gguf(const char * filename, int flag) {
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;       //LLAMA_FTYPE_MOSTLY_Q2_K
    _INFO("%s: saving to %s ftype=%d ......\n", __func__, filename,ftype);
    struct gguf_context * fctx = gguf_init_empty();
    int keyidx = -1;    
    
    // set arch
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_str(fctx, LLM_KV_GENERAL_NAME, ".");
    gguf_set_val_u32(fctx, kv(LLM_KV_VOCAB_SIZE), n_vocab);
    int llm_embd = hDict->lama_embed,latent_dim=hDict->latent_dim;        //hparams.n_embd
    if(hDict->nLevel>0)    assert(llm_embd>latent_dim && latent_dim>0);
    // set hparams
    const char*str = kv(LLM_KV_CONTEXT_LENGTH);
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              hparams.n_ctx                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            llm_embd                       );
    
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 hparams.n_layer                );
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         hparams.n_ff                   );
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        hparams.n_rot                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        hparams.n_head                 );    
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV),     hparams.n_head_kv              );

    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), hparams.f_norm_rms_eps         );
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              hparams.rope_freq_base         ); // TODO load in llama.cpp
    
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);
    // set vocab by copying from vocab_model gguf file
    gguf_set_val_str(fctx, kv(LLM_KV_TOKENIZER_MODEL), hDict->tokenizer_name.c_str());

    gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), hDict->tokens.data(), n_vocab);
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, hDict->scores, n_vocab);    
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, hDict->toktypes, n_vocab);
    if (hDict->tokenizer_name == "gpt2"){
        const char* sMERGES = kv(LLM_KV_TOKENIZER_MERGES);
        gguf_set_val_u32(fctx, sMERGES,  hDict->merges_keyidx              );
        keyidx = gguf_find_key(fctx, sMERGES);      //only for debug
        assert(hDict->merges.size()==hDict->n_merges);
        string word = hDict->merges[0];
        gguf_set_arr_str(fctx, sMERGES, hDict->merges.data(), hDict->n_merges);
        /*for (int i = 0; i < hDict->n_merges; i++) {        //only for debug
            const std::string word = gguf_get_arr_str(fctx, keyidx, i);
            GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);            
        }*/
    }   
    
    
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_BOS_ID), hDict->special_bos_id);
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_EOS_ID), hDict->special_eos_id);
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_UNK_ID), hDict->special_unk_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_SEP_ID), hDict->special_sep_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_PAD_ID), hDict->special_pad_id);      -1
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           1.0f / hparams.rope_freq_scale );
    gguf_set_val_u32(fctx, kv(LLM_KV_DICT_LATENT_DIM),             latent_dim                       );
    //more maybe from llama_chat_apply_template
    // add tensors
    gguf_add_tensor(fctx, hDict->tok_embeddings);   //4096*128256
    gguf_add_tensor(fctx, hDict->norm);             //4096
    gguf_add_tensor(fctx, hDict->output);           //4096*128256
    hDict->save_gguf(fctx, flag);    

    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        auto layer = dynamic_pointer_cast<lama_layer>(layers[i]); //layers[i];

        gguf_add_tensor(fctx, layer->attention_norm);
        gguf_add_tensor(fctx, layer->wq);
        if(layer->wk!=nullptr)
            gguf_add_tensor(fctx, layer->wk);
        if(layer->wv!=nullptr)
            gguf_add_tensor(fctx, layer->wv);
        gguf_add_tensor(fctx, layer->wo);
        if(layer->ffn_norm!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_norm);
        if(layer->ffn_gate!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_gate);
        if(layer->ffn_down!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_down);
        if(layer->ffn_up!=nullptr)
            gguf_add_tensor(fctx, layer->ffn_up);
    }

    const bool only_meta = false;
    gguf_write_to_file(fctx, filename, only_meta);
    gguf_free(fctx);
}

void LLaMeta::BuildTarget( struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,hGensor cur,hGensor _tNorm,hGensor _tOutput,hGensor KQ_pos, int flag)  {
    int n_tokens = hparams.n_ctx;
    auto train_params = hparams.common;
    const int N = n_tokens, n_past = 0;
    const float rms_norm_eps = hparams.f_norm_rms_eps;
    hGensor  t32 = nullptr;
    hGensor  t31   = ggml_rms_norm          (ctx, cur, rms_norm_eps);                    set_name(t31, "t31");     assert_shape_2d(t31, n_embd, N*n_batch);
    
    if(hDict->nLevel>0){
        t31 = hDict->DEC(ctx,t31);      //t31 = ggml_mul_mat(ctx, hDict->decoder, t31 );  
        set_name(t31, "embed_decoder");
        t32 = ggml_repeat            (ctx, _tNorm, t31); 
        n_embd = t32->ne[0];
    }   else
        t32   = ggml_repeat            (ctx, _tNorm, t31);                            
    set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
    hGensor  t33   = ggml_mul               (ctx, t32, t31);                             set_name(t33, "t33");     assert_shape_2d(t33, n_embd, N*n_batch);
    hGensor  t34   = ggml_mul_mat           (ctx, _tOutput, t33);                          set_name(t34, "t34");     assert_shape_2d(t34, n_vocab, N*n_batch);
    hGensor  t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);             set_name(t35, "t35");     assert_shape_3d(t35, n_vocab, N, n_batch);
    if(wiki!=nullptr && wiki->logits!=nullptr)    {   //GPT mode's input is different with TRAINING!!!
        t35 = ggml_add(ctx,t35,wiki->logits);
    }
    hGensor  t36   = ggml_cross_entropy_loss(ctx, t35, target_probs);                    set_name(t36, "t36");     assert_shape_1d(t36, 1);
    if(hDict->nLevel>0){
        n_embd = hparams.n_embd;
    }
    if (train_params.use_checkpointing) {
        checkpoints.push_back(t31);            checkpoints.push_back(t32);            checkpoints.push_back(t33);
        checkpoints.push_back(t34);            checkpoints.push_back(t35);            checkpoints.push_back(t36);
    }

    ggml_build_forward_expand(gf, t36);

    if (train_params.use_checkpointing) {
        ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx, gf, gb, true);
    }

    GGML_ASSERT(alloc != NULL);

    // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
    int n_leafs_before = gb->n_leafs;
    int n_nodes_before = gb->n_nodes;

    // output_ tensors
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t35, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36, 1.0f));
    // input gradient
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, t36->grad, 1.0f));
    GGML_ASSERT(t36->grad->data == NULL && t36->grad->view_src == NULL);
    ggml_set_input(t36->grad);
    // KQ_pos
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, KQ_pos, 1.0f));
    // allocating checkpoints in one block to reduce memory fragmentation
    // note: they will be freed in reverse order

    for (unsigned int i = 0; i < checkpoints.size(); ++i) {
        if (checkpoints[i]->data == NULL && checkpoints[i]->view_src == NULL) {
            ggml_set_input(checkpoints[i]);
        }
    }

    if (measure_only) {
        ggml_gallocr_reserve(alloc, gb);
    } else {
        ggml_gallocr_alloc_graph(alloc, gb);    //367,8088

        // set KQ_pos
        {
            int * data = (int *) KQ_pos->data;
            for (int i = 0; i < N; ++i) {
                data[i] = n_past + i;
            }
        }
    }

    // remove the additional nodes and leafs
    for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
        gb->leafs[i] = NULL;
    }
    for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
        gb->nodes[i] = NULL;
    }
    gb->n_leafs = n_leafs_before;
    gb->n_nodes = n_nodes_before;

    logits = t35;
    // return t36;
    loss = t36;

            /*  ???
    // make sure base model tensors data cannot be used in viewable operations
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, base->tok_embeddings, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, base->norm, 1.0f));
    ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, base->output, 1.0f));
    for (int il = 0; il < n_layer; ++il) {
        // struct my_llama_layer & layer = layers[il];
        auto layer = dynamic_pointer_cast<lama_layer>(layers[il]);
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->attention_norm, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->ffn_norm, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->wq, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->wk, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->wv, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->wo, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->ffn_gate, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->ffn_down, 1.0f));
        ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, layer->ffn_up, 1.0f));
    }*/
}

struct ggml_cgraph * llama_build_graph(llama_context & lctx,const llama_batch & batch,bool   worst_case);

void LLaMeta::LAMA::Decode(std::vector<llama_token>&embd_inp,int flag) {
    llama_token eos = llama_token_eos(lmodel),id;
    int i=0,n_consumed=0;
    std::vector<llama_token> embd;
    while ((int) embd_inp.size() > n_consumed) {
        id = embd_inp[n_consumed];
        // const std::string token_str = llama_token_to_piece(_ctx, id);
        // _INFO("%s",token_str.c_str());
        embd.push_back(embd_inp[n_consumed]);
        ++n_consumed;       
    }
    int n_eval = (int) embd.size(),n_past =0;
    auto batch = llama_batch_get_one(&embd[0], n_eval, n_past, 0);
    llama_decode(_ctx,batch );      n_past+=n_eval;
    llama_ctx_get_(_ctx,(void**)(&logits_out),10);  
}

void LLaMeta::LAMA::Answer(std::vector<llama_token>&embd_inp,int flag) {
    llama_token eos = llama_token_eos(lmodel),id;
    int i=0,n_consumed=0;
    std::vector<llama_token> embd;    
    /*  9493 -> 'when'   279 -> ' the' 16603 -> ' smoke'   374 -> ' is'  2133 -> ' going'  1523 -> ' down'    11 -> ','*/
    struct llama_sampling_params sparams;
    sparams.seed = 42;
    struct llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);
    _INFO("%s \"%s\" embd_inp.size(): %d, n_consumed: %d\n",__func__,"", (int) embd_inp.size(), n_consumed);
    while ((int) embd_inp.size() > n_consumed) {
        id = embd_inp[n_consumed];
        const std::string token_str = llama_token_to_piece(_ctx, id);
        _INFO("%s",token_str.c_str());
        embd.push_back(embd_inp[n_consumed]);
        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        llama_sampling_accept(ctx_sampling, _ctx, embd_inp[n_consumed], false);
        ++n_consumed;       
    }
    int n_eval = (int) embd.size(),n_past =0;
    auto batch = llama_batch_get_one(&embd[0], n_eval, n_past, 0);
            //     llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            // ggml_cgraph * gf = llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);
    if(0)   {
        gf = llama_build_graph(*_ctx, batch, false);
        res  = gf->nodes[gf->n_nodes - 1];
        GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor");
    }
        // struct ggml_tensor * embd = gf->nodes[gf->n_nodes - 2];
    llama_decode(_ctx,batch );      n_past+=n_eval;
    llama_ctx_get_(_ctx,(void**)(&logits_out),10);
    // auto tmaps = llama_internal_get_tensor_map(_ctx);       //tmpas contain all weights,not all tensors!
    // for(auto it : tmaps){
    //     if(it.first=="result_output")
    //         res = it.second;
    // }
    // assert(res!=nullptr);    

    id = llama_sampling_sample(ctx_sampling, _ctx, nullptr);      //539
    llama_sampling_accept(ctx_sampling, _ctx, id, true);
    const std::string token_str = llama_token_to_piece(_ctx, id);
    _INFO("%s => \"%s\"",__func__,token_str.c_str());

}

