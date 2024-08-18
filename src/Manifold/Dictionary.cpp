#include "Dictionary.hpp"
#include "gLAMA.hpp"
#include "../LLAMA/unicode.h"



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

void _T_repr_(hGensor t,const char*tab,char *buf,int flag=0x0);
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
    // n_vocab = hparams.n_vocab;      //Maybe 0!  would get correct value @LoadVocab!
    latent_dim = hparams.n_embd;
    if(hLM->hparams.nabla>3)
        assert(0);
    if(!hLM->hparams.vae.empty()){
    // if(hLM->hparams.nabla==3){
        dims = {hparams.n_embd, 256};
        // dims = {hparams.n_embd, 1024, 256};
        //dims = {hparams.n_embd,1024,256,64};       //little difference with {hparams.n_embd,1024,256,128}
        nLevel = dims.size()-1;   
        latent_dim = dims[nLevel];
        _INFO("%s symmetric=%d resi=%d tpNorm=%d opOut=%d nLevel=%d dims= ",__func__,(int)(isSymmetric),(int)(reserve_x),tpNorm,opOut,nLevel);
    }   else     {   /**/  
        latent_dim = hparams.dict_latent_dim;   //256;       
        _INFO("%s latent_dim=%d",__func__,latent_dim);
    }
    
    for(auto dim : dims)           {
        _INFO("%d ",dim);
    }
    _INFO("\n");
}

void ConsiceDict::InitVAE(int flag)  {
    if(nLevel==0){

    }  else if(nLevel>=1){
        isLoadTokenEmbed = true;
        InitMAEC(hLM->ctx,dims);
        // hMultiCoder hCoder = std::make_shared<MutliCoder>(hLM->ctx, hparams.n_embd, latent_dim);
        // MAEC.push_back(hCoder);
        // encoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, hparams.n_embd, latent_dim);     
        // decoder = ggml_new_tensor_2d(hLM->ctx, GGML_TYPE_F32, latent_dim, hparams.n_embd); 
    }    
    hLM->hparams.n_embd = latent_dim;        
}

void ConsiceDict::CreateEmbeddings(struct random_normal_distribution * rnd,int flag){
    assert(hLM!=nullptr);
    uint32_t n_embd  = hparams.n_embd;
    auto lama = hLM->GetRawModel( );  
    auto ctx = hLM->ctx;    
    if(nLevel==0){
        n_embd = latent_dim;
    }else{
        const uint32_t last_dim=dims[dims.size()-1];        
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
    }

    tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    norm           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    output         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);  
}

void ConsiceDict::Update_0(struct random_normal_distribution * rnd,int flag){
    const uint32_t n_embd  = hparams.n_embd;
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
        assert_shape_2d(tok_embeddings, hparams.n_embd, n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, n_vocab);              
    }else{

    }      
}

void ConsiceDict::Update_1(struct random_normal_distribution * rnd,int flag) {
    const uint32_t n_embd  = hparams.n_embd;

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
        ggml_set_param(hLM->ctx, norm);         hLM->nParams += ggml_nelements(norm);           
        hLM->Gensor2Map(norm);       // hLM->tensors[norm->name] = norm;
        hLM->InitGensor(hLM->ctx,output,         TN(LLM_TENSOR_OUTPUT), rnd);
        break;
    case LOAD_GRAD:     //bug!!!
        norm           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );
        if(norm->type!=GGML_TYPE_F32)   Gensor2float(hLM->ctx,norm);
        ggml_set_param(hLM->ctx, norm);         hLM->nParams += ggml_nelements(norm);           
        hLM->Gensor2Map(norm);       //hLM->tensors[norm->name] = norm;
        output         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  ); 
        if(output->type!=GGML_TYPE_F32)   {
            output->data = Gensor2float(hLM->ctx,output);       output->type = GGML_TYPE_F32;
        }
        ggml_set_param(hLM->ctx, output);     hLM->nParams += ggml_nelements(output);           
        hLM->Gensor2Map(output);       //hLM->tensors[output->name] = output;
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
        assert_shape_2d(tok_embeddings, hparams.n_embd, n_vocab);
        assert_shape_1d(norm,           hparams.n_embd);
        assert_shape_2d(output,         hparams.n_embd, n_vocab);              
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
    hLM->Gensor2Map(tok_embeddings); 
    // hLM->tensors[ggml_get_name(tok_embeddings)] = tok_embeddings;
    // hLM->tensors[ggml_get_name(norm)] = norm;
    // hLM->tensors[ggml_get_name(output)] = output;  
    assert(gensors.size()==0);          
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

void LLaMeta::save_gguf(const char * filename, int flag) {
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;       //LLAMA_FTYPE_MOSTLY_Q2_K
    _INFO("[save] saving gguf to %s ftype=%d ......\n", filename,ftype);
    struct gguf_context * fctx = gguf_init_empty();
    int keyidx = -1;    
    
    // set arch
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch);
    gguf_set_val_str(fctx, LLM_KV_GENERAL_NAME, ".");
    gguf_set_val_u32(fctx, kv(LLM_KV_VOCAB_SIZE), hDict->n_vocab);
    int llm_embd = hDict->lama_embed,latent_dim=hDict->latent_dim;        //hparams.n_embd
    if(hDict->nLevel>0)    assert(llm_embd>latent_dim && latent_dim>0);
    // set hparams
    const char*str = kv(LLM_KV_CONTEXT_LENGTH);
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              hparams.common.n_ctx                  );
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

    gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), hDict->tokens.data(), hDict->n_vocab);
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, hDict->scores, hDict->n_vocab);    
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, hDict->toktypes, hDict->n_vocab);
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