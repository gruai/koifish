#include "Dictionary.hpp"
#include "gLLM.hpp"
#include "unicode.h"



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

static const char * arch_str = "gruai";     //llm_arch_from_string
static char keybuf[512];
const char * kv(const char * key)   {    
    snprintf(keybuf, 512, key, arch_str);
    return keybuf;
};


string ConsiceDict::__repr__( string& suffix,string& prefix,int flag)     {
    char buf[5012]="\0";
    const char* _ops[]= {
        "ONLY_LOAD","RND_GRAD","LOAD_GRAD,","LOAD_GRAD_norm",
    };
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s[%s]:resi=%d tpNorm=%d opOut=\"%s\" nLevel=%d\n",prefix.c_str(),
        "ConsiceDict",(int)(reserve_x),tpNorm,_ops[opOut],nLevel);
    
    _T_repr_(tok_embeddings,tab,buf);   
    _T_repr_(_norm.w,tab,buf);              _T_repr_(_norm.b,tab,buf);   
    _T_repr_(_output.w,tab,buf);            _T_repr_(_output.b,tab,buf);   
    _T_repr_(out_u,tab,buf);
    _T_repr_(out_d,tab,buf);
    _T_repr_(out_v,tab,buf);
    if(nLevel>0){
        // sprintf(buf+strlen(buf),"%s\tdims=",tab);
        
        string s="\n",p=prefix+"\t";
        auto vae = MAEC[0];
        sprintf(buf+strlen(buf),"%s  [%s] x %ld\tdims=",tab,vae->Name().c_str(),MAEC.size());
        for(auto dim : dims)           {
            sprintf(buf+strlen(buf),"%d ",dim);
        }
        sprintf(buf+strlen(buf),"%s",vae->__repr__(s,p,0x0).c_str());  
    }
    // sprintf(buf+strlen(buf),"\n");

    sprintf(buf+strlen(buf),"%s",suffix.c_str());
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}


ConsiceDict::ConsiceDict(NLP_AutoRegressive *lama_,int flag) : VariationaAE(),hLM(lama_)   {
    assert(hLM->isValid());
    hparams = hLM->hparams;
    isDialect = hparams.dict_dialect == "on";
    isSVD = hparams.dict_logits == "svd";
    _norm.Init(lama_);              
    _output.Init(lama_); 
    reserve_x = true;
    isSymmetric = false;
    lama_embed = hparams.n_embd;
    // n_vocab = hparams.n_vocab;      //Maybe 0!  would get correct value @LoadVocab!
    latent_dim = hparams.n_embd;
    if(hLM->hparams.nabla>3)
        assert(0);
    if(!hLM->hparams.vae.empty()){
    // if(hLM->hparams.nabla==3){
        dims = {(int)hparams.n_embd, 256};
        // dims = {hparams.n_embd, 1024, 256};
        //dims = {hparams.n_embd,1024,256,64};       //little difference with {hparams.n_embd,1024,256,128}
        nLevel = dims.size()-1;   
        latent_dim = dims[nLevel];
        _INFO("%s symmetric=%d resi=%d tpNorm=%d opOut=%d nLevel=%d dims= ",__func__,(int)(isSymmetric),(int)(reserve_x),tpNorm,opOut,nLevel);
    }   else     {   /**/  
        if(hLM->hparams.wiki_actor!="copy") {
            latent_dim = hparams.dict_latent_dim;   //256;       
        }            
        _INFO("%s latent_dim=%d Dialect=%s",__func__,latent_dim,isDialect?"ON":"OFF");
    }
    if(hLM->hparams.wiki_actor!="copy") {
        hLM->hparams.n_embd = latent_dim;   //Reset n_embd just like nLayerX
        // hLM->hparams.SetHead(latent_dim);   // ???????
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
        InitMAEC(hLM->GetCTX(),dims);
        // hMultiCoder hCoder = std::make_shared<MutliCoder>(hLM->GetCTX(), hparams.n_embd, latent_dim);
        // MAEC.push_back(hCoder);
        // encoder = ggml_new_tensor_2d(hLM->GetCTX(), GGML_TYPE_F32, hparams.n_embd, latent_dim);     
        // decoder = ggml_new_tensor_2d(hLM->GetCTX(), GGML_TYPE_F32, latent_dim, hparams.n_embd); 
    }    
         
}

void ConsiceDict::CreateEmbeddings(int flag){
    assert(hLM!=nullptr);
    uint32_t n_embd = latent_dim,n_out=n_vocab;
    if(isDialect){
        n_out = tVocab();
    }
    // auto lama = hLM->GetRawModel( );  
    auto ctx = hLM->GetCTX();    
    if(nLevel==0){
        
    }else{
        const uint32_t last_dim=dims[dims.size()-1];        
        if(isLoadTokenEmbed) {
            const uint32_t n1 = isSymmetric ? n_embd : last_dim;
            if(opOut==RND_GRAD){
                _norm.w          = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n1);
                _output.w         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n1, n_out);  
            }else if(opOut==LOAD_GRAD_norm){
                _output.w         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n1, n_out);  
            }
            return;
        }
    }
    int group=hparams.Get({"model","target_group"},1);
    tok_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_out);
    _norm.w           = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    if(!isSVD){
        if(false){  // TO_DO maybe useful
            _output.w = tok_embeddings;
        }else{
            if(group==1)
                _output.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_out);  
            else{
                assert(n_embd%group==0);
                _output.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd/group, n_out);  
            }             
        }
           
    } else{
        out_u = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lo_rank, n_embd);   
        out_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lo_rank, n_out);
        out_d = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lo_rank, lo_rank); 
    }
}

std::string ConsiceDict::T2STR(TOKEN_ID tok,int flag ) { 
    string word = "";
    
    if(hLM!=nullptr){
        auto lama = hLM->lama();
        assert(lama!=nullptr);
        word = lama->T2STR(tok,flag);
    }        
    return word;   
}   

hGensor ConsiceDict::Embed2Output(struct ggml_context * ctx,hGensor t33,int flag)       { 
    hGensor  tOutput = nullptr;
    int group=hparams.Get({"model","target_group"},1);
    int n_embd=latent_dim,n_out=n_vocab,n_tokens=t33->ne[1],g_embd=n_embd/group;
    size_t nb0 = t33->nb[0],offset=0;       assert(nb0==4);  
    assert(n_embd%group==0);
    if(_output.w!=nullptr){    //1024 32000
        if(group>1){
            if(false){//expert version
                for(int i=0;i<group;i++){       
                    hGensor embd = ggml_view_2d(ctx, t33, g_embd, n_tokens, t33->nb[1], nb0*i*g_embd);  //ne0,ne1,nb1,offset
                    hGensor w = ggml_view_2d(ctx, _output.w, g_embd, n_vocab,_output.w->nb[1], nb0*i*g_embd);  //ne0,ne1,nb1,offset
                    hGensor expert = ggml_mul_mat(ctx, w, embd);        
                    // wB = _repeat(ctx,wB,expert);
                    tOutput = i==0 ? expert : ggml_add(ctx,tOutput,expert);       
                }                
            }/*else{
                assert(n_vocab%group==0);
                int ne1 = n_vocab/group;
                for(int i=0;i<group;i++){       
                    hGensor embd = ggml_view_2d(ctx, t33, g_embd, n_tokens, t33->nb[1], nb0*i*g_embd);  //ne0,ne1,nb1,offset
                    hGensor w = ggml_view_2d(ctx, _output.w, g_embd, ne1,_output.w->nb[1], offset);  //ne0,ne1,nb1,offset
                    offset += ggml_nelements(w)*nb0;
                    hGensor expert = ggml_mul_mat(ctx, w, embd);        
                    // wB = _repeat(ctx,wB,expert);                   
                    tOutput = i==0 ? expert : ggml_concat(ctx,tOutput,expert,0);       
                }    
            }*/
            hGensor embd = ggml_reshape_3d   (ctx, t33, n_embd/group, group, n_tokens);
            strcpy(embd->name, "");;        gTN(embd,"%s_group%d",t33->name,group);
            embd = ggml_permute(ctx,embd,0,2,1,3);
            assert(_output.w->ne[0]==n_embd/group);
            hGensor w = ggml_reshape_3d   (ctx, _output.w, _output.w->ne[0],n_vocab/group, group);            
            tOutput = ggml_mul_mat(ctx, w, embd);  
            tOutput = ggml_cont(ctx,ggml_permute(ctx,tOutput,0,2,1,3));
            tOutput = ggml_reshape_2d(ctx, tOutput, n_vocab, n_tokens);      //n_vocab, n_tokens
        }else
            tOutput = ggml_mul_mat(ctx, _output.w, t33);  
    }else{
        hGensor dv = ggml_mul_mat(ctx, out_d, out_v);
        hGensor svd = ggml_mul_mat(ctx, out_u, dv);  
        tOutput = ggml_mul_mat(ctx, svd, t33); 
    }
                              
    gTN(tOutput, "_output.w");  
    // assert_shape_2d(t34, n_vocab, N*n_batch);
    return tOutput;   
}

void ConsiceDict::Update_0(struct random_normal_distribution * rnd,int flag){
    const uint32_t n_embd  = hparams.n_embd;
    auto lama = hLM->GetRawModel( );  
    if(isLoadTokenEmbed) {
        bool isParam = false;
        // get tensors from llama_model (possibly mmapped)
        tok_embeddings = llama_get_model_tensor(lama, TN(LLM_TENSOR_TOKEN_EMBD));      
        if(isParam) nParams+=ggml_nelements(tok_embeddings);
        _norm.w           = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT_NORM));     
        if(isParam) nParams+=ggml_nelements(_norm.w);
        _output.w         = llama_get_model_tensor(lama, TN(LLM_TENSOR_OUTPUT));          
        if(isParam) nParams+=ggml_nelements(_output.w);
    }   else   {
        auto ctx = hLM->GetCTX();

        hLM->InitGensor(ctx,tok_embeddings, TN(LLM_TENSOR_TOKEN_EMBD), rnd);
        hLM->InitGensor(ctx,_norm.w,           TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        if(_output.w!=nullptr){
            if(_output.w!=tok_embeddings)
                hLM->InitGensor(ctx,_output.w,         TN(LLM_TENSOR_OUTPUT), rnd);
        }
        else{
            hLM->InitGensor(ctx,out_u,         "out_u", rnd);
            hLM->InitGensor(ctx,out_v,         "out_v", rnd);
            hLM->InitGensor(ctx,out_d,         "out_d", rnd);
        }
    }
    // ggml_tensor_dequant(ctx_build,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, n_vocab);
        assert_shape_1d(_norm.w,           hparams.n_embd);
        assert_shape_2d(_output.w,         hparams.n_embd, n_vocab);              
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
        _norm.w           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );       
        _output.w         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  );            
        break;
    case LOAD_GRAD_norm:    //bug@Optimizer::ggml_train
        _norm.w           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );
        assert(_norm.w->type==GGML_TYPE_F32);
        ggml_set_param(hLM->GetCTX(), _norm.w);         hLM->nParams += ggml_nelements(_norm.w);           
        hLM->Gensor2Map(_norm.w);       // hLM->tensors[_norm.w->name] = _norm.w;
        hLM->InitGensor(hLM->GetCTX(),_output.w,         TN(LLM_TENSOR_OUTPUT), rnd);
        break;
    case LOAD_GRAD:     //bug!!!
        _norm.w           = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT_NORM) );
        if(_norm.w->type!=GGML_TYPE_F32)   Gensor2float(hLM->GetCTX(),_norm.w);
        ggml_set_param(hLM->GetCTX(), _norm.w);         hLM->nParams += ggml_nelements(_norm.w);           
        hLM->Gensor2Map(_norm.w);       //hLM->tensors[_norm.w->name] = _norm.w;
        _output.w         = llama_get_model_tensor(lmodel,TN(LLM_TENSOR_OUTPUT)  ); 
        if(_output.w->type!=GGML_TYPE_F32)   {
            _output.w->data = Gensor2float(hLM->GetCTX(),_output.w);       _output.w->type = GGML_TYPE_F32;
        }
        ggml_set_param(hLM->GetCTX(), _output.w);     hLM->nParams += ggml_nelements(_output.w);           
        hLM->Gensor2Map(_output.w);       //hLM->tensors[_output.w->name] = _output.w;
        break;
    case RND_GRAD:
        hLM->InitGensor(hLM->GetCTX(),_norm.w,           TN(LLM_TENSOR_OUTPUT_NORM), rnd);
        hLM->InitGensor(hLM->GetCTX(),_output.w,         TN(LLM_TENSOR_OUTPUT), rnd);
        break;

    default:
        assert(0);
    }    
    assert(tok_embeddings!=nullptr && _norm.w!=nullptr && _output.w!=nullptr);
    // ggml_tensor_dequant(ctx_build,gensor,GGML_TYPE_F32);
    if(0){
        assert_shape_2d(tok_embeddings, hparams.n_embd, n_vocab);
        assert_shape_1d(_norm.w,           hparams.n_embd);
        assert_shape_2d(_output.w,         hparams.n_embd, n_vocab);              
    }
    int i = 0;
    for(auto map : MAEC){
        std::string name = TN(LLM_DICT_DOWN, i);    //"dict.0.down.weight"
        hLM->InitGensor(hLM->GetCTX(), map->encode,    TN(LLM_DICT_DOWN, i),     rnd); 
        if(map->decode!=nullptr)
            hLM->InitGensor(hLM->GetCTX(), map->decode,    TN(LLM_DICT_UP, i),       rnd);    
        i++;            
    }
    //ggml_set_param(hLM->GetCTX(), _norm.w);              hLM->nParams+=ggml_nelements(_norm.w);
    //_output.w is Q6k would fail @float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i)
    //ggml_set_param(hLM->GetCTX(), _output.w);            hLM->nParams+=ggml_nelements(_output.w);
    hLM->Gensor2Map(tok_embeddings); 
    // hLM->tensors[ggml_get_name(tok_embeddings)] = tok_embeddings;
    // hLM->tensors[ggml_get_name(_norm.w)] = _norm.w;
    // hLM->tensors[ggml_get_name(_output.w)] = _output.w;  
    assert(gensors.size()==0);          
}


void ConsiceDict::LoadVocab(const char*model_path,int flag)     {
    assert(std::filesystem::exists(model_path));
    string word;
    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_F16;   //LLAMA_FTYPE_ALL_F32;
    struct gguf_init_params params = {        false,NULL,    };
    struct gguf_context * vctx = gguf_init_from_file(model_path, params);

    token_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_LIST));
    if (token_idx == -1) {
        die("cannot find tokenizer vocab in model file");
    }
    n_vocab = gguf_get_arr_n(vctx, token_idx);
    int nTT = gguf_get_arr_n(vctx, token_idx);          assert(n_vocab==nTT);
    score_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_SCORES));
    if (score_idx == -1) {
        _INFO("%s cannot find tokenizer scores @%s",__func__,model_path);
        // die("cannot find tokenizer scores in model file");
    }else{
        scores = new float[nTT];
        memcpy(scores,gguf_get_arr_data(vctx, score_idx),sizeof(float)*nTT);          
    }
  

    toktype_idx = gguf_find_key(vctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
    if (toktype_idx == -1) {
        die("cannot find token type list in GGUF file");
    }
    // assert( nTT == gguf_get_arr_n(vctx, toktype_idx));
    toktypes = new int[nTT];
    memcpy(toktypes,gguf_get_arr_data(vctx, toktype_idx),sizeof(int)*nTT);    
    GGUF_GET_KEY(vctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));
    if (tokenizer_name == "llama") {
        // default special vocab
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
        // default special vocab
        special_bos_id = 11;        special_eos_id = 11;        special_unk_id = -1;
        special_sep_id = -1;        special_pad_id = -1;
    } else {
        fprintf(stderr, "%s: unknown tokenizer: '%s'", __func__, tokenizer_name.c_str());
        fprintf(stderr, "%s: using default tokenizer: 'llama'", __func__);
    }

    vocab.resize(n_vocab);
    for (uint32_t i = 0; i < n_vocab; i++) {
        vocab[i] = strdup(gguf_get_arr_str(vctx, token_idx, i));
    }
    // gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), vocab.data(), n_vocab);
    GGUF_GET_KEY(vctx, special_bos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_BOS_ID));
    GGUF_GET_KEY(vctx, special_eos_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_EOS_ID));
    GGUF_GET_KEY(vctx, special_unk_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_UNK_ID));
    GGUF_GET_KEY(vctx, special_sep_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_SEP_ID));
    GGUF_GET_KEY(vctx, special_pad_id, gguf_get_val_u32, GGUF_TYPE_UINT32, false, kv(LLM_KV_TOKENIZER_PAD_ID));

    gguf_free(vctx);
}

void CDict_CHAR::LoadVocab(const char*model_path,int flag)   {
    assert(strlen(model_path)==0 || std::filesystem::exists(model_path));
    string word;
    enum llama_ftype ftype = LLAMA_FTYPE_MOSTLY_Q8_0; 
    token_idx = -1;
    // n_vocab = len(chars);
    int nTT = n_vocab;
    score_idx = -1;
    if (score_idx == -1) {
        scores = nullptr;
    }else{
        scores = new float[nTT];        
    }
    toktype_idx = -1;
    
    toktypes = new int[nTT];
    // memcpy(toktypes,gguf_get_arr_data(vctx, toktype_idx),sizeof(int)*nTT); 
    tokenizer_name = "char_nano";
    vocab.resize(n_vocab);
    
    for (uint32_t i = 0; i < n_vocab; i++) {
        char a[2] = {(char)(i),'\0'};
        vocab[i] = strdup(a);
    }
}

void QKV_LAY::save_gguf(struct gguf_context *fctx, int flag){
    gguf_add_tensor(fctx, att_norm.w);
    gguf_add_tensor(fctx, Q.w);
    if(K.w!=nullptr)
        gguf_add_tensor(fctx, K.w);
    if(V.w!=nullptr)
        gguf_add_tensor(fctx, V.w);
    if(wo!=nullptr)
        gguf_add_tensor(fctx, wo);
    if(ffn_norm.w!=nullptr)
        gguf_add_tensor(fctx, ffn_norm.w);
    if(ffn_gate!=nullptr)
        gguf_add_tensor(fctx, ffn_gate);
    if(down.w!=nullptr)
        gguf_add_tensor(fctx, down.w);
    if(up.w!=nullptr)
        gguf_add_tensor(fctx, up.w);
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



