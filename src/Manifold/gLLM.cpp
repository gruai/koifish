/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General Language model  
 * 
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";
static const char * vendor = "gruai";     //llm_arch_from_string

// Fish* Fish::Copy(const Fish* src,int flag) { 
//     return nullptr;
// }

hFISH Fish::MakeInstance(const std::string nam_,struct CLI_params& params,int flag)   {
    hFISH fish = nullptr;
    switch(params.arch){
    case MODEL_ARCH::NLP_MAMBA:
        fish = std::make_shared<LLM_MAMBA>(nam_+"_mamba",params);
        break;
    case MODEL_ARCH::NLP_MOE:
        fish = std::make_shared<LLM_MOE>(nam_+"_moe",params);
        break;
    default:
        switch(params.nabla){
        case 1:
            fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params);
            break;
        case 2:
            // fish = std::make_shared<LLAMA_Brown>(nam_+"_brown",params);
            fish = std::make_shared<LLaMeta>(nam_,params);
            break;
        case 3:
            fish = std::make_shared<LLAMA_VAE>(nam_+"_vae",params);
            break;    
        default:
            assert(0);
        }     }
 
    hWIKI wiki = nullptr;
    if(params.tpWiki!="off")        //wiki is so heavy(ugly) that only load one instance here!
        wiki = std::make_shared<LAMA>(params);
    fish->Init( wiki );    
    fish->BuildGraph( );
    if(!fish->InitCTX())
        return nullptr;
    if(wiki!=nullptr){  //generate some sentence
        fish->gopt = GeneratOnPrompt::MakeInstance(params,wiki,fish.get(),flag);        
    }
    
    return fish;
}

hFISH Fish::MakeInstance(const std::string nam_,struct CLI_params& params,const Fish *hSrc_,int flag)   {
    hFISH fish = nullptr;
    switch(params.nabla){
    case 1:
        fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params);
        break;
    case 2:
        fish = std::make_shared<LLaMeta>(nam_,params);
        break;
    case 3:
        fish = std::make_shared<LLAMA_VAE>(nam_+"_vae",params);
        break;    
    default:
        assert(0);
    }  
    fish->isLocalInfer = flag==0x110;
    fish->graph_order = hSrc_->graph_order;
    hWIKI wiki = hSrc_->wiki;   //wiki is so heavy(ugly) that only one instance from hSrc!    
    assert(wiki!=nullptr);
    fish->Init( wiki );    
    fish->BuildGraph( );
    if(!fish->InitCTX())
        return nullptr;
    if(fish->isLocalInfer){  
    }else{
        fish->gopt = GeneratOnPrompt::MakeInstance(params,wiki,fish.get(),flag);        
    }
    
    return fish;
}

bool Fish::InitCTX(int flag) {
    auto& train_params = hparams.common;
    struct ggml_cgraph *cgraph = gb;
    if(gb==nullptr){        //  OnlyInfer
        cgraph = gf;
    }
    size_t max_work_size = ggml_graph_plan(cgraph, train_params.n_threads).work_size + GGML_OBJECT_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));

    // context for work buffer
    struct ggml_init_params ctx_work_params = {
        max_work_size, // mem_size
        NULL,          // mem_buffer    
        false,         // no_alloc
    };
    ctx_work = ggml_init(ctx_work_params);
    gb_plan = ggml_graph_plan(cgraph, train_params.n_threads);
    struct ggml_object * obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    gb_plan.work_data = (uint8_t *)ctx_work->mem_buffer + obj->offs;
    gf_plan = gb_plan;      //  ???
    return true;
}

//params!=src->params
LLaMeta::LLaMeta(const std::string& nam_,const LLaMeta* src,struct CLI_params params,int flag) : Fish(nam_,params){    
    // hDict = src->hDict;
    // tensors = src->tensors;
    // for(auto it : tensors){
    //     nParamsGGUF += ggml_nelements(it.second);
    // }

    graph_order = src->graph_order;
    //VAE's latent dim
    // if(hDict->nLevel>0)     {        
    //     hparams.n_embd = src->hparams.n_embd;
    //     // n_embd = hparams.n_embd;
    // }

    //  ugly ctx here
    // struct ggml_init_params ctx_model_params;
    // ctx_model_params.mem_size   = MostMemSize(0x0) ;
    // ctx_model_params.mem_buffer = NULL;
    // ctx_model_params.no_alloc   = true;
    // assert(ctx==nullptr);
    // ctx = ggml_init(ctx_model_params);

    // InitModel();
}

void Fish::SaveTrain(struct save_train_model * data, struct train_state * train) { 
    int64_t iter = train->opt->iter;
    _INFO("%s: iter_%ld\n", __func__, iter);
    string sBaseName = get_train_filename(data->fn_model_out.c_str(), data->pattern_fn_it.c_str(), data->fn_latest.c_str(), -1  );
    if (strlen(data->fn_checkpoint_out.c_str()) > 0) {
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model, train);
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_model_base, data->model, train);
    }
    if (strlen(data->fn_model_out.c_str()) > 0) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        vendor = "gruai";                 //llm_arch_from_string
        string sOut = "g_" + sBaseName; 
        save_gguf(sOut.c_str(),0x0);
    }
    if(1){  //only for debug
        vendor = "llama";
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
    // Fish::__repr__(suffix,prefix,flag);
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s(%s):nParams = %ld(%.6gM)",tab,name.c_str(),nParams,nParams/1.0e6);
    if(gb!=nullptr)
        sprintf(buf+strlen(buf),"\n%s  tensors=%ld gf=(%d %d)  gb=(%d %d) ",tab, gensors.size(),gf->n_nodes,gf->n_leafs,gb->n_nodes,gb->n_leafs);
    else
        sprintf(buf+strlen(buf),"\n%s  tensors=%ld gf=(%d %d) ",tab, gensors.size(),gf->n_nodes,gf->n_leafs);

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
    if(exLogits!=nullptr)
        _T_repr_(exLogits,"  ex_logits=",buf);    
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

/*
    struct llm_build_context llm(lctx, batch, cb, worst_case);  //worst_case=true    
        n_kv             (worst_case ? kv_self.size : kv_self.n),
        n_outputs        (worst_case ? n_tokens : lctx.n_outputs),
        n_outputs_enc    (worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
        kv_head          (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head),
        
    llm.init();
*/
LLM_MAMBA::LLM_MAMBA( const std::string& nam_,struct CLI_params params,int flag) : LLaMeta(nam_,params,flag)  {
    assert(arch==MODEL_ARCH::NLP_MAMBA);
    bool worst_case = true;
    // n_kv = (worst_case ? kv_self.size : kv_self.n);
    // n_outputs = (worst_case ? n_tokens : lctx.n_outputs);
    // n_outputs_enc = (worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd);
    // kv_head = (worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head);

    // isLoadTokenEmbed = true;
    // hparams.common.adam_alpha = 0.0001;     // 
}


hGensor LLM_MAMBA::build_layer_( int N,struct ggml_context *ctx_compute,hGensor inpL,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor KQ_pos,int flag) {
    int il = layer->id,nLay=layers.size();
    LAMA *lama = dynamic_cast<LAMA *>(wiki.get());      assert(lama!=nullptr);    
    
    /*  [4096,512] [4096]    */
    hGensor cur = ggml_rms_norm(ctx_compute, inpL, hparams.f_norm_rms_eps);     set_name(cur, "norm");
    hGensor t11 = ggml_repeat(ctx_compute, layer->attention_norm, cur);          
    set_name(t11, "t11");     //assert_shape_2d(t03, n_embd, N*n_batch);
    cur = ggml_mul(ctx, cur, t11);                    set_name(cur, "attn_norm");
//  cur = mamba_build_layer(ctx0,lctx,gf,cur,inpL,il,n_layer,n_tokens,kv_head,n_kv,n_outputs);
    cur = mamba_build_layer(ctx_compute, *(lama->_ctx), gf, cur,inpL, il, nLay,512);
    return cur;
}

//n_embd_head, n_head_kv
hGensor  LLaMeta::build_layer_( int N,struct ggml_context *ctx_compute,hGensor cur,std::shared_ptr<LLaMeta::lama_layer> layer,hGensor  KQ_pos,/*hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down,*/ int flag) {
    auto train_params = hparams.common;
    int n_vocab = hDict->n_vocab,n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd,n_head = hparams.n_head,n_ff = hparams.n_ff;
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;  
    const float kv_scale = 1.0f/sqrtf(float(hparams.n_embd)/hparams.n_head);
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



std::string LLaMeta::Name()     {   
    return "LAMA";  
}



//REF: ggml_compute_forward_dup_bytes
void LLaMeta::CopyWeight(const Fish* src,int flag) {
    auto gsrc = src->gf;
    size_t nx=0,nz,nT=0;
    
    for (int i = 0; i < gsrc->n_nodes; ++i) {
        hGensor t0 = gsrc->nodes[i],t1;
        if(strcmp(t0->name,"output.weight")==0){
            int j = 0;
        }
        const size_t type_size = ggml_type_size(t0->type);
        if (t0->flags & GGML_TENSOR_FLAG_PARAM) {
            t1 = GetGensor(t0->name);
            assert(t1!=nullptr && t0->type==t1->type);
            nz = ggml_nelements(t0);
            assert(nz==ggml_nelements(t1));
            memcpy(t1->data,t0->data,type_size*nz);
            nx += nz;       nT++;
        }
    }      
    assert(nx==src->nParams);   
}

/*
    would affect training process?
    hOPT->Compute would merge wiki->preLogits! (REF: DataLoader::update_batch)
*/
bool LLaMeta::LocalFeeling(std::vector<llama_token>&samp_tokens,vector<float>& result)  const {
    assert(target_probs->type == GGML_TYPE_F32);
    float *fLoss = (float*)(loss->data);    
    llama_token token;
    ggml_set_i32_nd(tokens_input, 0, 0, 0, 0, hDict->bos);
    size_t i,N=samp_tokens.size(),n_context=tokens_input->ne[0],nSampInBatch=tokens_input->ne[1];
    if(N>n_context)
        return false;
    assert(N>=0 && N<=n_context);     
    // assert(nSamp==1 && nSamp<=nSampInBatch);
    for(i=0;i<N;i++)  {
        token = samp_tokens[i];
        ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) 0, 0, 0, token);
    }
   
    if(isLocalInfer)
        ggml_graph_compute(gf, (ggml_cplan*)(&gf_plan));   
    else{
        ggml_graph_compute(gb, (ggml_cplan*)(&gb_plan));           
    }        

    assert(preLogits->type==GGML_TYPE_F32);
    assert(preLogits->ne[1]==n_context);
    size_t nz = ggml_nelements(preLogits),nTokenInWiki=preLogits->ne[0]; //preLogits->nb[0];

    //Causal Language Modeling
    size_t off = (N-1)*nTokenInWiki;      //(n_context-1)*nToken;
    float *out = (float*)(preLogits->data)+off;
    if(result.size()!=nTokenInWiki){
        result.clear( );     result.resize(nTokenInWiki);
    }
    memcpy(result.data(),out,sizeof(float)*nTokenInWiki);
    float sum=0;
    for(auto f : result)    sum+=f;
    return true;
}

void LLM_MAMBA::BuildTarget( struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,hGensor cur,hGensor _tNorm,hGensor _tOutput,hGensor KQ_pos, int flag)  {
    int n_vocab = hDict->n_vocab,n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    auto train_params = hparams.common;
    preLogits = ggml_reshape_3d(ctx, cur, n_vocab, n_ctx, n_batch);             set_name(preLogits, "preLogits");     
    assert_shape_3d(preLogits, n_vocab, n_ctx, n_batch);;
    loss = ggml_cross_entropy_loss(ctx, preLogits, target_probs);                    
    set_name(loss, "loss");     assert_shape_1d(loss, 1);
    ggml_build_forward_expand(gf, loss);
    if (train_params.use_checkpointing) {
        if(gb!=nullptr) 
            ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        if(gb!=nullptr){
            ggml_graph_cpy(gf, gb);
            ggml_build_backward_expand(ctx, gf, gb, true);            
        }
    }    
}

void LLaMeta::BuildTarget( struct ggml_context * ctx,ggml_gallocr_t& alloc,bool m_only,hGensor cur,hGensor _tNorm,hGensor _tOutput,hGensor KQ_pos, int flag)  {
    int n_vocab = hDict->n_vocab,n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    auto train_params = hparams.common;
    train_params.use_checkpointing = false;     // CYS_0826
    const int N = train_params.n_ctx, n_past = 0;
    const float rms_norm_eps = hparams.f_norm_rms_eps;
    hGensor  t32 = nullptr;
    hGensor  t31   = ggml_rms_norm          (ctx, cur, rms_norm_eps);                    set_name(t31, "t31");     
    assert_shape_2d(t31, hparams.n_embd, N*train_params.n_batch);
    
    if(hDict->nLevel>0){
        t31 = hDict->DEC(ctx,t31);      //t31 = ggml_mul_mat(ctx, hDict->decoder, t31 );  
        set_name(t31, "embed_decoder");
        t32 = ggml_repeat            (ctx, _tNorm, t31); 
        n_embd = t32->ne[0];
    }   else
        t32   = ggml_repeat            (ctx, _tNorm, t31);                            
    set_name(t32, "t32");     assert_shape_2d(t32, n_embd, N*n_batch);
    hGensor  t33   = ggml_mul               (ctx, t32, t31);                             set_name(t33, "t33");     
    assert_shape_2d(t33, n_embd, N*n_batch);
    hGensor  t34   = ggml_mul_mat           (ctx, _tOutput, t33);                          set_name(t34, "t34");     
    assert_shape_2d(t34, n_vocab, N*n_batch);
    hGensor  t35   = ggml_reshape_3d        (ctx, t34, n_vocab, N, n_batch);             set_name(t35, "t35");     
    assert_shape_3d(t35, n_vocab, N, n_batch);
    // no,no,no! 1) Softmax layers can be difficult to train since the gradients can vanish or explode  2) CrossEntropyLoss assumes logits on the input.
    //  t35 = ggml_soft_max_inplace(ctx,t35); 
    if(exLogits!=nullptr)    {   // preLogits = t35;
        t35 = ggml_add(ctx,t35,exLogits);
        //  WIKI::_LOGITS_SCALE
        // t35 = ggml_relu(ctx,t35);       //converge very slow, so strange!
        // t35 = ggml_mul(ctx,t35,exLogits);
    }
    hGensor  t36   = ggml_cross_entropy_loss(ctx, t35, target_probs);                    set_name(t36, "t36");     assert_shape_1d(t36, 1);
    if(isTrain())
        assert(t36->grad!=nullptr);
    if(hDict->nLevel>0){
        n_embd = hparams.n_embd;
    }
    if (train_params.use_checkpointing) {
        checkpoints.push_back(t31);            checkpoints.push_back(t32);            checkpoints.push_back(t33);
        checkpoints.push_back(t34);            checkpoints.push_back(t35);            checkpoints.push_back(t36);
    }

    ggml_build_forward_expand(gf, t36);

    if (train_params.use_checkpointing) {
        if(gb!=nullptr) 
            ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
    } else {
        if(gb!=nullptr){
            ggml_graph_cpy(gf, gb);
            ggml_build_backward_expand(ctx, gf, gb, true);            
        }
    }

    GGML_ASSERT(alloc != NULL);
    if(isLocalInfer){  
        GGML_ASSERT(alloc != NULL);
        assert(gb==nullptr);
        ggml_gallocr_alloc_graph(alloc, gf); 
        int * data = (int *) KQ_pos->data;
        for (int i = 0; i < N; ++i) {
            data[i] = n_past + i;
        }
    } else { // gb!=nullptr
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
        // allocating checkpoints in one block to reduce memory fragmentation they will be freed in reverse order    

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
    }
    preLogits = t35;
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

std::string LAMA::T2STR( int32_t tok,int flag )                    {   
    assert(_ctx!=nullptr);
    std::string str = llama_token_to_piece(_ctx, tok);
    return str;  
}

/**
 *  1.  llama_set_inputs(lctx, u_batch);        llama_graph_compute(lctx, gf, n_threads);
 *  2.  Extract logits and embeddings
 */
bool LAMA::Decode(std::vector<llama_token>&embd_inp,int start,int n_past,bool out_all) {
    assert(embd_inp.size()<=nCTX());
    llama_token eos = llama_token_eos(lmodel),id;
    int i=0,n_consumed=start;
    std::vector<llama_token> embd;
    while ((int) embd_inp.size() > n_consumed) {
        id = embd_inp[n_consumed];
        // const std::string token_str = llama_token_to_piece(_ctx, id);
        // _INFO("%s",token_str.c_str());
        embd.push_back(embd_inp[n_consumed]);
        ++n_consumed;       
    }
    int n_eval = (int) embd.size();
    struct llama_batch batch = {
        /*n_tokens       =*/ n_eval,
        /*tokens         =*/ embd.data(),
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
        /*all_pos_0      =*/ n_past,
        /*all_pos_1      =*/ 1,
        /*all_seq_id     =*/ 0,
    };//llama_batch_get_one(&embd[0], n_eval, n_past, 0);
    llama_ctx_set_(_ctx,&out_all,11);       
    llama_decode(_ctx,batch );        n_past+=n_eval;
    if(out_all){
        nOutToken = embd_inp.size();
    }else
        nOutToken = 1;
    
    // llama_ctx_get_(_ctx,(void**)(&logits_out),10);   
    // const float *logits = GetLogits(n_vocab,nOutToken,0);   

    return true;
}

const float *LAMA::GetLogits(int n_vocab,int n_ctx,int idx)   {
    /*if(cparams.logits_all){   llama_ctx_set_(_ctx,&out_all,11);       
        // assert(idx==0);
    }*/
/*
    _ctx->logits_all is updated by llama_ctx_set_(_ctx,&out_all,11)
    has no relation with cparams.logits_all !!!
*/
    if(idx==-1){ 
        // assert(_ctx->logits_all==false);
    }    
    const float *_logits = llama_get_logits_ith(_ctx, idx);     //return ctx->logits + j*ctx->model.hparams.n_vocab;
    if(0)   {        
        llama_ctx_get_(_ctx,(void**)(&_logits),10);  //ctx->logits;
    } 
    if(1){
        assert(n_ctx<=cparams.n_ctx || n_ctx==nOutToken);
        double nrm = NRM_2(_logits,n_vocab*n_ctx);
        if(nrm==0.0)    {
            _INFO("\n\n !!! %s |preLogits| is zero,so crazy! N=%dx%d \n\n",__func__,n_vocab,n_ctx);
        }     
        if(isnan(nrm) || isinf(nrm) )    {
            _INFO("\n\n !!! %s |preLogits| is NAN! N=%dx%d \n\n",__func__,nrm);
        }
    }
 
    return _logits;
}

void LAMA::Answer(std::vector<llama_token>&embd_inp,int flag) {
    // Answer_0(embd_inp,flag);     return;
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
        _INFO("%s ",token_str.c_str());
        embd.push_back(embd_inp[n_consumed]);
        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        llama_sampling_accept(ctx_sampling, _ctx, embd_inp[n_consumed], false);
        ++n_consumed;       
    }
    fflush(stdout);
    
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
    // llama_ctx_get_(_ctx,(void**)(&logits_out),10);  

    id = llama_sampling_sample(ctx_sampling, _ctx, nullptr);      //539
    llama_sampling_accept(ctx_sampling, _ctx, id, true);
    const std::string token_str = llama_token_to_piece(_ctx, id);
    _INFO("%s => \"%s\"",__func__,token_str.c_str());
}

LAMA::LAMA(CLI_params& hparams) {
    llama_mparams.n_gpu_layers = hparams.common.n_gpu_layers;
    llama_mparams.vocab_only = hparams.train=="scratch";     //need lmodel to tokenize training samples    

    teach = hparams.tpWiki=="logits" ? _LOGITS : hparams.tpWiki=="target" ? _TARGET : _OFF;
    if(llama_mparams.vocab_only || hparams.wiki_actor=="OnlyTokenizer"){
        isOnlyTokenizer = true;
        _INFO("%s: OnlyTokenizer.\n", __func__ );
        assert(teach==WIKI::_OFF);
    }        
    _INFO("%s: model base = '%s' nEmbd=%d wiki=%s\n", __func__, hparams.fn_model_base.c_str(),hparams.n_embd,hparams.tpWiki.c_str());
    lmodel = llama_load_model_from_file(hparams.fn_model_base.c_str(), llama_mparams);                
    n_vocab = llama_n_vocab(lmodel);
    // true or false?   If use logits distillation, must set it True!
    cparams.logits_all = true;
  
    _ctx = llama_new_context_with_model(lmodel, cparams);  
    // const auto & cp_ = _ctx->cparams;

    bos = llama_token_bos(llama_get_model(_ctx));
    eos = llama_token_eos(llama_get_model(_ctx));
    std::string sBos = llama_token_to_piece(_ctx, bos),sEos = llama_token_to_piece(_ctx, eos);
}


void LAMA::Reset(int flag)    {   
    // cparams.seed = 42;
    // llama_ctx_set_(_ctx,nullptr,42);
    cparams.n_threads = 20;
    cparams.n_threads_batch = 20;   
    if(1){
        _ctx = llama_ctx_reset_(_ctx,lmodel, cparams);
    }else{
        llama_free(_ctx);
        _ctx = llama_new_context_with_model(lmodel, cparams);  //need more work to simplify        
    }
}   

double WIKI::InductLogits(int nSampInBatch,std::vector<int32_t>& tok_ids,hGensor exLogits,hGensor target_probs,int flag)  {
    Reset();         //Timing bottleneck!!! for the crazy design of llama.cpp
    Decode(tok_ids,0,0x0,true);    
    const float *all_logits = GetLogits(n_vocab,tok_ids.size(),0),*logit;  
    size_t k,ld0=target_probs->nb[0],ld1=target_probs->nb[1],ld2=target_probs->nb[2],ld3=target_probs->nb[3],j,i;  
    int n_ctx = target_probs->ne[1];  
    double a1,a2,nrm=0;    
    float *p=teach == WIKI::_TARGET ? new float[n_vocab]:nullptr,*target ;  
    if(flag<0){    //only for old version
        assert(exLogits!=nullptr);
        target = (float*)exLogits->data+nSampInBatch*n_ctx*n_vocab;        
        nrm =  NRM_2(all_logits,n_ctx*n_vocab)/n_vocab;   
        memcpy((void*)target,(void*)all_logits,sizeof(float)*n_ctx*n_vocab);       //memcpy(g->data+off,(void*)(logits),ld2); 
    }else{    
        for (k=0; k<nSampInBatch; ++k) {        
            const float *from=all_logits+k*n_vocab;
            a1=NRM_2((float*)(from),n_ctx*n_vocab);          nrm=max(nrm,a1/n_vocab);     
            if(teach == WIKI::_TARGET){              
                assert(exLogits==nullptr);             
                for(j=0;j<n_ctx;j++){
                    logit = from+j*n_vocab;
                    target = (float*)target_probs->data+(k*n_ctx+j)*n_vocab;
                    /*SOFT_MAX_minus(n_vocab,target,logit);*/
                    SOFT_MAX(n_vocab,p,logit);
                    for(a1=0,a2=0,i=0;i<n_vocab;i++){
                        a1 += target[i];            a2 += p[i];
                        target[i] -= p[i];
                    }
                    // SOFT_MAX(n_vocab,p,target);     //  !!!No converge!!!   @home/cys/rnd/lic/log/eval/08_21_wiki_target_no_converge.info  
                    memcpy(target,p,sizeof(float)*n_vocab);
                    // todo - cys 20240821: MSE loss 
                }
            }else{
                assert(exLogits!=nullptr);
                // void *from=(void*)(all_logits)+k*ld1,*to=exLogits->data+k*ld2;
                target = (float*)exLogits->data+k*n_ctx*n_vocab;               
                memcpy((void*)target,(void*)from,sizeof(float)*n_ctx*n_vocab);       //memcpy(g->data+off,(void*)(logits),ld2);   
            }
        }    
    }
    delete[] p;
    return nrm;
}


void LLaMeta::InitEntryTensors(int flag) {
    auto train_params = hparams.common;
    int n_ctx = train_params.n_ctx,n_vocab = hDict->n_vocab,n_batch = train_params.n_batch;
    assert(n_ctx>0 && n_batch>0);
    struct ggml_init_params ctx_input_params = {// mem_size mem_buffer no_alloc
        ggml_tensor_overhead() * 2, NULL,true,                        
    };
    ctx_input = ggml_init(ctx_input_params);
    tokens_input  = ggml_new_tensor_2d(ctx_input, GGML_TYPE_I32, n_ctx, n_batch);
    target_probs  = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_ctx, n_batch);
    ggml_backend_buffer_t input_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx_input, ggml_backend_cpu_buffer_type());
    size_t max_input_size = ggml_backend_buffer_get_size(input_data);
    _INFO("%s: input_size(%d,%d) = %zu bytes (%.1f MB)\n", __func__, n_ctx, n_batch, max_input_size, (float) max_input_size / (1024.0f*1024.0f));
    // ggml_free(ctx_input);
}

bool Fish::OnTrainStep(struct train_opt_callback_data *data,DataLoader&loader, int accum_step, float *sched, int flag)    {
    LossCurve(0x0);
    assert(0);
    return false;
}

bool LLaMeta::lama_layer::InitFFN(const CLI_params&hparams, ggml_context *ctx, FFN_TYPE tpFFN, int flag)  {
    const uint32_t n_embd = hparams.n_embd, n_ctx = hparams.n_ctx(), n_ff = hparams.n_ff, n_batch = hparams.n_batch();  
    const uint32_t n_expert = hparams.n_expert;
    switch(tpFFN){
    case VAR_LAST:
    case SWIGLU:
        if(hparams.ZMUV_ratio>0)
            ffn_norm = nullptr;  
        else
            ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff);
        ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,   n_ff, n_embd);
        ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_ff); 
        if(tpFFN==VAR_LAST && isLast){/*i==n_layer-1*/
            eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_batch*n_ctx);   
        }
        break;
    case ONLY_LNormal:
    case ONLY_RMSNormal:
        ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        break;
    case VAR_0:
        eps = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_batch*n_ctx);   
        break;  
    case SMOE:{// MoE branch
        assert(n_expert>0 && hparams.n_expert_used>0) ; 
        ffn_gate_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd,   n_expert);
        // ffn_gate_inp = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});        
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / hparams.n_expert_used;
        ffn_gate_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp, n_expert);
        // ffn_gate_exps = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        ffn_down_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff_exp,   n_embd, n_expert); 
        // ffn_down_exps = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
        ffn_up_exps = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff_exp, n_expert); 
        // ffn_up_exps   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

        // Shared expert branch
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;
        ffn_gate_inp_shexp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); 
        // ffn_gate_inp_shexp = ml.create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd});
        ffn_gate_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff_shexp);
        // ffn_gate_shexp = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp});
        ffn_down_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff_shexp, n_embd);
        // ffn_down_shexp = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd});
        ffn_up_shexp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ff_shexp);
        // ffn_up_shexp   = ml.create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp});
    }
        break;
    default:
        TO_DO;
    } 
    return true;  
}