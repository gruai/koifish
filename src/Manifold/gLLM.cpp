/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  General AutoRegressive Language model  
 * 
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"
#include "Cache.hpp"
#include "Fish.hpp"
#include "../g_stddef.hpp"

static const char * LLM_KV_TRAINING_TYPE_TRAIN_MODEL     = "train_model";
static const char * LLM_KV_TRAINING_TYPE                 = "training.type";


shared_ptr<EDGE_DEVICES> EDGE_DEVICES::GetInstance(const CLI_params&hparams, int flag)       {
    static shared_ptr<EDGE_DEVICES> hEDS = std::make_shared<EDGE_DEVICES>(hparams,flag);
    return hEDS;
}

bool NLP_AutoRegressive::Init(const vector<hWIKI>& wikis_,int flag)     {
    auto train_params = hparams.common;
    if (train_params.seed == LLAMA_DEFAULT_SEED) {
        train_params.seed = time(NULL); 
    }    
    wikis = wikis_;  
    hOPT = std::make_shared<OPT_Adam>(this,hparams,flag);
    if(!InitDictTokenset()) //hDict
        return false;

    if(wikis.size()==0){
        _INFO("====== NO WIKI !!! ======\n");       //return false;
    }else{
        teach = wikis[0]->teach;  
    } 
    hCache = std::make_shared<KVCache>(this,0,0,0,0);
    
    hDistler = nullptr; //hparams.sigma=="" ? nullptr : std::make_shared<Distillation>(this,hparams,0x0);     //ADD SIGMA             

    // assert(ctx_build==nullptr);
    // ctx_build = InitCTX(MostMemSize(0x0));
    hEDS = EDGE_DEVICES::GetInstance(hparams);
    hOPT->hEDS = hEDS;
    
    InitModel(flag);
    hparams.Dump();         
    // if(hparams.is({"wiki","actor"},"copy")){
    //     bool bRet = wikis[0]->CopyGensors(this);      
    //     assert(bRet);
    // }

    // save_data.Init(hparams,GetRawModel());
    if(hOPT!=nullptr)   {
        hOPT->Dump(1);
    }   
    return true;         
}

NLP_AutoRegressive::NLP_AutoRegressive( const std::string& nam_, struct CLI_params params,ROLE_TYPE role_,int flag) : Fish(nam_,params,role_) {
    if(!hparams.is({"wiki","actor"},"copy"))  {
        
    }
    hparams.ffn_use_gate = true;
    int d = hparams.Get({"model","attention","dQKV"},4,false);
    isAttOnBC = d==3;       //d=4 much faster,nearly same

    string sT = params.KV({"model","attention","type"},"QKV",false);
    tpATT = sT=="brown" ? ATTENTION_TYPE::BROWN : sT=="off" ? ATTENTION_TYPE::OFF : ATTENTION_TYPE::QKV;
    tpFFN = (FFN_TYPE)(jKV(params.jConfig,{"model","ffn","type"},5,false));    
}   
//params!=src->params
NLP_AutoRegressive::NLP_AutoRegressive(const std::string& nam_,const NLP_AutoRegressive* src,struct CLI_params params,int flag) : Fish(nam_,params){    
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

string NLP_AutoRegressive::__repr__( string& suffix,string& prefix,int flag)         {
    // Fish::__repr__(suffix,prefix,flag);
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    auto gb=GetBackRaw(), gf=hForwTG->raw();;
    sprintf(buf+strlen(buf),"\n%s(%s):nParams = %ld(%.6gM)",tab,name.c_str(),nParams,nParams/1.0e6);
    if(gb!=nullptr)
        sprintf(buf+strlen(buf),"\n%s  tensors=%ld %s=(%d %d)  %s=(%d %d) ffn_use_gate=%d",tab, gensors.size(),
            hForwTG->name.c_str(),gf->n_nodes,gf->n_leafs,hBackTG->name.c_str(),gb->n_nodes,gb->n_leafs,
            hparams.ffn_use_gate);
    else
        sprintf(buf+strlen(buf),"\n%s  tensors=%ld gf=(%d %d) ",tab, gensors.size(),gf->n_nodes,gf->n_leafs);

    string s="\n",p=prefix+"\t";
    int i;
    _T_repr_(KQ_pos,"\n\tKQ_pos: ",buf);  
    _T_repr_(pos_embd,"\tpos_embd: ",buf); 
    _T_repr_(KQ_mask,"\tKQ_mask: ",buf);  
    sprintf(buf+strlen(buf),"%s",hDict->__repr__(s,p,0x0).c_str());
    bool isJModel =  !hparams.jModel.empty();
    if(isJModel){
        for(auto nn : neurons){
            if(nn->isGang())   continue;
            for(p=prefix, i=0;i<nn->level-1;i++)    p+="\t";
            // if(nn->level==1)
                sprintf(buf+strlen(buf),"%s\n",nn->__repr__(s,p,0x0).c_str());     
        }
    }else{
        
        int nLayer = layers.size();
        if(nLayer>0)    {
            auto layer = layers[0];
            sprintf(buf+strlen(buf),"%s  [%s] x %d\n",tab,layer->name.c_str(),nLayer);
            sprintf(buf+strlen(buf),"%s",layer->__repr__(s,p,0x0).c_str());     
            if(nLayer>1){
                sprintf(buf+strlen(buf),"%s  ......\n",tab);
                sprintf(buf+strlen(buf),"%s",layers[layers.size()-1]->__repr__(s,p,0x0).c_str());         
            }
        }
    }
    _T_repr_(target_probs,"  target_probs=",buf);  
    for(auto wiki : wikis){
        if(wiki->exLogits!=nullptr){
            string a = "   ex_logits@"+wiki->title+"=";
            _T_repr_(wiki->exLogits,a.c_str(),buf);   
        }
        if(wiki->t2t!=nullptr){
            string a = "   t2t@"+wiki->title+"=";
            _T_repr_(wiki->t2t,a.c_str(),buf);   
        }            
    }
     
    if(mom.embed2w!=nullptr)
        _T_repr_(mom.embed2w,"  gate=",buf); 
    _T_repr_(loss,"  loss=",buf);   

    sprintf(buf+strlen(buf),"%s",suffix.c_str()); 
    _INFO("%s",buf); 
    return buf;
}

string Ganglia::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s %s",tab,name.c_str());
    string s,p;
    for(auto child : ns){
        sprintf(buf+strlen(buf),"%s ",child->__repr__(s,p,0).c_str());
    }
    // sprintf(buf+strlen(buf),"\n");
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

string GeNeuron::__repr__( string& suffix,string& prefix,int flag)   {    
    char buf[5012]="\0";
    // const char*tab=prefix.c_str();
    // sprintf(buf+strlen(buf),"\n%s %s",tab,name.c_str());
    // if(flag>0)
    //     _INFO("%s",buf); 
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


string QKV_LAY::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    _T_repr_(att_norm.w,tab,buf);           _T_repr_(att_norm.b,tab,buf);
    _T_repr_(Q.w,tab,buf);                  _T_repr_(Q.b,tab,buf);       
    _T_repr_(K.w,tab,buf);                  _T_repr_(K.b,tab,buf);   
    _T_repr_(V.w,tab,buf);                  _T_repr_(V.b,tab,buf); 
    _T_repr_(proj.w,tab,buf);               _T_repr_(proj.b,tab,buf);   
    // _T_repr_(wk,tab,buf);       _T_repr_(wv,tab,buf);   
    _T_repr_(wo,tab,buf);   
    _T_repr_(ffn_norm.w,tab,buf);           _T_repr_(ffn_norm.b,tab,buf); 
    _T_repr_(ffn_gate,tab,buf);     
    _T_repr_(up.w,tab,buf);     _T_repr_(up.b,tab,buf);
    _T_repr_(down.w,tab,buf);   _T_repr_(down.b,tab,buf);     
    _T_repr_(eps,tab,buf);
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}

void NLP_AutoRegressive::build_inp_KQ_(struct ggml_context *ctx,bool isMask,bool causal) {
    char nam_[128];
    bool isFlash = hparams.isFlashAtten();     
    const uint32_t pad = isFlash ? 256u : 32u,cell_max=0;     //llama_kv_cache_cell_max(*cache)
    // auto cache = GetKVCache();
    // cache->n = std::min(cache->size, std::max(pad, GGML_PAD(cell_max, pad)));
    int n_kv=hCache==nullptr ? 512 : hCache->n_kv();
    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_tokens = n_batch*n_ctx,n_past=0;
    // const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        // KQ_pos - contains the positions
    KQ_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hparams.n_ctx());
    gTN(KQ_pos, "inp_pos");  
    ggml_set_input(KQ_pos);    
    int * data = (int *) KQ_pos->data;
    // @BuildComputeGraph After ggml_gallocr_alloc_graph(alloc, gb)!
    // for (int i = 0; i < n_tokens; ++i) {
    //     data[i] = n_past + i;
    // }
    if(isMask && 0){        //  nearly same if mask==nullptr
        auto dt = GGML_TYPE_F32;     
        auto pad = GGML_PAD(n_ctx, GGML_KQ_MASK_PAD);   //  (((x) + (n) - 1) & ~((n) - 1))
        // KQ_mask = causal
        //     ? ggml_new_tensor_2d(ctx, dt, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        //     : ggml_new_tensor_2d(ctx, dt, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        KQ_mask = ggml_new_tensor_2d(ctx, dt, n_ctx,pad );
        gTN(KQ_mask, "KQ_mask");      
        ggml_set_input(KQ_mask);        //  flags |= GGML_TENSOR_FLAG_INPUT;
        float*mask = (float*)(KQ_mask->data);
        KQ_mask = isFlash ? ggml_cast(ctx, KQ_mask, GGML_TYPE_F16) : KQ_mask;    
        // set data like llama_set_inputs    
    }
    return ;
}

hGensor SLP::UpdateGensor(int flag)  {
    hGensor h = w==nullptr ? nullptr : hFish->GetGensor (w->name);
    return h;
}

//n_embd_head, n_head_kv
hGensor NLP_AutoRegressive::build_layer_( int N,struct ggml_context *ctx_build,hGensor inpL,std::shared_ptr<QKV_LAY> layer,hGensor  KQ_pos,/*hGensor cur, hGensor wq, hGensor wk, hGensor wv, hGensor wo,
    hGensor attention_norm,hGensor KQ_pos,hGensor ffn_norm,hGensor ffn_up,hGensor ffn_gate,hGensor ffn_down,*/ int flag) {
    char nam_[128];
    auto train_params = hparams.common;
    int n_vocab = tVocab(),n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd,n_head = hparams.n_head(),n_ff = hparams.n_ff();
    int n_outputs = n_ctx;  //  ????
    const float f_norm_rms_eps  = hparams.f_norm_rms_eps;
    const float rope_freq_base  = hparams.rope_freq_base;
    const float rope_freq_scale = hparams.rope_freq_scale;  
    const float kv_scale = 1.0f/sqrtf(float(hparams.n_embd)/hparams.n_head());
    const int n_past = 0, n_head_kv=hparams.n_head_kv(),n_embd_head = hparams.n_embd_head();    
    hGensor wq = layer->Q.UpdateGensor();   //  UpdateGensor (layer->Q.w->name);                     
    hGensor wk = layer->K.UpdateGensor();   //layer->K.w==nullptr ? nullptr : UpdateGensor (layer->K.w->name);
    hGensor wv = layer->V.UpdateGensor();   //layer->V.w==nullptr ? nullptr : UpdateGensor (layer->V.w->name);
    hGensor ffn_norm = layer->ffn_norm.w==nullptr ? nullptr : UpdateGensor (layer->ffn_norm.w->name); 
    hGensor ffn_up = nullptr,ffn_gate=nullptr,ffn_down=nullptr;
    if(layer->up.w!=nullptr){                
        ffn_up = UpdateGensor (layer->up.w->name);
        if(layer->ffn_gate!=nullptr)    ffn_gate = UpdateGensor (layer->ffn_gate->name);
        ffn_down = UpdateGensor (layer->down.w->name);                
    }
    
    hGensor  ffn_inp = inpL;
    if(!layer->Q.Empty()){      //  if 0,   only for debug
        hGensor attention_norm = UpdateGensor (layer->att_norm.w->name);  
    //  rms_norm:   Root Mean Square Layer Normalization
        hGensor  t02 = ggml_rms_norm     (ctx_build, inpL, f_norm_rms_eps);                    
        sprintf(nam_,"norm_L-%d",layer->id),   gTN(t02, nam_);     assert_shape_2d(t02, n_embd, N*n_batch);
        hGensor  t03 = attention_norm;
        if(isTrain()){
            t03 = ggml_repeat(ctx_build, attention_norm, t02);              gTN(t03, "attnorm.w.repeat-%d",layer->id);     assert_shape_2d(t03, n_embd, N*n_batch);
        }
        hGensor  tBase = ggml_mul          (ctx_build, t02, t03);                               
        sprintf(nam_,"attn_norm-%d",layer->id),   gTN(tBase, nam_);     assert_shape_2d(tBase, n_embd, N*n_batch);
        if(isBias){

        }        
        hGensor wo = UpdateGensor (layer->wo->name);
        hBrownMotion hBrown = CreateBrownMotion(wq, wk, wv,layer);                       
        hGensor t16 = hBrown->Build(ctx_build , tBase,  KQ_pos);          // [128,17,6,1]   
        hGensor  t20 = ggml_mul_mat      (ctx_build, wo, t16);                          
        sprintf(nam_,"kqv_out-%d",layer->id),           gTN(t20, nam_);     assert_shape_2d(t20, n_embd, N*n_batch);
        if (layer->isLast && 0) {          //would crash if not set inp_out_ids       
            // skip computing output for unused tokens
            struct ggml_tensor * inp_out_ids = ggml_new_tensor_1d(ctx_build, GGML_TYPE_I32, n_outputs);   //n_outputs=(worst_case ? n_tokens : lctx.n_outputs),
            sprintf(nam_,"inp_out_ids"),     gTN(inp_out_ids, nam_);     // cb(lctx.inp_out_ids, "inp_out_ids", -1);
            ggml_set_input(inp_out_ids);
            t20   = ggml_get_rows(ctx_build,   t20, inp_out_ids);
            inpL = ggml_get_rows(ctx_build, inpL, inp_out_ids);
        }
        ffn_inp = ggml_add          (ctx_build, t20, inpL);  
    }                             
    sprintf(nam_,"ffn_inp-%d",layer->id),           gTN(ffn_inp, nam_);     assert_shape_2d(ffn_inp, n_embd, N*n_batch);
    hGensor  ffn = nullptr;
    switch(tpFFN)   {
    case VAR_LAST:
    case SWIGLU:    {
        hGensor  t22 = ggml_rms_norm     (ctx_build, ffn_inp, f_norm_rms_eps);                    
        sprintf(nam_,"norm_ffn-%d",layer->id),    gTN(t22, nam_);     assert_shape_2d(t22, n_embd, N*n_batch);
        ffn = t22;
        if(ffn_norm!=nullptr)       {
            hGensor  t23 = ffn_norm; //ggml_repeat       (ctx_build, ffn_norm, t22);    
             if(isTrain()){
                t23 = ggml_repeat       (ctx_build, ffn_norm, t22);    
                gTN(t23, "ffn_norm.repeat-%d",layer->id);     assert_shape_2d(t23, n_embd, N*n_batch);
            }      
            hGensor  t24 = ggml_mul          (ctx_build, t22, t23);                               
            sprintf(nam_,"ffn_norm-%d",layer->id),    gTN(t24, nam_);     assert_shape_2d(t24, n_embd, N*n_batch); 
            ffn = t24;                 
        }
          
        if(ffn_up!=nullptr){
            // hGensor  t22 = ggml_rms_norm     (ctx_build, t21, f_norm_rms_eps);                    gTN(t22, "t22");     assert_shape_2d(t22, n_embd, N*n_batch);
            // hGensor  t23 = ggml_repeat       (ctx_build, ffn_norm, t22);                    gTN(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            // hGensor  t24 = ggml_mul          (ctx_build, t23, t22);                               gTN(t24, "t24");     assert_shape_2d(t24, n_embd, N*n_batch);
            hGensor  t24 = ffn;
            hGensor  t25 = ggml_mul_mat      (ctx_build, ffn_up, t24);                      
            sprintf(nam_,"ffn_up-%d",layer->id),gTN(t25, nam_);     assert_shape_2d(t25, n_ff, N*n_batch);
            hGensor  t28;
            if(ffn_gate==nullptr){
                t28 = ggml_silu         (ctx_build, t25);      
                sprintf(nam_,"ffn_silu-%d",layer->id),         gTN(t28, nam_);      assert_shape_2d(t28, n_ff, N*n_batch);
            }else{
                hGensor t26 = ggml_mul_mat      (ctx_build, ffn_gate, t24);                    
                sprintf(nam_,"ffn_gate-%d",layer->id),  gTN(t26, nam_);     assert_shape_2d(t26, n_ff, N*n_batch); 
                hGensor t27 = ggml_silu         (ctx_build, t26);                 
                sprintf(nam_,"ffn_silu-%d",layer->id),         gTN(t27, nam_);      assert_shape_2d(t27, n_ff, N*n_batch);
                t28 = ggml_mul          (ctx_build, t27, t25);      
                sprintf(nam_,"ffn_gate_par-%d",layer->id),         gTN(t28, nam_);     assert_shape_2d(t28, n_ff, N*n_batch);      
            }
            hGensor  t29 = ggml_mul_mat      (ctx_build, ffn_down, t28);                    
            sprintf(nam_,"ffn_out-%d",layer->id),  gTN(t29, nam_);   assert_shape_2d(t29, n_embd, N*n_batch);
            hGensor  t30 = ggml_add          (ctx_build, t29, ffn_inp);   ffn = t30;
        }
        if(layer->eps!=nullptr){
            // hGensor  t300 = ffn!=nullptr ? ggml_rms_norm(ctx_build, ffn, f_norm_rms_eps) : ggml_rms_norm(ctx_build, t21, f_norm_rms_eps);
            randomize_tensor_normal(layer->eps, rnd);       gTN(layer->eps, "var_noise"); 
            hGensor  noise = ggml_scale_inplace(ctx_build, layer->eps, 0.001);
            ffn = ggml_add          (ctx_build, ffn,noise);     
        }else{
        } 
        break;
    }
    case VAR_0:
    case ONLY_LNormal:
    case ONLY_RMSNormal:    {
        assert(ffn_up==nullptr);
        hGensor  t22 = tpFFN==ONLY_LNormal ? ggml_norm(ctx_build, ffn_inp, f_norm_rms_eps) :
            ggml_rms_norm(ctx_build, ffn_inp, f_norm_rms_eps);
        gTN(t22, "t22");               assert_shape_2d(t22, n_embd, N*n_batch);   
        if(tpFFN==VAR_0)     {
            randomize_tensor_normal(layer->eps, rnd); 
            hGensor  noise = ggml_scale_inplace(ctx_build, layer->eps, 0.001);
            ffn = ggml_add          (ctx_build, t22,noise);     
            // ffn = t22;        
        }else{
            hGensor  t23 = ggml_repeat       (ctx_build, ffn_norm, t22);                    gTN(t23, "t23");     assert_shape_2d(t23, n_embd, N*n_batch);
            ffn = ggml_mul          (ctx_build, t23, t22);
        }
        break;
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
        // hGensor  tMu = ggml_mul          (ctx_build, layer->w_mu, t21);
        // hGensor  tVar = ggml_mul          (ctx_build, layer->w_var, t21);*/
        //  trick v1
                              
            // gTN(t30, "t30");     assert_shape_2d(t30, n_embd, N*n_batch);        
    }else{
        
    }       
    assert_shape_2d(ffn, n_embd, N*n_batch);
    sprintf(nam_,"l_out-%d",layer->id),  gTN(ffn, nam_);
    return ffn;          
}



std::string NLP_AutoRegressive::Name()     {   
    return "LAMA";  
}



//REF: ggml_compute_forward_dup_bytes
void Fish::CopyWeight(const Fish* src,int flag) {
    if(wiki_tutor!=nullptr) {
        assert(wiki_tutor==src->wiki_tutor);
        return;
    }
    auto gsrc = src->hForwTG->raw();
    size_t nx=0,nz,nT=0,type_size;
    vector<hGensor> tSrc;  
    hGensor t1;
    if(isTrain()){
        for (int i = 0; i < gsrc->n_nodes; ++i) {
            hGensor t0 = gsrc->nodes[i];
            if(strcmp(t0->name,"output.weight")==0){
                int j = 0;
            }
            type_size = ggml_type_size(t0->type);
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
        return;       
    }else if(!src->loadGensors.empty()){
        tSrc = src->loadGensors;
    }
    for(auto t0 : tSrc){
        t1 = GetGensor(t0->name);
        assert(t1!=nullptr && t0->type==t1->type);
        nz = ggml_nelements(t0);        type_size = ggml_type_size(t0->type);
        assert(nz==ggml_nelements(t1));
        memcpy(t1->data,t0->data,type_size*nz);
        nx += nz;       nT++;
    }
}

/*
    would affect training process?
*/
bool NLP_AutoRegressive::LocalFeeling(SampLoader *hLoader,vector<float>& result,int flag)  {
    assert(hOPT!=nullptr);
    
    assert(hLoader->all_samps.size()==1);
    auto hSamp = hLoader->all_samps[0];
    int i,nTok = hSamp->len,_nctx=hparams.n_ctx();
    assert(!hDict->tokenizer_add_bos);
    hOPT->Evaluate(*hLoader,-666);
    if(DUMP())
        _INFO("\t%s @\"%s\"\n",__func__,hLoader->sentence.c_str());
    assert(preLogits->type==GGML_TYPE_F32);
    assert(preLogits->ne[1]==nTok+1);  //???
    size_t nz = ggml_nelements(preLogits),nVocabInWiki=preLogits->ne[0]; //preLogits->nb[0];

    //out is last distribution of Causal Language Modeling
    size_t off = (nTok)*nVocabInWiki;      
    float *out = (float*)(preLogits->data)+off;
    if(result.size()!=nVocabInWiki){
        result.clear( );     result.resize(nVocabInWiki);
    }
    memcpy(result.data(),out,sizeof(float)*nVocabInWiki);
    float sum=0;
    for(auto f : result)    sum+=f;
    return true;
}

size_t NLP_AutoRegressive::tVocab(){
    assert(hDict!=nullptr);
    return hDict->tVocab( );
    //return hTokenset->nUnique;
}

hGensor Fish::BuildLoss( struct ggml_context * ctx,hGensor cur,int flag){
    if(hparams.debug.NO_loss)   {
        assert(cur==preLogits);
        out_node = cur;     return cur;
    }
    int n_ctx = hparams.n_ctx(),nCls = nClass(),n_batch = hparams.n_batch();
    if(hparams.is( {"model","target"},string("OneHot") )){
        target_probs = ggml_new_tensor_3d(ctx_build, GGML_TYPE_I32, 1, n_ctx, n_batch);
    }else
        target_probs = ggml_new_tensor_3d(ctx_build, GGML_TYPE_F32, nCls, n_ctx, n_batch);
    gTN(target_probs,      "targets");

    assert(loss==nullptr);
    hGensor  t36 = nullptr;    
    if(hparams.is({"model","target"},string("OneHot")))
        t36 = ggml_cross_entropy_loss_1(ctx, cur, target_probs);
    else
        t36 = ggml_cross_entropy_loss(ctx, cur, target_probs);          //  square_error_loss(ctx0, targets, logits);       
    gTN(t36, "loss");     assert_shape_1d(t36, 1);
    loss = t36;
    // if(isTrain())
    //     assert(loss->grad!=nullptr);
#ifdef GG_V12
    ggml_set_loss(loss);
#endif
    out_node = loss;
    
    return loss;
}

hGensor NLP_AutoRegressive::BuildTarget( struct ggml_context * ctx,hGensor cur,int flag)  {
    hGensor _tNorm = UpdateGensor (hDict->_norm.w->name); 
    int n_vocab = tVocab(),n_batch = hparams.common.n_batch,n_ctx = hparams.common.n_ctx,n_embd = hparams.n_embd;
    auto train_params = hparams.common;
    train_params.use_checkpointing = false;     // CYS_0826
    const int N = train_params.n_ctx, n_past = 0;
    const float rms_norm_eps = hparams.f_norm_rms_eps;
    hGensor  t32 = nullptr,wA = nullptr,wB = nullptr;
    hGensor  t31 = ggml_rms_norm(ctx, cur, rms_norm_eps);                    gTN(t31, "norm");     
    assert_shape_2d(t31, hparams.n_embd, N*train_params.n_batch);
    
    if(hDict->nLevel>0){
        t31 = hDict->DEC(ctx,t31);      //t31 = ggml_mul_mat(ctx, hDict->decoder, t31 );  
        gTN(t31, "embed_decoder");
        t32 = ggml_repeat            (ctx, _tNorm, t31); 
        n_embd = t32->ne[0];
    }   else{
        if(isTrain()){
            t32 = ggml_repeat(ctx, _tNorm, t31);   //_tNorm shoud same shape as t31 if has grad!
            gTN(t32, "_tNorm.repeat");     //assert_shape_2d(t32, n_embd, N*n_batch);
        }else
            t32 = _tNorm ;   
    }     
    hGensor  t33   = ggml_mul(ctx, t31, t32);                             gTN(t33, "result_norm");     
    assert_shape_2d(t33, n_embd, N*n_batch);
    if(role==ROLE_TYPE::SWARM_FOLLOWER){
        out_node = t33 ;        return out_node;
    }else    {
        if(role==ROLE_TYPE::SWARM_HEAD){
            t33 = mos.Build(hparams,ctx,t33);
        }
        hGensor t34 = hDict->Embed2Output(ctx,t33);
        // hGensor  t34   = ggml_mul_mat           (ctx, _tOutput, t33);                          gTN(t34, "t34");     
        assert_shape_2d(t34, n_vocab, N*n_batch);
        hGensor  t35 = n_batch==1 ? t34 : ggml_reshape_3d(ctx, t34, n_vocab, N, n_batch);             gTN(t35, "t35");     
        // no,no,no! 1) Softmax layers can be difficult to train since the gradients can vanish or explode  2) CrossEntropyLoss assumes logits on the input.
        //  t35 = ggml_soft_max_inplace(ctx,t35); 
        // preLogits = t35;
        if(!isLocalInfer){
            if(mom.embed2w!=nullptr)    {   
                assert(teach==WIKI::_LOGITS_GATE);
                t35 = build_gate(ctx,t33,t35,flag);
            }   else {
                for(auto wiki:wikis)    {   
                    if(wiki->t2t!=nullptr){
                        hGensor tEX1 = ggml_mul_mat(ctx, wiki->t2t, wiki->exLogits);                          
                        t35 = ggml_add(ctx,t35,tEX1);
                    }else if(wiki->exLogits!=nullptr){
                        hGensor tEX1 = ggml_soft_max(ctx,wiki->exLogits);
                        t35 = ggml_add(ctx,t35,tEX1);
                    }
                    
                    //  WIKI::_LOGITS_SCALE
                    // t35 = ggml_relu(ctx,t35);       //converge very slow, so strange!
                    // t35 = ggml_mul(ctx,t35,exLogits);        
                }
            }        
        }
        BuildLoss(ctx,t35);
        if (train_params.use_checkpointing) {
            checkpoints.push_back(t31);            checkpoints.push_back(t32);            checkpoints.push_back(t33);
            checkpoints.push_back(t34);            checkpoints.push_back(t35);            checkpoints.push_back(out_node);
        }    
        preLogits = t35;
    }

    // if(isTrain())
    //     assert(out_node->grad!=nullptr);
    if(hDict->nLevel>0){
        n_embd = hparams.n_embd;
    } 
    return out_node;
}




struct ggml_cgraph * llama_build_graph(llama_context & lctx,const llama_batch & batch,bool   worst_case);

std::string NLP_AutoRegressive::T2STR( const std::vector<TOKEN_ID>& toks,int flag )                    { 
    std::string str = "";
    for(auto tok : toks){
        if(tok==hDict->eos)
            break;
        std::string a = hDict->T2STR(tok,flag);
        str += a;
    }
    
    return str;  
}



bool NLP_AutoRegressive::InitDictTokenset(int flag)    {
    void *hLLM = nullptr;
    
    switch(hparams.ModelArch()){
    case MODEL_ARCH::NLP_GPT2:
    case MODEL_ARCH::NLP_GPT2_char:
        if(hparams.ModelArch()==MODEL_ARCH::NLP_GPT2_char){
            hDict = std::make_shared<CDict_CHAR>(this);
            hDict->LoadVocab("",0x0); 
        }else{
            hDict = std::make_shared<CDict_GPT2>(this);
            if(wikis.size()>0)  {   //lama()!= nullptr
                // hDict->LoadVocab(lama()->model_path.c_str(),0x0);       //  tokenizer_name == "gpt2"
                hDict->bos = wikis[0]->bos;             hDict->eos = wikis[0]->eos;  
                // hLLM = wikis[0]->lmodel;
            }  else{
                // hDict->LoadTokenizer("/home/cys/rnd/lic/models/gpt2_tokenizer.bin");
            }
        }
        hTokenset = std::make_shared<DataTokenSet>(hDict.get());
        
        break;
    default:
        hDict = std::make_shared<ConsiceDict>(this);
        if(wikis.size()>0)  {   
            hDict->n_vocab = wikis[0]->n_vocab;
            // hDict->LoadVocab(lama()->model_path.c_str(),0x0); 
            hDict->bos = wikis[0]->bos;             hDict->eos = wikis[0]->eos;  
            // hLLM = lama()->lmodel;
        }
        hTokenset = std::make_shared<DataTokenSet>(hDict.get());        
        break;
    }
        
    if(hparams.isOnlyGPT){

    }else{
        if(!hTokenset->Load(hparams,hLLM,0x0)){
            _ERROR("\n======== %s Failed to load tokenset@%s!========\n",__func__,hparams.fp_train_data.c_str());
            return false;
        };
        hDict->mapT2T = hTokenset->mapT2T;      hDict->dialect = hTokenset->dialect;
        for(auto wiki : wikis){
            wiki->mapT2T = hDict->mapT2T;       wiki->dialect = hDict->dialect;
        }       
    }     
    if(isTrain()){

    }

    return true;
}


/*
double WIKI::InductLogits(int nSampInBatch,std::vector<int32_t>& tok_ids,hGensor userLogits,hGensor target_probs,int flag)  {
    if(!isInduct())
        return -1.0;

    Reset();         //Timing bottleneck!!! for the crazy design of llama.cpp
    Decode(tok_ids,0,0x0,true);    
    const float *all_logits = GetLogits(n_vocab,tok_ids.size(),0),*logit; 
    size_t k,j,i,ldL=exLogits->ne[0];  
    int n_ctx = target_probs->ne[1],n_dialect=mapT2T.size(),token;  
    double a1,a2,nrm=0;    
    float *p=teach == WIKI::_TARGET ? new float[ldL]:nullptr,*target ;  
    if(flag<0){    //CHILD_0909_WIKIS
        hGensor logits = userLogits==nullptr ? exLogits : userLogits;
        assert(logits!=nullptr);
        target = (float*)logits->data+nSampInBatch*n_ctx*ldL;        
        nrm =  NRM_2(all_logits,n_ctx*ldL)/ldL;   
        if(logits->ne[0]==n_dialect){
            for(i=0; i<n_ctx; i++,target+=n_dialect,all_logits+=n_vocab){
                for(j=0;j<n_vocab;j++){
                    if(dialect[j]==0)       
                        continue;
                    token = mapT2T[j];
                    target[token] = all_logits[j];
                }                
            }
        }else
            memcpy((void*)target,(void*)all_logits,sizeof(float)*n_ctx*n_vocab);       //memcpy(g->data+off,(void*)(logits),ld2); 
    }else{    
        assert(0);
        for (k=0; k<nSampInBatch; ++k) {        
            const float *from=all_logits+k*n_vocab;
            a1=NRM_2((float*)(from),n_ctx*n_vocab);          nrm=max(nrm,a1/n_vocab);     
            if(teach == WIKI::_TARGET){              
                assert(exLogits==nullptr);             
                for(j=0;j<n_ctx;j++){
                    logit = from+j*n_vocab;
                    target = (float*)target_probs->data+(k*n_ctx+j)*n_vocab;
                    //  SOFT_MAX_minus(n_vocab,target,logit);
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
}*/


bool NLP_AutoRegressive::InitInput(struct ggml_context * ctx_build,bool isMask,int flag) {
    auto train_params = hparams.common;
    int n_ctx = train_params.n_ctx,n_vocab = tVocab(),n_batch = train_params.n_batch;
    assert(n_ctx>0 && n_batch>0);
    // struct ggml_init_params ctx_input_params = {// mem_size mem_buffer no_alloc
    //     ggml_tensor_overhead() * 2, NULL,true,                        
    // };
    // ctx_input = ggml_init(ctx_input_params);
    tokens_input = ggml_new_tensor_2d(ctx_build, GGML_TYPE_I32, n_ctx, n_batch);
    // if(hparams.is( {"model","target"},string("OneHot") )){
    //     target_probs = ggml_new_tensor_3d(ctx_build, GGML_TYPE_I32, 1,  n_ctx, n_batch);
    // }else
    //     target_probs = ggml_new_tensor_3d(ctx_build, GGML_TYPE_F32, n_vocab,  n_ctx, n_batch);
    // gTN(target_probs,      "targets");
    ggml_backend_buffer_t input_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx_build, ggml_backend_cpu_buffer_type());
    size_t max_input_size = ggml_backend_buffer_get_size(input_data);
    if(DUMP())
        _INFO("%s: input_size(%d,%d) = %zu bytes (%.1f MB)\n", __func__, n_ctx, n_batch, max_input_size, (float) max_input_size / (1024.0f*1024.0f));
    // ggml_free(ctx_input);
    gTN(tokens_input, "inp_tokens");
    
    // tokens_input would be set @UpdateBatch!
    if(n_batch>1){
        tBatch = ggml_reshape_1d(ctx_build, tokens_input, n_ctx*n_batch);   gTN(tokens_input, "inp_tokens_1d");
    } else
        tBatch = tokens_input;

    build_inp_KQ_(ctx_build,isMask);    
    return true;
}


bool NLP_AutoRegressive::CreateExlogists(hWIKI wiki,uint32_t n_ctx,uint32_t n_batch,int flag) {
    auto ctx=GetGGCTX();
    if(teach==WIKI::_OFF )
        return false;
    int64_t nV = tVocab();
    assert(wiki->n_vocab>=nV);
    if(!isLocalInfer){
        assert(wiki->exLogits==nullptr);
        
        if(wiki->n_vocab>nV){
            wiki->exLogits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nV,  n_ctx, n_batch);
            if(0){
            wiki->t2t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nV,  nV);
            sprintf(wiki->t2t->name,"t2t@%s",wiki->title.c_str());
            }
        }else{
            wiki->exLogits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, wiki->n_vocab,  n_ctx, n_batch);
        }
        tmpExLogis.push_back(wiki->exLogits); 
        return true;               
    }
    return false;
}

void NLP_AutoRegressive::Train(int flag)       {
    hOPT->BeforeTrain(hparams.common,tokens_input,0x0);
    if(!hOPT->PrepareData( hparams,flag ))
        return;
       
    GST_TIC(t0);
    print_build_info();
    
    SaveTrain("warmup" );      //warmup save
    Optimizer::RESULT result = hOPT->Search(ctx_work,loss,target_probs,hparams);  
    assert(result==Optimizer::OK);  
    if(ctx_work!=nullptr)  ggml_free(ctx_work);
    ggml_free(ctx_build);
    // ggml_free(ctx_input);
    // ggml_gallocr_free(alloc); 

    _INFO("%s: total training time: %.3g\n", __func__,GST_TOC(t0) );
}

struct ggml_cgraph *NLP_AutoRegressive::BuildRawGraph( struct ggml_context * ctx_,bool isBuild,int flag)       { 
    llama_model *model = GetRawModel( );
    assert(model!=nullptr);
    struct ggml_cgraph *gf_raw = ggml_new_graph_custom(ctx_, LLAMA_TRAIN_MAX_NODES, true);      ;
    gf_raw = _llama_raw_graph(model,gf_raw,hparams.prompt,false,0x0); 
    assert(gf_raw!=nullptr);
    for(int i=0;i<gf_raw->n_leafs;i++){     
        if(strcmp(gf_raw->leafs[i]->name,"inp_tokens")==0){  
            // tokens_input = gf_raw->nodes[i];     
            break;
        }
    }
    assert(isBuild);
    return gf_raw;
}

// TO DO :  refactor
void NLP_AutoRegressive::InitGensors(int flag){
    auto ctx=GetGGCTX();
    const uint32_t n_layer = layers.size();
    for (u_int i=0;i<n_layer;i++) {
        auto layer = dynamic_pointer_cast<QKV_LAY>(layers[i]);    
        if(!layer->Q.Empty()){    
            InitGensor(ctx,layer->att_norm.w, TN(LLM_TENSOR_ATTN_NORM, i), rnd);
            InitGensor(ctx,layer->Q.w,             TN(LLM_TENSOR_ATTN_Q, i), rnd);
            if(layer->K.w!=nullptr)      InitGensor(ctx,layer->K.w,TN(LLM_TENSOR_ATTN_K, i), rnd);
            if(layer->V.w!=nullptr)      InitGensor(ctx,layer->V.w,TN(LLM_TENSOR_ATTN_V, i), rnd);
            // InitGensor(ctx,layer->wo,             TN(LLM_TENSOR_ATTN_OUT, i), rnd);        
            InitGensor(ctx,layer->wo,             TN("blk.%d.attn_out", i), rnd);
        }
        if(layer->ffn_norm.w!=nullptr)
            InitGensor(ctx,layer->ffn_norm.w,       TN(LLM_TENSOR_FFN_NORM, i), rnd);
        if(layer->up.w!=nullptr){      
            if(layer->ffn_gate!=nullptr)          
            {   InitGensor(ctx,layer->ffn_gate,       TN(LLM_TENSOR_FFN_GATE, i), rnd); }
            InitGensor(ctx,layer->down.w,       TN(LLM_TENSOR_FFN_DOWN, i), rnd);
            InitGensor(ctx,layer->up.w,         TN(LLM_TENSOR_FFN_UP, i), rnd);                
        }
        if(layer->ffn_gate_inp!=nullptr){
            InitGensor(ctx,layer->ffn_gate_inp,             TN(LLM_TENSOR_FFN_GATE_INP, i), rnd);
            InitGensor(ctx,layer->ffn_gate_exps,            TN(LLM_TENSOR_FFN_GATE_EXPS, i), rnd);
            InitGensor(ctx,layer->ffn_down_exps,            TN(LLM_TENSOR_FFN_DOWN_EXPS, i), rnd);
            InitGensor(ctx,layer->ffn_up_exps,              TN(LLM_TENSOR_FFN_UP_EXPS, i), rnd);
            InitGensor(ctx,layer->ffn_gate_inp_shexp,       TN(LLM_TENSOR_FFN_GATE_INP_SHEXP, i), rnd);
            InitGensor(ctx,layer->ffn_gate_shexp,           TN(LLM_TENSOR_FFN_GATE_SHEXP, i), rnd);
            InitGensor(ctx,layer->ffn_down_shexp,           TN(LLM_TENSOR_FFN_DOWN_SHEXP, i), rnd);
            InitGensor(ctx,layer->ffn_up_shexp,             TN(LLM_TENSOR_FFN_UP_SHEXP, i), rnd);
        }
    }
    if(mom.embed2w!=nullptr){
        InitGensor(ctx,mom.embed2w,"gate_cys", rnd);
    }
    if(mos.gat_!=nullptr){
        InitGensor(ctx,mos.gat_,"gate_swarm", rnd);
    }
    for(auto wiki : wikis){
        if(wiki->t2t!=nullptr){
            InitGensor(ctx,wiki->t2t, nullptr, rnd);
        }            
    }
}

void NLP_AutoRegressive::InitModel(int flag){    
    const uint32_t n_ff = hparams.n_ff();
    auto train_params = hparams.common;
    uint32_t n_embd  = hparams.n_embd,n_ctx = train_params.n_ctx;        
    const uint32_t n_layer = hparams.nLayer();
    bool isJModel = !hparams.jModel.empty();
    auto ctx=GetGGCTX();
    if(arch==NLP_MAMBA)     {
        assert(hparams.n_head() == 0);
    }  else{
        hparams.n_rot = hparams.n_embd / hparams.n_head();    
    }
    
    _INFO("\nLLaMeta%s: init model embed=%d layer=%d ff=%d tpFFN=%d\n", __func__,n_embd,n_layer,n_ff,tpFFN);  
    _INFO("\t type of FFN=%s\n", tpFFN==FFN_TYPE::SWIGLU ? "MLP" : tpFFN==FFN_TYPE::VAR_LAST ? "Variation@last_layer" 
        : tpFFN==FFN_TYPE::ONLY_RMSNormal ? "RMS Normal" : "other");  
    _INFO("\t type of ATTENTION=%s P=%s \n",tpATT==ATTENTION_TYPE::BROWN ? "BROWN":"QKV",BROWN_Motion::Transfer_1?"Token":"Embed");
    if(isJModel){

    }else{
        for (int i=0;i<n_layer;i++) {
            auto  layer = std::make_shared<QKV_LAY>(this,i);
            layer->isLast = i==n_layer-1;
            layers.push_back(layer);        //typedef shared_ptr<layer> hLayer;
            layer->att_norm.w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            if(tpATT==ATTENTION_TYPE::OFF){
            }else if(tpATT==ATTENTION_TYPE::QKV){
                layer->Q.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                layer->K.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                layer->V.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);                
            }else{
                if(BROWN_Motion::Transfer_1){
                    layer->V.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);                
                    layer->Q.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                }else{
                    layer->Q.w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
                }
            }
            layer->wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            layer->CreateFFN(hparams,ctx,tpFFN);            
        }
    }
    tmpExLogis.clear();
    if(role==ROLE_TYPE::SWARM_FOLLOWER) {
    }else{
        for(auto wiki : wikis){
            CreateExlogists(wiki,n_ctx,train_params.n_batch);  
        }      
        if(role==ROLE_TYPE::SWARM_HEAD) {
            mos.Init(swarm,ctx,n_embd); 
        }      
    }

    if( tmpExLogis.size()>0 
            && !isLocalInfer) {
        // exLogits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_vocab,  n_ctx, train_params.n_batch);
        if(teach==WIKI::_LOGITS_GATE)   {               
            mom.embed2w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, tmpExLogis.size()+1);
        }   
    }
    nParams = 0;    
    
    if(isJModel){

    }else{
        hDict->CreateEmbeddings(0x0);
        hEDS->Alloc(ctx);  //back_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);        
        hDict->Update(rnd,0x0);      
        InitGensors(flag);        
    }
}

//Deprecated
void NLP_AutoRegressive::LoadTensors(struct llama_model * lama,int flag) {        
    nParams = 0;
    hDict->Update(nullptr,true);   //UpdateTokEmbed(lama,nullptr,true);
    
    // layers.resize(hparams.n_layer);
    for (uint32_t i = 0; i < hparams.n_layer_train; ++i) {
        // auto layer = dynamic_pointer_cast<QKV_LAY>(layers[i]);
        auto layer = std::make_shared<QKV_LAY>(this,i);
        layers.push_back(layer);        
        layer->att_norm.w = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_NORM, i));     nParams+=ggml_nelements(layer->att_norm.w);
        layer->Q.w             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_Q, i));        nParams+=ggml_nelements(layer->Q.w);
        layer->K.w             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_K, i));        nParams+=ggml_nelements(layer->K.w);
        layer->V.w             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_V, i));        nParams+=ggml_nelements(layer->V.w);
        layer->wo             = llama_get_model_tensor(lama, TN(LLM_TENSOR_ATTN_OUT, i));      nParams+=ggml_nelements(layer->wo);
        layer->ffn_norm.w       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_NORM, i));      nParams+=ggml_nelements(layer->ffn_norm.w);
        layer->ffn_gate       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_GATE, i));      nParams+=ggml_nelements(layer->ffn_gate);
        layer->down.w       = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_DOWN, i));      nParams+=ggml_nelements(layer->down.w);
        layer->up.w         = llama_get_model_tensor(lama, TN(LLM_TENSOR_FFN_UP, i));        nParams+=ggml_nelements(layer->up.w); 

        gensors.Insert({layer->att_norm.w,layer->Q.w,layer->K.w,layer->V.w,layer->wo,layer->ffn_norm.w,layer->ffn_gate,layer->down.w,layer->up.w});
        assert_shape_1d(layer->att_norm.w, hparams.n_embd);
        assert_shape_2d(layer->Q.w,             hparams.n_embd, hparams.n_embd);
        assert_shape_2d(layer->K.w,             hparams.n_embd, hparams.n_embd_gqa());
        assert_shape_2d(layer->V.w,             hparams.n_embd, hparams.n_embd_gqa());
        assert_shape_2d(layer->wo,             hparams.n_embd, hparams.n_embd);
        assert_shape_1d(layer->ffn_norm.w,       hparams.n_embd);
        assert_shape_2d(layer->ffn_gate,       hparams.n_embd, hparams.n_ff());
        assert_shape_2d(layer->down.w,       hparams.n_ff(),   hparams.n_embd);
        assert_shape_2d(layer->up.w,         hparams.n_embd, hparams.n_ff());
    }

}

