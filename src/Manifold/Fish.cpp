/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  Random swimming fish  
 * 
 *  \brief General Language model
 *  \author Yingshi Chen
 */
#include <set>
#include "Fish.hpp"
#include "gLLM.hpp"
#include "../g_stddef.hpp"

hFISH Fish::MakeInstance(const std::string nam_,struct CLI_params& params,vector<hWIKI> wikis,ROLE_TYPE role_,int flag)   {
    assert(wikis.size()>=0);
    hFISH fish = nullptr;
    switch(params.ModelArch()){
    case MODEL_ARCH::NLP_MAMBA:
        fish = std::make_shared<LLM_MAMBA>(nam_+"_mamba",params,role_);
        break;
    case MODEL_ARCH::NLP_GPT2:
    case MODEL_ARCH::NLP_GPT2_char:
        fish = std::make_shared<GPT2>(nam_+"_GPT2",params,role_);
        break;
    case MODEL_ARCH::NLP_MOE:
        fish = std::make_shared<LLM_MOE>(nam_+"_moe",params,role_);
        break;
    default:
        switch(params.nabla){
        case 1:
            fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params,role_);
            break;
        case 2:
            // fish = std::make_shared<LLAMA_Brown>(nam_+"_brown",params);
            fish = std::make_shared<NLP_AutoRegressive>(nam_,params,role_);
            break;
        case 3:
            fish = std::make_shared<LLAMA_VAE>(nam_+"_vae",params,role_);
            break;    
        default:
            assert(0);
        }     
    }
    
    fish->isLocalInfer = flag==0x110;   
    if(!fish->Init( wikis ))
        return nullptr;  
    if(!fish->Build( ) )
        return nullptr;
    if(fish->role==SWARM_FOLLOWER){

    }else{
        if(!wikis.empty()){  //generate some sentence
            if( params.gpt_every>0 && !fish->isLocalInfer)
                fish->gopt = GeneratOnPrompt::MakeInstance(params,wikis,fish.get(),flag);        
        }
    }
    // fish->Dump(0x0);
    return fish;
}

hFISH Fish::MakeSwarm(const std::string nam_,struct CLI_params& params,int flag)   {
    vector<hWIKI> wikis;
    if(params.tpWiki!="off") {//wiki is so heavy(ugly) that only load one instance here!
        for(auto path : params.fn_model_base){
            hWIKI wiki = std::make_shared<LAMA>(params,path);
            wikis.push_back(wiki);
        }        
    }       
    assert(wikis.size()>=0);

    int nSwarm = params.n_swarm,i;    
    for(i=0; i<nSwarm;i++){
        ROLE_TYPE role = i==nSwarm-1 ? SWARM_HEAD : SWARM_FOLLOWER;
        string title = role == SWARM_HEAD ? "Head_" : "Follower_";
        if(1 || role==SWARM_HEAD){
            hFISH fish = Fish::MakeInstance(title,params,wikis,role,0x0); 
            Fish::swarm.push_back(fish);            
        }
    }
    hSALP salp = std::make_shared<LogicSalp>(nam_,params);
    return salp;    
}

hFISH Fish::MakeInstance(const std::string nam_,struct CLI_params& params,const Fish *hSrc_,int flag)   {
    hFISH fish = nullptr;
    ROLE_TYPE role = ROLE_TYPE::COMMON;
    switch(params.nabla){
    case 1:
        fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params,role);
        break;
    case 2:
        fish = std::make_shared<NLP_AutoRegressive>(nam_,params,role);
        break;
    case 3:
        fish = std::make_shared<LLAMA_VAE>(nam_+"_vae",params,role);
        break;    
    default:
        assert(0);
    }  
    fish->isLocalInfer = flag==0x110;
    fish->graph_order = hSrc_->graph_order;
    //wiki is so heavy(ugly) that only one instance from hSrc!    
    fish->Init( hSrc_->wikis );    
    fish->Build( );

    if(fish->isTrain()){  
        fish->gopt = GeneratOnPrompt::MakeInstance(params,fish->wikis,fish.get(),flag);        
    }else{
        
    }
    
    return fish;
}

bool Fish::AfterBuild(bool isInitParam,int flag)   {        
    int64_t nx = 0;
    auto gf = GetForwRaw(),gb = GetBackRaw();
    if(isInitParam) {
        assert(rnd!=nullptr);
        // rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
    }
    printf("\n\n");
    assert(optParams.size()==0);
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & GGML_TENSOR_FLAG_PARAM) {
            GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);      
            auto node = gf->nodes[i];          
            optParams.push_back(node);
            // _INFO("Param_%-4d(op=%d)\t", optParams.size(), gf->nodes[i]->grad->op );
        }
    }
    isLoadCheckpoint = GGUF_Serialize(hparams.save.checkpoint_in,false,0x0);
    for (auto node : optParams) {  
        if(isLoadCheckpoint) {
            
        }else if(isInitParam){
            if (rnd != nullptr)
                randomize_tensor_normal(node, rnd);
            else
                ggml_set_zero(node);         
            _pt_cys_("",node,0x0);         printf("\n");       
        }
        
#ifndef NDEBUG
            
#endif
        
        // gg_print_tensor_("",gf->nodes[i],0);
        nx += ggml_nelements(node);        
    }   
    if(isTrain())  
        assert(nParams>0);
    else{
        assert(nParams==0);     assert(gb==nullptr);
    }
    if(nx != nParams){
        CHECK_SAME_TENSORS("Compare parameter tensors\t",optParams,xGensors); 
        _ERROR("%s nx(%ld)!=nParams(%ld)\t", __func__,nx,nParams );
        //  assert(0);
        // return false;
    }
    if (!hparams.only_write_model && hOPT!=nullptr) {
        hOPT->Prepare(nParams);
    }
    hOPT->Dump(1);
    szModel = ggml_used_mem(GetCTX()) + hEDS->sz; // ggml_backend_buffer_get_size(back_data);   // (float) (ggml_used_mem(ctx) + ggml_backend_buffer_get_size(back_data)) ;
          
    GGML_ASSERT(optParams.size() < GGML_MAX_PARAMS);

    if(role==SWARM_FOLLOWER){
        
    }else{
        
    }

    return true;
}

bool Fish::BeforeBuild(int flag)   {
    if(role==SWARM_HEAD){
        assert(swarm.size()>0);
    }else{
        
    }
    return true;
}

bool Fish::OnTrainStep(struct train_opt_callback_data *data,SampLoader&loader, int accum_step, float *sched, int flag)    {
    LossCurve(0x0);
    assert(0);
    return false;
}

static const char * vendor = "gruai";     //llm_arch_from_string
bool Fish::SaveTrain(string sX,int flag) { 
    assert(hOPT!=nullptr);
    int64_t iter = hOPT->iter;  //     train->opt->iter;
    // _INFO("%s: iter_%ld\n", __func__, iter);
    string sit = "IT",sOut;
    string sBaseName = hparams.save.model_out;  //get_train_filename(.c_str(),sit.c_str(), "", -1  );
    if (!hparams.save.checkpoint_out.empty()) {
        sOut = hparams.save.checkpoint_out+std::to_string(iter)+sX+".gguf";
        GGUF_Serialize(sOut,true,0x0);
        // GGUF_Serialize(sOut,false,0x0); //only for debug
    }
    
    if (!hparams.save.model_out.empty()) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        vendor = "gruai";                 //llm_arch_from_string
    }
    if(1){  //only for debug
        vendor = "llama";
        sOut = "l_" + sBaseName;     //hack  
    }
    
    return true;
}
/*
    1.  gguf_get_tensor_offset
*/
bool Fish::GGUF_Serialize(const std::string&path,  bool isSave, int flag){
try{
    char buf[1024];
    struct ggml_context * fctx_data = NULL;
    struct gguf_context * fctx = NULL;
    int n_kv = 0,n_tensors = 0;
    if(isSave){ //KV pairs
        fctx = gguf_init_empty();
        // struct ggml_init_params params = {128ull*1024ull*1024ull,NULL,false,};
        // fctx_data = ggml_init(params);
    }else{
        fctx = gguf_init_from_file(path.c_str(), {false,&fctx_data});
        if (!fctx) {
            _INFO("%s: failed to load '%s'\n", __func__, path.c_str());
            return false;
        }

        _INFO("%s: version=%d alignment=%zu offset=%zu\n", __func__, gguf_get_version(fctx),gguf_get_alignment(fctx),gguf_get_data_offset(fctx));
        n_kv = gguf_get_n_kv(fctx);
        _INFO("%s: n_kv: %d\n", __func__, n_kv);
        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(fctx, i);
            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
        if(0){// find kv string
            const char * findkey = "some.parameter.string";
            const int keyidx = gguf_find_key(fctx, findkey);
            if (keyidx == -1) {
                printf("%s: find key: %s not found.\n", __func__, findkey);
            } else {
                const char * key_value = gguf_get_val_str(fctx, keyidx);
                printf("%s: find key: %s found, kv[%d] value = %s\n", __func__, findkey, keyidx, key_value);
            }
        }
    }
    if(isSave){
        for(auto ps : optParams) {            
            gguf_add_tensor(fctx, ps);
        } 
        const bool only_meta = false;    
        gguf_write_to_file(fctx, path.c_str(), only_meta);
        size_t fsize = F_SIZE(path.c_str());
        _INFO("[save] @\"%s\" nT=%ld fsize=%gM\n",path.c_str(),optParams.size(),fsize/1.0e6);
    }else{
        n_tensors = gguf_get_n_tensors(fctx);
        if(isTrain() && n_tensors!=optParams.size()){      //  optParams maybe empty
            _INFO("%s nOptParams don't match(%d,%d) @%s!",__func__,n_tensors,optParams.size(),path.c_str());
            return false;
        }
        _INFO("[Serialize] n_tensors: %d\n", n_tensors);
        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name  (fctx, i);
            hGensor target = GetGensor(name);
            if(target==nullptr){
                return false;
            }
            if(!optParams.empty()){
                if(!(target->flags & GGML_TENSOR_FLAG_PARAM)){
                    return false;
                }
            }else{
                assert(!isTrain());
            }
            // const size_t offset = gguf_get_tensor_offset(fctx, i);
            // printf("%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
            struct ggml_tensor * cur = ggml_get_tensor(fctx_data, name);    //  cur = (struct ggml_tensor *)(mem_buffer + obj->offs);
            if(cur==nullptr){
                _INFO("%s failed to load tensor(%s) @%s!",__func__,name,path.c_str());
                return false;
            }else{
                
            }
            size_t nEle = ggml_nelements(cur),sz = ggml_nbytes(cur);
            assert(nEle == ggml_nelements(target)) ;
            memcpy(target->data,cur->data,sz);
            sprintf(buf,"\t%d d=%d sz=%ld",i,ggml_n_dims(cur),sz);
             _pt_cys_(buf,cur,0x0);    printf("\n");
            // _INFO("[Serialize]_%d\t%s: n_dims=%d sz = %ld\n",i,cur->name, ggml_n_dims(cur),sz);            
        }    
    }
    if(fctx_data!=NULL) ggml_free(fctx_data);
    gguf_free(fctx);
    return true;
}catch(...){
    return false;
}
}

/*void NLP_AutoRegressive::LoadGGUF(const std::string &filename, int flag)    {
    enum llama_ftype ftype = LLAMA_FTYPE_ALL_F32;       //LLAMA_FTYPE_MOSTLY_Q2_K
    _INFO("[save] %s to \"%s\" ftype=%d ......", name.c_str(), filename.c_str(), ftype);
    struct gguf_context * fctx = gguf_init_empty();
    int keyidx = -1;    
    
    // set arch_str
    gguf_set_val_str(fctx, LLM_KV_GENERAL_ARCHITECTURE, arch_str);
    gguf_set_val_str(fctx, LLM_KV_GENERAL_NAME, ".");
    gguf_set_val_u32(fctx, kv(LLM_KV_VOCAB_SIZE), hDict->n_vocab);
    int llm_embd = hDict->lama_embed,latent_dim=hDict->latent_dim;        //hparams.n_embd
    if(hDict->nLevel>0)    assert(llm_embd>latent_dim && latent_dim>0);
    // set hparams
    const char*str = kv(LLM_KV_CONTEXT_LENGTH);
    gguf_set_val_u32(fctx, kv(LLM_KV_CONTEXT_LENGTH),              hparams.common.n_ctx                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_EMBEDDING_LENGTH),            llm_embd                       );
    
    gguf_set_val_u32(fctx, kv(LLM_KV_BLOCK_COUNT),                 hparams.n_layer_train                );
    gguf_set_val_u32(fctx, kv(LLM_KV_FEED_FORWARD_LENGTH),         hparams.n_ff()                   );
    gguf_set_val_u32(fctx, kv(LLM_KV_ROPE_DIMENSION_COUNT),        hparams.n_rot                  );
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT),        hparams.n_head()                 );    
    gguf_set_val_u32(fctx, kv(LLM_KV_ATTENTION_HEAD_COUNT_KV),     hparams.n_head_kv()              );

    gguf_set_val_f32(fctx, kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS), hparams.f_norm_rms_eps         );
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_FREQ_BASE),              hparams.rope_freq_base         ); // TODO load in llama.cpp
    
    gguf_set_val_u32(fctx, LLM_KV_GENERAL_FILE_TYPE, ftype);
    // set vocab by copying from vocab_model gguf file
    gguf_set_val_str(fctx, kv(LLM_KV_TOKENIZER_MODEL), hDict->tokenizer_name.c_str());

    gguf_set_arr_str(fctx, kv(LLM_KV_TOKENIZER_LIST), hDict->vocab.data(), hDict->n_vocab);
    if(hDict->scores!=nullptr)
        gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_SCORES), GGUF_TYPE_FLOAT32, hDict->scores, hDict->n_vocab);    
    gguf_set_arr_data(fctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE), GGUF_TYPE_INT32, hDict->toktypes, hDict->n_vocab);
    if (hDict->tokenizer_name == "gpt2"){
        const char* sMERGES = kv(LLM_KV_TOKENIZER_MERGES);
        gguf_set_val_u32(fctx, sMERGES,  hDict->merges_keyidx              );
        keyidx = gguf_find_key(fctx, sMERGES);      //only for debug
        assert(hDict->merges.size()==hDict->n_merges);
        string word = hDict->merges[0];
        gguf_set_arr_str(fctx, sMERGES, hDict->merges.data(), hDict->n_merges);
        for (int i = 0; i < hDict->n_merges; i++) {        //only for debug
            const std::string word = gguf_get_arr_str(fctx, keyidx, i);
            GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);            
        }
    }    
    
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_BOS_ID), hDict->special_bos_id);
    gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_EOS_ID), hDict->special_eos_id);
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_UNK_ID), hDict->special_unk_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_SEP_ID), hDict->special_sep_id);      -1
    // gguf_set_val_u32(fctx, kv(LLM_KV_TOKENIZER_PAD_ID), hDict->special_pad_id);      -1
    gguf_set_val_f32(fctx, kv(LLM_KV_ROPE_SCALE_LINEAR),           1.0f / hparams.rope_freq_scale );
    gguf_set_val_u32(fctx, kv(LLM_KV_DICT_LATENT_DIM),             latent_dim                       );    
    //more maybe from llama_chat_apply_template


    for(auto ps : optParams) {            
        gguf_add_tensor(fctx, ps);
    } 
    _INFO("JModel n=%d\t",optParams.size());
    

    const bool only_meta = false;
    gguf_write_to_file(fctx, filename.c_str(), only_meta);
    gguf_free(fctx);

    size_t fsize = F_SIZE(filename.c_str());
    _INFO("\n[save] @\"%s\" fsize=%gM\n",filename.c_str(),fsize/1.0e6);
}
*/
bool Fish::LoadTrain(int flag) { 
    assert(hOPT!=nullptr);
    int64_t iter = hOPT->iter;  //     train->opt->iter;
    _INFO("%s: ......", __func__);

    auto fpCheck = hparams.save.checkpoint_in;
    if (fpCheck.empty()){
        _INFO("\r[LoadTrain] failed!  please set checkpoint path @\"checkpoint-in\"\n" );
        return false;
    }
               
    if(!GGUF_Serialize(fpCheck,false,0x0))
        return false;
    assert( vendor == "gruai" );
    _INFO("\r[LoadTrain] OK @\"%s\"\n",fpCheck.c_str() );
    return true;
}
void Fish::Statistic(int typ, int flag)     {   
    string suffix="", prefix="\t"; 
    auto gf = hForwTG->raw(),gb = hBackTG==nullptr? nullptr : hBackTG->raw();
    if(hparams.is({"gpt","c_graph"},string("raw"))){
        _INFO("raw graph\n");
    }
    int vQKV = hparams.Get({"model","attention","version"},0,false);
    _INFO("QKV version=%d\n",vQKV);
    
    ggml_graph_stat(gf);
    if(gb!=nullptr) ggml_graph_stat(gb);
    bool isDot = false;
    if (isDot)        {
        ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
        if(gb!=nullptr) ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
    }   else        {
        if(preLogits!=nullptr)
            // hForwTG->__repr__(suffix,prefix);   //preLogits = gf->nodes[gf->n_nodes - 2];
        if(gb!=nullptr){
            // hBackTG->__repr__(suffix,prefix);   //// TGraph("Backward",gb,true)
        }            
    }

    int nT = gensors.size(), nQ = 0, nF16 = 0;
    for (auto t : gensors)        {
        auto type = t.second->type;
        if (ggml_is_quantized(type))
            nQ++;
        if (type == GGML_TYPE_F16)
            nF16++;
    }
}

int Fish::BuildGraphFromRaw(int flag)   {
    int iRet = 0x0;
    bool isKeep = true;
    ctx_compute_params.mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
            (hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
    ctx_build = ggml_init(ctx_compute_params);
    
    struct ggml_cgraph *gf = BuildRawGraph( ctx_build,false ),*gb=nullptr;
    // preLogits = gf->nodes[gf->n_nodes - 1]; // the output is always the last tensor in the graph
    if(1){        
        alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        iRet = BuildComputeGraph(0,ctx_build,alloc,0x0);
    }else{
        // set<std::string> leaf_const={""};
        /*for (int i = 0; i < gf->n_leafs; i++) { //to float
            struct ggml_tensor * node = gf->leafs[i];
            if(strstr(node->name,"weight")==NULL)        
                continue;        
            optParams.push_back(node);   
        }    */
        
        hGensor last_embd = gf->nodes[gf->n_nodes - 2];
        // gf->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;      
        if(!isLocalInfer){
            gb = ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true);
        }
        if(gb!=nullptr){
            ggml_graph_cpy(gf, gb);        
            if (isKeep) {
                for (int i = 0; i < gf->n_nodes; i++) {
                    struct ggml_tensor * node = gf->nodes[i];
                    if (node->grad) {
                        node->grad = ggml_dup_tensor(ctx_build, node);
                        gf->grads[i] = node->grad;
                    }
                }
            }

            // remember original gradients which start with zero values
            struct ggml_hash_set zero_table = ggml_hash_set_new(gf->size);
            for (int i = 0; i < gf->n_nodes; i++) {
                if (gf->grads[i]) {
                    ggml_hash_insert(&zero_table, gf->grads[i]);
                }
            }

            for (int i = gf->n_nodes - 1; i >= 0; i--) {
                struct ggml_tensor * node = gf->nodes[i];
                // inplace operations to add gradients are not created by ggml_compute_backward
                // use allocator to automatically make inplace operations
                if (node->grad) {
                    // ggml_compute_backward(ctx_build, node, &zero_table);
                }
            }

            for (int i = 0; i < gf->n_nodes; i++) {
                struct ggml_tensor * node = gf->nodes[i];

                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    GGML_PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
                    ggml_build_forward_expand(gb, node->grad);
                }
            }

            ggml_hash_set_free(&zero_table);       
            ggml_graph_print(gb);  
        }
    }
      
    return iRet;
}

// If isParam, only alloc grad, no init!
void Fish::InitGensor(struct ggml_context *ctx, const char *name, hGensor gensor, bool isParam, int flag)    {
    assert(gensor!=nullptr);
    if (name != nullptr){
        ggml_set_name(gensor, name);        //    gTN0(w,"%s.w",name.c_str());
    }
        
    assert(gensor->data == nullptr);
    Gensor2Map(gensor);
    if (isParam && isTrain())        {
        ggml_set_param(ctx, gensor);
        gTN(gensor,"");
        nParams += ggml_nelements(gensor);
        assert(strlen(gensor->grad->name)>0);
        xGensors.push_back(gensor);
    }
    if(strcmp(gensor->name,"output.bias")==0) {   //only for debug
        // xn= gensor;     xxn = gensor->grad;
    }
}

void Fish::InitGensor(struct ggml_context *ctx, hGensor gensor, const char *name, struct random_normal_distribution *rnd, int flag){
    if (name != nullptr)    {
        ggml_set_name(gensor, name);
    }
        
    if(isTrain()){
        ggml_set_param(ctx, gensor);
        gTN(gensor,"");
    }
        
    if (gensor->data == nullptr)        {
        assert(0);
    }
    else        {
        if (rnd != nullptr)
            randomize_tensor_normal(gensor, rnd);
        else
            ggml_set_zero(gensor);
    }
    if (updateTMap)        {
        Gensor2Map(gensor);            
    }
    if(isTrain())   {
        xGensors.push_back(gensor);
        nParams += ggml_nelements(gensor);
    }
    if(strcmp(gensor->name,"output.bias")==0) {   //only for debug
        // xn= gensor;     xxn = gensor->grad;
    }
}

hGensor Fish::AddTensor(struct ggml_context *ctx,const std::string&key_,enum ggml_type tp,const SHAPE& shape,bool isParam,int flag){
    CHECK_SHAPE(shape);
    hGensor gensor = nullptr;
    if(shape.size()==4)  {
        gensor = ggml_new_tensor_4d(ctx, tp, shape[0], shape[1], shape[2], shape[3]);
    }else if(shape.size()==2)  {
        gensor = ggml_new_tensor_2d(ctx, tp, shape[0], shape[1]);        
    }else if(shape.size()==1)  {
        gensor = ggml_new_tensor_1d(ctx, tp, shape[0]);        
    }else{
        assert(0);
    }  
    // Gensor2Map(gg_tensor);
    InitGensor(ctx, key_.c_str(), gensor, isParam, 0x0);

    return gensor;   
}

#include "ggml-cuda.h"
#include "ggml-sycl.h"
#include "ggml-alloc.h"
extern "C" bool alloc_tensor_range(struct ggml_context * ctx,struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,ggml_backend_buffer_t ** buffers, size_t * n_buffers);
EDGE_DEVICES::EDGE_DEVICES(Fish *hF,struct ggml_context *ctx,int flag){
    hFish = hF;
    assert(back_data==nullptr);
    assert(workers.size()==0);
/*      need cuda support!
    int nDevice = ggml_backend_cuda_get_device_count();     //  ggml_cuda_init: found 1 CUDA devices:
    string sTp = hFish->hparams.KV({"train","device"},"");
    for (int device = 0; device < nDevice; ++device) {
        ggml_backend_t backend = ggml_backend_cuda_init(device);
        if (backend == nullptr) {
            _ERROR("%s: failed to initialize CUDA%d backend\n", __func__, device);
        }
        if(sTp=="onlycpu")
            continue;
        workers.push_back(backend);
        // char *guid = (char*)(backend->guid);
        _INFO("Fish::%s init CUDA backend @%p\n", __func__, backend);
    }*/
    cpu = ggml_backend_cpu_init();
    assert( cpu!=nullptr); 
    //  typedef uint8_t ggml_guid[16];     
    _INFO("Fish::%s init CPU backend @%p\n", __func__,cpu);
    workers.push_back(cpu);

    for (auto * backend : workers) {
        if (ggml_backend_is_cpu(backend)) {            // use host buffers for the CPU backend compute buffer
            //llama_default_buffer_type_cpu(true)
        /*
            static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
         {
             .get_name         =  ggml_backend_cuda_host_buffer_type_name,
             .alloc_buffer     =  ggml_backend_cuda_host_buffer_type_alloc_buffer,
             .get_alignment    =  ggml_backend_cpu_buffer_type()->iface.get_alignment,
             .get_max_size     =  NULL, // defaults to SIZE_MAX
             .get_alloc_size   =  ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
             .is_host          =  ggml_backend_cpu_buffer_type()->iface.is_host,
        },
         .context  =  nullptr,
        };*/
            //auto buft = ggml_backend_cuda_host_buffer_type();
            auto buft = ggml_backend_cpu_buffer_type();
            bufts.push_back(buft);
        } else {
            bufts.push_back(ggml_backend_get_default_buffer_type(backend));
        }
    }

    
    /*
        static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
         {
            ggml_backend_cpu_buffer_type_get_name,
             ggml_backend_cpu_buffer_type_alloc_buffer,
             ggml_backend_cpu_buffer_type_get_alignment,
            NULL, // defaults to SIZE_MAX
             NULL, // defaults to ggml_nbytes
            ggml_backend_cpu_buffer_type_is_host,
        },
         NULL,
    };
    */
}


string EDGE_DEVICES::__repr__( string& suffix,string& prefix,int flag)  {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    if(isOnlyCPU()){
        assert(workers.size()==1);
        sprintf(buf+strlen(buf),"OnlyCPU"); 
    }else{
        for (auto * backend : workers) {
            if (ggml_backend_is_cpu(backend)) {  
                sprintf(buf+strlen(buf),"CPU,"); 
            }else{
                sprintf(buf+strlen(buf),"GPU,");
            }
        }        
    }

    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
}

size_t EDGE_DEVICES::Alloc(struct ggml_context *ctx,int flag)    {
    auto buft = ggml_backend_cpu_buffer_type();
    // back_data = ggml_backend_alloc_ctx_tensors_from_buft(ctx,type );
    assert(ggml_get_no_alloc(ctx) == true);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    size_t max_size = ggml_backend_buft_get_max_size(buft);        //  SIZE_MAX

    ggml_backend_buffer_t * buffers = NULL;
    size_t n_buffers = 0;

    size_t cur_buf_size = 0;
    struct ggml_tensor * first = ggml_get_first_tensor(ctx);
    for (struct ggml_tensor * t = first; t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        size_t this_size = 0;
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }
        if(this_size>3096*1024*1024)        // huge tensor of "gate_exLogits_1" 524M        [ 32000  512  32  1 f32]
            assert(0);
        if (this_size > max_size) {
            fprintf(stderr, "%s: tensor %s is too large to fit in a %s buffer (tensor size: %zu, max buffer size: %zu)\n",__func__, t->name,
                    ggml_backend_buft_name(buft),this_size, max_size);
            for (size_t i = 0; i < n_buffers; i++) {
                ggml_backend_buffer_free(buffers[i]);
            }
            free(buffers);
            return 0;
        }
        
        if ((cur_buf_size + this_size) > max_size) {
            // allocate tensors in the current buffer
            if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
                return 0;
            }
            first = t;
            cur_buf_size = this_size;
        } else {
            cur_buf_size += this_size;
        }
    }

    // allocate remaining tensors
    if (cur_buf_size > 0) {
        if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
            return 0;
        }
    }

    if (n_buffers == 0) {
#ifndef NDEBUG
        fprintf(stderr, "%s: all tensors in the context are already allocated\n", __func__);
#endif
        return 0;
    }

    ggml_backend_buffer_t buffer;
    if (n_buffers == 1) {
        buffer = buffers[0];
    } else {
        assert(0);
        // buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
    }
    free(buffers);
    // return buffer;
    back_data = buffer;
    assert(back_data!=nullptr);     
    sz=ggml_backend_buffer_get_size(back_data);          //buffer->size;
    double sG = sz*1.0/1.0e9;
    return sz;
}   

bool EDGE_DEVICES::Build(bool isX,int flag)  {
    string sTp = hFish->hparams.KV({"train","device"},"");
    auto& train_params = hFish->hparams.common;
    struct ggml_cgraph *cgraph = hFish->GetBackRaw();   
    if(cgraph==nullptr){        //  OnlyInfer
        cgraph = hFish->GetForwRaw();
    }
    size_t max_work_size = 0;
    
// ggml_free(ctx_build);         ctx_build = nullptr;
    // context for work buffer

    struct ggml_cplan gf_plan,gb_plan;
    if(sTp=="onlycpu"){
        return hFish->ComputePlan(flag);
        // gb_plan = ggml_graph_plan(cgraph, train_params.n_threads,nullptr);
        // max_work_size = gb_plan.work_size + GGML_OBJECT_SIZE;
        //  
    }
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
    // hFish->ctx_work = ggml_init({max_work_size,NULL,false});         
    // struct ggml_object * obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    // gb_plan.work_data = (uint8_t *)ctx_work->mem_buffer + obj->offs;
    // gf_plan = gb_plan;      //  ???

    

    assert(sched==nullptr);
    int nMost = hFish->nMostNodes();
    sched = ggml_backend_sched_new(workers.data(), bufts.data(), workers.size(), nMost, false);
    bool full_offloat = true;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto cur = cgraph->nodes[i];
        if(sTp=="onlycpu")
            continue;
         if (0/*!lctx.cparams.offload_kqv*/) {
            if (strcmp(cur->name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                ggml_backend_sched_set_tensor_backend(sched, cur, cpu);
            }
        }   else if (strcmp(cur->name, "norm") == 0) {// norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        }   else {
           for (auto * backend : workers) {
                /*
                      1.ggml_backend_cuda_supports_op
                      2.GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
                            const int min_batch_size = 32;
                            return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) || (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);
                        }
                */
                bool isNotTiny = cur->ne[1]>=32 || cur->ne[2]>=32;
                bool isOffload = ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur);
                if ( isOffload  /* && ggml_backend_supports_buft(backend, lctx.model.buft_layer[il].buft) &&*/) {
                    ggml_backend_sched_set_tensor_backend(sched, cur, backend);
                    break;
                }
            }
        }       
    }

    if (!ggml_backend_sched_reserve(sched, cgraph)) {
        _ERROR("%s: failed to allocate compute buffers\n", __func__);
        return false;
    }
    for (size_t i = 0; i < workers.size(); i++) {
        ggml_backend_t backend = workers[i];
        size_t size = ggml_backend_sched_get_buffer_size(sched, backend);
        if (size > 1) {
            _INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(bufts[i]),
                    size / 1024.0 / 1024.0);
        }
    }

    // note: the number of splits during measure is higher than during inference due to the kv shift
    int n_splits = ggml_backend_sched_get_n_splits(sched);
    _INFO("%s: graph nodes  = %d, splites = %d\n", __func__, cgraph->n_nodes,n_splits);

    return true;
}

bool Fish::ComputePlan(int flag) {
    auto& train_params = hparams.common;
    struct ggml_cgraph *cgraph = GetBackRaw();
    if(cgraph==nullptr){        //  OnlyInfer
        cgraph = hForwTG->raw();
    }
    gb_plan = ggml_graph_plan(cgraph, train_params.n_threads,nullptr);
    size_t max_work_size = gb_plan.work_size + GGML_OBJECT_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
// ggml_free(ctx_build);         ctx_build = nullptr;
    ctx_work = ggml_init({max_work_size,NULL,false});    
    struct ggml_object * obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    gb_plan.work_data = (uint8_t *)ctx_work->mem_buffer + obj->offs;
    gf_plan = gb_plan;      //  ???
    return true;
}