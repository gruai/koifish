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
    case NLP_LLAMA:
        fish = std::make_shared<NLP_AutoRegressive>(nam_,params,role_);
        break;
    default:    //  more structures
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

Fish::Fish(const std::string&nam_,struct CLI_params params,ROLE_TYPE role_,int flag) : name(nam_),hparams(params),role(role_) {
    arch = params.ModelArch();
    
    string w = hparams.KV({"jmodel","parameter","debug_init_weight"});    // hack parameter only for debug
    bool isLoad = !hparams.save.checkpoint_in.empty();
    if(role==SWARM_FOLLOWER){
        tpInitWeight = INIT_WEIGHT::COPY_SWARM_HEAD;
    }else{
        tpInitWeight = w=="copy_wiki" ? INIT_WEIGHT::COPY_WIKI :
            isLoad ? INIT_WEIGHT::SERIALIZE :
            INIT_WEIGHT::RANDOM;        
    }

    
    if(jKEY(params.jConfig,{"train"}).empty())     {
        isLocalInfer = true;
    }
}

hFISH Fish::MakeSwarm(const std::string nam_,struct CLI_params& params,int flag)   {
    vector<hWIKI> wikis = WIKI::MakeInstance(nam_,params,0x0);
    // if(params.tpWiki!="off") {//wiki is so heavy(ugly) that only load one instance here!
    //     for(auto path : params.fn_model_base){
    //         hWIKI wiki = std::make_shared<LAMA>(params,path);
    //         wikis.push_back(wiki);
    //     }        
    // }       
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
            auto node = gf->nodes[i];          
            optParams.push_back(node);  
            nx += ggml_nelements(node);        
            // _INFO("Param_%-4d(op=%d)\t", optParams.size(), gf->nodes[i]->grad->op );
        }
    }
    bool bRet = false;
    switch (tpInitWeight)    {
    case SERIALIZE:
        isLoadCheckpoint = GGUF_Serialize(hparams.save.checkpoint_in,false,0x0);
        bRet = isLoadCheckpoint;
        break;
    case COPY_WIKI:
        assert(0);  //  Deprecated
        assert(hparams.is({"wiki","actor"},"copy") && wikis.size()>0);
        bRet = CopyGensors(wikis[0],0x0); 
        break;
    case COPY_SWARM_HEAD:
        bRet = true;
        break;
    default:    //random
        assert(!isLoadCheckpoint);
        for (auto node : optParams) {  
            if(isInitParam){
                if (rnd != nullptr)
                    randomize_tensor_normal(node, rnd);
                else
                    ggml_set_zero(node);  
            }
            _pt_cys_("",node,0x0);         printf("\n");
        }   
        bRet = true;
        break;
    }
    if(!bRet){
        _INFO("%s Failed to InitWeight from %s! tutor=%s\n",__func__,"",CSTR(wikis[0]));
        return false;
    }
    if(0){    // would copy weight from master
        _INFO_IF("%s found %d params\n", __func__,optParams.size()); 
        isLoadCheckpoint = GGUF_Serialize(hparams.save.checkpoint_in,false,0x0);
        bool isCopy = hparams.is({"wiki","actor"},"copy") && wikis.size()>0 && !isLoadCheckpoint;
        if(isCopy){
            if(wikis.empty()){
                _INFO("%s wikis is empty. So strange !!! \n",__func__,"");
                return false;
            }
            bool bRet = CopyGensors(wikis[0],0x0);      
            if(!bRet){
                _INFO("%s Failed to coyp gensors from %s! tutor=%s\n",__func__,"",CSTR(wikis[0]));
                return false;
            }
        }
        for (auto node : optParams) {  
            if(isLoadCheckpoint || isCopy) {
                
            }else if(isInitParam){
                if (rnd != nullptr)
                    randomize_tensor_normal(node, rnd);
                else
                    ggml_set_zero(node);  
            }
            _pt_cys_("",node,0x0);         printf("\n");  
    #ifndef NDEBUG
    #endif        
            // gg_print_tensor_("",gf->nodes[i],0);
            
        }   
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
    szModel = ggml_used_mem(GetGGCTX()) + hEDS->sz; // ggml_backend_buffer_get_size(back_data);   // (float) (ggml_used_mem(ctx) + ggml_backend_buffer_get_size(back_data)) ;
          
    GGML_ASSERT(optParams.size() < GGML_MAX_PARAMS);

    if(role==SWARM_FOLLOWER){
        
    }else{
        
    }
    
    if(!ComputePlan(flag)){
        return false;
    }
    
    return true;
}

void Fish::ClearGraph(int flag) {
    if (ctx_build!=nullptr) {
        ggml_free(ctx_build);       //      free(ctx->mem_buffer);
        ctx_build = nullptr;
        // ggml_gallocr_free(alloc);
    }
    hForwTG.reset();                hBackTG.reset();    
    neurons.clear();
    gensors.Clear();
    in_node = nullptr, out_node = nullptr, tBatch=nullptr;
    loss = nullptr, target_probs = nullptr, KQ_pos = nullptr, KQ_mask = nullptr, pos_embd=nullptr;
    preLogits = nullptr;        
    xn = nullptr,xxn = nullptr;  
    optParams.clear();      xGensors.clear();

    childs.clear();
    tmpExLogis.clear();
    // for (ggml_backend_buffer_t buf : bufs) {
    //     ggml_backend_buffer_free(buf);
    // }
    return;
}
bool Fish::UpdateNCTX(int _nctx,int flag){
    int ctx0 = hparams.n_ctx();
    if(ctx0==_nctx)
        return true;
    name = "4GPT_" + std::to_string(_nctx);
    graph_update = _nctx;
    _INFO("\n\n[UpdateNCTX] %d=>%d @%s\n",ctx0,_nctx,name.c_str());
    ClearGraph();
    hparams.SetNCTX(_nctx);
    if(!Build())
        return false;
    
    return true;
}

bool Fish::BeforeBuild(int flag)   {
    assert(ctx_build==nullptr);
    ctx_size = MostMemSize(0x0);
    ctx_build = InitCTX(ctx_size);

    if(role==SWARM_HEAD){
        assert(swarm.size()>0);
    }else{
        
    }
    return true;
}

bool Fish::Build(int flag)  {
    if(!BeforeBuild())
        return false;
    int iRet = 0x0;
    bool isInitParam = false, isJModel = !hparams.jModel.empty();
    if(hparams.is({"gpt","c_graph"},string("raw")) || hparams.ModelArch()==MODEL_ARCH::NLP_GPT2 
        || hparams.ModelArch()==MODEL_ARCH::NLP_GPT2_char){  //only for debug
        isInitParam = true;
        // int iRet = _llama_build_graph(GetRawModel(),&gf,&gb,0x0);          //bug
        iRet = BuildGraphFromRaw(0x0);
    }else{
        InitInput(ctx_build,true,flag); 
        if(isJModel){
            isInitParam = true;
            hForwTG = std::make_shared<TGraph>(this,"J_model",ctx_build,true);
            jToGraph(ctx_build,false,flag);
            assert(preLogits!=nullptr);            
            BuildLoss(ctx_build,preLogits);
            if(rnd==nullptr){   // InitModel
                rnd = init_random_normal_distribution(hparams.common.seed, 0.0f, 1.0f, -1.0f, +1.0f);
                size_t sz2 = hEDS->Alloc(ctx_build);
            }                
            // alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
            iRet = BuildComputeGraph(0,ctx_build,0x0);
        }else{
            enum ggml_cgraph_eval_order best_order = GGML_CGRAPH_EVAL_ORDER_COUNT;  //GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT;  //GGML_CGRAPH_EVAL_ORDER_COUNT;
            if(role==SWARM_FOLLOWER){
                BuildOperators(ctx_build,NULL,false,flag);   
            }else{
                size_t best_compute_size = SIZE_MAX;        
                graph_order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;     //only for debug
                if(graph_order==-1)   {// find best evaluation order
                    for (unsigned order = 0; order < (unsigned) GGML_CGRAPH_EVAL_ORDER_COUNT; ++order) {
                        // ctx_build = ggml_init(ctx_compute_params);
                        ggml_gallocr_t alloc_tmp = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                        BuildOperators(ctx_build,alloc_tmp,true,flag);
                        BuildComputeGraph(order,ctx_build,flag);       
                        size_t max_compute_size = ggml_gallocr_get_buffer_size(alloc_tmp, 0); // FIXME: this will still allocate the buffer
                        if (max_compute_size < best_compute_size) {
                            best_compute_size = max_compute_size;
                            best_order = hForwTG->Order();
                        }
                        ggml_gallocr_free(alloc_tmp);           //gf = nullptr;
                        // ggml_free(ctx_build);
                    }
                
                    size_t max_compute_size = best_compute_size;
                    _INFO("%s: compute_size = %zu bytes (%.1f MB)\n", __func__, max_compute_size, (float) max_compute_size / (1024.0f*1024.0f));
                    _INFO("%s: evaluation order = %s\n", __func__,
                        (best_order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? "LEFT_TO_RIGHT" :
                        (best_order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? "RIGHT_TO_LEFT" :
                        "invalid");
                    graph_order = best_order;
                }else{
                    assert(GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT<=graph_order && graph_order<=GGML_CGRAPH_EVAL_ORDER_COUNT);
                    best_order = (enum ggml_cgraph_eval_order)(graph_order);
                } 

                // alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                BuildOperators(ctx_build,NULL,false,flag);   
                if( BuildComputeGraph(best_order,ctx_build,flag)!=0x0)
                    return false;
            }
        }
    }
    assert(iRet==0x0);
    Statistic(0x0);
    if(!AfterBuild(isInitParam))
        return false;
    
    Dump(0x0);
    // ClearGraph();
#ifndef NDEBUG
    // ggml_graph_print(gf);    ggml_graph_print(gb);       //only for debug
#endif    
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
    if(path.empty())
        return false;
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
            _INFO_IF("%s: kv[%d]: key = %s\n", __func__, i, key);
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
        loadGensors.clear();
        for (int i = 0; i < n_tensors; ++i) {
            const char *name = gguf_get_tensor_name  (fctx, i);
            hGensor target = GetGensor(name);
            if(target==nullptr){
                return false;
            }
            loadGensors.push_back(target);
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
            if(nEle != ggml_nelements(target)) {
                assert(0);      continue;
            }
            if(target->type!=cur->type) {
                Gensor2float_(cur,(float*)target->data,0x0);
            }else
                memcpy(target->data,cur->data,sz);
            if(DUMP()){
                sprintf(buf,"\t%d d=%d sz=%ld",i,ggml_n_dims(cur),sz);
                _pt_cys_(buf,target,0x0);      printf("\n");
            }
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

bool Fish::LoadTrain(int flag) { 
    assert(hOPT!=nullptr);
    int64_t iter = hOPT->iter;  //     train->opt->iter;
    _INFO("%s: ......", __func__);

    auto fpCheck = hparams.save.checkpoint_in;
    bool isCopy = hparams.is({"wiki","actor"},"copy") && wikis.size()>0;
    if (fpCheck.empty()){
        if(wiki_tutor!=nullptr)
            return true;
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
    // _INFO("QKV version=%d\n",vQKV);
    
    // ggml_graph_stat(gf);
    // if(gb!=nullptr) ggml_graph_stat(gb);
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
    for (auto t : gensors.nag)        {
        auto type = t.second->type;
        if (ggml_is_quantized(type))
            nQ++;
        if (type == GGML_TYPE_F16)
            nF16++;
    }
    //  _INFO("%s cgraph(%d,%d) nQ=%d nF16=%d",__func__,cgraph->n_leafs,cgraph->n_nodes,nQ,nF16);
}

int Fish::BuildGraphFromRaw(int flag)   {
    int iRet = 0x0;
    bool isKeep = true;
    ctx_compute_params.mem_size = MostMemSize(0x0);
    // 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
    //         (hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
    ctx_build = ggml_init(ctx_compute_params);
    
    struct ggml_cgraph *gf = BuildRawGraph( ctx_build,false ),*gb=nullptr;
    // preLogits = gf->nodes[gf->n_nodes - 1]; // the output is always the last tensor in the graph
            
        // alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    iRet = BuildComputeGraph(0,ctx_build,0x0);
    
    
      
    return iRet;
}

// If isParam, only alloc grad, no init!
void Fish::InitGensor(struct ggml_context *ctx, const char *name, hGensor gensor, bool isParam, int flag)    {
    assert(gensor!=nullptr);
    if (name != nullptr){
        ggml_set_name(gensor, name);        //    gTN0(w,"%s.w",name.c_str());
    }
        
    assert(gensor->data == nullptr);
    gensors.Insert(gensor);
    if (isParam && isTrain())        {
        ggml_set_param(ctx, gensor);
        gTN(gensor,"");
        nParams += ggml_nelements(gensor);
        // assert(strlen(gensor->grad->name)>0);
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
        gensors.Insert(gensor);            
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
    // gensors.Insert(gg_tensor);
    InitGensor(ctx, key_.c_str(), gensor, isParam, 0x0);

    return gensor;   
}

#include "ggml-cuda.h"
#include "ggml-sycl.h"
#include "ggml-alloc.h"
extern "C" bool alloc_tensor_range(struct ggml_context * ctx,struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,ggml_backend_buffer_t ** buffers, size_t * n_buffers);
/*


*/
EDGE_DEVICES::EDGE_DEVICES(const CLI_params&hparams, int flag){
    // hFish = hF;
    assert(back_data==nullptr);
    assert(workers.size()==0);
    const size_t dev_count = 1; //ggml_backend_dev_count();
    _INFO("%s: %zu devices\n\n",__func__, dev_count);
    int n_ok=0,nT0=std::thread::hardware_concurrency(),nT=hparams.nThread();
    ggml_backend_t backend = nullptr;
    size_t free, total; 
    for (size_t i = 0; i < dev_count; ++i) {
#ifdef GG_V12  
        auto dev = ggml_backend_dev_get(i);
        devs.push_back(dev);
        ggml_backend_dev_memory(dev, &free, &total);
        printf("[EDGE_DEVICE]_%d %s:%s  memory: %zu MB (%zu MB free)\n", i, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev)
            ,total / 1024 / 1024, free / 1024 / 1024);   
        
        ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
#else
        backend = ggml_backend_cpu_init();
#endif
        assert(backend != NULL);
        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, nT);
            auto buft = ggml_backend_cpu_buffer_type();
            bufts.push_back(buft);
        } else {
            bufts.push_back(ggml_backend_get_default_buffer_type(backend));
        }
        workers.push_back(backend);
    }
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

    for (size_t i = 0; i < dev_count; ++i) {
        // Put the backend to be tested in front so that it's prioritized:
        std::vector<ggml_backend_t> backends_modded = {workers[i]};
        backends_modded.insert(backends_modded.end(), workers.begin(), workers.end());
/*
static size_t llama_model_max_nodes(const llama_model & model) {
    return std::max<size_t>(8192, model.tensors_by_name.size()*5);
}*/
        sched0 = ggml_backend_sched_new(
            backends_modded.data(), nullptr, backends_modded.size(), GGML_DEFAULT_GRAPH_SIZE/*2048*/, false);
        break;
        // std::pair<int, int> result = test_backend(backend_sched, backends[i]);
        // ggml_backend_sched_free(backend_sched);
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


EDGE_DEVICES::~EDGE_DEVICES(){
   for (auto backend : workers) {
        ggml_backend_free(backend); 
   }
   if(sched0!=nullptr)
        ggml_backend_sched_free(sched0);
    if(alloc_tmp!=nullptr)
            ggml_gallocr_free(alloc_tmp);
}

int EDGE_DEVICES::SetThread(int nThread,int flag)   {
    int nSet = 0;
    for(auto worker : workers){
        if (!ggml_backend_is_cpu(worker))   {
            ggml_backend_cpu_set_n_threads(worker, nThread);
            nSet ++;
            //ggml_backend_cpu_set_threadpool(hEDS->cpu, threadpool);
        // ggml_backend_cpu_set_abort_callback(hEDS->cpu, lctx.abort_callback, lctx.abort_callback_data);
        }
    }

#ifdef GGML_USE_BLAS
    if (lctx.backend_blas != nullptr) {
        ggml_backend_blas_set_n_threads(lctx.backend_blas, n_threads);
    }
#endif
    return 0x0;
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

bool alloc_tensor_range(struct ggml_context * ctx,
        struct ggml_tensor * first, struct ggml_tensor * last,
        ggml_backend_buffer_type_t buft, size_t size,
        ggml_backend_buffer_t ** buffers, size_t * n_buffers) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer == NULL) {
#ifndef NDEBUG
        _INFO("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
#endif
        for (size_t i = 0; i < *n_buffers; i++) {
            ggml_backend_buffer_free((*buffers)[i]);
        }
        free(*buffers);
        return false;
    }

    struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = first; t != last; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == NULL) {
            if (t->view_src == NULL) {
                ggml_tallocr_alloc(&tallocr, t);
            } else if (t->buffer == NULL) {
                ggml_backend_view_init(t);
            }
        } else {
            if (t->view_src != NULL && t->buffer == NULL) {
                // view of a pre-allocated tensor
                ggml_backend_view_init(t);
            }
        }
    }

    *buffers = (ggml_backend_buffer**)realloc(*buffers, sizeof(ggml_backend_buffer_t) * (*n_buffers + 1));
    (*buffers)[(*n_buffers)++] = buffer;

    return true;
}

bool EDGE_DEVICES::AllocGraph(hTGraph graph,int flag)    {
    bool bRet = false;    
    ggml_backend_sched_t backend_sched = GetSched();
    struct ggml_cgraph * cgraph = graph->raw();
    size_t nMost = cgraph->n_nodes + cgraph->n_leafs;
    ggml_backend_sched_reset(backend_sched); // clear allocation of previous graph
#ifdef GG_V12    
    // ggml_init_params params = {
    //         ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE,nullptr, true,
    // };
    // ggml_free(opt_ctx->ctx_copy);
    auto ctx_copy = InitCTX(0); //ggml_init({ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE,nullptr, true});    
    auto allocated_graph_copy = GG_dup_graph(ctx_copy, cgraph);
    bRet = ggml_backend_sched_alloc_graph(backend_sched, allocated_graph_copy);
#else
    if (flag==0x100) {  //measure_only
        assert(0);
        alloc_tmp = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        bRet = ggml_gallocr_reserve(alloc_tmp, cgraph);
    } else {
        /*  
            ggml_backend_sched_split_graph(sched, graph);
            if (!ggml_backend_sched_alloc_splits(sched)) {
                    ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)
                return false;
            }
            sched->is_alloc = true;
        */
        auto root = cgraph->nodes[0];
        // bRet = ggml_backend_sched_alloc_graph(backend_sched, cgraph);   //  crash @bitnet
        if(alloc_tmp!=nullptr)
            ggml_gallocr_free(alloc_tmp);
        alloc_tmp = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        bRet = ggml_gallocr_alloc_graph(alloc_tmp, cgraph);    //367,8088  
         
        ggml_backend_sched_reset(backend_sched);
    }
#endif
    return bRet;
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

bool Fish::ComputePlan(int flag) {
    auto& train_params = hparams.common;
    struct ggml_cgraph *cgraph = GetBackRaw();
    if(cgraph==nullptr){        //  OnlyInfer
        cgraph = hForwTG->raw();
    }
    gb_plan = ggml_graph_plan(cgraph, train_params.n_threads,nullptr);
    size_t max_work_size = gb_plan.work_size + GGML_OBJECT_MAX_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
// ggml_free(ctx_build);         ctx_build = nullptr;
    ctx_work = ggml_init({max_work_size,NULL,false});    
    struct ggml_object * obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    // gb_plan.work_data = (uint8_t *)ctx_work->mem_buffer + obj->offs;
    gb_plan.work_data = (uint8_t *)ggml_get_mem_buffer(ctx_work)+ obj->offs;
    gf_plan = gb_plan;      //  ???
    return true;
}

/**/
bool Fish::CopyGensors(hWIKI wiki,int flag)    {    
    _INFO("CopyGensors of %s ......hFish=%s",wiki->model_path.c_str(),name.c_str());
    int nT0 = wiki->tmaps.size(),nT1 = optParams.size(),nT2 = gensors.size(),x;
    size_t sz=0;

    if(nT0>nT2){
        return false;
    }
    for(auto it : wiki->tmaps){
        auto nam = it.first;
        hGensor src=it.second,dst = GetGensor(nam.c_str());
        size_t nElem = ggml_nelements(src),nbyte = ggml_nbytes(src);
        sz += nElem;
        if(strcmp(src->name,"blk.0.attn_q.weight")==0)   {   //only for debug
            x = 0;
        }
        if(dst==nullptr)
            return false;
        if(!ggml_are_same_shape(src,dst))
            return false;
        float *arr = (float*)(dst->data),a=arr[0];
        // _INFO("\t copy %s nbyte=%ld...\n",nam.c_str(),nbyte);
        // should replace by ggml_compute_forward_dup (memcpy only support CPU!)
        if(src->type==dst->type){
            memcpy(dst->data,src->data,nbyte);    
        }else if(dst->type==GGML_TYPE_F32)  {
            assert(ggml_is_quantized(src->type));
            Gensor2float_(src,(float*)(dst->data),flag);  
        }else{
            assert(0);
        }        
    }
    wiki_tutor = wiki.get();
    _INFO("\rCopyGensors of \"%s\" succeed!    N=%d sz=%.7gM \n",wiki->model_path.c_str(),nT0,sz/1.0e6);
    return true;
}