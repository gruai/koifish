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
        // if(!fish->ComputePlan())
        //     return nullptr;
        if(!wikis.empty()){  //generate some sentence
            CHILD_0909_WIKIS
            // fish->gopt = GeneratOnPrompt::MakeInstance(params,wikis,fish.get(),flag);        
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
    // if(!fish->ComputePlan())
    //     return nullptr;
    if(fish->isTrain()){  
        fish->gopt = GeneratOnPrompt::MakeInstance(params,fish->wikis,fish.get(),flag);        
    }else{
        
    }
    
    return fish;
}

bool Fish::ComputePlan(int flag) {
    auto& train_params = hparams.common;
    struct ggml_cgraph *cgraph = hBackTG->raw();
    if(cgraph==nullptr){        //  OnlyInfer
        cgraph = hForwTG->raw();
    }
    size_t max_work_size = ggml_graph_plan(cgraph, train_params.n_threads,nullptr).work_size + GGML_OBJECT_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
// ggml_free(ctx_build);         ctx_build = nullptr;
    // context for work buffer
    struct ggml_init_params ctx_work_params = {
        max_work_size, // mem_size
        NULL,          // mem_buffer    
        false,         // no_alloc
    };    

    ctx_work = ggml_init(ctx_work_params);
    gb_plan = ggml_graph_plan(cgraph, train_params.n_threads,nullptr);
    struct ggml_object * obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    gb_plan.work_data = (uint8_t *)ctx_work->mem_buffer + obj->offs;
    gf_plan = gb_plan;      //  ???
    return true;
}

bool Fish::AfterBuild(bool isInitParam,int flag)   {        
    int64_t nx = 0;
    auto gf = hForwTG->raw(),gb = hBackTG->raw();
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
            // wGensors.push_back(node);   
            if(isInitParam){
                if (rnd != nullptr)
                    randomize_tensor_normal(node, rnd);
                else
                    ggml_set_zero(node);                
            }
            _INFO("Param_%-4d(op=%d)\t", optParams.size(), gf->nodes[i]->grad->op );
#ifndef NDEBUG
            
#endif
            _pt_cys_("",node,0x0);         printf("\n");
            // gg_print_tensor_("",gf->nodes[i],0);
            nx += ggml_nelements(gf->nodes[i]);
        }
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
    szModel = ggml_used_mem(GetCTX()) + ggml_backend_buffer_get_size(back_data);   // (float) (ggml_used_mem(ctx) + ggml_backend_buffer_get_size(back_data)) ;
          
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
void Fish::SaveTrain(struct save_train_model * data, void *user_data,int flag) { 
    assert(hOPT!=nullptr);
    int64_t iter = hOPT->iter;  //     train->opt->iter;
    _INFO("%s: iter_%ld\n", __func__, iter);
    string sBaseName = get_train_filename(data->fn_model_out.c_str(), data->pattern_fn_it.c_str(), data->fn_latest.c_str(), -1  );
    if (strlen(data->fn_checkpoint_out.c_str()) > 0) {
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model, train);
        // save_checkpoint_file(get_train_filename(data->fn_checkpoint_out, data->pattern_fn_it, data->fn_latest, -1  ).c_str(), data->fn_model_base, data->model, train);
    }
    string sOut = "g_" + sBaseName; 
    if (strlen(data->fn_model_out.c_str()) > 0) {
        // save_llama_model_file(get_train_filename(data->fn_model_out, data->pattern_fn_it, data->fn_latest, iter).c_str(), data->fn_model_base, data->model);
        vendor = "gruai";                 //llm_arch_from_string
    }
    if(1){  //only for debug
        vendor = "llama";
        sOut = "l_" + sBaseName;     //hack  
    }
    save_gguf(sOut.c_str(),0x0);
    return;
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
    
    struct ggml_cgraph *gf = GetRawGraph( ctx_build,false ),*gb=nullptr;
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