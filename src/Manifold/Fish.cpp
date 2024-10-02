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
    switch(params.arch){
    case MODEL_ARCH::NLP_MAMBA:
        fish = std::make_shared<LLM_MAMBA>(nam_+"_mamba",params,role_);
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
            fish = std::make_shared<LLaMeta>(nam_,params,role_);
            break;
        case 3:
            fish = std::make_shared<LLAMA_VAE>(nam_+"_vae",params,role_);
            break;    
        default:
            assert(0);
        }     
    }
    
    fish->isLocalInfer = flag==0x110;   
    fish->Init( wikis );    
    fish->Build( );
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
        fish = std::make_shared<LLaMeta>(nam_,params,role);
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
    struct ggml_cgraph *cgraph = gb;
    if(gb==nullptr){        //  OnlyInfer
        cgraph = gf;
    }
    size_t max_work_size = ggml_graph_plan(cgraph, train_params.n_threads,nullptr).work_size + GGML_OBJECT_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float) max_work_size / (1024.0f*1024.0f));
// ggml_free(ctx_compute);         ctx_compute = nullptr;
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

void Fish::AfterBuild(int flag)   {
    int64_t nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->flags & GGML_TENSOR_FLAG_PARAM) {
            GGML_PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);                
            optParams.push_back(gf->nodes[i]);
            // opt_ps[np++] = gf->nodes[i];
            _INFO("%4d(op=%d)\t", optParams.size(), gf->nodes[i]->grad->op );
            gg_print_tensor_("",gf->nodes[i],0);
            nx += ggml_nelements(gf->nodes[i]);
        }
    }            
    GGML_ASSERT(optParams.size() < GGML_MAX_PARAMS);

    if(role==SWARM_FOLLOWER){
        
    }else{
        
    }
}

void Fish::BeforeBuild(int flag)   {
    if(role==SWARM_HEAD){
        assert(swarm.size()>0);
    }else{
        
    }
}

bool Fish::OnTrainStep(struct train_opt_callback_data *data,SampLoader&loader, int accum_step, float *sched, int flag)    {
    LossCurve(0x0);
    assert(0);
    return false;
}

static const char * vendor = "gruai";     //llm_arch_from_string
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

int Fish::BuildGraphFromRaw(int flag)   {
    int iRet = 0x0;
    bool isKeep = true;
    ctx_compute_params.mem_size = 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
            (hparams.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
    ctx_compute = ggml_init(ctx_compute_params);
    gf= ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);        //ggml_hash_set_reset(&cgraph->visited_hash_set);    
    // GetRawGraph(&gf);
    ggml_graph_print(gf);


    
    set<std::string> leaf_const={"inp_tokens"};
    for (int i = 0; i < gf->n_leafs; i++) { //to float
        struct ggml_tensor * node = gf->leafs[i];
        if(strstr(node->name,"weight")==NULL)        
            continue;        
        wGensors.push_back(node);
    }    
    // gf->order = GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT;      
    if(!isLocalInfer){
        gb = ggml_new_graph_custom(ctx_compute, LLAMA_TRAIN_MAX_NODES, true);
    }
    if(gb!=nullptr){
        ggml_graph_cpy(gf, gb);        
        if (isKeep) {
            for (int i = 0; i < gf->n_nodes; i++) {
                struct ggml_tensor * node = gf->nodes[i];
                if (node->grad) {
                    node->grad = ggml_dup_tensor(ctx_compute, node);
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
                // ggml_compute_backward(ctx_compute, node, &zero_table);
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
      
    return iRet;
}