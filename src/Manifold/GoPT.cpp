/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Generate some nonsense on Prompt
 *  \author Yingshi Chen
 */
#include "Optimizer.hpp"
#include "GoPT.hpp"
#include "Fish.hpp"
#include "gLLM.hpp"
#include <string>
#include <iostream>
#include <filesystem>
#include "../Utils/GST_rander.hpp"

namespace fs = std::filesystem;

#define LOG //

#ifdef __USE_GGML__
#else
std::vector<hWIKI> WIKI::MakeInstance(const std::string nam_,struct CLI_params& params,int flag){
    std::vector<hWIKI> wikis;
    if(params.tpWiki!="off") {//wiki is so heavy(ugly) that only load one instance here!
        for(auto path : params.fn_model_base){
            assert(0);
            // hWIKI wiki = std::make_shared<LAMA>(params,path);
            // wikis.push_back(wiki);
        }        
    }   
    return wikis;  
}
#endif

double WIKI::InductLogits(const CLI_params&config, int nSampInBatch,std::vector<TOKEN_ID>& tok_ids,struct ggml_tensor *target_probs,int flag) {
    if(!isInduct())
        return -1.0;
    
    Reset();         //Timing bottleneck!!! for the crazy design of llama.cpp
    Decode(tok_ids,0,0x0,true);    
    const float *all_logits = GetLogits(n_vocab,tok_ids.size(),0),*logit; 
    size_t k,j,i;    //exLogits->ne[0];  
    int n_ctx =config.n_ctx(),n_dialect=mapT2T.size(),token;  //target_probs->ne[1],
    double a1,a2,nrm=0;    
    float *p=teach == WIKI::_TARGET ? new float[n_vocab]:nullptr,*target=nullptr;  
    if(flag<0){    //CHILD_0909_WIKIS
        /*struct ggml_tensor * logits = userLogits==nullptr ? exLogits : userLogits;
        assert(logits!=nullptr);
        target = (float*)logits->data+nSampInBatch*n_ctx*ldL;        
        nrm =  NRM_2_(all_logits,n_ctx*ldL)/ldL;   
        if(logits->ne[0]==n_dialect){
            for(i=0; i<n_ctx; i++,target+=n_dialect,all_logits+=n_vocab){
                for(j=0;j<n_vocab;j++){
                    if(dialect[j]==0)       
                        continue;
                    token = mapT2T[j];
                    target[token] = all_logits[j];
                }                
            }
        }else*/{
            target = exLogits+nSampInBatch*n_ctx*n_vocab;        
            memcpy((void*)target,(void*)all_logits,sizeof(float)*n_ctx*n_vocab);       //memcpy(g->data+off,(void*)(logits),ld2); 
        }
    }else{    
        #ifdef __USE_GGML__
        for (k=0; k<nSampInBatch; ++k) {        
            const float *from=all_logits+k*n_vocab;
            a1=NRM_2_((float*)(from),n_ctx*n_vocab);          nrm=max(nrm,a1/n_vocab);     
            if(teach == WIKI::_TARGET){              
                assert(exLogits==nullptr);             
                for(j=0;j<n_ctx;j++){
                    logit = from+j*n_vocab;
                    target = (float*)target_probs->data+(k*n_ctx+j)*n_vocab;
                    //  SOFT_MAX_minus(n_vocab,target,logit);
                    // SOFT_MAX(n_vocab,p,logit);
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
                if(exLogits!=from){
                    // target = (float*)exLogits->data+k*n_ctx*n_vocab;          
                    target = (float*)exLogits+k*n_ctx*n_vocab;          
                    memcpy((void*)target,(void*)from,sizeof(float)*n_ctx*n_vocab);         
                }
                
            }
        }    
        #endif
    }
    delete[] p;
    return nrm;
}

static Grusoft::GRander rand_gopt(42*666);
int Sample_CDF_T(int n,float* logits,  float minp, float temperature, uint64_t *rng_seed,int flag=0x0) {
    float coin = rand_gopt.NextFloat_01();   //random_f32(rng_seed);
	// find max logit; we will use this to derive minp cutoff (in log space), since minp is scale-invariant (wrt softmax)
	float max_logit = -FLT_MAX;
	for (int i = 0; i < n; i++) {
		max_logit = logits[i] > max_logit ? logits[i] : max_logit;
	}

	// exp(logit / temp) <= exp(max_logit / temp) * minp -> logit <= max_logit + log(minp) * temp
	float logit_cutoff = max_logit + logf(minp) * temperature;

	// convert from logits to probabilities in-place while simultaneously doing (unscaled) softmax; we'll rescale later
	float* probs = logits;
	int fallback = 0;
	float cumulative_prob = 0.0f;
	for (int i = 0; i < n; i++) {
		if (logits[i] >= logit_cutoff) {
			probs[i] = expf((logits[i] - max_logit) / temperature);
			cumulative_prob += probs[i];
			fallback = i; // for fallback due to rounding errors
		} else {
			probs[i] = 0.0f;
		}
	}

	// sample from the truncated list
	float r = coin * cumulative_prob;
	float cdf = 0.0f;
	for (int i = 0; i < n; i++) {
		cdf += probs[i];
		if (r < cdf) {
			return i;
		}
	}
	return fallback; // in case of rounding errors
}

int Sample_CDF(int n,float*preP,uint64_t *rng_seed,int flag=0x0) {
    float sum=0,cdf=0,pMin,pMax,a;
    int j,next_token=-1;
    for (pMin=FLT_MAX,pMax=-FLT_MAX,j = 0; j < n; j++    ) {
        a = preP[j];
        pMin = min(a,pMin);     pMax = max(a,pMax);
    } 
    for (sum = 0, j = 0; j < n; j++)        {
        preP[j] = exp(preP[j]-pMax);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);
    float coin = rand_gopt.NextFloat_01();  //random_f32(rng_seed);
    for (cdf = 0, j = 0; j < n; j++)        {
        cdf += preP[j];
        if (coin < cdf / sum)            {
            next_token = j;
            break;
        }
    }
    return next_token;
}

int GGUF_list(CLI_params& config)  {
    std::vector<string> paths;
    std::string root = "/media/cys/E0/",path;
    root = "/home/cys/rnd/lic/models/";
    for (const auto & entry : fs::directory_iterator(root)){
         fs::path filePath = entry.path();
        if(filePath.extension() == ".gguf")
            paths.push_back(entry.path());
    }
    int nP = paths.size(),i=0;
    FILE *fp = fopen("./log/GGUF_list.log","wt");
    string sToken,info,suffix="\t",prefix;
    fprintf(fp,"%s LOAD %d @%s\n",__func__,nP,root.c_str());         fflush(fp);
    for(auto path : paths){
        // path = "/media/cys/E0/LLaMA3-8B_mmproj-Q4_1.gguf";      //only for debug
        auto param_1 = config;
        param_1.wiki_actor="OnlyTokenizer";     param_1.tpWiki=WIKI::_OFF;    
        // param_1.fn_model_base = path;
        
        // hFISH fish_0 = Fish::MakeInstance("GGUF_",param_1,0x0);
        // fish_0->Dump(0x0);
        GST_TIC(tic);
        fprintf(fp,"%d: \"%s\" ...\n",i++,path.c_str());         fflush(fp);
        try{
            hWIKI wiki = nullptr;       //  WIKI::MakeInstance
            assert(wiki!=nullptr);
            info = wiki==nullptr ? "" : wiki->__repr__(suffix,prefix);  
        }catch(const std::exception & e) {
            info =  std::string(e.what());
        }catch(...){
            info = "!!! UNKNOW EXCEPTION !!!";
        }
         
        fprintf(fp,"\t %s  T=%.3g\n",info.c_str(),GST_TOC(tic));    
        fflush(fp);
    }
    fclose(fp);
    return 0x0;
}
    
int run_caml(const char*prompt,int flag);

int Fish_bubble(CLI_params& config)  {  
    g_dump_level = 0;  
    config.wiki_actor = "copy";
    config.common.n_batch = 1;
    config.model.preLogits_dB = 1;
    DEBUG.graph_dump = 1;
    DEBUG.T_cuda_ver = 1;
    DEBUG.T_cpu = 0;
    // GTensor::tpPreLogits = typNUMBER::F32;

    arrHWIKI wikis = WIKI::MakeInstance("wikis",config,0x0);
#if !defined(NDEBUG)
    // config.common.n_ctx = 17;     config.common.n_batch = 1;      config.nLayerX=1;      //Only for debug
    // config.set();
#endif
    config.isOnlyGPT = true;
    hFISH fish = Fish::MakeInstance("BUBBLE_",config,wikis,Fish::ROLE_TYPE::COMMON,0x110);
    fish->GenSentence();
    return 666;
}

int fish_1(CLI_params& config)  {
    auto param_1 = config,param_2 = config;
    param_1.tpWiki = "logits";              param_1.common.n_batch = 1;
    param_2.tpWiki = "";                    param_2.common.n_batch = 1;    // 
    hFISH fish_0 = Fish::MakeInstance("BIG_",param_1,{},Fish::ROLE_TYPE::COMMON,0x0);
    hFISH fish_1 = Fish::MakeInstance("local_",param_2,fish_0.get(),0x110);
    fish_0->Dump(0x0);
    fish_1->Dump(0x0);

    vector<float> logits;           
    vector<TOKEN_ID> piffle;    
    // Need piffle to samploader
    fish_0->LocalFeeling(nullptr,logits); 
    fish_1->CopyWeight(fish_0.get());
    fish_1->LocalFeeling(nullptr,logits);
    return 666;
}

bool _LoadCheckPoint(CLI_params& config,arrHWIKI& wikis,int flag=0x0){ 
    if (config.checkpoint.in.empty())
        return false;
               
    hFISH fish = Fish::MakeInstance("Fish_",config,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
    if(fish==nullptr || !fish->LoadCheckPoint())
        return false;
    return true;
}

int GPT_work(CLI_params& config)  {
    //  GRUS_Get_SystemInfo
    _INFO("[%s] threads=%d \n%s\n",__func__,std::thread::hardware_concurrency(),"");    //llama_print_system_info()
    // ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);    
    DEBUG.SelfAttention_noraml = 0;
    DEBUG.NO_loss = true;
    DEBUG.graph_dump = 1;
    // config.wiki_actor="";    //only for debug

    config.isOnlyGPT = true;
    config.passLoadToken = true;
    bool isMakeFish = config.is({"wiki","actor"},"copy") || config.wiki_actor=="OnlyTokenizer";
    hFISH fish = nullptr;
    vector<hWIKI> wikis = WIKI::MakeInstance("",config,0x0);          
    if(config.fn_model_base.size()>0 && !isMakeFish){
        for(auto wiki : wikis){
            if(wiki->isOnlyTokenizer)
                assert( wiki->teach == WIKI::_OFF );
            else
                wiki->teach = WIKI::_LOGITS;
        }                     
    }

    if(isMakeFish){
        fish = Fish::MakeInstance("Fish_",config,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
        if(fish==nullptr || !fish->LoadCheckPoint()){
            _ERROR("%s has no WIKI or FISH!\n",__func__);
            return 0;
        }            
    }
    //  hGOPT gpt = std::make_shared<GeneratOnPrompt>(params,0x0);
    //  hGOPT gpt = std::make_shared<GOPT_infinite>(params,0x0); 
    hGOPT gpt = std::make_shared<GOPT_Metropolis>(config,wikis,fish.get(), 0x0);
    if (gpt->Init(config.prompt))    { 
        for (int i = 0; i < 10; i++)        { 
            if(!wikis.empty())
                wikis[0]->Reset();        //to get same results each run     
            gpt->Generate(i);
            // break;
        } 
    }   else    {
        return -1;
    }        
    
    return 666;
}

hGOPT GeneratOnPrompt::MakeInstance(struct CLI_params& config,arrHWIKI& wikis,const Fish *fish_0,int flag) {
    // gopt = std::make_shared<GeneratOnPrompt>(wiki,this,0x0);
    hGOPT gopt = std::make_shared<GOPT_Metropolis>(config,wikis,fish_0,0x0);
    if(gopt!=nullptr && gopt->Init(config.prompt)){
        // gopt->Generate(0); //only for debug  
    }else{
        gopt.reset();       gopt = nullptr;
    }   
    const char* promt = "when the smoke is going down,"; 
    std::vector<TOKEN_ID> some_inp = {9493,279,16603,374,2133,1523,11};      //const char* promt = "when the smoke is going down,";      
    
    // wiki->Decode(embd_inp,0,0x0);
    // wiki->Answer(embd_inp);      //only for debug
    return gopt;
}

std::string GeneratOnPrompt::GetPrompt(int flag) {   
    return config.prompt;  
} 

/*
    1.   prompt=>embd_inp
    2.  =>batch@Decode  =>batch@llama_decode
    3.  =>ubatch    then llama_build_graph

        
*/
void GeneratOnPrompt::InitInput(int flag){
    // if (params.chatml) {
    //     GetPrompt() = "<|im_start|>system\n" + GetPrompt() + "<|im_end|>";
    // }
    _INFO("[GPT] tokenize the prompt\n");    
    hGensor input = fish_1!=nullptr? fish_1->Input() : nullptr;
    TOKEN_ID bo=-1;
    if(input!=nullptr)
        dialogs->InitOneSamp(config.prompt,input,nullptr,0x110);
    switch(_arch)    {  
    case NLP_GPT2_char:        
        
        // embd_inp.clear();
        // embd_inp.push_back(0);
        // for(int i=0;i<GetPrompt().length();i++){
        //     int id = (int)(GetPrompt()[i]);       
        //     embd_inp.push_back(id);
        // }
        break;

    default:{
        int iRet =wiki0->STR2T(GetPrompt(),embd_inp);
        // const bool add_bos = llama_add_bos_token(model);
        // embd_inp = ::llama_tokenize(ctx, GetPrompt(), add_bos, true);
        bo = embd_inp[0];
        }
        break;
    }
    
    return;
}

GeneratOnPrompt::GeneratOnPrompt(struct gpt_params &par_, int flag) {    
    /*LOG("%s logits_all=%d\n", __func__,params.logits_all );
    llama_numa_init(params.numa);
    // prompt = GetPrompt();

    params.sparams.temp = 0.0;
    params.sparams.temp = 0.8;
    sparams = params.sparams;
    // compatible with LLAMA.cpp
    config.fn_model_base.push_back( params.model ); 
    n_predict = params.n_predict;*/
}

void GeneratOnPrompt::Clear()    {
    // write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);
#ifdef __USE_GGML__
    if (ctx_guidance) { llama_free(ctx_guidance); }
    if(wikis.empty()){
        llama_free(ctx);        
        // llama_free_model(model);            
    }

    llama_backend_free();
#endif

    FREE_a(_logits);
}
          //only for debug
GeneratOnPrompt::GeneratOnPrompt(CLI_params&cp_,arrHWIKI& wiki_, const Fish *hG_, int flag) : 
    config(cp_),fish_0(hG_), wikis(wiki_) {
    if(fish_0!=nullptr){
        auto gang_param = config;
        gang_param.tpWiki = "off";      assert(gang_param.tpWiki=="off");
        gang_param.common.n_batch = 1;
        fish_1 = (Fish*)fish_0;
        // fish_1 = Fish::MakeInstance("4GPT_",gang_param,wikis,Fish::ROLE_TYPE::SWARM_FOLLOWER,0x110);        //  isLocalInfer = flag==0x110;
        // fish_1->Dump(0x0);

        _arch = fish_0->arch;         
    }else{
        _arch = config.ModelArch();
    }
    if(!wikis.empty())
        wiki0 = wikis[0];
}

bool GeneratOnPrompt::Init(const std::string &prompt_, int flag)    {
    // std::tie(model, ctx) = llama_init_from_gpt_params(params);
    int n_vocab = 0;
    if(fish_1!=nullptr) {
        dialogs = std::make_shared<SampLoader>(fish_1,"gpt", true);       
        dialogs->Prepare(fish_1->hOPT.get(), fish_1->tsEval[0]); 
        dialogs->isRecycle = false;
        dialogs->type = SampLoader::TYPE::DT_EVAL;       
        n_vocab = fish_1->nClass();   
    }
    
    if (wikis.empty()){     CHILD_0909_WIKIS
        n_ctx = config.n_ctx();        
        _logits = new float[n_vocab];
        // dialogs->init(config.prompt.c_str(), B, T, 0, 1, 0);

        InitInput();
        return true;
    }else{ 
        n_ctx = wiki0->nCTX();     
        n_vocab = wiki0->n_vocab;  
    }
    _logits = new float[n_vocab];

    /*llama_backend_init(); // ggml_time_init(); 
    LAMA *lama = dynamic_cast<LAMA *>(wikis[0].get());
    if(lama==nullptr || !lama->isValid())
        return false;
    model = lama->lmodel;    
    int nTokens = llama_n_vocab(model), j;
    eos = llama_token_eos(model);       bos = llama_token_bos(model);
    _logits = new float[nTokens];
    assert(model != nullptr);
    ctx = lama->_ctx;
    assert(ctx != nullptr);
    if (model == NULL)    {
        LOG("%s: error: unable to load model\n", __func__);
        return false;
    }

    n_ctx_train = llama_n_ctx_train(model);
    n_ctx = llama_n_ctx(ctx);*/
    LOG("n_ctx: %d(%d)\n", n_ctx, n_ctx_train);
    if (n_ctx > n_ctx_train)    {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n",__func__, n_ctx_train, n_ctx);
    }
    LOG("\n");

    // prompt = prompt_;
    if (GetPrompt() != "")    {
        // GetPrompt() = prompt;
        Tokenize(flag);
    }

    // ga_n = params.grp_attn_n;               ga_w = params.grp_attn_w;
    // if (ga_n != 1)    {
    //     assert(ga_n > 0 && "grp_attn_n must be positive");                         // NOLINT
    //     assert(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
    //                                                                                     // assert(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
    //     // assert(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
    //     LOG("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    // }
    // LOG("\n\n");

    return true;
}

void GeneratOnPrompt::DisplayEmbd(bool input_echo, int n_consumed, int flag){
    NLP_AutoRegressive *dolphin = dynamic_cast<NLP_AutoRegressive *>(fish_1);
    std::string token_str;
    if (input_echo && display)    {
        for (auto id : tokens)        {            
            // switch(_arch)    {  
            // case NLP_GPT2_char:
            //     token_str = (char)(id); 
            //     break;
            // default:
            //     token_str = llama_token_to_piece(ctx, id);
            //     break;
            // }
            
            if (tokens.size() > 1)            {
                input_tokens.push_back(id);
            }            else            {
                output_tokens.push_back(id);
                output_ss << token_str;
            }
        }
        if(dolphin!=nullptr)    {
            _INFO("[Generate]_%d {%s}%s",tokens.size(),dialogs->sentence.c_str(),token_str.c_str());
        }else{
            // token_str = llama_token_to_piece(ctx, tokens[0]);
            printf("%s", token_str.c_str());
        }
            
        fflush(stdout);
    }
    // reset color to default if there is no pending user input
    if (input_echo && (int)embd_inp.size() == n_consumed)
    {
        // console::set_display(console::reset);
        display = true;
    }
}

/**/
int NLP_AutoRegressive::GenSentence(int flag)  {
    if(hOPT==nullptr)
        return -1;
    GST_TIC(tic);    
    if(gopt!=nullptr){
        hWIKI wiki = wikis[0];           assert(wiki != nullptr);
        wiki->Reset();
        return gopt->Generate(0x0);
    }
    uint64_t rng_seed = 42;
    std::string prompt = config.prompt;
    // prompt = LoadSomeText("config.fp_train_data",0x0);
    int genT = 64, nVocab = config.model.vocab_size, _nctx = config.n_ctx(), i, j;
    assert(genT <= _nctx);    
    double sum = 0, cdf = 0, tps=0,t0=GST_ms();
    hSampLoader hLoader = hOPT->val_loaders[0];
    if(hLoader->num_batches<=0 )    {
        hLoader->InitOneSamp(prompt,nullptr,this,0x110);        
        // hLoader->isRecycle = false;            hLoader->isNeedBOS = false;     
        // hLoader->UpdateBatch(0,this);
    } 
    // TokenEmbed* embed = GetNeuron<TokenEmbed>("TokenEmbed");    
    // embed->hBatch = hLoader->hBatch;        
    int nPrompToken = hLoader->nMostToken;
    // vector<TOKEN_ID>& piffle = hLoader->GetTokens( ),answer;
    // int nPrompToken = piffle.size();
    // nPrompToken = 1;			//	only for debug
    TOKEN_ID t = -1;
    _INFO("%s: <--- \n\t", __func__);
    float *logits = hCLS->Logits(true); //(float *)(preLogits->data)+i*nVocab;
    for (i = 0; i < nPrompToken+genT; i++)    {
        if(i<nPrompToken-1)   
            hOPT->SetPhase(Optimizer::P_PREFILL);
        else
            hOPT->SetPhase(Optimizer::P_GENERATE);
        // // LocalFeeling(piffle,preP);
        float fLos = hOPT->Evaluate(hLoader,-666);
        if(i<nPrompToken-1)   
            continue;        
        
        //t = Sample_CDF(nVocab,logits,&rng_seed);
        t = Sample_CDF_T(nVocab,logits,0.1,1,&rng_seed);
        hLoader->hBatch->Set(i+1,0,0,0,t);
        // piffle[i] = t;      answer.push_back(t); 
        string a = hDict->T2STR(hLoader->hBatch->host,i+2),b=hLoader->sentence,s=hDict->T2STR(t);   
        // _INFO("\r\t%s\t(%.*s)",a.c_str(),64,b.c_str());      
        _INFO("%s",s.c_str());   
        fflush(stdout);
        if (t == hDict->bos_id || t == hDict->eos_id || t == hDict->eot_id) {
            break;
        }
    }
    tps = genT/(GST_ms()-t0)*1000.0;
    _INFO("\n[Generate]---> tps=%g tAll=%gs\n", tps, GST_TOC(tic));
    return 0x0;
}

/*
    when the smoke is going down, that's when it's like, 'Abby, you really need to go to the doctor,' this was not the case with me or my son.
        We were between his second and third birthday and it was time to go to the doctor and not go to the doctor--and that was the reason he was going to the doctor and not going to the doctor.
        On the two sides of the age there was a lot of room for two oth
*/
TOKEN_ID GOPT_Metropolis::Sample(int idx, bool is_resampling)    {
    int j,nVocab = fish_1==nullptr ? wiki0->n_vocab : fish_1->nClass();   //, j;
    hSAMP samp = (dialogs==nullptr||dialogs->empty()) ? nullptr : dialogs->SampAt(0);
    
    hWIKI wiki = wikis.size()>0  ? wikis[0] : nullptr;
    WIKI::INDUCT_MODE teach = wiki==nullptr ? WIKI::_OFF : wiki->teach;    
    assert(idx==-1);
    const float *wLog = nullptr;
    float l1=0,sum1=0,l2=0,delta,a;   
    if(teach==WIKI::_OFF)  {

    }else{
        wLog = wiki->GetLogits(nVocab,1);   
        for (l2 = 0,j = 0; j < nVocab; j++    ) {   //  -8.23 -1.1958
            a = wLog[j];
            l2 += a*a;              _logits[j] = a;
            // pMin = min(a,pMin);     pMax = max(a,pMax);
        }        
    }

    if (fish_1 != nullptr)    {   //time bottleneck, so share wiki to reduce time & memory
        // SOFT_MAX(nTokens,_logits,wLog);     //soft merge ???        
        /*fish_1->UpdateNCTX(dialogs->nLeastCTX());
        fish_1->CopyWeight(fish_0);
        if (fish_1->LocalFeeling(&dialogs, x_logits,0))        {*/
            assert(x_logits.size()==nVocab);
            // SOFT_MAX(x_logits);
            switch(teach){
            case WIKI::_OFF:
                for (j = 0; j < nVocab; j++    ) {
                    a = x_logits[j];       
                    sum1 += a;   l1+=a*a;     
                    _logits[j] = a;
                }
                break;
            /*case WIKI::_LOGITS:
                for (j = 0; j < nTokens; j++    ) {
                    a = x_logits[j];                    l1+=a*a;        
                    _logits[j] = a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2)-1.0;
            break;*/
            case WIKI::_TARGET:
                SOFT_MAX(nVocab,_logits,wLog); 
                for (j = 0; j < nVocab; j++    ) {
                    a = x_logits[j];                    l1+=a*a;        
                    _logits[j] += a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2)-1.0;
            break;
            case WIKI::_LOGITS_SCALE:
                for (j = 0; j < nVocab; j++    ) {
                    a = x_logits[j];        
                    sum1 += a;              
                    l1+=a*a;     l2+=_logits[j]*_logits[j];
                    // a = max(a,0.f);      relu
                    _logits[j] *= a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2);  
                break;
            default:    //  WIKI::_LOGITS:
                for (j = 0; j < nVocab; j++    ) {
                    a = x_logits[j];        
                    sum1 += a;              
                    l1+=a*a;     l2+=_logits[j]*_logits[j];
                    _logits[j] += a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2);  
                // assert(fabs(sum1-1.0)<0.001);       //softmax->logits
            }
            delta_a += delta;
            delta_max = max(delta_max,delta);                       
        
    }
    int next_token = Sample_CDF(nVocab,_logits,&rng_state);    
    return next_token;
}

TOKEN_ID GeneratOnPrompt::Sample(int idx, bool is_resampling)    {
    TOKEN_ID id = 0;
    /*const llama_sampling_params &params = ctx_sampling->params;
    const float temp = params.temp, mirostat_tau = params.mirostat_tau, mirostat_eta = params.mirostat_eta;
    const int mirostat = params.mirostat;
    // const float         temp              = params.temp;
    const float dynatemp_range = params.dynatemp_range;
    const float dynatemp_exponent = params.dynatemp_exponent;
    const int32_t top_k = params.top_k;
    const float top_p = params.top_p;
    const float min_p = params.min_p;
    const float tfs_z = params.tfs_z;
    const float typical_p = params.typical_p;
    const std::vector<llama_sampler_type> &samplers_sequence = params.samplers_sequence;

    
    std::vector<float> original_logits, x_logits;
    auto ctx_main = ctx, ctx_cfg = ctx_guidance;
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, !is_resampling, &original_logits);
    if (!is_resampling)    {
        assert(!original_logits.empty());
    }
    if (fish_1 != nullptr)    {   //time bottleneck
        x_logits.resize(original_logits.size());
        if (fish_1->LocalFeeling(&dialogs, x_logits))        {
            std::transform(original_logits.begin(), original_logits.end(), x_logits.begin(),
                           original_logits.begin(), std::plus<float>());
        }
    }

    // Get a pointer to the logits
    float *logits = llama_get_logits_ith(ctx_main, idx);

    if (temp < 0.0)
    {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    }
    else if (temp == 0.0)
    {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    }
    else
    {
        if (mirostat == 1)
        {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        }
        else if (mirostat == 2)
        {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        }
        else
        {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            // sampler_queue(ctx_main, params, cur_p, min_keep);

            for (auto sampler_type : samplers_sequence)
            {
                switch (sampler_type)
                {
                case llama_sampler_type::TOP_K:
                    llama_sample_top_k(ctx_main, &cur_p, top_k, min_keep);
                    break;
                case llama_sampler_type::TFS_Z:
                    llama_sample_tail_free(ctx_main, &cur_p, tfs_z, min_keep);
                    break;
                case llama_sampler_type::TYPICAL_P:
                    llama_sample_typical(ctx_main, &cur_p, typical_p, min_keep);
                    break;
                case llama_sampler_type::TOP_P:
                    llama_sample_top_p(ctx_main, &cur_p, top_p, min_keep);
                    break;
                case llama_sampler_type::MIN_P:
                    llama_sample_min_p(ctx_main, &cur_p, min_p, min_keep);
                    break;
                case llama_sampler_type::TEMPERATURE:
                    if (dynatemp_range > 0)
                    {
                        float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                        float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                        llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                    }
                    else
                    {
                        llama_sample_temp(ctx_main, &cur_p, temp);
                    }
                    break;
                default:
                    break;
                }
            }

            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

            //{
            //    const int n_top = 10;
            //    LOG("top %d candidates:\n", n_top);

            //    for (int i = 0; i < n_top; i++) {
            //        const TOKEN_ID id = cur_p.data[i].id;
            //        (void)id; // To avoid a warning that id is unused when logging is disabled.
            //        LOG(" - %5d: '%12s' (%.3f)\n", id, llama_token_to_piece(ctx_main, id).c_str(), cur_p.data[i].p);
            //    }
            //}

            // LOG("sampled token: %5d: '%s'\n", id, llama_token_to_piece(ctx_main, id).c_str());
        }
    }

    if (ctx_sampling->grammar != NULL && !is_resampling)
    {
        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = {&single_token_data, 1, false};

        // Apply grammar constraints to the single token
        llama_sample_grammar(ctx_main, &single_token_data_array, ctx_sampling->grammar);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid)
        {
            assert(0);
            
            // Pass true for is_resampling
        }
    }

    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;*/

    return id;
}

void GeneratOnPrompt::OnAntiPrompt(int flag){
    /*if (!params.antiprompt.empty())
    {
        const int n_prev = 32;
        const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        // If we're not running interactively, the reverse prompt might be tokenized with some following characters
        // so we'll compensate for that by widening the search window a bit.
        for (std::string &antiprompt : params.antiprompt)
        {
            size_t extra_padding = params.interactive ? 0 : 2;
            size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                          ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                          : 0;

            if (last_output.find(antiprompt, search_start_pos) != std::string::npos)
            {
                if (params.interactive)
                {
                    is_interacting = true;
                }
                is_antiprompt = true;
                break;
            }
        }

        // check for reverse prompt using special tokens
        TOKEN_ID last_token = llama_sampling_last(ctx_sampling);
        for (std::vector<TOKEN_ID> ids : antiprompt_ids)
        {
            if (ids.size() == 1 && last_token == ids[0])
            {
                if (params.interactive)
                {
                    is_interacting = true;
                }
                is_antiprompt = true;
                break;
            }
        }

        if (is_antiprompt)
        {
            LOG("found antiprompt: %s\n", last_output.c_str());
        }
    }*/
}

std::string GeneratOnPrompt::T2STR(TOKEN_ID tok,int flag){
    NLP_AutoRegressive *dolphin = dynamic_cast<NLP_AutoRegressive *>(fish_1);
    std::string token_str;
    if(dolphin!=nullptr)
        token_str = dolphin->hDictVAE->T2STR(tok);
    else{
        token_str = wikis[0]->T2STR(tok);
    }
    return token_str;
}

bool GeneratOnPrompt::Inference(hSAMP samp,int& n_past,int flag)   {
    int n_eval = (int)tokens.size();
    // LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());
    bool bRet = false;
    if(fish_1!=nullptr){
        fish_1->UpdateNCTX(dialogs->nLeastCTX());
        fish_1->CopyWeight(fish_0);
        bRet = fish_1->LocalFeeling(dialogs, x_logits,0);      assert(bRet);
    }
    /*if(flag==0x100)     {   //  debug flag of "only fish"
        // vector<float> preP;
        // float fLos = fish_1->LocalFeeling(&dialogs,preP);
        // bRet = true;
    }else*/ if(wiki0!=nullptr && wiki0->teach!=WIKI::_OFF ){
        bRet = wiki0->Decode(tokens, 0, n_past,false);
        // bDecode = llama_decode(ctx, llama_batch_get_one(&tokens[0], n_eval, n_past, 0)) >= 0;
    }

    if (!bRet)            {
        LOG("%s_%d : failed @%s\n", __func__,samp->len,dialogs->sentence.c_str());
        return 1;
    }
    n_past += n_eval;
    // LOG("n_past = %d n_remain=%d\n", n_past,n_remain);
    return bRet;
}

int GeneratOnPrompt::Generate(int nJob, int flag)   {
    g_dump_level = 1;
    
    GST_TIC(tic);
    hSAMP samp = (dialogs==nullptr||dialogs->empty()) ? nullptr : dialogs->SampAt(0);
    output_tokens.clear();
    delta_max = 0;      delta_a = 0;
    string info = "only fish",sTok;
    if(wiki0!=nullptr)  {
        info = wiki0->teach==WIKI::_OFF ? "only fish":"WIKI"; 
        // ctx_sampling = nullptr;
    }   
    LOG("<--- GeneratOnPrompt %s job=%d logits_all=%d fish=%s teach=%d\n", info.c_str(),
        nJob,0,fish_1==nullptr?"":fish_1->Name().c_str(),wiki0==nullptr? -1 : wiki0->teach);
    rng_state = config.common.seed;
    // LOG("%s logits_all=%d\n", __func__, );  
    // bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past = 0,n_remain = n_predict,n_session_consumed = 0,ga_i = 0,max_embd_size = n_ctx-4;
    tokens.clear();                 // embd_guidance.clear();
    LOG("embd_inp.size(): %d \n", (int)embd_inp.size());
    tokens = embd_inp;
    _INFO("%s",config.prompt.c_str());
    
    while ((--n_remain >= 0 && !is_antiprompt) )    {
        if (tokens.empty())    
            break;        
        assert ((int)tokens.size() <= max_embd_size );
        // assert(ga_n == 1);
        // assert(ctx_guidance == nullptr);
        if(!Inference(samp,n_past))
            return 1;        
        /*// for (int i = 0; i < (int)tokens.size(); i += params.n_batch)        {
            int n_eval = (int)tokens.size();
            // LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());
            bool bDecode = false;
            if(flag==0x100)     {   //  "only fish"
                vector<float> preP;
                float fLos = fish_1->LocalFeeling(&dialogs,preP);
                bDecode = true;
            }else if(wiki!=nullptr){
                bDecode = wiki->Decode(tokens, 0, n_past,false);
            }else{
                // bDecode = llama_decode(ctx, llama_batch_get_one(&tokens[0], n_eval, n_past, 0)) >= 0;
            }
            if (!bDecode)            {
                LOG("%s : failed to eval\n", __func__);
                return 1;
            }
            n_past += n_eval;
            LOG("n_past = %d n_remain=%d\n", n_past,n_remain);*/
            // if (params.n_print > 0 && n_past % params.n_print == 0)                { // Display total tokens alongside total time
            //     LOG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
            // }
        // }

        if (!tokens.empty() )            {
            session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
            n_session_consumed = session_tokens.size();
        }
        
        tokens.clear();     //  kv cache only need one token
        TOKEN_ID tok = Sample(); // llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
        if(tok<0){
            _INFO("\t<E>");                break;
        }
        if(samp!=nullptr)   {
            dialogs->hTokens->Append(tok);        samp->len++;
        }
        sTok = T2STR(tok);
        if(samp!=nullptr)
            _INFO("[Generate]_%d {%s}%s\n",samp->len,dialogs->sentence.c_str(),sTok.c_str());
        else
        {   _INFO("%s",sTok.c_str());             fflush(stdout);       }
        tokens.push_back(tok);             
        
        // if not currently processing queued inputs;
        // if ((int)embd_inp.size() <= n_consumed)        {            // check for reverse prompt in the last n_prev tokens
        //     OnAntiPrompt(0x0);
        //     // if(ctx_sampling!=nullptr)   OnInteractive(n_past, n_consumed, n_remain, 0x0);
        // }
        // end of text token
        if (tok==eos    /*!tokens.empty() && tokens.back() == eos*/ )        {
            LOG(" [end of text]\n");
            break;
        }
    }
    
    _INFO("\n delta=%.3g(%.3g) T=%gs --------------->\n",delta_max,delta_a/n_predict, GST_TOC(tic));
    return 0x0;
}

int GeneratOnPrompt::Generate_v0(int nJob, int flag)   {
    
    return 0x0;
}

// #include "../../../llama.cpp/common/GG_dup_graph"
int GeneratOnPrompt::Tokenize(int flag) {
    // auto& embd_inp = prompt
    // const int n_ctx = llama_n_ctx(ctx);       
    if (!GetPrompt().empty()) {
        InitInput();
    } else {
        LOG("use session tokens\n");
        embd_inp = session_tokens;
    }
    assert (!embd_inp.empty());
    n_keep = (int)embd_inp.size();
    // LOG("prompt: \"%s\"\n", log_tostr(GetPrompt()));
    // LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    
    // negative prompt
    std::vector<TOKEN_ID> guidance_inp;
    int guidance_offset = 0;
    int original_prompt_len = 0;
    
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;   

    // LOGLN(
    //         "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu, embd_inp.size() %zu",
    //         log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        // LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context

    /*    // prefix & suffix for instruct mode
    inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", add_bos, true);
    inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n",    false,   true);

    LOG("inp_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_pfx).c_str());
    LOG("inp_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, inp_sfx).c_str());

    // chatml prefix & suffix
    cml_pfx = ::llama_tokenize(ctx, "\n<|im_start|>user\n", add_bos, true);
    cml_sfx = ::llama_tokenize(ctx, "<|im_end|>\n<|im_start|>assistant\n", false, true);

    LOG("cml_pfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_pfx).c_str());
    LOG("cml_sfx: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, cml_sfx).c_str());

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG("\n");
        LOG("%s: prompt: '%s'\n", __func__, GetPrompt().c_str());
        LOG("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (ctx_guidance) {
            LOG("\n");
            LOG("%s: negative prompt: '%s'\n", __func__, sparams.cfg_negative_prompt.c_str());
            LOG("%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int) guidance_inp.size(); i++) {
                LOG("%6d -> '%s'\n", guidance_inp[i], llama_token_to_piece(ctx, guidance_inp[i]).c_str());
            }
        }

        if (params.n_keep > add_bos) {
            LOG("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG("'\n");
        }
        LOG("\n");
    }
    LOG("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, 
        params.n_batch, params.n_predict, params.n_keep);*/
    // Handle_CtrlC();
    return 0x0;
}
