#include "console.h"
#include "GoPT.hpp"
#include "Fish.hpp"
#include "gLLM.hpp"
#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

static unsigned int random_u32(uint64_t *state){
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0,1)
float random_f32(uint64_t *state)   { 
    return (random_u32(state) >> 8) / 16777216.0f;
}

int Sample_CDF(int n,float*preP,uint64_t *rng_seed,int flag=0x0) {
    float sum=0,cdf=0;
    int j,next_token=-1;
    for (sum = 0, j = 0; j < n; j++)        {
        preP[j] = exp(preP[j]);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);
    float coin = random_f32(rng_seed);
    for (cdf = 0, j = 0; j < n; j++)        {
        cdf += preP[j];
        if (coin < cdf / sum)            {
            next_token = j;
            break;
        }
    }
    return next_token;
}

int GGUF_list(CLI_params& hparams)  {
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
        auto param_1 = hparams;
        param_1.wiki_actor="OnlyTokenizer";     param_1.tpWiki=WIKI::_OFF;    
        // param_1.fn_model_base = path;
        
        // hFISH fish_0 = Fish::MakeInstance("GGUF_",param_1,0x0);
        // fish_0->Dump(0x0);
        GST_TIC(tic);
        fprintf(fp,"%d: \"%s\" ...\n",i++,path.c_str());         fflush(fp);
        try{
            hWIKI wiki = wiki = std::make_shared<LAMA>(param_1,path);
            assert(wiki!=nullptr);
            info = wiki->__repr__(suffix,prefix);  
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

int Fish_bubble(CLI_params& hparams)  {    
    hparams.wiki_actor = "copy";
    arrHWIKI wikis = WIKI::MakeInstance("wikis",hparams,0x0);
#if !defined(NDEBUG)
    // hparams.common.n_ctx = 17;     hparams.common.n_batch = 1;      hparams.nLayerX=1;      //Only for debug
    // hparams.set();
#endif
    hparams.isOnlyGPT = true;
    hFISH fish = Fish::MakeInstance("BUBBLE_",hparams,wikis,Fish::ROLE_TYPE::COMMON,0x110);
    auto _gf = fish->GetForwRaw();         assert(_gf!=nullptr);
    if(0)
        ggml_graph_print(_gf);
    if(hparams.is({"gpt","c_graph"},string("raw")))
        ggml_graph_comp0(_gf,0x0);   //only for comparsion
    else{
        if(!fish->LoadTrain())
            return -1;
        fish->GenSentence();
    }
        
    return 666;
}

int fish_1(CLI_params& hparams)  {
    auto param_1 = hparams,param_2 = hparams;
    param_1.tpWiki = "logits";              param_1.common.n_batch = 1;
    param_2.tpWiki = "";                    param_2.common.n_batch = 1;    // 
    hFISH fish_0 = Fish::MakeInstance("BIG_",param_1,{},Fish::ROLE_TYPE::COMMON,0x0);
    hFISH fish_1 = Fish::MakeInstance("local_",param_2,fish_0.get(),0x110);
    fish_0->Dump(0x0);
    fish_1->Dump(0x0);

    vector<float> logits;           
    vector<llama_token> piffle;    
    fish_0->LocalFeeling(piffle,logits); 
    fish_1->CopyWeight(fish_0.get());
    fish_1->LocalFeeling(piffle,logits);
    return 666;
}

bool _LoadCheckPoint(CLI_params& hparams,arrHWIKI& wikis,int flag=0x0){ 
    if (hparams.save.checkpoint_in.empty())
        return false;
               
    hFISH fish = Fish::MakeInstance("Fish_",hparams,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
    if(fish==nullptr || !fish->LoadTrain())
        return false;
    return true;
}

int GPT_work(CLI_params& hparams)  {
    hFISH fish = nullptr;
    vector<hWIKI> wikis = WIKI::MakeInstance("",hparams,0x0); 
    if(!hparams.save.checkpoint_in.empty()){
        fish = Fish::MakeInstance("Fish_",hparams,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
        if(fish==nullptr || !fish->LoadTrain())
            return 0;
    }else{
        wikis.clear();
        gpt_params params;
        params.seed = 42,           params.sparams.seed = params.seed;
        if(hparams.fn_model_base.size()==0){
            _ERROR("%s has no WIKI models!",__func__);
            return -2;
        }
        params.model = hparams.fn_model_base[0];   //"./models/Meta-Llama-3-8B.Q2_K.gguf";
        params.prompt = hparams.prompt;
        params.escape = true;
        params.n_predict = 128;
        params.n_gpu_layers = 6;
        gpt_params_handle_model_default(params);
        if (params.escape)
        {
            string_process_escapes(params.prompt);
            string_process_escapes(params.input_prefix);
            string_process_escapes(params.input_suffix);
            string_process_escapes(params.sparams.cfg_negative_prompt);
            for (auto &antiprompt : params.antiprompt)        {
                string_process_escapes(antiprompt);
            }
        }
        LOG_TEE("%s\n", gpt_params_get_system_info(params).c_str());
        llama_numa_init(params.numa);
        hWIKI wiki = std::make_shared<LAMA>(hparams,params.model);      assert(wiki!=nullptr);
        wiki->teach = WIKI::_LOGITS;        
        wikis.push_back(wiki);
        //  hGOPT gpt = std::make_shared<GeneratOnPrompt>(params,0x0);
        //  hGOPT gpt = std::make_shared<GOPT_infinite>(params,0x0);    
        // hGOPT gpt = std::make_shared<GOPT_Metropolis>(params, 0x0); 
    }
    hGOPT gpt = std::make_shared<GOPT_Metropolis>(hparams,wikis,fish.get(), 0x0);
    // hWIKI wiki = wikis[0];
    if (gpt->Init(hparams.prompt))    { 
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

WIKI::WIKI(){

}
std::vector<hWIKI> WIKI::MakeInstance(const std::string nam_,struct CLI_params& params,int flag){
    std::vector<hWIKI> wikis;
    if(params.tpWiki!="off") {//wiki is so heavy(ugly) that only load one instance here!
        for(auto path : params.fn_model_base){
            hWIKI wiki = std::make_shared<LAMA>(params,path);
            wikis.push_back(wiki);
        }        
    }   
    return wikis;  
}

hGOPT GeneratOnPrompt::MakeInstance(struct CLI_params& hparams,arrHWIKI& wikis,const Fish *fish_0,int flag) {
    // gopt = std::make_shared<GeneratOnPrompt>(wiki,this,0x0);
    hGOPT gopt = std::make_shared<GOPT_Metropolis>(hparams,wikis,fish_0,0x0);
    if(gopt!=nullptr && gopt->Init(hparams.prompt)){
        // gopt->Generate(0); //only for debug  
    }else{
        gopt.reset();       gopt = nullptr;
    }   
    const char* promt = "when the smoke is going down,"; 
    std::vector<llama_token> some_inp = {9493,279,16603,374,2133,1523,11};      //const char* promt = "when the smoke is going down,";      
    
    // wiki->Decode(embd_inp,0,0x0);
    // wiki->Answer(embd_inp);      //only for debug
    return gopt;
}

void GeneratOnPrompt::InitEmbdInput(int flag){
            // if (params.chatml) {
            //     params.prompt = "<|im_start|>system\n" + params.prompt + "<|im_end|>";
            // }
    _INFO("[GPT] tokenize the prompt\n");    
    TOKEN_ID bo=-1;
    switch(_arch)    {  
    case NLP_GPT2_char:
        embd_inp.clear();
        embd_inp.push_back(0);
        for(int i=0;i<params.prompt.length();i++){
            int id = (int)(params.prompt[i]);       
            embd_inp.push_back(id);
        }
        break;
    default:{
        const bool add_bos = llama_add_bos_token(model);
        embd_inp = ::llama_tokenize(ctx, params.prompt, add_bos, true);
        bo = embd_inp[0];
        }
        break;
    }
    
    return;
}

GeneratOnPrompt::GeneratOnPrompt(struct gpt_params &par_, int flag) : params(par_){    
    LOG_TEE("%s logits_all=%d\n", __func__,params.logits_all );
    llama_numa_init(params.numa);
    prompt = params.prompt;

    params.sparams.temp = 0.0;
    params.sparams.temp = 0.8;
    sparams = params.sparams;
    // compatible with LLAMA.cpp
    hparams.fn_model_base.push_back( params.model ); 
    n_predict = params.n_predict;
}

void GeneratOnPrompt::Clear()    {
    if(ctx!=nullptr)
        llama_print_timings(ctx);
    // write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    if (ctx_guidance) { llama_free(ctx_guidance); }
    if(wikis.empty()){
        llama_free(ctx);        
        llama_free_model(model);            
    }

    if(ctx_sampling!=nullptr)
        llama_sampling_free(ctx_sampling);
    llama_backend_free();

    FREE_a(_logits);
}
          //only for debug
GeneratOnPrompt::GeneratOnPrompt(CLI_params&cp_,arrHWIKI& wiki_, const Fish *hG_, int flag) : 
    hparams(cp_),fish_0(hG_), wikis(wiki_) {
    if(fish_0!=nullptr){
        auto gang_param = hparams;
        gang_param.tpWiki = "off";      assert(gang_param.tpWiki=="off");
        gang_param.common.n_batch = 1;
        fish_1 = Fish::MakeInstance("4GPT_",gang_param,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
            //fish_1 = Fish::MakeInstance("4GPT_",gang_param,hG_,0x110);
        // fish_1 = Fish::MakeInstance("4GPT_",gang_param,0x0);
        fish_1->Dump(0x0);
        if(0){//only for debug
            vector<float> logits;           
            vector<llama_token> piffle;     
            fish_1->LocalFeeling(piffle,logits);
        } 
        _arch = fish_0->arch;         
    }else{
        _arch = hparams.ModelArch();
    }

    // fish_1 = nullptr;
    params.seed = 42; 
    params.sparams.seed = params.seed;
    if(!hparams.fn_model_base.empty())
        params.model = hparams.fn_model_base[0];   //"./models/Meta-Llama-3-8B.Q2_K.gguf";
    else
        params.model = hparams.KV({"arch"});
    params.prompt = hparams.prompt; 
    prompt = hparams.prompt; 
    //"when the smoke is going down,"; //"Building a website can be done in 10 simple steps:\\nStep 1:";
    params.escape = true;
    params.n_predict = 32;
    gpt_params_handle_model_default(params);
    if (params.escape)    {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        string_process_escapes(params.sparams.cfg_negative_prompt);
        for (auto &antiprompt : params.antiprompt)        {
            string_process_escapes(antiprompt);
        }
    }

    n_predict = params.n_predict;    
    LOG_TEE("%s wiki=%ld propmt=\"%s\" n_predict=%d logits_all=%d\n", __func__,wikis.size(),prompt.c_str(),n_predict,params.logits_all );
}

bool GeneratOnPrompt::Init(const std::string &prompt_, int flag)    {
    // std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (wikis.empty()){     CHILD_0909_WIKIS
        // hWIKI wiki = std::make_shared<LAMA>(hparams,hparams.fn_model_base[0]);
        // wikis.push_back(wiki);
        prompt = prompt_;
        n_ctx = hparams.n_ctx();
        int nTokens = fish_1->nClass();
        _logits = new float[nTokens];
        InitEmbdInput();
        return true;
    }else{        
    }
    
    llama_backend_init(); // ggml_time_init(); 
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
    assert(sparams.cfg_scale == 1.0);
    /*if (sparams.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }*/
    if (model == NULL)    {
        LOG_TEE("%s: error: unable to load model\n", __func__);
        return false;
    }

    n_ctx_train = llama_n_ctx_train(model);
    n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d(%d)\n", n_ctx, n_ctx_train);
    if (n_ctx > n_ctx_train)    {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",__func__, n_ctx_train, n_ctx);
    }
    LOG_TEE("\n");

    prompt = prompt_;
    if (prompt != "")    {
        params.prompt = prompt;
        Tokenize(flag);
    }

    ga_n = params.grp_attn_n;               ga_w = params.grp_attn_w;
    if (ga_n != 1)    {
        GGML_ASSERT(ga_n > 0 && "grp_attn_n must be positive");                         // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
                                                                                        // GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        // GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    return true;
}

void GeneratOnPrompt::DisplayEmbd(bool input_echo, llama_context *ctx, int n_consumed, int flag){
    if (input_echo && display)    {
        for (auto id : tokens)        {
            std::string token_str = ""; //llama_token_to_piece(ctx, id);
            switch(_arch)    {  
            case NLP_GPT2_char:
                token_str = (char)(id); 
                break;
            default:
                token_str = llama_token_to_piece(ctx, id);
                break;
            }
            // token_str = llama_token_to_piece(ctx, id);
            printf("%s", token_str.c_str());
            if (tokens.size() > 1)            {
                input_tokens.push_back(id);
            }            else            {
                output_tokens.push_back(id);
                output_ss << token_str;
            }
        }
        fflush(stdout);
    }
    // reset color to default if there is no pending user input
    if (input_echo && (int)embd_inp.size() == n_consumed)
    {
        console::set_display(console::reset);
        display = true;
    }
}

std::string LoadSomeText(const string&fpath,int flag)   {
    string txt="";
    FILE *fp = std::fopen(fpath.c_str(), "rt");
    if(fp==NULL)    return txt;

    std::fseek(fp, 42, SEEK_SET);
    const int n=2048;
    char buf[n];
    if( std::fread(buf, 1, n, fp)!=n ){
        return txt;
    }
    txt += buf;  
    return txt;
}

/**/
int NLP_AutoRegressive::GenSentence(int flag)  {
    if(hOPT==nullptr)
        return -1;
    assert(preLogits!=nullptr);
    GST_TIC(tic);
    hWIKI wiki = wikis[0];           assert(wiki != nullptr);
    if(gopt!=nullptr){
        wiki->Reset();
        return gopt->Generate(0x0);
    }
    uint64_t rng_seed = 42;
    std::string prompt = hparams.prompt;
    prompt = LoadSomeText(hparams.fp_train_data,0x0);
    int genT = 16, nVocab = preLogits->ne[0], _nctx = hparams.n_ctx(), i, j,pLen=0;
    assert(genT <= _nctx);
    pLen = std::min(_nctx,(int)(prompt.size()));
    
    double sum = 0, cdf = 0;
    /*vector<llama_token> piffle;       piffle.resize(_nctx);
    for (int i = 0; i < _nctx; ++i)    {
        piffle[i] = wiki->eos;
    }*/
   SampLoader *hLoader = &(hOPT->val_loader);
    if(hLoader->num_batches==0 )    {
        hLoader->InitOneSamp(prompt,Input(),0x110);
        // hSAMP samp = hOPT->val_loader.all_samps[0];
        // hOPT->val_loader.Samp2Batch(0,samp,Input(),nullptr,hparams.common);  
    } 
    vector<llama_token>& piffle = hLoader->GetTokens();
    assert(preLogits->type == GGML_TYPE_F32);
    vector<TOKEN_ID> answer;
    _INFO("%s: <--- \n\t", __func__);
    for (i = 1; i <= genT; i++)    {
        // LocalFeeling(piffle,preP);
        float fLos = hOPT->Evaluate(*hLoader ,-666);
        float *preP = (float *)(preLogits->data)+i*nVocab;
        TOKEN_ID t = Sample_CDF(nVocab,preP,&rng_seed);
        piffle[i] = t;      answer.push_back(t); 
        string a = hDict->T2STR(answer),b=hLoader->sentence,s=hDict->T2STR(t);   //    wiki->T2STR(next_token);
        _INFO("\r\t%s\t(%.*s)",a.c_str(),64,b.c_str());        //_INFO("%s+[%s] ",a.c_str(), answer.c_str());
        fflush(stdout);
    }
    _INFO("---> T=%g s\n", GST_TOC(tic));
    return 0x0;
}

/*
    when the smoke is going down, that's when it's like, 'Abby, you really need to go to the doctor,' this was not the case with me or my son.
        We were between his second and third birthday and it was time to go to the doctor and not go to the doctor--and that was the reason he was going to the doctor and not going to the doctor.
        On the two sides of the age there was a lot of room for two oth
*/
llama_token GOPT_Metropolis::Sample(int idx, bool is_resampling)    {
    std::vector<float> x_logits;
    auto ctx_main = ctx, ctx_cfg = ctx_guidance;
    int j,nTokens = fish_1==nullptr ? llama_n_vocab(model) : fish_1->nClass();   //, j;
    
    hWIKI wiki = wikis.size()>0  ? wikis[0] : nullptr;
    WIKI::INDUCT_MODE teach = wiki==nullptr ? WIKI::_OFF : wiki->teach;
    float *preP = new float[nTokens](), sum, cdf;
    assert(idx==-1);
    const float *wLog = nullptr;
    float l1=0,sum1=0,l2=0,delta,a,pMin=FLT_MAX,pMax=FLT_MIN;   
    if(teach==WIKI::_OFF)  {

    }else{
        wLog = wiki->GetLogits(nTokens,1);   
        for (l2 = 0,j = 0; j < nTokens; j++    ) {
            a = wLog[j];
            l2 += a*a;              _logits[j] = a;
            pMin = min(a,pMin);     pMax = max(a,pMax);
        }        
    }
 
    if (fish_1 != nullptr)    {   //time bottleneck, so share wiki to reduce time & memory
        // SOFT_MAX(nTokens,_logits,wLog);     //soft merge ???
        fish_1->CopyWeight(fish_0);
        std::vector<llama_token> inputs = embd_inp;
        inputs.insert( inputs.end(), output_tokens.begin(), output_tokens.end() );
        if (fish_1->LocalFeeling(inputs, x_logits))        {
            assert(x_logits.size()==nTokens);
            // SOFT_MAX(x_logits);
            switch(teach){
            case WIKI::_OFF:
                for (j = 0; j < nTokens; j++    ) {
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
                SOFT_MAX(nTokens,_logits,wLog); 
                for (j = 0; j < nTokens; j++    ) {
                    a = x_logits[j];                    l1+=a*a;        
                    _logits[j] += a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2)-1.0;
            break;
            case WIKI::_LOGITS_SCALE:
                for (j = 0; j < nTokens; j++    ) {
                    a = x_logits[j];        
                    sum1 += a;              
                    l1+=a*a;     l2+=_logits[j]*_logits[j];
                    // a = max(a,0.f);      relu
                    _logits[j] *= a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2);  
                break;
            default:    //  WIKI::_LOGITS:
                for (j = 0; j < nTokens; j++    ) {
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
        }   else{
            return -1;
        }
    }
    for (pMin=FLT_MAX,pMax=FLT_MIN,j = 0; j < nTokens; j++    ) {
        a = _logits[j];
        pMin = min(a,pMin);     pMax = max(a,pMax);
    }   
    int next_token = nTokens - 1;
    for (sum = 0, j = 0; j < nTokens; j++)    {
        preP[j] = exp(_logits[j]-pMax);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);   //1679583
    if(rng_state==42){  //only for debug
        assert(sum > 0 && sum < FLT_MAX);   //1679583
    }
    float coin = random_f32(&rng_state);     //0.339085
    for (cdf = 0, j = 0; j < nTokens; j++)    {
        cdf += preP[j];
        if (coin < cdf / sum)        {
            next_token = j;
            break;
        }
    }
    delete[] preP;
    return next_token;
}

llama_token GeneratOnPrompt::Sample(int idx, bool is_resampling)    {
    const llama_sampling_params &params = ctx_sampling->params;
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

    llama_token id = 0;
    std::vector<float> original_logits, x_logits;
    auto ctx_main = ctx, ctx_cfg = ctx_guidance;
    /*
        cur_p:  typedef struct llama_token_data_array {
            llama_token_data * data;            size_t size;            bool sorted;
        } llama_token_data_array;
    */
    auto cur_p = llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, !is_resampling, &original_logits);
    if (!is_resampling)    {
        GGML_ASSERT(!original_logits.empty());
    }
    if (fish_1 != nullptr)    {   //time bottleneck
        x_logits.resize(original_logits.size());
        if (fish_1->LocalFeeling(embd_inp, x_logits))        {
            std::transform(original_logits.begin(), original_logits.end(), x_logits.begin(),
                           original_logits.begin(), std::plus<float>());
        }
    }/**/

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
            //        const llama_token id = cur_p.data[i].id;
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
            /*LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, llama_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, true);*/
            // Pass true for is_resampling
        }
    }

    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}

void GeneratOnPrompt::OnAntiPrompt(int flag)
{
    if (!params.antiprompt.empty())
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
        llama_token last_token = llama_sampling_last(ctx_sampling);
        for (std::vector<llama_token> ids : antiprompt_ids)
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
    }
}

// deal with end of text token in interactive mode
void GeneratOnPrompt::OnInteractive(int &n_past, int &n_consumed, int &n_remain, int flag)
{
    if (llama_sampling_last(ctx_sampling) == eos)
    {
        LOG("found EOS token\n");
        if (params.interactive)
        {
            if (!params.antiprompt.empty())
            {
                // tokenize and inject first reverse prompt
                const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false, true);
                embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                is_antiprompt = true;
            }

            is_interacting = true;
            printf("\n");
        }        
        /*else if (params.instruct || params.chatml)
        {
            is_interacting = true;
        }*/
    }

    if (n_past > 0 && is_interacting)
    {
        LOG("waiting for user input\n");

        /*if (params.instruct || params.chatml)        {
            printf("\n> ");
        }*/

        if (params.input_prefix_bos)
        {
            LOG("adding input prefix BOS token\n");
            embd_inp.push_back(llama_token_bos(model));
        }

        std::string buffer;
        if (!params.input_prefix.empty())
        {
            LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
            printf("%s", params.input_prefix.c_str());
        }

        // color user input only
        console::set_display(console::user_input);
        display = params.display_prompt;

        std::string line;
        bool another_line = true;
        do
        {
            another_line = console::readline(line, params.multiline_input);
            buffer += line;
        } while (another_line);

        // done taking input, reset color
        console::set_display(console::reset);
        display = true;

        // Add tokens to tokens only if the input buffer is non-empty
        // Entering a empty line lets the user pass control back
        if (buffer.length() > 1)
        {
            // append input suffix if any
            if (!params.input_suffix.empty())
            {
                LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                printf("%s", params.input_suffix.c_str());
            }

            LOG("buffer: '%s'\n", buffer.c_str());

            const size_t original_size = embd_inp.size();

            if (params.escape)
            {
                string_process_escapes(buffer);
            }

            const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
            const auto line_inp = ::llama_tokenize(ctx, buffer, false, false);
            const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

            LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

            embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
            embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

            // instruct mode: insert response suffix
            /*if (params.instruct)
            {
                LOG("inserting instruction suffix\n");
                embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
            }
            // chatml mode: insert assistant chat suffix
            if (params.chatml)
            {
                LOG("inserting chatml suffix\n");
                embd_inp.insert(embd_inp.end(), cml_sfx.begin(), cml_sfx.end());
            }*/

            for (size_t i = original_size; i < embd_inp.size(); ++i)
            {
                const llama_token token = embd_inp[i];
                output_tokens.push_back(token);
                output_ss << llama_token_to_piece(ctx, token);
            }

            n_remain -= line_inp.size();
            LOG("n_remain: %d\n", n_remain);
        }
        else
        {
            LOG("empty line, passing control back\n");
        }

        input_echo = false; // do not echo this again
    }

    if (n_past > 0)
    {
        if (is_interacting)
        {
            llama_sampling_reset(ctx_sampling);
        }
        is_interacting = false;
    }
}

int GeneratOnPrompt::Generate(int nJob, int flag)   {
    GST_TIC(tic);
    output_tokens.clear();
    delta_max = 0;      delta_a = 0;
    hWIKI wiki = nullptr;
    string info = "only fish";
    if(wikis.size()>0)  {
        wiki = wikis[0];      CHILD_0909_WIKIS
        info = wiki->teach==WIKI::_OFF?"only fish":"WIKI"; 
        ctx_sampling = isCTXSampling ? llama_sampling_init(sparams) : nullptr;
    }   
    LOG_TEE("<--- GeneratOnPrompt %s job=%d logits_all=%d fish=%s teach=%d\n", info.c_str(),
        nJob,params.logits_all,fish_1==nullptr?"":fish_1->Name().c_str(),wiki==nullptr? -1 : wiki->teach);
    rng_state = params.seed;
    // LOG_TEE("%s logits_all=%d\n", __func__, );
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty())    {
        for (llama_token id : session_tokens)        {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens])
            {
                break;
            }
            n_matching_session_tokens++;
        }
        if (prompt.empty() && n_matching_session_tokens == embd_inp.size())
        {
            LOG_TEE("%s: using full prompt from session file\n", __func__);
        }
        else if (n_matching_session_tokens >= embd_inp.size())
        {
            LOG_TEE("%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (embd_inp.size() / 2))
        {
            LOG_TEE("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }
        else
        {
            LOG_TEE("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, n_matching_session_tokens, -1);
    }

    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past = 0,n_remain = params.n_predict,n_consumed = 0,n_session_consumed = 0,n_past_guidance = 0,ga_i = 0;

    tokens.clear();
    embd_guidance.clear();
    
    while ((n_remain != 0 && !is_antiprompt) || params.interactive)    {
        int iRet = 0;
        if(wiki!=nullptr && wiki->teach!=WIKI::_OFF)
            iRet = UpdateEmbed(nJob, n_past, n_remain, n_consumed, n_session_consumed, n_past_guidance, ga_i, 0x0);
        else
            iRet = 2;
        if (iRet == 0)
            break;
        if (iRet == 1)
            return 1;

        tokens.clear();
        embd_guidance.clear();
        if ((int)embd_inp.size() <= n_consumed && !is_interacting)        {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro)            {
                need_to_save_session = false;
                // llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
                // LOG("saved session to %s\n", path_session.c_str());
            }

            const llama_token id = Sample(); // llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            if(id<0){
                _INFO("\t<E>");
                break;
            }
            if(ctx_sampling!=nullptr)   {
                llama_sampling_accept(ctx_sampling, ctx, id, true);
                LOG("last=%s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());
            }
            tokens.push_back(id);            
            input_echo = true;      // echo this to console            
            --n_remain;             // decrement remaining sampling budget
            LOG("n_remain=%d\n", n_remain);
        }        else        {
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_consumed: %d\n", (int)embd_inp.size(), n_consumed);
            while ((int)embd_inp.size() > n_consumed)            {
                tokens.push_back(embd_inp[n_consumed]);
                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                if(ctx_sampling!=nullptr)   llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);

                ++n_consumed;
                if ((int)tokens.size() >= params.n_batch)                {
                    break;
                }
            }
        }
        DisplayEmbd(input_echo, ctx, n_consumed);

        // if not currently processing queued inputs;
        if ((int)embd_inp.size() <= n_consumed)        {            // check for reverse prompt in the last n_prev tokens
            OnAntiPrompt(0x0);
            if(ctx_sampling!=nullptr)   OnInteractive(n_past, n_consumed, n_remain, 0x0);
        }
        // end of text token
        if (!tokens.empty() && tokens.back() == eos )        {
            LOG_TEE(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0)        {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    // if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
    //     LOG_TEE("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
    //     llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    // }
    if(ctx_sampling!=nullptr)   {
        llama_sampling_free(ctx_sampling);    ctx_sampling = nullptr;
    }
    
    _INFO("\n delta=%.3g(%.3g) T=%gs --------------->\n",delta_max,delta_a/params.n_predict, GST_TOC(tic));
    return 0x0;
}

/*
    Decode(llama_decode) tokens in batches   tokens is typically prepared beforehand to fit within a batch, but not always
*/
int GeneratOnPrompt::UpdateEmbed(int nJob, int &n_past, int &n_remain, int &n_consumed, int &n_session_consumed, int &n_past_guidance, int &ga_i, int flag) {
    hWIKI wiki = wikis[0];      CHILD_0909_WIKIS
    if (!tokens.empty())    {
        // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
        // --prompt or --file which uses the same value.
        int max_embd_size = n_ctx - 4;

        // Ensure the input doesn't exceed the context size by truncating tokens if necessary.
        if ((int)tokens.size() > max_embd_size)        {
            const int skipped_tokens = (int)tokens.size() - max_embd_size;
            tokens.resize(max_embd_size);
            console::set_display(console::error);
            printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            console::set_display(console::reset);
            fflush(stdout);
        }
        // assert(ga_n == 1);
        assert(ctx_guidance == nullptr);

        for (int i = 0; i < (int)tokens.size(); i += params.n_batch)
        {
            int n_eval = min((int)tokens.size() - i, params.n_batch);
            LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());
            bool bDecode = wiki != nullptr ? wiki->Decode(tokens, i, n_past,false) : llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval, n_past, 0)) >= 0;
            if (!bDecode)            {
                LOG_TEE("%s : failed to eval\n", __func__);
                return 1;
            }
            n_past += n_eval;
            LOG("n_past = %d\n", n_past);
            if (params.n_print > 0 && n_past % params.n_print == 0)
            { // Display total tokens alongside total time
                LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
            }
        }

        if (!tokens.empty() && !path_session.empty())
        {
            session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
            n_session_consumed = session_tokens.size();
        }
    }
    return 2;
}

int GOPT_infinite::UpdateEmbed(int nJob, int &n_past, int &n_remain, int &n_consumed, int &n_session_consumed, int &n_past_guidance, int &ga_i, int flag)
{
    assert(0);      CHILD_0909_WIKIS
    /* const int n_ctx = llama_n_ctx(ctx);               // ctx->cparams.n_ctx;
    const int n_ctx_train = llama_n_ctx_train(model); // model->hparams.n_ctx_train;
    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (!tokens.empty())
    {
        // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
        // --prompt or --file which uses the same value.
        int max_embd_size = n_ctx - 4;

        // Ensure the input doesn't exceed the context size by truncating tokens if necessary.
        if ((int)tokens.size() > max_embd_size)
        {
            const int skipped_tokens = (int)tokens.size() - max_embd_size;
            tokens.resize(max_embd_size);
            console::set_display(console::error);
            printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            console::set_display(console::reset);
            fflush(stdout);
        }
        assert(ga_n == 1);
        if (ga_n == 1)
        {
            // infinite text generation via context shifting
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + (int)tokens.size() + std::max<int>(0, guidance_offset) > n_ctx)
            {
                if (params.n_predict == -2)
                {
                    LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                    return 0;
                }

                const int n_left = n_past - params.n_keep;
                const int n_discard = n_left / 2;

                LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                    n_past, n_left, n_ctx, params.n_keep, n_discard);

                llama_kv_cache_seq_rm(ctx, 0, params.n_keep, params.n_keep + n_discard);
                llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                n_past -= n_discard;

                if (ctx_guidance)
                {
                    n_past_guidance -= n_discard;
                }

                LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

                LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());

                LOG("clear session path\n");
                path_session.clear();
            }
        }
        else
        {
            // context extension via Self-Extend
            while (n_past >= ga_i + ga_w)
            {
                const int ib = (ga_n * ga_i) / ga_w;
                const int bd = (ga_w / ga_n) * (ga_n - 1);
                const int dd = (ga_w / ga_n) - ib * bd - ga_w;

                LOG("\n");
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib * bd, ga_i + ib * bd, n_past + ib * bd);
                LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n, (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
                LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, n_past + ib * bd, dd, ga_i + ib * bd + ga_w + dd, n_past + ib * bd + dd);

                llama_kv_cache_seq_add(ctx, 0, ga_i, n_past, ib * bd);
                llama_kv_cache_seq_div(ctx, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
                llama_kv_cache_seq_add(ctx, 0, ga_i + ib * bd + ga_w, n_past + ib * bd, dd);

                n_past -= bd;

                ga_i += ga_w / ga_n;

                LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
            }
        }

        // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
        if (n_session_consumed < (int)session_tokens.size())
        {
            size_t i = 0;
            for (; i < tokens.size(); i++)
            {
                if (tokens[i] != session_tokens[n_session_consumed])
                {
                    session_tokens.resize(n_session_consumed);
                    break;
                }

                n_past++;
                n_session_consumed++;

                if (n_session_consumed >= (int)session_tokens.size())
                {
                    ++i;
                    break;
                }
            }
            if (i > 0)
            {
                tokens.erase(tokens.begin(), tokens.begin() + i);
            }
        }
        // evaluate tokens in batches
        // tokens is typically prepared beforehand to fit within a batch, but not always
        assert(ctx_guidance == nullptr);
        for (int i = 0; i < (int)tokens.size(); i += params.n_batch)
        {
            if (i == 0)
                i = 0; // only for debug
            int n_eval = (int)tokens.size() - i;
            if (n_eval > params.n_batch)
            {
                n_eval = params.n_batch;
            }

            LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());
            bool bDecode = wiki != nullptr ? wiki->Decode(tokens, i, n_past,false) : llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval, n_past, 0)) >= 0;
            if (!bDecode)
            {
                LOG_TEE("%s : failed to eval\n", __func__);
                return 1;
            }

            n_past += n_eval;

            LOG("n_past = %d\n", n_past);
            // Display total tokens alongside total time
            if (params.n_print > 0 && n_past % params.n_print == 0)
            {
                LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
            }
        }

        if (!tokens.empty() && !path_session.empty())
        {
            session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
            n_session_consumed = session_tokens.size();
        }
    }*/
    return 2;
}