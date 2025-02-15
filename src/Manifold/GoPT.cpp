#include "../../llama.cpp/common/console.h"
#include "Optimizer.hpp"
#include "GoPT.hpp"
#include "Fish.hpp"
#include "gLLM.hpp"
#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#define LOG //

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
            hWIKI wiki = nullptr;       //  WIKI::MakeInstance
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
    if(hparams.is({"gpt","c_graph"},string("raw"))){
        //ggml_graph_comp0(_gf,0x0);   //only for comparsion
    }        
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
    vector<TOKEN_ID> piffle;    
    // Need piffle to samploader
    fish_0->LocalFeeling(nullptr,logits); 
    fish_1->CopyWeight(fish_0.get());
    fish_1->LocalFeeling(nullptr,logits);
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
    //  GRUS_Get_SystemInfo
    _INFO("[%s] threads=%d \n%s\n",__func__,std::thread::hardware_concurrency(),llama_print_system_info());
    ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);    
    DEBUG.SelfAttention_noraml = 0;
    DEBUG.NO_loss = true;
    DEBUG.graph_dump = 1;
    // hparams.wiki_actor="";    //only for debug

    hparams.isOnlyGPT = true;
    hparams.passLoadToken = true;
    bool isMakeFish = hparams.is({"wiki","actor"},"copy") || hparams.wiki_actor=="OnlyTokenizer";
    hFISH fish = nullptr;
    vector<hWIKI> wikis = WIKI::MakeInstance("",hparams,0x0);          
    if(hparams.fn_model_base.size()>0 && !isMakeFish){
        for(auto wiki : wikis){
            if(wiki->isOnlyTokenizer)
                assert( wiki->teach == WIKI::_OFF );
            else
                wiki->teach = WIKI::_LOGITS;
        }                     
    }

    if(isMakeFish){
        fish = Fish::MakeInstance("Fish_",hparams,wikis,Fish::ROLE_TYPE::COMMON,0x110);  
        if(fish==nullptr || !fish->LoadTrain()){
            _ERROR("%s has no WIKI or FISH!\n",__func__);
            return 0;
        }            
    }
    //  hGOPT gpt = std::make_shared<GeneratOnPrompt>(params,0x0);
    //  hGOPT gpt = std::make_shared<GOPT_infinite>(params,0x0); 
    hGOPT gpt = std::make_shared<GOPT_Metropolis>(hparams,wikis,fish.get(), 0x0);
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

hGOPT GeneratOnPrompt::MakeInstance(struct CLI_params& hparams,arrHWIKI& wikis,const Fish *fish_0,int flag) {
    // gopt = std::make_shared<GeneratOnPrompt>(wiki,this,0x0);
    hGOPT gopt = std::make_shared<GOPT_Metropolis>(hparams,wikis,fish_0,0x0);
    if(gopt!=nullptr && gopt->Init(hparams.prompt)){
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
    return hparams.prompt;  
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
        dialogs->InitOneSamp(hparams.prompt,input,0x110);
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
    hparams.fn_model_base.push_back( params.model ); 
    n_predict = params.n_predict;*/
}

void GeneratOnPrompt::Clear()    {
    // write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);

    if (ctx_guidance) { llama_free(ctx_guidance); }
    if(wikis.empty()){
        llama_free(ctx);        
        // llama_free_model(model);            
    }

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
        fish_1 = (Fish*)fish_0;
        // fish_1 = Fish::MakeInstance("4GPT_",gang_param,wikis,Fish::ROLE_TYPE::SWARM_FOLLOWER,0x110);        //  isLocalInfer = flag==0x110;
        // fish_1->Dump(0x0);

        _arch = fish_0->arch;         
    }else{
        _arch = hparams.ModelArch();
    }
    if(!wikis.empty())
        wiki0 = wikis[0];
}

bool GeneratOnPrompt::Init(const std::string &prompt_, int flag)    {
    // std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if(fish_1!=nullptr) {
        dialogs = std::make_shared<SampLoader>(fish_1,"gpt", true);       
        dialogs->Prepare(fish_1->hOPT.get(), fish_1->tsEval[0]); 
        dialogs->isRecycle = false;
        dialogs->type = SampLoader::TYPE::DT_EVAL;          
    }
    int n_vocab;
    if (wikis.empty()){     CHILD_0909_WIKIS
        n_ctx = hparams.n_ctx();
        n_vocab = fish_1->nClass();
        _logits = new float[n_vocab];
        // dialogs->init(hparams.prompt.c_str(), B, T, 0, 1, 0);

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
    prompt = LoadSomeText("hparams.fp_train_data",0x0);
    int genT = 16, nVocab = preLogits->ne[0], _nctx = hparams.n_ctx(), i, j,pLen=0;
    assert(genT <= _nctx);
    pLen = std::min(_nctx,(int)(prompt.size()));
    
    double sum = 0, cdf = 0;
    /*vector<TOKEN_ID> piffle;       piffle.resize(_nctx);
    for (int i = 0; i < _nctx; ++i)    {
        piffle[i] = wiki->eos;
    }*/
    hSampLoader hLoader = hOPT->val_loaders[0];
    if(hLoader->num_batches==0 )    {
        hLoader->InitOneSamp(prompt,nullptr,0x110);
    } 
    vector<TOKEN_ID>& piffle = hLoader->GetTokens();
    assert(preLogits->type == GGML_TYPE_F32);
    vector<TOKEN_ID> answer;
    _INFO("%s: <--- \n\t", __func__);
    for (i = 1; i <= genT; i++)    {
        // LocalFeeling(piffle,preP);
        float fLos = hOPT->Evaluate(hLoader ,-666);
        float *preP = (float *)(preLogits->data)+i*nVocab;
        TOKEN_ID t = Sample_CDF(nVocab,preP,&rng_seed);
        piffle[i] = t;      answer.push_back(t); 
        string a = hDict->T2STR(answer),b=hLoader->sentence,s=hDict->T2STR(t);   
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
TOKEN_ID GOPT_Metropolis::Sample(int idx, bool is_resampling)    {
    
    auto ctx_main = ctx, ctx_cfg = ctx_guidance;
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
        token_str = dolphin->hDict->T2STR(tok);
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
        ctx_sampling = nullptr;
    }   
    LOG("<--- GeneratOnPrompt %s job=%d logits_all=%d fish=%s teach=%d\n", info.c_str(),
        nJob,0,fish_1==nullptr?"":fish_1->Name().c_str(),wiki0==nullptr? -1 : wiki0->teach);
    rng_state = hparams.common.seed;
    // LOG("%s logits_all=%d\n", __func__, );  
    // bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past = 0,n_remain = n_predict,n_session_consumed = 0,ga_i = 0,max_embd_size = n_ctx-4;
    tokens.clear();                 // embd_guidance.clear();
    LOG("embd_inp.size(): %d \n", (int)embd_inp.size());
    tokens = embd_inp;
    _INFO("%s",hparams.prompt.c_str());
    
    while ((--n_remain >= 0 && !is_antiprompt) )    {
        if (tokens.empty())    
            break;        
        assert ((int)tokens.size() <= max_embd_size );
        // assert(ga_n == 1);
        assert(ctx_guidance == nullptr);
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
    /*GST_TIC(tic);
    output_tokens.clear();
    delta_max = 0;      delta_a = 0;
    hWIKI wiki = nullptr;
    string info = "only fish";
    if(wikis.size()>0)  {
        wiki = wikis[0];      CHILD_0909_WIKIS
        info = wiki->teach==WIKI::_OFF?"only fish":"WIKI"; 
        ctx_sampling = isCTXSampling ? llama_sampling_init(sparams) : nullptr;
    }   
    LOG("<--- GeneratOnPrompt %s job=%d logits_all=%d fish=%s teach=%d\n", info.c_str(),
        nJob,params.logits_all,fish_1==nullptr?"":fish_1->Name().c_str(),wiki==nullptr? -1 : wiki->teach);
    rng_state = params.seed;
    // LOG("%s logits_all=%d\n", __func__, );
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty())    {
        for (TOKEN_ID id : session_tokens)        {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens])
            {
                break;
            }
            n_matching_session_tokens++;
        }
        if (GetPrompt().empty() && n_matching_session_tokens == embd_inp.size())
        {
            LOG("%s: using full prompt from session file\n", __func__);
        }
        else if (n_matching_session_tokens >= embd_inp.size())
        {
            LOG("%s: session file has exact match for prompt!\n", __func__);
        }
        else if (n_matching_session_tokens < (embd_inp.size() / 2))
        {
            LOG("%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }
        else
        {
            LOG("%s: session file matches %zu / %zu tokens of prompt\n",
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

            const TOKEN_ID id = Sample(); // llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
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
        DisplayEmbd(input_echo, n_consumed);

        // if not currently processing queued inputs;
        if ((int)embd_inp.size() <= n_consumed)        {            // check for reverse prompt in the last n_prev tokens
            OnAntiPrompt(0x0);
            if(ctx_sampling!=nullptr)   OnInteractive(n_past, n_consumed, n_remain, 0x0);
        }
        // end of text token
        if (!tokens.empty() && tokens.back() == eos )        {
            LOG(" [end of text]\n");
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
    //     LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
    //     llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    // }
    if(ctx_sampling!=nullptr)   {
        llama_sampling_free(ctx_sampling);    ctx_sampling = nullptr;
    }
    
    _INFO("\n delta=%.3g(%.3g) T=%gs --------------->\n",delta_max,delta_a/params.n_predict, GST_TOC(tic));*/
    return 0x0;
}

#include "../../../llama.cpp/common/log.h"
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
