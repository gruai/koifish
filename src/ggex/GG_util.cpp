/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  Some auxiliary functions
 *  Unfortunately, llama.cpp removed training functions. I would continue to work hard to support and strengthen training.
 * 
 *  \brief 
 *  \author Yingshi Chen
 */
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../CLI_params.hpp"
#include "../ggex/GG_util.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "json.hpp"
#include <iostream>
#include <cstring>

#define ARG2STR(format,len)    {    va_list args;    va_start( args, format );    vsnprintf( buffer,len,format,args );    va_end(args);   assert(strlen(buffer)<=len);   }

int gTN(hGTensor cur, const char *format,...){
    int iRet = 0;
    if(strlen(cur->name)==0){
        ARG2STR(format,GGML_MAX_NAME);
        snprintf(cur->name, sizeof(cur->name), "%s", buffer);
        iRet+=1;
    }
    return iRet;
}


int gTN0(hGTensor cur,const char *format,... ){
    ARG2STR(format,GGML_MAX_NAME);
    snprintf(cur->name, sizeof(cur->name), "%s", buffer);
    return 0x0;
}

int gTN(struct ggml_tensor *cur,const char *format,... ){
    int iRet = 0;
    if(strlen(cur->name)==0){
        ARG2STR(format,GGML_MAX_NAME);
        // va_list args;
        // va_start( args, format );
        // vsnprintf( buffer,GGML_MAX_NAME,format,args );
        // va_end(args);
        // assert(strlen(buffer)<=GGML_MAX_NAME);
        ggml_format_name(cur,"%s",buffer);
        
        iRet+=1;
    }
    /*
        in ggml_compute_backward, some grad has no name!

        ggml_format_name(tensor->grad, "%s (grad)", tensor->name);
    */
#ifdef GG_V12
   CHILD_1218_GRAD      //  set name @BuildBackward
#else
    if(cur->grad && strlen(cur->grad->name)==0){
        assert(strlen(cur->name)<GGML_MAX_NAME);
        if(strcmp(cur->name,"inp_embd_rows")==0){
            int debug = 0;
        }
        ggml_format_name(cur->grad,"%s\"",cur->name);        
        iRet+=2;
    }
#endif
    return iRet;
}

int gTN0(struct ggml_tensor *cur,const char *format,... ){
    int iRet = 0;
    ARG2STR(format,GGML_MAX_NAME);
    // va_list args;
    // va_start( args, format );
    // vsnprintf( buffer,GGML_MAX_NAME,format,args );
    // va_end(args);
    // assert(strlen(buffer)<=GGML_MAX_NAME);
    ggml_format_name(cur,"%s",buffer);    
    iRet+=1;

    /*
        in ggml_compute_backward, some grad has no name!

        ggml_format_name(tensor->grad, "%s (grad)", tensor->name);
    */
    /*if(cur->grad && strlen(cur->grad->name)==0){
        assert(strlen(cur->name)<GGML_MAX_NAME);
        ggml_format_name(cur->grad,"%s\"",cur->name);        
        iRet+=2;
    }*/
    return iRet;
}

int clamp(const int v, const int min, const int max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

float fclamp(const float v, const float min, const float max) {
    return ((v < min) ? (min) : (v > max) ? (max) : v);
}

void assert_shape_1d(hGTensor  tensor, int64_t ne0){}
void assert_shape_2d(hGTensor  tensor, int64_t ne0, int64_t ne1){}
void assert_shape_3d(hGTensor  tensor, int64_t ne0, int64_t ne1, int64_t ne2){}
void assert_shape_4d(hGTensor  tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3){}

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == 1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == 1);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == 1);
}

void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == ne3);
}

struct random_uniform_distribution {
    std::mt19937 gen;
    std::uniform_real_distribution<float> rd;
};

struct random_normal_distribution * init_random_normal_distribution(    int seed, float mean, float std, float min, float max) {
    struct random_normal_distribution * rnd = (struct random_normal_distribution *) malloc(sizeof(struct random_normal_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
    return rnd;
}

struct random_uniform_distribution * init_random_uniform_distribution(int seed, float min, float max) {
    struct random_uniform_distribution * rnd = (struct random_uniform_distribution *) malloc(sizeof(struct random_uniform_distribution));
    rnd->gen = std::mt19937(seed);
    rnd->rd = std::uniform_real_distribution<float>{min, max};
    return rnd;
}

void free_random_normal_distribution (struct random_normal_distribution  * rnd) {
    free(rnd);
}

void free_random_uniform_distribution(struct random_uniform_distribution * rnd) {
    free(rnd);
}

float frand() {
    return (float)rand()/((float)(RAND_MAX) + 1.0f);
}

float frand_normal(struct random_normal_distribution * rnd) {
    return fclamp(rnd->rd(rnd->gen), rnd->min, rnd->max);
}

float frand_uniform(struct random_uniform_distribution * rnd) {
    return rnd->rd(rnd->gen);
}

struct ggml_tensor * tRAND(struct ggml_tensor * tensor, struct random_normal_distribution * rnd) {
    float scale = 1.0f; // xavier
    switch (ggml_n_dims(tensor)) {
        case 1:
            scale /= sqrtf((float) tensor->ne[0]);
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0]);
                *dst = scale * frand_normal(rnd);
            }
            break;
        case 2:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1]);
                    *dst = scale * frand_normal(rnd);
                }
            }
            break;
        case 3:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2]);
                        *dst = scale * frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            scale /= sqrtf((float) tensor->ne[0]+tensor->ne[1]);
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float * dst = (float *) ((char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3]);
                            *dst = scale * frand_normal(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(0 && "Unsupported tensor->n_dims");
    };
    return tensor;
}
hGTensor tRAND(hGTensor tensor, struct random_normal_distribution * rnd) {
    return nullptr;
}

JSON jKEY(const JSON& jConfig,const std::vector<std::string>&keys,int flag){
    assert(keys.size()>0);
    JSON cur = (JSON)jConfig;
    std::string key;
    for(int i=0;i<keys.size();i++){
        key = keys[i];
        if(cur.find(key)==cur.end())   {
            // _ERROR("\t NO %s @\"%s\";",__func__,key.c_str());
            cur.clear();
            return cur;
        }
        cur = cur[key];
    }
    if(cur.is_null() || cur.empty()){
        _ERROR("%s failed !!!",__func__);
    }
            
    return cur;
}
//bool jK2S(const JSON& jConfig,const std::vector<std::string>&keys,(const char*) &str,int flag=0x0);
bool jK2S(const JSON& jConfig,const std::vector<std::string>&keys,char** str,int flag=0x0){
    JSON cur = jKEY(jConfig,keys);
    if(cur.is_null() || cur.empty())    {
        assert(0);
        return false;
    }
    string val = cur.get<std::string>();
    // *str = val.c_str();
    strcpy(*str,val.c_str());
    return true;
}

void UpdateJConfig(JSON& jConfig,const std::string&jPath){
try{
    std::ifstream jfile(jPath);
    if(jfile.fail()){
        _INFO("\r\n%s  Failed to open %s",__func__,jPath.c_str());
    }
    jfile>>jConfig;
    std::string s = jConfig.dump();
}   catch(JSON::parse_error &e){
    _INFO("\r\n%s  Failed to open %s!!! ERR=%s",__func__,jPath.c_str(),e.what());
}   catch(...){
    _INFO("\r\n%s  Unknown exception @%s!!!",__func__,jPath.c_str());
}
}

std::string executable_name()   {
#if defined(PLATFORM_POSIX) || defined(__linux__) //check defines for your setup
    std::string sp;
    std::ifstream("/proc/self/comm") >> sp;
    return sp;
#elif defined(_WIN32)
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return buf;
#else
    static_assert(false, "unrecognized platform");
#endif
}
/**
 * 
*/
void train_print_usage(int argc, char ** argv, const struct CLI_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");

    // fprintf(stderr, "  --model-base FNAME         model path from which to load base model (default '%s')\n", params->fn_model_base);
    fprintf(stderr, "  --lora-out FNAME           path to save llama lora (default '%s')\n", params->save.model_out.c_str());
    fprintf(stderr, "  --only-write-lora          only save llama lora, don't do any training.  use this if you only want to convert a checkpoint to a lora adapter.\n");
    fprintf(stderr, "  --norm-rms-eps F           RMS-Norm epsilon value (default %f)\n", params->f_norm_rms_eps);
    fprintf(stderr, "  --rope-freq-base F         Frequency base for ROPE (default %f)\n", params->rope_freq_base);
    fprintf(stderr, "  --rope-freq-scale F        Frequency scale for ROPE (default %f)\n", params->rope_freq_scale);
    fprintf(stderr, "  --lora-alpha N             LORA alpha : resulting LORA scaling is alpha/r. (default %d)\n", params->lora_alpha);
    fprintf(stderr, "  --lora-r N                 LORA r: default rank. Also specifies resulting scaling together with lora-alpha. (default %d)\n", params->lora_r);
    fprintf(stderr, "  --rank-att-norm N          LORA rank for attention norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-ffn-norm N          LORA rank for feed-forward norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-out-norm N          LORA rank for output norm tensor, overrides default rank. Norm tensors should generally have rank 1.\n");
    fprintf(stderr, "  --rank-tok-embd N          LORA rank for token embeddings tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-out N               LORA rank for output tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wq N                LORA rank for wq tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wk N                LORA rank for wk tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wv N                LORA rank for wv tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-wo N                LORA rank for wo tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-ffn_gate N          LORA rank for ffn_gate tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-ffn_down N          LORA rank for ffn_down tensor, overrides default rank.\n");
    fprintf(stderr, "  --rank-ffn_up N            LORA rank for ffn_up tensor, overrides default rank.\n");

    // print_common_train_usage(argc, argv, &params->common);
}

bool CLI_params::operator!=(const CLI_params& other) const {
    return memcmp(this, &other, sizeof(other));
}

DEUG_SWITCH DEBUG;
void DEUG_SWITCH::Dump(int typ)    {
    _INFO("\t switch: datas=%d hyperparams=%d attn=%d\n", train_datas,train_hyperparams,SelfAttention_noraml);
}
void CLI_params::Dump( )    {
    _INFO("%s::CLI_params: \n", exec_name.c_str());
    // _INFO(" n_vocab: %u", n_vocab);
    _INFO(" n_ctx=%u", n_ctx());
    _INFO(" n_embd=%u", n_embd);        
    _INFO(" n_ff=%u", n_ff());
    _INFO(" n_head=%u", n_head());
    _INFO(" n_head_kv=%u", n_head_kv());
    _INFO(" n_layer=%u(%u)", nLayer(),n_layer_train);
    _INFO(" n_rot=%u\n", n_rot());       
    _INFO(" f_norm_rms_eps=%g", f_norm_rms_eps);   
    _INFO(" rope_freq_base=%g", rope_freq_base);   
    _INFO(" rope_freq_scale=%g", rope_freq_scale);
    _INFO("\n lora_r=%d lora_alpha=%g \n", lora_r,lora_alpha); 
    _INFO(" NABLA = %s\n", nabla==0? "" : nabla==3 ? "Embed+AutoEncoder" : (nabla==2 ? "" : "qkv") ); 
    // _INFO(" SIGMA = %s\n", sigma.c_str()); 
}

MODEL_ARCH CLI_params::ModelArch()   {  
    MODEL_ARCH arch = MODEL_ARCH::_X_;
    string info = jKV(jConfig,{"arch"},string(""),false); 
    assert(!info.empty());
    std::transform(info.begin(), info.end(), info.begin(), ::toupper);
    arch =  info=="MOE" ? NLP_MOE :
            info=="MAMBA" ? MODEL_ARCH::NLP_MAMBA : 
            info=="DEEPSEEK" ? MODEL_ARCH::NLP_DEEPSEEK : 
            info=="GPT2" ? MODEL_ARCH::NLP_GPT2 :
            info=="GPT2CHAR" ? MODEL_ARCH::NLP_GPT2_char :
            info=="LAMA" ? MODEL_ARCH::NLP_LLAMA :
            MODEL_ARCH::NLP_LLAMA;  

    return arch; 
}

void CLI_params::JModel2Params(int flag){
    // nlohmann::ordered_json jm = jKEY(jConfig,{"model"});
    jModel = jKEY(jConfig,{"model"});
    if(jModel.empty()){   
        return;
    }
    nLayerX = 1;    //at least 1 layer
    nLayerX = jKV(jConfig,{"model","parameter","Layer"},nLayerX );    
    assert(nLayerX<160 && nLayerX>0);
    // nearly same ???
    // if(nLayerX>0)
    //     common.residual_scale = 1.0f / sqrtf(2.0f * nLayerX);   
    assert(layerps.size()==0);
    auto jTrans = jKEY(jConfig,{"model","parameter","transformer"});
    if(!jTrans.empty()){
        int nH = jKV(jTrans,{"Head"},-1),nF = jKV(jTrans,{"Ffn"},-1),nE = jKV(jTrans,{"Embed"},-1),nC = jKV(jTrans,{"Ctx"},-1);
        if(nH>0 && nF>0){
            for(int i=0;i<nLayerX;i++)
                layerps.push_back(LAY_PARAM(nH,nH,nF));
        }else{
            assert(0);
        }
        if(nE>0)    {
            n_embd = nE;        assert(n_embd<160*1000 && n_embd>0);
        }else{
            assert(0);
        }
        if(nC>0)    {
            SetNCTX(nC);
            // common.n_ctx = nC; 
        }else{
            assert(0);
        }      
    }
    //  n_embd_head_k   n_embd_head_v   ???
}

uint32_t CLI_params::nThread() const {
    int nT0=std::thread::hardware_concurrency(),nT1=common.n_threads;
    return nT1;
}
void CLI_params::OnMostToken(size_t nMost,int flag){
    double a = nMost*1.0/n_ctx();    //nTokenInBatch();
    size_t nSamp = (size_t)(floor)(a);
    int iter_1 = (int)floor(a/n_batch());
    float rSample = common.rSubSample;
    common.nEpochIter = (int)ceil(nMost*rSample/nTokenInBatch());
    common.nMostIter = (int)floor(nMost*common.n_epochs*rSample/nTokenInBatch());
}    

std::string CLI_params::NameOnArch(std::string&name,int flag){
    string sx="";
    if(name=="layer")   
        sx="blk";
    switch(ModelArch()){
    case MODEL_ARCH::NLP_GPT2:  
    case MODEL_ARCH::NLP_GPT2_char:  {
    }
        break;
    default:
        
        break;
    }
    return sx;
}

void CLI_params::OnArch( ){
    int nH=-1;
    string info = jKV(jConfig,{"arch"},string("")); 
    bool isJModel = !jModel.empty();
    
    switch(ModelArch()){
    case MODEL_ARCH::NLP_GPT2:  
    case MODEL_ARCH::NLP_GPT2_char:  {
        //baby_GPT      dropout = 0.2
        // n_head = 6;             n_embd = 384;           dict_latent_dim=n_embd;
        //124M
        // nH = 12;         n_embd = 768;           DEBUG.dict_latent_dim = 768;
        n_embd_head_v = 64;         n_embd_head_k = 64;
        // n_embd = 128; dict_latent_dim = 128;        n_embd_head_v=n_embd_head_k=2; //only for debug        
        n_ctx_train = 1024;
        if(layerps.size()==0 && !isJModel){
            // TO_DO: why grad vanish @/home/cys/rnd/lic/log/gpt2/10_29_bug.info
            int n_ff0 = jKV(jConfig,{"model_v0","ffn","length"},3072,false),nLay=nLayer();
            for(int i=0;i<nLayer();i++){
                LAY_PARAM lay(nH,nH,n_ff0);
                layerps.push_back(lay);
            }
        }
        //  need new GEMM! cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
        modep.isNormalBias = true;
        modep.isSLPBias = true;         //nealy same
        modep.isPaddedCls = true;       //  ceil(n/128.0)*128

        int group=Get({"model_v0","target_group"},1);
        assert(group==1);
    }
        
        break;
    case NLP_QWEN2:
    case NLP_DEEPSEEK:
    /*
          `. llama_model_rope_type=LLAMA_ROPE_TYPE_NEOX
    */
        break;

    default:        
        _INFO("[ARCH]=%s\n",info.c_str());
        break;
    }    
}

struct train_params_ get_default_train_params_common() {
    struct train_params_ params;
    // params.print_usage = false;

    params.save_every = 10;

    params.seed       =   -1;

    params.n_ctx      =  128;
    params.n_threads  =    6;
    params.n_batch    =    8;
    params.n_gradient_accumulation = 1;
    params.n_epochs   = -1;
    params.n_gpu_layers = 0;

    params.custom_n_ctx = false;

    params.use_flash              = false;
    params.use_checkpointing      = true;

    params.sample_start           = "";
    params.include_sample_start   = false;
    params.escape                 = false;
    params.overlapping_samples    = false;
    params.fill_with_next_samples = false;
    params.separate_with_eos      = false;
    params.separate_with_bos      = true;
    params.sample_random_offsets  = false;
    params.force_reshuffle        = false;

    params.opt_past               = 0;
    params.opt_delta              = 1e-5f;
    params.opt_max_no_improvement = 0;

    params.warmup            =  600;
    params.lr_restart=0;

    // params.adam.n_iter         = -1;
    params.adam.alpha          = 1e-3f;
    params.adam.min_alpha      = 0;
    params.adam.decay          = 1e-1f;
    params.adam.decay_min_ndim = 2;
    params.adam.beta1          = 0.9f;
    params.adam.beta2          = 0.95f; //0.999f;
    params.adam.gclip          = 1.0f;
    params.adam.eps_loss       = 1e-5f;

    return params;
}

std::string CLI_params::GetDataPath(const string type,int flag){
    string fp = "";
try{
    if(type.empty())    //some tiny file for fast debug
        fp = jKV(jConfig,{"tiny_data","source"},fp );
    else{
        fp = jKV(jConfig,{"datasets",type,"source"},fp );
    }
    
}catch(...){

}
    if( !std::filesystem::exists(fp) ){
        _INFO("%s: warning: empty or not existing %s data file '%s'\n", __func__, type.c_str(), fp.c_str());
        fp = "";
    }    
    return fp;
}

/*
    Some trick
    1 Large batch size would decrease osillation
    2 Double batch+half layers =>  More accuracy in same training time
*/
bool CLI_params::InitJConfig(const std::string&jPath,int flag){
try{
    common = get_default_train_params_common();
    std::ifstream jfile(jPath);
    std::string info;

    
    if(jfile.fail()){
        _INFO("\r\n[%s] Failed to open \"%s\" !!!\n",__func__,jPath.c_str());
        return false;
    }
    jfile>>jConfig;
    std::string s = jConfig.dump(),s0;
    common.n_batch = jKV(jConfig,{      "train","batch"},common.n_batch );
    common.n_epochs = jKV(jConfig,{  "train","epoch"},common.n_epochs );
    common.nMostIter = jKV(jConfig,{  "train","adam-iter"},common.nMostIter );
    common.adam.alpha = jKV(jConfig,{   "train","learning-rate"},common.adam.alpha );
    common.n_gradient_accumulation = jKV(jConfig,{ "train","optimizatioin","grad_accumulation"  },common.n_gradient_accumulation );
    lars_ratio = jKV(jConfig,{          "train","optimizatioin","lars_ratio"},lars_ratio );
    ZMUV_ratio = jKV(jConfig,{          "train","optimizatioin","ZMUV_ratio"},ZMUV_ratio );

    // serial_path = jKV(jConfig,{"data","serialize_path"},s0 );
    string dict_type = jKV(jConfig,{"dict","type"},s0 );
    tpBatchSample = jKV(jConfig,{"data","tpBatchSample"},tpBatchSample ); 
    rSplit = jKV(jConfig,{"data","eval_split"},rSplit );
    string a = tpBatchSample=="stacking" ? tpBatchSample : "";
    
    std::vector<string> all_base;
    all_base = jKV_arr(jConfig,{"wiki","path"},all_base,false);
    for(auto path : all_base){
        if(path.empty() || path[0]=='#')
            continue;
        if( !std::filesystem::exists(path) ){
            _WARN("====== Failed to load model @\"%s\" !!! ======\n",path.c_str());
            continue;
        }
            
        if(model_title.empty()) //the first path is backbone
            model_title = remove_extension(base_name(path));
        fn_model_base.push_back(path);
    }
    if(model_title.empty()){
        model_title = jKV(jConfig,{"arch"},string(""));
    }
    // serial_path += a+"_["+model_title+dict_type+"]_";       //std::to_string(1.0-rSplit)
    JModel2Params(0x0);
    

    // eval_binpath = jKV(jConfig,{"data","eval_binpath"},s0 );   

    
    n_swarm = jKV(jConfig,{"train","swarm"},1 );
    common.save_every = jKV(jConfig,{"train","save-every"},common.save_every );
    common.dump_every = jKV(jConfig,{"train","dump-every"},common.dump_every );   
    // common.eval_every = jKV(jConfig,{"train","eval-every"},common.eval_every );    
    common.gpt_every = jKV(jConfig,{"train","gpt-every"},common.gpt_every );    
    // common.eval_every = common.eval_every<=0 ? 100000000 : common.eval_every;
    // if( eval_every>0 ){
    //     _INFO("\r\n%s  eval@every %d steps.",__func__,eval_every );
    // }
    common.rSubSample = jKV(jConfig,{"train","sample"},common.rSubSample ); 
    if(common.rSubSample<0)     common.rSubSample=1;
    if(common.rSubSample<1)     {
        common.lr_restart = 1;
    }
    
    common.seed = jKV(jConfig,{"seed"},common.seed ); 
    wiki_actor = jKV(jConfig,{"wiki","actor"},wiki_actor );
    wiki_logits = jKV(jConfig,{"wiki","logits"},wiki_logits );
    tpWiki = jKV(jConfig,{"wiki","induct"},tpWiki ); 
    nabla = jKV(jConfig,{"model_v0","nabla"},nabla );
    // sigma = jKV(jConfig,{"model_v0","sigma"},sigma ); 

    if(jModel.empty()){
        nFFX = jKV(jConfig,{"model_v0","ffn","length"},nFFX );  
        assert(nFFX<160*1024 && nFFX>0); 
        nLayerX = jKV(jConfig,{"model_v0","layer"},nLayerX );    
        assert(nLayerX<160 && nLayerX>0);    
        common.n_ctx = jKV(jConfig,{"model_v0","ctx"},common.n_ctx ); 
    }else{
        
    }
    
    common.custom_n_ctx = true;
    common.n_threads = jKV(jConfig,{"threads"},common.n_threads );
    common.n_gpu_layers = jKV(jConfig,{"n-gpu-layers"},common.n_gpu_layers );  
    
    // n_embd = jKV(jConfig,{"wiki","embd"},n_embd );

    save.model_out = jKV(jConfig,{"model-out"},save.model_out );  
    save.checkpoint_in = jKV(jConfig,{"checkpoint-in"},save.checkpoint_in );  
    save.checkpoint_out = jKV(jConfig,{"checkpoint-out"},save.checkpoint_out );    
    
    f_norm_rms_eps = jKV(jConfig,{"norm-rms-eps"},f_norm_rms_eps );
    rope_freq_base = jKV(jConfig,{"rope-freq-base"},rope_freq_base );
    rope_freq_scale = jKV(jConfig,{"rope-freq-scale"},rope_freq_scale );

    prompt = jKV(jConfig,{"gpt","prompt"},prompt );    

    dict_vae_dims = jKV(jConfig,{"dict","vae","dims"},dict_vae_dims ); 
    // DEBUG.dict_latent_dim = jKV(jConfig,{"dict","latent_dim"},DEBUG.dict_latent_dim ); 
    dict_dialect = jKV(jConfig,{"dict","dialect"},dict_dialect );
    dict_logits = jKV(jConfig,{"dict","logits"},dict_logits );
    
    vae = dict_vae_dims;        //hack

    test = jKV(jConfig,{"test"},test );    
    

/*
    on some ealy testing on finetune/distillation, it seems that less layers would get nealy same accuracy
*/
    
    tune = jKV(jConfig,{"lora","tune"},tune );         //"lora_tune"
    lora_r = jKV(jConfig,{"lora","rank"},lora_r );      //{"lora-r"}

    // train = jKV(jConfig,{"train"},train );   
    return true;
}   catch(JSON::parse_error &e){
    _INFO("\r\n%s  Failed to open %s!!! ERR=%s",__func__,jPath.c_str(),e.what());
    return false;
}   catch(...){
    _INFO("\r\n%s  Unknown exception @%s!!!",__func__,jPath.c_str());
    return false;
}
}

bool CLI_params::parse(int argc, char ** argv)  {
    bool invalid_param = false;
    std::string arg;
    common = get_default_train_params_common();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.length()>5 && arg.substr(arg.length()-5,arg.length())==".json") {   
#ifdef NDEBUG
        // arg = "/home/cys/rnd/lic/scripts/koifish.json";     //only for test
#endif         
            if(!InitJConfig(arg))
                return false;     
            break;
        }

        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--model-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            fn_model_base.push_back( argv[i] );
        } else if (arg == "--model-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            save.model_out = argv[i];
        } else if (arg == "--embd") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            n_embd = std::stoi(argv[i]);
        }else if (arg == "--lora-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            save.model_out = argv[i];
        } else if (arg == "--only-write-lora") {
            only_write_model = true;
        } else if (arg == "--learning-rate"){
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            common.adam.alpha = std::stof(argv[i]);
        }
        else if (arg == "--norm-rms-eps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            f_norm_rms_eps = std::stof(argv[i]);
            // custom_f_norm_rms_eps = true;
        } else if (arg == "--rope-freq-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            rope_freq_base = std::stof(argv[i]);
            // custom_rope_freq_base = true;
        } else if (arg == "--rope-freq-scale") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            rope_freq_scale = std::stof(argv[i]);
            // custom_rope_freq_scale = true;
        } else if (arg == "--lora-alpha") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            lora_alpha = std::stoi(argv[i]);
            // custom_lora_alpha = true;
        } else if (arg == "--lora-r") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            lora_r = std::stoi(argv[i]);
        } else if (arg == "--rank-att-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_attention_norm = std::stoi(argv[i]);
            // custom_n_rank_attention_norm = true;
        } else if (arg == "--rank-ffn-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_ffn_norm = std::stoi(argv[i]);
            // custom_n_rank_ffn_norm = true;
        } else if (arg == "--rank-out-norm") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_norm = std::stoi(argv[i]);
            // custom_n_rank_norm = true;
        } else if (arg == "--rank-tok-embd") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_tok_embeddings = std::stoi(argv[i]);
            // custom_n_rank_tok_embeddings = true;
        } else if (arg == "--rank-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_output = std::stoi(argv[i]);
            // custom_n_rank_output = true;
        } else if (arg == "--rank-wq") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_wq = std::stoi(argv[i]);
            // custom_n_rank_wq = true;
        } else if (arg == "--rank-wk") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_wk = std::stoi(argv[i]);
            // custom_n_rank_wk = true;
        } else if (arg == "--rank-wv") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_wv = std::stoi(argv[i]);
            // custom_n_rank_wv = true;
        } else if (arg == "--rank-wo") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_wo = std::stoi(argv[i]);
            // custom_n_rank_wo = true;
        } else if (arg == "--rank-ffn_gate") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_ffn_gate = std::stoi(argv[i]);
            // custom_n_rank_ffn_gate = true;
        } else if (arg == "--rank-ffn_down") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_ffn_down = std::stoi(argv[i]);
            // custom_n_rank_ffn_down = true;
        } else if (arg == "--rank-ffn_up") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // n_rank_ffn_up = std::stoi(argv[i]);
            // custom_n_rank_ffn_up = true;
        } else if (arg == "--nabla") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            nabla = std::stoi(argv[i]);
        } else if (arg == "--sigma") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            // sigma = argv[i];
        }else if (arg == "--layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            nLayerX = std::stoi(argv[i]);
        }else if (arg == "--tune") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            tune = (CLI_params::TUNE_ALG)(std::stoi(argv[i]));
        } else if (arg == "--train") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            train = argv[i];
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            train_print_usage(argc, argv, this);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        train_print_usage(argc, argv, this);
        exit(1);
    }
    exec_name = executable_name( );
    // finish_processing_train_args(&common);
    return true;
}


int Gensor_loab(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor ga,hGensor gb,int flag){
    printf("%s@%s <== %s x %s\n\t",__func__,w->name,ga->name,gb->name);
    auto shape = w->ne;
    int nIn=shape[0], nOut=shape[1], rank = nHeavy; //min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);        assert(nIn>0 && nOut>0 && ne00==nIn*nOut);
    assert(nIn>nHeavy && nOut>nHeavy && nHeavy>0);
    float *A=Gensor2float(ctx0,w,flag);
    auto svd=std::make_shared<LoSVD<float>>(A,nIn,nOut,rank,1.0e-3,0); //1.0e-3
    assert(ga->type==typNUMBER::F32 && gb->type==typNUMBER::F32);
    if(!svd->Build( ))  {
        return -1;
    }else{
        if(tELEM(ga)!=nIn*rank || tELEM(gb)!=nOut*rank)    {
            return -2;
        }
        svd->US((float *) ((char *) ga->data));     
        memcpy(gb->data,svd->V(),sizeof(float)*rank*nOut);
    }
    delete[] A;
    return 0x0;
}

int Gensor_SVD(struct ggml_context * ctx0,hGensor w,int nHeavy,hGensor U,hGensor D,hGensor V,int flag){
    printf("%s@%s \t ......",__func__,w->name);
      
    auto shape = w->ne;
    int nIn=shape[0], nOut=shape[1], rank = nHeavy; //min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);
    assert(nIn>0 && nOut>0 && ne00==nIn*nOut);
    assert(nIn>nHeavy && nOut>nHeavy && nHeavy>0);
    float *A=Gensor2float(ctx0,w,flag);
    GST_TIC(tic);
    auto svd=std::make_shared<LoSVD<float>>(A,nIn,nOut,rank,1.0e-3,0); //1.0e-3
    float t0 = GST_TOC(tic);
    if(!svd->Build( ))  {
        return -1;
    }else{
        //typNUMBER::F16 tensor would call ggml_vec_dot_f16 with GGML_SIMD acceleration
        /*if(compression==SVD_a)  {   //keep same graph
            float *approx = svd->Approx( );
            ggml_fp32_to_fp16_row(approx,(ggml_fp16_t*)w->data,nIn*nOut);
        }else*/{
            memcpy(U->data,svd->U(),sizeof(float)*nIn*rank);     
            memcpy(V->data,svd->V(),sizeof(float)*rank*nOut); 
            float *Sigma = svd->S(),*mD=(float *)(D->data);
            //memcpy(D->data,Sigma,sizeof(float)*rank);    
            memset(mD,0x0,sizeof(float)*rank*rank);
            for(int i=0;i<rank;i++)
                mD[i*rank+i] = Sigma[i];
                
        }       
    }
    svd.reset();
    delete[] A;

    return 0x0;
}


// float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
// void ggml_vec_max_f32(const int n, float * s, const float * x);
// inline void ggml_vec_scale_f32(const int n, float * y, const float   v);
float SOFT_MAX(const int n, float * y, const float * x) {
    float x1 = -INFINITY;
    int i;
    // ggml_vec_max_f32(n, &x1, x);
    for (i = 0; i < n; ++i) {
        x1 = MAX(x1, x[i]);
    }
    float sum = 0.0;
#ifdef GGML_SOFT_MAX_ACCELERATE
    x1 = -x1;
    vDSP_vsadd(S, 1, &x1, S, 1, Mup);
    vvexpf(S, S, &Mup);
    ggml_vec_sum_f32(Mup, &sum, S);
#else
    // sum = ggml_vec_soft_max_f32(n, y, x, x1);
    for (i = 0; i < n; ++i) {
        float val = expf(x[i] - x1);
        sum += (float)val;
        y[i] = val;
    }
#endif
    assert(sum > 0.0);
    sum = 1.0/sum;
    // ggml_vec_scale_f32(n, y, sum);   
    for (i = 0; i < n; ++i) {
        y[i] *= sum;
    }

#ifndef NDEBUG
    for (i = 0; i < n; ++i) {
        assert(!isnan(y[i]));        assert(!isinf(y[i]));
    }
#endif
    return x1;
}

//Py=Py-Px
float SOFT_MAX_minus(const int n, float * y, const float * x) {
    assert(0);
    float x1 = -INFINITY,a;
    int i;
    for (i = 0; i < n; ++i) {
        x1 = MAX(x1, x[i]);
    }
    float sum = 0.0;
    for (i = 0; i < n; ++i) {
        float val = expf(x[i] - x1);
        sum += (float)val;
        y[i] = val;
    }
    assert(sum > 0.0);
    sum = 1.0/sum;
    // ggml_vec_scale_f32(n, y, sum);   
    for (i = 0; i < n; ++i) {
        y[i] *= sum;
    }

#ifndef NDEBUG
    for (i = 0; i < n; ++i) {
        assert(!isnan(y[i]));        assert(!isinf(y[i]));
    }
#endif
    return x1;
}

/*
    for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
*/
float LOSS_cross_entropy_1(int n,const float*preP, int target,int&cand,int flag) {
    assert(target>=0 && target<n);
    float sum=0,loss=0,pMin,pMax,a;
    int j,next_token=-1;
    cand=-1;
    for (pMin=FLT_MAX,pMax=-FLT_MAX,j = 0; j < n; j++    ) {
        a = preP[j];
        if(a>pMax){
            pMax = a;   cand = j;
        }
        pMin = min(a,pMin);     //pMax = max(a,pMax);
    } 
    
    /*for (sum = 0, j = 0; j < n; j++)        { //  standard SOFTMAX
        preP[j] = exp(preP[j]-pMax);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);
    a = preP[target]/sum;   loss = -log(a); //  0.0430280194*/
    for (sum = 0,a = preP[target], j = 0; j < n; j++)        {  //faster & safer
        sum += exp(preP[j]-a);
    }assert(sum > 0 && sum < FLT_MAX);
    loss = log(sum);  
    return loss;
}

struct ggml_tensor * ggml_cross_entropy_loss_1(
        struct ggml_context         * ctx,
        struct ggml_tensor          * a,
        struct ggml_tensor          * b) {
#ifndef GG_V12
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op   = GGML_OP_CROSS_ENTROPY_LOSS_1;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
#endif
    return nullptr;
}

void _T_repr_(hGensor t,const char*tab,char *buf,const GENSOR_INFO&info){
    if(t==nullptr)      return;
    const char* A = "d";
    if(t->flags & GGML_TENSOR_FLAG_PARAM){
        A = "P";
    }else{
#ifndef GG_V12
        if(t->grad!=nullptr){
            A = "G";
        }
#endif
    }
    
    auto ne=t->ne;
    size_t n0 = strlen(buf),n1;         //char buf[64*1024]="\0";
    string suf,pref,sX = info.__repr__(suf,pref);        //    info.sX;
    sprintf(buf+strlen(buf),"%s %s %s %s \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab,sX.c_str(), A,
        t->name,ne[0], ne[1], ne[2], ne[3], cNameOf(t->type));
    n1= strlen(buf); 
}

void _T_repr_(hGensor t,const char*tab,char *buf,int typ){
    if(t==nullptr)      return;
    bool isInput = t->flags & GGML_TENSOR_FLAG_INPUT;
    string A = NameOf(t->type); //==typNUMBER::F16 ? "d16":"d";
    // if(t->grad!=nullptr){
    //     if(t->type==typNUMBER::F16)
    //         A = "P16";    
    //     else
    //         A = "P";
    // }
    if(isInput){
        A = "("+A+")";
    }
    size_t nElem = tELEM(t);
    auto ne=t->ne;
    switch(typ){
    case 1:
        sprintf(buf+strlen(buf),"%s%s '%s' \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab,A.c_str(),t->name,ne[0], ne[1], ne[2], ne[3], cNameOf(t->type));
        break;
    default:
        sprintf(buf+strlen(buf),"%s%s '%s' %.3g%s\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab, 
        A.c_str(),t->name,nElem>1000?nElem/1.0e6:nElem,nElem>1000?"M":"",ne[0], ne[1], ne[2], ne[3], cNameOf(t->type)); 
        break;
    }    
}

int CHECK_SAME_TENSORS(const string& desc,const std::vector<hGensor>&arrA,const std::vector<hGensor>&arrB,int flag){
    _INFO("\n======== %s @[%s]...",__func__,desc.c_str());
    size_t nA=arrA.size(),nB=arrB.size(),nDup=0,nMiss=0;
    bool isSame = arrA.size()==arrB.size();
    std::map<std::string, int> msg;
    int no=1;
    for(auto tA:arrA){
        if(msg.find(tA->name)!=msg.end()){
            _INFO("\tAA=\"%s\"",tA->name); 
            isSame = false;
        }
        msg[tA->name] = no;     
        no++;
    }
    for(auto tB:arrB){
        if(msg.find(tB->name)==msg.end()){
            isSame = false;    nMiss++; 
            _INFO("\tB_%d=\"%s\"",nMiss,tB->name); 
        }
        no = msg[tB->name];
        if(no<0){
            isSame = false;     nDup++;
        }
        msg[tB->name] = -no; 
    }
    for(auto ms : msg){
        if(ms.second>0){
            auto tA = arrA[ms.second-1];
            _INFO("A_%d=%s ",nMiss,tA->name); 
            isSame = false;     nMiss++;
        }
    }
    _INFO("%s======== %s @[%s] OK. A=%d B=%d \n",isSame?"\r":"\n", __func__,desc.c_str(),nA,nB);
    return 0x0;
}

size_t F_SIZE(const std::string&fpath,FILE *fp0,int flag) {
try{
    FILE *fp = fp0;
    if(fp0==NULL){
        fp = std::fopen(fpath.c_str(), "rb");
        assert(fp!=NULL);
#ifdef _WIN32
        int ret = _fseeki64(fp, 0, SEEK_END);
#else   
        int ret = std::fseek(fp, 0, SEEK_END);
#endif        
    }
    
#ifdef _WIN32
    __int64 ret = _ftelli64(fp);
#else
    long ret = std::ftell(fp);
#endif
    assert(ret != -1); 
    if(fp!=fp0)
        fclose(fp);
    return (size_t) ret;
}catch(...){
    assert(0);
    return 0x0;
}
}

struct ggml_context *InitCTX(size_t msize,int flag){
    struct ggml_init_params ctx_model_params;
    if(msize==0){
        msize = ggml_tensor_overhead() * 16384; //LLAMA_TRAIN_MAX_NODES;
    }
    ctx_model_params.mem_size   = msize;
    ctx_model_params.mem_buffer = NULL;
    ctx_model_params.no_alloc   = true;
    
    struct ggml_context * ctx = ggml_init(ctx_model_params);  
    return ctx;  
}

/*
static void ggml_compute_forward_scale_q(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == GGML_TASK_TYPE_INIT || params->type == GGML_TASK_TYPE_FINALIZE) {
        return;
    }
    GGML_TENSOR_UNARY_OP_LOCALS
    // scale factor
    float v;
    memcpy(&v, dst->op_params, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    // const size_t nb01 = src0->nb[1];
    // const size_t nb1 = dst->nb[1];    
    assert(ne00 % 32 == 0);
    float * wdata = (float *) params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;
    const enum ggml_type type = src0->type;
    const enum ggml_type dtype = dst->type;
    ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
    ggml_from_float_t const quantize_row_q = type_traits[dtype].from_float;
    for (int i1 = ir0; i1 < ir1; i1++) {
        if (dst->data != src0->data) {            // src0 is same shape as dst => same indices
            memcpy((char *)dst->data + i1*nb1, (char *)src0->data + i1*nb01, nc * sizeof(float));
        }
        void *data = dst->data + i1*nb1;
        dequantize_row_q(data, wdata, ne00);    // unquantize row from src0 to temp buffer    
        ggml_vec_scale_f32(nc, wdata, v);
        // ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*nb1), v);
        quantize_row_q(wdata, data, ne00);
    }
}*/

hGensor GradOf(struct ggml_cgraph *cgraph,hGensor node,int flag){
#ifdef GG_V12
    assert(cgraph->grads);
    assert(0);          return nullptr;
    /*const size_t igrad = ggml_hash_find(&cgraph->visited_hash_set, node);
    if(igrad == GGML_HASHSET_FULL){
        assert(0);      return nullptr;
    }
    
    if( ggml_bitset_get(cgraph->visited_hash_set.used, igrad) && cgraph->grads )
        return cgraph->grads[igrad];
    else
        return nullptr;*/
#elif defined _TENSOR_G_
    assert(0);
    return nullptr;
#else
    if(node->grad==nullptr){
        int maybe = 0;
    }
    return node->grad;
#endif
 }

bool GTensor::Dump(int type,int flag)  const{
    int i = 0;
    switch(type){
    default:
        _INFO(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n",
                    i,gg->ne[0], gg->ne[1],ggml_op_name(gg->op), ggml_get_name(gg));
    }
    return true;
}

extern "C" void gg_print_tensor_(const char* title, hGensor t, int n) {
    if(strlen(title)>0) printf("%s\n", title);
    size_t nz=0,nElems = 0;//t->size(),;
    float * data = (float *)t->data,a1=-FLT_MAX,a0=FLT_MAX;
    if(t->type!=typNUMBER::F32){
        data = new float[nElems];
        if(t->type==typNUMBER::F16){
            ggml_fp16_t *src_ = (ggml_fp16_t *)(t->data);
            for (int i = 0; i < nElems; i++) {
                data[i] = src_[i];
            }
        }else{  //need dequant
            data = nullptr;     n = 0;
        }
    }    
    double sum = 0.0;
    if(data!=nullptr){
        for (int i = 0; i < nElems; i++) {
            sum += data[i];
            if(data[i]==0)      nz++;
            a1 = std::max(a1,data[i]);      a0 = std::min(a0,data[i]);
        }        
    }
    // printf("sum:  %f\n\n", sum);
    if(nElems==1){
        printf("T%d:%s: %s\t data=%f \n",-1/*t->id*/,t->name, cNameOf(t->type),a0);
    }else
        printf("T%d:%s: %.4g(M)\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] sum=%g data=[%f : %f] rZ=%.3g%%\n", 
            -1/*t->id*/,t->name,nElems/1.0e6,t->ne[0], t->ne[1], t->ne[2], t->ne[3], cNameOf(t->type),sum,a0,a1,nz*100.0/nElems);
    if(n>0){
        printf("\t{");
        for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
            printf("%.5f ", data[i]);
            if (i != 0 && i % t->ne[0] == 0) {
                // printf("\n");
            }
        }
        printf("...");
        for (int i = 0; i < std::min((int) (t->ne[0]*t->ne[1]), n); i++) {
            printf("%.5f ", data[nElems - n + i]);
            if ((nElems - n + i) % t->ne[0] == 0) {
                // printf("\n");
            }
        }
        printf("}\n");
    }

    if(t->type!=typNUMBER::F32){
        delete[] data;
    } 
}

/*
    byte per element of this type
*/
double BPE(typNUMBER type) {
    double bpe = ggml_type_sizef((enum ggml_type)type);//deprecated    
    assert(bpe==1 || bpe==2 || bpe==4);
    return bpe;
}
const char *cNameOf(typNUMBER type){
    return ggml_type_name((enum ggml_type)type);
}
std::string NameOf(typNUMBER type){
    std::string name = cNameOf(type);
    return name;
}
bool isQuantized(typNUMBER type){
    return ggml_is_quantized((enum ggml_type)type);
}

double GTensor::bpe(){
    //  ggml_row_size()
    return BPE(type);   
}
size_t GTensor::size(int typ)  const       {   
    size_t nz=1;    
    for(auto a : shape)   nz*=a;  
    switch(typ){
    case 1:     //only for padded shape
        if(x_shape.size()>0){
            nz = 1;
            for(auto a : x_shape)   nz*=a;  
            assert(nz>1);            
        }
        break;
    default:
        break;
    }
    return nz;
    // return ggml_nelements(gg);     
}
#ifdef _TENSOR_G_


#else
    hGensor tSCAL(struct ggml_context * ctx,struct ggml_tensor  * a,float s,int flag)   {
        // hGensor b = ggml_scale_inplace( ctx,a,s);    // inplace operations are currently not supported!!!
        hGensor b = ggml_scale( ctx,a,s);
        gTN(b,"%s_s",a->name);        return b;
    }

    hGensor Permute(struct ggml_context * ctx_,struct ggml_tensor* cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4,bool isCont)    {
        hGensor q = ggml_permute(ctx_, cur, n1,n2,n3,n4);   
        gTN0(q,"%s.#",cur->name);     
        if(isCont)    {
            q = ggml_cont(ctx_,q);        
            gTN(q,"%s.#c",cur->name);           
        }
        return q;
    }

    hGensor TENSO(void* ctx0,int typ,SHAPE shape,int flag,const string&name){
        /*va_list args;   //  In C/C++, you can not determine the number of arguments that were passed to a "variadic" function!!!
        va_start( args, typ );
        SHAPE shape;
        int val=-1;
        while (val = va_arg(args, int))  {
            shape.push_back(val);
        }
        va_end(args);*/
        struct ggml_context *ctx=(struct ggml_context *)(ctx0);
        assert(ctx0!=nullptr);
        enum ggml_type type = (enum ggml_type)typ;
        struct ggml_tensor *gg=nullptr;
        switch(shape.size()){
        case 1:
            gg = ggml_new_tensor_1d(ctx, type, shape[0]);      
            break;
        case 2:
            gg = ggml_new_tensor_2d(ctx, type, shape[0],shape[1]);
            break;
        case 3:
            gg = ggml_new_tensor_3d(ctx, type, shape[0],shape[1],shape[2]);
            break;
        case 4:
            ggml_new_tensor_4d(ctx, type, shape[0],shape[1],shape[2],shape[3]); 
            break;
        default:
            assert(0);
            break;
        }
        return gg;
    }
#endif

void Gensor2float_(const hGensor w,float *A,int flag)   {
    size_t ne00 = tELEM(w),nbyte = tBYTE(w); 
    void *data_0 = w->data;
    enum ggml_type tp0 = (enum ggml_type)w->type;
    void  *src0_row = (void *) ((char *) w->data );
    assert(ggml_is_quantized(tp0));
    switch(w->type){
        // case typNUMBER::F16:
        //     ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
        //     break;
        case typNUMBER::F32:
            break;
        case typNUMBER::Q2_K:
            dequantize_row_q2_K((const block_q2_K*)src0_row, A, ne00);//-0.00318908691
            break;
        case typNUMBER::Q3_K:
            dequantize_row_q3_K((const block_q3_K*)src0_row, A, ne00);  
            break;
        case typNUMBER::Q4_K:
            dequantize_row_q4_K((const block_q4_K*)src0_row, A, ne00);  
            break;
        case typNUMBER::Q6_K:
            dequantize_row_q6_K((const block_q6_K*)src0_row, A, ne00);  
            break;
        case typNUMBER::Q8_0:
            dequantize_row_q8_0((const block_q8_0*)src0_row, A, ne00);  
            break;
        default:
            assert(0);
            // ggml_tensor_dequant(ctx0,w,typNUMBER::F32);       //memory leak@"float * wdata = malloc(sizeof(float)*ne00)" !!!
            // memcpy(A,w->data,sizeof(float)*ne00);
            // w->data = data_0;       w->type = tp0;
            break;
    }
}

void _TIME_INFO(const string&info,double fmillis,int flag) {
    _INFO("%s",info.c_str());
    if (fmillis < 1000.0f) {
        _INFO("%.1fms", (float) fmillis);
        return;
    }
    if (fmillis < 60*1000.0f) { //60sec
        _INFO("%.2f", fmillis/1000);
        return;
    }
    const int64_t one_sec = 1000, one_min = one_sec*60, one_hour = one_min*60, one_day = one_hour*24;

    int64_t millis  = (int64_t) fmillis;
    int64_t days    = millis/one_day;
    int64_t hours   = (millis - days*one_day)/one_hour;
    int64_t minutes = (millis - days*one_day - hours*one_hour)/one_min;
    int64_t seconds = (millis - days*one_day - hours*one_hour - minutes*one_min)/one_sec;

    // to print int64_t either cast to (long long int) or use macro PRId64 from <inttypes.h>
    if (days > 0) {
        _INFO("%lldd ", (long long int) days);
    }
    if(hours==0 && minutes==0){
        _INFO("%02ld", seconds);
    }else
        _INFO("%02lld:%02lld:%02lld", (long long int) hours, (long long int) minutes, (long long int) seconds);
}   

void ADAM_params_::Dump(int typ){
    _INFO("\tADAM lr=%g,beta=[%g-%g] decay=%g(%d) clip=%g(alg=%d)\n", alpha,beta1,beta2,
        decay,decay_min_ndim,gclip,clip_alg);
}

void MOEL_params_::Dump(int typ){

}

ggml_cgraph * GG_dup_graph(ggml_context * ctx, ggml_cgraph *src){
    assert(0);
    return nullptr;
}