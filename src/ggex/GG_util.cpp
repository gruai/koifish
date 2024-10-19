#include "../CLI_params.hpp"
#include "../ggex/GG_util.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "json.hpp"
#include <iostream>
#include <cstring>

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
    fprintf(stderr, "  --lora-out FNAME           path to save llama lora (default '%s')\n", params->fn_model_out.c_str());
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

    print_common_train_usage(argc, argv, &params->common);
}

bool CLI_params::operator!=(const CLI_params& other) const {
    return memcmp(this, &other, sizeof(other));
}

void CLI_params::Dump( )    {
    _INFO("%s::CLI_params: \n", exec_name.c_str());
    // _INFO(" n_vocab: %u", n_vocab);
    _INFO(" n_ctx:   %u", common.n_ctx);
    _INFO(" n_embd:  %u", n_embd);        
    _INFO(" n_ff:    %u", n_ff());
    _INFO(" n_head:  %u", n_head());
    _INFO(" n_head_kv:  %u", n_head_kv());
    _INFO(" n_layer: %u", n_layer);
    _INFO(" n_rot:   %u\n", n_rot);       
    _INFO(" f_norm_rms_eps:   %g", f_norm_rms_eps);   
    _INFO(" rope_freq_base:   %g", rope_freq_base);   
    _INFO(" rope_freq_scale:   %g", rope_freq_scale);
    _INFO("\n lora_r=%d  lora_alpha=%g \n", lora_r,lora_alpha); 
    _INFO(" NABLA = %s\n", nabla==0? "" : nabla==3 ? "Embed+AutoEncoder" : (nabla==2 ? "" : "qkv") ); 
    _INFO(" SIGMA = %s\n", sigma.c_str()); 
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
        _INFO("\r\n[%s] Failed to open %s !!!\n",__func__,jPath.c_str());
        return false;
    }
    jfile>>jConfig;
    std::string s = jConfig.dump(),s0;
    common.n_batch = jKV(jConfig,{      "train","batch"},common.n_batch );
    common.adam_n_iter = jKV(jConfig,{  "train","adam-iter"},common.adam_n_iter );
    common.adam_alpha = jKV(jConfig,{   "train","learning-rate"},common.adam_alpha );
    common.n_gradient_accumulation = jKV(jConfig,{ "train","optimizatioin","grad_accumulation"  },common.n_gradient_accumulation );
    lars_ratio = jKV(jConfig,{          "train","optimizatioin","lars_ratio"},lars_ratio );
    ZMUV_ratio = jKV(jConfig,{          "train","optimizatioin","ZMUV_ratio"},ZMUV_ratio );

    batch_sample = jKV(jConfig,{"data","batch_sample"},batch_sample ); 
    rSplit = jKV(jConfig,{"data","eval_split"},rSplit );
    std::vector<string> all_base;
    all_base = jKV_arr(jConfig,{"wiki","path"},all_base,false);
    for(auto path : all_base){
        if(path.empty() || path[0]=='#')
            continue;
        if( !std::filesystem::exists(path) )
            continue;
        if(model_title.empty()) //the first path is backbone
            model_title = remove_extension(base_name(path));
        fn_model_base.push_back(path);
    }
    
    serial_path = jKV(jConfig,{"data","serialize_path"},s0 );
    // eval_binpath = jKV(jConfig,{"data","eval_binpath"},s0 );   
    string a = batch_sample=="stacking" ? batch_sample : "";
    serial_path += a+"_["+model_title+"]_";       //std::to_string(1.0-rSplit)
    // eval_binpath += a+"_["+model_title+"]_.data";
    fp_train_data = jKV(jConfig,{"data","source"},fp_train_data );
    common.fn_train_data = "\0";
    assert( std::filesystem::exists(fp_train_data) );
    
    n_swarm = jKV(jConfig,{"train","swarm"},1 );
    common.save_every = jKV(jConfig,{"train","save-every"},common.save_every );
    eval_every = jKV(jConfig,{"train","eval-every"},eval_every );    
    gpt_every = jKV(jConfig,{"train","gpt-every"},gpt_every );    
    eval_every = eval_every<=0 ? 100000000 : eval_every;
    // if( eval_every>0 ){
    //     _INFO("\r\n%s  eval@every %d steps.",__func__,eval_every );
    // }
    common.seed = jKV(jConfig,{"seed"},common.seed );
    common.n_ctx = jKV(jConfig,{"model","ctx"},common.n_ctx );
    
     
    wiki_actor = jKV(jConfig,{"wiki","actor"},wiki_actor );
    wiki_logits = jKV(jConfig,{"wiki","logits"},wiki_logits );
    tpWiki = jKV(jConfig,{"wiki","induct"},tpWiki ); 

    info = jKV(jConfig,{"model","arch"},tpWiki ); 
    std::transform(info.begin(), info.end(), info.begin(), ::toupper);
    arch =  info=="MOE" ? NLP_MOE :
            info=="MAMBA" ? MODEL_ARCH::NLP_MAMBA : MODEL_ARCH::NLP_LLAMA;

    nabla = jKV(jConfig,{"model","nabla"},nabla );
    sigma = jKV(jConfig,{"model","sigma"},sigma ); 
    n_layer = jKV(jConfig,{"model","layer"},n_layer );
    
    common.custom_n_ctx = true;
    common.n_threads = jKV(jConfig,{"threads"},common.n_threads );
    common.n_gpu_layers = jKV(jConfig,{"n-gpu-layers"},common.n_gpu_layers );  
    
    // n_embd = jKV(jConfig,{"wiki","embd"},n_embd );

    fn_model_out = jKV(jConfig,{"model-out"},fn_model_out );    
    
    f_norm_rms_eps = jKV(jConfig,{"norm-rms-eps"},f_norm_rms_eps );
    rope_freq_base = jKV(jConfig,{"rope-freq-base"},rope_freq_base );
    rope_freq_scale = jKV(jConfig,{"rope-freq-scale"},rope_freq_scale );

    prompt = jKV(jConfig,{"gpt","prompt"},prompt );    

    dict_vae_dims = jKV(jConfig,{"dict","vae","dims"},dict_vae_dims ); 
    dict_latent_dim = jKV(jConfig,{"dict","latent_dim"},dict_latent_dim ); 
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

        if (consume_common_train_arg(argc, argv, &i, &common, &invalid_param)) {
            if (invalid_param) {
                break;
            } else if (common.print_usage) {
                train_print_usage(argc, argv, this);
                exit(0);
            }
        } else if (arg == "--model-base") {
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
            fn_model_out = argv[i];
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
            fn_model_out = argv[i];
        } else if (arg == "--only-write-lora") {
            only_write_model = true;
        } else if (arg == "--learning-rate"){
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            common.adam_alpha = std::stof(argv[i]);
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
            sigma = argv[i];
        }else if (arg == "--layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            n_layer = std::stoi(argv[i]);
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
    size_t ne00 = ggml_nelements(w);        assert(nIn>0 && nOut>0 && ne00==nIn*nOut);
    assert(nIn>nHeavy && nOut>nHeavy && nHeavy>0);
    float *A=Gensor2float(ctx0,w,flag);
    auto svd=std::make_shared<LoSVD<float>>(A,nIn,nOut,rank,1.0e-3,0); //1.0e-3
    assert(ga->type==GGML_TYPE_F32 && gb->type==GGML_TYPE_F32);
    if(!svd->Build( ))  {
        return -1;
    }else{
        if(ggml_nelements(ga)!=nIn*rank || ggml_nelements(gb)!=nOut*rank)    {
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
    size_t ne00 = ggml_nelements(w);
    assert(nIn>0 && nOut>0 && ne00==nIn*nOut);
    assert(nIn>nHeavy && nOut>nHeavy && nHeavy>0);
    float *A=Gensor2float(ctx0,w,flag);
    GST_TIC(tic);
    auto svd=std::make_shared<LoSVD<float>>(A,nIn,nOut,rank,1.0e-3,0); //1.0e-3
    float t0 = GST_TOC(tic);
    if(!svd->Build( ))  {
        return -1;
    }else{
        //GGML_TYPE_F16 tensor would call ggml_vec_dot_f16 with GGML_SIMD acceleration
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


// ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
// void ggml_vec_max_f32(const int n, float * s, const float * x);
// inline void ggml_vec_scale_f32(const int n, float * y, const float   v);
float SOFT_MAX(const int n, float * y, const float * x) {
    float x1 = -INFINITY;
    int i;
    // ggml_vec_max_f32(n, &x1, x);
    for (i = 0; i < n; ++i) {
        x1 = MAX(x1, x[i]);
    }
    ggml_float sum = 0.0;
#ifdef GGML_SOFT_MAX_ACCELERATE
    x1 = -x1;
    vDSP_vsadd(S, 1, &x1, S, 1, Mup);
    vvexpf(S, S, &Mup);
    ggml_vec_sum_f32(Mup, &sum, S);
#else
    // sum = ggml_vec_soft_max_f32(n, y, x, x1);
    for (i = 0; i < n; ++i) {
        float val = expf(x[i] - x1);
        sum += (ggml_float)val;
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
    ggml_float sum = 0.0;
    for (i = 0; i < n; ++i) {
        float val = expf(x[i] - x1);
        sum += (ggml_float)val;
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
struct ggml_tensor * ggml_flash_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        bool                  masked) {
    GGML_ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        is_node = true;
    }

    //struct ggml_tensor * result = ggml_dup_tensor(ctx, q);
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, q->ne);

    int32_t t = masked ? 1 : 0;
    ggml_set_op_params(result, &t, sizeof(t));

    result->op   = GGML_OP_FLASH_ATTN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;

    return result;
}
*/
// std::string jKV(const JSON& jConfig,const std::vector<std::string>&keys,const char* default_value,int flag){
//     std::string s1 = default_value;
//     std::string s2 = jKV(jConfig,keys,s1);
//     return s2;
// }

struct ggml_tensor * ggml_cross_entropy_loss_1(
        struct ggml_context         * ctx,
        struct ggml_tensor          * a,
        struct ggml_tensor          * b) {
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
}

void _T_repr_(hGensor t,const char*tab,char *buf,const GENSOR_INFO&info){
    if(t==nullptr)      return;
    const char* A = "d";
    if(t->grad!=nullptr){
        A = "P";
    }
    auto ne=t->ne;
    sprintf(buf+strlen(buf),"%s %s %s '%s' \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab,info.sX.c_str(), A,
        t->name,ne[0], ne[1], ne[2], ne[3], ggml_type_name(t->type));
}

void _T_repr_(hGensor t,const char*tab,char *buf,int typ){
    if(t==nullptr)      return;
    const char* A = "d";
    if(t->grad!=nullptr){
        A = "P";
    }
    auto ne=t->ne;
    switch(typ){
    case 1:
        sprintf(buf+strlen(buf),"%s%s '%s' \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab,A,t->name,ne[0], ne[1], ne[2], ne[3], ggml_type_name(t->type));
        break;
    default:
        sprintf(buf+strlen(buf),"%s%s '%s' %.3lf(M)\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n",tab, 
        A,t->name,ggml_nelements(t)/1.0e6,ne[0], ne[1], ne[2], ne[3], ggml_type_name(t->type)); 
        break;
    }
    
}