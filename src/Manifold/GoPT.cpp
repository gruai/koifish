/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Generate some nonsense on Prompt
 *  \author Yingshi Chen
 */
#include "GoPT.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#include "../Device/Pipe.hpp"
#include "../Utils/GST_Application.hpp"
#include "../Utils/GST_rander.hpp"
#include "Fish.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

namespace fs = std::filesystem;

#define LOG  //

#ifdef __USE_GGML__
#else
std::vector<hWIKI> WIKI::MakeInstance(const std::string nam_, struct CLI_params& params, int flag) {
    std::vector<hWIKI> wikis;
    if (params.tpWiki != "off") {  // wiki is so heavy(ugly) that only load one instance here!
        for (auto path : params.fn_model_base) {
            assert(0);
            // hWIKI wiki = std::make_shared<LAMA>(params,path);
            // wikis.push_back(wiki);
        }
    }
    return wikis;
}
#endif

double WIKI::InductLogits(const CLI_params& config, int nSampInBatch, std::vector<TOKEN_ID>& tok_ids, struct ggml_tensor* target_label, int flag) {
    if (!isInduct())
        return -1.0;

    Reset();  // Timing bottleneck!!! for the crazy design of llama.cpp
    Decode(tok_ids, 0, 0x0, true);
    const float *all_logits = GetLogits(n_vocab, tok_ids.size(), 0), *logit;
    size_t k, j, i;                                                // exLogits->ne[0];
    int n_ctx = config.n_ctx(), n_dialect = mapT2T.size(), token;  // target_label->ne[1],
    double a1, a2, nrm = 0;
    float *p = teach == WIKI::_TARGET ? new float[n_vocab] : nullptr, *target = nullptr;
    if (flag < 0) {  // CHILD_0909_WIKIS
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
        }else*/
        {
            target = exLogits + nSampInBatch * n_ctx * n_vocab;
            memcpy((void*)target, (void*)all_logits, sizeof(float) * n_ctx * n_vocab);  // memcpy(g->data+off,(void*)(logits),ld2);
        }
    } else {
#ifdef __USE_GGML__
        for (k = 0; k < nSampInBatch; ++k) {
            const float* from = all_logits + k * n_vocab;
            a1                = NRM_2_((float*)(from), n_ctx * n_vocab);
            nrm               = max(nrm, a1 / n_vocab);
            if (teach == WIKI::_TARGET) {
                assert(exLogits == nullptr);
                for (j = 0; j < n_ctx; j++) {
                    logit  = from + j * n_vocab;
                    target = (float*)target_label->data + (k * n_ctx + j) * n_vocab;
                    //  SOFT_MAX_minus(n_vocab,target,logit);
                    // SOFT_MAX(n_vocab,p,logit);
                    for (a1 = 0, a2 = 0, i = 0; i < n_vocab; i++) {
                        a1 += target[i];
                        a2 += p[i];
                        target[i] -= p[i];
                    }
                    // SOFT_MAX(n_vocab,p,target);     //  !!!No converge!!!   @home/cys/rnd/lic/log/eval/08_21_wiki_target_no_converge.info
                    memcpy(target, p, sizeof(float) * n_vocab);
                    // todo - cys 20240821: MSE loss
                }
            } else {
                assert(exLogits != nullptr);
                if (exLogits != from) {
                    // target = (float*)exLogits->data+k*n_ctx*n_vocab;
                    target = (float*)exLogits + k * n_ctx * n_vocab;
                    memcpy((void*)target, (void*)from, sizeof(float) * n_ctx * n_vocab);
                }
            }
        }
#endif
    }
    delete[] p;
    return nrm;
}

static Grusoft::GRander rand_gopt(42 * 666);
int Sample_CDF_T(int n, floatLogits* logits, float minp, float temperature, uint64_t* rng_seed, int flag = 0x0) {
    float coin = rand_gopt.NextFloat_01();  // random_f32(rng_seed);
    // find max logit; we will use this to derive minp cutoff (in log space), since minp is scale-invariant (wrt softmax)
    float max_logit = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        float a   = T2Float(logits + i);
        max_logit = a > max_logit ? a : max_logit;
    }

    // exp(logit / temp) <= exp(max_logit / temp) * minp -> logit <= max_logit + log(minp) * temp
    float logit_cutoff = max_logit + logf(minp) * temperature;

    // convert from logits to probabilities in-place while simultaneously doing (unscaled) softmax; we'll rescale later
    floatLogits* probs    = logits;
    int fallback          = 0;
    float cumulative_prob = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = T2Float(logits + i);  // logits[i];
        if (a >= logit_cutoff) {
            probs[i] = (floatLogits)expf((a - max_logit) / temperature);
            cumulative_prob += (float)(probs[i]);
            fallback = i;  // for fallback due to rounding errors
        } else {
            probs[i] = (floatLogits)0.0f;
        }
    }

    // sample from the truncated list
    float r   = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += (float)(probs[i]);
        if (r < cdf) {
            return i;
        }
    }
    return fallback;  // in case of rounding errors
}

int Sample_CDF(int n, float* preP, uint64_t* rng_seed, int flag = 0x0) {
    float sum = 0, cdf = 0, pMin, pMax, a;
    int j, next_token  = -1;
    for (pMin = FLT_MAX, pMax = -FLT_MAX, j = 0; j < n; j++) {
        a    = (float)(preP[j]);
        pMin = min(a, pMin);
        pMax = max(a, pMax);
    }
    for (sum = 0, j = 0; j < n; j++) {
        a       = (float)(preP[j]);
        preP[j] = exp(a - pMax);
        sum += a;
    }
    assert(sum > 0 && sum < FLT_MAX);
    float coin = rand_gopt.NextFloat_01();  // random_f32(rng_seed);
    for (cdf = 0, j = 0; j < n; j++) {
        cdf += (float)(preP[j]);
        if (coin < cdf / sum) {
            next_token = j;
            break;
        }
    }
    return next_token;
}

int GGUF_list(CLI_params& config) {
    std::vector<string> paths;
    std::string root = "/media/cys/E0/", path;
    root             = "/home/cys/rnd/lic/models/";
    for (const auto& entry : fs::directory_iterator(root)) {
        fs::path filePath = entry.path();
        if (filePath.extension() == ".gguf")
            paths.push_back(entry.path());
    }
    int nP = paths.size(), i = 0;
    FILE* fp = fopen("./log/GGUF_list.log", "wt");
    string sToken, info, suffix = "\t", prefix;
    fprintf(fp, "%s LOAD %d @%s\n", __func__, nP, root.c_str());
    fflush(fp);
    for (auto path : paths) {
        // path = "/media/cys/E0/LLaMA3-8B_mmproj-Q4_1.gguf";      //only for debug
        auto param_1       = config;
        param_1.wiki_actor = "OnlyTokenizer";
        param_1.tpWiki     = WIKI::_OFF;
        // param_1.fn_model_base = path;

        // hFISH fish_0 = Fish::MakeInstance("GGUF_",param_1,0x0);
        // fish_0->Dump(0x0);
        GST_TIC(tic);
        fprintf(fp, "%d: \"%s\" ...\n", i++, path.c_str());
        fflush(fp);
        try {
            hWIKI wiki = nullptr;  //  WIKI::MakeInstance
            assert(wiki != nullptr);
            info = wiki == nullptr ? "" : wiki->__repr__(suffix, prefix);
        } catch (const std::exception& e) {
            info = std::string(e.what());
        } catch (...) {
            info = "!!! UNKNOW EXCEPTION !!!";
        }

        fprintf(fp, "\t %s  T=%.3g\n", info.c_str(), GST_TOC(tic));
        fflush(fp);
    }
    fclose(fp);
    return 0x0;
}

int run_caml(const char* prompt, int flag);

int fish_1(CLI_params& config) {
    auto param_1 = config, param_2 = config;
    param_1.tpWiki         = "logits";
    param_1.common.n_batch = 1;
    param_2.tpWiki         = "";
    param_2.common.n_batch = 1;  //
    hFISH fish_0           = Fish::MakeInstance("BIG_", param_1, {}, Fish::ROLE_TYPE::COMMON, 0x0);
    hFISH fish_1           = Fish::MakeInstance("local_", param_2, fish_0.get(), 0x110);
    fish_0->Dump(0x0);
    fish_1->Dump(0x0);

    vector<float> logits;
    vector<TOKEN_ID> piffle;
    // Need piffle to samploader
    fish_0->LocalFeeling(nullptr, logits);
    fish_1->CopyWeight(fish_0.get());
    fish_1->LocalFeeling(nullptr, logits);
    return 666;
}

int GPT_work(CLI_params& config) {
    //  GRUS_Get_SystemInfo
    _INFO("[%s] threads=%d \n%s\n", __func__, std::thread::hardware_concurrency(), "");  // llama_print_system_info()
    // ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    DEBUG.SelfAttention_noraml = 0;
    DEBUG.NO_loss              = true;
    DEBUG.graph_dump           = 1;
    // config.wiki_actor="";    //only for debug

    config.isOnlyGPT     = true;
    config.passLoadToken = true;
    bool isMakeFish      = config.is({"wiki", "actor"}, "copy") || config.wiki_actor == "OnlyTokenizer";
    hFISH fish           = nullptr;
    vector<hWIKI> wikis  = WIKI::MakeInstance("", config, 0x0);
    if (config.fn_model_base.size() > 0 && !isMakeFish) {
        for (auto wiki : wikis) {
            if (wiki->isOnlyTokenizer)
                assert(wiki->teach == WIKI::_OFF);
            else
                wiki->teach = WIKI::_LOGITS;
        }
    }

    if (isMakeFish) {
        fish = Fish::MakeInstance("Fish_", config, wikis, Fish::ROLE_TYPE::COMMON, 0x110);
        if (fish == nullptr || fish->SAFETENSOR_Serialize(config.ckp_in[0], false)) { /*!fish->LoadCheckPoint_(config.ckp_in.data())*/
            _ERROR("%s has no WIKI or FISH!\n", __func__);
            return 0;
        }
    }
    //  hGENERATOR gpt = std::make_shared<GeneratOnPrompt>(params,0x0);
    //  hGENERATOR gpt = std::make_shared<GOPT_infinite>(params,0x0);
    hGENERATOR gpt = std::make_shared<GOPT_Metropolis>(config, wikis, fish.get(), 0x0);
    if (gpt->Init(config.prompt)) {
        for (int i = 0; i < 10; i++) {
            if (!wikis.empty())
                wikis[0]->Reset();  // to get same results each run
            gpt->Generate(i);
            // break;
        }
    } else {
        return -1;
    }

    return 666;
}

hGENERATOR GeneratOnPrompt::MakeInstance(struct CLI_params& config, arrHWIKI& wikis, const Fish* fish_0, int flag) {
    hGENERATOR gopt = nullptr;

    switch (config.ChatMode()) {
        case CHATML_ASSIST:
        case CHATML_THINK: {
            gopt = std::make_shared<GeneratOnPrompt>(config, wikis, fish_0, 0x0);
        } break;
        case CHAT_SMOKE: {
            gopt = std::make_shared<GOPT_Metropolis>(config, wikis, fish_0, 0x0);
            if (gopt != nullptr && gopt->Init(config.prompt)) {
                // gopt->Generate(0); //only for debug
            } else {
                gopt.reset();
                gopt = nullptr;
            }
            const char* promt              = "when the smoke is going down,";
            std::vector<TOKEN_ID> some_inp = {9493, 279, 16603, 374, 2133, 1523, 11};  // const char* promt = "when the smoke is going down,";
        } break;
        default:
            assert(0);
    }

    // wiki->Decode(embd_inp,0,0x0);
    // wiki->Answer(embd_inp);      //only for debug
    return gopt;
}

CHAT_MODE GeneratOnPrompt::ChatMode(int flag) const {
    if (config.isOnlyGPT) {
        return CHAT_MODE::CHATML_ASSIST;
    } else
        return CHAT_MODE::YABA;
}

std::string GeneratOnPrompt::GetPrompt(int flag) { return config.prompt; }

/*
    1.   prompt=>embd_inp
    2.  =>batch@Decode  =>batch@llama_decode
    3.  =>ubatch    then llama_build_graph


*/
void GeneratOnPrompt::InitInput(int flag) {
    // if (params.chatml) {
    //     GetPrompt() = "<|im_start|>system\n" + GetPrompt() + "<|im_end|>";
    // }
    _INFO("[GPT] tokenize the prompt\n");
    hGensor input = fish_1 != nullptr ? fish_1->Input() : nullptr;
    TOKEN_ID bo   = -1;
    if (input != nullptr)
        dialogs->InitOneSamp(config.prompt, input, nullptr, 0x110);
    switch (_arch) {
        case NLP_GPT2_char:

            // embd_inp.clear();
            // embd_inp.push_back(0);
            // for(int i=0;i<GetPrompt().length();i++){
            //     int id = (int)(GetPrompt()[i]);
            //     embd_inp.push_back(id);
            // }
            break;

        default: {
            int iRet = wiki0->STR2T(GetPrompt(), embd_inp);
            // const bool add_bos = llama_add_bos_token(model);
            // embd_inp = ::llama_tokenize(ctx, GetPrompt(), add_bos, true);
            bo = embd_inp[0];
        } break;
    }

    return;
}

// Deprecated
GeneratOnPrompt::GeneratOnPrompt(struct gpt_params& par_, int flag) {
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

void GeneratOnPrompt::Clear() {
    // write_logfile(ctx, params, model, input_tokens, output_ss.str(), output_tokens);
#ifdef __USE_GGML__
    if (ctx_guidance) {
        llama_free(ctx_guidance);
    }
    if (wikis.empty()) {
        llama_free(ctx);
        // llama_free_model(model);
    }

    llama_backend_free();
#endif

    // FREE_a(_logits);
}
// only for debug
GeneratOnPrompt::GeneratOnPrompt(CLI_params& cp_, arrHWIKI& wiki_, const Fish* hG_, int flag) : config(cp_), fish_0(hG_), wikis(wiki_) {
    if (fish_0 != nullptr) {
        samp_params       = config.chat_sampler;
        auto gang_param   = config;
        gang_param.tpWiki = "off";
        assert(gang_param.tpWiki == "off");
        gang_param.common.n_batch = 1;
        fish_1                    = (Fish*)fish_0;
        // fish_1 = Fish::MakeInstance("4GPT_",gang_param,wikis,Fish::ROLE_TYPE::SWARM_FOLLOWER,0x110);        //  isLocalInfer = flag==0x110;
        // fish_1->Dump(0x0);
        int n_vocab     = fish_1->nClass();
        nCanTopK        = (samp_params.top_k < n_vocab) ? samp_params.top_k : n_vocab;
        Head4Token* cls = ((Fish*)fish_0)->GetNeuron<Head4Token>("Head4Token", 0);

        cpuLogits.Init(n_vocab, true, cls->preLogits, 0x0);
        if (!samp_params.isSampleCPU) {
            gpuLogits.Init(n_vocab, false, cls->preLogits, 0x0);
        }
        rng_state = 20251021;
        _arch     = fish_0->arch;
    } else {
        _arch = config.ModelArch();
    }
    if (!wikis.empty())
        wiki0 = wikis[0];
}

bool GeneratOnPrompt::Init(const std::string& prompt_, int flag) {
    // std::tie(model, ctx) = llama_init_from_gpt_params(params);
    int n_vocab = 0;
    if (fish_1 != nullptr) {
        /*  may deprecated*/
        dialogs = std::make_shared<SampLoader>(fish_1, "gpt", true);
        dialogs->Prepare(fish_1->hOPT.get(), fish_1->tsEval[0]);
        dialogs->isRecycle = false;
        dialogs->type      = SampLoader::TYPE::DT_EVAL;
        n_vocab            = fish_1->nClass();
    }
    cpuLogits.Init(n_vocab, true, nullptr);
    assert(0);
    if (wikis.empty()) {
        CHILD_0909_WIKIS
        n_ctx = config.n_ctx();

        // _logits = new floatLogits[n_vocab];

        InitInput();
        return true;
    } else {
        n_ctx   = wiki0->nCTX();
        n_vocab = wiki0->n_vocab;
    }
    //_logits = new floatLogits[n_vocab];

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
    if (n_ctx > n_ctx_train) {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }
    LOG("\n");

    // prompt = prompt_;
    if (GetPrompt() != "") {
        // GetPrompt() = prompt;
        Tokenize(flag);
    }

    // ga_n = params.grp_attn_n;               ga_w = params.grp_attn_w;
    // if (ga_n != 1)    {
    //     assert(ga_n > 0 && "grp_attn_n must be positive");                         // NOLINT
    //     assert(ga_w % ga_n == 0 && "grp_attn_w must be a multiple of grp_attn_n"); // NOLINT
    //                                                                                     // assert(n_ctx_train % ga_w == 0     && "n_ctx_train must be a
    //                                                                                     multiple of grp_attn_w");    // NOLINT
    //     // assert(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
    //     LOG("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    // }
    // LOG("\n\n");

    return true;
}

void GeneratOnPrompt::DisplayEmbd(bool input_echo, int n_consumed, int flag) {
    NLP_AutoRegressive* dolphin = dynamic_cast<NLP_AutoRegressive*>(fish_1);
    std::string token_str;
    if (input_echo && display) {
        for (auto id : tokens) {
            // switch(_arch)    {
            // case NLP_GPT2_char:
            //     token_str = (char)(id);
            //     break;
            // default:
            //     token_str = llama_token_to_piece(ctx, id);
            //     break;
            // }

            if (tokens.size() > 1) {
                input_tokens.push_back(id);
            } else {
                output_tokens.push_back(id);
                output_ss << token_str;
            }
        }
        if (dolphin != nullptr) {
            _INFO("[Generate]_%d {%s}%s", tokens.size(), dialogs->sentence.c_str(), token_str.c_str());
        } else {
            // token_str = llama_token_to_piece(ctx, tokens[0]);
            printf("%s", token_str.c_str());
        }

        fflush(stdout);
    }
    // reset color to default if there is no pending user input
    if (input_echo && (int)embd_inp.size() == n_consumed) {
        // console::set_display(console::reset);
        display = true;
    }
}

/*
    when the smoke is going down, that's when it's like, 'Abby, you really need to go to the doctor,' this was not the case with me or my son.
        We were between his second and third birthday and it was time to go to the doctor and not go to the doctor--and that was the reason he was going to the
   doctor and not going to the doctor. On the two sides of the age there was a lot of room for two oth
*/
TOKEN_ID GOPT_Metropolis::Sample(int idx, bool is_resampling) {
    int j, nVocab = fish_1 == nullptr ? wiki0->n_vocab : fish_1->nClass();  //, j;
    hSAMP samp              = (dialogs == nullptr || dialogs->empty()) ? nullptr : dialogs->SampAt(0);
    float* _logits          = (float*)cpuLogits.logits;
    hWIKI wiki              = wikis.size() > 0 ? wikis[0] : nullptr;
    WIKI::INDUCT_MODE teach = wiki == nullptr ? WIKI::_OFF : wiki->teach;
    assert(idx == -1);
    const float* wLog = nullptr;
    float l1 = 0, sum1 = 0, l2 = 0, delta, a;
    if (teach == WIKI::_OFF) {
    } else {
        wLog = wiki->GetLogits(nVocab, 1);
        for (l2 = 0, j = 0; j < nVocab; j++) {  //  -8.23 -1.1958
            a = wLog[j];
            l2 += a * a;
            _logits[j] = a;
            // pMin = min(a,pMin);     pMax = max(a,pMax);
        }
    }

    if (fish_1 != nullptr) {  // time bottleneck, so share wiki to reduce time & memory
        assert(0);
        /*assert(x_logits.size() == nVocab);
        // SOFT_MAX(x_logits);
        switch (teach) {
            case WIKI::_OFF:
                for (j = 0; j < nVocab; j++) {
                    a = x_logits[j];
                    sum1 += a;
                    l1 += a * a;
                    _logits[j] = a;
                }
                break;
            case WIKI::_LOGITS:
                for (j = 0; j < nTokens; j++    ) {
                    a = x_logits[j];                    l1+=a*a;
                    _logits[j] = a;
                }
                delta = l2==0 ? 0 : sqrt(l1)/sqrt(l2)-1.0;
            break;
            case WIKI::_TARGET:
                SOFT_MAX(nVocab, _logits, wLog);
                for (j = 0; j < nVocab; j++) {
                    a = x_logits[j];
                    l1 += a * a;
                    _logits[j] += a;
                }
                delta = l2 == 0 ? 0 : sqrt(l1) / sqrt(l2) - 1.0;
                break;
            case WIKI::_LOGITS_SCALE:
                for (j = 0; j < nVocab; j++) {
                    a = x_logits[j];
                    sum1 += a,                    l1 += a * a;
                    l2 += _logits[j] * _logits[j];
                    // a = max(a,0.f);      relu
                    _logits[j] *= a;
                }
                delta = l2 == 0 ? 0 : sqrt(l1) / sqrt(l2);
                break;
            default:  //  WIKI::_LOGITS:
                for (j = 0; j < nVocab; j++) {
                    a = x_logits[j];
                    sum1 += a;
                    l1 += a * a;
                    l2 += _logits[j] * _logits[j];
                    _logits[j] += a;
                }
                delta = l2 == 0 ? 0 : sqrt(l1) / sqrt(l2);
                // assert(fabs(sum1-1.0)<0.001);       //softmax->logits
        }
        delta_a += delta;
        delta_max = max(delta_max, delta);*/
    }
    int next_token = Sample_CDF(nVocab, _logits, &rng_state);
    return next_token;
}

// only for debug
static inline unsigned int random_u32(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static inline float random_f32(uint64_t* state) { return (random_u32(state) >> 8) / 16777216.0f; }

static inline int sample_argmax(int n_vocab, float* logits) {
    int max_i   = 0;
    float max_p = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_p) {
            max_i = i;
            max_p = logits[i];
        }
    }
    return max_i;
}

/*static int compare_prob_desc(const void* a, const void* b) {
    ProbIndex* pa = (ProbIndex*)a;
    ProbIndex* pb = (ProbIndex*)b;
    if (pa->prob > pb->prob)
        return -1;
    if (pa->prob < pb->prob)
        return 1;
    return 0;
}*/

TOKEN_ID GeneratOnPrompt::Sample_cpu(int idx, bool isSorted) {
    TOKEN_ID id = 0;
    int n_vocab = fish_1 == nullptr ? wiki0->n_vocab : fish_1->nClass();
    if (samp_params.temperature == 0.0f || samp_params.top_k == 1) {
        return sample_argmax(n_vocab, cpuLogits.logits);
    }

    // nCanTopK = (samp_params.top_k < n_vocab) ? samp_params.top_k : n_vocab;
    if (isSorted) {
    } else {
        cpuLogits.UpdateLogits();
        // for (int i = 0; i < n_vocab; i++) {
        //     cpuLogits.logits[i]  = CU_T2FLoat<>(_logits[i]);
        //     cpuLogits.index[i] = i;
        // }
        cpuLogits.quick_select(n_vocab, nCanTopK);
        // floatLogits max_logit = cpuLogits.logits[0];  // Ch probindex[0].prob;
    }

    float prob_sum = 0.0f;
    for (int i = 0; i < nCanTopK; i++) {
        float prob          = expf(float(cpuLogits.logits[i] - cpuLogits.maxLogit) / samp_params.temperature);
        cpuLogits.logits[i] = prob;
        prob_sum += prob;
    }
    for (int i = 0; i < nCanTopK; i++) {
        cpuLogits.logits[i] /= prob_sum;
    }
    int nPick = nCanTopK;
    if (samp_params.top_p > 0.0f && samp_params.top_p < 1.0f) {
        cpuLogits.SortPair(nCanTopK);
        float cumulative_prob = 0.0f;
        int last_idx          = nCanTopK - 1;
        for (int i = 0; i < nCanTopK; i++) {
            cumulative_prob += float(cpuLogits.logits[i]);
            if (cumulative_prob > samp_params.top_p) {
                last_idx = i;
                break;
            }
        }
        nPick    = last_idx + 1;
        prob_sum = cumulative_prob;
    }

    float coin = random_f32(&rng_state) * prob_sum;  //  0.00294704828
    float cdf  = 0.0f;
    for (int i = 0; i < nPick; i++) {
        cdf += (float)(cpuLogits.logits[i]);
        if (coin < cdf) {
            return cpuLogits.index[i];
        }
    }
    id = cpuLogits.index[nPick - 1];  // probindex[nPick - 1].index;
    return id;
}

void GeneratOnPrompt::OnAntiPrompt(int flag) {
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

std::string GeneratOnPrompt::T2STR(TOKEN_ID tok, int flag) {
    NLP_AutoRegressive* dolphin = dynamic_cast<NLP_AutoRegressive*>(fish_1);
    std::string token_str;
    if (dolphin != nullptr)
        token_str = dolphin->hDict->T2STR(tok);
    else {
        token_str = wikis[0]->T2STR(tok);
    }
    return token_str;
}

bool GeneratOnPrompt::Inference(hSAMP samp, int& n_past, int flag) {
    int n_eval = (int)tokens.size();
    // LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, tokens).c_str());
    bool bRet = false;
    if (fish_1 != nullptr) {
        fish_1->UpdateNCTX(dialogs->nLeastCTX());
        fish_1->CopyWeight(fish_0);
        // bRet = fish_1->LocalFeeling(dialogs, cpuLogits.logits, 0);
        assert(bRet);
    }
    /*if(flag==0x100)     {   //  debug flag of "only fish"
        // vector<float> preP;
        // float fLos = fish_1->LocalFeeling(&dialogs,preP);
        // bRet = true;
    }else*/
    if (wiki0 != nullptr && wiki0->teach != WIKI::_OFF) {
        bRet = wiki0->Decode(tokens, 0, n_past, false);
        // bDecode = llama_decode(ctx, llama_batch_get_one(&tokens[0], n_eval, n_past, 0)) >= 0;
    }

    if (!bRet) {
        LOG("%s_%d : failed @%s\n", __func__, samp->len, dialogs->sentence.c_str());
        return 1;
    }
    n_past += n_eval;
    // LOG("n_past = %d n_remain=%d\n", n_past,n_remain);
    return bRet;
}

int GeneratOnPrompt::Generate(int nJob, int flag) {
    GST_TIC(tic);
    hSAMP samp = (dialogs == nullptr || dialogs->empty()) ? nullptr : dialogs->SampAt(0);
    output_tokens.clear();
    delta_max   = 0;
    delta_a     = 0;
    string info = "only fish", sTok;
    if (wiki0 != nullptr) {
        info = wiki0->teach == WIKI::_OFF ? "only fish" : "WIKI";
        // ctx_sampling = nullptr;
    }
    LOG("<--- GeneratOnPrompt %s job=%d logits_all=%d fish=%s teach=%d\n", info.c_str(), nJob, 0, fish_1 == nullptr ? "" : fish_1->Name().c_str(),
        wiki0 == nullptr ? -1 : wiki0->teach);
    rng_state = config.common.seed;
    // LOG("%s logits_all=%d\n", __func__, );
    // bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();
    int n_past = 0, n_remain = n_predict, n_session_consumed = 0, ga_i = 0, max_embd_size = n_ctx - 4;
    tokens.clear();  // embd_guidance.clear();
    LOG("embd_inp.size(): %d \n", (int)embd_inp.size());
    tokens = embd_inp;
    _INFO("%s", config.prompt.c_str());

    while ((--n_remain >= 0 && !is_antiprompt)) {
        if (tokens.empty())
            break;
        assert((int)tokens.size() <= max_embd_size);
        // assert(ga_n == 1);
        // assert(ctx_guidance == nullptr);
        if (!Inference(samp, n_past))
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

        if (!tokens.empty()) {
            session_tokens.insert(session_tokens.end(), tokens.begin(), tokens.end());
            n_session_consumed = session_tokens.size();
        }

        tokens.clear();           //  kv cache only need one token
        TOKEN_ID tok = Sample();  // llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
        if (tok < 0) {
            _INFO("\t<E>");
            break;
        }
        if (samp != nullptr) {
            dialogs->hTokens->Append(tok);
            samp->len++;
        }
        sTok = T2STR(tok);
        if (samp != nullptr)
            _INFO("[Generate]_%d {%s}%s\n", samp->len, dialogs->sentence.c_str(), sTok.c_str());
        else {
            _INFO("%s", sTok.c_str());
            fflush(stdout);
        }
        tokens.push_back(tok);

        // if not currently processing queued inputs;
        // if ((int)embd_inp.size() <= n_consumed)        {            // check for reverse prompt in the last n_prev tokens
        //     OnAntiPrompt(0x0);
        //     // if(ctx_sampling!=nullptr)   OnInteractive(n_past, n_consumed, n_remain, 0x0);
        // }
        // end of text token
        if (tok == eos /*!tokens.empty() && tokens.back() == eos*/) {
            LOG(" [end of text]\n");
            break;
        }
    }

    _INFO("\n delta=%.3g(%.3g) T=%gs --------------->\n", delta_max, delta_a / n_predict, GST_TOC(tic));
    return 0x0;
}

int GeneratOnPrompt::Generate_v0(int nJob, int flag) { return 0x0; }

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
    assert(!embd_inp.empty());
    n_keep = (int)embd_inp.size();
    // LOG("prompt: \"%s\"\n", log_tostr(GetPrompt()));
    // LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens

    // negative prompt
    std::vector<TOKEN_ID> guidance_inp;
    int guidance_offset     = 0;
    int original_prompt_len = 0;

    if ((int)embd_inp.size() > n_ctx - 4) {
        LOG("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;

    // LOGLN(
    //         "recalculate the cached logits (check): embd_inp.empty() %s, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu,
    //         embd_inp.size() %zu", log_tostr(embd_inp.empty()), n_matching_session_tokens, embd_inp.size(), session_tokens.size(), embd_inp.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        // LOGLN("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    return 0x0;
}

//  Keeping the prefill process uncompressed is crucial for performance maintenance
int Prefill(hFISH fish, int enable_thinking) { return 0x0; }

int OnEOS(hFISH fish, int flag = 0x0) {
    hChater gopt = fish->GetGenerator();
    _INFO("[MEMORY] %s\t%s\n", SUM::GPU_Info(0x0).c_str(), SUM::CPU_Info(0x0).c_str());
    _INFO("\t quant=%s\t DEBUG_switch={generate=%d %d} QKV=%d FFN=%d\n", SUM::sQuantInfo.c_str(), DEBUG.verGenerate, DEBUG.T_cuQK, DEBUG.verInferQKV,
          DEBUG.verInferFFN);
    _INFO("\n");  // next turn
    return 0x0;
}

/*
    hello
        Hello! How can I assist you today?
    More questions:
        1. Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?
        2. just keep asking 142857*7, would get some strange answers!
        3. prime factorization of 996   99997 failed @ 4B

        4. How many games did Arsenal FC go unbeaten during the 2003-2004 season of the English Premier League         correct_answer: "38"
        5. Write a function to print the Fibonacci sequence to the nth digit, but write and comment it like a pirate
        6. I get out on the top floor (third floor) at street level. How many stories is the building above the ground?
        7. "无知觉明",无是指什么? 知是指什么? 觉是指什么? 明是指什么?
        8. 天命玄鸟,降而生生. 玄鸟是什么鸟?
        9. 尚书·商书·胤征

    Common sense
        What is the capital of Shanghai?
        Who wrote the play Romeo and Juliet?
        In which year did the Titanic sink?
        What is the chemical symbol for the element gold?
        What is the longest river in the world?
*/
int Fish::Chat(int enable_thinking, LIFE_PHASE outer_phase, int flag) {
    Statistic(0x100);

    int seq_len           = config.chat_sampler.seq_len;
    int num_prompt_tokens = 0, user_turn = 1, next, token, generated_tokens = 0, nRound = 0;  // pos = 0,
    TOKENS prompt_tokens;
    hTokenizer tokenizer = GetTokenizer();
    double start_time = 0, eval = 0;
    string cur_answer, rendered_prompt;
    hChater gopt  = GetGenerator();
    hBATCH hBatch = GetCurBatch(true);
    assert(hBatch->hostToken->ne[0] >= seq_len);  // batch = hBatch->hostToken->ne[1] may >1
    GST_Application* hApp = GST_Application::GetInstance();
    // DEBUG.T_generate_most_layer = 1;
    DEBUG.verGenerate = DEBUG.cmd_p1;  // use this flag to comparse accu/time of different version
    // DEBUG.verGenerate     = 1;
    DEBUG.T_cuQK          = 0;
    DEBUG.T_kvcache_quant = 0;
    g_dump_level          = 1;

    while (hApp->iRunning() > 0) {
        if (user_turn) {
            num_prompt_tokens = hBatch->FillPrompt(this, DEBUG.prompts, {}, nRound);
            generated_tokens  = 0;
            cur_answer        = "";
            user_turn         = 0, nRound++;
            // hLoader->InitOneSamp(rendered_prompt, nullptr, fish.get(), 0x110);
            _INFO("\n");
            for (int i = 0; i < num_prompt_tokens - 1; i++) {  // prefill
                eval = Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
                hBatch->tok_pos++;
                if (hBatch->tok_pos == 1) {  // nRound == 2
                    DEBUG_HERE;
                    // K_EXIT(KOIFISH_EXIT_DEBUG);
                }
            }
        }
        if (hApp->iRunning() <= 0) {
            _WARN("\n%s[APP] Stop running! code=%d%s\t", COLOR_YELLOW, hApp->iRunning(), COLOR_RESET);
            break;
        }

        start_time = GST_ms();
        SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
        if (DEBUG.verGenerate) {  //    Deprecated
            QWEN3_PIPE qwen_pipe(shared_from_this(), 0x0);
            T_generate_(shared_from_this(), &qwen_pipe, config.model.tpActivation, 1);
        } else {
            eval = Evaluate(DL_BATCH_UPATE::BATCHofEMBED);
        }
        gopt->VerifyLogits();
        hBatch->tok_pos++;
        // K_EXIT(KOIFISH_EXIT_DEBUG);

        // _INFO(" %d[%d->%d]", pos, token, next), fflush(stdout);

        token = gopt->Sample(-1);  // 3347
        generated_tokens++;
        if (token == tokenizer->eos_id || hBatch->tok_pos >= seq_len) {  //  stop generation if get EOS token
            double elapsed_s = (double)(GST_ms() - start_time) / 1000.0;
            double tps       = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
            if (hBatch->tok_pos >= seq_len) {
                if (outer_phase == P_TRAIN)
                    return 0x0;
                _WARN("%scontext window full!%s\t", COLOR_YELLOW, COLOR_RESET);
            }
            _INFO("\n%s[%.2f tk/s, %d tokens in %.2fs(qkv=%.3fs ffn=%.3fs PreLogits=%.3fs X=%.3fs)]%s\n===================================\n", COLOR_GREEN, tps,
                  generated_tokens - 1, elapsed_s, SUM::tQKV_forw / 1.0e6, SUM::tFFN / 1.0e6, SUM::tPreLogits / 1.0e6, SUM::tX1 / 1.0e6, COLOR_RESET);

            user_turn = 1;
            cur_answer += "\t\t" + SUM::sQuantInfo;
            STR2FILE("chat.csv", cur_answer, nRound == 1 ? std::ofstream::out : std::ofstream::app);
            OnEOS(shared_from_this());
            if (nRound == DEBUG.prompts.size()) {  // only for debug
                return 0x0;
            }
            continue;
        }
        hBatch->Set(hBatch->tok_pos, 0, 0, 0, token);

        static int in_thinking_section = 0;
        static int in_bold_section     = 0;
        if (hBatch->tok_pos == num_prompt_tokens) {  // first token of the response
            in_thinking_section = enable_thinking;   // reset thinking state
            in_bold_section     = 0;                 // reset bold state
            if (in_thinking_section) {
                _INFO(COLOR_YELLOW);
            }
        }

        const char* piece = tokenizer->T2STR(token).c_str();  // decode(tokenizer, token);
        if (strcmp(piece, "</think>") == 0) {
            in_thinking_section = 0;
            if (!in_bold_section) {
                _INFO(COLOR_RESET);
            }
        } else {
            const char *current_pos = piece, *marker;
            while ((marker = strstr(current_pos, "**")) != NULL) {
                // print the text before the marker
                fwrite(current_pos, 1, marker - current_pos, stdout);

                // flip the bold state and change color accordingly
                in_bold_section = !in_bold_section;
                if (in_bold_section) {
                    _INFO(COLOR_BOLD_RED);
                } else if (in_thinking_section) {
                    _INFO(COLOR_YELLOW);
                } else {
                    _INFO(COLOR_RESET);
                }
                current_pos = marker + 2;  // Move past the "**"
            }
            // print any remaining text after the last marker
            if (token != tokenizer->eos_id) {
                _INFO("%s", current_pos);
                cur_answer += current_pos;
            }
        }

        fflush(stdout);
    }
    // free(prompt_tokens);
    return 0x0;
}

// Although each batch has B samples, but preLogits contains only dB samples(enough tokens since T is large)
int GeneratOnPrompt::SampleOnBatch(hBATCH hBatch, float* fLoss, int B, int T, SampLoader* hLoader, int flag) {
    try {
        Fish* dolphin = fish_1;
        assert(dolphin != nullptr);
        int nVocab      = dolphin->nClass();  // fish_1 == nullptr ? wiki0->n_vocab : fish_1->nClass();
        Head4Token* cls = ((Fish*)fish_0)->GetNeuron<Head4Token>("Head4Token", 0);
        assert(cls != nullptr);
        int nSampDB = cls->preLogits->ne[0];
        assert(T == hBatch->hostToken->ne[0] && T == cls->preLogits->ne[1]);
        assert(hBatch->hostToken->type == typNUMBER::I32);
        int* curToken        = (int*)(hBatch->hostToken->data) + (B - nSampDB) * T;
        float* curLoss       = fLoss + (B - nSampDB) * T;
        int generated_tokens = 0, token, target, predict, nMostLine = std::min(T, 32), nMostToken = nSampDB * T, i_sec, j;

        string fpath = "./log/SampleBatch_.csv", line_0, sP, sT;  //
        FILE* fp     = fopen(fpath.c_str(), "wt");
        if (fp == NULL) {
            _INFO("%s: warning: empty or not existing training data file '%s'\n", __func__, fpath.c_str());
            return false;
        }
        //    iiLoss.Add(mean_loss);    iiPPL.Add(ppl);
        fprintf(fp, "iter=generate-predict sample loss=%g ppl=%g\n", hLoader->iiLoss.Last(), hLoader->iiPPL.Last());
        hBITARR _dev = (hBITARR)(cls->preLogits->data), _host = (hBITARR)(cls->preLogits->host_data);
        std::vector<TOKEN_ID> toks_0, toks_1;
        std::vector<int> tic0 = hBatch->arrTic0, tic1 = hBatch->arrTic1;
        if (tic0.empty()) {
            j = nMostLine;
            while (j <= nMostToken) {
                tic1.push_back(j), tic0.push_back(j - nMostLine);
                j += nMostLine;
            }
        }

        for (int i_sec = 0; i_sec < tic1.size(); i_sec++) {
            // for (i = 0; i < nMostToken; i += nMostLine) {  //  preLogits contains only dB samples!!!
            toks_0.clear();
            for (j = tic0[i_sec]; j < tic1[i_sec]; j++) {
                token = curToken[j];
                toks_0.push_back(token);
                // toks_1.push_back(token);
            }
            D2H(_dev + sizeof(floatX) * nVocab * (j - 1), _host, sizeof(floatX) * nVocab);
            target  = j == nMostToken ? -1 : curToken[j];
            predict = Sample(-1);  // from cls->preLogits->host_data
            if (target == predict) {
                // assert(curLoss[j - 1] < 0.3);
            }

            line_0 = dolphin->hDict->T2STR(toks_0, 0x0);
            sP     = dolphin->hDict->T2STR(predict, 0x0);
            sT     = j == nMostToken ? "" : dolphin->hDict->T2STR(target, 0x0);
            fprintf(fp, "\n------ %d=\"%s\"(%d)-\"%s\"(%d) loss=%g\n\t\"%s\" => %s", i_sec, sP.c_str(), predict, sT.c_str(), target, curLoss[j - 1],
                    line_0.c_str(), sP.c_str());
            // fprintf(fp, "\n");
            // generated_tokens++;
        }

        fclose(fp);
        _INFO(">>>>>> Save SampleBatch_.csv @\"%s\"\n", fpath.c_str());
        return true;
    } catch (...) {
        return false;
    }

    return 0x0;
}