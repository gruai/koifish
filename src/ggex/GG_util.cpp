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
#ifdef __USE_GGML__
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#else
#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#include <cstring>
#include <iostream>

#include "../CLI_params.hpp"
#include "../Manifold/Serialize.hpp"
#include "../ggex/GG_util.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "json.hpp"

int g_dump_level = 1;

#define ARG2STR(format, len)                  \
    {                                         \
        va_list args;                         \
        va_start(args, format);               \
        vsnprintf(buffer, len, format, args); \
        va_end(args);                         \
        assert(strlen(buffer) <= len);        \
    }

int gTN(hGTensor cur, const char *format, ...) {
    int iRet = 0;
    if (strlen(cur->name) == 0) {
        ARG2STR(format, GTensor::MAX_NAME);
        snprintf(cur->name, sizeof(cur->name), "%s", buffer);
        iRet += 1;
    }
    return iRet;
}

int gTN0(hGTensor cur, const char *format, ...) {
    ARG2STR(format, GTensor::MAX_NAME);
    snprintf(cur->name, sizeof(cur->name), "%s", buffer);
    return 0x0;
}
#ifdef __USE_GGML__
int gTN(struct ggml_tensor *cur, const char *format, ...) {
    int iRet = 0;
    if (strlen(cur->name) == 0) {
        ARG2STR(format, GTensor::MAX_NAME);
        // va_list args;
        // va_start( args, format );
        // vsnprintf( buffer,GTensor::MAX_NAME,format,args );
        // va_end(args);
        // assert(strlen(buffer)<=GTensor::MAX_NAME);
        ggml_format_name(cur, "%s", buffer);

        iRet += 1;
    }
    /*
        in ggml_compute_backward, some grad has no name!

        ggml_format_name(tensor->grad, "%s (grad)", tensor->name);
    */
#ifdef GG_V12
    CHILD_1218_GRAD  //  set name @BuildBackward
#else
    if (cur->grad && strlen(cur->grad->name) == 0) {
        assert(strlen(cur->name) < GTensor::MAX_NAME);
        if (strcmp(cur->name, "inp_embd_rows") == 0) {
            int debug = 0;
        }
        ggml_format_name(cur->grad, "%s\"", cur->name);
        iRet += 2;
    }
#endif
        return iRet;
}

int gTN0(struct ggml_tensor *cur, const char *format, ...) {
    int iRet = 0;
    ARG2STR(format, GTensor::MAX_NAME);
    // va_list args;
    // va_start( args, format );
    // vsnprintf( buffer,GTensor::MAX_NAME,format,args );
    // va_end(args);
    // assert(strlen(buffer)<=GTensor::MAX_NAME);
    ggml_format_name(cur, "%s", buffer);
    iRet += 1;

    /*
        in ggml_compute_backward, some grad has no name!

        ggml_format_name(tensor->grad, "%s (grad)", tensor->name);
    */
    /*if(cur->grad && strlen(cur->grad->name)==0){
        assert(strlen(cur->name)<GTensor::MAX_NAME);
        ggml_format_name(cur->grad,"%s\"",cur->name);
        iRet+=2;
    }*/
    return iRet;
}
#endif

int clamp(const int v, const int min, const int max) { return ((v < min) ? (min) : (v > max) ? (max) : v); }

float fclamp(const float v, const float min, const float max) { return ((v < min) ? (min) : (v > max) ? (max) : v); }

void assert_shape_1d(hGTensor tensor, int64_t ne0) {}
void assert_shape_2d(hGTensor tensor, int64_t ne0, int64_t ne1) {}
void assert_shape_3d(hGTensor tensor, int64_t ne0, int64_t ne1, int64_t ne2) {}
void assert_shape_4d(hGTensor tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {}
#ifdef __USE_GGML__
void assert_shape_1d(struct ggml_tensor *tensor, int64_t ne0) {
    assert(tensor->ne[0] == ne0);
    assert(tensor->ne[1] == 1);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);
}

void assert_shape_2d(struct ggml_tensor *tensor, int64_t ne0, int64_t ne1) {
    assert(tensor->ne[0] == ne0);
    assert(tensor->ne[1] == ne1);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);
}

void assert_shape_3d(struct ggml_tensor *tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    assert(tensor->ne[0] == ne0);
    assert(tensor->ne[1] == ne1);
    assert(tensor->ne[2] == ne2);
    assert(tensor->ne[3] == 1);
}

void assert_shape_4d(struct ggml_tensor *tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    assert(tensor->ne[0] == ne0);
    assert(tensor->ne[1] == ne1);
    assert(tensor->ne[2] == ne2);
    assert(tensor->ne[3] == ne3);
}
#endif

#ifdef __USE_GGML__
struct ggml_tensor *tRAND(struct ggml_tensor *tensor, struct random_normal_distribution *rnd) {
    float scale = 1.0f;  // xavier
    switch (ggml_n_dims(tensor)) {
        case 1:
            scale /= sqrtf((float)tensor->ne[0]);
            for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                float *dst = (float *)((char *)tensor->data + i0 * tensor->nb[0]);
                *dst       = scale * frand_normal(rnd);
            }
            break;
        case 2:
            scale /= sqrtf((float)tensor->ne[0] + tensor->ne[1]);
            for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                    float *dst = (float *)((char *)tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1]);
                    *dst       = scale * frand_normal(rnd);
                }
            }
            break;
        case 3:
            scale /= sqrtf((float)tensor->ne[0] + tensor->ne[1]);
            for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                    for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                        float *dst = (float *)((char *)tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2]);
                        *dst       = scale * frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            scale /= sqrtf((float)tensor->ne[0] + tensor->ne[1]);
            for (int i3 = 0; i3 < tensor->ne[3]; i3++) {
                for (int i2 = 0; i2 < tensor->ne[2]; i2++) {
                    for (int i1 = 0; i1 < tensor->ne[1]; i1++) {
                        for (int i0 = 0; i0 < tensor->ne[0]; i0++) {
                            float *dst = (float *)((char *)tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3]);
                            *dst       = scale * frand_normal(rnd);
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
#endif

hGTensor tRAND(hGTensor tensor, struct random_normal_distribution *rnd) { return nullptr; }

JSON jKEY(const JSON &jConfig, const std::vector<std::string> &keys, int flag) {
    assert(keys.size() > 0);
    JSON cur = (JSON)jConfig;
    std::string key;
    for (int i = 0; i < keys.size(); i++) {
        key = keys[i];
        if (cur.find(key) == cur.end()) {
            // _ERROR("\t NO %s @\"%s\";",__func__,key.c_str());
            cur.clear();
            return cur;
        }
        cur = cur[key];
    }
    if (cur.is_null() || cur.empty()) {
        _ERROR("%s failed !!!", __func__);
    }

    return cur;
}
// bool jK2S(const JSON& jConfig,const std::vector<std::string>&keys,(const char*) &str,int flag=0x0);
bool jK2S(const JSON &jConfig, const std::vector<std::string> &keys, char **str, int flag = 0x0) {
    JSON cur = jKEY(jConfig, keys);
    if (cur.is_null() || cur.empty()) {
        assert(0);
        return false;
    }
    string val = cur.get<std::string>();
    // *str = val.c_str();
    strcpy(*str, val.c_str());
    return true;
}

void UpdateJConfig(JSON &jConfig, const std::string &jPath) {
    try {
        std::ifstream jfile(jPath);
        if (jfile.fail()) {
            _INFO("\r\n%s  Failed to open %s", __func__, jPath.c_str());
        }
        jfile >> jConfig;
        std::string s = jConfig.dump();
    } catch (JSON::parse_error &e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, jPath.c_str(), e.what());
    } catch (...) {
        _INFO("\r\n%s  Unknown exception @%s!!!", __func__, jPath.c_str());
    }
}

std::string EXE_name(int flag) {
#if defined(PLATFORM_POSIX) || defined(__linux__)  // check defines for your setup
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

void train_print_usage(int argc, char **argv, const struct CLI_params *params) {}

bool CLI_params::operator!=(const CLI_params &other) const { return memcmp(this, &other, sizeof(other)); }

DEUG_SWITCH DEBUG;
void DEUG_SWITCH::Dump(int typ) {
    _INFO("[DEBUG]: gemm=%d classifier=%d", T_GEMM, T_classifier_ver);
    _INFO("\n");
}
void CLI_params::Dump(int flag) {
    if (flag == 0x100) {  // only dump jConfig
        std::ofstream file("_koifish_tmp_config_.json");
        if (file.is_open()) {
            file << jConfig.dump(4);
            file.close();
        }
        return;
    }

    _INFO("%s::CLI_params: \n", exec_name.c_str());
    // _INFO(" n_vocab: %u", n_vocab);
    _INFO(" n_ctx=%u", n_ctx());
    _INFO(" embd=%u", nEmbed());
    _INFO(" n_ff=%u", n_ff());
    _INFO(" n_head=%u", n_head());
    _INFO(" n_head_kv=%u", n_head_kv());
    _INFO(" n_layer=%d(%d)", nLayer(), n_layer_train);
    // _INFO(" n_rot=%u\n", n_rot());
    _INFO(" f_norm_rms_eps=%g\n", model.norm_rms_eps);
    _INFO(" ROPE: type=%d freq_base=%g freq_scale=%g n_rot=%u\n", model.rope_type, model.rope_freq_base, model.rope_freq_scale, n_rot());
    // _INFO(" rope_freq_scale=%g\n", model.rope_freq_scale);
    // _INFO(" lora_r=%d lora_alpha=%d ", lora_r,lora_alpha);
    _INFO(" SepQKV: type=%d  \n", model.isSeparateQKV);
    // _INFO(" NABLA = %s\n", nabla==0? "" : nabla==3 ? "Embed+AutoEncoder" : (nabla==2 ? "" : "qkv") );
    // _INFO(" SIGMA = %s\n", sigma.c_str());
}

bool LoadJsonFile(const string &jPath, JSON &jObj, int flag) {
    try {
        std::ifstream jfile(jPath);
        std::string info;
        if (jfile.fail()) {
            _INFO("\r\n[%s] Failed to open \"%s\" !!!\n", __func__, jPath.c_str());
            return false;
        }
        jfile >> jObj;
        return true;
    } catch (...) {
        return false;
    }
}
/*
    "architectures": [
        "MistralForCausalLM"
    ],
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mistral",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "sliding_window": null,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.36.0",
    "use_cache": true,
    "vocab_size": 32000
*/
string MODEL_CARD::sWeight = ".weight", MODEL_CARD::sBias = ".bias";  //".w"
string MODEL_CARD::sNorm    = "_norm";                                //".norm"
string MODEL_CARD::sLayer   = "blk.";                                 //".norm"
string MODEL_CARD::sAttnOut = ".wo";                                  //  "_output";    //  "_cat"

MODEL_CARD::MODEL_CARD() {
#if defined(ENABLE_FP16)
    tpWeight = typNUMBER::F16, tpActivation = typNUMBER::F16, tpGradient = typNUMBER::F16;
#elif defined(ENABLE_BF16)
    tpWeight = typNUMBER::BF16, tpActivation = typNUMBER::BF16, tpGradient = typNUMBER::BF16;
#elif defined(ENABLE_FP8)
    tpWeight = typNUMBER::F8E5M2, tpActivation = typNUMBER::BF16, tpGradient = typNUMBER::BF16;
#elif defined(ENABLE_FP32)
    tpWeight = typNUMBER::F32, tpActivation = typNUMBER::F32, tpGradient = typNUMBER::F32;
#else
    assert(0);
#endif
}

MODEL_ARCH CLI_params::ModelArch() {
    MODEL_ARCH arch = MODEL_ARCH::_X_;
    string info     = "";
    if (model.empty()) {
        string s = jKV(jConfig, {"model", "arch"}, string(""), false);
        assert(!s.empty());
        info = s;
    } else {
        info = model.model_type;
    }
    if (model_title.empty())
        model_title = info;

    std::transform(info.begin(), info.end(), info.begin(), ::toupper);
    arch = info == "MOE"        ? NLP_MOE
           : info == "MAMBA"    ? MODEL_ARCH::NLP_MAMBA
           : info == "GUPPY"    ? MODEL_ARCH::NLP_GUPPY
           : info == "DEEPSEEK" ? MODEL_ARCH::NLP_DEEPSEEK
           : info == "QWEN2"    ? MODEL_ARCH::NLP_QWEN2
                                :
                             // info=="QWEN" ? MODEL_ARCH::NLP_QWEN :
               info == "GPT2"   ? MODEL_ARCH::NLP_GPT2
           : info == "GPT2CHAR" ? MODEL_ARCH::NLP_GPT2_char
           : info == "LAMA"     ? MODEL_ARCH::NLP_LLAMA
           : info == "MISTRAL"  ? MODEL_ARCH::NLP_MISTRAL
                                : MODEL_ARCH::NLP_LLAMA;

    return arch;
}

bool CLI_params::isValid(int flag) {
    uint32_t head_dim = n_embd_head();
    return true;
}

bool CLI_params::JModel2Params(int flag) {
    string key = "";
    try {
        jModel = jKEY(jConfig, {"model"});
        if (jModel.empty()) {
            return false;
        }
        nLayerX = 1;  // at least 1 layer
        nLayerX = jKV(jConfig, {"model", "parameter", "Layer"}, nLayerX);
        assert(nLayerX < 160 && nLayerX > 0);
        // nearly same ???
        // if(nLayerX>0)
        //     common.residual_scale = 1.0f / sqrtf(2.0f * nLayerX);
        assert(model.layerps.size() == 0);
        auto jTrans = jKEY(jConfig, {"model", "parameter", "transformer"});
        if (!jTrans.empty()) {
            int nH = jKV(jTrans, {"Head"}, -1), nF = jKV(jTrans, {"Ffn"}, -1), nE = -1, nC = jKV(jTrans, {"Ctx"}, -1);
            auto item = jKEY(jTrans, {"Embed"});
            std::vector<int> embeds;
            if (item.is_string()) {
                string sE = item.get<string>();
                G_S2TTT_(sE, embeds);
            } else {
                assert(item.is_number_integer());
                int n = item.get<int>();
                embeds.push_back(n);
            }
            assert(embeds.size() > 0 && embeds[0] > 0);
            token_embeds.push_back(embeds[0]);
            qkv_embeds = embeds;
            if (nF < 0)
                nF = nEmbed() * 4;

            if (nH > 0 && nF > 0) {
                for (int i = 0; i < nLayerX; i++) model.layerps.push_back(MODEL_CARD::LAY_PARAM(nH, nH, nF));
            } else {
                assert(0);
            }

            if (nC > 0) {
                SetNCTX(nC);
                // common.n_ctx = nC;
            } else {
                assert(0);
            }
        }
        // Mem 8484=>4772=>4838
        if (DEBUG.cmd_p1 == 1) {
            // scheduling.strategy = MEM_STRATEGY::MEM_SWAP_GUOKE;  // DEBUG.T_GEMM = -1;
            // common.method = "adamw";
            common.muon.isTransDown = false;
        } else {
            // common.method = "muon";
            // scheduling.strategy = MEM_STRATEGY::MEM_SWAP;
            // scheduling.strategy = MEM_STRATEGY::MEM_SWAP_GUOKE;
        }
        fuyou.Init(this, jConfig);
        // fuyou.InitSection(nLayer(), jKV(jConfig, {"model","fuyou", "branch"}, fuyou.nLayerInBranch));

        if (scheduling.strategy == MEM_STRATEGY::MEM_SWAP_GUOKE) {
            common.remater_ffn = 0;  // more memory, more time
        } else {                     //  remater_ffn=1: reduce memory & little slower
                                     // if(scheduling.nLayerInBranch>0)
                                     //     common.remater_ffn = 0;
                                     // else
            common.remater_ffn = 1;
            // common.remater_qkv = 1;
        }

        return true;
    } catch (JSON::parse_error &e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, key.c_str(), e.what());
        return false;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception @%s!!!", __func__, key.c_str());
        return false;
    }
}

bool CLI_params::isShareLayerOut() const {
    if (scheduling.strategy == MEM_STRATEGY::PRE_ALLOC_GPU || scheduling.strategy == MEM_STRATEGY::PRE_ALLOC_HOST_MAP)
        return false;
    return true;
}

// Deprecated, only for debug
bool SKDU_params::isUpdateParamV0() const {
    if (strategy == MEM_SWAP || strategy == MEM_SWAP_GUOKE)
        return false;

    return false;
}

bool SKDU_params::canSave(int iter, int flag) const {
    // if (LIB_iter4save > 0) {
    //     return iter >= LIB_iter4save;
    // }
    return true;
}

bool Fuyou_params::Init(CLI_params *hConfig, const JSON &jConfig, int flag) {
    filter_reload  = {"ffn", "attn.wq", "attn.wk", "attn.wv", "attn.wo"};
    nLayerInBranch = jKV(jConfig, {"model", "fuyou", "branch"}, nLayerInBranch);
    if (nLayerInBranch <= 0)
        return false;
    T_crossover = jKV(jConfig, {"model", "fuyou", "crossover"}, T_crossover);
    T_mutation  = jKV(jConfig, {"model", "fuyou", "mutation"}, T_mutation);
    social      = jKV(jConfig, {"model", "fuyou", "social"}, social);
    int nLayer  = hConfig->nLayer();
    assert(nLayer % nLayerInBranch == 0);
    string a  = "";  //"pso";
    a         = jKV(jConfig, {"model", "fuyou", "method"}, a);
    algorithm = NO_EVOL;
    for (auto an : Algo2Name) {
        if (an.second == a) {
            algorithm = an.first;
        }
    }
    assert(algorithm == Fuyou_params::NO_EVOL || Algo2Name[algorithm] == a);

    LIB_0           = 0;
    LIB_1           = 0;  // nLayerInBranch;
    nBranch         = nLayer / nLayerInBranch;
    LIB_iter_switch = jKV(jConfig, {"model", "fuyou", "switch"}, LIB_iter_switch);
    assert(LIB_iter_switch >= 1);

    if (ensemble == Fuyou_params::MULTI_SCALE) {
        LIB_iter_switch = 1;
    }
    if (nBranch > 1) {
        paramIsGuoke = true;  // DEBUG.x_str.empty();  // Reduce memory greaty, for example @@@0801_774M_section=9.info
    } else {
        paramIsGuoke = false;
    }
    return true;
}

bool Fuyou_params::isFirst(int layer, int flag) {
    if (layer <= nLayerInBranch)
        return true;
    return false;
}

// Deprecated
bool Fuyou_params::InitSection(int nLayer, int nLS, int nSwitch, int flag) {
    if (nLS <= 0)
        return false;
    assert(nLayer % nLS == 0);
    nLayerInBranch = nLS;
    LIB_0          = 0;
    LIB_1          = 0;  // nLayerInBranch;
    nBranch        = nLayer / nLayerInBranch;
    assert(nSwitch >= 1);
    LIB_iter_switch = nSwitch;  // 100,10

    return true;
}

int Fuyou_params::nWarmup(int flag) {
    // return LIB_iter_switch * nBranch;
    return LIB_iter_switch * (nBranch - 1);
}

uint32_t CLI_params::nThread() const {
    int nT0 = std::thread::hardware_concurrency(), nT1 = common.n_threads;
    return nT1;
}
uint32_t CLI_params::nEmbed(int flag) const {
    assert(token_embeds.size() > 0);
    if (flag == -1)
        return token_embeds[0];
    int no = token_embeds.size() - 1;
    return token_embeds[no];
}
void CLI_params::OnMostToken(size_t nMost, int flag) {
    double a          = nMost * 1.0 / n_ctx();  // nTokenInBatch();
    size_t nSamp      = (size_t)(floor)(a);
    int iter_1        = (int)floor(a / n_batch());
    float rSample     = common.rSubSample;
    common.nEpochIter = (int)ceil(nMost * rSample / nTokenInBatch());
    common.nMostIter  = (int)floor(nMost * common.n_epochs * rSample / nTokenInBatch());
}

void SKDU_params::Dump(int typ) const {
    _INFO("[Scheduling] MEM_STRATEGY=%s UpdateParam=V%d\n", MEM_STRATEGY_desc[strategy].c_str(), isUpdateParamV0() ? 0 : 1);
}

void Fuyou_params::Dump(int typ) const {
    string sType[] = {"AGGREGATION", "BEST", "RANDOM_1", "MULTI_SCALE"};  //@MODEL_ENSEMBLE
    string sAlgo   = Algo2Name.at(algorithm);
    _INFO("Explorer=[%s] ensembler=\"%s\" nSwitch=%d %s\n", sAlgo.c_str(), sType[ensemble].c_str(), LIB_iter_switch,
          paramIsGuoke ? "\"Only cur fuyou's params in GPU-memor!\"" : "\"All params in GPU-memory\"");
}

// LORA_ADAPT_W HIERARCH_LoRA::tpLORA = HIERARCH_LoRA::W_AB;      //  W0, AB, W_AB
void CLI_params::OnArch() {
    _INFO("[ARCH] sizeof(token)=%ld,sizeof(floatX)=%ld sizeof(Grad)=%d(%d)\n", sizeof(TOKEN_ID), sizeof(floatX), sizeof(floatGrad), sizeof(floatMV));
    int nH        = -1;
    bool isJModel = !jModel.empty();

    if (common.seed == -1) {
        common.seed = time(NULL);
    }
    _INFO("seed=%u\n", common.seed);
    srand(common.seed);

    switch (ModelArch()) {
        case MODEL_ARCH::NLP_GUPPY:
            model.isNormalBias       = true;
            model.isSLPBias          = true;  // nealy same
            model.isPaddedCls        = true;  // ceil(n/128.0)*128
            model.isSeparateQKV      = false;
            model.preLogits_dB       = 8;
            model.isFFNShareParam    = true;
            model.isEmbedWeightTying = true;
            DEBUG.check_tensor_norm  = true;
            model.Rope_version       = 0;  // 2025.5.7 some bugs
            break;
        case MODEL_ARCH::NLP_GPT2:
        case MODEL_ARCH::NLP_GPT2_char: {
            // DEBUG.cmd_p1 = 1;
            tpLORA = LORA_ADAPT_W::W0;  // refW_AB      W_AB
            if (tpLORA != LORA_ADAPT_W::W0) {
                fuyou.paramIsGuoke = true;  //
            }

            DEBUG.T_fuyou = 1;
            // fuyou.algorithm        = Fuyou_params::PARTICLE_SWARM;  //  PARTICLE_SWARM  GENE_MIX
            // fuyou.LIB_iter_switch  = 100;                           // 100,10
            // fuyou.ensemble = Fuyou_params::MULTI_SCALE;

            DEBUG.T_classifier_ver = 1;
            n_embd_head_v = 64, n_embd_head_k = 64, n_ctx_train = 1024;
            // _embd = 128; dict_latent_dim = 128;        n_embd_head_v=n_embd_head_k=2; //only for debug
            if (model.layerps.size() == 0 && !isJModel) {  //  deprecated
                int n_ff0 = jKV(jConfig, {"model_v0", "ffn", "length"}, 3072, false), nLay = nLayer();
                for (int i = 0; i < nLayer(); i++) {
                    MODEL_CARD::LAY_PARAM lay(nH, nH, n_ff0);
                    model.layerps.push_back(lay);
                }
            }

            model.tpPreLogits = typNUMBER::BF16;
            //  need new GEMM! cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
            model.isNormalBias  = true;
            model.isSLPBias     = true;  // nealy same
            model.isPaddedCls   = true;  //  ceil(n/128.0)*128
            model.isSeparateQKV = true;
            /* Very strange
                1. ver_1 would fail, why?
                2. ver_2 also fail
                2. ver_3 would stall after 700 steps
            */
            model.Rope_version       = 0;
            model.isEmbedWeightTying = true;
            model.preLogits_dB       = 8;

            int group = Get({"model_v0", "target_group"}, 1);
            assert(group == 1);
        }

        break;
        case NLP_QWEN2:
            // scheduling.strategy = MEM_STRATEGY::MEM_SWAP_GUOKE;  // 5.89 tps
            // scheduling.strategy     = MEM_STRATEGY::PRE_ALLOC_HOST_MAP;      //  6.53 tps
            model.isSeparateQKV = true, model.isBqkv = true;
            model.sNorm = ".norm", model.sLayer = "layers.";
            model.isEmbedWeightTying = true;  //  why false would cause nan
            // model.sAttnOut=".wqkv";
            break;
        case NLP_DEEPSEEK:
            model.sNorm = ".norm", model.sLayer = "layers.";
            break;
        case NLP_MISTRAL:
            model.sNorm  = ".norm";
            model.sLayer = "layers.";
            break;

        default:
            _INFO("[ARCH]=%s\n", "");
            break;
    }
}

TRAIN_CARD get_default_train_params_common() {
    TRAIN_CARD params;
    // params.print_usage = false;
    params.seed                    = -1;
    params.n_ctx                   = 128;
    params.n_threads               = 6;
    params.n_batch                 = 8;
    params.n_gradient_accumulation = 1;
    params.n_epochs                = -1;
    params.n_gpu_layers            = 0;

    params.custom_n_ctx = false;

    params.use_flash         = false;
    params.use_checkpointing = true;

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

    params.warmup     = 600;
    params.lr_restart = 0;

    // params.adam.n_iter         = -1;
    params.adam.alpha          = 1e-3f;
    params.adam.min_alpha      = 0;
    params.adam.decay          = 1e-1f;
    params.adam.decay_min_ndim = 2;
    params.adam.beta1          = 0.9f;
    params.adam.beta2          = 0.95f;  // 0.999f;
    params.adam.gclip          = 1.0f;
    params.adam.eps_loss       = 1e-5f;

    return params;
}

std::string CLI_params::GetDataPath(const string type, int flag) {
    string fp = "";
    try {
        if (type.empty())  // some tiny file for fast debug
            fp = jKV(jConfig, {"tiny_data", "source"}, fp);
        else {
            fp = jKV(jConfig, {"datasets", type, "source"}, fp);
        }

    } catch (...) {
    }
    if (!std::filesystem::exists(fp)) {
        _INFO("%s: warning: empty or not existing %s data file '%s'\n", __func__, type.c_str(), fp.c_str());
        fp = "";
    }
    return fp;
}

JSON CLI_params::ToJSON(int flag) {
    JSON json;
    json["config"] = jConfig;
    return json;
}

bool TRAIN_CARD::Init(CLI_params *hConfig, const JSON &jConfig, int flag) {
    method = jKV(jConfig, {"train", "optimizatioin", "method"}, method);

    n_batch   = jKV(jConfig, {"train", "batch"}, n_batch);
    n_epochs  = jKV(jConfig, {"train", "epoch"}, n_epochs);
    nMostIter = jKV(jConfig, {"train", "adam-iter"}, nMostIter);
    // why large "learning-rate" would fail, so strange!
    adam.alpha              = jKV(jConfig, {"train", "learning-rate"}, adam.alpha);
    adam.decay              = jKV(jConfig, {"train", "decay"}, adam.decay);
    n_gradient_accumulation = jKV(jConfig, {"train", "optimizatioin", "grad_accumulation"}, n_gradient_accumulation);

    dump_every = jKV(jConfig, {"train", "dump-every"}, dump_every);
    // eval_every = jKV(jConfig,{"train","eval-every"},eval_every );
    gpt_every = jKV(jConfig, {"train", "gpt-every"}, gpt_every);
    // eval_every = eval_every<=0 ? 100000000 : eval_every;
    // if( eval_every>0 ){
    //     _INFO("\r\n%s  eval@every %d steps.",__func__,eval_every );
    // }
    rSubSample = jKV(jConfig, {"train", "sample"}, rSubSample);
    if (rSubSample < 0)
        rSubSample = 1;
    if (rSubSample < 1) {
        lr_restart = 1;
    }

    seed = jKV(jConfig, {"seed"}, seed);

    custom_n_ctx = true;
    n_threads    = jKV(jConfig, {"threads"}, n_threads);
    n_gpu_layers = jKV(jConfig, {"n-gpu-layers"}, n_gpu_layers);
    return true;
}

/*
    Some trick
    1 Large batch size would decrease osillation
    2 Double batch+half layers =>  More accuracy in same training time
*/
bool CLI_params::InitJConfig(int flag) {
    try {
        char *env_str = getenv("PATH");
        env_str       = getenv("LD_LIBRARY_PATH");

        common = get_default_train_params_common();

        std::string s = jConfig.dump(), s0;
        common.Init(this, jConfig);

        lars_ratio = jKV(jConfig, {"train", "optimizatioin", "lars_ratio"}, lars_ratio);
        ZMUV_ratio = jKV(jConfig, {"train", "optimizatioin", "ZMUV_ratio"}, ZMUV_ratio);

        // serial_path = jKV(jConfig,{"data","serialize_path"},s0 );
        string dict_type = jKV(jConfig, {"dict", "type"}, s0);
        tpBatchSample    = jKV(jConfig, {"train", "batch_sample"}, tpBatchSample);
        rSplit           = jKV(jConfig, {"data", "eval_split"}, rSplit);
        // string a = tpBatchSample=="stacking" ? tpBatchSample : "";

        std::vector<string> all_base;
        all_base = jKV_arr(jConfig, {"wiki", "path"}, all_base, false);
        for (auto path : all_base) {
            if (path.empty() || path[0] == '#')
                continue;
            if (!std::filesystem::exists(path)) {
                _WARN("====== Failed to load model @\"%s\" !!! ======\n", path.c_str());
                continue;
            }

            if (model_title.empty())  // the first path is backbone
                model_title = remove_extension(base_name(path));
            fn_model_base.push_back(path);
        }
        // if(model_title.empty()){
        //     model_title = "Unknown";
        // }
        datatypes.arrTernary = jKV_arr(jConfig, {"model", "datatype", "ternary"}, datatypes.arrTernary, false);
        datatypes.arrTile    = jKV_arr(jConfig, {"model", "datatype", "tile"}, datatypes.arrTile, false);

        JModel2Params(0x0);

        model.InitHF(this, jConfig);

        n_swarm = jKV(jConfig, {"train", "swarm"}, 1);

        // common.seed = jKV(jConfig, {"seed"}, common.seed);
        wiki_actor  = jKV(jConfig, {"wiki", "actor"}, wiki_actor);
        wiki_logits = jKV(jConfig, {"wiki", "logits"}, wiki_logits);
        tpWiki      = jKV(jConfig, {"wiki", "induct"}, tpWiki);
        nabla       = jKV(jConfig, {"model_v0", "nabla"}, nabla);
        // sigma = jKV(jConfig,{"model_v0","sigma"},sigma );

        if (jModel.empty()) {
            nFFX = jKV(jConfig, {"model_v0", "ffn", "length"}, nFFX);
            assert(nFFX < 160 * 1024 && nFFX > 0);
            nLayerX = jKV(jConfig, {"model_v0", "layer"}, nLayerX);
            assert(nLayerX < 160 && nLayerX > 0);
            common.n_ctx = jKV(jConfig, {"model_v0", "ctx"}, common.n_ctx);
        } else {
        }

        checkpoint.model_out  = jKV(jConfig, {"model-out"}, checkpoint.model_out);
        checkpoint.in         = jKV(jConfig, {"checkpoint", "in"}, checkpoint.in);
        checkpoint.out        = jKV(jConfig, {"checkpoint", "out"}, checkpoint.out);
        checkpoint.save_every = jKV(jConfig, {"checkpoint", "save-every"}, checkpoint.save_every);

        // f_norm_rms_eps = jKV(jConfig,{"norm-rms-eps"},f_norm_rms_eps );
        // rope_freq_base = jKV(jConfig,{"rope-freq-base"},rope_freq_base );
        // rope_freq_scale = jKV(jConfig,{"rope-freq-scale"},rope_freq_scale );

        prompt = jKV(jConfig, {"gpt", "prompt"}, prompt);

        dict_vae_dims = jKV(jConfig, {"dict", "vae", "dims"}, dict_vae_dims);
        // DEBUG.dict_latent_dim = jKV(jConfig,{"dict","latent_dim"},DEBUG.dict_latent_dim );
        dict_dialect = jKV(jConfig, {"dict", "dialect"}, dict_dialect);
        dict_logits  = jKV(jConfig, {"dict", "logits"}, dict_logits);

        vae = dict_vae_dims;  // hack

        test = jKV(jConfig, {"test"}, test);

        /*
            on some ealy testing on finetune/distillation, it seems that less layers would get nealy same accuracy
        */
        // tune   = jKV(jConfig, {"lora", "tune"}, tune);    //"lora_tune"
        // lora_r = jKV(jConfig, {"lora", "rank"}, lora_r);  //{"lora-r"}

        DEBUG.x1         = jKV(jConfig, {"debug", "x"}, DEBUG.x1);
        DEBUG.x_str      = jKV(jConfig, {"debug", "x_str"}, DEBUG.x_str);
        DEBUG.N_mostiter = jKV(jConfig, {"debug", "most_iter"}, DEBUG.N_mostiter);

        SUM::nMostMemItem = jKV(jConfig, {"dump", "most_mem_item"}, SUM::nMostMemItem);

        dumpSwitch.train_time = jKV(jConfig, {"dump", "train_time"}, dumpSwitch.train_time);
        // train = jKV(jConfig,{"train"},train );
        return true;
    } catch (JSON::parse_error &e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, jsPath.c_str(), e.what());
        return false;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception @%s!!!", __func__, jsPath.c_str());
        return false;
    }
}

bool CLI_params::parse(int argc, char **argv) {
    std::string arg_prefix = "--", key, value;
    exec_name              = EXE_name();

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
        //     std::replace(arg.begin(), arg.end(), '_', '-');
        // }
        string sExt = arg.length() > 5 ? FILE_EXT(arg) : "";
        if (!sExt.empty()) {
            jsPath = arg;
            if (sExt == "json") {
                std::ifstream jfile(jsPath);
                if (jfile.fail()) {
                    _INFO("\r\n[%s] Failed to open \"%s\" !!!\n", __func__, jsPath.c_str());
                    return false;
                }
                jfile >> jConfig;
            } else if (sExt == "fish") {
                SAFETENSOR_Load_jconfig(arg, jConfig);
            } else if (sExt == "ck") {
                SAFETENSOR_Load_jconfig(arg, jConfig);
            }
            if (jConfig.empty())
                return false;
        } else if (arg == "--version") {
        } else if (arg == "p0") {
        } else if (arg == "p1") {
            DEBUG.cmd_p1 = 1;
            _INFO("******************* DEBUG.cmd_p1=%d ******************************\n", DEBUG.cmd_p1);
        } else if (arg == "p2") {
            DEBUG.cmd_p2 = 1;
        } else if (arg == "--hellaswag") {
            eval_metric = "hellaswag";
            assert(i + 1 < argc);
            JSON jEval;
            jEval["type"] = "hellaswag", jEval["glob"] = argv[++i];
            jEval["step"] = 0.01;
            // jEval["glob"] = argv[i++];
            jConfig["datasets_new"]["eval"] = jEval;
        } else if (arg == "--step") {
            eval_metric = "hellaswag";
            assert(i + 1 < argc);
            sscanf(argv[++i], "%f", &step);
        } else {
            _ERROR("error: invalid parameter for argument: %s\n", arg.c_str());
            train_print_usage(argc, argv, this);
            exit(1);
        }
    }
    DEBUG.T_GEMM = -1;  //  so many version of gemm

    if (!InitJConfig())
        return false;

    // finish_processing_train_args(&common);
    return true;
}

int Gensor_loab(struct ggml_context *ctx0, hGensor w, int nHeavy, hGensor ga, hGensor gb, int flag) {
    printf("%s@%s <== %s x %s\n\t", __func__, w->name, ga->name, gb->name);
    auto shape = w->ne;
    int nIn = shape[0], nOut = shape[1], rank = nHeavy;  // min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);
    assert(nIn > 0 && nOut > 0 && ne00 == nIn * nOut);
    assert(nIn > nHeavy && nOut > nHeavy && nHeavy > 0);
    float *A = Gensor2float(ctx0, w, flag);
    auto svd = std::make_shared<LoSVD<float>>("Gensor_loab", A, nIn, nOut, rank, 1.0e-3);  // 1.0e-3
    assert(ga->type == typNUMBER::F32 && gb->type == typNUMBER::F32);
    if (!svd->Build()) {
        return -1;
    } else {
        if (tELEM(ga) != nIn * rank || tELEM(gb) != nOut * rank) {
            return -2;
        }
        svd->US((float *)((char *)ga->data));
        memcpy(gb->data, svd->V(), sizeof(float) * rank * nOut);
    }
    delete[] A;
    return 0x0;
}

int Gensor_SVD(struct ggml_context *ctx0, hGensor w, int nHeavy, hGensor U, hGensor D, hGensor V, int flag) {
    printf("%s@%s \t ......", __func__, w->name);

    auto shape = w->ne;
    int nIn = shape[0], nOut = shape[1], rank = nHeavy;  // min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);
    assert(nIn > 0 && nOut > 0 && ne00 == nIn * nOut);
    assert(nIn > nHeavy && nOut > nHeavy && nHeavy > 0);
    float *A = Gensor2float(ctx0, w, flag);
    GST_TIC(tic);
    auto svd = std::make_shared<LoSVD<float>>("Gensor_SVD", A, nIn, nOut, rank, 1.0e-3);  // 1.0e-3
    float t0 = GST_TOC(tic);
    if (!svd->Build()) {
        return -1;
    } else {
        // typNUMBER::F16 tensor would call ggml_vec_dot_f16 with GGML_SIMD acceleration
        /*if(compression==SVD_a)  {   //keep same graph
            float *approx = svd->Approx( );
            ggml_fp32_to_fp16_row(approx,(ggml_fp16_t*)w->data,nIn*nOut);
        }else*/
        {
            memcpy(U->data, svd->U(), sizeof(float) * nIn * rank);
            memcpy(V->data, svd->V(), sizeof(float) * rank * nOut);
            float *Sigma = svd->S(), *mD = (float *)(D->data);
            // memcpy(D->data,Sigma,sizeof(float)*rank);
            memset(mD, 0x0, sizeof(float) * rank * rank);
            for (int i = 0; i < rank; i++) mD[i * rank + i] = Sigma[i];
        }
    }
    svd.reset();
    delete[] A;

    return 0x0;
}

/*
    return probability(from softmax transformantion) of the given index

    1. Softmax often seems to be over confident in it's prediction based on the way the softmax was trained to map the raw scores to probabilities.
*/
double P_softmax(int idx, float *logits, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        max_val = logits[i] > max_val ? logits[i] : max_val;
    }
    float partition = 0.0f;
    for (int i = 0; i < size; i++) {
        partition += expf(logits[i] - max_val);
    }
    //
    return expf(logits[idx] - max_val) / partition;
}

float SOFT_MAX(const int n, float *y, const float *x) {
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
    sum = 1.0 / sum;
    // ggml_vec_scale_f32(n, y, sum);
    for (i = 0; i < n; ++i) {
        y[i] *= sum;
    }

#ifndef NDEBUG
    for (i = 0; i < n; ++i) {
        assert(!isnan(y[i]));
        assert(!isinf(y[i]));
    }
#endif
    return x1;
}

// Py=Py-Px
float SOFT_MAX_minus(const int n, float *y, const float *x) {
    assert(0);
    float x1 = -INFINITY, a;
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
    sum = 1.0 / sum;
    // ggml_vec_scale_f32(n, y, sum);
    for (i = 0; i < n; ++i) {
        y[i] *= sum;
    }

#ifndef NDEBUG
    for (i = 0; i < n; ++i) {
        assert(!isnan(y[i]));
        assert(!isinf(y[i]));
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
float LOSS_cross_entropy_1(int n, const float *preP, int target, int &cand, int flag) {
    assert(target >= 0 && target < n);
    float sum = 0, loss = 0, pMin, pMax, a;
    int j, next_token   = -1;
    cand = -1;
    for (pMin = FLT_MAX, pMax = -FLT_MAX, j = 0; j < n; j++) {
        a = preP[j];
        if (a > pMax) {
            pMax = a;
            cand = j;
        }
        pMin = min(a, pMin);  // pMax = max(a,pMax);
    }

    /*for (sum = 0, j = 0; j < n; j++)        { //  standard SOFTMAX
        preP[j] = exp(preP[j]-pMax);
        sum += preP[j];
    }
    assert(sum > 0 && sum < FLT_MAX);
    a = preP[target]/sum;   loss = -log(a); //  0.0430280194*/
    for (sum = 0, a = preP[target], j = 0; j < n; j++) {  // faster & safer
        sum += exp(preP[j] - a);
    }
    assert(sum > 0 && sum < FLT_MAX);
    loss = log(sum);
    return loss;
}

struct ggml_tensor *ggml_cross_entropy_loss_1(struct ggml_context *ctx, struct ggml_tensor *a, struct ggml_tensor *b) {
#ifndef GG_V12
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct ggml_tensor *result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_1;
    result->grad   = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
#endif
    return nullptr;
}

void _T_repr_(hGensor t, const char *tab, char *buf, int typ) {
    if (t == nullptr)
        return;
    bool isInput = t->flags & GTensor::F_INPUT;
    string A     = NameOf(t->type);  //==typNUMBER::F16 ? "d16":"d";
    // if(t->grad!=nullptr){
    //     if(t->type==typNUMBER::F16)
    //         A = "P16";
    //     else
    //         A = "P";
    // }
    if (isInput) {
        A = "(" + A + ")";
    }
    size_t nElem = tELEM(t);
    auto ne      = t->ne;
    switch (typ) {
        case 1:
            sprintf(buf + strlen(buf), "%s%s '%s' \t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n", tab, A.c_str(), t->name, ne[0], ne[1], ne[2],
                    ne[3], cNameOf(t->type));
            break;
        default:
            sprintf(buf + strlen(buf), "%s%s '%s' %.3g%s\t[% " PRId64 " % " PRId64 " % " PRId64 " % " PRId64 " %s] \n", tab, A.c_str(), t->name,
                    nElem > 1000 ? nElem / 1.0e6 : nElem, nElem > 1000 ? "M" : "", ne[0], ne[1], ne[2], ne[3], cNameOf(t->type));
            break;
    }
}

int CHECK_SAME_TENSORS(const string &desc, const std::vector<hGensor> &arrA, const std::vector<hGensor> &arrB, int flag) {
    _INFO("\n======== %s @[%s]...", __func__, desc.c_str());
    size_t nA = arrA.size(), nB = arrB.size(), nDup = 0, nMiss = 0;
    bool isSame = arrA.size() == arrB.size();
    std::map<std::string, int> msg;
    int no = 1;
    for (auto tA : arrA) {
        if (msg.find(tA->name) != msg.end()) {
            _INFO("\tAA=\"%s\"", tA->name);
            isSame = false;
        }
        msg[tA->name] = no;
        no++;
    }
    for (auto tB : arrB) {
        if (msg.find(tB->name) == msg.end()) {
            isSame = false;
            nMiss++;
            _INFO("\tB_%d=\"%s\"", nMiss, tB->name);
        }
        no = msg[tB->name];
        if (no < 0) {
            isSame = false;
            nDup++;
        }
        msg[tB->name] = -no;
    }
    for (auto ms : msg) {
        if (ms.second > 0) {
            auto tA = arrA[ms.second - 1];
            _INFO("A_%d=%s ", nMiss, tA->name);
            isSame = false;
            nMiss++;
        }
    }
    _INFO("%s======== %s @[%s] %s. A=%d B=%d \n", isSame ? "\rPASSED " : "\nFailed !", __func__, desc.c_str(), "", nA, nB);
    return 0x0;
}

size_t F_SIZE(const std::string &fpath, FILE *fp0, int flag) {
    try {
        FILE *fp = fp0;
        if (fp0 == NULL) {
            fp = std::fopen(fpath.c_str(), "rb");
            assert(fp != NULL);
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
        if (fp != fp0)
            fclose(fp);
        return (size_t)ret;
    } catch (...) {
        assert(0);
        return 0x0;
    }
}

hGensor GradOf(struct ggml_cgraph *cgraph, hGensor node, int flag) {
#ifdef GG_V12
    assert(0);
    return nullptr;
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
    if (node->grad == nullptr) {
        int maybe = 0;
    }
    return node->grad;
#endif
}

dotprod_t fnDot(typNUMBER tp) {
    int wbit = BitPE(tp);
    return wbit == 4 ? dotprod_gf4 : wbit == 8 ? dotprod_fp8 : wbit == 16 ? dotprod_fp16 : dotprod_fp32;
}

/*
    byte per element of this type(maybe decimals rather than integers!)
*/
double BPE(typNUMBER type) {
    double bp = BitPE(type) / 8.0;
    // assert(bp == 1 || bp == 2 || bp == 4);
    return bp;
}

bool isQuantized(typNUMBER type) {
    if (type == typNUMBER::F8E5M2 || type == typNUMBER::F8E4M3 || type == typNUMBER::F16 || type == typNUMBER::BF16 || type == typNUMBER::F32 ||
        type == typNUMBER::I32 || type == typNUMBER::I8)
        return false;
    if (type == typNUMBER::T_SIGN || type == typNUMBER::T_BINARY || type == typNUMBER::T_BINARY_3 || type == typNUMBER::T_BINARY_TILE) {
        return true;
    }
    assert(0);
    exit(KOIFISH_UNSUPPORTED_DATATYPE);
}

const char *cNameOf(typNUMBER type) {
    static char buf[128];  // Not thread-safe if modified concurrently.
    if (type == typNUMBER::F8E5M2)
        return "F8E5M2";
    if (type == typNUMBER::F8E4M3)
        return "F8E4M3";
    if (type == typNUMBER::F16)
        return "F16(E5)";
    if (type == typNUMBER::BF16)
        return "BF16(E8)";
    if (type == typNUMBER::F32)
        return "float";
    if (type == typNUMBER::I32)
        return "I32";
    if (type == typNUMBER::I8)
        return "I8";
    if (type == typNUMBER::T_BINARY)
        return "BINARY";
    if (type == typNUMBER::T_BINARY_3)
        return "BINARY(3)";
    if (type == typNUMBER::T_SIGN)
        return "TERNARY";
    if (type == typNUMBER::T_BINARY_TILE) {
        sprintf(buf, "TILE(%dx%d)", THREAD_TILE_M, THREAD_TILE_N);
        return buf;  //"TILE(One float for each tile)";
    }

    assert(0 && "cNameOf of UNSUPPORTED_DATATYPE");
    exit(KOIFISH_UNSUPPORTED_DATATYPE);
}

std::string NameOf(typNUMBER type) {
    std::string name = cNameOf(type);
    return name;
}

// double GTensor::bpe() {
//     return BitPE(type) / 8.0;
// }
size_t GTensor::size(int typ) const {
    size_t nz = 1;
    for (auto a : shape) nz *= a;
    switch (typ) {
        case 1:  // only for padded shape
            if (x_shape.size() > 0) {
                assert(BIT_TEST(flags, F_PADDED));
                nz = 1;
                for (auto a : x_shape) nz *= a;
                assert(nz > 1);
            }
            break;
        default:
            break;
    }
    return nz;
}

void *GTensor::DataPad(void *src0, int flag) { return nullptr; }

#ifdef __USE_GGML__
hGensor tSCAL(struct ggml_context *ctx, struct ggml_tensor *a, float s, int flag) {
    // hGensor b = ggml_scale_inplace( ctx,a,s);    // inplace operations are currently not supported!!!
    hGensor b = ggml_scale(ctx, a, s);
    gTN(b, "%s_s", a->name);
    return b;
}

hGensor Permute(struct ggml_context *ctx_, struct ggml_tensor *cur, int64_t n1, int64_t n2, int64_t n3, int64_t n4, bool isCont) {
    hGensor q = ggml_permute(ctx_, cur, n1, n2, n3, n4);
    gTN0(q, "%s.#", cur->name);
    if (isCont) {
        q = ggml_cont(ctx_, q);
        gTN(q, "%s.#c", cur->name);
    }
    return q;
}

hGensor TENSO(void *ctx0, int typ, SHAPE shape, int flag, const string &name) {
    /*va_list args;   //  In C/C++, you can not determine the number of arguments that were passed to a "variadic" function!!!
    va_start( args, typ );
    SHAPE shape;
    int val=-1;
    while (val = va_arg(args, int))  {
        shape.push_back(val);
    }
    va_end(args);*/
    struct ggml_context *ctx = (struct ggml_context *)(ctx0);
    assert(ctx0 != nullptr);
    enum ggml_type type    = (enum ggml_type)typ;
    struct ggml_tensor *gg = nullptr;
    switch (shape.size()) {
        case 1:
            gg = ggml_new_tensor_1d(ctx, type, shape[0]);
            break;
        case 2:
            gg = ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
            break;
        case 3:
            gg = ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
            break;
        case 4:
            ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2], shape[3]);
            break;
        default:
            assert(0);
            break;
    }
    return gg;
}

void Gensor2float_(const hGensor w, float *A, int flag) {
    size_t ne00 = tELEM(w), nbyte = tBYTE(w);
    void *data_0       = w->data;
    enum ggml_type tp0 = (enum ggml_type)w->type;
    void *src0_row     = (void *)((char *)w->data);
    assert(ggml_is_quantized(tp0));
    switch (w->type) {
        // case typNUMBER::F16:
        //     ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
        //     break;
        case typNUMBER::F32:
            break;
        case typNUMBER::Q2_K:
            dequantize_row_q2_K((const block_q2_K *)src0_row, A, ne00);  //-0.00318908691
            break;
        case typNUMBER::Q3_K:
            dequantize_row_q3_K((const block_q3_K *)src0_row, A, ne00);
            break;
        case typNUMBER::Q4_K:
            dequantize_row_q4_K((const block_q4_K *)src0_row, A, ne00);
            break;
        case typNUMBER::Q6_K:
            dequantize_row_q6_K((const block_q6_K *)src0_row, A, ne00);
            break;
        case typNUMBER::Q8_0:
            dequantize_row_q8_0((const block_q8_0 *)src0_row, A, ne00);
            break;
        default:
            assert(0);
            // ggml_tensor_dequant(ctx0,w,typNUMBER::F32);       //memory leak@"float * wdata = malloc(sizeof(float)*ne00)" !!!
            // memcpy(A,w->data,sizeof(float)*ne00);
            // w->data = data_0;       w->type = tp0;
            break;
    }
}
#else
void Gensor2float_(const hGensor w, float *A, int flag) { assert(0); }
#endif

void ADAM_params_::Dump(int typ) {
    _INFO("\tADAM lr=%g,beta=[%g,%g] decay=%g(dim>=%d) clip=%g(alg=%d)\n", alpha, beta1, beta2, decay, decay_min_ndim, gclip, clip_alg);
}

MUON_params_::MUON_params_()    {
    tpDecay = 1;
    //lr_scale = 100.f;     // gradient would explode
}
void MUON_params_::Dump(int typ) {
    // float decay = DecayScale();
    _INFO("\t ldAB=%d lr_scale=%g tpDecay=%d mui=%g ep=(%g,%g) transDown=%d No grad_Clipping!\n", ldAB, lr_scale, tpDecay, mui, eps, eps_loss, isTransDown);
}

void MODEL_CARD::Dump(int typ) {
    // _INFO("\tMODEL card=%s\n", sCardPath.c_str());
}

ggml_cgraph *GG_dup_graph(ggml_context *ctx, ggml_cgraph *src) {
    assert(0);
    return nullptr;
}

#include "../Manifold/Fish.hpp"
#ifdef __USE_GGML__
#include "gguf.h"
#endif
/*
    1.  gguf_get_tensor_offset
*/
bool Fish::GGUF_Serialize(const std::string &path, bool isSave, int flag) {
#ifdef __USE_GGML__
    try {
        if (path.empty())
            return false;
        GST_TIC(tic);
        char buf[1024];
        struct ggml_context *fctx_data = NULL;
        struct gguf_context *fctx      = NULL;
        int n_kv = 0, n_tensors = 0;
        if (isSave) {  // KV pairs
            fctx = gguf_init_empty();
            // struct ggml_init_params params = {128ull*1024ull*1024ull,NULL,false,};
            // fctx_data = ggml_init(params);
        } else {
            fctx = gguf_init_from_file(path.c_str(), {false, &fctx_data});
            if (!fctx) {
                _INFO("%s: failed to load '%s'\n", __func__, path.c_str());
                return false;
            }

            _INFO("%s: version=%d alignment=%zu offset=%zu\n", __func__, gguf_get_version(fctx), gguf_get_alignment(fctx), gguf_get_data_offset(fctx));
            n_kv = gguf_get_n_kv(fctx);
            _INFO("%s: n_kv: %d\n", __func__, n_kv);
            for (int i = 0; i < n_kv; ++i) {
                const char *key = gguf_get_key(fctx, i);
                _INFO_IF("%s: kv[%d]: key = %s\n", __func__, i, key);
            }
            if (0) {  // find kv string
                const char *findkey = "some.parameter.string";
                const int keyidx    = gguf_find_key(fctx, findkey);
                if (keyidx == -1) {
                    printf("%s: find key: %s not found.\n", __func__, findkey);
                } else {
                    const char *key_value = gguf_get_val_str(fctx, keyidx);
                    printf("%s: find key: %s found, kv[%d] value = %s\n", __func__, findkey, keyidx, key_value);
                }
            }
        }
        if (isSave) {
            // if(!std::filesystem::exists(path)){
            //     _INFO("%s: failed to save @'%s'\n", __func__, path.c_str());
            //     return false;
            // }
            for (auto ps : optParams) {
                gguf_add_tensor(fctx, G(ps));
            }
            const bool only_meta = false;
            gguf_write_to_file(fctx, path.c_str(), only_meta);
            size_t fsize = F_SIZE(path.c_str());
            _INFO("[save] @\"%s\" nT=%ld fsize=%gM\tT=%.3g S\n", path.c_str(), optParams.size(), fsize / 1.0e6, GST_TOC(tic));
        } else {
            n_tensors = gguf_get_n_tensors(fctx);
            if (isTrain() && n_tensors != optParams.size()) {  //  optParams maybe empty
                _INFO("%s nOptParams don't match(%d,%d) @%s!", __func__, n_tensors, optParams.size(), path.c_str());
                return false;
            }
            _INFO("[Serialize] n_tensors: %d\n", n_tensors);
            loadGensors.clear();
            for (int i = 0; i < n_tensors; ++i) {
                const char *name = gguf_get_tensor_name(fctx, i);
                ggml_tensor *cur = ggml_get_tensor(fctx_data, name);
                if (cur == nullptr) {
                    _INFO("%s failed to load tensor(%s) @%s!", __func__, name, path.c_str());
                    return false;
                }

                hGensor target = GetGensor(name);
                if (target == nullptr) {
                    if (strcmp(name, "output.weight") == 0 && config.model.isEmbedWeightTying) {
                        continue;
                    } else
                        return false;
                }
                loadGensors.push_back(target);
                if (!optParams.empty()) {
                    if (!(target->flags & GTensor::F_PARAM)) {
                        return false;
                    }
                } else {
                    assert(!isTrain());
                }

#ifdef _TENSOR_G_
                target->CopyGG(cur);
#else
                size_t nEle = tELEM(cur), sz = tBYTE(cur);
                if (nEle != tELEM(target)) {
                    assert(0);
                    continue;
                }
                if (target->type != cur->type) {
                    Gensor2float_(cur, (float *)target->data, 0x0);
                } else
                    memcpy(target->data, cur->data, sz);
#endif
                if (DUMP()) {
                    sprintf(buf, "\t%d d=%d sz=%ld", i, tDIM(target), tBYTE(target));
                    // _pt_cys_(buf,target,0x0);      printf("\n");
                }
                // _INFO("[Serialize]_%d\t%s: n_dims=%d sz = %ld\n",i,cur->name, tDIM(cur),sz);
            }
        }
        if (fctx_data != NULL)
            ggml_free(fctx_data);
        gguf_free(fctx);
        return true;
    } catch (...) {
        return false;
    }
#else
    assert(0);
    return false;
#endif
}