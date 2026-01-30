/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some functions for CLI_params
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "../CLI_params.hpp"

#include <cstring>
#include <iostream>

#include "../Manifold/Serialize.hpp"
#include "../Tensor/GTensor.hpp"
#include "../Utils/GST_obj.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "json.hpp"

int g_dump_level = 1;
int g_dump_each  = 3;
static char buffer[GTensor::MAX_NAME];
#define ARG2STR(format, len)                  \
    {                                         \
        va_list args;                         \
        va_start(args, format);               \
        vsnprintf(buffer, len, format, args); \
        va_end(args);                         \
        assert(strlen(buffer) <= len);        \
    }

int gTN(hGTensor cur, const char* format, ...) {
    int iRet = 0;
    if (strlen(cur->name) == 0) {
        ARG2STR(format, GTensor::MAX_NAME);
        snprintf(cur->name, sizeof(cur->name), "%s", buffer);
        iRet += 1;
    }
    return iRet;
}

int gTN0(hGTensor cur, const char* format, ...) {
    ARG2STR(format, GTensor::MAX_NAME);
    snprintf(cur->name, sizeof(cur->name), "%s", buffer);
    return 0x0;
}

int clamp(const int v, const int min, const int max) { return ((v < min) ? (min) : (v > max) ? (max) : v); }

float fclamp(const float v, const float min, const float max) { return ((v < min) ? (min) : (v > max) ? (max) : v); }

void assert_shape_1d(hGTensor tensor, int64_t ne0) {}
void assert_shape_2d(hGTensor tensor, int64_t ne0, int64_t ne1) {}
void assert_shape_3d(hGTensor tensor, int64_t ne0, int64_t ne1, int64_t ne2) {}
void assert_shape_4d(hGTensor tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {}

hGTensor tRAND(hGTensor tensor, struct random_normal_distribution* rnd) { return nullptr; }

JSON jKEY(const JSON& jConfig, const std::vector<std::string>& keys, int flag) {
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
bool jK2S(const JSON& jConfig, const std::vector<std::string>& keys, char** str, int flag = 0x0) {
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

void UpdateJConfig(JSON& jConfig, const std::string& jPath) {
    try {
        std::ifstream jfile(jPath);
        if (jfile.fail()) {
            _WARN("\r\n%s  Failed to open %s", __func__, jPath.c_str());
        }
        jfile >> jConfig;
        std::string s = jConfig.dump();
    } catch (JSON::parse_error& e) {
        _WARN("\r\n%s  Failed to open %s!!! ERR=%s", __func__, jPath.c_str(), e.what());
    } catch (...) {
        _WARN("\r\n%s  Unknown exception @%s!!!", __func__, jPath.c_str());
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

void train_print_usage(int argc, char** argv, const struct CLI_params* params) {}

bool CLI_params::operator!=(const CLI_params& other) const { return memcmp(this, &other, sizeof(other)); }

DEUG_SWITCH DEBUG;
void DEUG_SWITCH::Dump(int typ) {
    _INFO("[DEBUG]: gemm=%d classifier=%d", T_GEMM, T_classifier_ver);
    _INFO("\n");
}
void CLI_params::Dump(int flag) {
    if (flag == 0x100) {  // only dump jConfig
        assert(!jConfig.empty());
        std::ofstream file("_koifish_tmp_config_.json");
        if (file.is_open()) {
            file << jConfig.dump(4);
            file.close();
        }
        return;
    }

    _INFO("%s::CLI_params: ", exec_name.c_str());
    // _INFO(" n_vocab: %u", n_vocab);
    _INFO("{ n_ctx=%u", n_ctx());
    _INFO(" embd=%u", nEmbed());
    _INFO(" n_ff=%u", n_ff());
    _INFO(" n_head=%u", n_head());
    _INFO(" n_head_kv=%u", n_head_kv());
    _INFO(" n_layer=%d(%d)", nLayer(), n_layer_train);
    _INFO(" f_norm_rms_eps=%g\n", model.norm_rms_eps);
    _INFO(" ROPE: type=%d freq_base=%g freq_scale=%g n_rot=%u\n", model.rope_type, model.rope_freq_base, model.rope_freq_scale, head_dim());
    // _INFO(" lora_r=%d lora_alpha=%d ", lora_r,lora_alpha);
    _INFO(" SepQKV: type=%d  }\n", model.isSeparateQKV);
    // _INFO(" NABLA = %s\n", nabla==0? "" : nabla==3 ? "Embed+AutoEncoder" : (nabla==2 ? "" : "qkv") );
    // _INFO(" SIGMA = %s\n", sigma.c_str());
}

bool LoadJsonFile(const string& jPath, JSON& jObj, int flag) {
    try {
        std::ifstream jfile(jPath);
        std::string info;
        if (jfile.fail()) {
            _WARN("[%s] Failed to open json@\"%s\" !!!\n", __func__, jPath.c_str());
            return false;
        }
        jfile >> jObj;
        return true;
    } catch (...) {
        _ERROR("[%s] Failed to open \"%s\" !!!\n", __func__, jPath.c_str());
        return false;
    }
}

string MODEL_CARD::sWeight = ".weight", MODEL_CARD::sBias = ".bias";  //".w"
string MODEL_CARD::sQzeros = ".qzeros", MODEL_CARD::sQscale = ".scales";

string MODEL_CARD::sLayer = "blk.";  //".norm"
// string MODEL_CARD::sAttnOut = ".wo";                                  //  "_output";    //  "_cat"
string MODEL_CARD::sEmbed = "embed", MODEL_CARD::sInvEmbed = "embed_inv";

MODEL_CARD::MODEL_CARD() {
#if defined(USE_FP16_BASELINE)
    tpWeight = typNUMBER::F16, tpActivation = typNUMBER::F16, tpGradient = typNUMBER::F16;
#elif defined(USE_BF16_BASELINE)
    tpWeight = typNUMBER::BF16, tpActivation = typNUMBER::BF16, tpGradient = typNUMBER::BF16;
#elif defined(USE_FP8_BASELINE)
    tpWeight = typNUMBER::F8E5M2, tpActivation = typNUMBER::BF16, tpGradient = typNUMBER::BF16;
    assert(0);
#endif
}

void QUANT_CARD::Dump(int typ) {
    //_INFO("\t[Quant]_<%d> \n\tMIQ=%d{%s} \n\tF8Ex=%d{%s} \n", default_bits, G_STR(filter_MIQ), G_STR(filter_WeightF8Ex));
}

void QUANT_CARD::Init4Neuron(const std::string& name, const JSON& jQuant, int flag) {
    type = NO_QUANT;
    if (jQuant.find("preQuant") != jQuant.end()) {
        type = PRE_QUANT;
        return;
    }
    string s0 = "";
    for (JSON::const_iterator it = jQuant.begin(); it != jQuant.end(); ++it) {
        auto k = it.key();
        if (!k.empty() && k[0] == '#')
            continue;
        if (k == "debug") {
            continue;
        }
        if (G_Has_(name, {k})) {
            const JSON& jQ = it.value();
            default_bits   = jKV(jQ, {"bits"}, default_bits, false);
            // quant.filter_MIQ        = jKV_arr(jConfig, {"quantizer", "MINI"}, quant.filter_MIQ, false);
            // quant.filter_WeightF8Ex = jKV_arr(jConfig, {"quantizer", "F8Ex"}, quant.filter_WeightF8Ex, false);
            if (default_bits != 4 && default_bits != 3 && default_bits != 8) {
                default_bits = 4;
                assert(0);
            }
            string norm_type = jKV(jQ, {"normal"}, s0);
            norm             = G_Aa(norm_type, "SINKHORN") ? NORMAL_MODE::SINKHORN : NORMAL_MODE::NO_NORMAL;
            //  params.type = method.mi == 0 ? (params.isNormalFloat ? QUANT_MODE::RTNf : QUANT_MODE::RTN) : QUANT_MODE::MINI;
            type = default_bits == 8 ? F8Ex : MINI;
        }
    }
}

bool QUANT_CARD::isValid() const {
    if (default_bits < 0 || default_bits > 8)
        return false;
    return true;
}
void QUANT_CARD::InitFromVendor(const JSON& jVendor, int flag) {
    int bits = 0, group = 0;
    // for QWEN3
    bits  = jKV(jVendor, {"bits"}, bits);
    group = jKV(jVendor, {"group_size"}, group);
    if (bits > 0)
        default_bits = bits;
    if (group > 0)
        T_group = group;
    isPreQuant = true;
}
JSON QUANT_CARD::ToJSON(int flag) {
    JSON jOut;
    jOut["quantizer"]["self_attn"]["bits"]     = default_bits;
    jOut["quantizer"]["self_attn"]["group"]    = T_group;
    jOut["quantizer"]["mlp"]["bits"]           = default_bits;
    jOut["quantizer"]["mlp"]["group"]          = T_group;
    jOut["quantizer"]["embed_tokens"]["bits"]  = default_bits;
    jOut["quantizer"]["embed_tokens"]["group"] = T_group;
    if (isPreQuant)
        jOut["preQuant"] = true;
    return jOut;
}

std::size_t QUANT_CARD::Hash(const QUANT_CARD& params, const std::type_info& ti, const std::string& desc) const {
    // Combine hashes of all elements in the tuple
    auto h1    = std::hash<int>{}(params.default_bits);
    auto h2    = std::hash<QUANT_MODE>{}(type);
    string sTi = ti.name();
    auto h3    = std::hash<std::string>{}(sTi);
    auto h4    = std::hash<std::string>{}(desc);

    // A simple way to combine hashes (boost-style or similar)
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
}

MODEL_ARCH CLI_params::ModelArch() {
    MODEL_ARCH arch = MODEL_ARCH::_X_;
    string info     = "";
    if (!model.isLoadCard()) {
        string s = jKV(jConfig, {"model", "arch"}, string(""), false);
        assert(!s.empty());
        info = s;
    } else {
        info = model.model_type;
    }
    if (model_title.empty())
        model_title = info;

    std::transform(info.begin(), info.end(), info.begin(), ::toupper);
    arch = info == "MOE"         ? NLP_MOE
           : info == "MAMBA"     ? MODEL_ARCH::NLP_MAMBA
           : info == "GUPPY"     ? MODEL_ARCH::NLP_GUPPY
           : info == "DEEPSEEK"  ? MODEL_ARCH::NLP_DEEPSEEK
           : info == "QWEN2"     ? MODEL_ARCH::NLP_QWEN2
           : info == "QWEN2.5"   ? MODEL_ARCH::NLP_QWEN2
           : info == "QWEN3"     ? MODEL_ARCH::NLP_QWEN3
           : info == "QWEN3_MOE" ? MODEL_ARCH::NLP_QWEN3
           : info == "GPT2"      ? MODEL_ARCH::NLP_GPT2
           : info == "GPT2CHAR"  ? MODEL_ARCH::NLP_GPT2_char
           : info == "LAMA"      ? MODEL_ARCH::NLP_LLAMA
           : info == "MISTRAL"   ? MODEL_ARCH::NLP_MISTRAL
                                 : MODEL_ARCH::NLP_LLAMA;

    return arch;
}

bool CLI_params::isValid(int flag) { return true; }

bool CLI_params::JModel2Params(int flag) {
    string key = "";
    try {
        jModel = jKEY(jConfig, {"model"});
        if (jModel.empty()) {
            _ERROR("%s  No \"model\" in jConfig!!!", __func__);
            return false;
        }
        jBackBone = jKEY(jModel, {"backbone"});
        if (jBackBone.empty()) {
            _ERROR("%s  No \"backbone\" in jConfig!!!", __func__);
            return false;
        }
        if (jModel.find("hf-card") != jModel.end())
            model.sCardPath = jKEY(jModel, {"hf-card"});
        if (jModel.find("token_bin_path") != jModel.end())  //  only for custom
            model.sTokenBinPath = jKEY(jModel, {"token_bin_path"});
        else {
            // model.sTokenBinPath = "./Datasets/climb-1b/tokenizer.dat";
        }

        jQuant = jKEY(jConfig, {"quantizer"});

        nLayerX = 1;  // at least 1 layer
        nLayerX = jKV(jConfig, {"model", "parameter", "Layer"}, nLayerX);
        assert(nLayerX < 160 && nLayerX > 0);
        model.max_pos_embeddings = jKV(jConfig, {"model", "parameter", "max_pos_embeddings"}, model.max_pos_embeddings);
        // nearly same ???
        // if(nLayerX>0)
        //     common.residual_scale = 1.0f / sqrtf(2.0f * nLayerX);
        assert(model.layerps.size() == 0);
        auto jTrans = jKEY(jConfig, {"model", "parameter", "transformer"});
        if (!jTrans.empty()) {
            int nH = jKV(jTrans, {"Head"}, -1), nF = jKV(jTrans, {"Ffn"}, -1), nE = -1, nC = jKV(jTrans, {"Ctx"}, -1), nHd = -1;
            assert(nH > 0);
            int nKV = jKV(jTrans, {"KVHead"}, nH);

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
            model.token_embeds.push_back(embeds[0]);
            model.qkv_embeds = embeds;
            if (nF < 0)
                nF = nEmbed() * 4;

            nHd = jKV(jTrans, {"head_dim"}, nHd);
            if (nHd == -1) {  //
                nHd = nEmbed() / nH;
            }
            assert(nHd > 0);

            if (nH > 0 && nF > 0) {
                for (int i = 0; i < nLayerX; i++) model.layerps.push_back(MODEL_CARD::LAY_PARAM(nH, nKV, nHd, nF));
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
            // common.muon.isTransDown = false;
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
    } catch (JSON::parse_error& e) {
        _WARN("%s  Failed to open %s!!! ERR=%s", __func__, key.c_str(), e.what());
        return false;
    } catch (...) {
        _WARN("%s  Unknown exception @%s!!!", __func__, key.c_str());
        return false;
    }
}

bool CLI_params::isShareLayerOut() const {
    // if(common.Empty())  //no training,only infer or evaluate
    //     return true;
    if (phase == LIFE_PHASE::P_GENERATE)
        return true;

    if (scheduling.strategy == MEM_STRATEGY::PRE_ALLOC_GPU || scheduling.strategy == MEM_STRATEGY::PRE_ALLOC_HOST_MAP)
        return false;
    return true;
}

/*
    REF 1.n_ctx_orig 2.n_ctx_train
*/
uint32_t CLI_params::n_ctx() const {
    int n = -1;
    switch (phase) {
        case P_GENERATE:  // case P_CHAT:
            n = chat_sampler.seq_len;
            if (n_ctx_train > 0)
                assert(n <= n_ctx_train);
            break;
        default:
            return common.n_ctx;
    }
    return n;
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

bool Fuyou_params::Init(CLI_params* hConfig, const JSON& jConfig, int flag) {
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
    assert(model.token_embeds.size() > 0);
    if (flag == -1)
        return model.token_embeds[0];
    int no = model.token_embeds.size() - 1;
    return model.token_embeds[no];
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
            break;
        case MODEL_ARCH::NLP_GPT2:
        case MODEL_ARCH::NLP_GPT2_char: {
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

            // if (model.layerps.size() == 0 && !isJModel) {  //  deprecated
            //     int n_ff0 = jKV(jConfig, {"model_v0", "ffn", "length"}, 3072, false), nLay = nLayer();
            //     int hd = nEmbed() / n_head();
            //     for (int i = 0; i < nLayer(); i++) {
            //         MODEL_CARD::LAY_PARAM lay(nH, nH, hd, n_ff0);
            //         model.layerps.push_back(lay);
            //     }
            // }

            // model.tpPreLogits = typNUMBER::BF16;
            //  need new GEMM! cuBLASLt requires bias in FP8 mode to be BF16... (sigh)
            model.isNormalBias       = true;
            model.isSLPBias          = true;  // nealy same
            model.isPaddedCls        = true;  //  ceil(n/128.0)*128
            model.isSeparateQKV      = true;
            model.qkv4dnn            = QKV_PACK::QQKKVV;
            model.rope_type          = ROPE_NONE;
            model.isEmbedWeightTying = true;
            model.preLogits_dB       = 8;
            model.fActSLP = GELU, model.fActFFN = GELU;

            int group = Get({"model_v0", "target_group"}, 1);
            assert(group == 1);
        } break;
        case NLP_DEEPSEEK:
            model.sLayer = "layers.";
            break;
        case NLP_MISTRAL:
            model.sLayer = "layers.";
            break;

        default:
            _INFO("[ARCH]=%s\n", "");
            break;
    }
}

TRAIN_CARD::TRAIN_CARD() {
    // seed                    = -1;
    // n_ctx                   = 128;
    // n_threads               = 6;
    n_batch = 1;
    // n_gradient_accumulation = 1;
    // n_epochs                = -1;
    // n_gpu_layers            = 0;

    custom_n_ctx = false;

    use_flash         = false;
    use_checkpointing = true;

    sample_start           = "";
    include_sample_start   = false;
    escape                 = false;
    overlapping_samples    = false;
    fill_with_next_samples = false;
    separate_with_eos      = false;
    separate_with_bos      = true;
    sample_random_offsets  = false;
    force_reshuffle        = false;

    opt_past               = 0;
    opt_delta              = 1e-5f;
    opt_max_no_improvement = 0;

    warmup     = 600;
    lr_restart = 0;

    // adam.n_iter         = -1;
    adam.alpha          = 1e-3f;
    adam.min_alpha      = 0;
    adam.decay          = 1e-1f;
    adam.decay_min_ndim = 2;
    adam.beta1          = 0.9f;
    adam.beta2          = 0.95f;  // 0.999f;
    adam.gclip          = 1.0f;
    adam.eps_loss       = 1e-5f;
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

void _JSON_remove_keys_(JSON& node, const std::string& prefix) {
    if (node.is_object()) {
        for (auto it = node.begin(); it != node.end();) {
            if (it.key().compare(0, prefix.size(), prefix) == 0) {
                string key = it.key();
                it         = node.erase(it);
            } else {
                _JSON_remove_keys_(it.value(), prefix);
                ++it;
            }
        }
    } else if (node.is_array()) {
        for (auto& element : node) {
            _JSON_remove_keys_(element, prefix);
        }
    }
}

JSON CLI_params::ToJSON(int type, int flag) {
    JSON json, jOut = jConfig;
    _JSON_remove_keys_(jOut, "#");
    switch (type) {
        case 0x100:  //@Fish::SAFETENSOR_Serialize
            jOut["checkpoint_in"] = jConfig["checkpoint_out"];
            jOut.erase("checkpoint_out");

            break;
        default:
            assert(0);
            break;
    }
    json["config"] = jOut;
    return json;
    /*
    std::ofstream file(file_name);
        if(!file.is_open()) {
            throw std::runtime_error(fmt::format("could not open file for writing {}", file_name));
        }

        std::vector<std::string> archs;
        if(config.Architecture == LLamaConfig::QWEN2) {
            archs = {"Qwen2ForCausalLM"};
        } else if (config.Architecture == LLamaConfig::LLAMA) {
            archs = {"LlamaForCausalLM"};
        }

        nlohmann::json config_json;
        config_json["architectures"] = std::move(archs);
        config_json["bos_token_id"] = config.BosTokenId;
        config_json["eos_token_id"] = config.EosTokenId;
        config_json["hidden_size"] = config.HiddenSize;
        config_json["intermediate_size"] = config.IntermediateSize;
        config_json["vocab_size"] = config.VocabSize;
        config_json["num_attention_heads"] = config.NumQueryHeads;
        config_json["num_key_value_heads"] = config.NumKeyValHeads;
        config_json["num_hidden_layers"] = config.NumLayers;
        config_json["max_position_embeddings"] = config.MaxPositionEmbeddings;
        config_json["rope_theta"] = config.RopeTheta;
        config_json["rms_norm_eps"] = config.RmsNormEps;
        config_json["tie_word_embeddings"] = config.TiedWordEmbeddings;
        config_json["torch_dtype"] = dtype_to_torch_str(config.DType);

        config_json["attention_dropout"] = 0.f;
        config_json["initializer_range"] = 0.02f;
        config_json["hidden_act"] = "silu";
        config_json["use_cache"] = true;
        if(config.Architecture == LLamaConfig::QWEN2) {
            config_json["model_type"] = "qwen2";
            config_json["max_window_layers"] = config.NumLayers;
            config_json["sliding_window"] = config.MaxPositionEmbeddings;
            config_json["use_sliding_window"] = false;
            config_json["use_mrope"] = false;
        } else if (config.Architecture == LLamaConfig::LLAMA) {
            config_json["model_type"] = "llama";
            config_json["attention_bias"] = false;
            config_json["mlp_bias"] = false;
        }

        file << config_json.dump(4);*/
}

bool TRAIN_CARD::Init(CLI_params* hConfig, const JSON& jConfig, int flag) {
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

CheckPoint_Params::CheckPoint_Params(const JSON& jData, const std::string& key, bool isSave, int flag) : jKey(key) {
    if (key.empty()) {  //  default checkpoints to store all info of current state
        //"state", "./hy-tmp/checkpoint/", -1, false
        jKey   = "._koifish_state_";
        sDir   = "./hy-tmp/checkpoint/";
        type   = CheckPoint_Params::STATE;
        format = CheckPoint_Params::KOIFISH;
        // isIn = false;
    } else {
        if (!jData.contains(key)) {
            _INFO("");
        }
        JSON jdata = jData[key];
        assert(!jdata.empty());
        // const std::string &tp, const std::string &p, int x, bool in
        bool isFind = false;
        std::string stp = jdata["type"];
        for (auto kv : CKP_desc) {
            if (stp == kv.second) {
                type   = kv.first;
                isFind = true;
            }
        }
        sDir       = jKV(jdata, {"path"}, sDir);
        save_every = jKV(jdata, {"save-every"}, save_every);
        if (isSave) {
            format = CheckPoint_Params::HF;
        }
        // isIn       = key == "in";  // hack
    }
    if (isSave) {  //  Verify ouput dir
        if (sDir.empty()) {
            sDir = "./hy-tmp/checkpoint/Koifish_";
            _INFO("[Save] path is empty! Koifish would save file to a tmp path@\"%s\"!\n", sDir.c_str());
        }
        VERIFY_DIR_EXIST(sDir, true);
    }
}

std::string CheckPoint_Params::FullPath(bool isSave, int flag) {
    string sOut = "", sExt = CKP_ext[type];
    switch (format) {
        case HF:
            if (isSave) {
                sOut = sDir + "model.safetensors";
            } else {
                sOut = sModelPath;
            }
            break;
        default:
            if (isSave) {
                if (!sX.empty()) {
                    sOut = sDir + sX;  //+ std::to_string(iter)
                } else
                    sOut = sDir + jKey;
                sOut += "." + sExt;
            } else {
                sOut = sModelPath;
            }
            break;
    }

    assert(!sOut.empty());
    return sOut;
}

bool CheckPoint_Params::SerialSnap(JSON& jConfig, bool isSave, int flag) {
    if (isSave) {
        JSON& jSnapshot       = jConfig["checkpoint_out"][jKey]["snapshot"];
        jSnapshot["curFuyou"] = curFuyou;
        jSnapshot["curIter"]  = curIter;
        jSnapshot["curEpoch"] = curEpoch;
        for (auto seed : seeds) {
            jSnapshot["seeds"][seed.first] = seed.second;
        }
        // JSON j_array = fuyou_filter_reload;
        jSnapshot["fuyou_filter"] = fuyou_filter_reload;

    } else {
        JSON& jSnapshot = jConfig["checkpoint_in"][jKey]["snapshot"];
        assert(!jSnapshot.empty());
        curFuyou            = jKV(jSnapshot, {"curFuyou"}, curFuyou);
        curIter             = jKV(jSnapshot, {"curIter"}, curIter);
        curEpoch            = jKV(jSnapshot, {"curEpoch"}, curEpoch);
        fuyou_filter_reload = jKV_arr(jSnapshot, {"fuyou_filter"}, fuyou_filter_reload);
    }
    return true;
}

bool CLI_params::InitChekcpoints(int argc, char** argv, const std::string& ckp_queue, int flag) {
    try {
        JSON jdata     = jKEY(jConfig, {ckp_queue});
        string type    = "", path;
        int save_every = -1;
        for (JSON::const_iterator it = jdata.begin(); it != jdata.end(); ++it) {
            auto k = it.key();
            if (!k.empty() && k[0] == '#')
                continue;
            if (k == "debug") {
                continue;
            }
            if (ckp_queue == "checkpoint_out")
                ckp_out.push_back(CheckPoint_Params(jdata, k, true));
            else {  //  "checkpoint_in" contains many tensor infos
                ckp_in.push_back(CheckPoint_Params(jdata, k, false));
            }
        }

        if (phase == P_EVAL_) {
            assert(!ckp_in.empty());
            assert(argc > 2);
            auto& fish_in      = ckp_in[ckp_in.size() - 1];
            fish_in.sModelPath = argv[1];
        } else {
            // default checkpoints to store all info of current state
            state = CheckPoint_Params(jConfig, "", true);
            state.Init();
            // checkpoints.push_back(state);
        }
    } catch (JSON::parse_error& e) {
        _ERROR("\r\n%s  Failed to open %s!!! ERR=%s", __func__, ckp_queue.c_str(), e.what());
        return false;
    } catch (...) {
        _ERROR("\r\n%s  Unknown exception @%s!!!", __func__, ckp_queue.c_str());
        return false;
    }
    return true;
}

bool CLI_params::ToJConfig(int flag) {
    try {
        assert(jConfig.empty());
        jConfig["version"]                                       = "0.1.0";
        jConfig["model"]["hf-card"]                              = model.sCardPath;
        jConfig["model"]["parameter"]["Layer"]                   = nLayer();
        jConfig["model"]["parameter"]["transformer"]["Ctx"]      = n_ctx();
        jConfig["model"]["parameter"]["transformer"]["Embed"]    = nEmbed();
        jConfig["model"]["parameter"]["transformer"]["Ffn"]      = n_ff();
        jConfig["model"]["parameter"]["transformer"]["Head"]     = n_head();
        jConfig["model"]["parameter"]["transformer"]["KVHead"]   = n_head_kv();
        jConfig["model"]["parameter"]["transformer"]["head_dim"] = head_dim();

        assert(jBackBone.empty() && "jBackBone is not empty!");
        jBackBone["embed_tokens"]["Embedding"] = JSON::array();
        jBackBone["layer"]["self_attn"]["QKV"] = JSON::array();
        jBackBone["layer"]["mlp"]["FFN"]       = JSON::array();
        jBackBone["norm"]["Normal"]            = JSON::array();
        jBackBone["output"]["CLASIFY"]         = JSON::array();
        jConfig["model"]["backbone"]           = jBackBone;

        if (jQuant.empty()) {
            if (DEBUG.test_quant) {
                jConfig["quantizer"]["self_attn"]["bits"]    = 3;
                jConfig["quantizer"]["mlp"]["bits"]          = 4;
                jConfig["quantizer"]["embed_tokens"]["bits"] = 4;
                jQuant                                       = jKEY(jConfig, {"quantizer"});
            }
        } else {                     // hf model has quantized, for example: Qwen3-32B-AWQ
            jConfig.update(jQuant);  // jConfig["quantizer"] = jQuant;  //
        }
        if (!jVendorQuant.empty()) {
            jConfig["vendor_quantizer"] = jVendorQuant;
        }

        chat_sampler.seq_len          = 1024;  // 512;
        jConfig["gpt"]["max_seq_len"] = chat_sampler.seq_len;
        // jConfig["debug"]["prompts"] = "hello";

        common.seed     = 42;
        jConfig["seed"] = common.seed;

        jModel = jKEY(jConfig, {"model"});

        if (DEBUG.test_quant) {  // some debug switch
            DEBUG.prompts = {"hello",
                             "What is the capital of Shanghai?",
                             "Who wrote the play Romeo and Juliet?",
                             "In which year did the Titanic sink?",
                             "What is the chemical symbol for the element gold?",
                             "What is the longest river in the world?",
                             "Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
                             "How many games did Arsenal FC go unbeaten during the 2003-2004 season of the English Premier League",
                             "I get out on the top floor (third floor) at street level. How many stories is the building above the ground?",
                             "天命玄鸟,降而生生. 玄鸟是什么鸟?"};
        } else {
        }

        Dump(0x100);
        return true;
    } catch (JSON::parse_error& e) {
        _INFO("\r\n%s  Failed!!! ERR=%s", __func__, e.what());
        return false;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception @%s!!!", __func__, jsPath.c_str());
        return false;
    }
}
/*
    Some trick
    1 Large batch size would decrease osillation
    2 Double batch+half layers =>  More accuracy in same training time
*/
bool CLI_params::InitJConfig(int flag) {
    try {
        char* env_str = getenv("PATH");
        env_str       = getenv("LD_LIBRARY_PATH");

        // common = get_default_train_params_common();

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

        chat_sampler.seq_len = jKV(jConfig, {"gpt", "max_seq_len"}, chat_sampler.seq_len);  // 128

        if (chat_sampler.seq_len <= 0)
            chat_sampler.seq_len = 8192;

        if (!JModel2Params(0x0))
            return false;
        if (!model.sCardPath.empty()) {
            if (!model.InitHugFace(this, jConfig, ""))
                return false;
        } else {
        }

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
        DEBUG.fLongTail  = jKV(jConfig, {"debug", "long_tail"}, DEBUG.fLongTail);
        DEBUG.prompts    = jKV_arr(jConfig, {"debug", "prompts"}, DEBUG.prompts);

        SUM::nMostMemItem    = jKV(jConfig, {"dump", "most_mem_item"}, SUM::nMostMemItem);
        SUM::nMinTensorAlloc = jKV(jConfig, {"dump", "min_tensor_alloc"}, SUM::nMinTensorAlloc);

        dumpSwitch.train_time     = jKV(jConfig, {"dump", "train_time"}, dumpSwitch.train_time);
        dumpSwitch.tensor_ref     = jKV(jConfig, {"dump", "tensor_ref"}, dumpSwitch.tensor_ref);
        dumpSwitch.train_csv_path = jKV(jConfig, {"dump", "train_csv_path"}, dumpSwitch.train_csv_path);
        // train = jKV(jConfig,{"train"},train );
        return true;
    } catch (JSON::parse_error& e) {
        _ERROR("\r\n%s  Failed to open %s!!! ERR=%s", __func__, jsPath.c_str(), e.what());
        return false;
    } catch (...) {
        _ERROR("\r\n%s  Unknown exception @%s!!!", __func__, jsPath.c_str());
        return false;
    }
}

bool CLI_params::parse(int argc, char** argv) {
    std::string arg_prefix = "--", key, value;
    exec_name              = EXE_name();
    string sExt            = argc > 1 ? FILE_EXT(argv[1]) : "";
    bool isJConfig         = false;
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
                SAFETENSOR_Load_jconfig(arg, jConfig, FSerial::FILE_FISH);
            } else if (sExt == "ck") {
                SAFETENSOR_Load_jconfig(arg, jConfig, FSerial::FILE_CHECKPOINT);
            }
            if (jConfig.empty())
                return false;
            isJConfig = true;
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
            jEval["samp"] = 1.0;
            // jEval["glob"] = argv[i++];
            jConfig["datasets_new"]["eval"] = jEval;
        } else if (arg == "--hf") {  // directory of hf model
            assert(i + 1 < argc);
            model.sCardPath = argv[++i];
            if (!VERIFY_DIR_EXIST(model.sCardPath, false)) {
                K_EXIT(KOIFISH_INVALID_ARGS_MODEL);
            }
        } else if (arg == "--prompts") {  // directory of hf model
            assert(i + 1 < argc);
            DEBUG.prompts = {argv[++i]};
        } else if (arg == "--tokenizer") {  // directory of tokenizer
            assert(i + 1 < argc);
            model.sTokenBinPath = argv[++i];
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
    if (isJConfig) {
        if (!InitJConfig())
            return false;
    } else {
        if (!model.InitHugFace(this, jConfig, model.sCardPath, 0x0))
            return false;
    }
    // Dump(0x100);

    switch (phase) {
        case P_GENERATE:
            break;
        case P_EVAL_:
            InitChekcpoints(argc, argv, "checkpoint_in");
            break;
        default:
            InitChekcpoints(argc, argv, "checkpoint_in");
            InitChekcpoints(argc, argv, "checkpoint_out");
            InitAllStates(0x0);
            break;
    }
    OnArch();
    return true;
}

int Gensor_loab(struct ggml_context* ctx0, hGensor w, int nHeavy, hGensor ga, hGensor gb, int flag) {
    printf("%s@%s <== %s x %s\n\t", __func__, w->name, ga->name, gb->name);
    auto shape = w->ne;
    int nIn = shape[0], nOut = shape[1], rank = nHeavy;  // min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);
    assert(nIn > 0 && nOut > 0 && ne00 == nIn * nOut);
    assert(nIn > nHeavy && nOut > nHeavy && nHeavy > 0);
    float* A = Gensor2float(ctx0, w, flag);
    auto svd = std::make_shared<LoSVD<float>>("Gensor_loab", A, nIn, nOut, rank, 1.0e-3);  // 1.0e-3
    assert(ga->type == typNUMBER::F32 && gb->type == typNUMBER::F32);
    if (!svd->Build()) {
        return -1;
    } else {
        if (tELEM(ga) != nIn * rank || tELEM(gb) != nOut * rank) {
            return -2;
        }
        svd->US((float*)((char*)ga->data));
        memcpy(gb->data, svd->V(), sizeof(float) * rank * nOut);
    }
    delete[] A;
    return 0x0;
}

int Gensor_SVD(struct ggml_context* ctx0, hGensor w, int nHeavy, hGensor U, hGensor D, hGensor V, int flag) {
    printf("%s@%s \t ......", __func__, w->name);

    auto shape = w->ne;
    int nIn = shape[0], nOut = shape[1], rank = nHeavy;  // min(64,min(nIn,nOut)/10);
    size_t ne00 = tELEM(w);
    assert(nIn > 0 && nOut > 0 && ne00 == nIn * nOut);
    assert(nIn > nHeavy && nOut > nHeavy && nHeavy > 0);
    float* A = Gensor2float(ctx0, w, flag);
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
            float *Sigma = svd->S(), *mD = (float*)(D->data);
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
template <typename T>
double P_softmax(int idx, T* logits, int size) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        float a = T2Float(logits + i);
        max_val = a > max_val ? a : max_val;
    }
    float partition = 0.0f;
    for (int i = 0; i < size; i++) {
        float a = T2Float(logits + i);  // logits[i];
        partition += expf(a - max_val);
    }
    //
    return expf(float(logits[idx]) - max_val) / partition;
}
template double P_softmax<floatLogits>(int idx, floatLogits* logits, int size);

float SOFT_MAX(const int n, float* y, const float* x) {
    float x1 = -INFINITY;
    int i;
    // ggml_vec_max_f32(n, &x1, x);
    for (i = 0; i < n; ++i) {
        x1 = std::max(x1, x[i]);
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
float SOFT_MAX_minus(const int n, float* y, const float* x) {
    assert(0);
    float x1 = -INFINITY, a;
    int i;
    for (i = 0; i < n; ++i) {
        x1 = std::max(x1, x[i]);
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

struct ggml_tensor* ggml_cross_entropy_loss_1(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
#ifndef GG_V12
    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    struct ggml_tensor* result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op     = GGML_OP_CROSS_ENTROPY_LOSS_1;
    result->grad   = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src[0] = a;
    result->src[1] = b;

    return result;
#endif
    return nullptr;
}

void _T_repr_(hGensor t, const char* tab, char* buf, int typ) {
    if (t == nullptr)
        return;
    bool isInput = t->flags & GTensor::F_INPUT;
    string A     = NameOf(t->type);  //==typNUMBER::F16 ? "d16":"d";
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

int CHECK_SAME_TENSORS(const string& desc, const std::vector<hGensor>& arrA, const std::vector<hGensor>& arrB, int flag) {
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

size_t F_SIZE(const std::string& fpath, FILE* fp0, int flag) {
    try {
        FILE* fp = fp0;
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

hGensor GradOf(struct ggml_cgraph* cgraph, hGensor node, int flag) {
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

const char* cNameOf(typNUMBER type) {
    static char buf[128];  // Not thread-safe if modified concurrently.
    if (type == typNUMBER::T_BINARY_TILE) {
        sprintf(buf, "TILE(%dx%d)", THREAD_TILE_M, THREAD_TILE_N);
        return buf;  //"TILE(One float for each tile)";
    } else
        return K_FLOATS[type].name.c_str();

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

void* GTensor::DataPad(void* src0, int flag) { return nullptr; }

void Gensor2float_(const hGensor w, float* A, int flag) { assert(0); }

void ADAM_params_::Dump(int typ) {
    _INFO("\tADAM lr=%g,beta=[%g,%g] decay=%g(dim>=%d) clip=%g(alg=%d)\n", alpha, beta1, beta2, decay, decay_min_ndim, gclip, clip_alg);
}

MUON_params_::MUON_params_() {
    tpDecay = 1;
    // lr_scale = 100.f;     // gradient would explode
}
void MUON_params_::Dump(int typ) {
    // float decay = DecayScale();
    _INFO("\t ldAB=%d lr_scale=%g tpDecay=%d mui=%g ep=(%g,%g) transDown=%d No grad_Clipping!\n", ldAB, lr_scale, tpDecay, mui, eps, eps_loss, isTransDown);
}

void MODEL_CARD::Dump(int typ) {
    // _INFO("\tMODEL card=%s\n", sCardPath.c_str());
}

ggml_cgraph* GG_dup_graph(ggml_context* ctx, ggml_cgraph* src) {
    assert(0);
    return nullptr;
}

#include "../Manifold/Fish.hpp"

/*
    1.  gguf_get_tensor_offset
*/
bool Fish::GGUF_Serialize(const std::string& path, bool isSave, int flag) {
#ifdef __USE_GGML__
    try {
        if (path.empty())
            return false;
        GST_TIC(tic);
        char buf[1024];
        struct ggml_context* fctx_data = NULL;
        struct gguf_context* fctx      = NULL;
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
                const char* key = gguf_get_key(fctx, i);
                _INFO_IF("%s: kv[%d]: key = %s\n", __func__, i, key);
            }
            if (0) {  // find kv string
                const char* findkey = "some.parameter.string";
                const int keyidx    = gguf_find_key(fctx, findkey);
                if (keyidx == -1) {
                    printf("%s: find key: %s not found.\n", __func__, findkey);
                } else {
                    const char* key_value = gguf_get_val_str(fctx, keyidx);
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
                const char* name = gguf_get_tensor_name(fctx, i);
                ggml_tensor* cur = ggml_get_tensor(fctx_data, name);
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
                    Gensor2float_(cur, (float*)target->data, 0x0);
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

// array: a bit stream from left to right
void BIT_SET_k(hBITARR array, size_t offset, BIT_8 elem, int bits) {
    size_t boff = offset * bits;
    for (int i = 0; i < bits; i++, boff++) {
        size_t id = boff / 8, shift = 7 - boff % 8, flag = 1 << shift;
        BIT_8 bit = (elem >> (bits - 1 - i)) & 0x1;
        if (bit)
            BIT_SET(array[id], flag);
        else
            BIT_RESET(array[id], flag);
    }
}
BIT_8 BIT_GET_k(hBITARR array, size_t offset, int bits) {
    BIT_8 elem  = 0x0;
    size_t boff = offset * bits;
    for (int i = 0; i < bits; i++, boff++) {
        size_t id = boff / 8, shift = 7 - boff % 8, flag = 1 << (bits - 1 - i);
        BIT_8 bit = (array[id] >> shift) & 0x1;
        if (bit)
            BIT_SET(elem, flag);
        // else
        //     BIT_RESET(elem, flag);
    }
    return elem;
}
void BIT_SET_4(hBITARR array, size_t offset, BIT_8 elem) {
    assert(0);
    return;
}

bool MODEL_CARD::InitHugFace(CLI_params* hConfig, const JSON& jConfig, const std::string& sCardPath_0, int flag) {
    bool isInitFromPath = !sCardPath_0.empty();
    int head_dim = -1, n_heads = -1, n_kv_heads = -1, n_FF = -1;
    if (!isInitFromPath) {
        sTokenBinPath = "./assets/tokenizer_151936.bin";
        n_layers      = hConfig->nLayer();
        for (int i = 0; i < n_layers; i++) {
            int nH = hConfig->n_head(i), nF = hConfig->n_ff(i);
            assert(nH > 0 && nF > 0);
        }

        string sTyp = jKVs(jConfig, {"model", "datatype", "weight"}, string(""));
        if (!sTyp.empty())
            tpWeight = tpNumOf(sTyp);
        // sTyp = jKVs(jConfig, {"model", "datatype", "embed"}, string(""));
        // if (!sTyp.empty())
        //     tpEmbed = tpNumOf(sTyp);
        head_dim = hConfig->head_dim();
        n_heads = hConfig->n_head(), n_kv_heads = hConfig->n_head_kv();
        // seq_len = hConfig->n_ctx_orig();

        sCardPath = jKV(jConfig, {"model", "hf-card"}, sCardPath);
    } else {
        sCardPath = sCardPath_0;
    }
    if (sCardPath.empty()) {
        return false;
    }
    string jPath = sCardPath + "config.json";
    if (!LoadJsonFile(jPath, jModelParam)) {
        return false;
    };
    sTokenJsonPath = sCardPath + "tokenizer.json";
    // LoadJsonFile(sTokenPath,jTokenizer);                             // }

    if (jModelParam.empty()) {
        sCardPath = "";
    } else {
        isEmbedWeightTying = jKV(jModelParam, {"tie_word_embeddings"}, isEmbedWeightTying);
        if (isEmbedWeightTying) {
            skip_st.push_back("lm_head.weight");  //  so strange: Qwen/Qwen3-0.6B has this & unsloth/Qwen3-0.6B-Base remove this
        }
        model_type = jKV(jModelParam, {"model_type"}, model_type);
        assert(model_type != "");
        num_attention_heads = jKV(jModelParam, {"num_attention_heads"}, n_heads);
        num_key_value_heads = jKV(jModelParam, {"num_key_value_heads"}, n_kv_heads);
        if (jModelParam.find("quantization_config") != jModelParam.end()) {
            hConfig->jVendorQuant = jModelParam["quantization_config"];
            QUANT_CARD quant;
            quant.InitFromVendor(jModelParam["quantization_config"]);
            hConfig->jQuant = quant.ToJSON();
        }
        vocab_size           = jKV(jModelParam, {"vocab_size"}, vocab_size);
        torch_dtype          = jKV(jModelParam, {"torch_dtype"}, torch_dtype);
        transformers_version = jKV(jModelParam, {"transformers_version"}, transformers_version);
        bos_token_id         = jKV(jModelParam, {"bos_token_id"}, bos_token_id);
        eos_token_id         = jKV(jModelParam, {"eos_token_id"}, eos_token_id);
        rope_theta           = jKV(jModelParam, {"rope_theta"}, rope_theta);
        hidden_size          = jKV(jModelParam, {"hidden_size"}, hidden_size);
        intermediate_size    = jKV(jModelParam, {"intermediate_size"}, intermediate_size);
        max_pos_embeddings   = jKV(jModelParam, {"max_position_embeddings"}, max_pos_embeddings);
        // rotary_dim           = jKVs(jModelParam, {"rope_scaling"}, rotary_dim);
        //
        int hd = -1;
        if (!jModelParam.contains("head_dim")) {  // some config.json of hf model don't have "head_dim"
            assert(hidden_size%num_attention_heads==0);
            hd = hidden_size/num_attention_heads;
            assert(hd > 0 && hd < 10240 && "Invalid head_dim @ computed InitHugFace");
        } else {
            hd = jKV(jModelParam, {"head_dim"}, head_dim);
            assert(hd > 0 && hd < 10240 && "Invalid head_dim @InitHugFace");
            if (hd != head_dim) {
                for (auto& lay : layerps) {
                    lay._head_dim = hd;
                }
            }
        }        

        if (!isInitFromPath)
            assert(num_attention_heads == n_heads && num_key_value_heads == n_kv_heads);
        else {
            n_layers         = jKV(jModelParam, {"num_hidden_layers"}, n_layers);
            hConfig->nLayerX = n_layers;
            token_embeds.push_back(hidden_size);
            if (hConfig->common.n_ctx == -1) {  //  max_position_embeddings(often the same as training context length)
                hConfig->common.n_ctx = max_pos_embeddings;
                hConfig->n_ctx_train  = max_pos_embeddings;
            }
            for (int i = 0; i < n_layers; i++) {
                layerps.push_back(MODEL_CARD::LAY_PARAM(num_attention_heads, num_key_value_heads, hd, intermediate_size));
            }
        }

        InitChatTemplate(hConfig);
    }

    LoadJsonFile(sCardPath + "model.safetensors.index.json", jSafetensorsIndex);
    if (!jSafetensorsIndex.empty()) {
        nTotalSize = jKV(jSafetensorsIndex, {"metadata", "total_size"}, nTotalSize);

        auto jMap = jSafetensorsIndex["weight_map"];
        for (JSON::iterator it = jMap.begin(); it != jMap.end(); ++it) {
            std::string key = it.key();
            st_map.insert(std::make_pair(key, nullptr));
        }
    }
    switch (hConfig->ModelArch()) {
        case NLP_QWEN3:
            if (nTotalSize == 8045591552) {     //  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
                hConfig->n_ctx_train = 262144;  //  Context Length: 262,144 natively
            }
            break;
        case NLP_QWEN2:
            break;
        default:
            break;
    }
    if (isInitFromPath) {
        hConfig->ToJConfig();
    }

    return true;  // isInitFromPath;
}