/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  General AutoRegressive Language model
 *
 *  \brief General Language model
 *  \author Yingshi Chen
 */

#include "gLLM.hpp"

#include "Fish.hpp"
#include "Optimizer.hpp"

shared_ptr<EDGE_DEVICES> EDGE_DEVICES::GetInstance(const CLI_params& config, int flag) {
    static shared_ptr<EDGE_DEVICES> hEDS = std::make_shared<EDGE_DEVICES>(config, flag);
    return hEDS;
}

bool NLP_AutoRegressive::Init(const vector<hWIKI>& wikis_, int flag) {
    auto train_params = config.common;
    if (train_params.seed == -1) {
        train_params.seed = time(NULL);
    }
    wikis = wikis_;

    if (hDict == nullptr) {
        if (!InitDictTokenset())  //  hDictVAE
            return false;
    }
    if (config.common.method == "muon")
        hOPT = std::make_shared<OPT_Muon>(this, config, flag);
    else
        hOPT = std::make_shared<OPT_Adam>(this, config, flag);
    if (wikis.size() == 0) {
        _INFO("====== NO WIKI !!! ======\n");  // return false;
    } else {
        teach = wikis[0]->teach;
    }

    hDistler   = nullptr;  // config.sigma=="" ? nullptr : std::make_shared<Distillation>(this,config,0x0);     //ADD SIGMA
    hEDS       = EDGE_DEVICES::GetInstance(config);
    hOPT->hEDS = hEDS;

    InitModel(flag);
    config.Dump();

    if (hOPT != nullptr) {
        // hOPT->Dump(1);
        if (config.isOnlyGPT || !isTrain())
            return true;
        if (!hOPT->PrepareData(config, flag))
            return false;
    }
    return true;
}

NLP_AutoRegressive::NLP_AutoRegressive(const std::string& nam_, struct CLI_params params, ROLE_TYPE role_, int flag) : Fish(nam_, params, role_) {
    if (!config.is({"wiki", "actor"}, "copy")) {
    }
    config.ffn_use_gate = true;
    int d               = config.Get({"model_v0", "attention", "dQKV"}, 4, false);
    isAttOnBC           = d == 3;  // d=4 much faster,nearly same

    string sT = params.KV({"model_v0", "attention", "type"}, "QKV", false);
    tpATT     = sT == "brown" ? ATTENTION_TYPE::BROWN : sT == "off" ? ATTENTION_TYPE::OFF : ATTENTION_TYPE::QKVs;
    tpFFN     = (FFN_TYPE)(jKV(params.jConfig, {"model_v0", "ffn", "type"}, 5, false));
}
// params!=src->params
NLP_AutoRegressive::NLP_AutoRegressive(const std::string& nam_, const NLP_AutoRegressive* src, struct CLI_params params, int flag) : Fish(nam_, params) {
    // hDictVAE = src->hDictVAE;
    // tensors = src->tensors;
    // for(auto it : tensors){
    //     nParamsGGUF += tELEM(it.second);
    // }

    graph_order = src->graph_order;
    // VAE's latent dim
    //  if(hDictVAE->nLevel>0)     {
    //      config.nEmbed() = src->config.nEmbed();
    //      // n_embd = config.nEmbed();
    //  }

    //  ugly ctx here
    // struct ggml_init_params ctx_model_params;
    // ctx_model_params.mem_size   = MostMemSize(0x0) ;
    // ctx_model_params.mem_buffer = NULL;
    // ctx_model_params.no_alloc   = true;
    // assert(ctx==nullptr);
    // ctx = ggml_init(ctx_model_params);

    // InitModel();
}

string NLP_AutoRegressive::__repr__(string& suffix, string& prefix, int flag) {
    // Fish::__repr__(suffix,prefix,flag);
    char buf[50120] = "\0";
    const char* tab = prefix.c_str();

    string s = "\n", p = prefix + "\t";
    int i;
    _T_repr_(KQ_pos, "\n\tKQ_pos: ", buf);
    _T_repr_(pos_embd, "\tpos_embd: ", buf);
    _T_repr_(KQ_mask, "\tKQ_mask: ", buf);
    sprintf(buf + strlen(buf), "%s", hDictVAE->__repr__(s, p, 0x0).c_str());
    bool isJModel = !config.jModel.empty();
    if (isJModel) {
        for (auto nn : neurons) {
            if (nn->isGang())
                continue;
            for (p = prefix, i = 0; i < nn->level - 1; i++) p += "\t";
            // if(nn->level==1)
            sprintf(buf + strlen(buf), "%s\n", nn->__repr__(s, p, 0x0).c_str());
        }
    } else {
        int nLayer = layers.size();
        if (nLayer > 0) {
            auto layer = layers[0];
            sprintf(buf + strlen(buf), "%s  [%s] x %d\n", tab, layer->name.c_str(), nLayer);
            sprintf(buf + strlen(buf), "%s", layer->__repr__(s, p, 0x0).c_str());
            if (nLayer > 1) {
                sprintf(buf + strlen(buf), "%s  ......\n", tab);
                sprintf(buf + strlen(buf), "%s", layers[layers.size() - 1]->__repr__(s, p, 0x0).c_str());
            }
        }
    }
    _T_repr_(target_probs, "  target(ID)=", buf);
    for (auto wiki : wikis) {
        if (wiki->exLogits != nullptr) {
            string a = "   ex_logits@" + wiki->title + "=";
            // _T_repr_(wiki->exLogits,a.c_str(),buf);
        }
        if (wiki->t2t != nullptr) {
            string a = "   t2t@" + wiki->title + "=";
            // _T_repr_(wiki->t2t,a.c_str(),buf);
        }
    }

    if (mom.embed2w != nullptr)
        _T_repr_(mom.embed2w, "  gate=", buf);
    _T_repr_(loss, "  loss=", buf);

    sprintf(buf + strlen(buf), "\tLAY=%d\t%s ", TGraph::curLayer, suffix.c_str());
    if (config.dumpSwitch.nn_structure)
        _INFO("%s", buf);
    return buf;
}

string Ganglia::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012]  = "\0";
    const char* tab = prefix.c_str();
    sprintf(buf + strlen(buf), "%s %s", tab, name.c_str());
    string s, p;
    for (auto child : ns) {
        sprintf(buf + strlen(buf), "%s ", child->__repr__(s, p, 0).c_str());
    }
    // sprintf(buf+strlen(buf),"\n");
    if (flag > 0)
        _INFO("%s", buf);
    return buf;
};

string GeNeuron::__repr__(string& suffix, string& prefix, int flag) {
    char buf[5012] = "\0";
    // const char*tab=prefix.c_str();
    // sprintf(buf+strlen(buf),"\n%s %s",tab,name.c_str());
    // if(flag>0)
    //     _INFO("%s",buf);
    return buf;
}

// void save_llama_model_file(const char * filename, const char * fn_model_base, struct llama_model * model) {
//     _INFO("%s: saving to %s\n", __func__, filename);
//     struct gguf_context * fctx = gguf_init_empty();

//     save_llama_model_gguf(fctx, fn_model_base, model);

//     // write file
//     const bool only_meta = false;
//     gguf_write_to_file(fctx, filename, only_meta);
//     gguf_free(fctx);
// }

/*
    // optionally save the session on first sample (for faster prompt loading next time)
    if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
        need_to_save_session = false;
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

        LOG("saved session to %s\n", path_session.c_str());
    }
*/

void NLP_AutoRegressive::build_inp_KQ_(void* ctx, bool isMask, bool causal) {
#ifdef __USE_GGML__
    char nam_[128];
    bool isFlash       = config.isFlashAtten();
    const uint32_t pad = isFlash ? 256u : 32u, cell_max = 0;  // llama_kv_cache_cell_max(*cache)
    // auto cache = GetKVCache();
    // cache->n = std::min(cache->size, std::max(pad, GGML_PAD(cell_max, pad)));
    int n_kv    = hCache == nullptr ? 512 : hCache->n_kv();
    int n_batch = config.n_batch(), n_ctx = config.n_ctx(), n_tokens = n_batch * n_ctx, n_past = 0;
    // const float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    // KQ_pos - contains the positions
    KQ_pos = GT(this, typNUMBER::I32, {n_ctx});
    gTN(KQ_pos, "inp_pos");
    tFLAG(KQ_pos, GTensor::F_INPUT);  //    ggml_set_input(KQ_pos);
    int* data = (int*)KQ_pos->data;
    // @BuildComputeGraph After ggml_gallocr_alloc_graph(alloc, gb)!
    // for (int i = 0; i < n_tokens; ++i) {
    //     data[i] = n_past + i;
    // }
    if (isMask && 0) {  //  nearly same if mask==nullptr
        auto dt  = typNUMBER::F32;
        auto pad = GGML_PAD(n_ctx, GGML_KQ_MASK_PAD);  //  (((x) + (n) - 1) & ~((n) - 1))
        // KQ_mask = causal
        //     ? GT(this, dt, n_kv,     GGML_PAD(n_tokens, GGML_KQ_MASK_PAD))
        //     : GT(this, dt, n_tokens, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
        KQ_mask = GT(this, dt, {n_ctx, pad});
        gTN(KQ_mask, "KQ_mask");
        tFLAG(KQ_mask, GTensor::F_INPUT);  // ggml_set_input(KQ_mask);        //  flags |= GGML_TENSOR_FLAG_INPUT;
        float* mask = (float*)(KQ_mask->data);
        //  KQ_mask = isFlash ? ggml_cast(ctx, KQ_mask, GGML_TYPE_F16) : KQ_mask;
    }
#endif
    return;
}

hGensor SLP::UpdateGensor(int flag) {
    hGensor h = w == nullptr ? nullptr : hFish->GetGensor(w->name);
    return h;
}

std::string NLP_AutoRegressive::Name() { return "LAMA"; }

// REF: ggml_compute_forward_dup_bytes
void Fish::CopyWeight(const Fish* src, int flag) {
    if (wiki_tutor != nullptr) {
        assert(wiki_tutor == src->wiki_tutor);
        return;
    }
#ifdef _TENSOR_G_
#else
    auto gsrc = src->hForwTG->raw();
    size_t nx = 0, nz, nT = 0, type_size;
    vector<hGensor> tSrc;
    hGensor t1;
    if (isTrain()) {
        for (int i = 0; i < gsrc->n_nodes; ++i) {
            hGensor t0 = gsrc->nodes[i];
            if (strcmp(t0->name, "output.weight") == 0) {
                int j = 0;
            }
            type_size = BPE(t0->type);
            if (t0->flags & GTensor::F_PARAM) {
                t1 = GetGensor(t0->name);
                assert(t1 != nullptr && t0->type == t1->type);
                nz = tELEM(t0);
                assert(nz == tELEM(t1));
                memcpy(t1->data, t0->data, type_size * nz);
                nx += nz;
                nT++;
            }
        }
        assert(nx == src->nParams);
        return;
    } else if (!src->loadGensors.empty()) {
        tSrc = src->loadGensors;
    }
    for (auto t0 : tSrc) {
        t1 = GetGensor(t0->name);
        assert(t1 != nullptr && t0->type == t1->type);
        nz        = tELEM(t0);
        type_size = BPE(t0->type);
        assert(nz == tELEM(t1));
        memcpy(t1->data, t0->data, type_size * nz);
        nx += nz;
        nT++;
    }
#endif
}

/*
    would affect training process?
*/
bool NLP_AutoRegressive::LocalFeeling(hSampLoader hLoader, vector<float>& result, int flag) {
    assert(hOPT != nullptr);
    auto preLogits = hCLS->preLogits;
    assert(hLoader->shard_samps.size() == 1);
    auto hSamp = hLoader->shard_samps[0];
    int i, nTok = hSamp->len, _nctx = config.n_ctx();
    // assert(!hDictVAE->hDict->tokenizer_add_bos);
    SetPhase(LIFE_PHASE::P_EVAL_);
    hOPT->EvaluateSamps(hLoader, -666);
    if (DUMP())
        _INFO("\t%s @\"%s\"\n", __func__, hLoader->sentence.c_str());
    assert(preLogits->type == typNUMBER::F32);
    assert(preLogits->ne[1] == nTok + 1);                           //???
    size_t nz = tELEM(preLogits), nVocabInWiki = preLogits->ne[0];  // preLogits->nb[0];

    // out is last distribution of Causal Language Modeling
    size_t off = (nTok)*nVocabInWiki;
    float* out = (float*)(preLogits->data) + off;
    if (result.size() != nVocabInWiki) {
        result.clear();
        result.resize(nVocabInWiki);
    }
    memcpy(result.data(), out, sizeof(float) * nVocabInWiki);
    float sum = 0;
    for (auto f : result) sum += f;
    return true;
}

size_t NLP_AutoRegressive::tVocab() {
    assert(hDict != nullptr);
    return hDict->nVocab();
}

hGensor Fish::BuildLoss(void* ctx, hGensor cur, int flag) {
    if (DEBUG.NO_loss) {
        assert(cur == hCLS->preLogits);
        out_node = cur;
        return cur;
    }
    int n_ctx = config.n_ctx(), nCls = nClass(), n_batch = config.n_batch();

    // cuLiteTest(B,T,C);

    assert(loss != nullptr);
    out_node = loss;

    return loss;
}

hGensor NLP_AutoRegressive::BuildTarget(void* ctx, hGensor cur, int flag) {
    assert(0);  // Deprecated
    hGensor _tNorm = UpdateGensor(hDictVAE->_norm.w->name);
    int n_vocab = tVocab(), n_batch = config.common.n_batch, n_ctx = config.common.n_ctx, n_embd = config.nEmbed();
    auto train_params              = config.common;
    train_params.use_checkpointing = false;  // CYS_0826
    const int N = train_params.n_ctx, n_past = 0;
    hGensor t32 = nullptr, wA = nullptr, wB = nullptr;
    return out_node;
}

std::string NLP_AutoRegressive::T2STR(const std::vector<TOKEN_ID>& toks, int nMost, int flag) {
    std::string str = "";
    int i           = 0;
    for (auto tok : toks) {
        if (tok == hDict->eos_id)
            break;
        std::string a = hDict->T2STR(tok, flag);
        str += a;
        if (++i >= nMost)
            break;
    }

    return str;
}

bool NLP_AutoRegressive::InitDictTokenset(int flag) {
    if (!Fish::InitDictTokenset(flag))
        return false;

    hDictVAE        = std::make_shared<DictVAE>(this);
    hDictVAE->hDict = hDict;
    assert(hDictVAE != nullptr && hDictVAE->isValid());
    return true;
}
bool Fish::InitDictTokenset(int flag) {
    void* hLLM = nullptr;
    // hDict = std::make_shared<GTokenizer>(this);     //  entence != prompt
    // hDict = std::make_shared<GTokenizer_Heap>(this);

    switch (config.ModelArch()) {
        case MODEL_ARCH::NLP_GPT2:
        case MODEL_ARCH::NLP_GPT2_char:
            if (config.ModelArch() == MODEL_ARCH::NLP_GPT2_char) {
                // hDictVAE = std::make_shared<CDict_CHAR>(this);
                // hDictVAE->LoadVocab("",0x0);
            } else {
                // hDictVAE = std::make_shared<CDict_GPT2>(this);
                hDict = std::make_shared<GTokenizer_GPT2>(this);
                if (wikis.size() > 0) {  // lama()!= nullptr
                    hDict->vocab.resize(wikis[0]->n_vocab);
                    hDict->bos_id = wikis[0]->bos;
                    hDict->eos_id = wikis[0]->eos;
                    // hLLM = wikis[0]->lmodel;
                } else {
                    int n_vocab = 50257;
                    // int n_vocab = CEIL_DIV(50257, 128) * 128;   //  50257 =>  50304
                    hDict->vocab.resize(n_vocab);
                    hDict->bos_id = 1;
                    hDict->eos_id = 2;
                }
            }
            // hTokenset = std::make_shared<DataTokenSet>(hDictVAE.get());
            break;
        case MODEL_ARCH::NLP_GUPPY:
            hDict->vocab.resize(50304);  //  50304       50257   ???
            hDict->bos_id = 1;
            hDict->eos_id = 2;
            break;
        case MODEL_ARCH::NLP_MISTRAL:
            // hDictVAE = std::make_shared<DictVAE>(this);
            hDict->vocab.resize(32000);
            hDict->bos_id = 1;
            hDict->eos_id = 2;
            break;
        case MODEL_ARCH::NLP_DEEPSEEK:
            // hDictVAE = std::make_shared<DictVAE>(this);
            hDict->vocab.resize(102400);
            hDict->bos_id = 100000;
            hDict->eos_id = 100001;
            break;
        case MODEL_ARCH::NLP_QWEN2:
            hDict = std::make_shared<GTokenizer_QWEN3>(this);
            hDict->vocab.resize(151936);
            break;
        case MODEL_ARCH::NLP_QWEN3:
            hDict = std::make_shared<GTokenizer_QWEN3>(this);
            hDict->vocab.resize(151936);
            // hDict->bos_id = 151643, hDict->eos_id = 151645;
            break;
        default:
            // hDictVAE = std::make_shared<DictVAE>(this);
            if (wikis.size() > 0) {
                // hDictVAE->n_vocab = wikis[0]->n_vocab;
                // hDictVAE->bos = wikis[0]->bos;             hDictVAE->eos = wikis[0]->eos;
            }
            // hTokenset = std::make_shared<DataTokenSet>(hDictVAE.get());
            break;
    }
    assert(hDict->nVocab() > 0);

    {
        tokenset = DataTokenSet::MakeInstance(config, hDict, isLocalInfer, 0x0);
        if (tokenset.empty()) {
            _ERROR("\n======== %s Failed to load tokenset!========\n", __func__);
            return false;
        };
    }
    tsTrain = tokenset[0];
    for (int i = 1; i < tokenset.size(); i++) {
        tsEval.push_back(tokenset[i]);
    }

    // hDictVAE->mapT2T = tsTrain->mapT2T;        hDictVAE->dialect = tsTrain->dialect;
    // for(auto wiki : wikis){
    //     wiki->mapT2T = hDictVAE->mapT2T;       wiki->dialect = hDictVAE->dialect;
    // }

    if (isTrain()) {
        if (tsTrain != nullptr && tsTrain->nMostTok > 0)
            config.OnMostToken(tsTrain->nMostTok);
    } else {  //  config.isOnlyGPT
        tsTrain = nullptr;
        tsEval  = tokenset;
        return true;  // may have no train !!!
    }

    return true;
}

bool NLP_AutoRegressive::InitInput(void* ctx_build, bool isMask, int flag) {
    auto train_params = config.common;
    int n_ctx = train_params.n_ctx, n_vocab = tVocab(), n_batch = train_params.n_batch;
    assert(n_ctx > 0 && n_batch > 0);
    SHAPE shape = {n_batch, n_ctx};

    // tokens_input copy values from Batch tensor
    tokens_input = GT(this, typNUMBER::I32, shape, GTensor::F_INPUT, "inp_tokens");  // gTN(tokens_input, "inp_tokens");
    tokens_input->Alloc();
    in_node = tokens_input;

    build_inp_KQ_(ctx_build, isMask);
    return true;
}

bool NLP_AutoRegressive::CreateExlogists(hWIKI wiki, uint32_t n_ctx, uint32_t n_batch, int flag) {
    auto ctx = GetGGCTX();
    if (teach == WIKI::_OFF)
        return false;
    int64_t nV = tVocab();
    assert(wiki->n_vocab >= nV);
    if (!isLocalInfer) {
        assert(wiki->exLogits == nullptr);
#ifndef _TENSOR_G_
        if (wiki->n_vocab > nV) {
            wiki->exLogits = GT(this, typNUMBER::F32, {nV, n_ctx, n_batch});
            if (0) {
                wiki->t2t = GT(this, typNUMBER::F32, {nV, nV});
                sprintf(wiki->t2t->name, "t2t@%s", wiki->title.c_str());
            }
        } else {
            wiki->exLogits = GT(this, typNUMBER::F32, {wiki->n_vocab, n_ctx, n_batch});
        }

        tmpExLogis.push_back(wiki->exLogits);
#else
        size_t nz      = nV * n_ctx * n_batch;
        wiki->exLogits = new float[nz];
#endif
        return true;
    }
    return false;
}

//  NLP_AutoRegressive::Train would call this function
void Fish::UpdateTernary(int flag) {
    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();
    string bit_tensors;
    int nTernary = 0;
    size_t nzT = 0, nzP = 0, nzBit = 0;
    for (auto t : hOPT->opt_ps) {
        nzP += t->size();
        nzBit += t->nByte() * 8.0;
        if (BIT_TEST(t->flags, GTensor::F_TERNARY)) {  // {"ffn_down.weight", "ffn_up.weight"}
            nTernary++;
            nzT += t->size();
            bit_tensors += t->name, bit_tensors += " ";
        }
        hRLS->GetTensorStatus(-1, t, 0x0);

        // t->DumpX(0x0);
    }
    double bpp = nzBit * 1.0 / nzP;  // bit per parameter
    _INFO("\n%s bit_per_parameter=%.4g szGama=%d TILEQ=(%d,%d) pQuant=%s tensor=%d(%.3g%%) \n", bit_tensors.empty() ? "[NO_bitQUANT]" : "[BIT_QUANT]", bpp,
          sizeof(floatGama), THREAD_TILE_M, THREAD_TILE_N, "W_SCALE", nTernary, nzT * 100.0f / nParams);
    if (!bit_tensors.empty()) {
        _INFO("\t@{%s}\n", bit_tensors.c_str());
    }
}

void NLP_AutoRegressive::Train(int flag) {
    // DEBUG.train_datas = 1;
    // DEBUG.train_hyperparams = 1;
    // DEBUG.back_graph_version = 1;        Verified at 05132025

    hOPT->BeforeTrain(tokens_input, 0x0);
    // RLS_BP *hRLS = hEDS->GetScheduler<RLS_BP>();
    // hRLS->Prepare(-1);
    UpdateTernary(flag);
    int64_t now = GST_ms();
    double ms   = 0;
    // print_build_info();

    Optimizer::RESULT result = hOPT->Search(nullptr, loss, target_probs, config);
    assert(result == Optimizer::OK || result == Optimizer::DID_NOT_CONVERGE);
    ms = GST_ms() - now;
    _INFO("\n[train]: ");
    _TIME_INFO("Total time=", ms);
    _INFO("\n\n");
}

#ifdef __USE_GGML__
#include "ggml-impl.h"
#endif
static struct random_normal_distribution* rnd = nullptr;
// TO DO :  refactor
void NLP_AutoRegressive::InitGensors(int flag) {
    /*auto ctx               = GetGGCTX();
    const uint32_t n_layer = layers.size();
    for (u_int i = 0; i < n_layer; i++) {
        auto layer = dynamic_pointer_cast<QKV_LAY>(layers[i]);
        if (!layer->Q.Empty()) {
            InitGensor(ctx, layer->att_norm.w, TN(LLM_TENSOR_ATTN_NORM, i), rnd);
            InitGensor(ctx, layer->Q.w, TN(LLM_TENSOR_ATTN_Q, i), rnd);
            if (layer->K.w != nullptr)
                InitGensor(ctx, layer->K.w, TN(LLM_TENSOR_ATTN_K, i), rnd);
            if (layer->V.w != nullptr)
                InitGensor(ctx, layer->V.w, TN(LLM_TENSOR_ATTN_V, i), rnd);
            // InitGensor(ctx,layer->wo,             TN(LLM_TENSOR_ATTN_OUT, i), rnd);
            InitGensor(ctx, layer->wo, TN("blk.%d.attn_out", i), rnd);
        }
        if (layer->ffn_norm.w != nullptr)
            InitGensor(ctx, layer->ffn_norm.w, TN(LLM_TENSOR_FFN_NORM, i), rnd);
        if (layer->up.w != nullptr) {
            if (layer->ffn_gate != nullptr) {
                InitGensor(ctx, layer->ffn_gate, TN(LLM_TENSOR_FFN_GATE, i), rnd);
            }
            InitGensor(ctx, layer->down.w, TN(LLM_TENSOR_FFN_DOWN, i), rnd);
            InitGensor(ctx, layer->up.w, TN(LLM_TENSOR_FFN_UP, i), rnd);
        }
        if (layer->ffn_gate_inp != nullptr) {
            InitGensor(ctx, layer->ffn_gate_inp, TN(LLM_TENSOR_FFN_GATE_INP, i), rnd);
            InitGensor(ctx, layer->ffn_gate_exps, TN(LLM_TENSOR_FFN_GATE_EXPS, i), rnd);
            InitGensor(ctx, layer->ffn_down_exps, TN(LLM_TENSOR_FFN_DOWN_EXPS, i), rnd);
            InitGensor(ctx, layer->ffn_up_exps, TN(LLM_TENSOR_FFN_UP_EXPS, i), rnd);
            InitGensor(ctx, layer->ffn_gate_inp_shexp, TN(LLM_TENSOR_FFN_GATE_INP_SHEXP, i), rnd);
            InitGensor(ctx, layer->ffn_gate_shexp, TN(LLM_TENSOR_FFN_GATE_SHEXP, i), rnd);
            InitGensor(ctx, layer->ffn_down_shexp, TN(LLM_TENSOR_FFN_DOWN_SHEXP, i), rnd);
            InitGensor(ctx, layer->ffn_up_shexp, TN(LLM_TENSOR_FFN_UP_SHEXP, i), rnd);
        }
    }
    if (mom.embed2w != nullptr) {
        InitGensor(ctx, mom.embed2w, "gate_cys", rnd);
    }
    if (mos.gat_ != nullptr) {
        InitGensor(ctx, mos.gat_, "gate_swarm", rnd);
    }
    for (auto wiki : wikis) {
        if (wiki->t2t != nullptr) {
#ifndef _TENSOR_G_
            InitGensor(ctx, wiki->t2t, nullptr, rnd);
#endif
        }
    }*/
}

void NLP_AutoRegressive::InitModel(int flag) {
    const uint32_t n_ff = config.n_ff();
    auto train_params   = config.common;
    int n_embd = config.nEmbed(), n_ctx = train_params.n_ctx;
    const uint32_t n_layer = config.nLayer();
    bool isJModel          = !config.jModel.empty();
    auto ctx               = GetGGCTX();
    if (arch == NLP_MAMBA) {
        assert(config.n_head() == 0);
    } else {
    }

    _INFO("\n Init model embed=%d layer=%d ff=%d tpFFN=%d\n", n_embd, n_layer, n_ff, tpFFN);
    _INFO("\t type of FFN=%s\n", tpFFN == FFN_TYPE::SWIGLU           ? "MLP"
                                 : tpFFN == FFN_TYPE::VAR_LAST       ? "Variation@last_layer"
                                 : tpFFN == FFN_TYPE::ONLY_RMSNormal ? "RMS Normal"
                                                                     : "other");
    // _INFO("\t type of ATTENTION=%s P=%s \n",tpATT==ATTENTION_TYPE::BROWN ? "BROWN":"QKV",BROWN_Motion::Transfer_1?"Token":"TokenEmbed");
    tmpExLogis.clear();
    if (role == ROLE_TYPE::SWARM_FOLLOWER) {
    } else {
        for (auto wiki : wikis) {
            CreateExlogists(wiki, n_ctx, train_params.n_batch);
        }
        if (role == ROLE_TYPE::SWARM_HEAD) {
            mos.Init(swarm, ctx, n_embd);
        }
    }

    if (tmpExLogis.size() > 0 && !isLocalInfer) {
        // exLogits = GT(this, typNUMBER::F32, n_vocab,  n_ctx, train_params.n_batch);
        if (teach == WIKI::_LOGITS_GATE) {
            mom.embed2w = GT(this, typNUMBER::F32, {n_embd, (int)(tmpExLogis.size()) + 1});
        }
    }
    nParams = 0;
}

void NLP_AutoRegressive::Dump(int type, int flag) {
    if (NOT_DUMP(5))
        return;
    fflush(stdout);
    int n_vocab = hDictVAE->hDict->nVocab(), n_batch = config.common.n_batch, n_ctx = config.common.n_ctx, n_embd = config.nEmbed();
    string suffix = "\n========\n", prefix;
    __repr__(suffix, prefix);
    config.Dump();  //        print_params(&config)
    hFuyou afu = GetFuyou(-1);
    if (afu != nullptr) {
        int nAfuParam = afu->nParams;
        _INFO("====== nParams = %ld(%.6gM nT=%ld) allParams = %ld(%.6gM nT=%ld) ======\n", nAfuParam, nAfuParam / 1.0e6, afu->ckpParams.size(), nParams,
              nParams / 1.0e6, optParams.size());
    } else
        _INFO("====== nParams = %ld(%.6gM nT=%ld) ======\n", nParams, nParams / 1.0e6, optParams.size());
    _INFO("\t nParams=%zu model_size = %zu bytes (%.1f MB)\n", nParams, szModel, szModel / (1024.0f * 1024.0f));
    _INFO("\t n_vocab=%d t_vocab=%d,n_batch=%d,n_ctx=%d,n_embd=%d,n_head=%d,n_rot=%d,n_ff=%d\n", n_vocab, tVocab(), n_batch, n_ctx, n_embd, config.n_head(),
          config.n_rot(), config.n_ff());
    _INFO("\t loader=%s\n", config.tpBatchSample.c_str());
    if (hOPT != nullptr) {
    } else {
        _INFO("hOPT is NULL\n");
    }
    if (config.lars_ratio > 0)
        _INFO("\t LARS(t_max=%g)\n", config.lars_ratio);

    std::sort(optParams.begin(), optParams.end(), [](const hGTensor& a, const hGTensor& b) { return strcmp(a->name,b->name)<0; });
    _INFO("====== Params Table ======\n");
    for (auto t : optParams) {
        t->DumpX(0x0);
    }
    switch (type) {
        case KOIFISH_OUTOF_GPUMEMORY:
            SUM::MemoryInfo(type);
            break;
        default:
            break;
    }
    fflush(stdout);
}

//  @GeNeuron::SetGuoke
bool Fuyou::Backward(hGensor cur, int flag) {
    bool isMix = true;
    for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
        GeNeuron* neuron = (GeNeuron*)((*it)->hOBJ);
        if (neuron->isShortcut)
            continue;
        if (neuron->isPassBack)
            continue;
        auto t0            = GST_ms();
        cur                = neuron->Ming(hRLS, cur);
        neuron->stat.tBack = GST_ms() - t0;
        if (isMix) {
            int debug = 0;
        }
    }
    // SYNC_DEVICE();
    return true;
}

float RAW_backward(Fish* fish, const int* hostInToken, int accum_steps, bool, int flag);
int Fish::BackwardOnRLS(int iter, int flag) {
    GTensor::delta->Zero();
    OutCLS* cls       = GetNeuron<OutCLS>("OutCLS", 0);
    GTensor::buff     = hCLS->preLogits->data;  // reused in many place!
    GTensor::buff_len = hCLS->preLogits->size() * sizeof(floatX);
    /*if (DEBUG.back_graph_version == 1) {
        int nAccum          = config.common.n_gradient_accumulation;
        bool isOnlyEvaluate = false;
        float loss          = RAW_backward(this, nullptr, nAccum, isOnlyEvaluate, flag);
        return 0x0;
    }*/

    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();
    hGensor cur  = cls->delta;
    hFuyou afu   = hRLS->afu;
    afu->Backward(cur);

    //  Head to follower
    if (iter >= config.fuyou.nWarmup() && hRLS->fuyouSwarm.size() > 1) {  //  memory
        if (DEBUG.T_fuyou == 0 || hRLS->isSwitchFuyou(iter + 1)) {
            hRLS->ExploreOptimization(iter);
        }
    }

    // afu update from Head
    /*if (hRLS->fuyouSwarm.size() > 1) {  //  memory
        if (DEBUG.T_fuyou == 0 || hRLS->isSwitchFuyou(iter)) {
            hRLS->ExploreOptimization(iter);
        }
    }*/
    return 0x0;
}

const CheckPoint_Params& Fish::SnapShot(int flag) const {
    if (isLocalInfer) {
        assert(config.ckp_in.size() > 0);
        return config.ckp_in[0];
    }
    return config.state;
}

hFuyou Fish::GetFuyou(int no, int flag) const {
    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();
    if (no == -1) {
        return hRLS->afu;
    }
    assert(no >= 0 && no < hRLS->fuyouSwarm.size());
    hFuyou fuyou = hRLS->fuyouSwarm[no];
    return fuyou;
}

float Fish::Evaluate(DL_BATCH_UPATE tpBatch, int flag) {
    hOptimizer hOPT     = GetOptimizer();
    hSampLoader hLoader = hOPT->val_loaders[0];
    // switch (type) {
    //     case 1:
    //     // TokenEmbed* embed = GetNeuron<TokenEmbed>("TokenEmbed", 0);
    //     // embed->hBatch     = hBatch;
    //         hLoader->num_batches = 1;
    //         break;
    //     default:
    //     break;
    // }
    float eval = hLoader->Evaluate(tpBatch, 0x0);
    return eval;
}

int Fish::ForwardOnRLS(int iter, int flag) {
    // if(DEBUG.back_graph_version==1)
    // { return ForwardOnNeuron_v0(flag);  }
    RLS_BP* hRLS = hEDS->GetScheduler<RLS_BP>();
    double now   = GST_ms();
    if (iter < 0 || isTrain())
        hRLS->Prepare(iter, 0);

    OutCLS* cls = GetNeuron<OutCLS>("OutCLS", 0);
    int L = config.nLayer(), nzLoss = cls->nzLoss, i, nFuyou = hRLS->fuyouSwarm.size();
    float *tmpLoss = nullptr, *loss = cls->hostLoss;
    vector<hFuyou> branches = hRLS->ActiveFuyous();

    /*vector<hFuyou> branches = {hRLS->afu};  // curTasks

    if (isAtPhase(LIFE_PHASE::P_EVAL_) || isAtPhase(LIFE_PHASE::P_GENERATE)) {
        if (config.fuyou.ensemble == Fuyou_params::RANDOM_1 && nFuyou > 1) {  //  random ensembling
            branches      = hRLS->fuyouSwarm;
            uint32_t pick = rand_coin.RandU32() % hRLS->fuyouSwarm.size();
            branches      = {hRLS->fuyouSwarm[pick]};
        } else if (config.fuyou.ensemble == Fuyou_params::AGGREGATION && nFuyou > 1) {
            branches = hRLS->fuyouSwarm;
        } else {
            // branches = {hRLS->fuyouSwarm[pick]};      //only for debug
        }
    }
        hRLS->curFuyous = branches;
        */
    int nB = branches.size(), curB = 0;
    if (nB > 1) {
        tmpLoss = new float[nzLoss]();
    } else {
        assert(nB > 0);
    }

    for (auto branch : branches) {
        hGensor cur = Input(), residual = nullptr;
        switch (phase) {
            case P_GENERATE:
                cur = GTensor::outL;
                break;
            default:
                break;
        }
        assert(!branch->empty());
        // GetNeuron<SelfAttention>("QKV", 0)->ManageMemory(DATA_PLACE::DEV_MEM);  //only for debug
        for (auto task : branch->Tasks()) {
            GeNeuron* neuron = (GeNeuron*)(task->hOBJ);
            if (neuron->name == "model.blk.10.attn" && curB > 0) {  //   model.inp_embd
                DEBUG_HERE;
            }
            if (neuron->isShortcut)
                continue;
            auto t0 = GST_ms();
            neuron->BeforeForward(iter);
            cur                = neuron->Ming(hRLS, cur);
            neuron->stat.tFore = GST_ms() - t0;
            // if (!SYNC_DEVICE(neuron->name, 1))
            //     assert(0);
        }
        curB++;
        if (tmpLoss != nullptr)
            for (i = 0; i < nzLoss; i++) tmpLoss[i] += loss[i];
        // break;
    }
    if (nB > 1) {
        for (i = 0; i < nzLoss; i++) {
            loss[i] = tmpLoss[i] / curB;
            assert(isValidF(loss + i) && loss[i] > 0 && loss[i] < 100.0);
        }
        delete[] tmpLoss;
    }
    // if (!SYNC_DEVICE("ForwardOnRLS", 1))
    //     assert(0);
    return 0x0;
}
int NLP_AutoRegressive::ForwardOnNeuron_v0(int flag) {
    int B, T, C, tpFuseNormal = config.Fuse_Normal, L = config.nLayer();
    GetBTC(B, T, C);
    hGensor cur         = nullptr;
    LayerNormal* lnf    = GetNeuron<LayerNormal>("LayerNormal", 0);
    TokenEmbed* embed   = GetNeuron<TokenEmbed>("TokenEmbed", 0);
    cur                 = embed->OnEmbed(Input(), 0x0);  //  ->Ming(nullptr,Input());
    SelfAttention *QKV0 = GetNeuron<SelfAttention>("SelfAttention", 0), *QKV = nullptr;

    if (tpFuseNormal == 1) {
        cur = QKV0->norm.cuFlow(cur);
    }

    FFN* ffn = nullptr;
    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);
        QKV                = GetNeuron<SelfAttention>("SelfAttention", l);
        ffn                = GetNeuron<FFN>("FFN", l);  // ffn->out = GTensor::delta;
        LayerNormal* hNorm = l + 1 != L ? &(GetNeuron<SelfAttention>("SelfAttention", l + 1)->norm) : lnf;
        ffn->fuseNorm      = tpFuseNormal == 1 ? hNorm : nullptr;
        QKV->fuseNorm      = tpFuseNormal == 1 ? &(ffn->norm) : nullptr;
        cur                = QKV->cuFlow(cur, flag);
        cur                = ffn->cuFlow(cur, 0x0);
        // residual = ffn->out;
    }
    if (tpFuseNormal == 0) {
        cur = lnf->cuFlow(cur);
    }
    OutCLS* cls = GetNeuron<OutCLS>("OutCLS", 0);
    cls->cuFlow(cur, flag);  // embed->w,

    PrintTensor<floatX>("output", ToX(cls->preLogits), true, B, T, C);
    // SYNC_DEVICE();
    return 0x0;
}