/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Random swimming fish that generated from AI
 *  \author Yingshi Chen
 */
#include "Fish.hpp"

#include "../g_stddef.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"

Fish::STAT Fish::stat;
void Fish::STAT::Dump(int typ, int flag) {
    if (NOT_DUMP(0))
        return;
    size_t sz         = 0x0;
    const char *title = "OPT";  //__func__
    switch (typ) {
        default:
            _INFO("\tTIME: qkv=%.3g(s),ffn=%.3g(s)\n", tQKV / 1.0e6, tFFN / 1.0e6);
    }
}

hFISH Fish::MakeInstance(const std::string nam_, struct CLI_params &params, vector<hWIKI> wikis, ROLE_TYPE role_, int flag) {
    assert(wikis.size() >= 0);
    hFISH fish = nullptr;
    switch (params.ModelArch()) {
        case MODEL_ARCH::NLP_MAMBA:
            fish = std::make_shared<LLM_MAMBA>(nam_ + "_mamba", params, role_);
            break;
        case MODEL_ARCH::NLP_DEEPSEEK:
            fish = std::make_shared<DeepSeek>(nam_ + "_DS", params, role_);
            break;
        case MODEL_ARCH::NLP_QWEN2:
            fish = std::make_shared<QWen>(nam_ + "_DS", params, role_);
            break;
        case MODEL_ARCH::NLP_MISTRAL:
            fish = std::make_shared<Mistral>(nam_ + "_mistral", params, role_);
            break;
        case MODEL_ARCH::NLP_GUPPY:
            fish = std::make_shared<Guppy>(nam_ + "_guppy", params, role_);
            break;
        case MODEL_ARCH::NLP_GPT2:
        case MODEL_ARCH::NLP_GPT2_char:
            fish = std::make_shared<GPT2>(nam_ + "_GPT2", params, role_);
            break;
        case MODEL_ARCH::NLP_MOE:
            fish = std::make_shared<LLM_MOE>(nam_ + "_moe", params, role_);
            break;
        case NLP_LLAMA:
            fish = std::make_shared<NLP_AutoRegressive>(nam_, params, role_);
            break;
        default:  //  more structures
            switch (params.nabla) {
                case 1:
                    // fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params,role_);
                    break;
                case 2:
                    // fish = std::make_shared<LLAMA_Brown>(nam_+"_brown",params);
                    fish = std::make_shared<NLP_AutoRegressive>(nam_, params, role_);
                    break;
                case 3:
                    fish = std::make_shared<LLAMA_VAE>(nam_ + "_vae", params, role_);
                    break;
                default:
                    assert(0);
            }
    }

    fish->isLocalInfer = flag == 0x110;
    if (!fish->Init(wikis))
        return nullptr;
    if (!fish->Build())
        return nullptr;
    if (fish->role == SWARM_FOLLOWER) {
    } else {
        if (!wikis.empty()) {  // generate some sentence
            if (params.common.gpt_every > 0 && !fish->isLocalInfer)
                fish->gopt = GeneratOnPrompt::MakeInstance(params, wikis, fish.get(), flag);
        }
    }
    // fish->Dump(0x0);
    return fish;
}

Fish::Fish(const std::string &nam_, struct CLI_params params, ROLE_TYPE role_, int flag) : name(nam_), config(params), role(role_) {
    arch = params.ModelArch();

    string w    = config.KV({"model", "parameter", "debug_init_weight"});  // hack parameter only for debug
    bool isLoad = !config.checkpoint.in.empty() || !config.model.empty();
    if (role == SWARM_FOLLOWER) {
        tpInitWeight = INIT_WEIGHT::COPY_SWARM_HEAD;
    } else {
        tpInitWeight = w == "copy_wiki" ? INIT_WEIGHT::COPY_WIKI : isLoad ? INIT_WEIGHT::SERIALIZE : INIT_WEIGHT::RANDOM;
    }
    rand_coin.Init(42);

    if (jKEY(params.jConfig, {"train"}).empty()) {
        isLocalInfer = true;
    }
}

bool Fish::isModel(std::vector<MODEL_ARCH> arcs, int flag) {
    MODEL_ARCH arc = config.ModelArch();
    for (auto arc0 : arcs) {
        if (arc == arc0)
            return true;
    }
    return false;
}
bool Fish::isTemporaryMemory(GeNeuron *neuron, int flag) const {
    assert(hEDS != nullptr);
    return !hEDS->hRLS->isResident(neuron);
}
bool Fish::isRemater(int flag) const {
    assert(hEDS != nullptr);
    if (hEDS->hRLS->isRemater)
        return true;
    return false;
}

bool Fish::isAtPhase(LIFE_PHASE ph) const { return GetOptimizer()->phase == ph; }

hFISH Fish::MakeSwarm(const std::string nam_, struct CLI_params &params, int flag) {
    vector<hWIKI> wikis = WIKI::MakeInstance(nam_, params, 0x0);
    // if(params.tpWiki!="off") {//wiki is so heavy(ugly) that only load one instance here!
    //     for(auto path : params.fn_model_base){
    //         hWIKI wiki = std::make_shared<LAMA>(params,path);
    //         wikis.push_back(wiki);
    //     }
    // }
    assert(wikis.size() >= 0);

    int nSwarm = params.n_swarm, i;
    for (i = 0; i < nSwarm; i++) {
        ROLE_TYPE role = i == nSwarm - 1 ? SWARM_HEAD : SWARM_FOLLOWER;
        string title   = role == SWARM_HEAD ? "Head_" : "Follower_";
        if (1 || role == SWARM_HEAD) {
            hFISH fish = Fish::MakeInstance(title, params, wikis, role, 0x0);
            Fish::swarm.push_back(fish);
        }
    }
    hPangpi salp = std::make_shared<Pangpi>(nam_, params);
    return salp;
}

hFISH Fish::MakeInstance(const std::string nam_, struct CLI_params &params, const Fish *hSrc_, int flag) {
    hFISH fish     = nullptr;
    ROLE_TYPE role = ROLE_TYPE::COMMON;
    switch (params.nabla) {
        case 1:
            // fish = std::make_shared<LLAMA_LORA>(nam_+"_lora",params,role);
            break;
        case 2:
            fish = std::make_shared<NLP_AutoRegressive>(nam_, params, role);
            break;
        case 3:
            fish = std::make_shared<LLAMA_VAE>(nam_ + "_vae", params, role);
            break;
        default:
            assert(0);
    }
    fish->isLocalInfer = flag == 0x110;
    fish->graph_order  = hSrc_->graph_order;
    // wiki is so heavy(ugly) that only one instance from hSrc!
    fish->Init(hSrc_->wikis);
    fish->Build();

    if (fish->isTrain()) {
        fish->gopt = GeneratOnPrompt::MakeInstance(params, fish->wikis, fish.get(), flag);
    } else {
    }

    return fish;
}
size_t Fish::MostMemSize(int typ) {
    size_t head_dim = config.n_embd_head();
    /*size_t kvbw = config.n_head_kv() * head_dim * kv_len * sizeof(KVT) + p->n_heads * kv_len * sizeof(float);

    uint64_t bw = 0;
    bw += p->head_dim * (p->n_heads + p->n_kv_heads * 2) * dim * dbits / 8; // QKV
    bw += kvbw * 2; // attn scoring and mixing
    bw += p->head_dim * p->n_heads * dim * dbits / 8; // attn output
    bw += 3 * (hidden_dim * dim * dbits / 8) * max(p->n_experts_ac, 1); // MLP
    bw *= p->n_layers;*/
    size_t bw = 1310227072;
    return bw;
}

bool Fish::UpdateParams(int flag) {
    size_t nx = 0;
    assert(optParams.size() == 0);
    for (auto it : gensors.infos) {
        auto t = it.first;
        if (BIT_TEST(t->flags, GTensor::GTensor::F_PARAM)) {
            if (t->isRefer())
                continue;
            optParams.push_back(t);
            nx += tELEM(t);  //
        }
    }
    nParams = nx;
    assert(optParams.size() < 20480);
    if (nx != nParams) {
        CHECK_SAME_TENSORS("Compare parameter tensors\t", optParams, xGensors);
        _ERROR("%s nx(%ld)!=nParams(%ld)\t", __func__, nx, nParams);
    }
    if (nParams == 0)
        exit(KOIFISH_ZERO_PARAMETERS);

    return true;
}

bool Fish::AfterBuild(bool isInitParam, int flag) {
    size_t n0 = 0, nInput = 0, i;
    if (isInitParam) {
        // assert(rnd!=nullptr);
    }
    _INFO("\n\n");
    // assert(optParams.size() == 0);
    for (auto it : gensors.infos) {
        auto t = it.first;
        if (BIT_TEST(t->flags, GTensor::GTensor::F_PARAM)) {
            if (t->isRefer())
                continue;
            // optParams.push_back(t);
            // nx += tELEM(t);
            n0++;                                                //
            if (G_Has_(t->name, config.datatypes.arrTernary)) {  // {"ffn_down.weight", "ffn_up.weight"}
                t->SetTernary(typNUMBER::T_BINARY_3);
            }
            if (G_Has_(t->name, config.datatypes.arrTile)) {  // {"ffn_down.weight", "ffn_up.weight"}
                t->SetTernary(typNUMBER::T_BINARY_TILE);
            }
        }
        if (BIT_TEST(t->flags, GTensor::F_INPUT)) {
            nInput++;
        }
    }
    // nParams = nx;
    // assert(optParams.size() < 20480);
    // if (nx != nParams) {
    //     CHECK_SAME_TENSORS("Compare parameter tensors\t", optParams, xGensors);
    //     _ERROR("%s nx(%ld)!=nParams(%ld)\t", __func__, nx, nParams);
    // }

    if (isTrain()) {
    } else {
        hOPT->SetPhase(LIFE_PHASE::P_EVAL_);
        assert(hBackTG == nullptr);
    }
    RLS_BP *hRLS = hEDS->GetScheduler<RLS_BP>();
    hRLS->Prepare(-1);  // Memory management
    for (auto t : hOPT->opt_ps) {
        hRLS->GetTensorStatus(-1, t, 0x0);
    }
    UpdateParams();

    if (tpInitWeight == SERIALIZE) {
        if (!LoadCheckPoint(flag))
            return false;
    }

    if (!config.only_write_model && hOPT != nullptr) {
        hOPT->Prepare(nParams);
    }
    hOPT->AfterBuild();
    // hOPT->Dump(1);
    if (role == SWARM_FOLLOWER) {
    } else {
    }

    if (!ComputePlan(flag)) {
        return false;
    }
#ifdef _TENSOR_G_

#else
    // ugly code!
    int *data = (int *)KQ_pos->data;
    for (int i = 0; i < config.n_ctx(); ++i) {
        data[i] = 0 + i;  // n_past=0
    }
#endif
    return true;
}

void Fish::Clear() {
    // AllocBuffer @Fish::jToGraph
    GTensor::FreeBuffer();
}

void Fish::ClearGraph(int flag) {
    hForwTG.reset();
    hBackTG.reset();
    neurons.clear();
    gensors.Clear();
    in_node = nullptr, out_node = nullptr;
    loss = nullptr, target_probs = nullptr, KQ_pos = nullptr, KQ_mask = nullptr, pos_embd = nullptr;

    xn = nullptr, xxn = nullptr;
    optParams.clear();
    xGensors.clear();

    childs.clear();
    tmpExLogis.clear();
    // for (ggml_backend_buffer_t buf : bufs) {
    //     ggml_backend_buffer_free(buf);
    // }
    return;
}
bool Fish::UpdateNCTX(int _nctx, int flag) {
    int ctx0 = config.n_ctx();
    if (ctx0 == _nctx)
        return true;
    name         = "4GPT_" + std::to_string(_nctx);
    graph_update = _nctx;
    _INFO("\n\n[UpdateNCTX] %d=>%d @%s\n", ctx0, _nctx, name.c_str());
    ClearGraph();
    config.SetNCTX(_nctx);
    if (!Build())
        return false;

    return true;
}

bool Fish::BeforeBuild(int flag) {
#ifdef __USE_GGML__
    assert(ctx_build == nullptr);
    ctx_size  = MostMemSize(0x0);
    ctx_build = InitCTX(ctx_size);
#endif
    if (role == SWARM_HEAD) {
        assert(swarm.size() > 0);
    } else {
    }
    return true;
}

bool Fish::Build(int flag) {
    if (!BeforeBuild())
        return false;
    int iRet         = 0x0;
    bool isInitParam = false, isJModel = !config.jModel.empty();
    assert(isJModel);
    isSymbolicAnalysis = true;

    /*if(config.ModelArch()==MODEL_ARCH::NLP_GPT2 || config.ModelArch()==MODEL_ARCH::NLP_GPT2_char){
        isInitParam = true;
        iRet = BuildGraphFromRaw(0x0);
    }else*/
    {
        InitInput(ctx_build, true, flag);
        //  isInitParam = true;     // would init param online, not here
        hForwTG = std::make_shared<TGraph>(this, "J_model", ctx_build, true);
        jToGraph(ctx_build, false, flag);
        assert(hCLS != nullptr);
        BuildLoss(ctx_build, hCLS->preLogits);
        iRet = BuildComputeGraph(0, ctx_build, 0x0);
    }

    assert(iRet == 0x0);
    Statistic(0x0);
    if (!AfterBuild(isInitParam))
        return false;

    Dump(0x0);
    isSymbolicAnalysis = false;
    return true;
}

static const char *vendor = "gruai";  // llm_arch_from_string
bool Fish::SaveTrain(string sX, int flag) {
    assert(hOPT != nullptr);
    if (config.checkpoint.out.empty()) {
        return false;
    }
    int iter = hOPT->iter;  //     train->opt->iter;
    // _INFO("%s: iter_%ld\n", __func__, iter);
    string sit       = "IT", sOut;
    string sBaseName = config.checkpoint.model_out;  // get_train_filename(.c_str(),sit.c_str(), "", -1  );
    bool isOK        = false;
    if (!sX.empty()) {
        sOut = config.checkpoint.out + sX + ".ck";  //+ std::to_string(iter)
    } else
        sOut = config.checkpoint.out + "latest" + ".ck";
    VERIFY_DIR_EXIST(sOut, true);

    if (!config.scheduling.canSave(iter, flag)) {
        return false;
    }
    for (auto t : optParams) {
        if (!t->isUpdateParam()) {
            return false;
        }
    }

    isOK = SAFETENSOR_Serialize(sOut, true);
    assert(isOK);
    if (isOK && sX == "warmup") {  // only for debug
        for (int i = 0; i < 1; i++) {
            isOK = SAFETENSOR_Serialize(sOut, false);
            assert(isOK);
            isOK = SAFETENSOR_Serialize(sOut, true);
            assert(isOK);
        }
    }
    // _INFO("[SAVE] @%s iter=%d\n", sX.c_str(), iter);

    return isOK;
}

bool Fish::LoadCheckPoint(int flag) {
    assert(tpInitWeight == INIT_WEIGHT::SERIALIZE);
    std::string fpCheck = config.checkpoint.in;
    if (!config.model.empty()) {
        isLoadCheckpoint = HF_Serialize(false, 0x0);
    } else {
        string type = FILE_EXT(fpCheck);
        bool isCopy = config.is({"wiki", "actor"}, "copy") && wikis.size() > 0;
        if (fpCheck.empty()) {
            // if(wiki_tutor!=nullptr)
            //     return true;
            return true;
        }

        _INFO("[CHECKPOINT]: load \"%s\", type=\"%s\" ......", fpCheck.c_str(), type.c_str());
        if (type == "fish") {
            isLoadCheckpoint = SAFETENSOR_Serialize(config.checkpoint.in, false, 0x0);
        } else if (type == "calm") {
            isLoadCheckpoint = CALM_Serialize(config.checkpoint.in, false, 0x0);
            // isLoadCheckpoint = YALM_Serialize(config.checkpoint.in,false,0x0);
        } else {
            // just try, may fail!
            isLoadCheckpoint = SAFETENSOR_Serialize(config.checkpoint.in, false, 0x0);
            //  Deprecated - Since koifish support 1-bit parameters, why need GGUF?
            // isLoadCheckpoint = GGUF_Serialize(config.checkpoint.in,false,0x0);
        }
    }

    if (!isLoadCheckpoint) {
        _INFO("\r[LoadCheckPoint] failed!  please check file @\"%s\"!\n", fpCheck.c_str());
        return false;
    }

    assert(vendor == "gruai");
    _INFO("\r[LoadCheckPoint] OK @\"%s\"\n", fpCheck.c_str());
    return true;
}
void Fish::Statistic(int typ, int flag) {
    string suffix = "", prefix = "\t";
    struct ggml_cgraph *gf = nullptr, *gb = nullptr;
#ifdef _TENSOR_G_
#else
    gf = hForwTG->raw(), gb = hBackTG == nullptr ? nullptr : hBackTG->raw();
#endif
    if (config.is({"gpt", "c_graph"}, string("raw"))) {
        _INFO("raw graph\n");
    }
    int vQKV = config.Get({"model_v0", "attention", "version"}, 0, false);
    // _INFO("QKV version=%d\n",vQKV);

    // ggml_graph_stat(gf);
    // if(gb!=nullptr) ggml_graph_stat(gb);
    bool isDot = false;
    if (isDot) {
        // ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
        // if(gb!=nullptr) ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
    } else {
        // if(preLogits!=nullptr)
        // hForwTG->__repr__(suffix,prefix);   //preLogits = gf->nodes[gf->n_nodes - 2];
        if (gb != nullptr) {
            // hBackTG->__repr__(suffix,prefix);   //// TGraph("Backward",gb,true)
        }
    }

    int nT = gensors.size(), nQ = 0, nF16 = 0;
    for (auto t : gensors.nag) {
        auto type = t.second->type;
        if (isQuantized(type))
            nQ++;
        if (type == typNUMBER::F16)
            nF16++;
    }
    //  _INFO("%s cgraph(%d,%d) nQ=%d nF16=%d",__func__,cgraph->n_leafs,cgraph->n_nodes,nQ,nF16);
}

int Fish::BuildGraphFromRaw(int flag) {
#ifdef __USE_GGML__
    int iRet                    = 0x0;
    bool isKeep                 = true;
    ctx_compute_params.mem_size = MostMemSize(0x0);
    // 2*LLAMA_TRAIN_MAX_NODES*ggml_tensor_overhead() +
    //         (config.common.use_checkpointing ? 3 : 2)*(GGML_OBJECT_SIZE+ggml_graph_overhead_custom(LLAMA_TRAIN_MAX_NODES, true));
    ctx_build = ggml_init(ctx_compute_params);

    struct ggml_cgraph *gf = BuildRawGraph(ctx_build, false), *gb = nullptr;
    // preLogits = gf->nodes[gf->n_nodes - 1]; // the output is always the last tensor in the graph

    // alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    iRet = BuildComputeGraph(0, ctx_build, 0x0);
    return iRet;
#else
    return -1;
#endif
}

// If isParam, only alloc grad, no init!
void Fish::InitGensor(void *ctx, const string &name, hGensor gensor, bool isParam, int flag) {
    assert(gensor != nullptr);
    if (!name.empty()) {
        gTN0(gensor, name.c_str());  //    gTN0(w,"%s.w",name.c_str());
    }

    if (isParam /*&& isTrain()*/) {
        gensor->SetFlag(GTensor::GTensor::F_PARAM);
        // gTN(gensor, "");        //  ?
        xGensors.push_back(gensor);
    }
    // if(strcmp(gensor->name,"output.bias")==0) {   //only for debug
    //     // xn= gensor;     xxn = gensor->grad;
    // }
}

void Fish::InitGensor(void *ctx, hGensor gensor, const char *name, struct random_normal_distribution *rnd, int flag) {
    assert(0);  // Deprecated
}

hGensor Fish::AddTensor(void *ctx, const std::string &key_, typNUMBER tp, const SHAPE &shape, bool isParam, int flag) {
    CHECK_SHAPE(shape);
    hGensor gensor = nullptr;
    if (shape.size() == 4) {
        gensor = GT(this, tp, shape, 0x0);
    } else if (shape.size() == 2) {
        gensor = GT(this, tp, shape, 0x0);
    } else if (shape.size() == 1) {
        gensor = GT(this, tp, shape, 0x0);
    } else {
        assert(0);
    }
    InitGensor(ctx, key_.c_str(), gensor, isParam, 0x0);

    return gensor;
}

/*

bool TGraph::SchedulerOnNeurons(int flag)    {
    for(auto n : hFish->neurons){
        n->SetDevice(0x0);  // cpu,gpu or other device?
    }
    // trial
    for(auto task : nodes){

    }
    return true;
}
*/

bool Fish::ComputePlan(int flag) {
#ifdef __USE_GGML__
    assert(0);
    auto &train_params         = config.common;
    struct ggml_cgraph *cgraph = GetBackRaw();
    if (cgraph == nullptr) {  //  OnlyInfer
        cgraph = hForwTG->raw();
    }
    gb_plan              = ggml_graph_plan(cgraph, train_params.n_threads, nullptr);
    size_t max_work_size = gb_plan.work_size + GGML_OBJECT_MAX_SIZE;
    _INFO("%s: work_size = %zu bytes (%.1f MB)\n", __func__, max_work_size, (float)max_work_size / (1024.0f * 1024.0f));
    // ggml_free(ctx_build);         ctx_build = nullptr;
    ctx_work                = ggml_init({max_work_size, NULL, false});
    struct ggml_object *obj = ggml_new_object(ctx_work, GGML_OBJECT_TYPE_WORK_BUFFER, gb_plan.work_size);
    // gb_plan.work_data = (uint8_t *)ggml_get_mem_buffer(ctx_work)+ obj->offs;
    gf_plan = gb_plan;  //  ???
    return true;
#else
    return true;
#endif
}

/**/
bool Fish::CopyGensors(hWIKI wiki, int flag) {
    _INFO("CopyGensors of %s ......hFish=%s", wiki->model_path.c_str(), name.c_str());
    int nT0 = wiki->tmaps.size(), nT1 = optParams.size(), nT2 = gensors.size(), x;
    size_t sz = 0;

    if (nT0 > nT2) {
        return false;
    }
    for (auto it : wiki->tmaps) {
        auto nam    = it.first;
        hGensor dst = GetGensor(nam.c_str()), src = nullptr;
#ifndef _TENSOR_G_
        src = it.second;
#endif
        size_t nElem = tELEM(src), nbyte = tBYTE(src);
        sz += nElem;
        if (strcmp(src->name, "blk.0.attn_q.weight") == 0) {  // only for debug
            x = 0;
        }
        if (dst == nullptr)
            return false;
        if (tELEM(src) != tELEM(dst))  // if(!ggml_are_same_shape(src,dst))
            return false;
        float *arr = (float *)(dst->data), a = arr[0];
        // _INFO("\t copy %s nbyte=%ld...\n",nam.c_str(),nbyte);
        // should replace by ggml_compute_forward_dup (memcpy only support CPU!)
        if (src->type == dst->type) {
            memcpy(dst->data, src->data, nbyte);
        } else if (dst->type == typNUMBER::F32) {
            assert(isQuantized(src->type));
            Gensor2float_(src, (float *)(dst->data), flag);
        } else {
            assert(0);
        }
    }
    wiki_tutor = wiki.get();
    _INFO("\rCopyGensors of \"%s\" succeed!    N=%d sz=%.7gM \n", wiki->model_path.c_str(), nT0, sz / 1.0e6);
    return true;
}

bool Fish::BeforeNextStep(int iter, int flag) {
    int nLayer = config.nLayer(), l;

    for (auto t : optParams) {
        // t->tile_r0 = t->tile_r1,        t->tile_c0 = t->tile_c1;
        t->tile_r1 = rand_coin.RandU32() % THREAD_TILE_M - THREAD_TILE_M / 2;
        t->tile_c1 = rand_coin.RandU32() % THREAD_TILE_N - THREAD_TILE_N / 2;
    }

    for (l = 0; l < nLayer; l++) {
        FFN *ffn           = GetNeuron<FFN>("FFN", l);
        SelfAttention *QKV = GetNeuron<SelfAttention>("QKV", l);

        bool isPass = rand_coin.NextCoin();
        // ffn->isShortcut = isPass;       // converge too slow
        // QKV->isShortcut = isPass;
    }
    return true;
}

int Fish::GetCurIter(int flag) const {
    if (hOPT == nullptr)
        return -1;
    return hOPT->GetITER();
}

bool Fish::AfterNextStep(int iter, int flag) {
    int nLayer = config.nLayer(), l;
    for (l = 0; l < nLayer; l++) {
        FFN *ffn           = GetNeuron<FFN>("FFN", l);
        SelfAttention *QKV = GetNeuron<SelfAttention>("QKV", l);
        SUM::tFFN += ffn->stat.tFore;
        SUM::tFFN += ffn->stat.tBack;
        SUM::tQKV += QKV->stat.tFore;
        SUM::tQKV += QKV->stat.tBack;
    }
    return true;
}