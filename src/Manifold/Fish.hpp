/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some key characteristics & design goals
 *      1.  Flex - it may have different structures/parameters at different phase!
 *      2.  Sparse
 *      3.  Lite - need much less resource than any other
 *      4.  Tough
 *
 *  \brief Fish - Life is just a random swimming fish.
 *  \author Yingshi Chen
 */

#pragma once

#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <threads.h>

#include <atomic>
#include <cassert>
#include <complex>
#include <map>
#include <memory>
#include <regex>
#include <stack>
#include <typeinfo>
#include <vector>

#include "../Device/EDevice.hpp"
#include "../Utils/GST_rander.hpp"
#include "../Utils/GST_util.hpp"
#include "../ggex/GG_util.hpp"
#include "GoPT.hpp"
#include "Neuron.hpp"
#include "SLP.hpp"
#include "TGraph.hpp"
// #include "../Utils/safetensors.hh"
#include "../Fuzi/Distillation.hpp"

using namespace std;

class Fish;
class Optimizer;
typedef shared_ptr<Optimizer> hOptimizer;
typedef shared_ptr<Fish> hFISH;
typedef vector<hFISH> tpSWARM;
class NLP_AutoRegressive;

struct MixOfModels {
    bool isRes = true, isSiLU = false;
    vector<hGensor> exs;
    int nTiB;  // number of tokens in batch
    // hGensor gate = nullptr;
    hGensor embed2w = nullptr;
    hGensor gat_    = nullptr;

    virtual hGensor Build(CLI_params &config, void *ctx, hGensor cur, int flag = 0x0) { return nullptr; }
    hGensor Forward(void *ctx, hGensor cur, hGensor w);
};

struct MixOfSwarm : public MixOfModels {
    virtual void Init(tpSWARM &swarm, void *ctx, int n_embd, int flag = 0x0);
    hGensor Build(CLI_params &config, void *ctx, hGensor cur, int flag = 0x0) override;
};

class Fish :public std::enable_shared_from_this<Fish> {
    Fish(const Fish &);
    Fish &operator=(const Fish &);

    struct JConfig {
        int ID = -1;
        JSON js;
        JConfig(const JSON &j, int id = -1) : js(j), ID(id) {}
    };

   protected:
    INIT_WEIGHT tpInitWeight = INIT_WEIGHT::RANDOM;

    std::string name;

    /*  wiki contains knowledge reflect the founation of our world
        wikis[0] is the backbone of FISH
    */
    vector<hWIKI> wikis;

    WIKI *wiki_tutor = nullptr;
    bool CopyGensors(hWIKI wiki, int flag);
    vector<hGensor> tmpExLogis;
    WIKI::INDUCT_MODE teach = WIKI::_LOGITS;

    // Generate some results on prompt
    hGOPT gopt = nullptr;

    // ggml_gallocr_t alloc;
    hTGraph hForwTG = nullptr, hBackTG = nullptr;
    int graph_order = -1, graph_update = -1;
    std::vector<hGensor> checkpoints;
    bool measure_only = false;
    void *ctx_build   = nullptr;  // user context of build graph

    // safetensors::safetensors_t safeTensors;
#ifdef __USE_GGML__
    struct ggml_cgraph *gb_tmp             = NULL;
    struct random_normal_distribution *rnd = nullptr;
    struct ggml_cplan gf_plan, gb_plan;
    struct ggml_context *ctx_work = nullptr;  // training ctx
    virtual bool BuildOperators(void *ctx, ggml_gallocr_t alloc, bool m_only, int flag = 0x0) {
        assert(0);
        return false;
    }
    struct ggml_init_params ctx_compute_params = {
        0,
        NULL,
        true,
    };

    virtual void Build(void *ctx0, ggml_gallocr_t &allocr, bool isOnlySymbol, int flag = 0x0) {
        assert(0);  //  Deprecated
    }
    virtual struct ggml_cgraph *BuildRawGraph(void *, bool isBuild, int flag = 0x0) { return nullptr; }
    virtual struct ggml_cgraph *GetForwRaw(int flag = 0x0) const {
        assert(hForwTG != nullptr);
        return hForwTG->raw();
    }
    virtual struct ggml_cgraph *GetBackRaw(int flag = 0x0) const {
        if (hBackTG == nullptr) {
            assert(isLocalInfer);
            return nullptr;
        }
        return hBackTG->raw();
    }
#endif
    std::vector<hNeuron> neurons, backbons;
    // @TGraph::TopoOrder
    GENSORS gensors;
    //  paramter tensors updated by hOPT    @Fish::AfterBuild
    vector<hGensor> optParams;
    vector<hGensor> loadGensors;
    std::vector<hGensor> xGensors;

    hEDevices hEDS = nullptr;

    bool updateTMap = false;
    // Ref@isTrain      No training process! only Evaluate/GPT/...
    bool isLocalInfer     = false;
    bool isLoadCheckpoint = false;
    // bool isBias()   const  {   return config.model.isBias; }
    bool isSymbolicAnalysis = false;

    int size       = 0;
    size_t nParams = 0, szModel = 0;

    hGensor in_node = nullptr, out_node = nullptr;  // maybe GPU tensor
    hGensor loss = nullptr, target_mask = nullptr, target_probs = nullptr, KQ_pos = nullptr, pos_embd = nullptr;
    hGensor KQ_mask    = nullptr;  //  mask for 1 head, it will be broadcasted to all heads
    TokenEmbed *hEmbed = nullptr;
    OutCLS *hCLS       = nullptr;  //  GetNeuron<OutCLS>("OutCLS",0);

    // hGensor gate=nullptr;      //create@InitModel update@
    MixOfModels mom;
    MixOfSwarm mos;
    // Fish can talk, at least it would bubble...
    hTokenizer hDict = nullptr;
    virtual bool InitDictTokenset(int flag = 0x0);
    DataTokens tokenset;
    hDataToken tsTrain = nullptr;  //  always only 1 train set!
    DataTokens tsEval;             //  support multiple eval set!

    hOptimizer hOPT;

    hDistillation hDistler;
    // performance
    int perf_runs       = 0;
    int64_t perf_cycles = 0, perf_time_us = 0;
    size_t ctx_size = 0;

    std::vector<hFISH> childs;

    // Only delete graph/neurons, keep OPT,DataLoader...
    virtual void ClearGraph(int flag = 0x0);
    virtual void Clear();

    virtual bool InitInput(void *ctx, bool isMask, int flag = 0x0) {
        assert(0);
        return false;
    }

    std::vector<std::string> to_quant, to_skip;

    virtual bool GGUF_Serialize(const std::string &path, bool isSave, int flag = 0x0);
    // Load model wight from hugging face model
    virtual bool HF_Serialize(bool isSave, int flag = 0x0);
    // Smart format of https://github.com/zeux/calm
    virtual bool CALM_Serialize(const std::string &path, bool isSave, int flag = 0x0);
    virtual bool YALM_Serialize(const std::string &path, bool isSave, int flag = 0x0);
    virtual bool SAFETENSOR_Serialize(const std::string &path, bool isSave, int flag = 0x0);

    MODEL_ARCH arch = MODEL_ARCH::_X_;
    Grusoft::GRander rand_coin;

   public:
    hGensor xn = nullptr, xxn = nullptr;  // only for debug
    struct STAT {
        double tFFN = 0, tQKV = 0;
        virtual void Dump(int typ, int flag = 0x0);
    };
    static STAT stat;

    struct CLI_params config;
    static tpSWARM swarm;

    enum ROLE_TYPE {
        COMMON,
        SWARM_HEAD,
        SWARM_FOLLOWER,
    };
    ROLE_TYPE role = ROLE_TYPE::COMMON;

    Fish() {}
    Fish(const std::string &nam_, struct CLI_params params, ROLE_TYPE role_ = COMMON, int flag = 0x0);
    Fish(const std::string &nam_, void *ctx_, int flag = 0x0) : name(nam_) /*,ctx(ctx_)*/ {
        assert(0);  // Deprecated
        _INFO("=== %s ===\n", __func__);
        // allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    }
    std::shared_ptr<Fish> SharedThis() { return shared_from_this(); }
    virtual bool isModel(std::vector<MODEL_ARCH> arcs, int flag = 0x0);
    virtual bool isRemater(int flag = 0x0) const;
    virtual bool isTemporaryMemory(GeNeuron *neuron, int flag = 0x0) const;

    bool isTrain() const { return !isLocalInfer; }
    bool isSymbolic() const { return isSymbolicAnalysis; }
    bool isAtPhase(LIFE_PHASE ph) const;
    bool hasWiki() { return wikis.size() > 0; }

    template <typename T>
    T *GetNeuron(const string &desc, int no = 0, int flag = 0x0) {
        int k = 0;
        for (auto n : neurons) {
            T *t = dynamic_cast<T *>(n.get());
            if (t == nullptr) {
                continue;
            }
            if (k++ == no) {
                return t;
            }
        }
        assert(0);
        return nullptr;
    }
    hOptimizer GetOptimizer() const {
        assert(hOPT != nullptr);
        return hOPT;
    }
    int GetCurIter(int flag = 0x0) const;
    // if type==1 return curBraches, otherwise, return allBraches
    int nBranch(int type) { return (GetScheduler<RLSchedule>())->nBranch(type); }
    template <typename T>
    T *GetScheduler() {
        T *hS = hEDS->GetScheduler<T>();
        return hS;
    }

    virtual int ForwardOnRLS(int iter, int flag);
    virtual int BackwardOnRLS(int iter, int flag);
    virtual ~Fish() { Clear(); }
    virtual std::string Name() { return name.c_str(); }
    virtual size_t MostMemSize(int typ = 0x0);
    virtual size_t Size(int flag = 0x0) { return ctx_size; }
    virtual size_t nMostNodes() {
        size_t n = std::max(size_t(8192), gensors.size() * 5);
        return n;
    }
    //  number of class (only valid for classification problem)
    virtual size_t nClass() {
        assert(0);
        return 0;
    }

    virtual bool Init(const vector<hWIKI> &wikis, int flag = 0x0) { throw "Fish::Init is ..."; }
    // shortcut parameter of LLM models
    virtual void GetBTC(int &B, int &T, int &C) {
        B = config.n_batch();
        C = config.nEmbed();
        T = config.n_ctx();
        assert(B > 0);
        assert(T > 0);
        assert(C > 0);
    };
    virtual hEDevices curDevice(int flag = 0x0) {
        assert(hEDS != nullptr);
        return hEDS;
    }
    virtual void *GetGGCTX(int typ = 0x0) {
        /*switch(typ){
        case 1:
            return ctx_build;
        case 2:
            return ctx_work;
        default:
            return ctx_build;
        }
        assert(0);*/
        return nullptr;
    }
    //  Activations would creat at Symbolic Analysis stage

    virtual bool Build(int flag = 0x0);
    virtual bool BeforeBuild(int flag = 0x0);
    virtual bool AfterBuild(bool isInitParam, int flag = 0x0);
    virtual bool UpdateNCTX(int _nctx, int flag = 0x0);
    // Koifish is so flex than it would has different parameters at different phase!
    virtual bool UpdateParams(int flag = 0x0);
    virtual void UpdateTernary(int flag = 0x0);

    virtual int BuildComputeGraph(int order, void *ctx, int flag);
    virtual hGensor BuildLoss(void *ctx, hGensor cur, int flag = 0x0);
    virtual hGensor BuildTarget(void *ctx, hGensor cur, int flag = 0x0) { return nullptr; }
    virtual hGensor GetGensor(const string &name, int flag = 0x0) { return gensors.Get(name, flag); }
    virtual GENSOR_INFO &GetGensorInfo(hGensor hP, int flag = 0x0) {
        assert(gensors.infos.find(hP) != gensors.infos.end());
        return gensors.infos[hP];
    }

    virtual string __repr__(string &suffix, string &prefix, int flag = 0x0) {
        _INFO("Ganlia (");
        prefix += "\t";
        suffix += "\n)\n";
        return "";
    }
    virtual string DebugInfo(int type = 0x0, int flag = 0x0) { return ""; }

    virtual void Dump(int type, int flag = 0x0) {}

    virtual void Statistic(int typ, int flag = 0x0);
    // virtual void CreateWiki(int flag=0x0)   {}

    // return target_probs = OutCLS->target
    virtual hGensor Target() { return target_probs; }
    virtual hGensor Output() {
        assert(out_node != nullptr);
        return out_node;
    }
    virtual hGensor Input() { return nullptr; }

    void UpdateTensors(int flag = 0x0) {
        UNUSED(flag);
        gensors.Clear();
        std::stack<hFISH> all_childs;
        for (auto child : childs) all_childs.push(child);
        while (!all_childs.empty()) {
            hFISH cur = all_childs.top();
            for (auto child : cur->childs) {
                all_childs.push(child);
            }
            gensors.Insert(cur->gensors.nag);
            all_childs.pop();
        }
        int nTensor = gensors.size();
    }

    // If isParam 1) call InitParam@huTensor::Alloc(random or serialize); 2) alloc grad if isTrain;
    void InitGensor(void *ctx, const string &name, hGensor gensor, bool isParam, int flag = 0);

    void InitGensor(void *ctx, hGensor gensor, const char *name, struct random_normal_distribution *rnd = nullptr, int flag = 0);

    void SetTensor(const int nx, const int ny, const std::vector<float> &arr_data, const char *name, int flag = 0x0) {
        assert(0);  //  Drepecated
        hGensor inp = GetGensor("inp");
        float *data = (float *)inp->data;  // ggml_get_data(inp);
        assert(data != nullptr);
        const int n = nx * ny;
        // assert(nx == n_img_size && ny == n_img_size);
        for (int k = 0; k < 3; k++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    data[k * n + y * nx + x] = arr_data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    void SetInput(const int nx, const int ny, const std::vector<float> &arr_data, int flag = 0x0) { SetTensor(nx, ny, arr_data, "inp", flag); }

    virtual void Neck(const std::string &key_, const SHAPE &shape, int flag = 0x0) { ; }
    // Deprecated
    hGensor AddTensor(const std::string &key_, typNUMBER tp, const SHAPE &shape, int flag = 0x0);
    hGensor AddTensor(void *ctx, const std::string &key_, typNUMBER tp, const SHAPE &shape, bool isParam, int flag = 0x0);

    std::vector<hLayer> layers;
    virtual void BeforeAddLayer() { ; }
    virtual void AddLayer(hLayer hLay, int flag = 0x0) {
        BeforeAddLayer();
        hLay->id = layers.size();
        layers.push_back(hLay);
        for (auto hN : hLay->neurons) {
            AfterAddNeuron(hN, flag);
        }
        AfterAddLayer(hLay);
    }
    virtual void AfterAddLayer(hLayer hLay, int flag = 0x0) { ; }
    virtual void AfterAddNeuron(hNeuron hN, int flag = 0x0) { ; }

    template <typename T>
    void AddLayer(hFISH graph, const std::string &key_, const SHAPE &shape, int flag) {  // neural layer with only one neruon
        hNeuron hN   = std::make_shared<T>(graph, key_, shape, flag);
        hLayer layer = std::make_shared<NeLayer>(key_, flag);
        layer->Add(hN);
        AddLayer(layer, flag);
    }

    virtual void Sparsing(int flag = 0x0);

    virtual bool BeforeNextStep(int iter, int flag = 0x0);
    virtual bool AfterNextStep(int iter, int flag = 0x0);

    virtual int GenSentence(int flag = 0x0) { return -1; }

    virtual bool ComputePlan(int flag = 0x0);
    int BuildGraphFromRaw(int flag);
    hNeuron J2Neuron(void *ctx_build, string &, int level, const JConfig &j, int flag);
    virtual int jToGraph(void *, bool isBuild, int flag = 0x0);

    virtual void Train(int flag = 0x0);
    virtual void Loss(int flag = 0x0) {}
    virtual double Eval_ppl(int flag = 0x0);

    virtual void CopyWeight(const Fish *src, int flag = 0x0);
    virtual bool LocalFeeling(hSampLoader hLoader, vector<float> &result, int flag = 0x0) { return false; }

    virtual bool isValid() { return true; }

    virtual void LossCurve(int flag = 0x0) {  // training accuracy curve
#ifdef _USE_WANDB_
        _WANDB_log(1.0);
#endif
    }
    static hFISH MakeInstance(const std::string nam_, struct CLI_params &params, vector<hWIKI> wikis, ROLE_TYPE role, int flag);
    static hFISH MakeSwarm(const std::string nam_, struct CLI_params &params, int flag);
    static hFISH MakeInstance(const std::string nam_, struct CLI_params &params, const Fish *hSrc_, int flag);
    // static Fish* Copy(const Fish* src,int flag=0x0);
    virtual bool SaveTrain(string sX, int flag = 0x0);
    virtual bool LoadCheckPoint(int flag = 0x0);

    friend class GeNeuron;
    friend class SLP;
    friend class LayerNormal;
    friend class NT_SAM;
    friend class SelfAttention;
    friend class ROPE;
    friend class OutCLS;
    friend class TokenEmbed;
    friend class SAM_encoder;
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class OPT_Adam;
    friend class Distillation;
    friend class DictVAE;
    friend class GeneratOnPrompt;
    friend class SampLoader;
    friend class WIKI;
    friend class KVCache;
    friend class TGraph;
    friend class EDGE_DEVICES;
    friend class RLS_BP;
};

/*
    鳑鲏
    1. similar idea @"Model Swarms: Collaborative Search to AdaptLLM Experts via Swarm Intelligence"
*/
struct Pangpi : public Fish {
    hFISH head = nullptr;

    typedef enum { BIT_MASK } SPACE_TYPE;
    SPACE_TYPE space = BIT_MASK;

    int x = 0;
    float fitness;  // greater fitness will have a greater probability of being selected for recombination.
    vector<double> position;
    // Pangpi(const int dim, int flag = 0x0);
    // Pangpi(const int dim, const vector<int>&picks, int flag = 0x0);
    Pangpi(const std::string &nam_, struct CLI_params params, int flag = 0x0);

    void Train(int flag = 0x0) override;

    int DIM() const { return position.size(); }

    virtual void Copy(const Pangpi *src, int flag = 0x0) {
        position = src->position;
        fitness  = src->fitness;
        x        = src->x;
    }

    // aA+b*B
    virtual void MixPosition(double alpha, const Pangpi *A, double beta, const Pangpi *B, int flag) {
        int dim = position.size(), i;
        for (i = 0; i < dim; i++) {
            position[i] = alpha * A->position[i] + beta * B->position[i];
        }
    }

    virtual void cross_over(const Pangpi *A, const Pangpi *B, int flag = 0x0);
    virtual void mutatioin(double T_mut, int flag = 0x0);
};
typedef shared_ptr<Pangpi> hPangpi;

enum FFN_TYPE {
    SWIGLU = 0,
    VANILLA,
    ONLY_LNormal,
    ONLY_RMSNormal,
    VAR_0,
    VAR_LAST,  // last layer with gaussian noise
    SMOE,      // Sparsely-Gated Mixture-of-Experts Layer
    GATE_CYS,
};
// Deprecated, repalce by SelfAttention+FFN
struct QKV_LAY : public NeLayer {
    hGensor eps = nullptr;
    LayerNormal att_norm, ffn_norm;
    SLP Q, K, V, proj, up, down;
    hGensor ffn_gate = nullptr;
    // attention
    hGensor wk = nullptr, wv = nullptr;
    hGensor wo = nullptr;

    // SMOE
    hGensor ffn_gate_inp = nullptr, ffn_gate_exps = nullptr, ffn_down_exps = nullptr, ffn_up_exps = nullptr;
    hGensor ffn_gate_inp_shexp = nullptr, ffn_gate_shexp = nullptr, ffn_down_shexp = nullptr, ffn_up_shexp = nullptr;

    // long rope factors
    hGensor rope_long = nullptr, rope_short = nullptr, rope_freqs = nullptr;
    hGensor rope_(bool isLong) const {
        if (rope_freqs != nullptr) {
            return rope_freqs;
        }
        if (isLong) {
            return rope_long;
        }
        return rope_short;
    }

    QKV_LAY(Fish *hF_, int id);
    int64_t parameter_count() {
        int64_t nx = 0;
        nx += att_norm.nElem();  // tELEM(attention_norm);

        nx += Q.nElem();
        nx += tELEM(wk);
        nx += tELEM(wv);
        nx += tELEM(wo);
        nx += ffn_norm.nElem();  // tELEM(ffn_norm);
        nx += tELEM(ffn_gate);
        nx += down.nElem();
        nx += up.nElem();  //(ffn_down); nx += tELEM(ffn_up);
        return nx;
    }
    virtual bool CreateFFN(const CLI_params &config, ggml_context *ctx, FFN_TYPE tpFFN, int flag = 0x0);
    string __repr__(string &suffix, string &prefix, int flag = 0x0) override;

    virtual void save_gguf(struct gguf_context *fctx, int flag);
};
typedef std::shared_ptr<QKV_LAY> hLQKV;