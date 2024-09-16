/**
 *  Copyright 2023-2024 by Grusoft
 *
 *  \brief A collection of neurons
 *  \author Yingshi Chen
 */

#pragma once

#include <cassert>
#include <complex>
#include <memory>
#include <vector>
#include <map>
#include <typeinfo>
#include <float.h>
#include <stdio.h>
#include <threads.h>
#include <atomic>
#include <inttypes.h>
#include <regex>
#include <stack>
using namespace std;

#include "../ggex/GG_util.hpp"
typedef struct ggml_tensor gensor;
typedef struct ggml_tensor *hGensor;
typedef std::map<std::string, struct ggml_tensor *> TENSORs;
typedef std::vector<int> SHAPE;

#include "train.h"
#include "TGraph.hpp"
#include "Optimizer.hpp"
#include "GPT.hpp"
#include "../lenda/util/GST_util.hpp"
#include "../Fuzi/Distillation.hpp"

class Fish;
typedef shared_ptr<Fish> hFISH;
struct NeLayer;

struct NP_
{ // Paramer of GeNeuron
    std::string type, title;
    SHAPE shape;
    NP_(const std::string &t, const std::string &n, SHAPE s) : type(t), title(n), shape(s) {}
};
class GeNeuron  {
    // GeNeuron(const GeNeuron&)=default;       for model.layers.resize(n_layer)@llama.cpp
    GeNeuron &operator=(const GeNeuron &) = default;

protected:
    hFISH hOrg = nullptr;
    COMPRESSIVE_SENSING compression = SKIP;
    SHAPE shape;

public:
    NeLayer *hLay = nullptr;
    std::string name;
    GeNeuron() {}
    GeNeuron(SHAPE shape_, hFISH hG_, int flag) : shape(shape_), hOrg(hG_) { ; }
    virtual ~GeNeuron() { ; }

    virtual bool Empty() { return shape.size() == 0; }
};
typedef shared_ptr<GeNeuron> hNeuron;

struct SLP : public GeNeuron
{ // single layer perceptron
    hGensor w = nullptr, b = nullptr;
    hGensor u = nullptr, s = nullptr, v = nullptr;
    SLP() {}
    SLP(Fish *ctx, const std::string &key_, const SHAPE &shape_, hFISH hG_, int flag) : GeNeuron(shape_, hG_, flag)
    {
        // compression = hOrg->params.compression;
        Build(ctx, key_, shape_, flag);
    }

    void Init(hGensor w_, hGensor b_, const SHAPE &shape_, int flag = 0x0);
    void Build(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag);
    hGensor Build_2(struct ggml_context *ctx0, hGensor cur, int flag = 0x0);
};
struct LayerNormal : public GeNeuron
{
    hGensor w = nullptr, b = nullptr;
    LayerNormal() {}
    LayerNormal(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag)
    {
        Build(ctx, key_, shape, flag);
    }
    void Build(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag);
};

struct SelfAttention : public GeNeuron
{
    SLP q, k, v;
    SLP proj;
    // SLP qkv;
    SelfAttention() {}
    SelfAttention(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag);
};

struct BROWN_Motion    {
    hGensor wq=nullptr;
    int n_embd=-1, n_head=-1, N=-1, n_batch=-1, n_rot=-1, n_ctx=-1, n_head_kv=-1, n_vocab=-1, n_ff=-1, n_past = 0;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;

    BROWN_Motion()  {}
    // BROWN_Motion(hGensor _wq, int _embd, int _head, int _N, int _batch, int _rot, int _ctx, int _head_kv, float f_eps, float rope_base, float rope_scale)
    //     : n_embd(_embd), n_head(_head), N(_N), n_batch(_batch), n_rot(_rot), n_ctx(_ctx), n_head_kv(_head_kv),
    //       f_norm_rms_eps(f_eps), rope_freq_base(rope_base), rope_freq_scale(rope_scale), wq(_wq)    {
            
    // }
    BROWN_Motion(hGensor _wq, struct CLI_params& hparams,int flags) : wq(_wq)   {
        f_norm_rms_eps  = hparams.f_norm_rms_eps;
        rope_freq_base  = hparams.rope_freq_base;
        rope_freq_scale = hparams.rope_freq_scale;  
        float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        // int n_embd_gqa = hparams.n_embd_gqa();
        int n_embd_head = hparams.n_embd_head( );
        n_head_kv=hparams.n_head_kv;
        // n_vocab = hparams.n_vocab;          
        n_batch  = hparams.common.n_batch;          
        n_ctx = hparams.common.n_ctx;              n_embd = hparams.n_embd;
        n_head = hparams.n_head,            n_rot = hparams.n_rot,                    n_ff = hparams.n_ff;
        N = n_ctx;
    }
    hGensor QKV_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    virtual hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos, bool use_flash);
};
typedef shared_ptr<BROWN_Motion> hBrownMotion;

struct QKV_Motion : public BROWN_Motion    {
    hGensor wk, wv;
    // int n_embd, n_head, N, n_batch, n_rot, n_ctx, n_head_kv, n_past = 0;
    // float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    QKV_Motion() {}
    // QKV_Motion(hGensor _wq, hGensor _wk, hGensor _wv, int _embd, int _head, int _N, int _batch, int _rot, int _ctx, int _head_kv, float f_eps, float rope_base, float rope_scale)
    //     : BROWN_Motion(_wq, _embd, _head, _N, _batch, _rot, _ctx, _head_kv, f_eps, rope_base, rope_scale), wk(_wk), wv(_wv)
    // {
    // }
    QKV_Motion(hGensor _wq, hGensor _wk, hGensor _wv, struct CLI_params& hparams,int flag)
        : BROWN_Motion(_wq, hparams,flag), wk(_wk), wv(_wv)
    {
    }

    // hGensor QKV_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos, bool use_flash)    override;
};

struct NeLayer
{ // Neural Layer with many neurons
    int id = -1;
    bool isLast = false;
    std::string name;
    std::vector<hNeuron> neurons;

    void Add(hNeuron hN, int flag = 0x0)
    {
        neurons.push_back(hN);
        hN->hLay = this; //???  
    }
    NeLayer(int id_) : name("NeLayer"),id(id_) {}
    NeLayer(const std::string &n_, int flag = 0x0) : name(n_) {}
    virtual ~NeLayer() {}
    // NeLayer* N_(const std::string& t,const std::string& n,SHAPE s)   {
    //     return this;
    // }
    virtual string __repr__( string& suffix,string& prefix,int flag=0x0)   {    return "";  }
};
typedef shared_ptr<NeLayer> hLayer;

struct save_train_model {
    std::string fn_checkpoint_out,fn_model_out,fn_model_base,pattern_fn_it,fn_latest;
    // struct llama_model * model=nullptr;
    void * model=nullptr;

    virtual void Init(CLI_params&params,void * model_,int flag=0x0)   {
        fn_checkpoint_out = params.common.fn_checkpoint_out;
        fn_model_out      = params.fn_model_out;
        pattern_fn_it     = params.common.pattern_fn_it;
        fn_latest         = params.common.fn_latest;
        model             = model_;
    }        
};

struct MixOfModels{
    // hGensor gate = nullptr;
    hGensor embed2w = nullptr;
    hGensor Forward(struct ggml_context * ctx,hGensor cur,hGensor w);
};
class Fish : public std::enable_shared_from_this<Fish>    {
    Fish(const Fish &);
    Fish &operator=(const Fish &);

protected:
    std::string name;

    /*  wiki contains knowledge reflect the founation of our world
        wikis[0] is the backbone of FISH
    */
    vector<hWIKI> wikis;
    vector<hGensor> tmpExLogis;
    WIKI::INDUCT_MODE teach=WIKI::_LOGITS;
    // Generate some results on prompt
    hGOPT gopt = nullptr;
    virtual int GenSentence(int flag=0x0);

    struct CLI_params hparams;
    save_train_model save_data;

    hTGraph hGraph;                            // compuation graph
    int graph_order=-1;
    struct ggml_cgraph *gf = NULL, *gb = NULL; // only for debug
    struct ggml_cplan gf_plan,gb_plan;

    std::map<std::string, struct ggml_tensor *> gensors;
    void Gensor2Map(struct ggml_tensor *gensor){
        auto key = ggml_get_name(gensor);
        assert(gensors.find(key) == gensors.end());
        gensors[key] = gensor;
    }   
    virtual hGensor GetGensor(const char *name, int flag = 0x0)    {
        assert(gensors.find(name) != gensors.end());
        return gensors[name];
    } 
        // 
    // std::vector<std::pair<std::string, struct ggml_tensor *>> tmaps;
    bool updateTMap = false;
    bool isLocalInfer = false;

    std::vector<uint8_t> work_buffer;
    // from GGML
    int size = 0; //,n_nodes=0,n_leafs=0;
    size_t nParams = 0, nParamsGGUF = 0, szModel = 0;

    hGensor in_node = nullptr, out_node = nullptr;
    hGensor loss = nullptr, target_probs = nullptr;
    hGensor preLogits = nullptr;        //no SOFTMAX
    
    //hGensor gate=nullptr;      //create@InitModel update@
    MixOfModels mom;

    hDataToken hTokenset=nullptr;
    hOptimizer hOPT;
    hDistillation hDistler;
    // performance
    int perf_runs = 0;
    int64_t perf_cycles = 0, perf_time_us = 0;
    struct ggml_context *ctx = nullptr;         // model ctx
    struct ggml_context *ctx_work = nullptr;    // training ctx
    struct ggml_context *ctx_input = nullptr; 
    size_t ctx_size = 0;
    
    std::vector<hFISH> childs;

    virtual void Clear() {
        if (ctx!=nullptr) {
            ggml_free(ctx);
        }
        if (ctx_input!=nullptr) {
            ggml_free(ctx_input);
        }
    }

    std::vector<std::string> to_quant, to_skip;

public:    
    MODEL_ARCH arch = MODEL_ARCH::_X_;

    Fish() {}
    Fish(const std::string&nam_,struct CLI_params params,int flag=0x0) : name(nam_),hparams(params) {
        arch = params.arch;
    }
    Fish(const std::string&nam_,struct ggml_context *ctx_, int flag = 0x0) : name(nam_),ctx(ctx_)    {
        GGML_PRINT("=== %s ===\n", __func__);
        // allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    }
    bool isTrain()  {
        return !isLocalInfer;
    }
    bool hasWiki()  {   return wikis.size()>0;  }

    virtual ~Fish() { Clear(); }
    virtual std::string Name()  {   return name.c_str();  }

    virtual size_t Size(int flag = 0x0) { return ctx_size; }

    virtual void Init(const vector<hWIKI>& wikis,int flag=0x0)          {   throw "Fish::Init is ...";           }       
    virtual void BuildGraph(int flag=0x0)               {   throw "Fish::BuildGraph is ...";     }

    virtual void BuildGraph(struct ggml_context *ctx0, ggml_gallocr_t &allocr, bool isOnlySymbol, int flag = 0x0)    {
        hGraph = std::make_shared<TGraph>(ctx0, GGML_DEFAULT_GRAPH_SIZE, false, isOnlySymbol);
        assert(out_node != nullptr && in_node != nullptr);
        hGraph->build_forward(out_node, true);
        hGraph->disconnect_node(in_node);
        ggml_gallocr_alloc_graph(allocr, hGraph->cgraph);
    }

    virtual string __repr__(string& suffix,string& prefix,int flag=0x0){
        _INFO( "Ganlia (" );
        prefix += "\t";
        suffix += "\n)\n";
        return "";
    }        
    
    virtual void Dump(int type,int flag=0x0)            {}

    virtual void Statistic(int typ, int flag = 0x0)     {
        ggml_graph_stat(gf);
        if(gb!=nullptr) ggml_graph_stat(gb);
        if (1)        {
            ggml_graph_dump_dot(gf, NULL, "opt-forward.dot");
            if(gb!=nullptr) ggml_graph_dump_dot(gb, gf, "opt-backward.dot");
        }   else        {
            ggml_graph_print(gf);
            if(gb!=nullptr) ggml_graph_print(gb);
        }

        int nT = gensors.size(), nQ = 0, nF16 = 0;
        for (auto t : gensors)        {
            auto type = t.second->type;
            if (ggml_is_quantized(type))
                nQ++;
            if (type == GGML_TYPE_F16)
                nF16++;
        }
    }
    // virtual void CreateWiki(int flag=0x0)   {}
    
    hGensor Target()    {   return nullptr;    }

    void UpdateTensors(int flag = 0x0)    {
        UNUSED(flag);
        gensors.clear();
        std::stack<hFISH> all_childs;
        for (auto child : childs)
            all_childs.push(child);
        while (!all_childs.empty())        {
            hFISH cur = all_childs.top();
            for (auto child : cur->childs)
            {
                all_childs.push(child);
            }
            gensors.insert(cur->gensors.begin(), cur->gensors.end());
            all_childs.pop();
        }
        int nTensor = gensors.size();
    }

    // If isParam, only alloc grad, no init!
    void InitGensor(struct ggml_context *ctx, const char *name, hGensor gensor, bool isParam, int flag = 0)    {
        if (name != nullptr)
            ggml_set_name(gensor, name);
        assert(gensor->data == nullptr);
        Gensor2Map(gensor);
        if (isParam && isTrain())        {
            ggml_set_param(ctx, gensor);
            nParams += ggml_nelements(gensor);
        }
    }

    void InitGensor(struct ggml_context *ctx, hGensor gensor, const char *name, struct random_normal_distribution *rnd = nullptr, int flag = 0)
    {
        if (name != nullptr)
            ggml_set_name(gensor, name);
        if(isTrain())
            ggml_set_param(ctx, gensor);
        if (gensor->data == nullptr)        {
            assert(0);
        }
        else
        {
            if (rnd != nullptr)
                randomize_tensor_normal(gensor, rnd);
            else
                ggml_set_zero(gensor);
        }
        if (updateTMap)        {
            Gensor2Map(gensor);            
        }
        if(isTrain())
            nParams += ggml_nelements(gensor);
    }

    

    hGensor get_tensor(const char *name, int flag = 0x0)    {
        return hGraph->get_tensor(name, flag); // from GGML
    }

    void SetTensor(const int nx, const int ny, const std::vector<float> &arr_data, const char *name, int flag = 0x0)    {
        hGensor inp = get_tensor("inp");
        float *data = (float *)ggml_get_data(inp);
        assert(data != nullptr);
        const int n = nx * ny;
        // GGML_ASSERT(nx == n_img_size && ny == n_img_size);
        for (int k = 0; k < 3; k++)
        {
            for (int y = 0; y < ny; y++)
            {
                for (int x = 0; x < nx; x++)
                {
                    data[k * n + y * nx + x] = arr_data[3 * (y * nx + x) + k];
                }
            }
        }
    }

    void SetInput(const int nx, const int ny, const std::vector<float> &arr_data, int flag = 0x0)    {
        SetTensor(nx, ny, arr_data, "inp", flag);
    }

    virtual void Neck(const std::string &key_, const SHAPE &shape, int flag = 0x0) { ; }

    hGensor AddTensor(const std::string &key_, enum ggml_type tp, const SHAPE &shape, int flag = 0x0);

    std::vector<hLayer> layers;
    virtual void BeforeAddLayer() { ; }
    virtual void AddLayer(hLayer hLay, int flag = 0x0)    {
        BeforeAddLayer();
        hLay->id = layers.size();
        layers.push_back(hLay);
        for (auto hN : hLay->neurons)
        {
            AfterAddNeuron(hN, flag);
        }
        AfterAddLayer(hLay);
    }
    virtual void AfterAddLayer(hLayer hLay, int flag = 0x0) { ; }
    virtual void AfterAddNeuron(hNeuron hN, int flag = 0x0) { ; }

    template <typename T>
    void AddLayer(hFISH graph, const std::string &key_, const SHAPE &shape, int flag)    { // neural layer with only one neruon
        hNeuron hN = std::make_shared<T>(graph, key_, shape, flag);
        hLayer layer = std::make_shared<NeLayer>(key_, flag);
        layer->Add(hN);
        AddLayer(layer, flag);
    }

    virtual void AddLayer(const std::string &key_, std::vector<NP_> nps, int flag = 0x0)    {
        hLayer layer = std::make_shared<NeLayer>(key_);
        for (auto param : nps)
        {
            hNeuron hN = nullptr;
            auto tp = param.type;
            if (param.type == "SelfAttention")
            {
                hN = std::make_shared<SelfAttention>(this, key_ + param.title, param.shape, flag);
            }
            else if (param.type == "LayerNormal")
            {
                hN = std::make_shared<LayerNormal>(this, key_ + param.title, param.shape, flag);
            }
            else if (param.type == "SLP")
            {
                hN = std::make_shared<SLP>(this, key_ + param.title, param.shape, shared_from_this(), flag);
            }
            else
            {
                assert(0);
            }
            layer->neurons.push_back(hN);
        }
        AddLayer(layer, flag = 0x0);
    }
    
    bool OnTrainStep(struct train_opt_callback_data *data0,SampLoader&loader, int accum_step, float *sched, int flag = 0x0);

    virtual bool InitCTX(int flag=0x0);
    virtual void Train(int flag = 0x0);
    virtual void Loss(int flag = 0x0) {}

    virtual void CopyWeight(const Fish* src,int flag = 0x0) {}
    virtual bool LocalFeeling(std::vector<llama_token>&tokens,vector<float>& result)   const  {   return false;   }

    virtual bool isValid()  {   return true;    }

    virtual void LossCurve(int flag = 0x0)
    { // training accuracy curve
#ifdef _USE_WANDB_
        _WANDB_log(1.0);
#endif
    }
    static hFISH MakeInstance(const std::string nam_,struct CLI_params& params,int flag);
    static hFISH MakeInstance(const std::string nam_,struct CLI_params& params,const Fish *hSrc_,int flag);
    // static Fish* Copy(const Fish* src,int flag=0x0);
    virtual void SaveTrain(struct save_train_model * data, struct train_state * train);
    virtual void save_gguf(const char * filename, int flag){}

    friend class GeNeuron;
    friend class SLP;
    friend class LayerNormal;
    friend class NT_SAM;
    friend class SAM_encoder;
    friend class Optimizer;
    friend class Distillation;
    friend class ConsiceDict;
    friend class GeneratOnPrompt;
    friend class LLaMeta;   
    friend class SampLoader;
};
