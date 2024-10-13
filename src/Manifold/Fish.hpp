/**
 *  Copyright 2023-2024 by Grusoft
 *
 *  \brief Fish - a collection of neurons
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
#include "GoPT.hpp"
#include "../lenda/util/GST_util.hpp"
#include "../Fuzi/Distillation.hpp"

class Fish;
typedef shared_ptr<Fish> hFISH;
typedef vector<hFISH> tpSWARM;    
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

enum FFN_TYPE {
    SWIGLU = 0,
    VANILLA,
    ONLY_LNormal,    
    ONLY_RMSNormal,
    VAR_0,
    VAR_LAST,       //last layer with gaussian noise 
    SMOE,           //Sparsely-Gated Mixture-of-Experts Layer
    GATE_CYS,
};
    
struct QKV_LAY : public NeLayer {
    hGensor eps=nullptr;
    hGensor attention_norm=nullptr;
        // attention
    hGensor wq=nullptr,wk=nullptr,wv=nullptr;
    hGensor wo=nullptr;
        // normalization
    hGensor  ffn_norm=nullptr,ffn_gate=nullptr,ffn_down=nullptr,ffn_up=nullptr;  
        //SMOE
    hGensor  ffn_gate_inp=nullptr,ffn_gate_exps=nullptr,ffn_down_exps=nullptr,ffn_up_exps=nullptr;  
    hGensor  ffn_gate_inp_shexp=nullptr,ffn_gate_shexp=nullptr,ffn_down_shexp=nullptr,ffn_up_shexp=nullptr;

    // long rope factors
    hGensor rope_long  = nullptr, rope_short = nullptr, rope_freqs = nullptr;
    hGensor rope_(bool isLong)  const {     
        if (rope_freqs != nullptr) {
            return rope_freqs;
        }
        if (isLong) {
            return rope_long;
        }
        return rope_short;
    }

    QKV_LAY(int id)  :  NeLayer(id)     {   name = "QKV_LAY";   }
    int64_t parameter_count() {
        int64_t nx = 0;
        nx += ggml_nelements(attention_norm);
        nx += ggml_nelements(wq);            nx += ggml_nelements(wk);            nx += ggml_nelements(wv);
        nx += ggml_nelements(wo);
        nx += ggml_nelements(ffn_norm); nx += ggml_nelements(ffn_gate); nx += ggml_nelements(ffn_down); nx += ggml_nelements(ffn_up);            
        return nx;
    }
    virtual bool CreateFFN(const CLI_params&hparams,ggml_context *ctx,FFN_TYPE tpFFN,int flag=0x0);
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;
};
typedef std::shared_ptr<QKV_LAY> hLQKV;
struct BROWN_Motion    {
    int version = 0;
    hGensor wq=nullptr;
    bool use_flash = true;   
    // int layer_id=-1;
    hLQKV lay;
    int rope_type=-1,n_embd_head=-1,n_tokens=-1,n_embd_head_k,n_embd_k_gqa=-1,n_embd_head_v,n_embd_v_gqa=-1;
    int n_embd=-1, n_head=-1, N=-1, n_batch=-1, n_rot=-1,n_ctx_orig=-1, n_ctx=-1, n_head_kv=-1, n_vocab=-1, n_ff=-1, n_past = 0;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    float f_max_alibi_bias;
    float attn_soft_cap;
    float beta_fast=32.0,beta_slow=1.0,ext_factor=0,attn_factor=1;

    BROWN_Motion()  {}
    // BROWN_Motion(hGensor _wq, int _embd, int _head, int _N, int _batch, int _rot, int _ctx, int _head_kv, float f_eps, float rope_base, float rope_scale)
    //     : n_embd(_embd), n_head(_head), N(_N), n_batch(_batch), n_rot(_rot), n_ctx(_ctx), n_head_kv(_head_kv),
    //       f_norm_rms_eps(f_eps), rope_freq_base(rope_base), rope_freq_scale(rope_scale), wq(_wq)    {
            
    // }
    BROWN_Motion(hGensor _wq, struct CLI_params& hparams,hLQKV lQKV,int flags) : wq(_wq),lay(lQKV)   {
        int layer_id = lay->id;
        version = hparams.Get({"model","attention","version"},version);

        f_norm_rms_eps  = hparams.f_norm_rms_eps;
        rope_freq_base  = hparams.rope_freq_base;
        rope_freq_scale = hparams.rope_freq_scale;  
        f_max_alibi_bias = hparams.f_max_alibi_bias;
        attn_soft_cap = hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f;
        use_flash = hparams.isFlashAtten();
        // float kv_scale = 1.0f/sqrtf(float(n_embd)/n_head);
        // int n_embd_gqa = hparams.n_embd_gqa();
        // int n_embd_head = hparams.n_embd_head( );
        n_head_kv=hparams.n_head_kv(layer_id);
        // n_vocab = hparams.n_vocab;          
        n_batch  = hparams.common.n_batch;          
        n_ctx_orig = hparams.n_ctx_orig();                  n_ctx = hparams.n_ctx();            n_tokens = n_ctx;             
        n_embd = hparams.n_embd;
        n_head = hparams.n_head(layer_id),                  n_rot = hparams.n_rot,              n_ff = hparams.n_ff(layer_id);
        n_embd_head = hparams.n_embd_head(layer_id);
        
        rope_type = hparams.rope_type;
        N = n_ctx;

        n_embd_head_k = hparams.n_embd_head_k;
        n_embd_k_gqa  = hparams.n_embd_k_gqa(layer_id);
        n_embd_head_v = hparams.n_embd_head_v;
        n_embd_v_gqa  = hparams.n_embd_v_gqa(layer_id);

        
    }
    hGensor QKV_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    virtual hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos,const llama_kv_cache & kv);
};
typedef shared_ptr<BROWN_Motion> hBrownMotion;

struct QKV_Motion : public BROWN_Motion    {
    hGensor wk, wv, inp_KQ_mask = nullptr;
      
    // int n_embd, n_head, N, n_batch, n_rot, n_ctx, n_head_kv, n_past = 0;
    // float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    QKV_Motion() {}
    // QKV_Motion(hGensor _wq, hGensor _wk, hGensor _wv, int _embd, int _head, int _N, int _batch, int _rot, int _ctx, int _head_kv, float f_eps, float rope_base, float rope_scale)
    //     : BROWN_Motion(_wq, _embd, _head, _N, _batch, _rot, _ctx, _head_kv, f_eps, rope_base, rope_scale), wk(_wk), wv(_wv)
    // {
    // }
    QKV_Motion(hGensor _wq, hGensor _wk, hGensor _wv,hGensor inp_mask, struct CLI_params& hparams,hLQKV lQKV,int flag)
        : BROWN_Motion(_wq, hparams,lQKV,flag), wk(_wk), wv(_wv),inp_KQ_mask(inp_mask)
    {
    }
    
    // hGensor QKV_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos,const llama_kv_cache & kv)    override;
};
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
    bool isRes = true,isSiLU=false;    
    vector<hGensor> exs;
    int nTiB;       //number of tokens in batch
    // hGensor gate = nullptr;
    hGensor embed2w = nullptr;
    hGensor gat_ = nullptr;

    virtual hGensor Build(CLI_params&hparams,struct ggml_context * ctx,hGensor cur,int flag=0x0)  { return nullptr; }
    hGensor Forward(struct ggml_context * ctx,hGensor cur,hGensor w);
};

struct MixOfSwarm : public MixOfModels{
    virtual void Init(tpSWARM&swarm,struct ggml_context *ctx,int n_embd,int flag=0x0);
    hGensor Build(CLI_params&hparams,struct ggml_context * ctx,hGensor cur,int flag=0x0)  override;
};

typedef llama_kv_cache KV_CACHE;

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

    struct CLI_params hparams;
    save_train_model save_data;

    // KV_CACHE kv_cache;
    
    hTGraph hGraph;                            // compuation graph
    int graph_order=-1;
    struct ggml_cgraph *gf = NULL, *gb = NULL; // only for debug
    struct ggml_cplan gf_plan,gb_plan;

    std::map<std::string, struct ggml_tensor *> gensors;
    std::vector<hGensor> wGensors;      //gensors with weight parameters
    
    void Gensor2Map(std::vector<hGensor> gensors){
        for(auto gensor : gensors){
            Gensor2Map(gensor);
        }
    }   
    void Gensor2Map(struct ggml_tensor *gensor){
        auto key = ggml_get_name(gensor);
        assert(gensors.find(key) == gensors.end());
        gensors[key] = gensor;
    }   
    
        // 
    // std::vector<std::pair<std::string, struct ggml_tensor *>> tmaps;
    bool updateTMap = false;
    bool isLocalInfer = false;

    std::vector<uint8_t> work_buffer;
    // from GGML
    int size = 0; //,n_nodes=0,n_leafs=0;
    size_t nParams = 0, szModel = 0;

    hGensor in_node = nullptr, out_node = nullptr;
    hGensor loss = nullptr, target_probs = nullptr, KQ_pos = nullptr, inp_KQ_mask = nullptr;
    hGensor preLogits = nullptr;        //no SOFTMAX
    
    //hGensor gate=nullptr;      //create@InitModel update@
    MixOfModels mom;
    MixOfSwarm  mos;

    hDataToken hTokenset=nullptr;
    hOptimizer hOPT;
    vector<hGensor> optParams;     //paramter tensors updated by hOPT
    hDistillation hDistler;
    // performance
    int perf_runs = 0;
    int64_t perf_cycles = 0, perf_time_us = 0;
    struct ggml_context *ctx = nullptr;         // model ctx
    struct ggml_context *ctx_work = nullptr;    // training ctx
    struct ggml_context *ctx_input = nullptr; 
    struct ggml_init_params ctx_compute_params = {0, NULL,true,};
    struct ggml_context * ctx_compute = nullptr;    //build graph
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
    static tpSWARM swarm;
    MODEL_ARCH arch = MODEL_ARCH::_X_;
    enum ROLE_TYPE    {
        COMMON,
        SWARM_HEAD,
        SWARM_FOLLOWER,
    };
    ROLE_TYPE role = ROLE_TYPE::COMMON;

    Fish() {}
    Fish(const std::string&nam_,struct CLI_params params,ROLE_TYPE role_=COMMON,int flag=0x0) : name(nam_),hparams(params),role(role_) {
        arch = params.arch;
        if(jKEY(params.jConfig,{"train"}).empty())     {
            isLocalInfer = true;
        }
    }
    Fish(const std::string&nam_,struct ggml_context *ctx_, int flag = 0x0) : name(nam_),ctx(ctx_)    {
        GGML_PRINT("=== %s ===\n", __func__);
        // allocr = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    }
    bool isTrain()  {
        return !isLocalInfer;
    }
    bool hasWiki()  {   return wikis.size()>0;  }
    virtual struct ggml_cgraph *GetRawGraph( struct ggml_context *,int flag=0x0)    {     return nullptr; }
    virtual KV_CACHE *GetKVCache()  {   return nullptr;    }

    virtual ~Fish() { Clear(); }
    virtual std::string Name()  {   return name.c_str();  }

    virtual size_t Size(int flag = 0x0) { return ctx_size; }

    virtual void Init(const vector<hWIKI>& wikis,int flag=0x0)          {   throw "Fish::Init is ...";           }       
    virtual void Build(int flag=0x0)               {   throw "Fish::Build is ...";     }
    virtual void BeforeBuild(int flag=0x0);
    virtual void AfterBuild(int flag=0x0);

    virtual void Build(struct ggml_context *ctx0, ggml_gallocr_t &allocr, bool isOnlySymbol, int flag = 0x0)    {
        hGraph = std::make_shared<TGraph>(ctx0, GGML_DEFAULT_GRAPH_SIZE, false, isOnlySymbol);
        assert(out_node != nullptr && in_node != nullptr);
        hGraph->build_forward(out_node, true);
        hGraph->disconnect_node(in_node);
        ggml_gallocr_alloc_graph(allocr, hGraph->cgraph);
    }
    virtual hGensor GetGensor(const char *name, int flag = 0x0)    {
        assert(gensors.find(name) != gensors.end());
        return gensors[name];
    } 

    virtual string __repr__(string& suffix,string& prefix,int flag=0x0){
        _INFO( "Ganlia (" );
        prefix += "\t";
        suffix += "\n)\n";
        return "";
    }        
    
    virtual void Dump(int type,int flag=0x0)            {}

    virtual void Statistic(int typ, int flag = 0x0);
    // virtual void CreateWiki(int flag=0x0)   {}
    
    virtual hGensor Target()    {   return nullptr;    }
    virtual hGensor Output()    {   assert(out_node!=nullptr);   return out_node;    }
    virtual hGensor Input()     {   return nullptr;    }
    virtual struct ggml_cgraph * ForwarGraph()      {   return gf;  }
    virtual struct ggml_cgraph * BackwardGraph()    {   return gb;  }

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

    virtual int GenSentence(int flag=0x0);

    virtual bool ComputePlan(int flag=0x0);
    int BuildGraphFromRaw(int flag);

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
    static hFISH MakeInstance(const std::string nam_,struct CLI_params& params,vector<hWIKI> wikis,ROLE_TYPE role, int flag);
    static hFISH MakeSwarm(const std::string nam_,struct CLI_params& params,int flag);
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
    friend class WIKI;
};

struct LogicSalp : public Fish {

    hFISH head=nullptr;

    typedef enum {
        BIT_MASK
    }SPACE_TYPE;
    SPACE_TYPE space=BIT_MASK;

    int x=0;
    float fitness;	//greater fitness will have a greater probability of being selected for recombination.
    vector<double> position;
    // LogicSalp(const int dim, int flag = 0x0);
    // LogicSalp(const int dim, const vector<int>&picks, int flag = 0x0);
    LogicSalp(const std::string& nam_, struct CLI_params params,int flag=0x0);
    // void BeforeBuild(int flag=0x0)   override;   
    // void AfterBuild(int flag=0x0)   override;   
    void Train(int flag = 0x0)  override;

    	
    int DIM() const		{ return position.size(); }

    virtual void Copy(const LogicSalp*src,int flag=0x0) {
        position = src->position;
        fitness = src->fitness;
        x = src->x;
    }

    //aA+b*B
    virtual void MixPosition(double alpha, const LogicSalp*A, double beta, const LogicSalp*B, int flag) {
        int dim = position.size(), i;
        for (i = 0; i < dim; i++) {
            position[i] = alpha*A->position[i] + beta*B->position[i];
        }
    }

    virtual void cross_over(const LogicSalp*A,  const LogicSalp*B,int flag=0x0);
    virtual void mutatioin(double T_mut,int flag=0x0);
};
typedef shared_ptr<LogicSalp> hSALP;