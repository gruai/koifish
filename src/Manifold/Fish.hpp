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
inline bool CHECK_SHAPE(const SHAPE&shape){
    bool isValid = shape.size()>0;
    for(auto s : shape){
        if(s<0) {
            isValid = false;        break;
        }
        if(s>1024*1024)        {
            isValid = false;        break;
        }
    }
    assert(isValid);
    return isValid;
}

#include "train.h"
#include "TGraph.hpp"
#include "Optimizer.hpp"
#include "GoPT.hpp"
#include "../lenda/util/GST_util.hpp"
#include "../Fuzi/Distillation.hpp"

class Fish;
class NLP_AutoRegressive;
class KVCache {
    void *lamakv=nullptr;
    int kv_n = -1;
    void init_lamakv(int n_batch);
protected:
    NLP_AutoRegressive *lam_ = nullptr;
public:
    KVCache(NLP_AutoRegressive *lam_,int max_batch_size, int max_seq_len, int n_kv_heads, int head_dim);

    void update(int batch_size, int start_pos, hGensor xk, hGensor xv);
    hGensor get(int batch_size, int start_pos, int seq_len);

    int n_kv();
    virtual hGensor SerialV(struct ggml_context *ctx,hGensor vCur,int il,bool isSave);
    virtual hGensor SerialK(struct ggml_context *ctx,hGensor vCur,int il,bool isSave);

};

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
    Fish *hOrg = nullptr;
    COMPRESSIVE_SENSING compression = SKIP;
    SHAPE shape;

public:
    NeLayer *hLay = nullptr;
    std::string name;
    GeNeuron() {}    
    GeNeuron(SHAPE shape_, Fish *hG_, int flag) : shape(shape_), hOrg(hG_) { ; }
    virtual ~GeNeuron() { ; }

    virtual void Init(Fish *hG_, int flag=0x0) {    hOrg=hG_;   }
    virtual bool Empty() { return shape.size() == 0; }
    virtual size_t nElem()  { return 0x0; }
};
typedef shared_ptr<GeNeuron> hNeuron;

// single layer perceptron
struct SLP : public GeNeuron    { 
    hGensor w = nullptr, b = nullptr;
    hGensor u = nullptr, s = nullptr, v = nullptr;
    SLP( ) {}
    SLP(Fish *ctx, const std::string &key_, const SHAPE &shape_, Fish *hG_, int flag) : GeNeuron(shape_, hG_, flag)    {
        // compression = hOrg->params.compression;
        Build(key_, shape_, flag);
    }

    // void Init(hGensor w_, hGensor b_, const SHAPE &shape_, int flag = 0x0);
    void Build(const std::string &key_, const SHAPE &shape, int flag);
    hGensor Build_2(struct ggml_context *ctx0, hGensor cur, int flag = 0x0);
    
    size_t nElem()  override;  
};
struct LayerNormal : public GeNeuron    {
    hGensor w = nullptr, b = nullptr;
    LayerNormal() {}
    LayerNormal(Fish *ctx, const std::string &key_, const SHAPE &shape, int flag)    {
        Build(key_, shape, flag);
    }
    void Build(const std::string &key_, const SHAPE &shape, int flag);
    hGensor Build_2(struct ggml_context * ctx0,hGensor cur,int flag);
    size_t nElem()  override;      
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
    LayerNormal att_norm,ffn_norm;
    SLP Q,up,down;
    hGensor  ffn_gate=nullptr;  
    // attention
    hGensor wk=nullptr,wv=nullptr;
    hGensor wo=nullptr;
      
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

    QKV_LAY(Fish *hF_,int id);
    int64_t parameter_count() {
        int64_t nx = 0;
        nx += att_norm.nElem();    //ggml_nelements(attention_norm);

        nx += Q.nElem();            nx += ggml_nelements(wk);            nx += ggml_nelements(wv);
        nx += ggml_nelements(wo);
        nx += ffn_norm.nElem();     //ggml_nelements(ffn_norm); 
        nx += ggml_nelements(ffn_gate);         nx += down.nElem();         nx += up.nElem();            //(ffn_down); nx += ggml_nelements(ffn_up);            
        return nx;
    }
    virtual bool CreateFFN(const CLI_params&hparams,ggml_context *ctx,FFN_TYPE tpFFN,int flag=0x0);
    string __repr__( string& suffix,string& prefix,int flag=0x0)   override;

    virtual void save_gguf(struct gguf_context *fctx, int flag);
};
typedef std::shared_ptr<QKV_LAY> hLQKV;
struct BROWN_Motion    {
    Fish *hFish_ = nullptr;
    std::shared_ptr<KVCache> kv;
    char nam_[128];
    //Transfer probability of Token or Transfer probability of Embed
    static bool Transfer_1;  

    int version = 0;
    bool isTrain = false;
    hGensor wq=nullptr,wv=nullptr;
    bool use_flash = true,use_cache = false;   
    // int layer_id=-1;
    hLQKV lay;
    int rope_type=-1,n_embd_head=-1,n_embd_head_k,n_embd_k_gqa=-1,n_embd_head_v,n_embd_v_gqa=-1;//n_tokens=-1,
    int n_embd=-1, n_head=-1, N=-1, n_batch=-1, n_rot=-1,n_ctx_orig=-1, n_ctx=-1, n_head_kv=-1, n_vocab=-1, n_ff=-1, n_past = 0;
    float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    float f_max_alibi_bias;
    float attn_soft_cap;
    float beta_fast=32.0,beta_slow=1.0,ext_factor=0,attn_factor=1;

    BROWN_Motion()  {}
    BROWN_Motion(Fish *hFish_,hGensor _wq, hGensor _wv,struct CLI_params& hparams,hLQKV lQKV,int flags);
    hGensor W_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    virtual hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos);

    virtual hGensor DiffusionOnEmbed(struct ggml_context *ctx, hGensor teb, hGensor KQ_pos);
    virtual hGensor DiffusionOnToken(struct ggml_context *ctx, hGensor teb, hGensor KQ_pos);
};
typedef shared_ptr<BROWN_Motion> hBrownMotion;

struct QKV_Motion : public BROWN_Motion    {
    hGensor wk=nullptr, KQ_mask=nullptr;
      
    // int n_embd, n_head, N, n_batch, n_rot, n_ctx, n_head_kv, n_past = 0;
    // float f_norm_rms_eps, rope_freq_base, rope_freq_scale;
    QKV_Motion() {}
    // QKV_Motion(hGensor _wq, hGensor _wk, hGensor _wv, int _embd, int _head, int _N, int _batch, int _rot, int _ctx, int _head_kv, float f_eps, float rope_base, float rope_scale)
    //     : BROWN_Motion(_wq, _embd, _head, _N, _batch, _rot, _ctx, _head_kv, f_eps, rope_base, rope_scale), wk(_wk), wv(_wv)
    // {
    // }
    QKV_Motion( Fish *hFish_,hGensor _wq, hGensor _wk, hGensor _wv,hGensor inp_mask, struct CLI_params& hparams,hLQKV lQKV,int flag)
        : BROWN_Motion(hFish_,_wq, _wv, hparams,lQKV,flag), wk(_wk), KQ_mask(inp_mask)
    {
    }
    hGensor vXkq(struct ggml_context *ctx, hGensor v,hGensor kq,int layer_id);
    // hGensor W_rope(struct ggml_context *ctx, hGensor cur, hGensor w, hGensor KQ_pos, SHAPE shape, int flag = 0x0);
    hGensor Build(struct ggml_context *ctx, hGensor t04, hGensor KQ_pos)    override;
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
    ggml_gallocr_t alloc;
    hTGraph hGraph;                            // compuation graph
    int graph_order=-1;
    struct ggml_cgraph *gf = NULL, *gb = NULL; // only for debug
    struct ggml_cgraph * gb_tmp = NULL;
    struct random_normal_distribution *rnd = nullptr;   
    ggml_backend_buffer_t back_data = NULL; 
    std::vector<struct ggml_tensor *> checkpoints;
    bool measure_only=false;  
    struct ggml_cplan gf_plan,gb_plan;

    std::map<std::string, struct ggml_tensor *> gensors;
   
    
    void Gensor2Map(std::vector<hGensor> gensors){
        for(auto gensor : gensors){
            Gensor2Map(gensor);
        }
    }   
    void Gensor2Map(struct ggml_tensor *gensor){
        const char* key = ggml_get_name(gensor);
        assert(strlen(key)>0);
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
    hGensor loss = nullptr, target_probs = nullptr, KQ_pos = nullptr, KQ_mask = nullptr;
    hGensor preLogits = nullptr;        //no SOFTMAX
    
    //hGensor gate=nullptr;      //create@InitModel update@
    MixOfModels mom;
    MixOfSwarm  mos;

    hDataToken hTokenset=nullptr;
    hOptimizer hOPT;
    vector<hGensor> optParams;     //paramter tensors updated by hOPT
    std::vector<hGensor> xGensors;      
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
        arch = params.ModelArch();
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
    virtual struct ggml_cgraph *GetRawGraph( struct ggml_context *,bool isBuild,int flag=0x0)    {     return nullptr; }
    std::shared_ptr<KVCache> hCache = nullptr;
    // virtual KVCache *GetKVCache()  {   return nullptr;    }

    virtual ~Fish() { Clear(); }
    virtual std::string Name()  {   return name.c_str();  }

    virtual size_t Size(int flag = 0x0) { return ctx_size; }

    virtual bool Init(const vector<hWIKI>& wikis,int flag=0x0)          {   throw "Fish::Init is ...";           }       
    virtual bool Build(int flag=0x0)               {   throw "Fish::Build is ...";     }
    virtual bool BeforeBuild(int flag=0x0);
    virtual bool AfterBuild(bool isInitParam,int flag=0x0);

    virtual void Build(struct ggml_context *ctx0, ggml_gallocr_t &allocr, bool isOnlySymbol, int flag = 0x0)    {
        hGraph = std::make_shared<TGraph>(ctx0, GGML_DEFAULT_GRAPH_SIZE, false, isOnlySymbol);
        assert(out_node != nullptr && in_node != nullptr);
        hGraph->build_forward(out_node, true);
        hGraph->disconnect_node(in_node);
        ggml_gallocr_alloc_graph(allocr, hGraph->cgraph);
    }
    virtual int BuildComputeGraph(int order,struct ggml_context * ctx,ggml_gallocr_t& alloc,int flag);
    virtual hGensor BuildLoss( struct ggml_context * ctx,hGensor cur,int flag=0x0); 
    virtual hGensor BuildTarget( struct ggml_context * ctx,hGensor cur,int flag=0x0)    {   return nullptr;   } 
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
    void InitGensor(struct ggml_context *ctx, const char *name, hGensor gensor, bool isParam, int flag = 0);

    void InitGensor(struct ggml_context *ctx, hGensor gensor, const char *name, struct random_normal_distribution *rnd = nullptr, int flag = 0);    

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
    // Deprecated
    hGensor AddTensor(const std::string &key_, enum ggml_type tp, const SHAPE &shape, int flag = 0x0);
    hGensor AddTensor(struct ggml_context *ctx,const std::string &key_, enum ggml_type tp, const SHAPE &shape, bool isParam,int flag = 0x0);

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
                hN = std::make_shared<SLP>(this, key_ + param.title, param.shape, this, flag);
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
    virtual void SaveTrain(struct save_train_model * data, void *user_data,int flag=0x0);
    virtual void save_gguf(const char * filename, int flag){}

    friend class GeNeuron;
    friend class SLP;
    friend class LayerNormal;
    friend class NT_SAM;
    friend class SAM_encoder;
    friend class Optimizer;
    friend class OPT_Adam;
    friend class Distillation;
    friend class ConsiceDict;
    friend class GeneratOnPrompt;
    friend class NLP_AutoRegressive;   
    friend class SampLoader;
    friend class WIKI;
    friend class KVCache;
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