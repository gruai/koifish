
/**
 *  Copyright 2023-2024 by Grusoft 
 *  
 *  Perceptrons 
 * 
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */
#include <set>
#include "Fish.hpp"
#include "gLLM.hpp"
#include "../g_stddef.hpp"

hNeuron GeNeuron::MakeInstance(Fish *hG_,struct ggml_context *ctx,const string& guid,const string& typ_0,const JSON& j,int flag){
    hNeuron nn=nullptr;
    string typ = typ_0;
    std::transform(typ.begin(), typ.end(), typ.begin(), ::toupper);
    vector<double> vals = j.get<vector<double> >();
    assert(vals.size()>=2);
    SHAPE shape={(int)(vals[0]),(int)(vals[1])}; 

    if(typ.rfind("EMBED", 0) == 0){
        nn = std::make_shared<Embed>();
    }else if(typ.rfind("LINEAR", 0) == 0){
        
        nn = std::make_shared<SLP>(hG_, guid, shape, flag);
    }else if(typ.rfind("QKV", 0) == 0){
        nn = std::make_shared<SelfAttention>();
    }else if(typ.rfind("DROPOUT", 0) == 0){
        nn = std::make_shared<Drop>();
    }else if(typ.rfind("SILU", 0) == 0){
        nn = std::make_shared<Relu>();
    }else{
        assert(0);
    }

    // nn->Init(nullptr,0x0);
    // nn->Build(name,shape,0x0);
    assert(nn->isValid());
    return nn;
}

GeNeuron::GeNeuron(const std::string &key_,SHAPE shape_, Fish *hG_, int flag) : name(key_),shape(shape_), hOrg(hG_) { 
    // name = "GeNeuron"; 
}

void LayerNormal::Build(const std::string&key_,const SHAPE& shape,int flag)    {
    isBias = hOrg->isBias;
    name = key_;
    struct ggml_context * ctx = hOrg->ctx;
    assert(shape.size()==1 && shape[0]>0 );
    int nIn=shape[0];
    w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
    if(isBias)  b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
    string sw = key_+".weight",sb=key_+".bias";
    bool isTrain = hOrg->isTrain();
    hOrg->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isBias)  hOrg->InitGensor(ctx,sb.c_str(),b,isTrain);
    // hOrg->InitGensor(ctx,w,sw.c_str(),hOrg->rnd);
    // hOrg->InitGensor(ctx,b,sb.c_str(),hOrg->rnd);
}

hGensor LayerNormal::Forward(struct ggml_context * ctx0,hGensor cur,int flag)    {    
    float f_norm_eps = hOrg->hparams.f_norm_eps;
    assert(cur!=nullptr);
    // TODO: implement ggml_norm backward
    // cur = ggml_norm(ctx0, cur, f_norm_eps);  
    const string prefix = sT+"."+cur->name;
    hGensor cur_norm = ggml_rms_norm(ctx0, cur, f_norm_eps);     
    ggml_set_name(cur_norm,_NAM_("%s_rms",prefix.c_str()));  
    hGensor  t03 = w;
    if(hOrg->isTrain()){
        t03 = ggml_repeat(ctx0, w, cur_norm);          
        ggml_set_name(t03,_NAM_("%s.r",w->name));    
        hOrg->Gensor2Map(t03);  
    }
    hGensor curw = ggml_mul(ctx0, cur_norm, t03);   
    ggml_set_name(curw,_NAM_("%s*w",prefix.c_str()));       
    if(b!=nullptr){
        if(hOrg->isTrain())
            cur = ggml_add(ctx0, curw, b); 
        else
            cur = ggml_add_inplace(ctx0, curw, b); 
        ggml_set_name(cur,_NAM_("%s+b",prefix.c_str()));   
    }
        
    return cur;
}
size_t LayerNormal::nElem()  {
    size_t nX=0; 
    nX += ggml_nelements(w);
    if(b!=nullptr)      
        nX += ggml_nelements(b);
    return nX;
}

hGensor GeNeuron::Forward(struct ggml_context *ctx_compute,hGensor cur,int flag){
    int tp=0;
    _INFO("\t %s\n",name.c_str());
    hGensor inp = cur;
    switch(tp){
    case 1:
        // cur = cur+inp;
        break;

    default:
        return cur;  
    }

    return cur;
}