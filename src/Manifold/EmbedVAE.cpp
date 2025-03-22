/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  Perceptrons 
 * 
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */
#include "Neuron.hpp"
#include <set>
#include "Fish.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"
#include "../g_stddef.hpp"
#include "../lenda/kernel/SVD.hpp"

TokenEmbed::TokenEmbed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag) : GeNeuron(key_,jit, hG_, flag)    {
    auto dims = hFish->config.embeds;
    nVocab = hG_->nClass();     
    latent = hFish->config.nEmbed(-1);
    shape.clear();
    /*if(jvals.size()==2){
        shape={(int)(jvals[0]),(int)(jvals[1])};
    }else{
        shape = {nCls,C};
    }    */
    assert(latent>0);
    isAddPos = type_info[type_info.length()-1] =='+'; 
}
TokenEmbed::~TokenEmbed(){
    FREE_a(workload_indices);       FREE_a(bucket_info);
}
bool TokenEmbed::InitBucket(size_t num_c_groups,int flag){
    if (bucket_info != NULL)
        return false;
    
    assert((size_t)(B * T) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    workload_indices = new int[B * T * num_c_groups];
    bucket_info = new int4[B * T * num_c_groups];
    return true;
}
bool TokenEmbed::SetMAEC(hMAEC mae_,int flag){
    maec = mae_;
    // latent = maec->nIn;
    for(auto ac : maec->codes){ //ugly design
        SLP&up=ac->up,&down=ac->down;
        out->AddSrc({up.w,up.b,up.out,down.w,down.b,down.out});
    }
    return true;
}

bool TokenEmbed::Build(int flag){ 
    struct ggml_context * ctx = hFish->GetGGCTX(1);
    int flagW=flag,n=nVocab;

    // InitMAC();
    assert(latent>0);
    typNUMBER tpData = GTensor::tpFloatX;

    bool isTrain = hFish->isTrain();
    string sw = name+MODEL_CARD::sWeight,sb=name+".pos"; 
#ifdef _TENSOR_G_ 
    if(hFish->config.modep.isPaddedCls) {
        flagW |= GTensor::F_PADDED;
        padded_nCls = ceil(n/128.0)*128;
    }else
        padded_nCls =n;   
    w = TENSO(ctx, tpData, {latent, padded_nCls},flagW); //padded_nCls
    if(padded_nCls>n)   w->x_shape = {latent,n};
#else
    w = TENSO(ctx, tpData, {latent, n});
#endif    
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isAddPos){
        n = hFish->config.n_ctx();
        b = TENSO(ctx, tpData, {latent, n});        
        sb = "position_embd.weight";
        hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
#ifdef _TENSOR_G_        
    
    SHAPE s3={B,T,latent};
    out = std::make_shared<huTensor>(name+".batch",s3,w->type,false);    
    // hFish->InitGensor(ctx,name+".batch",out,false);    
#endif    
    return true;
}
/*
   batch(tokens) embeddings from glob token embedding(w)
*/
hGensor TokenEmbed::Interact(struct ggml_context *ctx_,hGensor tokens,int flag){
    if(tokens==nullptr)  //symbolic analysis
        return GeNeuron::Interact(ctx_,tokens,flag);
    assert(tokens->type==typNUMBER::I32);
    string sw = name+"_rows";
    hGensor cur = nullptr;

#ifdef _TENSOR_G_
    if(hFish->isSymbolic()){            
        out->AddSrc({w,tokens,b});      cur=out;
    } else{
        // cur = w->GetRow(out,tokens,b);
        FUSE_cuda(tokens,nullptr,nullptr,0x0); //nullptr,nullptr,
        cur = out;
    }   
#else
    assert(w->ne[1]==shape[0]);
    cur = ggml_get_rows(ctx_, w, tokens);       gTN(cur, sw.c_str());   
    // hFish->xn = cur; hFish->xxn = cur->grad;         //  only for debug
    if(isAddPos)        {
        cur = ggml_add(ctx_, cur, b);  
    }
#endif
    cur = AfterForward(ctx_,cur,flag);
    return cur;
}
string TokenEmbed::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s {EMBED n=%d %s}",tab,nVocab,isAddPos?"+POS":"");    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

hGensor VarCoder::ENC(const hGensor x0){
#ifdef _TENSOR_G_
    hGensor x = nullptr;    //encode*x0;
    switch(tpNorm){
    case 0:
        x = x->Relu();  
        break;
    case 1:
        x = x->Silu();  
        break;
    case 2:
        x = x->Norm(1.0e-5);  
        break;
    }
#else
    hGensor x = ggml_mul_mat(ctx, encode, x0 );    
    switch(tpNorm){
    case 0:
        x = ggml_relu(ctx, x);  
        break;
    case 1:
        x = ggml_silu(ctx, x);  
        break;
    case 2:
        x = ggml_rms_norm(ctx, x,1.0e-5);  
        break;
    }
#endif
    if(isResi)      
        resi = x;  
        
    return x;
}

hGensor VarCoder::DEC(hGensor x){
    if(down.Empty())    //decode==nullptr
        return x;
#ifdef _TENSOR_G_
    if(resi!=nullptr){
        x += resi;
    }
    // x = decode*x;
    switch(tpNorm){
    case 0:
        x = x->Relu();  
        break;
    case 1:
        x = x->Silu();  
        break;
    case 2:
        x = x->Norm(1.0e-5);  
        break;
    }
#else
    if(resi!=nullptr){
        x = ggml_add(ctx, x, resi);
    }
    x = ggml_mul_mat(ctx, decode, x );    
    switch(tpNorm){
    case 0:
        x = ggml_relu(ctx, x);  
        break;
    case 1:
        x = ggml_silu(ctx, x);  
        break;
    case 2:
        x = ggml_rms_norm(ctx, x,1.0e-5);  
    }
#endif
    return x;
}

string VarCoder::__repr__( string& suffix,string& prefix,int flag)   {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"\n%s\t[%d=>%d]%s resi=%d tpNorm=%d",tab,nTop,nBottom,prefix.c_str(),isResi,tpNorm);
    // _T_repr_(encode,tab,buf);   
    // _T_repr_(decode,tab,buf);   
    // _T_repr_(norm,tab,buf);   
    // _T_repr_(resi,tab,buf);   
    if(flag>0)
        _INFO("%s",buf); 
    return buf;
}

VarCoder::VarCoder(Fish *hG_,std::vector<int>&dims,int level,bool isR,bool isB,int tpN,int flag) 
    : nTop(dims[level]),nBottom(dims[level+1]),isResi(isR),tpNorm(tpN) {
    isBias = isB;
    name = "AE_"+std::to_string(level);
    Init(hG_,0x0);
    assert(nTop>=nBottom && nBottom>0);
    Build(flag);
    /*encode = TENSO(nullptr, typNUMBER::F32, {nTop, nBottom});     
    if(isSym)
        decode = TENSO(nullptr, typNUMBER::F32, {nBottom, nTop}); 
    else{
        decode = nullptr;
        isResi = false;
    } */           
}
MAEC::MAEC(Fish *hG_, const std::string &key_, int flag) {
    name = "MAEC_"; //+key;
    Init(hG_,0x0);
    auto dims = hFish->config.embeds;
    if(dims.size()==1)      return ;

    nIn = dims[0];      nOut=dims[dims.size()-1];
    int reserve_x = 0;
    int nMap = dims.size()-1,tpNorm=-1;       assert(nMap>0);
    bool isSymmetric = hFish->config.modep.isEmbedWeightTying;
    
    isBias = false;
    codes.clear( );
    for(int i=0;i<nMap;i++){
        hVarCoder hCoder = std::make_shared<VarCoder>(hFish,dims,i,reserve_x,isBias,tpNorm);
        codes.push_back(hCoder);            
    }
    hVarCoder first=codes[0],last=codes[codes.size()-1];
    /*if(0){
        normE.BuildX(name+"norm_E",{nIn},hFish,flag);   
        first->down.out->AddSrc({normE.w,normE.b,normE.out,normE.rstd,normE.mean});
        normD.BuildX(name+"norm_D",{nIn},hFish,flag);   
        last->up.out->AddSrc({normD.w,normD.b,normD.out,normD.rstd,normD.mean});
        normE.delta = GTensor::delta;
        normD.delta = GTensor::delta;
    }*/

    return;
}
hGensor MAEC::ENC(hGensor cur,int flag){
    if(isForward()){
        if(!normE.Empty())  {
            normE.FUSE_cuda(cur);       
            cur=normE.out;
        }
        for(auto ac : codes){
            SLP &down = ac->down;
            //    cur->PrintX<floatX>("ac_in",0,-1);
            down.Forw(down.out,cur);  
            cur = down.out;      //cur->PrintX<floatX>("ac_out",0,-1);
        }
    }else{
        for (auto it = codes.rbegin(); it != codes.rend(); ++it)  {
            hVarCoder ac = *it;        //cur->PrintX<floatX>("ac_in",0,-1);
            SLP &down = ac->down;
            assert(down.inp!=nullptr);
            down.Back(down.delta,down.inp,cur,nullptr,(float*)GTensor::buff,0);  
            cur = down.delta;      
        }        
        if(!normE.Empty())  {
            normE.FUSE_cuda(normE.inp,(float*)GTensor::buff,cur);   
            cur = normE.delta; 
        }
    }
    return cur;
}

hGensor MAEC::DEC(hGensor cur,bool isForw,int flag){
    if(isForw){     //  !=isForward()
        for(auto ac : codes){
            SLP &up = ac->up;
            //    cur->PrintX<floatX>("ac_in",0,-1);
            up.Forw(up.out,cur);  
            cur = up.out;      //cur->PrintX<floatX>("ac_out",0,-1);
        }
    }else{
        for (auto it = codes.rbegin(); it != codes.rend(); ++it)  {
            hVarCoder ac = *it;        //cur->PrintX<floatX>("ac_in",0,-1);
            SLP &up = ac->up;
            assert(up.inp!=nullptr);
            up.Back(up.delta,up.inp,cur,nullptr,(float*)GTensor::buff,0);  
            cur = up.delta;      
        }        
    }
    return cur;
}
string MAEC::__repr__( string& suffix,string& prefix,int flag)    {
    string p = prefix+"\t";
    string info=p+name+"{ bias="+std::to_string(isBias);
    info += "\n"+normE.__repr__(suffix,p,flag);
    for(auto ac : codes){
        info+=ac->__repr__(suffix,p,flag);
    }
    info += "\n"+normD.__repr__(suffix,p,flag);
    info +=prefix+"\t}\n";
    return info;
};

VarCoder::VarCoder(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : GeNeuron(key_,jit, hG_, flag) {
    if(jvals.size()>=2){
        shape={(int)(jvals[0]),(int)(jvals[1])};
    }else{
        int n_ff = hFish->config.n_ff();
        shape = {C,n_ff};
    }    
    assert(shape[0]>0 && shape[1]>0); 
    nBottom = shape[0],nTop = shape[1];
}

FFN::FFN(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : VarCoder(hG_, key_,jit, flag)     {
    remater_ffn = hFish->config.common.remater_ffn;       //false;
    tpNorm = 2;
}
bool VarCoder::Build(int flag_0)   {
    int flag = flag_0;
    if(tpNorm>0)
        norm.BuildX(name+MODEL_CARD::sNorm,{nBottom},hFish,flag);   
    up.BuildX(name+"_up",{nBottom,nTop},hFish,flag | F_DELTA);  
    down.BuildX(name+"_down",{nTop,nBottom},hFish,flag | F_DELTA); 
    if(!isBias)   {
        up.b=nullptr;      down.b=nullptr;
    }
    
    return true;
}
bool FFN::Build(int flag_0)   {
    SHAPE sp={shape[0]},sp3,sp2;
    struct ggml_context * ctx_ = hFish->GetGGCTX(1);
    bool isTrain = hFish->isTrain();
    int flag = flag_0;
    latent=shape[1];
#ifdef _TENSOR_G_    
    // flag |= GeNeuron::F_BIAS; 
    assert(C==shape[0]);
    sp3 = {B,T,latent};
    sp2 = {B,T,C};
    // relu.out = std::make_shared<huTensor>(name+"_relu",sp3,GTensor::tpFloatX,false);   
#else    
    gate.BuildX(name+"_gate",{shape[0],shape[1]},hFish,flag);
#endif
         //layer->ffn_norm.sT="f";
    VarCoder::Build(flag);
    // up.BuildX(name+"_up",{shape[0],latent},hFish,flag);  
    // down.BuildX(name+"_down",{latent,shape[0]},hFish,flag);        
#ifdef _TENSOR_G_
    if(GTensor::scratch_ff1!=nullptr)   {
        assert(GTensor::scratch_ff1->size()>=up.out->size());
        // gelu_fusion = 1;     //  0 = none, 1 = forward, 2 = forward+backward (-1 => per-GPU default)
        if(remater_ffn){
            BIT_SET(up.out->flags,GTensor::F_NOALLOC); 
            out = std::make_shared<huTensor>(name+"_out",sp2,GTensor::tpFloatX,false);  
        }else{
            //out would be norm.out
        }
    }
           
    up.w->residual_scale = hFish->config.common.residual_scale;
    BIT_SET(down.out->flags,GTensor::F_NOALLOC);
#endif
    return true;
}

hGensor FFN::Interact(struct ggml_context * ctx_,hGensor inpL,int flag){    
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Interact(ctx_,nullptr,flag);
    }
    hGensor cur = nullptr;
    int iRet=-1;
#ifdef _TENSOR_G_
    cur = inpL;
    hGensor lastResi = inpL;
    if(hFish->isSymbolic()){      
        // out = inpL >> up >> relu >> down >> norm;  
        inpL >> up >> down >> norm ;  
        if(remater_ffn){
            norm.out>>this; //out->AddSrc(GENSOR_OP::Inst(norm.out),0x0);      
        }else
            out = norm.out;     //to save memory    
        cur = out;
    } else{ //  high performance fused operator
        float *inp1=TO<float>(down.out);
        FUSE_cuda(cur,nullptr,0x0); //nullptr,nullptr,
        cur = norm.out;
    } 
#else
    cur = norm.Interact(ctx_,inpL,0x0);
    gTN(cur,"%s.ffn_norm",name.c_str());      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = up.Interact(ctx_,cur,0x0);
    gTN(cur,"%s.ffn_up",name.c_str());//cb(cur, "ffn_up", il);
    
    // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
    cur = ggml_silu(ctx_, cur);                
    gTN(cur,"%s.ffn_silu",name.c_str());    
    if(!gate.Empty()){
        hGensor g = gate.Interact(ctx_,inpL,0x0);
        cur = ggml_mul(ctx_, cur, g);
        gTN(cur,"%s.ffn_gate",name.c_str());
    }    
    cur = down.Interact(ctx_,cur,0x0);
    gTN(cur,"%s.ffn_down",name.c_str());    //cb(cur, "ffn_down", il);
    cur = ggml_add(ctx_, cur, inpL);// add the input
    cur = AfterForward(ctx_,cur,flag);
    // if(!name.empty()){
    //     strcpy(cur->name,"");   gTN(cur,"%s",name.c_str());
    // }
    
#endif
    return cur;
}
string FFN::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s %s {l=%d}",tab,name.c_str(),shape[1]);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};