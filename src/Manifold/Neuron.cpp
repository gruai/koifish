
/**
 *  Copyright 2023-2025 by Grusoft  
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
#include "../lenda/kernel/SVD.hpp"

hNeuron GeNeuron::MakeInstance(Fish *hG_,struct ggml_context *ctx,const string& guid,JSON::const_iterator jit,int flag){
    hNeuron nn=nullptr;    
    // assert(j.is_object());
    auto typ_0 = jit.key();  
    auto v = jit.value();
    
    string typ = typ_0;
    std::transform(typ.begin(), typ.end(), typ.begin(), ::toupper);

    if(typ.rfind("EMBED", 0) == 0){
        nn = std::make_shared<Embed>(hG_, guid, jit, flag);
    }else if(typ.rfind("LINEAR", 0) == 0){        
        nn = std::make_shared<SLP>(hG_, guid, jit, flag);
    }else if(typ.rfind("GAU", 0) == 0){        
        nn = std::make_shared<GatedAttention>(hG_, guid, jit, flag);
    }/*else if(typ.rfind("QKV_ROPE", 0) == 0){
        nn = std::make_shared<QKV_rope>(hG_, guid, jit, flag);
    }*/else if(typ.rfind("BROWN", 0) == 0){
        nn = std::make_shared<BROWN_attn>(hG_, guid, jit, flag);
    }else if(typ.rfind("QKV", 0) == 0){
        nn = std::make_shared<SelfAttention>(hG_, guid, jit, flag);
    }else if(typ.rfind("DROPOUT", 0) == 0){
        nn = std::make_shared<Drop>(hG_, guid, jit, flag);
    }else if(typ.rfind("SILU", 0) == 0){
        nn = std::make_shared<Relu>(hG_, guid, jit, flag);
    }else if(typ.rfind("FFN", 0) == 0){
        nn = std::make_shared<FFN>(hG_, guid, jit, flag);
    }else if(typ.rfind("NORMAL", 0) == 0){
        nn = std::make_shared<LayerNormal>(hG_, guid, jit, flag);
    }else if(typ.rfind("CLASIFY", 0) == 0){
        nn = std::make_shared<OutCLS>(hG_, guid, jit, flag);
    }else{
        _ERROR("%s failed@[%s]",__func__,typ.c_str());
        assert(0);
    }

    // nn->Init(nullptr,0x0);
    nn->Build(0x0);
    assert(nn->isValid());
    return nn;
}

GeNeuron::GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag) : name(key_), hFish(hG_),ID(0) {    
try{   
    Init(hG_,0x0);

    if((*jit).contains(std::string{ "#id" })){
        ID = (*jit)["id"];
    }
    type_info = jit.key(); 
    if(jit->is_array()){
        jvals = jit->get<vector<double> >();
        // assert(jvals.size()>=2);
    }else{
        
    }
}catch(...){
    assert(0);
} 
}
void GeNeuron::Init(Fish *hG_, int flag) {    
    hFish=hG_;   
    auto& hparams = hG_->hparams; 
    n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_embd=hparams.n_embd;
    n_embd_head = hparams.n_embd_head();
    n_head=hparams.n_head();
    assert(n_embd_head*n_head==n_embd);
}

Ganglia::Ganglia(Fish *hG_,const string& key_,std::vector<hNeuron>& ns_,int flag) : ns(ns_)  {
    name="{"+key_+"}"; 
    hFish=hG_;
}



Relu::Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : GeNeuron(key_,jit, hG_, flag){

}
bool Relu::Build(int flag)   {
    return true;
};
hGensor Relu::Interact(struct ggml_context *ctx_,hGensor cur,int flag){
    return cur;
}

Drop::Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : GeNeuron(key_,jit, hG_, flag){
    
}
bool Drop::Build(int flag)   {
    return true;
};
hGensor Drop::Interact(struct ggml_context *ctx_,hGensor cur,int flag){
    return cur;
}

Embed::Embed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag) : GeNeuron(key_,jit, hG_, flag)    {
    int nCls = hG_->nClass();
    if(jvals.size()==2){
        shape={(int)(jvals[0]),(int)(jvals[1])};
    }else{
        shape = {nCls,n_embd};
    }
    
    assert(shape[1]>0);
    isAddPos = type_info[type_info.length()-1] =='+'; 
}
bool Embed::Build(int flag){
    assert(shape.size()==2);    
    struct ggml_context * ctx = hFish->GetGGCTX(1);
    int n=shape[0],latent=shape[1];
    if(n<=0){
        n = hFish->hparams.n_ctx();
    }
    assert(n>0 && latent>0);
    GTensor::tpDATA tpData = GTensor::tpFloatX;

    bool isTrain = hFish->isTrain();
    string sw = name+sWeight,sb=name+".pos"; 
    w = TENSO(ctx, tpData, {latent, n});
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isAddPos){
        n = hFish->hparams.n_ctx();
        b = TENSO(ctx, tpData, {latent, n});
        hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
#ifdef _TENSOR_CUD_        
    SHAPE s3={GTensor::B,GTensor::T,latent};
    out = std::make_shared<cuTensor>(name+".batch",s3,w->type,false);    
    // hFish->InitGensor(ctx,name+".batch",out,false);    
#endif    
    return true;
}
/*
   batch(tokens) embeddings from glob token embedding(w)
*/
hGensor Embed::Interact(struct ggml_context *ctx_,hGensor tokens,int flag){
    if(tokens==nullptr)  //symbolic analysis
        return GeNeuron::Interact(ctx_,tokens,flag);
    assert(tokens->type==GGML_TYPE_I32);
    string sw = name+"_rows";
    hGensor cur = nullptr;

#ifdef _TENSOR_CUD_
    if(hFish->isSymbolic()){            
        out->AddSrc({w,tokens,b});      cur=out;
    } else{
        cur = w->GetRow(out,tokens,b);
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
string Embed::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s {EMBED n=%d %s}",tab,shape[0],isAddPos?"+POS":"");    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

FFN::FFN(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : GeNeuron(key_,jit, hG_, flag)     {
    if(jvals.size()>=2){
        shape={(int)(jvals[0]),(int)(jvals[1])};
    }else{
        int n_ff = hFish->hparams.n_ff();
        shape = {n_embd,n_ff};
    }
    
    assert(shape[0]>0 && shape[1]>0);
    // up.Init(hG_,flag);       down.Init(hG_,flag);       relu.Init(hG_,flag); 
}
bool FFN::Build(int flag_0)   {
    SHAPE sp={shape[0]},sp3;
    struct ggml_context * ctx_ = hFish->GetGGCTX(1);
    bool isTrain = hFish->isTrain();
    int flag = flag_0,latent=shape[1];
#ifdef _TENSOR_CUD_
    flag |= GeNeuron::F_BIAS;       
    latent*=4;  //from kGPT
    assert(GTensor::C==shape[0]);
    sp3={GTensor::B,GTensor::T,latent};
    relu.out = std::make_shared<cuTensor>(name+"_relu",sp3,GTensor::tpFloatX,false);
    // hFish->InitGensor(nullptr,name+"_relu",relu.out,false);
#else    
    gate.BuildX(name+"_gate",{shape[0],shape[1]},hFish,flag);
#endif
    norm.BuildX(name+sNorm,sp,hFish,flag);        //layer->ffn_norm.sT="f";
    up.BuildX(name+"_up",{shape[0],latent},hFish,flag);    
    down.BuildX(name+"_down",{latent,shape[0]},hFish,flag);    
    return true;
}

hGensor FFN::Interact(struct ggml_context * ctx_,hGensor inpL,int flag){    
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Interact(ctx_,nullptr,flag);
    }
    hGensor cur = nullptr;
    int iRet=-1;
#ifdef _TENSOR_CUD_
    cur = inpL;
    hGensor lastResi = inpL;
    if(hFish->isSymbolic()){            
//         out = std::make_shared<cuTensor>(cur->shape,cur->type);        cur = out;        
//         _ffn = std::make_shared<cuTensor>(cur->shape,cur->type);    //B * T* C
//         cur = norm.Interact(ctx_,_ffn,0x0); 
//         //floatX* l_fch = acts.fch + l * B * T * 4*C;
        SHAPE sp={cur->shape[0],cur->shape[1],4*cur->shape[2]};    //B * T* 4*C
//         _latent = std::make_shared<cuTensor>(sp,cur->type);
// // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size

        out = inpL >> up >> relu >> down >> norm ;        
        cur = out;
    } else{ //  high performance fused operator
        iRet = FUSE_FFN(down.out,cur,up.out,up.w,up.b,relu.out,down.w,down.b,gelu_fusion,0);  cur=down.out; 
        cuLiteTest(GTensor::B,GTensor::T,GTensor::C,1);
        // iRet = FUSE_ResiNormal(out,down.out,lastResi,norm.out,norm.mean,norm.rstd,norm.w,norm.b,0x0);   
        // cuLiteTest(GTensor::B,GTensor::T,GTensor::C,1);  
        // cur = out;
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
    sprintf(buf+strlen(buf),"%s FFN",tab);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

MOE::MOE(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : GeNeuron(key_,jit, hG_, flag)     {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    isSiLU = true;
    //[ctx, E/H, H, n_batch
    // up.Init(hG_,flag);       down.Init(hG_,flag);       relu.Init(hG_,flag); 
}
bool MOE::Build(int flag)   {
    string sw = name+sWeight,sb=name+".bias";
    bool isTrain = hFish->isTrain();    
    int nIn=shape[0];
    struct ggml_context * ctx = hFish->GetGGCTX();
    //  [ctx, E/H, H, n_batch); ]
    w = TENSO(ctx, GGML_TYPE_F32, {n_embd_head,1,n_head,n_batch});
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
      
    return true;
}
hGensor MOE::Forward2(struct ggml_context * ctx_,hGensor inpL,hGensor wBase,int flag){   
    int n0=inpL->ne[0],n1=inpL->ne[1],n2=inpL->ne[2],n3=inpL->ne[3]; 
    hGensor cur = BeforeForward(ctx_,inpL,flag);
    if(cur==nullptr)  //    some operation like symolic analysis     
        return cur; 
#ifdef _TENSOR_CUD_
#else
    hGensor wp_ = ggml_mul_mat(ctx_,w,wBase ); //ggml_reshape_2d(ctx,v3,N, n_batch*n_embd)  
    gTN(wp_,"%s.trans",name.c_str());
    assert(wp_->ne[0]==1);
    wp_ = ggml_reshape_3d(ctx_, wp_, n1,n2,n3);   
    // w_ = ggml_reshape_2d(ctx_, w_, n_ctx,n_batch);   
    if(isSiLU){ //maybe useful
        wp_ = ggml_silu(ctx_,wp_);
    } 
    hGensor probs = ggml_soft_max(ctx_,wp_);             gTN(probs,"%s.probs",name.c_str());
    probs = ggml_reshape_4d(ctx_, wp_, 1,n1,n2,n3);  
    probs = ggml_repeat(ctx_, probs, cur); 
    // 
    cur = ggml_mul(ctx_,cur,probs);                     gTN(cur,"%s.moe",name.c_str());        
    cur = AfterForward(ctx_,cur);
    
#endif
    return cur;
}
string MOE::__repr__( string& suffix,string& prefix,int flag)    {
    return _repr_1(suffix,prefix,"MOE");
};

OutCLS::OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : GeNeuron(key_,jit, hG_, flag){
    int nEmbd=hFish->hparams.n_embd;
    // _target = hFish->Target();   //null now
    nCls=hFish->nClass();
    padded_nCls = ceil(nCls/128.0)*128;
    shape={nEmbd,nCls};
}
bool OutCLS::Build(int flag)   {
    SHAPE sp={shape[0]};
#ifdef _TENSOR_CUD_
    SHAPE sp2={GTensor::B,GTensor::T};
    target = std::make_shared<cuTensor>("target",sp2,GGML_TYPE_F32,false); 
    // hFish->InitGensor(nullptr,"target",target,false);           
    hFish->target_probs = target; 
    out = std::make_shared<cuTensor>("loss",sp2,GGML_TYPE_F32,false);     
    // hFish->InitGensor(nullptr,"loss",out,false);                
    hFish->loss = out;//
#else
    norm.BuildX(name+sNorm,sp,hFish,0x0);        //layer->ffn_norm.sT="f";
#endif
    string sCls = "";   //  ".cls"
    proj.BuildX(name+".probability",{shape[0],shape[1]},hFish,flag); 
    name += ".cls";      
    return true;
}
hGensor OutCLS::Interact(struct ggml_context * ctx_,hGensor inpL,int flag)    {
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Interact(ctx_,nullptr,flag);
    }
    int n_batch=hFish->hparams.n_batch(),n_ctx=hFish->hparams.n_ctx();
    hGensor cur = nullptr;
#ifdef _TENSOR_CUD_
    if(hFish->isSymbolic()){ 
        SHAPE sp={GTensor::B,GTensor::T},sp3={GTensor::B,GTensor::T,padded_nCls};
        inpL >> proj;         
        out->AddSrc({proj.out,target});            assert(target!=nullptr);
        hFish->preLogits = proj.out;
    } else{     
        assert(proj.b==nullptr);
        mean_loss = proj.out->FusedLoss(out,target,inpL,proj.w,nCls,0x0);        
    }
    cur = out;      return cur;
#else    
    cur = norm.Interact(ctx_,inpL,0x0);    
    gTN(cur,"result_norm");      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = proj.Interact(ctx_,cur,0x0);
    gTN(cur,"result_output");//cb(cur, "ffn_up", il);    
    // cur = ggml_silu(ctx_, cur);        
    if(n_batch>1){
        cur = ggml_reshape_3d(ctx_, cur, nCls, n_ctx, n_batch);
    }
    //  Need loss node from BuildLoss( struct ggml_context * ctx,hGensor cur,int flag)
#endif            
    cur = AfterForward(ctx_,cur,flag);           
    return cur;    
}

SLP::SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)      : GeNeuron(key_,jit, hG_, flag)    {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    // compression = hFish->params.compression;
    // Build(key_, shape_, flag);
}
bool SLP::Build(int flag)      {
    isBias = hFish->isBias || BIT_TEST(flag,F_BIAS);
    // shape = shape_;
    struct ggml_context *ctx = hFish->GetGGCTX();
    GTensor::tpDATA tpData = GTensor::tpFloatX;
    int bFlag = GTensor::F_PARAM;
    int nIn=shape[0],nOut=shape[1];
    if(shape.size()==2){    
        assert(shape[0]>0 && shape[1]>0);        
        w = TENSO(ctx, tpData, {nIn, nOut});
        if(isBias)  b = TENSO(ctx, tpData, {nOut});
    }else if(shape.size()==4)  {
        w = TENSO(ctx, tpData, shape);
        if(isBias)  b = TENSO(ctx, tpData,            {1,1,shape[3]});
    }else{
        assert(0);
    }

    string sw = name+sWeight,sb=name+".bias",so=name+".out"; 
    bool isTrain = hFish->isTrain();
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isBias)  hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
#ifdef _TENSOR_CUD_        
    SHAPE s3={GTensor::B,GTensor::T,nOut};
    out = std::make_shared<cuTensor>(so,s3,tpData,false);    
    // hFish->InitGensor(ctx,so.c_str(),out,false);    
#endif    
    if(compression==SVD){        
        assert(shape.size()==2);
        // SVD(w);
    }
    return true;
}
hGensor SLP::Interact(struct ggml_context * ctx0,hGensor cur,int flag)    {
    string prefix = ""; //sT+".";   //+
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::Interact(ctx0,cur,flag);
    }else{
        prefix = prefix+cur->name;
    }
    
    // compression = SVD_a; //SVD_a;    //SKIP;//hFish->params.compression;
    // if(1)   {   //only for debug
    //     float a[6*5] = {				
    //         8.79,  9.93,  9.83, 5.45,  3.16,
    //         6.11,  6.91,  5.04, -0.27,  7.98,
    //         -9.15, -7.93,  4.86, 4.85,  3.01,
    //         9.57,  1.64,  8.83, 0.74,  5.80,
    //         -3.49,  4.02,  9.80, 10.00,  4.27,
    //         9.84,  0.15, -8.99, -6.02, -5.31
    //     };
    //     auto svd=std::make_shared<LoSVD<float>>(a,6,5,5,0); //1.0e-3
    //     svd->Build( );
    // }
#ifdef _TENSOR_CUD_
    
#else
    if(compression==SVD || compression==SVD_a)        {   //A=UDV
        int nIn=shape[0], nOut=shape[1], rank = min(64,min(nIn,nOut)/10);
        float *A=new float[nIn*nOut];
        switch(w->type){
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
                break;
            case GGML_TYPE_F32:
                break;
            default:
                assert(0);
        }
        ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
        auto svd=std::make_shared<LoSVD<float>>(A,nIn,nOut,rank,0); //1.0e-3
        if(!svd->Build( ))  {
            compression = SKIP;
        }else{
            //GGML_TYPE_F16 tensor would call ggml_vec_dot_f16 with GGML_SIMD acceleration
            if(compression==SVD_a)  {   //keep same graph
                float *approx = svd->Approx( );
                ggml_fp32_to_fp16_row(approx,(ggml_fp16_t*)w->data,nIn*nOut);
            }else{  
                u = TENSO(ctx0, GGML_TYPE_F32, {nIn, rank});   
                memcpy(u->data,svd->U(),sizeof(float)*nIn*rank);     
                memcpy(v->data,svd->V(),sizeof(float)*nIn*rank); 
                v = TENSO(ctx0, GGML_TYPE_F32, {rank, nOut});
                // s = TENSO(ctx, GGML_TYPE_F16, nIn, nOut);
                
                cur = ggml_mul_mat(ctx0, u, cur);    
                cur = ggml_mul_mat(ctx0, v, cur);                      
            }       
        }
        delete[] A;
    }
    if(compression == SKIP || compression==SVD_a)  {
        cur = ggml_mul_mat(ctx0, w, cur);           //create temp GGML_OP_MUL_MAT tensor:  result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        gTN(cur,"%s*w",prefix.c_str());
    }
    if(b!=nullptr)  {        
        cur = ggml_add(ctx0, cur, b); 
        gTN(cur,"%s+b",prefix.c_str());
            // cur = ggml_add_inplace(ctx0, cur, b); 
    }
#endif
    cur = AfterForward(ctx0,cur,flag);
    // if(!name.empty()){
    //     gTN0(cur,"%s",name.c_str());
    // }
        
    return cur;
}

ROPE::ROPE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag)    : GeNeuron(key_,jit, hG_, flag) {
    assert(jvals.size()>=1 && jvals[0]>0);
    shape={(int)(jvals[0])};
    /*auto& hparams = hG_->hparams;
    n_rot = hparams.n_rot;
    rope_freq_base  = hparams.rope_freq_base;
    rope_freq_scale = hparams.rope_freq_scale;  
    KQ_pos = hFish->KQ_pos;*/
}
/*
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
*/
bool ROPE::Build(int flag)    {
    auto& hparams = hFish->hparams;
    n_rot = hparams.n_rot;
    rope_freq_base  = hparams.rope_freq_base;
    rope_freq_scale = hparams.rope_freq_scale;  
    KQ_pos = hFish->KQ_pos;    
    shape = {n_embd_head, n_head, n_ctx, n_batch};
    return true;
}

hGensor ROPE::Interact(struct ggml_context * ctx_,hGensor inpL,int flag)    {   
    hGensor cur = BeforeForward(ctx_,inpL,flag);
    if(cur==nullptr)  //    some operation like symolic analysis     
        return cur; 
    assert(cur->ne[0]==shape[0] && cur->ne[1]==shape[1] && cur->ne[2]==shape[2] && cur->ne[3]==shape[3]);
    string nam0 = name+"."+sT;
    // hGensor  t05 = w==nullptr ? cur : ggml_mul_mat(ctx_, w, cur);         
    // gTN(t05,"%s*w",name.c_str());   
    // hGensor  t06 = ggml_reshape_4d(ctx_, cur,shape[0],shape[1],shape[2],shape[3]); //n_embd_head, n_head, N, n_batch    
    // gTN(t06,"%s$",name.c_str());   //gTN(t06, "t06");            
    const int rope_mode = 0;
#ifdef _TENSOR_CUD_
#else
    hGensor  t07 = n_embd_head==1 ? cur :
        ggml_rope_ext(ctx_, cur, KQ_pos, nullptr, n_rot, rope_mode, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    gTN(t07,"%s_rope",nam0.c_str()); 
    // CYS_0826 hGensor  t07 = ggml_rope_custom(ctx,t06, KQ_pos, n_rot, 0, n_ctx, 0,rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f);
    if(flag==0){
        hGensor  t13 = ggml_permute      (ctx_, t07, 0, 2, 1, 3);    //  [24,6,512,32] => [24,512,6,32]
        gTN(t13,"%s_0213",t07->name);
        return t13;        
    }else{
        return t07;
    }
     
    cur = AfterForward(ctx_,cur);
#endif
    return cur;
}
string ROPE::__repr__( string& suffix,string& prefix,int flag)    {
    return _repr_1(suffix,prefix,"ROPE");
};

LayerNormal::LayerNormal(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag)    : GeNeuron(key_,jit, hG_, flag) {
    assert(jvals.size()>=1 && jvals[0]>0);
    shape={(int)(jvals[0])};
}
/*
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
*/
bool LayerNormal::Build(int flag)    {
    isBias = hFish->isBias || BIT_TEST(flag,F_BIAS);
    // name = key_;
    struct ggml_context * ctx = hFish->GetGGCTX();
    assert(shape.size()==1 && shape[0]>0 );
    string sw = name+sWeight,sb=name+".bias";
    bool isTrain = hFish->isTrain();    
    int nIn=shape[0];
    if(isAffineTrans){
        w = TENSO(ctx, GTensor::tpFloatX, {nIn},GTensor::F_PARAM);
        hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    }
    if(isBias)  {
        b = TENSO(ctx, GTensor::tpFloatX, {nIn},GTensor::F_PARAM);
        hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
#ifdef _TENSOR_CUD_        
    assert(nIn==GTensor::C);
    SHAPE sp={GTensor::B,GTensor::T},sp3={GTensor::B,GTensor::T,nIn};
    out = std::make_shared<cuTensor>(name+".out",sp3,GTensor::tpFloatX,false);    
    // hFish->InitGensor(ctx,name+".out",out,false);    
    mean = std::make_shared<cuTensor>(name+".mean",sp,GGML_TYPE_F32,false);       
    rstd = std::make_shared<cuTensor>(name+".rstd",sp,GGML_TYPE_F32,false); 
    // hFish->InitGensor(ctx,name+".mean",mean,false);    
    // hFish->InitGensor(ctx,name+".rstd",rstd,false);    
#else
 
#endif
    return true;
}



hGensor LayerNormal::Interact(struct ggml_context * ctx0,hGensor cur,int flag)    {   
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::Interact(ctx0,cur,flag);
    } 

    float f_norm_eps = hFish->hparams.f_norm_eps;
    assert(cur!=nullptr);
    // TODO: implement ggml_norm backward
    // cur = ggml_norm(ctx0, cur, f_norm_eps);  
    const string prefix = sT+"."+cur->name;
#ifdef _TENSOR_CUD_
    if(isForward()){
        if(hFish->isSymbolic()){            
            // out = std::make_shared<cuTensor>(cur->shape,cur->type);        cur = out;
  
            // out->AddSrc({cur,w,b});   
            // mean->AddSrc({cur,w,b});   
            // rstd->AddSrc({cur,w,b});   
        } else{
            cur = cur->Normal(out,mean,rstd,w,b);            
        } 
    }else{
        cur = cur->Normal(out,mean,rstd,w,b,false);   
    }
#else
    hGensor cur_norm = ggml_rms_norm(ctx0, cur, f_norm_eps);     
    ggml_set_name(cur_norm,_NAM_("%s_rms",prefix.c_str()));  
    hGensor  t03 = w;
    if(hFish->isTrain()){
        t03 = ggml_repeat(ctx0, w, cur_norm);          
        ggml_set_name(t03,_NAM_("%s.r",w->name));    
        // hFish->gensors.Insert(t03);  
    }
    hGensor curw = ggml_mul(ctx0, cur_norm, t03);   
    ggml_set_name(curw,_NAM_("%s*w",prefix.c_str()));       
    if(b!=nullptr){
        if(hFish->isTrain())
            cur = ggml_add(ctx0, curw, b); 
        else
            cur = ggml_add_inplace(ctx0, curw, b); 
        ggml_set_name(cur,_NAM_("%s+b",prefix.c_str()));   
    }else{
        cur = curw;
    }
    if(!name.empty()){
        strcpy(cur->name,"");   gTN(cur,"%s",name.c_str());
    }
#endif     
    return cur;
}
size_t LayerNormal::nElem()  {
    size_t nX=0; 
    nX += tELEM(w);
    if(b!=nullptr)      
        nX += tELEM(b);
    return nX;
}

hGensor GeNeuron::Backward(void *user_ctx_,hGensor cur,int flag)    {
    return nullptr;
}
hGensor GeNeuron::Interact(struct ggml_context *ctx_,hGensor cur,int flag){
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
hGensor GeNeuron::BeforeForward(struct ggml_context *ctx_,hGensor cur,int flag){
    int tp=0;
    if(cur==nullptr){
        _INFO("\t %s\n",name.c_str());
    }
    return cur;
}
hGensor GeNeuron::AfterForward(struct ggml_context *ctx_,hGensor cur,int flag){
    if(!name.empty()){
        gTN0(cur,"%s",name.c_str());
    }
    return cur;
}

void GeNeuron::BuildX(const std::string &key_, const SHAPE &shp_, Fish *hG_, int flag){
    if(hFish==hG_ && shp_==shape && name==key_){ //
        _INFO("%s is alread build!!!\n",name.c_str());
        assert(0);   return;
    }
    assert(hG_!=nullptr);
    Init(hG_,flag);

    name = key_;
    shape = shp_;
    
    bool bRet = Build(flag);
    assert(bRet);
}

bool GeNeuron::isValid()  {   
    if(w==nullptr)
        return false;
    return true;    
}

bool GeNeuron::isForward()  {   
    assert(hFish!=nullptr);
    bool isForward = !hFish->hOPT->isBackward;
    return isForward;    
}
#ifdef _TENSOR_CUD_
hGTensor operator>>(hGTensor t, const LayerNormal& norm){
    assert(t!=nullptr && norm.out!=nullptr);

    norm.out->AddSrc({t,norm.w,norm.b});
    // norm.mean->AddSrc({t,norm.w,norm.b});    //???
    // norm.rstd->AddSrc({t,norm.w,norm.b});    //???

    return norm.out;
}
hGTensor operator>>(hGTensor t, const SLP& slp){
    assert(t!=nullptr && slp.out!=nullptr);
    slp.out->AddSrc({t,slp.w,slp.b});
    return slp.out;
}
hGTensor operator>>(hGTensor t, const Relu& relu){
    assert(t!=nullptr && relu.out!=nullptr);
    relu.out->AddSrc({t});
    return relu.out;
}
#endif