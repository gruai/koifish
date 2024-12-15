
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
hGensor Relu::Forward(struct ggml_context *ctx_,hGensor cur,int flag){
    return cur;
}

Drop::Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : GeNeuron(key_,jit, hG_, flag){
    
}
bool Drop::Build(int flag)   {
    return true;
};
hGensor Drop::Forward(struct ggml_context *ctx_,hGensor cur,int flag){
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
    struct ggml_context * ctx = hFish->GetCTX(1);
    int n=shape[0],latent=shape[1];
    if(n<=0){
        n = hFish->hparams.n_ctx();
    }
    assert(n>0 && latent>0);

    bool isTrain = hFish->isTrain();
    string sw = name+sWeight,sb=name+".pos"; 
    w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, n);
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isAddPos){
        n = hFish->hparams.n_ctx();
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, n);
        hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
    
    return true;
}
hGensor Embed::Forward(struct ggml_context *ctx_,hGensor cur,int flag){
    if(cur==nullptr)  //symbolic analysis
        return GeNeuron::Forward(ctx_,cur,flag);
    
    string sw = name+"_samp";
    cur = ggml_get_rows(ctx_, w, cur);    gTN(cur, sw.c_str());   
    if(isAddPos){
        cur = ggml_add(ctx_, cur, b);  
    }
    if(!name.empty()){ //"inp_embd"
        gTN0(cur,"%s",name.c_str());
    }
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
bool FFN::Build(int flag)   {
    SHAPE sp={shape[0]};
    struct ggml_context * ctx_ = hFish->GetCTX(1);
    bool isTrain = hFish->isTrain();
    norm.BuildX(name+sNorm,sp,hFish,0x0);        //layer->ffn_norm.sT="f";
    up.BuildX(name+"_up",{shape[0],shape[1]},hFish,flag);
    gate.BuildX(name+"_gate",{shape[0],shape[1]},hFish,flag);
    down.BuildX(name+"_down",{shape[1],shape[0]},hFish,flag);    

    // gate = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, shape[0],shape[1]);   
    // hFish->InitGensor(ctx_,name+"_gate",gate,isTrain);
    return true;
}
hGensor FFN::Forward(struct ggml_context * ctx_,hGensor inpL,int flag){    
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_,nullptr,flag);
    }

    hGensor cur = norm.Forward(ctx_,inpL,0x0);
    gTN(cur,"%s.ffn_norm",name.c_str());      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = up.Forward(ctx_,cur,0x0);
    gTN(cur,"%s.ffn_up",name.c_str());//cb(cur, "ffn_up", il);
    
    // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
    cur = ggml_silu(ctx_, cur);                
    gTN(cur,"%s.ffn_silu",name.c_str());    
    if(!gate.Empty()){
        hGensor g = gate.Forward(ctx_,inpL,0x0);
        cur = ggml_mul(ctx_, cur, g);
        gTN(cur,"%s.ffn_gate",name.c_str());
    }    
    cur = down.Forward(ctx_,cur,0x0);
    gTN(cur,"%s.ffn_down",name.c_str());    //cb(cur, "ffn_down", il);
    cur = ggml_add(ctx_, cur, inpL);// add the input
    cur = AfterForward(ctx_,cur,flag);
    // if(!name.empty()){
    //     strcpy(cur->name,"");   gTN(cur,"%s",name.c_str());
    // }
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
    struct ggml_context * ctx = hFish->GetCTX();
    //  [ctx, E/H, H, n_batch); ]
    w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_head,1,n_head,n_batch);
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
      
    return true;
}
hGensor MOE::Forward2(struct ggml_context * ctx_,hGensor inpL,hGensor wBase,int flag){   
    int n0=inpL->ne[0],n1=inpL->ne[1],n2=inpL->ne[2],n3=inpL->ne[3]; 
    hGensor cur = BeforeForward(ctx_,inpL,flag);
    if(cur==nullptr)  //    some operation like symolic analysis     
        return cur; 

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
    return cur;
}
string MOE::__repr__( string& suffix,string& prefix,int flag)    {
    return _repr_1(suffix,prefix,"MOE");
};

OutCLS::OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : GeNeuron(key_,jit, hG_, flag){
    int nEmbd=hFish->hparams.n_embd;
    nCls=hFish->nClass();
    shape={nEmbd,nCls};
}
bool OutCLS::Build(int flag)   {
    SHAPE sp={shape[0]};
    norm.BuildX(name+sNorm,sp,hFish,0x0);        //layer->ffn_norm.sT="f";
    string sCls = "";   //  ".cls"
    proj.BuildX(name+sCls,{shape[0],shape[1]},hFish,flag); 
    name += ".cls";      
    return true;
}
hGensor OutCLS::Forward(struct ggml_context * ctx_,hGensor inpL,int flag)    {
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_,nullptr,flag);
    }
    int n_batch=hFish->hparams.n_batch(),n_ctx=hFish->hparams.n_ctx();
    hGensor cur = norm.Forward(ctx_,inpL,0x0);
    gTN(cur,"result_norm");      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = proj.Forward(ctx_,cur,0x0);
    gTN(cur,"result_output");//cb(cur, "ffn_up", il);    
    // cur = ggml_silu(ctx_, cur);    
    
    if(n_batch>1){
        cur = ggml_reshape_3d(ctx_, cur, nCls, n_ctx, n_batch);
    }
            
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
    isBias = hFish->isBias;
    // shape = shape_;
    struct ggml_context * ctx = hFish->GetCTX();
    if(shape.size()==2){    
        assert(shape[0]>0 && shape[1]>0);
        int nIn=shape[0],nOut=shape[1];
        w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nIn, nOut);
        if(isBias)  b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nOut);
    }else if(shape.size()==4)  {
        w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
        if(isBias)  b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,            1,            1, shape[3]);
    }else{
        assert(0);
    }
    string sw = name+sWeight,sb=name+".bias"; 
    bool isTrain = hFish->isTrain();
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isBias)  hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    
    if(compression==SVD){        
        assert(shape.size()==2);
        // SVD(w);
    }
    return true;
}
hGensor SLP::Forward(struct ggml_context * ctx0,hGensor cur,int flag)    {
    string prefix = ""; //sT+".";   //+
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx0,cur,flag);
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
                u = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nIn, rank);   
                memcpy(u->data,svd->U(),sizeof(float)*nIn*rank);     
                memcpy(v->data,svd->V(),sizeof(float)*nIn*rank); 
                v = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, nOut);
                // s = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, nIn, nOut);
                
                cur = ggml_mul_mat(ctx0, u, cur);    
                // cur = ggml_scale_inplace(ctx0, cur,1.0f/sqrt(float(n_embd)/n_head));  
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

hGensor ROPE::Forward(struct ggml_context * ctx_,hGensor inpL,int flag)    {   
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
    isBias = hFish->isBias;
    // name = key_;
    struct ggml_context * ctx = hFish->GetCTX();
    assert(shape.size()==1 && shape[0]>0 );
    string sw = name+sWeight,sb=name+".bias";
    bool isTrain = hFish->isTrain();    
    int nIn=shape[0];
    if(isAffineTrans){
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
        hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
    }
    if(isBias)  {
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
        hFish->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
    return true;
}

hGensor LayerNormal::Forward(struct ggml_context * ctx0,hGensor cur,int flag)    {   
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx0,cur,flag);
    } 

    float f_norm_eps = hFish->hparams.f_norm_eps;
    assert(cur!=nullptr);
    // TODO: implement ggml_norm backward
    // cur = ggml_norm(ctx0, cur, f_norm_eps);  
    const string prefix = sT+"."+cur->name;
    hGensor cur_norm = ggml_rms_norm(ctx0, cur, f_norm_eps);     
    ggml_set_name(cur_norm,_NAM_("%s_rms",prefix.c_str()));  
    hGensor  t03 = w;
    if(hFish->isTrain()){
        t03 = ggml_repeat(ctx0, w, cur_norm);          
        ggml_set_name(t03,_NAM_("%s.r",w->name));    
        hFish->gensors.Insert(t03);  
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
        
    return cur;
}
size_t LayerNormal::nElem()  {
    size_t nX=0; 
    nX += ggml_nelements(w);
    if(b!=nullptr)      
        nX += ggml_nelements(b);
    return nX;
}

hGensor GeNeuron::Forward(struct ggml_context *ctx_,hGensor cur,int flag){
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