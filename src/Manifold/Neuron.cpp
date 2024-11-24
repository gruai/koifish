
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
    }else if(typ.rfind("QKV_ROPE", 0) == 0){
        nn = std::make_shared<QKV_rope>(hG_, guid, jit, flag);
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
        assert(0);
    }

    // nn->Init(nullptr,0x0);
    nn->Build(0x0);
    assert(nn->isValid());
    return nn;
}

GeNeuron::GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag) : name(key_), hOrg(hG_),ID(0) {    
try{    
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
Ganglia::Ganglia(Fish *hG_,const string& key_,std::vector<hNeuron>& ns_,int flag) : ns(ns_)  {
    name="{"+key_+"}"; 
    hOrg=hG_;
}



Relu::Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : GeNeuron(key_,jit, hG_, flag){

}
bool Relu::Build(int flag)   {
    return true;
};
hGensor Relu::Forward(struct ggml_context *ctx_build,hGensor cur,int flag){
    return cur;
}

Drop::Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : GeNeuron(key_,jit, hG_, flag){
    
}
bool Drop::Build(int flag)   {
    return true;
};
hGensor Drop::Forward(struct ggml_context *ctx_build,hGensor cur,int flag){
    return cur;
}

Embed::Embed(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag) : GeNeuron(key_,jit, hG_, flag)    {
    assert(jvals.size()==2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[1]>0);    

    isAddPos = type_info[type_info.length()-1] =='+'; 
}
bool Embed::Build(int flag){
    assert(shape.size()==2);    
    struct ggml_context * ctx = hOrg->GetCTX(1);
    int n=shape[0],latent=shape[1];
    if(n<=0){
        n = hOrg->hparams.n_ctx();
    }
    assert(n>0 && latent>0);

    bool isTrain = hOrg->isTrain();
    string sw = name+".w",sb=name+".pos"; 
    w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, n);
    hOrg->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isAddPos){
        n = hOrg->hparams.n_ctx();
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, latent, n);
        hOrg->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
    
    return true;
}
hGensor Embed::Forward(struct ggml_context *ctx_build,hGensor cur,int flag){
    if(cur==nullptr)  //symbolic analysis
        return GeNeuron::Forward(ctx_build,cur,flag);
    
    string sw = name+"_samp";
    cur = ggml_get_rows(ctx_build, w, cur);    gTN(cur, sw.c_str());   
    if(isAddPos){
        cur = ggml_add(ctx_build, cur, b);  
    }
    if(!name.empty()){ //"inp_embd"
        gTN0(cur,"%s",name.c_str());
    }
    return cur;
}

FFN::FFN(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : GeNeuron(key_,jit, hG_, flag)     {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    // up.Init(hG_,flag);       down.Init(hG_,flag);       relu.Init(hG_,flag); 
}
bool FFN::Build(int flag)   {
    SHAPE sp={shape[0]};
    norm.BuildX(name+".norm",sp,hOrg,0x0);        //layer->ffn_norm.sT="f";
    up.BuildX(name+".up",{shape[0],shape[1]},hOrg,flag);   
  
    down.BuildX(name+".down",{shape[1],shape[0]},hOrg,flag);       
    return true;
}
hGensor FFN::Forward(struct ggml_context * ctx_build,hGensor inpL,int flag){    
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_build,nullptr,flag);
    }

    hGensor cur = norm.Forward(ctx_build,inpL,0x0);
    gTN(cur,"%s.ffn_norm",name.c_str());      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = up.Forward(ctx_build,cur,0x0);
    gTN(cur,"%s.ffn_up",name.c_str());//cb(cur, "ffn_up", il);
    // cur = ggml_gelu(ctx, cur);                cb(cur, "ffn_gelu", il);  //GGML_UNARY_OP_GELU:not implemented for backward
    cur = ggml_silu(ctx_build, cur);                
    gTN(cur,"%s.ffn_silu",name.c_str());    //cb(cur, "ffn_silu", il);     
    cur = down.Forward(ctx_build,cur,0x0);
    gTN(cur,"%s.ffn_down",name.c_str());    //cb(cur, "ffn_down", il);
    cur = ggml_add(ctx_build, cur, inpL);// add the input
    if(!name.empty()){
        strcpy(cur->name,"");   gTN(cur,"%s",name.c_str());
    }
    return cur;
}

OutCLS::OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : GeNeuron(key_,jit, hG_, flag){
    int nEmbd=hOrg->hparams.n_embd;
    nCls=hOrg->nClass();
    shape={nEmbd,nCls};
}
bool OutCLS::Build(int flag)   {
    SHAPE sp={shape[0]};
    norm.BuildX(name+".norm",sp,hOrg,0x0);        //layer->ffn_norm.sT="f";
    proj.BuildX(name+".cls",{shape[0],shape[1]},hOrg,flag);       
    return true;
}
hGensor OutCLS::Forward(struct ggml_context * ctx_build,hGensor inpL,int flag)    {
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_build,nullptr,flag);
    }
    int n_batch=hOrg->hparams.n_batch(),n_ctx=hOrg->hparams.n_ctx();
    hGensor cur = norm.Forward(ctx_build,inpL,0x0);
    gTN(cur,"result_norm");      // cb(cur, _NAM_("ffn_norm"), il);    
    cur = proj.Forward(ctx_build,cur,0x0);
    gTN(cur,"result_output");//cb(cur, "ffn_up", il);    
    // cur = ggml_silu(ctx_build, cur);    
    
    if(n_batch>1){
        cur = ggml_reshape_3d(ctx_build, cur, nCls, n_ctx, n_batch);
    }
            
    cur = AfterForward(ctx_build,cur,flag);           
    return cur;    
}

SLP::SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)      : GeNeuron(key_,jit, hG_, flag)    {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    // compression = hOrg->params.compression;
    // Build(key_, shape_, flag);
}
bool SLP::Build(int flag)      {
    isBias = hOrg->isBias;
    // shape = shape_;
    struct ggml_context * ctx = hOrg->GetCTX();
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
    string sw = name+".weight",sb=name+".bias"; 
    bool isTrain = hOrg->isTrain();
    hOrg->InitGensor(ctx,sw.c_str(),w,isTrain);
    if(isBias)  hOrg->InitGensor(ctx,sb.c_str(),b,isTrain);
    
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
    
    // compression = SVD_a; //SVD_a;    //SKIP;//hOrg->params.compression;
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
    if(!name.empty()){
        gTN0(cur,"%s",name.c_str());
    }
        
    return cur;
}

LayerNormal::LayerNormal(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag)    : GeNeuron(key_,jit, hG_, flag) {
    assert(jvals.size()>=1 && jvals[0]>0);
    shape={(int)(jvals[0])};
}
/*
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
*/
bool LayerNormal::Build(int flag)    {
    isBias = hOrg->isBias;
    // name = key_;
    struct ggml_context * ctx = hOrg->GetCTX();
    assert(shape.size()==1 && shape[0]>0 );
    string sw = name+".weight",sb=name+".bias";
    bool isTrain = hOrg->isTrain();    
    int nIn=shape[0];
    if(isAffineTrans){
        w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
        hOrg->InitGensor(ctx,sw.c_str(),w,isTrain);
    }
    if(isBias)  {
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nIn);
        hOrg->InitGensor(ctx,sb.c_str(),b,isTrain);
    }
    return true;
}

hGensor LayerNormal::Forward(struct ggml_context * ctx0,hGensor cur,int flag)    {   
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx0,cur,flag);
    } 

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

hGensor GeNeuron::Forward(struct ggml_context *ctx_build,hGensor cur,int flag){
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
hGensor GeNeuron::BeforeForward(struct ggml_context *ctx_build,hGensor cur,int flag){
    int tp=0;
    if(cur==nullptr){
        _INFO("\t %s\n",name.c_str());
    }
    return cur;
}
hGensor GeNeuron::AfterForward(struct ggml_context *ctx_build,hGensor cur,int flag){
    if(!name.empty()){
        gTN0(cur,"%s",name.c_str());
    }
    return cur;
}

void GeNeuron::BuildX(const std::string &key_, const SHAPE &shp_, Fish *hG_, int flag){
    if(hOrg==hG_ && shp_==shape && name==key_){ //
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