
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */
#include <set>
#include "Fish.hpp"
#include "Optimizer.hpp"
#include "gLLM.hpp"
#include "../g_stddef.hpp"
#include "../lenda/kernel/SVD.hpp"

hNeuron GeNeuron::MakeInstance(Fish *hG_,void *ctx,const string& guid,JSON::const_iterator jit,int flag){
    hNeuron nn=nullptr;    
    // assert(j.is_object());
    auto typ_0 = jit.key();  
    auto v = jit.value();
    
    string typ = typ_0;
    std::transform(typ.begin(), typ.end(), typ.begin(), ::toupper);

    if(typ.rfind("EMBED", 0) == 0){
        nn = std::make_shared<TokenEmbed>(hG_, guid, jit, flag);
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
    nn->Build(flag);
    assert(nn->isValid());
    return nn;
}

GeNeuron::GeNeuron(const std::string &key_,JSON::const_iterator jit, Fish *hG_, int flag) : name(key_), hFish(hG_),ID(0) {    
try{   
    Init(hG_,0x0);
    layer = TGraph::curLayer;          assert(layer>=0);
    if((*jit).contains(std::string{ "#id" })){
        ID = (*jit)["id"];
    }
    type_info = jit.key(); 
    if(jit->is_array()){
        for (const auto& item : *jit) {
            if(item.is_string()){
                string sVal = item.template get<string>();
                std::transform(sVal.begin(), sVal.end(), sVal.begin(), ::toupper);
                if(sVal=="SPARSE"){
                    isSparse = true;
                }
                if(sVal=="SVD"){
                    compression = SVD;      assert(0);      // need BLAS support
                }
            }else if(item.is_number()){
                double val = item.template get<double>();
                jvals.push_back(val);
            }
            
        }        
        // assert(jvals.size()>=2);
    }else{
        
    }
}catch(...){
    assert(0);
} 
}

GeNeuron::~GeNeuron() {   
    if(host_inp!=nullptr)
        cudaFreeHost(host_inp); 
    // FREE_a( host_inp );   
}

void GeNeuron::Init(Fish *hG_, int flag) {    
    hFish=hG_;   
    auto& config = hG_->config; 
    // n_batch=config.n_batch(),n_ctx=config.n_ctx(),n_embd=config.nEmbed();
    hG_->GetBTC(B,T,C);
    n_embd_head = config.n_embd_head();
    n_head=config.n_head();
    assert(n_embd_head*n_head==C);
    tpWeight = hFish->config.model.tpWeight;
    tpActivation = hFish->config.model.tpActivation;
    tpGradient = hFish->config.model.tpGradient;
}
void GeNeuron::SetRefer(const GeNeuron* src,bool isBias,int flag){
    assert(src!=nullptr);
    assert(w->data==nullptr);
    w->SetRefer(src->w);       
    if(isBias){
        if(b!=nullptr && src->b!=nullptr){
            b->SetRefer(src->b);  
        }
    }
}

string GeNeuron::_repr_1( string& suffix,string& prefix,string info,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s %s",tab,info.c_str());    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
}

Ganglia::Ganglia(Fish *hG_,const string& key_,std::vector<hNeuron>& ns_,int flag) : ns(ns_)  {
    name="{"+key_+"}"; 
    hFish=hG_;
}



Relu::Relu(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : SparseNeuron(key_,jit, hG_, flag){

}
bool Relu::Build(int flag)   {
    return true;
};
hGensor Relu::Ming(RLS_BP* ctx_,hGensor cur,int flag){
    return cur;
}

Drop::Drop(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)    : SparseNeuron(key_,jit, hG_, flag){
    
}
bool Drop::Build(int flag)   {
    return true;
};
hGensor Drop::Ming(RLS_BP* ctx_,hGensor cur,int flag){
    return cur;
}

MOE::MOE(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : SparseNeuron(key_,jit, hG_, flag)     {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    isSiLU = true;
    //[ctx, E/H, H, n_batch
    // up.Init(hG_,flag);       down.Init(hG_,flag);       relu.Init(hG_,flag); 
}
bool MOE::Build(int flag)   {
    string sw = name+MODEL_CARD::sWeight,sb=name+".bias";
    bool isTrain = hFish->isTrain();    
    int nIn=shape[0];
    void * ctx = hFish->GetGGCTX();
    //  [ctx, E/H, H, n_batch); ]
    w = GT(hFish, typNUMBER::F32, {n_embd_head,1,n_head,B});
    hFish->InitGensor(ctx,sw.c_str(),w,isTrain);
      
    return true;
}
hGensor MOE::Forward2(void * ctx_,hGensor inpL,hGensor wBase,int flag){   
    int n0=inpL->ne[0],n1=inpL->ne[1],n2=inpL->ne[2],n3=inpL->ne[3]; 
    hGensor cur = BeforeMing(nullptr,inpL,flag);
    if(cur==nullptr)  //    some operation like symolic analysis     
        return cur; 
#ifdef _TENSOR_G_
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
   
#endif
    return cur;
}
string MOE::__repr__( string& suffix,string& prefix,int flag)    {
    return _repr_1(suffix,prefix,"MOE");
};

OutSimilarity::OutSimilarity(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) 
    : OutCLS(hG_, key_, jit, flag)   {
        dB = 0;
}

//  The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning
OutEntropy::OutEntropy(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) 
    : OutCLS(hG_, key_, jit, flag)   {
        dB = 0;
}

OutCLS::OutCLS(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_,jit, hG_, flag){   
    int nEmbd=hFish->config.nEmbed();
    // _target = hFish->Target();   //null now          
    nCls=hFish->nClass();
    padded_nCls = (hFish->config.model.isPaddedCls) ? ceil(nCls/128.0)*128 : nCls;
    //reduce memory & some float error
    dB = hFish->config.model.preLogits_dB; //B 1 B/2;
    assert(B%dB==0);
    
    /*
    1. If true, reduces the memory,  often results in better and faster outcomes ???
    2. According to findings from OLMo the weight tying is beneficial for smaller models like 1B but for larger ones starting from 7B it starts to hurt the performance - instability in loss curves. 
        I don't know why it is not discussed in their paper, but one of the researchers is talking about it in TWIML AI podcast around 16:50: https://youtu.be/mwS9zPCv_dY?t=1010
    3. output softmax wants embeddings to be very large so their inner products will produce very different values.
        input embeddings want a much smaller range so they can have stable dynamics throughout training.
        All the "old" code bases had this scalar (usually sqrt(d)) but the llama arch dropped this when they started untying
    4. Use Weight Tying Only When the Distributional Hypothesis Holds   ???
    */
    // hFish->config.model.isEmbedWeightTying = false;
    
#ifdef _TENSOR_G_
    shape={nEmbd,padded_nCls};
    rLoss = 1.0f / (B * T );  //* grad_accum_steps 
    rLoss /= hG_->config.nGradAccumulate();
#else
    shape={nEmbd,nCls};
#endif
}

float *OutCLS::Logits(bool isToHost,int flag) {
    assert(preLogits!=nullptr);
    if(isToHost){
        assert(preLogits->host_data!=nullptr);
        
        return (float*)(preLogits->host_data);
    }else
        return TO<float>(preLogits);
    //(float *)(preLogits->data)+i*nVocab;
}  

bool OutCLS::Build(int flag)   {
    SHAPE sp={shape[0]};
    latent = hFish->config.nEmbed(-1);
    hEmbed = hFish->GetNeuron<TokenEmbed>("TokenEmbed",0);      assert(hEmbed!=nullptr);
    
    SHAPE sp2={B,T},sp3={dB,T,padded_nCls},sp4={B,T,latent};
    nzLoss = B*T;
    hostLoss = new float[nzLoss];
    target = std::make_shared<huTensor>(hFish,"target",sp2,typNUMBER::F32,false); 
    target->Alloc();
    // hFish->InitGensor(nullptr,"target",target,false);           
    hFish->target_probs = target; 
    out = std::make_shared<huTensor>(hFish,"loss",sp2,typNUMBER::F32,false);     
    auto tpL = hFish->config.model.tpPreLogits;
    if(hFish->config.isOnlyGPT) {
        preLogits = GT(hFish,tpL,{padded_nCls},0x0,"preLogits");   // std::make_shared<huTensor>(hFish,"preLogits",sp3,tpActivation,false); 
        preLogits->flags |= GTensor::F_HOSTDATA;    
    }else{
        preLogits = std::make_shared<huTensor>(hFish,"preLogits",sp3,tpL,false);  
    }

    delta = GTensor::bt4c;  // !=GTensor::delta
     
    // hFish->InitGensor(nullptr,"loss",out,false);                
    hFish->loss = out;//
    proj.BuildX(name,{shape[0],shape[1]},hFish,flag); proj.b = nullptr;
    if(!hFish->config.model.isEmbedWeightTying){        
        if(hFish->config.ModelArch()==NLP_GUPPY){
            // assert(FFN::first!=nullptr);            
        }
        proj.w->SetRefer(hEmbed->wInv);   
        //proj.w->SetRefer(hEmbed->w); 
    }else{
        proj.w->SetRefer(hEmbed->w);   
    }

    // norm.BuildX(name+sNorm,sp,hFish,0x0);        //layer->ffn_norm.sT="f";
    // proj.BuildX(name+".probability",{shape[0],shape[1]},hFish,flag); 

    string sCls = "";   //  ".cls"
    name += ".cls";      
    return true;
}
hGensor OutCLS::Ming(RLS_BP* ctx_,hGensor inpL,int flag)    {
    GeNeuron::BeforeMing(ctx_,nullptr,flag);

    int n_batch=hFish->config.n_batch(),n_ctx=hFish->config.n_ctx();
    hGensor cur = nullptr;

    if(hFish->isSymbolic()){ 
        if(!hFish->config.model.isEmbedWeightTying){
            //inpL >> proj;               // out->AddSrc({proj.out,target}); 
            out->AddSrc({inpL,proj.w,preLogits,target});       
        }else{
            out->AddSrc({inpL,proj.w,preLogits,target});            
        }
        assert(target!=nullptr);   
        cur = out; 
    } else{    
        cur = FUSE_cuda(inpL,0x0);     //return preLogits;        
        // hFish->hOPT->UpdateTrainLoss(-1,mean_loss);   
    }
    cur = AfterMing(ctx_,cur,flag);   
    return cur; 
}
string OutCLS::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s OutCLS{dB=%d x=%d}",tab,dB,padded_nCls);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

SLP::SLP(Fish *hG_, const std::string &key_, JSON::const_iterator jit,  int flag)      : SparseNeuron(key_,jit, hG_, flag)    {
    assert(jvals.size()>=2);
    shape={(int)(jvals[0]),(int)(jvals[1])};
    assert(shape[0]>0 && shape[1]>0);
    // compression = hFish->params.compression;
    // Build(key_, shape_, flag);
}

bool SLP::Build(int flag)      {
    delta = GTensor::delta;
    isBias = hFish->config.model.isSLPBias || BIT_TEST(flag,F_BIAS);
    // shape = shape_;
    void *ctx = hFish->GetGGCTX();
    typNUMBER tpData = tpWeight;
    int bFlag = 0x0;
    nIn=shape[0],nOut=shape[1];
    if(shape.size()==2){    
        assert(shape[0]>0 && shape[1]>0);        
        w = GT(hFish, tpData, {nIn, nOut});
        if(isBias)  b = GT(hFish, tpData, {nOut});
    }else if(shape.size()==4)  {
        w = GT(hFish, tpData, shape);
        if(isBias)  b = GT(hFish, tpData,            {1,1,shape[3]});
    }else{
        assert(0);
    }

    string sw = name+MODEL_CARD::sWeight,sb=name+".bias",so=name+".out"; 
    bool isTrain = hFish->isTrain();
    hFish->InitGensor(ctx,sw.c_str(),w,true);
    if(isBias)  
        hFish->InitGensor(ctx,sb.c_str(),b,true);
#ifdef _TENSOR_G_        
    if(b!=nullptr)  b->tpInit = INIT_WEIGHT::W_SKIP;
    SHAPE s3={B,T,nOut},s4={B,T,nIn};
    out = std::make_shared<huTensor>(hFish,so,s3,tpData,false); 
    if(BIT_TEST(flag,F_DELTA)){
        delta = GTensor::delta;
    }
        
    // hFish->InitGensor(ctx,so.c_str(),out,false);    
#endif    
    if(compression==SVD){        
        assert(shape.size()==2);
        // SVD(w);
    }
    return true;
}
hGensor SLP::Ming(RLS_BP* hRLS,hGensor cur,int flag)    {
    string prefix = ""; //sT+".";   //+
    if(cur==nullptr){   //symbolic analysis
        return GeNeuron::BeforeMing(hRLS,cur,flag);
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
#ifdef _TENSOR_G_
    
#else
    if(compression==SVD || compression==SVD_a)        {   //A=UDV
        int nIn=shape[0], nOut=shape[1], rank = min(64,min(nIn,nOut)/10);
        float *A=new float[nIn*nOut];
        switch(w->type){
            case GGML_TYPE_F16:
                ggml_fp16_to_fp32_row((ggml_fp16_t*)w->data,A,nIn*nOut);
                break;
            case typNUMBER::F32:
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
                u = TENSO(ctx0, typNUMBER::F32, {nIn, rank});   
                memcpy(u->data,svd->U(),sizeof(float)*nIn*rank);     
                memcpy(v->data,svd->V(),sizeof(float)*nIn*rank); 
                v = TENSO(ctx0, typNUMBER::F32, {rank, nOut});
                // s = GT(hFish, GGML_TYPE_F16, nIn, nOut);
                
                cur = ggml_mul_mat(ctx0, u, cur);    
                cur = ggml_mul_mat(ctx0, v, cur);                      
            }       
        }
        delete[] A;
    }
    if(compression == SKIP || compression==SVD_a)  {
        cur = ggml_mul_mat(ctx0, w, cur);           //create temp GGML_OP_MUL_MAT tensor:  result = ggml_new_tensor(ctx, typNUMBER::F32, 4, ne);
        gTN(cur,"%s*w",prefix.c_str());
    }
    if(b!=nullptr)  {        
        cur = ggml_add(ctx0, cur, b); 
        gTN(cur,"%s+b",prefix.c_str());
            // cur = ggml_add_inplace(ctx0, cur, b); 
    }
#endif
    cur = AfterMing(hRLS,cur,flag);
    // if(!name.empty()){
    //     gTN0(cur,"%s",name.c_str());
    // }
        
    return cur;
}

int SLP::Forw(float *rhs,float *lhs,int flag){
	float*bias = b==nullptr ? nullptr : TO<float>(b);
    if(compression==GBDT){
        // assert(nHot>0);
		// matmul_sparse_2(xb2, hb, (char*)w->w2 + moe_experts[e] * esize, NULL, hidden_dim, dim, nHot,hPicker->hot,hPicker->dTemp, dotprod);	
        assert(0);      return 0x0;
    }
			
	if(compression==SVD || compression==SVD_a)        {   //A=UDV        
        if(hSVD==nullptr && (layer>20) ) {  /*&& (layer==27)*/
            InitSVD();    
            if(hSVD->rHeavy()<1.5){
                _INFO("\t...drop!");
            }
        }    
    }
    if(hSVD!=nullptr && hSVD->rHeavy()>1.5){
        int nHeavy = hSVD->nHeavy;      
        dotprod_t fDot = fnDot(hSVD->tpOut);
        float *UX=new float[nHeavy],*U=hSVD->U();
        // cur = ggml_mul_mat(ctx0, u, cur);    
        D_matvec(UX, lhs, U, nullptr, nIn, nHeavy, fDot);
        assert( isValidF(nHeavy,UX) );
        // cur = ggml_mul_mat(ctx0, v, cur);   
        D_matvec(rhs, UX, hSVD->V(), bias, nHeavy, nOut, fDot);
        assert( isValidF(nOut,rhs) );
        delete[] UX;
    }else
	    D_matvec(rhs, lhs, w->data, bias, nIn, nOut, hFish->config.model.fDotW);
    return 0x0;
}

ROPE::ROPE(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag)    : SparseNeuron(key_,jit, hG_, flag) {
    assert(jvals.size()>=1 && jvals[0]>0);
    // shape={(int)(jvals[0])};
    Build(flag);
}
/*
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
*/
bool ROPE::Build(int flag)    {
    auto& config = hFish->config;
    n_rot = config.n_rot();
    int n_embed = config.nEmbed();
    head_dim=config.n_embd_head();
    int n_heads = config.n_head(),n_kv_heads = config.n_head_kv();
    q_dim = head_dim*n_heads, kv_dim = head_dim*n_kv_heads;
    assert(head_dim>0 && q_dim>0 && kv_dim>0);
    r_dim = config.model.rotary_dim;
    if(r_dim==0){
        r_dim = 128;    theta = 1000000;
    }else{
        n_ctx_orig = config.n_ctx_orig();
        freq_base  = config.model.rope_freq_base;
        freq_scale = config.model.rope_freq_scale;  
        theta = config.model.rope_theta;        
    }	
    int dim2 = head_dim/2;	
    float *fcos = new float[T*dim2],*fsin=new float[T*dim2];    
    for (int tid = 0; tid < dim2/2; tid++)    {
        // Compute the frequency for each tid
        float theta=10000,freq = 1.0f / powf(theta, (float)(tid) / dim2); // float powf(float base, float exponent);
        // Compute the cosine and sine for all values of 't'
        for (int t = 0; t < T; t++)        {
            fcos[t * dim2 + tid] = cosf(t * freq);
            fsin[t * dim2 + tid] = sinf(t * freq);
        }
    }
    hSin = GT(hFish, typNUMBER::F32, {T*dim2},0x0,name+".sin");     hSin->Alloc();
    hSin->SerialGP(fsin,nullptr,sizeof(float)*T*dim2,false);
    hCos = GT(hFish, typNUMBER::F32, {T*dim2},0x0,name+".cos");     hCos->Alloc();
    hCos->SerialGP(fcos,nullptr,sizeof(float)*T*dim2,false);
    delete[] fcos,fsin;  

    // KQ_pos = hFish->KQ_pos;    
    // shape = {n_embd_head, n_head, T, B};
    return true;
}

hGensor ROPE::Ming(RLS_BP* ctx_,hGensor inpL,int flag)    {   
    hGensor cur = BeforeMing(ctx_,inpL,flag);
    if(cur==nullptr)  //    some operation like symolic analysis     
        return cur; 
    assert(cur->ne[0]==shape[0] && cur->ne[1]==shape[1] && cur->ne[2]==shape[2] && cur->ne[3]==shape[3]);
    string nam0 = name+"."+sT;
    // hGensor  t05 = w==nullptr ? cur : ggml_mul_mat(ctx_, w, cur);         
    // gTN(t05,"%s*w",name.c_str());   
    // hGensor  t06 = ggml_reshape_4d(ctx_, cur,shape[0],shape[1],shape[2],shape[3]); //n_embd_head, n_head, N, n_batch    
    // gTN(t06,"%s$",name.c_str());   //gTN(t06, "t06");            
    const int rope_mode = 0;
#ifdef _TENSOR_G_
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
#endif
    return cur;
}

bool ROPE::Empty() const      { 
    // return !hFish->config.model.isRope; 
    return hSin==nullptr;
}
string ROPE::__repr__( string& suffix,string& prefix,int flag)    {
    return _repr_1(suffix,prefix,"ROPE");
};

LayerSoftmax::LayerSoftmax(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_,jit, hG_, flag) {
    /*
        float scale = 1.f / sqrtf(HS);
        int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
        softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);
    */
}

LayerNormal::LayerNormal(Fish *hG_, const std::string &key_, JSON::const_iterator jit, int flag) : SparseNeuron(key_,jit, hG_, flag) {
    // delta = GTensor::delta;
    // isRMS = true;
    if(jvals.size()==1){
        shape={(int)(jvals[0])};
    }else{
        shape = {C};
    }
    // assert(jvals.size()>=1 && jvals[0]>0);
    // shape={(int)(jvals[0])};
}
/*
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm
*/
bool LayerNormal::Build(int flag0)    {
    delta = GTensor::delta;
    int flag = flag0 | GTensor::F_RESIDENT;
    if(hFish->arch==MODEL_ARCH::NLP_GPT2 || hFish->arch==MODEL_ARCH::NLP_GUPPY)
        isRMS = false;
    // isRMS = name!="model.output_norm" ? false : true;
    isBias = hFish->config.model.isNormalBias || BIT_TEST(flag,F_BIAS) ;
/**
 * 猜测，center操作/全连接层的bias项，储存到的是关于数据的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以RMS不仅去掉了center操作，bias项也都去掉了。
 */
    if(isRMS){
        isBias = false;
    }
    // name = key_;
    void * ctx = hFish->GetGGCTX();
    assert(shape.size()==1 && shape[0]>0 );
    string sw = name+MODEL_CARD::sWeight,sb=name+".bias";
    bool isTrain = hFish->isTrain();    
    int nIn=shape[0];   
    if(isAffineTrans){
        w = GT(hFish, tpWeight, {nIn},flag);
        hFish->InitGensor(ctx,sw.c_str(),w,true);
    }
    if(isBias)  {
        b = GT(hFish, tpWeight, {nIn},flag);
        hFish->InitGensor(ctx,sb.c_str(),b,true);
    }
        
    w->tpInit = INIT_WEIGHT::FIX_1;    //always 1   ???   
    if(b!=nullptr)  b->tpInit=W_SKIP;
    // assert(nIn==C || nIn==hFish->config.nEmbed(-1));
    SHAPE sp={B,T},sp3={B,T,nIn};
    out = GT(hFish, tpActivation, {B,T,nIn},flag,name+".out");      //std::make_shared<huTensor>(hFish,name+".out",sp3,tpActivation,false);    
 
    if(isRMS){

    }else{
        mean = GT(hFish,typNUMBER::F32,{B,T},flag,name+".mean");    //std::make_shared<huTensor>(hFish,name+".mean",sp,typNUMBER::F32,false);  
    }         
    rstd = GT(hFish,typNUMBER::F32,{B,T},flag,name+".rstd");        //std::make_shared<huTensor>(hFish,name+".rstd",sp,typNUMBER::F32,false);    

    return true;
}
string LayerNormal::__repr__( string& suffix,string& prefix,int flag){
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s %s(%s%s%s) out=%s",tab, isRMS ? "RMS" : "LayerNormal",
        b==nullptr?"":"+b",mean==nullptr?"":"+mean",rstd==nullptr?"":"+rstd",
        out==nullptr?"NULL":out->name);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf; 
}

hGensor LayerNormal::Ming(RLS_BP* hRLS,hGensor cur,int flag)    {
    GeNeuron::BeforeMing(hRLS,cur,flag);

    float f_norm_eps = hFish->config.model.norm_eps;
    assert(cur!=nullptr);
    // TODO: implement ggml_norm backward
    // cur = ggml_norm(ctx0, cur, f_norm_eps);  
    const string prefix = sT+"."+cur->name;

    if(isForward()){
        if(hFish->isSymbolic()){            
            cur >> *this;       cur = out;
        } else{
            cur = FUSE_cuda(cur);   
            //cur = cur->Normal(out,mean,rstd,w,b);            
        } 
    }else{
        cur = FUSE_cuda(cur); 
        //cur->Normal(out,mean,rstd,w,b,false);   
    } 
    cur = AfterMing(hRLS,cur,flag);   
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
hGensor GeNeuron::BeforeMing(RLS_BP* hRLS,hGensor cur,int flag){
    assert(hRLS!=nullptr);
    if(hRLS->isRemater)   {      
        return cur;
    }
    vector arrT = PGensors();    
    DATA_PLACE old_place = place;
    if(hFish->isSymbolic()){ 
        // host_inp = new char[GTensor::outL->nByte()];        
        cudaHostAlloc(&host_inp, GTensor::outL->nByte(),0);
        
    }else{
        SYNC_DEVICE();  double now = GST_ms();
        ManageMemory(DATA_PLACE::DEV_MEM);
        if(isForward()){
            
        }else{
             if(old_place==DATA_PLACE::FREE_DEV){   //  Remater  
                hRLS->isRemater = true;
                Ming(hRLS,inp,flag);
                hRLS->isRemater = false;   
             }
        }
        SYNC_DEVICE();  SUM::tRemater += GST_ms()-now;   
    }
    
    return cur;
}
hGensor GeNeuron::Ming(RLS_BP* hRLS,hGensor cur,int flag){
    assert(0);
    return cur;
}
hGensor GeNeuron::AfterMing(RLS_BP* hRLS,hGensor cur,int flag){
    if(hRLS->isRemater)   {        
        return cur;
    }
    assert(hRLS!=nullptr);
    auto hOPT = hFish->hOPT;
    if(hFish->isSymbolic()){
        if(!name.empty()){
            gTN0(cur,"%s",name.c_str());
        }
    }else{
        if(isForward()){
        }else{  //After backward, UpdateParam
            if(!hFish->config.scheduling.isUpdateParamV0()){
                // if(strcmp(cur->name,"model.blk.11.attn")==0){
                //     int bug = 0x0;
                // }    
                for(auto t : PGensors()){
                    if(t->isRefer() || !t->isParam())     //
                        continue;
    
                    hOPT->UpdateTensorParam(t,nullptr,0.0); 
                    hRLS->SetTensorStatus(hOPT->GetITER(),t,RLSchedule::UPDATE_PARAM);               
                }  
            }
        }
        
        if(!hRLS->isResident(this)){           
            SYNC_DEVICE();  //  Otherwise, The timing of cudaFree would get much higher
            ManageMemory(DATA_PLACE::FREE_DEV);            
        }
        
    }
    return cur;
}

// std::vector<hGensor> GeNeuron::Gensors(bool isNoRef) {
//     std::vector<hGensor> gensors={  w,b };     
//     if(!isNoRef)   {
//         std::vector<hGensor> refs={  inp,out,delta };     
//         gensors.insert(gensors.end(), refs.begin(), refs.end());
//     }
//     return gensors;
// }

void GeNeuron::OnDebug(const std::string&info,int typ,int flag){
    dump_flag = -1;
}

std::vector<hGensor> GeNeuron::PGensors(bool isNoRef,int flag) {
    assert(out!=nullptr);
    vector arrT = {out};
    for(auto op : out->src ){
        if(op->_t==nullptr)
            continue;
        arrT.push_back(op->_t);
    }
    return arrT;
}
hGensor GeNeuron::GetGensor(const std::string &key,int flag) {    
    auto gensors = PGensors();  assert(gensors.size()>0);
    for(auto gensor:gensors){
        if(gensor==nullptr) continue;        
        if(strstr(gensor->name,key.c_str()) != NULL){
            return gensor; 
        }
    }
    assert(0);
    return nullptr;
};

// 天地本逆旅, 你我皆过客(Guoke)
int GeNeuron::SetGuoke(GeNeuron *hGuoke_,int flag){   
    std::vector<hGensor> gSrc, arrP=PGensors();
    if(hGuoke_!=nullptr){
        gSrc=hGuoke_->PGensors();
        assert(gSrc.size()==arrP.size());
    }
    int nG = 0,i,nT=arrP.size();    
    for(i=0;i<nT;i++){
        auto t = arrP[i];
        if(hGuoke_==nullptr){
            if(!BIT_TEST(t->flags,GTensor::F_RESIDENT))
                BIT_SET(t->flags,GTensor::F_RESIDENT);
            continue;
        }
        assert(gSrc[i]->isSameShape(t));
        assert(t!=gSrc[i]);
        if(t->isParam()){
            assert(gSrc[i]->isParam());
            continue;
        }
            
        if(t->isRefer()){
            continue;
        }
        t->SetRefer(gSrc[i]);
        nG++;
    }
    return nG;
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

bool GeNeuron::isOnlyInfer(){
    assert(hFish!=nullptr);
    return hFish->isLocalInfer;
}
bool GeNeuron::isForward()  {   
    assert(hFish!=nullptr);
    if(hFish->isRemater())
        return true;
    bool isForward = !hFish->hOPT->isBackward;
    return isForward;    
}

hGTensor operator>>(hGTensor t, const LayerNormal& norm){
    assert(t!=nullptr && norm.out!=nullptr);

    // norm.out->AddSrc({t,norm.w,norm.b});
    norm.out->AddSrc({t,norm.w,norm.b,norm.mean,norm.rstd});    // should alloc memory of mean&rstd
    return norm.out;
}
hGTensor operator>>(hGTensor t, const SLP& slp){
    assert(t!=nullptr );
    if(slp.Empty())
        return t;
    assert(slp.out!=nullptr);
    slp.out->AddSrc({t,slp.w,slp.b});
    return slp.out;
}
hGTensor operator>>(hGTensor t, const Relu& relu){
    assert(t!=nullptr && relu.out!=nullptr);
    relu.out->AddSrc({t});
    return relu.out;
}
hGTensor operator>>(hGTensor t, const GeNeuron* neuron){
    assert(t!=nullptr && neuron->out!=nullptr);
    neuron->out->AddSrc({t});
    return neuron->out;
}
hGTensor operator>>(hGTensor t, const SparseNeuron* neuron){
    assert(t!=nullptr && neuron->out!=nullptr);
    neuron->out->AddSrc({t});
    return neuron->out;
}



