/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#include <sched.h>
#include "SAM.hpp"
#include "TGraph.hpp"
#include "../ggex/GG_util.hpp"
#include "../lenda/kernel/SVD.hpp"
#include "llama_cys.h"

hGensor Fish::AddTensor(const std::string&key_,enum ggml_type tp,const SHAPE& shape,int flag){
    auto ctx=GetCTX();
    hGensor gg_tensor = nullptr;
    if(shape.size()==4)  {
        gg_tensor = ggml_new_tensor_4d(ctx, tp, shape[0], shape[1], shape[2], shape[3]);
    }else if(shape.size()==2)  {
        gg_tensor = ggml_new_tensor_2d(ctx, tp, shape[0], shape[1]);        
    }else if(shape.size()==1)  {
        gg_tensor = ggml_new_tensor_1d(ctx, tp, shape[0]);        
    }else{
        assert(0);
    }  
    Gensor2Map(gg_tensor);
    // assert(tensors.find(key_) == tensors.end());
    // tensors[key_] = gg_tensor;    
    return gg_tensor;   
}



QKV_LAY::QKV_LAY(Fish *hF_,int id)  :  NeLayer(id)     {   
    name = "QKV_LAY";   
    att_norm.Init(hF_,0x0),             ffn_norm.Init(hF_,0x0);
    Q.Init(hF_,0x0),        K.Init(hF_,0x0),            V.Init(hF_,0x0),        
    up.Init(hF_,0x0),       down.Init(hF_,0x0);
    proj.Init(hF_,0x0);
}

size_t SLP::nElem()  {
    size_t nX=0; 
    nX += ggml_nelements(w);
    if(b!=nullptr)      
        nX += ggml_nelements(b);
    return nX;
}

SelfAttention::SelfAttention(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : GeNeuron(key_,jit, hG_, flag)     {
    assert(hOrg!=nullptr);
    auto& hparams = hG_->hparams;
    f_max_alibi_bias = hparams.f_max_alibi_bias;
 
    // n_embd_head = hparams.n_embd_head_v;   
    n_embd_gqa  = hparams.n_embd_v_gqa();
    n_tokens=n_ctx*n_batch;
    KQ_mask=hOrg->KQ_mask,      KQ_pos=hOrg->KQ_pos;
    ID = 0;
    
    n_ff = hOrg->hparams.n_ff(ID);
    n_head_kv=hparams.n_head_kv(ID);        
    assert(n_embd_head*n_head==n_embd);
    if(jvals.size()>=3){
        shape={(int)(jvals[0]),(int)(jvals[1]),(int)(jvals[2])};
    }else{  //"attn":{"QKV":[]},  
        shape={n_embd,n_embd,n_head};
    }
    
    assert(shape[0]>0 && shape[1]>0 && shape[2]>0);
    // q.Init(hG_,flag);       k.Init(hG_,flag);       v.Init(hG_,flag);       proj.Init(hG_,flag);
}

bool SelfAttention::Build(int flag)   {
    /*layer->Q.BuildX(_NAM_("block.%d.Q",il),{n_embd, n_embd},0x0);                layer->Q.sT="q";
            layer->K.BuildX(_NAM_("block.%d.K",il),{n_embd, n_embd_gqa},0x0);            layer->K.sT="k";
            layer->V.BuildX(_NAM_("block.%d.V",il),{n_embd, n_embd_gqa},0x0);            layer->V.sT="v";
            Qcur = layer->Q.Forward(ctx_,cur,0x0);       
            Kcur = layer->K.Forward(ctx_,cur,0x0);       
            Vcur = layer->V.Forward(ctx_,cur,0x0);*/
    SHAPE sp={shape[0],shape[1]};
    norm.BuildX(name+".norm",{shape[0]},hOrg,0x0);        
    Q.BuildX(name+".Q",sp,hOrg,flag);          
    K.BuildX(name+".K",sp,hOrg,flag);              V.BuildX(name+".V",sp,hOrg,flag);
    rope.BuildX(name+".rope",sp,hOrg,flag);
    proj_cat.BuildX(name+".proj",sp,hOrg,flag);

    // tpTrans = RELU2;
    // moe.BuildX(name+".moe",sp,hOrg,flag);        //  why this would slow converge???
    return true;
}
string SelfAttention::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"{%s QKV%s%s E%d H%d x=%d trans=%d}",tab,
        moe.Empty()?"":"+moe",rope.Empty()?"":"+rope",
        n_embd,n_head,tpNormal,tpTrans);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

hGensor SelfAttention::MyAttention(struct ggml_context * ctx_,hGensor cur,int flag)   {
    float kq_scale = 1.0f/sqrtf(float(n_embd_head)),s;
    
    hGensor q,k;                  //  assert(KQ_mask!=nullptr);
    hGensor Qcur = Q.Forward(ctx_,cur,0x0);       
    hGensor Kcur = K.Forward(ctx_,cur,0x0);  
   
    // cb(Qcur, "Qcur", il);        cb(Kcur, "Kcur", il);        cb(Vcur, "Vcur", il);
    if(isAttOnBC){  // attenion on all tokens, memory would explode!
        Qcur = ggml_reshape_3d(ctx_, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx_, Kcur, n_embd_head, n_head, n_tokens);
        // Vcur = ggml_reshape_3d(ctx_, Vcur, n_embd_head, n_head, n_tokens);
    }else /**/ {
        Qcur = ggml_reshape_4d(ctx_, Qcur, n_embd_head, n_head, n_ctx,n_batch);
        Kcur = ggml_reshape_4d(ctx_, Kcur, n_embd_head, n_head, n_ctx,n_batch);
        // Vcur = ggml_reshape_4d(ctx_, Vcur, n_embd_head, n_head, n_ctx,n_batch);
    }    
    if(!rope.Empty()){
        rope.sT="Q";    Qcur = rope.Forward(ctx_,Qcur,0x1);     
        rope.sT="K";    Kcur = rope.Forward(ctx_,Kcur,0x1);  
    }
    if(tpNormal==1 /*&& n_embd_head>=1*/)    {   
        Qcur=ggml_rms_norm(ctx_,Qcur,1.0e-5);  Kcur=ggml_rms_norm(ctx_,Kcur,1.0e-5);    //Vcur=ggml_rms_norm(ctx_,Vcur,1.0e-5);  
        gTN(Qcur,"%s.Q4",name.c_str());         gTN(Kcur,"%s.K4",name.c_str());         //gTN(Vcur,"%s.V4",name.c_str()); 
    }

    q = ggml_permute(ctx_, Qcur, 0, 2, 1, 3);   //eh,ctx,h,b
    k = ggml_permute(ctx_, Kcur, 0, 2, 1, 3);  
    struct ggml_tensor * kq = ggml_mul_mat(ctx_, k, q);        //cb(kq, "kq", il);        
    switch(tpTrans){    //Get markov transition matrix from KQ
    case RELU2:
        kq = ggml_silu(ctx_,kq);                        gTN(kq,"%s.r2_0",name.c_str()); 
        kq = ggml_mul(ctx_,kq,kq);                      gTN(kq,"%s.r2_1",name.c_str()); 
        kq = ggml_scale(ctx_,kq,(1.0f/n_embd_head));    gTN(kq,"%s.r2_2",name.c_str()); 
        break;
    case SOFT_MAX:
    default:    //
        if(1)      {    //     may crash in some case! 
            kq = ggml_soft_max_ext(ctx_, kq, KQ_mask, kq_scale, f_max_alibi_bias);       //would 
        }else{  //wouls slow converge,why?
            hGensor  t16_1 = ggml_scale_inplace(ctx_, kq, kq_scale);        gTN(t16_1,"%s.161",name.c_str());     
            hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx_, t16_1, 0);    gTN(t16_2,"%s.162",name.c_str());        
            kq = ggml_soft_max_inplace(ctx_, t16_2);             
        }   
        break;  
    }
  
    gTN(kq,"%s.kq_soft_max_ext",name.c_str());            //cb(kq, "kq_soft_max_ext", il);  
    gTN0(q,"%s.q",name.c_str());         gTN0(k,"%s.k",name.c_str());  
    return kq;
}

hGensor SelfAttention::Forward(struct ggml_context * ctx_,hGensor inpL,int flag)    {
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_,nullptr,flag);
    }
    
    hGensor cur = norm.Forward(ctx_,inpL,0x0); // normal_mode==0 ?: inpL;
    hGensor kq = MyAttention(ctx_,cur,flag);
    hGensor Vcur = V.Forward(ctx_,cur,0x0),v; 
    Vcur = ggml_reshape_4d(ctx_, Vcur, n_embd_head, n_head, n_ctx,n_batch);
    v = ggml_cont(ctx_,ggml_permute(ctx_, Vcur, 1, 2, 0, 3));
           gTN0(v,"%s.v",name.c_str()); 
    /*if(isOnlinePush)    {
        ggml_build_forward_expand(gf, q);    ggml_build_forward_expand(gf, k);    ggml_build_forward_expand(gf, v);
    }*/   
    
    hGensor kqv = ggml_mul_mat(ctx_, v, kq);        // eh,ctx,h,b
    gTN(kqv,"%s.kqv",name.c_str());            
    if(!moe.Empty())
        kqv = moe.Forward(ctx_,kqv);
    hGensor kqv_merged = ggml_permute(ctx_, kqv, 0, 2, 1, 3); // eh,h,ctx,b
    gTN0(kqv_merged,"%s.kqv_merged",name.c_str());            //cb(kqv_merged, "kqv_merged", il);
    if(0){   //  back gradient is zero
        //cur = ggml_cont_2d(ctx_, kqv_merged, n_embd_head_v*n_head, n_tokens);
    }else{
        hGensor kqv_out = ggml_cont(ctx_, kqv_merged);              
        cur = ggml_reshape_2d(ctx_, kqv_out, n_embd, n_tokens);              
    }        
    gTN0(cur,"%s.kqv_merged_cont",name.c_str());//cb(cur, "kqv_merged_cont", il);
    
    cur = proj_cat.Forward(ctx_,cur,0x0);            //cb(cur, "attn_proj", il); 
    
    //if(isOnlinePush)            ggml_build_forward_expand(gf, cur);        

    if (isLast) {            // skip computing output for unused tokens
        // hGensor inp_out_name.c_str()s = nullptr;  //build_inp_out_name.c_str()s();
        // cur  = ggml_get_rows(ctx_,  cur, inp_out_name.c_str()s);
        // inpL = ggml_get_rows(ctx_, inpL, inp_out_name.c_str()s);
    }
    
    cur = ggml_add(ctx_, cur, inpL);  

    cur = AfterForward(ctx_,cur,flag);
    return cur;
}

BROWN_attn::BROWN_attn(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : SelfAttention(hG_,key_,jit,flag)     {
    auto& hparams = hG_->hparams;
    n_rot = hparams.n_rot;
    rope_freq_base  = hparams.rope_freq_base;
    rope_freq_scale = hparams.rope_freq_scale; 
    isRope = false; 
}
bool BROWN_attn::Build(int flag)   {
    // SelfAttention::Build(flag);           
    SHAPE sp={shape[0],shape[1]};
    norm.BuildX(name+".norm",{shape[0]},hOrg,0x0);        
    Q.BuildX(name+".tmp",{n_ctx,n_ctx,n_head,n_batch},hOrg,flag);   //transition as property
    proj_cat.BuildX(name+".proj",sp,hOrg,flag);   
    // moe.BuildX(name+".moe",sp,hOrg,flag);  
    return true;
}
hGensor BROWN_attn::Forward(struct ggml_context * ctx_,hGensor teb,int flag)    {
    assert_shape_2d(teb, n_embd, n_ctx*n_batch);
    hGensor cur=BeforeForward(ctx_,teb,flag);
    if(cur==nullptr)    return cur;

    cur = norm.Forward(ctx_,cur,0x0);
    const float kq_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1,N = n_ctx,n_past=0;;    
    hGensor v = cur,v3=nullptr,v4=nullptr, wv = nullptr, kqv_out=nullptr,prob;
    
    hGensor v_rope = ggml_reshape_4d(ctx_, cur, n_embd_head, n_head, N, n_batch);       gTN(v_rope,"%s.4",name.c_str()); 
    if(!isRope){
        v_rope = ggml_permute(ctx_, v_rope, 1,2,0,3);   gTN(v_rope,"%s.4p",name.c_str());    //  [ctx, E/H, H, n_batch); ]
        v = ggml_cont(ctx_,v_rope);
    }else{
        if(0)
            ;// v = W_rope(ctx_,cur,V.w,KQ_pos,{n_embd_head, n_head, N, n_batch},"v",0x1);   //24,6,32,3
        else{
            v_rope = ggml_rope_ext(ctx_, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            gTN(v_rope,"%s.rope_ext",name.c_str()); 
            v_rope = ggml_permute(ctx_, v_rope, 1,2,0,3);   //  [ctx, E/H, H, n_batch); ]
            v = ggml_cont(ctx_,v_rope);
        }
    }
    gTN(v,"%s.v4",name.c_str());     
    if(0)      {
        prob = ggml_soft_max_ext(ctx_, Q.w, KQ_mask, kq_scale, f_max_alibi_bias);       //would crash!
    }else{
        hGensor  t16_1 = ggml_scale_inplace        (ctx_, Q.w, kq_scale); 
        hGensor  t16_2 = ggml_diag_mask_inf_inplace(ctx_, t16_1, n_past); 
        prob = ggml_soft_max_inplace     (ctx_, t16_2);               
    }      
    // [32,24,6,3]x[32,32,6,3]  => [24,32,6,3]
    wv = ggml_mul_mat(ctx_, v, prob);        gTN(wv,"%s.wv",name.c_str());
    // experts mechanism
    if(!moe.Empty()){
        // v4 = ggml_reshape_4d   (ctx_, teb, n_embd_head, n_head, N, n_batch);
        // v4 = ggml_permute(ctx_, v4, 0,2,1,3); 
        // v4 = ggml_cont(ctx_,v4);
        // wv = moe.Forward2(ctx_,wv,v4);
        wv = moe.Forward(ctx_,wv);
    }        

    kqv_out = ggml_permute(ctx_, wv, 0,2,1,3);       //
    assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);  
    // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    gTN(kqv_out, "%s.kqv_out_rope",name.c_str());     

    kqv_out = ggml_cont(ctx_, kqv_out);              
    gTN(kqv_out, "%s.kqv_merged_cont",name.c_str());     
    kqv_out = ggml_reshape_2d   (ctx_, kqv_out, n_embd, N*n_batch);   // [768,17,1]  
    // if(isOnlinePush) ggml_build_forward_expand(gf_,kqv_out);  
    hGensor t20 = proj_cat.Forward(ctx_,kqv_out);                        
    gTN(t20, "%s.kqv_out",name.c_str());     assert_shape_2d(t20, n_embd, N*n_batch);
    cur = ggml_add          (ctx_, t20, teb);  /**/  
    
    cur = AfterForward(ctx_,cur,flag);
    return cur;
}

string BROWN_attn::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s BROWN_attn %s",tab,moe.Empty()?"":"+moe");    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

GatedAttention::GatedAttention(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : SelfAttention(hG_,key_,jit,flag)     {
    auto& hparams = hG_->hparams;
    shape = {n_embd,n_ff};
    tpTrans = RELU2;
}
bool GatedAttention::Build(int flag)   {
    norm.BuildX(name+".norm",{shape[0]},hOrg,0x0);        //layer->ffn_norm.sT="f";
    upU.BuildX(name+".upU",{shape[0],shape[1]},hOrg,flag);   
    upV.BuildX(name+".upV",{shape[0],shape[1]},hOrg,flag);     
    down.BuildX(name+".down",{shape[1],shape[0]},hOrg,flag);           
    if(1){
        SHAPE sp={n_embd,n_embd};
        Q.BuildX(name+".Q",sp,hOrg,flag);          
        K.BuildX(name+".K",sp,hOrg,flag);              
        rope.BuildX(name+".rope",sp,hOrg,flag);
    }
           
    return true;
}
hGensor GatedAttention::Forward(struct ggml_context * ctx_,hGensor inpL,int flag)    {
    if(inpL==nullptr){   //symbolic analysis
        return GeNeuron::Forward(ctx_,nullptr,flag);
    }
    
    hGensor cur = norm.Forward(ctx_,inpL,0x0),attn=nullptr;    
    gTN(cur,"%s.gau_norm",name.c_str());      // cb(cur, _NAM_("ffn_norm"), il); 
    // attn = MyAttention(ctx_,cur,0x0);       //  [c,c,H,B]   
    // cur = up.Forward(ctx_,cur,0x0);    
    hGensor Ucur = upU.Forward(ctx_,cur,0x0);  
    hGensor Vcur = upV.Forward(ctx_,cur,0x0);  
    hGensor u = ggml_silu(ctx_, Ucur); 
    hGensor v = ggml_silu(ctx_, Vcur);    
    
    if(attn!=nullptr){  //https://zhuanlan.zhihu.com/p/475393475
        v = ggml_mul_mat(ctx_,attn, v);
    }
 
    hGensor uv = ggml_mul(ctx_,u,v);
    cur = down.Forward(ctx_,uv,0x0);
    cur = ggml_add(ctx_, cur, inpL);// add the input

    cur = AfterForward(ctx_,cur,flag);
    return cur;
}

string GatedAttention::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s {GatedAttention }",tab);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};

/*
BROWN_v0::BROWN_v0(Fish* hG_,const std::string&key_,JSON::const_iterator jit,int flag) : SelfAttention(hG_,key_,jit,flag)     {
    auto& hparams = hG_->hparams;
    n_rot = hparams.n_rot;
    rope_freq_base  = hparams.rope_freq_base;
    rope_freq_scale = hparams.rope_freq_scale;  
}
bool BROWN_v0::Build(int flag)   {
    // SelfAttention::Build(flag);           
    SHAPE sp={shape[0],shape[1]};
    norm.BuildX(name+".norm",{shape[0]},hOrg,0x0);        
    Q.BuildX(name+".Q",sp,hOrg,flag);  
    if(Transfer_1)       
        V.BuildX(name+".V",{shape[0],1},hOrg,flag);  //w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);           
    // K.BuildX(name+".K",sp,hOrg,flag);              V.BuildX(name+".V",sp,hOrg,flag);
    proj_cat.BuildX(name+".proj",sp,hOrg,flag);       
           
    return true;
}
hGensor BROWN_v0::Forward(struct ggml_context * ctx_,hGensor teb,int flag)    {
    hGensor cur=BeforeForward(ctx_,teb,flag);

    const float kq_scale = 1.0f/sqrtf(float(n_embd)/n_head);
    int rope = 1,N = n_ctx,n_past=0;;    
    hGensor v = teb,v3=nullptr,v4=nullptr, t14 = nullptr, kqv_out=nullptr;
    assert_shape_2d(teb, n_embd, N*n_batch);
    if(0)
        v = W_rope(ctx_,cur,V.w,KQ_pos,{n_embd_head, n_head, N, n_batch},"v",0x1);   //24,6,32,3
    else{     
        hGensor v_rope = ggml_reshape_4d   (ctx_, teb, n_embd_head, n_head, N, n_batch);
        gTN(v_rope,"%s.teb",name.c_str());        
        v_rope = ggml_rope_ext(ctx_, v_rope, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        gTN(v_rope,"%s.rope_ext",name.c_str()); 
        v_rope = ggml_reshape_2d   (ctx_, v_rope, n_embd_head*n_head, N*n_batch);
        v = ggml_mul_mat(ctx_, v_rope, Q.w);    //[144,96]x[144,144]=>[96,144]
    }
    gTN(v,"%s.rope_wq",name.c_str());
    v3 = ggml_reshape_3d(ctx_, v, N, n_batch, n_embd);        gTN(v3, "%s.v3",name.c_str());       
    // experts mechanism
    hGensor probs = nullptr;
    if(V.w!=nullptr)   {
        hGensor w_trans = V.w;
        hGensor w_ = ggml_mul_mat(ctx_, w_trans,teb ); //ggml_reshape_2d(ctx,v3,N, n_batch*n_embd)  
        gTN(w_,"%s.wvte",name.c_str());
        w_ = ggml_reshape_2d(ctx_, w_, N,n_batch);   
        // if(isSiLU){ //maybe useful
        //     w_ = ggml_silu(ctx,w_);
        // } 
        probs = ggml_soft_max(ctx_,w_);              gTN(probs,"%s.probs",name.c_str());
        probs = ggml_repeat(ctx_, probs, v3); 
    }else
        probs = ggml_soft_max(ctx_,v3); 
    hGensor expert = v3;    //ggml_reshape_2d(ctx,v3,n_vocab,n_ctx*n_batch);
    // [32,3,144]x[32,3,144,1]
    hGensor kqv = ggml_mul(ctx_,expert,probs);       gTN(kqv,"%s.kqv",name.c_str());
    v4 = ggml_reshape_4d   (ctx_, kqv,N, n_batch,n_embd_head, n_head);
    kqv_out = ggml_permute(ctx_, v4, 2, 3, 0, 1);       // [24,6,512,32]  
    assert_shape_4d(kqv_out, n_embd_head, n_head, N, n_batch);  
    // kqv_out = ggml_rope_ext(ctx, kqv_out, KQ_pos, nullptr, n_rot, 0, n_ctx, rope_freq_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    gTN(kqv_out, "%s.kqv_out_rope",name.c_str());     

    kqv_out = ggml_cont(ctx_, kqv_out);              
    gTN(kqv_out, "%s.kqv_merged_cont",name.c_str());     
    kqv_out = ggml_reshape_2d   (ctx_, kqv_out, n_embd, N*n_batch);   // [768,17,1]  
    // if(isOnlinePush) ggml_build_forward_expand(gf_,kqv_out);  
    hGensor t20 = proj_cat.Forward(ctx_,kqv_out);                        
    gTN(t20, "%s.kqv_out",name.c_str());     assert_shape_2d(t20, n_embd, N*n_batch);
    cur = ggml_add          (ctx_, t20, teb);  
    
    cur = AfterForward(ctx_,cur,flag);
    return cur;
}

string BROWN_v0::__repr__( string& suffix,string& prefix,int flag)    {
    char buf[5012]="\0";
    const char*tab=prefix.c_str();
    sprintf(buf+strlen(buf),"%s BROWN_v0",tab);    
    if(flag>0)
        _INFO("%s",buf); 
    return buf;  
};*/



NT_SAM::NT_SAM(hFISH graph,const std::string&key_,const SHAPE& shape,bool is_global_,int flag)    :
    NeLayer(key_,flag),is_global_attn(is_global_)   {
    struct ggml_context * ctx = graph->GetCTX();         assert(ctx!=nullptr);
    assert(shape.size()==4 && shape[0]>0);
    nEmbed = shape[0];                          
    head_dim = shape[1];
    int n_img_embd = shape[2],n_window_size=shape[3],ld=is_global_attn ? 2*n_img_embd - 1 : 2*n_window_size - 1;
    nHead = nEmbed / head_dim;    
    /**/if (is_global_attn) {
        ld = 2*n_img_embd - 1;
    } else {
        ld = 2*n_window_size - 1;
    }
    norm1.BuildX(key_+".norm1",{nEmbed},nullptr,0x0);
    rel_pos_w = graph->AddTensor(key_+".attn.rel_pos_w",GGML_TYPE_F16,{head_dim,ld},0x0);
    rel_pos_h = graph->AddTensor(key_+".attn.rel_pos_h",GGML_TYPE_F16,{head_dim,ld},0x0);
    in_proj.BuildX(key_+".attn.qkv",{nEmbed, 3*nEmbed},nullptr,0x0);
    proj.BuildX(key_+".attn.proj",{nEmbed, nEmbed},nullptr,0x0);
    norm2.BuildX(key_+".norm2",{nEmbed},nullptr,0x0);
    mlp_lin1.BuildX(key_+".mlp.lin1",{nEmbed, 4*nEmbed},nullptr,0x0);
    mlp_lin2.BuildX(key_+".mlp.lin2",{4*nEmbed, nEmbed},nullptr,0x0);

    // graph->AddLayer(key_,{
    //                 NP_("SelfAttention",".self_attn",{n_enc_out_chans, n_enc_out_chans, n_enc_out_chans}),
    //                 NP_("SelfAttention",".cross_attn_token_to_image",{n_enc_out_chans, n_enc_out_chans/2, n_enc_out_chans}),
    //                 NP_("LayerNormal",".norm1",{n_enc_out_chans}),
    //                 NP_("SelfAttention",".cross_attn_image_to_token",{n_enc_out_chans, n_enc_out_chans/2, n_enc_out_chans}),
    //                 NP_("LayerNormal",".norm2",{n_enc_out_chans}),
    //                 NP_("LayerNormal",".norm3",{n_enc_out_chans}),
    //                 NP_("LayerNormal",".norm4",{n_enc_out_chans}),
    //                 NP_("SLP",".mlp.lin1",{n_enc_out_chans, 8*n_enc_out_chans}),
    //                 NP_("SLP",".mlp.lin2",{8*n_enc_out_chans,n_enc_out_chans}),
    //             } ;
}

hGensor NT_SAM::Build_(struct ggml_context * ctx0,hGensor inpL,float eps,
    int n_window_size,int n_enc_state,int n_enc_head_dim,int n_enc_head,int flag)    {
    hGensor cur;
// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
    {
        cur = ggml_norm(ctx0, inpL, eps);
        // cur = ln_0_w*cur + ln_0_b
        cur = ggml_mul(ctx0, cur, norm1.w);
        cur = ggml_add_inplace(ctx0, cur, norm1.b);
    }

    const int64_t w0 = cur->ne[1],h0 = cur->ne[2];
    if (!is_global_attn) {
        // local attention layer - apply window partition
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
        cur = ggml_win_part(ctx0, cur, n_window_size);
    }

    const int64_t W = cur->ne[1],H = cur->ne[2];
    {
        cur = in_proj.Forward(ctx0,cur);
            // cur = ggml_mul_mat(ctx0, in_proj.w, cur);
            // cur = ggml_add_inplace(ctx0, cur, in_proj.b);
        
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
        const int B = cur->ne[3];
        cur = ggml_reshape_4d(ctx0, cur, n_enc_state, 3, W*H, B);   //[768,3,196,25]
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));
        struct ggml_tensor * Q,* K,* V;
        Q = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 0*cur->nb[3]);
        Q = ggml_reshape_4d(ctx0, Q,   n_enc_head_dim, n_enc_head, W*H, B);
        Q = ggml_cont      (ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        Q = ggml_reshape_3d(ctx0, Q,   n_enc_head_dim, W*H, B*n_enc_head);

        K = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 1*cur->nb[3]);
        K = ggml_reshape_4d(ctx0, K,   n_enc_head_dim, n_enc_head, W*H, B);
        K = ggml_cont      (ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
        K = ggml_reshape_3d(ctx0, K,   n_enc_head_dim, W*H, B*n_enc_head);

        V = ggml_view_3d   (ctx0, cur, n_enc_state, W*H, B, cur->nb[1], cur->nb[2], 2*cur->nb[3]);
        V = ggml_reshape_4d(ctx0, V,   n_enc_head_dim, n_enc_head, W*H, B);
        V = ggml_cont      (ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
        V = ggml_reshape_3d(ctx0, V,   W*H, n_enc_head_dim, B*n_enc_head);

        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

        struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0,KQ,1.0f/sqrtf(n_enc_head_dim));

        struct ggml_tensor * rw = ggml_get_rel_pos(ctx0, rel_pos_w, W, W);
        struct ggml_tensor * rh = ggml_get_rel_pos(ctx0, rel_pos_h, H, H);

        struct ggml_tensor * q_r = ggml_reshape_4d(ctx0, Q, n_enc_head_dim, W, H, B*n_enc_head);

        struct ggml_tensor * rel_w = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_mul_mat(ctx0,
                        rw,
                        ggml_cont(ctx0, ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
                    0, 2, 1, 3));
        struct ggml_tensor * rel_h = ggml_mul_mat(ctx0, rh, q_r);

        struct ggml_tensor * attn = ggml_add_rel_pos_inplace(ctx0, KQ_scaled, rel_w, rel_h);

        struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, attn);

        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

        cur =
            ggml_reshape_4d(ctx0,
                    ggml_cont(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W*H, n_enc_head, B),
                            0, 2, 1, 3)),
                    n_enc_state, W, H, B);

        cur = ggml_mul_mat(ctx0, proj.w, cur);
        cur = ggml_add_inplace(ctx0, cur, proj.b);
    }

    if (!is_global_attn) {
        // local attention layer - reverse window partition
        cur = ggml_win_unpart(ctx0, cur, w0, h0, n_window_size);
    }

    cur = ggml_add_inplace(ctx0, cur, inpL);

    struct ggml_tensor * inpFF = cur;

    // feed-forward network
    {
        // norm
        {
            cur = ggml_norm(ctx0, inpFF, eps);

            // cur = mlp_ln_w*cur + mlp_ln_b
            cur = ggml_mul(ctx0, cur, norm2.w);
            cur = ggml_add_inplace(ctx0, cur, norm2.b);
        }

        // fully connected
        cur = ggml_mul_mat(ctx0, mlp_lin1.w, cur);
        cur = ggml_add_inplace(ctx0, cur, mlp_lin1.b);

        // GELU activation
        cur = ggml_gelu(ctx0, cur);

        // projection
        cur = ggml_mul_mat(ctx0, mlp_lin2.w, cur);
        cur = ggml_add_inplace(ctx0, cur, mlp_lin2.b);
    }

    inpL = ggml_add(ctx0, cur, inpFF);
    return inpL;
}

hGensor NT_SAM::Forward(hFISH graph,int nEmbed,int nHead,int W,int H,hGensor cur,int flag){
    struct ggml_context * ctx0 = graph->GetCTX();    
    cur = ggml_mul_mat(ctx0, in_proj.w, cur);
    cur = ggml_add_inplace(ctx0, cur, in_proj.b);
    int32_t head_dim = nEmbed / nHead; 

    // split qkv into separate tensors
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
    const int B = cur->ne[3];

    cur = ggml_reshape_4d(ctx0, cur, nEmbed, 3, W*H, B);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 3, 1, 2));

    struct ggml_tensor * Q;
    struct ggml_tensor * K;
    struct ggml_tensor * V;

    Q = ggml_view_3d   (ctx0, cur, nEmbed, W*H, B, cur->nb[1], cur->nb[2], 0*cur->nb[3]);
    Q = ggml_reshape_4d(ctx0, Q,   head_dim, nHead, W*H, B);
    Q = ggml_cont      (ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(ctx0, Q,   head_dim, W*H, B*nHead);

    K = ggml_view_3d   (ctx0, cur, nEmbed, W*H, B, cur->nb[1], cur->nb[2], 1*cur->nb[3]);
    K = ggml_reshape_4d(ctx0, K,   head_dim, nHead, W*H, B);
    K = ggml_cont      (ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(ctx0, K,   head_dim, W*H, B*nHead);

    V = ggml_view_3d   (ctx0, cur, nEmbed, W*H, B, cur->nb[1], cur->nb[2], 2*cur->nb[3]);
    V = ggml_reshape_4d(ctx0, V,   head_dim, nHead, W*H, B);
    V = ggml_cont      (ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
    V = ggml_reshape_3d(ctx0, V,   W*H, head_dim, B*nHead);

    struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

    struct ggml_tensor * KQ_scaled =
        ggml_scale_inplace(ctx0,
                KQ,
                1.0f/sqrtf(head_dim));

    struct ggml_tensor * rw = ggml_get_rel_pos(ctx0, rel_pos_w, W, W);
    struct ggml_tensor * rh = ggml_get_rel_pos(ctx0, rel_pos_h, H, H);

    struct ggml_tensor * q_r = ggml_reshape_4d(ctx0, Q, head_dim, W, H, B*nHead);

    struct ggml_tensor * rel_w = ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_mul_mat(ctx0,
                    rw,
                    ggml_cont(ctx0, ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
                0, 2, 1, 3));
    struct ggml_tensor * rel_h = ggml_mul_mat(ctx0, rh, q_r);

    struct ggml_tensor * attn = ggml_add_rel_pos_inplace(ctx0, KQ_scaled, rel_w, rel_h);

    struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, attn);

    struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

    cur =
        ggml_reshape_4d(ctx0,
                ggml_cont(ctx0,
                    ggml_permute(ctx0,
                        ggml_reshape_4d(ctx0, KQV, head_dim, W*H, nHead, B),
                        0, 2, 1, 3)),
                nEmbed, W, H, B);

    cur = ggml_mul_mat(ctx0, proj.w, cur);
    cur = ggml_add_inplace(ctx0, cur, proj.b);
    return cur;
}

/*
struct ggml_compute_state_shared {
    const struct ggml_cgraph * cgraph=nullptr;
    const struct ggml_cplan  * cplan=nullptr;
    int64_t perf_node_start_cycles=0;
    int64_t perf_node_start_time_us=0;
    const int n_threads=0;
    // synchronization primitives
    atomic_int n_active;  // num active threads
    atomic_int node_n=-1;    // active graph node
    atomic_int node_task=GGML_TASK_TYPE_FINALIZE; // active graph node task phase

    ggml_abort_callback abort_callback=NULL; // abort ggml_graph_compute when true
    void * abort_callback_data=NULL;
    ggml_compute_state_shared() {}
    ggml_compute_state_shared(const struct ggml_cgraph * cg,const struct ggml_cplan  * cp,int nt,int flag=0x0)
        :   cgraph(cg),cplan(cp),n_threads(nt),n_active(nt)  {
        
    }
    
};

typedef void * thread_ret_t;
typedef pthread_t ggml_thread_t;
#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

struct ggml_compute_state {
    ggml_thread_t thrd;
    int ith;
    struct ggml_compute_state_shared * shared;
};

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512
struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

struct ggml_context_container {
    bool used;

    struct ggml_context context;
};

struct ggml_state {
    struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
    struct ggml_numa_nodes numa;
};

// global state
static struct ggml_state g_state;
// static atomic_int g_state_barrier = 0;

#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static void ggml_graph_compute_thread_sync_node(int * node_n, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_node_n = * node_n;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * node_n = atomic_load(&state->shared->node_n);
        if (* node_n != last_node_n) break;
    }
}
static void ggml_graph_compute_thread_sync_task(int * task_phase, struct ggml_compute_state * state, const bool do_yield) {
    // wait for other threads to finish
    const int last_task_phase = * task_phase;

    while (true) {
        if (do_yield) {
            sched_yield();
        }

        * task_phase = atomic_load(&state->shared->node_task);
        if (* task_phase != last_task_phase) break;
    }
}*/
    
/*
    
*/
bool TGraph::TopoOrder(int flag)   {
    gimap.clear();          topo_nodes.clear();
    int pos=-1,nDup=0,i,no,nNode=cgraph->n_nodes,nLeaf=cgraph->n_leafs;
    hGensor cur,son;    
    // std::vector<hGensor> gensors;
    assert(sinks.size()>0);
    for(auto r : sinks){
        topo_nodes.push_back(r);        gimap[r] = GENSOR_INFO(0,0,-1,-1);    
        gimap[r].sX = r->name;
    }

    while(++pos<topo_nodes.size()) {
        cur = topo_nodes[pos];
        if(strcmp(cur->name,"result_output'")==0){      // only for debug
            int xxx = 0;
        }
        GENSOR_INFO info = gimap[cur];
        for (int i=0,no=0; i < GGML_MAX_SRC; ++i) {
            const int k =(order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :(order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) : i;
            if (!cur->src[k]) continue;
            son = cur->src[k];
            if(strcmp(son->name,"loss'")==0){           // only for debug
                int xxx = 0;
            }
            if(gimap.find(son) == gimap.end())  {                
                gimap[son] = GENSOR_INFO(topo_nodes.size(),info.level+1,pos,no++);
                gimap[son].sX = son->name;
                topo_nodes.push_back(son);
            }else{
                nDup++;
            }
                
        }        
    }
    
    // sort(gimap.begin(), gimap.end(), comp);
    return true;
}

string TGraph::__repr__(string& suffix,string& prefix,hGensor root_0,int flag) {
    const char*tab=prefix.c_str();     
    if(empty())   {
        _INFO("CGRAPH_%s is empty! root=%s",name.c_str(),root_0==nullptr?"":root_0->name); 
        return "";
    }
    string root_name = "";
    const size_t MAX_BUF=64*1024;
    char buf[MAX_BUF]="\0";
    sprintf(buf+strlen(buf),"\n CGRAPH_%s x=%d nodes=%d leafs=%d forward=(%d,%d)\n",name.c_str(), -1,cgraph->n_nodes, cgraph->n_leafs,nForwN,nForwL);
    
    // the output is always the last tensor in the graph
    int pos=-1,nDup=0,i,no,nNode=cgraph->n_nodes,nLeaf=cgraph->n_leafs,root_id=root_0==nullptr ? cgraph->n_nodes-1 : -1;
    hGensor root = root_0;
    if(root_0==nullptr){
        if(isBackward){

        }else
            root = cgraph->nodes[nNode-1];
    } 
#if !defined(NDEBUG)
#endif
    if(!root_name.empty()){
        for(int i=0;i<nNode;i++){     //pick root
            if(strcmp(cgraph->nodes[i]->name,root_name.c_str())==0){  //    l_out-1    inp_embd
                root = cgraph->nodes[i];     root_id=i;
                break;
            }
        }        
    }
    assert(root!=nullptr || sinks.size()>0);

    hGensor cur,son;    
    std::vector<hGensor> gensors,all_nodes;
    for(int i=0;i<nNode;i++)        {
        if(strcmp(cgraph->nodes[i]->name,"loss")==0)
            continue;
        all_nodes.push_back(cgraph->nodes[i]);
    }
    for(int i=0;i<nLeaf;i++)        all_nodes.push_back(cgraph->leafs[i]);    
    
    for(auto gensor:topo_nodes){
        gensors.push_back(gensor);
        _T_repr_(gensor,tab,buf,gimap[gensor]);     
        assert(strlen(buf)<MAX_BUF);
    }

    sprintf(buf+strlen(buf),"%s",suffix.c_str()); 
    _INFO("%s",buf); 
    int nMiss = all_nodes.size()-gensors.size();
    _INFO("CGRAPH_%s root=%d(%d) nPass=%ld(%d) nMiss=%d",name.c_str(),root_id,sinks.size(),gensors.size(),nDup,nMiss); 
    if(CHECK_SAME_TENSORS(name,gensors,all_nodes)!=0x0){   //       "loss"
        assert(has("loss")>=0);
        assert(0);  
    }
    return buf;
}   

TGraph::TGraph(Fish *hF_,const string&nam_,struct ggml_context *ctx_,bool isGrad,int flag) : hFish(hF_),ctx(ctx_),name(nam_)   {
    // hOPT = hFish->hOPT;
    cgraph = ggml_new_graph_custom(ctx, LLAMA_TRAIN_MAX_NODES, isGrad);
    /*
    const size_t obj_size = ggml_graph_nbytes(size, grads);
    struct ggml_object * obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_GRAPH, obj_size);
    struct ggml_cgraph * cgraph = (struct ggml_cgraph *) ((char *) ctx->mem_buffer + obj->offs);
    */
    nodes=cgraph->nodes;
    grads=cgraph->grads;
    leafs=cgraph->leafs;
}

void TGraph::PushBack(hGensor node,int flag) {
    assert(node!=nullptr && strlen(node->name)>0);
    const char*name = node->name;
    const int n0 = cgraph->n_nodes;
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != GGML_OP_NONE) {
            //GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    if (hash_insert(cgraph->visited_hash_set, node) == GGML_HASHSET_ALREADY_EXISTS) {
        return;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const int k =
            (order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i :
            (order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT) ? (GGML_MAX_SRC-1-i) :
            /* unknown order, just fall back to using i*/ i;
        if (node->src[k]) {
            if(strlen(node->src[k]->name)==0){  //  in ggml_compute_backward, some grad has no name!
                // assert(isBackward);
                ggml_format_name(node->src[k], "%s_%d",node->name,k);
            }
            PushBack(node->src[k]);
        }
    }

    if (node->op == GGML_OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        GGML_ASSERT(cgraph->n_leafs < cgraph->size);

        if (strlen(node->name) == 0) {
            ggml_format_name(node, "lef_%d",cgraph->n_leafs);
        }
        leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        GGML_ASSERT(cgraph->n_nodes < cgraph->size);
        if (strlen(node->name) == 0) {
            ggml_format_name(node, "stem_%d", cgraph->n_nodes);
        }
        nodes[cgraph->n_nodes] = node;
        if (grads) {
            grads[cgraph->n_nodes] = node->grad;
        }
        if(node->grad!=0x0 && strlen(node->grad->name)==0){
            ggml_format_name(node->grad, "%s_grad", node->name);
        }
        cgraph->n_nodes++;
    }

    if (cgraph->n_nodes > n0) {        
        assert(nodes[cgraph->n_nodes - 1] == node);
        if(node->grad && strlen(node->grad->name)==0){
            ggml_format_name(node->grad,"%s_grad",node->name);
        }
    }
}

bool TGraph::isSink(hGensor node,int flag){
    for(auto n : sinks){
        if(n==node)
            return true;
    }
    return false;
}

int Fish::BuildComputeGraph(int order,struct ggml_context * ctx,ggml_gallocr_t& alloc,int flag){
    // hForwTG->Clear();               hBackTG->Clear();        
    auto train_params = hparams.common;
    train_params.use_checkpointing = false;     // CYS_0826
    assert(ctx==ctx_build);
    struct ggml_cgraph *gf = hForwTG->raw(), *gb = nullptr;
    if(order>=0){   //order<0: we have build it in other way
        gf->order = (enum ggml_cgraph_eval_order) order;    
        hForwTG->PushBack(out_node);        hForwTG->sinks.push_back(out_node);
        // ggml_build_forward_expand(gf, out_node);  
    }
    if(!hForwTG->isSink(out_node)){
        _INFO("%s %s is not Sink!!!",__func__,out_node->name);
        return -1;
    }
    assert(hForwTG->isValid());
    hForwTG->TopoOrder();
    hForwTG->__repr__(out_node);  
    int n0=gf->n_nodes; 
    if(!isLocalInfer){       
        hBackTG = std::make_shared<TGraph>(this,hForwTG->name+".Backward",ctx_build,true);        hBackTG->isBackward = true; 
        gb = hBackTG->BuildBackward(ctx,gf);
        // make sure some tensors are not reallocated by inserting new temporary nodes depending on them
        int n_leafs_before = gb->n_leafs,n_nodes_before = gb->n_nodes;
        ggml_set_input(out_node->grad);
        hBackTG->PushBack(ggml_scale_inplace(ctx, KQ_pos, 1.0f));        
        
        bool isReforward = false;        //why???
        if(isReforward){// output_ tensors
            ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, preLogits, 1.0f));
            ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, loss, 1.0f));
            // input gradient
            ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, out_node->grad, 1.0f));
            assert(out_node->grad->data == NULL && out_node->grad->view_src == NULL);
            ggml_set_input(out_node->grad);
            // KQ_pos
            ggml_build_forward_expand(gb, ggml_scale_inplace(ctx, KQ_pos, 1.0f));
            // allocating checkpoints in one block to reduce memory fragmentation they will be freed in reverse order
            for (unsigned int i = 0; i < checkpoints.size(); ++i) {
                if (checkpoints[i]->data == NULL && checkpoints[i]->view_src == NULL) {
                    ggml_set_input(checkpoints[i]);
                }
            }        
        }
        if (measure_only) {
            ggml_gallocr_reserve(alloc, gb);
        } else {
            ggml_gallocr_alloc_graph(alloc, gb);    //367,8088  
        }
        // remove the additional nodes and leafs
        for (int i = n_leafs_before; i < gb->n_leafs; ++i) {
            gb->leafs[i] = NULL;
        }
        for (int i = n_nodes_before; i < gb->n_nodes; ++i) {
            gb->nodes[i] = NULL;
        } 
        gb->n_leafs = n_leafs_before;
        gb->n_nodes = n_nodes_before;

        
        // hBackTG->__repr__(out_node);        
    }  
    
    auto leaf0=gf->nodes[0];
    const int N = train_params.n_ctx, n_past = 0;
    if (train_params.use_checkpointing) {
        if(gb!=nullptr) {
            gb_tmp = train_params.use_checkpointing ? ggml_new_graph_custom(ctx_build, LLAMA_TRAIN_MAX_NODES, true) : NULL;
            ggml_build_backward_gradient_checkpointing(ctx, gf, gb, gb_tmp, checkpoints.data(), (int) checkpoints.size());
        }            
    } 

    assert(alloc != NULL);
    if(isLocalInfer){  //gb=nullptr
        GGML_ASSERT(alloc != NULL);
        assert(gb==nullptr);
        ggml_gallocr_alloc_graph(alloc, gf);         
        // int * data = (int *) KQ_pos->data;
        // for (int i = 0; i < N; ++i) {
        //     data[i] = n_past + i;
        // }
    } else { // gb!=nullptr
        
    }
    int * data = (int *) KQ_pos->data;
    for (int i = 0; i < N; ++i) {
        data[i] = n_past + i;
    }
    
    return 0x0;
}

int TGraph::has(const string&name,int flag){
    std::map<std::string, int> msg;
    int nLeaf = cgraph->n_leafs,nNode = cgraph->n_nodes,no=1,nDup=0;
    std::vector<hGensor> gensors,all_nodes;
    for(int i=0;i<nNode;i++)        {
        if(name==cgraph->nodes[i]->name){
            return i;
        }
        all_nodes.push_back(cgraph->nodes[i]);
    }
    for(int i=0;i<nLeaf;i++)        {
        all_nodes.push_back(cgraph->leafs[i]);
        if(name==cgraph->leafs[i]->name){
            return i+nNode;
        }
    }

    return -1;
}

bool TGraph::isValid(){
    std::map<std::string, int> msg;
    int nLeaf = cgraph->n_leafs,nNode = cgraph->n_nodes,no=1,nDup=0;
    std::vector<hGensor> gensors,all_nodes;
    for(int i=0;i<nNode;i++)        all_nodes.push_back(cgraph->nodes[i]);
    for(int i=0;i<nLeaf;i++)        all_nodes.push_back(cgraph->leafs[i]);

    for(auto tA:all_nodes){
        if(msg.find(tA->name)!=msg.end()){            
            int j = msg[tA->name];
            hGensor tB=all_nodes[j];
            _INFO("\tAA_[%d %d]=\"%s\" !!!\n",j,no,tA->name); 
            nDup++;
        }
        msg[tA->name] = no;     
        no++;
    }
    if(nDup>0)
        return false;
    return true;
}
/*
    struct ggml_hash_set {
        size_t size;
        uint32_t * used;
        struct ggml_tensor ** keys;
    };
    */
extern "C" void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, struct ggml_hash_set * zero_table);
struct ggml_cgraph * TGraph::BuildBackward(struct ggml_context * ctx_,struct ggml_cgraph *gf,int flag)   {    
    struct ggml_cgraph *gb=cgraph;
    assert(isBackward && gf!=nullptr);
    nForwN=gf->n_nodes,nForwL=gf->n_leafs;
    int n_grad=0,n_param=0,n_p0=0;
    size_t sz = gf->size;
    bool isKeep = false;
    assert(gb!=nullptr);
    hGensor xn = hFish->xn,xxn = hFish->xxn,root_f=gf->nodes[nForwN-1],root_b=nullptr;
    
    ggml_graph_cpy(gf, gb);   //copy all leafs/nodes/grads & visited_hash_set
    // ggml_build_backward_expand(ctx_, gf, gb, true);    return gb;    
         
    if (isKeep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];
            if (node->grad) {
                node->grad = ggml_dup_tensor(ctx_, node);
                gf->grads[i] = node->grad;      n_grad++;
            }            
        }
    }
    
    // remember original gradients which start with zero values
    struct ggml_hash_set zero_table = ggml_hash_set_new(gf->size);
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->grads[i]) {
            ggml_hash_insert(&zero_table, gf->grads[i]);
        }        
    } 
//  1 ggml_set_param would set FLAG & init grad(zero)  2 dst would create grad if its src has grad
    n_grad=0,n_param=0,n_p0=0;
    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];
        if(node==xn){   //only for debug
            int xxx = 0;
        }
        if (node->grad) {   //set src's grad
            if(node->grad->grad!=NULL)  //
                node->grad->grad = NULL;
                
            if(strlen(node->grad->name)==0){
                gTN(node,"");
            }
            if(i==6){   //  inp_pe_rms
                int xxx=0;
            }
            ggml_compute_backward(ctx_, node, &zero_table);
            // if(xn->grad!=xxn){
            //     int xxx=0;
            // }
            n_grad++;
        }
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_p0++;
        }
    }
    assert(isValid()); 
    root_b=gb->nodes[gb->n_nodes-1];
    for (int i = 0; i < gf->n_nodes; i++) {
        struct ggml_tensor * node = gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            // _INFO("%s: found root node %p\n", __func__, (void *) node);
            // ggml_build_forward_expand(gb, node->grad);
            assert( !(node->grad->flags & GGML_TENSOR_FLAG_PARAM) );
            gTN(node,"");            
            PushBack(node->grad);
            sinks.push_back(node->grad);
            n_param++;
        }
    }
    int n_bleafs = gb->n_leafs,n_bnodes = gb->n_nodes;
    ggml_hash_set_free(&zero_table);       

    TopoOrder();
    
    __repr__( ); 
    
    return gb;
}

void TGraph::Traverse(int flag){
    bool GGML_OP_HAS_INIT    [GGML_OP_COUNT] = { 0 };
    bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = { 0 };
    int node_n     = -1;
    while (true) {
        if (node_n != -1) {                /* FINALIZE */
            struct ggml_tensor * node = cgraph->nodes[node_n];
            if (GGML_OP_HAS_FINALIZE[node->op]) {
                //params.nth = ggml_get_n_tasks(node, n_threads);
            }
        }
        
        while (++node_n < cgraph->n_nodes) { // distribute new work or execute it direct if 1T
            GGML_PRINT_DEBUG_5("%s: %d/%d\n", __func__, node_n, cgraph->n_nodes);
            struct ggml_tensor * node = cgraph->nodes[node_n];
            if (1) {
                if (GGML_OP_HAS_INIT[node->op]) {
                    //params.type = GGML_TASK_TYPE_INIT;
                }
                if (GGML_OP_HAS_FINALIZE[node->op]) {
                    //params.type = GGML_TASK_TYPE_FINALIZE;
                    // ggml_compute_forward(&params, node);
                }
                // ggml_graph_compute_perf_stats_node(node, state->shared);
            } else {
                break;
            }
        }
    }
}

void * _graph_pass_thread(void * data) {    
    assert(0);
    return GGML_EXIT_SUCCESS;
}

int TGraph::compute_on_plan( struct ggml_cplan* cplan,int flag) {
    return ggml_graph_compute(cgraph, cplan);
    /*int compute_status = GGML_EXIT_ABORTED;
    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    if (cplan->work_size > 0) {
        GGML_ASSERT(cplan->work_data);
    }
    GST_TIC(t0);   
#ifdef GGML_USE_VULKAN
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_preallocate_buffers_graph_cpu_assist(cgraph->nodes[i]);
    }
    ggml_vk_preallocate_buffers_cpu_assist();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_vk_build_graph_cpu_assist(cgraph->nodes[i], i == cgraph->n_nodes - 1);
    }
#endif

    const int n_threads = cplan->n_threads;

    struct ggml_compute_state_shared state_shared(cgraph,cplan,n_threads);
        
    struct ggml_compute_state * workers = (struct ggml_compute_state *)alloca(sizeof(struct ggml_compute_state)*n_threads);
    
    // create thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; ++j) {
            workers[j].thrd=0;      workers[j].ith=j;   workers[j].shared=&state_shared;
            workers[j] = (struct ggml_compute_state) {
                .thrd   = 0,
                .ith = j,
                .shared = &state_shared,
            };
            if(isOnlySymbol){
                const int rc = pthread_create(&workers[j].thrd, NULL, _graph_pass_thread, &workers[j]);
                GGML_ASSERT(rc == 0);                UNUSED(rc);   
                // _graph_pass_thread(&workers[j]);
            }   else{
                // const int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
                const int rc = pthread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
                GGML_ASSERT(rc == 0);                UNUSED(rc);                
            }
        }
    }

    workers[0].ith = 0;
    workers[0].shared = &state_shared;

    const int64_t perf_start_cycles  = ggml_perf_cycles();
    const int64_t perf_start_time_us = ggml_perf_time_us();
    if(isOnlySymbol)
        _graph_pass_thread(&workers[0]);
    else
        compute_status = (size_t) ggml_graph_compute_thread(&workers[0]);

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    // join or kill thread pool
    if (n_threads > 1) {
        for (int j = 1; j < n_threads; j++) {
            const int rc = ggml_thread_join(workers[j].thrd, NULL);
            GGML_ASSERT(rc == 0);
        }
    }

#ifdef GGML_USE_VULKAN
    ggml_vk_graph_cleanup_cpu_assist();
#endif

    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

        cgraph->perf_runs++;
        cgraph->perf_cycles  += perf_cycles_cur;
        cgraph->perf_time_us += perf_time_us_cur;

        GGML_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, cgraph->perf_runs,
                (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
                (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
    }
    tCompute = GST_TOC(t0);
    return compute_status;*/
}

void s2layerinfo(const string&jkey,std::vector<string>&lays){
    lays.clear();
    const char* seps=" ,:;{}()\t=";
    string nam_0;
    char *token = strtok((char*) jkey.c_str(), seps);
    int no=0,nLay=1;
    while (token != NULL) {
        if(no==0)
            nam_0 = token;
        else{
            if(token[0]=='*'){
                if(sscanf(token+1,"%d",&nLay)==1)   {

                }else{

                }
            }
        }
        token = strtok(NULL, seps);     no++;
    }
    if(nLay>1){
        for(int i=0;i<nLay;i++){
            string name=nam_0+".L"+std::to_string(i);
            lays.push_back(name);
        }
    }else{
        lays.push_back(nam_0);
    }
}

hNeuron Fish::J2Neuron(struct ggml_context *ctx_,string& dad,int level,const JConfig& config,int flag){
    hNeuron hN=nullptr,cur=nullptr;
    std::vector<hNeuron> gang;    
    string k,nam_,prefix;
    std::vector<string> lay_names;
    int i,nLay;
    for(JSON::const_iterator it = config.js.begin(); it != config.js.end(); ++it)    {
        k =it.key();     
        if(!k.empty() && k[0]=='#')     
            continue;
        if(k=="parameter"){
            // BuildMacros();
            continue;
        }
        auto v=it.value();
        if(it->is_array()){

        }else if(it->is_structured())        {
            s2layerinfo(k,lay_names);
            int lay = 0;
            for(auto nam_ : lay_names){
                JConfig jLay(*it,lay++);
                prefix = dad.empty()?nam_:dad+"."+nam_;      //  ,  //nam_
                cur = J2Neuron(ctx_,prefix,level+1,jLay,flag);  
                gang.push_back(cur);        
            }   
            continue;       
        }
        else        {
            assert(0);          
        }       
        cur = GeNeuron::MakeInstance(this,ctx_,dad,it,flag);        
        cur->ID = config.ID;        cur->level = level+1;
        neurons.push_back(cur);  
        gang.push_back(cur);
    }
    assert(gang.size()>0);
    if(gang.size()>1)   {
        hN = std::make_shared<Ganglia>(this,dad,gang,flag);     hN->level = level;
        neurons.push_back(hN);  
    }else{
        assert(cur!=nullptr);
        hN = cur;
    }
    assert(hN->isValid());
    return hN;
}

/*

*/
int Fish::jToGraph( struct ggml_context *ctx_,bool isBuild,int flag)   {
    JConfig js(hparams.jModel);
    string sRoot;
    J2Neuron(ctx_,sRoot,0,js,flag);   //  "GPT2"
    // for(auto nn : neurons){    //symbolic analysis
    //     nn->Forward(ctx_,nullptr);
    // }

    int n_batch=hparams.n_batch(),n_ctx=hparams.n_ctx(),n_ctx_train=hparams.n_ctx_train,n_embd=hparams.n_embd;
    hGensor cur = tBatch; 
    for(auto nn : neurons){
        cur = nn->Forward(ctx_,cur);
    }
    preLogits = cur;
    // out_node = BuildLoss(ctx_,preLogits);
    return 0x0;
}