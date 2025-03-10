
/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief 
*  \author Yingshi Che
 */
#include "GTensor.hpp"
#include "GG_util.hpp"
#ifdef ENABLE_BF16
   typNUMBER GTensor::tpFloatX = typNUMBER::BF16;
#else
   typNUMBER GTensor::tpFloatX = typNUMBER::F32;
#endif

// int B=0,T=0,C=0;
hGTensor GTensor::scratch_bt4c=nullptr,GTensor::scratch_btc=nullptr,GTensor::scratch_output=nullptr,GTensor::scratch_ff1=nullptr;

float GTensor::rLARS(float s0,float T_lars,int flag)   {
   if( shape.size()<=1 )
      return s0;
      
   float eps = 1.0e-8;
   float trust_ratio =wnorm/(gnorm+eps);
   trust_ratio = std::min(trust_ratio,T_lars);
   float r = trust_ratio;
   return r;
}

GTensor::GTensor(SHAPE shape_,typNUMBER tpD_,bool isX,int flag) : shape(shape_),type(tpD_),flags(flag)      {
   int i=0;
   for(auto n : shape){
      ne[i++] = n;
      assert(n>0 && "");
   }
   for(i=shape.size();i<GGML_MAX_DIMS;i++)  
      ne[i]=1;
   double nB = BPE(type);    assert(nB>=1.0);
   // from ggml_new_tensor
   size_t szBlk = ggml_blck_size((enum ggml_type)type);
   nb[0] = BPE(type);       assert(szBlk==1);
   nb[1] = nb[0]*(ne[0]/szBlk);
   for (int i = 2; i < GGML_MAX_DIMS; i++) {
      nb[i] = nb[i - 1]*ne[i - 1];
   }

   
   szData = size()*nB;   
}

float GTensor::Get(int i,int flag)  const    {   
   assert(0);     return 0.f;
   // return ggml_get_f32_1d(gg, i); 
}
/*
   Only for gguf-serialize
*/
struct ggml_tensor* GTensor::GG( ) {   
   if(gg==nullptr){
      gg = new ggml_tensor();
#ifdef GG_V12
#else
      *gg = (struct ggml_tensor) {     // @ggml_new_tensor_impl
         /*.type         =*/ type,
         /*.backend      =*/ GGML_BACKEND_TYPE_CPU,
         /*.buffer       =*/ NULL,
         /*.ne           =*/ { ne[0],ne[1],ne[2],ne[3] },
         /*.nb           =*/ { nb[0],nb[1],nb[2],nb[3] },
         /*.op           =*/ GGML_OP_NONE,
         /*.op_params    =*/ { 0 },
         /*.flags        =*/ flags,
         /*.grad         =*/ NULL,
         /*.src          =*/ { NULL },
         /*.view_src     =*/ view_src,
         /*.view_offs    =*/ view_offs,
         /*.data         =*/ data,
         /*.name         =*/ { 0 },
         /*.extra        =*/ NULL,
         ///*.padding      =*/ { 0 },
      };
#endif
      gg->data = new char[szData];
      memcpy(gg->name,name,sizeof(char)*GGML_MAX_NAME);
   }
   size_t sz   = ggml_nbytes(gg);     // 154389504
   assert(sz==szData);
#ifdef _TENSOR_G_
    bool toHost = SerialGP(gg->data,nullptr,true,0x0);
    assert(toHost);
#endif
    
/*
   for (uint32_t i = 0; i < ctx->header.n_tensors; ++i) {
        struct gguf_tensor_info * info = &ctx->infos[i];

        const size_t size     = info->size;
        const size_t size_pad = GGML_PAD(size, ctx->alignment);

        gguf_bwrite_el(buf, info->data, size);

        if (size_pad != size) {
            uint8_t pad = 0;
            for (size_t j = 0; j < size_pad - size; ++j) {
                gguf_bwrite_el(buf, &pad, sizeof(pad));
            }
        }

        GGML_ASSERT(offset == info->offset);

        offset += size_pad;
    }
*/
   assert(isParam());
   return gg;  
}

bool GTensor::Alloc(int tpInit,int flag){
   assert(szData>0);
   data = new char[szData];
   if(isParam()){
      grad = new char[szData];
   }
   return true;
}
GTensor::~GTensor()  {
#ifdef _TENSOR_G_   
#else
   if(data!=nullptr)       
      delete[] (char*)data;
   if(grad!=nullptr)       
      delete[] (char*)grad;
#endif
}

void GTensor::AddSrc(const vector<hGTensor>& ts,int flag) {
   for(auto t : ts)   {
      if(t==nullptr) 
         continue;
      AddSrc(t);
   }      
}

void GTensor::Set(float a,int flag)    {   
   assert(!isEmpty());
   if(a==0){
      memset(data,0x0,szData);
   }else{
      assert(0);
   }
   //ggml_set_f32(gg, 1.0f); 
}
bool GTensor::OverWrite(struct ggml_tensor*gg_,bool isSrc,int flag){
   assert(size()==ggml_nelements(gg_));
   assert(type==(typNUMBER)gg_->type);
   if(isSrc){
      memcpy(data,gg_->data,szData);
   }else{
      memcpy(gg_->data,data,szData);
   }
   
   return true;
}
bool GTensor::OverWrite(hGTensor hGT,bool isSrc,int flag)  {   
   /*cuTensor *src = dynamic_cast<cuTensor *>(hGT.get());
   size_t nEle = size();
   assert(isSameShape(hGT));    
   if(src!=nullptr){
      // cudaCheck(cudaMemcpy(data, src->data, szData, cudaMemcpyHostToDevice));
      return true;
   }*/
   assert(0);
   return false;
}

hGTensor GTensor::Relu() 
{  auto cur=ggml_relu(nullptr, gg);  return NEW_(cur);  }
hGTensor GTensor::Silu() 
{  auto cur=ggml_silu(nullptr, gg);  return NEW_(cur);  }
hGTensor GTensor::Norm(float epsilon,int flag) 
{  auto cur=ggml_silu(nullptr, gg);  return NEW_(cur);  }

hGTensor GTensor::CrossEntropy( const hGTensor b,int flag )   	{
   auto cur = ggml_cross_entropy_loss(nullptr,gg, b->GG() );   
   // ggml_cross_entropy_loss_1(_ctx, cur, target_probs); 
   return GTensor::NEW_(cur);
}

hGTensor GTensor::GetRow(hGTensor, hGTensor tokens,hGTensor pos,int flag)   {
   assert(0);     //GGML VERSION
   // assert(ne[1]==shape[0]);
   // struct ggml_tensor *cur = ggml_get_rows(_ctx, gg, tokens->gg);       gTN(cur, name);   
   // if(pos!=nullptr)        {
   //    cur = ggml_add(_ctx, cur, pos->gg);  
   // } 
   // return GTensor::NEW_(cur);
   return nullptr;
}

hGensor GENSORS::Get(const string&name, int flag)    {        
   if(flag==0x100){    //  .weight=>.w
      for(auto ng:nag){
            if(strstr(name.c_str(),ng.first.c_str())!= NULL){
               return ng.second;
            }
      }
      return nullptr;
   }else{
      if(nag.find(name) == nag.end()){
         _ERROR("Failed to get tensor=%s nGensor=%d",name.c_str(),nag.size());  
         return nullptr;
      }
      return nag[name];
   }
   
} 

    // inline hGensor To4D(struct ggml_context * ctx_build,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4){
    //     cur = ggml_reshape_4d(ctx_build, cur, n1, n2,n3,n4);
    //     return cur;
    // }
    // inline hGensor Permute(struct ggml_context * ctx_,hGensor cur,int64_t n1,int64_t n2,int64_t n3,int64_t n4,bool isCont=true)    {
    //     hGensor q = ggml_permute(ctx_, cur, n1,n2,n3,n4);   
    //     gTN0(q,"%s.#",cur->name);     
    //     if(isCont)    {
    //         q = ggml_cont(ctx_,q);        
    //         gTN(q,"%s.#c",cur->name);           
    //     }
    //     return q;
    // }