/**
 *  Copyright 2023-2024 by Grusoft 
 * 
 *  \brief
 *  \author Yingshi Chen
 */
#pragma once
#include "Ganglia.hpp"

/**
 * A abstract model of Multi-level Encoder/Decoer
*/
class MutliCoder : public Ganglia   {
protected:
    int nTop=-1,nBottom=-1;
    
public:
    hGensor encode=nullptr,decode=nullptr;

    MutliCoder(struct ggml_context *ctx,int dim1,int dim2,int flag=0x0) : nTop(dim1),nBottom(dim2) {
        assert(nTop>nBottom && nBottom>0);
        encode = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nTop, nBottom);     
        decode = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nBottom, nTop); 
    }

    virtual hGensor ENC(struct ggml_context *ctx,hGensor x){
        x = ggml_mul_mat(ctx, encode, x );    

        return x;
    }

    virtual hGensor DEC(struct ggml_context *ctx,hGensor x){
        x = ggml_mul_mat(ctx, decode, x );    

        return x;
    }
};
typedef shared_ptr<MutliCoder> hMultiCoder;

class VariationaAE : public Ganglia   {
protected:
    int nRefine = 1;
    bool reserve_x = false;
    vector<hGensor> resi_x;
    vector<float> hier_norm;
    
    hGanglia callosum;

    virtual hGensor _build_coder( bool isDown,hGensor x=nullptr )        {
        /*x_hier = 0  
        if x is None:
            assert(self.first_embed is not None)        
            x=torch.eye(self.T_last).to(self.device)
            if self.first_embed is not None:
                x = self.first_embed(x,graph.MAEC.graphs[-1])*/
        // map_range = range(graph.MAEC.nRefine) if self.down else reversed(range(graph.MAEC.nRefine))
        vector<int> map_range;
        for( auto map_no : map_range )  {
            // hier_norm.push_back(torch.norm(x).item())            
            auto map = MAEC[map_no];
            //  x_1 = map.hier_feat(x)
            // x_hier = x_hier+self.hier_mlp[map_no](x_1)/graph.num_nodes*1.0e-6      
            if(!isDown)   
                x = map->DEC(ctx,x);      //up_pooling
            // if not self.down and hasattr(graph,"resi_x"):
            //     assert(x.shape==graph.resi_x[map_no].shape)
            //     x = (x + graph.resi_x[map_no])/2
            // combined=[x]
            /*for i in range(self.post):      # post-smoothing                
                x = self.conv_post[i](x,map )     
                if self.activation is not None:     
                    x = self.activation(x)
            # combined.append(x)*/
            

            if (reserve_x)
                resi_x.push_back(x);
            if (isDown)
                x = map->ENC(ctx,x);    //Pool_x(x,map.cluster)
            /*########## SVD ? QR 
            # if x.shape[0]>x.shape[1]:
            #     q,r =self.Feat_normal(x,mode='reduced')
            #     assert(q.shape==x.shape)
            #     x = q
            ##########?  why different post_conv at each refine is useless???
            # x = self.conv_post[map_no](x, edge_index)
            # x = self.activation(x)*/
        }
        /*# print(f"\thier_norm={''.join(f'{k:.3g} 'for k in graph.hier_norm)}\t")  
        if self.MLP is not None:
            for blk in self.MLP:
                x = blk(x)     
        if self.APPNP is not None:
            x = self.APPNP(x,graph.edge_index) */    
        
        return x;
    }vector<hMultiCoder> MAEC;    //  multi-level auto encoder
public:
    
    virtual hGensor ENC(struct ggml_context *ctx,hGensor x){
        hGensor cur = x;
        for(auto coder:MAEC)
            cur = coder->ENC(ctx, cur);
        return cur;
    }

    virtual hGensor DEC(struct ggml_context *ctx,hGensor x){
        hGensor cur = x;
        for(auto coder:MAEC)
            cur = coder->DEC(ctx, cur);
        return cur;
    }

    void BuildGraph(int flag=0x0)   override   { 
    }

    virtual void Build( hGensor x=nullptr )        {
        x = _build_coder(true,x);       //encoder
        callosum->BuildGraph( );      //  self.scale_MLP( torch.cat(combined, dim=1))                              //
        x = _build_coder(false,x);      //decoer
    }

    VariationaAE(   )       {  }
    VariationaAE(struct cwd_params params,int flag=0x0) : Ganglia(params)  {}
    virtual ~VariationaAE() {
    }  
};
typedef shared_ptr<VariationaAE> hVAE;

/**
 * LAMA+VAE

class VAE_LAMA : public VariationaAE {
protected:
    int lama_embed = 0,var_embed = 192;
public:
    VAE_LAMA( struct cwd_params params,int flag=0x0) : VariationaAE(params) {
        callosum = std::make_shared<LLAMA_VAE>(params);
    }

    virtual ~VAE_LAMA() {
    }  

    virtual void Init(int flag=0x0) override        {
        callosum->Init();
    }

    void BuildGraph(int flag=0x0)   override        {  
        callosum->BuildGraph( );
    }
    
    void Train(int flag=0x0)    override            {
        callosum->Train( );
    }
};
typedef shared_ptr<MutliCoder> hMultiCoder;*/