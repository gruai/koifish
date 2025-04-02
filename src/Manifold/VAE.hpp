/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief
 *  \author Yingshi Chen
 */
#pragma once
#include "Fish.hpp"

class VariationaAE : public Fish   {
protected:
    int nRefine = 1, tpNorm=2;
    bool isSymmetric = true;
    bool reserve_x = false;
    vector<hGensor> resi_x;
    vector<float> hier_norm;
    std::vector<int> dims;        
    hFISH callosum=nullptr;

    virtual hGensor _build_coder( bool isDown,hGensor x=nullptr )        {
        /*x_hier = 0  
        if x is None:
            assert(self.first_embed is not None)        
            x=torch.eye(self.T_last).to(self.device)
            if self.first_embed is not None:
                x = self.first_embed(x,graph.MAEC.graphs[-1])*/
        // map_range = range(graph.MAEC.nRefine) if self.down else reversed(range(graph.MAEC.nRefine))
        vector<int> map_range;
        auto ctx = GetGGCTX();
        for( auto map_no : map_range )  {
            // hier_norm.push_back(torch.norm(x).item())            
            auto map = MAEC[map_no];
            //  x_1 = map.hier_feat(x)
            // x_hier = x_hier+self.hier_mlp[map_no](x_1)/graph.num_nodes*1.0e-6      
            if(!isDown)   
                x = map->DEC(x);      //up_pooling
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
                x = map->ENC(x);    //Pool_x(x,map.cluster)
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
    }
    vector<hVarCoder> MAEC;    //  multi-level auto encoder

public:
    virtual int InitMAEC(void *ctx,const std::vector<int>& dims_,int flag=0x0) {
        dims = dims_;
        int nMap = dims.size()-1;       assert(nMap>0);
        MAEC.clear( );
        for(int i=0;i<nMap;i++){
            hVarCoder hCoder = std::make_shared<VarCoder>(this,dims,i,reserve_x,isSymmetric,tpNorm);
            MAEC.push_back(hCoder);            
        }

        return MAEC.size();
    }

    virtual void save_gguf(struct gguf_context *, int flag);

    virtual hGensor ENC(void *ctx,hGensor x){
        hGensor cur = x;
        for(auto coder:MAEC)
            cur = coder->ENC(cur);
        return cur;
    }

    virtual hGensor DEC(void *ctx,hGensor x){
        hGensor cur = x;
        for (auto it = MAEC.rbegin(); it != MAEC.rend(); ++it)
            cur = (*it)->DEC(cur);
        return cur;
    }

    /*
        Reparameterization trick to sample from N(mu, var) from        N(0,1).
        z = eps * std + mu
    */
    virtual void Latent()    {
        /*mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu*/
    }

    bool Build(int flag=0x0)   override   { 
        return false;
    }

    virtual void Build( hGensor x=nullptr )        {
        x = _build_coder(true,x);       //encoder
        callosum->Build( );      //  self.scale_MLP( torch.cat(combined, dim=1))                              //
        x = _build_coder(false,x);      //decoer
    }

    VariationaAE(   )       {  }
    VariationaAE(struct CLI_params params,bool isRes,int flag=0x0) : Fish("VariationaAE",params),reserve_x(isRes)  {}
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
    VAE_LAMA( struct CLI_params params,int flag=0x0) : VariationaAE(params) {
        callosum = std::make_shared<LLAMA_VAE>(params);
    }

    virtual ~VAE_LAMA() {
    }  

    virtual void Init(int flag=0x0) override        {
        callosum->Init();
    }

    void Build(int flag=0x0)   override        {  
        callosum->Build( );
    }
    
    void Train(int flag=0x0)    override            {
        callosum->Train( );
    }
};
typedef shared_ptr<VarCoder> hVarCoder;*/