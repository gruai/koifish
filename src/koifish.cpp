
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include "./Manifold/gLLM.hpp"
#include "./Manifold/VAE.hpp"
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int g_dump_level = 1;

int main(int argc, char ** argv) {
    // print_build_info();
    _INFO("[ARCH] token=%ld,floatX=%ld\n", sizeof(TOKEN_ID),sizeof(floatX));
    CLI_params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }

    params.OnArch();    
    // if(params.test=="GPT_work")      //  Deprecated
    //     return GPT_work(params);    
    if(params.test=="fish_1")
        return fish_1(params);   
    if(params.test=="GGUF_list")
        return GGUF_list(params);     
    if(params.test=="bubble")
        return Fish_bubble(params);   
    if(params.test=="token")
        return Fish_token(params);   
    // if(params.test=="tutor")
    //     return Tutor(params);   
    
    hFISH fish = nullptr;
    if(params.n_swarm>1)   {
        fish = Fish::MakeSwarm("Fish_",params,0x0);    
    }else {
        params.common.n_gpu_layers = 40;
        vector<hWIKI> wikis = WIKI::MakeInstance("",params,0x0);      
        if(wikis.size()==0){
            // _INFO("====== NO WIKI !!! ======\n");       return;
        }else if(params.wiki_actor=="copy" ){
            wikis[0]->CopyParams(params);      
        }   
        fish = Fish::MakeInstance("Fish_",params,wikis,Fish::ROLE_TYPE::COMMON,0x0);    
    } 
    if(fish && fish->isTrain())
        fish->Train( );

    return 0x0;    
}

