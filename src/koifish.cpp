
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

int main(int argc, char ** argv) {
    print_build_info();
    
    struct CLI_params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }   
    assert(params.nabla > 0);     

    if (params.common.seed == LLAMA_DEFAULT_SEED) {
        params.common.seed = time(NULL);
    }
    printf("\n%s: seed: %u\n", __func__, params.common.seed);
    srand(params.common.seed);

    if(params.test=="GPT_work")
        return GPT_work(params);    
    if(params.test=="fish_1")
        return fish_1(params);   
    if(params.test=="GGUF_list")
        return GGUF_list(params);     
    if(params.test=="bubble")
        return Fish_bubble(params);   
    if(params.test=="tutor")
        return Tutor(params);   
        
    hFISH fish = nullptr;
    if(params.n_swarm>1)   {
        fish = Fish::MakeSwarm("Fish_",params,0x0);    
    }else {
        vector<hWIKI> wikis = WIKI::MakeInstance("",params,0x0);        
        fish = Fish::MakeInstance("Fish_",params,wikis,Fish::ROLE_TYPE::COMMON,0x0);    
    } 
    if(fish->isTrain())
        fish->Train( );

    return 0x0;    
}