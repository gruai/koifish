
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include "./Manifold/gLAMA.hpp"
#include "./Manifold/VAE.hpp"
#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif



int main(int argc, char ** argv) {
    struct CLI_params params;
    if (!params.parse(argc, argv)) {
        return -1;
    }   
    assert(params.nabla > 0);     

    if (params.common.seed == LLAMA_DEFAULT_SEED) {
        params.common.seed = time(NULL);
    }
    printf("%s: seed: %u\n", __func__, params.common.seed);
    srand(params.common.seed);
    hGanglia fish = nullptr;
    switch(params.nabla){
    case 1:
        fish = std::make_shared<LLAMA_LORA>(params);
        break;
    case 2:
        fish = std::make_shared<LLAMA_Brown>(params);
        break;
    case 3:
        fish = std::make_shared<LLAMA_VAE>(params);
        break;
    
    default:
        assert(0);
    }  
    // fish->CreateWiki();
    fish->Init( );
    fish->BuildGraph( );
    fish->Train( );

    return 0x0;    
}