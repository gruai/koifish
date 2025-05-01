/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 *  
 *  \brief Evaluate metric(ppl)
 *  \author Yingshi Chen
 */
#include "Optimizer.hpp"
#include "GoPT.hpp"
#include "Fish.hpp"
#include "gLLM.hpp"
#include <string>
#include <iostream>
#include <filesystem>

float sample_prob(int idx, float* logits, int size);

int Fish_ppl(CLI_params& config)  {  
    g_dump_level = 0;  
    config.wiki_actor = "copy";     config.isOnlyGPT = true;
    config.common.remater_ffn = 1;
    config.common.n_batch = 1;
    config.model.preLogits_dB = 1;
    config.model.sparse.method = -1;
    // config.model.isEmbedWeightTying = false;    //QWEN32B

    DEBUG.graph_dump = 1;
    DEBUG.T_cuda_ver = 1;
    DEBUG.T_cpu = 2;
    GTensor::tpPreLogits = typNUMBER::F32;
    // return run_caml(config.prompt.c_str(),0x0);
    
    hFISH fish = Fish::MakeInstance("PPL_",config,{},Fish::ROLE_TYPE::COMMON,0x110);
    hOptimizer hOPT = fish->GetOptimizer();
    uint64_t rng_seed = 42;
    std::string prompt = LoadSomeText("/home/cys/rnd/lic/models/TinyStories-valid.txt",64*1024 );  //shakespeare.txt
    int nVocab = fish->config.model.vocab_size, _nctx = fish->config.n_ctx(), i, j;    
    hSampLoader hLoader = hOPT->val_loaders[0];
    if(hLoader->num_batches<=0 )    {
        hLoader->InitOneSamp(prompt,nullptr,fish.get(),0x110);                
    } 
    GST_util::tX = 0;
    int nTokens = hLoader->nMostToken;  
    nTokens = 200;  
    // assert(nTokens <= _nctx);
    TOKEN_ID t = -1;
    _INFO("\n====== %s: T_cpu=%d \n\t", __func__,DEBUG.T_cpu);
    OutCLS *hCLS = fish->GetNeuron<OutCLS>("OutCLS",0);
    float *logits = hCLS->Logits();
	double sum = 0, ss = 0, nz = 0, ppl = 0, pplerr = 0, tps = 0, t0 = GST_ms(),tAll = 0;
    vector<TOKEN_ID>& tokens = hLoader->GetTokens( );
    hOPT->SetPhase(Optimizer::P_GENERATE);
    for (i = 0; i+1 < nTokens; i++)    {  
        float fLos = hOPT->Evaluate(hLoader,-666);  
        double logprob = log(sample_prob(tokens[i + 1], logits, nVocab));

        sum += logprob;        ss += logprob * logprob;        nz ++;
        ppl = exp(-sum / nz);
        pplerr = ppl * sqrt((ss - sum * sum / nz) / nz / nz);
    }
    tAll = (GST_ms()-t0)/1000.0;            tps = nz/tAll;
    _INFO("[PPL] #ppl=%g±%.3f tps=%g(%d) T=%.3gs T_pick=%.3gs\n", ppl, pplerr, tps, nTokens,
        tAll,CS_Picker::tPick/1.0e6);
    Fish::stat.Dump(0x0);

    if(fish->config.model.isSparse()){
        // fish->Sparsing();
    }
    return 0x0;
}
