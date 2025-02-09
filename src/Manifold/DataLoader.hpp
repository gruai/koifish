/**
 *  Copyright 2023-2025 by Grusoft 
 *  
 *  Generate samples for train/Eval
 *      
 *  \brief samples from multiple dataset/tokenset/...
 *  \author Yingshi Chen
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "../ggex/GG_util.hpp"
struct ConsiceDict;
#include "../CLI_params.hpp"
#include "../g_stddef.hpp"
#include "TokenSet.hpp"
using namespace std;
// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define SHARD_HEADER_SIZE 256

class WIKI;
class Fish;
struct train_state;
class Optimizer;

class NLP_AutoRegressive;
class SampLoader;


class SampLoader;

class SampLoader   {
protected:  
    typedef std::string mt19937_state;
    //  Store tokens from source.  always in CPU
    shared_ptr<GTensor> hostBatch=nullptr,hostBatchMask=nullptr,hostTargetProbs=nullptr;     
    int tokens_per_iter = 0;
    // CLI_params hparams;
    std::string fp_data;
    std::string sentence="";
    std::vector<TOKEN_ID> samp_toks;
    bool sample_separation_eos,sample_separation_bos;
    // size_t _nctx=-1;
    std::vector<hSAMP> shard_samps;
    std::vector<hSAMP> cur_samps;
    // std::vector<size_t> idcs;      //would change epoch by epoch(shuffle,subsampling...)
    int64_t nShard() {
        return shard_samps.size();
    }   
    std::shared_ptr<ConsiceDict> hDict;
    hDataToken hTokens;

    bool isTarget_1 = false;
    bool isRecycle = true;
    bool isFixEvalSample = true;        // Need fix this to do some experiments
    mt19937_state shuffle_rng_state_current;
    mt19937_state shuffle_rng_state_next;
    size_t shuffle_sample_count=0,next_sample=0,shuffle_samples_hash = 0x0;

    NLP_AutoRegressive *dolphin=nullptr;
public:
    std::string batch_sample;       //  "stacking"
    std::string name;
    int num_batches=-1;    //number of batchs in each epoch

    int64_t len() {
        return shard_samps.size();
    }
    bool empty()    {   return len()==0;    }
    size_t nTokens()    {
        return hTokens->tokens.size();
    }
    int nLeastCTX(int flag=0x0);
    hSAMP SampAt(size_t idx_){
        assert(idx_<nShard());
        return shard_samps[idx_];
    }
    virtual hSAMP Next(bool isLoop = true);
    virtual bool isNextEpoch(int flag=0x0);
    virtual string IterInfo(int flag=0x0);
    vector<TOKEN_ID>& GetTokens()    {  return hTokens->tokens; }

    int32_t TokenAt(size_t pos){
        return hTokens->At(pos);
    }
    bool MaskAt(size_t pos,TOKEN_ID&mask);
    std::vector<std::string> curDeTexts;
    virtual hSAMP InitOneSamp(const string &prompt,hGensor input,int flag=0x0);
    virtual double DecodeVerify(hSAMP samp, hGensor tokens,hGensor logits,int flag=0x0);
    void Samp2Batch(int k,hSAMP samp,struct train_params_& params,int flag=0x0);

    enum TYPE{
        DT_TRAIN=1,DT_EVAL,DT_PREDICT,
        DT_MERGE
    };
    TYPE type = DT_TRAIN;
    
    Optimizer *hOPT = nullptr;

    SampLoader()    {}
    SampLoader(Fish *g_,const string&n,bool isNewTS,int flag=0x0);
    // virtual bool Init(Fish *g_,const string&n,bool isNewTS,int flag=0x0 ) ;
    virtual bool Prepare(hDataToken hT,int flag=0x0 ) ;    
    virtual size_t update_batch(int next_id,Fish* fish);
    virtual ~SampLoader( ) {
        
        
        if(!shard_samps.empty()){
            
        }
    }
#ifdef _DATA_LOADER_LITE_
#else
    virtual bool Serialize(const std::string&path, bool isSave, int flag=0x0);
    virtual void SetSamples(std::vector<size_t>& begin_,std::vector<size_t>& size_,
        bool isTrain,CLI_params& train_params,int flag=0x0);
    void Shuffle(int flag=0x0);
    bool TopoOrder(std::vector<size_t>&ids,std::mt19937& rng,int flag=0x0);
#endif
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class Fish;
    friend class GeneratOnPrompt;
    friend class OutCLS;
};
typedef shared_ptr<SampLoader> hSampLoader;

// class DataLoader_3D : public SampLoader  {
// protected:
// public:
//     int64_t update_batch(int next_id,Fish* fish)    override;
// };
#endif // DATALOADER_H