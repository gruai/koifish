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

#include "llmc_utils.h"
#include "llmc_rand.h"
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
#define HEADER_SIZE 256

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
    int num_batches;    //number of batchs in each epoch

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
    virtual hSAMP Next(bool isLoop = true){
        if(next_sample==nShard()){
            if(!hTokens->NextShard( ))
                return nullptr;      
            next_sample = 0;
        }
            
        int64_t idx_ = next_sample;
        assert(idx_<nShard());
        if(type == SampLoader::TYPE::DT_TRAIN) 
            next_sample ++;
        else{
            if(!isFixEvalSample)
                next_sample ++;
        }
        // next_sample++;
        return shard_samps[idx_];
    }
    vector<TOKEN_ID>& GetTokens()    {  return hTokens->tokens; }

    int32_t TokenAt(size_t pos){
        return hTokens->At(pos);
    }
    std::vector<std::string> curDeTexts;
    virtual hSAMP InitOneSamp(const string &prompt,struct ggml_tensor *input,int flag=0x0);
    virtual double DecodeVerify(hSAMP samp, struct ggml_tensor *tokens,struct ggml_tensor *logits,int flag=0x0);
    void Samp2Batch(int k,hSAMP samp,struct ggml_tensor *tokens_input,struct ggml_tensor *target_probs,struct train_params_& params,int flag=0x0);

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
};
typedef shared_ptr<SampLoader> hSampLoader;

class SampLoader_glob : public SampLoader {

public:    
    //compatible with train_opt_callback_data@LLAMA.cpp
    // struct train_opt_callback_data callback_data;
    // train_state *train=nullptr;
    //token source, in some case, only need lite vector
    
    // std::vector<TOKEN_ID> stokens;

    
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // batch and token information
    size_t B=-1;
    size_t T=-1;
    size_t num_tokens=0; // total number of tokens, may different with tokens.size()!
    size_t shard_num_samples=0;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob
    size_t current_shard_idx,current_sample_idx; 
    // file handle
    FILE* fpToken=nullptr;
    // data buffers
    uint16_t* buffer=nullptr;  
    int* inputs=nullptr,*targets=nullptr; 
    // random shuffle related variables
    mt19937_torch shuffle_rng;
    int should_shuffle;
    int* shard_indices=nullptr;
    int* intra_shard_indices=nullptr;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;

    SampLoader_glob()  {}
    virtual ~SampLoader_glob( ) {
        free(buffer);
        free(inputs);
        free(targets);
        if (should_shuffle) {
            free(shard_indices);
            free(intra_shard_indices);
        }
        if(fpToken!=nullptr)    {
            fcloseCheck(fpToken);
            globfree(&glob_result);
        }
    }

    

    void prepare_intra_shard_indices_( ) {
        // shuffle the examples inside the shards
        if (intra_shard_indices != NULL) {
            // in case shards have different number of samples / sizes
            free(intra_shard_indices);
        }
        intra_shard_indices = (int*)mallocCheck(shard_num_samples * sizeof(int));
        init_identity_permutation(intra_shard_indices, (int) shard_num_samples);
        random_permutation(intra_shard_indices, (int) shard_num_samples, &shuffle_rng);
    }
    int64_t PrepareShard(int id)    {   return 0x0; }
    void reset( ) {
        current_shard_idx = 0;
        current_sample_idx = 0;
        if (should_shuffle) {  // shuffle the shards
            random_permutation(shard_indices, (int) glob_result.gl_pathc, &shuffle_rng);
        }
        PrepareShard( (int) current_shard_idx);
        if (should_shuffle) {
            prepare_intra_shard_indices_( );
        }
    }

    void advance_( ) {
        if (current_shard_idx == glob_result.gl_pathc - 1) {
            // if we are at the last shard, we reset the loader and start a new epoch
            reset( );
            return;
        }

        // advance the loader by loading the next data shard and resetting the position
        current_shard_idx = (current_shard_idx + 1) % glob_result.gl_pathc;
        current_sample_idx = 0;
        PrepareShard( (int) current_shard_idx);

        if (should_shuffle) {
            prepare_intra_shard_indices_( );
        }
    }

    virtual void Init( const char* filename_pattern,size_t B_,size_t T_,int process_rank_,int num_processes_,int should_shuffle_);

    void load_batch( ) {
        assert(!should_shuffle || (should_shuffle && intra_shard_indices != NULL));
        assert(current_sample_idx < shard_num_samples);
        size_t idx = should_shuffle ? intra_shard_indices[current_sample_idx] : current_sample_idx;
        size_t global_batch_offset_bytes = idx * total_batch_size_bytes;
        int64_t current_offset = header_bytes + global_batch_offset_bytes + local_batch_offset_bytes;
        // read B*T+1 uint16_t tokens from the file into buffer
        fseekCheck(fpToken, (int) current_offset, SEEK_SET);
        freadCheck(buffer, sizeof(uint16_t), B*T+1, fpToken);
        // decode the buffer into inputs and targets (cast to int)
        for (int i = 0; i < B*T; i++) {
            inputs[i] = (int)buffer[i];
            targets[i] = (int)buffer[i+1];
        }
    }

    

    void next_batch( )  {
        // if the next batch would go past the end of the file, advance the loader
        if (current_sample_idx >= shard_num_samples) {
            advance_( );
        }
        load_batch( );
        current_sample_idx += 1;
    }

    void resume(size_t current_shard_idx_, size_t current_sample_idx_) {
        // used during model resumption (-y 1) flag
        current_shard_idx = current_shard_idx_;
        current_sample_idx = current_sample_idx_;
        // PrepareShard( (int) current_shard_idx);
    }


};

// class DataLoader_3D : public SampLoader  {
// protected:
// public:
//     int64_t update_batch(int next_id,Fish* fish)    override;
// };
#endif // DATALOADER_H