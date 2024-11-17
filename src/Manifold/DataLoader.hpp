/*
Implements:
- SampLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <memory>
#include <vector>
#include "llmc_utils.h"
#include "llmc_rand.h"
// #include "Dictionary.hpp"
struct ConsiceDict;
#include "../CLI_params.hpp"
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

class FSerial{
    FILE *_stream=NULL;
    bool _valid = false;
public:
    FSerial(const std::string& sPath,bool isSave,int flag){
    try{
        if(isSave){
            if((_stream=fopen(sPath.c_str(),"wb"))!=NULL){
                _valid = true;
            }
        }else{
            if((_stream=fopen(sPath.c_str(),"rb"))!=NULL){
                _valid = true;
            }
        }
        
    }catch(...){
        
        _valid = false;
    }
    }

    virtual ~FSerial(){
    try{
        // if(_stream!=NULL)    
        //     fclose(_stream);
    }   catch(...){

    }  
    }

    virtual bool isValid()  {   return _valid; }

    template<typename T>
    bool Serial(T *val,int nz,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        if(isSave){
            if(fwrite(val,sizeof(T),nz,_stream)!=nz)
                return false;
            fflush(_stream);
        }else{
            if(fread(val,sizeof(T),nz,_stream)!=nz)
                return false;
        }
        return true;
    }

    template<typename T>
    bool Serial(T &val,bool isSave,int flag=0x0){        
        return Serial(&val,1,isSave,flag);
    }
        
    template<typename T>
    bool Serial(std::vector<T>& arrT,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        size_t nT=arrT.size(),i;
        Serial(&nT,1,isSave);
        if(nT==0){
            return true;
        }
        if(isSave){
            if(fwrite(arrT.data(),sizeof(T),nT,_stream)!=nT)
                return false;
            fflush(_stream);
        }else{
            arrT.clear();       arrT.resize(nT);
            if(fread(arrT.data(),sizeof(T),nT,_stream)!=nT)
                return false;
            // T* buf = new T[nT];
            // size_t nRead = fread((void*)(buf),sizeof(T),nT,_stream);
            // if(nRead!=nT)
            //     return false;    
            // std::copy(buf, buf + nT, std::back_inserter(arrT));
            // delete[] buf;
        }
        return true;
    }

    template<typename T,typename Tchild>
    bool Serial_Vector(std::vector<T*>& arrT,bool isSave,int flag=0x0){
        if(!isValid())  return false;
        size_t nT=arrT.size(),i;
        Serial(&nT,1,isSave);
        if(isSave){
            for(auto obj0:arrT){
                Tchild *obj = dynamic_cast<Tchild*>(obj0);
                assert(obj!=nullptr);
                if(!obj->Serialize(*this,isSave,flag))
                    return false;
            }
        }else{
            arrT.clear();       arrT.resize(nT);
            for(i=0;i<nT;i++){
                Tchild *obj = new Tchild();
                if(!obj->Serialize(*this,isSave,flag))
                    return false;
                arrT[i] = obj;
            }
        }
        return true;
    }
    
};

class NLP_AutoRegressive;
class SampLoader;
class DataTokenSet    {
protected:
    ConsiceDict *hDict = nullptr;
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
    std::string fpath;
    size_t fsize=0,nUnique=0,nVocab=0,nDialect=0,nTokens;
    

    void seek(FILE *fp,size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else   
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }
    // int UniqueTokens(const std::vector<TOKEN_ID>& tokens,size_t n_1,int flag=0x0);
public:
    std::vector<TOKEN_ID> tokens;
    DataTokenSet(ConsiceDict *hDict);

    TOKEN_ID At(size_t pos){
        assert(pos<nTokens);
        int32_t token = clamp(tokens[pos], 0, (nVocab - 1));
        return token;
    }

    bool Serialize(const std::string&path,  bool isSave, int flag=0x0);
    
    virtual bool Load(struct CLI_params& hparams,void *hLLM,int flag=0x0);
    int UniqueTokens(size_t n_1,int flag=0x0);
    bool InitSamps(unsigned context_length,std::vector<size_t>& samples_begin,std::vector<size_t>&samples_size,int flag=0x0);

friend class NLP_AutoRegressive;
friend class Optimizer;
friend class SampLoader;
};
typedef std::shared_ptr<DataTokenSet> hDataToken;

class DTS_GPT2 : public DataTokenSet    {
public:
    DTS_GPT2(ConsiceDict *hDict) : DataTokenSet(hDict)    {

    }
    // int stream2token(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag=0x0)    override;
};

class SampLoader;
struct SAMP{
    size_t pos=-1,len=-1;       //  range is [pos,pos+len)
    size_t off_cycle=0;         // more random
    int jump = -1;   
    std::string desc;
    

    SAMP()  {}
    SAMP(size_t p,size_t l) : pos(p),len(l) {

    }
    virtual ~SAMP() {}

    bool Serialize(FSerial&S, bool isSave, int flag);

    void Refresh(SampLoader *loader,void *ctx,std::vector<int32_t>& tok_ids,int typ);
    virtual double UpdateTag(hDataToken hDT,int *tag,int step,bool flip,int flag=0x0);

    static size_t HASH(const char* fn, const std::vector<SAMP*>& samps) {
        std::hash<std::string> h_string;
        std::hash<unsigned long long> h_ull;
        size_t h = h_string(std::string(fn)),sample_count=samps.size();
        h = hash_combine(h, h_ull((unsigned long long) sample_count));
        for (auto samp : samps) {
            h = hash_combine(h, h_ull((unsigned long long) samp->pos));
            h = hash_combine(h, h_ull((unsigned long long) samp->len));
        }
        return h;
    }
}; 
// typedef std::shared_ptr<SAMP> hSAMP;
typedef SAMP* hSAMP;


class SampLoader   {
protected:  
    int tokens_per_iter = 0;
    CLI_params hparams;
    std::string fp_data;
    std::string sentence="";
    std::vector<int32_t> tok_ids;
    bool sample_separation_eos,sample_separation_bos;
    size_t n_ctx=-1;
    std::vector<hSAMP> all_samps;
    // std::vector<size_t> idcs;      //would change epoch by epoch(shuffle,subsampling...)
    int64_t N4Train() {
        return all_samps.size();
        // return idcs.size();
    }   
    std::shared_ptr<ConsiceDict> hDict=nullptr;
    bool isTarget_1 = false;
    mt19937_state shuffle_rng_state_current;
    mt19937_state shuffle_rng_state_next;
    size_t shuffle_sample_count=0,next_sample=0;

    NLP_AutoRegressive *lama=nullptr;
public:
    int64_t len() {
        return all_samps.size();
    }
    size_t nTokens()    {
        return tokens->nTokens;
    }
    hSAMP SampAt(size_t idx_){
        assert(idx_<N4Train());
        // size_t id = idcs[idx_];
        // assert(id<len());
        return all_samps[idx_];
    }

    int32_t TokenAt(size_t pos){
        return tokens->At(pos);
        /*size_t nToken = tokens.size();
        assert(pos<nToken);
        int32_t token = clamp(tokens[pos], 0, (n_vocab - 1));
        return token;*/
    }
    void Samp2Batch(int k,hSAMP samp,struct ggml_tensor *tokens_input,struct ggml_tensor *target_probs,struct train_params_common& params,int flag=0x0);

    enum TYPE{
        DT_TRAIN=1,DT_EVAL,DT_PREDICT,
        DT_MERGE
    };
    TYPE type = DT_TRAIN;

    Optimizer *hOPT = nullptr;
    std::string batch_sample;
    //compatible with train_opt_callback_data@LLAMA.cpp
    struct train_opt_callback_data callback_data;   
    size_t n_vocab = 0 ;
    int num_batches;    //number of batchs in each epoch

    train_state *train=nullptr;
    //token source
    hDataToken tokens = nullptr;      
    // std::vector<TOKEN_ID> tokens;
    // size_t n_unique_tokens=0;

    size_t shuffle_samples_hash = 0x0;

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
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx,current_sample_idx; 
    // file handle
    FILE* tokens_file=nullptr;
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

    SampLoader( )   {}
    virtual void Init(NLP_AutoRegressive *g_,int flag=0x0 ) ;
    virtual void Prepare(Optimizer *hOPT_,int flag=0x0 ) ;    

    virtual ~SampLoader( ) {
        free(buffer);
        free(inputs);
        free(targets);
        if (should_shuffle) {
            free(shard_indices);
            free(intra_shard_indices);
        }
        if(tokens_file!=nullptr)    {
            fcloseCheck(tokens_file);
            globfree(&glob_result);
        }
        if(!all_samps.empty()){
            
        }
    }


    int64_t load_shard_(int shard_index) {
        if (should_shuffle) {
            shard_index = shard_indices[shard_index];
        }
        // use the first glob match as the filename for now
        const char* filename = glob_result.gl_pathv[shard_index];
        // open the input file for reading. also only a single file can be opened at a time
        if (tokens_file != NULL) {
            fcloseCheck(tokens_file);
        }
        tokens_file = fopenCheck(filename, "rb");
        // validate the header
        int header[HEADER_SIZE];
        freadCheck(header, sizeof(int), HEADER_SIZE, tokens_file);
        if (header[0] != 20240520) {
            printf("Bad magic in the data file\n");
            printf("---> HINT: Are you passing in a correct file?\n");
            printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
            exit(EXIT_FAILURE);
        }
        if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
        int64_t ntok = header[2]; // number of tokens in the file
        assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
        // determine the file size and make sure it is consistent with the number of tokens
        fseekCheck(tokens_file, 0, SEEK_END); // seek to end of file
        file_size_bytes = ftell(tokens_file); // read the offset, i.e. file size
        fseekCheck(tokens_file, 0, SEEK_SET); // seek back to the beginning
        // we expect ntok in the file to be consistent with filesize, assert that is the case
        int64_t expected_file_size = HEADER_SIZE * sizeof(int) + ntok * sizeof(uint16_t);
        if (file_size_bytes != expected_file_size) {
            printf("Error: file size is not as expected\n");
            exit(EXIT_FAILURE);
        }
        // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
        shard_num_samples = (ntok * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes;
        return ntok;
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

    void reset( ) {
        current_shard_idx = 0;
        current_sample_idx = 0;
        if (should_shuffle) {  // shuffle the shards
            random_permutation(shard_indices, (int) glob_result.gl_pathc, &shuffle_rng);
        }
        load_shard_( (int) current_shard_idx);
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
        load_shard_( (int) current_shard_idx);

        if (should_shuffle) {
            prepare_intra_shard_indices_( );
        }
    }

    void init( const char* filename_pattern,size_t B_,size_t T_,
                int process_rank_,int num_processes_,int should_shuffle_) {
        process_rank = process_rank_;
        num_processes = num_processes_;
        B = B_;         T = T_;
        assert(B>0 && T>0);
        tokens_file = NULL;
        should_shuffle = should_shuffle_;
        header_bytes = HEADER_SIZE * sizeof(int);
        total_batch_size_bytes = ((num_processes * (B * T)) * sizeof(uint16_t));
        local_batch_offset_bytes = process_rank * B * T * sizeof(uint16_t);

        // glob to get the list of files matching the pattern, these are our data shards
        int glob_status = glob(filename_pattern, 0, NULL, &glob_result);
        if (glob_status != 0) {
            printf("Error: failed to glob pattern: %s\n", filename_pattern);
            exit(EXIT_FAILURE);
        }
        if (glob_result.gl_pathc == 0) {
            printf("Error: no files found matching the pattern: %s\n", filename_pattern);
            exit(EXIT_FAILURE);
        }

        if (should_shuffle) {
            manual_seed(&shuffle_rng, 42 + process_rank);
            shard_indices = (int*)mallocCheck(glob_result.gl_pathc * sizeof(int));
            init_identity_permutation(shard_indices, (int) glob_result.gl_pathc);
            intra_shard_indices = NULL;  // dynamically allocated allowing different shard sizes
        }

        // inspect and validate all shards so we don't get any runtime errors later
        // if too slow / too many shards, may wish to revisit later
        int64_t ntok_total = 0;
        for (int shard_index = 0; shard_index < glob_result.gl_pathc; shard_index++) {
            int64_t shard_ntok = load_shard_( shard_index);
            // we need at least one batch/shard, the way things are written right now.
            // can be relaxed a lot later.
            assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
            ntok_total += shard_ntok;
        }
        // debugging prints
        // printf("SampLoader: filename_pattern: %s\n", filename_pattern);
        // printf("SampLoader: Found %ld tokens across %zu shards\n", ntok_total, glob_result.gl_pathc);

        // allocate all the space we'll need
        buffer = (uint16_t*)mallocCheck((B * T + 1) * sizeof(uint16_t));
        inputs = (int*)mallocCheck(B * T * sizeof(int));
        targets = (int*)mallocCheck(B * T * sizeof(int));
        num_tokens = ntok_total;

        // reset the  to initialize it
        reset( );
    }

    void load_batch( ) {
        assert(!should_shuffle || (should_shuffle && intra_shard_indices != NULL));
        assert(current_sample_idx < shard_num_samples);
        size_t idx = should_shuffle ? intra_shard_indices[current_sample_idx] : current_sample_idx;
        size_t global_batch_offset_bytes = idx * total_batch_size_bytes;
        int64_t current_offset = header_bytes + global_batch_offset_bytes + local_batch_offset_bytes;
        // read B*T+1 uint16_t tokens from the file into buffer
        fseekCheck(tokens_file, (int) current_offset, SEEK_SET);
        freadCheck(buffer, sizeof(uint16_t), B*T+1, tokens_file);
        // decode the buffer into inputs and targets (cast to int)
        for (int i = 0; i < B*T; i++) {
            inputs[i] = (int)buffer[i];
            targets[i] = (int)buffer[i+1];
        }
    }

    int64_t update_batch(int next_id,Fish* fish);

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
        load_shard_( (int) current_shard_idx);
    }

#ifdef _DATA_LOADER_LITE_
#else
    virtual bool Serialize(const std::string&path, bool isSave, int flag=0x0);
    virtual void SetSamples(int nV,hDataToken hDT,std::vector<size_t>& begin_,std::vector<size_t>& size_,
        bool isTrain,CLI_params& train_params,int flag=0x0);
    void Shuffle(int flag=0x0);
    bool TopoOrder(std::vector<size_t>&ids,std::mt19937& rng,int flag=0x0);
#endif
    friend class NLP_AutoRegressive;
    friend class Optimizer;
    friend class Fish;
};

// class DataLoader_3D : public SampLoader  {
// protected:
// public:
//     int64_t update_batch(int next_id,Fish* fish)    override;
// };
#endif // DATALOADER_H