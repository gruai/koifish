/**
 *  Copyright 2023-2025 by Grusoft 
 *  
 *  Tokenset manager. Key component of speed & accuracy
 * 
 *  \brief Tokenset from files/hugging/...
 *  \author Yingshi Chen
 */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <memory>
#include <random>
#include <vector>
#include <glob.h>
#include "../CLI_params.hpp"
#include "../g_stddef.hpp"
#include "Serial.hpp"

struct ConsiceDict;
class DataTokenSet;
typedef std::shared_ptr<DataTokenSet> hDataToken;

struct SAMP{
    size_t pos=-1,len=-1;       //  range is [pos,pos+len)
    size_t off_cycle=0;         // more random
    int jump = 0;   
    TOKEN_ID last_target=-1;
    std::string desc;    

    SAMP()  {}
    SAMP(size_t p,size_t l) : pos(p),len(l) {

    }
    virtual ~SAMP() {}

    bool Serialize(FSerial&S, bool isSave, int flag);    
    // void Refresh(SampLoader *loader,void *ctx,std::vector<int32_t>& samp_toks,int typ);
    virtual double UpdateTag(hDataToken hDT,int *tag,int step,bool flip,int flag=0x0);

    static size_t HASH(const char* fn, const std::vector<SAMP*>& samps) {
        std::hash<std::string> h_string;
        std::hash<unsigned long long> h_ull;
        size_t h = h_string(std::string(fn)),sample_count=samps.size();
        h = HASH_combine(h, h_ull((unsigned long long) sample_count));
        for (auto samp : samps) {
            h = HASH_combine(h, h_ull((unsigned long long) samp->pos));
            h = HASH_combine(h, h_ull((unsigned long long) samp->len));
        }
        return h;
    }
}; 
// typedef std::shared_ptr<SAMP> hSAMP;
typedef SAMP* hSAMP;

class DataTokenSet    {
protected:
    string name;
    string serial_root;
    ConsiceDict *hDict = nullptr;
    std::map<TOKEN_ID, TOKEN_ID> mapT2T;
    std::vector<TOKEN_ID> dialect;
    std::string fpath;
    size_t fsize=0,nUnique=0,nVocab=0,nDialect=0;    
    std::vector<hSAMP> shard_samps;

    void seek(FILE *fp,size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else   
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        assert(ret == 0); // same
    }
    // int UniqueTokens(const std::vector<TOKEN_ID>& tokens,size_t n_1,int flag=0x0);
public:
    static std::vector<hDataToken> MakeInstance(struct CLI_params& params,ConsiceDict *hDict, int flag);

    std::vector<TOKEN_ID> tokens;
    DataTokenSet(ConsiceDict *hDict);

    TOKEN_ID At(size_t pos);

    bool Serialize(const std::string&path,  bool isSave, int flag=0x0);
    virtual bool NextShard(int flag=0x0)    {return true;}
    virtual bool Load(struct CLI_params& hparams,void *hLLM,int flag=0x0);
    virtual void Append(TOKEN_ID id,int flag=0x0);
    int UniqueTokens(size_t n_1,int flag=0x0);
    bool InitSamps(unsigned context_length,std::vector<size_t>& samples_begin,std::vector<size_t>&samples_size,int flag=0x0);

friend class NLP_AutoRegressive;
friend class Optimizer;
friend class SampLoader;
};
typedef std::vector<hDataToken> DataTokens;

class GlobTokenset : public DataTokenSet{
protected:
    std::vector<string> shard_paths;
    int shard_index=0;

    FILE* fpToken=nullptr;
    bool isShuffle = false;
    int64_t PrepareShard(int id,bool load=false, int flag=0x0);
    bool NextShard(int flag=0x0)    override;
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes,B=-1,T=-1;  // header size in bytes
    int64_t file_size_bytes,shard_num_samples=0;
public:
    GlobTokenset(JSON::const_iterator jit,ConsiceDict *hDict,int flag=0x0);
};

class DTS_GPT2 : public DataTokenSet    {
public:
    DTS_GPT2(ConsiceDict *hDict) : DataTokenSet(hDict)    {

    }
};