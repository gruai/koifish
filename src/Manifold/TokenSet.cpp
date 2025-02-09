/**
 *  Copyright 2023-2025 by Grusoft  
 * 
 *  \brief
 *  \author Yingshi Chen
 */

#include "TokenSet.hpp"
#include "DataLoader.hpp"
#include "gLLM.hpp"
#include "Optimizer.hpp"
#include "Dictionary.hpp"

bool GlobTokenset::NextShard(int flag)  {
    size_t iRet = OnShardFile(shard_index++,true);    
    if(iRet==size_t(-1))  
        return false;    
    
    return true;
}
    
bool GlobTokenset::Shard2Sample_hellaswag(int flag){
try{
    size_t B = GTensor::B,T = GTensor::T;
    int ASSUMED_NUM_COMPLETIONS=4,batch_dim_offset,nComplete=0;
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS),examples_per_process=nShardSamples,end_example_index=examples_per_process,start_example_index=0;
    uint16_t *buffer16 = new uint16_t[longest_example_bytes];
    assert (can_fit_examples > 0);
    tokens.resize(nShardSamples*ASSUMED_NUM_COMPLETIONS*T);
    masks.resize(tokens.size());
    int num_batches = CEIL_DIV(examples_per_process, can_fit_examples),id;
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    size_t header_bytes = SHARD_HEADER_SIZE * sizeof(int),sz=header_bytes;
    uint16_t example_header[3]; //<START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
    fseekCheck(fpShard, (int) header_bytes, SEEK_SET);
    for (id = start_example_index; id <end_example_index; id++) {        
        batch_dim_offset = id*ASSUMED_NUM_COMPLETIONS;
        freadCheck(&example_header[0], sizeof(uint16_t), 3, fpShard);
        assert(example_header[0] == 65535);         assert(example_header[2] == id); 
        sz += example_header[1];
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;        assert(remaining_bytes > 0); 
        //fseekCheck(fpShard, (int) remaining_bytes, SEEK_CUR);
        freadCheck(buffer16, sizeof(char), remaining_bytes, fpShard);
        hSAMP samp = new SAMP(0x0,0);        
        int l = (int)buffer16[0],nComplete = (int)buffer16[1],context_length = (int)buffer16[2];
        assert(l >= 0 && l < ASSUMED_NUM_COMPLETIONS); // we expect the label to be in [0, 4) for right now
        samp->label = l; 
        assert(nComplete == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
        // assert(batch_dim_offset + c <= B); // we expect to fit in the batch
        assert(context_length > 0 && context_length < T); // context is non-empty and up to T    
        uint16_t *context_tokens_start = (uint16_t *)(buffer16+3); // where the tokens start            
        for (int b = 0; b < nComplete; b++) {
            for (int i = 0; i < context_length; i++) {
                int boff = batch_dim_offset + b;
                int tok_cur = (int)context_tokens_start[i];
                tokens[boff * T + i] = tok_cur;
            }
        }
        // process the completions, insert them in their row, right after the (shared) context
        uint16_t *completions_iter = buffer16 + 3 + context_length;
        for (int c = 0; c < nComplete; c++) {
            int coff = batch_dim_offset + c;
            int completion_length = (int)completions_iter[0];
            uint16_t *completion_tokens_start = completions_iter + 1;
            assert(completion_length > 0 && context_length + completion_length < T); // things fit?
            for (int i = 0; i < completion_length; i++) {
                int tok_cur = (int)completion_tokens_start[i];                
                tokens[coff * T + context_length + i] = tok_cur;    // at inputs, the completions simply follow the context
                // at targets things start to get tricky
                // we expect the last context token to predict the first completion token
                // and then onwards from there.
                // targets[coff * T + context_length + i - 1] = tok_cur;
                // and at these positions, we want to set mask=1, because these are the
                // positions where we want to average the loss, in each row, to determine
                // its overall probability of following the context.
                masks[coff * T + context_length + i - 1] = 1;
            }
            completions_iter += 1 + completion_length; // move to the next completion
        }
        shard_samps.push_back(samp);
    }
    assert(sz==szFile);
    delete[] buffer16;
    return true;
}catch(...){
    return false;
}

}
bool GlobTokenset::Shard2Sample(int flag){
try{
    int szT=sizeof(uint16_t),nT = (szFile-header_bytes)/szT;
    assert(nT==nShardToks);
    tokens.resize(nT);
    fseekCheck(fpShard, (int) header_bytes, SEEK_SET);
    uint16_t *tmp16=new uint16_t[nT];
    if(fread(tokens.data(),szT,nT,fpShard)!=nT) {
        _INFO("Error: file size is not as expected\n");
        return 0x0;
    }else{
        nT = min(nT,1024);
        for(int i=0;i<nT;i++){
            assert(0<=tokens[i] && tokens[i]<nVocab);
        }
    }
    delete[] tmp16;
    // InitSamps
    int n_ctx = hDict->hparams.n_ctx();
    size_t nToken = tokens.size(),nFirst=std::min((size_t) n_ctx, nToken),step=1,n0=shard_samps.size();
    // samples_size.push_back(nFirst);
    size_t end = (nToken >= n_ctx) ? (nToken - n_ctx) : 0;
    if(end>10*1024*1024){
        step = n_ctx;
    }
    for (size_t sample_begin = 1; sample_begin < end; sample_begin+=step) {
        shard_samps.push_back(new SAMP(sample_begin,n_ctx));
    }
    
    // _INFO("\t%s %s: nSamp=%ld=>%ld nBach=%d\n", __func__,name.c_str(),n0,shard_samps.size(),nBatch()); 
    return true;
}catch(...){
    return false;
}
}


size_t GlobTokenset::OnShardFile(int id0, bool load, int flag) {
    int id = id0;
    id = id0%shard_paths.size();
    if (isShuffle) {
        
    }
    if(id>=shard_paths.size()){
        return -1;
    }
    // use the first glob match as the filename for now
    const char* filename = shard_paths[id].c_str();
    assert (fpShard == NULL);
    fpShard = fopenCheck(filename, "rb");
    // validate the header
    int header[SHARD_HEADER_SIZE];
    freadCheck(header, sizeof(int), SHARD_HEADER_SIZE, fpShard);
    if (header[0] != 20240520 && header[0] != 20240522) {
        printf("Bad magic in the data file\n---> HINT: Are you passing in a correct file?\n---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    fseekCheck(fpShard, 0, SEEK_END); // seek to end of file
    szFile = ftell(fpShard); // read the offset, i.e. file size
    nShardToks = 0;
    switch(header[0]){      //
    case 20240522:      //  hellaswag dataset
        tpSample = HellaSwag ;
        // int ASSUMED_NUM_COMPLETIONS =4,can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
        longest_example_bytes = header[3];
        nShardSamples = header[2];
        nShardToks = GTensor::B*GTensor::T;
        // label = (int*)mallocCheck(can_fit_examples * sizeof(int));
        break;
    default:    //  20240520
        nShardToks = header[2];         assert(nShardToks > 0); 
        // fseekCheck(fpShard, 0, SEEK_SET); // seek back to the beginning
        // we expect nTok0 in the file to be consistent with filesize, assert that is the case
        int64_t expected_file_size = SHARD_HEADER_SIZE * sizeof(int) + nShardToks * sizeof(uint16_t);
        if (szFile != expected_file_size) {
            printf("Error: file size is not as expected\n");
            exit(EXIT_FAILURE);
        }
        nShardSamples = (nShardToks * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes;// -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens

        break;
    }
    if(load){
        switch(tpSample){
        case HellaSwag:
            Shard2Sample_hellaswag(0x0);
            break;
        default:
            Shard2Sample(0x0);
            break;
        }
        _INFO("[shard-%d]@\"%s\": tokens=%.3g(M) nShardSamples=%ld(%ld) \n",id,filename,nShardToks/1.0e6,nShardSamples,shard_samps.size()); 
    }
    if(tpSample==RANDOM_GENERATE)    {        
        /*
        if(load){
            int szT=sizeof(uint16_t),nT = (file_size_bytes-header_bytes)/szT;
            assert(nT==nTok0);
            tokens.resize(nT);
            fseekCheck(fpShard, (int) header_bytes, SEEK_SET);
            uint16_t *tmp16=new uint16_t[nT];
            if(fread(tokens.data(),szT,nT,fpShard)!=nT) {
                _INFO("Error: file size is not as expected\n");
                return 0x0;
            }else{
                nT = min(nT,1024);
                for(int i=0;i<nT;i++){
                    assert(0<=tokens[i] && tokens[i]<nVocab);
                }
            }
            delete[] tmp16;
        }*/
    }    
    
    if(load){
        
    }
    if (fpShard != NULL) {
        fcloseCheck(fpShard);           fpShard=NULL;
    }
    return nShardToks;
}

DataTokenSet::DataTokenSet(ConsiceDict *hD) : hDict(hD)   {
    nVocab = hDict->n_vocab;
    assert(nVocab>0);
}

TOKEN_ID DataTokenSet::At(size_t pos){
    assert(pos<tokens.size());
    int32_t token = CLAMP(tokens[pos], 0, (nVocab - 1));
    return token;
}

bool DataTokenSet::Serialize(const std::string&path,  bool isSave, int flag){
try{
    FSerial S(path,isSave,flag);
    if(!S.isValid())
        return false;
    
    _INFO("%s %s@%s...",__func__,isSave?"save@":"load@",path.c_str());
    _CHECK( S.Serial(nVocab,isSave,flag) );
    _CHECK( S.Serial(nUnique,isSave,flag) );
    _CHECK( S.Serial(fsize,isSave,flag) );
    _CHECK( S.Serial(nDialect,isSave,flag) );
    _CHECK( S.Serial(tokens,isSave,flag) );
    if(nDialect>0){
        _CHECK( S.Serial(dialect,isSave,flag) );
        _CHECK( S.Serial(mapT2T,isSave,flag) );
    }
    if(isSave){

    }else{
        if(tokens.size()==0)    return false;
        for(auto token:tokens){
            if(token<0 || token>=nVocab){
                return false;
            }
        }
    }
    
    
    
    return true;
}catch(...){
    return false;
}
}

double DataTokenSet::LossOnResult(OutCLS *cls,int flag) {
    assert(cls!=nullptr);
    double mean_loss = 0;
    int *mask = nullptr,n=0,nzLoss=cls->nzLoss;
    float *loss = cls->hostLoss;
    if(hasMask()){
        // mask = TO<int>(hostBatchMask);
    }
    for (int i = 0; i < nzLoss; i++) {
        if(mask!=nullptr && mask[i]==0){
            continue;
        }
        mean_loss += loss[i];       n++;
    }   
    assert(n>0);
    mean_loss /= n;   
    return mean_loss; 
}