#include "DataLoader.hpp"
#ifdef _DATA_LOADER_LITE_
#else
#include "Fish.hpp"
#include "../ggex/GG_util.hpp"
#include "../LLAMA/common/common.h"


// int64_t DataLoader_3D::update_batch(int sample_id,Fish* fish)   {    
//     int64_t used_samples = 0;
//     return used_samples;
// }

void SAMP::Refresh(DataLoader *loader,void *ctx,std::vector<int32_t>& tok_ids,int typ)  {
    auto target_probs = loader->hOPT->hTargetProbs();
    int64_t n_vocab=target_probs->ne[0],n_tokens=target_probs->ne[1],nSampInBatch=target_probs->ne[2];
    std::string sentence;
    struct llama_context * lctx = static_cast<llama_context *>(ctx);
    if(typ==-1)     //batch_sample=="stacking"
        _INFO(" (%ld,%d)@\"%s...\"",pos,jump,sentence.c_str());     //sample_size
    else{
        tok_ids.clear();
        int nz=len+nSampInBatch,i;
        for(i=0;i<nz;i++){
            llama_token tok = loader->TokenAt(pos+i);
            tok_ids.push_back(tok);
            string word = llama_token_to_piece(lctx, tok);
            if(word=="\n" || word=="\r\n")  {
                word=" ";
            }
                
            sentence += word;
            if(i>64){
                
            }
        }   
        // _INFO("\t%ld@\"%.*s...\"\n",pos,64,sentence.c_str());     //sample_size
        desc = sentence;
    }
    
}

void DataLoader::Samp2Batch(int k,hSAMP samp,hGensor tokens_input,hGensor target_probs,struct train_params_common& params,int flag)    {   
    tok_ids.clear();        
    
    bool fill_with_next_samples=params.fill_with_next_samples;
    size_t starting=samp->pos+samp->jump;    
    ggml_set_i32_nd(tokens_input, 0, k, 0, 0, bos);
    for (int64_t i=0; i<n_ctx; ++i) {    
        llama_token token = eos;
        /*if (samp_off >= samp->len && fill_with_next_samples) { //true only arg == "--fill-with-next-samples"
            if (!sample_separation_eos) {
                // insert eos token to separate samples
                sample_separation_eos = true;
            } else if (!sample_separation_bos) {
                // insert bos token to separate samples
                sample_separation_bos = true;
                token = bos;
            } else {
                samp_off  = 0;
                size_t sample_idx   = (sample_id + used_samples) % samples_count;
                samp = SampAt(sample_idx);
                // sample_begin = shuffled_samples_begin[sample_idx];
                // sample_size  = shuffled_samples_size[sample_idx];
                ++used_samples;
            }
        }*/
        // note: no else-if here
        if (starting >= tokens.size()) {
            starting = 0;
        }
        assert(starting<tokens.size());
        token = clamp(tokens[starting], 0, (llama_token) (n_vocab - 1));
        ++starting;        
        ggml_set_f32_nd(target_probs,  token, (int) i, (int) k, 0, +1.0f);
        tok_ids.push_back(token);
        
        if (i+1<n_ctx) {
            ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) k, 0, 0, token);
            // sentence = llama_token_to_piece(lctx, token);
            // _INFO("%s,",sentence.c_str());
        }
    }
}

//Important!  this would update input & target_probs of model!
int64_t DataLoader::update_batch(int sample_id,Fish* fish){    
    struct train_params_common *params = &(hOPT->train_params);
    struct llama_context * lctx=(struct llama_context *)(hOPT->app_ctx);
    struct ggml_tensor   * tokens_input=hOPT->tokens_input;
    struct ggml_tensor   * target_probs=hOPT->hTargetProbs();
    int64_t samples_count = N4Train();
    // const llama_token    * train_data=tokens.data();
    size_t  k,n_train_data=tokens.size();
    sample_separation_eos=!params->separate_with_eos;
    sample_separation_bos=!params->separate_with_bos;
    double t_Samp = 0,nrm=0,a;
    bool                   sample_random_offsets=params->sample_random_offsets;
    // GGML_ASSERT(samples_count > 0);
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(target_probs));
    int64_t n_vocab  = target_probs->ne[0],nSampInBatch = tokens_input->ne[1],ld0,ld1,ld2,ld3;
    n_ctx = tokens_input->ne[0];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_ctx == target_probs->ne[1]);
    GGML_ASSERT(nSampInBatch  == target_probs->ne[2]);
    GST_TIC(T0);
    int used_samples = 0;
    ggml_set_f32(target_probs, 0.0f);
    bos = llama_token_bos(llama_get_model(lctx));
    eos = llama_token_eos(llama_get_model(lctx));
    std::string sBos = llama_token_to_piece(lctx, bos),sEos = llama_token_to_piece(lctx, eos);
    bool isLog = false;
    if(isLog) _INFO("BATCH_%ld ",sample_id);   //, samples_count,n_train_data);nSampe=(%ld/%ld)
    assert(target_probs->type == GGML_TYPE_F32);
    hSAMP samp = nullptr;
    hGensor exLogits=nullptr;
    if(fish->wiki!=nullptr && fish->exLogits!=nullptr) {
        exLogits = fish->exLogits;   // wiki->logits = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_ctx, n_batch);
        ld0=exLogits->nb[0],ld1=exLogits->nb[1],ld2=exLogits->nb[2],ld3=exLogits->nb[3];     
        assert(sizeof(float)*n_vocab*n_ctx==ld2);    
        assert(ld3==ld2*nSampInBatch);    
    }
    GST_TIC(tic);
    for (k=0; k<nSampInBatch; ++k) {
        size_t sample_idx = (sample_id + used_samples) % samples_count,samp_off=0;
        samp = SampAt(sample_idx);
        // samp->Refresh(this,lctx,0x0);
        GGML_ASSERT(samp->pos+samp->len-1 < n_train_data);          
        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);   
        if(batch_sample=="stacking"){
            samp->jump++;
        }else{            
            ++used_samples;   
        }
       
        Samp2Batch(k,samp,tokens_input,target_probs,*params);   //refresh tok_ids
        if(isLog && k<6 && batch_sample!="stacking"){
            sentence=llama_token_to_piece(lctx, tok_ids[0]);
            _INFO(" (%ld,%d)@\"%s...\"",samp->pos,samp->jump,sentence.c_str());     //sample_size
        }

        if(exLogits!=nullptr && batch_sample!="stacking"){ 
            nrm = fish->wiki->InductLogits(k,tok_ids,exLogits,target_probs,-1);   
            /*fish->wiki->Reset();         //Timing bottleneck!!! for the crazy design of llama.cpp
            fish->wiki->Decode(tok_ids,0,0x0,true); 
            const float *logits = fish->wiki->GetLogits(n_vocab,n_ctx,0);   
            double nrm = NRM_2(logits,n_vocab*n_ctx);
            if(nrm==0.0)    {
                _INFO("\n\n !!! %s |preLogits| is zero,so crazy! N=%dx%d \n\n",__func__,n_vocab,n_ctx);
            }  else if(k%10==0) {
                _INFO("\r %ld\t%ld@\"%s...\" nrm=%g\tT=%.4gs(%.4g)\t",k+1,samp->pos,sentence.c_str(),nrm,GST_TOC(tic),t_Samp);     //sample_size
            }     
            memcpy(exLogits->data+k*ld2,(void*)(logits),ld2);   */ 
        }        
    }
    t_Samp = GST_TOC(tic);
    if(isLog) _INFO("\tT=%g\n",GST_TOC(T0));
    if(batch_sample=="stacking"){
        if(fish->wiki->isInduct()){
            GST_TIC(tic);  
            samp->Refresh(this,lctx,tok_ids,0x0);          //refresh tok_ids               
            nrm = fish->wiki->InductLogits(nSampInBatch,tok_ids,exLogits,target_probs,0x0);           
            _INFO("\t stacking@%d\"%.*s...\" nrm=%g\tT=%.4gs\t\n",samp->pos,64,samp->desc.c_str(),nrm,GST_TOC(tic));             
        }
        used_samples = nSampInBatch;        //1    
    }
    return used_samples;
}

#define _CHECK(err)                                               \
    do {                                                            \
        bool err_ = (err);                                        \
        if (err_ != true) {                                   \
            fprintf(stderr, "!!! %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            throw("");                                                \
        }                                                           \
    } while (0)

template<>
bool FSerial::Serial(std::string &val,bool isSave,int flag){    
    if(!isValid())  
        return false;
    size_t nT=val.size(),i;
    Serial(&nT,1,isSave);
    if(nT==0){
        return true;
    }
    if(isSave){
        if(fwrite(val.c_str(),sizeof(char),nT,_stream)!=nT)
            return false;
    }else{
        char* buf = new char[nT];
        size_t nRead = fread((void*)(buf),sizeof(char),nT,_stream);
        if(nRead!=nT)
            return false;
        val = buf;      
        delete[] buf;
    }
    return true;
}

bool SAMP::Serialize(FSerial&S, bool isSave, int flag){
    if(!S.isValid())
        return false;

    _CHECK( S.Serial(pos,isSave,flag) );
    _CHECK( S.Serial(len,isSave,flag) );
    _CHECK( S.Serial(off_cycle,isSave,flag) );
    return true;
}


bool DataLoader::Serialize(const std::string&path, bool isSave, int flag){
try{
    FSerial S(path,isSave,flag);
    if(!S.isValid())
        return false;
    
    _INFO("%s %s@%s...",__func__,isSave?"save":"load",path.c_str());
    _CHECK( S.Serial(n_vocab,isSave,flag) );
    _CHECK( S.Serial(tokens,isSave,flag) );
    _CHECK( S.Serial(n_unique_tokens,isSave,flag) );
    _CHECK( S.Serial(shuffle_samples_hash,isSave,flag) );
    _CHECK( S.Serial(hOPT->train_params.seed,isSave,flag) );

    // _CHECK( S.Serial(samp_begin,isSave,flag) );
    // _CHECK( S.Serial(samp_size,isSave,flag) );
    // _CHECK( S.Serial(shuffled_samples_offs,isSave,flag) );
    // _CHECK( S.Serial(shuffled_samples_begin,isSave,flag) );
    // _CHECK( S.Serial(shuffled_samples_size,isSave,flag) );
    _CHECK( S.Serial(idcs,isSave,flag) );
    bool bRet = S.Serial_Vector<SAMP,SAMP>(all_samps,isSave,flag);
    _CHECK(bRet);
    _INFO("\r%s %s@\"%s\" ... OK. nSamp=%ld @[Datasets(nToken=%ld unique=%lld  hash=%llX]\n",__func__,isSave?"save":"load",path.c_str(),
        all_samps.size(),tokens.size(),n_unique_tokens,shuffle_samples_hash);
    if(isSave){
        
    }else{
        size_t nSample = all_samps.size();
        num_batches = nSample/hOPT->train_params.n_batch;
        num_batches = nSample==0 ? 0 : max(num_batches,1); 
        _INFO("\t nBatch in each epoch=%d\n", num_batches); 
    }
    if(type==TYPE::DT_TRAIN){
        assert(train!=nullptr);   
        if(isSave)    {

        }else{
            train->shuffle_rng_state_current = mt19937_seed_to_state(hOPT->train_params.seed);
            train->shuffle_sample_count = all_samps.size();
            train->shuffle_next_sample = 0;
            train->shuffle_samples_hash = shuffle_samples_hash; 
        }
        _INFO("\t%s@[%s]: hash=%lld\n", "train_state",path.c_str(), train->shuffle_samples_hash);  
    }
    return true;
}catch(...){
    return false;
}
}

/*
    @@@tokenize_file:
        out_samples_begin.push_back(0);
        out_samples_size.push_back(std::min((size_t) context_length, out_tokens.size()));
        size_t end = (out_tokens.size() >= context_length) ? (out_tokens.size() - context_length) : 0;
        for (size_t sample_begin = 1; sample_begin < end; ++sample_begin) {
            out_samples_begin.push_back(sample_begin);
            out_samples_size.push_back(context_length);
        }
*/
void DataLoader::SetSamples(int nV,std::vector<llama_token>& tokens_,std::vector<size_t>& samp_0,std::vector<size_t>& samp_L,
    bool isTrain,CLI_params& hparams,int flag)  {
    batch_sample = hparams.batch_sample;

    double rSplit = 1.0-hparams.rSplit;
    n_vocab = nV;    //hparams.n_vocab;
    tokens = tokens_;
    size_t nSample = samp_0.size(),pick=(size_t)(nSample*rSplit),i;
    //    assert(samp_begin.size() == samp_size.size());   
    if(isTrain){
        for(i=0;i<pick;i++){
            all_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }
        // samp_begin = std::vector<size_t>(samp_0.begin(),samp_0.begin()+pick);          
        // samp_size = std::vector<size_t>(samp_L.begin(),samp_L.begin()+pick);          
        Shuffle( );
    } else if(pick<nSample) {
        for(i=pick;i<nSample;i++){
            all_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }
        // samp_begin = std::vector<size_t>(samp_0.begin()+pick,samp_0.end());          
        // samp_size = std::vector<size_t>(samp_L.begin()+pick,samp_L.end()); 
        if(1){  //too many batch in eval-set, so just random pick
            Shuffle( );
        }else{
            assert(0);
            // shuffled_samples_begin = samp_begin;
            // shuffled_samples_size = samp_size;            
        }
    }
    nSample = all_samps.size();
    num_batches = nSample/hparams.common.n_batch;
    num_batches = nSample==0 ? 0 : max(num_batches,1);
    _INFO("%s@[%s]: tokens=%zu nSamp=%d nBach=%d\n", __func__,isTrain?"train":"eval", tokens.size(),all_samps.size(),num_batches); 
}

void DataLoader::Shuffle(int flag)  {
    size_t count = all_samps.size();
    assert(count>0);
    struct train_params_common& train_params = hOPT->train_params;
    if(n_unique_tokens==-1){    //  first run
        std::vector<size_t> token_noccurs;
        token_noccurs.resize(n_vocab, 0);   //params.n_vocab
        for (unsigned int i = 0; i < tokens.size(); ++i) {
            ++token_noccurs[tokens[i]];
        }
        n_unique_tokens = 0;
        for (unsigned int i = 0; i < token_noccurs.size(); ++i) {
            if (token_noccurs[i] == 0) continue;
            ++n_unique_tokens;
        }
        _INFO("%s: number of unique tokens: %d\n", __func__, n_unique_tokens);        
    }

    //  hash_combine(samples_begin,samples_size[i],sample_count
    shuffle_samples_hash = SAMP::HASH(train_params.fn_train_data.c_str(),all_samps);    
    // compute_samples_hash(train_params.fn_train_data.c_str(), samp_begin.data(), samp_size.data(), samp_size.size());
    const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != all_samps.size());
    if (changed_train_data) {
        _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (train_params.force_reshuffle) {
        _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((train->shuffle_rng_state_current == "") || changed_train_data || train_params.force_reshuffle) {
        train->shuffle_rng_state_current = mt19937_seed_to_state(train_params.seed);
        train->shuffle_sample_count = all_samps.size();
        train->shuffle_next_sample = 0;
        train->shuffle_samples_hash = shuffle_samples_hash;
    }
    
    // shuffled_samples_offs.resize(samp_begin.size());
    // shuffled_samples_begin.resize(samp_begin.size());
    // shuffled_samples_size.resize(samp_size.size());
        
    std::mt19937 rng;
    mt19937_set_state(rng, train->shuffle_rng_state_current);
    // sort indices by random value for each index
       
    std::vector<unsigned> rnd;    
    idcs.resize(count);    rnd.resize(count);
    for (unsigned i=0; i<count; ++i) {
        idcs[i] = i;
        rnd[i]  = rng();
    }
    std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b){
        // stable sort for reproducibility
        return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
    }); 
    /*epoch_samps.clear();       
    for (unsigned i=0; i<count; ++i) {
        hSAMP samp=all_samps[idcs[i]];
        samp->off = samp->len * ((double) rng() / (double) (rng.max()-1));
        epoch_samps.push_back(samp);
    }*/
    /*
    for (unsigned i=0; i<count; ++i) {
        shuffled_samples_offs[i] = (size_t) ((samp_size[idcs[i]] - 1) * ((double) rng() / (double) (rng.max()-1)));
    }    
    for (unsigned i=0; i<count; ++i) {
        shuffled_samples_begin[i] = samp_begin[idcs[i]];
    }
    for (unsigned i=0; i<count; ++i) {
        shuffled_samples_size[i] = samp_size[idcs[i]];
    }*/

    train->shuffle_rng_state_next = mt19937_get_state(rng); 
}
#endif