#include "DataLoader.hpp"
#ifdef _DATA_LOADER_LITE_
#else
#include "Ganglia.hpp"
#include "../ggex/GG_util.hpp"
#include "../LLAMA/common/common.h"

//Important!  this would update input & target_probs of model!
int64_t DataLoader::update_batch(int sample_id,std::shared_ptr<WIKI> wiki){    
    struct train_params_common *params = &(hOPT->train_params);
    struct llama_context * lctx=(struct llama_context *)(hOPT->app_ctx);
    struct ggml_tensor   * tokens_input=hOPT->tokens_input;
    struct ggml_tensor   * target_probs=hOPT->target_probs;
    int64_t samples_count=samp_size.size();
    // const llama_token    * train_data=tokens.data();
    size_t                 n_train_data=tokens.size();
    bool                   separate_with_eos=params->separate_with_eos;
    bool                   separate_with_bos=params->separate_with_bos;
    bool                   fill_with_next_samples=params->fill_with_next_samples;
    bool                   sample_random_offsets=params->sample_random_offsets;
    // GGML_ASSERT(samples_count > 0);
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(target_probs));
    int64_t n_vocab  = target_probs->ne[0],n_tokens = tokens_input->ne[0],n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_tokens == target_probs->ne[1]);
    GGML_ASSERT(n_batch  == target_probs->ne[2]);
    GST_TIC(T0);
    int64_t used_samples = 0;
    ggml_set_f32(target_probs, 0.0f);
    llama_token bos = llama_token_bos(llama_get_model(lctx));
    llama_token eos = llama_token_eos(llama_get_model(lctx));
    std::string sBos = llama_token_to_piece(lctx, bos),sEos = llama_token_to_piece(lctx, eos);

    
    // LLAMA_LOG_INFO("%s: sample_id=%d n_batch=%d n_train_samples=%zu\n", __func__, sample_id, n_batch, n_train_samples);
    _INFO("BATCH_%ld ",sample_id);   //, samples_count,n_train_data);nSampe=(%ld/%ld)
    for (int k=0; k<n_batch; ++k) {
        // LLAMA_LOG_INFO("%s: batch %d\n", __func__, k);
        std::vector<int32_t> tok_ids;
        size_t sample_idx   = (sample_id + used_samples) % samples_count;
        size_t sample_offs  = sample_random_offsets ? shuffled_samples_offs[sample_idx] : 0;
        size_t sample_begin = shuffled_samples_begin[sample_idx];
        size_t sample_size  = shuffled_samples_size[sample_idx];
        ++used_samples;
        assert(sample_offs==0);
        
        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);
        GGML_ASSERT(sample_begin+sample_size-1 < n_train_data);
        std::string sentence="";
        ggml_set_i32_nd(tokens_input, 0, k, 0, 0, bos);
        bool sample_separation_eos = !separate_with_eos;
        bool sample_separation_bos = !separate_with_bos;
        for (int64_t i=0; i<n_tokens; ++i) {
            llama_token token = eos;
            if (sample_offs >= sample_size && fill_with_next_samples) { //true only arg == "--fill-with-next-samples"
                if (!sample_separation_eos) {
                    // insert eos token to separate samples
                    sample_separation_eos = true;
                } else if (!sample_separation_bos) {
                    // insert bos token to separate samples
                    sample_separation_bos = true;
                    token = bos;
                } else {
                    // sample separation is done, continue with next sample
                    sample_separation_eos = !separate_with_eos;
                    sample_separation_bos = !separate_with_bos;
                    sample_offs  = 0;
                    sample_idx   = (sample_id + used_samples) % samples_count;
                    sample_begin = shuffled_samples_begin[sample_idx];
                    sample_size  = shuffled_samples_size[sample_idx];
                    ++used_samples;
                }
            }
            // note: no else-if here
            if (sample_offs < sample_size) {
                token = clamp(tokens[sample_begin+sample_offs], 0, (llama_token) (n_vocab - 1));
                ++sample_offs;
            }
            ggml_set_f32_nd(target_probs,  token, (int) i, (int) k, 0, +1.0f);
            tok_ids.push_back(token);
            
            if (i+1<n_tokens) {
                ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) k, 0, 0, token);
                sentence = llama_token_to_piece(lctx, token);
                // _INFO("%s,",sentence.c_str());
            }
        }
        if(k<6)
            _INFO(" %ld@\"%s...\"",sample_begin,sentence.c_str());     //sample_size
        if(wiki!=nullptr && wiki->logits!=nullptr){
            assert(target_probs->type == GGML_TYPE_F32);
            wiki->Decode(tok_ids,0,0x0);
            auto g=wiki->logits;   // wiki->logits = ggml_new_tensor_3d(ctx_input, GGML_TYPE_F32, n_vocab,  n_tokens, n_batch);
            size_t ld0=g->nb[0],ld1=g->nb[1],ld2=g->nb[2],ld3=g->nb[3],off=k*ld2;          
            assert(ld0==4); 
            float *logits = wiki->logits_out;   //n_vocab,nToken,
            // target=(float*)((char *)(target_probs->data)+i*ld1+k*ld2);
            assert(sizeof(float)*n_vocab*n_tokens==ld2 && off>=0);    
            memcpy(g->data+off,logits,ld2);                
            /*for (int64_t i=0; i<n_tokens; ++i) {
                llama_token token = tok_ids[i];
                for(int j=0;j<n_vocab;j++,logits++){                    
                    // target[j]=1.0f-*logits;
                    // void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
                    // ggml_set_f32_nd(target_probs, j, (int) i, (int) k, 0, +1.0f-*logits);
                }                    
            }*/
        }        
    }
    _INFO("\tT=%g\n",GST_TOC(T0));

    return used_samples;
}

#define _CHECK(err)                                               \
    do {                                                            \
        bool err_ = (err);                                        \
        if (err_ != true) {                                   \
            fprintf(stderr, "!!! %s error %d at %s:%d\n",  \
                #err, err_, __FILE__, __LINE__);                    \
            exit(1);                                                \
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

bool DataLoader::Serialize(const std::string&path, bool isSave, int flag){
    FSerial S(path,isSave,flag);
    if(!S.isValid())
        return false;
    
    _INFO("%s %s@%s...",__func__,isSave?"save":"load",path.c_str());
    _CHECK( S.Serial(n_vocab,isSave,flag) );
    _CHECK( S.Serial(tokens,isSave,flag) );
    _CHECK( S.Serial(n_unique_tokens,isSave,flag) );
    _CHECK( S.Serial(shuffle_samples_hash,isSave,flag) );
    _CHECK( S.Serial(hOPT->train_params.seed,isSave,flag) );

    _CHECK( S.Serial(samp_begin,isSave,flag) );
    _CHECK( S.Serial(samp_size,isSave,flag) );
    _CHECK( S.Serial(shuffled_samples_offs,isSave,flag) );
    _CHECK( S.Serial(shuffled_samples_begin,isSave,flag) );
    _CHECK( S.Serial(shuffled_samples_size,isSave,flag) );
    _INFO("\r%s %s@\"%s\" ... OK. N=%lld,hash=%lld\n",__func__,isSave?"save":"load",path.c_str(),
        n_unique_tokens,shuffle_samples_hash);
    if(isSave){
        
    }else{
        size_t nSample = samp_size.size();
        num_batches = nSample/hOPT->train_params.n_batch;
        num_batches = nSample==0 ? 0 : max(num_batches,1); 
        _INFO("%s@[%s]: tokens=%zu nSamp=%d nBach=%d\n", __func__,path.c_str(), tokens.size(),samp_size.size(),num_batches); 
    }
    if(type==TYPE::DT_TRAIN){
        assert(train!=nullptr);
        // _CHECK( S.Serial(train->train_its,isSave,flag) );
        // _CHECK( S.Serial(train->train_samples,isSave,flag) );
        // _CHECK( S.Serial(train->train_tokens,isSave,flag) );
        // _CHECK( S.Serial(train->train_epochs,isSave,flag) );
        // _CHECK( S.Serial(train->shuffle_samples_hash,isSave,flag) );
        // _CHECK( S.Serial(train->shuffle_rng_state_current,isSave,flag) );
        // _CHECK( S.Serial(train->shuffle_rng_state_next,isSave,flag) );
        // _CHECK( S.Serial(train->shuffle_sample_count,isSave,flag) );
        // _CHECK( S.Serial(train->shuffle_next_sample,isSave,flag) );   
        if(isSave)    {

        }else{
            train->shuffle_rng_state_current = mt19937_seed_to_state(hOPT->train_params.seed);
            train->shuffle_sample_count = samp_size.size();
            train->shuffle_next_sample = 0;
            train->shuffle_samples_hash = shuffle_samples_hash; 
        }
        _INFO("\t%s@[%s]: hash=%lld\n", "train_state",path.c_str(), train->shuffle_samples_hash);  
    }
    return true;
}

void DataLoader::SetSamples(std::vector<llama_token>& tokens_,std::vector<size_t>& samp_0,std::vector<size_t>& samp_L,
    bool isTrain,CLI_params& hparams,int flag)  {
    double rSplit = 1.0-hparams.rSplit;
    n_vocab = hparams.n_vocab;
    tokens = tokens_;
    size_t nSample = samp_0.size(),pick=(size_t)(nSample*rSplit);
    assert(samp_begin.size() == samp_size.size());   
    if(isTrain){
        samp_begin = std::vector<size_t>(samp_0.begin(),samp_0.begin()+pick);          
        samp_size = std::vector<size_t>(samp_L.begin(),samp_L.begin()+pick); 
         
        Shuffle( );
    } else if(pick<nSample) {
        samp_begin = std::vector<size_t>(samp_0.begin()+pick,samp_0.end());          
        samp_size = std::vector<size_t>(samp_L.begin()+pick,samp_L.end()); 
        if(1){  //too many batch in eval-set, so just random pick
            Shuffle( );
        }else{
            shuffled_samples_begin = samp_begin;
            shuffled_samples_size = samp_size;            
        }

    }
    nSample = samp_size.size();
    num_batches = nSample/hparams.common.n_batch;
    num_batches = nSample==0 ? 0 : max(num_batches,1);
    _INFO("%s@[%s]: tokens=%zu nSamp=%d nBach=%d\n", __func__,isTrain?"train":"eval", tokens.size(),samp_size.size(),num_batches); 
}

void DataLoader::Shuffle(int flag)  {
    struct train_params_common& train_params = hOPT->train_params;
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

    shuffle_samples_hash = compute_samples_hash(train_params.fn_train_data.c_str(), samp_begin.data(), samp_size.data(), samp_size.size());
    const bool changed_train_data = (shuffle_samples_hash != train->shuffle_samples_hash) || (train->shuffle_sample_count != samp_size.size());
    if (changed_train_data) {
        _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (train_params.force_reshuffle) {
        _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((train->shuffle_rng_state_current == "") || changed_train_data || train_params.force_reshuffle) {
        train->shuffle_rng_state_current = mt19937_seed_to_state(train_params.seed);
        train->shuffle_sample_count = samp_size.size();
        train->shuffle_next_sample = 0;
        train->shuffle_samples_hash = shuffle_samples_hash;
    }

    shuffled_samples_offs.resize(samp_begin.size());
    shuffled_samples_begin.resize(samp_begin.size());
    shuffled_samples_size.resize(samp_size.size());
    train->shuffle_rng_state_next = shuffle_samples(
        train->shuffle_rng_state_current,
        shuffled_samples_offs.data(),
        shuffled_samples_begin.data(),
        shuffled_samples_size.data(),
        samp_begin.data(),
        samp_size.data(),
        samp_size.size());
    }
#endif