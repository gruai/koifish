#include "DataLoader.hpp"
#ifdef _DATA_LOADER_LITE_
#else
#endif

#include "gLLM.hpp"
#include "Optimizer.hpp"
#include "../ggex/llmc_utils.h"
#include "Dictionary.hpp"

void mt19937_set_state(std::mt19937& rng, const std::string& rng_state) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state.exceptions(std::stringstream::failbit);
    s_rng_state.str(rng_state);
    s_rng_state >> rng;
}

std::string mt19937_get_state(const std::mt19937& rng) {
    std::stringstream s_rng_state;
    s_rng_state.imbue(std::locale::classic());
    s_rng_state << rng;
    return s_rng_state.str();
}

std::string mt19937_seed_to_state(unsigned seed) {
    std::mt19937 rng(seed);
    return mt19937_get_state(rng);
}
/*
void SAMP::Refresh(SampLoader *loader,void *ctx,std::vector<int32_t>& tok_ids,int typ)  {
    assert(0);      // Deprecated
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
            TOKEN_ID tok = loader->TokenAt(pos+i);
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
}*/


double SampLoader::DecodeVerify(hSAMP samp,hGensor tokens,hGensor logits,int flag)    {
    int nC = tokens->ne[0],nB = tokens->ne[1],b,c,j,cand=-1,nz=0;
    int _nvocab = hDict->n_vocab;
    assert( tokens->type== GGML_TYPE_I32 && tokens->ne[2]==1);
    double off=0,sum=0,avg,last_err=0;
    float *p = nullptr,p1,accu_self=0;
    if(logits!=nullptr) {
        assert( logits->type== GGML_TYPE_F32 );
        assert(logits->ne[1]==nC && logits->ne[2]==nB && _nvocab==logits->ne[0]);
        p = (float*)(logits->data);
    }
    
    assert(nC>0 && nB>0);
    int *t0 = (int*)tokens->data,*t=t0,nMatch=0,nMiss=0,target;
    for(b=0; b<nB; b++){
        string line;
        t++;                assert(*t>0 && *t<_nvocab);
        for(c=0; c<nC; c++,t++){        
            target = c==nC-1 ? samp->last_target : *t;
            off = LOSS_cross_entropy_1(_nvocab,p,target,cand);    
            p+=_nvocab;   sum += off*off;     nz++;
            if(cand==target){
                nMatch++;
            }else{
                nMiss++;
            }            
            line += hDict->T2STR(cand);
        }
        last_err = off;
        curDeTexts.push_back(line);
        break;
    }
    accu_self = nMatch*1.0/nz;
    avg =sqrt(sum/(nz));
    return last_err;
}

int SampLoader::nLeastCTX(int  flag) {   
    auto samp = SampAt(0);
    return hDict->tokenizer_add_bos? samp->len : samp->len+1;
}

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


void SampLoader::Samp2Batch(int k,hSAMP samp,struct train_params_& params,int flag)    {   
    samp_toks.clear();        
    auto dialect = hDict->dialect;
    bool fill_with_next_samples=params.fill_with_next_samples,isDialect = hDict->isDialect;
    size_t starting=samp->pos+samp->jump,_nctx = params.n_ctx,_nToken=nTokens();            //    tokens_input->ne[0];;    
    hostBatch->Set(0, k, 0, 0, hDict->bos);  //ggml_set_i32_nd(G(tokens_input), 0, k, 0, 0, hDict->bos);
    samp_toks.push_back(hDict->bos);
    for (int64_t i=0; i<_nctx; ++i) {    
        TOKEN_ID token = hDict->eos;        
        
        // eos ???
        if (starting >= _nToken) {
            if(isRecycle)
                starting = 0;
        }
        if(starting<_nToken)
            token = TokenAt(starting); 
        else    
            token = hDict->eos;
        if(isDialect){
            assert(dialect[token]>0);
            token = hDict->mapT2T[token];
        }
        if(DEBUG.train_datas==1)
            token = i;      //only for debug

        ++starting;   
        if(hostTargetProbs==nullptr){

        }else{
            if(isTarget_1)     {
#ifdef _TENSOR_CUD_
                hostTargetProbs->Set( (int) i, (int) k, 0, 0, token); 
#else
                hostTargetProbs->Set(0, (int) i, (int) k, 0, token); 
#endif
            }                           
            else    {
                hostTargetProbs->Set(token, (int) i, (int) k, 0, +1.0f);
            }        
        }
        samp_toks.push_back(token);
        
        if (i+1<_nctx) {
            hostBatch->Set( (int) (i + 1), (int) k, 0, 0, token);
        }else{
            samp->last_target = token;
        }
    }
}

//Important!  this would update input & target_probs of model!
size_t SampLoader::update_batch(int x,Fish* fish){    
    struct train_params_ _params = hOPT->TrainParams();
    assert(fish==hOPT->_fish);
    // struct llama_context * lctx=(struct llama_context *)(hOPT->app_ctx);
    cur_samps.clear();      
    hostBatch->Zero();          hostTargetProbs->Zero();
    int64_t nAllSamples_ = nShard();
    // const TOKEN_ID    * train_data=tokens.data();
    size_t  k,n_train_data = nTokens(); // tokens.size();
    sample_separation_eos=!_params.separate_with_eos;
    sample_separation_bos=!_params.separate_with_bos;
    double t_Samp = 0,nrm=0,a;
    bool sample_random_offsets=_params.sample_random_offsets;
    // assert(samples_count > 0);
    // assert(ggml_is_matrix(tokens_input));
    size_t nSampInBatch = fish->hparams.n_batch();  
    GST_TIC(T0);
    bool isLog = false;
    if(isLog) _INFO("BATCH_%ld ",next_sample);   
    hSAMP samp = nullptr;

    GST_TIC(tic);
    for (k=0; k<nSampInBatch; ++k) { 
        if(batch_sample=="stacking"){
            samp->jump++;
        }else{            
            samp = Next();  //SampAt((next_sample + k) % samples_count);  
        }
        if(samp==nullptr)   return 0x0;

        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);  
        if(GST_TOC(tic)>20){        //for long-time data-update
            _INFO("\r[%s] k=%d(%d) T=%.3g ...",__func__,k,nSampInBatch,GST_TOC(tic));   
            tic = Clock::now( );
        }
        cur_samps.push_back(samp);
        Samp2Batch(k,samp,_params);   
        if(isLog && k<6 && batch_sample!="stacking"){
            sentence = dolphin->T2STR(samp_toks,0x0);     //llama_token_to_piece(lctx, samp_toks[0]);
            _INFO(" (%ld,%d)@\"%s\"",samp->pos,samp->jump,sentence.c_str());     //sample_size
        }else if(type == SampLoader::TYPE::DT_EVAL)  {
            if(k==0){
                sentence = dolphin->T2STR(samp_toks,0x0); 
                // assert(raw_t[0]==hDict->bos);
            }
        }

        if(batch_sample!="stacking"){
            for(auto wiki : fish->wikis){
                // nrm = wiki->InductLogits(k,samp_toks,nullptr,G(target_probs),-1); 
                nrm = wiki->InductLogits(k,samp_toks,nullptr,nullptr,-1); 
            }
        }        
    }
    hGensor tokens_input=fish->Input(),target_probs=hOPT->hTargetProbs();    assert(tokens_input!=nullptr);
    int *raw_t = (int*)(tokens_input->data);
#ifdef _TENSOR_CUD_
    assert(isInRange((int*)(hostBatch->data),hostBatch->size(),0,hDict->n_vocab));
    tokens_input->OverWrite(hostBatch);
    if(target_probs!=nullptr)    {
        target_probs->OverWrite(hostTargetProbs);    
    }
#else
    hostBatch->OverWrite(tokens_input,false);
    if(target_probs!=nullptr)    {
        hostTargetProbs->OverWrite(target_probs,false);    
        // assert(nSampInBatch==target_probs->ne[2]);
        // ggml_set_f32(target_probs, 0.0f);
    }
#endif

    t_Samp = GST_TOC(tic);
    // Decode(tokens_input);       //only for debug
    if(isLog) _INFO("\tT=%g\n",GST_TOC(T0));
    if(batch_sample=="stacking"){
        CHILD_0909_WIKIS assert(0);
        /*if(fish->wiki->isInduct()){
            GST_TIC(tic);  
            samp->Refresh(this,lctx,samp_toks,0x0);          //refresh samp_toks               
            nrm = fish->wiki->InductLogits(nSampInBatch,samp_toks,exLogits,target_probs,0x0);           
            _INFO("\t stacking@%d\"%.*s...\" nrm=%g\tT=%.4gs\t\n",samp->pos,64,samp->desc.c_str(),nrm,GST_TOC(tic));             
        }*/
    }
    // if(type == SampLoader::TYPE::DT_TRAIN) 
    //     next_sample += nSampInBatch;
    // else{
    //     if(!isFixEvalSample)
    //         next_sample += nSampInBatch;
    // }
    return nSampInBatch;
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

std::vector<hDataToken> DataTokenSet::MakeInstance(struct CLI_params& params,ConsiceDict *hDict, int flag){
    DataTokens dts;
    JSON jdata = jKEY(params.jConfig,{"datasets"});
    if(jdata.empty()){      //Deprecated
        auto hTokenset = std::make_shared<DataTokenSet>(hDict);  
        // hTokenset->serial_root = "/home/cys/rnd/lic/datasets/story19M___[gpt2char]_"; //only for debug" 
        hTokenset->serial_root = "./datasets/story19M___[gpt2.Q8_0]_";
        // hTokenset->serial_root = "./datasets/story19M___[bitnet-m7-70m.Q8_0]_";
        dts.push_back(hTokenset);
    }else{
        for(JSON::const_iterator it = jdata.begin(); it != jdata.end(); ++it)    {
            auto k =it.key();     
            if(!k.empty() && k[0]=='#')     
                continue;
            if(k=="debug"){
                continue;
            }
            auto v=it.value();
            // auto hTokenset = std::make_shared<DataTokenSet>(hDict); 
            auto hTokenset = std::make_shared<GlobTokenset>(it,hDict);   
            dts.push_back(hTokenset);  
        }        
    }

    return dts;
}

GlobTokenset::GlobTokenset(JSON::const_iterator jit,ConsiceDict *hDict,int flag) : DataTokenSet(hDict)    {
    header_bytes = HEADER_SIZE * sizeof(int);
    int num_processes=1,process_rank=0;
    B = hDict->hparams.n_batch(),       T = hDict->hparams.n_ctx();
    total_batch_size_bytes = ((num_processes * (B * T)) * sizeof(uint16_t));
    local_batch_offset_bytes = process_rank * B * T * sizeof(uint16_t);

    auto k =jit.key();  
    auto v = jit.value();
    string pattern = v["glob"];
    glob_t glob_result;
    int glob_status = glob(pattern.c_str(), 0, NULL, &glob_result);
    if (glob_status != 0) {
        _INFO("%s Error: glob failed @\"%s\"\n", __func__,pattern.c_str());
        exit(EXIT_FAILURE);
    }
    if (glob_result.gl_pathc == 0) {
        _INFO("%s No files found matching the pattern: %s\n", __func__,pattern.c_str());
        exit(EXIT_FAILURE);
    }
    int nFile = 0;
    /*if (isShuffle) {
        manual_seed(&shuffle_rng, 42 + process_rank);
        shard_indices = (int*)mallocCheck(glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(shard_indices, (int) glob_result.gl_pathc);
        intra_shard_indices = NULL;  // dynamically allocated allowing different shard sizes
    }*/

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    nMostTok = 0;
    for (int id = 0; id < glob_result.gl_pathc; id++) {
        string sPath = glob_result.gl_pathv[id];
        shard_paths.push_back(sPath);
        int64_t shard_ntok = PrepareShard( id );
        nFile++;
        // assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        nMostTok += shard_ntok;
    }
    double nG = nMostTok/1.0e9;
    
    _INFO( "[%s] %s find %.8gG tokens @\"%s\"(%d files)\n", __func__,name.c_str(),pattern.c_str(),nG,nFile );
}

bool GlobTokenset::NextShard(int flag)  {
    PrepareShard(shard_index++,true);      
    // InitSamps
    int n_ctx = hDict->hparams.n_ctx();
    // double rSplit = 1.0-hDict->hparams.rSplit;
    // samples_begin.clear();      samples_size.clear();
    // samples_begin.push_back(0);
    size_t nToken = tokens.size(),nFirst=std::min((size_t) n_ctx, nToken),step=1;
    // samples_size.push_back(nFirst);
    size_t end = (nToken >= n_ctx) ? (nToken - n_ctx) : 0;
    if(end>10*1024*1024){
        step = n_ctx;
    }
    for (size_t sample_begin = 1; sample_begin < end; sample_begin+=step) {
        shard_samps.push_back(new SAMP(sample_begin,n_ctx));
    }
    size_t nSample = shard_samps.size();
    size_t num_batches = nSample/hDict->hparams.n_batch();
    num_batches = nSample==0 ? 0 : max(num_batches,(size_t)1);
    _INFO("\t%s@[%s]: tokens=%.3g(M) nSamp=%d nBach=%d\n", __func__,name.c_str(), nToken/1.0e6,shard_samps.size(),num_batches); 
    return true;
}

int64_t GlobTokenset::PrepareShard(int id, bool load, int flag) {
    if (isShuffle) {
        // id = shard_indices[id];
    }
    // use the first glob match as the filename for now
    const char* filename = shard_paths[id].c_str();
    if (fpToken != NULL) {
        fcloseCheck(fpToken);
    }
    fpToken = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, fpToken);
    if (header[0] != 20240520) {
        printf("Bad magic in the data file\n");
        printf("---> HINT: Are you passing in a correct file?\n");
        printf("---> HINT: The data encoding may have changed, re-run data prepro or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    int64_t nTok0 = header[2]; // number of tokens in the file
    assert(nTok0 > 0); // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    fseekCheck(fpToken, 0, SEEK_END); // seek to end of file
    file_size_bytes = ftell(fpToken); // read the offset, i.e. file size
    fseekCheck(fpToken, 0, SEEK_SET); // seek back to the beginning
    // we expect nTok0 in the file to be consistent with filesize, assert that is the case
    int64_t expected_file_size = HEADER_SIZE * sizeof(int) + nTok0 * sizeof(uint16_t);
    if (file_size_bytes != expected_file_size) {
        printf("Error: file size is not as expected\n");
        exit(EXIT_FAILURE);
    }
    if(load){
        int szT=sizeof(uint16_t),nT = (file_size_bytes-header_bytes)/szT;
        assert(nT==nTok0);
        tokens.resize(nT);
        fseekCheck(fpToken, (int) header_bytes, SEEK_SET);
        uint16_t *tmp16=new uint16_t[nT];
        if(fread(tokens.data(),szT,nT,fpToken)!=nT) {
            _INFO("Error: file size is not as expected\n");
            return 0x0;
        }else{
            nT = min(nT,1024);
            for(int i=0;i<nT;i++){
                assert(0<=tokens[i] && tokens[i]<nVocab);
            }
        }
        delete[] tmp16;
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    shard_num_samples = (nTok0 * sizeof(uint16_t) - sizeof(uint16_t)) / total_batch_size_bytes;
    if(load){
        _INFO("[shard]: tokens=%.3g(M) nSamp=%d @\"%s\"\n", nTok0/1.0e6,shard_num_samples,filename); 
    }
    return nTok0;
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

bool SampLoader::Serialize(const std::string&path, bool isSave, int flag){
try{
    FSerial S(path,isSave,flag);
    if(!S.isValid())
        return false;
    int _nvocab = hDict->n_vocab;
    uint32_t seed=hOPT->TrainParams().seed;
    _INFO("%s %s@%s...",__func__,isSave?"save@":"load@",path.c_str());
    _CHECK( S.Serial(_nvocab,isSave,flag) );
    // _CHECK( S.Serial(tokens,isSave,flag) );
    // _CHECK( S.Serial(n_unique_tokens,isSave,flag) );
    _CHECK( S.Serial(shuffle_samples_hash,isSave,flag) );
    _CHECK( S.Serial(seed,isSave,flag) );

    _CHECK( S.Serial(batch_sample,isSave,flag) );
    // _CHECK( S.Serial(ids,isSave,flag) );
    bool bRet = S.Serial_Vector<SAMP,SAMP>(shard_samps,isSave,flag);
    _CHECK(bRet);
    if(shard_samps.size()==0)
        return false;
    size_t nT = nTokens();
    _INFO("\r%s %s@\"%s\" ... OK. \r\n\tnSamp=%ld @[Datasets(nToken=%ld  hash=%llX]\n",__func__,isSave?"save":"load",path.c_str(),
        shard_samps.size(),nT,shuffle_samples_hash);
    for(auto samp : shard_samps){
        // if(id>=nT){
        //     _INFO("\t\tInvalid id(%ld) > nTokens=%d\n", id,nT);  
        //     return false;
        // }
    }
    if(isSave){
        
    }else{
        size_t nSample = shard_samps.size();
        num_batches = nSample/hOPT->TrainParams().n_batch;
        num_batches = nSample==0 ? 0 : max(num_batches,1); 
        _INFO("\t nBatch in each epoch=%d\n", num_batches); 
    }
    if(type==TYPE::DT_TRAIN){
        /*assert(train!=nullptr);   */
        if(isSave)    {

        }else{
            shuffle_rng_state_current = mt19937_seed_to_state(hOPT->TrainParams().seed);
            shuffle_sample_count = shard_samps.size();
            next_sample = 0;
            shuffle_samples_hash = shuffle_samples_hash; 
        }
        _INFO("\t%s@[%s]: hash=%lld\n", "train_state",path.c_str(), shuffle_samples_hash);  
    }
    return true;
}catch(...){
    return false;
}
}

SampLoader::SampLoader(Fish *g_,const string&n,bool isNewTS,int flag) {
    name = n;
    assert(g_!=nullptr);
    dolphin = dynamic_cast<NLP_AutoRegressive*>(g_);         
    if(dolphin==nullptr){
        assert(0);
        return ;
    }
    isTarget_1 = g_->hparams.is( {"model","target"},string("OneHot") );
    return ;
}

bool SampLoader::Prepare(hDataToken hT,int flag){
    bool isNewTS = hT==nullptr;
    hDict = dolphin->hDict;     
    if(isNewTS){
        hTokens = std::make_shared<DataTokenSet>(hDict.get());
    }else
        hTokens = hT;   //dolphin->hTokenset;
    assert(hTokens!=nullptr && hDict!=nullptr);  
    hOPT = dolphin->hOPT.get();
    assert(hOPT!=nullptr);

    if(hTokens!=nullptr){
        if(!hTokens->NextShard())
            return false;
        shard_samps = hTokens->shard_samps;
    }
    // bos = hDict->bos;
    // eos = hDict->eos;
    // Batch tensor would be set @UpdateBatch!
    struct train_params_ _params = hOPT->TrainParams();   
    int n_ctx = _params.n_ctx,n_batch = _params.n_batch;
    assert(n_ctx>0 && n_batch>0);
    SHAPE shape={n_ctx, n_batch},sp1={1, n_ctx, n_batch};
    hostBatch = std::make_shared<GTensor>(shape,GGML_TYPE_I32);    
#ifdef _TENSOR_CUD_
    sp1 = shape;
#endif
    if(isTarget_1){
        hostTargetProbs = std::make_shared<GTensor>( sp1,GGML_TYPE_I32);
    }else{
        sp1={hDict->n_vocab, n_ctx, n_batch};
        hostTargetProbs = std::make_shared<GTensor>( sp1,GGML_TYPE_F32);
    }
    hostBatch->Alloc();     hostTargetProbs->Alloc();
    return true;
}
            
/*
    @@@tokenize_file:
        samples_begin.push_back(0);
        samples_size.push_back(std::min((size_t) context_length, tokens.size()));
        size_t end = (tokens.size() >= context_length) ? (tokens.size() - context_length) : 0;
        for (size_t sample_begin = 1; sample_begin < end; ++sample_begin) {
            samples_begin.push_back(sample_begin);
            samples_size.push_back(context_length);
        }
*/
void SampLoader::SetSamples(std::vector<size_t>& samp_0,std::vector<size_t>& samp_L,bool isTrain,CLI_params& hp_,int flag)  {
    // hparams = hp_;
    batch_sample = dolphin->hparams.batch_sample;

    double rSplit = 1.0-dolphin->hparams.rSplit;
    // hTokens = hDT;
    size_t nSample = samp_0.size(),pick=(size_t)(nSample*rSplit),i;
    //    assert(samp_begin.size() == samp_size.size());   
    if(isTrain){
        for(i=0;i<pick;i++){
            shard_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }
          
        Shuffle( );
    } else if(pick<nSample) {
        for(i=pick;i<nSample;i++){
            shard_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }

        if(1){  //too many batch in eval-set, so just random pick
            Shuffle( );
        }else{
            assert(0);
            
        }
    }
    nSample = shard_samps.size();
    num_batches = nSample/dolphin->hparams.n_batch();
    num_batches = nSample==0 ? 0 : max(num_batches,1);
    _INFO("%s@[%s]: tokens=%zu nSamp=%d nBach=%d\n", __func__,isTrain?"train":"eval", nTokens(),shard_samps.size(),num_batches); 
}

double SAMP::UpdateTag(hDataToken hDT,int *tag,int step,bool do_mask,int flag)   {
    TOKEN_ID tok;
    int nFlip = 0;
    for(size_t t=pos;t<pos+len;t++){
        tok = hDT->tokens[t];
        assert(tag[tok]<=step);
        if(tag[tok]==step){

        }else{
            if(do_mask)            {    
                tag[tok]=step;      nFlip++;
            }
            else{   //self duplicate
                if(tag[tok]<0)  
                    continue;
                else{
                    tag[tok]*=-1;       nFlip++;
                }
            }
        }
    }
    if(!do_mask) {
        for(size_t t=pos;t<pos+len;t++){
            tok = hDT->tokens[t];
            if(tag[tok]<0)  
                tag[tok] = -tag[tok];
        }
    }
    return nFlip*1.0;
}

bool SampLoader::TopoOrder(std::vector<size_t>&ids,std::mt19937& rng,int flag)  {    
    bool isRepeated = true;
    size_t count = shard_samps.size(),i,j,k,jj,pick,seed,nPick=16,nLeft;
    size_t nSampInBatch=dolphin->hparams.n_batch(),nVocab=hTokens->nVocab,ctx=dolphin->hparams.n_ctx(),tib=dolphin->hparams.nTokenInBatch();
    if(count<nSampInBatch*10)
        return false;
    GST_TIC(tic);
    size_t nBatch = (size_t)(count/nSampInBatch*0.7),nz=0;    
    int step=1,*stp=new int[nVocab]();
    for(i=0;i<nVocab;i++)       stp[i]=step;

    hSAMP cur,next=nullptr;
    double rDup=0.0,avgDup=0,maxDup=0;
    for(i=0;i<nBatch;i++)    {  //for each batch
        seed = ids[i*nSampInBatch];
        cur = shard_samps[seed];
        step++;
        double rEx = cur->UpdateTag(hTokens,stp,step,true),r,rBest=FLT_MAX;        
        for(j=i*nSampInBatch+1;j<(i+1)*nSampInBatch;j++)  {
            if(0)   {   //10001/16/3 rDup=0.415(0.568)=>rDup=0.347(0.453)        10001/16/32 rDup=0.704(0.738)=>0.625(0.649)
                nLeft = count-j;        
                for(k=0,rBest=0;k<nPick;k++){
                    if(isRepeated)  {
                        jj = rng()%count;
                    }else       {
                        jj = j+rng()%nLeft;         assert(jj>=j && jj<count);
                    }
                    next = shard_samps[ids[jj]];
                    r = next->UpdateTag(hTokens,stp,step,false);      
                    if(r>rBest){
                        rBest = r;  pick = jj;
                    }
                }                
                std::swap(ids[j],ids[pick]);
            }
            next = shard_samps[ids[j]];
            r = next->UpdateTag(hTokens,stp,step,true);  
            assert(r==rBest || rBest==FLT_MAX);
            rEx+=r;
        }
        rDup = 1.0-rEx/tib;
        avgDup += rDup;     maxDup = max(maxDup,rDup);
        nz++;
        // if(nz>10000)  break;      //only for debug
    }
    avgDup /= nz;
    _INFO("SampLoader_%s nBatch=%ld rDup=%.3g(%.3g) T=%.3g(sec)\n",__func__,nz,avgDup,maxDup,GST_TOC(tic));
    delete[] stp;

    return true;
}

string SampLoader::IterInfo(int flag){
    char buffer[256];
    if(hTokens->shard_paths.size()>0)  {
        sprintf(buffer,"%zu/%zu@S%d",std::min(1 + next_sample, shuffle_sample_count), shuffle_sample_count,hTokens->shard_index);
    }else{
        sprintf(buffer,"sample@%zu/%zu",std::min(1 + next_sample, shuffle_sample_count), shuffle_sample_count);
    }
    
    return buffer;
}


bool SampLoader::isNextEpoch(int train_epochs){
    if(hTokens->shard_paths.size()>0)   
        return false;
    if( next_sample <shuffle_sample_count )
        return false;

    _INFO("%s: reshuffle samples. completed epochs: %llu\n", __func__, train_epochs);
    shuffle_rng_state_current =shuffle_rng_state_next;
    // train->shuffle_rng_state_next = shuffle_samples(
    //     train->shuffle_rng_state_current,data->shuffled_samples_offs,data->shuffled_samples_begin,
    //     data->shuffled_samples_size,data->samples_begin,data->samples_size,data->samples_count);
    
    Shuffle();           //SAMP_0816
    // train->shuffle_rng_state_next = shuffle_samples(
    //     train->shuffle_rng_state_current,loader->shuffled_samples_offs.data(),
    //     loader->shuffled_samples_begin.data(),loader->shuffled_samples_size.data(),
    //     loader->samp_begin.data(),loader->samp_size.data(),loader->samp_size.size());
    
    next_sample = 0;
    return true;
}

void SampLoader::Shuffle(int flag)  {
    size_t count = shard_samps.size(),i;
    assert(count>0);
    struct train_params_ _params = hOPT->TrainParams();
    //  hash_combine(samples_begin,samples_size[i],sample_count
        
    const bool changed_train_data = false;  //(shuffle_samples_hash != hOPT->shuffle_samples_hash) || (train->shuffle_sample_count != shard_samps.size());
    if (changed_train_data) {
        _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (_params.force_reshuffle) {
        _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((shuffle_rng_state_current == "") || changed_train_data || _params.force_reshuffle) {
        shuffle_rng_state_current = mt19937_seed_to_state(_params.seed);
        shuffle_sample_count = shard_samps.size();
        next_sample = 0;
        
    }
    
    // shuffled_samples_offs.resize(samp_begin.size());
    // shuffled_samples_begin.resize(samp_begin.size());
    // shuffled_samples_size.resize(samp_size.size());
        
    std::mt19937 rng;
    mt19937_set_state(rng, shuffle_rng_state_current);
    // sort indices by random value for each index
       
    std::vector<unsigned> rnd;    
    std::vector<size_t> ids;
    ids.resize(count);    rnd.resize(count);
    for (i=0; i<count; ++i) {
        ids[i] = i;
        rnd[i]  = rng();
    }
    std::sort(ids.begin(), ids.end(), [&rnd](size_t a, size_t b){
        // stable sort for reproducibility
        return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
    }); 
    std::vector<hSAMP> tSamps;      tSamps.resize(count);

    //  TopoOrder(ids,rng);     // May better, need more testing

    for(i=0; i<count; ++i){
        tSamps[i] = shard_samps[ids[i]];
    }
    shard_samps = tSamps;
    
    shuffle_samples_hash = SAMP::HASH(fp_data.c_str(),shard_samps);
    shuffle_rng_state_next = mt19937_get_state(rng); 
}

// mark each byte with its utf8 unit number.
// returns the number of utf8 characters.
// e.g. when bytes == '\x61\xD0\xB0\x62',
// then utf8_units will become [0,0,1,0]
// utf8_nunits will become [1,2,2,1] and 3 is returned.
// bytes where utf8_units is zero, are the begin of an utf8 character.
static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}
static size_t mark_utf8_units(const char* bytes, int * utf8_units, int * utf8_nunits, size_t count) {
    size_t offs = 0;
    size_t count_utf8 = 0;
    while(offs < count) {
        int len = (int) utf8_len(bytes[offs]);
        for (int i=0; i<len; ++i) {
            utf8_units[offs+i]  = i;
            utf8_nunits[offs+i] = len;
        }
        offs += len;
        ++count_utf8;
    }       
    return count_utf8;
}

int ConsiceDict::STR2T(const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag)  {   
    int n_tokens = -1;
    if(wiki_tutor!=nullptr) {
        n_tokens = wiki_tutor->STR2T(txt,btch,flag);
    }else{
        assert(0);
        /*llama_model *lam_ = dolphin->GetRawModel(); //    <llama_model *>(hLLM);
        assert(lam_!=nullptr);
            //  would call llama_tokenize_internal
        int n_tokens = llama_tokenize( lam_, txt,txt_len,btch.data(),(int) btch.size(),tokenizer_add_bos, false);*/
        
    }
    if(tokenizer_add_bos)
        assert(btch[0]==bos);
    return n_tokens;
}
std::string ConsiceDict::T2STR(TOKEN_ID tok,int flag ) { 
    string word = "";
    
    if(wiki_tutor!=nullptr){
        word = wiki_tutor->T2STR(tok,flag);
    }else{
        assert(0);
    }       
    return word;   
}   

bool DataTokenSet::Load(struct CLI_params& hparams,void *hLLM,int flag){
    if(hparams.passLoadToken)
        return true;
    auto arch = hparams.ModelArch();
    GST_TIC(tic);
    string batch_sample = hparams.KV({"data","batch_sample"} ); 
    // rSplit = jKV(jConfig,{"data","eval_split"},rSplit );
    string ssf = "./dataset/Serial/";
    string dict_type = hparams.KV({"dict","type"} );
    ssf += "_["+hparams.model_title+dict_type+"]_"+".tokenset";       //hparams.serial_path+
    ssf = serial_root+".tokenset"; //only for debug
    // string ssf = hparams.serial_path+".tokenset";     
    if( Serialize(ssf,false) ){
        
    }else{
        if(hLLM==nullptr && arch!=MODEL_ARCH::NLP_GPT2_char && arch!=MODEL_ARCH::NLP_GPT2)
            return false;
        fpath = hparams.GetDataPath(""); //fp_train_data.c_str();
        tokens.clear();        
        FILE *fp = std::fopen(fpath.c_str(), "rb");
        if (fp == NULL) {
            _INFO("%s: warning: empty or not existing training data file '%s'\n", __func__, fpath.c_str());
            return false;
        } else {
            seek(fp, 0, SEEK_END);
            fsize = F_SIZE(fpath,fp);
            seek(fp, 0, SEEK_SET);
        }   
        _INFO("[Load&Token]: @'%s' fsize=%.3g(M) ... ", fpath.c_str(),fsize/1.0e6);
        const int n_max_tokens_overhead = 1;    
        if(fsize+n_max_tokens_overhead*2>=INT_MAX){
            _INFO("\n%s reduce fsize from %ld=>%ld\n",__func__,fsize,INT_MAX-n_max_tokens_overhead*2);
            fsize = INT_MAX-n_max_tokens_overhead*2;            
        }
        char *buf=new char[fsize];        //buf.resize(fsize);
        errno = 0;
        std::size_t ret = std::fread(buf, fsize, 1, fp);
        if (ferror(fp)) {
            die_fmt("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
        size_t count_utf8 = 0;
        if(0)   {
            std::vector<int> utf8_units,utf8_nunits;
            utf8_units.resize(fsize);            utf8_nunits.resize(fsize);
            count_utf8 = mark_utf8_units(buf, utf8_units.data(), utf8_nunits.data(), fsize);
        }
        
        std::vector<TOKEN_ID> btch;
        size_t cur=0,step=10*1024*1024,len;
        btch.resize(step);
        while(cur<fsize){     
            GST_TIC(t0);
            len = min(step,fsize-cur);  
            int n_tokens = hDict->STR2T(buf+cur,len,btch,flag);
            // int n_tokens = llama_tokenize( lam_, buf+cur,len,btch.data(),(int) btch.size(),false, false);            
            if (n_tokens<=0) {
                 _INFO("Invalid n_tokens=%d @%ld!!!\n",n_tokens,cur);    
                 assert(n_tokens>0);            
            }
            for(int i=0;i<n_tokens;i++){
                auto t = btch[i];
                assert(t>=0 && t<nVocab);
                if(t<0 || t>=nVocab){
                    _ERROR("\n======== %s Invalid token(%d) @%d !========\n",__func__,t,tokens.size()+i);
                    return false;
                }
            }
            
            _INFO("\r\t tokenize %.3g%%\t[%ld:%ld]\tT=%.3g(s) ......",cur*100.0/fsize,cur,tokens.size(),GST_TOC(t0));
            tokens.insert(tokens.begin(),btch.begin(),btch.begin()+n_tokens);
            cur += len;
        }
        delete[] buf;
        
        UniqueTokens(-1);
        assert(nUnique<=nVocab);
        Serialize(ssf,true);
        
    }
    size_t nTokens = tokens.size();
    _INFO("\r[Load&Token]: @'%s' fsize=%.3g(M) nTokens=%.3g(M) nUnique=%ld T=%.3g(s)\t\t\t\n", 
        ssf.c_str(),fsize/1.0e6,nTokens/1.0e6,nUnique,GST_TOC(tic));
    
    return true;
}

hSAMP SampLoader::InitOneSamp(const string &prompt,hGensor input, int flag){
    const char *buf = prompt.c_str();
    std::vector<TOKEN_ID> btch;
    btch.resize(10*1024*1024);
    assert(hTokens!=nullptr);
    assert(hTokens->tokens.size()==0);
    hTokens->tokens.clear();
    int n_tokens = hDict->STR2T(buf,prompt.size(),btch,flag);
    int _nctx = dolphin->hparams.n_ctx();
    n_tokens = min(n_tokens,_nctx);         assert(n_tokens>0);

    hTokens->tokens.insert(hTokens->tokens.begin(),btch.begin(),btch.begin()+n_tokens);
    
    shard_samps.clear();
    shard_samps.push_back(new SAMP(0,n_tokens));    
    hSAMP samp = shard_samps[0];
    // assert(_nvocab==0);
    // _nvocab = hDict->n_vocab;
    num_batches = 1;
    sentence = hDict->T2STR(hTokens->tokens);

    // if(input!=nullptr)
    //     Samp2Batch(0,samp,input,nullptr,dolphin->hparams.common);  
    return samp;
}

bool DataTokenSet::InitSamps(unsigned context_length,std::vector<size_t>& samples_begin,std::vector<size_t>&samples_size,int flag){
    samples_begin.clear();      samples_size.clear();
    samples_begin.push_back(0);
    size_t nToken = tokens.size(),nFirst=std::min((size_t) context_length, nToken),step=1;
    samples_size.push_back(nFirst);
    size_t end = (nToken >= context_length) ? (nToken - context_length) : 0;
    if(end>10*1024*1024){
        step = context_length;
    }
    for (size_t sample_begin = 1; sample_begin < end; sample_begin+=step) {
        samples_begin.push_back(sample_begin);
        samples_size.push_back(context_length);
    }
    size_t nSamp = samples_begin.size();
    return true;
}

void DataTokenSet::Append(TOKEN_ID id,int flag){
    assert(id>=0 && id<hDict->n_vocab);
    tokens.push_back(id);
}
int DataTokenSet::UniqueTokens(size_t n_1,int flag){
    mapT2T.clear();
    // std::vector<size_t> token_noccurs;
    dialect.resize(nVocab, 0);   //params.n_vocab
    assert(nVocab>0);
    for (unsigned int i = 0; i < tokens.size(); ++i) {
        TOKEN_ID id = tokens[i];
        assert(id>=0 && id<nVocab);
        // if(id==28739)        //only for debug
        //     id=28739;
        ++dialect[id];
    }
    nUnique = 0;
    for (unsigned int i = 0; i < dialect.size(); ++i) {
        if (dialect[i] == 0) continue;
        TOKEN_ID id = dialect[i];
        mapT2T[i] = nUnique;
        ++nUnique;
    }    
    assert(mapT2T.size()==nUnique);
    return nUnique;
}



std::string shuffle_samples_X(
        const std::string & rng_state,size_t* shuffled_offs,
        size_t            * shuffled_begins,
        size_t            * shuffled_sizes,
        const size_t      * begins,
        const size_t      * sizes,
        size_t              count) {
    if (count == 0) return rng_state;

    std::mt19937 rng;
    mt19937_set_state(rng, rng_state);

    // sort indices by random value for each index
    std::vector<size_t> idcs;
    {
        std::vector<unsigned> rnd;
        idcs.resize(count);
        rnd.resize(count);
        for (unsigned i=0; i<count; ++i) {
            idcs[i] = i;
            rnd[i]  = rng();
        }

        std::sort(idcs.begin(), idcs.end(), [&rnd](size_t a, size_t b){
            // stable sort for reproducibility
            return (rnd[a] == rnd[b]) ? (a < b) : (rnd[a] < rnd[b]);
        });
    }

    // create random offsets
    for (unsigned i=0; i<count; ++i) {
        shuffled_offs[i] = (size_t) ((sizes[idcs[i]] - 1) * ((double) rng() / (double) (rng.max()-1)));
    }

    // reorder begins and sizes by sorted indices
    for (unsigned i=0; i<count; ++i) {
        shuffled_begins[i] = begins[idcs[i]];
    }

    for (unsigned i=0; i<count; ++i) {
        shuffled_sizes[i] = sizes[idcs[i]];
    }

    return mt19937_get_state(rng);
}