#include "DataLoader.hpp"
#ifdef _DATA_LOADER_LITE_
#else
#endif

#include "gLLM.hpp"
#include "../ggex/GG_util.hpp"
#include "common.h"
#include "Dictionary.hpp"



void SAMP::Refresh(SampLoader *loader,void *ctx,std::vector<int32_t>& tok_ids,int typ)  {
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
    
}

void SampLoader::Samp2Batch(int k,hSAMP samp,hGensor tokens_input,hGensor target_probs,struct train_params_common& params,int flag)    {   
    tok_ids.clear();        
    auto dialect = hDict->dialect;
    bool fill_with_next_samples=params.fill_with_next_samples,isDialect = hDict->isDialect;
    size_t starting=samp->pos+samp->jump;    
    ggml_set_i32_nd(tokens_input, 0, k, 0, 0, hDict->bos);
    for (int64_t i=0; i<n_ctx; ++i) {    
        TOKEN_ID token = hDict->eos;
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
        if (starting >= nTokens()) {
            starting = 0;
        }
        assert(starting<nTokens());
        token = TokenAt(starting); //clamp(tokens[starting], 0, (TOKEN_ID) (n_vocab - 1));
        if(isDialect){
            assert(dialect[token]>0);
            token = hDict->mapT2T[token];
        }
        ++starting;   
        if(isTarget_1)     
            ggml_set_f32_nd(target_probs,  0, (int) i, (int) k, 0, token);            
        else{
            ggml_set_f32_nd(target_probs,  token, (int) i, (int) k, 0, +1.0f);
        }           

        tok_ids.push_back(token);
        
        if (i+1<n_ctx) {
            ggml_set_i32_nd(tokens_input, (int) (i + 1), (int) k, 0, 0, token);
            // sentence = llama_token_to_piece(lctx, token);
            // _INFO("%s,",sentence.c_str());
        }
    }
}

//Important!  this would update input & target_probs of model!
int64_t SampLoader::update_batch(int x,Fish* fish){    
    struct train_params_common *params = &(hOPT->train_params);
    // struct llama_context * lctx=(struct llama_context *)(hOPT->app_ctx);
    
    struct ggml_tensor *tokens_input=hOPT->gang->Input();
    assert(tokens_input!=nullptr);
    struct ggml_tensor *target_probs=hOPT->hTargetProbs();
    int64_t samples_count = N4Train();
    // const TOKEN_ID    * train_data=tokens.data();
    size_t  k,n_train_data = nTokens(); // tokens.size();
    sample_separation_eos=!params->separate_with_eos;
    sample_separation_bos=!params->separate_with_bos;
    double t_Samp = 0,nrm=0,a;
    bool sample_random_offsets=params->sample_random_offsets;
    // GGML_ASSERT(samples_count > 0);
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(target_probs));
    int64_t n_vocab  = target_probs->ne[0],nSampInBatch = fish->hparams.n_batch();  // tokens_input->ne[1];  //'ld0,ld1,ld2,ld3;
    n_ctx = fish->hparams.n_ctx();            //    tokens_input->ne[0];
    GGML_ASSERT(n_vocab  == target_probs->ne[0]);
    GGML_ASSERT(n_ctx == target_probs->ne[1]);
    GGML_ASSERT(nSampInBatch  == target_probs->ne[2]);
    GST_TIC(T0);
    
    ggml_set_f32(target_probs, 0.0f);
    // bos = llama_token_bos(llama_get_model(lctx));
    // eos = llama_token_eos(llama_get_model(lctx));
    /*std::string sBos = llama_token_to_piece(lctx, bos),sEos = llama_token_to_piece(lctx, eos);*/
    bool isLog = false;
    if(isLog) _INFO("BATCH_%ld ",next_sample);   //, samples_count,n_train_data);nSampe=(%ld/%ld)
    // assert(target_probs->type == GGML_TYPE_F32);
    hSAMP samp = nullptr;

    GST_TIC(tic);
    for (k=0; k<nSampInBatch; ++k) { 
        if(batch_sample=="stacking"){
            samp->jump++;
        }else{            
            samp = SampAt((next_sample + k) % samples_count);  
        }
        // LLAMA_LOG_INFO("%s: sample_idx=%zu sample=%zu\n", __func__, sample_idx, sample);  
        if(GST_TOC(tic)>20){        //for long-time data-update
            _INFO("\r[%s] k=%d(%d) T=%.3g ...",__func__,k,nSampInBatch,GST_TOC(tic));   
            tic = Clock::now( );
        }
       
        Samp2Batch(k,samp,tokens_input,target_probs,*params);   //refresh tok_ids
        if(isLog && k<6 && batch_sample!="stacking"){
            sentence = lama->T2STR(tok_ids,0x0);     //llama_token_to_piece(lctx, tok_ids[0]);
            _INFO(" (%ld,%d)@\"%s\"",samp->pos,samp->jump,sentence.c_str());     //sample_size
        }

        if(batch_sample!="stacking"){ 
            // nrm = fish->wikis[0]->InductLogits(k,tok_ids,exLogits,target_probs,-1); 
            for(auto wiki : fish->wikis){
                nrm = wiki->InductLogits(k,tok_ids,nullptr,target_probs,-1); 
            }
        }        
    }
    t_Samp = GST_TOC(tic);
    if(isLog) _INFO("\tT=%g\n",GST_TOC(T0));
    if(batch_sample=="stacking"){
        CHILD_0909_WIKIS assert(0);
        /*if(fish->wiki->isInduct()){
            GST_TIC(tic);  
            samp->Refresh(this,lctx,tok_ids,0x0);          //refresh tok_ids               
            nrm = fish->wiki->InductLogits(nSampInBatch,tok_ids,exLogits,target_probs,0x0);           
            _INFO("\t stacking@%d\"%.*s...\" nrm=%g\tT=%.4gs\t\n",samp->pos,64,samp->desc.c_str(),nrm,GST_TOC(tic));             
        }*/
    }
    
    next_sample += nSampInBatch;
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

DataTokenSet::DataTokenSet(ConsiceDict *hD) : hDict(hD)   {
    nVocab = hDict->n_vocab;
    assert(nVocab>0);
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
    
    _INFO("%s %s@%s...",__func__,isSave?"save@":"load@",path.c_str());
    _CHECK( S.Serial(n_vocab,isSave,flag) );
    // _CHECK( S.Serial(tokens,isSave,flag) );
    // _CHECK( S.Serial(n_unique_tokens,isSave,flag) );
    _CHECK( S.Serial(shuffle_samples_hash,isSave,flag) );
    _CHECK( S.Serial(hOPT->train_params.seed,isSave,flag) );

    _CHECK( S.Serial(batch_sample,isSave,flag) );
    // _CHECK( S.Serial(ids,isSave,flag) );
    bool bRet = S.Serial_Vector<SAMP,SAMP>(all_samps,isSave,flag);
    _CHECK(bRet);
    if(all_samps.size()==0)
        return false;
    size_t nT = nTokens();
    _INFO("\r%s %s@\"%s\" ... OK. \r\n\tnSamp=%ld @[Datasets(nToken=%ld  hash=%llX]\n",__func__,isSave?"save":"load",path.c_str(),
        all_samps.size(),nT,shuffle_samples_hash);
    for(auto samp : all_samps){
        // if(id>=nT){
        //     _INFO("\t\tInvalid id(%ld) > nTokens=%d\n", id,nT);  
        //     return false;
        // }
    }
    if(isSave){
        
    }else{
        size_t nSample = all_samps.size();
        num_batches = nSample/hOPT->train_params.n_batch;
        num_batches = nSample==0 ? 0 : max(num_batches,1); 
        _INFO("\t nBatch in each epoch=%d\n", num_batches); 
    }
    if(type==TYPE::DT_TRAIN){
        /*assert(train!=nullptr);   */
        if(isSave)    {

        }else{
            shuffle_rng_state_current = mt19937_seed_to_state(hOPT->train_params.seed);
            shuffle_sample_count = all_samps.size();
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

void SampLoader::Init(NLP_AutoRegressive *g_,int flag) {
    lama = g_;      //need dict&tokenset but is nullptr now
    isTarget_1 = g_->hparams.is( {"model","target"},string("OneHot") );
}

void SampLoader::Prepare(Optimizer *hOPT_,int flag){
    hOPT = hOPT_;           assert(hOPT_!=nullptr);
    assert(lama!=nullptr);
    tokens = lama->hTokenset;
    hDict = lama->hDict;        
    assert(tokens!=nullptr && hDict!=nullptr);  
    // bos = hDict->bos;
    // eos = hDict->eos;
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
void SampLoader::SetSamples(int nV,hDataToken hDT,std::vector<size_t>& samp_0,std::vector<size_t>& samp_L,
    bool isTrain,CLI_params& hp_,int flag)  {
    hparams = hp_;
    batch_sample = hparams.batch_sample;

    double rSplit = 1.0-hparams.rSplit;
    n_vocab = nV;    //hparams.n_vocab;
    tokens = hDT;
    size_t nSample = samp_0.size(),pick=(size_t)(nSample*rSplit),i;
    //    assert(samp_begin.size() == samp_size.size());   
    if(isTrain){
        for(i=0;i<pick;i++){
            all_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }
          
        Shuffle( );
    } else if(pick<nSample) {
        for(i=pick;i<nSample;i++){
            all_samps.push_back(new SAMP(samp_0[i],samp_L[i]));
        }

        if(1){  //too many batch in eval-set, so just random pick
            Shuffle( );
        }else{
            assert(0);
            
        }
    }
    nSample = all_samps.size();
    num_batches = nSample/hparams.common.n_batch;
    num_batches = nSample==0 ? 0 : max(num_batches,1);
    _INFO("%s@[%s]: tokens=%zu nSamp=%d nBach=%d\n", __func__,isTrain?"train":"eval", nTokens(),all_samps.size(),num_batches); 
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
    size_t count = all_samps.size(),i,j,k,jj,pick,seed,nPick=16,nLeft;
    size_t nSampInBatch=hparams.n_batch(),nVocab=tokens->nVocab,ctx=hparams.n_ctx(),tib=hparams.nTokenInBatch();
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
        cur = all_samps[seed];
        step++;
        double rEx = cur->UpdateTag(tokens,stp,step,true),r,rBest=FLT_MAX;        
        for(j=i*nSampInBatch+1;j<(i+1)*nSampInBatch;j++)  {
            if(0)   {   //10001/16/3 rDup=0.415(0.568)=>rDup=0.347(0.453)        10001/16/32 rDup=0.704(0.738)=>0.625(0.649)
                nLeft = count-j;        
                for(k=0,rBest=0;k<nPick;k++){
                    if(isRepeated)  {
                        jj = rng()%count;
                    }else       {
                        jj = j+rng()%nLeft;         assert(jj>=j && jj<count);
                    }
                    next = all_samps[ids[jj]];
                    r = next->UpdateTag(tokens,stp,step,false);      
                    if(r>rBest){
                        rBest = r;  pick = jj;
                    }
                }                
                std::swap(ids[j],ids[pick]);
            }
            next = all_samps[ids[j]];
            r = next->UpdateTag(tokens,stp,step,true);  
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

void SampLoader::Shuffle(int flag)  {
    size_t count = all_samps.size(),i;
    assert(count>0);
    struct train_params_common& train_params = hOPT->train_params;
    //  hash_combine(samples_begin,samples_size[i],sample_count
        
    const bool changed_train_data = false;  //(shuffle_samples_hash != hOPT->shuffle_samples_hash) || (train->shuffle_sample_count != all_samps.size());
    if (changed_train_data) {
        _INFO("%s: train data seems to have changed. restarting shuffled epoch.\n", __func__);
    }
    if (train_params.force_reshuffle) {
        _INFO("%s: forced reshuffling of data. restarting with newly shuffled epoch.\n", __func__);
    }
    if ((shuffle_rng_state_current == "") || changed_train_data || train_params.force_reshuffle) {
        shuffle_rng_state_current = mt19937_seed_to_state(train_params.seed);
        shuffle_sample_count = all_samps.size();
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

    TopoOrder(ids,rng);     // May better, need more testing

    for(i=0; i<count; ++i){
        tSamps[i] = all_samps[ids[i]];
    }
    all_samps = tSamps;
    
    shuffle_samples_hash = SAMP::HASH(fp_data.c_str(),all_samps);
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

int ConsiceDict::stream2token(void *hLLM,const char*txt,int txt_len,std::vector<TOKEN_ID>& btch,int flag){
    llama_model *lam_ = static_cast<llama_model *>(hLLM);
    assert(lam_!=nullptr);
        //  would call llama_tokenize_internal
    int n_tokens = llama_tokenize( lam_, txt,txt_len,btch.data(),(int) btch.size(),false, false);
    return n_tokens;
}

bool DataTokenSet::Load(struct CLI_params& hparams,void *hLLM,int flag){
    GST_TIC(tic);
    string ssf = hparams.serial_path+".tokenset";     
    if( Serialize(ssf,false) ){
        
    }else{
        fpath = hparams.fp_train_data.c_str();
        assert( std::filesystem::exists(fpath) );
        // llama_model *lam_ = static_cast<llama_model *>(hLLM);
        // assert(lam_!=nullptr);
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
            int n_tokens = hDict->stream2token(hLLM,buf+cur,len,btch,flag);
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
    nTokens = tokens.size();
    _INFO("\r[Load&Token]: @'%s' fsize=%.3g(M) nTokens=%.3g(M) nUnique=%ld T=%.3g(s)\t\t\t\n", 
        ssf.c_str(),fsize/1.0e6,nTokens/1.0e6,nUnique,GST_TOC(tic));
    
    return true;
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