/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "TokenSet.hpp"

#include "../Manifold/Optimizer.hpp"
#include "../Manifold/gLLM.hpp"
#include "DataLoader.hpp"
#include "Dictionary.hpp"

PromptTokenset::PromptTokenset(JSON::const_iterator jit, hTokenizer hDict, int flag) : DataTokenSet(hDict) {
    auto k  = jit.key();
    auto v  = jit.value();
    sPrompt = v["prompt"];
    name    = sPrompt;
    // hDict->eos = 0;
}
PromptTokenset::PromptTokenset(const string& prompt, hTokenizer hDict, int flag) : DataTokenSet(hDict) {
    sPrompt = prompt;
    name    = sPrompt;
}

Tokenset_HellaSwag::Tokenset_HellaSwag(JSON::const_iterator jit, hTokenizer hDict, int flag) : GlobTokenset(jit, hDict, flag) {
    name = "HellaSwag";
    // rStepOfEval = 0.0;  //  no sample on evaluate
    auto k    = jit.key();
    auto v    = jit.value();
    rSampling = 0;
    rSampling = jKV(v, {"samp"}, rSampling);
    assert(rSampling > 0.0);
    int nFile = shard_paths.size();
    assert(nFile == 1);
}

GlobTokenset::GlobTokenset(JSON::const_iterator jit, hTokenizer hDict, int flag) : DataTokenSet(hDict) {
    header_bytes      = K_SHARD_HEADER_SIZE * sizeof(int);
    int num_processes = 1, process_rank = 0;
    B = hDict->config.n_batch(), T = hDict->config.n_ctx();
    total_batch_size   = num_processes * (B * T);
    local_batch_offset = process_rank * B * T;

    auto k       = jit.key();
    auto v       = jit.value();
    glob_pattern = v["glob"];
    name         = jKV(v, {"name"}, name);
    eval_every   = jKV(v, {"eval-every"}, eval_every);
    // rSampling    = 0.01;
    rSampling = jKV(v, {"samp"}, rSampling);
    if (v.find("most") == v.end())
        nMostShard = 100000;
    else
        nMostShard = jKV(v, {"most"}, nMostShard);
    if (nMostShard == 0)  // in some case, create a blank dataset
        return;
}

bool GlobTokenset::Init(int flag) {
    glob_t glob_result;
    int glob_status = glob(glob_pattern.c_str(), 0, NULL, &glob_result);
    if (glob_status != 0) {
        _ERROR("%s Error: glob failed to find \"%s\"\n", __func__, glob_pattern.c_str());
        K_EXIT(KOIFISH_LOAD_TOKENSET_GLOB);
    }
    if (glob_result.gl_pathc == 0) {
        _ERROR("%s No files found matching the pattern: %s\n", __func__, glob_pattern.c_str());
        K_EXIT(KOIFISH_LOAD_TOKENSET_GLOB);
    }
    int nFile = 0;
    nMostTok  = 0;
    for (int id = 0; id < glob_result.gl_pathc; id++) {
        string sPath = glob_result.gl_pathv[id];
        shard_paths.push_back(sPath);
        int64_t shard_ntok = OnShardFile(id);
        nFile++;
        // assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        nMostTok += shard_ntok;
        if (nMostShard > 0 && shard_paths.size() == nMostShard)
            break;
    }
    double nG = nMostTok / 1.0e9;
    if (nMostTok == 0) {
        assert(0 && "GlobTokenset::Failed to load tokens");
    } else
        _INFO("[%s] %s find %.8gG tokens @\"%s\"(%d files)\n", __func__, name.c_str(), glob_pattern.c_str(), nG, nFile);
    return true;
}

size_t DataTokenSet::nBatch(int flag) {
    size_t nSample  = shard_samps.size();  //  shard_samps init @Shard2Sample
    size_t nBatches = nSample / hDict->config.n_batch();
    nBatches        = nSample == 0 ? 0 : max(nBatches, (size_t)1);
    return nBatches;
}

bool GlobTokenset::LoadNextShard(SampLoader* hLoader, int flag) {
    if (shard_index > 0)
        _INFO("-------- End of shard_%d@\"%s\"-------- \n", shard_index, shard_paths[shard_index - 1].c_str());
    if (shard_index == shard_paths.size()) {
        hLoader->NextEpoch();
        shard_index = 0;
    }

    if (!hLoader->isLastShard) {
        size_t iRet = OnShardFile(shard_index++, true);
        if (iRet == size_t(-1))
            return false;
    }

    return true;
}

bool GlobTokenset::fp2Tokens(int flag) {
    try {
        int nT = (szFile - header_bytes) / bpToken;
        assert(nT == nShardToks);
        assert(fpShard != nullptr);
        // tokens.resize(nT);
        fseekCheck(fpShard, (long)header_bytes, SEEK_SET);
        hBITARR tmp = new BIT_8[nT * bpToken];  // TOKEN may 8/16/32 bit
        if (fread(tmp, bpToken, nT, fpShard) != nT) {
            _ERROR("file size is not as expected\n");
            return 0x0;
        }
        switch (bpToken) {
            case 2: {
                uint16_t* tmp16 = (uint16_t*)tmp;
                tokens.assign(tmp16, tmp16 + nT);
            } break;
            case 4: {
                int32_t* tmp32 = (int32_t*)tmp;
                tokens.assign(tmp32, tmp32 + nT);
            } break;
            default:
                assert(0);
                break;
        }
        delete[] tmp;
        return true;
    } catch (...) {
        return false;
    }
}
// shard type != token type
bool GlobTokenset::Shard2Sample(int id, int flag) {
    try {
        fp2Tokens(flag);
        /*int nT = (szFile - header_bytes) / bpToken;
        assert(nT == nShardToks);
        // tokens.resize(nT);
        fseekCheck(fpShard, (long)header_bytes, SEEK_SET);
        hBITARR tmp = new BIT_8[nT * bpToken];  // TOKEN may 8/16/32 bit
        if (fread(tmp, bpToken, nT, fpShard) != nT) {
            _ERROR("file size is not as expected\n");
            return 0x0;
        }
        switch (bpToken) {
            case 2: {
                uint16_t* tmp16 = (uint16_t*)tmp;
                tokens.assign(tmp16, tmp16 + nT);
            } break;
            case 4: {
                int32_t* tmp32 = (int32_t*)tmp;
                tokens.assign(tmp32, tmp32 + nT);
            } break;
            default:
                assert(0);
                break;
        }
        delete[] tmp;*/
        // InitSamps
        int n_ctx     = hDict->config.n_ctx(), len;
        size_t nToken = tokens.size(), nFirst = std::min((size_t)n_ctx, nToken), step = 1, n0 = shard_samps.size();
        // samples_size.push_back(nFirst);
        size_t end = (nToken >= n_ctx) ? (nToken - n_ctx) : 0;
        // if (end > 10 * 1024 * 1024) {
        step = n_ctx;
        // }
        float rSample = hDict->config.common.rSubSample;
        if (rSample > 0 && rSample < 1)
            step /= rSample;
        for (size_t sample_begin = 0; sample_begin < end; sample_begin += step) {
            len = std::min(n_ctx, (int)(end - sample_begin));
            if (len != n_ctx)  // to simplifi collate function
                continue;
            shard_samps.push_back(new SAMP(sample_begin, len));
        }
        n0 = shard_samps.size();
        if (n0 == 0) {
            assert(0);
        } else {
            _INFO("\n[shard \"%s\"]: %ld(tokens)=>%ld(samps) nBach=%d step=[%ld:%ld:%ld]\n", name.c_str(), nToken, shard_samps.size(), nBatch(), 0, end, step);
        }
        return true;
    } catch (...) {
        return false;
    }
}

size_t GlobTokenset::OnShardFile(int id0, bool load, int flag) {
    int id = id0;
    id     = id0 % shard_paths.size();
    if (isShuffle) {
    }
    if (id >= shard_paths.size()) {
        return -1;
    }
    size_t expected_file_size = 0x0;
    if (!VERIFY_DIR_EXIST(shard_paths[id])) {
        K_EXIT(KOIFISH_LOAD_SHARD_NULL);
    }
    const char* filename = shard_paths[id].c_str();
    if (GetShardInfo(id, flag)) {
    } else {  //  get meta(nShardToks) from head
        assert(fpShard == NULL);
        fpShard = fopen(filename, "rb");
        if (fpShard == NULL) {
            K_EXIT(KOIFISH_LOAD_SHARD_NULL);
        }
        // validate the header
        uint32_t header[K_SHARD_HEADER_SIZE];
        freadCheck(header, sizeof(int), K_SHARD_HEADER_SIZE, fpShard);
        if (header[1] != 1) {
            _ERROR("Bad version<%d> of data file\n", header[1]);
            K_EXIT(KOIFISH_LOAD_TOKENFILE_HEADER);
        }
        fseekCheck(fpShard, 0, SEEK_END);  // seek to end of file
        szFile     = ftell(fpShard);       // read the offset, i.e. file size
        nShardToks = 0;
        bpToken    = -1;
        switch (header[0]) {  //
            case 20240522:    //  hellaswag dataset
                tpSample = HellaSwag;
                bpToken  = 2;
                // int nMostCompletion =4,can_fit_examples = (int) (B / nMostCompletion);
                longest_example_bytes = header[3];
                nShardSamples         = header[2];
                nShardToks            = B * T;
                // label = (int*)mallocCheck(can_fit_examples * sizeof(int));
                break;
            case 20240520:  //
            case 20250520:  // qwen2.5:
                nShardToks = header[2];
                assert(nShardToks > 0);
                bpToken            = header[0] == 20240520 ? 2 : header[3];
                expected_file_size = K_SHARD_HEADER_SIZE * sizeof(int) + nShardToks * bpToken;
                if (szFile != expected_file_size) {
                    _ERROR("file size(%ld) is not as expected(%ld) @%s\n", expected_file_size, szFile, filename);
                    K_EXIT(EXIT_FAILURE);
                }
                // -1  due to us taking B*T+1 tokens but moving by B*T tokens
                nShardSamples = (nShardToks - 1) / total_batch_size;
                break;
            case 20251218:  // qwen3:
                nShardToks = header[2];
                assert(nShardToks > 0);
                bpToken            = /*header[0] == 20251218 ? 2 :*/ header[3];
                expected_file_size = K_SHARD_HEADER_SIZE * sizeof(int) + nShardToks * bpToken;
                if (szFile != expected_file_size) {
                    _ERROR("file size(%ld) is not as expected(%ld) @%s\n", expected_file_size, szFile, filename);
                    K_EXIT(EXIT_FAILURE);
                }
                // -1  due to us taking B*T+1 tokens but moving by B*T tokens
                nShardSamples = (nShardToks - 1) / total_batch_size;
                break;
            default:
                _ERROR("Bad magic<%d> in the data file @\"%s\"\n", header[0], filename);
                K_EXIT(KOIFISH_LOAD_TOKENFILE_HEADER);
                break;
        }
    }
    if (load) {
        if (Shard2Sample(0x0)) {
            _INFO("[shard \"%s\"_%d]@\"%s\": tokens=%.3g(M) nShardSamples=%ld(%ld) \n", name.c_str(), id + 1, filename, nShardToks / 1.0e6, nShardSamples,
                  shard_samps.size());
        } else {
            _WARN("[shard \"%s\"_%d]@\"%s\": tokens=%.3g(M) nShardSamples=%ld(%ld) \n", name.c_str(), id + 1, filename, nShardToks / 1.0e6, nShardSamples,
                  shard_samps.size());
        }
    }
    if (tpSample == RANDOM_GENERATE) {
        /*
        if(load){
            int nT = (file_size_bytes-header_bytes)/bpToken;
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

    if (load) {
    }
    if (fpShard != NULL) {
        fcloseCheck(fpShard);
        fpShard = NULL;
    }
    return nShardToks;
}

DataTokenSet::DataTokenSet(hTokenizer hD) : hDict(hD) {
    assert(hDict->isValid(true));
    nVocab = hDict->nVocab();
    assert(nVocab > 0);
}
DataTokenSet::~DataTokenSet() {
    for (auto hSamp : shard_samps) delete hSamp;
    shard_samps.empty();
}

TOKEN_ID DataTokenSet::At(size_t pos) {
    assert(pos < tokens.size());
    int32_t token = CLAMP(tokens[pos], 0, (nVocab - 1));
    return token;
}

bool DataTokenSet::Serialize(const std::string& path, bool isSave, int flag) {
    try {
        FSerial S(path, isSave, flag);
        if (!S.isValid())
            return false;

        _INFO("%s %s@%s...", __func__, isSave ? "save@" : "load@", path.c_str());
        CHECK_(S.Serial(nVocab, isSave, flag));
        CHECK_(S.Serial(nUnique, isSave, flag));
        CHECK_(S.Serial(fsize, isSave, flag));
        CHECK_(S.Serial(nDialect, isSave, flag));
        CHECK_(S.Serial(tokens, isSave, flag));
        if (nDialect > 0) {
            CHECK_(S.Serial(dialect, isSave, flag));
            CHECK_(S.Serial(mapT2T, isSave, flag));
        }
        if (isSave) {
        } else {
            if (tokens.size() == 0)
                return false;
            for (auto token : tokens) {
                if (token < 0 || token >= nVocab) {
                    return false;
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

// for each batch @Head4Token::cuFlow
float SampLoader::UpdateII(float mean_loss, int flag) {
    Fish* hFish = dolphin;
    // float alpha4g = 1.0f, mean_loss = 0.0f, logprob = 0.f;
    Head4Token* cls   = hFish->GetNeuron<Head4Token>("Head4Token", 0);
    TokenEmbed* embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    /*int i;
    for (logprob = 0, i = 0; i < B * T; i++) {
        mean_loss += hostLoss[i];
        logprob += -hostLoss[i];
    }
    mean_loss /= B * T;

    float ppl = exp(-logprob / (B * T));  //  just exp(mean_loss)*/
    float ppl = exp(mean_loss);

    iiLoss.Add(mean_loss);
    iiPPL.Add(ppl);
    iiLoss.Stat();
    iiPPL.Stat();

    // if (DEBUG.eval_Generate > 0) {
    //     hChater gopt = hFish->GetGenerator();
    //     gopt->SampleOnBatch(embed->hBatch, hostLoss, B, T, this);  // 1479
    // }
    return mean_loss;
}

double DataTokenSet::LossOnResult(hSampLoader hLoader, Head4Token* cls, int flag) {
    assert(cls != nullptr);
    double mean_loss = 0, sum = 0, ss = 0, ppl = 0, sigma = 0, logprob = 0;
    hBATCH hBatch = hLoader->hBatch;
    int *mask = hLoader->hBatch->mask32, n = 0, nzLoss = cls->nzLoss;
    int nVocab = cls->nCls, ldP = cls->padded_nCls;
    float* loss      = cls->hostLoss;
    TOKEN_ID* labels = TO<TOKEN_ID>(hLoader->hostLabel);  // hLoader->hBatch->hostToken
    float* logits    = nullptr;

    for (int i = 0; i < nzLoss; i++) {
        if (BIT_TEST(mask[i], MASK_FLAG::F_IGNORE_LOSS)) {
            assert(loss[i] == 0.0);
            continue;
        }
        if (!isValidF(loss[i])) {
            if (DEBUG.dump_LossDetail)
                hLoader->hBatch->DumpX(labels, cls->hostLoss);  
            // TOKENS tokens(hBatch->host_toks, hBatch->host_toks + nzLoss);
            // DumpTokens(hDict, tokens, -1);
            hSAMP samp = hLoader->cur_samps[i / ldP];
            samp->Dump(hDict, hLoader->GetTokens(), 0x0, "Invalid LossOnResult");
            TOKEN_ID spot = hBatch->host_toks[i];
            _ERROR("spot=%s(%d) loss=%g(%d)", hDict->Decode({spot}).c_str(), spot, loss[i], i);
            K_EXIT_NOW(KOIFISH_INVALID_LOSS);
        }

        mean_loss += loss[i];
        n++;

        /*if (0) {  //  Debug_PPL   t=921 -10.825027195036846   [1.30967237e-09,...,2.09547579e-09]
            if (logits == nullptr)
                logits = cls->fLogits();
            logprob = log(P_softmax(tokens[i], logits + ldP * i, nVocab));
            assert(fabs(logprob + loss[i]) < 1.0e-5 * fabs(logprob));
        } else*/
        {
            logprob = -loss[i];
        }
        sum += logprob;
        ss += logprob * logprob;
        ppl   = exp(-sum / n);
        sigma = ppl * sqrt((ss - sum * sum / n) / n / n);
    }
    assert(n > 0);
    mean_loss /= n;
    hLoader->UpdateII(mean_loss, 0x0);
    return mean_loss;
}

bool Tokenset_HellaSwag::Shard2Sample(int id, int flag) {
    try {
        size_t B = hDict->config.n_batch(), T = hDict->config.n_ctx();
        int batch_dim_offset, nComplete       = 0;
        int can_fit_examples = (int)(B / nMostCompletion), examples_per_process = nShardSamples, end_example_index = examples_per_process,
            start_example_index = 0;
        uint16_t* buffer16      = new uint16_t[longest_example_bytes];
        assert(can_fit_examples > 0);
        tokens.resize(nShardSamples * nMostCompletion * T);
        masks.resize(tokens.size());
        int num_batches = CEIL_DIV(examples_per_process, can_fit_examples), id;
        // now seek through the file to the start of that example
        // utilize <EXAMPLE_BYTES> for efficiency
        size_t header_bytes = K_SHARD_HEADER_SIZE * sizeof(int), sz = header_bytes;
        uint16_t example_header[3];  //<START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        fseekCheck(fpShard, (int)header_bytes, SEEK_SET);
        for (id = start_example_index; id < end_example_index; id++) {
            batch_dim_offset = id * nMostCompletion;
            freadCheck(&example_header[0], sizeof(uint16_t), 3, fpShard);
            assert(example_header[0] == 65535);
            assert(example_header[2] == id);
            sz += example_header[1];
            // skip to the next example, keeping in mind that we already read the header
            size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
            assert(remaining_bytes > 0);
            // fseekCheck(fpShard, (int) remaining_bytes, SEEK_CUR);
            freadCheck(buffer16, sizeof(char), remaining_bytes, fpShard);

            int l = (int)buffer16[0], nComplete = (int)buffer16[1], context_length = (int)buffer16[2];
            assert(l >= 0 && l < nMostCompletion);  // we expect the label to be in [0, 4) for right now
            // samp->label = l;
            hQuestion question = new QUESTION(l, batch_dim_offset, batch_dim_offset + nComplete);
            assert(nComplete == nMostCompletion);  // we expect 4 completions for now
            // assert(batch_dim_offset + c <= B); // we expect to fit in the batch
            assert(context_length > 0 && context_length < T);            // context is non-empty and up to T
            uint16_t* context_tokens_start = (uint16_t*)(buffer16 + 3);  // where the tokens start
            for (int b = 0; b < nComplete; b++) {
                for (int i = 0; i < context_length; i++) {
                    int boff             = batch_dim_offset + b;
                    int tok_cur          = (int)context_tokens_start[i];
                    tokens[boff * T + i] = tok_cur;
                }
            }
            // process the completions, insert them in their row, right after the (shared) context
            uint16_t* completions_iter = buffer16 + 3 + context_length;
            for (int c = 0; c < nComplete; c++) {
                int coff                          = batch_dim_offset + c;
                int completion_length             = (int)completions_iter[0];
                uint16_t* completion_tokens_start = completions_iter + 1;
                assert(completion_length > 0 && context_length + completion_length < T);  // things fit?
                for (int i = 0; i < completion_length; i++) {
                    int tok_cur                           = (int)completion_tokens_start[i];
                    tokens[coff * T + context_length + i] = tok_cur;  // at inputs, the completions simply follow the context
                    // at targets things start to get tricky
                    // we expect the last context token to predict the first completion token
                    // and then onwards from there.
                    // targets[coff * T + context_length + i - 1] = tok_cur;
                    // and at these positions, we want to set mask=1, because these are the
                    // positions where we want to average the loss, in each row, to determine
                    // its overall probability of following the context.
                    masks[coff * T + context_length + i - 1] = 1;
                }
                completions_iter += 1 + completion_length;  // move to the next completion
                hSAMP samp   = new SAMP(coff * T, T);
                samp->target = (void*)question;
                shard_samps.push_back(samp);
            }
            questions.push_back(question);
        }
        assert(sz == szFile);
        delete[] buffer16;
        return true;
    } catch (...) {
        return false;
    }
}

/*
    A lite version of Optimizer::Evaluate
 */
double SampLoader::Evaluate(DL_BATCH_UPATE tpBatch, int flag) {
    assert(hOPT != nullptr);
    Fish* hFish  = dolphin;
    RLS_BP* hRLS = hFish->GetScheduler<RLS_BP>();
    double tic = GST_ms(), tps, tRemain = 0.0, tpi = 0, relax = 0.9, dt, tCur, tLast;
    int i, nB = 0, step = StepOfEvaluate(), iter = hOPT->GetITER(), nMost = CEIL_DIV(num_batches, step);
    switch (tpBatch) {
        case BATCHofEMBED:
            nMost = 1;
            break;
        default:
            break;
    }
    // double a, a0 = DBL_MAX, a1 = -DBL_MAX, mean_loss = 0, ss = 0, sigma, sum = 0;
    hGensor target_label = hFish->Target();
    Head4Token* cls      = hFish->GetNeuron<Head4Token>("Head4Token", 0);
    cls->hLoader         = shared_from_this();
    TokenEmbed* embed    = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
    // hSAMP samp = nullptr;
    next_sample = 0;  // fix this to keep same acc on each experiment
    nEvalTokens = 0;
    tLast       = GST_ms();
    ClearII();
    for (int i = 0; i < nMost; i++) {
        if (tpBatch == SAMPLEofSHARD)
            CollateBatch(min(i * step, num_batches), hFish);
        // samp = cur_samps[0];
        embed->hBatch = GetCurBatch();
        hFish->ForwardOnRLS(iter, 0x0);

        nEvalTokens += embed->hBatch->nFillTokens(), nB++;
        tCur = GST_ms(), dt = tCur - tLast, tLast = tCur;
        tpi = tpi * (1.0 - relax) + dt * relax, tRemain = (nMost - i) * tpi;  //  ms
        if (i % 10 == 0 && tRemain > 60 * 1000) {
            _INFO("\r\t%d/%d a=[%.3g,%.3g] %.4gk/s ...\t", i, nMost, iiLoss.a0, iiLoss.a1, nEvalTokens / (tCur - tic));
            _TIME_INFO("remain=", tRemain), _INFO("        ");
        }
        if ((tCur - tic) / 1.0e3 > DEBUG.Time_most)
            break;
        // if (DEBUG.eval_Generate > 0) {
        //     hChater gopt = hFish->GetGenerator();
        //     gopt->SampleOnBatch(embed->hBatch);  // 1479
        // }
    }

    // _INFO("\n\t");
    SUM::tEval_1 = (GST_ms() - tic) / 1.0e3;
    switch (hFish->phase) {
        case P_GENERATE:
            break;
        case P_EVAL_:
            break;
        case P_TRAIN:
            UpdateStepInfos(iiLoss.average, nB);
            break;
    }
    // if (!hFish->isLocalInfer)
    //     UpdateStepInfos(iiLoss.average, nB);

    switch (tpBatch) {
        case BATCHofEMBED:
            nMost = 1;
            break;
        default:
            tps = nEvalTokens / SUM::tEval_1 / 1.0e3;
            _INFO("\t#%g±%.4f tps=%.3gK(%gM) a=[%g,%g] T=%g(sec)\n", "", iiLoss.average, iiLoss.sigma, tps, nEvalTokens / 1.0e6, iiLoss.a0, iiLoss.a1,
                  SUM::tEval_1);
            iiLoss.SaveToCSV(name + "_loss.csv", 0x0);
            break;
    }

    return iiLoss.average;
}

void SampLoader::UpdateStepInfos(float mean_loss, int nB, int flag) {
    int iter = hOPT->GetITER(), nFuyou = dolphin->nFuyou(1);
    float last          = stepis.Last();  //[eval]   Loss@Evaluation=7.302641 T=0.232s ======
    float train_last    = hOPT->trainInfos().Last();
    bool isFirst        = stepis.steps.empty();
    StepInfos::STEP stp = StepInfos::STEP(mean_loss, iter, hOPT->train_epochs);
    stepis.Add(stp);
    if (isFirst) {
        _INFO(" Loss@\"%s\"=%.3f nFuyou=%d ", sTokenSet().c_str(), mean_loss, nFuyou);
        return;
    }

    float delta = last - mean_loss, best = stepis.Best(), ee = 0;
    bool isOverfit = delta < 0 && abs(mean_loss - best) > best / 10;
    // if(isOverfit)   {
    //     _INFO(" !OVERFIT! ");
    // }
    double a = nB * hOPT->TrainParams().nTokenInBatch() / 1.0e6;
    _INFO(" Loss@\"%s\"=%.3f(%.2g) nBranch=%d nToken=%.3gM best=%.4f(%d) E2T=%.3g T=%g(%.3g)s x=%.3g\n", sTokenSet().c_str(), mean_loss, nFuyou, delta, a, best,
          stepis.best_id, mean_loss - train_last, SUM::tEval_1, SUM::tLoadData / 1000.0, ee);  //

    // if (wLog == nullptr) {
    // } else
    //     _INFO("\t Loss@Evaluation=%f delta=%g(%.5g) T=%gs\n", mean_loss, delta_max, delta_ / nB, GST_TOC(tic));
    string sX = "_loss=" + std::to_string(mean_loss);

    stepis.SaveToCSV("_info_.csv");
}

double Tokenset_HellaSwag::LossOnResult(hSampLoader hLoader, Head4Token* cls, int flag) {
    assert(cls != nullptr);
    double mean_loss = 0, a = 0, a_0 = DBL_MAX;
    // auto sp = hLoader->hostBatch->shape;
    // auto config = hFish->config;
    int nB = hLoader->B, nT = hLoader->T;  //,nB=sp[1],nT=sp[0]

    int *mask = nullptr, n = 0, nzLoss = cls->nzLoss, i = 0, t, b = 0, q, no = -1, nOK = 0, nQ = 0, s = 0;
    assert(nB % nMostCompletion == 0);

    float* loss = cls->hostLoss;
    TOKEN_ID token;
    quesInBatch.clear();
    while (s < hLoader->cur_samps.size()) {
        auto samp          = hLoader->cur_samps[s];
        hQuestion question = (hQuestion)(samp->target);
        quesInBatch.push_back(question);
        for (b = question->b0; b < question->b1; b++, s++) {
            samp = hLoader->cur_samps[s];
            assert(samp->target == question);
        }

        for (a_0 = DBL_MAX, no = -1, b = question->b0; b < question->b1; b++) {
            for (a = 0, n = 0, t = 0; t < nT; t++, i++) {
                if (!hLoader->isHostMask(i)) {
                    continue;
                }
                a += loss[i];
                n++;
            }
            assert(n > 0);
            a /= n;
            if (a < a_0) {
                a_0 = a;
                no  = b - question->b0;
            }
        }
        if (no == question->label) {
            nOK++;
        }
        nQ++;
    }
    assert(s == hLoader->cur_samps.size());
    assert(nQ == nB / nMostCompletion);
    mean_loss = nOK * 1.0 / nQ;
    return mean_loss;
}

Tokenset_JSONL::Tokenset_JSONL(JSON::const_iterator jit, hTokenizer hDict, const string& format_, int flag) : GlobTokenset(jit, hDict, flag) {
    name   = "Chat_JSONL";
    format = format_;
    // rStepOfEval = 0.0;  //  no sample on evaluate
    auto k          = jit.key();
    auto v          = jit.value();
    enable_thinking = hDict->config.chat_sampler.enable_thinking;
    assert(!enable_thinking && "Koifish don't support enable_thinking mode now!");
    // rSampling = jKV(v, {"samp"}, rSampling);
    std::vector<string> features;
    features = jKV_arr(v, {"x"}, features);
    if (G_Has_("multi_turn", features, false)) {
        multi_turn = true;
    }
    assert(rSampling == 1);
    assert(!glob_pattern.empty());
}

//  ChatML​ is a tokenization-friendly text formatthat encodes chat messages into a single string
std::string Tokenset_JSONL::toChatML(JSON& jMsg, int flag) {
    assert(jMsg.find("messages") != jMsg.end());
    /*{"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}*/
    //' <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of
    // France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe capital of France is Paris.<|im_end|>\n'
    //    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n\n\nThe
    //    capital of France is Paris.<|im_end|>\n"
    std::vector<ChatML_samp> lines;
    for (auto& msg : jMsg["messages"]) {
        std::string role    = msg["role"];
        std::string content = msg["content"];
        lines.push_back(ChatML_samp(role, content));
    }
    assert(hDict != nullptr);

    std::string result = hDict->config.chat_sampler.toChatML(lines);

    return result;
}

/*
    GetShardInfo would be called many times, for jsonl, only load json & get meta-info(then to sample) once
    too slow!
 */
bool Tokenset_JSONL::GetShardInfo_txt(int id, int flag) {
    std::string sPath;
    if (!messages.empty())
        return true;

    try {
        sPath = shard_paths[id];
        if (!VERIFY_DIR_EXIST(sPath)) {
            K_EXIT(KOIFISH_LOAD_SHARD_NULL);
        }
        JSON jData;
        std::ifstream file(sPath);
        file >> jData;
        size_t nMsg    = jData.size();
        int max_length = hDict->config.n_ctx(), nPad = 0, nDrop = 0;
        double tX, tSum, t0                                     = GST_us();
        for (auto& item : jData) {
            // _INFO("%s", item.dump().c_str());
            string msg;
            if (G_Aa(format, "OAI_message")) {
                assert(item.find("messages") != item.end());
                msg = toChatML(item);
            } else {
                // _INFO("%s\n", item.dump().c_str());
                assert(item.find("text") != item.end());
                msg = item["text"];
            }
            double t1   = GST_us();
            TOKENS curT = hDict->Encode(msg);
            tX += GST_us() - t1;
            if (curT.size() >= max_length) {  //  high-quality, clean data is far more valuable than noisy, truncated data.)
                nDrop++;
                continue;
            }
            nPad = max_length - curT.size();
            for (int i = curT.size(); i < max_length; i++) {
                curT.push_back(hDict->pad_id);
            }
            messages.push_back(msg);
            size_t begin = tokens.size();
            tokens.insert(tokens.end(), curT.begin(), curT.end());
            shard_samps.push_back(new SAMP(begin, curT.size(), nPad));
            if (messages.size() <= 2) {
                string msg_1 = hDict->Decode(curT);
                assert(msg_1.find(msg) == 0);
                DumpTokens(hDict, curT, -1);
            }
            if (messages.size() % 20 == 0) {
                tSum = (GST_us() - t0) / 1.0e6;
                _INFO("\r%ld(%ld)\tavg=%.5gs\tT=%.5g(%.5g) ...", messages.size(), tokens.size(), tSum / tokens.size(), tSum, tX / 1.0e6);
            }
        }
        if (nPad == 0 && tokens.size() >= max_length) {  // last-sample: training need extra 1 token as target
            shard_samps.pop_back();
            tokens.resize(tokens.size() - max_length);
        } else {
        }
        nShardSamples = shard_samps.size();
        nShardToks    = tokens.size();
        assert(nShardToks > 0);
        // messages=>tokens

    } catch (JSON::parse_error& e) {
        _ERROR("\r\n>>>>>> Tokenset_JSONL::OnShardFile @ %s!!! ERR=%s", sPath.c_str(), e.what());
        return false;
    } catch (...) {
        return false;
    }
    return nShardSamples > 0;
}

bool Tokenset_JSONL::GetShardInfo(int id, int flag) {
    std::filesystem::path path = shard_paths[id];
    if (!std::filesystem::exists(path)) {
        K_EXIT(KOIFISH_LOAD_SHARD_NULL);
    }
    isJsonTxt = !G_Aa(path.extension(), ".bin");
    if (isJsonTxt)
        return GetShardInfo_txt(id, flag);

    return isJsonTxt && nShardSamples > 0;
}

/**
 * '<|im_start|>system\nYou are a dog.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nFine<|im_end|>\n'
 *      1.  <|im_start|>systemis notstrictly required​ in ChatML / Qwen-style SFT!
 *
 *
 * */
bool Tokens2Samp_Chatml(hTokenizer hDict, const TOKENS& tokens, size_t& pos, ChatML_samp& meta, bool multi_turn, int flag) {
    bool think_closed = false;
    size_t answer_0 = 0, answer_1 = 0, nToken = tokens.size(), end;
    TOKEN_ID current = -1;
    do {
        while (++pos < nToken && tokens[pos] != hDict->assist_id);
        assert(tokens[pos] == hDict->assist_id);  //  Detect start of assistant turn: <|im_start|> followed by "assistant"
        while (++pos < nToken) {
            current = tokens[pos];
            if (current == hDict->id_think_close) {
                think_closed = true;
                TOKEN_ID a1 = tokens[pos + 1], a2 = tokens[pos + 2];  // tokenizer.decode(seq[pos+1]),tokenizer.decode(seq[pos+2])
                if (a1 == hDict->id_newline2)                         //"\n\n":    pos = pos+1
                    pos = pos + 1;
                if (a1 == hDict->id_newline && a2 == hDict->id_newline)
                    pos = pos + 2;
                answer_0 = pos + 1;
            } else if (current == hDict->id_im_end) {  // hDict->id_im_end
                answer_1     = pos;
                think_closed = false;
                end          = pos + 1;
                // TOKENS curT(tokens.begin() + answer_0, tokens.begin() + answer_1);
                // string ans = hDict->Decode(curT);
                meta.answers.push_back(std::make_tuple(answer_0 - meta.start, answer_1 - meta.start));
                break;
            } else {                 // Only enable loss for tokens after  inside assistant block
                if (think_closed) {  // Skip padding tokens
                    // a = tokenizer.decode(current)
                    if (current != hDict->pad_id) {
                        // labels[batch_idx, pos] = current
                        // true_answer += a
                    }
                }
            }
        }
        if (multi_turn && tokens[pos + 1] == meta.eoc) {
            pos++;
            break;
        }
    } while (current != meta.eoc && pos + 1 < nToken);

    if (!multi_turn)
        assert(meta.answers.size() <= 1);
    meta.end = pos + 1;
    assert(tokens[meta.end - 1] == meta.eoc);
    return true;
}

bool Tokenset_JSONL::Shard2Sample(int id, int flag) {
    // assert(hDict->isValid());
    int n_ctx = hDict->config.n_ctx(), len, pad_id = hDict->pad_id, nDrop = 0;
    int max_length = hDict->config.n_ctx();
    float rSample  = hDict->config.common.rSubSample;
    assert(!enable_thinking);
    if (isJsonTxt) {
        assert(shard_samps.size() > 0);  // in this trival case, all samps generated in GetShardInfo_txt
        _INFO("\n[shard \"%s\"]: %ld(tokens)=>%ld(samps) nBach=%d\n", name.c_str(), tokens.size(), shard_samps.size(), nBatch());
        return true;
    }
    try {
        fp2Tokens(flag);
        size_t nToken = tokens.size(), pos = -1;
        while (++pos < nToken) {
            assert(tokens[pos] == hDict->id_im_start);
            if (pos == 3376)
                DEBUG_HERE;
            ChatML_samp chatml(pos, multi_turn, multi_turn ? hDict->pad_id : hDict->id_im_end);
            if (!Tokens2Samp_Chatml(hDict, tokens, pos, chatml, multi_turn, flag)) {
                assert(0);
            }
            if (chatml.answers.size() == 0) {
                assert(0);  // so strange!
                continue;
            }
            TOKENS curT(tokens.begin() + chatml.start, tokens.begin() + chatml.end);
            // if (shard_samps.size() <= 8 && !chatml.answers.empty()) {
            //     DumpTokens(hDict, curT, -1);
            //     if (multi_turn) {
            //         for (auto [a, b] : chatml.answers) {
            //             string msg = hDict->Decode(TOKENS(curT.begin() + a, curT.begin() + b));
            //             _INFO("\"%s\"\t", msg.c_str());
            //         }
            //         _INFO("\n");
            //     }
            // }

            if (curT.size() > max_length) {
                nDrop++;
                continue;
            }
            int nPad = max_length - curT.size();
            for (int i = curT.size(); i < max_length; i++) {
                curT.push_back(hDict->pad_id);
            }
            hSAMP hSamp    = new SAMP(chatml.start, curT.size(), nPad);
            hSamp->answers = chatml.answers;
            if (shard_samps.size() <= 8 /*|| hSamp->pos == 3376*/)
                hSamp->Dump(hDict, tokens, 0x0);
            shard_samps.push_back(hSamp);

            // if (messages.size() % 20 == 0) {
            //     tSum = (GST_us() - t0) / 1.0e6;
            //     _INFO("\r%ld(%ld)\tavg=%.5gs\tT=%.5g(%.5g) ...", messages.size(), tokens.size(), tSum / tokens.size(), tSum, tX / 1.0e6);
            // }
        }

        /*
        if (rSample > 0 && rSample < 1)
            step /= rSample;
        for (size_t sample_begin = 0; sample_begin < end; sample_begin += step) {
            len = std::min(n_ctx, (int)(end - sample_begin));
            if (len != n_ctx)  // to simplifi collate function
                continue;
            shard_samps.push_back(new SAMP(sample_begin, len));
        }*/
        _INFO("\n[shard \"%s\"]: %ld(tokens)=>%ld(samps) nDrop=%d nBach=%d\n", name.c_str(), tokens.size(), shard_samps.size(), nDrop, nBatch());
        return true;
    } catch (...) {
        return false;
    }
}

void SAMP::Dump(hTokenizer hDict, const TOKENS& tokens, int type, const std::string& desc, int flag) {
    int nValidLen = len - pad_len;
    size_t start  = pos;
    if (type == 0x100)
        start = 0;
    assert(start + nValidLen <= tokens.size());
    TOKENS samp_tokens(tokens.begin() + start, tokens.begin() + start + nValidLen);
    DumpTokens(hDict, samp_tokens, 0, flag);
    _INFO("\n ------ range=[%lld:%lld pad=%lld] turn=%lld", pos, pos + nValidLen, pad_len, answers.size());
    if (answers.size() > 0) {  // multi_turn
        for (auto [a, b] : answers) {
            assert(a >= 0 && b < nValidLen);
            string msg = hDict->Decode(TOKENS(samp_tokens.begin() + a, samp_tokens.begin() + b));
            _INFO("\n\t[%lld:%lld]=\"%s\"\t", a, b, msg.c_str());
        }
        _INFO("\n");
    } else {
        assert(0);
    }
    _INFO(" ------ %s", desc.c_str());
}

void DumpTokens(hTokenizer hDict, const TOKENS& tokens, int nX, int flag) {
    std::string msg;
    int nPad = 0, PAD_ID = -1;
    if (hDict != nullptr) {
        msg    = hDict->Decode(tokens);
        PAD_ID = hDict->pad_id;
    }

    size_t pos = 0;
    while ((pos = msg.find('\n', pos)) != std::string::npos) {  //  escape_newlines
        msg.replace(pos, 1, "\\n");
        pos += 2;  // skip the inserted "\n"
    }
    _INFO("[%s]\n{'input_ids': tensor([[", msg.c_str());
    int i = 0;
    for (auto id : tokens) {
        _INFO("%7d,", id);
        if (++i % 9 == 0)
            _INFO("\n");
        if (id == PAD_ID) {
            _INFO("%7d <pad>...", id);
            break;
        }
    }
    _INFO("]]),\n");
    return;
}