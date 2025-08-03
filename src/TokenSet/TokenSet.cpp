/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
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

Tokenset_HellaSwag::Tokenset_HellaSwag(JSON::const_iterator jit, hTokenizer hDict, int flag) : GlobTokenset(jit, hDict, flag) {
    name = "HellaSwag";
    // rStepOfEval = 0.0;  //  no sample on evaluate
    auto k      = jit.key();
    auto v      = jit.value();
    rStepOfEval = 0;
    rStepOfEval = jKV(v, {"step"}, rStepOfEval);
    int nFile   = shard_paths.size();
    assert(nFile == 1);
}

GlobTokenset::GlobTokenset(JSON::const_iterator jit, hTokenizer hDict, int flag) : DataTokenSet(hDict) {
    header_bytes      = SHARD_HEADER_SIZE * sizeof(int);
    int num_processes = 1, process_rank = 0;
    B = hDict->config.n_batch(), T = hDict->config.n_ctx();
    total_batch_size_bytes   = ((num_processes * (B * T)) * sizeof(uint16_t));
    local_batch_offset_bytes = process_rank * B * T * sizeof(uint16_t);

    auto k         = jit.key();
    auto v         = jit.value();
    string pattern = v["glob"];
    name           = jKV(v, {"name"}, name);
    eval_every     = jKV(v, {"eval-every"}, eval_every);
    rStepOfEval    = 0.1;
    rStepOfEval    = jKV(v, {"step"}, rStepOfEval);
    if (v.find("mody") == v.end())
        nMostShard = 100000;
    else
        nMostShard = jKV(v, {"most"}, nMostShard);
    if (nMostShard == 0)  // in some case, create a blank dataset
        return;

    glob_t glob_result;
    int glob_status = glob(pattern.c_str(), 0, NULL, &glob_result);
    if (glob_status != 0) {
        _INFO("%s Error: glob failed @\"%s\"\n", __func__, pattern.c_str());
        exit(EXIT_FAILURE);
    }
    if (glob_result.gl_pathc == 0) {
        _INFO("%s No files found matching the pattern: %s\n", __func__, pattern.c_str());
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
        _INFO("[%s] %s find %.8gG tokens @\"%s\"(%d files)\n", __func__, name.c_str(), pattern.c_str(), nG, nFile);
}

size_t DataTokenSet::nBatch(int flag) {
    size_t nSample  = shard_samps.size();
    size_t nBatches = nSample / hDict->config.n_batch();
    nBatches        = nSample == 0 ? 0 : max(nBatches, (size_t)1);
    return nBatches;
}

bool GlobTokenset::LoadNextShard(SampLoader *hLoader, int flag) {
    if (shard_index > 0)
        _INFO("-------- End of shard_%d@\"%s\"-------- \n", shard_index, shard_paths[shard_index - 1].c_str());
    if (shard_index == shard_paths.size()) {
        hLoader->NextEpoch();
        shard_index = 0;
    }

    if (!hLoader->isLast) {
        size_t iRet = OnShardFile(shard_index++, true);
        if (iRet == size_t(-1))
            return false;
    }

    return true;
}

// shard type != token type
bool GlobTokenset::Shard2Sample(int flag) {
    try {
        int szT = sizeof(uint16_t), nT = (szFile - header_bytes) / szT;
        assert(nT == nShardToks);
        // tokens.resize(nT);
        fseekCheck(fpShard, (int)header_bytes, SEEK_SET);
        uint16_t *tmp16 = new uint16_t[nT];
        if (fread(tmp16, szT, nT, fpShard) != nT) {
            _INFO("Error: file size is not as expected\n");
            return 0x0;
        } /*else{
             nT = min(nT,1024);
             for(int i=0;i<nT;i++){
                 // assert(0<=tokens[i] && tokens[i]<nVocab);
             }
         }*/
        tokens.assign(tmp16, tmp16 + nT);
        delete[] tmp16;
        // InitSamps
        int n_ctx     = hDict->config.n_ctx(), len;
        size_t nToken = tokens.size(), nFirst = std::min((size_t)n_ctx, nToken), step = 1, n0 = shard_samps.size();
        // samples_size.push_back(nFirst);
        size_t end = (nToken >= n_ctx) ? (nToken - n_ctx) : 0;
        if (end > 10 * 1024 * 1024) {
            step = n_ctx;
        }
        float rSample = hDict->config.common.rSubSample;
        if (rSample > 0 && rSample < 1)
            step /= rSample;
        for (size_t sample_begin = 0; sample_begin < end; sample_begin += step) {
            len = std::min(n_ctx, (int)(end - sample_begin));
            shard_samps.push_back(new SAMP(sample_begin, len));
        }
        n0 = shard_samps.size();
        // _INFO("\t%s %s: nSamp=%ld=>%ld nBach=%d\n", __func__,name.c_str(),n0,shard_samps.size(),nBatch());
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
    // use the first glob match as the filename for now
    const char *filename = shard_paths[id].c_str();
    assert(fpShard == NULL);
    fpShard = fopenCheck(filename, "rb");
    // validate the header
    int header[SHARD_HEADER_SIZE];
    freadCheck(header, sizeof(int), SHARD_HEADER_SIZE, fpShard);
    if (header[0] != 20240520 && header[0] != 20240522) {
        printf(
            "Bad magic in the data file\n---> HINT: Are you passing in a correct file?\n---> HINT: The data encoding may have changed, re-run data prepro "
            "or refer again to README.\n");
        exit(EXIT_FAILURE);
    }
    if (header[1] != 1) {
        printf("Bad version in data file\n");
        exit(EXIT_FAILURE);
    }
    fseekCheck(fpShard, 0, SEEK_END);  // seek to end of file
    szFile     = ftell(fpShard);       // read the offset, i.e. file size
    nShardToks = 0;
    switch (header[0]) {  //
        case 20240522:    //  hellaswag dataset
            tpSample = HellaSwag;
            // int nMostCompletion =4,can_fit_examples = (int) (B / nMostCompletion);
            longest_example_bytes = header[3];
            nShardSamples         = header[2];
            nShardToks            = B * T;
            // label = (int*)mallocCheck(can_fit_examples * sizeof(int));
            break;
        default:  //  20240520
            nShardToks = header[2];
            assert(nShardToks > 0);
            // fseekCheck(fpShard, 0, SEEK_SET); // seek back to the beginning
            // we expect nTok0 in the file to be consistent with filesize, assert that is the case
            int64_t expected_file_size = SHARD_HEADER_SIZE * sizeof(int) + nShardToks * sizeof(uint16_t);
            if (szFile != expected_file_size) {
                printf("Error: file size is not as expected\n");
                exit(EXIT_FAILURE);
            }
            nShardSamples = (nShardToks * sizeof(uint16_t) - sizeof(uint16_t)) /
                            total_batch_size_bytes;  // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens

            break;
    }
    if (load) {
        Shard2Sample(0x0);
        _INFO("[shard-%d]@\"%s\": tokens=%.3g(M) nShardSamples=%ld(%ld) \n", id + 1, filename, nShardToks / 1.0e6, nShardSamples, shard_samps.size());
    }
    if (tpSample == RANDOM_GENERATE) {
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

    if (load) {
    }
    if (fpShard != NULL) {
        fcloseCheck(fpShard);
        fpShard = NULL;
    }
    return nShardToks;
}

DataTokenSet::DataTokenSet(hTokenizer hD) : hDict(hD) {
    nVocab = hDict->nVocab();
    assert(nVocab > 0);
}

TOKEN_ID DataTokenSet::At(size_t pos) {
    assert(pos < tokens.size());
    int32_t token = CLAMP(tokens[pos], 0, (nVocab - 1));
    return token;
}

bool DataTokenSet::Serialize(const std::string &path, bool isSave, int flag) {
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

double DataTokenSet::LossOnResult(hSampLoader hLoader, OutCLS *cls, int flag) {
    assert(cls != nullptr);
    double mean_loss = 0;
    int *mask = hLoader->hBatch->mask, n = 0, nzLoss = cls->nzLoss;
    float *loss = cls->hostLoss;
    TOKEN_ID token;
    // if(hasMask()){
    //     mask = TO<int>(hLoader->hostBatchMask);
    // }
    for (int i = 0; i < nzLoss; i++) {
        if (hLoader->isHostMask(i)) {
            continue;
        }
        mean_loss += loss[i];
        n++;
    }
    assert(n > 0);
    mean_loss /= n;
    return mean_loss;
}

bool Tokenset_HellaSwag::Shard2Sample(int flag) {
    try {
        size_t B = hDict->config.n_batch(), T = hDict->config.n_ctx();
        int batch_dim_offset, nComplete       = 0;
        int can_fit_examples = (int)(B / nMostCompletion), examples_per_process = nShardSamples, end_example_index = examples_per_process,
            start_example_index = 0;
        uint16_t *buffer16      = new uint16_t[longest_example_bytes];
        assert(can_fit_examples > 0);
        tokens.resize(nShardSamples * nMostCompletion * T);
        masks.resize(tokens.size());
        int num_batches = CEIL_DIV(examples_per_process, can_fit_examples), id;
        // now seek through the file to the start of that example
        // utilize <EXAMPLE_BYTES> for efficiency
        size_t header_bytes = SHARD_HEADER_SIZE * sizeof(int), sz = header_bytes;
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
            assert(context_length > 0 && context_length < T);             // context is non-empty and up to T
            uint16_t *context_tokens_start = (uint16_t *)(buffer16 + 3);  // where the tokens start
            for (int b = 0; b < nComplete; b++) {
                for (int i = 0; i < context_length; i++) {
                    int boff             = batch_dim_offset + b;
                    int tok_cur          = (int)context_tokens_start[i];
                    tokens[boff * T + i] = tok_cur;
                }
            }
            // process the completions, insert them in their row, right after the (shared) context
            uint16_t *completions_iter = buffer16 + 3 + context_length;
            for (int c = 0; c < nComplete; c++) {
                int coff                          = batch_dim_offset + c;
                int completion_length             = (int)completions_iter[0];
                uint16_t *completion_tokens_start = completions_iter + 1;
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
                samp->target = (void *)question;
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
 */
double DataTokenSet::Evaluate(Fish *hFish, hSampLoader loader0, int flag) {
    hSampLoader loader = loader0;
    if (loader == nullptr) {
        loader       = std::make_shared<SampLoader>(hFish, "Eval", false);
        loader->type = SampLoader::TYPE::DT_EVAL;
        loader->Prepare(nullptr, shared_from_this());
    }
    int i, nB = 0, step = loader->StepOfEvaluate(), iter = -1;
    double a, mean_loss = 0, now;
    hSAMP samp = nullptr;
    for (int i = 0; i < loader->num_batches; i += step) {
        loader->UpdateBatch(i, hFish);
        samp = loader->cur_samps[i];

        now               = GST_ms();
        OutCLS *cls       = hFish->GetNeuron<OutCLS>("OutCLS", 0);
        TokenEmbed *embed = hFish->GetNeuron<TokenEmbed>("TokenEmbed", 0);
        embed->hBatch     = loader->GetCurBatch();
        hFish->ForwardOnRLS(iter, 0x0);
        mean_loss += LossOnResult(loader, cls);  // loader->hTokens->LossOnResult(loader, cls);
        // mean_loss += GraphCompute(loader, _fish->hForwTG);
        nB++;
    }
    mean_loss /= nB;

    return mean_loss;
}

double Tokenset_HellaSwag::LossOnResult(hSampLoader hLoader, OutCLS *cls, int flag) {
    assert(cls != nullptr);
    double mean_loss = 0, a = 0, a_0 = DBL_MAX;
    // auto sp = hLoader->hostBatch->shape;
    // auto config = hFish->config;
    int nB = hLoader->B, nT = hLoader->T;  //,nB=sp[1],nT=sp[0]

    int *mask = nullptr, n = 0, nzLoss = cls->nzLoss, i = 0, t, b = 0, q, no = -1, nOK = 0, nQ = 0, s = 0;
    assert(nB % nMostCompletion == 0);

    float *loss = cls->hostLoss;
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