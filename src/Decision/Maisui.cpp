/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "Maisui.hpp"

#include "../Manifold/HotPicker.hpp"
#include "../Manifold/Optimizer.hpp"
#include "../Shoal/gLLM.hpp"
#include "../TokenSet/Batch.hpp"

Maisui::Maisui(const std::string& nam_, CLI_params params, int flag) : Fish(nam_, params) {
    //  @Fish::MakeSwarm
    int nSwarm = params.n_swarm, i, nEnbed = config.nEmbed();
    for (i = 0; i < nSwarm; i++) {  // support multiple backbone
        MODEL_ROLE role       = BACKBONE_LOGITS;
        string title          = "Backbone_";
        CLI_params paramSwarm = params;
        assert(paramSwarm.sft.baseArch == NTP_QWEN3);
        paramSwarm.common.SetOnlyInfer();
        paramSwarm.phase            = LIFE_PHASE::P_EVAL_;
        paramSwarm.model.sPadMask = "last"; //paramSwarm.common.isPadMask = true;
        hFISH fish                  = Fish::MakeInstance(title, paramSwarm, wikis, role, 0x0);

        // fish->isLocalInfer = true;
        Fish::swarm.push_back(fish);
        backbone = fish;
    }
    assert(backbone->hDict != nullptr);
    hDict = backbone->hDict;
}

bool Fish::InitProbers(CSTRING& sType, int nMostSamp, int flag) {
    // string sPickerName = "maisui";
    // int nEmbed         = config.nEmbed();
    // Head4Token* hCLS   = GetNeuron<Head4Token>("Head4Token", 0);
    // hCLS->InitFeatPicker(sPickerName, nEmbed);
    // probers.push_back(std::make_shared<PROBER>());
    // probers[0]->hPicker = Pickers[sPickerName];
    // return probers.size()>0;
    int nInject = 0;

    if (sType == "last_hidden") {
        // probers.push_back(std::make_shared<PROBER>("last_hidden", "Head4Token", -1, -1, PROBER::ACTIVATION));
        probers.push_back(std::make_shared<PROBER>("last_hidden", "LayerNormal", -1, -1, PROBER::ACTIVATION));
    } else if (sType == "prelogits") {
        probers.push_back(std::make_shared<PROBER>("prelogits", "Head4Token", -1, -1, PROBER::PARTIAL_PRELOGITS));
        // probers.push_back(std::make_shared<LogitsInfo>("prelogits", "Head4Token", -1, PROBER::PARTIAL_PRELOGITS));
    }
    for (auto prober : probers) {
        prober->Inject(shared_from_this());
        nInject += prober->neurons.size();
        arrPicker.push_back(prober->hPicker);
        prober->Dump(0x0);
    }
    // _INFO("[Prober] nAllInject=%d @", nInject);
    // if(nInject==0)
    //     return KOIFISH_PROBER_INJECT;

    return nInject > 0;
}

int Fish::CollateHidden(hDataToken coral, const std::string& sX, int flag) {
    assert(coral != nullptr);
    int nEmbed = config.nEmbed(), nSamp = coral->shard_samps.size(), nLoad = 0;
    assert(curTokenSet == nullptr);
    bool isCache = !DEBUG.fHiddenFeature.empty();
    // auto picker = Pickers[sPicker];
    for (auto prober : probers) {
        hFeatPicker picker = prober->hPicker;
        assert(picker != nullptr);
        picker->nPickSamp = 0;
        std::string sPath = _sLOG("%s_%s@[%s]_<%s>_N%d.dat", DEBUG.fHiddenFeature.c_str(), Name().c_str(), coral->name.c_str(), prober->name.c_str(), nSamp);
        if (isCache && picker->SerialFeatures(this, sPath, false)) {
            nLoad++;
        } else {
            //  off = i0 + i1 * ne[0] + i2 * ne[0] * ne[1] + i3 * ne[0] * ne[1] * ne[2]
            SHAPE spMost = {nSamp, picker->dim()};  // extra line for target
            if (picker->raw_feat != nullptr) {
                picker->raw_feat.reset();
            }
            picker->raw_feat = std::make_shared<GTensor>(this, spMost, typNUMBER::F32, true, GTensor::F_DEBUG);
            picker->raw_feat->Alloc();
        }
    }
    if (nLoad == probers.size()) {
        return 0x0;
    }

    DEBUG.Time_most = 60 * 1000;
    curTokenSet     = coral;
    SetPhase(P_EVAL_);
    _INFO("\n[Feature] Collate @\"%s\" ...", coral->name.c_str());
    // Evaluate({coral}, SAMPLEofSHARD);
    coral->loader->QiWu(SAMPLEofSHARD, 0x0);

    for (auto prober : probers) {
        auto picker = prober->hPicker;
        picker->CleanFeats();  // Shrink
        if (picker->empty()) {
            _ERROR("[Feature] Failed to pick anything!\n");
            return KOIFISH_HIDDEN_NULL;
        }
        if (isCache) {
            std::string sPath = _sLOG("%s_%s@[%s]_<%s>_N%d.dat", DEBUG.fHiddenFeature.c_str(), Name().c_str(), coral->name.c_str(), prober->name.c_str(), nSamp);
            picker->SerialFeatures(this, sPath, true);
        }
    }

    curTokenSet = nullptr;
    return 0x0;
}

void Maisui::Train(int flag) {
    InitSpecialTokensOfDecision(GetTokenizer());

    assert(backbone != nullptr);
    if (!backbone->InitProbers("last_hidden", -1))  //  "prelogits"   "last_hidden"
        return;
    arrPicker        = backbone->arrPicker;
    hDataToken coral = backbone->tsTrain;
    assert(coral != nullptr);
    backbone->CollateHidden(coral, sPickerName);
#ifdef __USE_GBDT__
    for (auto picker : arrPicker) {
        // hFeatPicker picker = backbone->Pickers[sPickerName];
        assert(picker != nullptr);

        picker->hTrainData = picker->Coral2Tabular(coral, backbone, arch, FeatsOnFold::DF_TRAIN);
    }
    if (!backbone->tsEval.empty()) {
        coral = backbone->tsEval[0];
        backbone->CollateHidden(coral, sPickerName);
        for (auto picker : arrPicker) {
            picker->hValData = picker->Coral2Tabular(coral, backbone, arch, FeatsOnFold::DF_EVAL);
        }
    }
#else
    NOT_IMPLEMENTED;
#endif
    arrPicker[0]->Train();

}

bool Fish::Init(const vector<hWIKI>& wikis_, int flag) {
    wikis = wikis_;
    hEDS  = EDGE_DEVICES::GetInstance(config);
    if (hDict == nullptr) {
        if (!InitTokenCoral())  //
            return false;
    }
    if (config.common.method == "gbdt") {
    }

    if (wikis.size() == 0) {
        _INFO("[WIKI] OFF\n");  // return false;
    } else {
        teach = wikis[0]->teach;
    }

    hDistler = nullptr;  // config.sigma=="" ? nullptr : std::make_shared<Distillation>(this,config,0x0);     //ADD SIGMA

    // InitModel(flag);
    config.Dump();

    if (hOPT != nullptr) {
        // hOPT->Dump(1);
        if (/*config.isOnlyGPT ||*/ !isTrain())
            return true;
        if (!hOPT->PrepareData(config, flag))
            return false;
    }
    return true;
}

bool Maisui::Init(const vector<hWIKI>& wikis_, int flag) {
    bool bRet = Fish::Init(wikis_, flag);
    return bRet;
}

size_t Maisui::nClass() const { return 2; }

bool Maisui::InitInput(void* ctx_build, bool isMask, int flag) { return true; }

bool Maisui::AllocBuffer(int flag) { return true; };

bool Maisui::Build(int flag) { return true; }

int Maisui::ChatOnSamps(int iter, int flag) {
    assert(gopt == nullptr && backbone->gopt == nullptr);
    backbone->config.chat_sampler.mode = CHATML_ASSIST;
    hDataToken coral                   = backbone->tsTrain;
    hChater gopt                       = GeneratOnPrompt::MakeInstance(coral, wikis, backbone.get(), flag);
    backbone->gopt                     = gopt;
    SetPhase(P_EVAL_);
    assert(gopt->hBatch == nullptr);
    gopt->hBatch = coral->GetNanny()->GetCurBatch();
    // coral->Evaluate({coral}, SAMPLEofSHARD);
    coral->GetNanny()->QiWu(SAMPLEofSHARD, 0x1000);
    return 0x0;
    int seq_len = curChatLen(LIMIT), next, token, generated_tokens = 0;
    hTokenizer tokenizer = GetTokenizer();
    double start_time = 0, eval = 0;

    ;
    hBATCH hBatch = gopt->hBatch;
    // DEBUG.T_generate_most_layer = 1;
    DEBUG.verGenerate     = DEBUG.cmd_p1;  // use this flag to comparse accu/time of different version
    DEBUG.T_cuQK          = 0;
    DEBUG.T_kvcache_quant = 0;
    // g_dump_level          = -1;
    string cur_answer, rendered_prompt;
    gopt->OnPrompt(iter, nullptr);
    int num_prompt_tokens = gopt->hBatch->nTokens(BATCH_INPUT::F_PROMPT_LEN);  // nPrefill;
    for (int i = 0; i < num_prompt_tokens - 1; i++) {                        // prefill
        eval = Evaluate({gopt->tsChat}, DL_BATCH_UPATE::BATCHofEMBED);
        gopt->hBatch->tok_pos++;
        if (gopt->hBatch->tok_pos == 1) {  // nRound == 2
            DEBUG_HERE;
            // K_EXIT(KOIFISH_EXIT_DEBUG);
        }
    }
    start_time = GST_ms();
    SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
    while (true) {
        eval = Evaluate({}, DL_BATCH_UPATE::BATCHofEMBED);
        gopt->OnLogits(nullptr, nullptr);
        hBatch->tok_pos++;
        // _INFO(" %d[%d->%d]", pos, token, next), fflush(stdout);
        token = gopt->Sample(hBatch);  // 3347
        generated_tokens++;
        if (token == tokenizer->S.eos || hBatch->tok_pos >= seq_len) {  //  stop generation if get EOS token
            double elapsed_s = (double)(GST_ms() - start_time) / 1000.0;
            double tps       = (generated_tokens > 0 && elapsed_s > 0) ? (generated_tokens - 1) / elapsed_s : 0.0;
            if (hBatch->tok_pos >= seq_len) {
                if (outer_phase == P_TRAIN)
                    return 0x0;
                _WARN("%scontext window full!%s\t", COLOR_YELLOW, COLOR_RESET);
            }
            _INFO("\n%s[%.2f tk/s, %d tokens in %.2fs(qkv=%.3fs ffn=%.3fs PreLogits=%.3fs X=%.3fs)]%s\n===================================\n", COLOR_GREEN, tps,
                  generated_tokens - 1, elapsed_s, SUM::tQKV_forw / 1.0e6, SUM::tFFN / 1.0e6, SUM::tPreLogits / 1.0e6, SUM::tX1 / 1.0e6, COLOR_RESET);

            cur_answer += "\t\t" + SUM::sQuantInfo;
            STR2FILE("chat.csv", cur_answer, std::ofstream::out);
            // OnEOS(shared_from_this());
            break;
        }
        hBatch->SetToken(hBatch->tok_pos, 0, 0, 0, token);

        static int in_thinking_section = 0;
        static int in_bold_section     = 0;
        if (hBatch->tok_pos == num_prompt_tokens) {                     // first token of the response
            in_thinking_section = config.chat_sampler.enable_thinking;  // reset thinking state
            in_bold_section     = 0;                                    // reset bold state
            if (in_thinking_section) {
                _INFO(COLOR_YELLOW);
            }
        }

        const char* piece = tokenizer->T2STR(token).c_str();  //
        if (strcmp(piece, "</think>") == 0) {
            in_thinking_section = 0;
            if (!in_bold_section) {
                _INFO(COLOR_RESET);
            }
        } else {
            const char *current_pos = piece, *marker;
            while ((marker = strstr(current_pos, "**")) != NULL) {
                // print the text before the marker
                fwrite(current_pos, 1, marker - current_pos, stdout);

                // flip the bold state and change colour accordingly
                in_bold_section = !in_bold_section;
                if (in_bold_section) {
                    _INFO(COLOR_BOLD_RED);
                } else if (in_thinking_section) {
                    _INFO(COLOR_YELLOW);
                } else {
                    _INFO(COLOR_RESET);
                }
                current_pos = marker + 2;  // Move past the "**"
            }
            // print any remaining text after the last marker
            if (token != tokenizer->S.eos) {
                _INFO("%s", current_pos);
                cur_answer += current_pos;
            }
        }

        fflush(stdout);
    }

    double tSample = (double)(GST_ms() - start_time) / 1000.0;
    gopt->AfterSample(iter, tSample);

    return 0x0;
}

int Fish::ChatOnCoral(hDataToken coral, int flag) {
    assert(coral != nullptr);
    // InitProbers("prelogits", -1);
    // assert(arrPicker.size() == 1);
    int nEvalTokens = 0, seq_len = curChatLen(LIMIT), next, token, logits0 = 0, logits1 = 1;
    assert(curTokenSet == nullptr);
    curTokenSet          = coral;
    DEBUG.Time_most      = 60 * 1000;
    hSampNanny hNany     = coral->loader;
    hTokenizer tokenizer = GetTokenizer();
    assert(isAtPhase({P_EVAL_CHAT}));
    hBATCH hBatch = hNany->GetCurBatch();
    gopt->hBatch  = hBatch;

    bool isOnly1Token = DEBUG.cmd_p1;  // true of false
    bool isDump       = false;         // iBatch % DEBUG.dump_Eval_each == 0;
    // hBatch->onlyLogits = true;
    // header->dump_flag  = -1;  // only for debug
    // DEBUG.T_generate_most_layer = 1;
    // g_dump_level          = -1;
    SUM::tX1 = 0.0, SUM::tQKV_forw = 0.0, SUM::tFFN = 0.0, SUM::tPreLogits = 0.0;
    double tic = GST_ms(), tps, tRemain = 0.0, tpi = 0, relax = 0.9, dt, tCur, tLast;
    int nB = 0, step = hNany->StepOfEvaluate(), nMost = CEIL_DIV(hNany->num_batches, step);
    for (int iBatch = 0; iBatch < nMost; iBatch++) {
        // if (i < 50) {  // only for debug
        //     Next();
        //     continue;
        // }
        hBatch->tok_pos = 0;
        if (hNany->CollateBatch(min(iBatch * step, hNany->num_batches), this) == 0)
            continue;
        gopt->curSamps = hBatch->curSamps;
        hSAMP hSamp    = gopt->curSamps[0];
        // Chat(0x0, 0x0);
        double start_time = GST_ms(), eval = 0;
        gopt->cur_answer = "";  //"\n[" + std::to_string(iBatch) + "] ";
        gopt->OnPrompt(iBatch, nullptr);
        if (isOnly1Token) {
            SetPhase(P_EVAL_, P_X, 0x100);
            hBatch->onlyLogits = true;
            logits0            = hSamp->nPrefill - 1;  // gopt->hBatch->nPrefill - 1
            logits1            = logits0 + 1;
            ForwardOnRLS(iBatch, 0x0);
        } else {                                             // generate many token uing kv-cache
            SetPhase(P_CHAT_1, P_X, 0x100);                  // hack
            for (int i = 0; i < hSamp->nPrefill - 1; i++) {  // prefill
                // eval = hNany->Evaluate(DL_BATCH_UPATE::BATCHofEMBED, 0x0);
                ForwardOnRLS(iBatch, 0x0);
                hBatch->tok_pos++;
            }
        }
        gopt->tPrefill = (double)(GST_ms() - start_time) / 1000.0;
        start_time     = GST_ms();
        gopt->BeforeGenerate();
        while (true) {
            // eval = hNany->Evaluate(DL_BATCH_UPATE::BATCHofEMBED, 0x0);
            if (!isOnly1Token)
                ForwardOnRLS(iBatch, 0x0);
            gopt->OnLogits(hSamp, arrPicker.empty() ? nullptr : arrPicker[0]);
            hBatch->tok_pos++;
            assert(hBatch->tok_pos <= seq_len);
            token = gopt->Sample(hBatch);  // 1866=true
            gopt->generated_tokens++;
            if (token == tokenizer->S.eos || hBatch->tok_pos >= seq_len) {  //  stop generation if get EOS token
                if (hBatch->tok_pos >= seq_len) {
                    if (outer_phase == P_TRAIN)
                        return 0x0;
                    _WARN("%scontext window full!%s\t", COLOR_YELLOW, COLOR_RESET);
                }
                break;
            }
            hBatch->SetToken(hBatch->tok_pos, 0, 0, 0, token);
            if (token != tokenizer->S.eos)
                gopt->cur_answer += tokenizer->T2STR(token);
            if (isOnly1Token)
                break;
        }

        gopt->tSample = (double)(GST_ms() - start_time) / 1000.0;
        gopt->DisplayAnswer(gopt->cur_answer, 0x0);
        gopt->AfterSample(iBatch + 1, 0.0, iBatch + 1 == nMost);
        SetPhase(P_EVAL_CHAT);

        nEvalTokens += hBatch->nFillTokens(), nB++;
        tCur = GST_ms(), dt = tCur - tLast, tLast = tCur;
        tpi = tpi * (1.0 - relax) + dt * relax, tRemain = (nMost - iBatch) * tpi;  //  ms

        if (isDump /*&& tRemain > 60 * 1000*/) {
            _INFO("\r\t%d/%d %.4gk/s ...\t", iBatch, nMost, nEvalTokens / (tCur - tic));
            _TIME_INFO("remain=", tRemain), _INFO("        ");
        }
        if ((tCur - tic) / 1.0e3 > DEBUG.Time_most) {
            _WARN("\n \t%d/%d SampNanny::Evaluate Out of time(%.3gs)\n", iBatch, nMost, DEBUG.Time_most);
            break;
        }
    }

    curTokenSet = nullptr;

    return 0x0;
}

/* Xoshiro256PlusSIMD       https://github.com/stephanfr/Xoshiro256PlusSIMD/blob/main/README.md
 * https://www.lomont.org/papers/2008/Lomont_PRNG_2008.pdf
 */
bool GeneratOnPrompt::OnLogits(hSAMP hSamp, shared_ptr<FeatPicker> hPicker, int flag) {
    int nV = -1;
    if (hSamp == nullptr) {                                                 //  logits0 == 0 && logits1 == -1
        D2H(hClsLogits->data, hClsLogits->host_data, hClsLogits->nByte());  //
        nV = hClsLogits->ne[1];
    } else {
        int logits0 = hSamp->nPrefill - 1, logits1 = logits0 + 1;  // nPrefill = 213
        nV = logits1 - logits0;
        // assert(fish_0->arrPicker.size()==1) ;
        // hFeatPicker logitPicker = fish_0->arrPicker[0];
        size_t off_0 = hClsLogits->Offset(0, logits0, 0, 0), off_1 = hClsLogits->Offset(0, logits1, 0, 0);
        assert(off_1 > off_0);
        D2H((hBITARR)hClsLogits->data + off_0, (hBITARR)hClsLogits->host_data, off_1 - off_0);
    }
    int ldH = hClsLogits->ne[0];
    // hClsLogits->Print("_logits",0,-1);
    // PrintTensor<floatLogits>("host", (floatLogits*)hClsLogits->host_data, false, ldH, nV, 1, 1, -1);
    // _INFO("\n%s(Invalid logits!)%s\n", COLOR_RED, COLOR_RESET);
    return true;
}