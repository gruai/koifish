/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Scheduling
 *  \author Yingshi Chen
 */
#include "Scheduler.hpp"

#include <sys/resource.h>

#include <functional>

#include "../ggex/GG_util.hpp"
#include "Fish.hpp"
#include "Optimizer.hpp"

LearnSKDU::LearnSKDU(TRAIN_CARD &train_params) : _params(train_params) {
    if (_params.lr_restart == 1)
        policy = COSINE_EPOCH;
    warmup = _params.warmup, mostIter = _params.nMostIter;
    if (policy == COSINE_EPOCH) {
        mostIter = _params.nEpochIter;
    }
    if (mostIter < warmup) {
        warmup = max(1, mostIter / 10);
        assert(warmup > 0);
    }
}

void LearnSKDU::Dump(int typ) { _INFO("\tLR policy=%s warmup=%d@%d\n", policy == COSINE ? "Cosine" : "Cosine_EPOCH", warmup, mostIter); }

float LearnSKDU::LearningRate(int64_t step, int flag) {
    float lr0                      = _params.LearningRate(), lr;
    float final_learning_rate_frac = 0.01, min_lr = lr0 * final_learning_rate_frac;
    if (policy == COSINE_EPOCH) {  //  9799        60999
        step = step % _params.nEpochIter;
    }
    switch (policy) {
        case WSD: {
            int iter_decay = mostIter * 0.9;
            if (step < warmup) {
                lr = lr0 * ((float)(step + 1)) / warmup;
            } else if (step > iter_decay) {
                lr = min_lr + (lr0 - min_lr) * (1.0 - (step - iter_decay) / (mostIter - iter_decay));
            } else {
                lr = lr0;
            }
        } break;
        default:
            if (step < warmup) {
                lr = lr0 * ((float)(step + 1)) / warmup;
            } else {
                float decay_ratio = ((float)(step - warmup)) / (mostIter - warmup);
                assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
                float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio));  // coeff starts at 1 and goes to 0
                assert(0.0f <= coeff && coeff <= 1.0f);
                float min_lr = lr0 * final_learning_rate_frac;
                lr           = min_lr + coeff * (lr0 - min_lr);
            }
            break;
    }

    return lr;
}

hGensor GeNeuron::OnInput(hGensor hIn, int flag) {
    assert(hFish != nullptr);
    bool isTemp = hFish->isTemporaryMemory(this);
    if (hFish->isRemater()) {
        assert(isTemp);
        hIn->SerialData(name, host_inp, false, dump_flag);
    }

    if (!hFish->hOPT->isBackward) {  // Forward
        if (isTemp) {
            double now = GST_ms();
            hIn->SerialData(name, host_inp, true, dump_flag);
            SUM::tUpload += GST_ms() - now;
        }
    } else {
    }
    return hIn;
}

bool RLS_BP::InitGUOKE(int flag) {
    bool isRefParam = hFish->config.fuyou.paramIsGuoke;  //
    int nLayer = hFish->config.nLayer(), nT = 0;
    if (params.strategy == MEM_STRATEGY::PRE_ALLOC_GPU) {
        int LIS = hFish->config.fuyou.nLayerInBranch;
        if (hFish->isLocalInfer) {
            for (auto task : afu->tasks) {
                GeNeuron *neuron = (GeNeuron *)(task->hOBJ);
                neuron->ManageMemory(DATA_PLACE::DEV_MEM);
            }
            return true;
        }
        if (LIS > 0) {
            int nSection = nLayer / LIS;  // brach or hierarch
            for (int i = 1; i < nSection; i++) {
                for (int l = 0; l < LIS; l++) {
                    int lay0 = l, lay1 = i * LIS + l;
                    SelfAttention *firstQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", lay0),
                                  *QKV      = hFish->GetNeuron<SelfAttention>("SelfAttention", lay1);
                    QKV->branch             = i;
                    nT += QKV->PickGensors().size();
                    nT_guoke += QKV->SetGuoke(firstQKV, isRefParam);
                    FFN *firstFFN = hFish->GetNeuron<FFN>("FFN", lay0), *ffn = hFish->GetNeuron<FFN>("FFN", lay1);
                    ffn->branch = i;
                    nT += ffn->PickGensors().size();
                    nT_guoke += ffn->SetGuoke(firstFFN, isRefParam);
                }
            }
            _INFO("[RLS_branch] \tGuoke=%d(%d) nSection=%d isRefParam=%d\t\n", nT_guoke, nT, nSection, isRefParam);
        }
    } else if (params.strategy == MEM_STRATEGY::MEM_SWAP_GUOKE) {
        FFN *firstFFN = hFish->GetNeuron<FFN>("FFN", 0);
        firstFFN->SetGuoke(nullptr, isRefParam);
        SelfAttention *firstQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", 0);
        firstQKV->SetGuoke(nullptr, isRefParam);
        for (int l = 1; l < nLayer; l++) {
            SelfAttention *QKV = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
            nT += QKV->PickGensors().size();
            nT_guoke += QKV->SetGuoke(firstQKV, isRefParam);
            FFN *ffn = hFish->GetNeuron<FFN>("FFN", l);
            nT += ffn->PickGensors().size();
            nT_guoke += ffn->SetGuoke(firstFFN, isRefParam);
        }
        _INFO("[RLS] \tGuoke=%d(%d)\t\n", nT_guoke, nT);
    }
    if (hFish->isLocalInfer) {
    } else {
        for (auto neuron : hFish->backbons) {
            neuron->ManageMemory(DATA_PLACE::DEV_MEM);
        }
    }

    return true;
}

/*
    1.  RLS_BP::Init call this once(SYMBOLIC)
    2.  RLS_BP::Prepare call this many times(DEV_MEM)
    3.  GeNeuron::BeforeMing/AfterMing call this many times!
*/
void GeNeuron::ManageMemory(DATA_PLACE target, int typ, int flag) {
    if (name == "model.blk.2.attn.wq.weight" && target != SYMBOLIC) {  //"model.blk.1.attn"&& target==FREE_DEV        "preLogits"   "model.output.cls"
        int debug = 0;
    }
    if (target == place && tReloads.empty())
        return;
    hOptimizer hOPT = hFish->hOPT;
    bool isSymbolic = target == SYMBOLIC;
    string op = "", stage = hOPT->isBackward ? "BACK" : "FORE";
    if (hFish->isRemater()) {
        stage = "Remater";
    }
    INIT_WEIGHT tpInitWeight = hFish->tpInitWeight;
    assert(out != nullptr);
    vector arrT = PickGensors();
    // size_t dev_mem = 0x0,host_mem = 0x0;
    double a = GTensor::szGlobalMaloc;
    int nX   = 0;
    for (auto t : arrT) {
        if (strcmp(t->name, "model.blk.0.attn.wq.weight") == 0 &&
            target == DEV_MEM) {  //"model.blk.1.attn"&& target==FREE_DEV        "preLogits"   "model.output.cls"
            int debug = 0;
        }
        if (t == nullptr)
            continue;
        switch (target) {
            case FREE_DEV:
                if (t->isRefer())
                    continue;
                t->Free(true);  //  hFish->config.scheduling.isParamResident
                op = "Free";
                break;
            default:
                if (isSymbolic) {
                    if (t->isRefer())
                        continue;
                    if (BIT_TEST(t->flags, GTensor::F_HOSTALLOC)) {
                        host_most_mem += t->mostMemory();
                    } else
                        dev_most_mem += t->mostMemory();
                    op = "Symbolic";
                } else {
                    t->Alloc(hOPT->GetITER(), flag);
                    op = "Alloc";
                    if (BIT_TEST(t->flags, GTensor::F_RELOAD)) {
                        bool isFree = !hFish->config.fuyou.isFirst(layer);
                        t->Serial_MMAP(true, false);  // no reset, should free all memory
                        if (isFree) {
                            t->Free();
                        }
                    }
                }
        }
    }
    if (op == "Alloc") {
        int bug = 0x0;
    }
    place = target;
    if (DUMP(0)) {
        size_t szFree, szTotal;
        cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
        _INFO("[RLS] %s %s@%d_%s(%.3gM) mGPU=%.6gM\n", name.c_str(), op.c_str(), hOPT->GetITER(), stage.c_str(), (GTensor::szGlobalMaloc - a) / 1.0e6,
              (szTotal - szFree) / 1.0e6);
    }
}

void RLSchedule::BeforeStart(int flag) { step = 0; }
bool RLSchedule::Planning(int flag) {
    assert(0);
    double costs = 0;
    int t        = 0;
    if (isPrefill) {
        for (auto node : curTasks()) {
            if (costs + node->cost > budget) {
                break;
            }
            node->begin = 0;
            costs += node->cost;
            t++;
        }
    }
    for (auto node : curTasks()) {
        if (node->isOn())
            continue;
        if (costs + node->cost > budget) {
            // costs = OffLoad(node.cost);
        }
        costs += node->cost;
        if (costs > budget) {
            assert(0);
            return false;
        }
        t++;
    }
    Verify(flag);
    return true;
}

RLS_BP::RLS_BP(EDGE_DEVICES *hED_, const CLI_params &config, int flag) : RLSchedule(hED_, config, flag) {
    params = config.scheduling;
    vector<TaskNode *> arrT;
    // if(config.fuyou.nLayerInBranch>0)
    int l_1 = config.nLayer();
    afu     = std::make_shared<Fuyou>("afu", this, hFish, arrT, 0, l_1, 0x0);
}

int RLS_BP::BeforeNextStep(int flag) {
    step++;
    return 0x0;
}
void RLS_BP::Dump(int typ) const {
    params.Dump(typ);
    size_t szFree, szTotal;
    cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
    _INFO("[RLS]\tnGuoke=%d(%.3G) Memory of GPU=%.6gM(free=%.6gM)\n", nT_guoke, szGuoke / 1.0e9, (szTotal - szFree) / 1.0e6, szFree / 1.0e6);
    auto &fuyou = hFish->config.fuyou;
    if (fuyouSwarm.size() > 1) {
        // hFuyou first = fuyouSwarm[0];

        _INFO("[Fuyou] n=%d ", fuyouSwarm.size());
        hFish->config.fuyou.Dump(0x0);
        _INFO("\n");
    }
}

bool RLS_BP::isResident(GeNeuron *neuron, int flag) {
    if (params.strategy == MEM_STRATEGY::PRE_ALLOC_GPU || params.strategy == MEM_STRATEGY::PRE_ALLOC_HOST_MAP)
        return true;
    if (dynamic_cast<TokenEmbed *>(neuron))
        return true;
    if (dynamic_cast<OutCLS *>(neuron))
        return true;
    if (dynamic_cast<LayerNormal *>(neuron))
        return true;
    if (dynamic_cast<FFN *>(neuron)) {
        // neuron->dump_flag = -1;
        //  return neuron->layer==1;
    }
    if (dynamic_cast<SelfAttention *>(neuron)) {
        // return neuron->layer==1;
        // return true;
    }

    // neuron->dump_flag = -1;
    return false;
}

void RLS_BP::Init(Fish *hF, std::vector<hNeuron> backbons, int flag) {
    hFish  = hF;
    budget = hDevices->mostRAM / 1.0e6;
    assert(budget > 0);

    for (auto n : backbons) {
        for (auto t : n->PickGensors()) {
            tMaps[t] = FLIP;
        }
        n->ManageMemory(DATA_PLACE::SYMBOLIC);
        double mem     = n->dev_most_mem / 1.0e6;
        TaskNode *node = new TaskNode(n->name, (void *)(n.get()), mem);
        afu->Add(node);  // curTasks.push_back(node);
    }
    // assert(curTasks.size() >= 2);
    TaskNode *last = afu->Last();  // curTasks[curTasks.size() - 1];
    _INFO("[RLS]\tInit [%s,...,%s]", afu->First()->name.c_str(), last->name.c_str());
    Dump(0x0);
}

Fuyou::Fuyou(const string &n, RLS_BP *hRL, Fish *hF, vector<TaskNode *> arrT, int l0, int l1, int flag) : name(n), hRLS(hRL), hFish(hF) {
    if (hFish != nullptr)
        params = hFish->config.fuyou;
    params.LIB_0 = l0;
    params.LIB_1 = l1;
    tasks        = arrT;
    if (tasks.size() == 0)
        return;
    std::hash<std::string> hasher;
    size_t hash = hasher(n);  // Returns a size_t (unsigned integer)
    rander.Init(hash);
    // auto allParams = hFish->optParams;
    string sT = "";
    nParams   = 0;
    for (auto task : tasks) {
        GeNeuron *neuron = (GeNeuron *)(task->hOBJ);
        for (auto t : neuron->PickGensors()) {
            if (BIT_TEST(t->flags, GTensor::F_PARAM)) {
                ckpParams.push_back(t);
                if (t->isRefer())  //  "model.out.weight"
                    continue;
                nParams += t->size();
                if (!hFish->isTrain())  //  loadCheckPoint
                    _INFO("\t%.5gM@%s\n", nParams / 1.0e6, t->name);
            }
        }

        bool isFy = dynamic_cast<FFN *>(neuron) || dynamic_cast<SelfAttention *>(neuron);
        if (!isFy)
            continue;

        for (auto t : neuron->PickGensors()) {
            if (BIT_TEST(t->flags, GTensor::F_PARAM)) {
                fParams.push_back(t);
                if (BIT_TEST(t->flags, GTensor::F_RELOAD)) {
                    tReloads.push_back(t);
                    sT += t->name + string(" ");
                }
            }
        }
    }
    _INFO("\t fuyou_%s nParam=%ld(T=%ld) nReload=%ld(%s)\n", name.c_str(), nParams, fParams.size(), tReloads.size(), sT.c_str());
    fflush(stdout);
}

bool Fuyou::UpdateFollower(std::shared_ptr<Fuyou> follower, int flag) {
    int nP = fParams.size(), i, nReload = 0, nFree = 0;
    assert(nP > 0 && fParams.size() == follower->fParams.size());
    // if (follower->params.paramIsGuoke) {
    //     for (auto t : follower->tReloads) {
    //         if (t->data == nullptr) {
    //             t->Serial_MMAP(false);
    //             nReload++;
    //         }
    //     }
    // }

    for (int i = 0; i < nP; i++) {
        auto tNext = follower->fParams[i], tHead = fParams[i];
        assert(tNext->isSameShape(tHead));
        if (tHead->data == tNext->data) {
            bool isReload = BIT_TEST(tNext->flags, GTensor::F_RELOAD) || BIT_TEST(tHead->flags, GTensor::F_RELOAD);
            assert(isReload);
        }
        Exploitation(tHead, tNext);
    }
    // if (nReload>0) {
    //     for (auto t : follower->tReloads) {
    //         t->Free(); nFree++;
    //     }
    //     assert(nFree==nReload);
    // }
    return true;
}
/**/
bool RLSchedule::ExploreOptimization(int iter, int flag) {
    if (fuyouSwarm.size() <= 1)
        return false;
    if (hFish->config.fuyou.algorithm == Fuyou_params::NO_EVOL)
        return false;

    double now           = GST_ms();
    hFuyou first         = fuyouSwarm[curFuyouID];  //  afu = fuyouSwarm[curFuyouID];
    vector<hFuyou> cands = fuyouSwarm;
    std::sort(cands.begin(), cands.end(),  // ugly because we don't have a typedef for the std::pair
              [](const hFuyou &a, const hFuyou &b) { return a->loss < b->loss; });
    hFuyou head = cands[0];
    head        = first;
    if (head->loss == FLT_MAX)
        return false;

    for (auto fuyou : fuyouSwarm) {
        if (fuyou == head)
            continue;
        head->UpdateFollower(fuyou);
    }
    if (DEBUG.T_fuyou == 1) {
        _INFO("[Fuyou] head=\"%s\" update algorithm=%d", head->name.c_str(), head->params.algorithm);
        _TIME_INFO(" t=", GST_ms() - now), _INFO("\n");
    }
    return true;
}

/*
bool RLSchedule::ExploreOpt_v1(int iter, int flag) {
    if (fuyouSwarm.size() <= 1)
        return false;

    double now = GST_ms();
    vector<hFuyou> cands;
    for (auto fuyou : fuyouSwarm) {
        if (fuyou == afu)
            continue;
        cands.push_back(fuyou);
    }
    std::sort(cands.begin(), cands.end(),  // ugly because we don't have a typedef for the std::pair
              [](const hFuyou &a, const hFuyou &b) { return a->loss < b->loss; });
    if (cands[0]->loss == FLT_MAX)
        return false;
    hFuyou head = cands[0];  //  afu = fuyouSwarm[curFuyouID];
    assert(head != afu);
    head->UpdateFollower(afu);
    bool doMore = iter >= hFish->config.fuyou.nWarmup();
    if(doMore){
        for(int i=1;i < cands.size(); i++){
            assert(head != cands[i]);
            head->UpdateFollower(cands[i]);
        }
    }

    if (DEBUG.T_fuyou == 1) {
        _INFO("[Fuyou] \"%s\" update from\"%s\",  algorithm=%d", afu->name.c_str(), head->name.c_str(), head->params.algorithm);
        _TIME_INFO(" t=", GST_ms() - now), _INFO("\n");
    }
    return true;
}*/

bool RLS_BP::InitBranch(int flag) {
    auto fy = hFish->config.fuyou;
    int L = hFish->config.nLayer(), t0 = fy.LIB_0, t1 = fy.LIB_1, LIS = fy.nLayerInBranch, nPass = 0;
    if (LIS <= 0) {
        assert(fuyouSwarm.size() == 0);
        fuyouSwarm = {afu};  // Ref RLS_BP::Init
        return true;
    }

    if (fuyouSwarm.size() > 0) {
        return true;
    }

    assert(L % LIS == 0);
    int nSwitch = L / LIS, fy0 = 0, fy1 = nSwitch;
    if (hFish->isLocalInfer) {  // only one best fuyou do infer
        int b = hFish->SnapShot().curFuyou;
        assert(b >= 0 && b < nSwitch);
        nSwitch = 1;
        fy0 = b, fy1 = b + 1;
        auto fy  = hFish->config.fuyou;
        fy.LIB_0 = b * LIS, fy.LIB_1 = fy.LIB_0 + LIS;
    }
    _INFO("\n[RLS_branch] branches=%d(%d/%d) ", nSwitch, L, LIS);
    // curTasks.clear();
    for (int b = fy0; b < fy1; b++) {
        int LIB_0 = b * LIS, LIB_1 = std::min((b + 1) * LIS, L);
        vector<TaskNode *> tasks;
        for (auto n : hFish->backbons) {
            bool isPass = true, isGrad = true;
            if (n->layer == 0 || n->layer > L)
                isPass = false;                                        // backbons.push_back(n);
            else if (LIB_0 <= n->layer - 1 && n->layer - 1 < LIB_1) {  // QKV,FFN
                isPass = false;
                isGrad = n->layer - 1 >= fy.LIB_1 - LIS;  // backbons.push_back(n);
                // n->isPassBack = n->layer - 1 < fy.LIB_1 - LIS;
                // if (n->isPassBack)
                //     nPass++;  // backbons.push_back(n);
            }

            if (isPass) {
                continue;
            }
            double mem     = n->dev_most_mem / 1.0e6;
            TaskNode *node = new TaskNode(n->name, (void *)(n.get()), mem);
            tasks.push_back(node);
            n->stat.Reset();
        }
        _INFO(" %d@{L%d:L%d}", tasks.size(), LIB_0, LIB_1);
        fuyouSwarm.push_back(std::make_shared<Fuyou>(std::to_string(b), this, hFish, tasks, LIB_0, LIB_1));
    }
    assert(fuyouSwarm.size() == nSwitch);
    if (hFish->isLocalInfer) {
        curFuyouID = 0;
        afu        = fuyouSwarm[curFuyouID];
        assert(afu->nParams == SUM::nzLoadParam);
    }
    _INFO("\n");
    return true;
}

bool RLS_BP::isUpdateBatch(int iter, int flag) {
    bool isUpdate = true;
    if (hFish->config.fuyou.ensemble == Fuyou_params::MULTI_SCALE)
        isUpdate = curFuyouID == 0;
    return isUpdate;
}

bool RLSchedule::isSwitchFuyou(int iter, int flag) { return iter % hFish->config.fuyou.LIB_iter_switch == 0; }
/*
    1 train LIB_0/LIB_1/LIS_2 ....
    2 train LIB_0/{LIB_0,LIB_1}/{LIB_0,LIB_1,LIS_2} ...
    3 train LIB_0/{LIB_0,LIB_1}/{LIB_0,LIB_1,LIS_2} ...  only last branch do back-propagation
 */
bool RLS_BP::UpdateBackbone(int iter, int flag) {
    auto fy = hFish->config.fuyou;
    int L = hFish->config.nLayer(), t0 = fy.LIB_0, t1 = fy.LIB_1, LIS = fy.nLayerInBranch, nPass = 0;
    assert(L % LIS == 0);
    string s = "\n", p = "\t";
    int nSwitch = L / LIS;
    hFuyou last = fuyouSwarm[curFuyouID];
    if (iter > 0)
        last->Serialize(true);
    // curFuyouID    = iter == -1 ? 0 : rand_branch.RandU32() % nSwitch;
    curFuyouID = iter == -1 ? 0 : (curFuyouID + 1) % nSwitch;
    // curBranch = 0;       // only for debug
    fy.LIB_0 = curFuyouID * LIS, fy.LIB_1 = fy.LIB_0 + LIS;
    assert(fy.LIB_1 <= L);
    // curTasks = fuyouSwarm[curFuyouID];
    afu = fuyouSwarm[curFuyouID];
    // if(iter>0)   //data is null now!!!
    //     afu->Serialize(false);
    afu->loss_0 = afu->loss;
    for (auto node : curTasks()) {
        GeNeuron *neuron = (GeNeuron *)(node->hOBJ);
        // _INFO("%s\n",neuron->__repr__(s,p).c_str());
        neuron->stat.Reset();
    }
    for (auto t : afu->tReloads) {
        t->Serial_MMAP(false);
    }

    // assert(curTasks.size() == LIS * 2 + 3);
    _INFO("[Section@%d] layer[%d-%d] tasks=%ld(nPassBack=%d) last_loss=%g(%g) N=(%d,%d,%d %d)\n", iter, fy.LIB_0, fy.LIB_1, curTasks().size(), nPass,
          last->loss, last->loss_0 - last->loss, SUM::nInitParam, SUM::nSaveParam, SUM::nLoadParam, SUM::nDogLeg);

    if (0) {
        size_t szFree, szTotal;
        cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
        // mGPU=%.6gM(free=%.6gM)(szTotal - szFree) / 1.0e6, szFree / 1.0e6);
    }

    return true;
    // if (LIS == -1) {
    //     assert(backbons.size() == 2 * L + 3);
    // }
}

vector<hFuyou> RLSchedule::ActiveFuyous(int flag) {
    assert(hFish != nullptr);
    vector<hFuyou> fuyous           = {afu};  // afu is set @UpdateBackbone
    Fuyou_params::ENSEMBLE ensemble = hFish->config.fuyou.ensemble;
    int nFuyou                      = fuyouSwarm.size();

    if (hFish->isAtPhase(LIFE_PHASE::P_EVAL_) || hFish->isAtPhase(LIFE_PHASE::P_GENERATE)) {
        if (ensemble == Fuyou_params::RANDOM_1 && nFuyou > 1) {  //  random ensembling
            uint32_t pick = rand() % nFuyou;                     // rand_fuyou.RandU32() % fuyouSwarm.size();
            fuyous        = {fuyouSwarm[pick]};
        } else if (ensemble == Fuyou_params::AGGREGATION && nFuyou > 1) {
            fuyous = fuyouSwarm;
        } else if (ensemble == Fuyou_params::FUYOU_BEST && nFuyou > 1) {
            // fuyous = fuyouSwarm;  // only for debug
        } else {
            // fuyous      = fuyouSwarm;      //only for debug
        }
    }
    return fuyous;
}

/*
    Fish::ForwardOnRLS would call this before each step
*/
bool RLS_BP::Prepare(int iter, int flag) {
    if (iter == -1) {  // only call once!
        InitBranch();
        InitGUOKE();
        Dump(0x0);
        /*for (auto node : curTasks()) {
            GeNeuron *neuron = (GeNeuron *)(node->hOBJ);
            neuron->ManageMemory(DATA_PLACE::DEV_MEM);
        }*/
    }

    switch (phase) {
        case LIFE_PHASE::P_EVAL_:
        case LIFE_PHASE::P_GENERATE:
            return true;
            break;
        case LIFE_PHASE::P_TRAIN:
            if (fuyouSwarm.size() > 1) {
                if (iter == -1 || isSwitchFuyou(iter)) {
                    UpdateBackbone(iter, flag);
                }
                //  gradient is released or zero at this time!
                // if (iter > hFish->config.fuyou.nWarmup() && DEBUG.T_fuyou==1 && isSwitchFuyou(iter)) {
                //     ExploreOptimization(iter);
                // }
            }
            break;
        default:
            break;
    }

    double costs = 0;
    int t        = 0;
    for (auto node : curTasks()) {
        // if (costs + node->cost > budget) {
        //     _INFO("[RLS]  Outof Budeget@\"%s\"!!! budget=%g,costs=%g+%g\n", node->name.c_str(), budget, costs, node->cost);
        //     assert(0);
        //     break;
        // }
        node->begin      = 0;
        GeNeuron *neuron = (GeNeuron *)(node->hOBJ);
        if (iter < 0 && isResident(neuron)) {
            resident_list += neuron->name + ", ";
        }
        /*  Refactor out since 09/10/2025
        bool isAlloc = false;
        switch (params.strategy) {
            case MEM_STRATEGY::MEM_SWAP_GUOKE:
            case MEM_STRATEGY::MEM_SWAP:
                isAlloc = isResident(neuron);  //|| iter>0
                break;
            default:
                isAlloc = true;
                break;
        }
        if (isAlloc) {
            neuron->ManageMemory(DATA_PLACE::DEV_MEM);
        }*/
        costs += node->cost;
        t++;
    }

    step = 0;
    if (iter < 0)
        _INFO("[RLS] resident={%s}\n", resident_list.c_str());
    if (DUMP(1) && iter <= 2 && phase != LIFE_PHASE::P_EVAL_) {
        size_t szFree, szTotal;
        cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);                
        _INFO("[MEMORY] mGPU=%.6gM(free=%.6gM) %s\n", (szTotal - szFree) / 1.0e6, szFree / 1.0e6, SUM::CPU_MemoryInfo().c_str());
    }
    fflush(stdout);
    assert(SUM::nInitParam == hFish->optParams.size());

    return true;
}