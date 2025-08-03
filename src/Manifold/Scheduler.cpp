/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Scheduling
 *  \author Yingshi Chen
 */
#include "Scheduler.hpp"

#include "../ggex/GG_util.hpp"
#include "Fish.hpp"
#include "Optimizer.hpp"

LearnSKDU::LearnSKDU(struct train_params_ &train_params) : _params(train_params) {
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

void LearnSKDU::Dump(int typ) { _INFO("\tLR policy=%s warmup=%d@%d\n", policy == COSINE ? "COSINE" : "COSINE_EPOCH", warmup, mostIter); }

float LearnSKDU::LearningRate(int64_t step, int flag) {
    float lr0                    = _params.LearningRate(), lr;
    int final_learning_rate_frac = 0;
    if (policy == COSINE_EPOCH) {
        step = step % _params.nEpochIter;
    }

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
    bool isRefParam = params.paramIsGuoke;
    int nLayer = hFish->config.nLayer(), nT = 0;
    if (params.strategy == MEM_STRATEGY::PRE_ALLOC_GPU) {
        int LIS = params.nLayerInBranch;
        if (LIS > 0) {
            int nSection = nLayer / LIS;
            for (int i = 1; i < nSection; i++) {
                for (int l = 0; l < LIS; l++) {
                    int lay0 = l, lay1 = i * LIS + l;
                    SelfAttention *firstQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", lay0);
                    SelfAttention *QKV      = hFish->GetNeuron<SelfAttention>("SelfAttention", lay1);
                    nT += QKV->PGensors().size();
                    nT_guoke += QKV->SetGuoke(firstQKV, isRefParam);
                    FFN *firstFFN = hFish->GetNeuron<FFN>("FFN", lay0);
                    FFN *ffn      = hFish->GetNeuron<FFN>("FFN", lay1);
                    nT += ffn->PGensors().size();
                    nT_guoke += ffn->SetGuoke(firstFFN, isRefParam);
                }
            }
            _INFO("[RLS_section] \tGuoke=%d(%d) nSection=%d\t\n", nT_guoke, nT, nSection);
        }
        return true;
    }

    if (params.strategy == MEM_STRATEGY::MEM_SWAP_GUOKE) {
        FFN *firstFFN = hFish->GetNeuron<FFN>("FFN", 0);
        firstFFN->SetGuoke(nullptr, isRefParam);
        SelfAttention *firstQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", 0);
        firstQKV->SetGuoke(nullptr, isRefParam);
        for (int l = 1; l < nLayer; l++) {
            SelfAttention *QKV = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
            nT += QKV->PGensors().size();
            nT_guoke += QKV->SetGuoke(firstQKV, isRefParam);
            FFN *ffn = hFish->GetNeuron<FFN>("FFN", l);
            nT += ffn->PGensors().size();
            nT_guoke += ffn->SetGuoke(firstFFN, isRefParam);
        }
        _INFO("[RLS] \tGuoke=%d(%d)\t\n", nT_guoke, nT);
    }

    return true;
}

void GeNeuron::ManageMemory(DATA_PLACE target, int typ, int flag) {
    if (name == "model.layers.0.mlp" && target != SYMBOLIC) {  //"model.blk.1.attn"&& target==FREE_DEV        "preLogits"   "model.output.cls"
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
    vector arrT = PGensors();
    // size_t dev_mem = 0x0,host_mem = 0x0;
    double a = GTensor::szGlobalMaloc;
    for (auto t : arrT) {
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
                    if (tpInitWeight == SERIALIZE)
                        t->tpInit = tpInitWeight;
                    t->Alloc(hOPT->GetITER(), flag);
                    if (BIT_TEST(t->flags, GTensor::F_RELOAD)) {
                        auto now = GST_us();
                        assert(t->host_data != nullptr);
                        t->SerialGP(t->host_data, nullptr, t->szData, false);  // 0x7ffda0c82300
                        SUM::tX1 += GST_us() - now;
                    }
                    op = "Alloc";
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

    /*if(!isSymbolic){
        place = DEV_MEM;
        if(GTensor::szGlobalMaloc-sz0!=xxx){   //only for debug
            ManageMemory(DATA_PLACE::SYMBOLIC);
        }
    }else{
        xxx = mem;
    }*/
    // return mem;
}

void RLSchedule::BeforeStart(int flag) { step = 0; }
bool RLSchedule::Planning(int flag) {
    assert(0);
    double costs = 0;
    int t        = 0;
    if (isPrefill) {
        for (auto node : curTasks) {
            if (costs + node->cost > budget) {
                break;
            }
            node->begin = 0;
            costs += node->cost;
            t++;
        }
    }
    for (auto node : curTasks) {
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

RLS_BP::RLS_BP(EDGE_DEVICES *hED_, const CLI_params &config, int flag) : RLSchedule(hED_, config, flag) { params = config.scheduling; }

int RLS_BP::BeforeNextStep(int flag) {
    step++;
    return 0x0;
}
void RLS_BP::Dump(int typ) const {
    params.Dump(typ);
    _INFO("\tnGuoke=%d(%.3G)\n", nT_guoke, szGuoke / 1.0e9);
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
        for (auto t : n->PGensors()) tensors[t] = FLIP;
        n->ManageMemory(DATA_PLACE::SYMBOLIC);
        double mem     = n->dev_most_mem / 1.0e6;
        TaskNode *node = new RLSchedule::TaskNode(n->name, (void *)(n.get()), mem);
        curTasks.push_back(node);
    }
    assert(curTasks.size() >= 2);
    TaskNode *last = curTasks[curTasks.size() - 1];
    _INFO("\tRLS_BP::Init [%s,...,%s]", curTasks[0]->name.c_str(), last->name.c_str());
    Dump(0x0);
}

bool RLS_BP::InitBranch(int flag) {
    int L = hFish->config.nLayer(), t0 = params.LIB_0, t1 = params.LIB_1, LIS = params.nLayerInBranch, nPass = 0;
    if(LIS<=0)
        return true;
        
    assert(L % LIS == 0);
    int nSwitch = L / LIS;
    curTasks.clear();
    for (int b = 0; b < nSwitch; b++) {
        int LIB_0 = b * LIS, LIB_1 = std::min((b + 1) * LIS, L);
        arrTask tasks;
        for (auto n : hFish->backbons) {
            bool isPass = true, isGrad = true;
            if (n->layer == 0 || n->layer > L)
                isPass = false;                                        // backbons.push_back(n);
            else if (LIB_0 <= n->layer - 1 && n->layer - 1 < LIB_1) {  // QKV,FFN
                isPass = false;
                isGrad = n->layer - 1 >= params.LIB_1 - LIS;  // backbons.push_back(n);
                // n->isPassBack = n->layer - 1 < params.LIB_1 - LIS;
                // if (n->isPassBack)
                //     nPass++;  // backbons.push_back(n);
            }
            if (isPass) {
                continue;
            }
            double mem     = n->dev_most_mem / 1.0e6;
            TaskNode *node = new RLSchedule::TaskNode(n->name, (void *)(n.get()), mem);
            tasks.push_back(node);
            n->stat.Reset();
        }
        allTasks.push_back(tasks);
    }
    assert(allTasks.size()==nSwitch);
    return true;
}
/*
    1 train LIB_0/LIB_1/LIS_2 ....
    2 train LIB_0/{LIB_0,LIB_1}/{LIB_0,LIB_1,LIS_2} ...
    3 train LIB_0/{LIB_0,LIB_1}/{LIB_0,LIB_1,LIS_2} ...  only last branch do back-propagation
 */
bool RLS_BP::UpdateBackbone(int iter, int flag) {
    int L = hFish->config.nLayer(), t0 = params.LIB_0, t1 = params.LIB_1, LIS = params.nLayerInBranch, nPass = 0;

    assert(L % LIS == 0);
    int nSwitch  = L / LIS;
    curBranch    = iter == -1 ? 0 : rand_branch.RandU32() % nSwitch;
    params.LIB_0 = curBranch * LIS, params.LIB_1 = params.LIB_0 + LIS;
    assert(params.LIB_1 <= L);
    curTasks = allTasks[curBranch];
    for(auto node : curTasks){
        GeNeuron *neuron = (GeNeuron *)(node->hOBJ);
        neuron->stat.Reset();
    }
    /*curTasks.clear();
    for (auto n : hFish->backbons) {  //  @Fish::jToGraph(void *ctx_, bool isBuild, int flag)
        bool isPass = true, isGrad = true;
        if (n->layer == 0 || n->layer > L)
            isPass = false;                                                      // backbons.push_back(n);
        else if (params.LIB_0 <= n->layer - 1 && n->layer - 1 < params.LIB_1) {  // QKV,FFN
            isPass = false;
            isGrad = n->layer - 1 >= params.LIB_1 - LIS;  // backbons.push_back(n);
            // n->isPassBack = n->layer - 1 < params.LIB_1 - LIS;
            // if (n->isPassBack)
            //     nPass++;  // backbons.push_back(n);
        }
        if (isPass) {
            continue;
        }
        double mem     = n->dev_most_mem / 1.0e6;
        TaskNode *node = new RLSchedule::TaskNode(n->name, (void *)(n.get()), mem);
        curTasks.push_back(node);
        n->stat.Reset();
    }*/
    size_t szFree, szTotal;
    cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
    // assert(curTasks.size() == LIS * 2 + 3);
    _INFO("[Section@%d] layer[%d-%d] curTasks=%ld(nPassBack=%d) mGPU=%.6gM(free=%.6gM)\n", iter, params.LIB_0, params.LIB_1, curTasks.size(), nPass,
          (szTotal - szFree) / 1.0e6, szFree / 1.0e6);

    return true;
    // if (LIS == -1) {
    //     assert(backbons.size() == 2 * L + 3);
    // }
}

/*
    Fish::ForwardOnRLS would call this before each step
*/
bool RLS_BP::Prepare(int iter, int flag) {
    switch (phase) {
        case LIFE_PHASE::P_EVAL_:
            return true;
            break;
        case LIFE_PHASE::P_TRAIN:
            if (params.nLayerInBranch > 0 && (iter == -1 || iter % params.LIB_iter_switch == 0)) {
                UpdateBackbone(iter, flag);
            }
        default:
            break;
    }

    double costs = 0;
    int t        = 0;
    for (auto node : curTasks) {
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
        }
        costs += node->cost;
        t++;
    }
    step = 0;
    if (iter < 0)
        _INFO("[RLS] resident={%s}\n", resident_list.c_str());
    if (DUMP(1) && iter <= 2 && phase != LIFE_PHASE::P_EVAL_) {
        size_t szFree, szTotal;
        cudaError_t err = cudaMemGetInfo(&szFree, &szTotal);
        _INFO("[MEMORY] mGPU=%.6gM(free=%.6gM)\n", (szTotal - szFree) / 1.0e6, szFree / 1.0e6);
    }
    return true;
}