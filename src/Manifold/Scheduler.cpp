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

bool RLS_BP::BeforeTrain(int flag) {
    if (params.strategy == SKDU_params::MEM_PRE_ALLOC)
        return true;
    if (params.strategy == SKDU_params::MEM_SWAP_GUOKE) {
        int nLayer = hFish->config.nLayer(), nG = 0, nT = 0;
        FFN *firstFFN = hFish->GetNeuron<FFN>("FFN", 0);
        firstFFN->SetGuoke(nullptr);
        SelfAttention *firstQKV = hFish->GetNeuron<SelfAttention>("SelfAttention", 0);
        firstQKV->SetGuoke(nullptr);
        for (int l = 1; l < nLayer; l++) {
            SelfAttention *QKV = hFish->GetNeuron<SelfAttention>("SelfAttention", l);
            nT += QKV->PGensors().size();
            nG += QKV->SetGuoke(firstQKV);
            FFN *ffn = hFish->GetNeuron<FFN>("FFN", l);
            nT += ffn->PGensors().size();
            nG += ffn->SetGuoke(firstFFN);
        }
        _INFO("[RLS] \tGuoke=%d(%d)\t\n", nG, nT);
    }

    return true;
}

void GeNeuron::ManageMemory(DATA_PLACE target, int typ, int flag) {
    if (name == "model.output.cls") {  //"model.blk.1.attn"&& target==FREE_DEV        "preLogits"
        int debug = 0;
    }
    if (target == place)
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
    double a = GTensor::szMaloc;
    for (auto t : arrT) {
        if (t == nullptr)
            continue;
        switch (target) {
            case FREE_DEV:
                if (t->isRefer())
                    continue;
                t->Free(hFish->config.scheduling.isParamResident);
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
        _INFO("[RLS] %s %s@%d_%s(%.3gM) mGPU=%.6gM\n", name.c_str(), op.c_str(), hOPT->GetITER(), stage.c_str(), (GTensor::szMaloc - a) / 1.0e6,
              (szTotal - szFree) / 1.0e6);
    }

    /*if(!isSymbolic){
        place = DEV_MEM;
        if(GTensor::szMaloc-sz0!=xxx){   //only for debug
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
        for (auto node : nodes) {
            if (costs + node->cost > budget) {
                break;
            }
            node->begin = 0;
            costs += node->cost;
            t++;
        }
    }
    for (auto node : nodes) {
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
void RLS_BP::Dump(int typ) const { params.Dump(typ); }
void RLS_BP::Init(Fish *hF, std::vector<hNeuron> backbons, int flag) {
    hFish  = hF;
    budget = hDevices->mostRAM / 1.0e6;
    assert(budget > 0);

    for (auto n : backbons) {
        for (auto t : n->PGensors()) tensors[t] = FLIP;
        n->ManageMemory(DATA_PLACE::SYMBOLIC);
        double mem = n->dev_most_mem / 1.0e6;
        Node *node = new RLSchedule::Node(n->name, (void *)(n.get()), mem);
        nodes.push_back(node);
    }
    assert(nodes.size() >= 2);
    Node *last = nodes[nodes.size() - 1];
    _INFO("\tRLS_BP::Init [%s,...,%s]", nodes[0]->name.c_str(), last->name.c_str());
}

bool RLS_BP::isResident(GeNeuron *neuron, int flag) {
    if (params.strategy == SKDU_params::MEM_PRE_ALLOC)
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

bool RLS_BP::Prepare(int iter, int flag) {
    double costs = 0;
    int t        = 0;

    for (auto node : nodes) {
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
            case SKDU_params::MEM_SWAP_GUOKE:
            case SKDU_params::MEM_SWAP:
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
    return true;
}