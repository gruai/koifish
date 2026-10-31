/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  非我无所取(qu)
 * 
 *  \brief Prober the emergency
 *  \author Yingshi Chen
 */

#pragma once

#include<queue>
#include "../Utils/GST_obj.hpp"
#include "../Utils/GST_util.hpp"

class GeNeuron;
class Fish;
class FeatPicker;
class GTensor;

/**
 * 1. supoort multi-thread
 */
// for each pair<value,id>
enum SORT_BY {
    NO_SORT,
    BY_VALUE,
    BY_ID,
};

struct TOPK_heap {
    int tid = -1;  // thread id, each thread for one tensor
    int dim = -1;

    float maxLogit  = 0.f;
    float lastLogit = 0.f;  // k-th largest value (smallest in heap)
    virtual bool isLarge(int i, int k, int flag = 0x0) { return false; }
    std::priority_queue<int> heap;
    std::vector<int> _picks;

    virtual int Select(int nPick, SORT_BY sort = SORT_BY::NO_SORT, int flag = 0x0);
    virtual float ValueAt(int k) {
        assert(0);
        return 0.0;
    }
};

// 
class PROBER : public std::enable_shared_from_this<PROBER>, public TOPK_heap {
    std::shared_ptr<Fish> hFish;

   public:
    typedef enum {
        ACTIVATION,
        GRAD,
        SCORE,
        PARTIAL_PRELOGITS,
    } FEAT_TYPE;
    FEAT_TYPE tpFeat = ACTIVATION;

    typedef enum {
        BEFORE_MING,
        AFTER_MING,
        AFTER_FORWARD,
        IN_FORWARD
    } COLLECT_STAGE;
    COLLECT_STAGE stage = AFTER_MING;

    // std::type_info tp;
    std::string sNeuron;
    std::string name;
    int nid_0, nid_1;       //  id of neurons
    shared_ptr<FeatPicker> hPicker = nullptr;
    std::vector<std::shared_ptr<GeNeuron>> neurons;

    PROBER() {}
    PROBER(CSTRING& nam_, CSTRING& sN, int n0, int n1, FEAT_TYPE tpF, int flag = 0x0);
    virtual bool Inject(std::shared_ptr<Fish> hFish, int flag = 0x0);
    virtual bool CollectFeats(GeNeuron* nn, shared_ptr<GTensor> inp_, int flag = 0);
    virtual void Dump(int typ, int flag=0x0) const;
};
typedef std::shared_ptr<PROBER> hPROBER;
typedef std::vector<hPROBER> arrPROBER;

struct QU  {
    int posOfTarget = -1;
    uint32_t token  = 0;
    float confi     = 0.;  // confidense
    string word     = "";
};


struct INSPECT {
    GeNeuron* hN = nullptr;
    INSPECT(GeNeuron* hN_, const std::string& info = "", int typ = 0x0, int flag = 0x0);
    virtual ~INSPECT();
};