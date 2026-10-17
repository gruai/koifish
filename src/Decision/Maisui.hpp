
/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Some key characteristics & design goals
 *      1.  Flex - it may have different structures/parameters at different phase!
 *      2.  Sparse
 *      3.  Lite - need much less resource than any other
 *      4.  Tough
 *
 *  \brief Fish - Life is just a random swimming fish(子在川上曰: "逝者如斯夫").
 *  \author Yingshi Chen
 */
#pragma once
#include "../Shoal/Fish.hpp"

struct DORT_wrap;

/*
    麦穗鱼
*/
class Maisui : public Fish {
   protected:
    hFISH backbone  = nullptr;  //  decision model readouts logits of backbone
    // DORT_wrap* dort = nullptr;
    string sPickerName = "maisui";
    int x = 0;
    float fitness;  // greater fitness will have a greater probability of being selected for recombination.
    vector<double> position;
    bool AllocBuffer(int flag = 0x0) override;
    bool Build(int flag = 0x0) override;
    // Maisui(const int dim, int flag = 0x0);
    // Maisui(const int dim, const vector<int>&picks, int flag = 0x0);
   public:
    Maisui(const std::string& nam_, struct CLI_params params, int flag = 0x0);
    size_t nClass() const override;
    bool Init(const vector<hWIKI>& wikis_, int flag = 0x0) override;
    void Train(int flag = 0x0) override;
    bool InitInput(void* ctx_build, bool isMask, int flag) override;
};
typedef shared_ptr<Maisui> hMaisui;