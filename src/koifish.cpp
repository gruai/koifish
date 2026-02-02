/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *
 *  \brief main
 *  \author Yingshi Chen
 */

#include <algorithm>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "./Manifold/gLLM.hpp"
#include "./Utils/GST_Application.hpp"
#include "./Utils/GST_log.hpp"
#include "./g_def_x.hpp"

class KoifishApp : public GST_Application {
   protected:
    hFISH fish = nullptr;

   public:
    KoifishApp(int argc, char* argv[]) : GST_Application(argc, argv) { name = "Koifish"; }
    virtual ~KoifishApp() {}

    int Swim() override {
        vector<hWIKI> wikis;  // reserved for hybrid llm training
        fish = Fish::MakeInstance("Fish_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x0);
        if (fish == nullptr) {
            _ERROR("[APP] %s is nullptr!!!", name.c_str());
            return KOIFISH_NULL_FISH;
        }
        switch (fish->phase) {
            case LIFE_PHASE::P_TRAIN:
                fish->Train();
                break;
            case LIFE_PHASE::P_EVAL_:
                _INFO("[eval] ");
                break;
            case LIFE_PHASE::P_PREFILL:
                _INFO("[prefill] " );
                break;
            case LIFE_PHASE::P_GENERATE:
                _INFO("[generate] ");
                break;
            default:
                break;
        }

        return KOIFISH_OK;
    }
};

int main(int argc, char* argv[]) {
    KoifishApp app(argc, argv);
    int iRet = app.Run();
    return iRet;
}