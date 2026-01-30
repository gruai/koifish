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

class KoifishApp : public GST_Application {
   protected:
    hFISH fish = nullptr;

   public:
    KoifishApp(int argc, char* argv[]) : GST_Application(argc, argv) { name = "Koifish"; }
    virtual ~KoifishApp() {
        fish  = nullptr;
        gBUFF = nullptr;
    }

    void Swim() override {
        if (params.n_swarm > 1) {
            fish = Fish::MakeSwarm("Fish_", params, 0x0);
        } else {
            // params.common.n_gpu_layers = 40;
            vector<hWIKI> wikis = WIKI::MakeInstance("", params, 0x0);
            if (wikis.size() == 0) {
                // _INFO("====== NO WIKI !!! ======\n");       return;
            } else if (params.wiki_actor == "copy") {
                wikis[0]->CopyParams(params);
            }
            fish = Fish::MakeInstance("Fish_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x0);
        }

        if (fish && fish->isTrain())
            fish->Train();
    }
};

int main(int argc, char* argv[]) {
    KoifishApp app(argc, argv);
    int iRet = app.Run();
    return iRet;
}