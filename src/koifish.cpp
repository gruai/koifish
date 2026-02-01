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
        
    }

    void Swim() override {
        vector<hWIKI> wikis;    //reserved for hybrid llm training
        fish = Fish::MakeInstance("Fish_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x0);

        if (fish && fish->isTrain())
            fish->Train();
    }
};

int main(int argc, char* argv[]) {
    KoifishApp app(argc, argv);
    int iRet = app.Run();
    return iRet;
}