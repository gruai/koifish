/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
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

#include "./Shoal/gLLM.hpp"
#include "./Utils/GST_Application.hpp"
#include "./Utils/GST_log.hpp"
#include "./g_def_x.hpp"

bool CmdLineParser::koifish_configure(int flag) {
    add_positional("config_file", "path of the config file", false);
    bool success = parse();
    return success;
}

class KoifishApp : public GST_Application {
   protected:
    hFISH fish = nullptr;

   public:
    KoifishApp(int argc, char* argv[]) : GST_Application("Koifish", argc, argv) {}
    virtual ~KoifishApp() {}

    bool ParseCmdline(int flag = 0x0) override {
        bool success = cli_parser.koifish_configure();
        if (!success)
            return false;

        return true;
    }

    int Swim() override {
        vector<hWIKI> wikis;  // reserved for hybrid llm training
        fish = Fish::MakeInstance("Fish_", params, wikis, MODEL_ROLE::COMMON, 0x0);
        if (fish == nullptr) {
            _ERROR("[APP] %s is nullptr!!!", name.c_str());
            return KOIFISH_NULL_FISH;
        }
        switch (fish->phase) {
            case LIFE_PHASE::P_TRAIN:
                fish->Train();
                break;
            case LIFE_PHASE::P_SFT:
                fish->Train();
                break;
            case LIFE_PHASE::P_EVAL_:
                _INFO("[eval] ");
                break;
            case LIFE_PHASE::P_PREFILL:
                _INFO("[prefill] ");
                break;
            case LIFE_PHASE::P_CHAT_1:
                _INFO("[chat_1] ");
                break;
            case LIFE_PHASE::P_CHAT_N:
                _INFO("[chat_N] ");
                break;
            default:
                break;
        }
        fish.reset();
        return KOIFISH_OK;
    }
};

int main(int argc, char* argv[]) {
    KoifishApp app(argc, argv);
    int iRet = app.Run();
    return iRet;
}