/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Perceptrons
 *
 *  \brief Neurons & Perceptrons
 *  \author Yingshi Chen
 */

#include <algorithm>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "./Manifold/VAE.hpp"
#include "./Manifold/gLLM.hpp"
#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

int main(int argc, char **argv) {
    try {
#ifdef _WIN32
        system("chcp 65001");       // Ensures that Unicode characters (Chinese, emojis, etc.) are displayed correctly in the console output.
#endif
        // print_build_info();
        _INFO("[ARCH] token=%ld,floatX=%ld\n", sizeof(TOKEN_ID), sizeof(floatX));
        string sEXE = EXE_name(), sExt = argc > 1 ? FILE_EXT(argv[1]) : "";
        CLI_params params;
        if (!params.parse(argc, argv)) {
            return -1;
        }

        params.OnArch();        

        hFISH fish = nullptr;
        if (params.n_swarm > 1) {
            fish = Fish::MakeSwarm("Fish_", params, 0x0);
        } else {
            params.common.n_gpu_layers = 40;
            vector<hWIKI> wikis        = WIKI::MakeInstance("", params, 0x0);
            if (wikis.size() == 0) {
                // _INFO("====== NO WIKI !!! ======\n");       return;
            } else if (params.wiki_actor == "copy") {
                wikis[0]->CopyParams(params);
            }
            fish = Fish::MakeInstance("Fish_", params, wikis, Fish::ROLE_TYPE::COMMON, 0x0);
        }
        if (fish && fish->isTrain())
            fish->Train();

        // CHECK_LAST_CUDA_ERROR();
        return 0x0;
    } catch (const std::exception &e) {
        _INFO("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char *info) {
        _INFO("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _INFO("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}

const char *GRUAI_KOIFISH_APP_NAME = "Koifish-alpha";
void GRUAI_KOIFISH_VERSION(char *str) {
    char sName[80];
    int i, nLen = (int)strlen(GRUAI_KOIFISH_APP_NAME), nFrame = 68, off;

    for (i = 0; i < nFrame; i++) sName[i] = i == 0 || i == nFrame - 1 ? '*' : ' ';
    sName[nFrame]     = '\n';
    sName[nFrame + 1] = '\0';
    off               = (nFrame - 2 - nLen) / 2 + 1;
    for (i = 0; i < nLen; i++) sName[i + off] = GRUAI_KOIFISH_APP_NAME[i];

    sprintf(str, "%s%s%s", "********************************************************************\n", sName,
            //"*                   for personal, non-commercial use.              *\n"
            "*  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen                  *\n"
            "*  SPDX-License-Identifier: MIT                                    *\n"
            "*  MAIL: gsp.cys@gmail.com                                         *\n"
            "********************************************************************\n");
}

#if (defined _WINDOWS) || (defined WIN32)
BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    char str_version[1000];
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            GRUAI_KOIFISH_VERSION(str_version);
            _INFO("%s", str_version);
            break;
        case DLL_THREAD_ATTACH:
            break;
        default:
            break;
    }

    return TRUE;
}
#else
// https://stackoverflow.com/questions/22763945/dll-main-on-windows-vs-attribute-constructor-entry-points-on-linux
__attribute__((constructor)) void dllLoad() {
    char str_version[1000];
    GRUAI_KOIFISH_VERSION(str_version);
    _INFO("%s", str_version);
    _INFO("\n");
}

__attribute__((destructor)) void dllUnload() {}
#endif
