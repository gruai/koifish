/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Application
 *  \author Yingshi Chen
 */

#include "GST_Application.hpp"

void CUDA_cleanup();

GST_Application::GST_Application(int argc, char* argv[]) {
#ifdef _WIN32
    system("chcp 65001");  // Ensures that Unicode characters (Chinese, emojis, etc.) are displayed correctly in the console output.
#endif
    //  register a cleanup function that will be automatically called when the program exits
    atexit(CUDA_cleanup);
    g_instance = this;
    SetupSignalHandlers();

    if (!params.parse(argc, argv)) {
        throw std::runtime_error("Invild arguments");
    }
}
bool GST_Application::g_running              = false;
GST_Application* GST_Application::g_instance = nullptr;

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
    GRUAI_KOIFISH_VERSION(str_version, 0x0);
    _INFO("%s", str_version);
    _INFO("\n");
}

__attribute__((destructor)) void dllUnload() {}
#endif