/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Application
 *  \author Yingshi Chen
 */

#include "GST_Application.hpp"

#include "GST_MemBuffer.hpp"

void CUDA_cleanup();
// A safe guard for cuda,supa, or other vendor device lib
class DeviceGuard {
   public:
    DeviceGuard() {}

    //  !!! Do not​ try to clean up CUDA in catch (...), static/ global destruction, ...
    ~DeviceGuard() { 
        CUDA_cleanup(); 
    }

   private:
};
std::string g_sAppName, g_sAppPath;

std::string Time2String(const std::chrono::system_clock::time_point& now, int flag = 0x0) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_buffer;
#ifdef _WIN32
    localtime_s(&tm_buffer, &time);
#else
    localtime_r(&time, &tm_buffer);
#endif
    char buf[1024];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        std::cout << "App name: " << basename(buf) << std::endl;
    }
    g_sAppPath = buf;
    g_sAppName = basename(buf);

    std::stringstream ss;
    ss << std::put_time(&tm_buffer, "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

GST_Application::GST_Application(int argc, char* argv[]) {
#ifdef _WIN32
    system("chcp 65001");  // Ensures that Unicode characters (Chinese, emojis, etc.) are displayed correctly in the console output.
#endif
    //  register a cleanup function that will be automatically called when the program exits
    // atexit(CUDA_cleanup);    //CUDA runtime unloading may happen BEFORE your atexit()!   Do not​ try to clean up CUDA in program exits
    g_instance = this;
    SetupSignalHandlers();

    start_time = std::chrono::system_clock::now();
    _INFO("🐠 started at: %s\n\n", Time2String(start_time).c_str());

    if (!params.parse(argc, argv)) {
        std::ostringstream oss;
        for (int i = 0; i < argc; ++i) {
            oss << argv[i];
            if (i != argc - 1)
                oss << " ";
        }
        _ERROR("[APP] exit now! It failed to parse arguments={%s}\n", oss.str().c_str());
        std::exit(EXIT_FAILURE);
    }
}

GST_Application ::~GST_Application() {
    end_time = std::chrono::system_clock::now();
    _INFO("🐠 end at: %s\n", Time2String(end_time).c_str());
}

void GST_Application ::Cleanup() {
    _INFO("[APP] %s Cleanup...\n", name.c_str());
    gBUFF     = nullptr;
    g_running = false;

    // Cleanup Linux-specific resources
    CleanupLogging();
}

int GST_Application ::Run() {
    try {
        if (!Initialize()) {
            throw std::runtime_error("Initialization failed");
        }
        DeviceGuard guard;
        Swim();
        Cleanup();

        return KOIFISH_OK;
    } catch (const SafeExit& e) {
        _ERROR("%s %s", e.what(), e.getFormattedInfo().c_str());
        return e.getExitCode();
    } catch (const std::exception& e) {
        _ERROR("%s", e.what());
        return KOIFISH_INTERNAL_EXCEPTION;
    } catch (const char* info) {
        _ERROR("%s", info);
        return KOIFISH_INTERNAL_EXCEPTION;
    } catch (...) {
        _ERROR("%s  Unknown exception !!!", __func__);
        return KOIFISH_INTERNAL_ERR;
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