/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Application
 *  \author Yingshi Chen
 */

#pragma once

#include <unistd.h>

#include <csignal>
#include <iostream>
#include <memory>

#include "../CLI_params.hpp"
#include "../g_def_x.hpp"
#include "GST_obj.hpp"
#include "GST_os.hpp"

struct AppResult {
    std::error_code error_code;
    std::string error_message;
    
    AppResult() {}    
   
    AppResult(const std::error_code& code, const std::string& message = "") 
        : error_code(code), error_message(message) {}
    
    bool success() const { return !error_code; }
    bool failed() const { return static_cast<bool>(error_code); }
    
    explicit operator bool() const { return success(); }
};

class GST_Application {
   private:
    static bool g_running;
    static GST_Application* g_instance;

   protected:
    string name = "GST_Application";
    CLI_params params;

   public:
    GST_Application(int argc, char* argv[]);

    virtual ~GST_Application() { Cleanup(); }

    // Similar to OnInitInstance
    virtual bool Initialize() {
        _INFO("[APP] %s Initialization.\tProcess ID=%d\n", name.c_str(), getpid());

        // Linux-specific initializations
        if (!InitializeLogging()) {
            std::cerr << "Logging initialization failed" << std::endl;
            return false;
        }

        if (!InitializeDaemon()) {
            std::cerr << "Daemon initialization failed" << std::endl;
            return false;
        }

        g_running = true;
        return true;
    }

    // Similar to OnExitInstance
    virtual void Cleanup() {
        _INFO("[APP] %s Cleanup...\n",name.c_str());
        g_running = false;

        // Cleanup Linux-specific resources
        CleanupDaemon();
        CleanupLogging();
    }

    virtual void Swim() {
        // Main loop
        while (g_running) {
            // Process events, handle signals, etc.
            ProcessEvents();
            usleep(100000);  // 100ms sleep
        }
    }

    virtual int Run() {
        try {
            if (!Initialize()) {
                throw std::runtime_error("Initialization failed");
            }
            Swim();
            return KOIFISH_OK;
        } catch (const SafeExit& e) {
            _ERROR("%s", e.what());
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

   private:
    void SetupSignalHandlers() {
        signal(SIGINT, SignalHandler);   // Ctrl+C
        signal(SIGTERM, SignalHandler);  // Termination signal
        signal(SIGHUP, SignalHandler);   // Hangup (reload config)
    }

    static void SignalHandler(int signal) {
        std::cout << "\nReceived signal: " << signal << std::endl;
        switch (signal) {
            case SIGINT:
            case SIGTERM:
                g_running = false;
                break;
            case SIGHUP:
                if (g_instance) {
                    g_instance->ReloadConfiguration();
                }
                break;
        }
    }

    bool InitializeLogging() {
        std::cout << "Initializing system logging..." << std::endl;
        // syslog, file logging, etc.
        return true;
    }

    bool InitializeDaemon() {
        std::cout << "Initializing daemon components..." << std::endl;
        // Daemon-specific initialization
        return true;
    }

    void CleanupLogging() { std::cout << "Cleaning up logging..." << std::endl; }

    void CleanupDaemon() { std::cout << "Cleaning up daemon..." << std::endl; }

    void ProcessEvents() {
        // Process application events
        static int counter = 0;
        if (counter++ % 10 == 0) {
            std::cout << "." << std::flush;
        }
    }

    void ReloadConfiguration() { std::cout << "Reloading configuration..." << std::endl; }
};
