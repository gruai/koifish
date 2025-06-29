/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUS SPARSE TEMPLATE	- OS/System Utility
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#include <fstream>
#include <iostream>
#include <memory>  //for shared_ptr
#include <string>
#ifndef _WIN32
#include <arpa/inet.h>
#include <dirent.h>
#else
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#include <winsock2.h>
#endif
extern inline void create_dir_if_not_exists(const char *dir) {
    if (dir == NULL) {
        return;
    }
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) == -1) {
            printf("ERROR: could not create directory: %s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("created directory: %s\n", dir);
    }
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
extern inline const std::string DATE(int fmt = 0x0) {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    std::string sDate = buf;
    sDate             = "\"" + sDate + "\"";
    return sDate;
}

#define cDATE DATE().c_str()