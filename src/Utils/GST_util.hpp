/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUS SPARSE TEMPLATE	- Utility
 *  \author Yingshi Chen
 */

#pragma once

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include <algorithm>
#include <complex>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>  //for shared_ptr
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>
using namespace std;

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(WIN32)

#define GST_NOW() (clock())
#define GST_TIC(tick) clock_t tick = clock();
#define GST_TOC(tick) ((clock() - (tick)) * 1.0f / CLOCKS_PER_SEC)

static int64_t timer_freq, timer_start;
inline void GST_time_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
inline int64_t GST_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}
inline int64_t GST_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
#include <chrono>
#include <thread>

typedef std::chrono::high_resolution_clock Clock;
#define GST_NOW() (Clock::now())
#define GST_TIC(tick) auto tick = Clock::now();
// #define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now( )-(tick)).count( ))/1000.0)
#define GST_TOC(tick) ((std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - (tick)).count()) / 1000000.0)

inline void GST_time_init(void) {}

//	A millisecond is a unit of time in the International System of Units equal to one thousandth of a second or 1000 microseconds
inline double GST_ms(void) {
    /*  CLOCK_REALTIME is affected by NTP, and can move forwards and backwards. CLOCK_MONOTONIC is not, and advances at one tick per tick.
If you want to compute the elapsed time between two events observed on the one machine without an intervening reboot, CLOCK_MONOTONIC is the best option.
    */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000.0 + (int64_t)ts.tv_nsec / 1000000.0;
}

// A microsecond is equal to 1000 nanoseconds or 1⁄1,000 of a millisecond
inline double GST_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000.0 + (int64_t)ts.tv_nsec / 1000.0;
}
#endif

#define TIMING_ms(a, sum)     \
    do {                      \
        double t0 = GST_ms(); \
        a;                    \
        sum += GST_ms() - t0; \
    } while (0)

/*
    Dataset 		 10/19/2014		cys
*/
struct Dataset {
    enum { TAG_ZERO = 0x10 };
    typedef double T;
    void _mean(T *x, int flag);
    void _normal(int nz, T *x, int type, int flag);

    int nMost, ldX, *tag, type;
    T *X;

    Dataset(int n, int ld, int tp = 0x0, double *x = nullptr);
    ~Dataset() { Clear(); /*operator delete[] (tag);		operator delete[] (X);*/ }
    void Clear();

    void Preprocess(int alg, int flag);

    int Shrink(int nTo, int flag);
    int nSample() const {
        if (nMost <= 0)
            return 0;
        int i;
        for (i = 0; i < nMost && tag[i] >= 0; i++);
        return i;
    }

    double *Sample(int k) const {
        assert(k >= 0 && k < nMost);
        return X + ldX * k;
    }
    void Copy(int i, const Dataset &hS, int no, int nSample = 1, int flag = 0x0);
//	double Cost( NeuralNet*hNN,int flag );
#ifdef _GST_MATLAB_
    static Dataset *ImportMat(const char *sPath, const char *sVar, int flag);
#endif

    int Load(const std::wstring sPath, int flag);
    int Save(const std::wstring sPath, int flag);
    int ToBmp(int epoch, int _x = 0, int flag = 0x0);
};
typedef Dataset *hDATASET;
double Err_Tag(int lenT, double *acti, int tag, int &nOK, int flag);
double Err_Auto(int lenT, double *in, double *out, int &nOK, int flag);

template <typename T>
std::string G_STR(const T &x) {
    std::stringstream ss;
    ss << x;
    return ss.str();
}

std::string EXE_name(int flag = 0x0);
std::string FILE_EXT(const std::string &path);
bool VERIFY_DIR_EXIST(const std::string &path, bool isCreate = false);
struct MEM_USAGE {
    void *hData = nullptr;
    size_t sz;
    string desc;
    MEM_USAGE(size_t sz_, string d_, void *hData = nullptr, int flag = 0x0) : sz(sz_), desc(d_) {}
};
struct SUM {
    static std::vector<MEM_USAGE> mems;

    static double tX, tX1, tRemater, tQKV, tFFN, tUpload, tLoadData, tEval_0, tEval_1;
    static size_t szUpload;
    static void Reset(string typ, int flag = 0x0);
    static void TimeInfo(int flag = 0x0);
    static void MemoryInfo(int type, int flag = 0x0);
};

// Discrete Distribution of array
struct Distri_ARRAY {
    std::vector<float> distri;
    double mean = 0, sigma = 0, sum = 0, ss = 0, average = 0;
    float a0 = FLT_MAX, a1 = -FLT_MAX;

    virtual void Clear() {
        distri.clear();
        mean = 0, sigma = 0, average = 0;
        sum = 0, ss = 0;
        a0 = DBL_MAX, a1 = -DBL_MAX;
    }
    virtual void Add(float a) {
        distri.push_back(a);
        sum += a;
        ss += a * a;
        a0 = std::min(a0, a), a1 = std::max(a1, a);
    }

    virtual void Stat() {
        int n = distri.size();
        assert(n>0)    ;
        average = sum / n;
        sigma     = std::max(0.0, ss / n - average * average);  // float point error
        sigma     = sqrt(sigma);
    }
};

#ifndef _WIN32
#include <arpa/inet.h>
#include <dirent.h>
#else
#pragma comment(lib, "Ws2_32.lib")  // Link Ws2_32.lib for socket functions
#include <winsock2.h>
#endif

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        fprintf(stderr,
                "---> HINT 1: dataset files/code have moved to dev/data recently (May 20, 2024). You may have to mv them from the legacy data/ dir to "
                "dev/data/(dataset), or re-run the data preprocessing script. Refer back to the main README\n");
        fprintf(stderr, "---> HINT 2: possibly try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n", file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

extern inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

extern inline void sclose_check(int sockfd, const char *file, int line) {
    if (close(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
extern inline void closesocket_check(int sockfd, const char *file, int line) {
    if (closesocket(sockfd) != 0) {
        fprintf(stderr, "Error: Failed to close socket at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
    if (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "Error: File write error at %s:%d\n", file, line);
        } else {
            fprintf(stderr, "Error: Partial write at %s:%d. Expected %zu elements, wrote %zu\n", file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Written elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

extern inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// check that all tokens are within range
extern inline void token_check(const int *tokens, int token_count, int vocab_size, const char *file, int line) {
    for (int i = 0; i < token_count; i++) {
        if (!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "Error: Token out of vocabulary at %s:%d\n", file, line);
            fprintf(stderr, "Error details:\n");
            fprintf(stderr, "  File: %s\n", file);
            fprintf(stderr, "  Line: %d\n", line);
            fprintf(stderr, "  Token: %d\n", tokens[i]);
            fprintf(stderr, "  Position: %d\n", i);
            fprintf(stderr, "  Vocab: %d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)

extern inline int find_max_step(const char *output_log_dir) {
    // find the DONE file in the log dir with highest step count
    if (output_log_dir == NULL) {
        return -1;
    }
    DIR *dir;
    struct dirent *entry;
    int max_step = -1;
    dir          = opendir(output_log_dir);
    if (dir == NULL) {
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "DONE_", 5) == 0) {
            int step = atoi(entry->d_name + 5);
            if (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    return max_step;
}

extern inline int ends_with_bin(const char *str) {
    // checks if str ends with ".bin". could be generalized in the future.
    if (str == NULL) {
        return 0;
    }
    size_t len         = strlen(str);
    const char *suffix = ".bin";
    size_t suffix_len  = strlen(suffix);
    if (len < suffix_len) {
        return 0;
    }
    int suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    return suffix_matches;
}

class GST_util {
   public:
    static int dump;
    static int verify;

    enum {
        VERIRY_KRYLOV_FORM = 1,
        VERIRY_UNITARY_MAT = 2,
        VERIRY_I_MAT       = 2,
        VERIRY_TRI_MAT     = 2,
    };
    static void print(const char *format, ...) {
        if (dump == 0)
            return;
        char buffer[1000];
        va_list args;
        va_start(args, format);
        //	vfprintf( stderr,format,args );
        vsnprintf(buffer, 1000, format, args);
        printf("%s", buffer);
        va_end(args);
    }
#ifdef _GST_MATLAB_
    static int LoadDoubleMAT(const char *sPath, const char *sVar, int *nRow, int *nCol, double **val, int flag);
#endif
    //	friend std::ostream& operator<<( std::ostream& os,const hGMAT &mat )	{	mat->dump(os);		return os; }
    template <typename T>
    static void Output(char *sPath, int m, int n, T *dat, int flag = 0x0) {
        std::ofstream file;
        file.open(sPath, std::ios::out);
        if (file.is_open()) {
            //???
            // hGMAT hMat= make_shared<Matrix<Z>>(m,n,dat,GeMAT::DATA_REFER );
            // file<< hMat.get() << endl;
            file.close();
        }
    }
};

inline bool isStrMatch(const string &target, const vector<string> &words) {
    for (auto w : words) {
        if (target.find(w) != std::string::npos)
            return true;
    }
    return false;
}