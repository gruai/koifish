/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief  Some functions of Download
 *  \author Yingshi Chen
 */

// ms_download.cpp  (multi-threaded version)
//
// Downloads the Qwen3.8-Flash-Next GSQ-RCO GGUF shards (default: IQ3_S, 2 shards) from ModelScope,
// the same way strata's setup.py does:
//
//   file list + size + SHA-256 : GET <ep>/api/v1/models/<repo>/repo/files?Recursive=true
//                                  -> Data.Files[] { Path, Size, Sha256 }
//   a file                     : GET <ep>/models/<repo>/resolve/master/<Path>
//
// Each file is split into fixed-size chunks. A pool of worker threads (one libcurl handle each) fetches
// chunks with HTTP Range requests and writes them at their own offset with pwrite() into a preallocated
// <file>.part.  <file>.part.map records which chunks are complete, so an interrupted run resumes at chunk
// level.  At the end the size and SHA-256 are checked against ModelScope's, and <file>.done is written.
//
// Build:  g++ -std=c++17 -O2 -pthread ms_download.cpp json11.cpp -lcurl -o ms_download
// Run:    ./ms_download --quant IQ3_S --out ./models --threads 8 --chunk-mb 32
//         ./ms_download --help

#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif

#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "../CLI_params.hpp"
#include "./GST_Serialize.hpp"
#include "./json/json12.hpp"

namespace fs = std::filesystem;

// static const char* kRepo       = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
// static const int kShards       = 2;
static const char* kDefaultEp  = "https://www.modelscope.cn";


static const int kChunkRetries = 8;
static int g_backoff_ms        = 2000;  // base of the retry backoff (a test lowers it)

// ------------------------------------------------------------------------------------------ SHA-256 (BEGIN)
class Sha256 {
   public:
    Sha256() { reset(); }

    void reset() {
        static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, init, sizeof(h_));
        len_  = 0;
        fill_ = 0;
    }

    void update(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        len_ += n;
        while (n > 0) {
            size_t take = std::min(n, sizeof(buf_) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == sizeof(buf_)) {
                block(buf_);
                fill_ = 0;
            }
        }
    }

    std::string hex() {  // finishes the hash; call reset() before reusing the object
        uint64_t bits     = len_ * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (fill_ != 56) update(&zero, 1);
        uint8_t lenbe[8];
        for (int i = 0; i < 8; i++) lenbe[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        update(lenbe, 8);
        char out[65];
        for (int i = 0; i < 8; i++) std::snprintf(out + 8 * i, 9, "%08x", static_cast<unsigned>(h_[i]));
        return std::string(out, 64);
    }

   private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block(const uint8_t* p) {
        static const uint32_t K[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
                                       0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
                                       0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                                       0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                                       0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
                                       0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                                       0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t v[8];
        std::memcpy(v, h_, sizeof(v));
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            uint32_t t1 = v[7] + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            uint32_t mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            uint32_t t2 = S0 + mj;
            v[7]        = v[6];
            v[6]        = v[5];
            v[5]        = v[4];
            v[4]        = v[3] + t1;
            v[3]        = v[2];
            v[2]        = v[1];
            v[1]        = v[0];
            v[0]        = t1 + t2;
        }
        for (int i = 0; i < 8; i++) h_[i] += v[i];
    }

    uint32_t h_[8];
    uint8_t buf_[64];
    size_t fill_;
    uint64_t len_;
};
// ------------------------------------------------------------------------------------------ SHA-256 (END)

static std::string sha256_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return "";
    Sha256 s;
    std::vector<char> buf(4 << 20);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = in.gcount();
        if (got > 0)
            s.update(buf.data(), static_cast<size_t>(got));
    }
    return s.hex();
}

static std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// ------------------------------------------------------------------------------------------ positional file writes
// Many threads write different offsets of one file; pwrite() (POSIX) / WriteFile with an explicit offset (Windows)
// does that without a shared file position or a lock.  (The Windows branch is untested.)
class PosFile {
   public:
    PosFile()                          = default;
    PosFile(const PosFile&)            = delete;
    PosFile& operator=(const PosFile&) = delete;
    ~PosFile() { close(); }

    bool open(const fs::path& p) {  // the file must exist
#ifdef _WIN32
        h_ = CreateFileW(p.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        return h_ != INVALID_HANDLE_VALUE;
#else
        fd_ = ::open(p.c_str(), O_WRONLY);
        return fd_ >= 0;
#endif
    }

    bool write_at(const char* buf, size_t n, int64_t off) {
#ifdef _WIN32
        while (n > 0) {
            OVERLAPPED ov{};
            ov.Offset     = static_cast<DWORD>(static_cast<uint64_t>(off) & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(static_cast<uint64_t>(off) >> 32);
            DWORD w = 0, want = static_cast<DWORD>(std::min<size_t>(n, 1u << 30));
            if (!WriteFile(h_, buf, want, &w, &ov))
                return false;
            buf += w;
            n -= w;
            off += w;
        }
        return true;
#else
        while (n > 0) {
            ssize_t w = ::pwrite(fd_, buf, n, static_cast<off_t>(off));
            if (w < 0) {
                if (errno == EINTR)
                    continue;
                return false;
            }
            buf += w;
            n -= static_cast<size_t>(w);
            off += w;
        }
        return true;
#endif
    }

    void close() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE)
            CloseHandle(h_);
        h_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
#endif
    }

   private:
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

// ------------------------------------------------------------------------------------------ HTTP
static void common_opts(CURL* h, const std::string& url) {
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_USERAGENT, "ms-downloader/2.0");
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);  // resolve/ answers with a redirect to the CDN
    curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_FAILONERROR, 1L);  // HTTP >= 400 is an error, its body is not written
}

static size_t str_cb(char* p, size_t sz, size_t n, void* ud) {
    static_cast<std::string*>(ud)->append(p, sz * n);
    return sz * n;
}

static bool http_get(const std::string& url, std::string& body, std::string& err) {
    CURL* h = curl_easy_init();
    if (!h) {
        err = "curl_easy_init failed";
        return false;
    }
    common_opts(h, url);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, str_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
    CURLcode rc = curl_easy_perform(h);
    if (rc != CURLE_OK)
        err = curl_easy_strerror(rc);
    curl_easy_cleanup(h);
    return rc == CURLE_OK;
}

static int64_t head_size(const std::string& url) {  // Content-Length of the file, -1 if unknown
    CURL* h = curl_easy_init();
    if (!h)
        return -1;
    common_opts(h, url);
    curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 60L);
    curl_off_t len = -1;
    if (curl_easy_perform(h) == CURLE_OK)
        curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len);
    curl_easy_cleanup(h);
    return len;
}

// ------------------------------------------------------------------------------------------ ModelScope
struct Meta {
    int64_t size = 0;
    std::string sha256;
};

// (size, sha256) of every file of the repository, as ModelScope publishes them; empty if it cannot be asked.
static std::map<std::string, Meta> fetch_meta(const std::string& ep, const std::string& repo) {
    std::map<std::string, Meta> m;
    std::string body, err;
    if (!http_get(ep + "/api/v1/models/" + repo + "/repo/files?Recursive=true", body, err)) {
        std::cerr << "  [!] cannot read the file list from ModelScope (" << err << "): sizes and SHA-256 are not checked\n";
        return m;
    }
    std::string perr;
    const JSON j = JSON::parse(body, perr);  // const: operator[] below never inserts keys
    if (!perr.empty()) {
        std::cerr << "  [!] unexpected file list from ModelScope (" << perr << ")\n";
        return m;
    }
    for (const JSON& f : j["Data"]["Files"].array_items()) {
        Meta x;
        x.size                      = static_cast<int64_t>(f["Size"].number_value());
        x.sha256                    = lower(f["Sha256"].string_value());
        m[f["Path"].string_value()] = x;
    }
    return m;
}

static std::string ms_url(const std::string& ep, const std::string& repo, const std::string& path) {
    return ep + "/models/" + repo + "/resolve/master/" + path;
}

// ------------------------------------------------------------------------------------------ chunk map
// <file>.part.map:  "v1 <total> <chunk>\n<one '0'/'1' per chunk>\n".  A chunk is marked only after all its bytes were
// written, and the map is replaced atomically (tmp + rename), so a crash or Ctrl-C at any moment loses at most the
// chunks that were in flight.
static bool load_map(const fs::path& p, int64_t total, int64_t chunk, std::vector<char>& done) {
    std::ifstream in(p);
    std::string tag, bits;
    int64_t t = 0, c = 0;
    if (!(in >> tag >> t >> c >> bits))
        return false;
    if (tag != "v1" || t != total || c != chunk || bits.size() != done.size())
        return false;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] != '0' && bits[i] != '1')
            return false;
        done[i] = bits[i] == '1';
    }
    return true;
}

static void save_map(const fs::path& p, int64_t total, int64_t chunk, const std::vector<char>& done) {
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        std::string bits(done.size(), '0');
        for (size_t i = 0; i < done.size(); ++i)
            if (done[i])
                bits[i] = '1';
        out << "v1 " << total << ' ' << chunk << '\n' << bits << '\n';
    }
    std::error_code ec;
    fs::rename(tmp, p, ec);
}

static int64_t chunk_len(int64_t total, int64_t chunk, size_t idx) { return std::min<int64_t>(chunk, total - static_cast<int64_t>(idx) * chunk); }

// ------------------------------------------------------------------------------------------ parallel download
struct Job {
    std::string url;
    PosFile* file = nullptr;
    int64_t total = 0;
    int64_t chunk = 0;
    fs::path mapPath;
    std::vector<char> done;         // guarded by mu
    std::vector<size_t> pending;    // chunk indexes still to fetch
    std::atomic<size_t> next{0};    // next index into `pending`
    std::atomic<int64_t> bytes{0};  // bytes of the file present (finished chunks + progress of running ones)
    std::atomic<int> running{0};    // worker threads still alive
    std::atomic<bool> failed{false};
    std::mutex mu;  // guards done, the map file and err
    std::string err;
};

static void set_error(Job& J, const std::string& msg) {
    std::lock_guard<std::mutex> g(J.mu);
    if (J.err.empty())
        J.err = msg;
    J.failed = true;
}

struct ChunkCtx {
    Job* J;
    CURL* h;
    int64_t start;  // file offset of the chunk
    int64_t len;    // chunk length
    int64_t got;    // bytes of this chunk already written
    bool checked = false;
    bool fatal   = false;  // not worth retrying
    std::string why;
};

static size_t chunk_write_cb(char* p, size_t sz, size_t n, void* ud) {
    auto* c            = static_cast<ChunkCtx*>(ud);
    const size_t bytes = sz * n;
    if (!c->checked) {
        c->checked = true;
        long code  = 0;
        curl_easy_getinfo(c->h, CURLINFO_RESPONSE_CODE, &code);
        // 206 = the Range request was honoured.  A 200 is acceptable only when the "range" is the whole file.
        const bool whole = c->start == 0 && c->got == 0 && c->len == c->J->total;
        if (code != 206 && !(code == 200 && whole)) {
            c->fatal = true;
            c->why =
                "the server answered HTTP " + std::to_string(code) + " to a Range request (it does not support ranges?); try --threads 1 --chunk-mb 1000000";
            return 0;
        }
    }
    if (c->got + static_cast<int64_t>(bytes) > c->len) {
        c->fatal = true;
        c->why   = "the server sent more bytes than the requested range";
        return 0;
    }
    if (!c->J->file->write_at(p, bytes, c->start + c->got)) {
        c->fatal = true;
        c->why   = std::string("write to the output file failed: ") + std::strerror(errno);
        return 0;
    }
    c->got += static_cast<int64_t>(bytes);
    c->J->bytes += static_cast<int64_t>(bytes);
    return bytes;
}

static int abort_cb(void* ud, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<Job*>(ud)->failed.load() ? 1 : 0;  // another worker gave up: stop quickly
}

// Fetches chunk `idx`, retrying (and resuming inside the chunk) on network errors.
static bool fetch_chunk(Job& J, CURL* h, size_t idx) {
    const int64_t start = static_cast<int64_t>(idx) * J.chunk;
    const int64_t len   = chunk_len(J.total, J.chunk, idx);
    int64_t got         = 0;
    std::string lastErr;
    for (int attempt = 0; attempt < kChunkRetries; ++attempt) {
        if (J.failed)
            return false;
        ChunkCtx c{&J, h, start, len, got};
        curl_easy_reset(h);
        common_opts(h, J.url);
        char range[64];
        std::snprintf(range, sizeof(range), "%lld-%lld", static_cast<long long>(start + got), static_cast<long long>(start + len - 1));
        curl_easy_setopt(h, CURLOPT_RANGE, range);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, chunk_write_cb);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, &c);
        curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, abort_cb);
        curl_easy_setopt(h, CURLOPT_XFERINFODATA, &J);
        curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1024L);  // below 1 KiB/s for 30 s: stalled, retry
        curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, 30L);

        CURLcode rc = curl_easy_perform(h);
        got         = c.got;
        if (rc == CURLE_OK && got == len)
            return true;
        if (c.fatal) {
            set_error(J, c.why);
            return false;
        }
        if (J.failed)
            return false;
        lastErr = rc != CURLE_OK ? curl_easy_strerror(rc) : "connection closed early";
        std::fprintf(stderr, "\n  [!] chunk %zu: %s (attempt %d/%d), retrying\n", idx, lastErr.c_str(), attempt + 1, kChunkRetries);
        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(g_backoff_ms * (attempt + 1), 20000)));
    }
    set_error(J, "chunk " + std::to_string(idx) + ": " + lastErr + " (gave up after " + std::to_string(kChunkRetries) + " attempts)");
    return false;
}

static void worker(Job* Jp) {
    Job& J  = *Jp;
    CURL* h = curl_easy_init();
    if (!h)
        set_error(J, "curl_easy_init failed");
    while (h && !J.failed) {
        size_t k = J.next.fetch_add(1);
        if (k >= J.pending.size())
            break;
        size_t idx = J.pending[k];
        if (!fetch_chunk(J, h, idx))
            break;
        std::lock_guard<std::mutex> g(J.mu);
        J.done[idx] = 1;
        save_map(J.mapPath, J.total, J.chunk, J.done);
    }
    if (h)
        curl_easy_cleanup(h);
    J.running--;
}

// Fills <part> (preallocated to `total` bytes) with `threads` parallel Range requests of `chunk` bytes each.
// Returns true when every chunk is on disk.  Resumes from <part>.map when it matches (same size and chunk size).
static bool download_chunked(const std::string& url, const fs::path& part, int64_t total, const std::string& label, int threads, int64_t chunk) {
    fs::path mapPath = part;
    mapPath += ".map";
    const size_t nchunks = static_cast<size_t>((total + chunk - 1) / chunk);

    Job J;
    J.url     = url;
    J.total   = total;
    J.chunk   = chunk;
    J.mapPath = mapPath;
    J.done.assign(nchunks, 0);

    std::error_code ec;
    bool resumed = load_map(mapPath, total, chunk, J.done) && fs::exists(part, ec) && static_cast<int64_t>(fs::file_size(part, ec)) == total;
    if (!resumed) {
        std::fill(J.done.begin(), J.done.end(), 0);
        fs::remove(part, ec);
        {
            std::ofstream create(part, std::ios::binary);  // create, then size it (sparse where the filesystem allows)
            if (!create) {
                std::fprintf(stderr, "  [X] cannot create %s\n", part.string().c_str());
                return false;
            }
        }
        fs::resize_file(part, static_cast<std::uintmax_t>(total), ec);
        if (ec) {
            std::fprintf(stderr, "  [X] cannot size %s to %lld bytes: %s\n", part.string().c_str(), (long long)total, ec.message().c_str());
            return false;
        }
        save_map(mapPath, total, chunk, J.done);
    }

    PosFile file;
    if (!file.open(part)) {
        std::fprintf(stderr, "  [X] cannot open %s for writing\n", part.string().c_str());
        return false;
    }
    J.file = &file;

    int64_t have = 0;
    for (size_t i = 0; i < nchunks; ++i) {
        if (J.done[i])
            have += chunk_len(total, chunk, i);
        else
            J.pending.push_back(i);
    }
    J.bytes = have;
    if (resumed && have > 0)
        std::printf("  %s: resuming, %.2f of %.2f GiB already on disk\n", label.c_str(), have / 1073741824.0, total / 1073741824.0);
    if (J.pending.empty())
        return true;

    const int nthreads = static_cast<int>(std::min<size_t>(static_cast<size_t>(threads), J.pending.size()));
    J.running          = nthreads;
    std::vector<std::thread> pool;
    for (int i = 0; i < nthreads; ++i) pool.emplace_back(worker, &J);

    // progress line, once a second, until every worker is done
    using clock   = std::chrono::steady_clock;
    auto lastT    = clock::now();
    int64_t lastB = J.bytes.load();
    while (J.running.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto now  = clock::now();
        double dt = std::chrono::duration<double>(now - lastT).count();
        if (dt < 1.0)
            continue;
        int64_t cur = J.bytes.load();
        std::printf("\r  %s  %5.1f%%  %.2f / %.2f GiB  %6.1f MB/s  %d threads   ", label.c_str(), 100.0 * cur / total, cur / 1073741824.0, total / 1073741824.0,
                    (cur - lastB) / dt / 1e6, nthreads);
        std::fflush(stdout);
        lastT = now;
        lastB = cur;
    }
    for (auto& t : pool) t.join();
    std::printf("\r  %s  %5.1f%%  %.2f / %.2f GiB                                   \n", label.c_str(), 100.0 * J.bytes.load() / total,
                J.bytes.load() / 1073741824.0, total / 1073741824.0);

    file.close();
    if (J.failed) {
        std::fprintf(stderr, "  [X] %s: %s\n", label.c_str(), J.err.c_str());
        return false;
    }
    return true;
}

//"  host  ModelScope host (default $MODELSCOPE_ENDPOINT or https://www.modelscope.cn)\n"
int DownloadGGUF(CSTRING& kRepo, std::vector<std::string>&paths, CSTRING& quant, fs::path outDir, int threads, int chunkMb, bool verify) {
    std::string www = std::getenv("MODELSCOPE_ENDPOINT") ? std::getenv("MODELSCOPE_ENDPOINT") : kDefaultEp;
    while (!www.empty() && www.back() == '/') www.pop_back();
    
    threads                  = std::max(1, std::min(threads, 64));
    chunkMb                  = std::max(1, std::min(chunkMb, 4096));
    const int64_t chunkBytes = static_cast<int64_t>(chunkMb) << 20;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    int rc = 0;
    do {
        // <quant>/Qwen3.8-Flash-Next-GSQ-RCO-<quant>-00001-of-00002.gguf, ...-00002-of-00002.gguf
        std::error_code ec;
        fs::create_directories(outDir, ec);
        if (ec) {
            std::cerr << "cannot create " << outDir << ": " << ec.message() << "\n";
            rc = 1;
            break;
        }

        std::printf("ModelScope: %s/models/%s (%s), %d threads x %d MB chunks\n", www.c_str(), kRepo.c_str(), quant.c_str(), threads, chunkMb);
        auto meta = fetch_meta(www, kRepo);

        // free disk space for files not started yet (a half-done file's space was checked when it was started)
        int64_t need = 0;
        for (auto& p : paths) {
            auto it = meta.find(p);
            if (it == meta.end())
                continue;
            fs::path dst  = outDir / fs::path(p).filename();
            fs::path part = dst;
            part += ".part";
            if (!fs::exists(dst, ec) && !fs::exists(part, ec))
                need += it->second.size;
        }
        if (need > 0) {
            auto sp = fs::space(outDir, ec);
            if (!ec && static_cast<int64_t>(sp.available) < need) {
                _ERROR( "No enough free disk space: need %.1f GB, have %.1f GB in %s\n", need / 1e9, sp.available / 1e9,outDir.string().c_str());
                rc = 1;
                break;
            }
        }

        for (auto& p : paths) {
            fs::path dst  = outDir / fs::path(p).filename();
            fs::path part = dst;
            part += ".part";
            fs::path mark = dst;
            mark += ".done";
            fs::path pmap = part;
            pmap += ".map";
            std::string name = dst.filename().string();

            if (fs::exists(dst, ec) && fs::exists(mark, ec)) {
                std::printf("  [ok] %s already downloaded\n", name.c_str());
                continue;
            }

            std::string url = ms_url(www, kRepo, p);
            auto it         = meta.find(p);
            int64_t total   = (it != meta.end() && it->second.size > 0) ? it->second.size : head_size(url);
            if (total <= 0) {
                std::fprintf(stderr, "  [X] %s: cannot find out the file size (needed to split it into chunks)\n", name.c_str());
                rc = 1;
                break;
            }

            if (!download_chunked(url, part, total, name, threads, chunkBytes)) {
                std::fprintf(stderr, "  [X] %s: download failed; run again to resume\n", name.c_str());
                rc = 1;
                break;
            }

            // verify size, then SHA-256, against what ModelScope publishes
            int64_t got = static_cast<int64_t>(fs::file_size(part, ec));
            if (got != total) {
                std::fprintf(stderr, "  [X] %s: size %lld, expected %lld\n", name.c_str(), (long long)got, (long long)total);
                fs::remove(part, ec);
                fs::remove(pmap, ec);
                rc = 1;
                break;
            }
            if (verify && it != meta.end() && !it->second.sha256.empty()) {
                std::printf("  verifying SHA-256 of %s ...\n", name.c_str());
                std::string h = sha256_file(part);
                if (h != it->second.sha256) {
                    std::fprintf(stderr, "  [X] %s: SHA-256 mismatch (got %s, expected %s); deleted, run again\n", name.c_str(), h.c_str(),
                                 it->second.sha256.c_str());
                    fs::remove(part, ec);
                    fs::remove(pmap, ec);
                    rc = 1;
                    break;
                }
            }

            fs::rename(part, dst, ec);
            if (ec) {
                std::cerr << "  [X] cannot rename " << part << ": " << ec.message() << "\n";
                rc = 1;
                break;
            }
            fs::remove(pmap, ec);
            std::ofstream(mark) << "ok\n";
            std::printf("  [ok] %s\n", name.c_str());
        }
        if (rc == 0)
            std::printf("Done. Shards are in %s\n", outDir.string().c_str());
    } while (false);

    curl_global_cleanup();
    return rc;
}
