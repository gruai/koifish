/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Serialization with many check
 *
 *  \brief Serialization
 *  \author Yingshi Chen
 */
#pragma once
#include <fcntl.h>
#include <stdlib.h>    // mkstemp
#include <sys/stat.h>  // fchmod
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../CLI_params.hpp"
#include "../Utils/GST_obj.hpp"

/**
 * Abstract class of file serialization
 * 1.  write to a temporary file in the same directory, then atomically rename it to the
 *
 */
namespace fs = std::filesystem;
class FSerial {
    FILE* _stream        = NULL;
    bool _valid          = false;
    std::string tmp_path = ".tmpXXXXXX", path;
    std::string dir      = ".";  //  Create temp file in the same directory as the target.
    int fd               = 0x0;

    template <typename T>
    void _fd_write_(const T* data0, size_t count, int flag = 0x0) {
        static_assert(std::is_trivially_copyable_v<T>, "raw byte I/O requires a trivially copyable type");
        const char* data = reinterpret_cast<const char*>(data0);
        size_t len       = sizeof(T) * count;
        while (len > 0) {
            ssize_t n = ::write(fd, data, len);
            if (n < 0) {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error(std::string("write: ") + std::strerror(errno));
            }
            data += n;
            len -= static_cast<size_t>(n);
        }
    }
    void _fd_flush_() {
        try {
            if (::fchmod(fd, 0644) != 0)
                throw std::runtime_error(std::string("fchmod: ") + std::strerror(errno));

            // Flush file data + metadata to disk.
            if (::fsync(fd) != 0)
                throw std::runtime_error(std::string("fsync: ") + std::strerror(errno));

            if (::close(fd) != 0) {  // close can report deferred errors
                fd = -1;
                throw std::runtime_error(std::string("close: ") + std::strerror(errno));
            }
            fd = -1;

            // Atomically replace the target.
            if (::rename(tmp_path.c_str(), path.c_str()) != 0)
                throw std::runtime_error(std::string("rename: ") + std::strerror(errno));
        } catch (...) {
            if (fd >= 0)
                ::close(fd);
            ::unlink(tmp_path.c_str());  // clean up temp file on failure
            throw;
        }

        // fsync the directory so the rename itself is durable.
        int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
        if (dfd >= 0) {
            ::fsync(dfd);
            ::close(dfd);
        }
    }

   public:
    enum STATUS {
        INIT_MMAP = 0x100,
        COPY_MMAP = 0x200  // copy mmap file of HF model to checkpoint
    };

    FSerial(const std::string& sPath, bool isSave, int flag) {
        try {
            if (isSave) {
                /*if ((_stream = fopen(sPath.c_str(), "wb")) != NULL) {
                    _valid = true;
                }*/
                path = sPath;
                if (auto pos = path.find_last_of('/'); pos != std::string::npos)
                    dir = (pos == 0) ? "/" : path.substr(0, pos);

                tmp_path = path + ".tmpXXXXXX";
                fd       = ::mkstemp(tmp_path.data());  // unique name, mode 0600, O_EXCL
                if (fd < 0)
                    throw std::runtime_error(std::string("mkstemp: ") + std::strerror(errno));
                _valid = true;
            } else {
                if ((_stream = fopen(sPath.c_str(), "rb")) != NULL) {
                    _valid = true;
                }
            }

        } catch (...) {
            _valid = false;
        }
    }

    virtual ~FSerial() {
        try {
            if (fd > 0x0) {
                _fd_flush_();
            }
            if (_stream != NULL)
                fclose(_stream);
        } catch (...) {
        }
    }

    virtual bool isValid() { return _valid; }

    template <typename T>
    bool Serial(T* val, int nz, bool isSave, int flag = 0x0) {
        if (!isValid())
            return false;
        if (isSave) {
            // if (fwrite(val, sizeof(T), nz, _stream) != nz)
            //     return false;
            // fflush(_stream);
            _fd_write_(val, nz);
        } else {
            if (fread(val, sizeof(T), nz, _stream) != nz)
                return false;
        }
        return true;
    }

    template <typename T>
    bool Serial(T& val, bool isSave, int flag = 0x0) {
        return Serial(&val, 1, isSave, flag);
    }

    template <typename T>
    bool Serial(std::vector<T>& arrT, bool isSave, int flag = 0x0) {
        if (!isValid())
            return false;
        size_t nT = arrT.size(), i;
        Serial(&nT, 1, isSave);
        if (nT == 0) {
            return true;
        }
        if (isSave) {
            // if (fwrite(arrT.data(), sizeof(T), nT, _stream) != nT)
            //     return false;
            // fflush(_stream);
            _fd_write_(arrT.data(), nT);
        } else {
            arrT.clear();
            arrT.resize(nT);
            if (fread(arrT.data(), sizeof(T), nT, _stream) != nT)
                return false;
            // T* buf = new T[nT];
            // size_t nRead = fread((void*)(buf),sizeof(T),nT,_stream);
            // if(nRead!=nT)
            //     return false;
            // std::copy(buf, buf + nT, std::back_inserter(arrT));
            // delete[] buf;
        }
        return true;
    }

    template <typename T, typename Tchild>
    bool Serial_Vector(std::vector<T*>& arrT, bool isSave, int flag = 0x0) {
        if (!isValid())
            return false;
        size_t nT = arrT.size(), i;
        Serial(&nT, 1, isSave);
        if (isSave) {
            for (auto obj0 : arrT) {
                Tchild* obj = dynamic_cast<Tchild*>(obj0);
                assert(obj != nullptr);
                if (!obj->Serialize(*this, isSave, flag))
                    return false;
            }
        } else {
            arrT.clear();
            arrT.resize(nT);
            for (i = 0; i < nT; i++) {
                Tchild* obj = new Tchild();
                if (!obj->Serialize(*this, isSave, flag))
                    return false;
                arrT[i] = obj;
            }
        }
        return true;
    }

    template <typename T, typename Tchild>
    bool Serial_hVector(std::vector<std::shared_ptr<T>>& arrT, bool isSave, int flag = 0x0) {
        if (!isValid())
            return false;
        size_t nT = arrT.size(), i;
        Serial(&nT, 1, isSave);
        if (isSave) {
            for (auto obj0 : arrT) {
                Tchild* obj = dynamic_cast<Tchild*>(obj0.get());
                assert(obj != nullptr);
                if (!obj->Serialize(*this, isSave, flag))
                    return false;
            }
        } else {
            arrT.clear();
            arrT.resize(nT);
            for (i = 0; i < nT; i++) {
                auto obj = std::make_shared<Tchild>();
                if (!obj->Serialize(*this, isSave, flag))
                    return false;
                arrT[i] = obj;
            }
        }
        return true;
    }
};

// only load jConfig from safeternsor model
bool SAFETENSOR_Load_jconfig(const std::string& path, JSON& jsConfig, FILE_FORMAT_TYPE tpFile, int flag = 0x0);
std::string LoadSomeText(const string& fpath, const int nMost = 2048, int flag = 0x0);
hBITARR SAFE_read_mmap(hBITARR dst, hBITARR mmp_data, size_t length, int flag = 0x0);

/**
 * Why not use O_TMPFILE(int fd = ::open(dir.c_str(), O_TMPFILE | O_WRONLY, 0644)?
 * It’s harder to read and debug.   O_TMPFILE is Linux-only (kernel 3.11+). It doesn’t exist on macOS, BSD, or Windows. mkstemp works on every POSIX system.
 */