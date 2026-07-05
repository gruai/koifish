/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *   hf-transformers compatible SafeTensors
 *
 *  Inspired from: https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../CLI_params.hpp"
#include "../Utils/GST_obj.hpp"
#include "./GTensor.hpp"

struct K_SafeTensors;

constexpr size_t kMaxDim = 8;  // must be equal to SAFETENSORS_C_MAX_DIM in `safetensors-c.h`
/**
 *  The HF's safetensors format cannot add custom fields directly to Tensor Entries. But Koifish would add many detial info(for example, quant method & bit).
    So Koifish format is not compliant with HF's safetensors format!

 *  1. HF's safetensors format
 *  a) Tensor Entry Structure
        Each key (e.g., "layer1.weight") maps to an object with these mandatory​ fields:
            dtype​("F16", "BF16", "I32"),shape​,data_offsets([start, end])
        Important:​ data_offsets are relative to the start of the data buffer (the section after the header), not the start of the file.
    b). __metadata__Section
        This is an optional key containing a dictionary of string-to-string​ pairs. There is no enforced schema, but common conventions include:
            format: Framework origin (e.g., "pt"for PyTorch).
            description: Human-readable model info.
            Custom keys: Any other metadata (e.g., "author", "version").
            Restriction:​ All values under __metadata__must be strings.

    2
*/
// using tensor_st = GTensor;

class K_SafeTensors {
   protected:
    int posix_fd = 0x0;

   public:
    CheckPoint_Params ckp;
    std::string header_str, sFolderPath;
    // GST_Dict(preserves the order of key insertion) is much simple than JSON
    GST_Dict<hGTensor> tensors;
    JSON metadata, jHeader;
    JSON jsConfig;

    Fish* hFish = nullptr;
    // std::vector<uint8_t> storage;  // would explode!     empty when mmap'ed
    size_t header_size{0};  // JSON size
    static string config_key_;
    bool mmaped{false};
    virtual void Clear(int flag = 0x0) {
        tensors.Clear();
        metadata.clear();
        jsConfig.clear();
        // storage.clear();
        header_size = 0x0;
    }
    virtual void UpdateMetaData(int flag = 0x0);

    K_SafeTensors(Fish* fish, CheckPoint_Params ckp_, const std::string& path, int flag = 0x0);
    //
    // Following members are set when mmaped.
    //
    const uint8_t* mmap_addr{nullptr};
    size_t mmap_size{0};
    const uint8_t* databuffer_addr{nullptr};  // [mmap_addr + header_size + 8]
    size_t databuffer_size{0};                // mmap_size - header_size - 8
    // opaque pointer to safetensors_file and safetensors_mmap
    void* st_file{nullptr};
    void* st_mmap{nullptr};

    size_t Register(shared_ptr<GTensor> t, size_t offset, FILE_FORMAT_TYPE format, int flag = 0x0);
    bool InitHeader(int flag = 0x0);
    void insertJS(const JSON& js, size_t dst_offset, int flag = 0x0) {
        jsConfig        = js;
        hGTensor tensor = std::make_shared<GTensor>();
        strcpy(tensor->name, config_key_.c_str());
        tensor->hFish           = hFish;
        tensor->msgpack         = JSON::to_msgpack(js);
        tensor->type            = typNUMBER::U8;           // dtype::kUINT8;
        size_t sz               = tensor->msgpack.size();  // storage.size();
        tensor->data_offsets[0] = dst_offset;
        tensor->data_offsets[1] = dst_offset + sz;
        tensor->shape           = {(int)sz};
        tensors.insert(config_key_, tensor);

        // storage.resize(dst_offset + sz);
        // memcpy(storage.data() + dst_offset, msgpack.data(), sz);  //  6441  [132...160]
    }
    int loadJS(hGTensor tensor, const uint8_t* bytes_ptr, size_t bytes_size, int flag = 0x0) {
        /*JSON jdesc = tensor.jDesc();
        if (jdesc.at("data_offsets").size() != 2) {
            return -3;
        }
        size_t offset_start = static_cast<size_t>(jdesc.at("data_offsets")[0]);  // 1544148992
        size_t offset_end   = static_cast<size_t>(jdesc.at("data_offsets")[1]);  // 1545276932*/
        size_t offset_start = tensor->data_offsets[0];
        size_t offset_end   = tensor->data_offsets[1];
        if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
            std::cerr << "bad offsets" << std::endl;
            return -1;
        }
        size_t szSrc = offset_end - offset_start;
        std::vector<uint8_t> msgpack;
        msgpack.resize(szSrc);
        memcpy(msgpack.data(), bytes_ptr + offset_start, szSrc);
        jsConfig = JSON::from_msgpack(msgpack);
        return 0x0;
    }

    bool initJS(int flag = 0x0) {
        try {
            const uint8_t* databuffer = nullptr;
            if (mmaped) {
                databuffer = databuffer_addr;  // st->mmap_addr + 8 + st->header_size;
            } else {
                assert(0);  // only mmap is support LLM
                // databuffer = storage.data();
            }
            hGTensor tensor = nullptr;
            if (!tensors.at(config_key_, tensor))  // may has no "__koifish__config__"
                return false;

            size_t start = tensor->data_offsets[0], end = tensor->data_offsets[1];
            // assert(end<nbytes);
            std::vector<uint8_t> msgpack = {databuffer + start, databuffer + end};
            size_t sz                    = msgpack.size();
            jsConfig                     = JSON::from_msgpack(msgpack);
            return true;
        } catch (JSON::parse_error& e) {
            _INFO("\r\n%s  Failed to initJS from safetensor[]!!! ERR=%s", __func__, e.what());
            return false;
        } catch (...) {
            return false;
        }
    }

    ~K_SafeTensors();

    // Validate data_offsets of all tensors in K_SafeTensors.
    bool validate_data_offsets(std::string& err) {
        bool valid{true};

        std::stringstream ss;
        size_t databuffersize;
        if (mmaped) {
            databuffersize = databuffer_size;
        } else {
            assert(0);
            // databuffersize = storage.size();
        }

        size_t ntensors{0};
        // Iterate with key insertion order.
        for (size_t i = 0; i < tensors.size(); i++) {
            std::string key = tensors.keys()[i];

            hGTensor tensor = tensors.at(i);
            // if (!tensors.at(i, &tensor)) {
            //     ss << "Internal error: Failed to get stensor at [" << i << "]\n";
            //     valid = false;
            //     continue;
            // }
            if (tensor->data_offsets[0] > tensor->data_offsets[1]) {
                ss << key << ".data_offsets.BEGIN " << tensor->data_offsets[0] << " must be less than or equal to data_offsets.END " << tensor->data_offsets[1]
                   << "\n";
                valid = false;
            }
            // size_t tensor_size = tensor->nByte();
            size_t tensor_size = tensor->nByte_CKP(ckp);
            if (tensor_size == 0) {  //  ???
                continue;
            }

            // data_offsets are absolute offset from the databuffer(file)
            if (tensor->data_offsets[0] > databuffersize) {
                ss << "Tensor `" << key << "`.data_offset.BEGIN " << tensor->data_offsets[0] << " exceeds databuffer size " << databuffersize << ".\n";
                valid = false;
            }

            if (tensor->data_offsets[1] > databuffersize) {
                ss << "Tensor `" << key << "`.data_offset.END " << tensor->data_offsets[1] << " exceeds databuffer size " << databuffersize << ".\n";
                valid = false;
            }

            size_t data_size = tensor->data_offsets[1] - tensor->data_offsets[0];

            if (tensor_size < data_size && data_size % tensor_size != 0) {  // [data,gm,gv]
                ss << "Data size mismatch. The size in Tensor `" << key << "` is " << tensor_size << ", but the size from data_offsets is " << data_size
                   << "\n";
                valid = false;
            }

            ntensors++;
            if (ntensors == tensors.size()) {
                // Last element's data_offsets[1] must be equal to databuffer size.
                if (tensor->data_offsets[1] != databuffersize) {
                    ss << "The last tensor's data_offset.END(" << tensor->data_offsets[1] << ") must be equal to databufer size " << databuffersize << ".\n";
                    valid = false;
                }
            }
        }

        if (!valid) {
            err = ss.str();
        }

        return valid;
    }

    bool _to_ofs(size_t& szAll, std::string* warn, std::string* err, int flag);
    bool _to_memory(std::vector<uint8_t>& buffer, std::string* warn, std::string* err);

    // Save safetensors to file,    return true upon success. `err` will be filled when false.
    bool Save(const std::string& filename, size_t& sz, std::string* warn, std::string* err, int flag);

    bool MMAP(const std::string& path, bool isSave = false, int flag = 0x0);
};

//
// Load safetensors from file.
// databuffer is copied to `K_storage`.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool load_from_file(const std::string& filename, K_SafeTensors* st, std::string* warn, std::string* err);

//
// Load safetensors data from memory.
// databuffer is copied to `K_storage`.
//
// @param[in] addr Memory address of safetensors data.
// @param[in] nbytes The size in bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
//
bool load_from_memory(const uint8_t* addr, const size_t nbytes, const std::string& filename, K_SafeTensors* st, std::string* warn, std::string* err);

//
// Load safetensors with memory mapping(i.e. zero-copy).
// databuffer is not copied to `K_SafeTensors` object, thus the app must hold
// file during `safetensor_t` object is live.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_file(const std::string& filename, K_SafeTensors* st, std::string* warn, std::string* err);

//
// Load safetensors from mmaped region.
// databuffer is not copied to `K_SafeTensors` object, thus the app must not
// free/unmap `addr` during `safetensor_t` object is live.
//
// @param[in] addr mmaped memory address of safetensors data.
// @param[in] nbytes mmap bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_memory(const uint8_t* arr, const size_t nbytes, const std::string& filename, K_SafeTensors* st, std::string* warn, std::string* err);

#if defined(SAFETENSORS_CPP_IMPLEMENTATION)

#include <cstring>
#include <fstream>
#include <memory>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h>  // for _fseeki64
#include <windows.h>
#endif

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// #define __MINIJSON_LIBERAL

// We recommended to use simdjson from_chars.
// Using strtod() is a fallback
#if defined(MINIJSON_USE_STRTOD)
// Use stdlib's strtod
#include <cstring>
#else

namespace minijson { namespace simdjson { namespace internal {

double from_chars(const char* first) noexcept;
double from_chars(const char* first, const char* end) noexcept;

char* to_chars(char* first, const char* last, double value);

}}}  // namespace minijson::simdjson::internal

#endif

namespace minijson {

namespace detail {

double from_chars(const char* p);
const char* my_strchr(const char* p, int ch);

}  // namespace detail

namespace detail {

//
// Usage:
//  - set_input()
//  - scan_string()
//    - success: use `token_buffer` string
//    - error: use `error_message`
//
struct string_parser {
    // input string must be UTF-8
    void set_input(const std::string& s) { _input = s; }

    bool scan_string();

    void reset() {
        if (_input.size()) {
            current = _input[0];
        } else {
            current = '\0';
        }
        curr_idx = 0;
        token_buffer.clear();
    }

    // fetch next token.
    unsigned char get() {
        if ((curr_idx + 1) < _input.size()) {
            curr_idx++;
            current = _input[curr_idx];
            return current;
        }
        current = '\0';
        return current;
    }

    bool eof() {
        if (_input.empty()) {
            return true;
        }

        if (curr_idx >= _input.size()) {
            return true;
        }

        return false;
    }

    void add(const unsigned char c) { token_buffer += c; }

    void add(const int i) {
        // use lower 8bit
        token_buffer += static_cast<unsigned char>(i & 0xff);
    }

    int get_codepoint();

    bool next_byte_in_range(const std::initializer_list<int> ranges);

    std::string error_message;
    std::string token_buffer;  // output

    unsigned char current{'\0'};
    size_t curr_idx{0};
    std::string _input;
};

}  // namespace detail

typedef enum {
    unknown_type,
    null_type,
    boolean_type,
    number_type,
    string_type,
    array_type,
    object_type,
} type;

typedef enum {
    no_error,
    undefined_error,
    invalid_token_error,
    unknown_type_error,
    memory_allocation_error,
    corrupted_json_error,
    duplicated_key_error,
} error;

class value;

typedef bool boolean;
typedef double number;
typedef std::string string;
typedef GST_Dict<value> object;
typedef std::vector<value> array;
typedef struct {
} null_t;

// null_t null;

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<null_t> {
    static constexpr uint32_t type_id() { return 0; }
};

template <>
struct TypeTraits<boolean> {
    static constexpr uint32_t type_id() { return 1; }
};

template <>
struct TypeTraits<number> {
    static constexpr uint32_t type_id() { return 2; }
};

template <>
struct TypeTraits<string> {
    static constexpr uint32_t type_id() { return 3; }
};

template <>
struct TypeTraits<object> {
    static constexpr uint32_t type_id() { return 4; }
};

template <>
struct TypeTraits<array> {
    static constexpr uint32_t type_id() { return 5; }
};

class value {
   private:
    type t;
    union {
        null_t n;
        boolean b;
        number d;
        std::string* s;
        array* a;
        object* o;
    } u;

    void _free_u() {
        if (t == string_type) {
            delete this->u.s;
            this->u.s = nullptr;
        }
        if (t == array_type) {
            delete this->u.a;
            this->u.a = nullptr;
        }
        if (t == object_type) {
            delete this->u.o;
            this->u.o = nullptr;
        }
    }

   public:
    value() : t(unknown_type), u() {}
    value(null_t n) : t(null_type), u() { u.n = n; }
    value(boolean b) : t(boolean_type), u() { u.b = b; }
    value(number d) : t(boolean_type), u() { u.d = d; }
    value(const char* s) : t(string_type), u() { u.s = new std::string(s); }
    value(const std::string& s) : t(string_type), u() { u.s = new std::string(s); }
    value(const array& a) : t(array_type), u() { u.a = new array(a); }
    value(const object& o) : t(object_type), u() { u.o = new object(o); }
    value(const value& v) : t(v.t), u() {
        if (t == array_type) {
            u.a  = new array();
            *u.a = *v.u.a;
        } else if (t == object_type) {
            u.o  = new object();
            *u.o = *v.u.o;
        } else if (t == string_type) {
            u.s  = new std::string();
            *u.s = *v.u.s;
        } else
            u.d = v.u.d;
    }
    ~value() { _free_u(); }

    template <typename T>
    bool is() const {
        if (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id() && t == null_type)
            return true;
        if (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id() && t == boolean_type)
            return true;
        if (TypeTraits<T>::type_id() == TypeTraits<number>::type_id() && t == number_type)
            return true;
        if (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id() && t == string_type)
            return true;
        if (TypeTraits<T>::type_id() == TypeTraits<array>::type_id() && t == array_type)
            return true;
        if (TypeTraits<T>::type_id() == TypeTraits<object>::type_id() && t == object_type)
            return true;
        return false;
    }

    template <typename T>
    const T* as() const {
        if ((t == array_type) && (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
            return reinterpret_cast<const T*>(u.a);
        }

        if ((t == object_type) && (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
            return reinterpret_cast<const T*>(u.o);
        }

        if ((t == string_type) && (TypeTraits<T>::type_id() == TypeTraits<std::string>::type_id())) {
            return reinterpret_cast<const T*>(u.s);
        }

        if ((t == null_type) && (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
            return reinterpret_cast<const T*>(&u.n);
        }

        if ((t == boolean_type) && (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
            return reinterpret_cast<const T*>(&u.b);
        }

        if ((t == number_type) && (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
            return reinterpret_cast<const T*>(&u.d);
        }

        return nullptr;
    }

    template <typename T>
    T* as() {
        if ((t == array_type) && (TypeTraits<T>::type_id() == TypeTraits<array>::type_id())) {
            return reinterpret_cast<T*>(u.a);
        }

        if ((t == object_type) && (TypeTraits<T>::type_id() == TypeTraits<object>::type_id())) {
            return reinterpret_cast<T*>(u.o);
        }

        if ((t == string_type) && (TypeTraits<T>::type_id() == TypeTraits<string>::type_id())) {
            return reinterpret_cast<T*>(u.s);
        }

        if ((t == null_type) && (TypeTraits<T>::type_id() == TypeTraits<null_t>::type_id())) {
            return reinterpret_cast<T*>(&u.n);
        }

        if ((t == boolean_type) && (TypeTraits<T>::type_id() == TypeTraits<boolean>::type_id())) {
            return reinterpret_cast<T*>(&u.b);
        }

        if ((t == number_type) && (TypeTraits<T>::type_id() == TypeTraits<number>::type_id())) {
            return reinterpret_cast<T*>(&u.d);
        }

        return nullptr;
    }

    null_t& operator=(null_t& n) {
        t   = null_type;
        u.n = n;
        return u.n;
    }
    boolean& operator=(boolean b) {
        t   = boolean_type;
        u.b = b;
        return u.b;
    }
    number& operator=(number d) {
        t   = number_type;
        u.d = d;
        return u.d;
    }
    const std::string& operator=(const char* s) {
        _free_u();
        t   = string_type;
        u.s = new std::string(s);
        return *u.s;
    }
    const std::string& operator=(const std::string& s) {
        _free_u();
        t   = string_type;
        u.s = new std::string(s);
        return *u.s;
    }
    const object& operator=(const object& o) {
        _free_u();
        t   = object_type;
        u.o = new object(o);
        return *u.o;
    }
    const array& operator=(const array& a) {
        _free_u();
        t   = array_type;
        u.a = new array(a);
        return *u.a;
    }
    const value& operator=(const value& v) {
        _free_u();
        t = v.t;
        if (t == array_type) {
            u.a = new array(*v.u.a);
        } else if (t == object_type) {
            u.o = new object(*v.u.o);
        } else if (t == string_type) {
            u.s = new std::string(*v.u.s);
        } else
            u.d = v.u.d;
        return *this;
    }

    std::string type_name() const {
        if (t == array_type) {
            return "array";
        }

        if (t == object_type) {
            return "object";
        }

        if (t == string_type) {
            return "string";
        }

        if (t == null_type) {
            return "null";
        }

        if (t == boolean_type) {
            return "boolean";
        }

        if (t == number_type) {
            return "number";
        }

        return "[[invalid]]";
    }

    std::string str(const char* p) const {
        std::stringstream ss;
        ss << '"';
        while (*p) {
            if (*p == '\n') {
                ss << "\\n";
            } else if (*p == '\r') {
                ss << "\\r";
            } else if (*p == '\t') {
                ss << "\\t";
            } else if (detail::my_strchr("\"", *p)) {
                ss << "\\" << *p;
            } else {
                ss << *p;
            }
            p++;
        }
        ss << '"';
        return ss.str();
    }

    /*std::string str() const {
        std::stringstream ss;
        if (t == unknown_type) {
            ss << "undefined";
        } else if (t == null_type) {
            ss << "null";
        } else if (t == boolean_type) {
            ss << (u.b ? "true" : "false");
        } else if (t == number_type) {
            ss << double(u.d);
        } else if (t == string_type) {
            ss << str(u.s->c_str());
        } else if (const array* pa = as<array>()) {
            array::const_iterator i;
            ss << "[";
            // array a = get<array>();
            for (i = pa->begin(); i != pa->end(); i++) {
                if (i != pa->begin())
                    ss << ", ";
                ss << i->str();
            }
            ss << "]";
        } else if (auto po = as<object>()) {
            // object::const_iterator i;
            ss << "{";
            // object o = get<object>();
            for (size_t i = 0; i < po->size(); i++) {
                if (i > 0)
                    ss << ", ";
                ss << "\"" << po->keys()[i] << "\"";

                value v;
                if (po->at(i, &v)) {
                    ss << ": " << v.str();
                } else {
                    // TODO: report error
                    ss << ": null";
                }
            }
            ss << "}";
        }
        return ss.str();
    }*/
};

#define MINIJSON_SKIP(i)                             \
    while (*i && detail::my_strchr("\r\n \t", *i)) { \
        i++;                                         \
    }

template <typename Iter>
inline error parse_object(Iter& i, value& v) {
    object o;
    i++;
    MINIJSON_SKIP(i)
    if (!(*i)) {
        return corrupted_json_error;
    }
    if (*i != '\x7d') {
        while (*i) {
            value vk, vv;
            error e = parse_string(i, vk);
            if (e != no_error)
                return e;
            MINIJSON_SKIP(i)
            if (!(*i)) {
                return corrupted_json_error;
            }
            if (*i != ':')
                return invalid_token_error;
            i++;
            e = parse_any(i, vv);
            if (e != no_error)
                return e;

            auto ps = vk.as<std::string>();
            if (!ps) {
                return unknown_type_error;
            }

            if (o.count(*ps)) {
                return duplicated_key_error;
            }
            o.insert(*ps, vv);

            MINIJSON_SKIP(i)
            if (!(*i)) {
                return corrupted_json_error;
            }
            if (*i == '\x7d')
                break;
            if (*i != ',')
                return invalid_token_error;
            i++;
            MINIJSON_SKIP(i)
            if (!(*i)) {
                return corrupted_json_error;
            }
#ifdef __MINIJSON_LIBERAL
            if (*i == '\x7d')
                break;
#endif
        }
    }
    v = value(o);
    i++;
    return no_error;
}

template <typename Iter>
inline error parse_array(Iter& i, value& v) {
    array a;
    i++;
    MINIJSON_SKIP(i)
    if (!(*i)) {
        return corrupted_json_error;
    }
    if (*i != ']') {
        while (*i) {
            value va;
            error e = parse_any(i, va);
            if (e != no_error)
                return e;
            a.push_back(va);
            MINIJSON_SKIP(i)
            if (!(*i)) {
                return corrupted_json_error;
            }
            if (*i == ']')
                break;
            if (*i != ',')
                return invalid_token_error;
            i++;
            MINIJSON_SKIP(i)
            if (!(*i)) {
                return corrupted_json_error;
            }
#ifdef __MINIJSON_LIBERAL
            if (*i == '\x7d')
                break;
#endif
        }
    }
    v = value(a);
    i++;
    return no_error;
}

template <typename Iter>
inline error parse_null(Iter& i, value& v) {
    Iter p = i;
    if (*i == 'n' && *(i + 1) == 'u' && *(i + 2) == 'l' && *(i + 3) == 'l') {
        i += 4;
        v = null_t();
    }
    if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
        i = p;
        return undefined_error;
    }
    return no_error;
}

template <typename Iter>
inline error parse_boolean(Iter& i, value& v) {
    Iter p = i;
    if (*i == 't' && *(i + 1) == 'r' && *(i + 2) == 'u' && *(i + 3) == 'e') {
        i += 4;
        v = static_cast<boolean>(true);
    } else if (*i == 'f' && *(i + 1) == 'a' && *(i + 2) == 'l' && *(i + 3) == 's' && *(i + 4) == 'e') {
        i += 5;
        v = static_cast<boolean>(false);
    }
    if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
        i = p;
        return undefined_error;
    }
    return no_error;
}

template <typename Iter>
inline error parse_number(Iter& i, value& v) {
    Iter p = i;

    if (*i == '-') {
        i++;
    }

#define MINIJSON_IS_NUM(x) ('0' <= x && x <= '9')
#define MINIJSON_IS_ALNUM(x) (('0' <= x && x <= '9') || ('a' <= x && x <= 'f') || ('A' <= x && x <= 'F'))
    if (*i == '0' && *(i + 1) == 'x' && MINIJSON_IS_ALNUM(*(i + 2))) {
        i += 3;
        while (MINIJSON_IS_ALNUM(*i)) i++;
        v = static_cast<number>(detail::from_chars(p));
    } else {
        while (MINIJSON_IS_NUM(*i)) i++;
        if (*i == '.') {
            i++;
            if (!MINIJSON_IS_NUM(*i)) {
                i = p;
                return invalid_token_error;
            }
            while (MINIJSON_IS_NUM(*i)) i++;
        }
        if (*i == 'e') {
            i++;
            if (!MINIJSON_IS_NUM(*i)) {
                i = p;
                return invalid_token_error;
            }
            while (MINIJSON_IS_NUM(*i)) i++;
        }
        v = static_cast<number>(detail::from_chars(p));
    }
    if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
        i = p;
        return invalid_token_error;
    }
    return no_error;
}

template <typename Iter>
inline error parse_string(Iter& i, value& v) {
    if (*i != '"')
        return invalid_token_error;

    Iter s = i;
    char t = *i++;  // = '"'
    Iter p = i;

#if 0
  std::stringstream ss;
  while (*i && *i != t) {
    if (*i == '\\' && *(i + 1)) {
      i++;
      if (*i == 'n')
        ss << "\n";
      else if (*i == 'r')
        ss << "\r";
      else if (*i == 't')
        ss << "\t";
      else
        ss << *i;
    } else {
      ss << *i;
    }
    i++;
  }
#else
    // read until '"'
    while (*i && *i != t) {
        if (*i == '\\' && *(i + 1)) {
            i++;
        }
        i++;
    }

#endif
    if (!*i)
        return invalid_token_error;
    if (i < p) {
        return corrupted_json_error;
    }

#if 0
  v = std::string(p, size_t(i - p));

  i++;
  if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
    i = p;
    return invalid_token_error;
  }

#else

    i++;
    if (*i && nullptr == detail::my_strchr(":,\x7d]\r\n ", *i)) {
        i = p;
        return invalid_token_error;
    }

    // include first and last '"' char
    std::string buf(s, size_t(i - s));

    detail::string_parser str_parser;
    str_parser.set_input(buf);

    if (!str_parser.scan_string()) {
        // TODO: error message
        // str_parser.error_message;
        return invalid_token_error;
    } else {
        v = str_parser.token_buffer;
    }

#endif

    return no_error;
}

template <typename Iter>
inline error parse_any(Iter& i, value& v) {
    MINIJSON_SKIP(i)
    if (*i == '\x7b')
        return parse_object(i, v);
    if (*i == '[')
        return parse_array(i, v);
    if (*i == 't' || *i == 'f')
        return parse_boolean(i, v);
    if (*i == 'n')
        return parse_null(i, v);
    if ((*i == '-') || ('0' <= *i && *i <= '9'))
        return parse_number(i, v);
    if (*i == '"')
        return parse_string(i, v);
    return invalid_token_error;
}

template <typename Iter>
inline error parse(Iter& i, value& v) {
    return parse_any(i, v);
}

#undef MINIJSON_SKIP

inline const char* errstr(error e) {
    const char* s = "unknown error";
    switch (e) {
        case no_error: {
            s = "no error";
            break;
        }
        case undefined_error: {
            s = "undefined";
            break;
        }
        case invalid_token_error: {
            s = "invalid token";
            break;
        }
        case unknown_type_error: {
            s = "unknown type";
            break;
        }
        case memory_allocation_error: {
            s = "memory allocation error";
            break;
        }
        case corrupted_json_error: {
            s = "input is corrupted";
            break;
        }
        case duplicated_key_error: {
            s = "duplicated key found";
            break;
        }
            // default: return "unknown error";
    }

    return s;
}

}  // namespace minijson

// Max header(JSON) size. 100 MB as done in original safetensors implementation.
constexpr size_t kMaxJSONSize = 1024ull * 1024ull * 100ull;

namespace detail {

#ifdef _WIN32
std::wstring UTF8ToWchar(const std::string& str) {
    int wstr_size = MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), nullptr, 0);
    std::wstring wstr(size_t(wstr_size), 0);
    MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), &wstr[0], int(wstr.size()));
    return wstr;
}

std::string WcharToUTF8(const std::wstring& wstr) {
    int str_size = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()), nullptr, 0, nullptr, nullptr);
    std::string str(size_t(str_size), 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()), &str[0], int(str.size()), nullptr, nullptr);
    return str;
}
#endif

bool ReadWholeFile(std::vector<unsigned char>* out, std::string* err, const std::string& filepath, void*) {
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
    if (asset_manager) {
        AAsset* asset = AAssetManager_open(asset_manager, filepath.c_str(), AASSET_MODE_STREAMING);
        if (!asset) {
            if (err) {
                (*err) += "File open error : " + filepath + "\n";
            }
            return false;
        }
        size_t size = AAsset_getLength(asset);
        if (size == 0) {
            if (err) {
                (*err) += "Invalid file size : " + filepath + " (does the path point to a directory?)";
            }
            return false;
        }
        out->resize(size);
        AAsset_read(asset, reinterpret_cast<char*>(&out->at(0)), size);
        AAsset_close(asset);
        return true;
    } else {
        if (err) {
            (*err) += "No asset manager specified : " + filepath + "\n";
        }
        return false;
    }
#else
#ifdef _WIN32
#if defined(__GLIBCXX__)  // mingw
    int file_descriptor = _wopen(UTF8ToWchar(filepath).c_str(), _O_RDONLY | _O_BINARY);
    __gnu_cxx::stdio_filebuf<char> wfile_buf(file_descriptor, std::ios_base::in);
    std::istream f(&wfile_buf);
#elif defined(_MSC_VER) || defined(_LIBCPP_VERSION)
    // For libcxx, assume _LIBCPP_HAS_OPEN_WITH_WCHAR is defined to accept
    // `wchar_t *`
    std::ifstream f(UTF8ToWchar(filepath).c_str(), std::ifstream::binary);
#else
    // Unknown compiler/runtime
    std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
#else
    std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
    if (!f) {
        if (err) {
            (*err) += "File open error : " + filepath + "\n";
        }
        return false;
    }

    // For directory(and pipe?), peek() will fail(Posix gnustl/libc++ only)
    f.peek();
    if (!f) {
        if (err) {
            (*err) += "File read error. Maybe empty file or invalid file : " + filepath + "\n";
        }
        return false;
    }

    f.seekg(0, f.end);
    size_t sz = static_cast<size_t>(f.tellg());

    // std::cout << "sz = " << sz << "\n";
    f.seekg(0, f.beg);

    if (int64_t(sz) < 0) {
        if (err) {
            (*err) += "Invalid file size : " + filepath + " (does the path point to a directory?)";
        }
        return false;
    } else if (sz == 0) {
        if (err) {
            (*err) += "File is empty : " + filepath + "\n";
        }
        return false;
    } else if (sz >= (std::numeric_limits<std::streamoff>::max)()) {
        if (err) {
            (*err) += "Invalid file size : " + filepath + "\n";
        }
        return false;
    }

    out->resize(sz);
    f.read(reinterpret_cast<char*>(&out->at(0)), static_cast<std::streamsize>(sz));

    return true;
#endif
}
/*
bool parse_metadata(const ::minijson::value& v, GST_Dict<std::string>& dst, std::string* err) {
    if (auto po = v.as<::minijson::object>()) {
        for (size_t i = 0; i < po->size(); i++) {
            ::minijson::value ov;
            if (!po->at(i, &ov)) {
                if (err) {
                    (*err) += "[Internal error] Invalid object found in __metadata__, at index " + std::to_string(i) + ".\n";
                }
                return false;
            }

            if (auto so = ov.as<std::string>()) {
                if (dst.count(po->keys()[i])) {
                    // This should not be happen though
                    if (err) {
                        (*err) += "Duplicate key `" + po->keys()[i] + "` found in __metadata__.\n";
                    }
                    return false;
                }

                dst.insert(po->keys()[i], *so);
            } else {
                if (err) {
                    (*err) += "`" + po->keys()[i] + "` must be string value.\n";
                }
                return false;
            }
        }
    } else {
        if (err) {
            (*err) += "`__metadata__` value must be JSON object.\n";
        }
        return false;
    }

    return true;
}*/

bool parse_dtype(const ::minijson::value& v, typNUMBER& dtype, std::string* err) {
    if (auto so = v.as<std::string>()) {
        std::string name = *so;
        dtype            = tpNumOf(name);
        if (dtype == typNUMBER::T_OTHER) {
            (*err) += "Unknown `dtype` string: " + *so + ".\n";
        }
        return dtype != typNUMBER::T_OTHER;
    } else {
        if (err) {
            (*err) += "`dtype` item should be string type but got " + v.type_name() + ".\n";
        }
        return false;
    }

    return true;
}

bool parse_shape(const ::minijson::value& v, std::vector<size_t>& dst, std::string* err) {
    // NOTE:
    // - Empty tensors (tensors with 1 dimension being 0) are allowed
    // - [] is allowed(0-Rank tensor = merely a scalar)
    if (auto pa = v.as<::minijson::array>()) {
        ::minijson::array::const_iterator i;

        for (i = pa->begin(); i != pa->end(); i++) {
            if (auto pn = i->as<::minijson::number>()) {
                if (dst.size() >= kMaxDim) {
                    if (err) {
                        (*err) += "`shape` length must be less than " + std::to_string(kMaxDim) + " but got " + std::to_string(dst.size()) + ".\n";
                    }
                    return false;
                }

                dst.push_back(size_t(*pn));

            } else {
                if (err) {
                    (*err) += "Array item in `shape` must be number type, but got " + i->type_name() + ".\n";
                }
                return false;
            }
        }
    } else {
        if (err) {
            (*err) += "`shape` value must be JSON array, but got " + v.type_name() + ".\n";
        }
        return false;
    }

    return true;
}

bool parse_data_offsets(const ::minijson::value& v, std::array<size_t, 2>& dst, std::string* err) {
    if (auto pa = v.as<::minijson::array>()) {
        ::minijson::array::const_iterator i;
        size_t cnt = 0;

        for (i = pa->begin(); i != pa->end(); i++) {
            if (auto pn = i->as<::minijson::number>()) {
                if (cnt >= 2) {
                    if (err) {
                        (*err) += "`data_offsets` length must be 2.\n";
                    }
                    return false;
                }

                dst[cnt] = size_t(*pn);

                cnt++;

            } else {
                if (err) {
                    (*err) += "Array item in `data_offsets` must be number type, but got " + i->type_name() + ".\n";
                }
                return false;
            }
        }

        if (cnt != 2) {
            if (err) {
                (*err) += "`data_offsets` length must be 2.\n";
            }
            return false;
        }
    } else {
        if (err) {
            (*err) += "`data_offsets` value must be JSON array, but got " + v.type_name() + ".\n";
        }
        return false;
    }

    return true;
}

// From llama.cpp
#if defined(_WIN32)
static std::string safetensors_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, err,
                                 MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

struct safetensors_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE* fp{nullptr};
    size_t size{0};
    mutable bool _valid{false};
    std::string _err;

    safetensors_file(const char* fname, const char* mode) {
        fp = std::fopen(fname, mode);
        if (fp == nullptr) {
            _err   = "failed to open safetensors @" + std::string(fname) + ":\t" + std::string(strerror(errno)) + "!\n";
            _valid = false;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
            _valid = true;
        }
    }

    ~safetensors_file() {
        if (fp) {
            std::fclose(fp);
            fp = nullptr;
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            // this really shouldn't fail
            _valid = false;
            return 0;
        }

        return (size_t)ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        if (ret == 0) {
            _valid = false;
        }
    }

    bool& is_valid() const { return _valid; }

    const std::string& get_error() const { return _err; }
};

struct safetensors_mmap {
    uint8_t* addr{nullptr};
    size_t size{0};

    bool _valid{false};
    std::string _warn;
    std::string _err;

    const bool is_valid() const { return _valid; }

    const std::string& get_error() const { return _err; }

    const std::string& get_warning() const { return _warn; }

    safetensors_mmap(const safetensors_mmap&) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    //  support multiple mmap @ different files
    safetensors_mmap(struct safetensors_file* file, int __prot, size_t prefetch = (size_t)-1 /* -1 = max value */, bool numa = false) {
        size      = file->size;
        int fd    = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) {
            prefetch = 0;
        }
#ifdef __linux__
        if (prefetch) {
            flags |= MAP_POPULATE;
        }
#endif
        // addr = reinterpret_cast<uint8_t *>(mmap(NULL, file->size, PROT_READ, flags, fd, 0));
        addr = reinterpret_cast<uint8_t*>(mmap(NULL, file->size, __prot, flags, fd, 0));
        if (addr == MAP_FAILED) {
            _valid = false;
            _err   = "mmap failed: " + std::string(strerror(errno)) + "\n";
            size   = 0;
            addr   = nullptr;
            return;
        }

        if (prefetch > 0) {
            // Advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                _warn += "posix_madvise(.., POSIX_MADV_WILLNEED) failed: " + std::string(strerror(errno)) + "\n";
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                _warn += "posix_madvise(.., POSIX_MADV_RANDOM) failed: " + std::string(strerror(errno)) + "\n";
            }
        }

        _valid = true;
    }

    ~safetensors_mmap() {
        if (_valid) {
            munmap(addr, size);
        }
        size   = 0;
        addr   = nullptr;
        _valid = false;
    }

#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    safetensors_mmap(struct safetensors_file* file, bool prefetch = true, bool numa = false) {
        (void)numa;

        size = file->size;

        HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        DWORD error     = GetLastError();

        if (hMapping == NULL) {
            // TODO: get error message
            _err   = "CreateFileMappingA failed: " + safetensors_format_win_err(error) + "\n";
            _valid = false;
            size   = 0;
            addr   = nullptr;
            return;
        }

        addr  = reinterpret_cast<uint8_t*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
        error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            _err = "MapViewOfFile failed: " + safetensors_format_win_err(error) + "\n";
        }

#if _WIN32_WINNT >= _WIN32_WINNT_WIN8
        if (prefetch) {
            // PrefetchVirtualMemory is only present on Windows 8 and above, so we
            // dynamically load it
            BOOL(WINAPI * pPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = reinterpret_cast<decltype(pPrefetchVirtualMemory)>(GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes  = (SIZE_T)size;
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    _warn += "PrefetchVirtualMemory failed: " + safetensors_format_win_err(GetLastError()) + "\n";
                }
            }
        }
#endif
    }
    ~safetensors_mmap() {
        if (!UnmapViewOfFile(addr)) {
            _warn += "UnmapViewOfFile failed: " + safetensors_format_win_err(GetLastError()) + "\n";
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    safetensors_mmap(struct safetensors_file* file, bool prefetch = true, bool numa = false) {
        (void)file;
        (void)prefetch;
        (void)numa;

        _valid = false;
        _err   = "mmap not supported\n";
        addr   = nullptr;
        size   = 0;
    }
#endif
};

}  // namespace detail

K_SafeTensors::~K_SafeTensors() {
    if (st_mmap) {
        detail::safetensors_mmap* p = reinterpret_cast<detail::safetensors_mmap*>(st_mmap);
        delete p;
        st_mmap = nullptr;
    }

    if (st_file) {
        detail::safetensors_file* p = reinterpret_cast<detail::safetensors_file*>(st_file);
        delete p;
        st_file = nullptr;
    }
}

bool mmap_from_file(const std::string& filename, K_SafeTensors* st, std::string& warn, std::string& err, int __prot) {
    if (!st) {
        return false;
    }
    detail::safetensors_file* pf = new detail::safetensors_file(filename.c_str(), "r+b");
    // detail::safetensors_file *pf = new detail::safetensors_file(filename.c_str(), "rb");
    if (!pf->is_valid()) {
        err += pf->get_error();
        delete pf;
        return false;
    }
    // TODO: prefetch, numa
    detail::safetensors_mmap* pm = new detail::safetensors_mmap(pf, __prot);
    if (!pm->is_valid()) {
        err += pm->get_error();
        delete pm;
        return false;
    }

    // bool ret = mmap_from_memory(pm->addr, pm->size, filename, st, &warn, &err);
    uint64_t json_size     = *(uint64_t*)pm->addr;
    st->header_size        = json_size;
    const char* json_bytes = reinterpret_cast<const char*>(pm->addr) + sizeof(uint64_t);
    std::string json_str(json_bytes, json_size);
    // _INFO(">>>>>> header of%s= \n\t%s", filename.c_str(), json_str.c_str());
    st->jHeader = JSON::parse(json_str);
    for (auto& [key, value] : st->jHeader.items()) {
        //_INFO();
        if (key == "__metadata__")
            st->metadata = value;
        else {
            hGTensor tensor = std::make_shared<GTensor>(st->hFish, key, value);
            st->tensors.insert(key, tensor);
        }
    }

    // st->tensors     = std::move(tensors);

    // size_t databuffer_size = pm->size - st->header_size - sizeof(uint64_t);
    st->mmaped    = true;
    st->mmap_addr = pm->addr, st->mmap_size = pm->size;
    st->databuffer_addr = st->mmap_addr + sizeof(uint64_t) + st->header_size;
    st->databuffer_size = st->mmap_size - (sizeof(uint64_t) + st->header_size);
    // st->mmap_addr       = pm->addr;
    // st->mmap_size       = pm->size;
    // st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
    // st->databuffer_size = st->mmap_size - (8 + st->header_size);
    // retain pointer
    st->st_file = pf;
    st->st_mmap = pm;
    // st->mmaped = true;
    st->initJS();
    return true;
}

bool mmap_from_memory(const uint8_t* addr, const size_t nbytes, const std::string& filename, K_SafeTensors* st, std::string* warn, std::string* err) {
    if (!addr) {
        return false;
    }

    if (nbytes < 16) {
        return false;
    }

    if (!st) {
        return false;
    }
    uint64_t json_size     = *(uint64_t*)addr;
    const char* json_bytes = reinterpret_cast<const char*>(addr) + sizeof(uint64_t);
    // void* bytes_ptr   = (char*)data + sizeof(uint64_t) + json_size;
    // size_t bytes_size = size - sizeof(uint64_t) - json_size;
    std::string json_str(json_bytes, json_size);

    st->metadata = JSON::parse(json_str);
    // if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn, err)) {
    //     return false;
    // }

    size_t databuffer_size = nbytes - st->header_size - 8;
    st->mmaped             = true;
    st->mmap_addr          = addr;
    st->mmap_size          = nbytes;
    st->databuffer_addr    = st->mmap_addr + 8 + st->header_size;
    st->databuffer_size    = st->mmap_size - (8 + st->header_size);

    return true;
}

std::string HF_dtype2str(const typNUMBER dtype);
#endif

