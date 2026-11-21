/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Thanks the work of safetensors(https://github.com/syoyo/safetensors-cpp)
 *
 *  \brief Serialization
 *  \author Yingshi Chen
 */

#include "../Tensor/Safetensors.hpp"

std::string K_SafeTensors::config_key_ = "__koifish__config__";

#ifdef _nlohmann_json_
std::vector<uint8_t> inline J_to_msgpack(const JSON& js) { return JSON::to_msgpack(js); }
JSON inline J_from_msgpack(const std::vector<uint8_t>& msgpack) { return JSON::from_msgpack(msgpack); }
#else
std::vector<uint8_t> J_to_msgpack(const json11::Json& j);
json11::Json J_from_msgpack(const std::vector<uint8_t>& buf);
#endif

bool K_SafeTensors::initJS(int flag) {
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
        jsConfig                     = J_from_msgpack(msgpack);  // JSON::from_msgpack(msgpack);
        return true;
    } /*catch (JSON::parse_error& e) {
        _INFO("\r\n%s  Failed to initJS from safetensor[]!!! ERR=%s", __func__, e.what());
        return false;
    }*/
    catch (...) {
        _ERROR("\r\n%s  Failed to initJS from safetensor[]!!!", __func__);
        return false;
    }
}

void K_SafeTensors::insertJS(const JSON& js, size_t dst_offset, int flag) {
    jsConfig        = js;
    hGTensor tensor = std::make_shared<GTensor>();
    strcpy(tensor->name, config_key_.c_str());
    tensor->hFish           = hFish;
    tensor->msgpack         = J_to_msgpack(js);        // JSON::to_msgpack(js);
    tensor->type            = typNUMBER::U8;           // dtype::kUINT8;
    size_t sz               = tensor->msgpack.size();  // storage.size();
    tensor->data_offsets[0] = dst_offset;
    tensor->data_offsets[1] = dst_offset + sz;
    tensor->shape           = {(int)sz};
    tensors.insert(config_key_, tensor);

    // storage.resize(dst_offset + sz);
    // memcpy(storage.data() + dst_offset, msgpack.data(), sz);  //  6441  [132...160]
}

int K_SafeTensors::loadJS(hGTensor tensor, const uint8_t* bytes_ptr, size_t bytes_size, int flag) {
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
    jsConfig = J_from_msgpack(msgpack);
    return 0x0;
}