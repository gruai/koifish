/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Thanks the work of safetensors(https://github.com/syoyo/safetensors-cpp)
 *
 *  \brief Serialization
 *  \author Yingshi Chen
 */
#include "Serialize.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <set>

#include "../Tensor/GeQuant.hpp"
#include "../TokenSet/Dictionary.hpp"
#include "../Utils/GST_os.hpp"
#include "Fish.hpp"

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "../Tensor/safetensors.hh"

std::string safetensors::safetensors_t::config_key_ = "__json__config__";
/*
     support multiple mmap @ different files
     1. AfterBuild->HF_Serialize
     2. AfterBuild->SaveTrain->SAFETENSOR_Serialize
     3.
*/
bool SAFETENSOR_mmap(const std::string& path, safetensors::safetensors_t& st, int flag) {
    _INFO("\n>>>>>> SAFETENSOR_mmap mmap@ %s \"%s\" %s f=%d......", COLOR_ORANGE, path.c_str(), COLOR_RESET, flag);
    try {
        std::string warn, err;
        int __prot = PROT_READ | PROT_WRITE;                                             // PROT_READ
        bool ret   = safetensors::mmap_from_file(path.c_str(), &st, warn, err, __prot);  //   safetensors::load_from_file();
        int nT     = (int)(st.tensors.size());
        // assert(nT > 1);
        if (warn.size()) {
            _WARN(">>>>>> WARN: SAFETENSOR_mmap@\"%s\" \"%s\"\n", path.c_str(), warn.c_str());  // std::cout << "WARN: " << warn << "\n";
        }
        if (!ret) {
            _ERROR("\n>>>>>> ERR: SAFETENSOR_mmap@\"%s\" \"%s\"\n", path.c_str(), err.c_str());
            return false;
        }
        if (!safetensors::validate_data_offsets(st, err)) {
            _ERROR(">>>>>> Invalid data_offsets: \"%s\"\n", err.c_str());
            // std::cerr << err << "\n";
            return false;
        }

        if (st.metadata.size()) {
            for (size_t i = 0; i < st.metadata.size(); i++) {
                std::string key = st.metadata.keys()[i], value;
                st.metadata.at(i, &value);
            }
        }
        const uint8_t* databuffer{nullptr};
        if (st.mmaped) {
            databuffer = st.databuffer_addr;  // st->mmap_addr + 8 + st->header_size;
        } else {
            assert(0);
            //  databuffer = st.storage.data();
        }

        // Print Tensor info & value.
        for (size_t i = 0; i < nT; i++) {
            std::string key = st.tensors.keys()[i];
            if (key == safetensors::safetensors_t::config_key_) {
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                st.loadJS(tensor, databuffer, st.mmap_size);
                if (true) {  // only for debug
                    JSON jsConfig = st.jsConfig["CLI_params"]["config"];
                    std::ofstream file("./hy-tmp/_koifish_tmp_config_.json");
                    if (file.is_open()) {
                        file << jsConfig.dump(4);
                        file.close();
                    }
                }
                continue;
            }
        }
        size_t szIFS = std::filesystem::file_size(path);
        _INFO("\r>>>>>> SAFETENSOR_mmap mmap@ %s \"%s\" %s f=%d nT=%d fsize=%.7gM\n", COLOR_ORANGE, path.c_str(), COLOR_RESET, flag, nT, szIFS / 1.0e6);
        return true;
    } catch (JSON::parse_error& e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, path.c_str(), e.what());
        return false;
    } catch (...) {
        return false;
    }
}

bool SAFETENSOR_Load_jconfig(const std::string& path, JSON& jsConfig, FSerial::FILE_TYPE tpFile, int flag) {
    try {
        safetensors::safetensors_t st;
        bool bLoad = SAFETENSOR_mmap(path, st, flag);
        if (!bLoad)
            return false;
        if (st.jsConfig.empty()) {
            _INFO(">>>>>> \"%s\" has no jConfig! \n", path.c_str());
            return false;
        }

        // jsConfig = st.jsConfig;
        // assert(!jsConfig.empty());
        // std::ofstream o("_koifish_tmp_config_.json");
        // o << std::setw(4) << jsConfig << std::endl;
        jsConfig = st.jsConfig["CLI_params"]["config"];
        if (jsConfig.empty()) {
            assert(0);
            return false;
        }
        switch (tpFile) {
            case FSerial::FILE_FISH:
            case FSerial::FILE_CHECKPOINT:

                break;
            default:
                break;
        }

        return true;
    } catch (JSON::parse_error& e) {
        _ERROR("\r\n SAFETENSOR_Load_jconfig failed to open %s!!! ERR=%s", path.c_str(), e.what());
        return false;
    } catch (...) {
        _ERROR("\r\n SAFETENSOR_Load_jconfig Unknown exception @%s!!!", path.c_str());
        return false;
    }
}

bool Fuyou::Serialize(bool isSave, int flag) {
    for (auto t : tReloads) {
        assert(BIT_TEST(t->flags, GTensor::F_RELOAD));
        assert(t->host_data != nullptr);
        auto now = GST_us();
        assert(t->host_data != nullptr);
        t->Serial_Quant_MMAP(isSave);
        SUM::tLoadParam += GST_us() - now;
    }

    return true;
}

using namespace safetensors;

/*
    Called by SAFETENSOR_Serialize
*/
bool safetensors::save_to_ofs(const safetensors_t& st, std::ofstream& ofs, size_t& szAll, std::string* warn, std::string* err, int flag) {
    bool isInitMMap = BIT_TEST(flag, FSerial::INIT_MMAP);
    bool isCopyMMap = BIT_TEST(flag, FSerial::COPY_MMAP);
    try {
        fflush(stdout);
        // _INFO(">>>>>> saveto_ofs ......\n");
        // directly serialize JSON string.
        std::stringstream ss;
        // By default, std::stringstream does not throw exceptions for stream failures (e.g., failbit).
        ss.exceptions(std::ios::failbit | std::ios::badbit);  // to make it throw on errors.
        // NOTE: The last offset **must** be the end of the file,
        // so write __metadata__ first(if metadata part exists)

        std::string _err;
        // if (!validate_data_offsets(st, _err)) {
        //     if (err) {
        //         (*err) += "Invalid safensors is provided.\n";
        //         (*err) += _err;
        //     }
        //     return false;
        // }

        ss << "{";
        if (st.metadata.size()) {
            ss << "\"__metadata__\": {";
            size_t nmeta = 0;
            for (size_t i = 0; i < st.metadata.size(); i++) {
                std::string key = st.metadata.keys()[i];
                std::string value;
                st.metadata.at(i, &value);

                if (nmeta > 0) {
                    ss << ", ";
                }
                ss << "\"" + key + "\": \"" << value << "\"";
                nmeta++;
            }
            ss << "}";

            if (st.tensors.size()) {
                ss << ", ";
            }
        }

        size_t ntensors = 0, nInit = 0;
        for (size_t i = 0; i < st.tensors.size(); i++) {
            std::string key = st.tensors.keys()[i];
            safetensors::tensor_t tensor;
            st.tensors.at(i, &tensor);
            // _INFO("\r\t %d/%d\t\"%s\"", i, st.tensors.size(), key.c_str());
            fflush(stdout);
            if (tensor.shape.size() > safetensors::kMaxDim) {
                if (err) {
                    (*err) += key + ".shape is too large.\n";
                    (*err) += _err;
                }
                return false;
            }

            if (ntensors > 0) {
                ss << ", ";
            }
            ss << "\"" << key << "\": {";
            ss << "\"dtype\": \"" << safetensors::get_dtype_str(tensor.dtype) << "\", ";
            ss << "\"shape\": [";
            for (size_t i = 0; i < tensor.shape.size(); i++) {
                if (i > 0) {
                    ss << ", ";
                }
                ss << tensor.shape[i];
            }
            ss << "]";
            ss << ", \"data_offsets\": [" << tensor.data_offsets[0] << ", " << tensor.data_offsets[1] << "]";
            ss << "}";
            ntensors++;
        }
        ss << "}";

        std::string header_str = ss.str();
        uint64_t header_size   = header_str.size();  // do not include '\n'
        const void* databuffer_addr{nullptr};
        size_t databuffer_size{0};
        if (st.mmaped) {
            databuffer_size = st.databuffer_size;
            databuffer_addr = st.databuffer_addr;
        } else {
            databuffer_size = szAll, databuffer_addr = nullptr;
            // databuffer_size = st.storage.size();
            // databuffer_addr = reinterpret_cast<const void *>(st.storage.data());
        }

        // make databuffer addr start from the multiple of 8.
        size_t pad_bytes = 0;
        if ((header_size % 8) != 0) {
            pad_bytes = 8 - (header_size % 8);
        }
        // printf("header_size = %d\n", int(header_size));
        // printf("pad_bytes = %d\n", int(pad_bytes));
        size_t padded_header_size = header_size + pad_bytes;  // 20856
        size_t szOFS              = 8 + padded_header_size + databuffer_size;
        szAll                     = szOFS;
        fflush(stdout);
        // _INFO(">>>>>> saveto_ofs ......sz=%.6gM...", szAll / 1.0e6);
        // std::vector<uint8_t> buffer;
        // buffer.resize(8 + padded_header_size + databuffer_size);
        // size_t szDst = buffer.size();  //  248972672,  248951808
        // // write padded header_size
        // memcpy(buffer.data(), &padded_header_size, sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&padded_header_size), sizeof(size_t));
        // // write header
        // memcpy(buffer.data() + 8, header_str.data(), header_size);
        ofs.write(reinterpret_cast<const char*>(header_str.data()), header_size);
        // // Use whitespace for trailing padding.
        // memset(buffer.data() + 8 + header_size, 0x20, pad_bytes);
        std::vector<uint8_t> pad;
        pad.resize(pad_bytes);
        memset(pad.data(), 0x20, pad_bytes);
        ofs.write(reinterpret_cast<const char*>(pad.data()), pad_bytes);
        // memcpy(buffer.data() + 8 + padded_header_size, databuffer_addr, databuffer_size);
        if (databuffer_addr != nullptr)  // mmap file
            ofs.write(reinterpret_cast<const char*>(databuffer_addr), databuffer_size);
        else {
            size_t dst_offset = 0, sz;
            for (size_t i = 0; i < st.tensors.size(); i++) {
                std::string key = st.tensors.keys()[i];
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                assert(dst_offset == tensor.data_offsets[0]);
                if (key != safetensors::safetensors_t::config_key_) {
                    GTensor* t = (GTensor*)(tensor.hUserData);
                    sz         = tensor.data_offsets[1] - dst_offset;  // t->nByte() * 3;
                    assert(sz == t->nByte() || sz == t->nByte() * 3);
                    hBITARR tmp = new BIT_8[sz]();
                    if (t->GetDataX() == nullptr) {
                        if (isInitMMap) {
                            nInit++;
                        } else if (isCopyMMap) {
                            assert(t->host_data != nullptr);
                            SAFE_read_mmap(tmp, (hBITARR)(t->host_data), t->nByte());
                            t->tpInit = INIT_WEIGHT::SERIALIZE;
                        } else {
                            _ERROR("[ST_SERIALIZE] \"%s\" is empty!\n", t->name);
                            return false;
                        }
                        // memset(databuffer_size + dst_offset, 0x0, sz);
                    } else {
                        assert(t->data != nullptr && t->gm != nullptr);
                        t->SerialData("", tmp, true);
                    }
                    ofs.write(reinterpret_cast<const char*>(tmp), sz);
                    delete[] tmp;
                } else {
                    sz              = tensor.msgpack.size();
                    const char* tmp = (const char*)tensor.msgpack.data();
                    ofs.write(tmp, sz);
                }
                dst_offset += sz;
            }
        }

        fflush(stdout);
        // _INFO(">>>>>> saveto_ofs ......OK\n");
        return true;
    } catch (const std::exception& e) {
        _INFO("\n!!!!saveto_ofs excetioin=%s sz=%ld!!!\n", e.what(), szAll);
        return false;
    } catch (...) {
        _INFO("\n!!!!saveto_ofs Unknown exception occurred sz=%ld!!!\n", szAll);
        return false;
    }
}

static safetensors::safetensors_t all_states;
void CheckPoint_Params::Init(int flag) {
    switch (type) {
        case STATE:
            // To read/write mmap many times, a hack for the poor design of safetensors
            hUserData = &all_states;
            break;
        default:
            break;
    }
}

void HST2JSON(safetensors::safetensors_t* hst, int flag = 0x0) {
    JSON jSafeTensors;
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        safetensors::tensor_t tensor;
        hst->tensors.at(i, &tensor);
        if (key == safetensors::safetensors_t::config_key_) {
            continue;
        }
        JSON jdesc    = tensor.jDesc();
        jdesc["name"] = key;
        jSafeTensors.push_back(jdesc);
    }
    assert(!jSafeTensors.empty());
    std::ofstream file("_safetensors_.json");
    if (file.is_open()) {
        file << jSafeTensors.dump(4);
        file.close();
    }
    return;
}

int Fish::SAFETENSOR2Gensors(const std::string& path, safetensors::safetensors_t* hst, int flag) {
    int nSerialT = 0;
    hst->Clear();
    bool bLoad = SAFETENSOR_mmap(path, *hst, flag);  //*hst
    if (!bLoad)
        return false;
    const uint8_t* databuffer{nullptr};
    if (hst->mmaped) {
        databuffer = hst->databuffer_addr;  // safeTensors->mmap_addr + 8 + safeTensors->header_size;
    } else {
        assert(0);
        // databuffer = safeTensors.storage.data();
    }
    HST2JSON(hst, flag);
    // Print Tensor info & value.
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        safetensors::tensor_t tensor;
        hst->tensors.at(i, &tensor);
        if (key == safetensors::safetensors_t::config_key_) {
            continue;
        }
        // if (isOnlyVocab)
        //     continue;

        if (G_Has_(key, config.model.skip_st))
            continue;

        hGensor target = GetGensor(key);  //  "model.embed.weight"    model.layers.0.attn_norm.weight
        if (target == nullptr) {
            _ERROR("\t[SERIAL] Failed @%s!\n", key.c_str());
            continue;
        }
        auto ginfo = GetGensorInfo(target);
        JSON jdesc = tensor.jDesc();
        if (target->SerialJSON(key, jdesc, (void*)databuffer, hst->mmap_size) != 0) {
            return false;
        }
        if (G_Has_(key, {"model.embed_tokens.weight"})) {  // model.embed_tokens.weight
            DEBUG_HERE;
            // target->Print("wte@ST", 4, -1);
        }

        if (DUMP() || config.dumpSwitch.tensor_load > 0) {
            tensor.Dump("  >>>>  " + key, databuffer);
            _INFO("  >>>>  [%d] typ=%s\t data=%p grad=%p \t sz=%ld @%s\n", nSerialT, cNameOf(target->type), target->data, target->grad, tBYTE(target),
                  target->name);
        }
        // if(strcmp(target->name,"model.layers.27.mlp.norm.weight")==0){   //only for debug model.output.weight
        //     target->Print(key,0,-1);                //PrintTensor<f8e5>("wout",target->data,target->ne[0],dim);
        // }

        nSerialT++;
    }
    return nSerialT;
}

/*
    Save
        call safetensors::save_to_file
*/
bool Fish::SAFETENSOR_Serialize(CheckPoint_Params& ckp, bool isSave, int flag) {
    double t0   = GST_ms();
    string path = ckp.FullPath(isSave), jPath = path + ".json";
    if (path.empty()) {
        _INFO("\r\n%s failed: empty path!!! ", __func__);
        return false;
    }

    size_t data_offset_base = 0, nInit = 0, szOFS = 0;
    std::string warn, err;
    
    vector<hGensor> curParams = optParams;
    safetensors::safetensors_t st, *hst = (safetensors::safetensors_t*)(ckp.hUserData);
    switch (ckp.type) {
        case CheckPoint_Params::STATE:
            assert(hst != nullptr);  // hst=&all_states, so mmp would keep open
            break;
        case CheckPoint_Params::BEST:
            curParams = GetFuyou(-1)->ckpParams;
            hst       = &st;
            break;
        case CheckPoint_Params::FULL:
            // assert(0);
            hst = &st;
            break;
        default:
            assert(0);
    }

    try {
        if (isSave) {
            fflush(stdout);
            JSON jsConfig;
            _INFO(">>>>>> ST_SERIALIZE save @\"%s\" nInit=%ld ......", path.c_str(), nInit);
            jsConfig["vendor"]              = "gruai";
            jsConfig["CLI_params"]          = config.ToJSON(0x100);
            jsConfig["tokenizer"]["tokens"] = "";
            if (jsConfig["CLI_params"]["config"].contains("model") && jsConfig["CLI_params"]["config"]["model"].contains("hf-card")) {
                jsConfig["CLI_params"]["config"]["model"]["#origin-hf-card"] = jsConfig["CLI_params"]["config"]["model"]["hf-card"];
                jsConfig["CLI_params"]["config"]["model"].erase("hf-card");
                jsConfig["CLI_params"]["config"]["model"]["arch"] = config.model.model_type; //"QWEN3";
                jsConfig["CLI_params"]["config"]["model"]["parameter"]["max_pos_embeddings"] = config.model.max_pos_embeddings;   //:32768            
            }

            hst->Clear();
            size_t dst_offset = 0;
            if(curParams.size()==0){
                _WARN("%s SAFETENSOR: Save_Params=0 @\"%s\"",path.c_str());
            }
            for (auto t : curParams) {
                if (G_Has_(t->name, {"model.embed_tokens.weight"})) {  // model.embed_tokens.weight
                    DEBUG_HERE;
                    // t->Print("wte@save", 4, -1);
                }

                size_t sz = t->nByte();  // expand
                if (ckp.type == CheckPoint_Params::STATE)
                    sz *= 3;
                assert(sz > 0);
                safetensors::tensor_t tensor;
                tensor.dtype           = t->type == typNUMBER::F32 ? typNUMBER::F32 : t->type == typNUMBER::F16 ? typNUMBER::F16 : typNUMBER::BF16;
                tensor.hUserData       = t.get();
                tensor.data_offsets[0] = dst_offset;
                tensor.data_offsets[1] = dst_offset + sz;
                tensor.shape.insert(tensor.shape.end(), t->shape.begin(), t->shape.end());
                hst->tensors.insert(t->name, tensor);

                jsConfig["tensors"][t->name] = dst_offset;  // tensor->Dump(100,"");
                // jsConfig["tensors"]["offset"] = dst_offset;
                dst_offset += sz;
            }
            hst->insertJS(jsConfig, dst_offset);
            // __metadata__
            hst->metadata.insert("vendor", "gruai");
            bool ret = safetensors::save_to_file(*hst, path, dst_offset, &warn, &err, flag);
            if (warn.size()) {
                std::cout << "WARN: " << warn << "\n";
            }
            if (!ret) {
                std::cerr << "Failed to write safetensor data to " << path << "\n";
                if (err.size()) {
                    std::cout << "ERR: " << err << "\n";
                }
                return false;
            }
            szOFS = std::filesystem::file_size(path);
            //  save json file
            std::ofstream o(jPath);
            o << std::setw(4) << jsConfig << std::endl;
            _INFO("\r>>>>>> ST_SERIALIZE save @\"%s\" nInit=%ld sz=%.6gM flag=%d T=%.4gs\n", path.c_str(), nInit, szOFS / 1.0e6, flag,
                  (GST_ms() - t0) / 1000.0);
            return true;
        } else {
            int nSerialT = SAFETENSOR2Gensors(path, hst, 0x0);
            _INFO(">>>>>> ST_SERIALIZE load@\"%s\" nSerialT=%d iter=%d\n", path.c_str(), nSerialT, flag);
            return true;
        }
    } catch (JSON::parse_error& e) {
        _INFO("\r\n%s  Failed to serialize @\"%s\"!!! ERR=%s", __func__, jPath.c_str(), e.what());
        return false;
    } catch (...) {
        _INFO("\r\n%s  Failed to serialize @\"%s\"!!! ", __func__, path.c_str());
        return false;
    }
}

void* MMAP_json(JSON& header, void** objs, size_t* objs_nz, const std::string& path, bool isSave, int flag) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return nullptr;
    }
    size_t size = st.st_size;
    void* data  = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return nullptr;
    }
#ifdef __linux__
    // increases readahead buffer size, resulting in faster cold loads
    posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
#endif
    close(fd);  // fd can be closed after mmap returns without invalidating the mapping

    // Parse the metadata JSON and the tensors
    if (size < sizeof(uint64_t)) {
        munmap(data, size);
        return nullptr;
    }

    uint64_t json_size = *(uint64_t*)data;
    if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
        munmap(data, size);
        return nullptr;
    }

    char* json_ptr    = (char*)data + sizeof(uint64_t);
    void* bytes_ptr   = (char*)data + sizeof(uint64_t) + json_size;
    size_t bytes_size = size - sizeof(uint64_t) - json_size;

    std::string json_str(json_ptr, json_size);
    header   = JSON::parse(json_str);
    *objs    = bytes_ptr;
    *objs_nz = bytes_size;
    return data;
}

bool Fish::HF_Serialize(bool isSave, int flag) {
    std::vector<std::string> paths = FilesOfDir(config.model.sCardPath, {"safetensors"}, 0x0);
    if (paths.empty())
        return false;

    int nSerialT = 0;
    std::vector<safetensors::safetensors_t*> st_mmfs;
    for (auto path : paths) {
        safetensors::safetensors_t* hst = new safetensors::safetensors_t();
        st_mmfs.push_back(hst);
        nSerialT += SAFETENSOR2Gensors(path, hst, 0x0);
    }
    if (isTrain()) {  // otherwise, st would release mmap memory!
        SaveTrain(config.state, true, FSerial::COPY_MMAP);
    }
    for (auto hst : st_mmfs)  // release all mmf resource
        delete hst;

    if (!config.model.st_map.empty()) {  //  "model.safetensors.index.json"
        assert(nSerialT == config.model.st_map.size());
        for (auto kv : config.model.st_map) {
            hGensor target = GetGensor(kv.first);
            assert(target->data != nullptr);
            kv.second = target->data;
        }
    }
    _LOG(nSerialT == 0 ? DUMP_ERROR : DUMP_INFO, ">>>>>> HF_Serialize load@\"%s\" nSerialT=%d iter=%d\n", config.model.sCardPath.c_str(), nSerialT, flag);
    // if(!config.quant.filter_MIQ.empty())
    //     throw SafeExit("", KOIFISH_EXIT_DEBUG, SafeExit::ExitReason::SYSTEM_FAILURE, __func__);
    for (auto t : optParams) {
        assert(!BIT_TEST(t->flags, GTensor::F_MMAP));
        if (!isTrain()) {
            assert(t->host_data == nullptr);
        }

        // assert(t->szUse>0 && t->data!=nullptr);  //bias maybe null
    }
    if (nSerialT == 0) {  //  !=optParams.size()
        return false;     // bias maybe null
    }

    return true;
}

bool MODEL_CARD::InitChatTemplate(CLI_params* hConfig, int flag) {
    // string fPrompt = sCardPath + "template_user_thinking.txt", fSysPromt = sCardPath + "template_system_thinking.txt";
    // if (enable_thinking) {  // load the "thinking" versions of the templates

    // } else {  // load the standard versions of the templates
    //     fPrompt = sCardPath + "template_user.txt", fSysPromt = sCardPath + "template_system.txt";
    // }
    // prompt_template        = FILE2STR(fPrompt);
    // system_prompt_template = FILE2STR(fSysPromt);

    if (enable_thinking) {  //
        prompt_template        = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
        system_prompt_template = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    } else {  //
        prompt_template        = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        system_prompt_template = "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }

    return true;
}
bool MODEL_CARD::InitHugFace(CLI_params* hConfig, const JSON& jConfig, const std::string& sCardPath_0, int flag) {
    bool isInitFromPath = !sCardPath_0.empty();
    int head_dim = -1, n_heads = -1, n_kv_heads = -1, n_FF = -1;
    if (!isInitFromPath) {
        sTokenBinPath = "./assets/tokenizer_151936.bin";
        n_layers      = hConfig->nLayer();
        for (int i = 0; i < n_layers; i++) {
            int nH = hConfig->n_head(i), nF = hConfig->n_ff(i);
            assert(nH > 0 && nF > 0);
        }

        string sTyp = jKVs(jConfig, {"model", "datatype", "weight"}, string(""));
        if (!sTyp.empty())
            tpWeight = tpNumOf(sTyp);
        // sTyp = jKVs(jConfig, {"model", "datatype", "embed"}, string(""));
        // if (!sTyp.empty())
        //     tpEmbed = tpNumOf(sTyp);
        head_dim = hConfig->head_dim();
        n_heads = hConfig->n_head(), n_kv_heads = hConfig->n_head_kv();
        // seq_len = hConfig->n_ctx_orig();

        sCardPath = jKV(jConfig, {"model", "hf-card"}, sCardPath);
    } else {
        sCardPath = sCardPath_0;
    }
    if (sCardPath.empty()) {
        return false;
    }
    string jPath = sCardPath + "config.json";
    if (!LoadJsonFile(jPath, jModelParam)) {
        return false;
    };
    sTokenJsonPath = sCardPath + "tokenizer.json";
    // LoadJsonFile(sTokenPath,jTokenizer);                             // }

    if (jModelParam.empty()) {
        sCardPath = "";
    } else {
        isEmbedWeightTying = jKV(jModelParam, {"tie_word_embeddings"}, isEmbedWeightTying);
        if (isEmbedWeightTying) {
            skip_st.push_back("lm_head.weight");  //  so strange: Qwen/Qwen3-0.6B has this & unsloth/Qwen3-0.6B-Base remove this
        }
        model_type = jKV(jModelParam, {"model_type"}, model_type);
        assert(model_type != "");
        num_attention_heads = jKV(jModelParam, {"num_attention_heads"}, n_heads);
        num_key_value_heads = jKV(jModelParam, {"num_key_value_heads"}, n_kv_heads);
        if (jModelParam.find("quantization_config") != jModelParam.end()) {
            hConfig->jVendorQuant = jModelParam["quantization_config"];
            QUANT_CARD quant;
            quant.InitFromVendor(jModelParam["quantization_config"]);
            hConfig->jQuant = quant.ToJSON();
        }

        int hd = jKV(jModelParam, {"head_dim"}, head_dim);
        if (hd != head_dim) {
            for (auto& lay : layerps) {
                lay._head_dim = hd;
            }
        }
        vocab_size           = jKV(jModelParam, {"vocab_size"}, vocab_size);
        torch_dtype          = jKV(jModelParam, {"torch_dtype"}, torch_dtype);
        transformers_version = jKV(jModelParam, {"transformers_version"}, transformers_version);
        bos_token_id         = jKV(jModelParam, {"bos_token_id"}, bos_token_id);
        eos_token_id         = jKV(jModelParam, {"eos_token_id"}, eos_token_id);
        rope_theta           = jKV(jModelParam, {"rope_theta"}, rope_theta);
        hidden_size          = jKV(jModelParam, {"hidden_size"}, hidden_size);
        intermediate_size    = jKV(jModelParam, {"intermediate_size"}, intermediate_size);
        max_pos_embeddings   = jKV(jModelParam, {"max_position_embeddings"}, max_pos_embeddings);
        // rotary_dim           = jKVs(jModelParam, {"rope_scaling"}, rotary_dim);
        //

        if (!isInitFromPath)
            assert(num_attention_heads == n_heads && num_key_value_heads == n_kv_heads);
        else {
            n_layers         = jKV(jModelParam, {"num_hidden_layers"}, n_layers);
            hConfig->nLayerX = n_layers;
            token_embeds.push_back(hidden_size);
            if (hConfig->common.n_ctx == -1) {  //  max_position_embeddings(often the same as training context length)
                hConfig->common.n_ctx = max_pos_embeddings;
                hConfig->n_ctx_train  = max_pos_embeddings;
            }
            for (int i = 0; i < n_layers; i++) layerps.push_back(MODEL_CARD::LAY_PARAM(num_attention_heads, num_key_value_heads, hd, intermediate_size));
        }

        InitChatTemplate(hConfig);
    }

    LoadJsonFile(sCardPath + "model.safetensors.index.json", jSafetensorsIndex);
    if (!jSafetensorsIndex.empty()) {
        nTotalSize = jKV(jSafetensorsIndex, {"metadata", "total_size"}, nTotalSize);

        auto jMap = jSafetensorsIndex["weight_map"];
        for (JSON::iterator it = jMap.begin(); it != jMap.end(); ++it) {
            std::string key = it.key();
            st_map.insert(std::make_pair(key, nullptr));
        }
    }
    switch (hConfig->ModelArch()) {
        case NLP_QWEN3:
            if (nTotalSize == 8045591552) {     //  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
                hConfig->n_ctx_train = 262144;  //  Context Length: 262,144 natively
            }
            break;
        case NLP_QWEN2:
            break;
        default:
            break;
    }
    if (isInitFromPath) {
        hConfig->ToJConfig();
    }

    return true;  // isInitFromPath;
}

/*Deprecated*/
bool Fish::YALM_Serialize(const std::string& path, bool isSave, int flag) {
    try {
        if (isSave) {
        } else {
            std::vector<std::string> files;
            DIR* dir = opendir(path.c_str());
            if (dir == nullptr) {
                std::cout << "failed to open directory" << std::endl;
                return -1;
            }
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string filename = entry->d_name;
                // Skip . and .. directory entries
                if (filename != "." && filename != "..") {
                    files.push_back(path + "/" + filename);
                }
            }
            closedir(dir);
            if (files.empty()) {
                std::cout << "no files found" << std::endl;
                return -1;
            }
            std::sort(files.begin(), files.end());

            // Read first file with metadata
            /*if (CALM_Serialize(files[0], true) != 0) {
                std::cout << "failed to read metadata" << std::endl;
                return -1;
            }
            // Read remaining files without metadata
            for (size_t i = 1; i < files.size(); i++) {
                if (CALM_Serialize(files[i], false) != 0) {
                    std::cout << "failed to read file " << files[i] << std::endl;
                    return -1;
                }
            }*/

            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

std::string LoadSomeText(const string& fpath, const int nMost, int flag) {
    assert(nMost > 0);
    string txt = "";
    FILE* fp   = std::fopen(fpath.c_str(), "rt");
    if (fp == NULL)
        return txt;

    // std::fseek(fp, 42, SEEK_SET);
    char buf[nMost + 1] = "\0";
    size_t sz           = std::fread(buf, 1, nMost, fp);
    buf[sz]             = '\0';
    txt                 = buf;
    return txt;
}

// memcpy(tmpData, host_data, szData), msync(host_data, szData, MS_SYNC);
hBITARR SAFE_read_mmap(hBITARR dst, hBITARR mmp_data, size_t length, int flag) {
    if (msync(mmp_data, length, MS_SYNC) == -1) {
        // perror("msync failed");
        //  Continue anyway - data might still be valid
    }
    // Memory barrier to ensure we read after sync
    std::atomic_thread_fence(std::memory_order_acquire);
    memcpy(dst, mmp_data, length);

    // Verify the copy (optional but recommended)
    if (memcmp(dst, mmp_data, length) != 0) {
        std::cerr << "WARNING: Copy verification failed!" << std::endl;
    }

    return dst;
}