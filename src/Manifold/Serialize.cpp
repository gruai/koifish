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
#include "../Tensor/Safetensors.hpp"

std::string K_SafeTensors::config_key_ = "__koifish__config__";

tensor_st::tensor_st(JSON& jDesc, int flag) {
    string jtext = jDesc.dump(4);
    try {
        string info = jDesc["dtype"];
        dtype       = tpNumOf(info);
        assert(jDesc.contains("shape"));
        shape = jKV_arr(jDesc, {"shape"}, shape);
        assert(jDesc.contains("data_offsets"));
        std::vector<size_t> offsets;
        offsets = jKV_arr(jDesc, {"data_offsets"}, offsets);
        assert(offsets.size() == 2);
        data_offsets = {offsets[0], offsets[1]};
    } catch (JSON::parse_error& e) {
        _ERROR("@%s  ERR=%s jtext=%s", __func__, e.what(), jtext.c_str());
        return;
    } catch (...) {
        _ERROR("@%s  jtext=%s", __func__, jtext.c_str());
        return;
    }
}

/**
 * 1. "dtype" field in Hugging Face .safetensors files does not natively support Q4 (4-bit integer quantization). But Koifish's _state_ file support this file
 */
std::string tensor_st::hf_dtype(const typNUMBER dtype) {
    string name = K_FLOATS[dtype].name;
    switch (dtype) {
        case typNUMBER::F32:
            return "F32";
        case typNUMBER::F16:
            return "F16";
        case typNUMBER::BF16:
            return "BF16";
        case typNUMBER::I32:
            return "I32";
        case typNUMBER::I8:
            return "I8";
        case typNUMBER::F8E4M3:
            return "F8_E4M3";
        case typNUMBER::F8E5M2:
            return "F8_E5M2";
        case typNUMBER::U8:
            return "U8";
        default:
            assert(0 && "Hugging Face .safetensors files does not natively support this dtype");
    }
    return name;
}

/*
     support multiple mmap @ different files
     1. AfterBuild->HF_Serialize
     2. AfterBuild->SaveTrain->SAFETENSOR_Serialize
     3.
*/
bool K_SafeTensors::MMAP(const std::string& path, bool isSave, int flag) {
    assert(!isSave);  // only support load now
    _INFO("\n>>>>>> SAFETENSOR_mmap mmap@ %s \"%s\" %s f=%d......", COLOR_ORANGE, path.c_str(), COLOR_RESET, flag);
    try {
        std::string warn, err;
        int __prot = PROT_READ | PROT_WRITE;                                 // PROT_READ
        bool ret   = mmap_from_file(path.c_str(), this, warn, err, __prot);  //   load_from_file();
        int nT     = (int)(tensors.size());
        // assert(nT > 1);
        if (warn.size()) {
            _WARN(">>>>>> SAFETENSOR_mmap@\"%s\" \"%s\"\n", path.c_str(), warn.c_str());  // std::cout << "WARN: " << warn << "\n";
        }
        if (!ret) {
            _ERROR("\n>>>>>> SAFETENSOR_mmap@\"%s\" \"%s\"\n", path.c_str(), err.c_str());
            return false;
        }
        if (!validate_data_offsets(err)) {
            _ERROR(">>>>>> Invalid data_offsets: \"%s\"\n", err.c_str());
            // std::cerr << err << "\n";
            return false;
        }

        const uint8_t* databuffer{nullptr};
        if (mmaped) {
            databuffer = databuffer_addr;  // st->mmap_addr + 8 + st->header_size;
        } else {
            assert(0);
            //  databuffer = storage.data();
        }

        for (size_t i = 0; i < nT; i++) {
            std::string key = tensors.keys()[i];
            if (key == K_SafeTensors::config_key_) {
                tensor_st tensor;
                tensors.at(i, &tensor);
                loadJS(tensor, databuffer, mmap_size);
                if (true) {  // only for debug
                    assert(jsConfig.contains("CLI_params"));
                    assert(jsConfig["CLI_params"].contains("config"));
                    JSON jsC = jsConfig["CLI_params"]["config"];
                    std::ofstream file("./hy-tmp/_koifish_tmp_config_.json");
                    if (file.is_open()) {
                        file << jsC.dump(4);
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
        _ERROR("\r\n%s  Failed to open %s!!! ERR=%s", __func__, path.c_str(), e.what());
        return false;
    } catch (...) {
        _ERROR("\r\nK_SafeTensors::MMAP  Failed to load %s!!!", path.c_str());
        return false;
    }
}

bool SAFETENSOR_Load_jconfig(const std::string& path, JSON& jsConfig, FSerial::FILE_TYPE tpFile, int flag) {
    try {
        K_SafeTensors st(nullptr, {});
        bool bLoad = st.MMAP(path, false, flag);
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

/*
    Called by SAFETENSOR_Serialize
*/
bool K_SafeTensors::_to_ofs(std::ofstream& ofs, size_t& szAll, std::string* warn, std::string* err, int flag) {
    bool isInitMMap = BIT_TEST(flag, FSerial::INIT_MMAP);
    bool isCopyMMap = BIT_TEST(flag, FSerial::COPY_MMAP);
    try {
        fflush(stdout);
        InitHeader();
        uint64_t header_size = header_str.size();  // 39226
        assert(header_size > 0);
        const void* databuffer_addr{nullptr};
        size_t databuffer_size{0};
        if (mmaped) {
            databuffer_size = databuffer_size;
            databuffer_addr = databuffer_addr;
        } else {
            databuffer_size = szAll, databuffer_addr = nullptr;
            // databuffer_size = storage.size();
            // databuffer_addr = reinterpret_cast<const void *>(storage.data());
        }

        size_t pad_bytes = 0;
        if (0) {  // make databuffer addr start from the multiple of 8.
            if ((header_size % 8) != 0) {
                pad_bytes = 8 - (header_size % 8);
            }
        }
        size_t padded_header_size = header_size + pad_bytes;  // 36539
        size_t szOFS = 8 + padded_header_size + databuffer_size, nInit = 0;
        szAll = szOFS;
        fflush(stdout);
        ofs.write(reinterpret_cast<const char*>(&padded_header_size), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(header_str.data()), header_size);
        if (pad_bytes > 0) {  // // Use whitespace for trailing padding.
            std::vector<uint8_t> pad;
            pad.resize(pad_bytes);
            memset(pad.data(), 0x20, pad_bytes);
            ofs.write(reinterpret_cast<const char*>(pad.data()), pad_bytes);
        }
        // memcpy(buffer.data() + 8 + padded_header_size, databuffer_addr, databuffer_size);
        if (databuffer_addr != nullptr)  // mmap file
            ofs.write(reinterpret_cast<const char*>(databuffer_addr), databuffer_size);
        else {
            size_t dst_offset = 0, sz;
            for (size_t i = 0; i < tensors.size(); i++) {
                std::string key = tensors.keys()[i];
                tensor_st tensor;
                tensors.at(i, &tensor);
                assert(dst_offset == tensor.data_offsets[0]);
                if (key != K_SafeTensors::config_key_) {
                    GTensor* t = (GTensor*)(tensor.hUserData);                    
                    sz         = tensor.data_offsets[1] - dst_offset;  // t->nByte() * 3;
                    if (sz != t->nByte_CKP(ckp)) {
                        size_t sz1 = t->nByte_CKP(ckp);
                        assert(0);
                    }
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
//  [bug]
bool K_SafeTensors::_to_memory(std::vector<uint8_t>& buffer, std::string* warn, std::string* err) {
    std::stringstream ss;
    std::string _err;
    if (!validate_data_offsets(_err)) {
        if (err) {
            (*err) += "Invalid safensors is provided.\n" + _err;
        }
        return false;
    }

    InitHeader();

    uint64_t header_size = header_str.size();  // do not include '\n'

    const void* databuffer_addr{nullptr};
    size_t databuffer_size{0};
    if (mmaped) {
        databuffer_size = databuffer_size;
        databuffer_addr = databuffer_addr;
    } else {
        assert(0);
        // databuffer_size = storage.size();
        // databuffer_addr = reinterpret_cast<const void *>(storage.data());
    }

    // make databuffer addr start from the multiple of 8.
    size_t pad_bytes = 0;
    if ((header_size % 8) != 0) {
        pad_bytes = 8 - (header_size % 8);
    }
    // printf("header_size = %d\n", int(header_size));
    // printf("pad_bytes = %d\n", int(pad_bytes));
    size_t padded_header_size = header_size + pad_bytes;  // 20856
    buffer.resize(8 + padded_header_size + databuffer_size);
    size_t szDst = buffer.size();  //  248972672,  248951808
    // write padded header_size
    memcpy(buffer.data(), &padded_header_size, sizeof(size_t));
    // write header
    memcpy(buffer.data() + 8, header_str.data(), header_size);
    // Use whitespace for trailing padding.
    memset(buffer.data() + 8 + header_size, 0x20, pad_bytes);
    memcpy(buffer.data() + 8 + padded_header_size, databuffer_addr, databuffer_size);

    return true;
}

static K_SafeTensors all_states(nullptr, {});
// To read/write mmap many times, a hack for the poor design of safetensors
void CLI_params::InitAllStates(int flag) {
    state          = CheckPoint_Params(jConfig, "", true);
    all_states.ckp = state;
    state.hAllST   = &all_states;
}

void CheckPoint_Params::Init(int flag) {
    // switch (type) {
    //     case STATE:
    //         // To read/write mmap many times, a hack for the poor design of safetensors
    //         hUserData = &all_states;
    //         break;
    //     default:
    //         break;
    // }
}

void HST2JSON(K_SafeTensors* hst, int flag = 0x0) {
    JSON jSafeTensors;
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        tensor_st tensor;
        hst->tensors.at(i, &tensor);
        if (key == K_SafeTensors::config_key_) {
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

int Fish::SAFETENSOR2Gensors(const std::string& path, K_SafeTensors* hst, int flag) {
    int nSerialT = 0;
    hst->Clear();
    bool bLoad = hst->MMAP(path, false, flag);  //*hst
    if (!bLoad)
        return false;
    const uint8_t* databuffer{nullptr};
    if (hst->mmaped) {
        databuffer = hst->databuffer_addr;  // safeTensors->mmap_addr + 8 + safeTensors->header_size;
    } else {
        assert(0);
        // databuffer = safeTensors.storage.data();
    }
    HST2JSON(hst, flag);  //  "_safetensors_.json"
    // Print Tensor info & value.
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        tensor_st tensor;
        hst->tensors.at(i, &tensor);
        if (key == K_SafeTensors::config_key_) {
            continue;
        }
        // if (isOnlyVocab)
        //     continue;

        if (G_Has_(key, config.model.skip_st))
            continue;

        hGensor target = GetGensor(key);  //  "model.embed.weight"    "model.embed_tokens.weight"
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
    assert(nSerialT > 0);
    return nSerialT;
}

K_SafeTensors::K_SafeTensors(Fish* fish, CheckPoint_Params ckp_, int flag) : ckp(ckp_) { UpdateMetaData(flag); }

void K_SafeTensors::UpdateMetaData(int flag) {
    //  jHeader["__metadata__"] = metadata;
    // metadata["vendor"] = "gruai";
    metadata["format"] = "pt";
    metadata["writer"] = "koifish";
}

size_t K_SafeTensors::Register(hGensor t, size_t offset, int flag) {
    size_t sz = t->nByte_CKP(ckp);  // may expand
    // if (ckp.type == CheckPoint_Params::STATE)
    //     sz = t->nByte_CKP();
    assert(sz > 0);

    tensor_st tensor;
    tensor.dtype           = t->type;  // == typNUMBER::F32 ? typNUMBER::F32 : t->type == typNUMBER::F16 ? typNUMBER::F16 : typNUMBER::BF16;
    tensor.hUserData       = t.get();
    tensor.data_offsets[0] = offset;
    tensor.data_offsets[1] = offset + sz;
    tensor.shape.insert(tensor.shape.end(), t->shape.begin(), t->shape.end());

    tensors.insert(t->name, tensor);
    return offset + sz;
}

/*
    Save
        call Save
*/
bool Fish::SAFETENSOR_Serialize(CheckPoint_Params& ckp, bool isSave, int flag) {
    double t0   = GST_ms();
    string path = ckp.FullPath(isSave), jPath = path + "_detail.json";
    if (path.empty()) {
        _INFO("\r\n%s failed: empty path!!! ", __func__);
        return false;
    }

    size_t data_offset_base = 0, nInit = 0, szOFS = 0;
    std::string warn, err;

    vector<hGensor> curParams = optParams;
    K_SafeTensors st(this, ckp), *hst = (K_SafeTensors*)(ckp.hAllST);
    switch (ckp.type) {
        case CheckPoint_Params::STATE:
            assert(hst != nullptr);  // hst=&all_states, so mmp would keep open
            break;
        case CheckPoint_Params::BEST:
            if (config.fuyou.isON()) {
                curParams = GetFuyou(-1)->ckpParams;
                assert(curParams.size() > 0 && "curFuyou is empty!");
            }
            hst = &st;
            break;
        case CheckPoint_Params::FULL:
            // assert(0);
            hst = &st;
            break;
        default:
            assert(0);
    }

    fflush(stdout);
    try {
        if (isSave) {
            JSON jsConfig;
            _INFO(">>>>>> ST_SERIALIZE save @\"%s\" nInit=%ld ......", path.c_str(), nInit);
            jsConfig["vendor"]              = "gruai";
            jsConfig["CLI_params"]          = config.ToJSON(0x100);
            jsConfig["tokenizer"]["tokens"] = "";
            if (jsConfig["CLI_params"]["config"].contains("model") && jsConfig["CLI_params"]["config"]["model"].contains("hf-card")) {
                jsConfig["CLI_params"]["config"]["model"]["#origin-hf-card"] = jsConfig["CLI_params"]["config"]["model"]["hf-card"];
                jsConfig["CLI_params"]["config"]["model"].erase("hf-card");
                jsConfig["CLI_params"]["config"]["model"]["arch"]                            = config.model.model_type;          //"QWEN3";
                jsConfig["CLI_params"]["config"]["model"]["parameter"]["max_pos_embeddings"] = config.model.max_pos_embeddings;  //: 32768
            }

            hst->Clear();
            hst->UpdateMetaData();
            size_t dst_offset = 0;
            if (curParams.size() == 0) {
                _WARN("\r\n%s SAFETENSOR: Save_Params=0!!! @\"%s\"", path.c_str());
            }
            for (auto t : curParams) {
                if (G_Has_(t->name, {"model.layers.0.mlp.down_proj.qweight"})) {  // model.embed_tokens.weight
                    DEBUG_HERE;
                    // t->Print("wte@save", 4, -1);
                }
                if (t->isRefer())  //  "model.out.weight"
                    continue;
                jsConfig["tensors"][t->name] = dst_offset;  // tensor->Dump(100,"");
                dst_offset                   = hst->Register(t, dst_offset);
            }
            if (ckp.format == CheckPoint_Params::KOIFISH) {
                hst->insertJS(jsConfig, dst_offset);
            }
            // HST2JSON(hst, flag);        //  "_safetensors_.json"
            bool ret = hst->Save(path, dst_offset, &warn, &err, flag);
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
            if (!jPath.empty()) {  //  save json file with more info
                std::ofstream o(jPath);
                o << std::setw(4) << jsConfig << std::endl;
            }
            _INFO("\r>>>>>> ST_SERIALIZE save @\"%s\"(\"%s\") nInit=%ld sz=%.6gM flag=%d T=%.4gs\n", path.c_str(), jPath.empty() ? "" : "+json", nInit,
                  szOFS / 1.0e6, flag, (GST_ms() - t0) / 1000.0);
            return true;
        } else {
            int nSerialT = SAFETENSOR2Gensors(path, hst, 0x0);
            _LOG(nSerialT == 0 ? DUMP_ERROR : DUMP_INFO, ">>>>>> ST_SERIALIZE load@\"%s\" nSerialT=%d iter=%d\n", path.c_str(), nSerialT, flag);
            return nSerialT > 0;
        }
    } catch (JSON::parse_error& e) {
        _INFO("\r\n%s  Failed to serialize @\"%s\"!!! ERR=%s", __func__, jPath.c_str(), e.what());
        return false;
    } catch (...) {
        _INFO("\r\n%s  Failed to serialize @\"%s\"!!! ", __func__, path.c_str());
        return false;
    }
}

bool K_SafeTensors::InitHeader(int flag) {
    try {
        // std::stringstream ss;
        // // By default, std::stringstream does not throw exceptions for stream failures (e.g., failbit).
        // ss.exceptions(std::ios::failbit | std::ios::badbit);  // to make it throw on errors.
        // std::string _err;
        jHeader.clear();
        if (metadata.size() == 0)
            return true;
        jHeader["__metadata__"] = metadata;
        // ss << "{";
        // ss << metadata;
        // if (tensors.size()) {
        //     ss << ", ";
        // }
        size_t ntensors = 0;
        for (size_t i = 0; i < tensors.size(); i++) {
            std::string key = tensors.keys()[i];
            tensor_st tensor;
            tensors.at(i, &tensor);
            jHeader[key] = tensor.jDesc();
            ntensors++;
        }
        // ss << "}";
        header_str = jHeader.dump();  //.dump(4);
        // _INFO("%s", jHeader.dump().c_str());

        JSON jsObj;
        jsObj = JSON::parse(header_str);  //  verify
        return true;
    } catch (JSON::parse_error& e) {
        _ERROR("\r\n>>>>>> InitHeader failed @ %s!!! ERR=%s", header_str.c_str(), e.what());
        return false;
    } catch (...) {
        return false;
    }
}

/**
 * [todo] GDS/NVMe direct I/O > MMAP(like https://github.com/xaskasdf/ntransformer)
 */
bool Fish::HF_Serialize(bool isSave, int flag) {
    std::vector<std::string> paths = FilesOfDir(config.model.sCardPath, {"safetensors"}, 0x0);
    if (paths.empty())
        return false;

    int nSerialT = 0;
    std::vector<K_SafeTensors*> st_mmfs;
    for (auto path : paths) {
        K_SafeTensors* hst = new K_SafeTensors(this, {});
        st_mmfs.push_back(hst);
        nSerialT += SAFETENSOR2Gensors(path, hst, 0x0);
    }
    if (isTrain()) {  // otherwise, st would release mmap memory!
        SaveTrain(config.state, true, FSerial::COPY_MMAP);
    }
    for (auto hst : st_mmfs)  // release all mmf resource
        delete hst;

    if (!config.model.st_map.empty()) {  //  "model.safetensors.index.json"
        for (auto kv : config.model.st_map) {
            hGensor target = GetGensor(kv.first);
            assert(target->data != nullptr);
            kv.second = target->data;
        }
        assert(nSerialT == config.model.st_map.size());
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