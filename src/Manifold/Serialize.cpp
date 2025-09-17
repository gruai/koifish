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

#include "../TokenSet/Dictionary.hpp"
#include "../Utils/GST_os.hpp"
#include "Fish.hpp"

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "../Tensor/safetensors.hh"

std::string safetensors::safetensors_t::config_key_ = "__json__config__";

bool SAFETENSOR_Load(const std::string &path, safetensors::safetensors_t &st, int flag) {
    _INFO("\n>>>>>> ST_SERIALIZE load@\"%s\" f=%d......\n", path.c_str(), flag);
    try {
        std::string warn, err;
        int __prot = PROT_READ | PROT_WRITE;                                             // PROT_READ
        bool ret   = safetensors::mmap_from_file(path.c_str(), &st, warn, err, __prot);  //   safetensors::load_from_file();
        if (warn.size()) {
            _INFO(">>>>>> WARN: %s\n", warn.c_str());  // std::cout << "WARN: " << warn << "\n";
        }
        if (!ret) {
            _INFO(">>>>>> ERR: %s\n", err.c_str());
            return false;
        }
        if (!safetensors::validate_data_offsets(st, err)) {
            std::cerr << "Invalid data_offsets\n";
            std::cerr << err << "\n";
            return false;
        }

        if (st.metadata.size()) {
            for (size_t i = 0; i < st.metadata.size(); i++) {
                std::string key = st.metadata.keys()[i], value;
                st.metadata.at(i, &value);
            }
        }
        const uint8_t *databuffer{nullptr};
        if (st.mmaped) {
            databuffer = st.databuffer_addr;  // st->mmap_addr + 8 + st->header_size;
        } else {
            assert(0);
            //  databuffer = st.storage.data();
        }
        int nT = st.tensors.size();
        assert(nT > 0);
        // Print Tensor info & value.
        for (size_t i = 0; i < nT; i++) {
            std::string key = st.tensors.keys()[i];
            if (key == safetensors::safetensors_t::config_key_) {
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                st.loadJS(tensor, databuffer, st.mmap_size);
                if (true) {
                    JSON jsConfig = st.jsConfig["CLI_params"]["config"];
                    std::ofstream file("_koifish_tmp_config_.json");
                    if (file.is_open()) {
                        file << jsConfig.dump(4);
                        file.close();
                    }
                }
                continue;
            }
        }

        return true;
    } catch (JSON::parse_error &e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, path.c_str(), e.what());
        return false;
    } catch (...) {
        return false;
    }
}

bool SAFETENSOR_Load_jconfig(const std::string &path, JSON &jsConfig, int flag) {
    safetensors::safetensors_t st;
    bool bLoad = SAFETENSOR_Load(path, st, flag);
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

    return true;
}

bool Fuyou::Serialize(bool isSave, int flag) {
    for (auto t : tReloads) {
        assert(BIT_TEST(t->flags, GTensor::F_RELOAD));
        assert(t->host_data != nullptr);
        auto now = GST_us();
        assert(t->host_data != nullptr);
        t->Serial_MMAP(isSave);
        SUM::tLoadParam += GST_us() - now;
    }

    return true;
}

using namespace safetensors;

bool safetensors::save_to_ofs(const safetensors_t &st, std::ofstream &ofs, size_t &szAll, std::string *warn, std::string *err, int flag) {
    bool isInitMMap = BIT_TEST(flag, FSerial::INIT_MMAP);
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
        const void *databuffer_addr{nullptr};
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
        ofs.write(reinterpret_cast<const char *>(&padded_header_size), sizeof(size_t));
        // // write header
        // memcpy(buffer.data() + 8, header_str.data(), header_size);
        ofs.write(reinterpret_cast<const char *>(header_str.data()), header_size);
        // // Use whitespace for trailing padding.
        // memset(buffer.data() + 8 + header_size, 0x20, pad_bytes);
        std::vector<uint8_t> pad;
        pad.resize(pad_bytes);
        memset(pad.data(), 0x20, pad_bytes);
        ofs.write(reinterpret_cast<const char *>(pad.data()), pad_bytes);
        // memcpy(buffer.data() + 8 + padded_header_size, databuffer_addr, databuffer_size);
        if (databuffer_addr != nullptr)  // mmap file
            ofs.write(reinterpret_cast<const char *>(databuffer_addr), databuffer_size);
        else {
            size_t dst_offset = 0, sz;
            for (size_t i = 0; i < st.tensors.size(); i++) {
                std::string key = st.tensors.keys()[i];
                safetensors::tensor_t tensor;
                st.tensors.at(i, &tensor);
                assert(dst_offset == tensor.data_offsets[0]);
                if (key != safetensors::safetensors_t::config_key_) {
                    GTensor *t = (GTensor *)(tensor.hUserData);
                    sz  = t->nByte() * 3;
                    assert(tensor.data_offsets[1] - dst_offset == sz);
                    char *tmp = new char[sz]();
                    if (t->GetDataX() == nullptr) {
                        if (isInitMMap) {
                            nInit++;
                        } else {
                            _INFO("[ST_SERIALIZE] \"%s\" is empty!\n", t->name);
                            return false;
                        }
                        // memset(databuffer_size + dst_offset, 0x0, sz);
                    } else {
                        assert(t->data != nullptr && t->gm != nullptr);
                        t->SerialData("", tmp, true);
                    }
                    ofs.write(reinterpret_cast<const char *>(tmp), sz);
                    delete[] tmp;
                } else {
                    sz = tensor.msgpack.size();
                    const char *tmp = (const char *)tensor.msgpack.data();
                    ofs.write(tmp, sz);
                }
                dst_offset += sz;
            }
        }

        fflush(stdout);
        // _INFO(">>>>>> saveto_ofs ......OK\n");
        return true;
    } catch (const std::exception &e) {
        _INFO("\n!!!!saveto_ofs excetioin=%s sz=%ld!!!\n", e.what(), szAll);
        return false;
    } catch (...) {
        _INFO("\n!!!!saveto_ofs Unknown exception occurred sz=%ld!!!\n", szAll);
        return false;
    }
}

static safetensors::safetensors_t safeTensors;
bool Fish::SAFETENSOR_Serialize(const std::string &path, bool isSave, int flag) {
    double t0               = GST_ms();
    string jPath            = path + ".json";
    size_t data_offset_base = 0, nInit = 0, szOFS = 0;
    std::string warn, err;
    JSON jsConfig;
    try {
        if (isSave) {
            fflush(stdout);
            _INFO(">>>>>> ST_SERIALIZE save @\"%s\" nInit=%ld ......", path.c_str(), nInit);

            bool isInitMMap                 = BIT_TEST(flag, FSerial::INIT_MMAP);
            jsConfig["vendor"]              = "gruai";
            jsConfig["CLI_params"]          = config.ToJSON();
            jsConfig["tokenizer"]["tokens"] = "";
            safeTensors.Clear();
            size_t dst_offset = 0;
            for (auto t : optParams) {
                std::vector<floatX> weight;
                // size_t dst_offset = safeTensors.storage.size();
                size_t sz = t->nByte();  // expand
                sz *= 3;
                assert(sz > 0);
                /*safeTensors.storage.resize(dst_offset + sz);
                char *dst = (char *)(safeTensors.storage.data());
                if (t->GetDataX() == nullptr) {
                    if (isInitMMap) {
                        nInit++;
                    } else {
                        _INFO("[ST_SERIALIZE] \"%s\" is empty!\n", t->name);
                        return false;
                    }
                    // memset(dst + dst_offset, 0x0, sz);
                } else {
                    assert(0);  // should never call this!
                    assert(t->data != nullptr && t->gm != nullptr);
                    // D2H(t->data, (char *)dst + dst_offset, t->szData);
                    // D2H(t->gm, (char *)dst + dst_offset + t->szData, t->szM + t->szV);
                }*/
                // memcpy(safeTensors.storage.data() + dst_offset, t->data, sz);
                safetensors::tensor_t tensor;
                tensor.dtype           = t->type == typNUMBER::F32   ? safetensors::dtype::kFLOAT32
                                         : t->type == typNUMBER::F16 ? safetensors::dtype::kFLOAT16
                                                                     : safetensors::dtype::kBFLOAT16;
                tensor.hUserData       = t.get();
                tensor.data_offsets[0] = dst_offset;
                tensor.data_offsets[1] = dst_offset + sz;
                tensor.shape.insert(tensor.shape.end(), t->shape.begin(), t->shape.end());
                safeTensors.tensors.insert(t->name, tensor);

                jsConfig["tensors"][t->name]  = "";  // tensor->Dump(100,"");
                jsConfig["tensors"]["offset"] = dst_offset;
                dst_offset += sz;
            }
            safeTensors.insertJS(jsConfig,dst_offset);
            // __metadata__
            safeTensors.metadata.insert("vendor", "gruai");
            bool ret = safetensors::save_to_file(safeTensors, path, dst_offset, &warn, &err, flag);
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

            //  save json file
            std::ofstream o(jPath);
            o << std::setw(4) << jsConfig << std::endl;
            _INFO("\r>>>>>> ST_SERIALIZE save @\"%s\" nInit=%ld sz=%.6gM flag=%d T=%.4gs\n", path.c_str(), nInit, szOFS / 1.0e6, flag,
                  (GST_ms() - t0) / 1000.0);
            return true;
        } else {
            int nSerialT = 0;
            safeTensors.Clear();
            bool bLoad = SAFETENSOR_Load(path, safeTensors, flag);
            if (!bLoad)
                return false;
            const uint8_t *databuffer{nullptr};
            if (safeTensors.mmaped) {
                databuffer = safeTensors.databuffer_addr;  // safeTensors->mmap_addr + 8 + safeTensors->header_size;
            } else {
                assert(0);
                // databuffer = safeTensors.storage.data();
            }

            // Print Tensor info & value.
            for (size_t i = 0; i < safeTensors.tensors.size(); i++) {
                std::string key = safeTensors.tensors.keys()[i];
                safetensors::tensor_t tensor;
                safeTensors.tensors.at(i, &tensor);
                if (key == safetensors::safetensors_t::config_key_) {
                    /*    safeTensors.loadJS(tensor, databuffer, safeTensors.mmap_size);
                        jsConfig = safeTensors.jsConfig["CLI_params"]["config"];
                        std::ofstream file("ST_SERIALIZE.json");
                        if (file.is_open()) {
                            file << jsConfig.dump(4);
                            file.close();
                        }*/
                    continue;
                }
                hGensor target = GetGensor(key);  //  "model.embed.weight"    model.layers.0.attn_norm.weight
                if (target == nullptr) {
                    _INFO("\t[SERIAL] Failed @%s!\n", key.c_str());
                    continue;
                }
                JSON jdesc = tensor.jDesc();
                if (target->SerialJSON(key, jdesc, (void *)databuffer, safeTensors.mmap_size) != 0) {
                    return false;
                }
                if (DUMP()) {
                    tensor.Dump(key, databuffer);
                    _INFO("  >>>>  %d typ=%s\t data=%p grad=%p \t sz=%ld @%s\n", nSerialT, cNameOf(target->type), target->data, target->grad, tBYTE(target),
                          target->name);
                }
                // if(strcmp(target->name,"model.layers.27.mlp.norm.weight")==0){   //only for debug model.output.weight
                //     target->Print(key,0,-1);                //PrintTensor<f8e5m2_t>("wout",target->data,target->ne[0],dim);
                // }

                nSerialT++;
            }
            // config.model.Init(jsConfig)
            _INFO("\n>>>>>> ST_SERIALIZE load@\"%s\" nSerialT=%d iter=%d\n", path.c_str(), nSerialT, flag);
            return true;
        }
    } catch (JSON::parse_error &e) {
        _INFO("\r\n%s  Failed to open %s!!! ERR=%s", __func__, jPath.c_str(), e.what());
        return false;
    } catch (...) {
        return false;
    }
}

void *MMAP_json(JSON &header, void **objs, size_t *objs_nz, const std::string &path, bool isSave, int flag) {
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
    void *data  = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
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

    uint64_t json_size = *(uint64_t *)data;
    if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
        munmap(data, size);
        return nullptr;
    }

    char *json_ptr    = (char *)data + sizeof(uint64_t);
    void *bytes_ptr   = (char *)data + sizeof(uint64_t) + json_size;
    size_t bytes_size = size - sizeof(uint64_t) - json_size;

    std::string json_str(json_ptr, json_size);
    header   = JSON::parse(json_str);
    *objs    = bytes_ptr;
    *objs_nz = bytes_size;
    return data;
}

bool Fish::HF_Serialize(bool isSave, int flag) { return false; }

bool MODEL_CARD::OnJsonCALM(CLI_params *hConfig, const std::string &path, const JSON &meta, int flag) {
    sTokenPath = path;
    dim        = jKVs(meta, {"dim"}, dim);  // atoi(meta["dim"]);
    assert(dim == hConfig->nEmbed());
    hidden_dim = jKVs(meta, {"hidden_dim"}, hidden_dim);  // atoi(meta["hidden_dim"]);
    n_layers   = jKVs(meta, {"n_layers"}, n_layers);      // atoi(meta["n_layers"]);
    assert(n_layers == hConfig->nLayer());
    n_heads    = jKVs(meta, {"n_heads"}, n_heads);        // atoi(meta["n_heads"]);
    n_kv_heads = jKVs(meta, {"n_kv_heads"}, n_kv_heads);  // atoi(meta["n_kv_heads"]);
    head_dim   = jKVs(meta, {"head_dim"}, head_dim);
    assert(dim == head_dim * n_heads);
    layerps.clear();
    for (int i = 0; i < n_layers; i++) {
        LAY_PARAM lay(n_heads, n_kv_heads, hidden_dim);
        layerps.push_back(lay);
    }
    string info = "";
    info        = jKVs(meta, {"dtype"}, info);
    tpWeight    = tpNumOf(info);
    int wbit    = BitPE(tpWeight);
    fDotW       = fnDot(tpWeight);

    jModelParam  = meta;
    vocab_size   = jKVs(meta, {"vocab_size"}, 0);
    bos_token_id = jKVs(meta, {"bos_token_id"}, 0);
    eos_token_id = jKVs(meta, {"eos_token_id"}, 0);
    //
    // const char* max_seq_len = meta["max_seq_len"];
    // config.seq_len = max_seq_len && atoi(max_seq_len) < 4096 ? atoi(max_seq_len) : 4096;
    seq_len = jKVs(meta, {"max_seq_len"}, 0);
    seq_len = std::min(seq_len, 4096);  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly
                                        // specified if (context) { 	config.seq_len = context;
                                        // }

    // config.rope_theta = atof(meta["rope_theta"]);
    // config.rotary_dim = atoi(meta["rotary_dim"]);
    rope_theta = jKVs(meta, {"rope_theta"}, rope_theta);
    rotary_dim = jKVs(meta, {"rotary_dim"}, rotary_dim);
    // if (meta["n_experts"]) {
    // 	config.n_experts = atoi(meta["n_experts"]);
    // 	config.n_experts_ac = atoi(meta["n_experts_active"]);
    // }
    n_experts    = jKVs(meta, {"n_experts"}, n_experts);
    n_experts_ac = jKVs(meta, {"n_experts_active"}, n_experts_ac);
    // const char* norm_eps = meta["norm_eps"];
    norm_eps = jKVs(meta, {"norm_eps"}, norm_eps);

    // const char* act_type = meta["act_type"];
    act_type = jKVs(meta, {"act_type"}, act_type);  // act_type && strcmp(act_type, "gelu"] == 0;

    // const char* norm_type = meta["norm_type"];
    norm_type = jKVs(meta, {"norm_type"}, norm_type);  // act_type && strcmp(act_type, "gelu"] == 0;
    // config.norm_ln = norm_type && strncmp(norm_type, "layernorm", 9) == 0;  // note: we currently don't support layernorm bias
    // config.norm_par = norm_type && strcmp(norm_type, "layernorm_par"] == 0; // note: we currently don't support layernorm bias

    // const char* qkv_clip = meta["qkv_clip"];
    // config.qkv_clip = qkv_clip ? atof(qkv_clip) : FLT_MAX;
    clip_qkv = jKVs(meta, {"qkv_clip"}, clip_qkv);

    assert(hConfig->isValid());
    return true;
}

bool MODEL_CARD::InitHF(CLI_params *hConfig, const JSON &jConfig, int flag) {
    n_layers = hConfig->nLayer();
    for (int i = 0; i < n_layers; i++) {
        int nH = hConfig->n_head(i), nF = hConfig->n_ff(i);
        assert(nH > 0 && nF > 0);
    }

    string sTyp = jKVs(jConfig, {"model", "datatype", "weight"}, string(""));
    if (!sTyp.empty())
        tpWeight = tpNumOf(sTyp);
    sTyp = jKVs(jConfig, {"model", "datatype", "embed"}, string(""));
    if (!sTyp.empty())
        tpEmbed = tpNumOf(sTyp);
    head_dim   = hConfig->n_embd_head();
    n_kv_heads = hConfig->n_head_kv();
    seq_len    = hConfig->n_ctx();
    assert(seq_len < 4096);  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly
                             // specified if (context) { 	config.seq_len = context;

    sCardPath = jKV(jConfig, {"model_card"}, sCardPath);
    if (sCardPath.empty()) {
        return false;
    }
    string jPath = sCardPath + "config.json";
    LoadJsonFile(jPath, jModelParam);
    sTokenPath = sCardPath + "tokenizer.json";
    // LoadJsonFile(sTokenPath,jTokenizer);                             // }

    if (jModelParam.empty()) {
        sCardPath = "";
    } else {
        model_type = jKV(jModelParam, {"model_type"}, model_type);
        if (model_type == "")
            sCardPath = "";
        else {
            vocab_size           = jKV(jModelParam, {"vocab_size"}, vocab_size);
            torch_dtype          = jKV(jModelParam, {"torch_dtype"}, torch_dtype);
            transformers_version = jKV(jModelParam, {"transformers_version"}, transformers_version);
            bos_token_id         = jKV(jModelParam, {"bos_token_id"}, bos_token_id);
            eos_token_id         = jKV(jModelParam, {"eos_token_id"}, eos_token_id);
        }
    }

    return empty();
}

bool Fish::CALM_Serialize(const std::string &path, bool isOnlyVocab, int flag) {
    try {
        JSON header;
        size_t objs_size;
        int nSerialT = 0;
        void *objs, *data = MMAP_json(header, &objs, &objs_size, path, false, flag);
        hGTensor tokens = std::make_shared<GTensor>(), scores = std::make_shared<GTensor>();
        if (data == nullptr)
            return false;

        for (auto &[key, val] : header.items()) {
            if (key == "__metadata__") {
                JSON metadata = val;
                std::cout << "\n\t__metadata__ " << metadata << std::endl << std::endl;
                config.model.OnJsonCALM(&config, path, metadata, 0x0);

                // InitDictTokenset();
                if (isOnlyVocab)
                    continue;
            } else if (key == "tokenizer.tokens") {
                tokens->SerialJSON(key, val, objs, objs_size, flag | GTensor::F_NOALLOC);
            } else if (key == "tokenizer.scores") {
                scores->SerialJSON(key, val, objs, objs_size, flag | GTensor::F_NOALLOC);
            } else {
                if (isOnlyVocab)
                    continue;

                hGensor target = GetGensor(key);  //  "model.embed.weight"    model.layers.0.attn_norm.weight
                if (target == nullptr) {
                    _INFO("\t[SERIAL] Failed @%s!\n", key.c_str());
                    continue;
                }

                if (target->SerialJSON(key, val, objs, objs_size) != 0) {
                    munmap(data, size);
                    return -1;
                }  //

                if (G_Has_(target->name, {"mlp.w1.weight"})) {  // "layers.27.mlp.w1.weight" wk.weight wq.weight wv.weight wo.weight ,"w2.weight","w3.weight"
                    // BIT_SET(target->flags, GTensor::F_TERNARY_);
                    // target->ToTernary();
                }
                if (DUMP()) {
                    _INFO("  >>>>  %d typ=%s\t data=%p grad=%p \t sz=%ld @%s\n", nSerialT, cNameOf(target->type), target->data, target->grad, tBYTE(target),
                          target->name);
                }
                // if(strcmp(target->name,"model.layers.27.mlp.norm.weight")==0){   //only for debug model.output.weight
                //     target->Print(key,0,-1);                //PrintTensor<f8e5m2_t>("wout",target->data,target->ne[0],dim);
                // }
                nSerialT++;
            }
        }
        _INFO("[SERIAL] n_T=%d\n", nSerialT);
        hDict->InitFrom(this, tokens, scores, 0x0);
        tokens.reset();
        scores.reset();
        return true;
    } catch (...) {
        return false;
    }
}
/**/
bool Fish::YALM_Serialize(const std::string &path, bool isSave, int flag) {
    try {
        if (isSave) {
        } else {
            std::vector<std::string> files;
            DIR *dir = opendir(path.c_str());
            if (dir == nullptr) {
                std::cout << "failed to open directory" << std::endl;
                return -1;
            }
            struct dirent *entry;
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
            if (CALM_Serialize(files[0], true) != 0) {
                std::cout << "failed to read metadata" << std::endl;
                return -1;
            }
            // Read remaining files without metadata
            for (size_t i = 1; i < files.size(); i++) {
                if (CALM_Serialize(files[i], false) != 0) {
                    std::cout << "failed to read file " << files[i] << std::endl;
                    return -1;
                }
            }

            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

std::string LoadSomeText(const string &fpath, const int nMost, int flag) {
    assert(nMost > 0);
    string txt = "";
    FILE *fp   = std::fopen(fpath.c_str(), "rt");
    if (fp == NULL)
        return txt;

    // std::fseek(fp, 42, SEEK_SET);
    char buf[nMost + 1] = "\0";
    size_t sz           = std::fread(buf, 1, nMost, fp);
    buf[sz]             = '\0';
    txt                 = buf;
    return txt;
}
