/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
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
#include "../Utils/GST_Application.hpp"
#include "../Utils/GST_os.hpp"
#include "Fish.hpp"

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "../Tensor/Safetensors.hpp"

std::string K_SafeTensors::config_key_ = "__koifish__config__";

// from json desc entry of each tensor in some checkpoint files
GTensor::GTensor(Fish* _hFish, const string& _name, JSON& jEntry, int flag) {
    // assert(_hFish != nullptr);       //  when SAFETENSOR_Load_jconfig, _hFish is still nullptr
    strcpy(name, _name.c_str());
    hFish        = _hFish;
    string jtext = jEntry.dump(4);
    try {
        string info = jEntry["dtype"];
        type        = tpNumOf(info);
        assert(jEntry.contains("shape"));
        shape = jKV_arr(jEntry, {"shape"}, shape);
        assert(jEntry.contains("data_offsets"));
        std::vector<size_t> offsets;
        offsets = jKV_arr(jEntry, {"data_offsets"}, offsets);
        assert(offsets.size() == 2);
        data_offsets = {offsets[0], offsets[1]};
        szData       = jKV(jEntry, {"szData"}, szData, false);
        szGama       = jKV(jEntry, {"szGama"}, szGama, false);
        if (szData + szGama > 0) {
            // assert(szData + szGama == offsets[1] - offsets[0]);
        }
    } catch (JSON::parse_error& e) {
        _ERROR("@%s  ERR=%s jtext=%s", __func__, e.what(), jtext.c_str());
        return;
    } catch (...) {
        _ERROR("@%s  jtext=%s", __func__, jtext.c_str());
        return;
    }
}
JSON GTensor::jDesc(K_SafeTensors* st, int flag) {
    JSON js;
    typNUMBER jsType = type;
    SHAPE jsShape    = shape;
    string info      = K_FLOATS[type].name;
    // if (hFish->config.model.ckp_format == CKP_HF) {
    if (st->ckp.format == CKP_HF) {  //  HF don't support following type!
        if (type == typNUMBER::Q4) {
            jsType = typNUMBER::I32;
            assert(shape[0] % 8 == 0);
            jsShape[0] /= 8;
        } else if (type == typNUMBER::T_BINARY) {
            assert(0 && "Many tools(lm_eval...) failed to load these packed-I32 format quant model!");
            /*jsType = typNUMBER::I32;
            assert(shape[0] % 32 == 0);
            jsShape[0] /= 32;*/
        } else if (type == typNUMBER::T_SIGN || type == typNUMBER::Q2) {
            jsType = typNUMBER::BF16;
            // Even its easy to save in packed-I32 format, many tools(lm_eval...) failed to load these quant model!
            /*jsType = typNUMBER::I32;
            assert(shape[0] % 16 == 0);
            jsShape[0] /= 16;*/
        }
        info = HF_dtype2str(jsType);
    }
    assert(!info.empty());
    js["dtype"]        = info;
    js["shape"]        = jsShape;
    js["data_offsets"] = {data_offsets[0], data_offsets[1]};
    if (st->ckp.format == CKP_HF) {
    } else {  // only kifish support more items!
        js["szGama"] = szGama;
        js["szData"] = szData;
        if (st->ckp.state_type != CheckPoint_Params::STATE && (szData + szGama) > 0) {  // name!=K_SafeTensors::config_key_
            assert(szData + szGama == data_offsets[1] - data_offsets[0]);
        }
    }

    return js;
}

void* JSON2SRC(K_SafeTensors* hst, const JSON& jval, const void* bytes_ptr, const size_t bytes_size, size_t& szSrc, GTensor* tensor) {
    if (jval.at("data_offsets").size() != 2) {
        return nullptr;
    }
    size_t offset_start = static_cast<size_t>(jval.at("data_offsets")[0]);  // 1544148992
    size_t offset_end   = static_cast<size_t>(jval.at("data_offsets")[1]);  // 1545276932
    if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
        _ERROR("bad offsets");
        return nullptr;
    }
    size_t szData = tensor->nByte_CKP(hst->ckp);
    szSrc         = offset_end - offset_start;
    if (szData > szSrc || szSrc % szData != 0) {
        _WARN("GTensor::LoadParam_ failed! size mismach[%ld!=%ld] @%s", szData * 3, szSrc, tensor->name);
        return nullptr;
    }

    void* src = (char*)bytes_ptr + offset_start;
    return src;
}
SHAPE JSON2SHAPE(const JSON& jval) {
    SHAPE spJ;
    if (jval.at("shape").size() > 4) {
        _ERROR("JSON2SHAPE: shape exceeds 4 dimensions");
    }
    spJ          = jKV_arr(jval, {"shape"}, spJ);
    size_t numel = SHAPE2NZ(spJ);
    return spJ;
}

/*
    ---- stage one
    1. Load param from HF "model.safetensors"
        a .weight
        b AWQ model (.qweight, .zeros, .scales)
    2. Load param from Koifish state file
        a load bf16 weight
        b load quantized weight (dequant it or not)

    ---- stage two
    1. quant weight (q4->q2 is also quant)
*/
int GTensor::LoadParam(K_SafeTensors* hst, const std::string& name_, hGTensor kunsor, void* bytes_ptr, size_t bytes_size, int flag) {
    //  "tokenizer.tokens" model.layers.0.mlp.down_proj.qweight   "model.embed_tokens.weight" t->name,
    if (G_Has_(name, {"model.embed_tokens.weight"})) {  //  model.layers.0.self_attn.q_proj.qweight
        DEBUG_HERE;
    }
    if (kunsor == nullptr) {  // load from host_data of mmp
        assert(host_data != nullptr && data != nullptr);
        Serial_Quant_MMAP(false, false, LOAD_ONLY_W);
        //  G_NORM_STAT<bf16>(size(), (bf16*)host_data, disq.sum_2, disq.sum_1, disq.nrm_1);
        return 0x0;
    }
    // from HF "model.safetensors"
    if (strcmp(name, name_.c_str()) != 0) {
        strcpy(name, name_.c_str());
    }
    SHAPE spJ = kunsor->shape;  // JSON2SHAPE(jval);
    // std::string dtype_str = jval.value("dtype", "");
    typNUMBER tpMMP = kunsor->type;
    if (tpMMP == typNUMBER::I32 && hQuant != nullptr) {
        tpMMP = hQuant->bit2typ();
        if (hst->ckp.format == FILE_FORMAT_TYPE::CKP_KOIFISH && tpMMP == typNUMBER::Q2) {
            assert(0);
            tpMMP = typNUMBER::T_SIGN;
        }
        size_t nByte = K_FLOATS[tpMMP].nByte(SHAPE2NZ(shape));
        assert(nByte == SHAPE2NZ(spJ) * 4 && name);
    } else {
        if (hQuant != nullptr) {  // *.scales F16
            DEBUG_HERE;
        } else
            assert(shape == spJ && name);
    }

    ReShape(shape, tpMMP);
    if (hst->ckp.format == FILE_FORMAT_TYPE::CKP_KOIFISH) {
        assert(szData == kunsor->szData);
        szGama = kunsor->szGama;
    }
    size_t szSrc = kunsor->data_offsets[1] - kunsor->data_offsets[0];
    void* src    = (hBITARR)bytes_ptr + kunsor->data_offsets[0];
    /*void* src_1  = JSON2SRC(hst, kunsor->jDesc(hst), bytes_ptr, bytes_size, szSrc, this);
    if (src != src_1) {
        DEBUG_HERE;
    }*/
    if (BIT_TEST(flag, F_NOALLOC)) {  //  "tokenizer.tokens","tokenizer.scores"
        data = src;                   // ((char*)(src))[szSrc-1]    (char*)bytes_ptr + offset_end-1
        BIT_SET(flags, F_MMAP);
        /*if (DEBUG.T_cpu == 1) {       // reserve for future CPU version
            data = src;
            BIT_SET(flags, F_MMAP);
        }*/
    } else {
        assert(data == nullptr);
        host_data = src;
        if (!hFish->isTrain()) {  // otherwize, mmap file is free & host_data is invalid
            tpInit = W_SKIP;
            Alloc(-1, flag);
            Serial_Quant_MMAP(false, false);
            host_data = nullptr;  // mmap file would release
        } else {
            if (DEBUG.save_GlobalSate <= 0) {
                tpInit = W_SKIP;
                Alloc(-1, flag);
                Serial_Quant_MMAP(false, false, LOAD_ONLY_W);
                SUM::nInitParam++;
                host_data = nullptr;             // mmap file would release
            } else if (type == typNUMBER::BF16)  // get original disq of tensor stored in mmp
                G_NORM_STAT<bf16>(size(), (bf16*)src, disq.sum_2, disq.sum_1, disq.nrm_1);
        }
        // if (G_Has_(name, hFish->config.quant.filter_WeightF8Ex)) {  // model.embed_tokens.weight    only for debug
        //     ToF8Ex(0x0);
        // }
    }
    if (strlen(name) > 0 && flag > 0)
        DumpX(0);
    if (G_Has_(name, {"model.layers.0.input_layernorm.weight"})) {
        // Print(name,0,-1);
    }
    return 0;
}
//  "./checkpoints/._koifish_state_.ckp" @K_SafeTensors::Register_ ckp.type == CheckPoint_Params::STATE
size_t GTensor::nByte_CKP(const CheckPoint_Params& ckp, int flag) const {
    size_t sz = nByte();
    if (sz == 0) {
        // assert(K_SafeTensors::config_key_==(string)(name));
    }
    if (hFish != nullptr && flag >= 0) {
        string opt = hFish->config.common.method;
        if (hFish->isTrain() && ckp.state_type == CheckPoint_Params::STATE) {
            if (opt == "muon") {
                if (hFish->config.common.muon.isAdamW((void*)this))
                    sz *= 3;
                else
                    sz *= 2;  // only gm
            } else
                sz *= 3;  // gm+gv
        }
    } else {  //  @SAFETENSOR_Load_jconfig
    }

    if (szGama > 0) {
        sz += szGama;
    }

    return sz;
}

/**
 * 1. "dtype" field in Hugging Face .safetensors files does not natively support Q4 (4-bit integer quantization). But Koifish's _state_ file support this file
 */
std::string HF_dtype2str(const typNUMBER dtype) {
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
     1. AfterBuild->SafeTensors_Serialize
     2. AfterBuild->SaveTrain->SAFETENSOR_Serialize
     3.
*/
bool K_SafeTensors::MMAP(const std::string& path, bool isSave, int flag) {
    assert(!isSave);  // only support load now
    _INFO(">>>>>> SAFETENSOR_mmap mmap@ %s \"%s\" %s f=%d......", COLOR_ORANGE, path.c_str(), COLOR_RESET, flag);
    try {
        ckp.format = FormatOfFile(path);
        if (ckp.format == FILE_FISH) {
            ckp.format = CKP_KOIFISH;
        }

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
                hGTensor tensor = tensors.at(i);
                loadJS(tensor, databuffer, mmap_size);
                string sJsonPath = path;
                std::replace(sJsonPath.begin(), sJsonPath.end(), '/', '_');
                if (true) {  // only for debug
                    assert(jsConfig.contains("CLI_params"));
                    assert(jsConfig["CLI_params"].contains("config"));
                    // JSON2FILE("./log/@[" + sJsonPath + "].json");
                    JSON jsC = jsConfig["CLI_params"]["config"];
                    std::ofstream file("./log/@[" + sJsonPath + "].json");
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

bool SAFETENSOR_Load_jconfig(const std::string& path, JSON& jsConfig, FILE_FORMAT_TYPE tpFile, int flag) {
    try {
        K_SafeTensors st(nullptr, {}, path);
        bool bLoad = st.MMAP(path, false, flag);
        if (!bLoad) {
            _ERROR("\r\n SAFETENSOR_Load_jconfig failed to MMAP %s!!!", path.c_str());
            return false;
        }

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
            case FILE_FISH:
            case FILE_CHECKPOINT:

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

        size_t pad_bytes = 0, nExpDataX = 0;
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
            size_t dst_offset = 0, szTmp, szRead;
            for (size_t i = 0; i < tensors.size(); i++) {
                int nExp        = 0;
                std::string key = tensors.keys()[i];
                hGTensor t      = tensors.at(i);
                assert(dst_offset == t->data_offsets[0]);
                if (key != K_SafeTensors::config_key_) {
                    // GTensor* t = (GTensor*)(tensor.hUserData);
                    szTmp = t->data_offsets[1] - dst_offset;  // t->nByte() * 3;
                    if (szTmp != t->nByte_CKP(ckp)) {
                        size_t sz1 = t->nByte_CKP(ckp);
                        if (ckp.format == FILE_FORMAT_TYPE::CKP_HF && K_FLOATS[t->type].bis <= 4) {
                            nExp = szTmp / sz1;
                            nExpDataX++;
                        } else
                            assert(0);
                    }
                    hBITARR tmp = new BIT_8[szTmp]();
                    if (G_Has_(t->name, {"model.layers.27.mlp.down_proj.weight"})) {  // model.embed_tokens.weight  model.layers.0.self_attn.q_proj
                        DEBUG_HERE;
                        // t->Print("wte@save", 4, -1);
                    }
                    // t->Serial_Quant_MMAP(true,false,SAVE_TO_TMPDATA, tmp);
                    if (t->GetDataX() == nullptr) {  // copy data@"model.safetensors" to state file
                        if (isInitMMap) {
                            nInit++;
                        } else if (isCopyMMap) {
                            assert(t->host_data != nullptr);
                            szRead = t->nByte_CKP(ckp, -1);
                            assert(szRead <= szTmp);
                            SAFE_read_mmap(tmp, (hBITARR)(t->host_data), szRead);
                            t->tpInit = INIT_WEIGHT::SERIALIZE;
                        } else {
                            _ERROR("[ST_SERIALIZE] \"%s\" is empty!\n", t->name);
                            return false;
                        }
                        // memset(databuffer_size + dst_offset, 0x0, sz);
                    } else {
                        assert(t->data != nullptr && t->gm != nullptr);
                        // t->Print(t->name, 0, -1);    //only for debug
                        if (nExp > 0) { //nExp = szTmp / sz1
                            D2H(gBUFF->tmpTernary->data, tmp, szTmp);
                        } else {
                            t->SerialGamaData("", tmp, true, szTmp);
                        }
                        if (t->GetDynamicQuant() != nullptr) {
                            DEBUG_HERE;
                        }
                        // t->Print(t->name, 0, -1);
                    }
                    ofs.write(reinterpret_cast<const char*>(tmp), szTmp);
                    delete[] tmp;
                } else {
                    szTmp           = t->msgpack.size();
                    const char* tmp = (const char*)(t->msgpack.data());
                    ofs.write(tmp, szTmp);
                }
                dst_offset += szTmp;
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

static K_SafeTensors all_states(nullptr, {}, "all_states");
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

void HST2JSON(const std::string& path, K_SafeTensors* hst, int flag = 0x0) {
    JSON jSafeTensors;
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        hGTensor tensor = hst->tensors.at(i);
        if (key == K_SafeTensors::config_key_) {
            continue;
        }
        JSON jdesc    = tensor->jDesc(hst);
        jdesc["name"] = key;
        jSafeTensors.push_back(jdesc);
    }
    JSON jX;
    jX["ckp"]  = hst->ckp.sModelPath;
    jX["path"] = hst->sFolderPath;
    jX["APP"]  = g_sAppPath;

    jX["Time"] = GST_timeStr();
    jSafeTensors.push_back(jX);

    assert(!jSafeTensors.empty());
    std::ofstream file(path);
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
    HST2JSON("./_safetensors_.json", hst, flag);
    // Print Tensor info & value.
    for (size_t i = 0; i < hst->tensors.size(); i++) {
        std::string key = hst->tensors.keys()[i];
        hGTensor kunsor = hst->tensors.at(i);  // tensor info from .kun
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
            return -1;
        }
        auto ginfo = GetGensorInfo(target);
        // JSON jdesc = tensor->jDesc(hst);
        if (G_Has_(key, {"model.layers.27.mlp.down_proj.weight"})) {  // model.embed_tokens.weight model.layers.0.mlp.down_proj.weight
            DEBUG_HERE;
            // target->Print("wte@ST", 4, -1);
        }

        if (target->LoadParam(hst, key, kunsor, (void*)databuffer, hst->mmap_size) != 0) {
            return false;
        }
        // if (target->szGama > 0)
        //     target->Print(target->name, 4, -1);

        if (DUMP() || config.dumpSwitch.tensor_load > 0 /*|| target->data == nullptr*/) {
            // tensor.Dump("  >>>>  " + key, databuffer);
            _INFO("  >>>>  [%d] typ=%s\t data=%p grad=%p \t sz=%ld @%s\n", nSerialT, cNameOf(target->type), target->data, target->grad, tBYTE(target),
                  target->name);
        }
        // if (G_Has_(target->name, {"model.embed_tokens.weight"})) {  //  model.layers.0.self_attn.q_proj.qweight
        //     target->Print(key, 0, -1);
        // }

        nSerialT++;
    }
    assert(nSerialT > 0);
    return nSerialT;
}

K_SafeTensors::K_SafeTensors(Fish* fish, CheckPoint_Params ckp_, const std::string& path, int flag) : ckp(ckp_), sFolderPath(path) {
    hFish = fish;
    if (fish != nullptr) {
        // ckp_format = fish->config.model.ckp_format;
        // ckp_format = FormatOfFile(path);
    } else {
    }
    UpdateMetaData(flag);
}

void K_SafeTensors::UpdateMetaData(int flag) {
    //  jHeader["__metadata__"] = metadata;
    // metadata["vendor"] = "gruai";
    metadata["format"] = "pt";
    metadata["writer"] = "koifish";
}

size_t K_SafeTensors::Register(hGensor t, size_t offset, FILE_FORMAT_TYPE format, int flag) {
    size_t sz = t->nByte_CKP(ckp);  // may expand
    assert(t->hFish != nullptr);
    assert(strlen(t->name) > 0);
    if (G_Has_(t->name, {"model.layers.1.self_attn.q_proj"})) {
        DEBUG_HERE;
    }
    assert(sz > 0);
    if (format == CKP_HF) {  //  "Many tools(lm_eval...) failed to load these packed-I32 format quant model!"
        if (t->type == typNUMBER::Q4) {
            sz *= 4;
        } else if (t->type == typNUMBER::T_BINARY) {
            sz *= 16;
        } else if (t->type == typNUMBER::T_SIGN || t->type == typNUMBER::Q2) {
            sz *= 8;
        }
    }
    t->data_offsets[0] = offset;
    t->data_offsets[1] = offset + sz;
    tensors.insert(t->name, t);
    // _INFO("\tRegister %ld@%s\n", sz, t->name);
    return offset + sz;
}

/*
    [todo] GDS/NVMe direct I/O > MMAP(like https://github.com/xaskasdf/ntransformer)
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
    K_SafeTensors st(this, ckp, path), *hst = (K_SafeTensors*)(ckp.hAllST);
    switch (ckp.state_type) {
        case CheckPoint_Params::STATE:
            assert(hst != nullptr);  // hst=&all_states, so mmp would keep open
            if (hst->hFish == nullptr)
                hst->hFish = this;
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
            _INFO("<<<<<< ST_SERIALIZE save @\"%s\" nInit=%ld ......", path.c_str(), nInit);
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
                if (G_Has_(t->name, {"model.layers.27.mlp.down_proj.weight"})) {  // model.embed_tokens.weight  model.layers.0.self_attn.q_proj
                    DEBUG_HERE;
                    // t->Print("wte@save", 4, -1);
                }
                if (t->isRefer())  //  "model.out.weight"
                    continue;
                jsConfig["tensors"][t->name] = dst_offset;  // tensor->Dump(100,"");
                dst_offset                   = hst->Register(t, dst_offset, hst->ckp.format);
            }
            if (ckp.format == CKP_KOIFISH) {
                hst->insertJS(jsConfig, dst_offset);
            }
            HST2JSON("_safetensors_.json", hst, flag);  //
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
            szOFS = std::filesystem::file_size(path);  //  1499827103
            if (!jPath.empty()) {                      //  save json file with more info
                std::ofstream o(jPath);
                o << std::setw(4) << jsConfig << std::endl;
            }
            _INFO("\r<<<<<< ST_SERIALIZE save @\"%s\"(\"%s\") nInit=%ld sz=%.6gM flag=%d T=%.4gs\n", path.c_str(), jPath.empty() ? "" : "+json", nInit,
                  szOFS / 1.0e6, flag, (GST_ms() - t0) / 1000.0);
            return true;
        } else {
            int nSerialT = SAFETENSOR2Gensors(path, hst, 0x0);
            _LOG(nSerialT == 0 ? DUMP_ERROR : DUMP_INFO, ">>>>>> ST_SERIALIZE load@\"%s\" nSerialT=%d iter=%d\n", path.c_str(), nSerialT, flag);
            return nSerialT > 0;
        }
    } catch (JSON::parse_error& e) {
        _ERROR("\r\n%s  Failed to serialize @\"%s\"!!! ERR=%s", __func__, jPath.c_str(), e.what());
        return false;
    } catch (...) {
        _ERROR("\r\n%s  Failed to serialize @\"%s\"!!! ", __func__, path.c_str());
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
            hGTensor tensor = tensors.at(i);
            jHeader[key]    = tensor->jDesc(this);
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
 *  ckp maybe nullptr to support args of HF/Safetensors
 */
bool Fish::LoadFolderOfST(int stType, int flag) {
    std::string sFolder;
    bool isFromCKP = !config.ckp_in.empty();  // Load checkpoint
    CheckPoint_Params ckp;                    //  may be {}
    if (!isFromCKP) {                         // otherwise, load HF/Safetensors
        assert(!config.model.sSTPath.empty());
        sFolder = config.model.sSTPath;
    } else {
        ckp     = config.ckp_in[0];
        sFolder = ckp.sModelPath;  // FullPath(false);
        // assert(config.model.tpInitWeight == INIT_WEIGHT::SERIALIZE);
    }

    std::vector<std::string> paths = FilesOfDir(sFolder, {"safetensors", "kun"}, 0x0);
    if (paths.empty()){
        _WARN("\r[LoadFolderOfST] failed!  please check {\"safetensors\", \"kun\"} files @\"%s\"!\n", sFolder.c_str());
        return false;
    }
        

    isLoadCheckpoint = true;  //  HF/Safetensors is also checkpoint
    int nSerialT     = 0, curSerialT;
    std::vector<K_SafeTensors*> st_mmfs;
    for (auto path : paths) {
        K_SafeTensors* hst = new K_SafeTensors(this, ckp, sFolder);
        st_mmfs.push_back(hst);
        curSerialT = SAFETENSOR2Gensors(path, hst, 0x0);
        if (curSerialT <= 0) {
            isLoadCheckpoint = false;
            break;
        }
        nSerialT += curSerialT;
    }
    if (nSerialT == 0) {           //  !=optParams.size()
        isLoadCheckpoint = false;  // bias maybe null
    }
    if (isLoadCheckpoint) {
        _LOG(nSerialT == 0 ? DUMP_ERROR : DUMP_INFO, ">>>>>> SafeTensors_Serialize load@\"%s\" OK. nSerialT=%d iter=%d\n\n", config.model.sCardPath.c_str(),
             nSerialT, flag);

        if (isTrain()) {  // otherwise, st would release mmap memory!
            SaveTrain(config.state, true, FSerial::COPY_MMAP);
        }
        for (auto hst : st_mmfs)  // release all mmf resource
            delete hst;

        if (!config.model.st_index_map.empty()) {  //  "model.safetensors.index.json"     "model.embed_tokens.weight"
                                                   /*for (auto kv : config.model.st_index_map) {
                                                      hGensor target = GetGensor(kv.first);
                                                      if(target->data == nullptr){
                                                          assert(0);
                                                      }
                                                      kv.second = target->data;
                                                  }*/
            assert(nSerialT == config.model.st_index_map.size());
        }

        // if(!config.quant.filter_MIQ.empty())
        //     throw SafeExit("", KOIFISH_EXIT_DEBUG, SafeExit::ExitReason::SYSTEM_FAILURE, __func__);
        for (auto t : optParams) {
            assert(!BIT_TEST(t->flags, GTensor::F_MMAP));
            if (!isTrain()) {
                assert(t->host_data == nullptr);
            }
            // assert(t->szUse>0 && t->data!=nullptr);  //bias maybe null
        }
    }

    // std::string fpCheck = sSTPath;  //ckp != nullptr ? ckp->sModelPath : "";
    if (!isLoadCheckpoint) {
        _WARN("\r[LoadFolderOfST] failed!  please check {\"safetensors\", \"kun\"}  files @\"%s\"!\n", sFolder.c_str());
        return false;
    }
    // if (ckp != nullptr)
    //     UpdateCheckPoint(*ckp, false);
    config.fuyou.filter_reload = {};  // ckp.fuyou_filter_reload;        // Don't reload

    // assert(vendor == "gruai");
    _INFO("\r[LoadFolderOfST] OK @\"%s\"\n", sFolder.c_str());
    return true;
}


/**
The exact behavior can vary based on model version, configuration, and prompt design
1. enable_thinking
    If ask "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n": The model generates a response starting with <think>(internal thinking) because that's how it was trained for reasoning tasks
    If ask "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n": The model sees that the "thinking" part (<think>followed by \n\n</think>\n\n) is already provided​ in the prompt
The model interprets this as: the assistant has already done its thinking, and now it's at the point of producing the final answer
Since the thinking is complete (marked by </think>), the model just continues with the actual response.

2. For raw prompt(like "hello") alone without <|im_start|> / <|im_end|> / `` template markers
Qwen3’s tokenizer + generation pipeline doesn’t detect a valid chat turn boundary
The model falls into:auto-completing random internal reasoning tokens spitting out garbled thought fragments, repeated words, nonsense rambling
it’s trying to "continue raw text" instead of "respond to user chat"
 */
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