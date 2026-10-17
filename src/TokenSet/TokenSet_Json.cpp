/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#include "../Manifold/Optimizer.hpp"
#include "../Shoal/gLLM.hpp"
#include "../Tensor/Safetensors.hpp"
#include "../Utils/GST_obj.hpp"
#include "../Utils/json/simdjson.h"
#include "TokenSet.hpp"
using namespace simdjson;
#include "DataLoader.hpp"
#include "Dictionary.hpp"

// #define _USE_ARROW_
#ifdef _USE_ARROW_
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#endif


/**
 * 1) If val is obect(not string), just return val.raw_json()! So some 
 * "state": {
        "ticket": {
            "channel": "email",
            "body": "..."
        }
    }
        would contain some trash, which would lower down the accuracy
 *  */ 
static std::string value_to_raw_or_string(ondemand::value val, bool& was_object) {
    was_object            = false;
    ondemand::json_type t = val.type();
    if (t == ondemand::json_type::string) {
        std::string_view sv = val.get_string();
        return std::string(sv);
    }
    if (t == ondemand::json_type::object)
        was_object = true;
    // raw_json() returns the exact source bytes for this value — works for object/array/number/bool alike, no extra parsing cost
    std::string_view raw = val.raw_json();
    return std::string(raw);
}

bool Tokenset_JEV::GetShardInfo(int id, int flag) {
    shard_samps.clear();
    std::filesystem::path path = shard_paths[id];
    if (!std::filesystem::exists(path)) {
        K_EXIT(KOIFISH_LOAD_SHARD_NULL);
    }
    _INFO(">>>>>> SIMDJSON_mmap shard@ %s \"%s\" %s f=%d......\n", COLOR_ORANGE, path.c_str(), COLOR_RESET, flag);
    double start = GST_ms();
    // std::vector<Record> records;
    int nErrDoc = 0;
    padded_memory_map mapped(path.c_str());
    ondemand::parser parser;
    constexpr size_t batch_size = 1024 * 1024;  // 1 MB batches
    ondemand::document_stream stream;
    auto error = parser.iterate_many(mapped.view(), batch_size).get(stream);
    //  parser.iterate(mapped.view().substr(offset))
    if (error) {
        return false;
    }
    for (auto doc : stream) {
        try {
            if (doc.error()) {
                nErrDoc++;
                continue;  // Skip malformed document
            }
            string state_raw, qid, value;
            MAP_S2S q_kv;
            bool is_obj               = false;
            ondemand::value state_val = doc["state"];
            state_raw                 = value_to_raw_or_string(state_val, is_obj);
            if(is_obj){
                _WARN("\nstate=%s", state_raw.c_str());
            }
            // ---- "questions" : object with dynamic keys ----
            ondemand::object questions_obj = doc["questions"];
            for (auto field : questions_obj) {
                std::string_view key     = field.unescaped_key();
                qid                      = std::string(key);
                ondemand::object question = field.value();
                for (auto kv : question) {
                    std::string_view key     = kv.unescaped_key();
                    value = value_to_raw_or_string(kv.value(), is_obj);
                    q_kv.emplace(std::string(key), value);
                }
                //  each question gen One sample
                
                std::shared_ptr<SAMP_JEV> rec = std::make_shared<SAMP_JEV>(state_raw, qid, q_kv);
                if(rec->tpDecisoin!=SAMP_JEV::NOUL)
                    continue;
                shard_samps.push_back(rec);  // one rec may generate multiple samps
                // assert(!is_obj);
            }

            /*// ---- "_meta" : fairly fixed schema ----
                ondemand::object meta = doc["_meta"].get_object();
                for (auto field : meta) {
                    std::string_view key = field.unescaped_key();
                    if (key == "row") {
                        rec->meta_row = field.value().get_int64();
                    } else if (key == "source") {
                        rec->meta_source = std::string(field.value().get_string().value());
                    } else if (key == "id") {
                        rec->meta_id = std::string(field.value().get_string().value());
                    } else if (key == "split") {
                        // may be null in some rows
                        auto v = field.value();
                        if (v.is_null()) {
                            rec->meta_split.clear();
                        } else {
                            rec->meta_split = std::string(v.get_string().value());
                        }
                    }
                    // ignore other meta fields (text_sha256, row_sha256, repo, revision, ...)
                }
            */   

        } catch (const simdjson::simdjson_error& e) {
            _ERROR("simdjson error @%s\n", e.what());
            nErrDoc++;
            continue;  // skip this document, keep going
        } catch (const std::exception& e) {
            _ERROR("simdjson error @%s\n", e.what());
            nErrDoc++;
            continue;
        }
    }
    double tJson  = (GST_ms() - start) / 1000.0;
    nShardSamples = shard_samps.size();
    _INFO("\n>>>>>> SIMDJSON_mmap shard@ %s \"%s\" %s OK! nSamp=%d(tps=%.3gK) tJson=%.3g\n", COLOR_ORANGE, path.c_str(), COLOR_RESET, nShardSamples,
          nShardSamples / tJson / 1000.0, tJson);
    return nShardSamples > 0;
}
