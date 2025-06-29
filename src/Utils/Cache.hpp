/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief KV cache
 *  \author Yingshi Chen
 */
#pragma once
#include <memory>

#include "../g_float.hpp"

class GTensor;
typedef std::shared_ptr<GTensor> hGTensor;
/*
    https://github.com/ggerganov/llama.cpp/discussions/7887
    https://github.com/ggerganov/llama.cpp/discussions/7625
*/
class Fish;
class NLP_AutoRegressive;

class KVCache {
    void *lamakv = nullptr;
    int kv_n     = -1;
    void init_lamakv(int n_batch);

   protected:
    Fish *_fish = nullptr;

    typNUMBER tpCache = typNUMBER::BF16;
    hGTensor key      = nullptr;  // (layer, seq_len, dim)
    hGTensor value    = nullptr;  // (layer, seq_len, dim)
                                  // void *raw_key=nullptr,*raw_val=nullptr;
   public:
    enum CTYPE { KV_KEY = 1, KV_VAL };
    KVCache(Fish *, int max_batch_size = 0, int max_seq_len = 0, int flag = 0x0);

    void update(int batch_size, int start_pos, hGTensor xk, hGTensor xv);
    hGTensor get(int batch_size, int start_pos, int seq_len);

    int n_kv();
    virtual hGTensor SerialV(void *ctx, hGTensor vCur, int il, bool isSave);
    virtual hGTensor SerialK(void *ctx, hGTensor vCur, int il, bool isSave);

    void *Get(KVCache::CTYPE typ, int flag = 0x0);
};
typedef std::shared_ptr<KVCache> hKVCache;