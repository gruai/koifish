/**
 *  Copyright 2023-2024 by Grusoft
 *
 *  \brief KV cache
 *  \author Yingshi Chen
 */

/*
    https://github.com/ggerganov/llama.cpp/discussions/7887
    https://github.com/ggerganov/llama.cpp/discussions/7625
*/
#include "Fish.hpp"
class KVCache {
public:
    KVCache(int max_batch_size, int max_seq_len, int n_kv_heads, int head_dim);

    void update(int batch_size, int start_pos, hGensor xk, hGensor xv);
    hGensor get(int batch_size, int start_pos, int seq_len);

private:
    hGensor cache_k;
    hGensor cache_v;
};