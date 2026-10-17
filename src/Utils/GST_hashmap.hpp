/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <string>

#include "../g_float.hpp"
//

/**
 * 1. phmap::flat_hash_map is unordered! designed as a drop-in replacement for std::unordered_map, and it does not maintain the insertion or iteration order of
 * its elements
 * 2. Order-map is phmap::btree_map
 */
#include "../Utils/parallel_hashmap/phmap.h"

#define MAP_ phmap::flat_hash_map

using MAP_VOCAB = phmap::flat_hash_map<std::string, int>;
// typedef std::unordered_map<std::string, int> MAP_VOCAB;

// map one token to another token(dialect)
using MAP_T2T = phmap::flat_hash_map<TOKEN_ID, TOKEN_ID>;
using MAP_I2I = phmap::flat_hash_map<int, int>;
using MAP_S2S = phmap::flat_hash_map<std::string, std::string>;

template <typename K, typename V>
V* HT_FIND(phmap::flat_hash_map<K, V*>& HT, K key) {
    if (HT.find(key) == HT.end())
        return nullptr;
    else
        return HT[key];
}