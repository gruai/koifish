/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief
 *  \author Yingshi Chen
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "../g_float.hpp"
//

/**
 * 1. phmap::flat_hash_map is unordered! designed as a drop-in replacement for std::unordered_map, and it does not maintain the insertion or iteration order of
 * its elements
 * 2. Order-map is phmap::btree_map
 */
#include "../Utils/parallel_hashmap/phmap.h"

#define MAP_ phmap::flat_hash_map
//   generally much faster than std::unordered_set, but it is not safer
#define UNORDER_SET_ phmap::flat_hash_set   

using MAP_VOCAB = phmap::flat_hash_map<std::string, int>;
// typedef std::unordered_map<std::string, int> MAP_VOCAB;

// map one token to another token(dialect)
using MAP_T2T = phmap::flat_hash_map<TOKEN_ID, TOKEN_ID>;
using MAP_I2I = phmap::flat_hash_map<int, int>;
using MAP_S2S = phmap::flat_hash_map<std::string, std::string>;

using MAP_S2CLASS = phmap::flat_hash_map<std::string, std::type_index>;

template <typename K, typename V>
V* HT_FIND(phmap::flat_hash_map<K, V*>& HT, K key) {
    if (HT.find(key) == HT.end())
        return nullptr;
    else
        return HT[key];
}

// Maitain the insertion order!
template <class K, class V>
class vector_map {
   public:
    using key_type       = K;
    using mapped_type    = V;
    using value_type     = std::pair<K, V>;  // not pair<const K, V>, so the vector stays erasable
    using size_type      = std::size_t;
    using iterator       = typename std::vector<value_type>::iterator;
    using const_iterator = typename std::vector<value_type>::const_iterator;

    vector_map() = default;
    vector_map(std::initializer_list<value_type> init) { insert(init.begin(), init.end()); }
    template <class It>
    vector_map(It first, It last) {
        insert(first, last);
    }

    // ---- iteration (insertion order) ----
    iterator begin() { return items_.begin(); }
    iterator end() { return items_.end(); }
    const_iterator begin() const { return items_.begin(); }
    const_iterator end() const { return items_.end(); }
    const_iterator cbegin() const { return items_.cbegin(); }
    const_iterator cend() const { return items_.cend(); }

    // ---- capacity ----
    bool empty() const { return items_.empty(); }
    size_type size() const { return items_.size(); }
    void reserve(size_type n) {
        items_.reserve(n);
        index_.reserve(n);
    }

    // ---- lookup ----
    iterator find(const K& k) {
        auto it = index_.find(k);
        return it == index_.end() ? items_.end() : items_.begin() + it->second;
    }
    const_iterator find(const K& k) const {
        auto it = index_.find(k);
        return it == index_.end() ? items_.end() : items_.begin() + it->second;
    }
    size_type count(const K& k) const { return index_.count(k); }

    V& at(const K& k) {
        auto it = find(k);
        if (it == end())
            throw std::out_of_range("vector_map::at: key not found");
        return it->second;
    }
    const V& at(const K& k) const {
        auto it = find(k);
        if (it == end())
            throw std::out_of_range("vector_map::at: key not found");
        return it->second;
    }

    // ---- modifiers ----
    template <class... Args>
    std::pair<iterator, bool> try_emplace(const K& k, Args&&... args) {
        auto it = index_.find(k);
        if (it != index_.end())
            return {items_.begin() + it->second, false};
        items_.emplace_back(std::piecewise_construct, std::forward_as_tuple(k), std::forward_as_tuple(std::forward<Args>(args)...));
        index_.emplace(k, items_.size() - 1);
        return {items_.end() - 1, true};
    }

    V& operator[](const K& k) { return try_emplace(k).first->second; }

    // Like std::map: keeps the existing entry if the key is already present.
    template <class... Args>
    std::pair<iterator, bool> emplace(Args&&... args) {
        value_type v(std::forward<Args>(args)...);
        auto it = index_.find(v.first);
        if (it != index_.end())
            return {items_.begin() + it->second, false};
        items_.push_back(std::move(v));
        index_.emplace(items_.back().first, items_.size() - 1);
        return {items_.end() - 1, true};
    }
    template <class P>
    std::pair<iterator, bool> insert(P&& p) {
        return emplace(std::forward<P>(p));
    }
    template <class It>
    void insert(It first, It last) {
        for (; first != last; ++first) emplace(first->first, first->second);
    }

    iterator erase(const_iterator pos) {
        size_type i = static_cast<size_type>(pos - items_.cbegin());
        index_.erase(items_[i].first);
        auto r = items_.erase(items_.begin() + i);
        for (size_type j = i; j < items_.size(); ++j) index_[items_[j].first] = j;
        return r;
    }
    size_type erase(const K& k) {
        auto it = find(k);
        if (it == end())
            return 0;
        erase(const_iterator(it));
        return 1;
    }
    void clear() {
        items_.clear();
        index_.clear();
    }

    // ---- comparison ----
    // Equality ignores order (JSON semantics: {"a":1,"b":2} == {"b":2,"a":1}).
    friend bool operator==(const vector_map& a, const vector_map& b) {
        if (a.size() != b.size())
            return false;
        for (const auto& e : a.items_) {
            auto it = b.find(e.first);
            if (it == b.end() || !(it->second == e.second))
                return false;
        }
        return true;
    }
    friend bool operator!=(const vector_map& a, const vector_map& b) { return !(a == b); }

    // Ordering is done on key-sorted views, so it matches the old std::map behaviour
    // and stays consistent with operator==.
    friend bool operator<(const vector_map& a, const vector_map& b) {
        auto sorted = [](const vector_map& m) {
            std::vector<const value_type*> v;
            v.reserve(m.size());
            for (const auto& e : m.items_) v.push_back(&e);
            std::sort(v.begin(), v.end(), [](const value_type* x, const value_type* y) { return x->first < y->first; });
            return v;
        };
        auto sa = sorted(a), sb = sorted(b);
        return std::lexicographical_compare(sa.begin(), sa.end(), sb.begin(), sb.end(), [](const value_type* x, const value_type* y) {
            if (x->first < y->first)
                return true;
            if (y->first < x->first)
                return false;
            return x->second < y->second;
        });
    }
    friend bool operator>(const vector_map& a, const vector_map& b) { return b < a; }
    friend bool operator<=(const vector_map& a, const vector_map& b) { return !(b < a); }
    friend bool operator>=(const vector_map& a, const vector_map& b) { return !(a < b); }

   private:
    std::vector<value_type> items_;  // insertion order
    phmap::flat_hash_map<K, size_type> index_;
    // std::unordered_map<K, size_type> index_; // key -> position, O(1) lookup
};