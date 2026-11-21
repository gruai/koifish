/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief GRUSOFT TEMPLATE	- Memory/Buffer
 *  \author Yingshi Chen
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

#include "../g_def_x.hpp"
#include "../g_float.hpp"
#include "GST_obj.hpp"

using namespace std;

class GTensor;
class Fish;

template <class T>
class intrusive_ptr;
template <class T>
class weak_intrusive_ptr;

// Base class: the reference counts live INSIDE the object.
//   refcount_  = number of intrusive_ptr (strong owners)
//   weakcount_ = number of weak_intrusive_ptr, +1 while refcount_ > 0
//                (the strong owners collectively hold one weak reference)
// When refcount_ hits 0   -> release_resources() runs (free the heavy payload early).
// When weakcount_ hits 0  -> the object itself is deleted.
class intrusive_ptr_target {
   public:
    intrusive_ptr_target() noexcept = default;
    // Copying/moving an object must never copy its reference counts.
    intrusive_ptr_target(const intrusive_ptr_target&) noexcept {}
    intrusive_ptr_target& operator=(const intrusive_ptr_target&) noexcept { return *this; }
    virtual ~intrusive_ptr_target() {
        assert(refcount_.load() == 0 && "object destroyed while strong refs exist");
        assert(weakcount_.load() == 0 && "object destroyed while weak refs exist");
    }

   protected:
    // Called once, when the last strong reference goes away. Must not throw.
    virtual void release_resources() {}

   private:
    template <class T>
    friend class intrusive_ptr;
    template <class T>
    friend class weak_intrusive_ptr;

    static void release_weak(intrusive_ptr_target* b) noexcept {
        if (b->weakcount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            delete b;
    }
    static void release_strong(intrusive_ptr_target* b) noexcept {
        if (b->refcount_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            b->release_resources();
            release_weak(b);  // drop the "strong owners" weak reference
        }
    }

    mutable std::atomic<size_t> refcount_{0};
    mutable std::atomic<size_t> weakcount_{0};
};

template <class T>
class intrusive_ptr final {
    static_assert(std::is_base_of_v<intrusive_ptr_target, T>, "T must derive from intrusive_ptr_target");
    template <class U>
    friend class intrusive_ptr;
    template <class U>
    friend class weak_intrusive_ptr;

    T* p_ = nullptr;

    static intrusive_ptr_target* base(T* p) noexcept { return static_cast<intrusive_ptr_target*>(p); }

   public:
    using element_type = T;

    intrusive_ptr() noexcept = default;
    intrusive_ptr(std::nullptr_t) noexcept {}

    intrusive_ptr(const intrusive_ptr& o) noexcept : p_(o.p_) {
        if (p_)
            base(p_)->refcount_.fetch_add(1, std::memory_order_relaxed);
    }
    intrusive_ptr(intrusive_ptr&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }

    template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    intrusive_ptr(const intrusive_ptr<U>& o) noexcept : p_(o.p_) {
        if (p_)
            base(p_)->refcount_.fetch_add(1, std::memory_order_relaxed);
    }
    template <class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    intrusive_ptr(intrusive_ptr<U>&& o) noexcept : p_(o.p_) {
        o.p_ = nullptr;
    }

    ~intrusive_ptr() {
        if (p_)
            intrusive_ptr_target::release_strong(base(p_));
    }

    intrusive_ptr& operator=(intrusive_ptr rhs) noexcept {
        swap(rhs);
        return *this;
    }  // copy+move

    // ---- observers -------------------------------------------------------
    T* get() const noexcept { return p_; }
    T& operator*() const noexcept { return *p_; }
    T* operator->() const noexcept { return p_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }
    size_t use_count() const noexcept { return p_ ? base(p_)->refcount_.load(std::memory_order_relaxed) : 0; }
    // number of weak_intrusive_ptr observers (approximate under concurrent mutation)
    size_t weak_use_count() const noexcept {
        if (!p_)
            return 0;
        size_t w = base(p_)->weakcount_.load(std::memory_order_relaxed);
        return w ? w - 1 : 0;
    }
    bool unique() const noexcept { return use_count() == 1; }

    // ---- modifiers -------------------------------------------------------
    void swap(intrusive_ptr& o) noexcept { std::swap(p_, o.p_); }
    void reset() noexcept {
        intrusive_ptr tmp;
        swap(tmp);
    }  // tmp's dtor drops the reference

    // Give up ownership WITHOUT decrementing: the caller now owns one strong ref
    // (e.g. stored in a C API / PyObject slot) and must call reclaim() later.
    T* release() noexcept {
        T* r = p_;
        p_   = nullptr;
        return r;
    }

    // ---- factories -------------------------------------------------------
    // Allocate + construct; the only way to start a fresh object's life.
    template <class... Args>
    static intrusive_ptr make(Args&&... args) {
        T* raw = new T(std::forward<Args>(args)...);
        base(raw)->refcount_.store(1, std::memory_order_relaxed);
        base(raw)->weakcount_.store(1, std::memory_order_relaxed);  // the "+1"
        intrusive_ptr r;
        r.p_ = raw;
        return r;
    }
    // Adopt a pointer that already carries one owned strong ref (inverse of release()).
    static intrusive_ptr reclaim(T* p) noexcept {
        intrusive_ptr r;
        r.p_ = p;
        return r;
    }
    // Create an owning ptr from a NON-owning raw pointer (e.g. `this`): bumps the count.
    // Replacement for enable_shared_from_this. Object must currently have refcount > 0.
    static intrusive_ptr reclaim_copy(T* p) noexcept {
        if (p) {
            assert(base(p)->refcount_.load() > 0 && "reclaim_copy on an object nobody owns");
            base(p)->refcount_.fetch_add(1, std::memory_order_relaxed);
        }
        return reclaim(p);
    }
};

template <class T, class... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
    return intrusive_ptr<T>::make(std::forward<Args>(args)...);
}

template <class T, class U>
bool operator==(const intrusive_ptr<T>& a, const intrusive_ptr<U>& b) noexcept {
    return a.get() == b.get();
}
template <class T, class U>
bool operator!=(const intrusive_ptr<T>& a, const intrusive_ptr<U>& b) noexcept {
    return a.get() != b.get();
}
template <class T>
bool operator==(const intrusive_ptr<T>& a, std::nullptr_t) noexcept {
    return !a;
}
template <class T>
bool operator!=(const intrusive_ptr<T>& a, std::nullptr_t) noexcept {
    return static_cast<bool>(a);
}
template <class T>
bool operator<(const intrusive_ptr<T>& a, const intrusive_ptr<T>& b) noexcept {
    return std::less<T*>()(a.get(), b.get());
}

// Non-owning observer: keeps the *object* (control block) alive, not its payload.
template <class T>
class weak_intrusive_ptr final {
    static_assert(std::is_base_of_v<intrusive_ptr_target, T>);
    T* p_ = nullptr;
    static intrusive_ptr_target* base(T* p) noexcept { return static_cast<intrusive_ptr_target*>(p); }
    void retain_() noexcept {
        if (p_)
            base(p_)->weakcount_.fetch_add(1, std::memory_order_relaxed);
    }

   public:
    weak_intrusive_ptr() noexcept = default;
    weak_intrusive_ptr(const intrusive_ptr<T>& s) noexcept : p_(s.get()) { retain_(); }
    weak_intrusive_ptr(const weak_intrusive_ptr& o) noexcept : p_(o.p_) { retain_(); }
    weak_intrusive_ptr(weak_intrusive_ptr&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ~weak_intrusive_ptr() {
        if (p_)
            intrusive_ptr_target::release_weak(base(p_));
    }
    weak_intrusive_ptr& operator=(weak_intrusive_ptr rhs) noexcept {
        std::swap(p_, rhs.p_);
        return *this;
    }

    // Atomically "upgrade" to a strong pointer; null if the object already died.
    intrusive_ptr<T> lock() const noexcept {
        if (!p_)
            return {};
        size_t n = base(p_)->refcount_.load(std::memory_order_relaxed);
        do {
            if (n == 0)
                return {};
        } while (!base(p_)->refcount_.compare_exchange_weak(n, n + 1, std::memory_order_acquire, std::memory_order_relaxed));
        return intrusive_ptr<T>::reclaim(p_);  // adopt the ref we just added
    }
    bool expired() const noexcept { return !p_ || base(p_)->refcount_.load(std::memory_order_relaxed) == 0; }
    size_t use_count() const noexcept { return p_ ? base(p_)->refcount_.load(std::memory_order_relaxed) : 0; }
    void reset() noexcept {
        weak_intrusive_ptr tmp;
        std::swap(p_, tmp.p_);
    }
};


namespace std {
template <class T>
struct hash<intrusive_ptr<T>> {
    size_t operator()(const intrusive_ptr<T>& p) const noexcept { return std::hash<T*>()(p.get()); }
};
}  // namespace std

struct cuDeviceFree {
    void operator()(void* p) const;
};
//
using cuPTR = std::unique_ptr<void, cuDeviceFree>;

struct hostDeviceFree {
    void operator()(void* p) const;
};
using hostPTR = std::unique_ptr<void, hostDeviceFree>;

class GST_TensorBuffer {
   protected:
    enum TYPE { ACTIVATION, WEIGHT, MOMENT, GRAD, TEMP, OTHER, COUNT };
    // size_t szA = 0, szW = 0, szG = 0, szMoment = 0, szTemp = 0, szOther = 0;
    size_t szSum[COUNT] = {};

    struct Allocation {
        TYPE type;
        void* hData = nullptr;
        size_t sz;
        string desc;
        SourceLocation where = SourceLocation::current();
        Allocation() {}
        Allocation(size_t sz_, string d_, SourceLocation where_, void* hData = nullptr, int flag = 0x0);
    };

    int nMostMemItem       = 1024;
    Fish* hFish            = nullptr;
    void* buffer_          = nullptr;
    size_t total_size_     = 0x0;
    size_t current_offset_ = 0x0;
    GST_Dict<Allocation> records;
    // phmap::btree_map<const void*, Allocation> records_map; // replace with order-map?
    std::vector<GTensor*> zTensors;

   public:
    //   huTensor bt4c,delta, ...
    shared_ptr<GTensor> bt4c = nullptr, delta = nullptr, gate_delta = nullptr, tmpDelta = nullptr, scratch = nullptr, tmpFF1 = nullptr, tmpW = nullptr,
                        tmpGW = nullptr, tmpQout = nullptr, tmpKout = nullptr, residual = nullptr, tmpTernary = nullptr, outL = nullptr, Qlen = nullptr,
                        KVlen = nullptr;

    GST_TensorBuffer(Fish* hFish, int flag = 0x0);
    virtual ~GST_TensorBuffer();

    // Prevent copying
    GST_TensorBuffer(const GST_TensorBuffer&)            = delete;
    GST_TensorBuffer& operator=(const GST_TensorBuffer&) = delete;

    bool Prepare(int flag = 0x0);

    shared_ptr<GTensor> RegisterTensor(Fish* hFish, const string& name_, SHAPE shape, typNUMBER tpD_, SourceLocation& where_, int flag = 0x0);
    void* Register(size_t sz_, string d_, void* hData_, SourceLocation& where_, int flag = 0x0);
    bool Free(void* hObj, int flag = 0x0);

    virtual bool Clear(int flag = 0x0);

    void Dump(int type, int flag = 0x0) const;

    size_t get_used_memory() const { return current_offset_; }
    size_t get_total_memory() const { return total_size_; }
    size_t get_available_memory() const { return total_size_ - current_offset_; }

   private:
    std::mutex _m;
};
typedef shared_ptr<GST_TensorBuffer> hTensorBuffer;

void GlobalMemoryInfo(int type, int flag);

extern hTensorBuffer gBUFF;