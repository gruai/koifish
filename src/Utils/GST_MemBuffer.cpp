/**
 *  SPDX-FileCopyrightText: 2019-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  v0.1        need many refactor
 *
 *  \brief GRUSOFT TEMPLATE	- Memory/Buffer using the RAII (Resource Acquisition Is Initialization) principle
 *  \author Yingshi Chen
 */

#include "GST_MemBuffer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

#include "../Shoal/Fish.hpp"
#include "../Tensor/GTensor.hpp"
#include "GST_log.hpp"
#include "GST_obj.hpp"

// intrusive_ptr maybe useful 
enum class Device { CPU, CUDA };

// ---------------- raw memory: owning pointer with a function-pointer deleter ----------------
using DataPtr = std::unique_ptr<void, void (*)(void*)>;
static void host_free(void* p) { std::free(p); }
#ifdef USE_CUDA
static void dev_free(void* p) { cudaFree(p); }
#endif
static DataPtr allocate(Device d, size_t nbytes) {
    if (d == Device::CPU) {
        void* p = std::malloc(nbytes ? nbytes : 1);
        if (!p) throw std::bad_alloc();
        return DataPtr(p, &host_free);
    }
#ifdef USE_CUDA
    void* p = nullptr;
    if (cudaMalloc(&p, nbytes) != cudaSuccess) throw std::bad_alloc();
    return DataPtr(p, &dev_free);
#else
    throw std::runtime_error("built without CUDA");
#endif
}

// ---------------- StorageImpl: owns the bytes ----------------
class StorageImpl final : public intrusive_ptr_target {
public:
    static inline std::atomic<long> live_storages{0};   // debugging counters
    static inline std::atomic<long> live_bytes{0};

    StorageImpl(size_t nbytes, Device dev) : data_(allocate(dev, nbytes)), nbytes_(nbytes), device_(dev) {
        ++live_storages;
        live_bytes += static_cast<long>(nbytes);
    }
    ~StorageImpl() override { free_(); --live_storages; }

    void* data() const { return data_.get(); }
    size_t nbytes() const { return nbytes_; }
    Device device() const { return device_; }

protected:
    // Last *strong* owner is gone: free the memory now, even if a weak_intrusive_ptr
    // still keeps this small control object alive.
    void release_resources() override { free_(); }

private:
    void free_() {
        if (data_) { live_bytes -= static_cast<long>(nbytes_); data_.reset(); }
    }
    DataPtr data_;
    size_t nbytes_;
    Device device_;
};
using Storage = intrusive_ptr<StorageImpl>;

// ---------------- TensorImpl: metadata + a refcounted reference to Storage ----------------
class TensorImpl final : public intrusive_ptr_target {
public:
    TensorImpl(Storage s, std::vector<int64_t> sizes, std::vector<int64_t> strides, int64_t offset)
        : storage_(std::move(s)), sizes_(std::move(sizes)), strides_(std::move(strides)), offset_(offset) {}

    Storage storage_;                 // strong ref: this is what keeps the bytes alive
    std::vector<int64_t> sizes_, strides_;
    int64_t offset_;                  // in elements

protected:
    void release_resources() override { storage_.reset(); }   // like c10::TensorImpl
};

// ---------------- Tensor: a cheap, copyable handle (float32 only, to stay small) ----------------
class Tensor {
    intrusive_ptr<TensorImpl> impl_;
    explicit Tensor(intrusive_ptr<TensorImpl> i) : impl_(std::move(i)) {}

    static std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sz) {
        std::vector<int64_t> st(sz.size());
        int64_t acc = 1;
        for (int64_t i = static_cast<int64_t>(sz.size()) - 1; i >= 0; --i) { st[i] = acc; acc *= sz[i]; }
        return st;
    }
    Tensor make_view(std::vector<int64_t> sz, std::vector<int64_t> st, int64_t off) const {
        // NEW TensorImpl, SAME Storage (storage refcount +1, no data copied)
        return Tensor(make_intrusive<TensorImpl>(impl_->storage_, std::move(sz), std::move(st), off));
    }

public:
    Tensor() = default;                                       // copy/move = refcount bump only

    static Tensor empty(std::vector<int64_t> sizes, Device dev = Device::CPU) {
        int64_t n = 1;
        for (auto s : sizes) n *= s;
        auto st = contiguous_strides(sizes);
        auto storage = make_intrusive<StorageImpl>(static_cast<size_t>(n) * sizeof(float), dev);
        return Tensor(make_intrusive<TensorImpl>(std::move(storage), std::move(sizes), std::move(st), 0));
    }

    const std::vector<int64_t>& sizes() const { return impl_->sizes_; }
    const std::vector<int64_t>& strides() const { return impl_->strides_; }
    int64_t numel() const {
        return std::accumulate(sizes().begin(), sizes().end(), int64_t{1}, std::multiplies<int64_t>());
    }
    bool defined() const { return static_cast<bool>(impl_); }

    float* data_ptr() const {                                  // device pointer if Device::CUDA!
        return static_cast<float*>(impl_->storage_->data()) + impl_->offset_;
    }
    float* host_ptr() const {
        if (impl_->storage_->device() != Device::CPU) throw std::runtime_error("not a host tensor");
        return data_ptr();
    }

    // ---- views: share the Storage ----
    Tensor select(int64_t dim, int64_t index) const {
        auto sz = sizes(); auto st = strides();
        int64_t off = impl_->offset_ + index * st[dim];
        sz.erase(sz.begin() + dim); st.erase(st.begin() + dim);
        return make_view(std::move(sz), std::move(st), off);
    }
    Tensor narrow(int64_t dim, int64_t start, int64_t len) const {
        auto sz = sizes(); auto st = strides();
        sz[dim] = len;
        return make_view(std::move(sz), std::move(st), impl_->offset_ + start * st[dim]);
    }
    Tensor transpose(int64_t d0, int64_t d1) const {
        auto sz = sizes(); auto st = strides();
        std::swap(sz[d0], sz[d1]); std::swap(st[d0], st[d1]);
        return make_view(std::move(sz), std::move(st), impl_->offset_);
    }

    // ---- ops that touch data (CPU) ----
    template <class F>
    void for_each_index(F&& f) const {                         // odometer over sizes()
        const auto& sz = sizes(); const auto& st = strides();
        std::vector<int64_t> idx(sz.size(), 0);
        int64_t n = numel();
        for (int64_t k = 0; k < n; ++k) {
            int64_t off = 0;
            for (size_t d = 0; d < sz.size(); ++d) off += idx[d] * st[d];
            f(k, off);
            for (int64_t d = static_cast<int64_t>(sz.size()) - 1; d >= 0; --d)
                if (++idx[d] < sz[d]) break; else idx[d] = 0;
        }
    }
    Tensor& fill_(float v) { float* p = host_ptr(); for_each_index([&](int64_t, int64_t off) { p[off] = v; }); return *this; }
    Tensor& arange_() { float* p = host_ptr(); for_each_index([&](int64_t k, int64_t off) { p[off] = float(k); }); return *this; }
    Tensor clone() const {                                     // deep copy -> NEW Storage
        Tensor out = empty(sizes(), impl_->storage_->device());
        const float* src = host_ptr(); float* dst = out.host_ptr();
        for_each_index([&](int64_t k, int64_t off) { dst[k] = src[off]; });
        return out;
    }
    float at(std::initializer_list<int64_t> ix) const {
        int64_t off = impl_->offset_; size_t d = 0;
        for (auto i : ix) off += i * strides()[d++];
        return static_cast<float*>(impl_->storage_->data())[off];
    }

    // ---- introspection ----
    size_t impl_use_count() const { return impl_.use_count(); }                  // # Tensor handles sharing this view
    size_t storage_use_count() const { return impl_->storage_.use_count(); }     // # TensorImpls sharing the bytes
    bool shares_storage_with(const Tensor& o) const { return impl_->storage_ == o.impl_->storage_; }
    weak_intrusive_ptr<StorageImpl> weak_storage() const { return weak_intrusive_ptr<StorageImpl>(impl_->storage_); }

    // ---- C-API / language-binding style round trip (e.g. store in a PyObject / void* slot) ----
    TensorImpl* release_impl() { return impl_.release(); }
    static Tensor reclaim_impl(TensorImpl* p) { return Tensor(intrusive_ptr<TensorImpl>::reclaim(p)); }
};

static void print_state(const char* tag) {
    std::printf("  [%s] live storages=%ld, live bytes=%ld\n", tag,
                StorageImpl::live_storages.load(), StorageImpl::live_bytes.load());
}

// intrusive_ptr maybe useful 
int intrusive_ptr_usecase() {
    std::printf("1) create + handle copy\n");
    Tensor a = Tensor::empty({4, 6});
    a.arange_();
    Tensor b = a;                                   // handle copy: same TensorImpl
    std::printf("  a,b share one TensorImpl: impl_use_count=%zu, storage_use_count=%zu\n",
                a.impl_use_count(), a.storage_use_count());
    print_state("after create");

    std::printf("2) views share the Storage (no data copied)\n");
    Tensor row  = a.select(0, 1);                   // a[1, :]
    Tensor cols = a.narrow(1, 2, 3);                // a[:, 2:5]
    Tensor t    = a.transpose(0, 1);                // a.T
    std::printf("  impl_use_count(a)=%zu (a,b), storage_use_count=%zu (a,row,cols,t)\n",
                a.impl_use_count(), a.storage_use_count());
    std::printf("  row shares storage with a: %d\n", a.shares_storage_with(row));
    row.fill_(-1.f);                                // write through the view...
    std::printf("  after row.fill_(-1): a[1,0]=%g a[1,5]=%g a[0,0]=%g  (aliasing works)\n",
                a.at({1, 0}), a.at({1, 5}), a.at({0, 0}));
    std::printf("  t[2,1]=%g == a[1,2]=%g\n", t.at({2, 1}), a.at({1, 2}));

    std::printf("3) clone = new Storage\n");
    Tensor c = a.clone();
    std::printf("  c shares storage with a: %d\n", c.shares_storage_with(a));
    print_state("after clone");

    std::printf("4) weak observer: payload dies with the last strong ref\n");
    auto weak = a.weak_storage();
    std::printf("  expired=%d, strong=%zu\n", weak.expired(), weak.use_count());
    b = Tensor(); a = Tensor(); row = Tensor(); cols = Tensor();
    std::printf("  after dropping a,b,row,cols: expired=%d (t still alive)\n", weak.expired());
    t = Tensor();
    std::printf("  after dropping t:            expired=%d, lock()=%s\n", weak.expired(),
                weak.lock() ? "non-null" : "null");
    print_state("payload of a freed, c's kept");   // only c's 96 bytes remain
    // `weak` still holds the (tiny) StorageImpl control object; the 96-byte payload is already freed.

    std::printf("5) raw-pointer round trip (what a Python/C binding does)\n");
    TensorImpl* raw = c.release_impl();             // handle gives up ownership, count unchanged
    std::printf("  c.defined()=%d, raw->refcount still owned by the 'foreign' slot\n", c.defined());
    Tensor c2 = Tensor::reclaim_impl(raw);          // adopt it back
    std::printf("  reclaimed: numel=%lld, impl_use_count=%zu\n", (long long)c2.numel(), c2.impl_use_count());

    std::printf("6) thread-safety: 8 threads copy/drop handles concurrently\n");
    {
        Tensor shared = Tensor::empty({1024});
        std::vector<std::thread> ts;
        for (int i = 0; i < 8; ++i)
            ts.emplace_back([&] {
                for (int k = 0; k < 100000; ++k) { Tensor h = shared; Tensor v = h.narrow(0, 0, 10); (void)v; }
            });
        for (auto& th : ts) th.join();
        std::printf("  impl_use_count=%zu storage_use_count=%zu (expected 1 and 1)\n",
                    shared.impl_use_count(), shared.storage_use_count());
    }

    c2 = Tensor();
    weak.reset();
    print_state("end");
    bool ok = StorageImpl::live_storages == 0 && StorageImpl::live_bytes == 0;
    std::printf("%s\n", ok ? "NO LEAKS" : "LEAK DETECTED");
    return ok ? 0 : 1;
}

hTensorBuffer gBUFF = nullptr;

GST_TensorBuffer::~GST_TensorBuffer() {
    Clear();

    records.Clear();
    /* Free the main buffer
    if (buffer_ != nullptr) {
        cudaFree(buffer_);
        buffer_ = nullptr;
    }*/
}
/**
 * 1. cudaMemGetInfo reports:
        Memory allocated by your program (via cudaMalloc, cudaMemcpy, etc.).
        Additional memory reserved by the CUDA runtime (e.g., for kernels, libraries like cuBLAS).
        Example: If your app allocates 500MB, cudaMemGetInfo might report 600MB due to overhead.
    2. nvidia-smi reports:
        Actual physical memory in use by all processes (including other CUDA apps, graphics compositors, etc.).
        Excludes reserved but unused memory (e.g., fragmentation, CUDA context pools)
*/
void GST_TensorBuffer::Dump(int type, int flag) const {
    size_t sz0, sz1;
    cudaError_t err = cudaMemGetInfo(&sz0, &sz1);
    std::vector<Allocation> mems;

    std::sort(mems.begin(), mems.end(),  // ugly because we don't have a typedef for the std::pair
              [](const Allocation& a, const Allocation& b) { return a.sz > b.sz; });
    size_t szNow = 0, i = 0, szFree = 0;
    double mUsed = (sz1 - sz0) / 1.0e6;
    for (auto mem : mems) {
        szNow += mem.sz;
    }
    size_t szA = szSum[ACTIVATION], szW = szSum[WEIGHT], szG = szSum[GRAD], szMoment = szSum[MOMENT], szTemp = szSum[TEMP], szOther = szSum[OTHER];
    szFree = szA + szW + szG + szMoment + szTemp + szOther - szNow;
    _INFO(
        "[MEMORY] Current usage statistics:  %ld mem-blocks sum=%.3gG(%.3gG) \n\tactivation=%.5gM weight=%.5gM grad=%.5gM moments=%.5gM temp=%.5gM "
        "other=%.5gM\n",
        mems.size(), szNow / 1.0e9, szFree / 1.0e9, szA / 1.0e6, szW / 1.0e6, szG / 1.0e6, szMoment / 1.0e6, szTemp / 1.0e6, szOther / 1.0e6);
    _INFO("\tcurBrach=%.5gM mUsed==%.5gM shadow=%s\n", (szA + szW + szG + szMoment) / 1.0e6, mUsed, hFish->config.distill.isKeepShadoW() ? "ON" : "off");
    int nDump = type == KOIFISH_MISS_MEMBLOCK ? mems.size() : type == KOIFISH_OUTOF_GPUMEMORY ? 32 : nMostMemItem;
    if (nDump > 0) {  // decsend by memory size
        size_t szTotal = 0;
        for (auto mem : mems) {
            szTotal += mem.sz;
            _INFO("\t%ld\t%6gM  @%s \t%.3gG\n", i++, mem.sz / 1.0e6, mem.desc.c_str(), szTotal / 1.0e9);
            if (i > nDump)
                break;
        }
        _INFO("\n");
    }
}

GST_TensorBuffer::Allocation::Allocation(size_t sz_, string d_, SourceLocation where_, void* hData_, int flag)
    : sz(sz_), desc(d_), hData(hData_), where(where_) {
    assert(!desc.empty());
    if (G_Has_(d_, {"ROPE.sincos", "tmpGateDelta"})) {
        DEBUG_HERE;
    }
    char last = desc[desc.length() - 1];
    switch (last) {
        case 'a':
            if (desc.substr(0, 3) == "tmp") {
                type = TYPE::TEMP;
            } else {
                type = TYPE::ACTIVATION;
            }
            break;
        case 'w':
            type = TYPE::WEIGHT;
            break;
        case 'g':
            type = TYPE::GRAD;
            break;
        case 'm':
            type = TYPE::MOMENT;
            break;
        case 't':
            type = TYPE::TEMP;
            break;
        default:
            type = TYPE::OTHER;
            break;
    }
}

void GlobalMemoryInfo(int type, int flag) {}

void* GST_TensorBuffer::Register(size_t sz_, string d_, void* hData_, SourceLocation& where_, int flag) {
    try {
        std::lock_guard<std::mutex> lk(_m);
        Allocation alloc(sz_, d_, where_, hData_, flag);
        records.insert(d_, alloc);
        szSum[alloc.type] += alloc.sz;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}
shared_ptr<GTensor> GST_TensorBuffer::RegisterTensor(Fish* hFish, const string& name_, SHAPE shape, typNUMBER tpD_, SourceLocation& where_, int flag) {
    hGTensor tensor = GT(hFish, tpD_, shape);
    zTensors.push_back(tensor.get());
    return tensor;
}

/**/
bool GST_TensorBuffer::Free(void* hObj, int flag) {
    std::lock_guard<std::mutex> lk(_m);
    for (size_t id = 0; id < records.size(); id++) {
        Allocation alloc;
        records.at(id, alloc);
        if (alloc.hData == hObj) {
            records.erase(alloc.desc);
            return true;
        }
    }
    //_WARN("[Free] failed to find record of %p! records=%ld.\n", hObj, records.size());
    // assert(0 && "GST_TensorBuffer::FreeMem failed");
    return false;
}
/*
 */
bool GST_TensorBuffer::Clear(int flag) {
    try {
        if (bt4c == nullptr)  // hack
            return true;
        bt4c = nullptr, delta = nullptr, tmpDelta = nullptr, outL = nullptr, scratch = nullptr, tmpFF1 = nullptr, tmpW = nullptr, tmpGW = nullptr,
        residual   = nullptr;
        tmpTernary = nullptr;
        gate_delta = nullptr;
        return true;
    } catch (const std::exception& e) {
        _WARN("%s", e.what());
        fflush(stdout);
        return -1000;
    } catch (const char* info) {
        _WARN("%s", info);
        fflush(stdout);
        return -1001;
    } catch (...) {
        _WARN("\r\n%s  Unknown exception !!!", __func__);
        fflush(stdout);
        return -2001;
    }
}


/*
1. RAII (Resource Acquisition Is Initialization)
2. Throw exceptions from constructors​ if resource acquisition fails

class SafeCUDAMemory {
private:
    void* ptr;

public:
    SafeCUDAMemory(size_t size) : ptr(nullptr) {
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA malloc failed");
        }
    }

    ~SafeCUDAMemory() {
        if (ptr != nullptr) {
            // Note: Don't throw from destructor!
            cudaFree(ptr);
        }
    }

    // Prevent copying
    SafeCUDAMemory(const SafeCUDAMemory&) = delete;
    SafeCUDAMemory& operator=(const SafeCUDAMemory&) = delete;

    // Allow moving
    SafeCUDAMemory(SafeCUDAMemory&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    void* get() const { return ptr; }
};
*/