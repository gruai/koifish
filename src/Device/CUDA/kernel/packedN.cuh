/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Align;   Packed data
 *      [todo] PTX assembly instruction
 *
 *  \author Yingshi Chen
 */
#pragma once

#include <functional>

#include "../Device/CUDA/cuda_def.hpp"
#include "../g_def_x.hpp"
#include "../g_float.hpp"
#include "utils.cuh"

#if defined(__CUDACC__)
#define ALIGN(n) __align__(n)
#else
#define ALIGN(n) alignas(n)
#endif

// float4? which is better?
template <typename T, int N>
union ALIGN(sizeof(T) * N) ablock {
    T v[N];
};

enum class TransferMode {
    DEFAULT,
    LDG,
    LU,
    CS,  //  memory load operation with sequential consistency (seq_cst) memory ordering.
};

#define LDMATRIX_X2(R0, R1, addr) asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))
#define CVTA_TO_SHARED_PTX(addr, smem_ptr) asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(addr) : "l"(smem_ptr));

#define LDG32_GUARD_PTX(reg, ptr, guard)      \
    {                                         \
        asm volatile(                         \
            "{.reg .pred p;\n\t"              \
            "setp.ne.u32 p, %2, 0;\n\t"       \
            "@p ld.global.f32 %0, [%1];}\n\t" \
            : "=f"(reg)                       \
            : "l"(ptr), "r"(guard));          \
    }

#define LDG32_GUARD_MOV0_PTX(reg, ptr, guard) \
    {                                         \
        asm volatile(                         \
            "{.reg .pred p;\n\t"              \
            "setp.ne.u32 p, %2, 0;\n\t"       \
            "@!p mov.b32 %0, 0;\n\t"          \
            "@p ld.global.f32 %0, [%1];}\n\t" \
            : "=f"(reg)                       \
            : "l"(ptr), "r"(guard));          \
    }

#define STS128_PTX(reg0, reg1, reg2, reg3, addr)                                                                                \
    {                                                                                                                           \
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n\t" : : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)); \
    }

#define LDS128_PTX(reg0, reg1, reg2, reg3, addr)                                                                                   \
    {                                                                                                                              \
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n\t" : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3) : "l"(addr)); \
    }

#define STS32_PTX(reg, addr)                                               \
    {                                                                      \
        asm volatile("st.shared.f32 [%0], %1;\n" : : "l"(addr), "f"(reg)); \
    }

#define STG32_GUARD_PTX(reg, ptr, guard)       \
    {                                          \
        asm volatile(                          \
            "{.reg .pred p;\n\t"               \
            "setp.ne.u32 p, %2, 0;\n\t"        \
            "@p st.global.f32 [%0], %1;}\n\t"  \
            :                                  \
            : "l"(ptr), "f"(reg), "r"(guard)); \
    }

__device__ __forceinline__ uint32_t ld_shared(const uint32_t* __restrict__ ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int4 ld_shared(const int4* __restrict__ ptr) {
    int4 ret;
    asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_shared(const float* __restrict__ ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) { asm volatile("st.shared.f32 [%0], %1;" ::"l"(ptr), "f"(val)); }

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) { asm volatile("st.shared.u32 [%0], %1;" ::"l"(ptr), "r"(val)); }

template <TransferMode Mode>
struct Transfer {};

template <>
struct Transfer<TransferMode::DEFAULT> {
    template <class T>
    __host__ __device__ static void call(T* dst, const T* src) {
        *dst = *src;
    }
};

template <>
struct Transfer<TransferMode::LDG> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldg(src);
    }
};

template <>
struct Transfer<TransferMode::LU> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldlu(src);
    }
};

template <>
struct Transfer<TransferMode::CS> {
    template <class T>
    __device__ static void call(T* dst, const T* src) {
        *dst = __ldcs(src);
    }
};

/*!
 * \brief Copies `NBytes` from `src` to `dst`, using `CopyType` to perform memory access.
 * \details
 * This means that pointers need to be aligned according to `CopyType`'s requirements,
 * and copies are (most likely) be performed using vectorized access according to
 * `CopyType`.
 * The ranges `[src, src+NBytes)` and `[dst, dst + NBytes)` must be non-overlapping.
 *
 * This function is used to implement `memcpy_aligned`, and generally not intended to
 * be used directly.
 */
template <class CopyType, int NBytes, TransferMode Mode, class TrueType>
__host__ __device__ void memcpy_as(TrueType* __restrict__ dst, const TrueType* __restrict__ src) {
    static_assert(NBytes % sizeof(TrueType) == 0, "Number of bytes must be a multiple of the true type size");
    static_assert(NBytes % sizeof(CopyType) == 0, "Number of bytes must be a multiple of the copy type size");

    // in order to do simple byte-level copying, the underlying type must be trivially copyable (i.e., compatible
    // with memcpy)
    static_assert(std::is_trivially_copyable_v<TrueType>, "TrueType must be trivially copyable");
    const auto* read_address = reinterpret_cast<const CopyType*>(src);
    auto* write_address      = reinterpret_cast<CopyType*>(dst);
#pragma unroll
    for (int i = 0; i < NBytes; i += sizeof(CopyType)) {
        Transfer<Mode>::call(write_address, read_address);
        ++read_address;
        ++write_address;
    }
}
template <std::size_t Count, TransferMode Mode, class T>
__host__ __device__ void memcpy_aligned(T* dst, const T* src, std::integral_constant<std::size_t, Count> = {}) {
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    constexpr const int NBytes = sizeof(T) * Count;
    // ideally, we'd just use a simple memcpy, like below, but that does
    // not always generate vectorized loads
    // std::memcpy(values, __builtin_assume_aligned(address, bytes), bytes);

    if constexpr (NBytes % sizeof(int4) == 0) {
        memcpy_as<int4, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int2) == 0) {
        memcpy_as<int2, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(int1) == 0) {
        memcpy_as<int1, NBytes, Mode>(dst, src);
    } else if constexpr (NBytes % sizeof(short1) == 0) {
        memcpy_as<short1, NBytes, Mode>(dst, src);
    } else {
        memcpy_as<char1, NBytes, Mode>(dst, src);
    }
}

/*!
 * \brief Assume an array of objects of `size` bytes each, what is the alignment of an individual element of that array.
 * \details Assume that the array itself starts at a 16-byte aligned address,
 * what is the worst-case alignment of any object. E.g., for objects of 4 bytes, * alignment is 4, for 6 bytes it is 2, etc.
 */
constexpr __host__ __device__ std::size_t alignment_from_size(std::size_t size) {
    for (int i = 2; i <= 16; i *= 2) {
        if ((size % i) != 0) {
            return i / 2;
        }
    }
    return 16;
}
/*
    Packed data structure that forces the compiler to use 128-bit loads/stores
    in GPUs that support (the LDG.128 and STS.128 instructions)
    This is a bit similar to the use of float4 in the case of 32-bit floats, but supports arbitrary precision.
*/
template <class T, std::size_t ElemCount>
class alignas(alignment_from_size(sizeof(T) * ElemCount)) PackedN {
    static_assert(std::is_trivial_v<T>, "Only trivial types are supported");
    T values[ElemCount] = {};
    //  [lesson]  Default member initializers (like = nullptr) inside a class definition are not allowed in CUDA device code — especially when the class is used
    //  in __device__or __global__functions. T* data_src;          = nullptr;

   public:
    static constexpr const std::size_t size  = ElemCount;
    static constexpr const std::size_t bytes = ElemCount * sizeof(T);

    __host__ __device__ explicit PackedN() {
        //  typ128::zeros() =  constant(static_cast<T>(0.f));
        for (int k = 0; k < size; ++k) {
            values[k] = (T)0.f;
        }
    }

    __host__ __device__ explicit PackedN(int4 bits) {
        static_assert(sizeof(bits) == sizeof(values), "Size mismatch.");
        memcpy(&values, &bits, sizeof(bits));
    }

    __host__ __device__ explicit PackedN(const T* src, int flag = 0x0) {
        memcpy_aligned<size, TransferMode::DEFAULT>(values, src);
        // memcpy(&values, &bits, sizeof(bits));
    }

    //  stochastic rounding
    __host__ __device__ void Randf(int seed, int flag = 0x0) {
        for (int k = 0; k < size; ++k) {
            values[k] = CU_Float2T<T>(values[k], seed);
        }
    }

    // Accumulated x2 to sum, sum is not initialized!, the caller should initialize sum to 0 before the first call!
    __host__ __device__ float X2(float& sum) const {
        // float sum = 0.0f;
        for (int k = 0; k < size; ++k) {
            float a = CU_T2Float(values + k);
            sum += a * a;
        }
        return sum;
    }
    static __host__ __device__ float X2(float& sum, const T* src) {
        const PackedN a4 = load_cs(src);
        a4.X2(sum);
        return sum;
    }
    static __host__ __device__ float X2(float& sum, const T* src, const size_t& start, const size_t& end) {
        for (size_t off4 = start; off4 < end; off4 += size) {
            X2(sum, src + off4);
        }
        return sum;
    }

    static int nThreadOfBlock(int N, int bit, int nT0 = 1024) {  // CU_T4B_BIG
        int nT = nT0;
        while (!(N % nT == 0 && (N / nT) % size == 0)) {
            nT /= 2;
        }
        assert(nT > 1);
        return nT;
    }

    __host__ __device__ void Set(T a = (T)0.f) {
        for (int k = 0; k < size; ++k) {
            values[k] = a;
        }
    }

    __host__ __device__ void Scale(const T& s) {
        for (int k = 0; k < size; ++k) {
            values[k] *= s;
        }
    }

    __host__ __device__ void Hadamard(const T& s, const PackedN& w4) {
        for (int k = 0; k < size; ++k) {
            values[k] = values[k] * s * w4.values[k];
        }
    }

    // Linear exploration a + s * (b - a);  internal is float computation for better precision, and the result is cast back to T.
    __host__ __device__ void Lerp(const T* src1, const T* src2, float s) {
        const auto a_1 = load(src1);
        const auto a_2 = load(src2);
        for (int k = 0; k < size; k++) {
            //  fmaf(t, b, fmaf(-t, a, a));
            float a = CU_T2Float(a_1.values + k), b = CU_T2Float(a_2.values + k);
            values[k] = a + s * (b - a);
        }
    }
    __host__ __device__ void Lerp(const T* src1, float s) {
        const auto a_1 = load(src1);
        for (int k = 0; k < size; k++) {
            //  fmaf(t, b, fmaf(-t, a, a));
            float a = CU_T2Float(a_1.values + k), b = CU_T2Float(values + k);
            values[k] = a + s * (b - a);
        }
    }

    __host__ __device__ void Add(const T* src) {
        const auto a4 = load(src);
        for (int k = 0; k < size; k++) {
            values[k] += a4.values[k];
        }
    }
    __host__ __device__ void Add2(const T* src1, const T* src2) {
        const auto a_1 = load(src1);
        const auto a_2 = load(src2);
        for (int k = 0; k < size; k++) {
            values[k] += a_1.values[k] + a_2.values[k];
        }
    }

    // stochastic rounding
    __host__ __device__ void Add2(float s1, const T* src1, float s2, const T* src2, int seed = 0) {
        const auto a_1 = load(src1);
        const auto a_2 = load(src2);
        for (int k = 0; k < size; k++) {
            float a = CU_T2Float(a_1.values + k), b = CU_T2Float(a_2.values + k);
            float sum = s1 * a + s2 * b;
            values[k] = CU_Float2T<T>(sum, seed);
        }
    }
    __host__ __device__ void AddTo(T* dst) {
        PackedN a4 = load(dst);
        for (int k = 0; k < size; k++) {
            values[k] += a4.values[k];
        }
        store(dst);
    }
    //
    __host__ __device__ void AddFloat(const float* src, int flag = 0x0) {
        int bFloat = 4, off = 0;
        // v1
        for (int o = 0; o < size / bFloat; ++o) {
            for (int i = 0; i < bFloat; ++i, ++src, ++off) {
                values[off] = (T)(*src + (float)values[off]);
            }
        }
        //  v0
        // using f128 = PackedN<float, 4>;
        // for (int o = 0; o < size / bFloat; ++o) {
        //     f128 a4 = f128::load(src + o * bFloat);
        //     for (int i = 0; i < bFloat; ++i) {
        //         int x         = o * bFloat + i;
        //         values[x] = (T)(a4[i] + (float)values[x]);
        //     }
        // }
    }

    constexpr static __host__ __device__ PackedN constant(T value) {
        PackedN result;
        for (int k = 0; k < size; ++k) {
            result.values[k] = value;
        }
        return result;
    }

    constexpr static __host__ __device__ PackedN zeros() { return constant(static_cast<T>(0.f)); }
    constexpr static __host__ __device__ void zeros(T* dst) { zeros().store(dst); }

    constexpr static __host__ __device__ PackedN ones() { return constant(1.f); }

    template <class U>
    constexpr static __host__ __device__ PackedN from(PackedN<U, ElemCount> other) {
        PackedN<T, ElemCount> result;
        for (int i = 0; i < ElemCount; ++i) {
            result[i] = static_cast<T>(other[i]);
        }
        return result;
    }

    constexpr __host__ __device__ T& operator[](int index) { return values[index]; }

    constexpr __host__ __device__ const T& operator[](int index) const { return values[index]; }

    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(values), "Size mismatch.");
        memcpy(&bits, &values, sizeof(bits));
        return bits;
    }

    static __host__ __device__ PackedN load(const T* address) {
        PackedN result;
        memcpy_aligned<size, TransferMode::DEFAULT>(result.values, address);
        return result;
    }

    static __device__ PackedN load_ldg(const T* address) {
        PackedN result;
        memcpy_aligned<size, TransferMode::LDG>(result.values, address);
        return result;
    }

    static __device__ PackedN load_lu(const T* address) {
        PackedN result;
        memcpy_aligned<size, TransferMode::LU>(result.values, address);
        return result;
    }
    // loading a value that other threads may be writing to with a corresponding __stcs()(store with sequential consistency), and you need a guaranteed,
    // globally consistent view of when that load happens relative to other memory operations in your kernel.
    static __device__ PackedN load_cs(const T* address) {
        PackedN result;
        memcpy_aligned<size, TransferMode::CS>(result.values, address);
        return result;
    }

    __host__ __device__ void store(T* dst) { memcpy_aligned<size, TransferMode::DEFAULT>(dst, values); }

    template <typename Tx>
    __host__ __device__ void storeX(Tx* dst) const {
        assert(0);  // how to instaniate for the case T=float
        for (int k = 0; k < size; ++k) {
            dst[k] = (Tx)(values[k]);
        }
    }
};
typedef PackedN<float, 4> f128;
typedef PackedN<floatX, 16 / sizeof(floatX)> X128;
typedef PackedN<float, 2> f64;
typedef PackedN<floatX, 8 / sizeof(floatX)> x64;

template <typename T>
using Packed128 = PackedN<T, 16 / sizeof(T)>;

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// load_ a PackedN from an aligned memory address      reinterpret_cast is not SAFE!
template <class T>
__device__ inline Packed128<T> load128(const T* address) {
    return Packed128<T>{*reinterpret_cast<const int4*>(address)};
}
// load_ a PackedN from an aligned memory address with streaming cache hint
template <class T>
__device__ inline Packed128<T> load128cs(const T* address) {
    return Packed128<T>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store_ a PackedN to an aligned memory address
template <class T>
__device__ inline void store128(T* target, Packed128<T> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store_ a PackedN to an aligned memory address with streaming cache hint
template <class T>
__device__ void store128cs(T* target, Packed128<T> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store_ a PackedN to an aligned memory address while caching in L2 but bypassing L1
template <class T>
__device__ void store128cg(T* target, Packed128<T> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

/**
 * Task allocation - One thread, one data
 *
 * no sync between wrap, block ...
 */
template <typename Typ>
struct TASKA_1p1 {
    using typ128 = PackedN<Typ, 16 / sizeof(Typ)>;
    cudaStream_t stream;
    size_t smem = 0x0;
    int nTask = 0, N = 0;
    int block3 = 0, tpb = 512;  //  tpb is the number of threads in one block
    int grid3 = 0, nBlock = 0;  // grid3=(nBlock, 1, 1), nBlock is the total number of blocks

    TASKA_1p1(int N_, cudaStream_t stream_, int flag = 0x0) : nTask(1), N(N_), stream(stream_) {
        if (N % typ128::size != 0)
            return;
        block3 = tpb;
        nBlock = CEIL_DIV(N, tpb * typ128::size);
        grid3  = nBlock;
    }
    bool isValid() const {
        if (nBlock <= 0)
            return false;
        assert(stream != nullptr);
        assert(nBlock * tpb * typ128::size >= N);
        return true;
    }
};
//  One thread per data
// #define T1p1(CU_kernel) CU_kernel<<<taska.grid3, taska.block3, taska.smem, main_stream>>>
// template< template<typename> class CU_kernel, typename Typ>
// void T1p1(CU_kernel kernel, const TASKA_1p1<Typ>& taska, int flag) {
//     if (taska.isValid()) {
//         kernel<Typ><<<taska.grid3, taska.block3, taska.smem, taska.stream>>>(taska);
//     }
// }
template <typename KernelType, typename Typ, typename... Args>
void T1p1(KernelType kernel, const TASKA_1p1<Typ>& taska, Args&&... args) {
    assert(taska.isValid());
    kernel<<<taska.grid3, taska.block3, taska.smem, taska.stream>>>(taska, std::forward<Args>(args)...);
    // cudaDeviceSynchronize();
    // cudaCheck(cudaGetLastError());
}

/**
 * Task allocation - Single Wrap, Multiple Data
 *
 *  There are total nTask & Each task has C elements
 *
 * 1.   grid_size = deviceProp.multiProcessorCount * 2(blocks/SM)
 * 2.   Each wrap(32 threads) for one task, so tasks of each round is warpsInGrid
 * 3.   wrap_stride = 32*x128::size elements
 */
template <typename Typ>
struct TASKA_SWMD {
    using typ128 = PackedN<Typ, 16 / sizeof(Typ)>;
    int nTask    = 0;
    int block3   = 512;
    int grid3 = 0, nBlock = 0;  // grid3=(nBlock, 1, 1), nBlock is the total number of blocks
    int ldC0 = 0x0, ldC = 0x0;
    int wrap_stride        = 0x0;  // number of elements by one wrap
    int C_nStride          = 0;    // C_nStride*wrap_stride = C
    int warpsInBlock       = 0;    // number of warps in one block
    int warpsInGrid        = 0;    // number of warps in grid. = nBlock * warpsInBlock;
    size_t smem            = 0x0;
    unsigned int wrap_mask = 0xffffffff;

    TASKA_SWMD(int nTask_, int C, int nSM, int flag = 0x0) : nTask(nTask_), ldC0(C) {
        using x128     = PackedN<Typ, 16 / sizeof(Typ)>;
        int block_size = block3;
        warpsInBlock   = block_size / WARP_SIZE;
        if (flag == 0x100) {  //  each task on 1 warp
            assert(nTask % warpsInBlock == 0);
            nBlock = nTask / warpsInBlock;
        } else {
            nBlock = BLOCKS_PER_SM * nSM;  // Streaming Multiprocessor
        }
        grid3       = nBlock;     // grid3=(nBlock, 1, 1)
        int TpW     = WARP_SIZE;  // threads per wrap
        wrap_stride = TpW * x128::size;
        if (ldC0 < wrap_stride) {
            assert(ldC0 % x128::size == 0);
            int nT    = ldC0 / x128::size;
            wrap_mask = (1u << nT) - 1;
        }
        ldC = CEIL_DIV(C, wrap_stride) * wrap_stride;
        assert(ldC >= C);
        smem = (2 * ldC + 2 * (block_size - 32) * f128::size) * sizeof(float) * 2;  // 23552

        warpsInGrid = nBlock * warpsInBlock;

        C_nStride = CEIL_DIV(C, wrap_stride);  // + 2;
    }

    //(warpThreadIdx * typ128::size) + (c * smp.wrap_stride);
    int __host__ __device__ IdxInC(int c, int warpThreadIdx) {
        int idx = warpThreadIdx * typ128::size + c * wrap_stride;
        assert(idx < ldC);
        return idx;
    }

    // template <typename Func, typename... Args>
    // void Run(Func&& func, Args&&... args) {
    //     std::forward<Func>(func)(std::forward<Args>(args)...);
    // }
};

//  Single Wrap, Multiple Data
#define SWMD(CU_kernel) CU_kernel<<<smp.grid3, smp.block3, smp.smem, main_stream>>>
//  Single Wrap, 1 head
#define SW1H(CU_kernel) CU_kernel<<<smp.grid3, smp.block3, smp.smem, main_stream>>>
//  Single Wrap, 1 token
#define SW1T(CU_kernel) CU_kernel<<<smp.grid3, smp.block3, smp.smem, main_stream>>>

/**
 * @brief Provides information about Pack2 of elements for a given type.
 *
 * @tparam T The type for which to provide Pack2 information.
 */
template <typename T>
struct Pack2 {
    /**
     * @brief The number of elements packed together.
     *
     * @return constexpr int representing number of elements within the type.
     */
    static __device__ inline constexpr int num() { return 1; }
    /**
     * @brief Packs a single T element twice (replicated) into its packed type.
     *
     * @param i[in] The element to pack.
     * @return The packed type.
     */
    static __device__ inline constexpr T pack(const bf16& i);
};
template <>
struct Pack2<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type   = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16& i) { return bf16_2{i, i}; }
};
template <>
struct Pack2<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type   = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16& i) { return bf16_2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<half> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type   = half_2;
    static __device__ inline constexpr half_2 pack(const half& i) { return half_2{i, i}; }
};
template <>
struct Pack2<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type   = half_2;
    static __device__ inline constexpr half_2 pack(const half& i) { return half_2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<float> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type   = float2;
    static __device__ inline constexpr float2 pack(const float& i) { return float2{i, i}; }
};
template <>
struct Pack2<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type   = float2;
    static __device__ inline constexpr float2 pack(const float& i) { return float2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<char> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = char;
    using packed_type   = char2;
    static __device__ inline constexpr char2 pack(const char& i) { return char2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<char2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = char;
    using packed_type   = char2;
    static __device__ inline constexpr char2 pack(const char& i) { return char2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<int> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type   = int2;
    static __device__ inline constexpr int2 pack(const int& i) { return int2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<int2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type   = int2;
    static __device__ inline constexpr int2 pack(const int& i) { return int2{i, i}; }  // this replication makes code cleaner later.
};
struct uint64_2 {
    uint64_t x, y;
};
template <>
struct Pack2<uint64_t> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = uint64_t;
    using packed_type   = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t& i) { return uint64_2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<uint64_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = uint64_t;
    using packed_type   = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t& i) { return uint64_2{i, i}; }  // this replication makes code cleaner later.
};
template <>
struct Pack2<float4> {
    static __device__ inline constexpr int num() { return 4; }
};
template <>
struct Pack2<int4> {
    static __device__ inline constexpr int num() { return 4; }
};

#ifdef CUDA_HOPPER
template <>
struct Pack2<fp8e4m3> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e4m3;
    using packed_type   = fp8e4m3_4;
};
template <>
struct Pack2<fp8e4m3_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e4m3;
    using packed_type   = fp8e4m3_4;
};
template <>
struct Pack2<fp8e5m2> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = fp8e5m2;
    using packed_type   = fp8e5m2_4;
};
template <>
struct Pack2<fp8e5m2_4> {
    static __device__ inline constexpr int num() { return 4; }
    using unpacked_type = fp8e5m2;
    using packed_type   = fp8e5m2_4;
};
#endif

template <typename Typ>
__global__ void CU_add3(TASKA_1p1<Typ> taska, Typ* out_, const Typ* inp1, const Typ* inp2, int flag = 0x0) {
    using typ128 = PackedN<Typ, 16 / sizeof(Typ)>;
    int idx      = (blockIdx.x * blockDim.x + threadIdx.x) * typ128::size;
    if (idx >= taska.N)
        return;
    typ128 out;
    out.Add2(inp1 + idx, inp2 + idx);
    out.store(out_ + idx);
}


/*
    Performs a deterministic x2
        atomicAdd in CUDA is not deterministic because it involves race conditions when multiple threads attempt to modify the same memory location
   simultaneously.
*/
template <class T, class Tout>
__global__ static void CU_x2_atomic(Tout* out, const T* x0, size_t N) {
    using x128 = PackedN<T, 16 / sizeof(T)>;
    assert(N % x128::size == 0);
    int idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x, off = idx * x128::size;
    if (off >= N) {  // guard
        return;
    }
    assert(blockDim.x <= 1024);

    // float a         = x0[idx];
    // float sum       = a * a;
    float sum = 0.f;
    x128::X2(sum, x0 + off);
    float block_sum = blockReduce_v0<warpReduceSum>(sum, true);
    if (tid == 0) {
        // printf("%g + %g\n",*out,block_sum);
        atomicAdd(out, (Tout)block_sum);
    }
}