/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  \brief Align;   Packed data
 *  \author Yingshi Chen
 */
#pragma Once

#include "../g_float.hpp"

#if defined(__CUDACC__)
    #define ALIGN(n) __align__(n)
#else
    #define ALIGN(n) ALIGN(n)
#endif

// float4? which is better?
template <typename T, int N>
union ALIGN(sizeof(T) * N) ablock {
	T v[N];
};

/*
    Packed128 data structure that forces the compiler to use 128-bit loads/stores
    in GPUs that support (the LDG.128 and STS.128 instructions)
    This is a bit similar to the use of float4 in the case of 32-bit floats, but supports arbitrary precision.
*/
template<typename T>
struct ALIGN(16) Packed128 {
    static constexpr const size_t size = sizeof(int4) / sizeof(T);
    T payload[size];

    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__  static Packed128 constant(T value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    __device__ T& operator[](int index) {
        return payload[index];
    }
    __device__ const T& operator[](int index) const {
        return payload[index];
    }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
};
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
// load a Packed128 from an aligned memory address      reinterpret_cast is not SAFE!
template<class T>
__device__ inline Packed128<T> load128(const T* address) {
    return Packed128<T>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class T>
__device__ inline Packed128<T> load128cs(const T* address) {
    return Packed128<T>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class T>
__device__ inline void store128(T* target, Packed128<T> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template<class T>
__device__ void store128cs(T* target, Packed128<T> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class T>
__device__ void store128cg(T* target, Packed128<T> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

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

__device__  __forceinline__ uint32_t ld_shared(const uint32_t* __restrict__ ptr) {
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ int4 ld_shared(const int4* __restrict__ ptr) {
    int4 ret;
    asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];" : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
    return ret;
}

__device__  __forceinline__ float ld_shared(const float* __restrict__ ptr) {
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(const float* ptr, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(const uint32_t* ptr, uint32_t val) {
    asm volatile("st.shared.u32 [%0], %1;" :: "l"(ptr), "r"(val));
}
/**
 * @brief Provides information about Pack2 of elements for a given type.
 *
 * @tparam T The type for which to provide Pack2 information.
 */
template<typename T> 
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
    static __device__ inline constexpr T pack(const bf16 &i);
};
template<> struct Pack2<bf16> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; }
};
template<> struct Pack2<bf16_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = bf16;
    using packed_type = bf16_2;
    static __device__ inline constexpr bf16_2 pack(const bf16 &i) { return bf16_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<half> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; }
};
template<> struct Pack2<half_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = half;
    using packed_type = half_2;
    static __device__ inline constexpr half_2 pack(const half &i) { return half_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<float> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; }
};
template<> struct Pack2<float2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = float;
    using packed_type = float2;
    static __device__ inline constexpr float2 pack(const float &i) { return float2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<char> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = char;
    using packed_type = char2;
    static __device__ inline constexpr char2 pack(const char &i) { return char2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<char2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = char;
    using packed_type = char2;
    static __device__ inline constexpr char2 pack(const char &i) { return char2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<int> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<int2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = int;
    using packed_type = int2;
    static __device__ inline constexpr int2 pack(const int &i) { return int2{i, i}; } // this replication makes code cleaner later.
};
struct uint64_2 { uint64_t x, y; };
template<> struct Pack2<uint64_t> {
    static __device__ inline constexpr int num() { return 1; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<uint64_2> {
    static __device__ inline constexpr int num() { return 2; }
    using unpacked_type = uint64_t;
    using packed_type = uint64_2;
    static __device__ inline constexpr uint64_2 pack(const uint64_t &i) { return uint64_2{i, i}; } // this replication makes code cleaner later.
};
template<> struct Pack2<float4> {
    static __device__ inline constexpr int num() { return 4; }
};
template<> struct Pack2<int4> {
    static __device__ inline constexpr int num() { return 4; }
};

#ifdef CUDA_HOPPER
    template<> struct Pack2<fp8e4m3> {
        static __device__ inline constexpr int num() { return 1; }
        using unpacked_type = fp8e4m3;
        using packed_type = fp8e4m3_4;
    };
    template<> struct Pack2<fp8e4m3_4> {
        static __device__ inline constexpr int num() { return 4; }
        using unpacked_type = fp8e4m3;
        using packed_type = fp8e4m3_4;
    };
    template<> struct Pack2<fp8e5m2> {
        static __device__ inline constexpr int num() { return 1; }
        using unpacked_type = fp8e5m2;
        using packed_type = fp8e5m2_4;
    };
    template<> struct Pack2<fp8e5m2_4> {
        static __device__ inline constexpr int num() { return 4; }
        using unpacked_type = fp8e5m2;
        using packed_type = fp8e5m2_4;
    };
#endif