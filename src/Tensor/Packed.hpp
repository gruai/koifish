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

// load a Packed128 from an aligned memory address
template<class T>
__device__ Packed128<T> load128(const T* address) {
    return Packed128<T>{*reinterpret_cast<const int4*>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template<class T>
__device__ Packed128<T> load128cs(const T* address) {
    return Packed128<T>{__ldcs(reinterpret_cast<const int4*>(address))};
}
// store a Packed128 to an aligned memory address
template<class T>
__device__ void store128(T* target, Packed128<T> value) {
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