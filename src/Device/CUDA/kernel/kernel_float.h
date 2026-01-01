/*
 * Kernel Float: Header-only library for vector types and reduced precision floating-point math.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /**
  * @//github.com/KernelTuner/kernel_float
  */
//================================================================================
// this file has been auto-generated, do not modify its contents!
// date: 2025-01-27 16:26:28.827757
// git hash: 09dc82096e4c013a079f0e315da1ccce17453c93
//================================================================================

#ifndef KERNEL_FLOAT_MACROS_H
#define KERNEL_FLOAT_MACROS_H

#ifdef __HIPCC__
#include "hip/hip_runtime.h"
#endif

// clang-format off
#ifdef __CUDACC__
    #define KERNEL_FLOAT_IS_CUDA (1)
    #define KERNEL_FLOAT_DEVICE    __forceinline__ __device__

    #ifdef __CUDA_ARCH__
        #define KERNEL_FLOAT_INLINE    __forceinline__ __device__
        #define KERNEL_FLOAT_IS_DEVICE (1)
    #else  // __CUDA_ARCH__
        #define KERNEL_FLOAT_INLINE  __forceinline__ __host__
        #define KERNEL_FLOAT_IS_HOST (1)
    #endif  // __CUDA_ARCH__
#elif defined(__HIPCC__)
    #define KERNEL_FLOAT_IS_HIP (1)
    #define KERNEL_FLOAT_DEVICE  __attribute__((always_inline)) __device__
    #define KERNEL_FLOAT_INLINE  __attribute__((always_inline)) __host__ __device__

    #ifdef __HIP_DEVICE_COMPILE__
        #define KERNEL_FLOAT_IS_DEVICE (1)
    #else
        #define KERNEL_FLOAT_IS_HOST (1)
    #endif

#else
    #define KERNEL_FLOAT_INLINE  inline
    #define KERNEL_FLOAT_IS_HOST (1)
#endif

#ifndef KERNEL_FLOAT_FP16_AVAILABLE
    #define KERNEL_FLOAT_FP16_AVAILABLE (1)
#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#ifndef KERNEL_FLOAT_BF16_AVAILABLE
    #define KERNEL_FLOAT_BF16_AVAILABLE (1)
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#ifndef KERNEL_FLOAT_FP8_AVAILABLE
    #ifdef __CUDACC_VER_MAJOR__
        #define KERNEL_FLOAT_FP8_AVAILABLE (__CUDACC_VER_MAJOR__ >= 12)
    #else  // __CUDACC_VER_MAJOR__
        #define KERNEL_FLOAT_FP8_AVAILABLE (0)
    #endif  // __CUDACC_VER_MAJOR__
#endif  // KERNEL_FLOAT_FP8_AVAILABLE

#define KERNEL_FLOAT_ASSERT(expr) \
    do {                          \
    } while (0)
#define KERNEL_FLOAT_UNREACHABLE __builtin_unreachable()

// Somet utility macros
#define KERNEL_FLOAT_CONCAT_IMPL(A, B) A##B
#define KERNEL_FLOAT_CONCAT(A, B)      KERNEL_FLOAT_CONCAT_IMPL(A, B)
#define KERNEL_FLOAT_CALL(F, ...)      F(__VA_ARGS__)

// TOOD: check if this way is support across all compilers
#if defined(__has_builtin) && 0  // Seems that `__builtin_assume_aligned` leads to segfaults
    #if __has_builtin(__builtin_assume_aligned)
        #define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) static_cast <TYPE*>(
                __builtin_assume_aligned(static_cast <TYPE*>(PTR), (ALIGNMENT)))
    #else
        #define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) (PTR)
    #endif
#else
    #define KERNEL_FLOAT_ASSUME_ALIGNED(TYPE, PTR, ALIGNMENT) (PTR)
#endif

#define KERNEL_FLOAT_MAX_ALIGNMENT (32)

#if KERNEL_FLOAT_FAST_MATH
    #define KERNEL_FLOAT_POLICY ::kernel_float::fast_policy;
#endif

                // clang-format on

#endif  // KERNEL_FLOAT_MACROS_H
#ifndef KERNEL_FLOAT_CORE_H
#define KERNEL_FLOAT_CORE_H

namespace kernel_float {
    template <size_t... Is>
    struct index_sequence {
        static constexpr size_t size = sizeof...(Is);
    };

    namespace detail {
    template <size_t N>
    struct make_index_sequence_impl {};

    // Benchmarks show that it is much faster to predefine all possible index sequences instead of doing something
    // recursive with variadic templates.
#define KERNEL_FLOAT_INDEX_SEQ(N, ...)            \
    template <>                                   \
    struct make_index_sequence_impl<N> {          \
        using type = index_sequence<__VA_ARGS__>; \
    };

    KERNEL_FLOAT_INDEX_SEQ(0)
    KERNEL_FLOAT_INDEX_SEQ(1, 0)
    KERNEL_FLOAT_INDEX_SEQ(2, 0, 1)
    KERNEL_FLOAT_INDEX_SEQ(3, 0, 1, 2)
    KERNEL_FLOAT_INDEX_SEQ(4, 0, 1, 2, 3)
    KERNEL_FLOAT_INDEX_SEQ(5, 0, 1, 2, 3, 4)
    KERNEL_FLOAT_INDEX_SEQ(6, 0, 1, 2, 3, 4, 5)
    KERNEL_FLOAT_INDEX_SEQ(7, 0, 1, 2, 3, 4, 5, 6)
    KERNEL_FLOAT_INDEX_SEQ(8, 0, 1, 2, 3, 4, 5, 6, 7)
    KERNEL_FLOAT_INDEX_SEQ(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
    KERNEL_FLOAT_INDEX_SEQ(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    KERNEL_FLOAT_INDEX_SEQ(11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    KERNEL_FLOAT_INDEX_SEQ(12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    KERNEL_FLOAT_INDEX_SEQ(13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    KERNEL_FLOAT_INDEX_SEQ(14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    KERNEL_FLOAT_INDEX_SEQ(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    KERNEL_FLOAT_INDEX_SEQ(16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    KERNEL_FLOAT_INDEX_SEQ(17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

    }  // namespace detail

    template <size_t N>
    using make_index_sequence = typename detail::make_index_sequence_impl<N>::type;

    namespace detail {
    template <typename T>
    struct decay_impl {
        using type = T;
    };

    template <typename T>
    struct decay_impl<const T> {
        using type = T;
    };

    template <typename T>
    struct decay_impl<const T&> {
        using type = T;
    };

    template <typename T>
    struct decay_impl<T&> {
        using type = T;
    };

    template <typename T>
    struct decay_impl<T&&> {
        using type = T;
    };
    }  // namespace detail

    template <typename T>
    using decay_t = typename detail::decay_impl<T>::type;

    template <typename A, typename B>
    struct promote_type {};

    template <typename T>
    struct promote_type<T, T> {
        using type = T;
    };

    template <typename T>
    struct promote_type<void, T> {
        using type = T;
    };

    template <typename T>
    struct promote_type<T, void> {
        using type = T;
    };

    template <>
    struct promote_type<void, void> {
        using type = void;
    };

#define KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, U) \
    template <>                                 \
    struct promote_type<T, U> {                 \
        using type = T;                         \
    };                                          \
    template <>                                 \
    struct promote_type<U, T> {                 \
        using type = T;                         \
    };

    // T + bool becomes T
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(char, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed char, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed short, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed int, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed long, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed long long, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned char, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned short, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned int, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned long, bool)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned long long, bool)

    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, float)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, float)
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, double)

#define KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(T)                \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, char)               \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed char)        \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed short)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed int)         \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed long)        \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, signed long long)   \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned char)      \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned short)     \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned int)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned long)      \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, unsigned long long) \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(T, bool)

    KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(float)
    KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(double)
    KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(long double)

#define KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(T, U)       \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(signed T, signed U) \
    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(unsigned T, unsigned U)

    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(short, char)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(int, char)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(int, short)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, char)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, short)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long, int)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, char)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, short)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, int)
    KERNEL_FLOAT_DEFINE_PROMOTED_INTEGRAL(long long, long)

    template <typename T>
    struct promote_type<T*, T*> {
        using type = T*;
    };

#define KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(I) \
    template <typename T>                       \
    struct promote_type<T*, I> {                \
        using type = T*;                        \
    };                                          \
    template <typename T>                       \
    struct promote_type<I, T*> {                \
        using type = T*;                        \
    };

    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(char)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed char)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed short)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed int)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed long)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(signed long long)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned char)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned short)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned int)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned long)
    KERNEL_FLOAT_DEFINE_PROMOTED_POINTER(unsigned long long)

    // half precision
    //    KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(half)
    //    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half, bool)
    //    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, half)
    //    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, half)
    //    KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(long double, half)

    namespace detail {
    template <typename... Ts>
    struct multi_promote_type;

    template <typename T>
    struct multi_promote_type<T> {
        using type = T;
    };

    template <typename A, typename B>
    struct multi_promote_type<A, B> : promote_type<A, B> {};

    template <typename A, typename B, typename C, typename... Rest>
    struct multi_promote_type<A, B, C, Rest...> : multi_promote_type<typename promote_type<A, B>::type, C, Rest...> {};

    }  // namespace detail

    template <typename... Ts>
    using promote_t = typename detail::multi_promote_type<decay_t<Ts>...>::type;

    namespace detail {

    template <typename A, typename B>
    struct is_same_type_impl {
        static constexpr bool value = false;
    };

    template <typename A>
    struct is_same_type_impl<A, A> {
        static constexpr bool value = true;
    };
    }  // namespace detail

    template <typename A, typename B>
    static constexpr bool is_same_type = detail::is_same_type_impl<A, B>::value;

    namespace detail {
    template <typename From, typename To, typename Common = To>
    struct is_implicit_convertible_impl {
        static constexpr bool value = false;
    };

    template <typename From, typename To>
    struct is_implicit_convertible_impl<From, To, typename promote_type<From, To>::type> {
        static constexpr bool value = true;
    };
    }  // namespace detail

    template <typename From, typename To>
    static constexpr bool is_implicit_convertible = detail::is_implicit_convertible_impl<decay_t<From>, decay_t<To>>::value;

    namespace detail {
    template <typename T>
    KERNEL_FLOAT_INLINE T& declval() {
        while (1);
    }
    }  // namespace detail

    template <typename F, typename... Args>
    using result_t = decltype((detail::declval<F>())(detail::declval<Args>()...));

    namespace detail {
    template <bool, typename T>
    struct enable_if_impl {};

    template <typename T>
    struct enable_if_impl<true, T> {
        using type = T;
    };
    }  // namespace detail

    template <bool C, typename T = void>
    using enable_if_t = typename detail::enable_if_impl<C, T>::type;

    KERNEL_FLOAT_INLINE
    constexpr size_t round_up_to_power_of_two(size_t n) {
        size_t result = 1;
        while (result < n) {
            result *= 2;
        }
        return result;
    }

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_BASE_H
#define KERNEL_FLOAT_BASE_H

#if KERNEL_FLOAT_IS_HIP
#include <hip/hip_vector_types.h>
#endif

namespace kernel_float {

template <typename T, size_t N, size_t Alignment = alignof(T)>
struct alignas(Alignment) aligned_array {
    KERNEL_FLOAT_INLINE
    T* data() { return items_; }

    KERNEL_FLOAT_INLINE
    const T* data() const { return items_; }

    T items_[N] = {};
};

template <typename T, size_t Alignment>
struct alignas(Alignment) aligned_array<T, 1, Alignment> {
    KERNEL_FLOAT_INLINE
    aligned_array(T item = {}) : item_(item) {}

    KERNEL_FLOAT_INLINE
    operator T() const { return item_; }

    KERNEL_FLOAT_INLINE
    T* data() { return &item_; }

    KERNEL_FLOAT_INLINE
    const T* data() const { return &item_; }

    T item_ = {};
};

template <typename T, size_t Alignment>
struct aligned_array<T, 0, Alignment> {
    KERNEL_FLOAT_INLINE
    T* data() { while (true); }

    KERNEL_FLOAT_INLINE
    const T* data() const { while (true); }
};

KERNEL_FLOAT_INLINE
static constexpr size_t compute_max_alignment(size_t total_size, size_t min_align) {
    if (total_size % 32 == 0 || min_align >= 32) {
        return 32;
    } else if (total_size % 16 == 0 || min_align == 16) {
        return 16;
    } else if (total_size % 8 == 0 || min_align == 8) {
        return 8;
    } else if (total_size % 4 == 0 || min_align == 4) {
        return 4;
    } else if (total_size % 2 == 0 || min_align == 2) {
        return 2;
    } else {
        return 1;
    }
}

template <typename T, size_t N>
using vector_storage = aligned_array<T, N, compute_max_alignment(sizeof(T) * N, alignof(T))>;

template <size_t... Ns>
struct extent;

template <size_t N>
struct extent<N> {
    static constexpr size_t number_of_elements = N;
};

template <typename E>
static constexpr size_t extent_size = E::number_of_elements;

namespace detail {
// Indicates that elements of type `T` offer less precision than floats, thus operations
// on elements of type `T` can be performed by upcasting them to ` float`.
template <typename T>
struct allow_float_fallback {
    static constexpr bool value = false;
};

template <>
struct allow_float_fallback<float> {
    static constexpr bool value = true;
};
}  // namespace detail

template <typename T>
struct into_vector_impl {
    using value_type  = T;
    using extent_type = extent<1>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, 1> call(const T& input) { return vector_storage<T, 1>{input}; }
};

template <typename T, size_t N>
struct into_vector_impl<T[N]> {
    using value_type  = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const T (&input)[N]) { return call(input, make_index_sequence<N>()); }

    private:
    template <size_t... Is>
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(const T (&input)[N], index_sequence<Is...>) {
        return {input[Is]...};
    }
};

template <typename V>
struct into_vector_impl<const V> : into_vector_impl<V> {};

template <typename V>
struct into_vector_impl<V&> : into_vector_impl<V> {};

template <typename V>
struct into_vector_impl<const V&> : into_vector_impl<V> {};

template <typename V>
struct into_vector_impl<V&&> : into_vector_impl<V> {};

template <typename T, size_t N, size_t A>
struct into_vector_impl<aligned_array<T, N, A>> {
    using value_type  = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> call(const aligned_array<T, N, A>& input) { return input; }
};

#define KERNEL_FLOAT_DEFINE_VECTOR_TYPE(T, T1, T2, T3, T4)                        \
    template <>                                                                   \
    struct into_vector_impl<::T1> {                                               \
        using value_type  = T;                                                    \
        using extent_type = extent<1>;                                            \
                                                                                    \
        KERNEL_FLOAT_INLINE                                                       \
        static vector_storage<T, 1> call(::T1 v) { return {v.x}; }                \
    };                                                                            \
                                                                                    \
    template <>                                                                   \
    struct into_vector_impl<::T2> {                                               \
        using value_type  = T;                                                    \
        using extent_type = extent<2>;                                            \
                                                                                    \
        KERNEL_FLOAT_INLINE                                                       \
        static vector_storage<T, 2> call(::T2 v) { return {v.x, v.y}; }           \
    };                                                                            \
                                                                                    \
    template <>                                                                   \
    struct into_vector_impl<::T3> {                                               \
        using value_type  = T;                                                    \
        using extent_type = extent<3>;                                            \
                                                                                    \
        KERNEL_FLOAT_INLINE                                                       \
        static vector_storage<T, 3> call(::T3 v) { return {v.x, v.y, v.z}; }      \
    };                                                                            \
                                                                                    \
    template <>                                                                   \
    struct into_vector_impl<::T4> {                                               \
        using value_type  = T;                                                    \
        using extent_type = extent<4>;                                            \
                                                                                    \
        KERNEL_FLOAT_INLINE                                                       \
        static vector_storage<T, 4> call(::T4 v) { return {v.x, v.y, v.z, v.w}; } \
    };

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(char, char1, char2, char3, char4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(short, short1, short2, short3, short4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(int, int1, int2, int3, int4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long, long1, long2, long3, long4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(long long, longlong1, longlong2, longlong3, longlong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned char, uchar1, uchar2, uchar3, uchar4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned short, ushort1, ushort2, ushort3, ushort4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned int, uint1, uint2, uint3, uint4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long, ulong1, ulong2, ulong3, ulong4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong1, ulonglong2, ulonglong3, ulonglong4)

KERNEL_FLOAT_DEFINE_VECTOR_TYPE(float, float1, float2, float3, float4)
KERNEL_FLOAT_DEFINE_VECTOR_TYPE(double, double1, double2, double3, double4)

template <typename T, typename E, typename S = vector_storage<T, E::number_of_elements>>
struct vector;

template <typename T, typename E, typename S>
struct into_vector_impl<vector<T, E, S>> {
    using value_type  = T;
    using extent_type = E;

    KERNEL_FLOAT_INLINE
    static vector_storage<T, extent_size<E>> call(const vector<T, E, S>& input) { return input.storage(); }
};

template <typename T>
struct preferred_vector_size {
    static constexpr size_t value = 1;
};

template <typename V>
struct vector_traits;

template <typename T, typename E, typename S>
struct vector_traits<vector<T, E, S>> {
    using value_type   = T;
    using extent_type  = E;
    using storage_type = S;
    using vector_type  = vector<T, E, S>;
};

template <typename V>
using vector_value_type = typename into_vector_impl<V>::value_type;

template <typename V>
using vector_extent_type = typename into_vector_impl<V>::extent_type;

template <typename V>
static constexpr size_t vector_size = extent_size<vector_extent_type<V>>;

template <typename V>
using into_vector_type = vector<vector_value_type<V>, vector_extent_type<V>>;

template <typename V>
using vector_storage_type = vector_storage<vector_value_type<V>, vector_size<V>>;

template <typename... Vs>
using promoted_vector_value_type = promote_t<vector_value_type<Vs>...>;

template <typename V>
KERNEL_FLOAT_INLINE vector_storage_type<V> into_vector_storage(V&& input) {
    return into_vector_impl<V>::call(static_cast<V&&>(input));
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_APPLY_H
#define KERNEL_FLOAT_APPLY_H

namespace kernel_float {
namespace detail {

template <typename... Es>
struct broadcast_extent_impl;

template <typename E>
struct broadcast_extent_impl<E> {
    using type = E;
};

template <size_t N>
struct broadcast_extent_impl<extent<N>, extent<N>> {
    using type = extent<N>;
};

template <size_t N>
struct broadcast_extent_impl<extent<1>, extent<N>> {
    using type = extent<N>;
};

template <size_t N>
struct broadcast_extent_impl<extent<N>, extent<1>> {
    using type = extent<N>;
};

template <>
struct broadcast_extent_impl<extent<1>, extent<1>> {
    using type = extent<1>;
};

template <typename A, typename B, typename C, typename... Rest>
struct broadcast_extent_impl<A, B, C, Rest...> : broadcast_extent_impl<typename broadcast_extent_impl<A, B>::type, C, Rest...> {};

}  // namespace detail

template <typename... Es>
using broadcast_extent = typename detail::broadcast_extent_impl<Es...>::type;

template <typename... Vs>
using broadcast_vector_extent_type = broadcast_extent<vector_extent_type<Vs>...>;

template <typename From, typename To>
static constexpr bool is_broadcastable = is_same_type<broadcast_extent<From, To>, To>;

template <typename V, typename To>
static constexpr bool is_vector_broadcastable = is_broadcastable<vector_extent_type<V>, To>;

namespace detail {

template <typename T, typename From, typename To>
struct broadcast_impl;

template <typename T, size_t N>
struct broadcast_impl<T, extent<1>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(const vector_storage<T, 1>& input) {
        vector_storage<T, N> output;
        for (size_t i = 0; i < N; i++) {
            output.data()[i] = input.data()[0];
        }
        return output;
    }
};

template <typename T, size_t N>
struct broadcast_impl<T, extent<N>, extent<N>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(vector_storage<T, N> input) { return input; }
};

template <typename T>
struct broadcast_impl<T, extent<1>, extent<1>> {
    KERNEL_FLOAT_INLINE static vector_storage<T, 1> call(vector_storage<T, 1> input) { return input; }
};

}  // namespace detail

/**
 * Takes the given vector `input` and extends its size to a length of `N`. This is only valid if the size of `input`
 * is 1 or `N`.
 *
 * Example
 * =======
 * ```
 * vec<float, 1> a = {1.0f};
 * vec<float, 5> x = broadcast<5>(a);  // Returns [1.0f, 1.0f, 1.0f, 1.0f, 1.0f]
 *
 * vec<float, 5> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
 * vec<float, 5> y = broadcast<5>(b);  // Returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f]
 * ```
 */
template <size_t N, typename V>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, extent<N>> broadcast(const V& input, extent<N> new_size = {}) {
    using T = vector_value_type<V>;
    return detail::broadcast_impl<T, vector_extent_type<V>, extent<N>>::call(into_vector_storage(input));
}

/**
 * Takes the given vector `input` and extends its size to the same length as vector `other`. This is only valid if the
 * size of `input` is 1 or the same as `other`.
 */
template <typename V, typename R>
KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<R>> broadcast_like(const V& input, const R& other) {
    return broadcast(input, vector_extent_type<R>{});
}

/**
 * The accurate_policy is designed for computations where maximum accuracy is essential. This policy ensures that all
 * operations are performed without any approximations or optimizations that could potentially alter the precise
 * outcome of the computations
 */
struct accurate_policy {};

/**
 * The fast_policy is intended for scenarios where performance and execution speed are more critical than achieving
 * the utmost accuracy. This policy leverages optimizations to accelerate computations, which may involve
 * approximations that slightly compromise precision.
 */
struct fast_policy {
    using fallback_policy = accurate_policy;
};

/**
 * This template policy allows developers to specify a custom degree of approximation for their computations. By
 * adjusting the `Level` parameter, you can fine-tune the balance between accuracy and performance to meet the
 * specific needs of your application. Higher values mean more precision.
 */
template <int Level = -1>
struct approx_level_policy {
    using fallback_policy = approx_level_policy<>;
};

template <>
struct approx_level_policy<> {
    using fallback_policy = fast_policy;
};

/**
 * The approximate_policy serves as the default approximation policy, providing a standard level of approximation
 * without requiring explicit configuration. It balances accuracy and performance, making it suitable for
 * general-purpose use cases where neither extreme precision nor maximum speed is necessary.
 */
using approx_policy = approx_level_policy<>;

/**
 * The `default_policy` acts as the standard computation policy. It can be configured externally using the
 * `KERNEL_FLOAT_GLOBAL_POLICY` macro. If `KERNEL_FLOAT_GLOBAL_POLICY` is not defined, default to `accurate_policy`.
 */
#if defined(KERNEL_FLOAT_GLOBAL_POLICY)
using default_policy = KERNEL_FLOAT_GLOBAL_POLICY;
#elif defined(KERNEL_FLOAT_POLICY)
using default_policy = KERNEL_FLOAT_POLICY;
#else
using default_policy = accurate_policy;
#endif

namespace detail {

template <typename F, typename Output, typename... Args>
struct invoke_impl {
    KERNEL_FLOAT_INLINE static Output call(F fun, Args... args) { return fun(args...); }
};

template <typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_impl;

template <typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_base_impl : apply_impl<typename Policy::fallback_policy, F, N, Output, Args...> {};

template <typename Policy, typename F, size_t N, typename Output, typename... Args>
struct apply_impl : apply_base_impl<Policy, F, N, Output, Args...> {};

// Only for `accurate_policy` do we implement `apply_impl`, the others will fall back to `apply_base_impl`.
template <typename F, size_t N, typename Output, typename... Args>
struct apply_impl<accurate_policy, F, N, Output, Args...> {
    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
#pragma unroll
        for (size_t i = 0; i < N; i++) {
            output[i] = invoke_impl<F, Output, Args...>::call(fun, args[i]...);
        }
    }
};

template <typename Policy, typename F, size_t N, typename Output, typename... Args>
struct map_impl {
    static constexpr size_t packet_size = preferred_vector_size<Output>::value;
    static constexpr size_t remainder   = N % packet_size;

    KERNEL_FLOAT_INLINE static void call(F fun, Output* output, const Args*... args) {
        if constexpr (N / packet_size > 0) {
#pragma unroll
            for (size_t i = 0; i < N - remainder; i += packet_size) {
                apply_impl<Policy, F, packet_size, Output, Args...>::call(fun, output + i, (args + i)...);
            }
        }

        if constexpr (remainder > 0) {
#pragma unroll
            for (size_t i = N - remainder; i < N; i++) {
                apply_impl<Policy, F, 1, Output, Args...>::call(fun, output + i, (args + i)...);
            }
        }
    }
};

template <typename F, size_t N, typename Output, typename... Args>
using default_map_impl = map_impl<default_policy, F, N, Output, Args...>;

}  // namespace detail

template <typename F, typename... Args>
using map_type = vector<result_t<F, vector_value_type<Args>...>, broadcast_vector_extent_type<Args...>>;

/**
 * Apply the function `F` to each element from the vector `input` and return the results as a new vector.
 *
 * Examples
 * ========
 * ```
 * vec<float, 4> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * vec<float, 4> squared = map([](auto x) { return x * x; }, input); // [1.0f, 4.0f, 9.0f, 16.0f]
 * ```
 */
template <typename Accuracy = default_policy, typename F, typename... Args>
KERNEL_FLOAT_INLINE map_type<F, Args...> map(F fun, const Args&... args) {
    using Output = result_t<F, vector_value_type<Args>...>;
    using E      = broadcast_vector_extent_type<Args...>;
    vector_storage<Output, extent_size<E>> result;

    detail::map_impl<Accuracy, F, extent_size<E>, Output, vector_value_type<Args>...>::call(
        fun, result.data(),
        (detail::broadcast_impl<vector_value_type<Args>, vector_extent_type<Args>, E>::call(into_vector_storage(args)).data())...);

    return result;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_APPLY_H
#ifndef KERNEL_FLOAT_COMPLEX_TYPE_H
#define KERNEL_FLOAT_COMPLEX_TYPE_H

namespace kernel_float {

template <typename T>
struct alignas(2 * alignof(T)) complex_type_storage {
    T re;
    T im;
};

template <typename T>
struct complex_type : complex_type_storage<T> {
    using base_type = complex_type_storage<T>;

    template <typename T2>
    KERNEL_FLOAT_INLINE complex_type(complex_type<T2> that) : base_type(that.real(), that.imag()) {}

    KERNEL_FLOAT_INLINE
    complex_type(T real = {}, T imag = {}) : base_type(real, imag) {}

    KERNEL_FLOAT_INLINE
    T real() const { return this->re; }

    KERNEL_FLOAT_INLINE
    T imag() const { return this->im; }

    KERNEL_FLOAT_INLINE
    T norm() const { return real() * real() + imag() * imag(); }

    KERNEL_FLOAT_INLINE
    complex_type conj() const { return {real(), -imag()}; }
};

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> v) {
    return v;
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, complex_type<T> b) {
    return {a.real() + b.real(), a.imag() + b.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(T a, complex_type<T> b) {
    return {a + b.real(), b.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator+(complex_type<T> a, T b) {
    return {a.real() + b, a.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, complex_type<T> b) {
    return (a = a + b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator+=(complex_type<T>& a, T b) {
    return (a = a + b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> v) {
    return {-v.real(), -v.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, complex_type<T> b) {
    return {a.real() - b.real(), a.imag() - b.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(T a, complex_type<T> b) {
    return {a - b.real(), -b.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator-(complex_type<T> a, T b) {
    return {a.real() - b, a.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, complex_type<T> b) {
    return (a = a - b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator-=(complex_type<T>& a, T b) {
    return (a = a - b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, complex_type<T> b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(complex_type<T> a, T b) {
    return {a.real() * b, a.imag() * b};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator*=(complex_type<T>& a, complex_type<T> b) {
    return (a = a * b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator*=(complex_type<T>& a, T b) {
    return (a = a * b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator*(T a, complex_type<T> b) {
    return {a * b.real(), a * b.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, complex_type<T> b) {
    T normi = T(1) / b.norm();

    return {(a.real() * b.real() + a.imag() * b.imag()) * normi, (a.imag() * b.real() - a.real() * b.imag()) * normi};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(complex_type<T> a, T b) {
    return a * (T(1) / b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> operator/(T a, complex_type<T> b) {
    T normi = T(1) / b.norm();

    return {a * b.real() * normi, -a * b.imag() * normi};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>* operator/=(complex_type<T>& a, complex_type<T> b) {
    return (a = a / b);
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T>& operator/=(complex_type<T>& a, T b) {
    return (a = a / b);
}

template <typename T>
KERNEL_FLOAT_INLINE T real(complex_type<T> v) {
    return v.real();
}

template <typename T>
KERNEL_FLOAT_INLINE T imag(complex_type<T> v) {
    return v.imag();
}

template <typename T>
KERNEL_FLOAT_INLINE T abs(complex_type<T> v) {
    return hypot(v.real(), v.imag());
}

template <typename T>
KERNEL_FLOAT_INLINE T arg(complex_type<T> v) {
    return atan2(v.imag(), v.real());
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> sqrt(complex_type<T> v) {
    T radius = abs(v);
    T cosA   = v.real() / radius;

    complex_type<T> out = {sqrt(radius * (cosA + T(1)) * T(.5)), sqrt(radius * (T(1) - cosA) * T(.5))};

    // signbit should be false if x.y is negative
    if (v.imag() < 0) {
        out = complex_type<T>{out.real, -out.im};
    }

    return out;
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> norm(complex_type<T> v) {
    return v.real() * v.real() + v.imag() * v.imag();
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> conj(complex_type<T> v) {
    return {v.real(), -v.imag()};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> exp(complex_type<T> v) {
    // TODO: Handle nan and inf correctly
    T e = exp(v.real());
    T a = v.imag();
    return complex_type<T>(e * cos(a), e * sin(a));
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> log(complex_type<T> v) {
    return {log(abs(v)), arg(v)};
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, T b) {
    return exp(a * log(b));
}

template <typename T>
KERNEL_FLOAT_INLINE complex_type<T> pow(complex_type<T> a, complex_type<T> b) {
    return exp(a * log(b));
}

template <typename L, typename R>
struct promote_type<complex_type<L>, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

template <typename L, typename R>
struct promote_type<complex_type<L>, R> {
    using type = complex_type<promote_t<L, R>>;
};

template <typename L, typename R>
struct promote_type<L, complex_type<R>> {
    using type = complex_type<promote_t<L, R>>;
};

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_UNOPS_H
#define KERNEL_FLOAT_UNOPS_H

namespace kernel_float {

enum struct RoundingMode { ANY, DOWN, UP, NEAREST, TOWARD_ZERO };

namespace ops {

template <typename T, typename R, RoundingMode m = RoundingMode::ANY, typename = void>
struct cast;

template <typename T, RoundingMode m>
struct cast<T, T, m> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept { return input; }
};

template <typename T>
struct cast<T, T> {
    KERNEL_FLOAT_INLINE T operator()(T input) noexcept { return input; }
};

template <typename T, typename R>
struct cast<T, R> {
    KERNEL_FLOAT_INLINE R operator()(T input) noexcept {
        if constexpr (detail::allow_float_fallback<T>::value || detail::allow_float_fallback<R>::value) {
            return cast<float, R>{}(cast<T, float>{}(input));
        } else {
            return R(input);
        }
    }
};

template <>
struct cast<float, float> {
    KERNEL_FLOAT_INLINE float operator()(float input) noexcept { return input; }
};

template <RoundingMode m>
struct cast<float, float, m> {
    KERNEL_FLOAT_INLINE float operator()(float input) noexcept { return input; }
};

template <typename T>
struct cast<T, float> {
    KERNEL_FLOAT_INLINE float operator()(T input) noexcept { return float(input); }
};

template <typename T>
struct cast<float, T> {
    KERNEL_FLOAT_INLINE T operator()(float input) noexcept { return T(input); }
};

}  // namespace ops

/**
 * Cast the elements of the given vector `input` to a different type `R`.
 *
 * This function casts each element of the input vector to a different data type specified by
 * template parameter `R`.
 *
 * Optionally, the rounding mode can be set using the `Mode` template parameter. The default mode is `ANY`, which
 * uses the fastest rounding mode available.
 *
 * Example
 * =======
 * ```
 * vec<float, 4> input {1.2f, 2.7f, 3.5f, 4.9f};
 * auto casted = cast<int>(input); // [1, 2, 3, 4]
 * ```
 */
template <typename R, RoundingMode Mode = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, vector_extent_type<V>> cast(const V& input) {
    using F = ops::cast<vector_value_type<V>, R, Mode>;
    return map(F{}, input);
}

#define KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)                                                        \
    template <typename Accuracy = default_policy, typename V>                                      \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> NAME(const V& input) { \
        using F = ops::NAME<vector_value_type<V>>;                                                 \
        return ::kernel_float::map<Accuracy>(F{}, input);                                          \
    }

#define KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                         \
    namespace ops {                                                   \
    template <typename T>                                             \
    struct NAME {                                                     \
        KERNEL_FLOAT_INLINE T operator()(T input) { return T(EXPR); } \
    };                                                                \
    }                                                                 \
                                                                        \
    KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)

#define KERNEL_FLOAT_DEFINE_UNARY_OP(NAME, OP, EXPR)                           \
    KERNEL_FLOAT_DEFINE_UNARY(NAME, EXPR)                                      \
                                                                                \
    template <typename T, typename E, typename S>                              \
    KERNEL_FLOAT_INLINE vector<T, E> operator OP(const vector<T, E, S>& vec) { \
        return NAME(vec);                                                      \
    }

KERNEL_FLOAT_DEFINE_UNARY_OP(negate, -, -input)
KERNEL_FLOAT_DEFINE_UNARY_OP(bit_not, ~, ~input)
KERNEL_FLOAT_DEFINE_UNARY_OP(logical_not, !, (ops::cast<bool, T>{}(!ops::cast<T, bool>{}(input))))

#define KERNEL_FLOAT_DEFINE_UNARY_STRUCT(NAME, EXPR_F64, EXPR_F32)                       \
    namespace ops {                                                                      \
    template <typename T, typename = void>                                               \
    struct NAME;                                                                         \
                                                                                            \
    template <typename T>                                                                \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> {                \
        KERNEL_FLOAT_INLINE T operator()(T input_arg) {                                  \
            float input = ops::cast<T, float>{}(input_arg);                              \
            return ops::cast<decltype(EXPR_F32), T>{}(EXPR_F32);                         \
        }                                                                                \
    };                                                                                   \
                                                                                            \
    template <>                                                                          \
    struct NAME<double> {                                                                \
        KERNEL_FLOAT_INLINE double operator()(double input) { return double(EXPR_F64); } \
    };                                                                                   \
    }

#define KERNEL_FLOAT_DEFINE_UNARY_MATH(NAME)                             \
    KERNEL_FLOAT_DEFINE_UNARY_STRUCT(NAME, ::NAME(input), ::NAME(input)) \
    KERNEL_FLOAT_DEFINE_UNARY_FUN(NAME)

KERNEL_FLOAT_DEFINE_UNARY_MATH(sin)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cos)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tan)
KERNEL_FLOAT_DEFINE_UNARY_MATH(asin)
KERNEL_FLOAT_DEFINE_UNARY_MATH(acos)
KERNEL_FLOAT_DEFINE_UNARY_MATH(atan)

KERNEL_FLOAT_DEFINE_UNARY_MATH(sinh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cosh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tanh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(acosh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(asinh)
KERNEL_FLOAT_DEFINE_UNARY_MATH(atanh)

KERNEL_FLOAT_DEFINE_UNARY_MATH(exp)
KERNEL_FLOAT_DEFINE_UNARY_MATH(exp2)
KERNEL_FLOAT_DEFINE_UNARY_MATH(exp10)
KERNEL_FLOAT_DEFINE_UNARY_MATH(expm1)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log2)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log10)
KERNEL_FLOAT_DEFINE_UNARY_MATH(log1p)

KERNEL_FLOAT_DEFINE_UNARY_MATH(erf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfinv)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfc)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfcx)
KERNEL_FLOAT_DEFINE_UNARY_MATH(erfcinv)
KERNEL_FLOAT_DEFINE_UNARY_MATH(normcdf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(lgamma)
KERNEL_FLOAT_DEFINE_UNARY_MATH(tgamma)

KERNEL_FLOAT_DEFINE_UNARY_MATH(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rsqrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(cbrt)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rcbrt)

KERNEL_FLOAT_DEFINE_UNARY_MATH(abs)
KERNEL_FLOAT_DEFINE_UNARY_MATH(floor)
KERNEL_FLOAT_DEFINE_UNARY_MATH(round)
KERNEL_FLOAT_DEFINE_UNARY_MATH(ceil)
KERNEL_FLOAT_DEFINE_UNARY_MATH(trunc)
KERNEL_FLOAT_DEFINE_UNARY_MATH(rint)

// There are not support on HIP
#if !KERNEL_FLOAT_IS_HIP
KERNEL_FLOAT_DEFINE_UNARY_MATH(isnan)
KERNEL_FLOAT_DEFINE_UNARY_MATH(isinf)
KERNEL_FLOAT_DEFINE_UNARY_MATH(isfinite)
#endif

// CUDA offers special reciprocal functions (rcp), but only on the device.
#if KERNEL_FLOAT_IS_DEVICE
KERNEL_FLOAT_DEFINE_UNARY_STRUCT(rcp, __drcp_rn(input), __frcp_rn(input))
#else
KERNEL_FLOAT_DEFINE_UNARY_STRUCT(rcp, 1.0 / input, 1.0f / input)
#endif

KERNEL_FLOAT_DEFINE_UNARY_FUN(rcp)

#define KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(NAME)                                                          \
    template <typename V>                                                                                 \
    KERNEL_FLOAT_INLINE vector<vector_value_type<V>, vector_extent_type<V>> fast_##NAME(const V& input) { \
        return ::kernel_float::map<fast_policy>(ops::NAME<vector_value_type<V>>{}, input);                \
    }

KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(sin)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(cos)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(tan)

KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(exp)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(log)

KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(sqrt)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(rcp)
KERNEL_FLOAT_DEFINE_UNARY_FUN_FAST(rsqrt)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(T, F, EXPR_F32)                       \
    namespace detail {                                                                \
    template <>                                                                       \
    struct apply_impl<fast_policy, ops::F<T>, 1, T, T> {                              \
        KERNEL_FLOAT_INLINE static void call(ops::F<T>, T* result, const T* inputs) { \
            T input = inputs[0];                                                      \
            *result = EXPR_F32;                                                       \
        }                                                                             \
    };                                                                                \
    }

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, exp, __expf(input))
// KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, exp2, __exp2f(input)) // Seems to be missing?
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, exp10, __exp10f(input))

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, log, __logf(input))
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, log2, __log2f(input))
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, log10, __log10f(input))

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, sin, __sinf(input))
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, cos, __cosf(input))
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN(float, tan, __tanf(input))

// This PTX is only supported on CUDA
#if KERNEL_FLOAT_IS_CUDA
#define KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(T, F, INSTR, REG)                         \
    namespace detail {                                                                    \
    template <>                                                                           \
    struct apply_impl<fast_policy, ops::F<T>, 1, T, T> {                                  \
        KERNEL_FLOAT_INLINE static void call(ops::F<T> fun, T* result, const T* inputs) { \
            asm(INSTR " %0, %1;" : "=" REG(*result) : REG(*inputs));                      \
        }                                                                                 \
    };                                                                                    \
    }

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(double, rcp, "rcp.approx.ftz.f64", "d")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(double, rsqrt, "rsqrt.approx.ftz.f64", "d")

KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, exp2, "ex2.approx.f32", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, sqrt, "sqrt.approx.f32", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, rcp, "rcp.approx.f32", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, rsqrt, "rsqrt.approx.f32", "f")
KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, tanh, "tanh.approx.f32", "f")

// These are no longer necessary due to the KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_FUN above
// KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, sin, "sin.approx.f32", "f")
// KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, cos, "cos.approx.f32", "f")
// KERNEL_FLOAT_DEFINE_UNARY_FAST_IMPL_PTX(float, log2, "lg2.approx.f32", "f")
#endif

#define KERNEL_FLOAT_FAST_F32_MAP(F) F(exp) F(exp2) F(exp10) F(log) F(log2) F(log10) F(sin) F(cos) F(tan) F(rcp) F(rsqrt) F(sqrt)

#else
#define KERNEL_FLOAT_FAST_F32_MAP(F)
#endif

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_UNOPS_H
#ifndef KERNEL_FLOAT_CAST_H
#define KERNEL_FLOAT_CAST_H

namespace kernel_float {

namespace detail {
/**
 * Convert vector of element type `T` and extent type `E` to vector of element type `T2` and extent type `E2`.
 *  Specialization exist for the cases where `T==T2` and/or `E==E2`.
 */
template <typename T, typename E, typename T2, typename E2, RoundingMode M = RoundingMode::ANY>
struct convert_impl {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, extent_size<E2>> call(vector_storage<T, extent_size<E>> input) {
        using F = ops::cast<T, T2, M>;
        vector_storage<T2, extent_size<E>> intermediate;
        detail::default_map_impl<F, extent_size<E>, T2, T>::call(F{}, intermediate.data(), input.data());
        return detail::broadcast_impl<T2, E, E2>::call(intermediate);
    }
};

// T == T2, E == E2
template <typename T, typename E, RoundingMode M>
struct convert_impl<T, E, T, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, extent_size<E>> call(vector_storage<T, extent_size<E>> input) { return input; }
};

// T == T2, E != E2
template <typename T, typename E, typename E2, RoundingMode M>
struct convert_impl<T, E, T, E2, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, extent_size<E2>> call(vector_storage<T, extent_size<E>> input) {
        return detail::broadcast_impl<T, E, E2>::call(input);
    }
};

// T != T2, E == E2
template <typename T, typename E, typename T2, RoundingMode M>
struct convert_impl<T, E, T2, E, M> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T2, extent_size<E>> call(vector_storage<T, extent_size<E>> input) {
        using F = ops::cast<T, T2, M>;

        vector_storage<T2, extent_size<E>> result;
        detail::default_map_impl<F, extent_size<E>, T2, T>::call(F{}, result.data(), input.data());
        return result;
    }
};
}  // namespace detail

template <typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector_storage<R, N> convert_storage(const V& input, extent<N> new_size = {}) {
    return detail::convert_impl<vector_value_type<V>, vector_extent_type<V>, R, extent<N>, M>::call(into_vector_storage(input));
}

/**
 * Cast the values of the given input vector to type `R` and then broadcast the result to the given size `N`.
 *
 * Example
 * =======
 * ```
 * int a = 5;
 * vec<float, 3> x = convert<float, 3>(a);  // returns [5.0f, 5.0f, 5.0f]
 *
 * float b = 5.0f;
 * vec<float, 3> x = convert<float, 3>(b);  // returns [5.0f, 5.0f, 5.0f]
 *
 * vec<int, 3> c = {1, 2, 3};
 * vec<float, 3> x = convert<float, 3>(c);  // returns [1.0f, 2.0f, 3.0f]
 * ```
 */
template <typename R, size_t N, RoundingMode M = RoundingMode::ANY, typename V>
KERNEL_FLOAT_INLINE vector<R, extent<N>> convert(const V& input, extent<N> new_size = {}) {
    return convert_storage<R, N, M, V>(input, new_size);
}

template <typename T, RoundingMode M = RoundingMode::ANY>
struct AssignConversionProxy {
    KERNEL_FLOAT_INLINE
    explicit AssignConversionProxy(T&& ptr) : ptr_(std::forward<T>(ptr)) {}

    template <typename U>
    KERNEL_FLOAT_INLINE AssignConversionProxy& operator=(U&& values) {
        ptr_ = detail::convert_impl<vector_value_type<U>, vector_extent_type<U>, vector_value_type<T>, vector_extent_type<T>, M>::call(
            into_vector_storage(values));

        return *this;
    }

    private:
    T ptr_;
};

/**
 * Takes a vector reference and gives back a helper object. This object allows you to assign
 * a vector of a different type to another vector while performing implicit type conversion.
 *
 * For example, if `x = expression;` does not compile because `x` and `expression` are
 * different vector types, you can use `cast_to(x) = expression;` to make it work.
 *
 * Example
 * =======
 * ```
 * vec<float, 2> x;
 * vec<double, 2> y = {1.0, 2.0};
 * cast_to(x) = y;  // Normally, `x = y;` would give an error, but `cast_to` fixes that.
 * ```
 */
template <RoundingMode M = RoundingMode::ANY, typename T>
KERNEL_FLOAT_INLINE AssignConversionProxy<T, M> cast_to(T&& input) {
    return AssignConversionProxy<T, M>(std::forward<T>(input));
}

/**
 * Returns a vector containing `N` copies of `value`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = fill<3>(42); // returns [42, 42, 42]
 * ```
 */
template <size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> fill(T value, extent<N> = {}) {
    vector_storage<T, 1> input = {value};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector containing `N` copies of `T(0)`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = zeros<int, 3>(); // returns [0, 0, 0]
 * ```
 */
template <typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> zeros(extent<N> = {}) {
    vector_storage<T, 1> input = {T{}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector containing `N` copies of `T(1)`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = ones<int, 3>(); // returns [1, 1, 1]
 * ```
 */
template <typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> ones(extent<N> = {}) {
    vector_storage<T, 1> input = {T{1}};
    return detail::broadcast_impl<T, extent<1>, extent<N>>::call(input);
}

/**
 * Returns a vector filled with `value` having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = fill_like(a, 42); // returns [42, 42, 42]
 * ```
 */
template <typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> fill_like(const V&, T value) {
    return fill(value, E{});
}

/**
 * Returns a vector filled with zeros having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = zeros_like(a); // returns [0, 0, 0]
 * ```
 */
template <typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> zeros_like(const V& = {}) {
    return zeros<T>(E{});
}

/**
 * Returns a vector filled with ones having the same type and size as input vector `V`.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> a = {1, 2, 3};
 * vec<int, 3> b = ones_like(a); // returns [1, 1, 1]
 * ```
 */
template <typename V, typename T = vector_value_type<V>, typename E = vector_extent_type<V>>
KERNEL_FLOAT_INLINE vector<T, E> ones_like(const V& = {}) {
    return ones<T>(E{});
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_BINOPS_H
#define KERNEL_FLOAT_BINOPS_H

namespace kernel_float {

template <typename F, typename L, typename R>
using zip_type = map_type<F, L, R>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements.
 *
 * Example
 * =======
 * ```
 * vec<bool, 3> make_negative = {true, false, true};
 * vec<int, 3> input = {1, 2, 3};
 * vec<int, 3> output = zip([](bool b, int n){ return b ? -n : +n; }, make_negative, input); // returns [-1, 2, -3]
 * ```
 */
template <typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_type<F, L, R> zip(F fun, const L& left, const R& right) {
    return ::kernel_float::map(fun, left, right);
}

template <typename F, typename L, typename R>
using zip_common_type =
    vector<result_t<F, promoted_vector_value_type<L, R>, promoted_vector_value_type<L, R>>, broadcast_vector_extent_type<L, R>>;

/**
 * Combines the elements from the two inputs (`left` and `right`)  element-wise, applying a provided binary
 * function (`fun`) to each pair of corresponding elements. The elements are promoted to a common type before applying
 * the binary function.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> a = {1.0f, 2.0f, 3.0f};
 * vec<int, 3> b = {4, 5, 6};
 * vec<float, 3> c = zip_common([](float x, float y){ return x + y; }, a, b); // returns [5.0f, 7.0f, 9.0f]
 * ```
 */
template <typename Accuracy = default_policy, typename F, typename L, typename R>
KERNEL_FLOAT_INLINE zip_common_type<F, L, R> zip_common(F fun, const L& left, const R& right) {
    using T = promoted_vector_value_type<L, R>;
    using O = result_t<F, T, T>;
    using E = broadcast_vector_extent_type<L, R>;

    vector_storage<O, extent_size<E>> result;

    detail::map_impl<Accuracy, F, extent_size<E>, O, T, T>::call(
        fun, result.data(), detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(into_vector_storage(left)).data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(into_vector_storage(right)).data());

    return result;
}

#define KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)                                                                             \
    template <typename Accuracy = default_policy, typename L, typename R, typename C = promoted_vector_value_type<L, R>> \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, R> NAME(L&& left, R&& right) {                                  \
        return zip_common<Accuracy>(ops::NAME<C>{}, static_cast<L&&>(left), static_cast<R&&>(right));                    \
    }

#define KERNEL_FLOAT_DEFINE_BINARY(NAME, EXPR, EXPR_F64, EXPR_F32)                                                                     \
    namespace ops {                                                                                                                    \
    template <typename T, typename = void>                                                                                             \
    struct NAME {                                                                                                                      \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) { return ops::cast<decltype(EXPR), T>{}(EXPR); }                             \
    };                                                                                                                                 \
                                                                                                                                        \
    template <>                                                                                                                        \
    struct NAME<double> {                                                                                                              \
        KERNEL_FLOAT_INLINE double operator()(double left, double right) { return ops::cast<decltype(EXPR_F64), double>{}(EXPR_F64); } \
    };                                                                                                                                 \
                                                                                                                                        \
    template <typename T>                                                                                                              \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> {                                                              \
        KERNEL_FLOAT_INLINE T operator()(T left_, T right_) {                                                                          \
            float left  = ops::cast<T, float>{}(left_);                                                                                \
            float right = ops::cast<T, float>{}(right_);                                                                               \
            return ops::cast<decltype(EXPR_F32), T>{}(EXPR_F32);                                                                       \
        }                                                                                                                              \
    };                                                                                                                                 \
    }                                                                                                                                  \
                                                                                                                                        \
    KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)

#define KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(NAME, OP, EXPR_F64, EXPR_F32)                                                   \
    KERNEL_FLOAT_DEFINE_BINARY(NAME, left OP right, EXPR_F64, EXPR_F32)                                                        \
                                                                                                                                \
    template <typename L, typename R, typename C = promote_t<L, R>, typename E1, typename E2>                                  \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E1>, vector<R, E2>> operator OP(const vector<L, E1>& left,     \
                                                                                                const vector<R, E2>& right) {  \
        return zip_common(ops::NAME<C>{}, left, right);                                                                        \
    }                                                                                                                          \
    template <typename L, typename R, typename C = promote_t<L, vector_value_type<R>>, typename E>                             \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, vector<L, E>, R> operator OP(const vector<L, E>& left, const R& right) { \
        return zip_common(ops::NAME<C>{}, left, right);                                                                        \
    }                                                                                                                          \
    template <typename L, typename R, typename C = promote_t<vector_value_type<L>, R>, typename E>                             \
    KERNEL_FLOAT_INLINE zip_common_type<ops::NAME<C>, L, vector<R, E>> operator OP(const L& left, const vector<R, E>& right) { \
        return zip_common(ops::NAME<C>{}, left, right);                                                                        \
    }

#define KERNEL_FLOAT_DEFINE_BINARY_OP(NAME, OP) KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(NAME, OP, left OP right, left OP right)

KERNEL_FLOAT_DEFINE_BINARY_OP(add, +)
KERNEL_FLOAT_DEFINE_BINARY_OP(subtract, -)
KERNEL_FLOAT_DEFINE_BINARY_OP(divide, /)
KERNEL_FLOAT_DEFINE_BINARY_OP(multiply, *)
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(modulo, %, ::fmod(left, right), ::fmodf(left, right))

KERNEL_FLOAT_DEFINE_BINARY_OP(equal_to, ==)
KERNEL_FLOAT_DEFINE_BINARY_OP(not_equal_to, !=)
KERNEL_FLOAT_DEFINE_BINARY_OP(less, <)
KERNEL_FLOAT_DEFINE_BINARY_OP(less_equal, <=)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater, >)
KERNEL_FLOAT_DEFINE_BINARY_OP(greater_equal, >=)

// clang-format off
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_and, &, bool(left) && bool(right), bool(left) && bool(right))
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_or, |, bool(left) | bool(right), bool(left) | bool(right))
KERNEL_FLOAT_DEFINE_BINARY_OP_FALLBACK(bit_xor, ^, bool(left) ^ bool(right), bool(left) ^ bool(right))
// clang-format on

// clang-format off
template<template<typename, typename=void> typename F, typename T, typename E, typename R>
static constexpr bool is_vector_assign_allowed =
is_vector_broadcastable<R, E> &&
is_implicit_convertible<
result_t<
F<promote_t<T, vector_value_type<R>>>,
    T,
    vector_value_type<R>
>,
T
>;
// clang-format on

#define KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(NAME, OP)                                                                  \
    template <typename T, typename E, typename R, typename = enable_if_t<is_vector_assign_allowed<ops::NAME, T, E, R>>> \
    KERNEL_FLOAT_INLINE vector<T, E>& operator OP(vector<T, E>& lhs, const R& rhs) {                                    \
        using F = ops::NAME<T>;                                                                                         \
        lhs     = zip_common(F{}, lhs, rhs);                                                                            \
        return lhs;                                                                                                     \
    }

KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(add, +=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(subtract, -=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(divide, /=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(multiply, *=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(modulo, %=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_and, &=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_or, |=)
KERNEL_FLOAT_DEFINE_BINARY_ASSIGN_OP(bit_xor, ^=)

namespace ops {
template <typename T>
struct min {
    KERNEL_FLOAT_INLINE T operator()(T left, T right) {
        auto cond = less<T>{}(left, right);
        return cast<decltype(cond), bool>{}(cond) ? left : right;
    }
};

template <typename T>
struct max {
    KERNEL_FLOAT_INLINE T operator()(T left, T right) {
        auto cond = greater<T>{}(left, right);
        return cast<decltype(cond), bool>{}(cond) ? left : right;
    }
};

template <>
struct min<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) { return ::fmin(left, right); }
};

template <>
struct max<double> {
    KERNEL_FLOAT_INLINE double operator()(double left, double right) { return ::fmax(left, right); }
};

template <>
struct min<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) { return ::fminf(left, right); }
};

template <>
struct max<float> {
    KERNEL_FLOAT_INLINE float operator()(float left, float right) { return ::fmaxf(left, right); }
};
}  // namespace ops

KERNEL_FLOAT_DEFINE_BINARY_FUN(min)
KERNEL_FLOAT_DEFINE_BINARY_FUN(max)

#define KERNEL_FLOAT_DEFINE_BINARY_MATH(NAME)                                                                    \
    namespace ops {                                                                                              \
    template <typename T, typename = void>                                                                       \
    struct NAME;                                                                                                 \
                                                                                                                    \
    template <typename T>                                                                                        \
    struct NAME<T, enable_if_t<detail::allow_float_fallback<T>::value>> {                                        \
        KERNEL_FLOAT_INLINE T operator()(T left, T right) { return T(::NAME(left, right)); }                     \
    };                                                                                                           \
                                                                                                                    \
    template <>                                                                                                  \
    struct NAME<double> {                                                                                        \
        KERNEL_FLOAT_INLINE double operator()(double left, double right) { return double(::NAME(left, right)); } \
    };                                                                                                           \
    }                                                                                                            \
                                                                                                                    \
    KERNEL_FLOAT_DEFINE_BINARY_FUN(NAME)

KERNEL_FLOAT_DEFINE_BINARY_MATH(copysign)
KERNEL_FLOAT_DEFINE_BINARY_MATH(fmod)
KERNEL_FLOAT_DEFINE_BINARY_MATH(nextafter)
KERNEL_FLOAT_DEFINE_BINARY_MATH(pow)
KERNEL_FLOAT_DEFINE_BINARY_MATH(remainder)

KERNEL_FLOAT_DEFINE_BINARY(hypot, ops::sqrt<T>()(left* left + right * right), ::hypot(left, right), ::hypotf(left, right))

#if KERNEL_FLOAT_IS_DEVICE
KERNEL_FLOAT_DEFINE_BINARY(rhypot, (ops::rcp<T>(ops::hypot<T>()(left, right))), ::rhypot(left, right), ::rhypotf(left, right))
#else
KERNEL_FLOAT_DEFINE_BINARY(rhypot, (ops::rcp<T>(ops::hypot<T>()(left, right))), (double(1) / ::hypot(left, right)), (float(1) / ::hypotf(left, right)))
#endif

namespace ops {
template <>
struct add<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) { return left || right; }
};

template <>
struct multiply<bool> {
    KERNEL_FLOAT_INLINE bool operator()(bool left, bool right) { return left && right; }
};
};  // namespace ops

namespace detail {
template <typename Policy, typename T, size_t N>
struct apply_base_impl<Policy, ops::divide<T>, N, T, T, T> {
    KERNEL_FLOAT_INLINE static void call(ops::divide<T>, T* result, const T* lhs, const T* rhs) {
        T rhs_rcp[N];

        // Fast way to perform division is to multiply by the reciprocal
        apply_impl<Policy, ops::rcp<T>, N, T, T>::call({}, rhs_rcp, rhs);
        apply_impl<Policy, ops::multiply<T>, N, T, T, T>::call({}, result, lhs, rhs_rcp);
    }
};

#if KERNEL_FLOAT_IS_DEVICE
template <>
struct apply_impl<fast_policy, ops::divide<float>, 1, float, float, float> {
    KERNEL_FLOAT_INLINE static void call(ops::divide<float>, float* result, const float* lhs, const float* rhs) {
        *result = __fdividef(*lhs, *rhs);
    }
};
#endif
}  // namespace detail

namespace detail {
// Override `pow` using `log2` and `exp2`
template <typename Policy, typename T, size_t N>
struct apply_base_impl<Policy, ops::pow<T>, N, T, T, T> {
    KERNEL_FLOAT_INLINE static void call(ops::pow<T>, T* result, const T* lhs, const T* rhs) {
        T lhs_log[N];
        T result_log[N];

        // Fast way to perform power function is using log2 and exp2
        apply_impl<Policy, ops::log2<T>, N, T, T>::call({}, lhs_log, lhs);
        apply_impl<Policy, ops::multiply<T>, N, T, T, T>::call({}, result_log, lhs_log, rhs);
        apply_impl<Policy, ops::exp2<T>, N, T, T, T>::call({}, result, result_log);
    }
};
}  // namespace detail

template <typename L, typename R, typename T = promoted_vector_value_type<L, R>>
KERNEL_FLOAT_INLINE zip_common_type<ops::divide<T>, T, T> fast_divide(const L& left, const R& right) {
    using E = broadcast_vector_extent_type<L, R>;
    vector_storage<T, extent_size<E>> result;

    detail::map_impl<fast_policy, ops::divide<T>, extent_size<E>, T, T, T>::call(
        ops::divide<T>{}, result.data(),
        detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(into_vector_storage(left)).data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(into_vector_storage(right)).data());

    return result;
}

namespace detail {
template <typename T>
struct cross_impl {
    KERNEL_FLOAT_INLINE
    static vector<T, extent<3>> call(const vector_storage<T, 3>& av, const vector_storage<T, 3>& bv) {
        auto a                  = av.data();
        auto b                  = bv.data();
        vector<T, extent<6>> v0 = {a[1], a[2], a[0], a[2], a[0], a[1]};
        vector<T, extent<6>> v1 = {b[2], b[0], b[1], b[1], b[2], b[0]};
        vector<T, extent<6>> rv = v0 * v1;

        auto r                  = rv.data();
        vector<T, extent<3>> r0 = {r[0], r[1], r[2]};
        vector<T, extent<3>> r1 = {r[3], r[4], r[5]};
        return r0 - r1;
    }
};
};  // namespace detail

/**
 * Calculates the cross-product between two vectors of length 3.
 */
template <typename L, typename R, typename T = promoted_vector_value_type<L, R>,
            typename = enable_if_t<(vector_size<L> == 3 && vector_size<R> == 3)>>
KERNEL_FLOAT_INLINE vector<T, extent<3>> cross(const L& left, const R& right) {
    return detail::cross_impl<T>::call(convert_storage<T, 3>(left), convert_storage<T, 3>(right));
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_CONSTANT
#define KERNEL_FLOAT_CONSTANT

namespace kernel_float {

/**
 * `constant<T>` represents a constant value of type `T`.
 *
 * The object has the property that for any binary operation involving
 * a `constant<T>` and a value of type `U`, the constant is automatically
 * cast to also be of type `U`.
 *
 * For example:
 * ```
 * float a = 5;
 * constant<double> b = 3;
 *
 * auto c = a + b; // The result will be of type `float`
 * ```
 */
template <typename T = double>
struct constant {
    /**
     * Create a new constant from the given value.
     */
    KERNEL_FLOAT_INLINE
    constexpr constant(T value = {}) : value_(value) {}

    KERNEL_FLOAT_INLINE
    constexpr constant(const constant<T>& that) : value_(that.value_) {}

    /**
     * Create a new constant from another constant of type `R`.
     */
    template <typename R>
    KERNEL_FLOAT_INLINE explicit constexpr constant(const constant<R>& that) {
        auto f = ops::cast<R, T>();
        value_ = f(that.get());
    }

    /**
     * Return the value of the constant
     */
    KERNEL_FLOAT_INLINE
    constexpr T get() const { return value_; }

    KERNEL_FLOAT_INLINE
    constexpr operator T() const { return value_; }

    private:
    T value_;
};

// Deduction guide for `constant<T>`
#if defined(__cpp_deduction_guides)
template <typename T>
constant(T&&) -> constant<decay_t<T>>;
#endif

template <typename T = double>
KERNEL_FLOAT_INLINE constexpr constant<T> make_constant(T value) {
    return value;
}

template <typename L, typename R>
struct promote_type<constant<L>, constant<R>> {
    using type = constant<typename promote_type<L, R>::type>;
};

template <typename L, typename R>
struct promote_type<constant<L>, R> {
    using type = typename promote_type<L, R>::type;
};

template <typename L, typename R>
struct promote_type<L, constant<R>> {
    using type = typename promote_type<L, R>::type;
};

namespace ops {
template <typename T, typename R>
struct cast<constant<T>, R> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept { return cast<T, R>{}(input); }
};

template <typename T, typename R, RoundingMode m>
struct cast<constant<T>, R, m> {
    KERNEL_FLOAT_INLINE R operator()(const T& input) noexcept { return cast<T, R, m>{}(input); }
};
}  // namespace ops

#define KERNEL_FLOAT_CONSTANT_DEFINE_OP(OP)                                                          \
    template <typename L, typename R>                                                                \
    KERNEL_FLOAT_INLINE auto operator OP(const constant<L>& left, const R& right) {                  \
        auto f = ops::cast<L, vector_value_type<R>>();                                               \
        return f(left.get()) OP right;                                                               \
    }                                                                                                \
                                                                                                        \
    template <typename L, typename R>                                                                \
    KERNEL_FLOAT_INLINE auto operator OP(const L& left, const constant<R>& right) {                  \
        auto f = ops::cast<R, vector_value_type<L>>();                                               \
        return left OP f(right.get());                                                               \
    }                                                                                                \
                                                                                                        \
    template <typename L, typename R, typename E>                                                    \
    KERNEL_FLOAT_INLINE auto operator OP(const constant<L>& left, const vector<R, E>& right) {       \
        auto f = ops::cast<L, R>();                                                                  \
        return f(left.get()) OP right;                                                               \
    }                                                                                                \
                                                                                                        \
    template <typename L, typename R, typename E>                                                    \
    KERNEL_FLOAT_INLINE auto operator OP(const vector<L, E>& left, const constant<R>& right) {       \
        auto f = ops::cast<R, L>();                                                                  \
        return left OP f(right.get());                                                               \
    }                                                                                                \
                                                                                                        \
    template <typename L, typename R, typename T = promote_t<L, R>>                                  \
    KERNEL_FLOAT_INLINE constant<T> operator OP(const constant<L>& left, const constant<R>& right) { \
        auto fl = ops::cast<L, T>();                                                                 \
        auto fr = ops::cast<R, T>();                                                                 \
        return fl(left.get()) OP fr(right.get());                                                    \
    }

KERNEL_FLOAT_CONSTANT_DEFINE_OP(+)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(-)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(*)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(/)
KERNEL_FLOAT_CONSTANT_DEFINE_OP(%)

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_ITERATE_H
#define KERNEL_FLOAT_ITERATE_H

namespace kernel_float {

/**
 * Apply the function fun for each element from input.
 *
 * Example
 * =======
 * ```
 * for_each(range<int, 3>(), [&](auto i) {
 *    printf("element: %d\n", i);
 * });
 * ```
 */
template <typename V, typename F>
KERNEL_FLOAT_INLINE void for_each(V&& input, F fun) {
    auto storage = into_vector_storage(input);

#pragma unroll
    for (size_t i = 0; i < vector_size<V>; i++) {
        fun(storage.data()[i]);
    }
}

namespace detail {
template <typename T, size_t N>
struct range_impl {
    template <typename F>
    KERNEL_FLOAT_INLINE static vector_storage<T, N> call(F fun) {
        vector_storage<T, N> result;

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            result.data()[i] = fun(i);
        }

        return result;
    }
};
}  // namespace detail

/**
 * Generate vector consisting of the result `fun(0), ..., fun(N-1)`
 *
 * Example
 * =======
 * ```
 * // Returns [0.0f, 2.0f, 4.0f]
 * vec<float, 3> vec = range<3>([](auto i){ return float(i * 2.0f); });
 * ```
 */
template <size_t N, typename F, typename T = result_t<F, size_t>>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range(F fun) {
    return detail::range_impl<T, N>::call(fun);
}

/**
 * Generate vector consisting of the numbers `0, ..., N-1` of type `T`
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2]
 * vec<float, 3> vec = range<float, 3>();
 * ```
 */
template <typename T, size_t N>
KERNEL_FLOAT_INLINE vector<T, extent<N>> range() {
    return detail::range_impl<T, N>::call(ops::cast<size_t, T>());
}

/**
 * Takes a vector `vec<T, N>` and returns a new vector consisting of the numbers ``0, ..., N-1`` of type ``T``
 *
 * Example
 * =======
 * ```
 * auto input = vec<float, 3>(5.0f, 10.0f, -1.0f);
 * auto indices = range_like(input);  // returns [0.0f, 1.0f, 2.0f]
 * ```
 */
template <typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> range_like(const V& = {}) {
    return range<vector_value_type<V>, vector_size<V>>();
}

/**
 * Takes a vector of size ``N`` and returns a new vector consisting of the numbers ``0, ..., N-1``. The data type used
 * for the indices is given by the first template argument, which is `size_t` by default. This function is useful when
 * needing to iterate over the indices of a vector.
 *
 * Example
 * =======
 * ```
 * // Returns [0, 1, 2] of type size_t
 * vec<size_t, 3> a = each_index(float3(6, 4, 2));
 *
 * // Returns [0, 1, 2] of type int.
 * vec<int, 3> b = each_index<int>(float3(6, 4, 2));
 *
 * vec<float, 3> input = {1.0f, 2.0f, 3.0f, 4.0f};
 * for (auto index: each_index<int>(input)) {
 *   printf("%d] %f\n", index, input[index]);
 * }
 * ```
 */
template <typename T = size_t, typename V>
KERNEL_FLOAT_INLINE vector<T, vector_extent_type<V>> each_index(const V& = {}) {
    return range<T, vector_size<V>>();
}

namespace detail {
template <typename V, typename T = vector_value_type<V>, size_t N = vector_size<V>>
struct flatten_impl {
    using value_type             = typename flatten_impl<T>::value_type;
    static constexpr size_t size = N * flatten_impl<T>::size;

    template <typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, N> storage = into_vector_storage(input);

#pragma unroll
        for (size_t i = 0; i < N; i++) {
            flatten_impl<T>::call(output + flatten_impl<T>::size * i, storage.data()[i]);
        }
    }
};

template <typename T>
struct flatten_impl<T, T, 1> {
    using value_type             = T;
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE
    static void call(T* output, const T& input) { *output = input; }

    template <typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) {
        *output = ops::cast<T, U>{}(input);
    }
};
}  // namespace detail

template <typename V>
using flatten_value_type = typename detail::flatten_impl<V>::value_type;

template <typename V>
static constexpr size_t flatten_size = detail::flatten_impl<V>::size;

template <typename V>
using flatten_type = vector<flatten_value_type<V>, extent<flatten_size<V>>>;

/**
 * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
 *
 * Example
 * =======
 * ```
 * vec<float2, 3> input = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
 * vec<float, 6> result = flatten(input); // returns [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]
 * ```
 */
template <typename V>
KERNEL_FLOAT_INLINE flatten_type<V> flatten(const V& input) {
    vector_storage<flatten_value_type<V>, flatten_size<V>> output;
    detail::flatten_impl<V>::call(output.data(), input);
    return output;
}

namespace detail {
template <typename U, typename V = U, typename T = vector_value_type<V>>
struct concat_base_impl {
    static constexpr size_t size = vector_size<V>;

    KERNEL_FLOAT_INLINE static void call(U* output, const V& input) {
        vector_storage<T, size> storage = into_vector_storage(input);

        for (size_t i = 0; i < size; i++) {
            output[i] = ops::cast<T, U>{}(storage.data()[i]);
        }
    }
};

template <typename U, typename T>
struct concat_base_impl<U, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(U* output, const T& input) { *output = ops::cast<T, U>{}(input); }
};

template <typename T>
struct concat_base_impl<T, T, T> {
    static constexpr size_t size = 1;

    KERNEL_FLOAT_INLINE static void call(T* output, const T& input) { *output = input; }
};

template <typename... Vs>
struct concat_impl {};

template <typename V, typename... Vs>
struct concat_impl<V, Vs...> {
    using value_type             = typename promote_type<vector_value_type<V>, typename concat_impl<Vs...>::value_type>::type;
    static constexpr size_t size = concat_base_impl<V>::size + concat_impl<Vs...>::size;

    template <typename U>
    KERNEL_FLOAT_INLINE static void call(U* output, const V& input, const Vs&... rest) {
        concat_base_impl<U, V>::call(output, input);
        concat_impl<Vs...>::call(output + concat_base_impl<U, V>::size, rest...);
    }
};

template <>
struct concat_impl<> {
    using value_type             = void;
    static constexpr size_t size = 0;

    template <typename U>
    KERNEL_FLOAT_INLINE static void call(U* output) {}
};
}  // namespace detail

template <typename... Vs>
using concat_value_type = promote_t<typename detail::concat_impl<Vs...>::value_type>;

template <typename... Vs>
static constexpr size_t concat_size = detail::concat_impl<Vs...>::size;

template <typename... Vs>
using concat_type = vector<concat_value_type<Vs...>, extent<concat_size<Vs...>>>;

/**
 * Concatenates the provided input values into a single one-dimensional vector.
 *
 * This function works in three steps:
 * - All input values are converted into vectors using the `into_vector` operation.
 * - The resulting vectors' elements are then promoted into a shared value type.
 * - The resultant vectors are finally concatenated together.
 *
 * For instance, when invoking this function with arguments of types `float, double2, double`:
 * - After the first step: `vec<float, 1>, vec<double, 2>, vec<double, 1>`
 * - After the second step: `vec<double, 1>, vec<double, 2>, vec<double, 1>`
 * - After the third step: `vec<double, 4>`
 *
 * Example
 * =======
 * ```
 * double vec1 = 1.0;
 * double3 vec2 = {2.0, 3.0, 4.0);
 * double4 vec3 = {5.0, 6.0, 7.0, 8.0};
 * vec<double, 8> concatenated = concat(vec1, vec2, vec3); // contains [1, 2, 3, 4, 5, 6, 7, 8]
 *
 * int num1 = 42;
 * float num2 = 3.14159;
 * int2 num3 = {-10, 10};
 * vec<float, 4> concatenated = concat(num1, num2, num3); // contains [42, 3.14159, -10, 10]
 * ```
 */
template <typename... Vs>
KERNEL_FLOAT_INLINE concat_type<Vs...> concat(const Vs&... inputs) {
    vector_storage<concat_value_type<Vs...>, concat_size<Vs...>> output;
    detail::concat_impl<Vs...>::call(output.data(), inputs...);
    return output;
}

template <typename V, typename... Is>
using select_type = vector<vector_value_type<V>, extent<concat_size<Is...>>>;

/**
 * Selects elements from the this vector based on the specified indices.
 *
 * Example
 * =======
 * ```
 * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
 * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
 *
 * vec<int, 4> indices = {0, 4, 4, 2};
 * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
 * ```
 */
template <typename V, typename... Is>
KERNEL_FLOAT_INLINE select_type<V, Is...> select(const V& input, const Is&... indices) {
    using T                   = vector_value_type<V>;
    static constexpr size_t N = vector_size<V>;
    static constexpr size_t M = concat_size<Is...>;

    vector_storage<size_t, M> index_set;
    detail::concat_impl<Is...>::call(index_set.data(), indices...);

    vector_storage<T, N> inputs = into_vector_storage(input);
    vector_storage<T, M> outputs;
    for (size_t i = 0; i < M; i++) {
        size_t j = index_set.data()[i];

        if (j < N) {
            outputs.data()[i] = inputs.data()[j];
        }
    }

    return outputs;
}

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_MEMORY_H
#define KERNEL_FLOAT_MEMORY_H

namespace kernel_float {
namespace detail {
template <typename T, size_t N, typename Is = make_index_sequence<N>>
struct copy_impl;

template <typename T, size_t N, size_t... Is>
struct copy_impl<T, N, index_sequence<Is...>> {
    KERNEL_FLOAT_INLINE
    static vector_storage<T, N> load(const T* input, const size_t* offsets, const bool* mask) {
        return {(mask[Is] ? input[offsets[Is]] : T{})...};
    }

    KERNEL_FLOAT_INLINE
    static void store(T* outputs, const T* inputs, const size_t* offsets, const bool* mask) {
        ((mask[Is] ? outputs[offsets[Is]] = inputs[Is] : T{}), ...);
    }
};
}  // namespace detail

/**
 * Load the elements from the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be loaded and ``false`` indicates that the value should be skipped. This can be used
 * to prevent reading out of bounds.
 *
 * ```
 * // Load 2 elements at data[0] and data[8], skip data[2] and data[4]
 * vec<T, 4> values = read(data, make_vec(0, 2, 4, 8), make_vec(true, false, false, true));
 * ```
 */
template <typename T, typename I, typename M = bool, typename E = broadcast_vector_extent_type<I, M>>
KERNEL_FLOAT_INLINE vector<T, E> read(const T* ptr, const I& indices, const M& mask = true) {
    return detail::copy_impl<T, extent_size<E>>::load(ptr, convert_storage<size_t>(indices, E()).data(),
                                                        convert_storage<bool>(mask, E()).data());
}

/**
 * Store the elements from the vector `values` in the buffer ``ptr`` at the locations specified by ``indices``.
 *
 * The ``mask`` should be a vector of booleans where ``true`` indicates that the value should
 * be store and ``false`` indicates that the value should be skipped. This can be used
 * to prevent writing out of bounds.
 *
 * ```
 * // Store 2 elements at data[0] and data[8], skip data[2] and data[4]
 * auto values = make_vec(42, 13, 87, 12);
 * auto mask = make_vec(true, false, false, true);
 * write(data, make_vec(0, 2, 4, 8), values, mask);
 * ```
 */
template <typename T, typename V, typename I, typename M = bool, typename E = broadcast_vector_extent_type<V, I, M>>
KERNEL_FLOAT_INLINE void write(T* ptr, const I& indices, const V& values, const M& mask = true) {
    return detail::copy_impl<T, extent_size<E>>::store(ptr, convert_storage<T>(values, E()).data(),
                                                        convert_storage<size_t>(indices, E()).data(), convert_storage<bool>(mask, E()).data());
}

/**
 * Load ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * ```
 * // Load 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<T, 4> values = read<4>(data);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * vec<T, 4> values = read<4>(data + 10);
 * ```
 */
template <size_t N, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> read(const T* ptr) {
    return read(ptr, range<size_t, N>());
}

/**
 * Store ``N`` elements at the location ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * ```
 * // Store 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * write(data, values);
 *
 * // Store 4 elements at locations data[10], data[11], data[12], data[13]
 * write(data + 10, values);
 * ```
 */
template <typename V, typename T>
KERNEL_FLOAT_INLINE void write(T* ptr, const V& values) {
    static constexpr size_t N = vector_size<V>;
    write(ptr, range<size_t, N>(), values);
}

namespace detail {
/**
 * Returns the greatest common divisor of `a` and `b`.
 */
KERNEL_FLOAT_INLINE
constexpr size_t gcd(size_t a, size_t b) { return b == 0 ? a : gcd(b, a % b); }

/**
 * Returns true if a pointer having alignment of `a` bytes also has an alignment of `b` bytes. Returns false otherwise.
 */
KERNEL_FLOAT_INLINE
constexpr size_t alignment_divisible(size_t a, size_t b) {
    return gcd(a, KERNEL_FLOAT_MAX_ALIGNMENT) % gcd(b, KERNEL_FLOAT_MAX_ALIGNMENT) == 0;
}

template <typename T, size_t N, size_t Alignment, typename = void>
struct copy_aligned_impl {
    static constexpr size_t K           = N > 8 ? 8 : (N > 4 ? 4 : (N > 2 ? 2 : 1));
    static constexpr size_t Alignment_K = gcd(Alignment, sizeof(T) * K);

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        copy_aligned_impl<T, K, Alignment>::load(output, input);
        copy_aligned_impl<T, N - K, Alignment_K>::load(output + K, input + K);
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        copy_aligned_impl<T, K, Alignment>::store(output, input);
        copy_aligned_impl<T, N - K, Alignment_K>::store(output + K, input + K);
    }
};

template <typename T, size_t Alignment>
struct copy_aligned_impl<T, 0, Alignment> {
    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {}

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {}
};

template <typename T, size_t Alignment>
struct copy_aligned_impl<T, 1, Alignment> {
    using storage_type = T;

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) { output[0] = input[0]; }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) { output[0] = input[0]; }
};

template <typename T, size_t Alignment>
struct copy_aligned_impl<T, 2, Alignment, enable_if_t<(Alignment > sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(Alignment, 2 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0]            = storage.v0;
        output[1]            = storage.v1;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) { *reinterpret_cast<storage_type*>(output) = storage_type{input[0], input[1]}; }
};

template <typename T, size_t Alignment>
struct copy_aligned_impl<T, 4, Alignment, enable_if_t<(Alignment > 2 * sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(Alignment, 4 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1, v2, v3;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0]            = storage.v0;
        output[1]            = storage.v1;
        output[2]            = storage.v2;
        output[3]            = storage.v3;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        *reinterpret_cast<storage_type*>(output) = storage_type{input[0],  //
                                                                input[1], input[2], input[3]};
    }
};

template <typename T, size_t Alignment>
struct copy_aligned_impl<T, 8, Alignment, enable_if_t<(Alignment > 4 * sizeof(T))>> {
    static constexpr size_t storage_alignment = gcd(Alignment, 8 * sizeof(T));
    struct alignas(storage_alignment) storage_type {
        T v0, v1, v2, v3, v4, v5, v6, v7;
    };

    KERNEL_FLOAT_INLINE
    static void load(T* output, const T* input) {
        storage_type storage = *reinterpret_cast<const storage_type*>(input);
        output[0]            = storage.v0;
        output[1]            = storage.v1;
        output[2]            = storage.v2;
        output[3]            = storage.v3;
        output[4]            = storage.v4;
        output[5]            = storage.v5;
        output[6]            = storage.v6;
        output[7]            = storage.v7;
    }

    KERNEL_FLOAT_INLINE
    static void store(T* output, const T* input) {
        *reinterpret_cast<storage_type*>(output) = storage_type{input[0],  //
                                                                input[1], input[2], input[3], input[4], input[5], input[6], input[7]};
    }
};

}  // namespace detail

/**
 * Load ``N`` elements at the locations ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * It is assumed that ``ptr`` is maximum aligned such that all ``N`` elements can be loaded at once using a vector
 * operation. If the pointer is not aligned, undefined behavior will occur.
 *
 * ```
 * // Load 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<T, 4> values = read_aligned<4>(data);
 *
 * // Load 4 elements at locations data[12], data[13], data[14], data[15]
 * vec<T, 4> values2 = read_aligned<4>(data + 12);
 * ```
 */
template <size_t Align, size_t N = Align, typename T>
KERNEL_FLOAT_INLINE vector<T, extent<N>> read_aligned(const T* ptr) {
    static constexpr size_t alignment = detail::gcd(Align * sizeof(T), KERNEL_FLOAT_MAX_ALIGNMENT);
    vector_storage<T, N> result;
    detail::copy_aligned_impl<T, N, alignment>::load(result.data(), KERNEL_FLOAT_ASSUME_ALIGNED(const T, ptr, alignment));
    return result;
}

/**
 * Store ``N`` elements at the locations ``ptr[0], ptr[1], ptr[2], ...``.
 *
 * It is assumed that ``ptr`` is maximum aligned such that all ``N`` elements can be loaded at once using a vector
 * operation. If the pointer is not aligned, undefined behavior will occur.
 *
 * ```
 * // Store 4 elements at locations data[0], data[1], data[2], data[3]
 * vec<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
 * write_aligned(data, values);
 *
 * // Load 4 elements at locations data[10], data[11], data[12], data[13]
 * write_aligned(data + 10, values);
 * ```
 */
template <size_t Align, typename V, typename T>
KERNEL_FLOAT_INLINE void write_aligned(T* ptr, const V& values) {
    static constexpr size_t N         = vector_size<V>;
    static constexpr size_t alignment = detail::gcd(Align * sizeof(T), KERNEL_FLOAT_MAX_ALIGNMENT);

    return detail::copy_aligned_impl<T, N, alignment>::store(KERNEL_FLOAT_ASSUME_ALIGNED(T, ptr, alignment),
                                                                convert_storage<T, N>(values).data());
}

/**
 * @brief A reference wrapper that allows reading/writing a vector of type `T`and length `N` with optional data
 * conversion.
 *
 * @tparam T The type of the elements as seen from the user's perspective.
 * @tparam N The number of elements in the vector.
 * @tparam U The underlying storage type. Defaults to the same type as T.
 * @tparam Align  The alignment constraint for read and write operations.
 */
template <typename T, size_t N, typename U = T, size_t Alignment = alignof(U)>
struct vector_ref {
    static constexpr size_t access_alignment = detail::gcd(Alignment, KERNEL_FLOAT_MAX_ALIGNMENT);
    static_assert(access_alignment >= alignof(U), "invalid alignment for pointer type");

    using pointer_type = U*;
    using value_type   = T;
    using vector_type  = vector<value_type, extent<N>>;

    /**
     * Constructs a vector_ref to manage access to a raw data pointer.
     *
     * @param data Pointer to the raw data this vector_ref will manage.
     */
    KERNEL_FLOAT_INLINE explicit vector_ref(pointer_type data) : data_(data) {}

    /**
     * Reads data from the underlying raw pointer, converting it to type `T`.
     *
     * @return vector_type A vector of type vector_type containing the read and converted data.
     */
    KERNEL_FLOAT_INLINE vector_type read() const {
        vector_storage<U, N> result;
        detail::copy_aligned_impl<U, N, access_alignment>::load(result.data(), KERNEL_FLOAT_ASSUME_ALIGNED(const U, data_, access_alignment));

        return convert<value_type, N>(result);
    }

    /**
     * Writes data to the underlying raw pointer, converting it from the input vector if necessary.
     *
     * @tparam V The type of the input vector, defaults to `T`.
     * @param values The values to be written.
     */
    template <typename V = vector_type>
    KERNEL_FLOAT_INLINE void write(const V& values) const {
        detail::copy_aligned_impl<U, N, access_alignment>::store(KERNEL_FLOAT_ASSUME_ALIGNED(U, data_, access_alignment),
                                                                    convert_storage<U, N>(values).data());
    }

    /**
     * Conversion operator that is shorthand for `read()`.
     */
    KERNEL_FLOAT_INLINE operator vector_type() const { return read(); }

    /**
     * Assignment operator that is shorthand for `write(values)`.
     */
    template <typename V>
    KERNEL_FLOAT_INLINE vector_ref operator=(const V& values) const {
        write(values);
        return *this;
    }

    /**
     * Gets the raw data pointer managed by this vector_ref
     */
    KERNEL_FLOAT_INLINE pointer_type get() const { return data_; }

    private:
    pointer_type data_ = nullptr;
};

/**
 * Specialization for `vector_ref` if the backing storage is const.
 */
template <typename T, size_t N, typename U, size_t Alignment>
struct vector_ref<T, N, const U, Alignment> {
    static constexpr size_t access_alignment = detail::gcd(Alignment, KERNEL_FLOAT_MAX_ALIGNMENT);
    static_assert(access_alignment >= alignof(U), "invalid alignment for pointer type");

    using pointer_type = const U*;
    using value_type   = T;
    using vector_type  = vector<value_type, extent<N>>;

    KERNEL_FLOAT_INLINE explicit vector_ref(pointer_type data) : data_(data) {}

    KERNEL_FLOAT_INLINE vector_type read() const {
        vector_storage<U, N> result;
        detail::copy_aligned_impl<U, N, access_alignment>::load(result.data(), KERNEL_FLOAT_ASSUME_ALIGNED(const U, data_, access_alignment));

        return convert_storage<value_type, N>(result);
    }

    KERNEL_FLOAT_INLINE operator vector_type() const { return read(); }

    KERNEL_FLOAT_INLINE pointer_type get() const { return data_; }

    private:
    pointer_type data_ = nullptr;
};

#define KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(OP, OP_ASSIGN)                                                                        \
    template <typename T, size_t N, typename U, size_t Alignment, typename V>                                                   \
    KERNEL_FLOAT_INLINE vector_ref<T, N, U, Alignment> operator OP_ASSIGN(vector_ref<T, N, U, Alignment> ptr, const V& value) { \
        ptr.write(ptr.read() OP value);                                                                                         \
        return ptr;                                                                                                             \
    }

KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(+, +=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(-, -=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(*, *=)
KERNEL_FLOAT_VECTOR_REF_ASSIGN_OP(/, /=)

template <typename T, size_t N, typename U, size_t Alignment>
struct into_vector_impl<vector_ref<T, N, U, Alignment>> {
    using value_type  = T;
    using extent_type = extent<N>;

    KERNEL_FLOAT_INLINE
    static vector_storage<value_type, N> call(const vector_ref<T, N, U, Alignment>& reference) { return reference.read(); }
};

/**
 * A wrapper for a pointer that enables vectorized access and supports type conversions..
 *
 * The `vector_ptr<T, N, U>` type is designed to function as if its a `vec<T, N>*` pointer, allowing of reading and
 * writing `vec<T, N>` elements. However, the actual type of underlying storage is a pointer of type `U*`, where
 * automatic conversion is performed between `T` and `U` when reading/writing items.
 *
 * For example, a `vector_ptr<double, N, half>`  is useful where the data is stored in low precision (here 16 bit)
 * but it should be accessed as if it was in a higher precision format (here 64 bit).
 *
 * @tparam T The type of the elements as viewed by the user.
 * @tparam N The alignment of T in number of elements.
 * @tparam U The underlying storage type, defaults to T.
 * @tparam Alignment The assumed alignment of the underlying U* pointer.
 */
template <typename T, size_t N, typename U = T, size_t Alignment = sizeof(U) * N>
struct vector_ptr {
    static constexpr size_t offset_alignment = detail::gcd(Alignment, sizeof(U) * N);
    using pointer_type                       = U*;
    using value_type                         = decay_t<T>;

    /**
     * Default constructor sets the pointer to `NULL`.
     */
    vector_ptr() = default;

    /**
     * Constructor from a given pointer. It is up to the user to assert that the pointer is aligned to `Align` elements.
     */
    KERNEL_FLOAT_INLINE explicit vector_ptr(pointer_type p) : data_(p) {}

    /**
     * Constructs a vector_ptr from another vector_ptr with potentially different alignment and type. This constructor
     * only allows conversion if the alignment of the source is greater than or equal to the alignment of the target.
     */
    template <typename T2, size_t N2, size_t A2>
    KERNEL_FLOAT_INLINE vector_ptr(vector_ptr<T2, N2, U, A2> p, enable_if_t<detail::alignment_divisible(A2, Alignment), int> = {})
        : data_(p.get()) {}

    /**
     * Shorthand for `at(0)`.
     */
    KERNEL_FLOAT_INLINE const vector_ref<value_type, N, U, Alignment> operator*() const {
        return vector_ref<value_type, N, U, Alignment>{data_};
    }

    /**
     * Accesses a reference to a vector at a specific index with optional alignment considerations.
     *
     * @tparam N The number of elements in the vector to access, defaults to N.
     * @param index The index at which to access the vector.
     */
    template <size_t K = N>
    KERNEL_FLOAT_INLINE vector_ref<value_type, K, U, offset_alignment> at(size_t index) const {
        return vector_ref<T, K, U, offset_alignment>{data_ + index * N};
    }

    /**
     * Shorthand for `at(index)`.
     */
    KERNEL_FLOAT_INLINE vector_ref<value_type, N, U, offset_alignment> operator[](size_t index) const { return at(index); }

    /**
     * Accesses a vector at a specific index.
     *
     * @tparam K The number of elements to read, defaults to `N`.
     * @param index The index from which to read the data.
     */
    template <size_t K = N>
    KERNEL_FLOAT_INLINE vector<value_type, extent<K>> read(size_t index = 0) const {
        return this->template at<K>(index).read();
    }

    /**
     * @brief Writes data to a specific index.
     *
     * @tparam K The number of elements to write, defaults to `N`.
     * @tparam V The type of the values being written.
     * @param index The index at which to write the data.
     * @param values The vector of values to write.
     */
    template <size_t K = N, typename V>
    KERNEL_FLOAT_INLINE void write(size_t index, const V& values) const {
        this->template at<K>(index).write(values);
    }

    /**
     * Gets the raw data pointer managed by this `vector_ptr`.
     */
    KERNEL_FLOAT_INLINE pointer_type get() const { return data_; }

    private:
    pointer_type data_ = nullptr;
};

/**
 * Specialization for `vector_ptr` if the backing storage is const.
 */
template <typename T, size_t N, typename U, size_t Alignment>
struct vector_ptr<T, N, const U, Alignment> {
    static constexpr size_t offset_alignment = detail::gcd(Alignment, sizeof(U) * N);
    using pointer_type                       = const U*;
    using value_type                         = decay_t<T>;

    vector_ptr() = default;

    KERNEL_FLOAT_INLINE explicit vector_ptr(pointer_type p) : data_(p) {}

    template <typename T2, size_t N2, size_t A2>
    KERNEL_FLOAT_INLINE vector_ptr(vector_ptr<T2, N2, const U, A2> p, enable_if_t<detail::alignment_divisible(A2, Alignment), int> = {})
        : data_(p.get()) {}

    template <typename T2, size_t N2, size_t A2>
    KERNEL_FLOAT_INLINE vector_ptr(vector_ptr<T2, N2, U, A2> p, enable_if_t<detail::alignment_divisible(A2, Alignment), int> = {})
        : data_(p.get()) {}

    KERNEL_FLOAT_INLINE vector_ref<value_type, N, const U, Alignment> operator*() const {
        return vector_ref<value_type, N, const U, Alignment>{data_};
    }

    template <size_t K = N>
    KERNEL_FLOAT_INLINE vector_ref<value_type, K, const U, offset_alignment> at(size_t index) const {
        return vector_ref<value_type, K, const U, offset_alignment>{data_ + index * N};
    }

    template <size_t K = N>
    KERNEL_FLOAT_INLINE vector<value_type, extent<K>> read(size_t index = 0) const {
        return this->template at<K>(index).read();
    }

    KERNEL_FLOAT_INLINE const vector<value_type, extent<N>> operator[](size_t index) const { return read(index); }

    KERNEL_FLOAT_INLINE pointer_type get() const { return data_; }

    private:
    pointer_type data_ = nullptr;
};

template <typename T, size_t N, typename U, size_t A>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U, detail::gcd(A, N * sizeof(U))> operator+(vector_ptr<T, N, U, A> p, size_t i) {
    return vector_ptr<T, N, U, detail::gcd(A, N * sizeof(U))>{p.get() + i * N};
}

template <typename T, size_t N, typename U, size_t A>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U, detail::gcd(A, N * sizeof(U))> operator+(size_t i, vector_ptr<T, N, U, A> p) {
    return p + i;
}

template <typename T, size_t N, typename U, size_t A, typename = enable_if_t<(N * sizeof(U)) % A == 0>>
KERNEL_FLOAT_INLINE vector_ptr<T, N, U, A>& operator+=(vector_ptr<T, N, U, A>& p, size_t i) {
    return p = p + i;
}

/**
 * Creates a `vector_ptr<T, N>` from a raw pointer `T*` by asserting a specific alignment `N`.
 *
 * @tparam N The alignment constraint for the vector_ptr. Defaults to KERNEL_FLOAT_MAX_ALIGNMENT.
 * @tparam T The type of the elements pointed to by the raw pointer.
 */
template <size_t N, typename T>
KERNEL_FLOAT_INLINE vector_ptr<T, N> assert_aligned(T* ptr) {
    return vector_ptr<T, N>{ptr};
}

// Doxygen cannot deal with the `assert_aligned` being defined twice, we ignore the second definition.
/// @cond IGNORE
/**
 * Creates a `vector_ptr<T, 1>` from a raw pointer `T*`. The alignment is assumed to be KERNEL_FLOAT_MAX_ALIGNMENT.
 *
 * @tparam T The type of the elements pointed to by the raw pointer.
 */
template <typename T>
KERNEL_FLOAT_INLINE vector_ptr<T, 1, T, KERNEL_FLOAT_MAX_ALIGNMENT> assert_aligned(T* ptr) {
    return vector_ptr<T, 1, T, KERNEL_FLOAT_MAX_ALIGNMENT>{ptr};
}
/// @endcond

template <typename T, size_t N = 1, typename U = T, size_t Align = N>
using vec_ptr = vector_ptr<T, N, U, Align * sizeof(U)>;

#if defined(__cpp_deduction_guides)
template <typename T>
vector_ptr(T*) -> vector_ptr<T, 1, T>;

template <typename T>
vector_ptr(const T*) -> vector_ptr<T, 1, const T>;

#if __cpp_deduction_guides >= 201907L
template <typename T>
vec_ptr(T*) -> vec_ptr<T, 1, T>;

template <typename T>
vec_ptr(const T*) -> vec_ptr<T, 1, const T>;
#endif
#endif

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_MEMORY_H
#ifndef KERNEL_FLOAT_TRIOPS_H
#define KERNEL_FLOAT_TRIOPS_H

namespace kernel_float {

namespace ops {
template <typename T>
struct conditional {
    KERNEL_FLOAT_INLINE T operator()(bool cond, T true_value, T false_value) {
        if (cond) {
            return true_value;
        } else {
            return false_value;
        }
    }
};
}  // namespace ops

/**
 * Return elements chosen from `true_values` and `false_values` depending on `cond`.
 *
 * This function broadcasts all arguments to the same size and then promotes the values of `true_values` and
 * `false_values` into the same type. Next, it casts the values of `cond` to booleans and returns a vector where
 * the values are taken from `true_values` where the condition is true and `false_values` otherwise.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @param false_values The vector of values to choose from when the condition is false.
 * @return A vector containing selected elements as per the condition.
 */
template <typename C, typename L, typename R, typename T = promoted_vector_value_type<L, R>, typename E = broadcast_vector_extent_type<C, L, R>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values, const R& false_values) {
    using F = ops::conditional<T>;
    vector_storage<T, extent_size<E>> result;

    detail::default_map_impl<F, extent_size<E>, T, bool, T, T>::call(
        F{}, result.data(), detail::convert_impl<vector_value_type<C>, vector_extent_type<C>, bool, E>::call(into_vector_storage(cond)).data(),
        detail::convert_impl<vector_value_type<L>, vector_extent_type<L>, T, E>::call(into_vector_storage(true_values)).data(),
        detail::convert_impl<vector_value_type<R>, vector_extent_type<R>, T, E>::call(into_vector_storage(false_values)).data());

    return result;
}

/**
 * Selects elements from `true_values` depending on `cond`.
 *
 * This function returns a vector where the values are taken from `true_values` where `cond` is `true` and `0` where
 * `cond is `false`.
 *
 * @param cond The condition used for selection.
 * @param true_values The vector of values to choose from when the condition is true.
 * @return A vector containing selected elements as per the condition.
 */
template <typename C, typename L, typename T = vector_value_type<L>, typename E = broadcast_vector_extent_type<C, L>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond, const L& true_values) {
    vector<T, extent<1>> false_values = T{};
    return where(cond, true_values, false_values);
}

/**
 * Returns a vector having the value `T(1)` where `cond` is `true` and `T(0)` where `cond` is `false`.
 *
 * @param cond The condition used for selection.
 * @return A vector containing elements as per the condition.
 */
template <typename T = bool, typename C, typename E = vector_extent_type<C>>
KERNEL_FLOAT_INLINE vector<T, E> where(const C& cond) {
    return cast<T>(cast<bool>(cond));
}

namespace ops {
template <typename T>
struct fma {
    KERNEL_FLOAT_INLINE T operator()(T a, T b, T c) { return ops::add<T>{}(ops::multiply<T>{}(a, b), c); }
};
}  // namespace ops

namespace detail {
template <typename T, size_t N>
struct apply_impl<accurate_policy, ops::fma<T>, N, T, T, T, T> {
    KERNEL_FLOAT_INLINE
    static void call(ops::fma<T>, T* output, const T* a, const T* b, const T* c) {
        T temp[N];
        apply_impl<accurate_policy, ops::multiply<T>, N, T, T, T>::call({}, temp, a, b);
        apply_impl<accurate_policy, ops::add<T>, N, T, T, T>::call({}, output, temp, c);
    }
};
}  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
namespace ops {
template <>
struct fma<float> {
    KERNEL_FLOAT_INLINE float operator()(float a, float b, float c) { return __fmaf_rn(a, b, c); }
};

template <>
struct fma<double> {
    KERNEL_FLOAT_INLINE double operator()(double a, double b, double c) { return __fma_rn(a, b, c); }
};
}  // namespace ops
#endif

/**
 * Computes the result of `a * b + c`. This is done in a single operation if possible for the given vector type.
 */
template <typename A, typename B, typename C, typename T = promoted_vector_value_type<A, B, C>,
            typename E = broadcast_vector_extent_type<A, B, C>>
KERNEL_FLOAT_INLINE vector<T, E> fma(const A& a, const B& b, const C& c) {
    using F = ops::fma<T>;
    vector_storage<T, extent_size<E>> result;

    detail::default_map_impl<F, extent_size<E>, T, T, T, T>::call(
        F{}, result.data(), detail::convert_impl<vector_value_type<A>, vector_extent_type<A>, T, E>::call(into_vector_storage(a)).data(),
        detail::convert_impl<vector_value_type<B>, vector_extent_type<B>, T, E>::call(into_vector_storage(b)).data(),
        detail::convert_impl<vector_value_type<C>, vector_extent_type<C>, T, E>::call(into_vector_storage(c)).data());

    return result;
}

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_TRIOPS_H
#ifndef KERNEL_FLOAT_REDUCE_H
#define KERNEL_FLOAT_REDUCE_H

namespace kernel_float {
namespace detail {

template <size_t N>
struct reduce_recur_impl;

template <typename F, size_t N, typename T, typename = void>
struct reduce_impl {
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input) { return reduce_recur_impl<N>::call(fun, input); }
};

template <size_t N>
struct reduce_recur_impl {
    static constexpr size_t K = round_up_to_power_of_two(N) / 2;

    template <typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input) {
        vector_storage<T, K> temp;
        default_map_impl<F, N - K, T, T, T>::call(fun, temp.data(), input, input + K);

        if constexpr (N < 2 * K) {
#pragma unroll
            for (size_t i = N - K; i < K; i++) {
                temp.data()[i] = input[i];
            }
        }

        return reduce_impl<F, K, T>::call(fun, temp.data());
    }
};

template <>
struct reduce_recur_impl<0> {};

template <>
struct reduce_recur_impl<1> {
    template <typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input) {
        return input[0];
    }
};

template <>
struct reduce_recur_impl<2> {
    template <typename F, typename T>
    KERNEL_FLOAT_INLINE static T call(F fun, const T* input) {
        return fun(input[0], input[1]);
    }
};

}  // namespace detail

/**
 * Reduce the elements of the given vector ``input`` into a single value using
 * the function ``fun``. This function should be a binary function that takes
 * two elements and returns one element. The order in which the elements
 * are reduced is not specified and depends on both the reduction function and
 * the vector type.
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {5, 2, 1};
 * int y = reduce(x, [](int a, int b) { return a + b; }); // returns 5+2+1=8
 * ```
 */
template <typename F, typename V>
KERNEL_FLOAT_INLINE vector_value_type<V> reduce(F fun, const V& input) {
    return detail::reduce_impl<F, vector_size<V>, vector_value_type<V>>::call(fun, into_vector_storage(input).data());
}

/**
 * Find the minimum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = min(x);  // Returns 0
 * ```
 */
template <typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T min(const V& input) {
    return reduce(ops::min<T>{}, input);
}

/**
 * Find the maximum element in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = max(x);  // Returns 5
 * ```
 */
template <typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T max(const V& input) {
    return reduce(ops::max<T>{}, input);
}

/**
 * Sum the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = sum(x);  // Returns 8
 * ```
 */
template <typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T sum(const V& input) {
    return reduce(ops::add<T>{}, input);
}

/**
 * Multiply the items in the given vector ``input``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = product(x);  // Returns 5*0*2*1*0 = 0
 * ```
 */
template <typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T product(const V& input) {
    return reduce(ops::multiply<T>{}, input);
}

/**
 * Check if all elements in the given vector ``input`` are non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template <typename V>
KERNEL_FLOAT_INLINE bool all(const V& input) {
    return reduce(ops::bit_and<bool>{}, cast<bool>(input));
}

/**
 * Check if any element in the given vector ``input`` is non-zero. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 */
template <typename V>
KERNEL_FLOAT_INLINE bool any(const V& input) {
    return reduce(ops::bit_or<bool>{}, cast<bool>(input));
}

/**
 * Count the number of non-zero items in the given vector ``input``. An element ``v`` is considered
 * non-zero if ``bool(v)==true``.
 *
 * Example
 * =======
 * ```
 * vec<int, 5> x = {5, 0, 2, 1, 0};
 * int y = count(x);  // Returns 3 (5, 2, 1 are non-zero)
 * ```
 */
template <typename T = int, typename V>
KERNEL_FLOAT_INLINE T count(const V& input) {
    return sum(cast<T>(cast<bool>(input)));
}

namespace detail {
template <typename T, size_t N>
struct dot_impl {
    KERNEL_FLOAT_INLINE
    static T call(const T* left, const T* right) {
        static constexpr size_t K = preferred_vector_size<T>::value;
        T result                  = {};

        if constexpr (N / K > 0) {
            T accum[K] = {T{}};
            apply_impl<default_policy, ops::multiply<T>, K, T, T, T>::call({}, accum, left, right);

#pragma unroll
            for (size_t i = 1; i < N / K; i++) {
                apply_impl<default_policy, ops::fma<T>, K, T, T, T, T>::call(ops::fma<T>{}, accum, left + i * K, right + i * K, accum);
            }

            result = reduce_impl<ops::add<T>, K, T>::call({}, accum);
        }

        if constexpr (N % K > 0) {
            for (size_t i = N - N % K; i < N; i++) {
                apply_impl<default_policy, ops::fma<T>, 1, T, T, T, T>::call({}, &result, left + i, right + i, &result);
            }
        }

        return result;
    }
};
}  // namespace detail

/**
 * Compute the dot product of the given vectors ``left`` and ``right``
 *
 * Example
 * =======
 * ```
 * vec<int, 3> x = {1, 2, 3};
 * vec<int, 3> y = {4, 5, 6};
 * int y = dot(x, y);  // Returns 1*4+2*5+3*6 = 32
 * ```
 */
template <typename L, typename R, typename T = promoted_vector_value_type<L, R>>
KERNEL_FLOAT_INLINE T dot(const L& left, const R& right) {
    using E = broadcast_vector_extent_type<L, R>;
    return detail::dot_impl<T, extent_size<E>>::call(convert_storage<T>(left, E{}).data(), convert_storage<T>(right, E{}).data());
}

namespace detail {
template <typename T, size_t N>
struct magnitude_impl {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) { return ops::sqrt<T>{}(detail::dot_impl<T, N>::call(input, input)); }
};

template <typename T>
struct magnitude_impl<T, 0> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) { return T{}; }
};

template <typename T>
struct magnitude_impl<T, 1> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) { return ops::abs<T>{}(input[0]); }
};

template <typename T>
struct magnitude_impl<T, 2> {
    KERNEL_FLOAT_INLINE
    static T call(const T* input) { return ops::hypot<T>()(input[0], input[1]); }
};

// The 3-argument overload of hypot is only available on host from C++17
// #if defined(__cpp_lib_hypot) && KERNEL_FLOAT_IS_HOST
// template<>
// struct magnitude_impl<float, 3> {
//    static float call(const float* input) {
//        return ::hypot(input[0], input[1], input[2]);
//    }
//};
//
// template<>
// struct magnitude_impl<double, 3> {
//    static double call(const double* input) {
//        return ::hypot(input[0], input[1], input[2]);
//    }
//};
// #endif

}  // namespace detail

/**
 * Compute the magnitude of the given input vector. This calculates the square root of the sum of squares, also
 * known as the Euclidian norm, of a vector.
 *
 * Example
 * =======
 * ```
 * vec<float, 3> x = {2, 3, 6};
 * float y = mag(x);  // Returns sqrt(2*2 + 3*3 + 6*6) = 7
 * ```
 */
template <typename V, typename T = vector_value_type<V>>
KERNEL_FLOAT_INLINE T mag(const V& input) {
    return detail::magnitude_impl<T, vector_size<V>>::call(into_vector_storage(input).data());
}
}  // namespace kernel_float

#endif  // KERNEL_FLOAT_REDUCE_H
#ifndef KERNEL_FLOAT_VECTOR_H
#define KERNEL_FLOAT_VECTOR_H

namespace kernel_float {

/**
 * Container that store fixed number of elements of type ``T``.
 *
 * It is not recommended to use this class directly, instead, use the type `vec<T, N>` which is an alias for
 * `vector<T, extent<N>, vector_storage<T, E>>`.
 *
 * @tparam T The type of the values stored within the vector.
 * @tparam E The size of this vector. Should be of type `extent<N>`.
 * @tparam S The object's storage class. Should be the type `vector_storage<T, E>`
 */
template <typename T, typename E, class S>
struct vector : public S {
    using self_type    = vector<T, E, S>;
    using value_type   = T;
    using extent_type  = E;
    using storage_type = S;

    // Copy another `vector<T, E>`
    vector(const vector&) = default;

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const storage_type& storage) : storage_type(storage) {}

    // Copy anything of type `storage_type`
    KERNEL_FLOAT_INLINE
    vector(const value_type& input = {}) : storage_type(detail::broadcast_impl<T, kernel_float::extent<1>, E>::call(input)) {}

    // For all other arguments, we convert it using `convert_storage` according to broadcast rules
    template <typename U, enable_if_t<is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE vector(U&& input) : storage_type(convert_storage<T>(input, extent_type{})) {}

    template <typename U, enable_if_t<!is_implicit_convertible<vector_value_type<U>, T>, int> = 0>
    KERNEL_FLOAT_INLINE explicit vector(U&& input) : storage_type(convert_storage<T>(input, extent_type{})) {}

    // List of `N` (where N >= 2), simply pass forward to the storage
    template <typename A, typename B, typename... Rest, typename = enable_if_t<sizeof...(Rest) + 2 == extent_size<E>>>
    KERNEL_FLOAT_INLINE vector(const A& a, const B& b, const Rest&... rest) : storage_type{T(a), T(b), T(rest)...} {}

    /**
     * Returns the number of elements in this vector.
     */
    KERNEL_FLOAT_INLINE
    static constexpr size_t size() { return extent_size<E>; }

    /**
     * Returns a reference to the underlying storage type.
     */
    KERNEL_FLOAT_INLINE
    storage_type& storage() { return *this; }

    /**
     * Returns a reference to the underlying storage type.
     */
    KERNEL_FLOAT_INLINE
    const storage_type& storage() const { return *this; }

    /**
     * Returns an instance of the `extent_type` for this vector.
     */
    KERNEL_FLOAT_INLINE
    extent_type extent() const { return {}; }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    T* data() { return storage().data(); }

    /**
     * Returns a pointer to the underlying storage data.
     */
    KERNEL_FLOAT_INLINE
    const T* data() const { return storage().data(); }

    KERNEL_FLOAT_INLINE
    const T* cdata() const { return this->data(); }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& at(size_t i) { return *(this->data() + i); }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& at(size_t i) const { return *(this->data() + i); }

    /**
     * Returns a reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T& operator[](size_t i) { return at(i); }

    /**
     * Returns a constant reference to the item at index `i`.
     */
    KERNEL_FLOAT_INLINE
    const T& operator[](size_t i) const { return at(i); }

    KERNEL_FLOAT_INLINE
    T& operator()(size_t i) { return at(i); }

    KERNEL_FLOAT_INLINE
    const T& operator()(size_t i) const { return at(i); }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    T* begin() { return this->data(); }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* begin() const { return this->data(); }

    /**
     * Returns a pointer to the first element.
     */
    KERNEL_FLOAT_INLINE
    const T* cbegin() const { return this->data(); }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    T* end() { return this->data() + size(); }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* end() const { return this->data() + size(); }

    /**
     * Returns a pointer to one past the last element.
     */
    KERNEL_FLOAT_INLINE
    const T* cend() const { return this->data() + size(); }

    /**
     * Copy the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    T get(size_t x) const { return at(x); }

    /**
     * Set the element at index `i`.
     */
    KERNEL_FLOAT_INLINE
    void set(size_t x, T value) { at(x) = static_cast<T&&>(value); }

    /**
     * Selects elements from the this vector based on the specified indices.
     *
     * Example
     * =======
     * ```
     * vec<float, 6> input = {0, 10, 20, 30, 40, 50};
     * vec<float, 4> vec1 = select(input, 0, 4, 4, 2); // [0, 40, 40, 20]
     *
     * vec<int, 4> indices = {0, 4, 4, 2};
     * vec<float, 4> vec2 = select(input, indices); // [0, 40, 40, 20]
     * ```
     */
    template <typename... Is>
    KERNEL_FLOAT_INLINE select_type<self_type, Is...> select(const Is&... indices) {
        return kernel_float::select(*this, indices...);
    }

    /**
     * Cast the elements of this vector to type `R` and returns a new vector.
     */
    template <typename R, RoundingMode Mode = RoundingMode::ANY>
    KERNEL_FLOAT_INLINE vector<R, extent_type> cast() const {
        return kernel_float::cast<R, Mode>(*this);
    }

    /**
     * Broadcast this vector into a new size `(Ns...)`.
     */
    template <size_t... Ns>
    KERNEL_FLOAT_INLINE vector<T, kernel_float::extent<Ns...>> broadcast(kernel_float::extent<Ns...> new_size = {}) const {
        return kernel_float::broadcast(*this, new_size);
    }

    /**
     * Apply the given function `F` to each element of this vector and returns a new vector with the results.
     */
    template <typename F>
    KERNEL_FLOAT_INLINE vector<result_t<F, T>, E> map(F fun) const {
        return kernel_float::map(fun, *this);
    }

    /**
     * Reduce the elements of the given vector input into a single value using the function `F`.
     *
     * This function should be a binary function that takes two elements and returns one element. The order in which
     * the elements are reduced is not specified and depends on the reduction function and the vector type.
     */
    template <typename F>
    KERNEL_FLOAT_INLINE T reduce(F fun) const {
        return kernel_float::reduce(fun, *this);
    }

    /**
     * Flattens the elements of this vector. For example, this turns a `vec<vec<int, 2>, 3>` into a `vec<int, 6>`.
     */
    KERNEL_FLOAT_INLINE flatten_type<vector> flatten() const { return kernel_float::flatten(*this); }

    /**
     * Apply the given function `F` to each element of this vector.
     */
    template <typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const {
        return kernel_float::for_each(*this, fun);
    }

    /**
     * Returns the result of `this + lhs * rhs`.
     *
     * The operation is performed using a single `kernel_float::fma` call, which may be faster then perform
     * the addition and multiplication separately.
     */
    template <typename L, typename R, typename T2 = promote_t<T, vector_value_type<L>, vector_value_type<R>>,
                typename E2 = broadcast_extent<E, vector_extent_type<L>, vector_extent_type<R>>>
    KERNEL_FLOAT_INLINE vector<T2, E2> add_mul(const L& lhs, const R& rhs) const {
        return ::kernel_float::fma(lhs, rhs, *this);
    }

    template <typename L, typename R, typename T2 = promote_t<T, vector_value_type<L>, vector_value_type<R>>,
                typename E2 = broadcast_extent<E, vector_extent_type<L>, vector_extent_type<R>>>
    KERNEL_FLOAT_INLINE vector<T2, E2> fma(const L& lhs, const R& rhs) const {
        return ::kernel_float::fma(lhs, rhs, *this);
    }
};

/**
 * Convert the given `input` into a vector. This function can perform one of the following actions:
 *
 * - For vectors `vec<T, N>`, it simply returns the original vector.
 * - For primitive types `T` (e.g., `int`, `float`, `double`), it returns a `vec<T, 1>`.
 * - For array-like types (e.g., `std::array<T, N>`, `T[N]`), it returns `vec<T, N>`.
 * - For vector-like types (e.g., `int2`, `dim3`), it returns `vec<T, N>`.
 */
template <typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_vec(V&& input) {
    return into_vector_impl<V>::call(static_cast<V&&>(input));
}

template <typename T>
using scalar = vector<T, extent<1>>;

template <typename T, size_t N>
using vec = vector<T, extent<N>>;

// clang-format off
template<typename T> using vec1 = vec<T, 1>;
template<typename T> using vec2 = vec<T, 2>;
template<typename T> using vec3 = vec<T, 3>;
template<typename T> using vec4 = vec<T, 4>;
template<typename T> using vec5 = vec<T, 5>;
template<typename T> using vec6 = vec<T, 6>;
template<typename T> using vec7 = vec<T, 7>;
template<typename T> using vec8 = vec<T, 8>;
// clang-format on

#define KERNEL_FLOAT_VECTOR_ALIAS(NAME, T) \
    template <size_t N>                    \
    using v##NAME = vec<T, N>;             \
    using NAME##1 = vec<T, 1>;             \
    using NAME##2 = vec<T, 2>;             \
    using NAME##3 = vec<T, 3>;             \
    using NAME##4 = vec<T, 4>;             \
    using NAME##5 = vec<T, 5>;             \
    using NAME##6 = vec<T, 6>;             \
    using NAME##7 = vec<T, 7>;             \
    using NAME##8 = vec<T, 8>;

KERNEL_FLOAT_VECTOR_ALIAS(int, int)
KERNEL_FLOAT_VECTOR_ALIAS(float, float)
KERNEL_FLOAT_VECTOR_ALIAS(double, double)

/**
 * Create a vector from a variable number of input values.
 *
 * The resulting vector type is determined by promoting the types of the input values into a common type.
 * The number of input values determines the dimension of the resulting vector.
 *
 * Example
 * =======
 * ```
 * auto v1 = make_vec(1.0f, 2.0f, 3.0f); // Creates a vec<float, 3> [1.0f, 2.0f, 3.0f]
 * auto v2 = make_vec(1, 2, 3, 4);       // Creates a vec<int, 4> [1, 2, 3, 4]
 * ```
 */
template <typename... Args>
KERNEL_FLOAT_INLINE vec<promote_t<Args...>, sizeof...(Args)> make_vec(Args&&... args) {
    using T = promote_t<Args...>;
    return vector_storage<T, sizeof...(Args)>{T(args)...};
};

#if defined(__cpp_deduction_guides)
// Deduction guide for `vector`
template <typename... Args>
vector(Args&&... args) -> vector<promote_t<Args...>, extent<sizeof...(Args)>>;

// Deduction guides for aliases are only supported from C++20
#if __cpp_deduction_guides >= 201907L
template <typename... Args>
vec(Args&&... args) -> vec<promote_t<Args...>, sizeof...(Args)>;
#endif
#endif

}  // namespace kernel_float

#endif
#ifndef KERNEL_FLOAT_FP16_H
#define KERNEL_FLOAT_FP16_H

#if KERNEL_FLOAT_FP16_AVAILABLE
// #define CUDA_NO_HALF (1)
// #define __CUDA_NO_HALF_OPERATORS__ (1)
// #define __CUDA_NO_HALF2_OPERATORS__ (1)
// #define __CUDA_NO_HALF_CONVERSIONS__ (1)

#if KERNEL_FLOAT_IS_CUDA
#include <cuda_fp16.h>
#elif KERNEL_FLOAT_IS_HIP
#include <hip/hip_fp16.h>
#endif

namespace kernel_float {

using half_t  = ::__half;
using half2_t = ::__half2;

template <>
struct preferred_vector_size<half_t> {
    static constexpr size_t value = 2;
};

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(half_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, half_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, half_t)

template <>
struct into_vector_impl<half2_t> {
    using value_type  = half_t;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<half_t, 2> call(half2_t input) { return {input.x, input.y}; }
};

namespace detail {
template <>
struct allow_float_fallback<half_t> {
    static constexpr bool value = true;
};
}  // namespace detail

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)                                              \
    namespace ops {                                                                                \
    template <>                                                                                    \
    struct NAME<half_t> {                                                                          \
        KERNEL_FLOAT_INLINE half_t operator()(half_t input) { return FUN1(input); }                \
    };                                                                                             \
    }                                                                                              \
    namespace detail {                                                                             \
    template <>                                                                                    \
    struct apply_impl<accurate_policy, ops::NAME<half_t>, 2, half_t, half_t> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<half_t>, half_t* result, const half_t* a) { \
            half2_t r = FUN2(half2_t{a[0], a[1]});                                                 \
            result[0] = r.x, result[1] = r.y;                                                      \
        }                                                                                          \
    };                                                                                             \
    }
#else
#define KERNEL_FLOAT_FP16_UNARY_FUN(NAME, FUN1, FUN2)
#endif

KERNEL_FLOAT_FP16_UNARY_FUN(sin, hsin, h2sin)
KERNEL_FLOAT_FP16_UNARY_FUN(cos, hcos, h2cos)

KERNEL_FLOAT_FP16_UNARY_FUN(exp, hexp, h2exp)
KERNEL_FLOAT_FP16_UNARY_FUN(exp2, hexp2, h2exp2)
KERNEL_FLOAT_FP16_UNARY_FUN(exp10, hexp10, h2exp10)
KERNEL_FLOAT_FP16_UNARY_FUN(log, hlog, h2log)
KERNEL_FLOAT_FP16_UNARY_FUN(log2, hlog2, h2log2)
KERNEL_FLOAT_FP16_UNARY_FUN(log10, hlog10, h2log2)

KERNEL_FLOAT_FP16_UNARY_FUN(sqrt, hsqrt, h2sqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(rsqrt, hrsqrt, h2rsqrt)
KERNEL_FLOAT_FP16_UNARY_FUN(rcp, hrcp, h2rcp)

KERNEL_FLOAT_FP16_UNARY_FUN(abs, __habs, __habs2)
KERNEL_FLOAT_FP16_UNARY_FUN(floor, hfloor, h2floor)
KERNEL_FLOAT_FP16_UNARY_FUN(ceil, hceil, h2ceil)
KERNEL_FLOAT_FP16_UNARY_FUN(rint, hrint, h2rint)
KERNEL_FLOAT_FP16_UNARY_FUN(trunc, htrunc, h2trunc)
KERNEL_FLOAT_FP16_UNARY_FUN(negate, __hneg, __hneg2)

#if KERNEL_FLOAT_IS_DEVICE
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)                                                              \
    namespace ops {                                                                                                 \
    template <>                                                                                                     \
    struct NAME<half_t> {                                                                                           \
        KERNEL_FLOAT_INLINE half_t operator()(half_t left, half_t right) const {                                    \
            return ops::cast<decltype(FUN1(left, right)), half_t>{}(FUN1(left, right));                             \
        }                                                                                                           \
    };                                                                                                              \
    }                                                                                                               \
    namespace detail {                                                                                              \
    template <>                                                                                                     \
    struct apply_impl<accurate_policy, ops::NAME<half_t>, 2, half_t, half_t, half_t> {                              \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<half_t>, half_t* result, const half_t* a, const half_t* b) { \
            half2_t r = FUN2(half2_t{a[0], a[1]}, half2_t{b[0], b[1]});                                             \
            result[0] = r.x, result[1] = r.y;                                                                       \
        }                                                                                                           \
    };                                                                                                              \
    }
#else
#define KERNEL_FLOAT_FP16_BINARY_FUN(NAME, FUN1, FUN2)
#endif

// There are not available in HIP
#if KERNEL_FLOAT_IS_CUDA
KERNEL_FLOAT_FP16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_FP16_BINARY_FUN(max, __hmax, __hmax2)
#endif

KERNEL_FLOAT_FP16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_FP16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_FP16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_FP16_BINARY_FUN(divide, __hdiv, __h2div)

KERNEL_FLOAT_FP16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_FP16_BINARY_FUN(not_equal_to, __hneu, __hneu2)
KERNEL_FLOAT_FP16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_FP16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_FP16_BINARY_FUN(greater_equal, __hge, __hgt2)

#if KERNEL_FLOAT_IS_DEVICE
namespace ops {
template <>
struct fma<half_t> {
    KERNEL_FLOAT_INLINE half_t operator()(half_t a, half_t b, half_t c) const { return __hfma(a, b, c); }
};
}  // namespace ops

namespace detail {
template <>
struct apply_impl<accurate_policy, ops::fma<half_t>, 2, half_t, half_t, half_t, half_t> {
    KERNEL_FLOAT_INLINE static void call(ops::fma<half_t>, half_t* result, const half_t* a, const half_t* b, const half_t* c) {
        half2_t r = __hfma2(half2_t{a[0], a[1]}, half2_t{b[0], b[1]}, half2_t{c[0], c[1]});
        result[0] = r.x, result[1] = r.y;
    }
};

// clang-format off
#define KERNEL_FLOAT_FAST_FP16_DISPATCH(OP)                                                         \
template<size_t N>                                                                              \
struct apply_impl<fast_policy, ops::OP<half_t>, N, half_t, half_t> {                            \
KERNEL_FLOAT_INLINE static void                                                             \
call(ops::OP<half_t>, half_t* output, const half_t* input) {                                \
float v[N];                                                                             \
map_impl<fast_policy, ops::cast<half_t, float>, N, float, half_t>::call({}, v, input);  \
map_impl<fast_policy, ops::OP<float>, N, float, float>::call({}, v, v);                 \
map_impl<fast_policy, ops::cast<float, half_t>, N, half_t, float>::call({}, output, v); \
}                                                                                           \
};
// clang-format on

KERNEL_FLOAT_FAST_F32_MAP(KERNEL_FLOAT_FAST_FP16_DISPATCH)
}  // namespace detail
#endif

#define KERNEL_FLOAT_FP16_CAST(T, TO_HALF, FROM_HALF)                        \
    namespace ops {                                                          \
    template <>                                                              \
    struct cast<T, half_t> {                                                 \
        KERNEL_FLOAT_INLINE half_t operator()(T input) { return TO_HALF; }   \
    };                                                                       \
    template <>                                                              \
    struct cast<half_t, T> {                                                 \
        KERNEL_FLOAT_INLINE T operator()(half_t input) { return FROM_HALF; } \
    };                                                                       \
    }

// Only CUDA has a special `__double2half` intrinsic
#if KERNEL_FLOAT_IS_HIP
#define KERNEL_FLOAT_FP16_CAST_FWD(T) KERNEL_FLOAT_FP16_CAST(T, static_cast<_Float16>(input), static_cast<T>(input))

KERNEL_FLOAT_FP16_CAST_FWD(double)
KERNEL_FLOAT_FP16_CAST_FWD(float)

KERNEL_FLOAT_FP16_CAST_FWD(char)
KERNEL_FLOAT_FP16_CAST_FWD(signed char)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned char)

KERNEL_FLOAT_FP16_CAST_FWD(signed short)
KERNEL_FLOAT_FP16_CAST_FWD(signed int)
KERNEL_FLOAT_FP16_CAST_FWD(signed long)
KERNEL_FLOAT_FP16_CAST_FWD(signed long long)

KERNEL_FLOAT_FP16_CAST_FWD(unsigned short)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned int)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned long)
KERNEL_FLOAT_FP16_CAST_FWD(unsigned long long)
#else
KERNEL_FLOAT_FP16_CAST(double, __double2half(input), double(__half2float(input)));
KERNEL_FLOAT_FP16_CAST(float, __float2half(input), __half2float(input));

// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_FP16_CAST(char, __int2half_rn(input), (char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed char, __int2half_rn(input), (signed char)__half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned char, __int2half_rn(input), (unsigned char)__half2int_rz(input));

KERNEL_FLOAT_FP16_CAST(signed short, __short2half_rn(input), __half2short_rz(input));
KERNEL_FLOAT_FP16_CAST(signed int, __int2half_rn(input), __half2int_rz(input));
KERNEL_FLOAT_FP16_CAST(signed long, __ll2half_rn(input), (signed long)(__half2ll_rz(input)));
KERNEL_FLOAT_FP16_CAST(signed long long, __ll2half_rn(input), __half2ll_rz(input));

KERNEL_FLOAT_FP16_CAST(unsigned short, __ushort2half_rn(input), __half2ushort_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned int, __uint2half_rn(input), __half2uint_rz(input));
KERNEL_FLOAT_FP16_CAST(unsigned long, __ull2half_rn(input), (unsigned long)(__half2ull_rz(input)));
KERNEL_FLOAT_FP16_CAST(unsigned long long, __ull2half_rn(input), __half2ull_rz(input));
#endif

KERNEL_FLOAT_VECTOR_ALIAS(half, half_t)
// KERNEL_FLOAT_TYPE_ALIAS(float16x, half_t)
// KERNEL_FLOAT_TYPE_ALIAS(f16x, half_t)

}  // namespace kernel_float

#endif

#endif  // KERNEL_FLOAT_FP16_H
#ifndef KERNEL_FLOAT_BF16_H
#define KERNEL_FLOAT_BF16_H

#if KERNEL_FLOAT_BF16_AVAILABLE
// #define CUDA_NO_BFLOAT16 (1)
// #define __CUDA_NO_BFLOAT16_OPERATORS__ (1)
// #define __CUDA_NO_BFLOAT162_OPERATORS__ (1)
// #define __CUDA_NO_BFLOAT16_CONVERSIONS__ (1)

#if KERNEL_FLOAT_IS_CUDA
#include <cuda_bf16.h>
#elif KERNEL_FLOAT_IS_HIP
#include <hip/hip_bf16.h>
#endif

namespace kernel_float {

#if KERNEL_FLOAT_IS_CUDA
using bfloat16_t   = __nv_bfloat16;
using bfloat16x2_t = __nv_bfloat162;
#elif KERNEL_FLOAT_IS_HIP
using bfloat16_t   = __hip_bfloat16;
using bfloat16x2_t = __hip_bfloat162;
#endif

#if KERNEL_FLOAT_IS_CUDA && __CUDA_ARCH__ >= 800
#define KERNEL_FLOAT_BF16_OPS_SUPPORTED 1
#endif

template <>
struct preferred_vector_size<bfloat16_t> {
    static constexpr size_t value = 2;
};

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(bfloat16_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, bfloat16_t)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, bfloat16_t)

template <>
struct into_vector_impl<bfloat16x2_t> {
    using value_type  = bfloat16_t;
    using extent_type = extent<2>;

    KERNEL_FLOAT_INLINE
    static vector_storage<bfloat16_t, 2> call(bfloat16x2_t input) { return {input.x, input.y}; }
};

namespace detail {
template <>
struct allow_float_fallback<bfloat16_t> {
    static constexpr bool value = true;
};
};  // namespace detail

#define KERNEL_FLOAT_BF16_UNARY_FUN(NAME, FUN1, FUN2)                                                          \
    namespace ops {                                                                                            \
    template <>                                                                                                \
    struct NAME<bfloat16_t> {                                                                                  \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t input) { return FUN1(input); }                    \
    };                                                                                                         \
    }                                                                                                          \
    namespace detail {                                                                                         \
    template <>                                                                                                \
    struct apply_impl<accurate_policy, ops::NAME<bfloat16_t>, 2, bfloat16_t, bfloat16_t> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<bfloat16_t>, bfloat16_t* result, const bfloat16_t* a) { \
            bfloat16x2_t r = FUN2(bfloat16x2_t{a[0], a[1]});                                                   \
            result[0] = r.x, result[1] = r.y;                                                                  \
        }                                                                                                      \
    };                                                                                                         \
    }

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_BF16_UNARY_FUN(sin, ::hsin, ::h2sin)
KERNEL_FLOAT_BF16_UNARY_FUN(cos, ::hcos, ::h2cos)

KERNEL_FLOAT_BF16_UNARY_FUN(exp, ::hexp, ::h2exp)
KERNEL_FLOAT_BF16_UNARY_FUN(exp2, ::hexp2, ::h2exp2)
KERNEL_FLOAT_BF16_UNARY_FUN(exp10, ::hexp10, ::h2exp10)
KERNEL_FLOAT_BF16_UNARY_FUN(log, ::hlog, ::h2log)
KERNEL_FLOAT_BF16_UNARY_FUN(log2, ::hlog2, ::h2log2)
KERNEL_FLOAT_BF16_UNARY_FUN(log10, ::hlog10, ::h2log2)

KERNEL_FLOAT_BF16_UNARY_FUN(sqrt, ::hsqrt, ::h2sqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(rsqrt, ::hrsqrt, ::h2rsqrt)
KERNEL_FLOAT_BF16_UNARY_FUN(rcp, ::hrcp, ::h2rcp)

KERNEL_FLOAT_BF16_UNARY_FUN(abs, ::__habs, ::__habs2)
KERNEL_FLOAT_BF16_UNARY_FUN(floor, ::hfloor, ::h2floor)
KERNEL_FLOAT_BF16_UNARY_FUN(ceil, ::hceil, ::h2ceil)
KERNEL_FLOAT_BF16_UNARY_FUN(rint, ::hrint, ::h2rint)
KERNEL_FLOAT_BF16_UNARY_FUN(trunc, ::htrunc, ::h2trunc)
KERNEL_FLOAT_BF16_UNARY_FUN(negate, ::__hneg, ::__hneg2)

// For some reason, HIP struggles with the functions `__habs` and `__hneg`. We define them here using bitwise ops.
// For CUDA, we can just use the regular bfloat16 functions (see above).
#elif KERNEL_FLOAT_IS_HIP
KERNEL_FLOAT_INLINE __hip_bfloat16 hip_habs(const __hip_bfloat16 a) {
__hip_bfloat16 res = a;
res.data &= 0x7FFF;
return res;
}

KERNEL_FLOAT_INLINE __hip_bfloat16 hip_hneg(const __hip_bfloat16 a) {
__hip_bfloat16 res = a;
res.data ^= 0x8000;
return res;
}

KERNEL_FLOAT_INLINE __hip_bfloat162 hip_habs2(const __hip_bfloat162 a) { return {hip_habs(a.x), hip_habs(a.y)}; }

KERNEL_FLOAT_INLINE __hip_bfloat162 hip_hneg2(const __hip_bfloat162 a) { return {hip_hneg(a.x), hip_hneg(a.y)}; }

KERNEL_FLOAT_BF16_UNARY_FUN(abs, hip_habs, hip_habs2)
KERNEL_FLOAT_BF16_UNARY_FUN(negate, hip_hneg, hip_hneg2)
#endif

#define KERNEL_FLOAT_BF16_BINARY_FUN(NAME, FUN1, FUN2)                                                                              \
    namespace ops {                                                                                                                 \
    template <>                                                                                                                     \
    struct NAME<bfloat16_t> {                                                                                                       \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t left, bfloat16_t right) const {                                        \
            return ops::cast<decltype(FUN1(left, right)), bfloat16_t>{}(FUN1(left, right));                                         \
        }                                                                                                                           \
    };                                                                                                                              \
    }                                                                                                                               \
    namespace detail {                                                                                                              \
    template <>                                                                                                                     \
    struct apply_impl<accurate_policy, ops::NAME<bfloat16_t>, 2, bfloat16_t, bfloat16_t, bfloat16_t> {                              \
        KERNEL_FLOAT_INLINE static void call(ops::NAME<bfloat16_t>, bfloat16_t* result, const bfloat16_t* a, const bfloat16_t* b) { \
            bfloat16x2_t r = FUN2(bfloat16x2_t{a[0], a[1]}, bfloat16x2_t{b[0], b[1]});                                              \
            result[0] = r.x, result[1] = r.y;                                                                                       \
        }                                                                                                                           \
    };                                                                                                                              \
    }

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_BF16_BINARY_FUN(add, __hadd, __hadd2)
KERNEL_FLOAT_BF16_BINARY_FUN(subtract, __hsub, __hsub2)
KERNEL_FLOAT_BF16_BINARY_FUN(multiply, __hmul, __hmul2)
KERNEL_FLOAT_BF16_BINARY_FUN(divide, __hdiv, __h2div)
KERNEL_FLOAT_BF16_BINARY_FUN(min, __hmin, __hmin2)
KERNEL_FLOAT_BF16_BINARY_FUN(max, __hmax, __hmax2)

KERNEL_FLOAT_BF16_BINARY_FUN(equal_to, __heq, __heq2)
KERNEL_FLOAT_BF16_BINARY_FUN(not_equal_to, __hneu, __hneu2)
KERNEL_FLOAT_BF16_BINARY_FUN(less, __hlt, __hlt2)
KERNEL_FLOAT_BF16_BINARY_FUN(less_equal, __hle, __hle2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater, __hgt, __hgt2)
KERNEL_FLOAT_BF16_BINARY_FUN(greater_equal, __hge, __hgt2)
#endif

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
namespace ops {
template <>
struct fma<bfloat16_t> {
    KERNEL_FLOAT_INLINE bfloat16_t operator()(bfloat16_t a, bfloat16_t b, bfloat16_t c) const { return __hfma(a, b, c); }
};
}  // namespace ops

namespace detail {
template <>
struct apply_impl<accurate_policy, ops::fma<bfloat16_t>, 2, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t> {
    KERNEL_FLOAT_INLINE static void call(ops::fma<bfloat16_t>, bfloat16_t* result, const bfloat16_t* a, const bfloat16_t* b,
                                            const bfloat16_t* c) {
        bfloat16x2_t r = __hfma2(bfloat16x2_t{a[0], a[1]}, bfloat16x2_t{b[0], b[1]}, bfloat16x2_t{c[0], c[1]});
        result[0] = r.x, result[1] = r.y;
    }
};

// clang-format off
#define KERNEL_FLOAT_FAST_BF16_DISPATCH(OP)                                                                 \
template<size_t N>                                                                                      \
struct apply_impl<fast_policy, ops::OP<bfloat16_t>, N, bfloat16_t, bfloat16_t> {                        \
KERNEL_FLOAT_INLINE static void                                                                     \
call(ops::OP<bfloat16_t>, bfloat16_t* output, const bfloat16_t* input) {                            \
float v[N];                                                                                     \
map_impl<fast_policy, ops::cast<bfloat16_t, float>, N, float, bfloat16_t>::call({}, v, input);  \
map_impl<fast_policy, ops::OP<float>, N, float, float>::call({}, v, v);                         \
map_impl<fast_policy, ops::cast<float, bfloat16_t>, N, bfloat16_t, float>::call({}, output, v); \
}                                                                                                   \
};
// clang-format on

KERNEL_FLOAT_FAST_F32_MAP(KERNEL_FLOAT_FAST_BF16_DISPATCH)
}  // namespace detail
#endif

#define KERNEL_FLOAT_BF16_CAST(T, TO_HALF, FROM_HALF)                            \
    namespace ops {                                                              \
    template <>                                                                  \
    struct cast<T, bfloat16_t> {                                                 \
        KERNEL_FLOAT_INLINE bfloat16_t operator()(T input) { return TO_HALF; }   \
    };                                                                           \
    template <>                                                                  \
    struct cast<bfloat16_t, T> {                                                 \
        KERNEL_FLOAT_INLINE T operator()(bfloat16_t input) { return FROM_HALF; } \
    };                                                                           \
    }

KERNEL_FLOAT_BF16_CAST(float, __float2bfloat16(input), __bfloat162float(input))
KERNEL_FLOAT_BF16_CAST(double, __double2bfloat16(input), __bfloat162float(input))

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
// clang-format off
// there are no official char casts. Instead, cast to int and then to char
KERNEL_FLOAT_BF16_CAST(char, __int2bfloat16_rn(input), (char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(signed char, __int2bfloat16_rn(input), (signed char)__bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned char, __int2bfloat16_rn(input), (unsigned char)__bfloat162int_rz(input));

KERNEL_FLOAT_BF16_CAST(signed short, __short2bfloat16_rn(input), __bfloat162short_rz(input));
KERNEL_FLOAT_BF16_CAST(signed int, __int2bfloat16_rn(input), __bfloat162int_rz(input));
KERNEL_FLOAT_BF16_CAST(signed long, __ll2bfloat16_rn(input), (signed long)(__bfloat162ll_rz(input)));
KERNEL_FLOAT_BF16_CAST(signed long long, __ll2bfloat16_rn(input), __bfloat162ll_rz(input));

KERNEL_FLOAT_BF16_CAST(unsigned short, __ushort2bfloat16_rn(input), __bfloat162ushort_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned int, __uint2bfloat16_rn(input), __bfloat162uint_rz(input));
KERNEL_FLOAT_BF16_CAST(unsigned long, __ull2bfloat16_rn(input), (unsigned long)(__bfloat162ull_rz(input)));
KERNEL_FLOAT_BF16_CAST(unsigned long long, __ull2bfloat16_rn(input), __bfloat162ull_rz(input));
// clang-format on
#endif

#if KERNEL_FLOAT_IS_CUDA
// KERNEL_FLOAT_BF16_CAST(
//     bool,
//     __nv_bfloat16_raw {input ? (unsigned short)0 : (unsigned short)0x3C00},
//     (__nv_bfloat16_raw(input).x & 0x7FFF) != 0);
#elif KERNEL_FLOAT_IS_HIP
KERNEL_FLOAT_BF16_CAST(bool, __hip_bfloat16{input ? (unsigned short)0 : (unsigned short)0x3C00}, (__hip_bfloat16(input).data & 0x7FFF) != 0);
#endif

KERNEL_FLOAT_VECTOR_ALIAS(bfloat16x, bfloat16_t)
// KERNEL_FLOAT_TYPE_ALIAS(float16x, bfloat16_t)
// KERNEL_FLOAT_TYPE_ALIAS(f16x, bfloat16_t)

}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE

namespace kernel_float {
template <>
struct promote_type<bfloat16_t, half_t> {
    using type = float;
};

template <>
struct promote_type<half_t, bfloat16_t> {
    using type = float;
};

}  // namespace kernel_float

#endif  // KERNEL_FLOAT_FP16_AVAILABLE
#endif

#endif  // KERNEL_FLOAT_BF16_H
#pragma once

namespace kernel_float {

namespace approx {

static_assert(sizeof(unsigned int) * 8 == 32, "invalid size of unsigned int");
static_assert(sizeof(unsigned short) * 8 == 16, "invalid size of unsigned short");
using uint32_t = unsigned int;
using uint16_t = unsigned short;

template <typename T, typename U>
KERNEL_FLOAT_DEVICE T transmute(const U& input) {
    static_assert(sizeof(T) == sizeof(U), "types must have equal size");
    T result{};
    ::memcpy(&result, &input, sizeof(T));
    return result;
}

KERNEL_FLOAT_DEVICE uint32_t bitwise_if_else(uint32_t condition, uint32_t if_true, uint32_t if_false) {
    uint32_t result;

#if KERNEL_FLOAT_IS_CUDA
    // equivalent to (condition & if_true) | ((~condition) & if_false)
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(condition), "r"(if_true), "r"(if_false));
#else
result = (condition & if_true) | ((~condition) & if_false);
#endif
    return result;
}

template <typename T, typename T2>
KERNEL_FLOAT_DEVICE T2 eval_poly_recur(T2 y, T2 x) {
    return y;
}

template <typename T, typename T2, typename... TRest>
KERNEL_FLOAT_DEVICE T2 eval_poly_recur(T2 y, T2 x, T coef, TRest... coefs) {
    y = __hfma2(x, y, T2{coef, coef});
    return eval_poly_recur<T>(y, x, coefs...);
}

template <typename T, typename T2, typename... TRest>
KERNEL_FLOAT_DEVICE T2 eval_poly(T2 x, T coef, TRest... coefs) {
    return eval_poly_recur<T>(T2{coef, coef}, x, coefs...);
}

#define KERNEL_FLOAT_DEFINE_POLY(NAME, N, ...)     \
    template <typename T>                          \
    struct NAME<T, N> {                            \
        template <typename T2>                     \
        static KERNEL_FLOAT_DEVICE T2 call(T2 x) { \
            return eval_poly<T>(x, __VA_ARGS__);   \
        }                                          \
    };

template <typename T, size_t N>
struct sin_poly : sin_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 1, 1.365)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 2, -21.56, 5.18)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 3, 53.53, -38.06, 6.184)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 4, -56.1, 77.94, -41.1, 6.277)
KERNEL_FLOAT_DEFINE_POLY(sin_poly, 5, 32.78, -74.5, 81.4, -41.34, 6.28)

template <typename T, size_t N>
struct cos_poly : cos_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 1, 0.0)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 2, -8.0, 0.6943)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 3, 38.94, -17.5, 0.9707)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 4, -59.66, 61.12, -19.56, 0.9985)
KERNEL_FLOAT_DEFINE_POLY(cos_poly, 5, 45.66, -82.4, 64.7, -19.73, 1.0)

template <typename T, size_t N>
struct asin_poly : asin_poly<T, 5> {};
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 1, 1.531)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 2, -0.169, 1.567)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 3, 0.05167, -0.2057, 1.57)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 4, -0.02103, 0.077, -0.2129, 1.57)
KERNEL_FLOAT_DEFINE_POLY(asin_poly, 5, 0.009796, -0.03772, 0.0857, -0.2142, 1.57)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_DEVICE half2_t flipsign(half2_t input, half2_t sign) {
    // Flip signbit of input when sign<0
    uint32_t result;

#if KERNEL_FLOAT_IS_CUDA
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(result) : "r"(0x80008000), "r"(transmute<uint32_t>(sign)), "r"(transmute<uint32_t>(input)));
#else
result = uint32_t(transmute<uint32_t>(sign) & 0x80008000) ^ transmute<uint32_t>(input);
#endif

    return transmute<half2_t>(result);
}

KERNEL_FLOAT_DEVICE uint32_t half2_gt_mask(half2_t a, half2_t b) {
    uint32_t val;
#if KERNEL_FLOAT_IS_CUDA
    uint32_t ai = *(reinterpret_cast<const uint32_t*>(&a));
    uint32_t bi = *(reinterpret_cast<const uint32_t*>(&b));
    asm("{ set.gt.u32.f16x2 %0,%1,%2;\n}" : "=r"(val) : "r"(ai), "r"(bi));
#else
val = transmute<uint32_t>(make_short2(a.x > b.x ? ~0 : 0, a.y > b.y ? ~0 : 0));
#endif
    return val;
}

KERNEL_FLOAT_INLINE half2_t make_half2(half x) { return {x, x}; }

KERNEL_FLOAT_DEVICE half2_t normalize_trig_input(half2_t x) {
    /* Using rint is too slow. Round using floating-point magic instead. */
    // half2_t x = arg * make_half2(-0.15915494309);
    // return __hfma2(arg, make_half2(0.15915494309),  h2rint(x));

    // 1/(2pi) = 0.15915494309189535
    static constexpr double ONE_OVER_TWOPI = 0.15915494309189535;
    static constexpr double OFFSET         = -2042.0;

    half2_t ws = __hfma2(x, make_half2(-ONE_OVER_TWOPI), make_half2(-OFFSET)) + make_half2(OFFSET);
    return __hfma2(x, make_half2(ONE_OVER_TWOPI), ws);
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t cos(half2_t x) {
    half2_t xf = normalize_trig_input(x);
    return cos_poly<half, Iter + 1>::call(__hmul2(xf, xf));
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t sin(half2_t x) {
    half2_t xf = normalize_trig_input(x);
    return sin_poly<half, Iter>::call(__hmul2(xf, xf)) * xf;
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t rcp(half2_t x) {
    // Flip bits
    uint32_t m = ~transmute<uint32_t>(x);

    // Multiply by bias (add contant)
    half2_t y = transmute<half2_t>(uint32_t(0x776d776d) + m);

#pragma unroll
    for (int i = 0; i < Iter; i++) {
        // y += y * (1 - x * y)
        y = __hfma2(y, __hfma2(-x, y, make_half2(1.0)), y);
    }

    return y;
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t rsqrt(half2_t x) {
    // A small number added such that rsqrt(0) does not return NaN
    static constexpr double EPS = 0.00000768899917602539;

    // Set top and bottom bits for both halfs, then shift by 1, then invert
    uint32_t r = ~((uint32_t(transmute<uint32_t>(x) >> 1)) | ~uint32_t(0x3fff3fff));

    // Add bias
    static constexpr uint32_t BIAS = 0x199c199c;
    half2_t y                      = transmute<half2_t>(uint32_t(r) + BIAS);

    // Newton-Raphson iterations
#pragma unroll
    for (int i = 0; i < Iter; i++) {
        half2_t half_x     = __hfma2(make_half2(-0.5), x, make_half2(-EPS));
        half2_t correction = __hfma2(half_x, y * y, make_half2(0.5));
        y                  = __hfma2(correction, y, y);  // y += y * correction
    }

    return y;
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t sqrt(half2_t x) {
    if (Iter == 1) {
        half2_t y = rsqrt<0>(x);

        // This method uses only 4 muls, instead of 5 muls when using `arg * approx_rsqrt<1>(arg)`
        half2_t xy = x * y;
        return xy * __hfma2(make_half2(-0.5) * y, xy, make_half2(1.5));
    }

    return x * rsqrt<Iter>(x);
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t asin(half2_t x) {
    static constexpr double HALF_PI = 1.57079632679;
    auto abs_x                      = __habs2(x);
    auto v                          = asin_poly<half, Iter + 1>::call(abs_x);
    auto abs_y                      = __hfma2(-v, sqrt<Iter>(make_half2(1) - abs_x), make_half2(HALF_PI));
    return flipsign(abs_y, x);
}

template <int Iter>
KERNEL_FLOAT_DEVICE half2_t acos(half2_t x) {
    static constexpr double HALF_PI = 1.57079632679;
    return make_half2(HALF_PI) - asin<Iter>(x);
}

template <int Deg>
KERNEL_FLOAT_DEVICE half2_t exp(half2_t x) {
    half2_t y;

    if (Deg == 0) {
        // Bring the value to range [32, 64]
        // 1.442 = 1/log(2)
        // 46.969 = 32.5/log(2)
        half2_t m = __hfma2(x, make_half2(1.442), make_half2(46.9375));

        // Transmute to int, shift higher mantissa bits into exponent field.
        y = transmute<half2_t>((transmute<uint32_t>(m) & 0x03ff03ff) << 5);
    } else {
        // Add a large number to round to an integer
        half2_t v = __hfma2(x, make_half2(1.442), make_half2(1231.0));

        // The exponent is now in the lower 5 bits. Shift that into the exponent field.
        half2_t exp = transmute<half2_t>((transmute<uint32_t>(v) & 0x001f001f) << 10);

        // The fractional part can be obtained from "1231-v".
        // 0.6934 = log(2)
        half2_t frac = __hfma2(make_half2(1231.0) - v, make_half2(0.6934), x);

        // This is the Taylor expansion of "exp(x)-1" around 0
        half2_t adjust;
        if (Deg == 1) {
            adjust = frac;
        } else if (Deg == 2) {
            // adjust = frac + 0.5 * frac^2
            adjust = __hfma2(frac, __hmul2(frac, make_half2(0.5)), frac);
        } else /* if (Deg == 2) */ {
            // adjust = frac + 0.5 * frac^2 + 1/6 * frac^3
            adjust = __hfma2(frac, __hmul2(__hfma2(frac, make_half2(0.1666), make_half2(0.5)), frac), frac);
        }

        // result = exp * (adjust + 1)
        y = __hfma2(exp, adjust, exp);
    }

    // Values below -10.39 (= -15*log(2)) become zero
    uint32_t zero_mask = half2_gt_mask(x, make_half2(-10.390625));
    return transmute<half2_t>(zero_mask & transmute<uint32_t>(y));
}

template <int = 0>
KERNEL_FLOAT_DEVICE half2_t log(half2_t arg) {
    // Shift exponent field into mantissa bits. Fill exponent bits with 0x5000 (= 32.0)
    uint32_t bits = bitwise_if_else(0x03ff03ff, transmute<uint32_t>(arg) >> 5, 0x50005000);

    // 0.6934 = log(2)
    // 32.53 = 46.969*log(2)
    return __hfma2(transmute<half2_t>(bits), make_half2(0.6934), make_half2(-32.53125));
}

template <int Deg>
KERNEL_FLOAT_DEVICE half2_t tanh(half2_t x) {
    if (Deg == 0) {
        return x * rcp<0>(make_half2(0.2869) + __habs2(x));
    } else {
        auto c0  = make_half2(0.4531);
        auto c1  = make_half2(0.5156);
        auto x2b = __hfma2(x, x, c1);
        return (x * x2b) * rcp<Deg>(__hfma2(x2b, __habs2(x), c0));
    }
}

#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_DEVICE bfloat16x2_t make_bfloat162(bfloat16_t x) { return {x, x}; }

KERNEL_FLOAT_DEVICE bfloat16x2_t make_bfloat162(double x) { return {__double2bfloat16(x), __double2bfloat16(x)}; }

KERNEL_FLOAT_DEVICE bfloat16x2_t normalize_trig_input(bfloat16x2_t x) {
    static constexpr double ONE_OVER_TWOPI = 0.15915494309189535;
    static constexpr double OFFSET         = -2042.0;

    bfloat16x2_t ws = __hadd2(__hfma2(x, make_bfloat162(-ONE_OVER_TWOPI), make_bfloat162(-OFFSET)), make_bfloat162(OFFSET));
    return __hfma2(x, make_bfloat162(ONE_OVER_TWOPI), ws);
}

template <int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t cos(bfloat16x2_t x) {
    bfloat16x2_t xf = normalize_trig_input(x);
    return cos_poly<bfloat16_t, Iter + 1>::call(__hmul2(xf, xf));
}

template <int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t sin(bfloat16x2_t x) {
    bfloat16x2_t xf = normalize_trig_input(x);
    return __hmul2(sin_poly<bfloat16_t, Iter>::call(__hmul2(xf, xf)), xf);
}

template <int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t rcp(bfloat16x2_t x) {
    bfloat16x2_t y = transmute<bfloat16x2_t>(uint32_t(0x7ef07ef0) + ~transmute<uint32_t>(x));

#pragma unroll
    for (int i = 0; i < Iter; i++) {
        y = __hfma2(y, __hfma2(__hneg2(x), y, make_bfloat162(1.0)), y);
    }

    return y;
}

template <int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t rsqrt(bfloat16x2_t x) {
    // Set top and bottom bits for both halfs, then shift by 1, then invert
    uint32_t r = ~((uint32_t(transmute<uint32_t>(x) >> 1)) | ~uint32_t(0x3fff3fff));

    // Add bias (0x1f36)
    bfloat16x2_t y = transmute<bfloat16x2_t>(uint32_t(r) + uint32_t(0x1f361f36));

    // Newton-Raphson iterations
#pragma unroll
    for (int i = 0; i < Iter; i++) {
        bfloat16x2_t half_x     = __hmul2(make_bfloat162(-0.5), x);
        bfloat16x2_t correction = __hfma2(half_x, __hmul2(y, y), make_bfloat162(0.5));
        y                       = __hfma2(correction, y, y);  // y += y * correction
    }

    return y;
}

template <int Iter>
KERNEL_FLOAT_DEVICE bfloat16x2_t sqrt(bfloat16x2_t x) {
    return __hmul2(x, rsqrt<Iter>(x));
}

template <int = 0>
KERNEL_FLOAT_DEVICE bfloat16x2_t exp(bfloat16x2_t arg) {
    static constexpr float SCALE   = 1.44272065994 / 256.0;
    static constexpr float OFFSET  = 382.4958400542335;
    static constexpr float MINIMUM = 382;

    float a = fmaxf(fmaf(__bfloat162float(arg.x), SCALE, OFFSET), MINIMUM);
    float b = fmaxf(fmaf(__bfloat162float(arg.y), SCALE, OFFSET), MINIMUM);

    return {transmute<bfloat16_t>(uint16_t(transmute<uint32_t>(a))), transmute<bfloat16_t>(uint16_t(transmute<uint32_t>(b)))};
}
#endif
}  // namespace approx

namespace detail {
template <int Level, typename F, typename T>
struct apply_impl<approx_level_policy<Level>, F, 1, T, T> {
    KERNEL_FLOAT_INLINE static void call(F fun, T* output, const T* input) {
        T in2[2], out2[2];
        in2[0] = input[0];
        apply_impl<approx_level_policy<Level>, F, 2, T, T>::call(fun, out2, in2);
        output[0] = out2[0];
    }
};
}  // namespace detail

#define KERNEL_FLOAT_DEFINE_APPROX_IMPL(T, FUN, DEFAULT_LEVEL)                                                                        \
    namespace detail {                                                                                                                \
    template <int Degree>                                                                                                             \
    struct apply_impl<approx_level_policy<Degree>, ops::FUN<T>, 2, T, T> {                                                            \
        KERNEL_FLOAT_INLINE static void call(ops::FUN<T>, T* output, const T* input) {                                                \
            auto res  = approx::FUN<Degree>({input[0], input[1]});                                                                    \
            output[0] = res.x;                                                                                                        \
            output[1] = res.y;                                                                                                        \
        }                                                                                                                             \
    };                                                                                                                                \
                                                                                                                                        \
    template <>                                                                                                                       \
    struct apply_impl<approx_policy, ops::FUN<T>, 2, T, T> : apply_impl<approx_level_policy<DEFAULT_LEVEL>, ops::FUN<T>, 2, T, T> {}; \
    }

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, sin, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, cos, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, rsqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, sqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, rcp, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, exp, 0)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, log, 0)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, asin, 2)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, acos, 2)
#endif

#if KERNEL_FLOAT_BF16_OPS_SUPPORTED
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, cos, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, sin, 4)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, rcp, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, rsqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, sqrt, 1)
KERNEL_FLOAT_DEFINE_APPROX_IMPL(bfloat16_t, exp, 0)
// KERNEL_FLOAT_DEFINE_APPROX_IMPL(half_t, log, 0)
#endif

#define KERNEL_FLOAT_DEFINE_APPROX_FUN(FUN)                                             \
    template <int Level = -1, typename V>                                               \
    KERNEL_FLOAT_INLINE into_vector_type<V> approx_##FUN(const V& args) {               \
        return map<approx_level_policy<Level>>(ops::FUN<vector_value_type<V>>{}, args); \
    }

KERNEL_FLOAT_DEFINE_APPROX_FUN(sin)
KERNEL_FLOAT_DEFINE_APPROX_FUN(cos)
KERNEL_FLOAT_DEFINE_APPROX_FUN(rsqrt)
KERNEL_FLOAT_DEFINE_APPROX_FUN(sqrt)
KERNEL_FLOAT_DEFINE_APPROX_FUN(rcp)
KERNEL_FLOAT_DEFINE_APPROX_FUN(exp)
KERNEL_FLOAT_DEFINE_APPROX_FUN(log)

}  // namespace kernel_float
#ifndef KERNEL_FLOAT_FP8_H
#define KERNEL_FLOAT_FP8_H

#if KERNEL_FLOAT_FP8_AVAILABLE
#include <cuda_fp8.h>

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_fp8_e4m3)

KERNEL_FLOAT_DEFINE_PROMOTED_FLOAT(__nv_fp8_e5m2)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(float, __nv_fp8_e5m2)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(double, __nv_fp8_e5m2)

namespace detail {
template <>
struct allow_float_fallback<__nv_fp8_e4m3> {
    static constexpr bool value = true;
};

template <>
struct allow_float_fallback<__nv_fp8_e5m2> {
    static constexpr bool value = true;
};
}  // namespace detail
}  // namespace kernel_float

#define KERNEL_FLOAT_FP8_CAST(T)                                                             \
    namespace ops {                                                                          \
    template <>                                                                              \
    struct cast<T, __nv_fp8_e4m3> {                                                          \
        KERNEL_FLOAT_INLINE __nv_fp8_e4m3 operator()(T v) const { return __nv_fp8_e4m3(v); } \
    };                                                                                       \
                                                                                                \
    template <>                                                                              \
    struct cast<__nv_fp8_e4m3, T> {                                                          \
        KERNEL_FLOAT_INLINE T operator()(__nv_fp8_e4m3 v) const { return T(v); }             \
    };                                                                                       \
                                                                                                \
    template <>                                                                              \
    struct cast<T, __nv_fp8_e5m2> {                                                          \
        KERNEL_FLOAT_INLINE __nv_fp8_e5m2 operator()(T v) const { return __nv_fp8_e5m2(v); } \
    };                                                                                       \
                                                                                                \
    template <>                                                                              \
    struct cast<__nv_fp8_e5m2, T> {                                                          \
        KERNEL_FLOAT_INLINE T operator()(__nv_fp8_e5m2 v) const { return T(v); }             \
    };                                                                                       \
    }

#define KERNEL_FLOAT_FP8_CAST2(T, FP8_TY, FP8_INTERP)                                            \
    namespace detail {                                                                           \
    template <>                                                                                  \
    struct apply_impl<accurate_policy, ops::cast<T, FP8_TY>, 2, FP8_TY, T> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::cast<T, FP8_TY>, FP8_TY* result, const T* v) { \
            __half2_raw x;                                                                       \
            memcpy(&x, v, 2 * sizeof(T));                                                        \
            __nv_fp8x2_storage_t y = __nv_cvt_halfraw2_to_fp8x2(x, __NV_NOSAT, FP8_INTERP);      \
            memcpy(result, &y, 2 * sizeof(FP8_TY));                                              \
        }                                                                                        \
    };                                                                                           \
    template <>                                                                                  \
    struct apply_impl<accurate_policy, ops::cast<FP8_TY, T>, 2, T, FP8_TY> {                     \
        KERNEL_FLOAT_INLINE static void call(ops::cast<FP8_TY, T>, T* result, const FP8_TY* v) { \
            __nv_fp8x2_storage_t x;                                                              \
            memcpy(&x, v, 2 * sizeof(FP8_TY));                                                   \
            __half2_raw y = __nv_cvt_fp8x2_to_halfraw2(x, FP8_INTERP);                           \
            memcpy(result, &y, 2 * sizeof(T));                                                   \
        }                                                                                        \
    };                                                                                           \
    }

namespace kernel_float {
KERNEL_FLOAT_FP8_CAST(double)
}  // namespace kernel_float

#if KERNEL_FLOAT_FP16_AVAILABLE

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half_t, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(half_t, __nv_fp8_e5m2)

KERNEL_FLOAT_FP8_CAST(half_t)
KERNEL_FLOAT_FP8_CAST2(half_t, __nv_fp8_e4m3, __NV_E4M3)
KERNEL_FLOAT_FP8_CAST2(half_t, __nv_fp8_e5m2, __NV_E5M2)

}  // namespace kernel_float
#endif  // KERNEL_FLOAT_FP16_AVAILABLE

#if KERNEL_FLOAT_BF16_AVAILABLE

namespace kernel_float {
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(bfloat16_t, __nv_fp8_e4m3)
KERNEL_FLOAT_DEFINE_PROMOTED_TYPE(bfloat16_t, __nv_fp8_e5m2)

KERNEL_FLOAT_FP8_CAST(bfloat16_t)
KERNEL_FLOAT_FP8_CAST2(bfloat16_t, __nv_fp8_e4m3, __NV_E4M3)
KERNEL_FLOAT_FP8_CAST2(bfloat16_t, __nv_fp8_e5m2, __NV_E5M2)
}  // namespace kernel_float
#endif  // KERNEL_FLOAT_BF16_AVAILABLE

#endif  // KERNEL_FLOAT_FP8_AVAILABLE
#endif  // KERNEL_FLOAT_FP8_H
#ifndef KERNEL_FLOAT_PRELUDE_H
#define KERNEL_FLOAT_PRELUDE_H

namespace kernel_float { namespace prelude {
namespace kf = ::kernel_float;

template <typename T>
using kscalar = vector<T, extent<1>>;

template <typename T, size_t N>
using kvec = vector<T, extent<N>>;

// clang-format off
template<typename T> using kvec1 = kvec<T, 1>;
template<typename T> using kvec2 = kvec<T, 2>;
template<typename T> using kvec3 = kvec<T, 3>;
template<typename T> using kvec4 = kvec<T, 4>;
template<typename T> using kvec5 = kvec<T, 5>;
template<typename T> using kvec6 = kvec<T, 6>;
template<typename T> using kvec7 = kvec<T, 7>;
template<typename T> using kvec8 = kvec<T, 8>;
// clang-format on

#define KERNEL_FLOAT_TYPE_ALIAS(NAME, T)     \
    template <size_t N>                      \
    using k##NAME    = vector<T, extent<N>>; \
    using k##NAME##1 = vec<T, 1>;            \
    using k##NAME##2 = vec<T, 2>;            \
    using k##NAME##3 = vec<T, 3>;            \
    using k##NAME##4 = vec<T, 4>;            \
    using k##NAME##5 = vec<T, 5>;            \
    using k##NAME##6 = vec<T, 6>;            \
    using k##NAME##7 = vec<T, 7>;            \
    using k##NAME##8 = vec<T, 8>;

KERNEL_FLOAT_TYPE_ALIAS(char, char)
KERNEL_FLOAT_TYPE_ALIAS(short, short)
KERNEL_FLOAT_TYPE_ALIAS(int, int)
KERNEL_FLOAT_TYPE_ALIAS(long, long)
KERNEL_FLOAT_TYPE_ALIAS(longlong, long long)

KERNEL_FLOAT_TYPE_ALIAS(uchar, unsigned char)
KERNEL_FLOAT_TYPE_ALIAS(ushort, unsigned short)
KERNEL_FLOAT_TYPE_ALIAS(uint, unsigned int)
KERNEL_FLOAT_TYPE_ALIAS(ulong, unsigned long)
KERNEL_FLOAT_TYPE_ALIAS(ulonglong, unsigned long long)

KERNEL_FLOAT_TYPE_ALIAS(float, float)
KERNEL_FLOAT_TYPE_ALIAS(f32x, float)
KERNEL_FLOAT_TYPE_ALIAS(float32x, float)

KERNEL_FLOAT_TYPE_ALIAS(double, double)
KERNEL_FLOAT_TYPE_ALIAS(f64x, double)
KERNEL_FLOAT_TYPE_ALIAS(float64x, double)

#if KERNEL_FLOAT_FP16_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(half, half_t)
KERNEL_FLOAT_TYPE_ALIAS(f16x, half_t)
KERNEL_FLOAT_TYPE_ALIAS(float16x, half_t)
#endif

#if KERNEL_FLOAT_BF16_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(bfloat16x, bfloat16_t)
KERNEL_FLOAT_TYPE_ALIAS(bf16x, bfloat16_t)
#endif

#if KERNEL_FLOAT_BF8_AVAILABLE
KERNEL_FLOAT_TYPE_ALIAS(float8x, __nv_fp8_e4m3)
KERNEL_FLOAT_TYPE_ALIAS(float8_e4m3x, __nv_fp8_e4m3)
KERNEL_FLOAT_TYPE_ALIAS(float8_e5m2x, __nv_fp8_e5m2)
#endif

template <size_t N>
static constexpr extent<N> kextent = {};

template <typename... Args>
KERNEL_FLOAT_INLINE kvec<promote_t<Args...>, sizeof...(Args)> make_kvec(Args&&... args) {
    return ::kernel_float::make_vec(static_cast<Args&&>(args)...);
};

template <typename V>
KERNEL_FLOAT_INLINE into_vector_type<V> into_kvec(V&& input) {
    return ::kernel_float::into_vec(static_cast<V&&>(input));
}

template <typename T = double>
using kconstant = constant<T>;

template <typename T = double>
KERNEL_FLOAT_INLINE constexpr kconstant<T> kconst(T value) {
    return value;
}

KERNEL_FLOAT_INLINE
static constexpr kconstant<double> operator""_c(long double v) { return static_cast<double>(v); }

KERNEL_FLOAT_INLINE
static constexpr kconstant<long long int> operator""_c(unsigned long long int v) { return static_cast<long long int>(v); }

// Deduction guides for aliases are only supported from C++20
#if defined(__cpp_deduction_guides) && __cpp_deduction_guides >= 201907L
template <typename T>
kscalar(T&&) -> kscalar<decay_t<T>>;

template <typename... Args>
kvec(Args&&...) -> kvec<promote_t<Args...>, sizeof...(Args)>;

template <typename T>
kconstant(T&&) -> kconstant<decay_t<T>>;
#endif

}}  // namespace kernel_float::prelude

#endif
#ifndef KERNEL_FLOAT_TILING_H
#define KERNEL_FLOAT_TILING_H

namespace kernel_float {

template <size_t... Ns>
struct block_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    block_size(dim3 thread_index) {
        if (rank > 0 && size(0) > 1) {
            thread_index_[0] = thread_index.x;
        }

        if (rank > 1 && size(1) > 1) {
            thread_index_[1] = thread_index.y;
        }

        if (rank > 2 && size(2) > 1) {
            thread_index_[2] = thread_index.z;
        }
    }

    KERNEL_FLOAT_INLINE
    size_t thread_index(size_t axis) const { return axis < rank ? thread_index_[axis] : 0; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }

    private:
    unsigned int thread_index_[rank] = {0};
};

template <size_t... Ns>
struct unravel_block_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    unravel_block_size(dim3 thread_index) { thread_index_ = thread_index.x; }

    KERNEL_FLOAT_INLINE
    size_t thread_index(size_t axis) const {
        size_t product_up_to_axis = 1;
#pragma unroll
        for (size_t i = 0; i < axis; i++) {
            product_up_to_axis *= size(i);
        }

        return (thread_index_ / product_up_to_axis) % size(axis);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }

    private:
    unsigned int thread_index_ = 0;
};

template <size_t... Ns>
struct tile_size {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis, size_t block_size = 0) {
        size_t sizes[rank] = {Ns...};
        return axis < rank ? sizes[axis] : 1;
    }
};

template <size_t... Ns>
struct tile_factor {
    static constexpr size_t rank = sizeof...(Ns);

    KERNEL_FLOAT_INLINE
    static constexpr size_t size(size_t axis, size_t block_size) {
        size_t factors[rank] = {Ns...};
        return block_size * (axis < rank ? factors[axis] : 1);
    }
};

namespace dist {
template <size_t N, size_t K>
struct blocked_impl {
    static constexpr bool is_exhaustive      = N % K == 0;
    static constexpr size_t items_per_thread = (N / K) + (is_exhaustive ? 0 : 1);

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) {
        return is_exhaustive || (local_to_global(thread_index, local_index) < N);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) { return thread_index * items_per_thread + local_index; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) { return global_index % items_per_thread; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) { return global_index / items_per_thread; }
};

struct blocked {
    template <size_t N, size_t K>
    using type = blocked_impl<N, K>;
};

template <size_t M, size_t N, size_t K>
struct cyclic_impl {
    static constexpr bool is_exhaustive      = N % (K * M) == 0;
    static constexpr size_t items_per_thread = ((N / (K * M)) + (is_exhaustive ? 0 : 1)) * M;

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) {
        return is_exhaustive || (local_to_global(thread_index, local_index) < N);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) {
        return (local_index / M) * M * K + thread_index * M + (local_index % M);
    }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) { return (global_index / (M * K)) * M + (global_index % M); }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) { return (global_index / M) % K; }
};

struct cyclic {
    template <size_t N, size_t K>
    using type = cyclic_impl<1, N, K>;
};

template <size_t M>
struct block_cyclic {
    template <size_t N, size_t K>
    using type = cyclic_impl<M, N, K>;
};

template <size_t N>
struct replicate_impl {
    static constexpr bool is_exhaustive      = true;
    static constexpr size_t items_per_thread = N;

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) { return true; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) { return local_index; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) { return global_index; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) { return 0; }
};

struct replicate {
    template <size_t N, size_t K>
    using type = replicate_impl<N>;
};

template <size_t N, size_t Root, size_t K>
struct centralize_impl {
    static_assert(Root < K, "index of root thread cannot exceed thread block size");
    static constexpr bool is_exhaustive      = K == 1;
    static constexpr size_t items_per_thread = N;

    KERNEL_FLOAT_INLINE
    static constexpr bool local_is_present(size_t thread_index, size_t local_index) { return K == 1 || thread_index == Root; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t local_to_global(size_t thread_index, size_t local_index) { return local_index; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_local(size_t global_index) { return global_index; }

    KERNEL_FLOAT_INLINE
    static constexpr size_t global_to_owner(size_t global_index) { return Root; }
};

struct at_thread0 {
    template <size_t N, size_t K>
    using type = centralize_impl<N, 0, K>;
};

template <size_t I>
struct at_thread {
    template <size_t N, size_t K>
    using type = centralize_impl<N, I, K>;
};
}  // namespace dist

template <typename... Ds>
struct distributions {};

namespace detail {
template <size_t I, typename D>
struct instantiate_distribution_impl {
    template <size_t N, size_t K>
    using type = dist::cyclic::type<N, K>;
};

template <typename First, typename... Rest>
struct instantiate_distribution_impl<0, distributions<First, Rest...>> {
    template <size_t N, size_t K>
    using type = typename First::template type<N, K>;
};

template <size_t I, typename First, typename... Rest>
struct instantiate_distribution_impl<I, distributions<First, Rest...>> : instantiate_distribution_impl<I + 1, distributions<Rest...>> {};

template <typename TileDim, typename BlockDim, typename Distributions, typename = make_index_sequence<TileDim::rank>>
struct tiling_impl;

template <typename TileDim, typename BlockDim, typename Distributions, size_t... Is>
struct tiling_impl<TileDim, BlockDim, Distributions, index_sequence<Is...>> {
    template <size_t I>
    using dist_type =
        typename instantiate_distribution_impl<I, Distributions>::template type<TileDim::size(I, BlockDim::size(I)), BlockDim::size(I)>;

    static constexpr size_t rank             = TileDim::rank;
    static constexpr size_t items_per_thread = (dist_type<Is>::items_per_thread * ... * 1);
    static constexpr bool is_exhaustive      = (dist_type<Is>::is_exhaustive && ...);

    template <typename IndexType>
    KERNEL_FLOAT_INLINE static vector_storage<IndexType, rank> local_to_global(const BlockDim& block, size_t item) {
        vector_storage<IndexType, rank> result;
        ((result.data()[Is] = dist_type<Is>::local_to_global(block.thread_index(Is), item % dist_type<Is>::items_per_thread),
            item /= dist_type<Is>::items_per_thread),
            ...);
        return result;
    }

    KERNEL_FLOAT_INLINE
    static bool local_is_present(const BlockDim& block, size_t item) {
        bool is_present = true;
        ((is_present &= dist_type<Is>::local_is_present(block.thread_index(Is), item % dist_type<Is>::items_per_thread),
            item /= dist_type<Is>::items_per_thread),
            ...);
        return is_present;
    }
};
};  // namespace detail

template <typename T>
struct tiling_iterator;

/**
    * Represents a tiling where the elements given by `TileDim` are distributed over the
    * threads given by `BlockDim` according to the distributions given by `Distributions`.
    *
    * The template parameters should be the following:
    *
    * * ``TileDim``: Should be an instance of ``tile_size<...>``. For example,
    *   ``tile_size<16, 16>`` represents a 2-dimensional 16x16 tile.
    * * ``BlockDim``: Should be an instance of ``block_dim<...>``. For example,
    *    ``block_dim<16, 4>`` represents a thread block having X dimension 16
    *    and Y-dimension 4 for a total of 64 threads per block.
    * * ``Distributions``: Should be an instance of ``distributions<...>``. For example,
    *   ``distributions<dist::cyclic, dist::blocked>`` will distribute elements in
    *   cyclic fashion along the X-axis and blocked fashion along the Y-axis.
    * * ``IndexType``: The type used for index values (``int`` by default)
    */
template <typename TileDim, typename BlockDim, typename Distributions = distributions<>, typename IndexType = int>
struct tiling {
    using self_type  = tiling<TileDim, BlockDim, Distributions, IndexType>;
    using impl_type  = detail::tiling_impl<TileDim, BlockDim, Distributions>;
    using block_type = BlockDim;
    using tile_type  = TileDim;

    static constexpr size_t rank       = tile_type::rank;
    static constexpr size_t num_locals = impl_type::items_per_thread;

    using index_type = IndexType;
    using point_type = vector<index_type, extent<rank>>;

    __forceinline__ __device__ tiling() : block_(dim3(threadIdx)) {}

    KERNEL_FLOAT_INLINE
    tiling(BlockDim block, vec<index_type, rank> offset = {}) : block_(block), offset_(offset) {}

    /**
        * Returns the number of items per thread in the tiling.
        *
        * Note that this method is ``constexpr`` and can be called at compile-time.
        */
    KERNEL_FLOAT_INLINE
    static constexpr size_t size() { return impl_type::items_per_thread; }

    /**
        * Checks if the tiling is exhaustive, meaning all items are always present for all threads. If this returns
        * `true`, then ``is_present`` will always true for any given index.
        *
        * Note that this method is ``constexpr`` and can thus be called at compile-time.
        */
    KERNEL_FLOAT_INLINE
    static constexpr bool all_present() { return impl_type::is_exhaustive; }

    /**
        * Checks if a specific item is present for the current thread based on the distribution strategy. Not always
        * is the number of items stored per thread equal to the number of items _owned_ by each thread (for example,
        * if the tile size is not divisible by the block size). In this case, ``is_present`` will return `false` for
        * certain items.
        */
    KERNEL_FLOAT_INLINE
    bool is_present(size_t item) const { return all_present() || impl_type::local_is_present(block_, item); }

    /**
        * Returns the global coordinates of a specific item for the current thread.
        */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> at(size_t item) const { return impl_type::template local_to_global<index_type>(block_, item) + offset_; }

    /**
        * Returns the global coordinates of a specific item along a specified axis for the current thread.
        */
    KERNEL_FLOAT_INLINE
    index_type at(size_t item, size_t axis) const { return axis < rank ? at(item)[axis] : index_type{}; }

    /**
        * Returns the global coordinates of a specific item for the current thread (alias of ``at``).
        */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> operator[](size_t item) const { return at(item); }

    /**
        * Returns a vector of global coordinates of all items present for the current thread.
        */
    KERNEL_FLOAT_INLINE
    vector<vector<index_type, extent<rank>>, extent<num_locals>> local_points() const {
        return range<num_locals>([&](size_t i) { return at(i); });
    }

    /**
        * Returns a vector of coordinate values along a specified axis for all items present for the current thread.
        */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<num_locals>> local_points(size_t axis) const {
        return range<num_locals>([&](size_t i) { return at(i, axis); });
    }

    /**
        * Returns a vector of boolean values representing the result of ``is_present`` of the items for the current thread.
        */
    KERNEL_FLOAT_INLINE
    vector<bool, extent<num_locals>> local_mask() const {
        return range<num_locals>([&](size_t i) { return is_present(i); });
    }

    /**
        * Returns the thread index (position) along a specified axis for the current thread.
        */
    KERNEL_FLOAT_INLINE
    index_type thread_index(size_t axis) const { return index_type(block_.thread_index(axis)); }

    /**
        * Returns the size of the block (number of threads) along a specified axis.
        *
        * Note that this method is ``constexpr`` and can thus be called at compile-time.
        */
    KERNEL_FLOAT_INLINE
    static constexpr index_type block_size(size_t axis) { return index_type(block_type::size(axis)); }

    /**
        * Returns the size of the tile along a specified axis.
        *
        * Note that this method is ``constexpr`` and can thus be called at compile-time.
        */
    KERNEL_FLOAT_INLINE
    static constexpr index_type tile_size(size_t axis) { return index_type(tile_type::size(axis, block_size(axis))); }

    /**
        * Returns the offset of the tile along a specified axis.
        */
    KERNEL_FLOAT_INLINE
    index_type tile_offset(size_t axis) const { return index_type(offset_[axis]); }

    /**
        * Returns a vector of thread indices for all axes.
        */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> thread_index() const {
        return range<rank>([&](size_t i) { return thread_index(i); });
    }

    /**
        * Returns a vector of block sizes for all axes.
        */
    KERNEL_FLOAT_INLINE
    static vector<index_type, extent<rank>> block_size() {
        return range<rank>([&](size_t i) { return block_size(i); });
    }

    /**
        * Returns a vector of tile sizes for all axes.
        */
    KERNEL_FLOAT_INLINE
    static vector<index_type, extent<rank>> tile_size() {
        return range<rank>([&](size_t i) { return tile_size(i); });
    }

    /**
        * Returns the offset of the tile for all axes.
        */
    KERNEL_FLOAT_INLINE
    vector<index_type, extent<rank>> tile_offset() const {
        return range<rank>([&](size_t i) { return tile_offset(i); });
    }

    /**
        * Returns an iterator pointing to the beginning of the tiling.
        */
    KERNEL_FLOAT_INLINE
    tiling_iterator<tiling> begin() const { return {*this, 0}; }

    /**
        * Returns an iterator pointing to the end of the tiling.
        */
    KERNEL_FLOAT_INLINE
    tiling_iterator<tiling> end() const { return {*this, num_locals}; }

    /**
        * Applies a provided function to each item present in the tiling for the current thread.
        * The function should take an index and a ``vector`` of global coordinates as arguments.
        */
    template <typename F>
    KERNEL_FLOAT_INLINE void for_each(F fun) const {
#pragma unroll
        for (size_t i = 0; i < num_locals; i++) {
            if (is_present(i)) {
                fun(i, at(i));
            }
        }
    }

    /**
        * Adds ``offset`` to all points of this tiling and returns a new tiling.
        */
    KERNEL_FLOAT_INLINE friend tiling operator+(const tiling& self, const vector<index_type, extent<rank>>& offset) {
        return tiling{self.block_, self.offset_ + offset};
    }

    /**
        * Adds ``offset`` to all points of this tiling and returns a new tiling.
        */
    KERNEL_FLOAT_INLINE friend tiling operator+(const vector<index_type, extent<rank>>& offset, const tiling& self) { return self + offset; }

    /**
        * Adds ``offset`` to all points of this tiling.
        */
    KERNEL_FLOAT_INLINE friend tiling& operator+=(tiling& self, const vector<index_type, extent<rank>>& offset) { return self = self + offset; }

    private:
    BlockDim block_;
    vector<index_type, extent<rank>> offset_;
};

template <typename T>
struct tiling_iterator {
    using value_type = vector<typename T::index_type, extent<T::rank>>;

    KERNEL_FLOAT_INLINE
    tiling_iterator(const T& inner, size_t position = 0) : inner_(&inner), position_(position) {
        while (position_ < T::num_locals && !inner_->is_present(position_)) {
            position_++;
        }
    }

    KERNEL_FLOAT_INLINE
    value_type operator*() const { return inner_->at(position_); }

    KERNEL_FLOAT_INLINE
    tiling_iterator& operator++() { return *this = tiling_iterator(*inner_, position_ + 1); }

    KERNEL_FLOAT_INLINE
    tiling_iterator operator++(int) {
        tiling_iterator old = *this;
        this ++;
        return old;
    }

    KERNEL_FLOAT_INLINE
    friend bool operator==(const tiling_iterator& a, const tiling_iterator& b) { return a.position_ == b.position_; }

    KERNEL_FLOAT_INLINE
    friend bool operator!=(const tiling_iterator& a, const tiling_iterator& b) { return !operator==(a, b); }

    size_t position_ = 0;
    const T* inner_;
};

template <size_t TileDim, size_t BlockDim, typename D = dist::cyclic, typename IndexType = int>
using tiling_1d = tiling<tile_size<TileDim>, block_size<BlockDim>, distributions<D>, IndexType>;

                // clang-format off
#define KERNEL_FLOAT_TILING_FOR_IMPL1(ITER_VAR, TILING, POINT_VAR, _)      \
        _Pragma("unroll")                                                         \
        for (size_t ITER_VAR = 0; ITER_VAR < (TILING).size(); ITER_VAR++)         \
            if (POINT_VAR = (TILING).at(ITER_VAR); (TILING).is_present(ITER_VAR)) \

#define KERNEL_FLOAT_TILING_FOR_IMPL2(ITER_VAR, TILING, INDEX_VAR, POINT_VAR)      \
        KERNEL_FLOAT_TILING_FOR_IMPL1(ITER_VAR, TILING, POINT_VAR, _) \
            if (INDEX_VAR = ITER_VAR; true)

#define KERNEL_FLOAT_TILING_FOR_IMPL(ITER_VAR, TILING, A, B, N, ...) \
    KERNEL_FLOAT_CALL(KERNEL_FLOAT_CONCAT(KERNEL_FLOAT_TILING_FOR_IMPL, N), ITER_VAR, TILING, A, B)

/**
 * Iterate over the points in a ``tiling<...>`` using a for loop.
 *
 * There are two ways to use this macro. Using the 1 variable form:
 * ```
 * auto t = tiling<tile_size<16, 16>, block_size<4, 4>>;
 *
 * KERNEL_FLOAT_TILING_FOR(t, auto point) {
 *  printf("%d,%d\n", point[0], point[1]);
 * }
 * ```
 *
 * Or using the 2 variables form:
 * ```
 * auto t = tiling<tile_size<16, 16>, block_size<4, 4>>;
 *
 * KERNEL_FLOAT_TILING_FOR(t, auto index, auto point) {
 *  printf("%d] %d,%d\n", index, point[0], point[1]);
 * }
 * ```
 */
#define KERNEL_FLOAT_TILING_FOR(...) \
    KERNEL_FLOAT_TILING_FOR_IMPL(KERNEL_FLOAT_CONCAT(__tiling_index_variable__, __LINE__), __VA_ARGS__, 2, 1)
                // clang-format on

                }  // namespace kernel_float

#endif  // KERNEL_FLOAT_TILING_H