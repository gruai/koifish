/**
 *  SPDX-FileCopyrightText: 2023-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT  
 * 
 *  Some codes from KT/llama.c
 * 
 *  \brief Tile
 *  \author Yingshi Chen
 */
#pragma Once

#include "./Packed.hpp"

// #ifdef CUDA_HOPPER
//     #define DEFAULT_ALIGN ALIGN(128)
// #else
    #define DEFAULT_ALIGN ALIGN(16)
// #endif

template<typename _T, int _rows, int _cols> 
struct DEFAULT_ALIGN TILE   {

};

template<typename T> constexpr int TILE_COL_DIM = sizeof(T) == 1 ? 32 : 16;
template<typename T> constexpr int TILE_ROW_DIM = 16;
template<typename _T, int _rows, int _cols>
struct DEFAULT_ALIGN cuTILE : public TILE<_T,_rows,_cols> {
    using T = Pack2<_T>::unpacked_type;
    using T2 = Pack2<_T>::packed_type;
    using dtype = T; ///< Data type of the elements in the tile.

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows          = _rows;
    static constexpr int underlying_cols          = _cols;
    static constexpr int underlying_height        = _rows / TILE_ROW_DIM<T>;
    static constexpr int underlying_width         = _cols / TILE_COL_DIM<T>;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static_assert(rows % TILE_ROW_DIM<T> == 0, "Rows must be divisible by the tile dimension");
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static_assert(cols % TILE_COL_DIM<T> == 0, "Cols must be divisible by the tile dimension");
    static constexpr int height              = _rows / TILE_ROW_DIM<T>; ///< Height of the tile in terms of 16-element subtiles.
    static constexpr int width               = _cols / TILE_COL_DIM<T>; ///< Width of the tile in terms of 16-element subtiles.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static_assert(Pack2<dtype>::num() == 1); // must be a 1-packed type (e.g. float, bf16, etc)

    static constexpr int swizzle_bytes = (
        sizeof(dtype) == 1 ? (  // Add FP8 case
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 2 ? (
            underlying_width%4 == 0 ? 128 :
            underlying_width%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            underlying_width%2 == 0 ? 128 : 64
        ) : -1
    );

    // wgmma layout with swizzling
    dtype data[rows*cols]; ///< Raw data storage for the tile.

    __device__ static inline T* idx(T *ptr, int2 coord) { // naive row-major coord default
        int r = coord.x, c = coord.y; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    }
    __device__ static inline uint32_t idx(uint32_t ptr, int2 coord) {
        int r = coord.x, c = coord.y; // alias
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c/subtile_cols;
        const uint32_t addr = ptr + sizeof(T)*(outer_idx*rows*subtile_cols + r*subtile_cols + c%subtile_cols);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (addr ^ swizzle);
    }

    __device__ inline       dtype& operator[](const int2 &rowcol)       {
        return *idx(data, rowcol);
    }
    __device__ inline const dtype& operator[](const int2 &rowcol) const {
        return *(const dtype*)idx((dtype*)data, rowcol);
    }
    __device__ inline       dtype& operator[](int idx)       {
        return data[idx];
    }
    __device__ inline const dtype& operator[](int idx) const {
        return data[idx];
    }

    // vector types
    using col_vec = sv<dtype, rows>; ///< Column vector type for this tile
    using row_vec = sv<dtype, cols>; ///< Row vector type for this tile
    template<int subtile_rows, int subtile_cols> using subtile = st_subtile<
        cuTILE<T, rows, cols>, subtile_rows, subtile_cols
    >; ///< A templated subtile type wrapper for this tile.
};


template<int _height, int _width> using st_bf = st<bf16,  _height, _width>;
template<int _height, int _width> using st_hf = st<half,  _height, _width>;
template<int _height, int _width> using st_fl = st<float, _height, _width>;
template<int _height, int _width> using st_fl8_e4m3 = st<fp8e4m3, _height, _width>;
template<int _height, int _width> using st_fl8_e5m2 = st<fp8e5m2, _height, _width>;