cmake_minimum_required(VERSION 3.22)
project(swardfish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Build all targets as position-independent code (-fPIC)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# ---------------------------------------------------------------- Strata
set(STRATA_ROOT "${CMAKE_SOURCE_DIR}/third_party/strata" CACHE PATH "Strata source root")
set(STRATA_BUILD_DIR "${STRATA_ROOT}/build" CACHE PATH "Strata build directory")
set(STRATA_LIB_NAMES strata_engine strata_kernels strata_kernels_cpu strata_core
                     strata_spec strata_prefill strata_quant strata_qwen35 strata_pf_own)

set(_archives "")
foreach(_n IN LISTS STRATA_LIB_NAMES)
  list(APPEND _archives "${STRATA_BUILD_DIR}/lib${_n}.a")
endforeach()

# CUDA arch for the Strata sub-build: "native" is fine for CMake >= 3.24,
# otherwise pass a number (e.g. 89) via -DSTRATA_CUDA_ARCH=89
set(STRATA_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA arch for Strata")

find_program(NINJA_EXECUTABLE ninja)
if(NINJA_EXECUTABLE)
  set(_strata_gen Ninja)
else()
  set(_strata_gen "${CMAKE_GENERATOR}")
endif()

# Configure + build Strata at *build* time, only if an archive is missing.
set(_strata_stamp "${STRATA_BUILD_DIR}/.strata_built")
add_custom_command(
  OUTPUT ${_archives} "${_strata_stamp}"
  COMMAND ${CMAKE_COMMAND} -S "${STRATA_ROOT}/engine" -B "${STRATA_BUILD_DIR}"
          -G "${_strata_gen}"
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
          -DSTRATA_ENABLE_CUDA=ON
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON
          -DCMAKE_CUDA_ARCHITECTURES=${STRATA_CUDA_ARCH}
          -DSTRATA_BUILD_TESTS=OFF
  COMMAND ${CMAKE_COMMAND} --build "${STRATA_BUILD_DIR}" --target strata  
  COMMAND ${CMAKE_COMMAND} -E touch "${_strata_stamp}"
  WORKING_DIRECTORY "${STRATA_ROOT}"
  COMMENT "Building Strata (static libs missing or out of date)"
  USES_TERMINAL
  VERBATIM)

add_custom_target(strata_build DEPENDS ${_archives})

# Imported targets so ordering/dependencies are tracked properly
add_library(strata_all INTERFACE)
add_dependencies(strata_all strata_build)

add_compile_definitions(STRATA_VERSION="0.1.57")
add_compile_definitions(STRATA_ENABLE_CUDA STRATA_HAVE_DFLASH2)

add_library(${PROJECT_NAME} SHARED strata_generate.cpp)
add_dependencies(${PROJECT_NAME} strata_build)
target_include_directories(${PROJECT_NAME} PRIVATE "${STRATA_ROOT}/engine/include")

# ggml headers live in the Strata build tree, which may not exist at configure time,
# so reference the path directly instead of using file(GLOB)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${STRATA_BUILD_DIR}/_deps/strata_llamacpp-src/ggml/include")

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--whole-archive ${_archives} -Wl,--no-whole-archive
    CUDA::cuda_driver CUDA::cudart CUDA::cublas
    Threads::Threads
    ${CMAKE_DL_LIBS})

set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY LINK_DEPENDS ${_archives})
target_link_options(${PROJECT_NAME} PRIVATE -Wl,--no-undefined)
