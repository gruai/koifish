cmake_minimum_required (VERSION 3.3)
project(ggml VERSION 0.1.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)
    include(cmake/GitVars.cmake)
    include(cmake/BuildTypes.cmake)
else()
    set(GGML_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# options

option(BUILD_SHARED_LIBS            "ggml: build shared libs" ${BUILD_SHARED_LIBS_DEFAULT})

option(GGML_ALL_WARNINGS            "ggml: enable all compiler warnings"                   OFF)
option(GGML_ALL_WARNINGS_3RD_PARTY  "ggml: enable all compiler warnings in 3rd party libs" OFF)

option(GGML_SANITIZE_THREAD         "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS        "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED      "ggml: enable undefined sanitizer" OFF)

# option(GGML_BUILD_TESTS             "ggml: build tests"    ${GGML_STANDALONE})
# option(GGML_BUILD_EXAMPLES          "ggml: build examples" ${GGML_STANDALONE})

option(GGML_TEST_COVERAGE           "ggml: enable test coverage" OFF)

option(GGML_PERF                    "ggml: enable perf timings"          OFF)
option(GGML_NO_ACCELERATE           "ggml: disable Accelerate framework" OFF)
option(GGML_OPENBLAS                "ggml: use OpenBLAS"                 OFF)
option(GGML_CLBLAST                 "ggml: use clBLAST"                  OFF)
option(GGML_HIPBLAS                 "ggml: use hipBLAS"                  OFF)
option(GGML_CUBLAS                  "ggml: use cuBLAS"                   ON)
option(GGML_METAL                   "ggml: use Metal"                    OFF)

option(GGML_CUDA_FORCE_DMMV                 "ggml: use dmmv instead of mmvq CUDA kernels"     OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
set(GGML_CUDA_DMMV_X      "32" CACHE STRING "ggml: x stride for dmmv CUDA kernels")
set(GGML_CUDA_MMV_Y        "1" CACHE STRING "ggml: y block size for mmv CUDA kernels")
option(GGML_CUDA_F16                        "ggml: use 16 bit floats for some calculations"   OFF)
set(GGML_CUDA_KQUANTS_ITER "2" CACHE STRING "ggml: iters./thread per block for Q2_K/Q6_K")
set(GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
# sanitizers

if (GGML_SANITIZE_THREAD)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

if (GGML_SANITIZE_ADDRESS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
endif()

if (GGML_SANITIZE_UNDEFINED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}     -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

# instruction set specific
option(GGML_AVX                     "ggml: enable AVX"                                     ON)
option(GGML_AVX2                    "ggml: enable AVX2"                                    ON)
option(GGML_AVX512                  "ggml: enable AVX512"                                  OFF)
option(GGML_AVX512_VBMI             "ggml: enable AVX512-VBMI"                             OFF)
option(GGML_AVX512_VNNI             "ggml: enable AVX512-VNNI"                             OFF)
option(GGML_FMA                     "ggml: enable FMA"                                     ON)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(GGML_F16C                "ggml: enable F16C"                                    ON)
endif()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=native")

# warning flags

if (GGML_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags   -Wall -Wpedantic -Wformat=2 -Wno-unused -Wstrict-prototypes -Wno-unused-variable -Wunused-parameter)
        set(cxx_flags -Wall -Wpedantic -Wformat=2 -Wno-unused-variable -Wunused-parameter)
    else()
        # todo : windows
    endif()

    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
        "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )
endif()

if (NOT MSVC)
    # TODO: temporary disabled until we figure out ggml-metal.m
    #add_compile_options(
    #    "$<$<COMPILE_LANGUAGE:C>:-Werror=vla>"
    #    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=vla>"
    #    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-Werror=vla>"
    #)
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity
# are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "DragonFly")
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

if (WHISPER_PERF)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DGGML_PERF)
endif()

# dependencies

set(CMAKE_C_STANDARD   11)
set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)

# main

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif ()

if (GGML_BUILD_TESTS)
    if (GGML_TEST_COVERAGE)
        if (CMAKE_C_COMPILER_ID MATCHES "Clang")
            set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fprofile-instr-generate -fcoverage-mapping")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
        else()
            message(WARNING "Test coverage is only supported for Clang")
        endif()
    endif()
endif()

if (GGML_OPENBLAS)
    set(OPENBLAS_INCLUDE_SEARCH_PATHS
        /usr/include
        /usr/include/openblas
        /usr/include/openblas-base
        /usr/local/include
        /usr/local/include/openblas
        /usr/local/include/openblas-base
        /opt/OpenBLAS/include
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/include
        )
    find_path(OPENBLAS_INC NAMES cblas.h PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
    find_library(OPENBLAS_LIB NAMES openblas libopenblas HINTS ${OPENBLAS_INC}/../lib/)
    if (OPENBLAS_LIB)
        message(STATUS "OpenBLAS found  @OPENBLAS_INC=${OPENBLAS_INC}")

        set(GGML_EXTRA_LIBS  ${GGML_EXTRA_LIBS}  ${OPENBLAS_LIB})
        set(GGML_EXTRA_INCS  ${GGML_EXTRA_INCS}  ${OPENBLAS_INC})
        set(GGML_EXTRA_FLAGS ${GGML_EXTRA_FLAGS} -DGGML_USE_OPENBLAS)
        add_compile_definitions(GGML_USE_OPENBLAS)
    else()
        message(WARNING "OpenBLAS not found!!!  @OPENBLAS_INC=${OPENBLAS_INC} ")
    endif()
endif()

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/DEBUG/)
    # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE_cuda/)
    # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/DEBUG_cuda/)
ELSE()
    # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE/)
    set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE_cuda/)
ENDIF()

set(GG_LIB  -Wl,--start-group 
            ${GG_LIB_PATH}/libllama.so 
            ${GG_LIB_PATH}/libggml.so 
            ${GG_LIB_PATH}/libcommon.a -Wl,--end-group )    
find_library(GGML_LIB NAMES libggml.so HINTS ${GG_LIB_PATH})
IF(NOT GGML_LIB)
    message(FATAL_ERROR "GGML_LIB library not found @${GG_LIB_PATH}")
endif()

# find_package(pybind11 REQUIRED)
option(DELTA_WANDB                  "DELTA: wandb support"                                  OFF)
if (DELTA_WANDB)
    add_subdirectory(src/pybind11)
    set(DELTA_LIB pybind11::embed)
    set(DELTA_cc -D_USE_WANDB_)
endif()
if(GGML_OPENBLAS)
    set(DELTA_cc ${DELTA_cc} -D_USE_OPENBLAS_)
    add_subdirectory(src/lenda/util)
    add_subdirectory(src/lenda/kernel)
else()
endif()
message(STATUS "DELTA GG_LIB=${GG_LIB_PATH} cc=${DELTA_cc} lib=${DELTA_LIB} PWD=${CMAKE_SOURCE_DIR}")

add_subdirectory(src/Manifold)
add_subdirectory(src/Fuzi)
add_subdirectory(src/ggex)
# add_subdirectory(src/SAM)
# add_subdirectory(src/finetune)
# add_subdirectory(src/Train)
 
set(sources   ${TGraph_sources} ${util_sources} ${kernel_sources} ${Fuzi_sources} ${ggex_sources})
# add_executable(finetune ${CMAKE_CURRENT_SOURCE_DIR}/src/finetune/finetune.cpp ${sources})
add_executable(koifish ${CMAKE_CURRENT_SOURCE_DIR}/src/koifish.cpp ${sources} )
# add_executable(train2 ${CMAKE_CURRENT_SOURCE_DIR}/src/Train/train-text-from-scratch.cpp ${sources} )

# add_executable(gpt2 ${CMAKE_CURRENT_SOURCE_DIR}/src/train_gpt2.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/Manifold/DataLoader.cpp )
# target_compile_options(gpt2 PRIVATE -fopenmp -D_DATA_LOADER_LITE_)
# target_link_libraries(gpt2 PRIVATE m gomp)
# target_include_directories(gpt2 PRIVATE ${CMAKE_SOURCE_DIR}/src/LLAMA ${CMAKE_SOURCE_DIR}/src/ggex ${CMAKE_SOURCE_DIR}/src/Manifold)

# add_executable(SAM ${CMAKE_CURRENT_SOURCE_DIR}/src/SAM/main.cpp ${TGraph_sources} ${util_sources} ${kernel_sources})
# pybind11_add_module(gquant main_bind.cpp ${kernel_sources} ${SpMV_sources} ${util_sources} )

foreach(prog koifish )    #finetune koifish train2
    install(TARGETS ${prog} RUNTIME)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src ${CMAKE_SOURCE_DIR}/llama.cpp/common ${CMAKE_SOURCE_DIR}/llama.cpp/include ${CMAKE_SOURCE_DIR}/llama.cpp/src ${CMAKE_SOURCE_DIR}/src/ggex/)
    target_link_libraries(${prog} PRIVATE ${GG_LIB} ${CMAKE_THREAD_LIBS_INIT} ${OPENBLAS_LIB} ${DELTA_LIB})
    target_compile_features(${prog} PRIVATE cxx_std_17)
    target_compile_options(${prog} PRIVATE  -DLLAMA_API_INTERNAL ${cxx_flags} ${DELTA_cc})
    # target_link_libraries(${prog} PRIVATE pybind11::embed)
endforeach(prog)

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


