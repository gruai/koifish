cmake_minimum_required (VERSION 3.14)
project(koifish VERSION 0.2.0)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(PWD ${CMAKE_CURRENT_SOURCE_DIR})

option(K_USECUDA                    "KOIFISH: CUDA"                 ON)
option(K_DEBUGCUDA                  "KOIFISH: DEBUG_CUDA[slow!]"                 OFF)
option(K_USE_CUDNN                  "KOIFISH: use CUDNN libs"         ON)
option(K_USE_CCACHE                 "KOIFISH: use ccache if available"       ON)
option(K_USE_GGML                   "KOIFISH: use GGML libs"        OFF)
OPTION(K_USE_OPENMP                 "KOIFISH: Enable OpenMP"                 ON)
OPTION(K_USE_OPENBLAS               "KOIFISH: Enable OpenBLAS"               OFF)
OPTION(K_USE_GBDT                   "KOIFISH: Enable Gradient boosting trees"               OFF)
OPTION(K_USE_EIGEN                  "KOIFISH: Enable Eigen lib"               OFF)
OPTION(K_TESTING                    "KOIFISH: Do some testing"               OFF)
# option(K_WANDB                      "KOIFISH: wandb support"        OFF)

# set (CMAKE_C_FLAGS "-std=c11")
# set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
# set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

# set (CMAKE_CXX_FLAGS "-std=c++11")
# set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
# set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

# set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

# add_compile_options (
#     -Wall
#     -Werror
#     -Wextra
#     -Wswitch-default
#     -Wfloat-equal
#     -Wshadow
#     -Wcast-qual
# )
add_compile_definitions(_TENSOR_G_)

if(K_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    # Ignore unknown #pragma warning
    if( (CMAKE_CXX_COMPILER_ID MATCHES "[cC][lL][aA][nN][gG]")
      OR (CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif()

if (K_USE_CCACHE)
    find_program(_CCACHE_FOUND ccache)

    if (_CCACHE_FOUND)
        # TODO: should not be set globally
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with K_USE_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with K_USE_CCACHE=OFF")
    endif ()
endif()

set(DELTA_cc ${DELTA_cc} -mf16c -mavx2 -mfma -std=c++20)
# if (K_WANDB)
#     add_subdirectory(src/pybind11)
#     set(DELTA_LIB pybind11::embed)
#     set(DELTA_cc ${DELTA_cc} -D_USE_WANDB_)
# endif()
if(K_USE_OPENBLAS)
    add_compile_definitions(_USE_OPENBLAS_)
    set(MORE_INC /home/cys/Downloads/OpenBLAS-0.3.27/build/generated/ /home/cys/Downloads/OpenBLAS-0.3.27/build/ )
    set(OPENBLAS_LIB  /home/cys/Downloads/OpenBLAS-0.3.27/libopenblas_haswellp-r0.3.27.so  )  
    set(blas_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/BLAS_t.cpp )
    # add_subdirectory(src/lenda/util)
    add_subdirectory(src/lenda/kernel)
else()
endif()

if(K_USE_EIGEN)
    add_subdirectory(src/lenda/Cheby)
    set(MORE_INC ${MORE_INC} /home/cys/rnd/gssx/gsp_3.0/library/eigen-3.4.0/)
endif()

if(K_USECUDA)
    # if target == '4090':
    #     cuda_flags.append('-DCUDA_ADA')
    #     cuda_flags.append('-arch=sm_89')
    # elif target == 'h100':
    #     cuda_flags.append('-DCUDA_HOPPER')
    #     cuda_flags.append('-arch=sm_90a')
    # elif target == 'a100':
    #     cuda_flags.append('-DCUDA_Ampere')
    #     cuda_flags.append('-arch=sm_80')
    # else:
    #     raise ValueError(f'Target {target} not supported')
    cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        # set(cdir _cuda)
        # set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89")  #BF16  Ref:    https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        set(CMAKE_CUDA_ARCHITECTURES "80") 
        add_compile_definitions(__USE_CUDA__)
        message(STATUS "CUDAA ARCH=${CMAKE_CUDA_ARCHITECTURES} DIR=${CUDA_TOOLKIT_ROOT_DIR}")
        enable_language(CUDA)
        
        set(CUDA_FLAGS -use_fast_math  -DENABLE_BF16 --expt-relaxed-constexpr -std=c++20 )
        # -mavx512f -mavx512vl  for identifier "__builtin_ia32_ldtilecfg" is undefined
        #  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()
        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")
        list(APPEND CUDA_CXX_FLAGS ${ARCH_FLAGS})
        list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument
        if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
            list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
        endif()
        if(K_DEBUGCUDA)
            message(STATUS "DEBUG_CUDA - CUDA with debug option, very slow!!!")
            list(APPEND CUDA_FLAGS -G -O0 ${CUDA_CXX_FLAGS_JOINED})
        endif()
        message(STATUS "CUDA_FLAGS=${CUDA_FLAGS}")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
        set(CUDA_LIB  cudart cublas cublasLt )   
    else()
        message(WARNING "CUDA not found")
    endif()
endif()

if(K_USE_CUDNN)    
    set(CUDA_LIB  ${CUDA_LIB} cudnn )   
    set(CUDA_INC  ${CMAKE_SOURCE_DIR}/cudnn-frontend/include ${CMAKE_SOURCE_DIR}/src/Device/CUDA/TK/include  ) 
    # set(CUDA_FLAGS ${CUDA_FLAGS} -DENABLE_CUDNN )
    add_compile_definitions(ENABLE_CUDNN)  
endif()

if(K_USE_GGML)
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/)
    # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/DEBUG${cdir}/)
    ELSE()
    set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/)
    # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE${cdir}/)
    ENDIF()     
    find_library(GGML_LIB NAMES libggml.so HINTS ${GG_LIB_PATH})
    IF(NOT GGML_LIB)
        message(FATAL_ERROR "GGML_LIB library not found @${GG_LIB_PATH}")
    endif()
    add_compile_definitions(__USE_GGML__)
    set(GG_LIB  ${GG_LIB_PATH}/libcommon.a ${GG_LIB_PATH}/libllama.so ${GG_LIB_PATH}/libggml.so ${GG_LIB_PATH}/libggml-cpu.so ${GG_LIB_PATH}/libggml-base.so)   
    set(GG_INC ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src ${CMAKE_SOURCE_DIR}/llama.cpp/include )
    message(STATUS "KOIFISH GG_LIB=${GG_LIB} GG_INC=${GG_INC}")
endif()

if(K_USE_GBDT)
    add_compile_definitions(_USE_GBDT_)
    add_subdirectory(src/GBDT)
endif()

message(STATUS "KOIFISH cc=${DELTA_cc} lib=${DELTA_LIB} PWD=${CMAKE_SOURCE_DIR}")

add_subdirectory(src/Tensor)
add_subdirectory(src/Manifold)
add_subdirectory(src/Fuzi)
add_subdirectory(src/Device)
add_subdirectory(src/ggex)
add_subdirectory(src/Models)
add_subdirectory(src/Utils)

add_subdirectory(src/TokenSet)

set(sources   ${TGraph_sources} ${util_sources} ${tensor_sources} ${kernel_sources} ${cheby_sources} ${Fuzi_sources} ${ggex_sources} ${device_sources} ${model_sources} 
    ${token_sources} ${mort_sources} ${blas_sources} ${TEST_SRCS} )
add_executable(koifish ${CMAKE_CURRENT_SOURCE_DIR}/src/koifish.cpp ${sources} )
add_executable(bubble ${CMAKE_CURRENT_SOURCE_DIR}/src/bubble.cpp ${sources} )
message(STATUS "KOIFISH sources=${sources}")


foreach(prog koifish bubble)    
    install(TARGETS ${prog} RUNTIME)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/src/Manifold/ ${CMAKE_SOURCE_DIR}/src/TokenSet/ ${CMAKE_SOURCE_DIR}/src/Utils/ ${CMAKE_SOURCE_DIR}/src/Device/CUDA 
        ${GG_INC} ${CUDA_INC} ${MORE_INC} )

    target_link_libraries(${prog} PRIVATE ${GG_LIB} ${CMAKE_THREAD_LIBS_INIT} ${OPENBLAS_LIB} ${DELTA_LIB} ${CUDA_LIB} ${GFLAGS_LIBRARIES} pthread )
    target_compile_features(${prog} PRIVATE cxx_std_17)
    target_compile_options(${prog} PRIVATE ${cxx_flags} ${DELTA_cc})
    # target_link_libraries(${prog} PRIVATE pybind11::embed)
endforeach(prog)

if (K_TESTING)
    add_subdirectory(tests)
    find_package(gflags REQUIRED)
    add_executable(k_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp ${sources} )
    foreach(prog k_test )    
        target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/src/Manifold/ ${CMAKE_SOURCE_DIR}/src/TokenSet/ ${CMAKE_SOURCE_DIR}/src/Utils/ ${CMAKE_SOURCE_DIR}/src/Device/CUDA 
            ${GG_INC} ${CUDA_INC} ${MORE_INC} )

        target_link_libraries(${prog} PRIVATE ${GG_LIB} ${CMAKE_THREAD_LIBS_INIT} ${OPENBLAS_LIB} ${DELTA_LIB} ${CUDA_LIB} ${GFLAGS_LIBRARIES} pthread )
        target_compile_features(${prog} PRIVATE cxx_std_17)
        target_compile_options(${prog} PRIVATE ${cxx_flags} ${DELTA_cc})
        # target_link_libraries(${prog} PRIVATE pybind11::embed)
    endforeach(prog)

endif()



