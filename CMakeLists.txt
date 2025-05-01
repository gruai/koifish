cmake_minimum_required (VERSION 3.14)
project(koifish VERSION 0.2.0)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(PWD ${CMAKE_CURRENT_SOURCE_DIR})

# find_package(pybind11 REQUIRED)
option(DELTA_WANDB                  "KOIFISH: wandb support"        OFF)
option(K_USECUDA                    "KOIFISH: CUDA"                 ON)
option(K_DEBUGCUDA                   "KOIFISH: DEBUG_CUDA[slow!]"                 OFF)
option(K_TENSOR_CUDNN               "KOIFISH: CUDNN TENSON"         ON)
option(K_USE_CCACHE                 "KOIFISH: use ccache if available"       ON)
option(K_USE_GGML                   "KOIFISH: use GGML libs"        OFF)

if (K_USE_CCACHE)
    find_program(_CCACHE_FOUND ccache)

    if (_CCACHE_FOUND)
        # TODO: should not be set globally
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF")
    endif ()
endif()

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

find_package(OpenMP)


if(K_TENSOR_CUDNN)
    add_compile_definitions(_TENSOR_G_)  
    set(CUDA_LIB  cudart cudnn cublasLt  )   
    set(CUDA_INC  ${CUDA_INC} ~/cudnn-frontend/include  ) 
endif()

if(K_USECUDA)
    cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        # set(cdir _cuda)
        # set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89")  #BF16  Ref:    https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        set(CMAKE_CUDA_ARCHITECTURES "80;90") 
        add_compile_definitions(__USE_CUDA__)
        message(STATUS "CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
        enable_language(CUDA)
        
        set(CUDA_FLAGS -use_fast_math -DENABLE_CUDNN -DENABLE_BF16 )
        # -mavx512f -mavx512vl  for identifier "__builtin_ia32_ldtilecfg" is undefined
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
            message(STATUS "CUDA with debug option, very slow!!!")
            list(APPEND CUDA_FLAGS -G ${CUDA_CXX_FLAGS_JOINED})
        endif()
        message(STATUS "CUDA_FLAGS=${CUDA_FLAGS}")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
    else()
        message(WARNING "CUDA not found")
    endif()
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
    set(DELTA_cc ${DELTA_cc} -D__USE_GGML__)    
    set(GG_LIB  ${GG_LIB_PATH}/libcommon.a ${GG_LIB_PATH}/libllama.so ${GG_LIB_PATH}/libggml.so ${GG_LIB_PATH}/libggml-cpu.so ${GG_LIB_PATH}/libggml-base.so)   
    set(GG_INC ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src ${CMAKE_SOURCE_DIR}/llama.cpp/include )
    message(STATUS "KOIFISH GG_LIB=${GG_LIB} GG_INC=${GG_INC}")
endif()

message(STATUS "KOIFISH cc=${DELTA_cc} lib=${DELTA_LIB} PWD=${CMAKE_SOURCE_DIR}")

add_subdirectory(src/Manifold)
add_subdirectory(src/Fuzi)
add_subdirectory(src/Device/CUDA)
add_subdirectory(src/ggex)
add_subdirectory(src/Models)
add_subdirectory(src/Utils)
 
set(sources   ${TGraph_sources} ${util_sources} ${kernel_sources} ${Fuzi_sources} ${ggex_sources} ${device_sources} ${model_sources} )
add_executable(koifish ${CMAKE_CURRENT_SOURCE_DIR}/src/koifish.cpp ${sources} )
message(STATUS "KOIFISH sources=${sources}")


foreach(prog koifish )    #finetune koifish train2
    install(TARGETS ${prog} RUNTIME)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/src/Device/CUDA ${GG_INC} ${CUDA_INC} )

    target_link_libraries(${prog} PRIVATE ${GG_LIB} ${CMAKE_THREAD_LIBS_INIT} ${OPENBLAS_LIB} ${DELTA_LIB} ${CUDA_LIB} pthread)
    target_compile_features(${prog} PRIVATE cxx_std_17)
    target_compile_options(${prog} PRIVATE  -DLLAMA_API_INTERNAL ${cxx_flags} ${DELTA_cc})
    # target_link_libraries(${prog} PRIVATE pybind11::embed)
endforeach(prog)

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


