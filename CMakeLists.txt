cmake_minimum_required (VERSION 3.14)
project(koifish VERSION 0.2.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)

# find_package(pybind11 REQUIRED)
option(DELTA_WANDB                  "KOIFISH: wandb support"                                  OFF)
option(K_REBUILD_LLAMA          "KOIFISH: rebuild llama"                                  ON)

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

if(K_REBUILD_LLAMA)
    add_subdirectory(llama.cpp)
    set(GG_LIB common llama ) 
else()
    IF(CMAKE_BUILD_TYPE MATCHES "Debug")
        set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/DEBUG/)
        # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/DEBUG_cuda/)
    ELSE()
        # set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE/)
        set(GG_LIB_PATH ${CMAKE_SOURCE_DIR}/bin/RELEASE_cuda/)
    ENDIF()
    set(GG_LIB  -Wl,--start-group ${GG_LIB_PATH}/libllama.so ${GG_LIB_PATH}/libggml.so ${GG_LIB_PATH}/libcommon.a -Wl,--end-group )    
    find_library(GGML_LIB NAMES libggml.so HINTS ${GG_LIB_PATH})
    IF(NOT GGML_LIB)
        message(FATAL_ERROR "GGML_LIB library not found @${GG_LIB_PATH}")
    endif()
endif()

message(STATUS "KOIFISH GG_LIB=${GG_LIB} cc=${DELTA_cc} lib=${DELTA_LIB} PWD=${CMAKE_SOURCE_DIR}")

add_subdirectory(src/Manifold)
add_subdirectory(src/Fuzi)
add_subdirectory(src/ggex)
 
set(sources   ${TGraph_sources} ${util_sources} ${kernel_sources} ${Fuzi_sources} ${ggex_sources})
add_executable(koifish ${CMAKE_CURRENT_SOURCE_DIR}/src/koifish.cpp ${sources} )

foreach(prog koifish )    #finetune koifish train2
    install(TARGETS ${prog} RUNTIME)
    target_include_directories(${prog} PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/src ${CMAKE_SOURCE_DIR}/llama.cpp/common ${CMAKE_SOURCE_DIR}/llama.cpp/include ${CMAKE_SOURCE_DIR}/llama.cpp/src ${CMAKE_SOURCE_DIR}/src/ggex/)
    target_link_libraries(${prog} PRIVATE ${GG_LIB} ${CMAKE_THREAD_LIBS_INIT} ${OPENBLAS_LIB} ${DELTA_LIB} pthread)
    target_compile_features(${prog} PRIVATE cxx_std_17)
    target_compile_options(${prog} PRIVATE  -DLLAMA_API_INTERNAL ${cxx_flags} ${DELTA_cc})
endforeach(prog)

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


