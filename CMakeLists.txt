cmake_minimum_required (VERSION 3.27)
unset(ENV{CFLAGS})
unset(ENV{CXXFLAGS})
unset(ENV{CPPFLAGS})
unset(ENV{LDFLAGS})

project(koifish VERSION 0.2.0)

message(STATUS "CONDA_ENV      = $ENV{CONDA_DEFAULT_ENV}")
message(STATUS "Python_which = $ENV{CONDA_PREFIX}/bin/python")

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
set(PWD ${CMAKE_CURRENT_SOURCE_DIR})

# Limit parallel link jobs to prevent memory exhaustion
set(CMAKE_JOB_POOLS link_pool=2) # Try 2 or 4 depending on your RAM
set(CMAKE_JOB_POOL_LINK link_pool)

option(K_USE_CCACHE                 "KOIFISH: use ccache if available"       ON)

option(K_NPU_LIB                    "KOIFISH: libs run on NPU(cu,amd,...)"  ON)
# option(K_USE_GGML                   "KOIFISH: use GGML libs"        OFF)
option(K_USE_OPENMP                 "KOIFISH: Enable OpenMP"                 ON)
option(K_USE_OPENBLAS               "KOIFISH: Enable OpenBLAS"               OFF)
option(K_USE_GBDT                   "KOIFISH: Enable Gradient boosting trees"               ON)
option(K_USE_EIGEN                  "KOIFISH: Enable Eigen lib"               OFF)
# option(K_USE_TVM                    "KOIFISH: Enable TVM lib"               OFF)

option(K_TESTING                    "KOIFISH: Do some testing"               OFF)
option(K_WATCHING                   "KOIFISH: Do some watching"              OFF)
# option(K_USE_CUFILE "Enable cuFile support for optimized file I/O" OFF)
# option(K_USE_NVML "Enable nvml support for detailed GPU stats reporting" OFF)
# option(K_USE_MPI "Enable MPI backend for multi-GPU support" OFF)

# option(K_USE_ARROW                  "KOIFISH: Enable arrow for parquet"      ON)
include(CMakeDependentOption)
cmake_dependent_option(K_USE_ARROW    "KOIFISH: Enable arrow for parquet"    ON     "USEARROW"      OFF )

if (K_USE_ARROW)
set(MY_CXX_STANDARD 20)
message(STATUS "arrow lib needs c++20, so set CXX_STANDARD=20")
else()
set(MY_CXX_STANDARD 17)
message(STATUS "Set CXX_STANDARD=17")
endif()


# set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
# set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")
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
set(FISH_INC ${CMAKE_SOURCE_DIR}/src/Manifold/ ${CMAKE_SOURCE_DIR}/src/TokenSet/ ${CMAKE_SOURCE_DIR}/src/Utils/ )
find_package(Threads REQUIRED)

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

set(DELTA_cc ${DELTA_cc} -mf16c -mavx2 -mfma)
# set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-lto")   #   Fix the Underlying DWARF Debug Info
# -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer

if(K_USE_OPENBLAS)
    add_compile_definitions(_USE_OPENBLAS_)
    set(MORE_INC ${MORE_INC} /home/cys/Downloads/OpenBLAS-0.3.27/build/generated/ /home/cys/Downloads/OpenBLAS-0.3.27/build/ )
    set(OPENBLAS_LIB  /home/cys/Downloads/OpenBLAS-0.3.27/libopenblas_haswellp-r0.3.27.so  )  
    set(blas_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/BLAS_t.cpp )
    # add_subdirectory(src/lenda/util)
    add_subdirectory(src/lenda/kernel)
else()
endif()

if(K_USE_ARROW)
    add_compile_definitions(_USE_ARROW_)
    set(ARROW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin/arrow/lib")
    set(ARROW_INC "${CMAKE_CURRENT_SOURCE_DIR}/bin/arrow/include")
    # find_package(Arrow REQUIRED)
    # find_package(Parquet REQUIRED)    
    # set(ARROW_LIB  Arrow::arrow_shared Parquet::parquet_shared  )  
    #  snappy zstd lz4 brotli
    set(ARROW_LIB ${ARROW_DIR}/libparquet.so  ${ARROW_DIR}/libarrow.so)
else()
endif()

if(K_WATCHING)
    find_package(X11 REQUIRED)
    # set(WATCH_LIB  /home/cys/rnd/lic/bin/X/libwatch.so X11::X11) 

    find_package(OpenGL REQUIRED)
    set(VTK_DIR ~/Github/vtk-src/vtk-static)     
    find_package(VTK REQUIRED COMPONENTS
        CommonCore
        CommonDataModel      
        IOXML
        ImagingSources
        FiltersSources       
        RenderingCore
        RenderingOpenGL2
        RenderingContextOpenGL2
        RenderingFreeType
        InteractionStyle
        ChartsCore
        RenderingAnnotation
        InteractionWidgets
        IOVideo
        RenderingContext2D
    )
    message(STATUS "VTK_VERSION = ${VTK_VERSION}")
    add_compile_definitions(__USE_WATCHING__)    
    set(WATCH_LIB ${VTK_LIBRARIES} X11::X11 OpenGL::GL pthread dl) 
    add_subdirectory(X/WATCH/)
endif()

if(K_USE_EIGEN)
    add_subdirectory(src/lenda/Cheby)
    set(MORE_INC ${MORE_INC} /home/cys/rnd/gssx/gsp_3.0/library/eigen-3.4.0/)
endif()


if(K_NPU_LIB)
    set(NPU_LIB npu_lib)    
else()
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        # set(cdir _cuda)
        # set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89")  BF16  Ref:    https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        # SM_89+ for FP8 support
        set(CMAKE_CUDA_ARCHITECTURES "89") 
        # compile for native GPU if no GPU has been specified
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
        add_compile_definitions(__USE_CUDA__ _USE_CUDA_FLOAT_)
        message(STATUS "CUDA ARCH=${CMAKE_CUDA_ARCHITECTURES} DIR=${CUDA_TOOLKIT_ROOT_DIR} INCLUDE=${CUDAToolkit_INCLUDE_DIRS}")
        enable_language(CUDA)
        
        set(CUDA_FLAGS -use_fast_math  -DENABLE_BF16 --expt-relaxed-constexpr -g0 )
        # -mavx512f -mavx512vl  for identifier "__builtin_ia32_ldtilecfg" is undefined
        #  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills
        # -std=c++20
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
        set(CUDA_LIB  cudart cublas cublasLt ) #cuda_driver  
    else()
        message(WARNING "CUDA not found")
    endif()

    set(MORE_INC ${MORE_INC} ${CMAKE_SOURCE_DIR}/third_party/cutlass/include/ ${CMAKE_SOURCE_DIR}/src/Device/tile_kernel/ ${CMAKE_SOURCE_DIR}/cudnn-frontend/include ${CMAKE_SOURCE_DIR}/src/Device/CUDA/TK/include)
    set(NPU_LIB  ${CUDA_LIB} cudnn )  
    add_compile_definitions(__USE_CUDNN__)  
endif()

if(K_USE_GBDT)
    add_compile_definitions(__USE_GBDT__)
    add_subdirectory(src/GBDT)
endif()

message(STATUS "KOIFISH cc=${DELTA_cc} CXX=${MY_CXX_STANDARD} SHARED = ${BUILD_SHARED_LIBS} PWD=${CMAKE_SOURCE_DIR}")

add_subdirectory(src/Tensor)
add_subdirectory(src/Shoal)
add_subdirectory(src/Manifold)
add_subdirectory(src/Decision)
add_subdirectory(src/Fuzi)
add_subdirectory(src/Utils)
add_subdirectory(src/Dynamics)
add_subdirectory(src/TokenSet)
add_subdirectory(src/Utils/oniguruma)
add_subdirectory(src/Device)

set(sources   ${manifold_sources} ${decision_sources} ${util_sources} ${tensor_sources} ${kernel_sources} ${cheby_sources} ${watch_sources} ${Fuzi_sources} ${ggex_sources} ${device_sources} ${model_sources} 
    ${token_sources} ${mort_sources} ${blas_sources} ${dynamics_sources} ${TEST_SRCS} )
add_executable(koifish ${CMAKE_CURRENT_SOURCE_DIR}/src/koifish.cpp ${sources} )
add_executable(bubble ${CMAKE_CURRENT_SOURCE_DIR}/src/bubble.cpp ${sources} )
# add_executable(pangpi ${CMAKE_CURRENT_SOURCE_DIR}/src/pangpi.cpp ${sources} )
message(STATUS "KOIFISH sources=${sources} device_sources=${device_sources}  NPU_LIB=${NPU_LIB}")

add_library(prog_deps INTERFACE)
target_link_libraries(prog_deps
  INTERFACE
    ${GG_LIB}    ${OPENBLAS_LIB}  ${WATCH_LIB}   ${TVM_LIB}    ${GFLAGS_LIBRARIES} ${ARROW_LIB} ${NPU_LIB}
    gbdt
    onig  
    Threads::Threads
)
add_compile_definitions(_USE_CUDA_FLOAT_)
find_package(CUDAToolkit REQUIRED)
target_include_directories(prog_deps
  INTERFACE ${FISH_INC} ${GG_INC} ${MORE_INC} ${ARROW_INC} ${CUDAToolkit_INCLUDE_DIRS}
)
get_target_property(KOIFISH_LINK_LIBRARIES prog_deps INTERFACE_LINK_LIBRARIES)
message(STATUS "KOIFISH_INC_DIRS = ${FISH_INC} ${GG_INC} MORE_INC=${MORE_INC} ARROW_INC=${ARROW_INC}")
message(STATUS "KOIFISH_LINK_LIBRARIES = ${KOIFISH_LINK_LIBRARIES}")


foreach(prog koifish bubble)    
    install(TARGETS ${prog} RUNTIME)
    set_target_properties(${prog} PROPERTIES
        CXX_STANDARD ${MY_CXX_STANDARD}
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    # target_compile_features(${prog} PRIVATE cxx_std_17)
    target_compile_options(${prog} PRIVATE ${cxx_flags} ${DELTA_cc})

    target_compile_definitions(${prog} PRIVATE  KOIFISH_CONDA_ENV="$ENV{CONDA_DEFAULT_ENV}")        
    target_link_libraries(${prog} PRIVATE prog_deps )
endforeach(prog)

if (K_TESTING)
    add_subdirectory(tests)
    add_subdirectory(tests/fp6)
    find_package(gflags REQUIRED)
    add_executable(k_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp ${sources} ${FP6_sources} )
    foreach(prog k_test )    
        target_include_directories(${prog} PRIVATE ${FISH_INC} ${GG_INC} ${MORE_INC} )

        target_link_libraries(${prog} PRIVATE prog_deps )
        target_compile_features(${prog} PRIVATE cxx_std_17)
        target_compile_options(${prog} PRIVATE ${cxx_flags} ${DELTA_cc})
        # target_link_libraries(${prog} PRIVATE pybind11::embed)
    endforeach(prog)

endif()



