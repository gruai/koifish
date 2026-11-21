cmake_minimum_required (VERSION 3.27)
project(npu_lib LANGUAGES CXX CUDA)

option(K_USECUDA                    "KOIFISH: CUDA"                 ON)
option(K_DEBUGCUDA                  "KOIFISH: DEBUG_CUDA[slow!]"                 OFF)
option(K_USE_CUDNN                  "KOIFISH: use CUDNN libs"         ON)
option(K_USE_TILELANG               "KOIFISH: Enable TileLang"              ON)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(Pcu ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/)

if(K_USE_TILELANG)
    set(Ptl ${CMAKE_CURRENT_SOURCE_DIR}/tile_kernel/)
    set(device_sources  ${Ptl}/gemm_v0.cu ${Ptl}/qkv_v0.cu ${Ptl}/header_cls.cu ${Ptl}/utils.cu)
    set(TL_INC ${Ptl} ${CMAKE_SOURCE_DIR}/third_party/cutlass/include) 
    add_compile_definitions(__USE_TILELANG__)
    set(K_USE_CUDNN ON)
else()
    # set(K_USE_CUDNN ON)
endif()

if(K_USECUDA)
  set(device_sources ${device_sources} ${PWD}/EDevice.cpp ${PWD}/Pipe.cpp  ${Pcu}/E_GPU.cpp ${Pcu}/NeuronFuse.cu ${Pcu}/Activation.cu 
    ${Pcu}/QKV.cu ${Pcu}/QKV_cudnn.cu 
    ${Pcu}/kernel/rope.cu ${Pcu}/huTensor.cu ${Pcu}/T.cu 
    ${Pcu}/kernel/gemm.cu ${Pcu}/kernel/abc_bit.cu ${Pcu}/kernel/quantizer.cu 
    ${Pcu}/Optimizer.cu
    ${Pcu}/Generate.cu
    ${PWD}/SelfAttention.cpp
    )
endif()

# set_property(GLOBAL PROPERTY JOB_POOLS cuda_pool=1)
# set_source_files_properties(${Pcu}/QKV_cudnn.cu PROPERTIES JOB_POOL_COMPILE cuda_pool)

message(STATUS "K_NPU_LIB=${K_NPU_LIB} K_USECUDA=${K_USECUDA} device_sources=${device_sources}")
if(K_NPU_LIB)
  if(K_USECUDA)
      include(CheckLanguage)
      check_language(CUDA)
      if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
      else()
        message(FATAL_ERROR "CUDA compiler (nvcc) not found. Please check your CUDA installation.")
      endif()
      # set(CMAKE_CUDA_ARCHITECTURES "89")

      find_package(CUDAToolkit REQUIRED)
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
          set(CUDA_LIB  cudart cublas cublasLt )   
      else()
          message(WARNING "CUDA not found")
      endif()
  endif()

  if(K_USE_CUDNN)    
      set(CUDA_LIB  ${CUDA_LIB} cudnn )   
      set(CUDA_INC  ${CMAKE_SOURCE_DIR}/cudnn-frontend/include ${CMAKE_SOURCE_DIR}/src/Device/CUDA/TK/include  ) 
      add_compile_definitions(__USE_CUDNN__)  
      message(STATUS "CUDNN INC=${CUDA_INC}")
  endif()
  add_library(npu_lib SHARED ${device_sources})

  # Key: enable separable compilation (required for cross-file device linking)
  set_target_properties(npu_lib PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      POSITION_INDEPENDENT_CODE ON      # If the final program is PIE or a shared library, this is needed
  )
 
  message(STATUS "NPULIB_INC_DIRS = ${FISH_INC} ${CUDA_INC} ${TL_INC} ${CMAKE_CURRENT_SOURCE_DIR}")
  target_include_directories(npu_lib PRIVATE  ${FISH_INC} ${CUDA_INC} ${TL_INC} ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

  # Link the CUDA runtime (PUBLIC so the parent also automatically links it)
  target_link_libraries(npu_lib PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cublasLt)
  target_link_libraries(npu_lib PRIVATE cudnn)
else()
  set(device_sources ${device_sources} PARENT_SCOPE )
endif()
