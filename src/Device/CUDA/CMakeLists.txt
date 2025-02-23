set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
if(K_TENSOR_CUDNN)
  set(device_sources  ${PWD}/E_GPU.cpp ${PWD}/cuFuse.cu ${PWD}/cuTensor.cu ${PWD}/GPT2.cu
    PARENT_SCOPE )
endif()
