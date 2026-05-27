set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(Pcu ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/)

if(K_USE_TILELANG)
  set(Ptl ${CMAKE_CURRENT_SOURCE_DIR}/tile_kernel/)
  set(device_sources  ${Ptl}/gemm_v0.cu ${Ptl}/qkv_v0.cu ${Ptl}/header_cls.cu ${Ptl}/utils.cu)
endif()

if(K_USECUDA)
  set(device_sources ${device_sources} ${PWD}/EDevice.cpp ${PWD}/Pipe.cpp  ${Pcu}/E_GPU.cpp ${Pcu}/NeuronFuse.cu ${Pcu}/Activation.cu ${Pcu}/QKV.cu ${Pcu}/kernel/rope.cu ${Pcu}/huTensor.cu ${Pcu}/T.cu 
    ${Pcu}/kernel/gemm.cu ${Pcu}/kernel/abc_bit.cu ${Pcu}/kernel/quantizer.cu 
    ${Pcu}/Optimizer.cu
    ${Pcu}/Generate.cu
    ${PWD}/SelfAttention.cpp
    PARENT_SCOPE )
endif()

