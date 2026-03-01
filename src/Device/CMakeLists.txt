set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(Pcu ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/)

if(K_USECUDA)
  set(device_sources  ${PWD}/EDevice.cpp ${PWD}/Pipe.cpp  ${Pcu}/E_GPU.cpp ${Pcu}/NeuronFuse.cu ${Pcu}/QKV.cu ${Pcu}/kernel/rope.cu ${Pcu}/huTensor.cu ${Pcu}/T.cu 
    ${Pcu}/kernel/gemm.cu ${Pcu}/kernel/abc_bit.cu ${Pcu}/kernel/quantizer.cu 
    ${Pcu}/Optimizer.cu
    ${Pcu}/Generate.cu
    ${PWD}/SelfAttention.cpp
    PARENT_SCOPE )
endif()
