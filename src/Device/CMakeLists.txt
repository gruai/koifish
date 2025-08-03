set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(Pcu ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/)
set(Pca ${CMAKE_CURRENT_SOURCE_DIR}/../../X/CALM/)
if(K_USECUDA)
  set(device_sources  ${PWD}/EDevice.cpp ${Pcu}/E_GPU.cpp ${Pcu}/NeuronFuse.cu ${Pcu}/QKV.cu ${Pcu}/huTensor.cu ${Pcu}/T.cu 
    ${Pcu}/kernel/gemm.cu ${Pcu}/kernel/abc_bit.cu 
    ${Pcu}/Optimizer.cu
    ${Pcu}/Generate.cu 
                # ${Pca}run_caml.cpp ${Pca}/tensors.c ${Pca}/tokenizer.c  ${Pca}/infer.cu ${Pca}/infer.c 
    PARENT_SCOPE )
endif()
