set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(Pcu ${CMAKE_CURRENT_SOURCE_DIR}/CUDA/)
if(K_USECUDA)
  set(device_sources  ${PWD}/EDevice.cpp ${Pcu}/E_GPU.cpp ${Pcu}/NeuronFuse.cu ${Pcu}/QKV.cu ${Pcu}/huTensor.cu ${Pcu}/Transformer.cu ${Pcu}/T.cu 
    ${Pcu}/Optimizer.cu 
                      # ${PWDc}run_caml.cpp ${PWDc}/tensors.c ${PWDc}/tokenizer.c  ${PWDc}/infer.cu ${PWDc}/infer.c 
    PARENT_SCOPE )
endif()
