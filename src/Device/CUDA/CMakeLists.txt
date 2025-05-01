set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(PWDc ${CMAKE_CURRENT_SOURCE_DIR}/../../../X/CALM/)
if(K_USECUDA)
  set(device_sources  ${PWD}/E_GPU.cpp ${PWD}/NeuronFuse.cu ${PWD}/QKV.cu ${PWD}/huTensor.cu ${PWD}/Transformer.cu ${PWD}/T.cu 
                      # ${PWDc}run_caml.cpp ${PWDc}/tensors.c ${PWDc}/tokenizer.c  ${PWDc}/infer.cu ${PWDc}/infer.c 
    PARENT_SCOPE )
endif()
