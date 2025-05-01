# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(TGraph_sources  ${PWD}/Fish.cpp ${PWD}/TGraph.cpp ${PWD}/Optimizer.cpp ${PWD}/Evaluate.cpp 
                    ${PWD}/gLLM.cpp ${PWD}/gLLMx.cpp ${PWD}/MoE.cpp ${PWD}/GoPT.cpp 
                    ${PWD}/Salp.cpp ${PWD}/Neuron.cpp ${PWD}/SparseNeuron.cpp ${PWD}/Scheduler.cpp ${PWD}/EmbedVAE.cpp ${PWD}/Serialize.cpp ${PWD}/Transformer.cpp
PARENT_SCOPE )

add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)
