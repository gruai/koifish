# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(TGraph_sources  ${PWD}/Fish.cpp ${PWD}/TGraph.cpp ${PWD}/Optimizer.cpp ${PWD}/Attention.cpp 
                    ${PWD}/gLLM.cpp ${PWD}/MoE.cpp ${PWD}/GoPT.cpp ${PWD}/DataLoader.cpp ${PWD}/Dictionary.cpp 
                    ${PWD}/Salp.cpp 
PARENT_SCOPE )

add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)
