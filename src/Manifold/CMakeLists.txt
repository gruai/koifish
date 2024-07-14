# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(TGraph_sources  ${PWD}/TGraph.cpp ${PWD}/Optimizer.cpp ${PWD}/Attention.cpp 
                    ${PWD}/gLAMA.cpp ${PWD}/GPT.cpp 
PARENT_SCOPE )

add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)
