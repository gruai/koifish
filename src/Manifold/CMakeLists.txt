# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(manifold_sources  ${PWD}/TGraph.cpp ${PWD}/Optimizer.cpp                      
                    ${PWD}/Neuron.cpp ${PWD}/SparseNeuron.cpp ${PWD}/Scheduler.cpp ${PWD}/Serialize.cpp 

PARENT_SCOPE )

add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)
