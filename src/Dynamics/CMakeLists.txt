# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(dynamics_sources  ${PWD}/Langevin.cpp ${PWD}/Transition.cpp PARENT_SCOPE )

add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)
