# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(SpMV_sources ${PWD}/Matrix.cpp ${PWD}/GVMAT.cpp ${PWD}/Pattern.cpp 
PARENT_SCOPE )
add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter)

