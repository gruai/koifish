# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(util_sources ${PWD}/BLAS_t.cpp ${PWD}/GST_util.cpp 
PARENT_SCOPE )
add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter -D_USE_OPENBLAS_)

