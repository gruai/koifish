# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(cheby_sources ${PWD}/ChebTools.cpp
PARENT_SCOPE )
# add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter -D_USE_OPENBLAS_)

