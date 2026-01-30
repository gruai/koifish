# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(util_sources ${PWD}/GST_util.cpp ${PWD}/GST_float.cpp ${PWD}/Cache.cpp ${PWD}/GST_rander.cpp 
                ${PWD}/CLI_params.cpp ${PWD}/GST_MemBuffer.cpp ${PWD}/GST_Application.cpp
PARENT_SCOPE )
add_compile_options(-Wunused -Wextra -Wcast-qual -Wno-unused-variable -Wunused-parameter -D_USE_OPENBLAS_)

