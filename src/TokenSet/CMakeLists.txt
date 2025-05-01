# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(token_sources  ${PWD}/DataLoader.cpp ${PWD}/Dictionary.cpp ${PWD}/TokenSet.cpp 
PARENT_SCOPE )


