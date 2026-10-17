# Add this directory for internal users.
include_directories(.)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(token_sources  ${PWD}/DataLoader.cpp ${PWD}/Dictionary.cpp ${PWD}/TokenSet.cpp ${PWD}/TokenSet_Json.cpp ${PWD}/HF_Tokenizer.cpp 
PARENT_SCOPE )

# set_source_files_properties(
#     ${PWD}/TokenSet_Json.cpp
#     TARGET_DIRECTORY koifish
#     PROPERTIES COMPILE_FLAGS "-g1"
# )
