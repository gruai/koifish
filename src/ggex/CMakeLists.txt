# if (GGML_ALL_WARNINGS)
#   if (NOT MSVC)
#       set(cxx_flags
#           # TODO(marella): Add other warnings.
#           -Wpedantic
#           -Wunused-variable
#           -Wno-unused-function
#           -Wno-multichar
#       )
#       add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>")
#   endif()
# endif()

# add_library(ggex STATIC common_sample.cpp common-ggml.cpp GG_util.cpp)
# target_link_libraries(ggex PRIVATE ggml)
# target_include_directories(ggex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(PWD ${CMAKE_CURRENT_SOURCE_DIR})
set(ggex_sources  ${PWD}/common_sample.cpp  ${PWD}/GG_util.cpp  ${PWD}/GTensor.cpp  
  PARENT_SCOPE )

