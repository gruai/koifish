# Minimal CMakeLists.txt for oniguruma library (UTF-8 only)
cmake_minimum_required(VERSION 3.5)
project(oniguruma VERSION 6.9.10 LANGUAGES C)

# Check headers for config.h generation
include(CheckIncludeFiles)
include(CheckTypeSize)

check_include_files(alloca.h HAVE_ALLOCA_H)
check_include_files(stdint.h HAVE_STDINT_H)
check_include_files(sys/times.h HAVE_SYS_TIMES_H)
check_include_files(sys/time.h HAVE_SYS_TIME_H)
check_include_files(sys/types.h HAVE_SYS_TYPES_H)
check_include_files(unistd.h HAVE_UNISTD_H)
check_include_files(inttypes.h HAVE_INTTYPES_H)
check_type_size(int SIZEOF_INT)
check_type_size(long SIZEOF_LONG)
check_type_size("long long" SIZEOF_LONG_LONG)
check_type_size("void*" SIZEOF_VOIDP)


# Generate config.h
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/config.h
"#ifndef CONFIG_H
#define CONFIG_H
#cmakedefine HAVE_ALLOCA_H 1
#cmakedefine HAVE_STDINT_H 1
#cmakedefine HAVE_SYS_TIMES_H 1
#cmakedefine HAVE_SYS_TIME_H 1
#cmakedefine HAVE_SYS_TYPES_H 1
#cmakedefine HAVE_UNISTD_H 1
#cmakedefine HAVE_INTTYPES_H 1
#define SIZEOF_INT ${SIZEOF_INT}
#define SIZEOF_LONG ${SIZEOF_LONG}
#define SIZEOF_LONG_LONG ${SIZEOF_LONG_LONG}
#define SIZEOF_VOIDP ${SIZEOF_VOIDP}
#define PACKAGE \"onig\"
#define PACKAGE_VERSION \"6.9.10\"
#define VERSION \"6.9.10\"
#endif
")


configure_file(${CMAKE_CURRENT_BINARY_DIR}/config.h ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# Core sources only (UTF-8 support)
set(_SRCS
  src/regint.h src/regparse.h src/regenc.h src/st.h
  src/regerror.c src/regparse.c src/regcomp.c src/regexec.c
  src/regenc.c src/regsyntax.c
  src/st.c src/onig_init.c
  src/unicode.c src/ascii.c src/utf8.c
  src/unicode_unfold_key.c
  src/unicode_fold1_key.c src/unicode_fold2_key.c src/unicode_fold3_key.c
)

add_library(onig ${_SRCS})

target_include_directories(onig PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

target_compile_definitions(onig PUBLIC ONIG_STATIC)
