cmake_minimum_required(VERSION 3.5)

project(hwinfo)

set(PWD ${CMAKE_CURRENT_SOURCE_DIR}/src/)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

set(hw_file 
    ${PWD}/hwinfo.cpp
    # platform independent definitions
    ${PWD}/PCIMapper.cpp
    ${PWD}/battery.cpp
    ${PWD}/cpu.cpp
    ${PWD}/gpu.cpp
    ${PWD}/ram.cpp
    ${PWD}/os.cpp
    ${PWD}/mainboard.cpp
    ${PWD}/disk.cpp
    # apple specific definitions
    ${PWD}/apple/utils/filesystem.cpp
    ${PWD}/apple/battery.cpp
    ${PWD}/apple/cpu.cpp
    ${PWD}/apple/gpu.cpp
    ${PWD}/apple/ram.cpp
    ${PWD}/apple/os.cpp
    ${PWD}/apple/mainboard.cpp
    ${PWD}/apple/disk.cpp
    # linux specific definitions
    ${PWD}/linux/utils/filesystem.cpp
    ${PWD}/linux/battery.cpp
    ${PWD}/linux/cpu.cpp
    ${PWD}/linux/gpu.cpp
    ${PWD}/linux/ram.cpp
    ${PWD}/linux/os.cpp
    ${PWD}/linux/mainboard.cpp
    ${PWD}/linux/disk.cpp
    # windows specific definitions
    ${PWD}/windows/utils/filesystem.cpp
    ${PWD}/windows/utils/wmi_wrapper.cpp
    ${PWD}/windows/battery.cpp
    ${PWD}/windows/cpu.cpp
    ${PWD}/windows/gpu.cpp
    ${PWD}/windows/ram.cpp
    ${PWD}/windows/os.cpp
    ${PWD}/windows/mainboard.cpp
    ${PWD}/windows/disk.cpp
PARENT_SCOPE )


