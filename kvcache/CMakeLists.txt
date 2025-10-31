cmake_minimum_required(VERSION 3.25)
project(kvcache LANGUAGES CXX CUDA)

# Set policy to handle c-ares CMake version requirement
if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Suppress gRPC always_inline warnings that we can't fix in third-party code
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-attributes)
endif()

find_package(Threads REQUIRED)

add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(tools)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
