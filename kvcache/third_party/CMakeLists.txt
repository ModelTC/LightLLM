include(FetchContent)

# Check if grpc directory already exists locally
set(LOCAL_GRPC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/grpc")

if(EXISTS "${LOCAL_GRPC_DIR}/CMakeLists.txt")
    message(STATUS "Using local gRPC from ${LOCAL_GRPC_DIR}")
    # Add the local grpc directory directly
    set(protobuf_INSTALL OFF)
    set(utf8_range_ENABLE_INSTALL OFF)
    set(FETCHCONTENT_QUIET OFF)
    add_subdirectory(${LOCAL_GRPC_DIR} grpc-build)
else()
    message(STATUS "Local gRPC not found, downloading from GitHub...")
    # Or we may use `find_package(gRPC CONFIG REQUIRED)` to look up a system-installed one.
    # See: https://github.com/grpc/grpc/blob/master/src/cpp/README.md#cmake
    FetchContent_Declare(
        gRPC
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG v1.71.0
        GIT_SHALLOW ON
        SYSTEM
        FIND_PACKAGE_ARGS CONFIG NAMES gRPC
        SOURCE_DIR "${LOCAL_GRPC_DIR}")
    set(protobuf_INSTALL OFF)
    set(utf8_range_ENABLE_INSTALL OFF)

    set(FETCHCONTENT_QUIET OFF)
    FetchContent_MakeAvailable(gRPC)
endif()
