# include(GNUInstallDirs)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATH_SUFFIXES include)
find_library(NCCL_LIBRARY NAMES libnccl.so.2 nccl PATH_SUFFIXES lib lib64)

if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "Found NCCL include directory: ${NCCL_INCLUDE_DIR}")
    message(STATUS "Found NCCL library: ${NCCL_LIBRARY}")
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NCCL
        DEFAULT_MSG
        NCCL_LIBRARY
        NCCL_INCLUDE_DIR)
    mark_as_advanced(NCCL_LIBRARY NCCL_INCLUDE_DIR)
else()
    message(SEND_ERROR "Cannot find NCCL, you may manually add NCCL_HOME path to CMAKE_PREFIX_PATH")
endif()

if(NCCL_FOUND AND NOT TARGET nccl::nccl)
    add_library(nccl::nccl SHARED IMPORTED)
    set_target_properties(nccl::nccl PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NCCL_INCLUDE_DIR}
        IMPORTED_LOCATION ${NCCL_LIBRARY})
endif()
