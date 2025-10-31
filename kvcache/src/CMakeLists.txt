set(name ${PROJECT_NAME})

# dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Python 3 REQUIRED COMPONENTS Development Interpreter)
find_package(Boost REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(NCCL REQUIRED)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# sources list
set(source
    pybind.cpp
    kernels/gather.cu
    kernels/scatter.cu)

# library
Python_add_library(${name} MODULE ${header} ${source} WITH_SOABI)
target_include_directories(${name} PRIVATE .)
target_compile_definitions(${name} PRIVATE MODULE_NAME=${name} TORCH_EXTENSION_NAME=${name})
target_compile_features(${name} PRIVATE cxx_std_17)
target_link_libraries(${name} PRIVATE CUDA::cudart nccl::nccl Python::Python Boost::boost ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(${name} PROPERTIES
  CXX_EXTENSIONS OFF
  PREFIX "")

if(DEFINED PY_BUILD_CMAKE_IMPORT_NAME)
  install(TARGETS ${name} DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})
endif()
