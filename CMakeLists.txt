cmake_minimum_required(VERSION 3.28)

# Set CUDA compiler before project() declaration
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)

# Enable CUDA first
enable_language(CUDA)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Then declare project with languages
project(DeepL LANGUAGES CXX CUDA)

# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 17)

# Make sure CUDA can find its headers
include_directories("/usr/local/cuda/include")  # Add this line

set(PYTHON_MODULE_NAME deeplearning)
set(CPP_LIBRARY_NAME deeplearning_cpp)
set(CUDA_SOURCES src/tensor.cu)
set(CPP_SOURCES src/Graph.cpp src/Layer.cpp src/Loss.cpp src/Optimizer.cpp src/Utils.cpp src/GradMode.cpp)

list(APPEND CMAKE_PREFIX_PATH "/home/ak6913/miniconda3/envs/deeplnrn/lib/python3.12/site-packages/pybind11/share/cmake")

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Python module
pybind11_add_module(${PYTHON_MODULE_NAME} src/deeplearning.cpp ${CUDA_SOURCES} ${CPP_SOURCES})
set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Add CUDA include directories explicitly
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}  # Add this line
    ${CMAKE_SOURCE_DIR}/src    
)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)

# C++ library
add_library(${CPP_LIBRARY_NAME} STATIC ${CUDA_SOURCES} ${CPP_SOURCES})
set_target_properties(${CPP_LIBRARY_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${CPP_LIBRARY_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}  # Add this line
    ${CMAKE_SOURCE_DIR}/src    
)
target_link_libraries(${CPP_LIBRARY_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)