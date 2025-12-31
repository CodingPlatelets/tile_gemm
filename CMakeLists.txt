cmake_minimum_required(VERSION 3.18)
project(tile_gemm LANGUAGES CUDA CXX)

# C++ and CUDA standard settings (compatible with gcc7.5)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architecture (sm_70 for V100, adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES 70)

# CUDA runtime shared (for gpgpu-sim)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart shared")

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find CUDA Toolkit and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library sources (shared between main and test)
set(LIB_SOURCES
    src/tile_gemm.cu
    src/sparse_to_dense.cu
    src/tile_reader.cpp
)

# Create a static library for shared code
add_library(tile_gemm_lib STATIC ${LIB_SOURCES})
target_link_libraries(tile_gemm_lib CUDA::cublas)

# Main executable
add_executable(tile_gemm src/main.cu)
target_link_libraries(tile_gemm tile_gemm_lib CUDA::cublas)

# Test executable
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness tile_gemm_lib CUDA::cublas)

# Set output directory
set_target_properties(tile_gemm test_correctness
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Custom targets for running
add_custom_target(run
    COMMAND tile_gemm ${CMAKE_SOURCE_DIR}/test_data
    DEPENDS tile_gemm
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tile_gemm with test_data"
)

add_custom_target(run_test
    COMMAND test_correctness ${CMAKE_SOURCE_DIR}/test_data
    DEPENDS test_correctness
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running correctness test"
)

add_custom_target(run_test_verbose
    COMMAND test_correctness ${CMAKE_SOURCE_DIR}/test_data -v
    DEPENDS test_correctness
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running correctness test (verbose)"
)
