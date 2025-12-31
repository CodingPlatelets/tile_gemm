#!/bin/bash
# CMake build script for tile_gemm
# This script builds the project and creates compile_commands.json symlink for clangd

set -e

BUILD_DIR="cmake_build"

# Create build directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Configure with CMake
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
cmake --build . -j$(nproc)

# Create symlink for compile_commands.json in project root (for clangd)
cd ..
if [ -f "${BUILD_DIR}/compile_commands.json" ]; then
    ln -sf ${BUILD_DIR}/compile_commands.json compile_commands.json
    echo "Created symlink: compile_commands.json -> ${BUILD_DIR}/compile_commands.json"
fi

echo ""
echo "Build complete! Executables are in ${BUILD_DIR}/"
echo "  - ${BUILD_DIR}/tile_gemm"
echo "  - ${BUILD_DIR}/test_correctness"

