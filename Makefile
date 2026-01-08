# CUDA Tile GEMM Makefile
# Compatible with CUDA 11.0+

# CUDA compiler
NVCC = nvcc

# Compiler flags
# for gpgpu sim
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_70 --cudart shared
INCLUDES = -I./include -I./third_party/cutlass/include
LIBS =

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
TEST_DIR = tests

# Source files
CUDA_SOURCES = $(SRC_DIR)/main.cu $(SRC_DIR)/tile_gemm.cu $(SRC_DIR)/sparse_to_dense.cu
CPP_SOURCES = $(SRC_DIR)/tile_reader.cpp

# Object files (library components, without main)
LIB_OBJECTS = $(BUILD_DIR)/tile_gemm.o $(BUILD_DIR)/sparse_to_dense.o $(BUILD_DIR)/tile_reader.o $(BUILD_DIR)/cutlass_gemm.o

# Main program object
MAIN_OBJECT = $(BUILD_DIR)/main.o

# Target executable (in build directory)
TARGET = $(BUILD_DIR)/tile_gemm

# Test target
TEST_TARGET = $(BUILD_DIR)/test_correctness

# Default target
all: $(BUILD_DIR) $(TARGET)

# Build with tests
all_with_tests: $(BUILD_DIR) $(TARGET) $(TEST_TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link main program
$(TARGET): $(MAIN_OBJECT) $(LIB_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Link test program
$(TEST_TARGET): $(BUILD_DIR)/test_correctness.o $(LIB_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LIBS)

# Compile main
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Compile CUDA sources
$(BUILD_DIR)/tile_gemm.o: $(SRC_DIR)/tile_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/sparse_to_dense.o: $(SRC_DIR)/sparse_to_dense.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD_DIR)/cutlass_gemm.o: $(SRC_DIR)/cutlass_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Compile C++ sources
$(BUILD_DIR)/tile_reader.o: $(SRC_DIR)/tile_reader.cpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Compile test sources
$(BUILD_DIR)/test_correctness.o: $(TEST_DIR)/test_correctness.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ $<

# Build only tests
test: $(BUILD_DIR) $(TEST_TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Run main program with test data
run: $(TARGET)
	$(TARGET) ./test_data

# Run correctness test
run_test: $(TEST_TARGET)
	$(TEST_TARGET) ./test_data

# Run correctness test with verbose output
run_test_verbose: $(TEST_TARGET)
	$(TEST_TARGET) ./test_data -v

.PHONY: all all_with_tests test clean run run_test run_test_verbose
