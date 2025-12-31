#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#include "tile_reader.h"
#include "tile_gemm.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <data_folder_path> [output_file]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  data_folder_path  Path to the data folder containing meta.txt and tile folders" << std::endl;
    std::cout << "  output_file       Output file path (default: result.txt in data_folder)" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " ./test_data" << std::endl;
    std::cout << "  " << program_name << " ./test_data ./output/result.txt" << std::endl;
}

void printDeviceInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string data_path = argv[1];
    std::string output_path;
    
    if (argc >= 3) {
        output_path = argv[2];
    } else {
        output_path = data_path + "/result.txt";
    }
    
    // 打印设备信息
    printDeviceInfo();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. 读取数据
    std::cout << "\n[Step 1] Reading tile data from: " << data_path << std::endl;
    
    TileReader reader(data_path);
    if (!reader.readAll()) {
        std::cerr << "Error reading data: " << reader.getError() << std::endl;
        return 1;
    }
    
    const MatrixData& matrix_A = reader.getMatrixA();
    const MatrixData& matrix_B = reader.getMatrixB();
    
    std::cout << "  Matrix A: " << matrix_A.meta.rows << " x " << matrix_A.meta.cols 
              << " (" << matrix_A.meta.num_tile_rows << " x " << matrix_A.meta.num_tile_cols << " tiles)" << std::endl;
    std::cout << "  Matrix B: " << matrix_B.meta.rows << " x " << matrix_B.meta.cols
              << " (" << matrix_B.meta.num_tile_rows << " x " << matrix_B.meta.num_tile_cols << " tiles)" << std::endl;
    std::cout << "  A tiles loaded: " << matrix_A.tiles.size() << std::endl;
    std::cout << "  B tiles loaded: " << matrix_B.tiles.size() << std::endl;
    
    // 2. 初始化GEMM计算器
    std::cout << "\n[Step 2] Initializing cuBLAS..." << std::endl;
    
    TileGEMM gemm;
    if (!gemm.init()) {
        std::cerr << "Error initializing cuBLAS: " << gemm.getError() << std::endl;
        return 1;
    }
    std::cout << "  cuBLAS initialized successfully" << std::endl;
    
    // 3. 执行分块矩阵乘法
    std::cout << "\n[Step 3] Computing tile-based GEMM..." << std::endl;
    
    int result_rows = matrix_A.meta.rows;
    int result_cols = matrix_B.meta.cols;
    int result_size = result_rows * result_cols;
    
    float* h_result = new float[result_size];
    
    auto compute_start = std::chrono::high_resolution_clock::now();
    
    if (!gemm.compute(matrix_A, matrix_B, h_result, result_rows, result_cols)) {
        std::cerr << "Error computing GEMM: " << gemm.getError() << std::endl;
        delete[] h_result;
        return 1;
    }
    
    auto compute_end = std::chrono::high_resolution_clock::now();
    auto compute_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start);
    
    std::cout << "  GEMM completed in " << compute_duration.count() << " ms" << std::endl;
    std::cout << "  Result matrix: " << result_rows << " x " << result_cols << std::endl;
    
    // 4. 保存结果
    std::cout << "\n[Step 4] Saving result to: " << output_path << std::endl;
    
    if (!saveResultToFile(h_result, result_rows, result_cols, output_path)) {
        std::cerr << "Error saving result" << std::endl;
        delete[] h_result;
        return 1;
    }
    
    // 计算总时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 打印结果矩阵的前几个元素（用于验证）
    std::cout << "\nResult matrix preview (top-left 4x4):" << std::endl;
    int preview_rows = std::min(4, result_rows);
    int preview_cols = std::min(4, result_cols);
    for (int i = 0; i < preview_rows; ++i) {
        for (int j = 0; j < preview_cols; ++j) {
            // 列主序
            int idx = j * result_rows + i;
            std::cout << std::fixed << std::setprecision(4) << h_result[idx] << "\t";
        }
        std::cout << std::endl;
    }
    
    delete[] h_result;
    return 0;
}

