#include <iostream>
#include <string>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "tile_reader.h"
#include "sparse_to_dense.h"
#include "tile_gemm.h"
#include "cutlass_gemm.h"

// 测试配置
const float EPSILON = 1e-5f;

// 将MatrixData转换为CPU上的稠密矩阵
void matrixDataToDenseCPU(const MatrixData& matrix_data, float* h_matrix) {
    int total_rows = matrix_data.meta.rows;
    int total_cols = matrix_data.meta.cols;
    int total_size = total_rows * total_cols;
    
    // 初始化为零
    memset(h_matrix, 0, total_size * sizeof(float));
    
    // 遍历所有tiles，填充到完整矩阵
    for (const auto& pair : matrix_data.tiles) {
        const TileData& tile = pair.second;
        
        // 填充tile中的元素
        for (const auto& elem : tile.elements) {
            int global_row = elem.global_row;
            int global_col = elem.global_col;
            
            // 验证坐标有效性
            if (global_row >= 0 && global_row < total_rows &&
                global_col >= 0 && global_col < total_cols) {
                // 列主序存储
                int idx = global_col * total_rows + global_row;
                h_matrix[idx] = elem.value;
            }
        }
    }
}

// 直接矩阵乘法（将所有tile合并后使用 CUTLASS 计算）
bool directGemm(const MatrixData& matrix_A, const MatrixData& matrix_B,
                float* h_result, int result_rows, int result_cols) {
    // 初始化 CUTLASS GEMM
    CutlassGemm cutlass_gemm;
    if (!cutlass_gemm.init()) {
        std::cerr << "Failed to initialize CUTLASS GEMM" << std::endl;
        return false;
    }
    
    // 将A和B矩阵的所有tiles合并为完整的稠密矩阵
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    
    if (!matrixTilesToDense(matrix_A, &d_A)) {
        std::cerr << "Failed to convert A matrix to dense" << std::endl;
        return false;
    }
    
    if (!matrixTilesToDense(matrix_B, &d_B)) {
        std::cerr << "Failed to convert B matrix to dense" << std::endl;
        cudaFree(d_A);
        return false;
    }
    
    // 分配结果矩阵
    int result_size = result_rows * result_cols;
    cudaError_t err = cudaMalloc(&d_C, result_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for result" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return false;
    }
    cudaMemset(d_C, 0, result_size * sizeof(float));
    
    // 执行GEMM: C = A * B
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    int M = matrix_A.meta.rows;  // A的行数
    int N = matrix_B.meta.cols;  // B的列数
    int K = matrix_A.meta.cols;  // A的列数 = B的行数
    
    if (!cutlass_gemm.gemm(d_A, M,
                           d_B, K,
                           d_C, M,
                           M, N, K,
                           alpha, beta)) {
        std::cerr << "CUTLASS GEMM failed: " << cutlass_gemm.getError() << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return false;
    }
    
    // 复制结果到CPU
    cudaMemcpy(h_result, d_C, result_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return true;
}

// 分块矩阵乘法（使用我们的tile算法）
bool tiledGemm(const MatrixData& matrix_A, const MatrixData& matrix_B,
               float* h_result, int result_rows, int result_cols) {
    TileGEMM gemm;
    if (!gemm.init()) {
        std::cerr << "Failed to initialize TileGEMM: " << gemm.getError() << std::endl;
        return false;
    }
    
    if (!gemm.compute(matrix_A, matrix_B, h_result, result_rows, result_cols)) {
        std::cerr << "TileGEMM compute failed: " << gemm.getError() << std::endl;
        return false;
    }
    
    return true;
}

// 比较两个矩阵是否相等
bool compareResults(const float* result1, const float* result2, 
                    int rows, int cols, float epsilon) {
    int mismatch_count = 0;
    float max_diff = 0.0f;
    int max_diff_row = 0, max_diff_col = 0;
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = j * rows + i;  // 列主序
            float diff = std::fabs(result1[idx] - result2[idx]);
            
            if (diff > epsilon) {
                mismatch_count++;
                if (diff > max_diff) {
                    max_diff = diff;
                    max_diff_row = i;
                    max_diff_col = j;
                }
            }
        }
    }
    
    if (mismatch_count > 0) {
        std::cout << "  Mismatch count: " << mismatch_count << " / " << (rows * cols) << std::endl;
        std::cout << "  Max difference: " << max_diff << " at (" << max_diff_row << ", " << max_diff_col << ")" << std::endl;
        std::cout << "  Direct value: " << result1[max_diff_col * rows + max_diff_row] << std::endl;
        std::cout << "  Tiled value: " << result2[max_diff_col * rows + max_diff_row] << std::endl;
        return false;
    }
    
    return true;
}

// 打印矩阵（用于调试）
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
    int preview_rows = std::min(8, rows);
    int preview_cols = std::min(8, cols);
    
    for (int i = 0; i < preview_rows; ++i) {
        for (int j = 0; j < preview_cols; ++j) {
            int idx = j * rows + i;  // 列主序
            printf("%8.4f ", matrix[idx]);
        }
        if (cols > preview_cols) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > preview_rows) std::cout << "..." << std::endl;
    std::cout << std::endl;
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <data_folder_path> [-v]" << std::endl;
    std::cout << "  -v: verbose mode (print matrices)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string data_path = argv[1];
    bool verbose = (argc >= 3 && std::string(argv[2]) == "-v");
    
    std::cout << "========================================" << std::endl;
    std::cout << "Tile GEMM Correctness Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 读取数据
    std::cout << "\n[1] Loading data from: " << data_path << std::endl;
    
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
    
    int result_rows = matrix_A.meta.rows;
    int result_cols = matrix_B.meta.cols;
    int result_size = result_rows * result_cols;
    
    // 在verbose模式下打印原始矩阵
    if (verbose) {
        std::cout << "\n[1.1] Original matrices:" << std::endl;
        
        // 转换并打印A矩阵
        int A_size = matrix_A.meta.rows * matrix_A.meta.cols;
        float* h_A = new float[A_size];
        matrixDataToDenseCPU(matrix_A, h_A);
        printMatrix(h_A, matrix_A.meta.rows, matrix_A.meta.cols, "Matrix A");
        delete[] h_A;
        
        // 转换并打印B矩阵
        int B_size = matrix_B.meta.rows * matrix_B.meta.cols;
        float* h_B = new float[B_size];
        matrixDataToDenseCPU(matrix_B, h_B);
        printMatrix(h_B, matrix_B.meta.rows, matrix_B.meta.cols, "Matrix B");
        delete[] h_B;
    }
    
    // 分配结果内存
    float* direct_result = new float[result_size];
    float* tiled_result = new float[result_size];
    
    // 测试1: 直接矩阵乘法
    std::cout << "\n[2] Running direct GEMM (merge all tiles, then multiply)..." << std::endl;
    if (!directGemm(matrix_A, matrix_B, direct_result, result_rows, result_cols)) {
        std::cerr << "Direct GEMM failed!" << std::endl;
        delete[] direct_result;
        delete[] tiled_result;
        return 1;
    }
    std::cout << "  Direct GEMM completed." << std::endl;
    
    if (verbose) {
        printMatrix(direct_result, result_rows, result_cols, "Direct Result (C = A * B)");
    }
    
    // 测试2: 分块矩阵乘法
    std::cout << "\n[3] Running tiled GEMM (our algorithm)..." << std::endl;
    if (!tiledGemm(matrix_A, matrix_B, tiled_result, result_rows, result_cols)) {
        std::cerr << "Tiled GEMM failed!" << std::endl;
        delete[] direct_result;
        delete[] tiled_result;
        return 1;
    }
    std::cout << "  Tiled GEMM completed." << std::endl;
    
    if (verbose) {
        printMatrix(tiled_result, result_rows, result_cols, "Tiled Result (C = A * B)");
    }
    
    // 比较结果
    std::cout << "\n[4] Comparing results (epsilon = " << EPSILON << ")..." << std::endl;
    bool match = compareResults(direct_result, tiled_result, result_rows, result_cols, EPSILON);
    
    std::cout << "\n========================================" << std::endl;
    if (match) {
        std::cout << "TEST PASSED: Results match!" << std::endl;
    } else {
        std::cout << "TEST FAILED: Results do not match!" << std::endl;
    }
    std::cout << "========================================" << std::endl;
    
    // 清理
    delete[] direct_result;
    delete[] tiled_result;
    
    return match ? 0 : 1;
}
