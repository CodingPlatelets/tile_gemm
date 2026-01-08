#include "tile_gemm.h"
#include "sparse_to_dense.h"
#include "cuda_utils.h"
#include <fstream>
#include <iomanip>
#include <cstring>

// 使用带error_msg的版本
#define CUDA_CHECK(call) CUDA_CHECK_MSG(call, error_msg)

TileGEMM::TileGEMM() : initialized(false) {
}

TileGEMM::~TileGEMM() {
}

bool TileGEMM::init() {
    if (!cutlass_gemm.init()) {
        error_msg = "Failed to initialize CUTLASS GEMM: " + cutlass_gemm.getError();
        return false;
    }
    initialized = true;
    return true;
}

bool TileGEMM::gemmTile(const float* d_A, const float* d_B, float* d_C,
                        int M, int N, int K) {
    if (!initialized) {
        error_msg = "TileGEMM not initialized";
        return false;
    }
    
    // 使用 CUTLASS 执行 GEMM
    // C = alpha * A * B + beta * C
    // 矩阵使用列主序存储
    // A: M x K, B: K x N, C: M x N
    
    const float alpha = 1.0f;
    const float beta = 1.0f;  // 累加到C
    
    // 调用 CUTLASS GEMM
    // lda = M (列主序下 A 的 leading dimension)
    // ldb = K (列主序下 B 的 leading dimension)
    // ldc = M (列主序下 C 的 leading dimension)
    if (!cutlass_gemm.gemm(d_A, M,     // A 和其 leading dimension
                           d_B, K,     // B 和其 leading dimension
                           d_C, M,     // C 和其 leading dimension
                           M, N, K,    // 矩阵尺寸
                           alpha, beta)) {
        error_msg = "CUTLASS GEMM failed: " + cutlass_gemm.getError();
        return false;
    }
    
    return true;
}

bool TileGEMM::compute(const MatrixData& matrix_A, const MatrixData& matrix_B,
                       float* h_result, int result_rows, int result_cols) {
    if (!initialized) {
        error_msg = "TileGEMM not initialized";
        return false;
    }
    
    // 分配GPU内存用于结果矩阵
    float* d_result;
    int result_size = result_rows * result_cols;
    CUDA_CHECK(cudaMalloc(&d_result, result_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_result, 0, result_size * sizeof(float)));
    
    // 分块矩阵乘法: C[i][j] = sum_k(A[i][k] * B[k][j])
    // 遍历A的tile行和B的tile列
    for (int i = 0; i < matrix_A.meta.num_tile_rows; ++i) {
        for (int j = 0; j < matrix_B.meta.num_tile_cols; ++j) {
            
            // 获取第一个有效的A[i][*]来确定C tile的行位置
            int C_row_start = -1;
            int C_tile_rows = -1;
            for (int k = 0; k < matrix_A.meta.num_tile_cols; ++k) {
                auto it = matrix_A.tiles.find(std::make_pair(i, k));
                if (it != matrix_A.tiles.end()) {
                    C_row_start = it->second.start_row;
                    C_tile_rows = it->second.height;
                    break;
                }
            }
            
            // 获取第一个有效的B[*][j]来确定C tile的列位置
            int C_col_start = -1;
            int C_tile_cols = -1;
            for (int k = 0; k < matrix_B.meta.num_tile_rows; ++k) {
                auto it = matrix_B.tiles.find(std::make_pair(k, j));
                if (it != matrix_B.tiles.end()) {
                    C_col_start = it->second.start_col;
                    C_tile_cols = it->second.width;
                    break;
                }
            }
            
            // 如果没有有效的tile，跳过
            if (C_row_start < 0 || C_col_start < 0 || C_tile_rows <= 0 || C_tile_cols <= 0) {
                continue;
            }
            
            // 分配临时C块内存并初始化为0
            float* d_C_tile;
            int C_tile_size = C_tile_rows * C_tile_cols;
            CUDA_CHECK(cudaMalloc(&d_C_tile, C_tile_size * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_C_tile, 0, C_tile_size * sizeof(float)));
            
            // 累加所有 A[i][k] * B[k][j]
            for (int k = 0; k < matrix_A.meta.num_tile_cols; ++k) {
                // 获取A[i][k]和B[k][j] tiles
                auto it_A = matrix_A.tiles.find(std::make_pair(i, k));
                auto it_B = matrix_B.tiles.find(std::make_pair(k, j));
                
                // 如果任一tile不存在，跳过（视为零矩阵）
                if (it_A == matrix_A.tiles.end() || it_B == matrix_B.tiles.end()) {
                    continue;
                }
                
                const TileData& tile_A = it_A->second;
                const TileData& tile_B = it_B->second;
                
                // 验证维度兼容性：A的列数必须等于B的行数
                if (tile_A.width != tile_B.height) {
                    cudaFree(d_C_tile);
                    cudaFree(d_result);
                    error_msg = "Tile dimension mismatch: A[" + std::to_string(i) + "][" + std::to_string(k) + 
                                "].width(" + std::to_string(tile_A.width) + ") != B[" + std::to_string(k) + 
                                "][" + std::to_string(j) + "].height(" + std::to_string(tile_B.height) + ")";
                    return false;
                }
                
                // 将tiles转换为稠密格式
                float* d_A_tile;
                float* d_B_tile;
                
                if (!tileToDenseGPU(tile_A, &d_A_tile)) {
                    cudaFree(d_C_tile);
                    cudaFree(d_result);
                    error_msg = "Failed to convert A tile to dense";
                    return false;
                }
                
                if (!tileToDenseGPU(tile_B, &d_B_tile)) {
                    cudaFree(d_A_tile);
                    cudaFree(d_C_tile);
                    cudaFree(d_result);
                    error_msg = "Failed to convert B tile to dense";
                    return false;
                }
                
                // 执行GEMM: C_tile += A_tile * B_tile
                // M = A.height (C的行数)
                // N = B.width (C的列数)
                // K = A.width = B.height (内维)
                if (!gemmTile(d_A_tile, d_B_tile, d_C_tile,
                             tile_A.height, tile_B.width, tile_A.width)) {
                    cudaFree(d_A_tile);
                    cudaFree(d_B_tile);
                    cudaFree(d_C_tile);
                    cudaFree(d_result);
                    return false;
                }
                
                // 释放临时tile内存
                cudaFree(d_A_tile);
                cudaFree(d_B_tile);
            }
            
            // 将C_tile复制到结果矩阵的对应位置
            // 需要逐列复制（因为是列主序）
            for (int col = 0; col < C_tile_cols; ++col) {
                int global_col = C_col_start + col;
                int src_offset = col * C_tile_rows;
                int dst_offset = global_col * result_rows + C_row_start;
                
                CUDA_CHECK(cudaMemcpy(d_result + dst_offset,
                                     d_C_tile + src_offset,
                                     C_tile_rows * sizeof(float),
                                     cudaMemcpyDeviceToDevice));
            }
            
            cudaFree(d_C_tile);
        }
    }
    
    // 将结果从GPU复制到CPU
    CUDA_CHECK(cudaMemcpy(h_result, d_result, result_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    cudaFree(d_result);
    return true;
}

bool saveResultToFile(const float* h_result, int rows, int cols,
                      const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open output file: " << output_path << std::endl;
        return false;
    }
    
    // 写入矩阵尺寸
    file << rows << " " << cols << std::endl;
    
    // 写入矩阵数据（行主序输出，方便阅读）
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 数据是列主序存储的
            int idx = j * rows + i;
            file << h_result[idx];
            if (j < cols - 1) {
                file << " ";
            }
        }
        file << std::endl;
    }
    
    file.close();
    std::cout << "Result saved to: " << output_path << std::endl;
    return true;
}
