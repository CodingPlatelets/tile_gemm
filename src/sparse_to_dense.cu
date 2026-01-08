#include "sparse_to_dense.h"
#include "cuda_utils.h"
#include <cstring>

// 使用stderr版本的宏
#define CUDA_CHECK(call) CUDA_CHECK_STDERR(call)

void tileToDenseCPU(const TileData& tile, float* h_dense) {
    int total_size = tile.height * tile.width;
    
    // 初始化为零（以防元素不完整）
    memset(h_dense, 0, total_size * sizeof(float));
    
    // 填充元素
    // 元素的坐标是全局坐标，需要转换为tile内的局部坐标
    for (const auto& elem : tile.elements) {
        // 计算局部坐标
        int local_row = elem.global_row - tile.start_row;
        int local_col = elem.global_col - tile.start_col;
        
        // 验证坐标有效性
        if (local_row >= 0 && local_row < tile.height &&
            local_col >= 0 && local_col < tile.width) {
            // 列主序存储
            int idx = local_col * tile.height + local_row;
            h_dense[idx] = elem.value;
        }
    }
}

bool tileToDenseGPU(const TileData& tile, float** d_dense) {
    int total_size = tile.height * tile.width;
    
    // 在CPU上转换
    float* h_dense = new float[total_size];
    tileToDenseCPU(tile, h_dense);
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(d_dense, total_size * sizeof(float)));
    
    // 复制到GPU
    CUDA_CHECK(cudaMemcpy(*d_dense, h_dense, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    delete[] h_dense;
    return true;
}

bool matrixTilesToDense(const MatrixData& matrix_data, float** d_matrix) {
    int total_rows = matrix_data.meta.rows;
    int total_cols = matrix_data.meta.cols;
    int total_size = total_rows * total_cols;
    
    // 在CPU上分配并初始化完整矩阵
    float* h_matrix = new float[total_size];
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
    
    // 分配GPU内存
    cudaError_t err = cudaMalloc(d_matrix, total_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        delete[] h_matrix;
        return false;
    }
    
    // 复制到GPU
    err = cudaMemcpy(*d_matrix, h_matrix, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(*d_matrix);
        delete[] h_matrix;
        return false;
    }
    
    delete[] h_matrix;
    return true;
}
