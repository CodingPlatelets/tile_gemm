#ifndef SPARSE_TO_DENSE_H
#define SPARSE_TO_DENSE_H

#include "tile_reader.h"
#include <cuda_runtime.h>

// 将单个tile转换为稠密矩阵（在GPU上）
// tile: 输入的tile数据（包含所有元素）
// d_dense: 输出的GPU稠密矩阵指针
// 注意：调用者负责释放GPU内存
bool tileToDenseGPU(const TileData& tile, float** d_dense);

// 将单个tile转换为稠密矩阵（在CPU上）
// tile: 输入的tile数据（包含所有元素）
// h_dense: 输出的CPU稠密矩阵指针（已分配内存，大小为height * width）
void tileToDenseCPU(const TileData& tile, float* h_dense);

// 将整个矩阵的所有tiles合并为一个完整的稠密矩阵
// matrix_data: 输入的矩阵数据（包含所有tiles）
// d_matrix: 输出的GPU稠密矩阵指针
// 注意：调用者负责释放GPU内存
bool matrixTilesToDense(const MatrixData& matrix_data, float** d_matrix);

#endif // SPARSE_TO_DENSE_H
