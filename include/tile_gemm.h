#ifndef TILE_GEMM_H
#define TILE_GEMM_H

#include "tile_reader.h"
#include "cutlass_gemm.h"

// GEMM计算器类
class TileGEMM {
public:
    TileGEMM();
    ~TileGEMM();
    
    // 初始化 GEMM 引擎
    bool init();
    
    // 执行完整的分块矩阵乘法 C = A * B
    // matrix_A: A矩阵数据
    // matrix_B: B矩阵数据
    // h_result: 输出结果矩阵（CPU内存，由调用者分配）
    // result_rows: 结果矩阵行数
    // result_cols: 结果矩阵列数
    bool compute(const MatrixData& matrix_A, const MatrixData& matrix_B,
                 float* h_result, int result_rows, int result_cols);
    
    // 执行单个tile的GEMM: C += A * B
    // d_A: A矩阵tile（GPU内存）
    // d_B: B矩阵tile（GPU内存）
    // d_C: C矩阵tile（GPU内存，累加结果）
    // M: A的行数，C的行数
    // N: B的列数，C的列数
    // K: A的列数，B的行数
    bool gemmTile(const float* d_A, const float* d_B, float* d_C,
                  int M, int N, int K);
    
    // 获取错误信息
    const std::string& getError() const { return error_msg; }

private:
    CutlassGemm cutlass_gemm;
    bool initialized;
    std::string error_msg;
};

// 将结果矩阵保存到文件
bool saveResultToFile(const float* h_result, int rows, int cols, 
                      const std::string& output_path);

#endif // TILE_GEMM_H

