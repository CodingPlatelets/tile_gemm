#ifndef CUTLASS_GEMM_H
#define CUTLASS_GEMM_H

#include <cuda_runtime.h>
#include <string>

// CUTLASS GEMM 包装器类
// 提供与 cuBLAS 类似的接口，用于执行单精度浮点 GEMM 操作
class CutlassGemm {
public:
    CutlassGemm();
    ~CutlassGemm();
    
    // 初始化（兼容性接口，CUTLASS 不需要显式初始化）
    bool init();
    
    // 执行 GEMM 操作: C = alpha * A * B + beta * C
    // 所有矩阵均为列主序（column-major）存储
    // A: M x K 矩阵（设备内存）
    // B: K x N 矩阵（设备内存）
    // C: M x N 矩阵（设备内存）
    // M: A 的行数，C 的行数
    // N: B 的列数，C 的列数
    // K: A 的列数，B 的行数
    // alpha, beta: 标量系数
    bool gemm(const float* d_A, int lda,
              const float* d_B, int ldb,
              float* d_C, int ldc,
              int M, int N, int K,
              float alpha, float beta);
    
    // 获取错误信息
    const std::string& getError() const { return error_msg; }

private:
    bool initialized;
    std::string error_msg;
};

#endif // CUTLASS_GEMM_H
