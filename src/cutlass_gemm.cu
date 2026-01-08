#include "cutlass_gemm.h"
#include <iostream>

// CUTLASS 头文件
#include "cutlass/gemm/device/gemm.h"

// 定义 CUTLASS GEMM 类型
// 使用列主序（ColumnMajor）布局以匹配原有 cuBLAS 数据格式
using ColumnMajor = cutlass::layout::ColumnMajor;

// CUTLASS GEMM 配置
// Element type: float
// Layout: ColumnMajor for all matrices (A, B, C)
// Compute type: float
using CutlassGemmOp = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    ColumnMajor,                     // LayoutA
    float,                           // ElementB
    ColumnMajor,                     // LayoutB
    float,                           // ElementC
    ColumnMajor,                     // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassSimt,      // Operator class (SIMT for compatibility)
    cutlass::arch::Sm70              // Architecture (compute capability 7.0 for V100)
>;

CutlassGemm::CutlassGemm() : initialized(false) {
}

CutlassGemm::~CutlassGemm() {
}

bool CutlassGemm::init() {
    // CUTLASS 不需要显式初始化，这是为了兼容原有接口
    initialized = true;
    return true;
}

bool CutlassGemm::gemm(const float* d_A, int lda,
                       const float* d_B, int ldb,
                       float* d_C, int ldc,
                       int M, int N, int K,
                       float alpha, float beta) {
    if (!initialized) {
        error_msg = "CutlassGemm not initialized";
        return false;
    }
    
    // 设置 GEMM 参数
    CutlassGemmOp::Arguments args(
        {M, N, K},                    // 问题尺寸
        {d_A, lda},                   // TensorRef A
        {d_B, ldb},                   // TensorRef B
        {d_C, ldc},                   // TensorRef C
        {d_C, ldc},                   // TensorRef D (输出)
        {alpha, beta}                 // 标量参数
    );
    
    // 实例化 CUTLASS GEMM
    CutlassGemmOp gemm_op;
    
    // 检查问题尺寸是否支持
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        error_msg = "CUTLASS GEMM cannot implement the given problem size";
        return false;
    }
    
    // 初始化 GEMM 操作
    status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        error_msg = "CUTLASS GEMM initialization failed";
        return false;
    }
    
    // 执行 GEMM
    status = gemm_op();
    if (status != cutlass::Status::kSuccess) {
        error_msg = "CUTLASS GEMM execution failed";
        return false;
    }
    
    // 等待 CUDA kernel 完成
    cudaError_t cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        error_msg = std::string("CUDA synchronization failed: ") + 
                    cudaGetErrorString(cuda_status);
        return false;
    }
    
    return true;
}
