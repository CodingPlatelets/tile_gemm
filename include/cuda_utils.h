#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// CUDA错误检查宏 - 打印到stderr版本
#define CUDA_CHECK_STDERR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA错误检查宏 - 存储到error_msg变量版本
#define CUDA_CHECK_MSG(call, error_msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            error_msg = std::string("CUDA Error: ") + cudaGetErrorString(err); \
            return false; \
        } \
    } while(0)

#endif // CUDA_UTILS_H

