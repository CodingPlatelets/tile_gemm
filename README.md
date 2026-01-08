# Tile-based Matrix Multiplication with CUDA

基于 tile 分块的矩阵乘法 CUDA 实现，支持从文件夹读取矩阵数据，使用 CUTLASS 开源库执行分块 GEMM。

## 功能特性

- 从文件夹自动扫描并读取 tile 数据
- 自动从元素坐标推断 tile 位置
- 支持不同形状的 tile（只需保证相乘的 tile 维度兼容）
- 支持 A 和 B 矩阵有不同数量的 tile
- 使用 CUTLASS 开源库执行分块 GEMM
- 包含正确性测试（对比直接乘法和分块乘法结果）
- 支持 CUDA 11.0+，兼容 gcc7.5+

## 依赖

- CUDA 11.0 或更高版本
- CUTLASS v2.11.0（通过 Git 子模块管理）
- gcc 7.5 或更高版本

## 获取项目

本项目使用 Git 子模块管理 CUTLASS 依赖。克隆时请使用：

```bash
# 方法 1：克隆时同时获取子模块（推荐）
git clone --recursive <your-repo-url>

# 方法 2：如果已经克隆，执行以下命令获取子模块
git submodule update --init --recursive
```

**注意**: 如果不使用 `--recursive` 选项，`third_party/cutlass` 目录将是空的。

## 编译

### 方式一：使用 Makefile

```bash
# 编译主程序
make

# 编译测试程序
make test

# 编译主程序和测试程序
make all_with_tests
```

可执行文件生成在 `build/` 目录中。

### 方式二：使用 CMake（推荐用于 IDE 代码提示）

```bash
# 使用构建脚本（会自动生成 compile_commands.json）
./build_cmake.sh

# 或手动构建
mkdir -p cmake_build && cd cmake_build
cmake ..
make -j$(nproc)

# 创建 compile_commands.json 符号链接（供 clangd 使用）
ln -sf cmake_build/compile_commands.json compile_commands.json
```

CMake 构建的可执行文件在 `cmake_build/` 目录中。

## 运行

### 主程序

```bash
# Makefile 构建
./build/tile_gemm <数据文件夹路径> [输出文件路径]

# CMake 构建
./cmake_build/tile_gemm <数据文件夹路径> [输出文件路径]
```

例如：
```bash
./build/tile_gemm ./test_data
./build/tile_gemm ./test_data ./output/result.txt
```

或使用 make：
```bash
make run
```

### 正确性测试

```bash
# 运行测试
make run_test

# 运行测试（详细模式，显示原始矩阵和结果矩阵）
make run_test_verbose
```

测试会：
1. 将所有 tile 合并为完整矩阵，使用 CUTLASS 直接计算
2. 使用我们的分块算法计算
3. 比较两种方法的结果是否一致

## 数据格式

### 文件夹结构

```
data_folder/
├── meta.txt              # 元数据文件（只需全局矩阵尺寸）
├── data_folder_A/        # A矩阵的tiles
│   ├── tile_0/
│   │   └── tile_0.txt
│   ├── tile_1/
│   │   └── tile_1.txt
│   └── ...
└── data_folder_B/        # B矩阵的tiles
    ├── tile_0/
    │   └── tile_0.txt
    ├── tile_1/
    │   └── tile_1.txt
    └── ...
```

### meta.txt 格式

只需指定全局矩阵尺寸：

```
A <rows> <cols>
B <rows> <cols>
```

例如：
```
A 8 6
B 6 10
```

### tile_x.txt 格式

```
<tile_height> <tile_width> <nnz>
<global_row> <global_col> <value>
<global_row> <global_col> <value>
...
```

- 第一行：tile 的高度、宽度、非零元素数
- 后续行：每个元素的全局坐标和值（共 height × width 行）
- 程序会自动从全局坐标推断 tile 的起始位置和逻辑位置

例如（一个 4×3 的 tile，起始位置为全局坐标 (0,0)）：
```
4 3 4
0 0 1.0
0 1 0.0
0 2 0.0
1 0 0.0
1 1 1.0
1 2 0.0
2 0 0.0
2 1 0.0
2 2 1.0
3 0 0.0
3 1 0.0
3 2 0.0
```

## 分块矩阵乘法规则

对于 C = A × B：
- A 矩阵被分为 (M × K) 个 tile 块
- B 矩阵被分为 (K × N) 个 tile 块
- 结果 C 的第 (i,j) 块 = Σ_k (A[i][k] × B[k][j])

**重要**：A[i][k] 的宽度必须等于 B[k][j] 的高度（矩阵乘法基本要求）

## 命令一览

### Makefile 命令

| 命令 | 说明 |
|------|------|
| `make` | 编译主程序 |
| `make test` | 编译测试程序 |
| `make all_with_tests` | 编译主程序和测试程序 |
| `make run` | 运行主程序（使用 test_data） |
| `make run_test` | 运行正确性测试 |
| `make run_test_verbose` | 运行测试（详细输出，显示矩阵） |
| `make clean` | 清理所有编译产物 |

### CMake 命令

| 命令 | 说明 |
|------|------|
| `./build_cmake.sh` | 构建并生成 compile_commands.json |
| `cmake --build cmake_build` | 编译 |
| `cmake --build cmake_build --target run` | 运行主程序 |
| `cmake --build cmake_build --target run_test` | 运行测试 |
| `cmake --build cmake_build --target run_test_verbose` | 运行测试（详细） |

## 项目结构

```
tile_gemm/
├── include/
│   ├── tile_reader.h        # 文件读取接口
│   ├── sparse_to_dense.h    # 稀疏转稠密工具
│   ├── tile_gemm.h          # GEMM计算接口
│   ├── cutlass_gemm.h       # CUTLASS GEMM包装器
│   └── cuda_utils.h         # CUDA工具宏
├── src/
│   ├── tile_reader.cpp      # 文件读取实现
│   ├── sparse_to_dense.cu   # 稀疏转稠密CUDA实现
│   ├── tile_gemm.cu         # Tile GEMM实现
│   ├── cutlass_gemm.cu      # CUTLASS GEMM封装
│   └── main.cu              # 主程序入口
├── tests/
│   └── test_correctness.cu  # 正确性测试
├── test_data/               # 测试数据
├── third_party/
│   └── cutlass/             # CUTLASS 库
├── build/                   # Makefile 编译输出目录
├── cmake_build/             # CMake 编译输出目录
├── Makefile
├── CMakeLists.txt
├── build_cmake.sh           # CMake 构建脚本
└── README.md
```

## 清理

```bash
# 清理 Makefile 构建
make clean

# 清理 CMake 构建
rm -rf cmake_build compile_commands.json
```
