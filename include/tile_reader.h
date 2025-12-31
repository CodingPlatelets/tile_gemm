#ifndef TILE_READER_H
#define TILE_READER_H

#include <string>
#include <vector>
#include <map>

// 矩阵元素（包含全局坐标）
struct MatrixElement {
    int global_row;    // 全局行坐标
    int global_col;    // 全局列坐标
    float value;
};

// 单个Tile的数据结构
struct TileData {
    int tile_index;     // tile的顺序索引
    int tile_row;       // tile在分块矩阵中的行索引（自动计算）
    int tile_col;       // tile在分块矩阵中的列索引（自动计算）
    int start_row;      // tile在全局矩阵中的起始行（从元素坐标推断）
    int start_col;      // tile在全局矩阵中的起始列（从元素坐标推断）
    int height;         // tile的高度
    int width;          // tile的宽度
    int nnz;            // 非零元素个数
    std::vector<MatrixElement> elements;  // 所有元素列表（height * width个）
};

// 矩阵元数据
struct MatrixMeta {
    int rows;           // 全局矩阵行数
    int cols;           // 全局矩阵列数
    int num_tile_rows;  // tile行数（动态计算）
    int num_tile_cols;  // tile列数（动态计算）
};

// 完整的矩阵数据（包含所有tiles）
struct MatrixData {
    MatrixMeta meta;
    std::map<std::pair<int, int>, TileData> tiles;  // key: (tile_row, tile_col)
};

// 读取完整的数据集
class TileReader {
public:
    TileReader(const std::string& base_path);
    
    // 读取所有数据
    bool readAll();
    
    // 获取A矩阵数据
    const MatrixData& getMatrixA() const { return matrix_A; }
    
    // 获取B矩阵数据
    const MatrixData& getMatrixB() const { return matrix_B; }
    
    // 获取错误信息
    const std::string& getError() const { return error_msg; }

private:
    // 读取meta.txt文件（只读取全局矩阵尺寸）
    bool readMeta();
    
    // 扫描并读取单个矩阵的所有tiles
    bool readMatrixTiles(const std::string& matrix_path, MatrixData& matrix_data);
    
    // 读取单个tile文件
    bool readTileFile(const std::string& tile_path, int tile_index, TileData& tile);
    
    // 计算tile的逻辑位置（tile_row, tile_col）
    void computeTilePositions(MatrixData& matrix_data);

    std::string base_path;
    std::string folder_name;  // 父文件夹名称
    MatrixData matrix_A;
    MatrixData matrix_B;
    std::string error_msg;
};

#endif // TILE_READER_H
