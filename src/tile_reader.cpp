#include "tile_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <set>
#include <climits>

TileReader::TileReader(const std::string& base_path) : base_path(base_path) {
    // 提取文件夹名称
    size_t pos = base_path.rfind('/');
    if (pos != std::string::npos && pos < base_path.length() - 1) {
        folder_name = base_path.substr(pos + 1);
    } else if (pos == std::string::npos) {
        folder_name = base_path;
    } else {
        // 路径以/结尾，去掉最后的/再找
        std::string temp = base_path.substr(0, pos);
        size_t pos2 = temp.rfind('/');
        if (pos2 != std::string::npos) {
            folder_name = temp.substr(pos2 + 1);
        } else {
            folder_name = temp;
        }
    }
}

bool TileReader::readAll() {
    // 1. 读取meta文件（只读取全局矩阵尺寸）
    if (!readMeta()) {
        return false;
    }
    
    // 2. 扫描并读取A矩阵的tiles
    std::string path_A = base_path + "/" + folder_name + "_A";
    if (!readMatrixTiles(path_A, matrix_A)) {
        return false;
    }
    
    // 3. 扫描并读取B矩阵的tiles
    std::string path_B = base_path + "/" + folder_name + "_B";
    if (!readMatrixTiles(path_B, matrix_B)) {
        return false;
    }
    
    // 4. 计算tile的逻辑位置
    computeTilePositions(matrix_A);
    computeTilePositions(matrix_B);
    
    return true;
}

bool TileReader::readMeta() {
    std::string meta_path = base_path + "/meta.txt";
    std::ifstream file(meta_path);
    
    if (!file.is_open()) {
        error_msg = "Cannot open meta file: " + meta_path;
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 跳过空行和注释
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        
        if (key == "A") {
            // 全局矩阵尺寸: A rows cols
            iss >> matrix_A.meta.rows >> matrix_A.meta.cols;
        } else if (key == "B") {
            // 全局矩阵尺寸: B rows cols
            iss >> matrix_B.meta.rows >> matrix_B.meta.cols;
        }
    }
    
    file.close();
    
    // 验证数据
    if (matrix_A.meta.rows <= 0 || matrix_A.meta.cols <= 0 ||
        matrix_B.meta.rows <= 0 || matrix_B.meta.cols <= 0) {
        error_msg = "Invalid matrix dimensions in meta file";
        return false;
    }
    
    if (matrix_A.meta.cols != matrix_B.meta.rows) {
        error_msg = "Matrix dimensions incompatible for multiplication: A.cols != B.rows";
        return false;
    }
    
    return true;
}

bool TileReader::readMatrixTiles(const std::string& matrix_path, MatrixData& matrix_data) {
    DIR* dir = opendir(matrix_path.c_str());
    if (!dir) {
        error_msg = "Cannot open directory: " + matrix_path;
        return false;
    }
    
    // 收集所有tile文件夹
    std::vector<std::pair<int, std::string>> tile_folders;  // (index, folder_name)
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // 跳过 . 和 ..
        if (name == "." || name == "..") {
            continue;
        }
        
        // 检查是否是tile文件夹 (tile_0, tile_1, ...)
        if (name.find("tile_") != 0) {
            continue;
        }
        
        // 解析tile索引
        try {
            int index = std::stoi(name.substr(5));
            tile_folders.push_back(std::make_pair(index, name));
        } catch (...) {
            continue;
        }
    }
    closedir(dir);
    
    // 按索引排序
    std::sort(tile_folders.begin(), tile_folders.end());
    
    // 临时存储tiles（用start_row, start_col作为key）
    std::vector<TileData> temp_tiles;
    
    // 读取每个tile文件
    for (const auto& tf : tile_folders) {
        int tile_index = tf.first;
        const std::string& folder_name = tf.second;
        
        std::string tile_file_path = matrix_path + "/" + folder_name + "/" + folder_name + ".txt";
        
        TileData tile;
        if (!readTileFile(tile_file_path, tile_index, tile)) {
            return false;
        }
        
        temp_tiles.push_back(tile);
    }
    
    // 临时存储到matrix_data（使用dummy key，后续computeTilePositions会重新组织）
    for (size_t i = 0; i < temp_tiles.size(); ++i) {
        // 暂时使用(start_row, start_col)作为key，后面会重新计算
        matrix_data.tiles[std::make_pair(temp_tiles[i].start_row, temp_tiles[i].start_col)] = temp_tiles[i];
    }
    
    return true;
}

bool TileReader::readTileFile(const std::string& tile_path, int tile_index, TileData& tile) {
    std::ifstream file(tile_path);
    
    if (!file.is_open()) {
        error_msg = "Cannot open tile file: " + tile_path;
        return false;
    }
    
    tile.tile_index = tile_index;
    
    // 读取第一行：height width nnz
    file >> tile.height >> tile.width >> tile.nnz;
    
    // 读取所有元素（height * width 个）
    int total_elements = tile.height * tile.width;
    tile.elements.reserve(total_elements);
    
    int min_row = INT_MAX, min_col = INT_MAX;
    
    for (int i = 0; i < total_elements; ++i) {
        MatrixElement elem;
        if (!(file >> elem.global_row >> elem.global_col >> elem.value)) {
            error_msg = "Failed to read element " + std::to_string(i) + " from file: " + tile_path;
            return false;
        }
        tile.elements.push_back(elem);
        
        // 推断start_row和start_col
        min_row = std::min(min_row, elem.global_row);
        min_col = std::min(min_col, elem.global_col);
    }
    
    tile.start_row = min_row;
    tile.start_col = min_col;
    
    file.close();
    return true;
}

void TileReader::computeTilePositions(MatrixData& matrix_data) {
    // 收集所有唯一的start_row和start_col值
    std::set<int> unique_rows, unique_cols;
    
    for (const auto& pair : matrix_data.tiles) {
        unique_rows.insert(pair.second.start_row);
        unique_cols.insert(pair.second.start_col);
    }
    
    // 创建排序后的行列映射
    std::vector<int> sorted_rows(unique_rows.begin(), unique_rows.end());
    std::vector<int> sorted_cols(unique_cols.begin(), unique_cols.end());
    std::sort(sorted_rows.begin(), sorted_rows.end());
    std::sort(sorted_cols.begin(), sorted_cols.end());
    
    // 创建start_row -> tile_row 和 start_col -> tile_col 的映射
    std::map<int, int> row_to_tile_row, col_to_tile_col;
    for (size_t i = 0; i < sorted_rows.size(); ++i) {
        row_to_tile_row[sorted_rows[i]] = i;
    }
    for (size_t i = 0; i < sorted_cols.size(); ++i) {
        col_to_tile_col[sorted_cols[i]] = i;
    }
    
    // 更新tile的逻辑位置并重新组织map
    std::map<std::pair<int, int>, TileData> new_tiles;
    
    for (auto& pair : matrix_data.tiles) {
        TileData& tile = pair.second;
        tile.tile_row = row_to_tile_row[tile.start_row];
        tile.tile_col = col_to_tile_col[tile.start_col];
        
        new_tiles[std::make_pair(tile.tile_row, tile.tile_col)] = tile;
    }
    
    matrix_data.tiles = new_tiles;
    
    // 更新tile数量
    matrix_data.meta.num_tile_rows = sorted_rows.size();
    matrix_data.meta.num_tile_cols = sorted_cols.size();
}
