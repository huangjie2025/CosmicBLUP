#pragma once
/// @file string_utils.h
/// String and file utilities used by CosmicBLUP.

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <functional>

namespace cosmic {

/// Trim whitespace from both ends of a string, returning a copy.
inline std::string trim_copy(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

/// Detect delimiter (comma or whitespace) from the first line of a file.
inline char detect_delim(const std::string& first_line) {
    return (first_line.find(',') != std::string::npos) ? ',' : ' ';
}

/// Check if a string represents an integer.
inline bool is_integer_string(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!isdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

/// Check if a token represents a missing value (NA, nan, ., null, empty).
inline bool is_missing_token(const std::string& s) {
    std::string t = trim_copy(s);
    if (t.empty()) return true;
    std::string lower = t;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower == "na" || lower == "nan" || lower == "." || lower == "null");
}

/// Generate a label from an inverse matrix filename by stripping extensions and "inv" suffix.
inline std::string make_inv_label(const std::string& inv_file) {
    std::string name = std::filesystem::path(inv_file).filename().string();
    auto ends_with_ci = [](const std::string& str, const std::string& suf) -> bool {
        if (str.size() < suf.size()) return false;
        std::string tail = str.substr(str.size() - suf.size());
        std::string a = tail, b = suf;
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
        return a == b;
    };
    auto strip_ci = [&](const std::string& suf) {
        if (ends_with_ci(name, suf)) name.erase(name.size() - suf.size());
    };
    strip_ci(".bin");
    strip_ci(".txt");
    strip_ci(".sparse");
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.size() > 3 && lower.rfind("inv") == lower.size() - 3) {
        if (lower != "ainv" && lower != "ginv" && lower != "hinv") name.erase(name.size() - 3);
    }
    if (!name.empty() && name.back() == '.') name.pop_back();
    return name;
}

/// Auto-detect model type from inverse matrix filename.
/// Returns "ssgblup", "gblup", or "pblup" based on filename content.
inline std::string auto_model_from_inv(const std::string& inv_path, const std::string& explicit_model) {
    if (!explicit_model.empty()) return explicit_model;
    std::string lower = inv_path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.find("hinv") != std::string::npos) return "ssgblup";
    if (lower.find("ginv") != std::string::npos) return "gblup";
    if (lower.find("ainv") != std::string::npos) return "pblup";
    return "pblup";
}

/// Detect inverse matrix and ID files from a path (file or directory).
/// Returns (matrix_path, id_path).
inline std::pair<std::string, std::string> detect_inv_files(const std::string& path) {
    namespace fs = std::filesystem;
    if (fs::is_regular_file(path)) {
        std::string dir = fs::path(path).parent_path().string();
        if (dir.empty()) dir = ".";
        std::string idcand;
        for (auto& p : fs::directory_iterator(dir)) {
            std::string name = p.path().filename().string();
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("id") != std::string::npos && p.is_regular_file()) { idcand = p.path().string(); break; }
        }
        return {path, idcand};
    }
    if (fs::is_directory(path)) {
        std::vector<std::string> matrices, ids;
        for (auto& p : fs::directory_iterator(path)) {
            if (!p.is_regular_file()) continue;
            std::string name = p.path().filename().string();
            std::string lower = name;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower.find("inv") != std::string::npos) matrices.push_back(p.path().string());
            if (lower.find("id") != std::string::npos) ids.push_back(p.path().string());
        }
        auto read_dim_quick = [](const std::string& f) -> std::pair<int,int> {
            std::ifstream file(f); std::string line; int rows=0, cols=0;
            while (getline(file, line)) {
                std::string s = trim_copy(line);
                if (s.empty()) continue;
                std::vector<std::string> tokens;
                if (s.find(',') != std::string::npos) {
                    std::string tok; std::stringstream ss(s);
                    while (getline(ss, tok, ',')) tokens.push_back(trim_copy(tok));
                } else {
                    std::string tok; std::stringstream ss(s);
                    while (ss >> tok) tokens.push_back(tok);
                }
                if (cols == 0) cols = (int)tokens.size();
                rows++;
            }
            return {rows, cols};
        };
        auto count_lines = [](const std::string& f) -> int {
            std::ifstream file(f); std::string line; int n=0;
            while (getline(file, line)) { std::string s = trim_copy(line); if (!s.empty()) n++; }
            return n;
        };
        std::string matrix, idfile;
        for (const auto& m : matrices) {
            auto dim = read_dim_quick(m); int n = dim.first;
            for (const auto& idc : ids) {
                int lines = count_lines(idc);
                if (lines == n) { matrix = m; idfile = idc; break; }
            }
            if (!matrix.empty()) break;
        }
        return {matrix, idfile};
    }
    return {std::string(), std::string()};
}

} // namespace cosmic
