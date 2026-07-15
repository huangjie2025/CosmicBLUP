#pragma once
/// @file matrix_io.h
/// Matrix I/O utilities used by CosmicBLUP.
/// Provides reading/writing inverse matrices (A/G/H), ID lists, variance files,
/// and matrix inversion with eigen-bending.

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "matrix_adapter.h"

namespace cosmic {

/// Read a sparse inverse matrix from text or binary file.
/// Supports dense format, triplet format (row col val), legacy lower-triangle
/// float binary, and COSMIC_UPPER_PACKED binary with metadata header.
/// Auto-detects 0/1-based indexing and symmetrizes if needed.
/// @param filename  Path to matrix file (.txt, .sparse, or .bin)
/// @param abs_zero_threshold  Threshold below which values are treated as zero
/// @param apply_sign_fix  Enable inverse-style sign correction heuristics
Eigen::SparseMatrix<double> readInvMatrix(const std::string& filename,
                                          double abs_zero_threshold = 1e-9,
                                          bool apply_sign_fix = true);

/// Read an inverse matrix as AbstractMatrix (sparse or mmap-backed).
/// @param filename  Path to matrix file
/// @param use_mmap  If true, use memory-mapped access
AbstractMatrix* readInvMatrixAbstract(const std::string& filename, bool use_mmap);

/// Read an ID list file (one ID per line). Returns map from ID to 1-based index.
std::map<std::string, int> readIdList(const std::string& filename);

/// Read variance components from file. Returns (sigma2_u, sigma2_e).
std::pair<double, double> readVars(const std::string& filename);

/// Write a dense matrix in triplet text format (lower triangle).
void write_matrix_txt(const std::string& path,
                      const Eigen::MatrixXd& M,
                      const std::vector<std::string>& ids);

/// Write a dense matrix in binary format using COSMIC_UPPER_PACKED metadata
/// header + ID file. Reader still accepts the legacy lower-triangle float
/// format for compatibility.
void write_matrix_bin(const std::string& bin_path,
                      const std::string& id_path,
                      const Eigen::MatrixXd& M,
                      const std::vector<std::string>& ids);

/// Invert a positive semi-definite matrix with eigen-bending.
/// Adds ridge to diagonal if ridge > 0, then clamps small eigenvalues.
Eigen::MatrixXd invert_psd_matrix(Eigen::MatrixXd M, double ridge = 0.0);

/// Factorize a matrix with progressive jitter for numerical stability.
/// Tries jitter values: 0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2.
/// Returns true if a valid (non-degenerate) factorization was found.
bool factorize_with_jitter(const Eigen::MatrixXd& matrix,
                           Eigen::LDLT<Eigen::MatrixXd>& solver);

/// Fix suspicious sign patterns in inverse matrices (HIBLUP compatibility).
void fixMatrixSigns(Eigen::SparseMatrix<double>& A, const std::string& filename);

} // namespace cosmic
