#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace cosmic {
namespace genmatrix {

void complete_missing_parents_add_phantoms(std::vector<int>& dam, std::vector<int>& sire);
void ml(const std::vector<int>& dam, const std::vector<int>& sire, std::vector<double>& f, std::vector<double>& dii, int g = 0, int fmiss = 0);
Eigen::SparseMatrix<double> buildA(const std::vector<int>& dam, const std::vector<int>& sire);
Eigen::MatrixXd buildA_dense(const std::vector<int>& dam, const std::vector<int>& sire);
Eigen::SparseMatrix<double> invertA(const Eigen::SparseMatrix<double>& A);
Eigen::SparseMatrix<double> invertA_henderson(const std::vector<int>& dam, const std::vector<int>& sire);
Eigen::SparseMatrix<double> invertA_henderson_fixed(const std::vector<int>& dam, const std::vector<int>& sire);
Eigen::SparseMatrix<double> invert_sparse_matrix(const Eigen::SparseMatrix<double>& M);

} // namespace genmatrix
} // namespace cosmic
