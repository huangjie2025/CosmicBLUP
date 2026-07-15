#pragma once
#include <vector>
#include <cstddef>
#include <Eigen/Dense>

namespace cosmic {
namespace grm {

void invertSPD(std::vector<double> &A, std::size_t n, double ridge, std::vector<double> &Ainv);

// Compute explicit APY approximation of G inverse
Eigen::MatrixXd compute_explicit_apy_inverse(const Eigen::MatrixXd& G, const std::vector<int>& core_status, double ridge);

} // namespace grm
} // namespace cosmic
