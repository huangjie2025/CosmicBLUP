#pragma once

// Upper-packed binary matrix format (COSMIC_UPPER_PACKED).
//
// Header layout (little-endian):
//   magic[32]      = "COSMIC_UPPER_PACKED" + NUL padding
//   version   i32  = 1
//   n         i64  = matrix dimension
//   index_base i32 = 1
//   storage_code i32 = 1 (upper-packed)
//   dtype_code i32   = 1 (float64)
//   endian_code i32  = 1 (little-endian)
// Payload: n*(n+1)/2 doubles, upper-column-major:
//   A(0,0),
//   A(0,1), A(1,1),
//   A(0,2), A(1,2), A(2,2), ...
//
// Reader expands symmetrically. Trailing bytes are rejected.

#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace cosmic {
namespace io {

void write_upper_packed_binary(const std::string& path,
                               const Eigen::MatrixXd& matrix);

void write_upper_packed_binary(const std::string& path,
                               const Eigen::SparseMatrix<double>& matrix);

Eigen::MatrixXd read_upper_packed_binary(const std::string& path);

}  // namespace io
}  // namespace cosmic
