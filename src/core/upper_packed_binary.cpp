#include "upper_packed_binary.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace cosmic {
namespace io {
namespace {

constexpr std::int32_t kVersion = 1;
constexpr std::int32_t kIndexBase = 1;
constexpr std::int32_t kStorageCodeUpperPacked = 1;
constexpr std::int32_t kDtypeCodeFloat64 = 1;
constexpr std::int32_t kEndianCodeLittle = 1;

std::array<char, 32> expected_magic() {
  std::array<char, 32> magic{};
  const char* magic_text = "COSMIC_UPPER_PACKED";
  std::memcpy(magic.data(), magic_text, std::strlen(magic_text));
  return magic;
}

void write_header(std::ofstream& out, std::int64_t n) {
  const auto magic = expected_magic();
  out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(&kIndexBase), sizeof(kIndexBase));
  out.write(reinterpret_cast<const char*>(&kStorageCodeUpperPacked), sizeof(kStorageCodeUpperPacked));
  out.write(reinterpret_cast<const char*>(&kDtypeCodeFloat64), sizeof(kDtypeCodeFloat64));
  out.write(reinterpret_cast<const char*>(&kEndianCodeLittle), sizeof(kEndianCodeLittle));
}

void check_square(int rows, int cols, const std::string& path) {
  if (rows != cols) {
    throw std::runtime_error("upper-packed binary output requires a square matrix: " + path);
  }
}

template <typename T>
void read_exact(std::ifstream& in, T& value, const std::string& path) {
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("truncated upper-packed binary matrix file: " + path);
  }
}

}  // namespace

void write_upper_packed_binary(const std::string& path,
                               const Eigen::MatrixXd& matrix) {
  check_square(matrix.rows(), matrix.cols(), path);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write upper-packed binary matrix file: " + path);
  }
  const std::int64_t n = static_cast<std::int64_t>(matrix.rows());
  write_header(out, n);
  for (int j = 0; j < matrix.cols(); ++j) {
    for (int i = 0; i <= j; ++i) {
      const double value = matrix(i, j);
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
}

void write_upper_packed_binary(const std::string& path,
                               const Eigen::SparseMatrix<double>& matrix) {
  check_square(matrix.rows(), matrix.cols(), path);
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write upper-packed binary matrix file: " + path);
  }
  const std::int64_t n = static_cast<std::int64_t>(matrix.rows());
  write_header(out, n);
  for (int j = 0; j < matrix.cols(); ++j) {
    for (int i = 0; i <= j; ++i) {
      const double value = matrix.coeff(i, j);
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
}

Eigen::MatrixXd read_upper_packed_binary(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read upper-packed binary matrix file: " + path);
  }

  std::array<char, 32> magic{};
  in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!in || magic != expected_magic()) {
    throw std::runtime_error("unsupported binary matrix header: " + path);
  }

  std::int32_t version = 0;
  std::int64_t n = 0;
  std::int32_t index_base = 0;
  std::int32_t storage_code = 0;
  std::int32_t dtype_code = 0;
  std::int32_t endian_code = 0;
  read_exact(in, version, path);
  read_exact(in, n, path);
  read_exact(in, index_base, path);
  read_exact(in, storage_code, path);
  read_exact(in, dtype_code, path);
  read_exact(in, endian_code, path);

  if (version != kVersion ||
      index_base != kIndexBase ||
      storage_code != kStorageCodeUpperPacked ||
      dtype_code != kDtypeCodeFloat64 ||
      endian_code != kEndianCodeLittle) {
    throw std::runtime_error("unsupported upper-packed binary matrix metadata: " + path);
  }
  if (n <= 0 || n > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("invalid upper-packed binary matrix dimension: " + path);
  }

  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(static_cast<int>(n), static_cast<int>(n));
  for (int j = 0; j < matrix.cols(); ++j) {
    for (int i = 0; i <= j; ++i) {
      double value = 0.0;
      read_exact(in, value, path);
      matrix(i, j) = value;
      matrix(j, i) = value;
    }
  }
  char trailing = '\0';
  if (in.read(&trailing, 1)) {
    throw std::runtime_error("upper-packed binary matrix file has trailing bytes: " + path);
  }
  return matrix;
}

}  // namespace io
}  // namespace cosmic
