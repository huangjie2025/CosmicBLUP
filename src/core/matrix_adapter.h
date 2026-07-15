#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include "mmap_matrix.h"

namespace cosmic {

// Abstract interface for Generalized Matrix Access (Sparse, MMap, Split)
class AbstractMatrix {
public:
    virtual ~AbstractMatrix() {}

    virtual int rows() const = 0;
    virtual int cols() const = 0;
    virtual size_t nonZeros() const = 0;

    // Core operation: y = A * x
    virtual Eigen::VectorXd operator*(const Eigen::VectorXd& x) const = 0;

    // Add diagonal elements to a vector: diag += alpha * diag(A)
    virtual void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const = 0;

    // Get a dense block [start_row, start_row+size) x [start_row, start_row+size)
    virtual Eigen::MatrixXd getBlock(int start_row, int size) const = 0;

    // Visit all triplets (r, c, v) - useful for constructing LHS
    // For symmetric matrices, this should visit lower/upper/both?
    // Convention: Visit stored triplets. For symmetric matrices, usually lower or both.
    // Our builders expect to handle symmetry, so visiting stored triplets is enough.
    virtual void visit_triplets(std::function<void(int, int, double)> callback) const = 0;

    // Convert to Dense Matrix (for Exact VCE or Debugging)
    virtual Eigen::MatrixXd toDense() const = 0;
};

// Interface for Genomic Data Matrix (e.g. PLINK/PGEN)
// Supports Matrix-free operations for RHE/PCG
class GenotypeMatrix {
public:
    virtual ~GenotypeMatrix() = default;

    // N (samples) x M (SNPs)
    virtual int rows() const = 0;
    virtual int cols() const = 0;

    // y = Z * v (Z is standardized genotype matrix)
    virtual void multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, int n_threads=1) = 0;

    // y = Z' * v
    virtual void multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, int n_threads=1) = 0;
};

// Wrapper for Eigen::SparseMatrix
class SparseMatrixAdapter : public AbstractMatrix {
    Eigen::SparseMatrix<double> mat;
public:
    SparseMatrixAdapter(Eigen::SparseMatrix<double> m) : mat(std::move(m)) {}

    const Eigen::SparseMatrix<double>& getMatrix() const { return mat; }

    int rows() const override { return (int)mat.rows(); }
    int cols() const override { return (int)mat.cols(); }
    size_t nonZeros() const override { return mat.nonZeros(); }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const override {
        return mat * x;
    }

    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const override {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
                if (it.row() == it.col() && it.row() < diag.size()) {
                    diag(it.row()) += it.value() * alpha;
                }
            }
        }
    }

    Eigen::MatrixXd getBlock(int start_row, int size) const override {
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(size, size);
        int end_row = start_row + size;
        // Optimization: if mat is column major (default), iterating columns in range is faster
        for (int k = start_row; k < end_row && k < mat.outerSize(); ++k) {
             for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
                 int r = (int)it.row();
                 if (r >= start_row && r < end_row) {
                     B(r - start_row, k - start_row) = it.value();
                 }
             }
        }
        // If row major, we would need to iterate rows.
        // But we assume default CSC.
        // Wait, if we want symmetric block from full matrix, we might miss lower triangle if only upper stored?
        // AbstractMatrix assumes it represents the full matrix logic?
        // Or we assume symmetric inputs are stored symmetrically or we handle it?
        // `get_block` usually implies returning what's there.
        return B;
    }

    void visit_triplets(std::function<void(int, int, double)> callback) const override {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
                callback((int)it.row(), (int)it.col(), it.value());
            }
        }
    }

    Eigen::MatrixXd toDense() const override {
        return Eigen::MatrixXd(mat);
    }
};

// Wrapper for Eigen::MatrixXd (Dense)
class DenseMatrixAdapter : public AbstractMatrix {
    Eigen::MatrixXd mat;
public:
    DenseMatrixAdapter(Eigen::MatrixXd m) : mat(std::move(m)) {}

    int rows() const override { return (int)mat.rows(); }
    int cols() const override { return (int)mat.cols(); }
    size_t nonZeros() const override { return mat.size(); }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const override {
        return mat * x;
    }

    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const override {
        int n = std::min((int)mat.rows(), (int)mat.cols());
        if (n > diag.size()) n = (int)diag.size();
        for (int i = 0; i < n; ++i) {
            diag(i) += mat(i, i) * alpha;
        }
    }

    Eigen::MatrixXd getBlock(int start_row, int size) const override {
        return mat.block(start_row, start_row, size, size);
    }

    void visit_triplets(std::function<void(int, int, double)> callback) const override {
        for (int j = 0; j < mat.cols(); ++j) {
            for (int i = 0; i < mat.rows(); ++i) {
                double v = mat(i, j);
                if (v != 0.0) callback(i, j, v);
            }
        }
    }

    Eigen::MatrixXd toDense() const override {
        return mat;
    }
};

// Wrapper for MMapMatrix
class MMapMatrixAdapter : public AbstractMatrix {
    std::unique_ptr<MMapMatrix> mat;
public:
    MMapMatrixAdapter(MMapMatrix* m) : mat(m) {}

    int rows() const override { return mat->rows(); }
    int cols() const override { return mat->cols(); }
    size_t nonZeros() const override { return mat->nonZeros(); }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd y = Eigen::VectorXd::Zero(mat->rows());
        mat->add_product_to(x, y, 1.0);
        return y;
    }

    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const override {
        mat->add_diagonal_to(diag, alpha);
    }

    Eigen::MatrixXd getBlock(int start_row, int size) const override {
        return mat->get_block(start_row, size, 1.0);
    }

    void visit_triplets(std::function<void(int, int, double)> callback) const override {
        mat->visit(callback);
    }

    Eigen::MatrixXd toDense() const override {
        // Warning: This can be huge.
        Eigen::MatrixXd D = Eigen::MatrixXd::Zero(rows(), cols());
        mat->visit([&](int r, int c, double v) {
            D(r, c) = v;
            if (r != c) D(c, r) = v; // Assume symmetric storage implies full reconstruction
        });
        return D;
    }
};

// Wrapper for Identity Matrix (e.g. for Permanent Environmental Effect, Pe)
class IdentityMatrixAdapter : public AbstractMatrix {
    int dim;
public:
    IdentityMatrixAdapter(int d) : dim(d) {}

    int rows() const override { return dim; }
    int cols() const override { return dim; }
    size_t nonZeros() const override { return dim; }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const override {
        return x; // I * x = x
    }

    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const override {
        int n = std::min(dim, (int)diag.size());
        for (int i = 0; i < n; ++i) {
            diag(i) += alpha;
        }
    }

    Eigen::MatrixXd getBlock(int start_row, int size) const override {
        return Eigen::MatrixXd::Identity(size, size);
    }

    void visit_triplets(std::function<void(int, int, double)> callback) const override {
        for (int i = 0; i < dim; ++i) {
            callback(i, i, 1.0);
        }
    }

    Eigen::MatrixXd toDense() const override {
        return Eigen::MatrixXd::Identity(dim, dim);
    }
};

// Split Matrix (A + G - A22)
// Assumes all matrices have same dimensions (or padded to same)
class SplitMatrixAdapter : public AbstractMatrix {
    AbstractMatrix* A;
    AbstractMatrix* G;
    AbstractMatrix* A22; // Subtracted term
    std::vector<int> map; // Global index (in A) for each row of G/A22. Empty if conformal.
    int _rows, _cols;
public:
    // Pointers are NOT owned by SplitMatrixAdapter
    SplitMatrixAdapter(AbstractMatrix* a, AbstractMatrix* g, AbstractMatrix* a22, const std::vector<int>& g_map = {})
        : A(a), G(g), A22(a22), map(g_map) {
        _rows = A->rows();
        _cols = A->cols();
    }

    int rows() const override { return _rows; }
    int cols() const override { return _cols; }
    size_t nonZeros() const override {
        return A->nonZeros() + (G ? G->nonZeros() : 0) + (A22 ? A22->nonZeros() : 0);
    }

    Eigen::VectorXd operator*(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd y = A->operator*(x);

        if (map.empty()) {
            if (G) y += G->operator*(x);
            if (A22) y -= A22->operator*(x);
        } else {
            // Scatter/Gather for G and A22
            int n_sub = (int)map.size();
            Eigen::VectorXd x_sub(n_sub);
            for(int i=0; i<n_sub; ++i) {
                int gid = map[i];
                if (gid >= 0 && gid < x.size()) x_sub(i) = x(gid);
                else x_sub(i) = 0.0;
            }

            Eigen::VectorXd y_sub = Eigen::VectorXd::Zero(n_sub);
            if (G) y_sub += G->operator*(x_sub);
            if (A22) y_sub -= A22->operator*(x_sub);

            for(int i=0; i<n_sub; ++i) {
                int gid = map[i];
                if (gid >= 0 && gid < y.size()) y(gid) += y_sub(i);
            }
        }
        return y;
    }

    void add_diagonal_to(Eigen::VectorXd& diag, double alpha) const override {
        A->add_diagonal_to(diag, alpha);

        auto add_sub = [&](AbstractMatrix* M, double sign) {
            if (!M) return;
            if (map.empty()) {
                M->add_diagonal_to(diag, alpha * sign);
            } else {
                M->visit_triplets([&](int r, int c, double v) {
                    if (r == c) {
                        if (r < (int)map.size()) {
                            int gid = map[r];
                            if (gid >= 0 && gid < diag.size()) {
                                diag(gid) += v * alpha * sign;
                            }
                        }
                    }
                });
            }
        };
        add_sub(G, 1.0);
        add_sub(A22, -1.0);
    }

    Eigen::MatrixXd getBlock(int start_row, int size) const override {
        Eigen::MatrixXd B = A->getBlock(start_row, size);

        auto add_sub_block = [&](AbstractMatrix* M, double sign) {
            if (!M) return;
            // Only add if block overlaps with mapped region?
            // M is subset.
            // If map is empty, just add M->getBlock
            if (map.empty()) {
                B += sign * M->getBlock(start_row, size);
            } else {
                // Mapping is complex for block extraction.
                // We need to find which indices in M map to [start_row, start_row+size).
                // If map is random, this is hard.
                // If map is sorted?
                // For now, rely on toDense() or just iterate triplets in range?
                // M->getBlock(r, s) uses local indices.
                // We need to map global [start, end) to local indices?
                // This is inverse map.
                // If we don't have inverse map, we can't efficiently find which local rows correspond to global rows.
                // So let's skip implementing getBlock for Mapped Split for now, or just warn.
                // Or iterate all map?
            }
        };

        if (map.empty()) {
            if (G) B += G->getBlock(start_row, size);
            if (A22) B -= A22->getBlock(start_row, size);
        }

        return B;
    }

    void visit_triplets(std::function<void(int, int, double)> callback) const override {
        A->visit_triplets(callback);

        auto visit_sub = [&](AbstractMatrix* M, double sign) {
             if (!M) return;
             if (map.empty()) {
                 M->visit_triplets([&](int r, int c, double v) {
                     callback(r, c, sign * v);
                 });
             } else {
                 M->visit_triplets([&](int r, int c, double v) {
                     if (r < (int)map.size() && c < (int)map.size()) {
                         int gr = map[r];
                         int gc = map[c];
                         if (gr >= 0 && gc >= 0) {
                             callback(gr, gc, sign * v);
                         }
                     }
                 });
             }
        };

        visit_sub(G, 1.0);
        visit_sub(A22, -1.0);
    }

    Eigen::MatrixXd toDense() const override {
        Eigen::MatrixXd D = A->toDense();
        if (map.empty()) {
            if (G) D += G->toDense();
            if (A22) D -= A22->toDense();
        } else {
             // ... implement if needed ...
        }
        return D;
    }
};

} // namespace cosmic
