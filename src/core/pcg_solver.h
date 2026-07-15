#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <type_traits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// Do NOT use 'using namespace' in headers to avoid pollution
// Use fully qualified names instead

static inline void spmv_row_major(const Eigen::SparseMatrix<double, Eigen::RowMajor>& Arow,
                                  const Eigen::VectorXd& x,
                                  Eigen::VectorXd& y) {
    // Eigen's operator* for sparse matrices is heavily optimized and will use BLAS/MKL if available
    y = Arow * x;
}

// Abstract Preconditioner Interface
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void build() = 0;
    virtual Eigen::VectorXd apply(const Eigen::VectorXd& v) const = 0;
};

// Jacobi (Diagonal) Preconditioner
class JacobiPreconditioner : public Preconditioner {
private:
    Eigen::VectorXd diag_inv;
    Eigen::VectorXd diag_copy;
public:
    JacobiPreconditioner(const Eigen::VectorXd& diag) : diag_copy(diag) {}
    JacobiPreconditioner(int n) { diag_inv = Eigen::VectorXd::Ones(n); }

    void build() override {
        if (diag_copy.size() > 0) {
            int n = (int)diag_copy.size();
            diag_inv = Eigen::VectorXd(n);
            for (int i = 0; i < n; ++i) {
                double d = diag_copy(i);
                diag_inv(i) = (std::fabs(d) > 1e-12) ? (1.0 / d) : 1.0;
            }
        }
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& v) const override {
        return diag_inv.array() * v.array();
    }
};

// Block Jacobi Preconditioner (for Fixed Effects)
class BlockJacobiPreconditioner : public Preconditioner {
private:
    int p_fixed;
    int n;
    Eigen::MatrixXd A11_dense;
    Eigen::LDLT<Eigen::MatrixXd> A11_ldlt;
    Eigen::VectorXd A22_diag_inv;
    bool valid_A11 = false;

    Eigen::VectorXd diag_copy;

public:
    BlockJacobiPreconditioner(const Eigen::MatrixXd& A11, const Eigen::VectorXd& diag, int p)
        : p_fixed(p), A11_dense(A11), diag_copy(diag) {
        n = (int)diag.size();
    }

    void build() override {
        if (A11_dense.size() > 0) {
            // Add a small ridge to A11 to prevent singularity issues
            double max_diag = 0.0;
            for(int i=0; i<p_fixed; ++i) {
                max_diag = std::max(max_diag, std::abs(A11_dense(i,i)));
            }
            double ridge = std::max(1e-8 * max_diag, 1e-6);
            for(int i=0; i<p_fixed; ++i) A11_dense(i,i) += ridge;

            A11_ldlt.compute(A11_dense);
            if (A11_ldlt.info() == Eigen::Success) {
                valid_A11 = true;
            } else {
                std::cerr << "Warning: Block Jacobi A11 factorization failed. Fallback to diagonal for fixed effects.\n";
            }
        }
        if (diag_copy.size() > 0) {
            int q = n - p_fixed;
            if (q > 0) {
                A22_diag_inv = Eigen::VectorXd(q);
                for(int i=0; i<q; ++i) {
                    double d = diag_copy(p_fixed + i);
                    A22_diag_inv(i) = (std::fabs(d) > 1e-12) ? (1.0 / d) : 1.0;
                }
            }
        }
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& v) const override {
        Eigen::VectorXd r(n);
        if (p_fixed > 0) {
            if (valid_A11) {
                r.head(p_fixed) = A11_ldlt.solve(v.head(p_fixed));
            } else {
                // Fallback to diagonal
                if (diag_copy.size() > 0) {
                    for(int i=0; i<p_fixed; ++i) {
                        double d = diag_copy(i);
                        r(i) = (std::fabs(d) > 1e-14) ? v(i)/d : v(i);
                    }
                } else {
                    r.head(p_fixed) = v.head(p_fixed);
                }
            }
        }
        if (n > p_fixed) {
            r.tail(n - p_fixed) = A22_diag_inv.array() * v.tail(n - p_fixed).array();
        }
        return r;
    }
};

// Incomplete Cholesky Preconditioner (IC0)
class IC0Preconditioner : public Preconditioner {
private:
    Eigen::IncompleteCholesky<double> ic;
    bool valid = false;

public:
    IC0Preconditioner(const Eigen::SparseMatrix<double>& A) {
        ic.compute(A);
        if (ic.info() == Eigen::Success) {
            valid = true;
        } else {
            std::cerr << "Warning: IC0 factorization failed. Using Identity.\n";
        }
    }

    void build() override {}

    Eigen::VectorXd apply(const Eigen::VectorXd& v) const override {
        if (!valid) return v;
        return ic.solve(v);
    }
};

// Pedigree Preconditioner (for ssGBLUP)
// Uses Ainv as M. M*z = r => Ainv * z = r => z = A * r.
// But A is dense. We have Ainv sparse.
// So M = Ainv. We want to solve M z = r.
// Ainv z = r.
// z = (Ainv)^-1 r = A r.
// Wait, preconditioning means solving M z = r.
// If we choose M = Ainv, then z = A r. Computing A r is dense.
// We want M such that M ~ (Hinv).
// If Hinv ~ Ainv, then we choose M = Ainv.
// Then we solve Ainv z = r.
// This is sparse solve!
// So we need to factorize Ainv.
class PedigreePreconditioner : public Preconditioner {
private:
    const Eigen::SparseMatrix<double>* Ainv;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    int p_fixed;
    Eigen::VectorXd diag_inv_fixed;
    bool valid = false;

    // Optional Fixed Effects Block Solver
    Eigen::MatrixXd A11_dense;
    Eigen::LDLT<Eigen::MatrixXd> A11_ldlt;
    bool valid_A11 = false;

public:
    PedigreePreconditioner(const Eigen::SparseMatrix<double>& Ainv_ref, int p)
        : Ainv(&Ainv_ref), p_fixed(p) {}

    void setFixedBlock(const Eigen::MatrixXd& A11) {
        A11_dense = A11;
        A11_ldlt.compute(A11_dense);
        if (A11_ldlt.info() == Eigen::Success) {
            valid_A11 = true;
        } else {
            std::cerr << "Warning: PedigreePreconditioner A11 factorization failed. Using Identity for fixed effects.\n";
        }
    }

    void build() override {
        if (Ainv && Ainv->rows() > 0) {
            solver.compute(*Ainv);
            if (solver.info() == Eigen::Success) valid = true;
            else std::cerr << "Warning: Pedigree preconditioner factorization failed.\n";
        }
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& v) const override {
        if (!valid) return v; // Fallback to identity

        Eigen::VectorXd r = v;

        // Apply to fixed effects part
        if (p_fixed > 0 && valid_A11) {
            r.head(p_fixed) = A11_ldlt.solve(v.head(p_fixed));
        }

        // Apply to random effects part
        if (v.size() > p_fixed) {
            Eigen::VectorXd r_u = v.tail(v.size() - p_fixed);
            Eigen::VectorXd z_u = solver.solve(r_u);
            r.tail(v.size() - p_fixed) = z_u;
        }
        return r;
    }
};

// Cluster Block Jacobi Preconditioner (for GBLUP)
// Divides the random effects into blocks (e.g., 1000 animals per block)
// and applies the inverse of these blocks.
// M = [ I_p  0  ... ]
//     [ 0   B_1 ... ]
//     [ 0   0   B_2 ]
class ClusterBlockJacobiPreconditioner : public Preconditioner {
private:
    int p_fixed;
    int n;
    int block_size;
    std::vector<Eigen::LDLT<Eigen::MatrixXd>> blocks_ldlt;
    bool valid = false;

    // Optional Fixed Effects Block Solver
    Eigen::MatrixXd A11_dense;
    Eigen::LDLT<Eigen::MatrixXd> A11_ldlt;
    bool valid_A11 = false;

public:
    // Takes a vector of dense blocks for the random effects
    ClusterBlockJacobiPreconditioner(const std::vector<Eigen::MatrixXd>& blocks, int p)
        : p_fixed(p) {

        if (blocks.empty()) return;
        block_size = (int)blocks[0].rows();
        blocks_ldlt.resize(blocks.size());

        // Build factorizations in parallel
        #pragma omp parallel for schedule(dynamic)
        for(int i=0; i<(int)blocks.size(); ++i) {
            blocks_ldlt[i].compute(blocks[i]);
            // We ignore failures for individual blocks, just fallback to diagonal implicitly?
            // LDLT solves even if singular (pivot), but accuracy might be bad.
        }
        valid = true;
    }

    void setFixedBlock(const Eigen::MatrixXd& A11) {
        A11_dense = A11;
        A11_ldlt.compute(A11_dense);
        if (A11_ldlt.info() == Eigen::Success) {
            valid_A11 = true;
        } else {
            std::cerr << "Warning: ClusterBlockJacobiPreconditioner A11 factorization failed. Using Identity for fixed effects.\n";
        }
    }

    void build() override {
        // Already built in constructor
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& v) const override {
        if (!valid) return v;

        Eigen::VectorXd r = v; // Copy
        int n_total = (int)v.size();

        // Apply fixed effects
        if (p_fixed > 0 && valid_A11) {
            r.head(p_fixed) = A11_ldlt.solve(v.head(p_fixed));
        }

        int n_random = n_total - p_fixed;

        if (n_random <= 0) return r;

        int n_blocks = (int)blocks_ldlt.size();

        // Apply random blocks
        #pragma omp parallel for schedule(static)
        for(int i=0; i<n_blocks; ++i) {
            int start_idx = p_fixed + i * block_size;
            int current_block_size = block_size;

            // Last block might be smaller?
            // The constructor assumed uniform blocks passed in.
            // If the last block is smaller, it should have been passed as smaller matrix.
            // But we store LDLT of matrix.
            // Let's assume the passed blocks MATCH the segmentation.

            // Check bounds
            if (start_idx >= n_total) continue;

            int actual_size = blocks_ldlt[i].rows(); // Get actual size from LDLT
            if (start_idx + actual_size > n_total) actual_size = n_total - start_idx; // Should match

            Eigen::VectorXd seg = v.segment(start_idx, actual_size);
            Eigen::VectorXd res = blocks_ldlt[i].solve(seg);

            // Scatter back
            r.segment(start_idx, actual_size) = res;
        }

        return r;
    }
};

template <typename MatrixType>
class PCGSolver {
private:
    const MatrixType& A;
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_row;
    Eigen::VectorXd b;
    int n;
    std::string precond_type = "diag";
    int p_fixed = 0;
    bool quiet = false;
    bool row_built = false;

    std::shared_ptr<Preconditioner> precond;

    // Pre-allocated work vectors
    Eigen::VectorXd x, r, z, p, Ap;

public:
    int last_iter = 0;
    double last_rel_res = 0.0;
    int restart_count = 0;

    PCGSolver(const MatrixType& lhs, const Eigen::VectorXd& rhs,
              const std::string& precond_t = "diag", int p_fixed_in = 0)
        : A(lhs), b(rhs), n((int)rhs.size()), precond_type(precond_t), p_fixed(p_fixed_in) {

        std::cerr << "DEBUG: PCGSolver Constructor. n=" << n << ", precond=" << precond_type << "\n";

        // Pre-allocate work vectors
        x.resize(n);
        r.resize(n);
        z.resize(n);
        p.resize(n);
        Ap.resize(n);

        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
            bool has_nan = false;
            for(int i=0; i<A.nonZeros(); ++i) {
                if (!std::isfinite(A.valuePtr()[i])) has_nan = true;
            }
            if (has_nan) std::cerr << "CRITICAL ERROR: Matrix A in PCG contains NaNs!\n";
        }

        // Setup preconditioner
        if (precond_type == "none") {
            // No preconditioner - will be set externally via setPreconditioner()
        } else if (precond_type == "block_jacobi") {
            if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
                Eigen::VectorXd diag = A.diagonal();
                Eigen::MatrixXd A11 = A.block(0,0,p_fixed,p_fixed);
                precond = std::make_shared<BlockJacobiPreconditioner>(A11, diag, p_fixed);
            } else {
                // Fallback to Jacobi
                precond_type = "diag";
            }
        }

        if (precond_type == "pedigree") {
             precond_type = "diag";
        }

        if (precond_type == "ic0") {
            if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
                precond = std::make_shared<IC0Preconditioner>(A);
            } else {
                precond_type = "diag";
            }
        }

        if (precond_type == "diag") {
            if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
                Eigen::VectorXd diag = A.diagonal();
                precond = std::make_shared<JacobiPreconditioner>(diag);
            } else {
                Eigen::VectorXd d(n);
                for(int i=0;i<n;++i) d(i) = A.coeff(i,i);
                precond = std::make_shared<JacobiPreconditioner>(d);
            }
        }

        if (precond) {
             std::cerr << "DEBUG: Calling precond->build()...\n";
             precond->build();
             std::cerr << "DEBUG: precond->build() done.\n";
        }
        std::cerr << "DEBUG: PCGSolver Constructor Done.\n";
    }

    // Setter for custom preconditioner
    void setPreconditioner(std::shared_ptr<Preconditioner> p) {
        precond = p;
        if (precond) precond->build();
    }

    void setQuiet(bool q) { quiet = q; }

    Eigen::VectorXd solve_with_rhs(const Eigen::VectorXd& rhs_in, double tol = 1e-8, int max_iter = 5000, int report_every = 100) {
        b = rhs_in; return solve(tol, max_iter, report_every);
    }

    void buildRowCopyIfNeeded() {
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
            if (!row_built) { A_row = Eigen::SparseMatrix<double, Eigen::RowMajor>(A); row_built = true; }
        }
    }

    void perform_mult(const Eigen::VectorXd& p, Eigen::VectorXd& Ap) {
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>>) {
             spmv_row_major(A_row, p, Ap);
        } else if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double, Eigen::RowMajor>>) {
             spmv_row_major(A, p, Ap);
        } else {
             A.multiply(p, Ap);
        }
    }

    Eigen::VectorXd solve(double tol = 1e-10, int max_iter = 1000, int report_every = 50, const Eigen::VectorXd& initial_guess = Eigen::VectorXd()) {
        buildRowCopyIfNeeded();
        bool is_small = (n < 1000);

        if (!b.allFinite()) {
            std::cerr << "CRITICAL ERROR: RHS contains NaNs before PCG solve!\n";
            return b;
        }

        if (initial_guess.size() == n) {
            x = initial_guess;
        } else {
            x.setZero();
        }

        perform_mult(x, Ap);

        r = b - Ap;
        z = precond ? precond->apply(r) : r;
        p = z;
        double rsold = r.dot(z);
        double bnorm = b.norm(); if (bnorm == 0.0) bnorm = 1.0;

        bool restarted = false;
        restart_count = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        if (!quiet && !is_small) {
#ifdef _OPENMP
            std::cerr << "OpenMP enabled, threads=" << omp_get_max_threads() << "\n";
#else
            std::cerr << "Single-threaded run\n";
#endif
            std::cerr << "PCG: n=" << n << ", nnz=" << A.nonZeros() << "\n";
        }

        for (int iter = 0; iter < max_iter; ++iter) {
            perform_mult(p, Ap);

            double denom = p.dot(Ap);

            if (!quiet && !is_small && iter == 0) {
                std::cerr << "DEBUG PCG Iter 0: rsold=" << rsold << ", denom=" << denom << ", p.norm()=" << p.norm() << ", r.norm()=" << r.norm() << "\n";
            }

            if (std::fabs(denom) < 1e-300) {
                if (!restarted) {
                    // Reset to simple diag precond if breakdown
                    // But here we just restart CG
                    z = precond ? precond->apply(r) : r;
                    p = z;
                    rsold = r.dot(z);
                    restarted = true;
                    restart_count += 1;
                    continue;
                }
                if (!quiet && !is_small) std::cerr << "PCG breakdown\n";
                break;
            }
            double alpha = rsold / denom;

            x.noalias() += alpha * p;
            r.noalias() -= alpha * Ap;

            double res_norm = r.norm(); double rel_res = res_norm / bnorm;

            if (!quiet && !is_small && (iter + 1) % report_every == 0) {
                 auto now = std::chrono::high_resolution_clock::now();
                 std::cerr << "Iter " << std::setw(4) << iter + 1
                      << " | residual=" << std::scientific << std::setprecision(3) << res_norm
                      << " | rel_res=" << rel_res
                      << " | elapsed(ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(now - t0).count() << "\n";
            }

            if (rel_res < tol) {
                if (!quiet && !is_small) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    std::cerr << "PCG 收敛: iterations=" << iter + 1 << ", rel_res=" << rel_res << "\n";
                    std::cerr << "time(ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "\n";
                }
                last_iter = iter + 1; last_rel_res = rel_res;
                return x;
            }
            z = precond ? precond->apply(r) : r;
            double rsnew = r.dot(z);
            double beta = rsnew / rsold;
            p = z + beta * p;
            rsold = rsnew;
            if (!x.allFinite() || !r.allFinite()) {
                if (!quiet) std::cerr << "数值问题在迭代 " << iter+1 << "\n";
                break;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!quiet && !is_small) std::cerr << "PCG 达到最大迭代次数未收敛，elapsed(ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "\n";
        last_iter = max_iter;
        perform_mult(x, Ap);
        last_rel_res = (b.norm() > 0 ? (b - Ap).norm() / bnorm : 0.0);
        return x;
    }
};
