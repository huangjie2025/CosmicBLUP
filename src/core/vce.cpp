#include "vce.h"
#include "stcg_vce.h"
#include "v_matrix.h"
#include "implicit_mme.h"
#include "adaptive_trace_estimator.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <map>
#include <Eigen/SparseCholesky>
#include <Eigen/Eigenvalues>

namespace cosmic {

// ============================================================================
// Strategy Policy Implementation
// ============================================================================

std::string to_string(VceStrategy s) {
    switch (s) {
        case VceStrategy::VBased:     return "V-based";
        case VceStrategy::MMEBased:   return "MME-based";
        case VceStrategy::MatrixFree: return "Matrix-free";
    }
    return "Unknown";
}

std::string to_string(SolverType s) {
    switch (s) {
        case SolverType::DenseLDLT:      return "Dense-LDLT";
        case SolverType::SparseLDLT:     return "Sparse-LDLT";
        case SolverType::PCG:            return "PCG";
        case SolverType::DenseVCholesky: return "V-Cholesky";
        case SolverType::DenseVEigen:    return "V-Eigen";
        case SolverType::DenseVLowRank:  return "V-LowRank";
        case SolverType::DenseVPCG_SLQ:  return "V-PCG/SLQ";
        case SolverType::STCG:           return "STCG";
    }
    return "Unknown";
}

std::string to_string(TraceMode m) {
    switch (m) {
        case TraceMode::Exact: return "Exact";
        case TraceMode::Fdiff: return "Finite-Diff";
        case TraceMode::Hutch: return "Hutchinson";
        case TraceMode::SLQ:   return "SLQ";
    }
    return "Unknown";
}

std::string to_string(PrecondType p) {
    switch (p) {
        case PrecondType::None:        return "None";
        case PrecondType::Jacobi:      return "Jacobi";
        case PrecondType::BlockJacobi: return "Block-Jacobi";
        case PrecondType::IC0:         return "IC(0)";
        case PrecondType::Pedigree:    return "Pedigree";
    }
    return "Unknown";
}

std::string to_string(TaskType t) {
    switch (t) {
        case TaskType::VCE:           return "VCE";
        case TaskType::Prediction:    return "Prediction";
        case TaskType::GWASNull:      return "GWAS-Null";
        case TaskType::Repeatability: return "Repeatability";
        case TaskType::RRM:           return "RRM";
    }
    return "Unknown";
}

SolverPolicy choose_policy(const PolicyInput& input) {
    SolverPolicy policy;
    int dim = input.mme_dim;
    int n = input.n_records;
    int nc = input.n_components;

    // Estimate memory for V-based route: V is n x n doubles
    size_t v_mem_bytes = (size_t)n * n * 8;
    size_t v_mem_mb = v_mem_bytes / (1024 * 1024);
    const size_t mem_limit_mb = 8000;  // 8 GB threshold for V matrix

    // ---- Step 1: Choose strategy (V-based / MME-based / Matrix-free) ----

    // RRM task: always MME-based with sparse solver
    if (input.task_type == TaskType::RRM) {
        policy.strategy = VceStrategy::MMEBased;
        if (dim < 5000) {
            policy.solver = SolverType::DenseLDLT;
            policy.trace_mode = TraceMode::Exact;
            policy.reason = "RRM + small dim=" + std::to_string(dim) + " -> Dense-LDLT";
        } else {
            policy.solver = SolverType::SparseLDLT;
            policy.trace_mode = TraceMode::Exact;
            policy.reason = "RRM + dim=" + std::to_string(dim) + " -> Sparse-LDLT";
        }
        policy.preconditioner = PrecondType::None;
        return policy;
    }

    // GWAS null model: lightweight, V-based for small n, MME for larger
    if (input.task_type == TaskType::GWASNull) {
        if (n <= 10000 && v_mem_mb < mem_limit_mb) {
            policy.strategy = VceStrategy::VBased;
            policy.solver = SolverType::DenseVEigen;
            policy.trace_mode = TraceMode::Exact;
            policy.preconditioner = PrecondType::None;
            policy.reason = "GWAS null + n=" + std::to_string(n) + " -> V-Eigen (fast null)";
        } else {
            policy.strategy = VceStrategy::MMEBased;
            policy.solver = SolverType::SparseLDLT;
            policy.trace_mode = TraceMode::Exact;
            policy.preconditioner = PrecondType::None;
            policy.reason = "GWAS null + n=" + std::to_string(n) + " -> Sparse-LDLT";
        }
        return policy;
    }

    // Repeatability + large n + genotype operator -> STCG
    if (input.task_type == TaskType::Repeatability && n > 80000 && input.has_genotype_operator) {
        policy.strategy = VceStrategy::MatrixFree;
        policy.solver = SolverType::STCG;
        policy.trace_mode = TraceMode::Hutch;
        policy.preconditioner = PrecondType::Jacobi;
        policy.reason = "Repeatability + large n=" + std::to_string(n) + " + genotype operator -> STCG";
        return policy;
    }

    // Repeatability + medium n -> V-based (aggregated mean)
    if (input.task_type == TaskType::Repeatability && n > 20000 && n <= 80000) {
        policy.strategy = VceStrategy::VBased;
        policy.solver = SolverType::DenseVEigen;
        policy.trace_mode = TraceMode::Exact;
        policy.preconditioner = PrecondType::None;
        policy.reason = "Repeatability + medium n=" + std::to_string(n) + " -> V-Eigen (aggregated)";
        return policy;
    }

    if (input.has_pedigree_sparse_inverse && dim > 200000) {
        policy.strategy = VceStrategy::MatrixFree;
        policy.solver = SolverType::PCG;
        policy.trace_mode = TraceMode::Hutch;
        policy.preconditioner = PrecondType::Pedigree;
        policy.reason = "Large dim=" + std::to_string(dim) + " + sparse Ainv -> Matrix-free PCG";
        return policy;
    }

    // Small sample: V-based route (memory-safe)
    if (n <= 5000 && nc <= 1 && v_mem_mb < mem_limit_mb) {
        policy.strategy = VceStrategy::VBased;
        if (n <= 2000) {
            policy.solver = SolverType::DenseVCholesky;
            policy.reason = "Small n=" + std::to_string(n) + " + single component -> V-Cholesky";
        } else {
            policy.solver = SolverType::DenseVEigen;
            policy.reason = "Medium n=" + std::to_string(n) + " + single component -> V-Eigen";
        }
        policy.trace_mode = TraceMode::Exact;
        policy.preconditioner = PrecondType::None;
        return policy;
    }

    // Medium sample: V-based with approximation (check memory)
    if (n <= 20000 && n <= 5000 * nc && v_mem_mb < mem_limit_mb) {
        policy.strategy = VceStrategy::VBased;
        if (n <= 8000) {
            policy.solver = SolverType::DenseVEigen;
            policy.reason = "n=" + std::to_string(n) + " + multi-component -> V-Eigen";
        } else if (n <= 80000) {
            policy.solver = SolverType::DenseVLowRank;
            policy.reason = "n=" + std::to_string(n) + " -> V-LowRank";
        } else {
            policy.solver = SolverType::DenseVPCG_SLQ;
            policy.reason = "n=" + std::to_string(n) + " -> V-PCG/SLQ";
        }
        policy.trace_mode = TraceMode::Exact;
        policy.preconditioner = PrecondType::None;
        return policy;
    }

    // ---- Step 2: MME-based route (default for most cases) ----
    policy.strategy = VceStrategy::MMEBased;

    if (dim < 5000) {
        policy.solver = SolverType::DenseLDLT;
        policy.trace_mode = TraceMode::Exact;
        policy.preconditioner = PrecondType::None;
        policy.reason = "dim=" + std::to_string(dim) + " < 5K -> Dense-LDLT";
    } else if (dim < 200000 && input.lhs_density > 0.0) {
        policy.solver = SolverType::SparseLDLT;
        policy.trace_mode = TraceMode::Exact;
        policy.preconditioner = PrecondType::None;
        policy.reason = "dim=" + std::to_string(dim) + " + density=" +
                        std::to_string(input.lhs_density).substr(0,6) + " -> Sparse-LDLT";
    } else {
        // Large: PCG
        policy.solver = SolverType::PCG;
        policy.trace_mode = TraceMode::Hutch;
        if (input.has_pedigree_sparse_inverse) {
            policy.preconditioner = PrecondType::Pedigree;
        } else if (nc > 1 || dim > 500000) {
            policy.preconditioner = PrecondType::BlockJacobi;
        } else {
            policy.preconditioner = PrecondType::Jacobi;
        }
        policy.reason = "dim=" + std::to_string(dim) + " > 200K -> PCG + " + to_string(policy.preconditioner);
    }

    return policy;
}

void print_solver_report(const SolverReport& r, std::ostream& os) {
    os << "\n========================================\n";
    os << "  Cosmic Solver Report\n";
    os << "========================================\n";
    os << "  Task:           " << to_string(r.task) << "\n";
    os << "  Strategy:       " << to_string(r.strategy) << "\n";
    os << "  Solver:         " << to_string(r.solver) << "\n";
    os << "  Trace mode:     " << to_string(r.trace_mode) << "\n";
    os << "  Preconditioner: " << to_string(r.preconditioner) << "\n";
    if (!r.strategy_reason.empty())
        os << "  Reason:         " << r.strategy_reason << "\n";
    if (r.fallback_triggered)
        os << "  Fallback:       YES - " << r.fallback_reason << "\n";
    os << "  MME dim:        " << r.mme_dim << "\n";
    os << "  LHS density:    " << r.lhs_density << "\n";
    os << "  Components:     " << r.n_components << "\n";
    if (r.pcg_iterations > 0)
        os << "  PCG iters:      " << r.pcg_iterations << "\n";
    if (r.vce_iterations > 0)
        os << "  VCE iters:      " << r.vce_iterations << "\n";
    os << "========================================\n\n";
}

// ============================================================================
// Existing code
// ============================================================================

static bool is_identity_sparse(const Eigen::SparseMatrix<double>& M, double tol = 1e-12) {
    if (M.rows() != M.cols()) return false;
    const int n = (int)M.rows();
    if (M.nonZeros() != n) return false;
    std::vector<int> cnt(n, 0);
    for (int k = 0; k < M.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            if (it.row() != it.col()) return false;
            if (std::abs(it.value() - 1.0) > tol) return false;
            int i = it.row();
            if (i < 0 || i >= n) return false;
            cnt[i] += 1;
            if (cnt[i] > 1) return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (cnt[i] != 1) return false;
    }
    return true;
}

static Eigen::Matrix2d project_spd_2x2(const Eigen::Matrix2d& A, double eps_rel = 1e-8, double eps_abs = 1e-12) {
    Eigen::Matrix2d S = 0.5 * (A + A.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(S);
    if (es.info() != Eigen::Success) {
        double scale = std::max({std::abs(S(0, 0)), std::abs(S(1, 1)), 1.0});
        S(0, 0) += eps_rel * scale + eps_abs;
        S(1, 1) += eps_rel * scale + eps_abs;
        return 0.5 * (S + S.transpose());
    }
    Eigen::Vector2d eval = es.eigenvalues();
    double max_eval = std::max(eval.maxCoeff(), 0.0);
    double floor = std::max(eps_abs, eps_rel * max_eval);
    eval(0) = std::max(eval(0), floor);
    eval(1) = std::max(eval(1), floor);
    Eigen::Matrix2d Q = es.eigenvectors();
    return Q * eval.asDiagonal() * Q.transpose();
}

// Generic SPD projection for arbitrary n x n symmetric matrix.
// Used by multi-component AI-REML (c+1 >= 3 parameters).
// Clips negative eigenvalues to a floor relative to the largest eigenvalue.
static Eigen::MatrixXd project_spd(const Eigen::MatrixXd& A, double eps_rel = 1e-8, double eps_abs = 1e-12) {
    Eigen::MatrixXd S = 0.5 * (A + A.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    if (es.info() != Eigen::Success) {
        // Fallback: add diagonal jitter
        double scale = 1.0;
        for (int i = 0; i < S.rows(); ++i) scale = std::max(scale, std::abs(S(i, i)));
        S += (eps_rel * scale + eps_abs) * Eigen::MatrixXd::Identity(S.rows(), S.cols());
        return 0.5 * (S + S.transpose());
    }
    Eigen::VectorXd eval = es.eigenvalues();
    double max_eval = std::max(eval.maxCoeff(), 0.0);
    double floor = std::max(eps_abs, eps_rel * max_eval);
    for (int i = 0; i < eval.size(); ++i) {
        eval(i) = std::max(eval(i), floor);
    }
    Eigen::MatrixXd Q = es.eigenvectors();
    return Q * eval.asDiagonal() * Q.transpose();
}

// Helper to compute selected inverse elements using Takahashi
// Returns sparse matrix containing elements of Z = L^-T D^-1 L^-1 corresponding to pattern of L + I
// Modified: Accepts optional Ainv pattern to augment the computation (Extended Takahashi)
static void compute_sparse_inverse_subset(const Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver,
                                          Eigen::SparseMatrix<double>& Z_subset,
                                          const AbstractMatrix* Ainv_ptr = nullptr,
                                          int p_offset = 0) {
    // 1. Access L and D and Permutation
    Eigen::SparseMatrix<double> L = solver.matrixL();
    Eigen::VectorXd D = solver.vectorD();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = solver.permutationP();
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P_inv = P.inverse();

    int n = (int)L.rows();

    // Z will hold the sparse inverse subset.
    Z_subset.resize(n, n);
    Z_subset.setZero();

    // Pre-processing for Extended Takahashi (Symbolic Propagation)
    // We need to ensure Z is computed for all (r, c) in P^T * Ainv * P
    std::vector<std::vector<int>> extra_reqs(n);
    bool use_extension = (Ainv_ptr != nullptr); // Enable for all sizes (Sparse map supports it now)

    if (use_extension) {
//         if (n <= 20000 || n % 1000 == 0) std::cout << "DEBUG: Pre-processing Extended Takahashi requirements... n=" << n << ", p=" << p_offset << std::endl;

        std::vector<bool> req_mat((size_t)n * n, false);

        // 1. Initial Requirements from Ainv
        try {
            long long cnt = 0;
            const auto& indices = P.indices();
            int size_P = (int)indices.size();

            Ainv_ptr->visit_triplets([&](int r, int c, double v) {
                if (r >= c) { // Lower triangle
                    if (p_offset + r >= size_P || p_offset + c >= size_P) return;

                    int r_perm = indices(p_offset + r);
                    int c_perm = indices(p_offset + c);

                    int u = std::max(r_perm, c_perm);
                    int v_idx = std::min(r_perm, c_perm);

                    if (u != v_idx) {
                        req_mat[(size_t)v_idx * n + u] = true;
                    }
                }
                cnt++;
            });
//             std::cout << "DEBUG: Scanned " << cnt << " triplets." << std::endl;
        } catch (...) {
            std::cerr << "Exception in visit_triplets" << std::endl;
        }

        // 3. Propagate Requirements (Forward 0 -> n-1)
//         std::cout << "DEBUG: Starting Symbolic Propagation..." << std::endl;

        long long added_reqs = 0;
        try {
            for (int i = 0; i < n; ++i) {
//                 if (i % 1000 == 0) std::cout << "DEBUG: Propagating i=" << i << " total_added=" << added_reqs << std::endl;

                // Get L_{ji} non-zeros
                std::vector<int> L_rows;
                for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
                    if (it.row() > i) L_rows.push_back(it.row());
                }

                // For each required row r in column i
                for (int r = i + 1; r < n; ++r) {
                    if (!req_mat[(size_t)i * n + r]) continue;

                    // For each j where L_{ji} != 0
                    for (int j : L_rows) {
                        int u = std::max(r, j);
                        int v_idx = std::min(r, j);

                        if (u != v_idx) {
                            if (!req_mat[(size_t)v_idx * n + u]) {
                                req_mat[(size_t)v_idx * n + u] = true;
                                added_reqs++;
                            }
                        }
                    }
                }
            }
        } catch (...) { std::cerr << "Error in propagation" << std::endl; }
//         std::cout << "DEBUG: Propagated " << added_reqs << " requirements." << std::endl;

        // Final Sort for consumption
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (req_mat[(size_t)i * n + j]) {
                    extra_reqs[i].push_back(j);
                }
            }
        }
    }

    // Use dense matrix for Z if n is small enough (e.g. <= 25000) to fit in RAM (~5GB).
    // This avoids map overhead which is critical if dense blocks exist.
    if (n <= 25000) {
//         std::cout << "DEBUG: Using Dense Storage for Takahashi (n=" << n << ")..." << std::endl;
        // Use vector as flat 2D array.
        // Initialize with 0.0.
        std::vector<double> Z_dense;
        try {
            Z_dense.resize((size_t)n * n, 0.0);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Error: Not enough memory for dense Takahashi buffer. Falling back to sparse map." << std::endl;
            goto fallback_map;
        }

        auto get_Z = [&](int r, int c) -> double {
            return Z_dense[(size_t)r * n + c];
        };
        auto set_Z = [&](int r, int c, double v) {
            Z_dense[(size_t)r * n + c] = v;
            Z_dense[(size_t)c * n + r] = v;
        };

        for (int i = n - 1; i >= 0; --i) {
             // Collect indices k where L_ki != 0
             std::vector<int> k_indices;
             std::vector<double> L_vals;
             for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
                 if (it.row() > i) {
                     k_indices.push_back(it.row());
                     L_vals.push_back(it.value());
                 }
             }

             // Merge L pattern with Extra Requirements
             std::vector<int> calc_indices = k_indices;
             if (use_extension && !extra_reqs[i].empty()) {
                 std::sort(extra_reqs[i].begin(), extra_reqs[i].end());
                 extra_reqs[i].erase(std::unique(extra_reqs[i].begin(), extra_reqs[i].end()), extra_reqs[i].end());

                 // Merge
                 for(int req : extra_reqs[i]) {
                     calc_indices.push_back(req);
                 }
                 std::sort(calc_indices.begin(), calc_indices.end());
                 calc_indices.erase(std::unique(calc_indices.begin(), calc_indices.end()), calc_indices.end());
             }

             // Compute Z_{ki} for all k in calc_indices
             for (int k : calc_indices) {
                 double sum = 0.0;
                 // Sum over j where L_{ji} != 0
                 for (size_t j_idx = 0; j_idx < k_indices.size(); ++j_idx) {
                     int j = k_indices[j_idx];
                     double l_ji = L_vals[j_idx];
                     // Z_kj is symmetric
                     sum += get_Z(k, j) * l_ji;
                 }
                 set_Z(k, i, -sum);
             }

             double sum_diag = 0.0;
             for (size_t idx = 0; idx < k_indices.size(); ++idx) {
                 int k = k_indices[idx];
                 double l_ki = L_vals[idx];
                 sum_diag += get_Z(k, i) * l_ki;
             }
             set_Z(i, i, (1.0 / D(i)) - sum_diag);
        }

        std::vector<Eigen::Triplet<double>> triplets;
        // Diagonal
        for(int i=0; i<n; ++i) {
            triplets.emplace_back(i, i, get_Z(i, i));
        }

        // We need to return triplets for BOTH Pattern(L) AND Pattern(Ainv).
        // Actually, returning EVERYTHING we computed is safer/better.
        // But Z_subset is usually expected to be sparse.
        // If Ainv is sparse, Union is sparse.
        // So we iterate over non-zeros of Z_dense? No, that's O(N^2).
        // We iterate over the computed indices.
        // But we didn't store them structure-wise.
        // Re-construct logic:
        // Iterate i, then iterate calc_indices (re-merge or store).
        // Or just iterate L and extra_reqs.

        for (int i = 0; i < n; ++i) {
             std::vector<int> k_indices;
             for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
                 if (it.row() > i) k_indices.push_back(it.row());
             }

             std::vector<int> calc_indices = k_indices;
             if (use_extension && !extra_reqs[i].empty()) {
                 for(int req : extra_reqs[i]) calc_indices.push_back(req);
                 std::sort(calc_indices.begin(), calc_indices.end());
                 calc_indices.erase(std::unique(calc_indices.begin(), calc_indices.end()), calc_indices.end());
             }

             for(int k : calc_indices) {
                 double val = get_Z(k, i);
                 if (val != 0.0) { // Should be non-zero usually
                     triplets.emplace_back(k, i, val);
                     triplets.emplace_back(i, k, val);
                 }
             }
        }

        Z_subset.setFromTriplets(triplets.begin(), triplets.end());
//         if (true) std::cout << "DEBUG: Takahashi (Dense) done." << std::endl;
        return;
    }

fallback_map:
    // std::cout << "DEBUG: Using Sparse Map Storage for Takahashi..." << std::endl;
    std::vector<std::map<int, double>> Z_rows(n);

    // Pre-fill Z_rows with structure of L (to avoid map allocations during loop)
    // Actually, L structure is enough?
    // Takahashi might introduce non-zeros not in L?
    // "Selected inversion computes entries of A^-1 corresponding to nonzeros of L."
    // Yes, exact structural match.

    // Optimization: The inner loop `sum Z_kj * L_ji` runs over j > i.
    // We want to compute Z_ki for a specific k.
    // Z_ki = - sum_{j > i} Z_kj * L_ji.
    // Note: Z_kj is symmetric. We need (k, j).
    // If k >= j, look in Z_rows[k][j].
    // If k < j, look in Z_rows[j][k].

    for (int i = n - 1; i >= 0; --i) {
        // if (i % 1000 == 0) std::cout << "DEBUG: Takahashi i=" << i << std::endl;
        // 1. Compute Off-Diagonal Z_ki for k > i

        // Collect indices k where L_ki != 0 (L structure)
        std::vector<int> k_indices;
        std::vector<double> L_vals;
        for (Eigen::SparseMatrix<double>::InnerIterator it(L, i); it; ++it) {
            if (it.row() > i) {
                k_indices.push_back(it.row());
                L_vals.push_back(it.value());
            }
        }

        // Merge L pattern with Extra Requirements
        std::vector<int> calc_indices = k_indices;
        if (use_extension && !extra_reqs[i].empty()) {
             std::sort(extra_reqs[i].begin(), extra_reqs[i].end());
             extra_reqs[i].erase(std::unique(extra_reqs[i].begin(), extra_reqs[i].end()), extra_reqs[i].end());
             for(int req : extra_reqs[i]) {
                 calc_indices.push_back(req);
             }
             std::sort(calc_indices.begin(), calc_indices.end());
             calc_indices.erase(std::unique(calc_indices.begin(), calc_indices.end()), calc_indices.end());
        }

        // For each k in calc_indices, compute Z_ki
        for (int k : calc_indices) {
            double sum = 0.0;

            // sum_{j > i} Z_kj * L_ji
            // We only need terms where L_ji != 0 (iterate L structure k_indices)
            for (size_t j_idx = 0; j_idx < k_indices.size(); ++j_idx) {
                int j = k_indices[j_idx];
                double l_ji = L_vals[j_idx];

                // Get Z_kj
                double z_kj = 0.0;
                int r = std::max(k, j);
                int c = std::min(k, j);
                if (Z_rows[r].count(c)) z_kj = Z_rows[r][c];

                sum += z_kj * l_ji;
            }

            Z_rows[k][i] = -sum; // Store Z_ki
        }

        // 2. Compute Diagonal Z_ii
        // Z_ii = 1/D_ii - sum_{k > i} Z_ki * L_ki
        // Only sum over k where L_ki != 0
        double sum_diag = 0.0;
        for (size_t idx = 0; idx < k_indices.size(); ++idx) {
            int k = k_indices[idx];
            double l_ki = L_vals[idx];

            // Z_ki is in Z_rows[k][i] (since k > i)
            double z_ki = 0.0;
            if (Z_rows[k].count(i)) z_ki = Z_rows[k][i];

            sum_diag += z_ki * l_ki;
        }

        Z_rows[i][i] = (1.0 / D(i)) - sum_diag;
    }

    // Convert Z_rows to SparseMatrix
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
        for (auto& kv : Z_rows[i]) {
            triplets.emplace_back(i, kv.first, kv.second);
            if (i != kv.first) triplets.emplace_back(kv.first, i, kv.second); // Symmetric
        }
    }

    Z_subset.resize(n, n);
    Z_subset.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << "DEBUG: Takahashi done." << std::endl;
}

// --- Pedigree Sampler Implementation ---

static inline std::string trim(const std::string& s) {
    size_t first = s.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, (last - first + 1));
}

void PedigreeSampler::load(const std::string& ped_file, const std::map<std::string, int>& id_map) {
    std::cout << "Loading pedigree for sampler from: " << ped_file << std::endl;
    std::ifstream file(ped_file);
    if (!file.is_open()) throw std::runtime_error("Cannot open pedigree file: " + ped_file);

    int n = (int)id_map.size();
    ped.assign(n, {-1, -1});
    D_sqrt.assign(n, 1.0); // Default to sqrt(1) if no parents

    // We need to process pedigree in topological order (parents before children).
    // Usually pedigree files are sorted. We assume sorted or we might fail.
    // Actually, L*z sampling only requires L.
    // A = T * D * T' where T is lower triangular with 1s on diagonal and -0.5 for parents.
    // L = T * sqrt(D).
    // So u = T * sqrt(D) * z.
    // T is inverse of (I - P), where P relates child to parents.
    // Actually Ainv = (I-P)' Dinv (I-P).
    // So A = (I-P)^-1 D (I-P')^-1.
    // Let L = (I-P)^-1 * sqrt(D). Then A = L L'.
    // u = L * z  => (I-P) * u = sqrt(D) * z.
    // So u_i = 0.5 * u_s + 0.5 * u_d + sqrt(D_ii) * z_i.
    // This is the recursive formula for sampling!
    // We just need to iterate from oldest to youngest.

    std::string line;
    int loaded = 0;
    while(std::getline(file, line)) {
        std::string row = trim(line);
        if (row.empty()) continue;
        std::stringstream ss(row);
        std::string id_s, sire_s, dam_s;
        ss >> id_s >> sire_s >> dam_s;

        if (id_map.find(id_s) == id_map.end()) continue;
        int i = id_map.at(id_s);

        int s = -1, d = -1;
        if (id_map.count(sire_s)) s = id_map.at(sire_s);
        if (id_map.count(dam_s)) d = id_map.at(dam_s);

        ped[i] = {s, d};

        // Compute D_sqrt
        // F_i = 0.5 * (F_s + F_d) ... Wait, we usually approximate F=0 if not calculating inbreeding.
        // Standard approximation:
        // Both parents known: var = 0.5 * sigma_u^2
        // One parent known: var = 0.75 * sigma_u^2
        // Neither known: var = 1.0 * sigma_u^2
        // If we ignore inbreeding (F=0):
        // phi = 0.5 for full term, but here D is Mendelian sampling variance fraction.
        // D_ii = 1 - 0.25*(1+Fs) - 0.25*(1+Fd)
        // If F=0:
        // 2 parents: 1 - 0.25 - 0.25 = 0.5
        // 1 parent: 1 - 0.25 = 0.75
        // 0 parents: 1.0

        double val = 1.0;
        if (s != -1 && d != -1) val = 0.5;
        else if (s != -1 || d != -1) val = 0.75;

        D_sqrt[i] = std::sqrt(val);
        loaded++;
    }
    std::cout << "Pedigree loaded. " << loaded << " animals matched." << std::endl;
}

void PedigreeSampler::sample(Eigen::VectorXd& u) {
    int n = (int)ped.size();
    if (u.size() != n) u.resize(n);

    // 1. Generate z ~ N(0, 1)
    Eigen::VectorXd z(n);
    rng.fill_normal(z);

    // 2. Compute u recursively: u_i = 0.5*u_s + 0.5*u_d + D_sqrt[i]*z_i
    // Assumption: ped is sorted such that parents appear before children (or at least have lower indices if we process by index).
    // The id_map usually maps by order of appearance in Ainv/Ped file.
    // We assume 0..n-1 is a valid topological order or close to it.
    // If not, we might need to sort. But for efficiency, let's assume valid order.

    for (int i = 0; i < n; ++i) {
        double u_val = D_sqrt[i] * z[i];
        int s = ped[i].s;
        int d = ped[i].d;

        if (s != -1 && s < i) u_val += 0.5 * u[s];
        if (d != -1 && d < i) u_val += 0.5 * u[d];

        // If s > i, it means parent comes later. This breaks the DAG assumption.
        // In most breeding evaluations, animals are renumbered 1..N by birth date.

        u[i] = u_val;
    }
}

// --- AI_REML Implementation ---

AI_REML::AI_REML(const std::vector<GenRecord>& recs_ref,
                 const FixedDesignG& fd_ref,
                 const std::vector<RandomComponent>& comps,
                 const Eigen::VectorXd& y_ref,
                 VCEConfig cfg)
    : VCESolver(recs_ref, fd_ref, comps, y_ref, cfg) {}

AI_REML::AI_REML(const std::vector<GenRecord>& recs_ref,
                 const FixedDesignG& fd_ref,
                 const AbstractMatrix* Ainv_ref,
                 const Eigen::VectorXd& y_ref,
                 VCEConfig cfg)
    : VCESolver(recs_ref, fd_ref, Ainv_ref, y_ref, cfg) {}

AI_REML::~AI_REML() = default;

void AI_REML::solve() {
    if (!initialized) initialize();

    // Fill and print solver report (IASBLUP-style)
    if (config.print_report) {
        solver_report_.task = TaskType::VCE;
        solver_report_.mme_dim = (int)y.size();
        solver_report_.n_components = (int)vars_u.size() + 1;
        // Determine actual strategy from current state
        if (sparse_direct_solver) {
            solver_report_.strategy = VceStrategy::MMEBased;
            solver_report_.solver = SolverType::SparseLDLT;
            solver_report_.trace_mode = TraceMode::Exact;
        } else if (direct_solver) {
            solver_report_.strategy = VceStrategy::MMEBased;
            solver_report_.solver = SolverType::DenseLDLT;
            solver_report_.trace_mode = TraceMode::Exact;
        } else if (implicit_mme) {
            solver_report_.strategy = VceStrategy::MatrixFree;
            solver_report_.solver = SolverType::PCG;
            solver_report_.trace_mode = TraceMode::Hutch;
        } else {
            solver_report_.strategy = VceStrategy::MMEBased;
            solver_report_.solver = SolverType::PCG;
            solver_report_.trace_mode = TraceMode::Hutch;
        }
        solver_report_.strategy_reason = "dim=" + std::to_string(solver_report_.mme_dim) +
            ", components=" + std::to_string(solver_report_.n_components);
        print_solver_report(solver_report_);
    }

    // HI Strategy: Initialize with HE Regression if requested
    if (config.use_he_init && vars_u.size() == 1) {
        if (false) std::cout << "  [Init] Running HE Regression for initialization..." << std::endl;
        HE_Regression he_solver(*recs_ptr, *fd_ptr, components, y, config);
        // Temporarily disable verbose for HE to keep log clean
        // Actually, HE output is useful.
        he_solver.solve();
        if (he_solver.getVarU() > 0 && he_solver.getVarE() > 0) {
            vars_u[0] = he_solver.getVarU();
            var_e = he_solver.getVarE();
            if (false) std::cout << "  [Init] HE Initialized: Vg=" << vars_u[0] << ", Ve=" << var_e << std::endl;
        } else {
            if (false) std::cout << "  [Init] HE Regression failed or negative variance. Using defaults." << std::endl;
        }
    }

    double start_vu = vars_u.empty() ? 1.0 : vars_u[0];

    if (config.verbose) {
        LOG_INFO("Total " << vars_u.size() + 1 << " variance components need to be estimated.");
        LOG_INFO("Variance components estimation using: AI(" << config.max_iter << ")");
        LOG_INFO("Running ...");
        LOG_INFO("Alg.\tIter.\tLogL.\tV(G)\tV(e)");
    }

    history.clear();
    converged = false;
    iterations_run = 0;
    last_diff = std::numeric_limits<double>::quiet_NaN();

    for (int iter = 0; iter < config.max_iter; ++iter) {
        double old_ve = var_e;
        std::vector<double> old_vus = vars_u;

        run_ai_iteration(iter);

        double current_vu = vars_u.empty() ? 0.0 : vars_u[0];
        if (config.verbose) {
                std::string logL_str = "N/A";
                std::string method_label = "AI";
                if (!history.empty()) {
                    std::istringstream iss(history.back());
                    std::string method, iter_str, ll_str;
                    if (iss >> method >> iter_str >> ll_str) {
                        logL_str = ll_str;
                        method_label = method;
                    }
                }
                LOG_INFO("[" << method_label << "]\t" << iter + 1 << "\t" << logL_str << "\t"
                              << std::fixed << std::setprecision(5) << current_vu << "\t" << var_e);
            }

        // Convergence Check
        double diff = std::abs(var_e - old_ve);
        for(size_t k=0; k<vars_u.size(); ++k) diff += std::abs(vars_u[k] - old_vus[k]);

        last_diff = diff;
        iterations_run = iter + 1;

        if (diff < config.tol) {
               if (config.verbose) LOG_INFO("[Converged?] Yes!\n");
                converged = true;
               break;
          }
    }
}

void AI_REML::initialize() {
    // std::cerr << "DEBUG: AI_REML::initialize called! Ainv_ptr=" << Ainv_ptr << std::endl;
    // std::cout << "Initializing AI-REML components..." << std::endl;

    // Unified Builder Initialization
    if (recs_ptr && fd_ptr) {
//         std::cout << "DEBUG: recs_ptr and fd_ptr are valid. Creating MMELHSBuilder..." << std::endl;
//         std::cout << "DEBUG: components.size() = " << components.size() << std::endl;
        for (size_t i = 0; i < components.size(); ++i) {
//             std::cout << "DEBUG: component[" << i << "].Qinv = " << components[i].Qinv << std::endl;
        }
        mme_builder = std::make_unique<MMELHSBuilder>(*recs_ptr, *fd_ptr, components);
//         std::cout << "DEBUG: MMELHSBuilder created." << std::endl;

        // Pre-calculate Xty and Zty using Builder's transpose multiply
        // We need 'y' vector. If y is empty in base class (new constructor), extract from recs.
        if (y.size() == 0) {
            int n = (int)recs_ptr->size();
            // We assume y is provided or we can't solve.
            // If y is empty, we might need to populate it from recs, but VCESolver::y is const ref.
            // For now, assume y is valid.
        }

//         std::cout << "DEBUG: Building full RHS..." << std::endl;
        // Build RHS = [X'y; Z'y]
        Eigen::VectorXd full_rhs = mme_builder->build_rhs(*recs_ptr, *fd_ptr, mme_builder->get_dim());
//         std::cout << "DEBUG: RHS built." << std::endl;
        int p = mme_builder->get_p();
        int q = mme_builder->get_q_total();
        Xty = full_rhs.head(p);
        Zty = full_rhs.tail(q);

    } else {
        throw std::runtime_error("AI_REML requires raw records (GenRecord) and FixedDesignG for Unified Builder. Legacy mode is deprecated.");
    }

//     std::cout << "DEBUG: Checking factorization logic..." << std::endl;
    // TODO: Support factorization for multiple components if needed.
    if (!Ainv_factorized && components.size() == 1) {
        const AbstractMatrix* Ainv_ptr = components[0].Qinv;
        // std::cerr << "DEBUG: Checking factorization. Rows=" << Ainv_ptr->rows() << std::endl;
        // Increase limit to 100k to support moderately large pedigree matrices
        if (Ainv_ptr->rows() > 100000) {
             // std::cout << "    [Init] Skipping Ainv factorization (Size > 100000). Using approximate Hessian sampling." << std::endl;
             use_dense_Ainv = false;
             Ainv_factorized = false;
        } else {
            const SparseMatrixAdapter* sp_adapter = dynamic_cast<const SparseMatrixAdapter*>(Ainv_ptr);
            if (!sp_adapter) {
                // std::cout << "    [Init] Skipping factorization (Abstract Matrix, not in-memory sparse)." << std::endl;
                Ainv_factorized = false;
            } else {
                const auto& Ainv_ref = sp_adapter->getMatrix();

                // DEBUG: Enable Ainv factorization (LDLT)
                bool enable_ainv_fact = true;
                // if (false) std::cout << "    [Init] Factorizing Ainv for correct variance sampling..." << std::endl;

                double density = (Ainv_ref.rows() > 0 && Ainv_ref.cols() > 0) ? (double)Ainv_ref.nonZeros() / (double(Ainv_ref.rows()) * double(Ainv_ref.cols())) : 0.0;

//                 std::cerr << "DEBUG: Ainv_ref.rows() = " << Ainv_ref.rows() << ", Density = " << density << std::endl;

                // Use Dense Factorization if matrix is small (up to 5000 rows)
                // 16k rows (SSGBLUP benchmark) = ~2GB RAM, which is risky for Dense LDLT on some systems.
                // We prefer Sparse Factorization for 16k unless it fails.
                bool force_dense = (Ainv_ref.rows() < 5000);

                // If density is high and size is VERY large, skip factorization to avoid crash
                // Increased limit to 50k to allow Sparse Factorization for 16k SSGBLUP
                bool skip_factorization = (Ainv_ref.rows() >= 50000 && density > 0.001);

                if (skip_factorization) {
                     // std::cout << "    [Init] Skipping factorization (Rows=" << Ainv_ref.rows() << ", Density=" << density << "). Will use EM mode." << std::endl;
                     Ainv_factorized = false;
                     use_dense_Ainv = false;
                } else if (enable_ainv_fact && force_dense) {
//                      std::cerr << "DEBUG: Entering Dense Block. Rows=" << Ainv_ref.rows() << std::endl;
                     // std::cout << "    [Init] Using Dense Factorization (Rows=" << Ainv_ref.rows() << ", Density=" << density << ")..." << std::endl;
                     use_dense_Ainv = true;

//                      std::cerr << "DEBUG: Converting Sparse to Dense..." << std::endl;
                     Eigen::MatrixXd Ad = Ainv_ref;
//                      std::cerr << "DEBUG: Dense conversion done. Checking for NaNs..." << std::endl;
                       if (!Ad.allFinite()) {
                           std::cerr << "ERROR: Dense matrix contains NaNs or Infs!" << std::endl;
                           throw std::runtime_error("Dense matrix contains NaNs or Infs");
                       }
//                        std::cerr << "DEBUG: Matrix finite. Starting Dense LDLT..." << std::endl;

                       try {
                          // Use LDLT for robustness
                          // Construct empty first, then compute to isolate errors
                          Ainv_dense_chol = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>();
//                           std::cerr << "DEBUG: Allocated LDLT object. Computing..." << std::endl;
                          Ainv_dense_chol->compute(Ad);
                       } catch (const std::bad_alloc& e) {
                           std::cerr << "EXCEPTION: Memory allocation failed for Dense LDLT: " << e.what() << std::endl;
                           Ainv_factorized = false;
                           use_dense_Ainv = false;
                       } catch (const std::exception& e) {
                           std::cerr << "EXCEPTION during Dense LDLT: " << e.what() << std::endl;
                           Ainv_factorized = false;
                           use_dense_Ainv = false;
                       } catch (...) {
                           std::cerr << "UNKNOWN EXCEPTION during Dense LDLT" << std::endl;
                           Ainv_factorized = false;
                           use_dense_Ainv = false;
                       }

//                        std::cerr << "DEBUG: Dense LDLT returned." << std::endl;

                       if (Ainv_dense_chol->info() != Eigen::Success) {
                           std::cerr << "Warning: Ainv dense factorization failed. Info=" << Ainv_dense_chol->info() << std::endl;
                           // Try to recover with sparse? Or just fail?
                           // Let's try sparse as backup
                           use_dense_Ainv = false;
                           Ainv_factorized = false;
                       } else {
                           Ainv_factorized = true;
                           // std::cout << "    [Init] Dense Ainv factorization success." << std::endl;
                       }
                  } else if (enable_ainv_fact) {
                       use_dense_Ainv = false;
//                        std::cerr << "DEBUG: Starting Sparse SimplicialLDLT..." << std::endl;
                       try {
                           Ainv_sparse_chol = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
                           Ainv_sparse_chol->compute(Ainv_ref);
//                            std::cerr << "DEBUG: Sparse SimplicialLDLT returned. Info=" << Ainv_sparse_chol->info() << std::endl;
                       } catch (const std::exception& e) {
                           std::cerr << "EXCEPTION during Sparse SimplicialLDLT: " << e.what() << std::endl;
                           throw;
                       } catch (...) {
                           std::cerr << "UNKNOWN EXCEPTION during Sparse SimplicialLDLT" << std::endl;
                           throw;
                       }
                     if (Ainv_sparse_chol->info() != Eigen::Success) {
                          std::cerr << "Warning: Ainv sparse factorization failed (LDLT). Variance estimates may be inaccurate." << std::endl;
                          if (Ainv_sparse_chol->info() == Eigen::NumericalIssue) std::cerr << "  Reason: Numerical Issue" << std::endl;
                          else if (Ainv_sparse_chol->info() == Eigen::InvalidInput) std::cerr << "  Reason: Invalid Input" << std::endl;

                          // Fallback to Dense LDLT if small enough
                          if (Ainv_ref.rows() < 25000) {
                               // std::cout << "    [Init] Retrying with Dense Factorization (Rows=" << Ainv_ref.rows() << ")..." << std::endl;
                               use_dense_Ainv = true;
                               Eigen::MatrixXd Ad = Ainv_ref;
                               Ainv_dense_chol = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>(Ad);
                               if (Ainv_dense_chol->info() == Eigen::Success) {
                                   Ainv_factorized = true;
                                   // std::cout << "    [Init] Dense Ainv factorization success." << std::endl;
                               } else {
                                   std::cerr << "Warning: Dense Ainv factorization also failed." << std::endl;
                                   Ainv_factorized = false;
                               }
                          } else {
                              Ainv_factorized = false;
                          }
                     } else {
                          Ainv_factorized = true;
                          // std::cout << "    [Init] Ainv factorization success (LDLT)." << std::endl;
                     }
                } else {
                     // std::cout << "    [Init] Ainv factorization disabled (DEBUG)." << std::endl;
                     Ainv_factorized = false;
                }
            }
        }
    }

    // Initialize Matrix-Free MME if mode is MC or explicitly requested
    std::string mode_str = config.vce_mode;
    std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
    if (mode_str == "mc" || config.algorithm == VCEAlgorithm::Fdiff) {
        std::vector<double> init_lambdas;
        for (double vu : vars_u) {
            init_lambdas.push_back(var_e / std::max(vu, 1e-9));
        }
        implicit_mme = std::make_unique<ImplicitMME>(*recs_ptr, *fd_ptr, components, init_lambdas);
    }

    initialized = true;
}

void AI_REML::build_lhs(const std::vector<double>& lambdas) {
    if (implicit_mme) {
        implicit_mme->lambdas = lambdas;
        implicit_mme->computeDiagonal();
        lhs_built = true;
        return;
    }

    if (mme_builder) {
        // Unified Builder Path
        // std::cout << "    [build_lhs] Updating LHS (Unified Builder)..." << std::endl;

        // Determine if we have cross-component covariances
        bool use_cov_matrix = false;
        Eigen::MatrixXd Lambda = Eigen::MatrixXd::Zero(vars_u.size(), vars_u.size());
        for (size_t i = 0; i < vars_u.size(); ++i) {
            Lambda(i, i) = var_e / std::max(vars_u[i], 1e-9);
            // Future: Inject off-diagonal covariance here when multi-trait / maternal cov is estimated
        }

        // This updates the internal matrix in mme_builder
        const auto& lhs = mme_builder->build_lhs(Lambda);

        // Setup solvers using builder's LHS
        int dim = (int)lhs.rows();

        // Pre-compute decomposition for Direct Solver if matrix is small or sparse
        // Use Dense Direct Solver for N < 10000 (handles ssGBLUP with dense G-block)
        // A 10000x10000 dense matrix is ~800MB, manageable on typical workstations.
        double lhs_density = (lhs.rows() > 0) ? (double)lhs.nonZeros() / (double(lhs.rows()) * double(lhs.rows())) : 0.0;
        bool use_dense_direct = (dim < 10000) || (lhs_density > 0.05 && dim < 15000);

        if (use_dense_direct) {
            if (!direct_solver) direct_solver = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>();
            current_LHS_dense = lhs;
            current_LHS_dense_valid = true;
            direct_solver->compute(current_LHS_dense);
        } else {
            current_LHS_dense_valid = false;
            // Use Sparse Direct for medium size ONLY if sparse enough
            // Increased density limit to 0.05 and size to 200k for PBLUP support
            bool enable_sparse_direct = (dim < 200000 && lhs_density < 0.05);
            // Do not enable sparse direct if mode is explicitly MC to avoid expensive factorization
            std::string mode_str = config.vce_mode;
            std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::tolower);
            if (mode_str == "mc") {
                enable_sparse_direct = false;
                // std::cout << "    [build_lhs] VCE Mode is MC. Skipping Sparse Direct Solver." << std::endl;
            }

            if (enable_sparse_direct) {
                if (!sparse_direct_solver) sparse_direct_solver = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();

                if (!sparse_symbolic_done) {
                    // First time: full compute (symbolic + numeric)
                    sparse_direct_solver->compute(lhs);
                    sparse_symbolic_done = true;
                } else {
                    // Subsequent calls: reuse symbolic factorization, only update numeric
                    // The sparsity pattern is unchanged (fast_update_ready guarantees this)
                    sparse_direct_solver->factorize(lhs);
                }

                if (sparse_direct_solver->info() != Eigen::Success) {
                    std::cerr << "Warning: Sparse Direct Solver factorization failed." << std::endl;
                    sparse_symbolic_done = false;  // Reset so next attempt does full compute
                } else {
                    Eigen::SparseMatrix<double> L_mat = sparse_direct_solver->matrixL();
                    long long nnz_L = L_mat.nonZeros();
                    // Threshold: allow up to 500 non-zeros per row (vs 150 default).
                    // ssGBLUP H-inverse has a dense G-inverse block that causes high fill-in,
                    // but the sparse solver is still valuable for exact trace (Takahashi) and
                    // log-det computation. Discarding it forces noisy stochastic trace in EM,
                    // which causes VCE divergence.
                    long long max_fillin = (long long)dim * 500;
                    if (nnz_L > max_fillin) {
                        std::cerr << "    [build_lhs] Fill-in explosion in Cholesky L: " << nnz_L
                                  << " nnz (threshold " << max_fillin << "). Switching to PCG." << std::endl;
                        sparse_direct_solver.reset(); // Destroy solver to force fallback
                        sparse_symbolic_done = false;
                    } else {
                        // std::cout << "    [build_lhs] Sparse Direct Solver factorized. Density=" << lhs_density << ", L_nnz=" << nnz_L << std::endl;
                    }
                }
            } else {
                 // if (config.verbose && mode_str != "mc") std::cout << "    [build_lhs] Sparse Direct Solver skipped. Dim=" << dim << ", Density=" << lhs_density << std::endl;
            }
        }

        lhs_built = true;
        return;
    }

    // --- Legacy Path (for reml_1d or non-builder mode) ---
    if (!mme_builder) {
         throw std::runtime_error("Legacy VCE mode (without raw records) is no longer supported for AI_REML. Please use Unified Constructor.");
    }
    // These are not needed if we throw above, but to be safe and clean:
    // int p = mme_builder->get_p();
    // int q = mme_builder->get_q();

}

Eigen::VectorXd AI_REML::solve_lhs(const Eigen::VectorXd& rhs, const Eigen::VectorXd& initial_guess) {
    if (!lhs_built) throw std::runtime_error("LHS not built!");

    if (implicit_mme) {
        int p = mme_builder->get_p();
        ::PCGSolver<ImplicitMME> pcg(*implicit_mme, rhs, config.pcg_precond, p);
        pcg.setQuiet(true);
        if (initial_guess.size() == rhs.size()) {
            return pcg.solve(config.pcg_tol, config.pcg_max_iter, 0, initial_guess);
        } else {
            return pcg.solve(config.pcg_tol, config.pcg_max_iter);
        }
    }

    // Determine which matrix to use
    // If mme_builder is present, use its LHS
    // Otherwise fallback (but we threw error in build_lhs if no builder, so this is safe)

    const Eigen::SparseMatrix<double>* lhs_ptr = nullptr;
    if (mme_builder) {
        lhs_ptr = &mme_builder->get_lhs();
    } else {
        throw std::runtime_error("Legacy solve_lhs not supported");
    }

    int dim = (int)lhs_ptr->rows();

    // Use Direct Solver for small/medium matrices for speed and robustness
    if (dim < 10000 && direct_solver) {
        if(direct_solver->info() == Eigen::Success) {
            return direct_solver->solve(rhs);
        } else {
             // std::cout << "    [solve_lhs] Direct solver failed. Falling back to PCG." << std::endl;
        }
    } else if (dim < 200000 && sparse_direct_solver) {
        if(sparse_direct_solver->info() == Eigen::Success) {
            // // std::cout << "    [solve_lhs] Using Sparse Direct Solver." << std::endl;
            return sparse_direct_solver->solve(rhs);
        }
    }

    // Use Block Jacobi if fixed effects exist, otherwise Diagonal
    int p = mme_builder->get_p();
    std::string precond = (config.pcg_precond != "diag") ? config.pcg_precond : ((p > 0) ? "block_jacobi" : "diag");

    // Build or reuse cached PCG components
    // The sparsity pattern of LHS doesn't change across VCE iterations (fast_update_ready),
    // so we can reuse the Row-major conversion and preconditioner factorization.
    bool need_rebuild = !pcg_cache_valid || (pcg_cache_p_fixed != p) || (pcg_cache_precond_type != precond);

    // Always update the Row-major matrix values
    // Note: CSC and Row-major have different internal value ordering, so we must use
    // Eigen's assignment operator which handles the conversion correctly.
    // If structure is unchanged, Eigen optimizes this to a value-only copy.
    cached_A_row = Eigen::SparseMatrix<double, Eigen::RowMajor>(*lhs_ptr);

    if (need_rebuild) {
        // Build preconditioner
        if (precond == "ic0") {
            cached_precond = std::make_shared<IC0Preconditioner>(*lhs_ptr);
        } else if (precond == "block_jacobi") {
            Eigen::VectorXd diag = lhs_ptr->diagonal();
            Eigen::MatrixXd A11 = lhs_ptr->block(0,0,p,p);
            cached_precond = std::make_shared<BlockJacobiPreconditioner>(A11, diag, p);
        } else {
            Eigen::VectorXd diag = lhs_ptr->diagonal();
            cached_precond = std::make_shared<JacobiPreconditioner>(diag);
        }
        if (cached_precond) cached_precond->build();

        pcg_cache_valid = true;
        pcg_cache_p_fixed = p;
        pcg_cache_precond_type = precond;
    }

    // For IC0 preconditioner, we need to refactorize with updated values
    // (IC0 depends on numerical values, not just sparsity pattern)
    if (precond == "ic0" && !need_rebuild) {
        // IC0 must be refactored each time because values change
        cached_precond = std::make_shared<IC0Preconditioner>(*lhs_ptr);
        cached_precond->build();
    }
    // For Jacobi/BlockJacobi, rebuild is cheap so we do it every time
    if (precond != "ic0") {
        if (precond == "block_jacobi") {
            Eigen::VectorXd diag = lhs_ptr->diagonal();
            Eigen::MatrixXd A11 = lhs_ptr->block(0,0,p,p);
            cached_precond = std::make_shared<BlockJacobiPreconditioner>(A11, diag, p);
        } else {
            Eigen::VectorXd diag = lhs_ptr->diagonal();
            cached_precond = std::make_shared<JacobiPreconditioner>(diag);
        }
        cached_precond->build();
    }

    // Create PCGSolver with cached Row-major matrix and custom preconditioner
    // Use "none" to skip building a default preconditioner; we set it via setPreconditioner()
    ::PCGSolver<Eigen::SparseMatrix<double, Eigen::RowMajor>> pcg(cached_A_row, rhs, "none", 0);
    pcg.setPreconditioner(cached_precond);
    pcg.setQuiet(true);

    // Warm start if possible
    if (last_solution.size() == rhs.size()) {
        last_solution = pcg.solve(config.pcg_tol, config.pcg_max_iter, 0, last_solution);
        return last_solution;
    } else if (initial_guess.size() == rhs.size()) {
        last_solution = pcg.solve(config.pcg_tol, config.pcg_max_iter, 0, initial_guess);
        return last_solution;
    } else {
        last_solution = pcg.solve(config.pcg_tol, config.pcg_max_iter);
        return last_solution;
    }
}

Eigen::VectorXd AI_REML::ApplyP(const Eigen::VectorXd& v) {
    if (!mme_builder) throw std::runtime_error("ApplyP requires mme_builder");
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();

    // P v = R^{-1} (v - W C^{-1} W^T R^{-1} v)
    // Here R = I * var_e, so R^{-1} = (1 / var_e) * I

    // 1. Scale v
    Eigen::VectorXd scaled_v = v / var_e;

    // 2. rhs = W^T scaled_v
    Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd rhs = mme_builder->mult_transpose_design(scaled_v, *recs_ptr, *fd_ptr);

    // 3. Solve MME: C x = rhs
    // We suppress PCG logs during ApplyP trace estimation
    Eigen::VectorXd x = solve_lhs(rhs);

    // 4. t = W x
    Eigen::VectorXd beta = x.head(p);
    Eigen::VectorXd u = x.tail(q);
    Eigen::VectorXd t = mme_builder->mult_design(beta, u, *recs_ptr, *fd_ptr);

    // 5. Return (v - t) / var_e
    return (v - t) / var_e;
}

double AI_REML::estimate_trace_Qinv_Cuu(int component_idx, int n_samples) {
    if (!mme_builder) throw std::runtime_error("estimate_trace_Qinv_Cuu requires initialized mme_builder");
    int p = mme_builder->get_p();
    int q_total = mme_builder->get_q_total();
    const auto& qs = mme_builder->get_qs();

    if (component_idx < 0 || component_idx >= (int)qs.size()) throw std::runtime_error("Invalid component index");

    int q_k = qs[component_idx];
    int q_offset = 0;
    for(int i=0; i<component_idx; ++i) q_offset += qs[i];

    if (direct_solver && direct_solver->info() == Eigen::Success && q_k <= 5000) {
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(p + q_total, q_k);
        rhs.block(p + q_offset, 0, q_k, q_k).setIdentity();
        Eigen::MatrixXd sol = direct_solver->solve(rhs);
        Eigen::MatrixXd Ckk = sol.block(p + q_offset, 0, q_k, q_k);

        double trace = 0.0;
        components[component_idx].Qinv->visit_triplets([&](int r, int c, double v) {
            trace += v * Ckk(c, r);
        });
        return trace;
    }

    if (sparse_direct_solver && sparse_direct_solver->info() == Eigen::Success && q_k <= 5000) {
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(p + q_total, q_k);
        rhs.block(p + q_offset, 0, q_k, q_k).setIdentity();
        Eigen::MatrixXd sol = sparse_direct_solver->solve(rhs);
        Eigen::MatrixXd Ckk = sol.block(p + q_offset, 0, q_k, q_k);

        double trace = 0.0;
        components[component_idx].Qinv->visit_triplets([&](int r, int c, double v) {
            trace += v * Ckk(c, r);
        });
        return trace;
    }

    auto op = [&](const Eigen::VectorXd& z) -> Eigen::VectorXd {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p + q_total);
        rhs.segment(p + q_offset, q_k) = z;
        Eigen::VectorXd sol = solve_lhs(rhs);
        Eigen::VectorXd Cuu_z_k = sol.segment(p + q_offset, q_k);
        return components[component_idx].Qinv->operator*(Cuu_z_k);
    };

    int min_samp = std::max(30, n_samples / 3);
    int max_samp = std::max(100, n_samples);
    return AdaptiveTraceEstimator::estimate(q_k, op, min_samp, max_samp, 0.05);
}

double AI_REML::estimate_trace_Cuu(int n_samples) {
    if (!mme_builder) throw std::runtime_error("estimate_trace_Cuu requires initialized mme_builder");
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();

    auto op = [&](const Eigen::VectorXd& z) -> Eigen::VectorXd {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(p + q);
        rhs.tail(q) = z;
        Eigen::VectorXd sol = solve_lhs(rhs);
        return sol.tail(q);
    };

    int min_samp = std::max(30, n_samples / 3);
    int max_samp = std::max(100, n_samples);
    return AdaptiveTraceEstimator::estimate(q, op, min_samp, max_samp, 0.05);
}

void AI_REML::run_ai_iteration(int iter) {
    // Mode Logic based on VCEAlgorithm enum
    bool use_ai = false;

    // Default Hybrid Strategy: EM for first 'em_max_iter' iterations, then AI
    if (config.algorithm == VCEAlgorithm::EMAI) {
        use_ai = (iter >= config.em_max_iter);
    } else if (config.algorithm == VCEAlgorithm::AI) {
        use_ai = true;
    } else if (config.algorithm == VCEAlgorithm::EM) {
        use_ai = false;
    } else {
        // Fallback for other modes (MC, HE, Exact should use their own solve() methods)
        use_ai = (iter >= 3);
    }

    // ----- Multi-component AI dispatch (Path A) -----
    // For multi-component models (vars_u.size() > 1), we always use the
    // standard EM path for variance component updates (guaranteed monotonic
    // convergence on near-confounded likelihood ridges). The AI matrix is
    // computed separately at convergence for standard errors (see
    // calculate_SE). This matches the approach used by ASReml's EM mode.
    if (vars_u.size() > 1) {
        use_ai = false;
    }

    // Single-component AI requires Ainv factorization OR a direct MME solver
    // (the data-driven AI matrix path uses solve_lhs for RHS vectors).
    if (use_ai && vars_u.size() == 1 && !Ainv_factorized && !sparse_direct_solver && !direct_solver) {
        use_ai = false;
    }

    std::string mode = config.vce_mode;
    // Lowercase conversion (keep for legacy compatibility if used elsewhere)
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

    double start_ve = var_e;
    std::vector<double> start_vus = vars_u;

    const double y_mean = y.size() > 0 ? y.mean() : 0.0;
    const double y_var = y.size() > 1
        ? std::max(1e-12, (y.array() - y_mean).square().sum() / (double)(y.size() - 1))
        : std::max(1e-12, y.squaredNorm());
    const double variance_floor = 1e-12;
    const double variance_ceiling = std::max(1.0, y_var * 1e6);
    auto stabilize_variance = [&](double proposed, double previous) {
        if (!std::isfinite(proposed) || proposed <= 0.0) return std::max(previous, variance_floor);
        double lower = variance_floor;
        double upper = variance_ceiling;
        if (std::isfinite(previous) && previous > 0.0) {
            lower = std::max(lower, previous * 0.1);
            upper = std::min(upper, previous * 10.0);
        }
        if (lower > upper) lower = variance_floor;
        return std::min(std::max(proposed, lower), upper);
    };

    // Check for Dense Exact Mode condition
    if (!mme_builder) throw std::runtime_error("run_ai_iteration requires initialized mme_builder");
    int n_total = mme_builder->get_p() + mme_builder->get_q_total();

    // Debug info
      if (false && iter == 0) {
          std::cout << "  [Optimization Check] N=" << n_total << ", Threshold=5000, Force=" << (config.force_dense ? "Yes" : "No") << std::endl;
      }

    // Store solution for LogL
    Eigen::VectorXd current_sol;

    // Check if we can use Exact Trace (Sparse Direct Solver Available)
    // If Sparse Direct Solver is used (e.g. PBLUP with small-medium size), we can do Exact AI-REML efficiently too.
    bool use_sparse_exact = (sparse_direct_solver != nullptr);

    if (mode == "mc") {
        use_sparse_exact = false;
        // Do not force dense if mode is mc
    }

    // TRACE: decision point (kept for diagnostics; remove before production)
    // std::cerr << "  [TRACE run_ai_iteration] iter=" << iter ...

    if ((config.force_dense || n_total < 5000 || use_sparse_exact) && mode != "mc") {
        if (config.verbose && iter == 0) std::cout << "  [Optimization] Switching to Exact AI-REML (Dense or Sparse)." << std::endl;

        bool success = false;
        // Dense exact mode only supports single random component; use AI-REML for multi-component
        bool use_dense_exact = (config.force_dense || (n_total < 5000 && !use_sparse_exact)) && vars_u.size() <= 1;
        if (use_dense_exact) {
             success = run_dense_exact_step(iter);
        } else {
             success = run_aireml_step(iter);
        }

        if (success) {
             return;
        } else {
             // If Exact Step failed, check mode.
             // If mode is strictly AI, we might want to stop or retry with damping.
             // But Exact mode failure usually means numerical issues or non-convergence.
             // For now, if Exact fails, we fall back to standard logic below.
             if (false) std::cout << "  [Optimization] Exact Step failed. Fallback to standard." << std::endl;
        }
    } else {
        // Logic handled at top of function via VCEAlgorithm enum.
        // Legacy string mode ignored here as enum takes precedence.
    }

    if (use_ai && current_sol.size() == 0) {
        bool success = run_aireml_step(iter);
        if (!success) {
             if (config.algorithm == VCEAlgorithm::AI) {
                 // In pure AI mode, failure is critical — but if LogL is unavailable
                 // (PCG/matrix-free path), fall back to EM to make progress.
                 if (!direct_solver && !sparse_direct_solver) {
                     use_ai = false;  // fall back to EM for this iteration
                 }
             } else {
                 // Hybrid or fallback allowed
                 if (false) std::cout << "  [Iter " << iter << "] AI Step failed. Falling back to EM..." << std::endl;
                 use_ai = false;
             }
        } else {
             // Success path
             if (direct_solver || sparse_direct_solver) {
                  std::vector<double> lambdas;
                  for (double vu : vars_u) lambdas.push_back(var_e / std::max(vu, 1e-9));
                  build_lhs(lambdas);
                  int p = mme_builder->get_p();
                  int q = mme_builder->get_q_total();
                  Eigen::VectorXd rhs(p + q);
                  rhs << Xty, Zty;
                  current_sol = solve_lhs(rhs);
             }

             // --- Robustness Check (Monotonicity) ---
             // Implementation of "Robust AI" (Knight 2008)
             // Check if the AI step caused a decrease in Likelihood.
             // If so, reject the step and fall back to EM (or use damping, but EM is safer).
             // TODO: calc_logL_internal needs update for multiple components.
             // For now assume single component or skip.

             /*
             if (iter > 0 && last_logL > -1e19) {
                 double new_logL = calc_logL_internal();
                 if (new_logL < last_logL) {
                      if (false) std::cout << "  [Robust Check] AI step decreased LogL (" << last_logL << " -> " << new_logL << "). Reverting to EM (Robust AI)." << std::endl;
                      vars_u = start_vus;
                      var_e = start_ve;
                      use_ai = false;
                      current_sol.resize(0);
                 }
             }
             */
        }
    }

    if (!use_ai) {
        // --- EM Implementation (Multi-Component) ---
        int n = (int)y.size();
        int p = mme_builder->get_p();
        int q_total = mme_builder->get_q_total();
        const auto& qs = mme_builder->get_qs();

        std::vector<double> lambdas;
        for (double vu : vars_u) lambdas.push_back(var_e / std::max(vu, 1e-9));

        // Build LHS once per iteration
        build_lhs(lambdas);

        if (false) std::cout << "  [Iter " << iter << "] Solving MME for data (EM)..." << std::endl;
        // 1. Solve MME for data
        Eigen::VectorXd rhs(p + q_total);
        rhs << Xty, Zty;
        Eigen::VectorXd sol = solve_lhs(rhs);
        current_sol = sol;

        Eigen::VectorXd u_all = sol.tail(q_total);
        Eigen::VectorXd beta_hat = sol.head(p);

        // Residuals
        Eigen::VectorXd e_hat;
        Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_all, *recs_ptr, *fd_ptr);
        e_hat = y - y_hat;

        double e_sse = e_hat.dot(e_hat);

        if (false) std::cout << "  [Iter " << iter << "] Estimating traces (EM)..." << std::endl;

        // 2. Estimate Traces & Update
        std::vector<double> tr_Qinv_Cuu(vars_u.size(), 0.0);
        int n_samp = (direct_solver || sparse_direct_solver) ? std::max(50, config.mc_samples) : std::max(5, config.mc_samples / 2);

        for (size_t k = 0; k < vars_u.size(); ++k) {
             // Extract u_k
             int offset = 0;
             for(size_t i=0; i<k; ++i) offset += qs[i];
             Eigen::VectorXd u_k = u_all.segment(offset, qs[k]);

             // Compute u_k' Qinv_k u_k
             Eigen::VectorXd Qinv_uk = components[k].Qinv->operator*(u_k);
             double u_Qinv_u = u_k.dot(Qinv_uk);

             // Estimate Trace
             bool use_exact = (vars_u.size() == 1 && sparse_direct_solver && sparse_direct_solver->info() == Eigen::Success);

             double tr_val = 0.0;
             if (use_exact) {
                  if (false) std::cout << "    [EM Step] Using Exact Trace (Takahashi)..." << std::endl;
                  Eigen::SparseMatrix<double> Z_subset;
                  compute_sparse_inverse_subset(*sparse_direct_solver, Z_subset, components[0].Qinv, p);
                  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P_mat = sparse_direct_solver->permutationP();
                  components[0].Qinv->visit_triplets([&](int r, int c, double v) {
                      int r_perm = P_mat.indices()(p + r);
                      int c_perm = P_mat.indices()(p + c);
                      double val_C = Z_subset.coeff(r_perm, c_perm);
                      tr_val += v * val_C;
                  });
             } else {
                  tr_val = estimate_trace_Qinv_Cuu((int)k, n_samp);
             }
             tr_Qinv_Cuu[k] = tr_val;

              // Update var_u[k]
              double vu_new = (u_Qinv_u + tr_val * var_e) / qs[k];
              vars_u[k] = stabilize_variance(vu_new, start_vus[k]);
        }

        // Update Ve
        double sum_lambda_tr = 0.0;
        for (size_t k = 0; k < vars_u.size(); ++k) {
            sum_lambda_tr += lambdas[k] * tr_Qinv_Cuu[k];
        }
        double tr_C_WdW = (double)(p + q_total) - sum_lambda_tr;
        double ve_new = (e_sse + var_e * tr_C_WdW) / n;

        var_e = stabilize_variance(ve_new, start_ve);
        if (!vars_u.empty()) var_u_legacy = vars_u[0];

        // Invalidate LHS
        lhs_built = false;
    }

    // --- LogL Calculation ---
    double logL = 0.0;
    bool logL_valid = false;

    // Always compute LogL for monitoring
    if (current_sol.size() == 0) {
         std::vector<double> lambdas;
         for (double vu : vars_u) lambdas.push_back(var_e / std::max(vu, 1e-9));
         build_lhs(lambdas);
         int p = mme_builder->get_p();
         int q = mme_builder->get_q_total();
         Eigen::VectorXd rhs(p + q);
         rhs << Xty, Zty;
         current_sol = solve_lhs(rhs);
    }

    if ((direct_solver || sparse_direct_solver) && current_sol.size() > 0) {
         double log_det_C = 0.0;
         bool ok_det = false;
         if (direct_solver && direct_solver->info() == Eigen::Success) {
             const Eigen::VectorXd D = direct_solver->vectorD();
             ok_det = true;
             for (int i = 0; i < D.size(); ++i) {
                 double d = D(i);
                 if (!(std::isfinite(d)) || d <= 0.0) { ok_det = false; break; }
                 log_det_C += std::log(d);
             }
         } else if (sparse_direct_solver && sparse_direct_solver->info() == Eigen::Success) {
             const Eigen::VectorXd D = sparse_direct_solver->vectorD();
             ok_det = true;
             for (int i = 0; i < D.size(); ++i) {
                 double d = D(i);
                 if (!(std::isfinite(d)) || d <= 0.0) { ok_det = false; break; }
                 log_det_C += std::log(d);
             }
         }
         if (!ok_det) {
             logL_valid = false;
         } else {

         // yPy = (y'y - sol' RHS)/Ve
         // y'y computed per call (must NOT be static: different AI_REML instances
         // in the same process would otherwise reuse a stale yty).
         double yty = y.dot(y);

         int p = mme_builder->get_p();
         int q = mme_builder->get_q_total();
         Eigen::VectorXd rhs(p + q); rhs << Xty, Zty;
         double sol_rhs = current_sol.dot(rhs);

         double yPy = (yty - sol_rhs) / var_e;

         double n = (double)y.size();

         double log_det_G = 0.0;
         if (vars_u.size() == components.size()) {
             const auto& qs = mme_builder->get_qs();
             for(size_t k=0; k<vars_u.size(); ++k) {
                 log_det_G += (double)qs[k] * std::log(vars_u[k]);
             }
         } else {
              // Fallback
              log_det_G = (double)q * std::log(vars_u[0]);
          }

             logL = -0.5 * (log_det_C + (n - p - q) * std::log(var_e) + log_det_G + yPy);
             logL_valid = std::isfinite(logL);
         }
    }

    if (logL_valid) last_logL = logL;

    // Store current solution for warm start
    if (current_sol.size() > 0) {
        final_solution = current_sol;
    }

    double var_u_legacy = (!vars_u.empty()) ? vars_u[0] : 0.0;
    std::ostringstream ss;
    ss << (use_ai ? "AI" : "EM") << "\t" << (iter + 1) << "\t";
    if (logL_valid) ss << std::fixed << std::setprecision(2) << logL;
    else ss << "0.00";
    ss << "\t" << std::setprecision(5) << var_u_legacy << "\t" << var_e;
    history.push_back(ss.str());
}

Eigen::Vector2d AI_REML::compute_stochastic_gradient(double vu, double ve, const std::vector<Eigen::VectorXd>& z_samples) {
    if (!mme_builder) throw std::runtime_error("compute_stochastic_gradient requires initialized mme_builder");
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();
    int n = (int)y.size();
    const AbstractMatrix* Ainv_ptr = components.empty() ? nullptr : components[0].Qinv;
    if (!Ainv_ptr) throw std::runtime_error("compute_stochastic_gradient requires Qinv");

    // Rebuild LHS with perturbed variances
    if (vu < 1e-9) vu = 1e-9;
    if (ve < 1e-9) ve = 1e-9;
    std::vector<double> lambdas = {ve / vu};
    build_lhs(lambdas);

    // 1. Data Term
    Eigen::VectorXd rhs_data(p + q);
    rhs_data << Xty, Zty;
    Eigen::VectorXd sol_data = solve_lhs(rhs_data);
    Eigen::VectorXd beta_hat = sol_data.head(p);
    Eigen::VectorXd u_hat = sol_data.tail(q);

    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_hat, *recs_ptr, *fd_ptr);
    Eigen::VectorXd e_hat = y - y_hat;

    double data_u = u_hat.dot(Ainv_ptr->operator*(u_hat)) / (vu * vu);
    double data_e = e_hat.squaredNorm() / (ve * ve);

    // 2. Trace Term using ApplyP
    double tr_P = 0.0;
    int N = (int)z_samples.size();

    #pragma omp parallel reduction(+:tr_P)
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd Pz = ApplyP(z_samples[i]);
            tr_P += z_samples[i].dot(Pz);
        }
    }
    tr_P /= N;

    // Analytical relation for single random component: tr(P V) = n - p
    // V = Z A Z^T * vu + I * ve
    // tr(P Z A Z^T) * vu + tr(P) * ve = n - p
    // tr(P Z A Z^T) = (n - p - tr_P * ve) / vu
    double tr_u = (n - p - tr_P * ve) / vu;
    double tr_e = tr_P;

    double grad_u = -0.5 * tr_u + 0.5 * data_u;
    double grad_e = -0.5 * tr_e + 0.5 * data_e;

    return Eigen::Vector2d(grad_u, grad_e);
}

AI_REML::AI_Terms AI_REML::compute_AI_terms_fdiff(int n_samples) {
    AI_Terms terms;
    int n = (int)y.size();

    // Generate fixed random vectors for Hutchinson trace estimation in N-space
    std::vector<Eigen::VectorXd> z_samples(n_samples);
    unsigned long base_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    RNG rng(base_seed);
    for (int i = 0; i < n_samples; ++i) {
        z_samples[i] = Eigen::VectorXd(n);
        rng.fill_rademacher(z_samples[i]);
    }

    double vu = vars_u[0];
    double ve = var_e;

    if (!std::isfinite(vu) || vu < 1e-9) vu = 1e-9;
    if (!std::isfinite(ve) || ve < 1e-9) ve = 1e-9;

    // Base Gradient
    Eigen::Vector2d g0 = compute_stochastic_gradient(vu, ve, z_samples);
    terms.grad_u = g0(0);
    terms.grad_e = g0(1);

    // Finite Difference Perturbations (e.g., 1e-4 relative)
    double delta_u = vu * 1e-4;
    double delta_e = ve * 1e-4;
    if (delta_u < 1e-8) delta_u = 1e-8;
    if (delta_e < 1e-8) delta_e = 1e-8;

    // Perturb Vu
    Eigen::Vector2d gu = compute_stochastic_gradient(vu + delta_u, ve, z_samples);
    // Perturb Ve
    Eigen::Vector2d ge = compute_stochastic_gradient(vu, ve + delta_e, z_samples);

    // Approximate negative Hessian (AI Matrix)
    double AI_uu = -(gu(0) - g0(0)) / delta_u;
    double AI_ue = -(gu(1) - g0(1)) / delta_u;
    double AI_eu = -(ge(0) - g0(0)) / delta_e;
    double AI_ee = -(ge(1) - g0(1)) / delta_e;

    // Symmetrize
    double AI_ue_sym = 0.5 * (AI_ue + AI_eu);

    terms.AI_uu = AI_uu;
    terms.AI_ue = AI_ue_sym;
    terms.AI_ee = AI_ee;

    // Restore LHS to base variances
    std::vector<double> lambdas = {ve / std::max(vu, 1e-9)};
    build_lhs(lambdas);

    return terms;
}

AI_REML::AI_Terms AI_REML::compute_AI_terms(int n_samples) {
    AI_Terms terms;
    int n = (int)y.size();
    if (!mme_builder) throw std::runtime_error("compute_AI_terms requires initialized mme_builder");

    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();
    const AbstractMatrix* Ainv_ptr = components.empty() ? nullptr : components[0].Qinv;

    double sum_uu = 0.0;
    double sum_ue = 0.0;
    double sum_ee = 0.0;
    double sum_tr_Ainv_Cuu = 0.0; // Needed for Gradient

    // ... comments ...

    // Loop 1: R^q samples (Gradient & AI_uu, AI_ue)
    bool used_ainv_sampling = false;
    #pragma omp parallel reduction(+:sum_tr_Ainv_Cuu, sum_uu, sum_ue)
    {
        unsigned long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        #ifdef _OPENMP
        seed += omp_get_thread_num() * 112233;
        #endif
        RNG local_rng(seed);

        #pragma omp for
        for (int k=0; k<n_samples; ++k) {
            Eigen::VectorXd z(q);
            local_rng.fill_rademacher(z);

            // 1. Gradient term: tr(Ainv Cuu)
            Eigen::VectorXd rhs1(p + q);
            rhs1.setZero();
            rhs1.tail(q) = z;
            Eigen::VectorXd sol1 = solve_lhs(rhs1);
            Eigen::VectorXd u_s = sol1.tail(q);
            sum_tr_Ainv_Cuu += z.dot(Ainv_ptr->operator*(u_s));

            // 2. AI Terms: AI_uu, AI_ue
            // We need samples u0 ~ N(0, G) to correctly estimate trace with G
            Eigen::VectorXd u0(q);
            if (Ainv_factorized) {
                if (!used_ainv_sampling && k==0) {
                     #pragma omp critical
                     used_ainv_sampling = true;
                }
                if (use_dense_Ainv) {
                    // Dense LDLT: Ainv = P L D L^T P^T
                    // u0 = P L^{-T} D^{-1/2} z

                    Eigen::VectorXd D = Ainv_dense_chol->vectorD();
                    Eigen::VectorXd z_scaled = z;
                    for(int i=0; i<D.size(); ++i) {
                        if(D(i) <= 0) {
                            z_scaled(i) = 0;
                        } else {
                            z_scaled(i) /= std::sqrt(D(i));
                        }
                    }

                    // matrixU() is L^T. solve() computes L^{-T} z_scaled
                    Eigen::VectorXd tmp = Ainv_dense_chol->matrixU().solve(z_scaled);

                    // Apply Permutation P
                    u0 = Ainv_dense_chol->transpositionsP() * tmp;
                } else {
                    // Sparse LDLT: Ainv = P U D U^T P^T
                    // We want u0 ~ N(0, Ainv^-1) = N(0, P U^-T D^-1 U^-1 P^T)
                    // Let u0 = P U^-T D^-1/2 z
                    // Step 1: Scale z by D^-1/2
                    Eigen::VectorXd D = Ainv_sparse_chol->vectorD();
                    Eigen::VectorXd z_scaled = z;
                    bool possible_indefinite = false;
                    for(int i=0; i<D.size(); ++i) {
                        if(D(i) <= 0) {
                            z_scaled(i) = 0; // Handle indefinite case safely (or throw?)
                            possible_indefinite = true;
                        } else {
                            z_scaled(i) /= std::sqrt(D(i));
                        }
                    }
                    if(possible_indefinite && k==0 && config.verbose) {
                        std::cerr << "Warning: Indefinite Ainv detected during sampling!" << std::endl;
                    }

                    // Step 2: Solve U^T y = z_scaled (Wait, matrixU is U)
                    // We need U^-T? No, A = (P U D U^T P^T)^-1 = P U^-T D^-1 U^-1 P^T
                    // Cov(u) = P U^-T D^-1/2 D^-1/2 U^-1 P^T
                    // So u = P U^-T D^-1/2 z
                    // matrixU() returns U.
                    // We need to apply U^-T.
                    // Eigen SimplicialLDLT doesn't provide solveTranspose() for matrixU easily?
                    // Actually matrixU() returns a triangular view.
                    // solve() does U^-1.
                    // We need U^-T.
                    // Wait, SimplicialLDLT: A = P L D L^T P^T (Lower) or P U^T D U P^T (Upper)?
                    // Eigen doc: "SimplicialLDLT ... decomposes A = P L D L^* P^T"
                    // It uses Lower by default? Or depends on template?
                    // The default SimplicialLDLT<SparseMatrix> may use the lower triangle.
                    // If Lower: Ainv = P L D L^T P^T.
                    // A = P L^-T D^-1 L^-1 P^T.
                    // u = P L^-T D^{-1/2} z.
                    // matrixL() returns L.
                    // L.transpose().solve(z_scaled) gives L^-T z_scaled.

                    // Let's check if we have matrixL or matrixU.
                    // The previous code used matrixU().
                    // If the factorization computed L, matrixU() might be L^T?
                    // "matrixU() returns the upper triangular factor U such that A = P L D U P^-1" No.
                    // For SimplicialLDLT, matrixL() returns L. matrixU() returns L^T (adjoint).

                    // So matrixU() is L^T.
                    // We want L^-T.
                    // (L^T)^-1 = L^-T.
                    // So matrixU().solve() computes (L^T)^-1 = L^-T.
                    // So `matrixU().solve(z_scaled)` IS CORRECT for `L^-T`.

                    Eigen::VectorXd tmp = Ainv_sparse_chol->matrixU().solve(z_scaled);
                    u0 = Ainv_sparse_chol->permutationP().inverse() * tmp;
                }
            } else {
                u0 = z; // Fallback G=I
            }

            // v = Z u0
            Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(p);
            Eigen::VectorXd v = mme_builder->mult_design(zero_beta, u0, *recs_ptr, *fd_ptr);

            Eigen::VectorXd Pv = ApplyP(v);

            sum_ue += Pv.squaredNorm();

            Eigen::VectorXd full = mme_builder->mult_transpose_design(Pv, *recs_ptr, *fd_ptr);
            Eigen::VectorXd ZtPv = full.tail(q);

            if (Ainv_factorized) {
                if (use_dense_Ainv) {
                     Eigen::VectorXd Gk = Ainv_dense_chol->solve(ZtPv);
                     sum_uu += ZtPv.dot(Gk);
                } else {
                     Eigen::VectorXd Gk = Ainv_sparse_chol->solve(ZtPv);
                     sum_uu += ZtPv.dot(Gk);
                }
            } else {
                sum_uu += ZtPv.squaredNorm();
            }
        }
    }

     if (config.verbose) {
         // std::cout << "    [Trace Est] Used Ainv Sampling: " << (used_ainv_sampling ? "Yes" : "No (G=I assumed!)") << std::endl;
     }

     // Loop 2: R^n samples (for AI_ee)
     #pragma omp parallel reduction(+:sum_ee)
     {
         unsigned long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
         #ifdef _OPENMP
         seed += omp_get_thread_num() * 445566;
         #endif
         RNG local_rng(seed);

         #pragma omp for
         for (int k=0; k<n_samples; ++k) {
             Eigen::VectorXd z(n);
             local_rng.fill_rademacher(z);

             Eigen::VectorXd Pz = ApplyP(z);

             // AI_ee contribution: Pz' * Pz
             sum_ee += Pz.squaredNorm();
         }
     }

     // Scale by 0.5 and average
     terms.AI_uu = 0.5 * (sum_uu / n_samples);
     terms.AI_ue = 0.5 * (sum_ue / n_samples);
     terms.AI_ee = 0.5 * (sum_ee / n_samples);

     // Gradients
    double tr_Ainv_Cuu = sum_tr_Ainv_Cuu / n_samples;

     // Gradient identities:
     // dL/du = -0.5 * tr(P Z Z') + 0.5 * u' G^-2 u
     // dL/de = -0.5 * tr(P) + 0.5 * e' R^-2 e

     // Correct formula for score (Gilmour et al. 1995):
     // S_u = -0.5 tr(Z' P Z) + 0.5 y' P Z Z' P y
     // S_e = -0.5 tr(P) + 0.5 y' P P y

     // We have data parts calculated outside.
     // We need to return the negative trace parts here.

     // tr(Z' P Z) = 1/var_u * (q - lambda * tr(Ainv * Cuu))
    // tr(P) = 1/var_e * (n - p - (q - lambda * tr(Ainv * Cuu)))

    double var_u = (!vars_u.empty()) ? vars_u[0] : 1.0;
    double effective_dim = (var_e/var_u) * tr_Ainv_Cuu;
    if (effective_dim > (double)q - 1e-6) effective_dim = (double)q - 1e-6;

    double tr_PZZ = (q - effective_dim) / var_u;
        double tr_P = (n - p - q + effective_dim) / var_e;

        if (config.verbose) {
             // std::cout << "    [Gradient Parts] tr(Ainv*Cuu)=" << tr_Ainv_Cuu
             //          << " EffDim=" << effective_dim
             //          << " tr(PZZ)=" << tr_PZZ << " tr(P)=" << tr_P << std::endl;
        }

        terms.grad_u = -0.5 * tr_PZZ;
        terms.grad_e = -0.5 * tr_P;

     // AI Matrix components:
     // AI_uu = 0.5 * tr(P Z Z' P Z Z')
     // AI_ue = 0.5 * tr(P Z Z' P)
     // AI_ee = 0.5 * tr(P P)

     // Our stochastic estimates:
     // Sample z in R^q:
     // E[ |Z' P Z z|^2 ] = tr( (Z' P Z)' (Z' P Z) ) = tr( Z' P Z Z' P Z ) = tr( P Z Z' P Z Z' ).
     // This matches 2 * AI_uu.
     // E[ |P Z z|^2 ] = tr( (P Z)' (P Z) ) = tr( Z' P P Z ) = tr( P Z Z' P ).
     // This matches 2 * AI_ue.

     // Sample z in R^n:
     // E[ |P z|^2 ] = tr( P' P ) = tr( P P ).
     // This matches 2 * AI_ee.

     // So our stochastic sums (sum_uu, sum_ue, sum_ee) are sum of squared norms.
     // sum_uu ~ n_samples * 2 * AI_uu.
     // So AI_uu = 0.5 * (sum_uu / n_samples). Correct.

     return terms;
 }

// ============================================================================
// Multi-component AI-REML (Path A): supports c >= 1 random components + residual.
// Estimates gradient and (c+1)x(c+1) AI matrix via Hutchinson stochastic trace,
// using ApplyP (MME solve) for implicit P action and per-component Qinv factorization.
// Returns terms.grad_vec (trace part only) and terms.AI_mat (full AI matrix).
// ============================================================================
AI_REML::AI_Terms AI_REML::compute_AI_terms_multi(int n_samples) {
    AI_Terms terms;
    int c = (int)vars_u.size();
    int num_params = c + 1;
    int n = (int)y.size();
    if (!mme_builder) throw std::runtime_error("compute_AI_terms_multi requires mme_builder");
    int p = mme_builder->get_p();
    int q_total = mme_builder->get_q_total();
    const auto& qs = mme_builder->get_qs();

    // Compute q_offsets for slicing Z'v into per-component blocks
    std::vector<int> q_offsets(c, 0);
    for (int k = 1; k < c; ++k) q_offsets[k] = q_offsets[k - 1] + qs[k - 1];

    // Factorize each Qinv_k locally (single-threaded setup).
    // For Pe (identity) or large matrices, factorization may be skipped; g_k = w_k fallback.
    std::vector<std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>> qinv_sparse(c);
    std::vector<std::unique_ptr<Eigen::LDLT<Eigen::MatrixXd>>> qinv_dense(c);
    std::vector<bool> use_dense_qinv(c, false);
    std::vector<bool> qinv_factorized(c, false);

    for (int k = 0; k < c; ++k) {
        const AbstractMatrix* Qinv_k = components[k].Qinv;
        if (!Qinv_k) continue;
        const SparseMatrixAdapter* sp = dynamic_cast<const SparseMatrixAdapter*>(Qinv_k);
        int rows = Qinv_k->rows();
        if (sp && rows > 0 && rows < 50000) {
            // Sparse factorization
            qinv_sparse[k] = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
            qinv_sparse[k]->compute(sp->getMatrix());
            if (qinv_sparse[k]->info() == Eigen::Success) {
                qinv_factorized[k] = true;
            } else if (rows < 8000) {
                // Fallback to dense
                use_dense_qinv[k] = true;
                Eigen::MatrixXd Qd = sp->getMatrix();
                qinv_dense[k] = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>();
                qinv_dense[k]->compute(Qd);
                qinv_factorized[k] = (qinv_dense[k]->info() == Eigen::Success);
            }
        } else if (rows > 0 && rows < 8000) {
            // Try dense directly for small matrices (e.g., dense GRM inverse)
            // Use AbstractMatrix::operator* to build dense column by column
            Eigen::MatrixXd Qd = Eigen::MatrixXd::Zero(rows, rows);
            bool built = false;
            // Attempt sparse triplet extraction via visit_triplets
            Qinv_k->visit_triplets([&](int r, int col, double v) {
                Qd(r, col) = v;
                built = true;
            });
            if (built || rows < 100) {
                use_dense_qinv[k] = true;
                qinv_dense[k] = std::make_unique<Eigen::LDLT<Eigen::MatrixXd>>();
                qinv_dense[k]->compute(Qd);
                qinv_factorized[k] = (qinv_dense[k]->info() == Eigen::Success);
            }
        }
        // If not factorized, solve_qinv falls back to g_k = w_k (assumes G_k = I)
    }

    // Helper: solve Qinv_k g = w  =>  g = G_k w = Qinv_k^{-1} w
    auto solve_qinv = [&](int k, const Eigen::VectorXd& w) -> Eigen::VectorXd {
        if (!qinv_factorized[k]) return w; // Fallback: G_k = I
        if (use_dense_qinv[k]) return qinv_dense[k]->solve(w);
        return qinv_sparse[k]->solve(w);
    };

    // Stochastic accumulators (trace parts)
    Eigen::VectorXd grad_trace = Eigen::VectorXd::Zero(num_params);
    Eigen::MatrixXd AI_acc = Eigen::MatrixXd::Zero(num_params, num_params);

    Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(p);

    // Use a fixed seed for reproducibility within a single solve() call
    unsigned long base_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    #pragma omp parallel
    {
        Eigen::VectorXd local_grad_trace = Eigen::VectorXd::Zero(num_params);
        Eigen::MatrixXd local_AI = Eigen::MatrixXd::Zero(num_params, num_params);

        unsigned long seed = base_seed;
        #ifdef _OPENMP
        seed += omp_get_thread_num() * 112233;
        #endif
        RNG local_rng(seed);

        #pragma omp for schedule(dynamic, 8)
        for (int s = 0; s < n_samples; ++s) {
            // Sample z ~ Rademacher(n)
            Eigen::VectorXd z(n);
            local_rng.fill_rademacher(z);

            // 1. Pz = P z (1 MME solve via ApplyP)
            Eigen::VectorXd Pz = ApplyP(z);

            // 2. Z' z  =>  [X'z; Z_1'z; ...; Z_c'z]
            Eigen::VectorXd Ztz_full = mme_builder->mult_transpose_design(z, *recs_ptr, *fd_ptr);

            // 3. For each k: v_k = dV_k z = Z_k * (G_k (Z_k' z))
            //    Note: dV_k/d(sigma2_k) = Z_k G_k Z_k' (does NOT include sigma2_k).
            //    Pv_k = P v_k (1 MME solve per component)
            std::vector<Eigen::VectorXd> Pv(c);
            for (int k = 0; k < c; ++k) {
                Eigen::VectorXd w_k = Ztz_full.segment(p + q_offsets[k], qs[k]);
                Eigen::VectorXd g_k = solve_qinv(k, w_k); // G_k w_k = Qinv_k^{-1} w_k
                // u_full = [0; ...; g_k at offset q_offsets[k]; ...; 0]
                Eigen::VectorXd u_full = Eigen::VectorXd::Zero(q_total);
                u_full.segment(q_offsets[k], qs[k]) = g_k;
                // v_k = Z_k g_k  (dV_k z, no sigma2_k factor)
                Eigen::VectorXd v_k = mme_builder->mult_design(zero_beta, u_full, *recs_ptr, *fd_ptr);
                // Pv_k = P v_k
                Pv[k] = ApplyP(v_k);

                // Gradient trace part: tr(P dV_k) = E[(Pz)' (dV_k z)] = (Pz)' v_k
                local_grad_trace(k) += Pz.dot(v_k);
            }
            // Residual trace: tr(P) = E[(Pz)' z]
            local_grad_trace(c) += Pz.dot(z);

            // 4. AI matrix: AI(i,j) = 0.5 * tr(P dV_i P dV_j) = 0.5 * E[(Pv_i)' (Pv_j)]
            for (int i = 0; i < c; ++i) {
                for (int j = i; j < c; ++j) {
                    local_AI(i, j) += Pv[i].dot(Pv[j]);
                }
                // AI(i, e) = 0.5 * tr(P dV_i P) = 0.5 * E[(Pv_i)' (Pz)]
                local_AI(i, c) += Pv[i].dot(Pz);
            }
            // AI(e, e) = 0.5 * tr(P P) = 0.5 * E[(Pz)' (Pz)]
            local_AI(c, c) += Pz.dot(Pz);
        }

        #pragma omp critical
        {
            grad_trace += local_grad_trace;
            AI_acc += local_AI;
        }
    }

    // Average over samples
    grad_trace /= (double)n_samples;
    AI_acc /= (double)n_samples;

    // Scale AI by 0.5 (factor in AI = 0.5 * tr(...))
    AI_acc *= 0.5;

    // Symmetrize AI matrix
    for (int i = 0; i < num_params; ++i) {
        for (int j = i + 1; j < num_params; ++j) {
            AI_acc(j, i) = AI_acc(i, j);
        }
    }

    terms.grad_vec = grad_trace;
    terms.AI_mat = AI_acc;
    return terms;
}

// ============================================================================
// Multi-component Data-Driven AI matrix (Gilmour et al. 1995).
// Exact (non-stochastic): uses current MME solution to form v_i = dV_i * Py,
// then AI_ij = 0.5 * v_i' P v_j. Requires (c+1) extra MME solves.
// ============================================================================
AI_REML::AI_Terms AI_REML::compute_AI_terms_multi_datadriven() {
    AI_Terms terms;
    int c = (int)vars_u.size();
    int num_params = c + 1;
    int n = (int)y.size();
    if (!mme_builder) throw std::runtime_error("compute_AI_terms_multi_datadriven requires mme_builder");
    int p = mme_builder->get_p();
    int q_total = mme_builder->get_q_total();
    const auto& qs = mme_builder->get_qs();

    std::vector<int> q_offsets(c, 0);
    for (int k = 1; k < c; ++k) q_offsets[k] = q_offsets[k - 1] + qs[k - 1];

    // LHS is already built and factorized by the caller (run_aireml_step_multi step 1).
    // We need the current solution (beta_hat, u_all, e_hat) to form v_i.
    // Re-solve to get the solution (caller already did this, but we redo for self-containedness).
    Eigen::VectorXd rhs(p + q_total);
    rhs << Xty, Zty;
    Eigen::VectorXd sol = solve_lhs(rhs);
    if (!sol.allFinite()) {
        terms.AI_mat = Eigen::MatrixXd::Zero(num_params, num_params);
        return terms;
    }
    Eigen::VectorXd beta_hat = sol.head(p);
    Eigen::VectorXd u_all = sol.tail(q_total);
    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_all, *recs_ptr, *fd_ptr);
    Eigen::VectorXd e_hat = y - y_hat;

    // w = P y = e / sigma_e^2
    Eigen::VectorXd w = e_hat / std::max(var_e, 1e-12);

    // Form v_i = (dV/dtheta_i) * Py for each parameter.
    // From MME: u_k = sigma2_k * K_k * Z_k' * (Py) = sigma2_k * K_k * Z_k' * w,
    //   so (dV/dsigma2_k) * Py = Z_k * K_k * Z_k' * w = Z_k * (u_k / sigma2_k).
    // For residual: (dV/dsigma2_e) * Py = I * w = w.
    std::vector<Eigen::VectorXd> v(num_params);
    Eigen::VectorXd beta_zero = Eigen::VectorXd::Zero(p);
    for (int k = 0; k < c; ++k) {
        Eigen::VectorXd u_k = u_all.segment(q_offsets[k], qs[k]);
        double sigma2_k = std::max(vars_u[k], 1e-12);
        Eigen::VectorXd u_full = Eigen::VectorXd::Zero(q_total);
        u_full.segment(q_offsets[k], qs[k]) = u_k / sigma2_k;
        v[k] = mme_builder->mult_design(beta_zero, u_full, *recs_ptr, *fd_ptr);
    }
    v[c] = w;

    // Solve P v_i for each i: MME solve with RHS [X'v_i; Z'v_i]
    // Then Pv_i = (v_i - X*beta* - Z*u*) / sigma_e^2
    std::vector<Eigen::VectorXd> Pv(num_params);
    for (int i = 0; i < num_params; ++i) {
        Eigen::VectorXd rhs_i = mme_builder->mult_transpose_design(v[i], *recs_ptr, *fd_ptr);
        Eigen::VectorXd sol_i = solve_lhs(rhs_i);
        if (!sol_i.allFinite()) {
            terms.AI_mat = Eigen::MatrixXd::Zero(num_params, num_params);
            return terms;
        }
        Eigen::VectorXd b_i = sol_i.head(p);
        Eigen::VectorXd u_i = sol_i.tail(q_total);
        Eigen::VectorXd yhat_i = mme_builder->mult_design(b_i, u_i, *recs_ptr, *fd_ptr);
        Pv[i] = (v[i] - yhat_i) / std::max(var_e, 1e-12);
    }

    // AI matrix: AI_ij = 0.5 * v_i' P v_j
    Eigen::MatrixXd AI_mat = Eigen::MatrixXd::Zero(num_params, num_params);
    for (int i = 0; i < num_params; ++i) {
        for (int j = i; j < num_params; ++j) {
            AI_mat(i, j) = 0.5 * v[i].dot(Pv[j]);
            AI_mat(j, i) = AI_mat(i, j);
        }
    }

    terms.AI_mat = AI_mat;
    // Gradient not computed here (caller uses analytical gradient traces)
    terms.grad_vec = Eigen::VectorXd::Zero(num_params);
    return terms;
}

// ============================================================================
// Multi-component AI-REML step (Path A).
// Handles c >= 1 random components + residual. Returns false on numerical
// failure so the caller (run_ai_iteration) can fall back to an EM step.
// ============================================================================
bool AI_REML::run_aireml_step_multi(int iter) {
    int c = (int)vars_u.size();
    if (c < 1) return false;
    int num_params = c + 1;

    // Safeguard against NaN/Inf variances
    for (int k = 0; k < c; ++k) {
        if (!std::isfinite(vars_u[k]) || vars_u[k] <= 0.0) vars_u[k] = 1e-6;
    }
    if (!std::isfinite(var_e) || var_e <= 0.0) var_e = 1e-6;

    int n = (int)y.size();
    if (!mme_builder) throw std::runtime_error("run_aireml_step_multi requires mme_builder");
    int p = mme_builder->get_p();
    int q_total = mme_builder->get_q_total();
    const auto& qs = mme_builder->get_qs();

    std::vector<int> q_offsets(c, 0);
    for (int k = 1; k < c; ++k) q_offsets[k] = q_offsets[k - 1] + qs[k - 1];

    // 1. Build LHS with current lambdas
    std::vector<double> lambdas(c);
    for (int k = 0; k < c; ++k) lambdas[k] = var_e / std::max(vars_u[k], 1e-9);
    build_lhs(lambdas);

    // 2. Solve MME for data: C [beta; u] = [X'y; Z'y]
    Eigen::VectorXd rhs(p + q_total);
    rhs << Xty, Zty;
    Eigen::VectorXd sol = solve_lhs(rhs);
    if (!sol.allFinite()) return false;

    Eigen::VectorXd beta_hat = sol.head(p);
    Eigen::VectorXd u_all = sol.tail(q_total);

    // 3. Residuals: e = y - X*beta - Z*u
    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_all, *recs_ptr, *fd_ptr);
    Eigen::VectorXd e_hat = y - y_hat;
    if (!e_hat.allFinite()) return false;

    // 4. Data terms (Gilmour et al. 1995 data-driven score)
    //    data_k = u_k' Qinv_k u_k / sigma2_k^2  (random component k)
    //    data_e = e'e / sigma2_e^2              (residual)
    Eigen::VectorXd data_terms = Eigen::VectorXd::Zero(num_params);
    std::vector<double> uQinvu(c, 0.0);
    for (int k = 0; k < c; ++k) {
        Eigen::VectorXd u_k = u_all.segment(q_offsets[k], qs[k]);
        Eigen::VectorXd Qinv_u_k = components[k].Qinv->operator*(u_k);
        uQinvu[k] = u_k.dot(Qinv_u_k);
        data_terms(k) = uQinvu[k] / (vars_u[k] * vars_u[k]);
    }
    double ete = e_hat.dot(e_hat);
    data_terms(c) = ete / (var_e * var_e);

    // 5. Analytical gradient traces via MME inverse (well-conditioned).
    int n_samp_trace = std::max(30, config.mc_samples);
    std::vector<double> tr_Qinv_Cuu(c, 0.0);
    double sum_eff_dim = 0.0;
    for (int k = 0; k < c; ++k) {
        tr_Qinv_Cuu[k] = estimate_trace_Qinv_Cuu(k, n_samp_trace);
        double lambda_k = var_e / std::max(vars_u[k], 1e-9);
        sum_eff_dim += lambda_k * tr_Qinv_Cuu[k];
    }

    Eigen::VectorXd grad_trace = Eigen::VectorXd::Zero(num_params);
    for (int k = 0; k < c; ++k) {
        grad_trace(k) = (qs[k] - var_e * tr_Qinv_Cuu[k]) / std::max(vars_u[k], 1e-9);
    }
    grad_trace(c) = ((double)n - (double)p - (double)q_total + sum_eff_dim) / std::max(var_e, 1e-9);

    // 6. Assemble gradient: grad(k) = -0.5 * trace(k) + 0.5 * data(k)
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_params);
    for (int k = 0; k < num_params; ++k) {
        grad(k) = -0.5 * grad_trace(k) + 0.5 * data_terms(k);
    }

    // 7. Compute EM step (always reliable, used as fallback / reference).
    //    EM update (same formula as standard EM path in run_ai_iteration):
    //    var_u_k_new = (u'G^{-1}u + var_e * tr_Qinv_Cuu_k) / q_k
    //    var_e_new   = (e'e + var_e * tr_C_WdW) / n
    //    where tr_C_WdW = (p + q_total) - sum(lambda_k * tr_Qinv_Cuu_k)
    Eigen::VectorXd em_delta = Eigen::VectorXd::Zero(num_params);
    for (int k = 0; k < c; ++k) {
        double vu_new = (uQinvu[k] + var_e * tr_Qinv_Cuu[k]) / std::max((double)qs[k], 1.0);
        em_delta(k) = vu_new - vars_u[k];
    }
    {
        double tr_C_WdW = (double)(p + q_total) - sum_eff_dim;
        double ve_new = (ete + var_e * tr_C_WdW) / std::max((double)n, 1.0);
        em_delta(c) = ve_new - var_e;
    }

    // 8. AI matrix is NOT computed here (would require c+1 extra MME solves
    //    per iteration). It is computed once at convergence in calculate_SE()
    //    using the data-driven method of Gilmour et al. (1995). This avoids
    //    O(iter * (c+1)) wasted MME solves since the AI matrix is only needed
    //    for standard errors, not for the EM-based variance updates.
    //    On near-confounded likelihood ridges (small samples with multiple
    //    random effects), the AI Newton direction can be unreliable even when
    //    its global cosine with the EM direction is positive. We therefore use
    //    EM steps for robust monotonic convergence and rely on the AI matrix
    //    only for standard errors (as in ASReml's EM mode).

    // 9. Use EM step for variance component updates (monotonic, reliable).

    // 10. Step size control: clamp to [0.1x, 10x] of previous variance.
    const double y_mean_local = y.size() > 0 ? y.mean() : 0.0;
    const double y_var_local = y.size() > 1
        ? std::max(1e-12, (y.array() - y_mean_local).square().sum() / (double)(y.size() - 1))
        : std::max(1e-12, y.squaredNorm());
    const double vc_floor = 1e-12;
    const double vc_ceil = std::max(1.0, y_var_local * 1e6);
    auto stabilize = [&](double proposed, double previous) -> double {
        if (!std::isfinite(proposed) || proposed <= 0.0) return std::max(previous, vc_floor);
        double lower = vc_floor;
        double upper = vc_ceil;
        if (std::isfinite(previous) && previous > 0.0) {
            lower = std::max(lower, previous * 0.1);
            upper = std::min(upper, previous * 10.0);
        }
        if (lower > upper) lower = vc_floor;
        return std::min(std::max(proposed, lower), upper);
    };

    // EM step: delta is the change, new_var = var + delta = em_target
    for (int k = 0; k < c; ++k) {
        double vu_new = vars_u[k] + em_delta(k);
        vars_u[k] = stabilize(vu_new, vars_u[k]);
    }
    double ve_new = var_e + em_delta(c);
    var_e = stabilize(ve_new, var_e);

    // Convergence metric: sum of absolute changes (consistent with standard EM path)
    {
        double diff = std::abs(em_delta(c));
        for (int k = 0; k < c; ++k) diff += std::abs(em_delta(k));
        last_diff = diff;
    }

    if (config.verbose && config.log_stream) {
        *config.log_stream << "  [AI-Multi Iter " << iter << "] step=EM"
                          << " Ve=" << var_e;
        for (int k = 0; k < c; ++k) *config.log_stream << " Vu[" << k << "]=" << vars_u[k];
        *config.log_stream << " (Diff=" << last_diff << ")" << std::endl;
    }

    if (getenv("CS_AI_MULTI_DEBUG")) {
        std::cerr << "[AI-Multi iter=" << iter << "] step=EM"
                  << " Diff=" << last_diff
                  << " Ve=" << var_e;
        for (int k = 0; k < c; ++k) std::cerr << " Vu[" << k << "]=" << vars_u[k];
        std::cerr << std::endl;
    }

    return true;
}

bool AI_REML::run_aireml_step(int iter) {
    if (vars_u.size() != 1) return false;

    // Safeguard against NaN/Inf variances entering AI step
    if (!std::isfinite(vars_u[0]) || vars_u[0] <= 0.0) vars_u[0] = 1e-6;
    if (!std::isfinite(var_e) || var_e <= 0.0) var_e = 1e-6;

    double var_u = vars_u[0];
    const AbstractMatrix* Ainv_ptr = components[0].Qinv;

    std::vector<double> lambdas = {var_e / std::max(var_u, 1e-9)};
    int n = (int)y.size();
    if (!mme_builder) throw std::runtime_error("run_aireml_step requires initialized mme_builder");
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();

    // 1. Build LHS
    build_lhs(lambdas);

    // 2. Solve Data
    if (false) std::cout << "  [Iter " << iter << "] AI-REML Step... Calling solve_lhs for data." << std::endl;
    Eigen::VectorXd rhs(p + q);
    rhs << Xty, Zty;
    Eigen::VectorXd sol = solve_lhs(rhs);
    if (false) std::cout << "  [Iter " << iter << "] solve_lhs returned. Computing residuals..." << std::endl;
    Eigen::VectorXd u_hat = sol.tail(q);
    Eigen::VectorXd beta_hat = sol.head(p);

    Eigen::VectorXd e_hat;
    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_hat, *recs_ptr, *fd_ptr);
    e_hat = y - y_hat;
    if (false) std::cout << "  [Iter " << iter << "] e_hat norm = " << e_hat.norm() << std::endl;

    // 3. Compute Gradients (Data part)
    // dL/dVu = -0.5 tr(P Z G Z') + 0.5 y' P Z G Z' P y
    // dL/dVe = -0.5 tr(P) + 0.5 y' P P y

    // y' P Z G Z' P y = u_hat' Ainv u_hat / var_u^2 (since G = Ainv^-1) -> NO!
    // G = A * var_u. G^-1 = Ainv / var_u.
    // u' G^-1 u = u' (Ainv / var_u) u = u' Ainv u / var_u.
    // BUT we need y' P Z A Z' P y = u' Ainv u / var_u^4 ?
    // Derivation: y' P Z = u' Ainv / var_u^2.
    // Term = (u' Ainv / var_u^2) A (Ainv u / var_u^2) = u' Ainv u / var_u^4.

    double data_term_u = u_hat.dot(components[0].Qinv->operator*(u_hat)) / (var_u * var_u);
    double data_term_e = e_hat.squaredNorm() / (var_e * var_e);

    // 4. Compute Stochastic Terms
    AI_Terms terms;

    // Check if we can use Exact Trace (Sparse Direct Solver Available)
    // Enabled Exact/Takahashi path for speed on sparse matrices.
    // MODIFIED: We now allow Data-Driven AI Matrix (Gilmour et al.) even if we don't have Takahashi inverse subset.
    // The Data-Driven AI Matrix only requires solving for specific RHS vectors, which we can do with ANY solver (Direct or Iterative).
    // The Gradient term `tr_Ainv_Cuu` still requires Trace(Ainv * Cuu).
    // For that, we EITHER need Takahashi (sparse_direct_solver) OR Stochastic estimation.
    // However, the AI Matrix itself is the most critical part for convergence speed.
    // So we will use Data-Driven AI Matrix + Stochastic Trace for Gradient if Direct Solver is not available.
    // If Direct Solver IS available, we use Exact Trace + Data-Driven AI Matrix (Fully Exact).

    // Force Stochastic Trace for validation of user's query
    bool solver_supports_exact = (sparse_direct_solver && sparse_direct_solver->info() == Eigen::Success);

    // Apply trace_mode policy (IASBLUP-style unified decision)
    // "auto" -> use solver capability to decide
    // "exact" -> force exact (requires sparse/dense solver)
    // "fdiff" -> force finite difference
    // "hutch" -> force Hutchinson stochastic
    // "slq"   -> force SLQ
    std::string tm = config.trace_mode;
    std::transform(tm.begin(), tm.end(), tm.begin(), ::tolower);

    bool use_exact_trace;
    if (tm == "exact") {
        use_exact_trace = true;
        if (!solver_supports_exact) {
            LOG_WARN("trace_mode=exact requested but no direct solver available, falling back to stochastic");
            use_exact_trace = false;
        }
    } else if (tm == "fdiff") {
        use_exact_trace = false;
        config.algorithm = VCEAlgorithm::Fdiff;  // Override to fdiff
    } else if (tm == "hutch" || tm == "slq") {
        use_exact_trace = false;
    } else {
        // "auto" (default): use exact if solver supports it
        use_exact_trace = solver_supports_exact;
    }

    // Enable Data-Driven AI Matrix if forced exact OR if we have sparse solver and smallish problem
    bool use_data_driven_ai = use_exact_trace || config.force_exact;
    if (config.algorithm == VCEAlgorithm::Fdiff) {
        use_data_driven_ai = false; // Force fallback to Fdiff
    }

    if (use_data_driven_ai) {
        if (false) std::cout << "    [Exact] Using Data-Driven AI Matrix (Gilmour et al. 1995)..." << std::endl;

        // --- A. Gradient Traces (Exact or Stochastic) ---
        double tr_Ainv_Cuu = 0.0;

        if (use_exact_trace) {
             // std::cout << "    [Trace] Using Exact Takahashi Inversion for Gradient..." << std::endl;
             if (false) std::cout << "    [Debug] Calling Takahashi with Ainv_ptr=" << Ainv_ptr << ", p=" << p << " (CHECK ME)" << std::endl;
             Eigen::SparseMatrix<double> Z_subset;
             // Revert to Standard Takahashi (assuming Pattern(L) covers Ainv due to 1e-50 fix)
             // FIX: Must pass Ainv_ptr and p to enable Extended Takahashi!
             compute_sparse_inverse_subset(*sparse_direct_solver, Z_subset, Ainv_ptr, p);
             Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P_mat = sparse_direct_solver->permutationP();

             double sum_diag = 0.0;
             double sum_lower = 0.0;
             double sum_upper = 0.0;

             Ainv_ptr->visit_triplets([&](int r, int c, double v) {
                  int r_perm = P_mat.indices()(p + r);
                  int c_perm = P_mat.indices()(p + c);

                  // Use a more robust lookup or assume it's there
                  // Z_subset is symmetric, so (r,c) or (c,r) works.
                  double val_C = Z_subset.coeff(r_perm, c_perm);
                  double val = v * val_C;

                  if (r == c) {
                      sum_diag += val;
                  } else if (r > c) {
                      sum_lower += val;
                  } else {
                      sum_upper += val;
                  }
             });

             // Robust Trace Logic: Handle Triangular or Symmetric storage
             // Prefer Lower Triangle if available (Standard for Pedigree/HIBLUP)
             if (std::abs(sum_lower) > 1e-20) {
                 tr_Ainv_Cuu = sum_diag + 2.0 * sum_lower;
                 // Debug Info
                 if (config.verbose) {
                      // Check for asymmetry/missing data
                      if (std::abs(sum_upper) > 1e-20 && std::abs(sum_lower - sum_upper) > 1e-6 * std::abs(sum_lower)) {
                           // std::cout << "    [Trace] Notice: Matrix storage asymmetry detected. Using Lower Triangle * 2." << std::endl;
                           std::cout << "            SumLower=" << sum_lower << ", SumUpper=" << sum_upper << std::endl;
                      } else if (std::abs(sum_upper) < 1e-20) {
                           // Pure Lower
                      }
                 }
             } else {
                 // Fallback to Upper if Lower is empty
                 tr_Ainv_Cuu = sum_diag + 2.0 * sum_upper;
                 if (config.verbose && std::abs(sum_upper) > 1e-20) {
                     // std::cout << "    [Trace] Notice: Matrix treated as Upper Triangular." << std::endl;
                 }
             }

             // Debugging Trace Discrepancy
             if (config.verbose) {
                 // std::cout << "    [Trace] Exact Trace Value (FIXED CHECK): " << tr_Ainv_Cuu << std::endl;

                 // Compute Stochastic Trace for comparison
                 double tr_stoch = estimate_trace_Qinv_Cuu(0, std::max(50, config.mc_samples));
                 // std::cout << "    [Trace] Stoch Trace: " << tr_stoch << " (Diff: " << std::abs(tr_Ainv_Cuu - tr_stoch) << ")" << std::endl;

                 if (std::abs(tr_Ainv_Cuu - tr_stoch) > std::abs(tr_stoch) * 0.5 + 10.0) { // Increased tolerance
                       // std::cout << "    [Trace] WARNING: Discrepancy persists! Checking logic..." << std::endl;
                       std::cout << "            SumDiag=" << sum_diag << ", SumLower=" << sum_lower << ", SumUpper=" << sum_upper << std::endl;
                       std::cout << "            Trusting Exact Trace (Takahashi) over noisy Stochastic Trace." << std::endl;
                       // DO NOT FALLBACK: tr_Ainv_Cuu = tr_stoch;
                  }
             }
        } else {
             // Stochastic Trace for Gradient
             // We need this if we are using PCG or if Sparse Factorization failed
             // But usually for PBLUP/SSGBLUP we rely on Sparse Factorization.
             // If we are here, likely we have a solver.
             // If not, we fall back to stochastic trace.
             // std::cout << "    [Trace] Using Stochastic Estimation for Gradient (" << config.mc_samples << " samples)..." << std::endl;
             tr_Ainv_Cuu = estimate_trace_Qinv_Cuu(0, config.mc_samples);
        }

        // --- B. Use Exact AI Matrix (Data-Driven / Average Information) ---
        if (false) std::cout << "    [Exact] Computing Data-Driven AI Matrix (2 extra solves)..." << std::endl;

        // 1. Prepare vectors
        // w = P y = e_hat / var_e
        Eigen::VectorXd w = e_hat / var_e;

        // v_u = Z A Z' P y = Z A Z' w
        // Derived as: v_u = Z u_hat / var_u
        Eigen::VectorXd beta_zero = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd Zu_hat = mme_builder->mult_design(beta_zero, u_hat, *recs_ptr, *fd_ptr);
        Eigen::VectorXd v_u = Zu_hat / var_u;

        // v_e = dV_e P y = I w = w
        Eigen::VectorXd v_e = w;

        // 2. Solve for P v_u
        // rhs_u = [X' v_u; Z' v_u]
        Eigen::VectorXd rhs_u_full = mme_builder->mult_transpose_design(v_u, *recs_ptr, *fd_ptr);
        // rhs_u_full /= var_e; // INCORRECT: LHS is already without var_e factor.
        Eigen::VectorXd sol_u = solve_lhs(rhs_u_full);
        Eigen::VectorXd b_u = sol_u.head(p);
        Eigen::VectorXd u_u = sol_u.tail(q);
        Eigen::VectorXd y_hat_u = mme_builder->mult_design(b_u, u_u, *recs_ptr, *fd_ptr);
        Eigen::VectorXd Pv_u = (v_u - y_hat_u) / var_e;

        // 3. Solve for P v_e
        // rhs_e = [X' v_e; Z' v_e]
        Eigen::VectorXd rhs_e_full = mme_builder->mult_transpose_design(v_e, *recs_ptr, *fd_ptr);
        // rhs_e_full /= var_e; // INCORRECT
        Eigen::VectorXd sol_e = solve_lhs(rhs_e_full);
        Eigen::VectorXd b_e = sol_e.head(p);
        Eigen::VectorXd u_e = sol_e.tail(q);
        Eigen::VectorXd y_hat_e = mme_builder->mult_design(b_e, u_e, *recs_ptr, *fd_ptr);
        Eigen::VectorXd Pv_e = (v_e - y_hat_e) / var_e;

        // 4. Compute AI terms
        // AI_ij = 0.5 * v_i' P v_j
        double val_AI_uu = 0.5 * v_u.dot(Pv_u);
        double val_AI_ee = 0.5 * v_e.dot(Pv_e);
        double val_AI_ue = 0.5 * v_u.dot(Pv_e);

        if (config.verbose) {
             // std::cout << "    [Debug AI] w norm: " << w.norm() << std::endl;
             // std::cout << "    [Debug AI] v_u norm: " << v_u.norm() << std::endl;
             // std::cout << "    [Debug AI] v_e norm: " << v_e.norm() << std::endl;
             // std::cout << "    [Debug AI] Pv_u norm: " << Pv_u.norm() << std::endl;
             // std::cout << "    [Debug AI] Pv_e norm: " << Pv_e.norm() << std::endl;
        }

        terms.AI_uu = val_AI_uu;
        terms.AI_ue = val_AI_ue;
        terms.AI_ee = val_AI_ee;

        // Correct Gradient Calculation for Vu (EM-Consistent)
        // dL/dVu = (1 / 2*Vu^2) * (u' Ainv u + tr(Ainv Cuu)*Ve - q*Vu)
        // This ensures the gradient is zero exactly when Vu = EM_Update.

        double u_Ainv_u = u_hat.dot(Ainv_ptr->operator*(u_hat));
        double term_tr = tr_Ainv_Cuu * var_e;

        // Gradient w.r.t Vu
        terms.grad_u = 0.5 * (u_Ainv_u + term_tr - q * var_u) / (var_u * var_u);

        // Correct Gradient Calculation for Ve (EM-Consistent)
        // dL/dVe = (1 / 2*Ve^2) * (e'e + Vu * (q - lambda*tr) - n*Ve) ?
        // Standard form: dL/dVe = -0.5 tr(P) + 0.5 y' P P y
        // tr(P) = (n - p - q + lambda*tr) / Ve
        // y' P P y = e'e / Ve^2
        // dL/dVe = 0.5 * (e'e/Ve^2 - (n-p-q + lambda*tr)/Ve)
        //        = 0.5/Ve^2 * (e'e - Ve*(n-p-q + lambda*tr))

        double lambda_tr = (var_e/var_u) * tr_Ainv_Cuu;
        if (lambda_tr > q - 1e-6) lambda_tr = q - 1e-6;

        // MODIFICATION: Use older version logic for gradient calculation to match v1.28
        // The older version used the standard AI-REML gradient form:
        // dL/dVe = -0.5 * tr(P) + 0.5 * y' P P y
        // tr(P) = (N - rank(X) - rank(Z) + tr(Ainv * Cuu) * lambda) / Ve ??
        // Actually, tr(P) = (n - p - q + lambda * tr(Ainv * Cuu)) / Ve is an approximation if G is not full rank.
        // Let's stick to the current logic as it seems mathematically sound, but ensure tr_Ainv_Cuu is correct.

        terms.grad_e = 0.5 * (e_hat.squaredNorm() - var_e * (n - p - q + lambda_tr)) / (var_e * var_e);

    } else {
        // Fallback for PCG / Dense
        // Use more samples for AI Matrix to ensure stability (min 300 for PBLUP/SSGBLUP)
        // For large systems, stochastic noise in AI matrix can lead to divergence.
        int n_samp = std::max(300, config.mc_samples * 2);
        // std::cout << "    [Stochastic] Estimating AI Matrix with " << n_samp << " samples..." << std::endl;
        if (config.algorithm == VCEAlgorithm::Fdiff) {
             terms = compute_AI_terms_fdiff(n_samp);
             use_data_driven_ai = true; // fdiff computes full gradient
        } else {
             terms = compute_AI_terms(n_samp);
        }
     }

     // 5. Build AI Matrix
    // Note: terms.grad_u already includes the data term component if calculated above.
    // If we used the EM-Consistent block, we computed the FULL gradient.
    // If we used the Stochastic block, terms.grad_u is only the TRACE part.
    // We need to handle this.

    double grad_u, grad_e;

    if (use_data_driven_ai) {
        grad_u = terms.grad_u; // Full gradient
        grad_e = terms.grad_e; // Full gradient
    } else {
        grad_u = terms.grad_u + 0.5 * data_term_u;
        grad_e = terms.grad_e + 0.5 * data_term_e;
    }
    Eigen::Matrix2d AI;
    AI(0,0) = terms.AI_uu;
    AI(0,1) = terms.AI_ue;
    AI(1,0) = terms.AI_ue;
    AI(1,1) = terms.AI_ee;
    AI = project_spd_2x2(AI);
    last_AI_mat = AI;

    // 6. Update
    // theta_new = theta + AI^-1 * grad
    double det = AI.determinant();

     // DEBUG: Print Gradients and AI Matrix
     if (config.verbose) {
         // std::cout << "    [AI Step] Grad=[" << grad_u << ", " << grad_e << "]" << std::endl;
         // std::cout << "    [AI Step] AI=[[" << AI(0,0) << ", " << AI(0,1) << "], [" << AI(1,0) << ", " << AI(1,1) << "]]" << std::endl;
     }

     Eigen::Vector2d grad(grad_u, grad_e);
     Eigen::Vector2d delta;
     Eigen::LLT<Eigen::Matrix2d> llt_AI(AI);
     if (llt_AI.info() == Eigen::Success) {
         delta = llt_AI.solve(grad);
     } else {
         if (std::abs(det) < 1e-20) {
             // std::cout << "    [AI Step] AI Matrix singular. Using Gradient Ascent step." << std::endl;
             // Simple Gradient Ascent with scaling
             // Delta = Grad * LearningRate
             // Use 10% relative change max
             double step_size = 0.1;
             // Scale gradient so max component changes variance by step_size * var
             double max_grad_rel = std::max(std::abs(grad(0))/var_u, std::abs(grad(1))/var_e);
             if (max_grad_rel > 0) {
                 double scale = step_size / max_grad_rel;
                 delta(0) = grad(0) * scale;
                 delta(1) = grad(1) * scale;
             } else {
                 delta.setZero();
             }
         } else {
            Eigen::LLT<Eigen::MatrixXd> lltAI(AI);
            if(lltAI.info() == Eigen::Success) {
                delta = lltAI.solve(grad);
            } else {
                delta = AI.inverse() * grad;
            }
         }
     }

     // Check for NaN
     if (!std::isfinite(delta(0)) || !std::isfinite(delta(1))) {
         // std::cout << "    [AI Step] Delta is NaN/Inf. Abort." << std::endl;
         return false;
     }

     // DEBUG: Delta
     // std::cout << "    [AI Step] Delta=[" << delta(0) << ", " << delta(1) << "]" << std::endl;

     // Check if Newton step is ridiculous (Singular/Ill-conditioned AI)
     // If step is > 100x current variance, the AI matrix was likely near-singular.
     // Fallback to Gradient Ascent.
     if (std::abs(delta(0)) > 100.0 * var_u || std::abs(delta(1)) > 100.0 * var_e) {
          // std::cout << "    [AI Step] Newton Delta ridiculous (" << delta(0) << "," << delta(1) << "). Aborting AI step." << std::endl;
          return false;
     }

     double vu_new = var_u + delta(0);
     double ve_new = var_e + delta(1);

     // --- Robust AI: LogL-based monotonicity check (Knight 2008) ---
     // Save starting point for potential revert
     double start_vu = var_u;
     double start_ve = var_e;

     // Compute current LogL (LHS is already built for current vars at top of function)
     double current_logL = calc_logL_internal();
     if (!std::isfinite(current_logL)) current_logL = -1e18;

     // PCG path: if LogL can't be computed (no direct solver), we cannot verify
     // monotonicity. Fall back to EM (which is monotonic by construction and
     // doesn't need LogL). This prevents AI divergence in matrix-free mode.
     bool logL_unavailable = (!direct_solver && !sparse_direct_solver);
     if (logL_unavailable) {
         if (!vars_u.empty()) vars_u[0] = start_vu;
         var_e = start_ve;
         lhs_built = false;
         return false;  // signal failure so caller falls back to EM
     }

     // Damping / Bounds with LogL monotonicity enforcement
     // Strategy: Step halving until LogL improves (or bounds satisfied as last resort)
     int max_halving = 15;
     bool step_ok = false;
     double best_logL = current_logL;
     double best_vu = var_u, best_ve = var_e;

     for (int k=0; k<max_halving; ++k) {
        vu_new = var_u + delta(0);
        ve_new = var_e + delta(1);

        // Reject negative variances
        if (vu_new <= 0.0 || ve_new <= 0.0) {
            delta *= 0.5;
            continue;
        }

        // Reject ridiculously huge steps (likely singular AI)
        if (std::abs(delta(0)) > 5.0 * var_u || std::abs(delta(1)) > 5.0 * var_e) {
            delta *= 0.5;
            continue;
        }

        // Evaluate LogL at candidate point
        if (!vars_u.empty()) vars_u[0] = vu_new;
        var_e = ve_new;
        lhs_built = false;
        double new_logL = calc_logL_internal();

        if (std::isfinite(new_logL) && new_logL > best_logL) {
            // LogL improved: accept this step
            best_logL = new_logL;
            best_vu = vu_new;
            best_ve = ve_new;
            step_ok = true;
            break;
        }

        // LogL did not improve: halve and retry
        // Restore vars for next evaluation
        if (!vars_u.empty()) vars_u[0] = start_vu;
        var_e = start_ve;
        lhs_built = false;
        delta *= 0.5;
     }

     if (!step_ok) {
        // Could not find an improving step. Restore originals and signal failure
        // so the caller can fall back to EM (monotonic).
        if (!vars_u.empty()) vars_u[0] = start_vu;
        var_e = start_ve;
        lhs_built = false;
        return false;
    }

    // Accept the best step found
    if (!vars_u.empty()) vars_u[0] = best_vu;
    var_e = best_ve;
    last_logL = best_logL;

    lhs_built = false;
    return true;
}

void AI_REML::initialize_dense() {
    if (dense_initialized) return;
    if (vars_u.size() > 1) throw std::runtime_error("Dense Exact Mode not supported for multiple components yet.");

    int n = (int)y.size();
    int p, q;

    if (mme_builder) {
        p = mme_builder->get_p();
        q = mme_builder->get_q_total();

        X_dense = Eigen::MatrixXd::Zero(n, p);
        Z_dense = Eigen::MatrixXd::Zero(n, q);

        for(int i=0; i<n; ++i) {
            // Fill X
            for(const auto& pr : fd_ptr->rows[i]) {
                if(pr.first < p) X_dense(i, pr.first) = pr.second;
            }
            // Fill Z
            if((*recs_ptr)[i].aid > 0) {
                int u_idx = (*recs_ptr)[i].aid - 1;
                if(u_idx < q) Z_dense(i, u_idx) = 1.0;
            }
        }
    } else {
        throw std::runtime_error("initialize_dense requires initialized mme_builder");
    }

    const AbstractMatrix* Ainv_ptr = components[0].Qinv;
    Eigen::MatrixXd Ainv_dense = Ainv_ptr->toDense();
    Eigen::LLT<Eigen::MatrixXd> lltAinv(Ainv_dense);
    if(lltAinv.info() == Eigen::Success) {
        A_dense = lltAinv.solve(Eigen::MatrixXd::Identity(Ainv_dense.rows(), Ainv_dense.cols()));
    } else {
        A_dense = Ainv_dense.inverse();
    }

    W_dense.resize(n, p + q);
    W_dense << X_dense, Z_dense;

    dense_rec_to_anim.assign(n, -1);
    std::vector<int> rec_counts(n, 0);

    if (mme_builder) {
         for(int i=0; i<n; ++i) {
             if((*recs_ptr)[i].aid > 0) {
                 int u_idx = (*recs_ptr)[i].aid - 1;
                 if(u_idx < q) {
                     dense_rec_to_anim[i] = u_idx;
                     rec_counts[i] += 1; // Assuming 1 entry per row for Z usually
                 }
             }
         }
    } else {
        // Should be unreachable due to throw above
    }

    bool is_incidence = true;
    for (int i = 0; i < n; ++i) {
        if (rec_counts[i] > 1) {
            is_incidence = false;
            break;
        }
    }

    ZAZt_dense.resize(n, n);
    if (is_incidence) {
        for(int i=0; i<n; ++i) {
            int u_i = dense_rec_to_anim[i];
            if (u_i == -1) {
                ZAZt_dense.row(i).setZero();
                continue;
            }
            for(int j=0; j<n; ++j) {
                int u_j = dense_rec_to_anim[j];
                if (u_j != -1) {
                    ZAZt_dense(i, j) = A_dense(u_i, u_j);
                } else {
                    ZAZt_dense(i, j) = 0.0;
                }
            }
        }
    } else {
        ZAZt_dense = Z_dense * A_dense * Z_dense.transpose();
    }

    dense_initialized = true;
}

bool AI_REML::run_dense_exact_step(int iter) {
    if (vars_u.size() > 2) return false;
    if (!dense_initialized) initialize_dense();

    double var_u = (!vars_u.empty()) ? vars_u[0] : 1.0;
    double lambda = var_e / var_u;
    std::vector<double> lambdas = {lambda};
    int n = (int)y.size();
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();
    int dim = p + q;
    const AbstractMatrix* Ainv_ptr = components[0].Qinv;

    // log|A| from Ainv (A = Ainv^{-1}) — cached per-instance in log_det_A_member
    // (must NOT be static: a stale value would be reused across AI_REML instances).
    if (!log_det_A_member_valid) {
         Eigen::MatrixXd Ad = Ainv_ptr->toDense();
         Eigen::LDLT<Eigen::MatrixXd> ldlt_Ainv(Ad);
         double log_det_Ainv = ldlt_Ainv.vectorD().array().log().sum();
         log_det_A_member = -log_det_Ainv;
         log_det_A_member_valid = true;
    }
    double log_det_A = log_det_A_member;

    // Helper to compute LogL for candidate variances
    auto compute_logL = [&](double vu, double ve) -> double {
        double lam = ve / vu;
        build_lhs({lam});
        DenseSolver solver(config.dense_tier);
        if (current_LHS_dense_valid) {
            current_LHS_dense = Eigen::MatrixXd(mme_builder->get_lhs());
            if (!solver.compute(current_LHS_dense)) return -1e18;
        } else {
            if (mme_builder) {
                if (!solver.compute(Eigen::MatrixXd(mme_builder->get_lhs()))) return -1e18;
            } else {
                return -1e18;
            }
        }

        double ld_C = solver.logDeterminant();

        Eigen::VectorXd rhs_tmp(dim);
        rhs_tmp << Xty, Zty;
        Eigen::VectorXd s = solver.solve(rhs_tmp);
        Eigen::VectorXd b = s.head(p);
        Eigen::VectorXd u = s.tail(q);
        Eigen::VectorXd e = y - X_dense * b - Z_dense * u;
        double yPy = y.dot(e) / ve;

        double m2 = ld_C + (n - p - q) * std::log(ve) + q * std::log(vu) + yPy + log_det_A;
        if (config.verbose) {
            std::cout << "    [DEBUG] n=" << n << ", p=" << p << ", q=" << q
                      << ", ld_C=" << ld_C << ", ve=" << ve << ", vu=" << vu
                      << ", yPy=" << yPy << ", log_det_A=" << log_det_A
                      << ", m2=" << m2 << std::endl;
        }
        return -0.5 * m2;
    };

    // 1. Build LHS & Factorize
    build_lhs(lambdas);
    DenseSolver solver(config.dense_tier);

    if (current_LHS_dense_valid) {
        current_LHS_dense = Eigen::MatrixXd(mme_builder->get_lhs());
        if (!solver.compute(current_LHS_dense)) return false;
    } else {
        if (mme_builder) {
             if (!solver.compute(Eigen::MatrixXd(mme_builder->get_lhs()))) return false;
        } else {
             // Legacy
             return false;
        }
    }

    double current_log_det_C = solver.logDeterminant();

    // 2. Solve for current estimates
    Eigen::VectorXd rhs(dim);
    rhs << Xty, Zty;
    Eigen::VectorXd sol = solver.solve(rhs);

    Eigen::VectorXd beta_hat = sol.head(p);
    Eigen::VectorXd u_hat = sol.tail(q);

    // 3. Compute Residuals
    Eigen::VectorXd e_hat = y - X_dense * beta_hat - Z_dense * u_hat;

    // Calculate Current LogL
    double current_yPy = y.dot(e_hat) / var_e;
    double current_m2 = current_log_det_C + (n - p - q) * std::log(var_e) + q * std::log(var_u) + current_yPy + log_det_A;
    double current_logL = -0.5 * current_m2;

    if (false) std::cout << "    [Dense Exact] Current LogL: " << current_logL << std::endl;

    // 5. Construct P Matrix
    // P = 1/sigma_e^2 * (I - H)
    // H = W * C_inv * W' * (1/sigma_e^2) ? No.
    // Let's use the relation: P = 1/sigma_e^2 * (I - 1/sigma_e^2 * W * C_inv * W')
    // W = [X Z]
    // H_star = W * C_inv * W'
    Eigen::MatrixXd C_inv_Wt(dim, n);
    for (int i = 0; i < n; ++i) {
        C_inv_Wt.col(i) = solver.solve(W_dense.row(i).transpose());
    }
    Eigen::MatrixXd H_star = W_dense * C_inv_Wt;

    // P = (I - H_star) / var_e
    Eigen::MatrixXd P = -H_star / var_e;
    P.diagonal().array() += 1.0 / var_e;

    // 6. Compute Matrices for Trace
    // M_u = P * Z * A * Z'
    // M_e = P

    // ZAZt = Z * A * Z'
    // OPTIMIZED: Use sparsity of Z (incidence) to fill ZAZt directly from A_dense
    Eigen::MatrixXd M_u = P * ZAZt_dense;

    // 7. Compute Exact AI Terms
    // AI_uu = 0.5 * tr(M_u * M_u)
    // AI_ue = 0.5 * tr(M_u * P)
    // AI_ee = 0.5 * tr(P * P)

    // OPTIMIZATION: Avoid O(N^3) matrix multiplication for trace.
    // Use O(N^2) element-wise operations.

    // AI_uu = 0.5 * sum_i sum_j (M_u(i,j) * M_u(j,i))
    double sum_uu = 0.0;
    // AI_ue = 0.5 * sum_i sum_j (M_u(i,j) * P(j,i)). Since P symmetric, P(j,i)=P(i,j).
    double sum_ue = 0.0;
    // AI_ee = 0.5 * sum_i sum_j (P(i,j)^2)
    double sum_ee = P.squaredNorm(); // squaredNorm is sum(x_ij^2)

    // We can fuse loops for M_u and P
    // M_u is dense (n x n)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double mu_ij = M_u(i, j);
            double mu_ji = M_u(j, i);
            sum_uu += mu_ij * mu_ji;

            double p_ij = P(i, j);
            sum_ue += mu_ij * p_ij;
        }
    }

    double AI_uu = 0.5 * sum_uu;
    double AI_ue = 0.5 * sum_ue;
    double AI_ee = 0.5 * sum_ee;

    // 8. Compute Gradients
    // grad_i = -0.5 * tr(P * dV/di) + 0.5 * y' * P * dV/di * P * y
    // dV/du = ZAZt
    // dV/de = I

    // tr(P * dV/du) = tr(M_u)
    // tr(P * dV/de) = tr(P)

    // Score part: y' P ... P y = (P y)' ... (P y)
    // Py = e_hat / var_e
    Eigen::VectorXd Py = e_hat / var_e;

    // score_u = Py' * ZAZt * Py
    double score_u_quad = Py.dot(ZAZt_dense * Py);
    // score_e = Py' * I * Py = Py.squaredNorm()
    double score_e_quad = Py.squaredNorm();

    double grad_u = -0.5 * M_u.trace() + 0.5 * score_u_quad;
    double grad_e = -0.5 * P.trace() + 0.5 * score_e_quad;

    // 9. Update
    Eigen::Matrix2d AI;
    AI << AI_uu, AI_ue, AI_ue, AI_ee;
    AI = project_spd_2x2(AI);
    last_AI_mat = AI;

    Eigen::Vector2d grad(grad_u, grad_e);

    Eigen::Vector2d delta;
    Eigen::LLT<Eigen::Matrix2d> llt_AI(AI);
    if (llt_AI.info() == Eigen::Success) {
        delta = llt_AI.solve(grad);
    } else {
        if (std::abs(AI.determinant()) < 1e-20) {
            if (false) std::cout << "    [Dense Exact] AI Matrix singular. Step failed." << std::endl;
            return false;
        }
        delta = AI.inverse() * grad;
    }

    double vu_new = var_u + delta(0);
    double ve_new = var_e + delta(1);

    // 10. Damping / Check bounds with LogL
    int max_halving = 6;
    bool step_ok = false;
    bool logL_computed = false;
    double new_logL = current_logL;

    for (int k=0; k<max_halving; ++k) {
        vu_new = var_u + delta(0);
        ve_new = var_e + delta(1);

        if (vu_new > 0 && ve_new > 0) {
            new_logL = compute_logL(vu_new, ve_new);
            logL_computed = true;
            if (new_logL > current_logL || k == max_halving - 1) {
                if (new_logL < current_logL && config.verbose) {
                    std::cout << "    [Dense Exact] Warning: LogL decreased (" << current_logL << " -> " << new_logL << "). Accepting anyway as last resort." << std::endl;
                } else if (config.verbose) {
                    std::cout << "    [Dense Exact] LogL improved: " << current_logL << " -> " << new_logL << std::endl;
                }
                step_ok = true;
                current_logL = new_logL;
                break;
            } else {
                if (false) std::cout << "    [Dense Exact] LogL decreased (" << current_logL << " -> " << new_logL << "). Damping." << std::endl;
            }
        }
        delta *= 0.5;
    }

    if (step_ok) {
        if (!vars_u.empty()) {
            vars_u[0] = vu_new;
        }
        var_e = ve_new;
        lhs_built = false;

        if (!logL_computed) {
            new_logL = compute_logL(vu_new, ve_new);
            current_logL = new_logL;
        }
        double logL = current_logL;

        // Push to history
        std::ostringstream ss;
        ss << "AI(Exact)" << "\t" << (iter + 1) << "\t" << std::fixed << std::setprecision(2) << logL << "\t" << std::setprecision(5) << var_u << "\t" << var_e;
        history.push_back(ss.str());

        return true;
    }

    return false;
}

void AI_REML::calculate_SE() {
    if (config.verbose) std::cout << "Calculating Standard Errors..." << std::endl;

    // Multi-component SE from AI matrix (Path A).
    // When run_aireml_step_multi was used, last_AI_mat is cached.
    // Otherwise (standard EM path), compute the AI matrix now at convergence.
    if (vars_u.size() > 1) {
        int c = (int)vars_u.size();
        int num_params = c + 1;

        // If no cached AI matrix, compute it now at the converged point.
        if (last_AI_mat.rows() != num_params || last_AI_mat.cols() != num_params) {
            if (config.verbose) std::cout << "  [SE-Multi] Computing AI matrix at convergence for SE." << std::endl;
            // Build LHS with converged lambdas and compute AI terms.
            std::vector<double> lambdas(c);
            for (int k = 0; k < c; ++k) lambdas[k] = var_e / std::max(vars_u[k], 1e-9);
            build_lhs(lambdas);
            AI_Terms terms = compute_AI_terms_multi_datadriven();
            if (terms.AI_mat.allFinite()) {
                last_AI_mat = project_spd(terms.AI_mat);
            }
        }

        if (last_AI_mat.rows() == num_params && last_AI_mat.cols() == num_params) {
            // Invert AI matrix to get asymptotic covariance of theta
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(last_AI_mat);
            if (es.info() == Eigen::Success) {
                Eigen::VectorXd evals = es.eigenvalues();
                double max_eval = std::max(evals.maxCoeff(), 1e-12);
                for (int i = 0; i < evals.size(); ++i) {
                    evals(i) = std::max(evals(i), 1e-10 * max_eval);
                }
                Eigen::MatrixXd AI_inv = es.eigenvectors() * evals.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();
                vars_u_se.resize(c);
                for (int k = 0; k < c; ++k) {
                    vars_u_se[k] = std::sqrt(std::max(0.0, AI_inv(k, k)));
                }
                var_e_se = std::sqrt(std::max(0.0, AI_inv(c, c)));
                if (config.verbose) {
                    std::cout << "  [SE-Multi] Computed SEs from AI matrix (" << num_params << "x" << num_params << ")." << std::endl;
                }
                return;
            }
        }
        if (config.verbose) std::cout << "  [SE-Multi] No valid AI matrix; skipping SE calculation." << std::endl;
        return;
    }

    Eigen::Matrix2d AI;
    bool has_exact_ai = false;

    if (last_AI_mat.rows() == 2 && last_AI_mat.cols() == 2) {
        if (false) std::cout << "    [SE] Using exact AI matrix from last iteration." << std::endl;
        AI = last_AI_mat;
        has_exact_ai = true;
    }

    double term_uu = 0.0, term_ue = 0.0, term_ee = 0.0;

    if (!has_exact_ai) {
        if (false) std::cout << "    [SE] Exact AI matrix not found. Estimating using Stochastic Trace Estimation..." << std::endl;
        int n_samples_trace = 200;

        std::vector<double> lambdas;
        for(double vu : vars_u) lambdas.push_back(var_e / vu);
        int n = (int)y.size();

        if (!mme_builder) throw std::runtime_error("calculate_SE requires initialized mme_builder");
        int q = mme_builder->get_q_total();
        int p = mme_builder->get_p();

        build_lhs(lambdas);

        double sum_uu = 0.0, sum_ue = 0.0, sum_ee = 0.0;

        #pragma omp parallel reduction(+:sum_uu, sum_ue)
        {
            unsigned long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            #ifdef _OPENMP
            seed += omp_get_thread_num() * 112233;
            #endif
            RNG local_rng(seed);

            #pragma omp for
            for (int k=0; k<n_samples_trace; ++k) {
                Eigen::VectorXd z(q);
                local_rng.fill_rademacher(z);

                Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(p);
                Eigen::VectorXd v = mme_builder->mult_design(zero_beta, z, *recs_ptr, *fd_ptr);

                Eigen::VectorXd rhs = mme_builder->mult_transpose_design(v, *recs_ptr, *fd_ptr);
                Eigen::VectorXd sol = solve_lhs(rhs);

                Eigen::VectorXd beta_hat = sol.head(p);
                Eigen::VectorXd u_hat = sol.tail(q);

                Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_hat, *recs_ptr, *fd_ptr);
                Eigen::VectorXd e_hat = v - y_hat;
                Eigen::VectorXd Pv = e_hat / var_e;

                sum_ue += Pv.squaredNorm();

                Eigen::VectorXd full_trans = mme_builder->mult_transpose_design(Pv, *recs_ptr, *fd_ptr);
                Eigen::VectorXd ZtPv = full_trans.tail(q);
                sum_uu += ZtPv.squaredNorm();
            }
        }

        term_uu = 0.5 * (sum_uu / n_samples_trace);
        term_ue = 0.5 * (sum_ue / n_samples_trace);

        #pragma omp parallel reduction(+:sum_ee)
        {
            unsigned long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            #ifdef _OPENMP
            seed += omp_get_thread_num() * 445566;
            #endif
            RNG local_rng(seed);

            #pragma omp for
            for (int k=0; k<n_samples_trace; ++k) {
                Eigen::VectorXd z_vec(n);
                local_rng.fill_rademacher(z_vec);

                Eigen::VectorXd rhs = mme_builder->mult_transpose_design(z_vec, *recs_ptr, *fd_ptr);
                Eigen::VectorXd sol = solve_lhs(rhs);

                Eigen::VectorXd beta_hat = sol.head(p);
                Eigen::VectorXd u_hat = sol.tail(q);

                Eigen::VectorXd y_hat = mme_builder->mult_design(beta_hat, u_hat, *recs_ptr, *fd_ptr);
                Eigen::VectorXd e_hat = z_vec - y_hat;
                Eigen::VectorXd Pz = e_hat / var_e;

                sum_ee += Pz.squaredNorm();
            }
        }

        term_ee = 0.5 * (sum_ee / n_samples_trace);

        AI(0,0) = term_uu;
        AI(0,1) = term_ue;
        AI(1,0) = term_ue;
        AI(1,1) = term_ee;
        AI = project_spd_2x2(AI);
    } else {
        term_uu = AI(0,0);
        term_ue = AI(0,1);
        term_ee = AI(1,1);
    }

    Eigen::Matrix2d Cov;
    Eigen::LLT<Eigen::Matrix2d> llt(AI);
    if (llt.info() == Eigen::Success) {
        Cov = llt.solve(Eigen::Matrix2d::Identity());
    } else {
        Cov = AI.inverse();
    }

    double se_var_u = std::sqrt(std::max(0.0, Cov(0,0)));
    double se_var_e = std::sqrt(std::max(0.0, Cov(1,1)));

    var_e_se = se_var_e;
    if (vars_u_se.size() < vars_u.size()) vars_u_se.resize(vars_u.size());
    if (!vars_u_se.empty()) vars_u_se[0] = se_var_u;

    if (config.verbose) {
        std::cout << "AI Matrix: [[" << term_uu << ", " << term_ue << "], [" << term_ue << ", " << term_ee << "]]" << std::endl;
        std::cout << "SE(Vu)=" << se_var_u << ", SE(Ve)=" << se_var_e << std::endl;
    }
}

// --- MCEM_REML Implementation ---

void MCEM_REML::solve() {
    if (components.size() != 1) {
        throw std::runtime_error("MC-EM REML currently supports single variance component only.");
    }
    if (!initialized) initialize();

    double start_vu = vars_u.empty() ? 1.0 : vars_u[0];
    std::cout << "Starting MC-EM REML VCE..." << std::endl;
    std::cout << "Initial vars: Ve=" << var_e << ", Vu=" << start_vu << std::endl;

    history.clear();
    converged = false;
    iterations_run = 0;
    last_diff = std::numeric_limits<double>::quiet_NaN();

    for (int iter = 0; iter < config.max_iter; ++iter) {
        double old_ve = var_e;
        double old_vu = vars_u[0];

        run_em_iteration(iter);

        if (config.verbose) {
            std::cout << "Iter " << iter + 1
                      << ": Ve=" << var_e
                      << ", Vu=" << vars_u[0] << std::endl;
        }

        // Record history
        std::ostringstream ss;
        ss << (iter + 1) << "\t" << vars_u[0] << "\t" << var_e;
        history.push_back(ss.str());

        double diff = std::abs(var_e - old_ve) + std::abs(vars_u[0] - old_vu);
        last_diff = diff;
        iterations_run = iter + 1;
        if (diff < config.tol) {
              if (config.verbose) std::cout << "[Converged?] Yes!\n" << std::endl;
              converged = true;
              break;
          }
    }
}

void MCEM_REML::initialize() {
    // std::cout << "Initializing MME components..." << std::endl;
    if (recs_ptr && fd_ptr) {
        mme_builder = std::make_unique<MMELHSBuilder>(*recs_ptr, *fd_ptr, components);
        Eigen::VectorXd full_rhs = mme_builder->build_rhs(*recs_ptr, *fd_ptr, mme_builder->get_dim());
        int p = mme_builder->get_p();
        int q = mme_builder->get_q_total();
        Xty = full_rhs.head(p);
        Zty = full_rhs.tail(q);

        // Ensure samplers are set
        if (samplers.size() < vars_u.size()) {
            samplers.resize(vars_u.size());
            for(size_t k=0; k<vars_u.size(); ++k) samplers[k] = std::make_shared<IdentitySampler>(rng);
        }

    } else {
         throw std::runtime_error("MCEM_REML requires raw records. Legacy mode deprecated.");
    }

    initialized = true;
}

Eigen::VectorXd MCEM_REML::solve_mme(const Eigen::VectorXd& rhs_beta, const Eigen::VectorXd& rhs_u, const std::vector<double>& lambdas) {
    if (!mme_builder) throw std::runtime_error("MME Builder not initialized");

    const auto& LHS = mme_builder->build_lhs(lambdas);
    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();
    int dim = p + q;

    Eigen::VectorXd RHS(dim);
    RHS.head(p) = rhs_beta;
    RHS.tail(q) = rhs_u;

    std::string precond = (p > 0) ? "block_jacobi" : "diag";
    ::PCGSolver<Eigen::SparseMatrix<double>> pcg(LHS, RHS, precond, p);
    pcg.setQuiet(true);
    return pcg.solve(config.pcg_tol, config.pcg_max_iter);
}

void MCEM_REML::run_em_iteration(int iter) {
    if (vars_u.empty()) vars_u.push_back(1.0);
    double var_u = vars_u[0];

    if (!std::isfinite(var_u) || var_u <= 0.0) var_u = 1e-6;
    if (!std::isfinite(var_e) || var_e <= 0.0) var_e = 1e-6;
    std::vector<double> lambdas = {var_e / var_u};

    Eigen::VectorXd sol = solve_mme(Xty, Zty, lambdas);

    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();

    Eigen::VectorXd beta_hat = sol.head(p);
    Eigen::VectorXd u_hat = sol.tail(q);

    double sum_u_Ainv_u = 0.0;
    double sum_e_e = 0.0;

    int n_samples = config.mc_samples;

    for (int k = 0; k < n_samples; ++k) {
        Eigen::VectorXd u_sim(q);
        samplers[0]->sample(u_sim); // standardized
        u_sim *= std::sqrt(var_u); // Scale

        Eigen::VectorXd e_sim(y.size());
        rng.fill_normal(e_sim);
        e_sim *= std::sqrt(var_e);

        // y_sim = Z * u_sim + e_sim
        Eigen::VectorXd zero_beta = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd y_sim = mme_builder->mult_design(zero_beta, u_sim, *recs_ptr, *fd_ptr) + e_sim;

        // Xt_ysim, Zt_ysim
        Eigen::VectorXd rhs_sim = mme_builder->mult_transpose_design(y_sim, *recs_ptr, *fd_ptr);
        Eigen::VectorXd Xt_ysim = rhs_sim.head(p);
        Eigen::VectorXd Zt_ysim = rhs_sim.tail(q);

        Eigen::VectorXd sol_sim = solve_mme(Xt_ysim, Zt_ysim, lambdas);
        Eigen::VectorXd u_sim_hat = sol_sim.tail(q);

        Eigen::VectorXd u_k = u_hat + (u_sim - u_sim_hat);

        double val_u = u_k.dot(components[0].Qinv->operator*(u_k));
        sum_u_Ainv_u += val_u;

        Eigen::VectorXd beta_sim_hat = sol_sim.head(p);

        // e_hat = y - X * beta_hat - Z * u_hat
        Eigen::VectorXd y_hat_pred = mme_builder->mult_design(beta_hat, u_hat, *recs_ptr, *fd_ptr);
        Eigen::VectorXd e_hat = y - y_hat_pred;

        // e_sim_hat = y_sim - X * beta_sim_hat - Z * u_sim_hat
        Eigen::VectorXd y_sim_hat_pred = mme_builder->mult_design(beta_sim_hat, u_sim_hat, *recs_ptr, *fd_ptr);
        Eigen::VectorXd e_sim_hat = y_sim - y_sim_hat_pred;

        Eigen::VectorXd e_k = e_hat + (e_sim - e_sim_hat);

        sum_e_e += e_k.dot(e_k);
    }

    double new_var_u = (sum_u_Ainv_u / n_samples) / q;
    double new_var_e = (sum_e_e / n_samples) / y.size();

    double alpha = 0.3;
    if (iter < 5) alpha = 0.5;

    vars_u[0] = alpha * new_var_u + (1.0 - alpha) * var_u;
    var_e = alpha * new_var_e + (1.0 - alpha) * var_e;

    if (vars_u[0] < 1e-6) vars_u[0] = 1e-6;
    if (var_e < 1e-6) var_e = 1e-6;

    var_u_legacy = vars_u[0];
}

double AI_REML::calc_logL_internal() {
    double logL = -1e20;

    std::vector<double> lambdas;
    for (double vu : vars_u) {
        lambdas.push_back(var_e / std::max(vu, 1e-9));
    }

    // Ensure LHS is built for current vars
    build_lhs(lambdas);

    int p = mme_builder->get_p();
    int q = mme_builder->get_q_total();
    Eigen::VectorXd rhs(p + q);
    rhs << Xty, Zty;

    // Solve
    Eigen::VectorXd sol = solve_lhs(rhs);

    // Determinant
    double log_det_C = 0.0;
    bool ok_det = false;

    if (direct_solver && direct_solver->info() == Eigen::Success) {
         const Eigen::VectorXd D = direct_solver->vectorD();
         ok_det = true;
         for (int i = 0; i < D.size(); ++i) {
             double d = D(i);
             if (!(std::isfinite(d)) || d <= 0.0) { ok_det = false; break; }
             log_det_C += std::log(d);
         }
    } else if (sparse_direct_solver && sparse_direct_solver->info() == Eigen::Success) {
         const Eigen::VectorXd D = sparse_direct_solver->vectorD();
         ok_det = true;
         for (int i = 0; i < D.size(); ++i) {
             double d = D(i);
             if (!(std::isfinite(d)) || d <= 0.0) { ok_det = false; break; }
             log_det_C += std::log(d);
         }
    }

    if (ok_det) {
         double yty = y.dot(y);
         double sol_rhs = sol.dot(rhs);
         double yPy = (yty - sol_rhs) / var_e;
         double n = (double)y.size();

         double log_det_G = 0.0;
         const auto& qs = mme_builder->get_qs();
         for(size_t k=0; k<vars_u.size(); ++k) log_det_G += qs[k] * std::log(std::max(vars_u[k], 1e-9));

         logL = -0.5 * (log_det_C + (n - p - q) * std::log(var_e) + log_det_G + yPy);
    }

    return logL;
}

// --- Factory and New Algorithms ---

std::unique_ptr<VCESolver> VCESolver::create(const std::vector<GenRecord>& recs,
                                           const FixedDesignG& fd,
                                           const std::vector<RandomComponent>& comps,
                                           const Eigen::VectorXd& y,
                                           VCEConfig cfg) {
    if (cfg.algorithm == VCEAlgorithm::MC) {
        return std::make_unique<MCEM_REML>(recs, fd, comps, y, cfg);
    } else if (cfg.algorithm == VCEAlgorithm::HE) {
        return std::make_unique<HE_Regression>(recs, fd, comps, y, cfg);
    } else if (cfg.algorithm == VCEAlgorithm::Exact) {
        return std::make_unique<Exact_LMM>(recs, fd, comps, y, cfg);
    } else if (cfg.algorithm == VCEAlgorithm::VMatrix) {
        return std::make_unique<VMatrixAI_REML>(recs, fd, comps, y, cfg);
    } else if (cfg.algorithm == VCEAlgorithm::STCG) {
        return std::make_unique<STCG_VCE>(recs, fd, comps, y, cfg);
    } else {
        // Default to AI_REML (handles AI, EM, EMAI, HI)
        return std::make_unique<AI_REML>(recs, fd, comps, y, cfg);
    }
}

void VCESolver::solveDiagonal(const Eigen::VectorXd& D,
                              const Eigen::VectorXd& Uty,
                              const Eigen::MatrixXd& UtX,
                              double& out_var_g,
                              double& out_var_e,
                              double& out_se_g,
                              double& out_se_e,
                              Eigen::VectorXd* out_beta,
                              Eigen::VectorXd* out_beta_se) {
    // Exact LMM for Diagonalized System (Null Model)
    // Model: y ~ N(X*beta, var_g * D + var_e * I)

    int n = (int)D.size();
    int p = (int)UtX.cols();
    double n_eff = (double)(n - p);

    // Helper Lambda for NegLogL
    auto calc_neg_logL = [&](double delta) -> double {
        Eigen::VectorXd V_diag = D.array() + delta;
        Eigen::VectorXd V_inv_diag = V_diag.cwiseInverse();

        // Weighted Least Squares for beta
        // (UtX' diag(V_inv) UtX) beta = UtX' diag(V_inv) Uty

        Eigen::MatrixXd Xt_Vinv_X = UtX.transpose() * V_inv_diag.asDiagonal() * UtX;
        Eigen::VectorXd Xt_Vinv_y = UtX.transpose() * V_inv_diag.asDiagonal() * Uty;

        Eigen::LDLT<Eigen::MatrixXd> solver(Xt_Vinv_X);
        Eigen::VectorXd beta = solver.solve(Xt_Vinv_y);

        Eigen::VectorXd resid = Uty - UtX * beta;
        double quad = (resid.array().square() * V_inv_diag.array()).sum();

        double log_det_V = V_diag.array().log().sum();
        double log_det_XtVinvX = solver.vectorD().array().log().sum();

        double sigma_g2 = quad / n_eff;

        // REML LogL
        double logL = -0.5 * (log_det_V + log_det_XtVinvX + n_eff * std::log(sigma_g2) + n_eff);
        return -logL;
    };

    // Golden Section Search
    double min_log_delta = -5.0;
    double max_log_delta = 5.0;
    double gr = (std::sqrt(5.0) + 1.0) / 2.0;

    double c = max_log_delta - (max_log_delta - min_log_delta) / gr;
    double d = min_log_delta + (max_log_delta - min_log_delta) / gr;

    double fc = calc_neg_logL(std::pow(10.0, c));
    double fd = calc_neg_logL(std::pow(10.0, d));

    double tol = 1e-5;
    while (std::abs(c - d) > tol) {
        if (fc < fd) {
            max_log_delta = d;
            d = c;
            fd = fc;
            c = max_log_delta - (max_log_delta - min_log_delta) / gr;
            fc = calc_neg_logL(std::pow(10.0, c));
        } else {
            min_log_delta = c;
            c = d;
            fc = fd;
            d = min_log_delta + (max_log_delta - min_log_delta) / gr;
            fd = calc_neg_logL(std::pow(10.0, d));
        }
    }

    double best_delta = std::pow(10.0, (min_log_delta + max_log_delta) / 2.0);

    // Final Calculation
    Eigen::VectorXd V_diag = D.array() + best_delta;
    Eigen::VectorXd V_inv_diag = V_diag.cwiseInverse();
    Eigen::MatrixXd Xt_Vinv_X = UtX.transpose() * V_inv_diag.asDiagonal() * UtX;
    Eigen::VectorXd Xt_Vinv_y = UtX.transpose() * V_inv_diag.asDiagonal() * Uty;

    Eigen::LDLT<Eigen::MatrixXd> solver(Xt_Vinv_X);
    Eigen::VectorXd beta = solver.solve(Xt_Vinv_y);

    Eigen::VectorXd resid = Uty - UtX * beta;
    double quad = (resid.array().square() * V_inv_diag.array()).sum();

    out_var_g = quad / n_eff;
    out_var_e = out_var_g * best_delta;

    // Simplified SE Calculation
    out_se_g = out_var_g * std::sqrt(2.0/n_eff);
    out_se_e = out_var_e * std::sqrt(2.0/n_eff);

    if (out_beta) *out_beta = beta;
    if (out_beta_se) {
        Eigen::MatrixXd cov_beta = solver.solve(Eigen::MatrixXd::Identity(p, p));
        out_beta_se->resize(p);
        for(int i=0; i<p; ++i) (*out_beta_se)(i) = std::sqrt(cov_beta(i,i) * out_var_g);
    }
}

// --- HE Regression ---

void HE_Regression::solve() {
    std::cout << "Starting HE Regression (Haseman-Elston)..." << std::endl;

    if (components.size() != 1) {
        throw std::runtime_error("HE Regression currently supports single variance component only.");
    }

    if (!mme_builder) {
        mme_builder = std::make_unique<MMELHSBuilder>(*recs_ptr, *fd_ptr, components);
    }

    const AbstractMatrix* Qinv_ptr = components[0].Qinv;
    bool is_large = (Qinv_ptr->rows() > 10000);

    // Check if we can use Dense HE (explicit G)
    // If matrix is small (<10k), Dense HE is fast and exact.
    // If large, we MUST use Randomized HE.

    if (is_large || config.force_exact == false) {
        if (is_large && config.verbose) std::cout << "  [HE] Matrix size " << Qinv_ptr->rows() << " -> Using Randomized HE (RHE)." << std::endl;
        run_randomized_he();
    } else {
        if (false) std::cout << "  [HE] Matrix size " << Qinv_ptr->rows() << " -> Using Dense HE." << std::endl;
        run_dense_he();
    }
}

void HE_Regression::run_dense_he() {
    int n = (int)y.size();
    int p = mme_builder->get_p();

    // 1. OLS for Fixed Effects
    Eigen::VectorXd full_rhs = mme_builder->build_rhs(*recs_ptr, *fd_ptr, mme_builder->get_dim());
    Eigen::VectorXd Xty = full_rhs.head(p);

    // Build X'X (lambda=1 dummy)
    mme_builder->build_lhs({1.0});
    Eigen::SparseMatrix<double> LHS = mme_builder->get_lhs();
    Eigen::MatrixXd XtX = LHS.topLeftCorner(p, p);

    // Solve OLS
    Eigen::VectorXd beta_ols = XtX.ldlt().solve(Xty);

    // 2. Compute Residuals y_adj = y - X * beta_ols
    Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(mme_builder->get_q_total());
    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_ols, zero_u, *recs_ptr, *fd_ptr);
    Eigen::VectorXd resid = y - y_hat;

    // 3. Regression: resid_i * resid_j ~ A_ij * var_g
    const AbstractMatrix* Qinv_ptr = components[0].Qinv;
    Eigen::MatrixXd Qinv_dense = Qinv_ptr->toDense();
    Eigen::MatrixXd A;
    Eigen::LLT<Eigen::MatrixXd> lltQinv(Qinv_dense);
    if(lltQinv.info() == Eigen::Success) {
        A = lltQinv.solve(Eigen::MatrixXd::Identity(Qinv_dense.rows(), Qinv_dense.cols()));
    } else {
        A = Qinv_dense.inverse();
    }

    double sum_A2 = 0.0;
    double sum_Ay = 0.0;

    // Only use off-diagonal elements to estimate var_g (avoids var_e contamination)
    for(int i=0; i<n; ++i) {
        for(int j=0; j<i; ++j) {
            double A_ij = A(i, j);
            double y_prod = resid(i) * resid(j);
            sum_A2 += A_ij * A_ij;
            sum_Ay += A_ij * y_prod;
        }
    }

    double var_g_est = 0.0;
    if (sum_A2 > 1e-12) {
        var_g_est = sum_Ay / sum_A2;
    }

    // Estimate var_e using diagonal
    // E[y_i^2] = A_ii * var_g + var_e
    double sum_e = 0.0;
    for(int i=0; i<n; ++i) {
        sum_e += (resid(i)*resid(i) - A(i,i) * var_g_est);
    }
    double var_e_est = sum_e / n;

    if (var_g_est < 1e-6) var_g_est = 1e-6;
    if (var_e_est < 1e-6) var_e_est = 1e-6;

    vars_u[0] = var_g_est;
    var_e = var_e_est;

    std::cout << "  [HE Dense] Results: Vg=" << var_g_est << ", Ve=" << var_e_est << std::endl;
    converged = true;
}

void HE_Regression::run_randomized_he() {
    // Randomized HE (Wu et al. 2018 / GCTA fast-HE) adapted for A^-1
    // We solve the system of moments:
    // Tr(A^2) * Vg + Tr(A) * Ve = y'Ay
    // Tr(A) * Vg + N * Ve = y'y
    //
    // Note: y here is projected y (P y) or residuals?
    // RHE usually works on phenotypes directly or residuals.
    // Let's use residuals e = y - X*beta_ols.

    int n = (int)y.size();
    int p = mme_builder->get_p();

    // 1. OLS for Fixed Effects
    Eigen::VectorXd full_rhs = mme_builder->build_rhs(*recs_ptr, *fd_ptr, mme_builder->get_dim());
    Eigen::VectorXd Xty = full_rhs.head(p);
    mme_builder->build_lhs({1.0});
    Eigen::SparseMatrix<double> LHS = mme_builder->get_lhs();
    Eigen::MatrixXd XtX = LHS.topLeftCorner(p, p);
    Eigen::VectorXd beta_ols = XtX.ldlt().solve(Xty);

    Eigen::VectorXd zero_u = Eigen::VectorXd::Zero(mme_builder->get_q_total());
    Eigen::VectorXd y_hat = mme_builder->mult_design(beta_ols, zero_u, *recs_ptr, *fd_ptr);
    Eigen::VectorXd resid = y - y_hat;

    // 2. Estimate Traces Stochastically
    // We need Tr(A), Tr(A^2) and y'Ay.
    // A = Qinv^-1 or G.

    // Check if we have Genomic Matrix (Matrix-Free)
    GenotypeMatrix* geno_mat = components[0].geno_mat;

    int k = std::max(50, config.mc_samples); // Number of random vectors
    double tr_A = 0.0;
    double tr_AA = 0.0;

    RNG local_rng(12345);

    if (geno_mat) {
        if (false) std::cout << "  [RHE] Using Genomic Matrix-Free Multiplication..." << std::endl;

        int n_threads = 1;
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #endif

        for(int i=0; i<k; ++i) {
            Eigen::VectorXd z(n);
            local_rng.fill_rademacher(z);

            // Compute Az = G z = 1/M * Z * Z' * z
            Eigen::VectorXd u(geno_mat->cols());
            geno_mat->multiply_Zt_v(z, u, n_threads);

            Eigen::VectorXd Az(n);
            geno_mat->multiply_Z_v(u, Az, n_threads);
            Az /= (double)geno_mat->cols();

            // Tr(A) = E[z' Az]
            tr_A += z.dot(Az);

            // Tr(A^2) = z' A A z = |Az|^2
            tr_AA += Az.squaredNorm();
        }
    } else {
        // Factorize Qinv once (Sparse Inverse)
        const AbstractMatrix* Qinv_ptr = components[0].Qinv;
        if (!sparse_solver) {
            if (false) std::cout << "  [RHE] Factorizing Qinv (Sparse)..." << std::endl;
            const SparseMatrixAdapter* sp = dynamic_cast<const SparseMatrixAdapter*>(Qinv_ptr);
            if (sp) {
                sparse_solver = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
                sparse_solver->compute(sp->getMatrix());
                if (sparse_solver->info() != Eigen::Success) {
                    std::cerr << "  [RHE] Factorization failed. Cannot run RHE." << std::endl;
                    return;
                }
            } else {
                std::cerr << "  [RHE] Qinv is not sparse adapter??" << std::endl;
                return;
            }
        }

        for(int i=0; i<k; ++i) {
            Eigen::VectorXd z(n);
            local_rng.fill_rademacher(z);

            // Compute A_sys z = Z * Qinv^-1 * Z' * z
            Eigen::VectorXd Ztz = mme_builder->mult_transpose_design(z, *recs_ptr, *fd_ptr).tail(mme_builder->get_q_total());
            Eigen::VectorXd Qinv_Ztz = sparse_solver->solve(Ztz);
            Eigen::VectorXd zero_b = Eigen::VectorXd::Zero(mme_builder->get_p());
            Eigen::VectorXd Az = mme_builder->mult_design(zero_b, Qinv_Ztz, *recs_ptr, *fd_ptr);

            tr_A += z.dot(Az);
            tr_AA += Az.squaredNorm();
        }
    }

    tr_A /= k;
    tr_AA /= k;

    // 3. Compute y' A y
    // y = resid
    double yAy = 0.0;
    if (geno_mat) {
        int n_threads = 1;
        #ifdef _OPENMP
        n_threads = omp_get_max_threads();
        #endif

        Eigen::VectorXd u(geno_mat->cols());
        geno_mat->multiply_Zt_v(resid, u, n_threads);

        // y' G y = 1/M * y' Z Z' y = 1/M * |Z'y|^2 = 1/M * |u|^2
        yAy = u.squaredNorm() / (double)geno_mat->cols();
    } else {
        Eigen::VectorXd Zt_resid = mme_builder->mult_transpose_design(resid, *recs_ptr, *fd_ptr).tail(mme_builder->get_q_total());
        Eigen::VectorXd Qinv_Zt_resid = sparse_solver->solve(Zt_resid);
        yAy = Zt_resid.dot(Qinv_Zt_resid);
    }

    double yy = resid.dot(resid);

    // 4. Solve 2x2 System
    // [ Tr(AA)  Tr(A) ] [ Vg ] = [ yAy ]
    // [ Tr(A)   N     ] [ Ve ] = [ yy  ]

    Eigen::Matrix2d M;
    M << tr_AA, tr_A,
         tr_A,  (double)n;

    Eigen::Vector2d rhs;
    rhs << yAy, yy;

    Eigen::Vector2d sol = M.inverse() * rhs;

    double vg = sol(0);
    double ve = sol(1);

    if (vg < 1e-6) vg = 1e-6;
    if (ve < 1e-6) ve = 1e-6;

    vars_u[0] = vg;
    var_e = ve;

    std::cout << "  [RHE Sparse] Results: Vg=" << vg << ", Ve=" << ve << std::endl;
    converged = true;
}


// --- Exact LMM ---

double Exact_LMM::calc_neg_logL(double delta, const Eigen::MatrixXd& X_rot, const Eigen::VectorXd& y_rot, int n, int p) {
    Eigen::VectorXd V_diag = D.array() + delta;
    Eigen::VectorXd V_inv_diag = V_diag.cwiseInverse();

    // Weighted Least Squares for beta
    // X' V^-1 X beta = X' V^-1 y
    // (X_rot' diag(V_inv) X_rot) beta = X_rot' diag(V_inv) y_rot

    Eigen::MatrixXd Xt_Vinv_X = X_rot.transpose() * V_inv_diag.asDiagonal() * X_rot;
    Eigen::VectorXd Xt_Vinv_y = X_rot.transpose() * V_inv_diag.asDiagonal() * y_rot;

    Eigen::LDLT<Eigen::MatrixXd> solver(Xt_Vinv_X);
    Eigen::VectorXd beta = solver.solve(Xt_Vinv_y);

    Eigen::VectorXd resid = y_rot - X_rot * beta;
    double quad = (resid.array().square() * V_inv_diag.array()).sum();

    double log_det_V = V_diag.array().log().sum();
    double log_det_XtVinvX = 0.0;

    // For REML
    Eigen::VectorXd L_diag = solver.vectorD();
    log_det_XtVinvX = L_diag.array().log().sum(); // Solver computes L D L'

    double n_eff = (double)(n - p);
    double sigma_g2 = quad / n_eff;

    double logL = -0.5 * (log_det_V + log_det_XtVinvX + n_eff * std::log(sigma_g2) + n_eff);
    return -logL; // Minimize negative logL
}

void Exact_LMM::solve() {
    std::cout << "Starting Exact LMM (Eigen Decomposition)..." << std::endl;

    if (components.size() != 1) {
        throw std::runtime_error("Exact LMM currently supports single variance component only.");
    }

    int n = (int)y.size();
    if (!mme_builder) {
        mme_builder = std::make_unique<MMELHSBuilder>(*recs_ptr, *fd_ptr, components);
    }

    // 1. Get G Matrix (Dense)
    // We assume Qinv is provided. We need G.
    const AbstractMatrix* Qinv_ptr = components[0].Qinv;
    if (Qinv_ptr->rows() > 20000) {
        throw std::runtime_error("Exact LMM requires dense G; matrix is too large.");
    }

    if (!decomposed) {
        std::cout << "  [Exact] Computing G = Qinv^-1..." << std::endl;
        Eigen::MatrixXd Qinv_dense = Qinv_ptr->toDense();
        Eigen::MatrixXd G;
        Eigen::LLT<Eigen::MatrixXd> lltQinv(Qinv_dense);
        if(lltQinv.info() == Eigen::Success) {
            G = lltQinv.solve(Eigen::MatrixXd::Identity(Qinv_dense.rows(), Qinv_dense.cols()));
        } else {
            G = Qinv_dense.inverse();
        }

        // 2. Eigen Decomposition of G = U D U'
        std::cout << "  [Exact] Eigen Decomposition of G..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        D = es.eigenvalues(); // eigenvalues
        U = es.eigenvectors(); // eigenvectors
        decomposed = true;
    }

    // 3. Rotate y and X
    // y_rot = U' * y
    // X_rot = U' * X
    Eigen::VectorXd y_rot = U.transpose() * y;

    int p = mme_builder->get_p();
    // Build X dense
    Eigen::MatrixXd X(n, p);
    // Fill X from fd_ptr
    X.setZero();
    for(int i=0; i<n; ++i) {
        for(const auto& pr : fd_ptr->rows[i]) {
            if(pr.first < p) X(i, pr.first) = pr.second;
        }
    }

    Eigen::MatrixXd X_rot = U.transpose() * X;

    // 4. Optimize Delta (Golden Section Search)
    // Range for log10(delta): [-5, 5] usually covers most biological cases
    double min_log_delta = -5.0;
    double max_log_delta = 5.0;

    // Golden Section Search
    double gr = (std::sqrt(5.0) + 1.0) / 2.0;
    double c = max_log_delta - (max_log_delta - min_log_delta) / gr;
    double d = min_log_delta + (max_log_delta - min_log_delta) / gr;

    double fc = calc_neg_logL(std::pow(10.0, c), X_rot, y_rot, n, p);
    double fd = calc_neg_logL(std::pow(10.0, d), X_rot, y_rot, n, p);

    double tol = 1e-5;
    while (std::abs(c - d) > tol) {
        if (fc < fd) {
            max_log_delta = d;
            d = c;
            fd = fc;
            c = max_log_delta - (max_log_delta - min_log_delta) / gr;
            fc = calc_neg_logL(std::pow(10.0, c), X_rot, y_rot, n, p);
        } else {
            min_log_delta = c;
            c = d;
            fc = fd;
            d = min_log_delta + (max_log_delta - min_log_delta) / gr;
            fd = calc_neg_logL(std::pow(10.0, d), X_rot, y_rot, n, p);
        }
    }

    double best_log_delta = (min_log_delta + max_log_delta) / 2.0;
    double best_delta = std::pow(10.0, best_log_delta);

    // Recalculate Final Results
    Eigen::VectorXd V_diag = D.array() + best_delta;
    Eigen::VectorXd V_inv_diag = V_diag.cwiseInverse();
    Eigen::MatrixXd Xt_Vinv_X = X_rot.transpose() * V_inv_diag.asDiagonal() * X_rot;
    Eigen::VectorXd Xt_Vinv_y = X_rot.transpose() * V_inv_diag.asDiagonal() * y_rot;
    Eigen::VectorXd beta = Xt_Vinv_X.ldlt().solve(Xt_Vinv_y);
    Eigen::VectorXd resid = y_rot - X_rot * beta;
    double quad = (resid.array().square() * V_inv_diag.array()).sum();
    double sigma_g2 = quad / (double)(n - p);

    vars_u[0] = sigma_g2;
    var_e = sigma_g2 * best_delta;

    std::cout << "Exact LMM Results (Golden Section): Vg=" << vars_u[0] << ", Ve=" << var_e << " (Delta=" << best_delta << ")" << std::endl;
    converged = true;
}

}
