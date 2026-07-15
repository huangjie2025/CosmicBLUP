#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "design.h"
#include "matrix_adapter.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

class ImplicitMME {
public:
    const std::vector<GenRecord>& recs;
    const FixedDesignG& fd;

    // Multiple Random Components
    std::vector<RandomComponent> components;
    std::vector<double> lambdas;

    int n_records;
    int p_fixed;
    int q_total; // Total size of all random effects
    int dim;
    std::vector<int> qs; // Size of each random effect
    std::vector<int> q_offsets; // Start index of each random effect

    // Diagonal for preconditioner
    Eigen::VectorXd diag;

    // New Constructor for Multiple Components
    ImplicitMME(const std::vector<GenRecord>& recs_,
                const FixedDesignG& fd_,
                const std::vector<RandomComponent>& components_,
                const std::vector<double>& lambdas_)
        : recs(recs_), fd(fd_), components(components_), lambdas(lambdas_)
    {
        n_records = (int)recs.size();
        p_fixed = fd.p;

        q_total = 0;
        for (const auto& comp : components) {
            int q = comp.Qinv ? comp.Qinv->rows() : 0;
            qs.push_back(q);
            q_offsets.push_back(q_total);
            q_total += q;
        }

        dim = p_fixed + q_total;
        computeDiagonal();
    }

    // Legacy Constructor
    ImplicitMME(const std::vector<GenRecord>& recs_,
                const FixedDesignG& fd_,
                const AbstractMatrix* Qinv_,
                double lambda_)
        : recs(recs_), fd(fd_)
    {
        n_records = (int)recs.size();
        p_fixed = fd.p;

        RandomComponent rc;
        rc.Qinv = Qinv_;
        components.push_back(rc);
        lambdas.push_back(lambda_);

        int q = Qinv_ ? Qinv_->rows() : 0;
        qs.push_back(q);
        q_offsets.push_back(0);
        q_total = q;

        dim = p_fixed + q_total;
        computeDiagonal();
    }

    // Legacy constructor support (if needed, but we should update callers)
    // Actually we will update callers in mme.cpp to use the above constructor.

    int rows() const { return dim; }
    int cols() const { return dim; }
    long long nonZeros() const {
        long long est = (long long)n_records * 10;
        for (const auto& comp : components) {
            if (comp.Qinv) est += comp.Qinv->nonZeros();
        }
        return est;
    }

    double coeff(int i, int j) const {
        if (i != j) return 0.0;
        if (i < 0 || i >= dim) return 0.0;
        return diag(i);
    }

    void computeDiagonal() {
        diag = Eigen::VectorXd::Zero(dim);

        // 1. Add Qinv diagonal (scaled by lambda)
        for (size_t k = 0; k < components.size(); ++k) {
            const auto& comp = components[k];
            if (comp.Qinv) {
                Eigen::VectorXd d_q = Eigen::VectorXd::Zero(qs[k]);
                comp.Qinv->add_diagonal_to(d_q, lambdas[k]);
                diag.segment(p_fixed + q_offsets[k], qs[k]) += d_q;
            }
        }

        // 2. Add X'R^{-1}X and Z'R^{-1}Z diagonal
        for (int i = 0; i < n_records; ++i) {
            const auto& nz = fd.rows[i];
            for (const auto& pair : nz) {
                int c = pair.first;
                double v = pair.second;
                diag(c) += v * v;
            }

            for (size_t k = 0; k < components.size(); ++k) {
                int u_idx_local = -1;
                const auto& comp = components[k];
                if (!comp.id_map.empty()) {
                    u_idx_local = comp.id_map[i];
                } else if (recs[i].aid > 0) {
                    u_idx_local = recs[i].aid - 1;
                }

                if (u_idx_local >= 0 && u_idx_local < qs[k]) {
                    double cov_val = comp.covar_map.empty() ? 1.0 : comp.covar_map[i];
                    diag(p_fixed + q_offsets[k] + u_idx_local) += cov_val * cov_val;
                }
            }
        }
    }

    Eigen::MatrixXd getFixedEffectsBlock() const {
        Eigen::MatrixXd A11 = Eigen::MatrixXd::Zero(p_fixed, p_fixed);
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif
        std::vector<Eigen::MatrixXd> thread_A11(num_threads, Eigen::MatrixXd::Zero(p_fixed, p_fixed));

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            auto& local_A11 = thread_A11[tid];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int i = 0; i < n_records; ++i) {
                const auto& nz = fd.rows[i];
                for (const auto& c1 : nz) {
                    if (c1.first >= p_fixed) continue;
                    for (const auto& c2 : nz) {
                        if (c2.first >= p_fixed) continue;
                        local_A11(c1.first, c2.first) += c1.second * c2.second;
                    }
                }
            }
        }

        for (int t = 0; t < num_threads; ++t) {
            A11 += thread_A11[t];
        }
        return A11;
    }

    // Extract a random effects block [start, start+size) (relative to u)
    Eigen::MatrixXd getRandomBlock(int start_u_idx, int size) const {
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(size, size);

        // Find which component this block belongs to
        int comp_idx = -1;
        for (size_t k = 0; k < components.size(); ++k) {
            if (start_u_idx >= q_offsets[k] && start_u_idx < q_offsets[k] + qs[k]) {
                comp_idx = k;
                break;
            }
        }

        if (comp_idx != -1) {
            const auto& comp = components[comp_idx];
            int local_start = start_u_idx - q_offsets[comp_idx];
            if (comp.Qinv) {
                B = comp.Qinv->getBlock(local_start, size);
                B *= lambdas[comp_idx];
            }
        }

        for(int i=0; i<size; ++i) {
            int global_u = start_u_idx + i;
            if (global_u < q_total) {
                B(i, i) = diag(p_fixed + global_u);
            }
        }

        return B;
    }

    // Optimized multiply (formerly perform_op)
    // Computes y = A * x where A is the MME matrix
    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
        y.setZero();

        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#endif

        std::vector<Eigen::VectorXd> thread_y_beta(num_threads, Eigen::VectorXd::Zero(p_fixed));
        std::vector<Eigen::VectorXd> thread_y_u;

        bool use_thread_local_u = ((long long)q_total * 8 * num_threads) < (1024LL * 1024 * 1024 * 4); // 4GB limit

        if (use_thread_local_u) {
            thread_y_u.resize(num_threads, Eigen::VectorXd::Zero(q_total));
        }

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif

            // Iterate records
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int i = 0; i < n_records; ++i) {
                double val = 0.0;

                // 1. Compute y_hat[i] = X[i]*beta + Z[i]*u
                const auto& nz = fd.rows[i];
                for (const auto& pair : nz) {
                    val += pair.second * x(pair.first);
                }

                for (size_t k = 0; k < components.size(); ++k) {
                    int u_idx_local = -1;
                    const auto& comp = components[k];
                    if (!comp.id_map.empty()) {
                        u_idx_local = comp.id_map[i];
                    } else if (recs[i].aid > 0) {
                        u_idx_local = recs[i].aid - 1;
                    }

                    if (u_idx_local >= 0 && u_idx_local < qs[k]) {
                        int u_idx_global = p_fixed + q_offsets[k] + u_idx_local;
                        double cov_val = comp.covar_map.empty() ? 1.0 : comp.covar_map[i];
                        val += cov_val * x(u_idx_global);
                    }
                }

                // 2. Scatter to y_beta (thread local)
                for (const auto& pair : nz) {
                    thread_y_beta[tid](pair.first) += pair.second * val;
                }

                // 3. Scatter to y_u
                for (size_t k = 0; k < components.size(); ++k) {
                    int u_idx_local = -1;
                    const auto& comp = components[k];
                    if (!comp.id_map.empty()) {
                        u_idx_local = comp.id_map[i];
                    } else if (recs[i].aid > 0) {
                        u_idx_local = recs[i].aid - 1;
                    }

                    if (u_idx_local >= 0 && u_idx_local < qs[k]) {
                        int u_idx_global = p_fixed + q_offsets[k] + u_idx_local;
                        double cov_val = comp.covar_map.empty() ? 1.0 : comp.covar_map[i];

                        if (use_thread_local_u) {
                            thread_y_u[tid](q_offsets[k] + u_idx_local) += cov_val * val;
                        } else {
#ifdef _OPENMP
#pragma omp atomic
#endif
                            y(u_idx_global) += cov_val * val;
                        }
                    }
                }
            }
        }

        // Reduce thread_y_beta
        for (int t = 0; t < num_threads; ++t) {
            y.head(p_fixed) += thread_y_beta[t];
        }

        // Reduce thread_y_u if used
        if (use_thread_local_u) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < q_total; ++i) {
                double sum = 0.0;
                for (int t = 0; t < num_threads; ++t) {
                    sum += thread_y_u[t](i);
                }
                y(p_fixed + i) += sum;
            }
        }

        // Add Regularization terms: lambda * Qinv * u
        for (size_t k = 0; k < components.size(); ++k) {
            const auto& comp = components[k];
            if (comp.Qinv) {
                Eigen::VectorXd u_k = x.segment(p_fixed + q_offsets[k], qs[k]);
                Eigen::VectorXd Qu_k = comp.Qinv->operator*(u_k);
                y.segment(p_fixed + q_offsets[k], qs[k]) += lambdas[k] * Qu_k;
            }
        }
    }
};

} // namespace cosmic
