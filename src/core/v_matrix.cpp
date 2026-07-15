#include "v_matrix.h"
#include "logger.h"
#include "dense_solver.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

namespace cosmic {

void VMatrixAI_REML::initialize() {
    if (initialized) return;

    if (config.verbose && config.log_stream) {
        *config.log_stream << "Initializing V-Matrix AI-REML Solver..." << std::endl;
    } else if (config.verbose) {
        std::cout << "Initializing V-Matrix AI-REML Solver..." << std::endl;
    }

    n = y.size();

    // Determine number of fixed effects (p)
    p = 0;
    if (fd_ptr) {
        for (int i = 0; i < n; ++i) {
            for (const auto& pr : fd_ptr->rows[i]) {
                if (pr.first >= p) p = pr.first + 1;
            }
        }

        X_dense = Eigen::MatrixXd::Zero(n, p);
        for (int i = 0; i < n; ++i) {
            for (const auto& pr : fd_ptr->rows[i]) {
                X_dense(i, pr.first) = pr.second;
            }
        }
    } else {
        throw std::runtime_error("VMatrixAI_REML requires FixedDesignG");
    }

    int c_comps = components.size();
    GZt.resize(c_comps);
    ZGZt.resize(c_comps);

    for (int k = 0; k < c_comps; ++k) {
        if (!components[k].Qinv) throw std::runtime_error("RandomComponent missing Qinv");
        int q_k = components[k].Qinv->rows();

        Eigen::MatrixXd Z_k = Eigen::MatrixXd::Zero(n, q_k);
        for (int i = 0; i < n; ++i) {
            int aid = -1;
            if (!components[k].id_map.empty()) {
                aid = components[k].id_map[i];
            } else if (recs_ptr) {
                aid = (*recs_ptr)[i].aid - 1;
            }
            if (aid >= 0 && aid < q_k) {
                double val = 1.0;
                if (!components[k].covar_map.empty()) {
                    val = components[k].covar_map[i];
                }
                Z_k(i, aid) = val;
            }
        }

        if (config.verbose) {
            std::cout << "  [V-Matrix] Computing G*Z' for component " << k + 1 << " (dim " << q_k << " x " << n << ")..." << std::endl;
        }

        // Assemble Qinv as a sparse matrix directly
        std::vector<Eigen::Triplet<double>> triplets;
        components[k].Qinv->visit_triplets([&](int r, int c, double v) {
            // Ensure we only process the lower triangle to avoid duplicating if the matrix provides both
            if (r >= c) {
                triplets.emplace_back(r, c, v);
                if (r > c) {
                    triplets.emplace_back(c, r, v); // explicitly add upper part for SimplicialLDLT
                }
            }
        });

        Eigen::SparseMatrix<double> Qinv_sparse(q_k, q_k);
        Qinv_sparse.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::MatrixXd G_k_Zt;

        // Attempt sparse factorization first (much faster and memory efficient for pedigrees)
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(Qinv_sparse);

        if (solver.info() == Eigen::Success) {
            G_k_Zt = solver.solve(Z_k.transpose());
        } else {
            // Fallback to Dense (for Dense GRMs)
            if (config.verbose) {
                std::cout << "  [V-Matrix] Sparse decomposition failed/unsuitable, falling back to dense matrix inversion..." << std::endl;
            }
            Eigen::MatrixXd Qinv_dense = Eigen::MatrixXd(Qinv_sparse);
            Eigen::MatrixXd G_k;
            Eigen::LLT<Eigen::MatrixXd> llt(Qinv_dense);
            if (llt.info() == Eigen::Success) {
                G_k = llt.solve(Eigen::MatrixXd::Identity(q_k, q_k));
            } else {
                G_k = Qinv_dense.inverse();
            }
            G_k_Zt = G_k * Z_k.transpose();
        }

        GZt[k] = G_k_Zt;
        ZGZt[k] = Z_k * G_k_Zt;
    }

    initialized = true;
}

bool VMatrixAI_REML::run_aireml_step(int iter) {
    int c_comps = vars_u.size();

    // 1. Build V matrix
    Eigen::MatrixXd V = Eigen::MatrixXd::Identity(n, n) * var_e;
    for (int k = 0; k < c_comps; ++k) {
        V += ZGZt[k] * vars_u[k];
    }

    // 2. Invert V using Five-Tier DenseSolver
    DenseSolver solver(config.dense_tier);
    if (!solver.compute(V)) {
        if (config.verbose) std::cout << "  [V-Matrix] DenseSolver compute failed." << std::endl;
        return false;
    }

    Eigen::MatrixXd V_inv = Eigen::MatrixXd::Zero(n, n);
    // Construct explicit inverse for the rest of the exact logic
    // For PCG_SLQ or LowRankSVD, we can still form the inverse column by column if needed
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd ei = Eigen::VectorXd::Zero(n);
        ei(i) = 1.0;
        V_inv.col(i) = solver.solve(ei);
    }

    double log_det_V = solver.logDeterminant();

    // 3. Build P matrix
    Eigen::MatrixXd Xt_Vinv = X_dense.transpose() * V_inv;
    Eigen::MatrixXd Xt_Vinv_X = Xt_Vinv * X_dense;

    Eigen::MatrixXd XtVinvX_inv;
    double log_det_XtVinvX = 0.0;

    if (p > 0) {
        Eigen::LLT<Eigen::MatrixXd> llt_XtVinvX(Xt_Vinv_X);
        if (llt_XtVinvX.info() != Eigen::Success) return false;

        XtVinvX_inv = llt_XtVinvX.solve(Eigen::MatrixXd::Identity(p, p));
        for (int i = 0; i < p; ++i) log_det_XtVinvX += 2.0 * std::log(llt_XtVinvX.matrixL()(i, i));
    }

    Eigen::MatrixXd P = V_inv;
    if (p > 0) {
        P -= Xt_Vinv.transpose() * XtVinvX_inv * Xt_Vinv;
    }

    Eigen::VectorXd Py = P * y;

    // 4. Compute Log Likelihood
    double yPy = y.dot(Py);
    double current_logL = -0.5 * (log_det_V + log_det_XtVinvX + yPy);
    last_logL = current_logL;

    last_P = P;
    last_Py = Py;
    last_XtVinvX_inv = XtVinvX_inv;
    last_Xt_Vinv = Xt_Vinv;

    if (config.verbose) {
        if (config.log_stream) {
            *config.log_stream << "  [V-Matrix Iter " << iter << "] LogL: " << std::fixed << std::setprecision(4) << current_logL << std::endl;
        } else {
            std::cout << "  [V-Matrix Iter " << iter << "] LogL: " << std::fixed << std::setprecision(4) << current_logL << std::endl;
        }
    }

    // 5. Compute Gradients and AI Matrix
    int num_params = c_comps + 1; // c variance components + 1 residual variance
    Eigen::VectorXd grad = Eigen::VectorXd::Zero(num_params);
    Eigen::MatrixXd AI = Eigen::MatrixXd::Zero(num_params, num_params);

    std::vector<Eigen::MatrixXd> dV(num_params);
    for (int k = 0; k < c_comps; ++k) dV[k] = ZGZt[k];
    dV[c_comps] = Eigen::MatrixXd::Identity(n, n);

    std::vector<Eigen::MatrixXd> P_dV(num_params);
    for (int k = 0; k < num_params; ++k) {
        P_dV[k] = P * dV[k];
        double tr_P_dV = P_dV[k].trace();
        double yP_dV_Py = Py.dot(dV[k] * Py);
        grad(k) = -0.5 * tr_P_dV + 0.5 * yP_dV_Py;
    }

    for (int i = 0; i < num_params; ++i) {
        for (int j = i; j < num_params; ++j) {
            double val = 0.5 * Py.dot(dV[i] * P_dV[j] * Py);
            AI(i, j) = val;
            AI(j, i) = val;
        }
    }

    last_AI = AI;

    // 6. Update Variances
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AI);
    Eigen::VectorXd evals = es.eigenvalues();
    for (int i = 0; i < evals.size(); ++i) {
        if (evals(i) < 1e-8) evals(i) = 1e-8;
    }
    Eigen::MatrixXd AI_inv = es.eigenvectors() * evals.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();

    Eigen::VectorXd delta = AI_inv * grad;

    // 7. Step size control (damping)
    double step_size = 1.0;
    bool step_ok = false;
    for (int step = 0; step < 5; ++step) {
        bool all_positive = true;
        for (int k = 0; k < c_comps; ++k) {
            if (vars_u[k] + step_size * delta(k) <= 1e-8) all_positive = false;
        }
        if (var_e + step_size * delta(c_comps) <= 1e-8) all_positive = false;

        if (all_positive) {
            step_ok = true;
            break;
        }
        step_size *= 0.5;
    }

    if (!step_ok) {
        // Force positive
        for (int k = 0; k < c_comps; ++k) {
            vars_u[k] = std::max(1e-6, vars_u[k] + step_size * delta(k));
        }
        var_e = std::max(1e-6, var_e + step_size * delta(c_comps));
    } else {
        for (int k = 0; k < c_comps; ++k) vars_u[k] += step_size * delta(k);
        var_e += step_size * delta(c_comps);
    }

    double max_delta = delta.cwiseAbs().maxCoeff() * step_size;
    last_diff = max_delta;

    return true;
}

void VMatrixAI_REML::solve() {
    if (!initialized) initialize();

    if (config.verbose) {
        if (config.log_stream) {
            *config.log_stream << "\nStarting V-Matrix AI-REML Optimization..." << std::endl;
            *config.log_stream << "  Individuals (n): " << n << std::endl;
            *config.log_stream << "  Fixed Effects (p): " << p << std::endl;
            *config.log_stream << "  Random Components (c): " << vars_u.size() << std::endl;
        } else {
            std::cout << "\nStarting V-Matrix AI-REML Optimization..." << std::endl;
            std::cout << "  Individuals (n): " << n << std::endl;
            std::cout << "  Fixed Effects (p): " << p << std::endl;
            std::cout << "  Random Components (c): " << vars_u.size() << std::endl;
        }
    }

    converged = false;
    for (int iter = 1; iter <= config.max_iter; ++iter) {
        iterations_run = iter;
        bool ok = run_aireml_step(iter);
        if (!ok) {
            std::cerr << "  [V-Matrix] Step failed at iteration " << iter << std::endl;
            break;
        }

        if (config.verbose) {
            std::cout << "  Iter " << iter << ": Ve = " << var_e;
            for (size_t k = 0; k < vars_u.size(); ++k) {
                std::cout << ", Vu[" << k << "] = " << vars_u[k];
            }
            std::cout << " (Diff = " << last_diff << ")" << std::endl;
        }

        if (last_diff < config.tol) {
            converged = true;
            if (config.verbose) std::cout << "  [V-Matrix] Converged!" << std::endl;
            break;
        }
    }

    if (converged) {
        compute_blup();
    }

    calculate_SE();
}

void VMatrixAI_REML::compute_blup() {
        if (config.verbose) {
            if (config.log_stream) *config.log_stream << "  [V-Matrix] Computing BLUP solutions (GEBV/fixed effects)..." << std::endl;
            else std::cout << "  [V-Matrix] Computing BLUP solutions (GEBV/fixed effects)..." << std::endl;
        }

        // beta = (X' V^{-1} X)^{-1} X' V^{-1} y
        Eigen::VectorXd beta;
        if (p > 0) {
            beta = last_XtVinvX_inv * last_Xt_Vinv * y;
        } else {
            beta = Eigen::VectorXd::Zero(0);
        }

        int c_comps = vars_u.size();
        int q_total = 0;
        for (int k = 0; k < c_comps; ++k) {
            q_total += components[k].Qinv->rows();
        }

        final_solution.resize(p + q_total);
        if (p > 0) {
            final_solution.head(p) = beta;
        }

        int offset = p;
        for (int k = 0; k < c_comps; ++k) {
            // u_k = sigma2_k * G_k * Z_k' * P * y
            Eigen::VectorXd u_k = vars_u[k] * GZt[k] * last_Py;
            int q_k = u_k.size();
            final_solution.segment(offset, q_k) = u_k;
            offset += q_k;
        }
    }

void VMatrixAI_REML::calculate_SE() {
    if (last_AI.rows() > 0) {
        Eigen::MatrixXd AI_inv = last_AI.inverse();
        int c_comps = vars_u.size();
        vars_u_se.resize(c_comps);
        for (int k = 0; k < c_comps; ++k) {
            vars_u_se[k] = std::sqrt(std::max(0.0, AI_inv(k, k)));
        }
        var_e_se = std::sqrt(std::max(0.0, AI_inv(c_comps, c_comps)));
    }
}

}