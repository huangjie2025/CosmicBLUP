#include "mixed_model_solver.h"
#include "logger.h"
#include "mme_builder.h"
#include "pcg_solver.h"
#include "matrix_free_pcg.h"
#include "implicit_mme.h"
#include "matrix_adapter.h"
#include "plink_reader.h"
#include "pgen_reader.h"
#include <algorithm>
#include <cmath>

// Kronecker product helper (avoids Eigen unsupported module dependency)
static Eigen::MatrixXd kron(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    int m = A.rows(), n = A.cols(), p = B.rows(), q = B.cols();
    Eigen::MatrixXd C(m * p, n * q);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C.block(i * p, j * q, p, q) = A(i, j) * B;
    return C;
}

#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
#define COSMICBLUP_HAS_WEAK_OPENBLAS 1
extern "C" void openblas_set_num_threads(int num_threads) __attribute__((weak));
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

using namespace std;
using namespace Eigen;

SolveResult GSSolve::solve(const Context& ctx) {
    SolveResult result;
    const auto& recs = ctx.recs;
    const auto& fd = ctx.fd;
    const auto& cfg = ctx.cfg;
    const auto& vce_result = ctx.vce_result;

    double sigma2_e = ctx.run_vce ? vce_result.sigma2_e : ctx.sigma2_e_init;
    double sigma2_u = (ctx.run_vce && !vce_result.vars_u.empty()) ? vce_result.vars_u[0] : ctx.sigma2_u_init;

    // Build MME components
    std::vector<RandomComponent>& mme_components = result.mme_components;
    int pe_component_idx = -1;

    if (ctx.stcg_mode && ctx.stcg_ga_dim) {
        RandomComponent rc;
        rc.Qinv = ctx.stcg_ga_dim;
        mme_components.push_back(rc);
        if (ctx.PEinv) {
            RandomComponent rc_pe;
            rc_pe.Qinv = ctx.PEinv;
            pe_component_idx = (int)mme_components.size();
            mme_components.push_back(rc_pe);
        }
    } else if (cfg.rrm_model) {
        mme_components = ctx.base_components;
    } else {
        if (ctx.Qinv) { RandomComponent rc; rc.Qinv = ctx.Qinv; mme_components.push_back(rc); }
        if (ctx.GDinv) { RandomComponent rc; rc.Qinv = ctx.GDinv; mme_components.push_back(rc); }
        if (ctx.GEinv) { RandomComponent rc; rc.Qinv = ctx.GEinv; mme_components.push_back(rc); }
        if (ctx.PEinv) {
            RandomComponent rc;
            rc.Qinv = ctx.PEinv;
            pe_component_idx = (int)mme_components.size();
            mme_components.push_back(rc);
        }
        if (cfg.mat_effect) { RandomComponent rc; rc.Qinv = ctx.Qinv; rc.id_map = ctx.mat_id_map; mme_components.push_back(rc); }
        for (size_t i = 0; i < ctx.generic_rand_invs.size(); ++i) {
            RandomComponent rc;
            rc.Qinv = ctx.generic_rand_invs[i].get();
            rc.id_map = ctx.generic_rand_maps[i];
            mme_components.push_back(rc);
        }
    }

    bool build_matrix = !ctx.force_implicit;

    // Compute lambdas
    std::vector<double>& mme_lambdas = result.mme_lambdas;
    if (ctx.run_vce && !vce_result.vars_u.empty()) {
        for (double vu : vce_result.vars_u) {
            mme_lambdas.push_back((vu > 1e-9) ? (sigma2_e / vu) : 0.0);
        }
    } else {
        std::vector<double> user_vars(mme_components.size(), sigma2_u);
        if (cfg.var_priors.size() > 1) {
            for (size_t k = 0; k < std::min(mme_components.size(), cfg.var_priors.size() - 1); ++k) {
                user_vars[k] = cfg.var_priors[k];
            }
        }
        for (double vu : user_vars) {
            mme_lambdas.push_back((vu > 1e-9) ? (sigma2_e / vu) : 0.0);
        }
    }

    // RRM block-K setup
    Eigen::MatrixXd rrm_mme_lambda;
    if (cfg.rrm_model && !mme_components.empty()) {
        result.rrm_additive_K = Eigen::MatrixXd::Zero(ctx.rrm_additive_component_count, ctx.rrm_additive_component_count);
        result.rrm_pe_K = Eigen::MatrixXd::Zero(ctx.rrm_pe_component_count, ctx.rrm_pe_component_count);
        for (int k = 0; k < ctx.rrm_additive_component_count; ++k) {
            double diag_val = (k < (int)mme_lambdas.size() && mme_lambdas[k] > 1e-12) ? (sigma2_e / mme_lambdas[k]) : 0.0;
            result.rrm_additive_K(k, k) = diag_val;
        }
        for (int k = 0; k < ctx.rrm_pe_component_count; ++k) {
            int comp_idx = ctx.rrm_additive_component_count + k;
            double diag_val = (comp_idx < (int)mme_lambdas.size() && mme_lambdas[comp_idx] > 1e-12) ? (sigma2_e / mme_lambdas[comp_idx]) : 0.0;
            result.rrm_pe_K(k, k) = diag_val;
        }
    }

    MMELHSBuilder mme_builder(recs, fd, mme_components, build_matrix);

    VectorXd rhs = mme_builder.build_rhs(recs, fd, mme_builder.get_dim());
    VectorXd sol;
    std::unique_ptr<SimplicialLDLT<SparseMatrix<double>>> direct_solver;
    bool used_pe_absorption = false;
    int reduced_dim = mme_builder.get_dim();

    if (ctx.stcg_mode && vce_result.solution.size() == rhs.size()) {
        sol = vce_result.solution;
        build_matrix = false;
        LOG_INFO("Using STCG V-implicit solution for BLUE/BLUP; skipping explicit MME solve.");
    } else if (build_matrix) {
        SparseMatrix<double> lhs = mme_builder.build_lhs(mme_lambdas);

        bool can_absorb_pe = (pe_component_idx >= 0
                              && pe_component_idx < (int)mme_components.size()
                              && pe_component_idx < (int)mme_lambdas.size());
        if (can_absorb_pe) {
            const auto& qs = mme_builder.get_qs();
            int pe_start = mme_builder.get_p();
            for (int i = 0; i < pe_component_idx; ++i) pe_start += qs[i];
            int pe_dim = qs[pe_component_idx];
            int full_dim = mme_builder.get_dim();

            std::vector<int> full_to_reduced(full_dim, -1);
            std::vector<int> reduced_to_full;
            reduced_to_full.reserve(full_dim - pe_dim);
            for (int idx = 0; idx < full_dim; ++idx) {
                if (idx < pe_start || idx >= pe_start + pe_dim) {
                    full_to_reduced[idx] = (int)reduced_to_full.size();
                    reduced_to_full.push_back(idx);
                }
            }

            VectorXd rhs_reduced(reduced_to_full.size());
            for (size_t i = 0; i < reduced_to_full.size(); ++i) rhs_reduced((int)i) = rhs(reduced_to_full[i]);

            std::vector<Triplet<double>> reduced_trips;
            reduced_trips.reserve(lhs.nonZeros());

            for (int col = 0; col < lhs.outerSize(); ++col) {
                if (col >= pe_start && col < pe_start + pe_dim) continue;
                int new_col = full_to_reduced[col];
                for (SparseMatrix<double>::InnerIterator it(lhs, col); it; ++it) {
                    int row = it.row();
                    if (row >= pe_start && row < pe_start + pe_dim) continue;
                    reduced_trips.emplace_back(full_to_reduced[row], new_col, it.value());
                }
            }

            std::vector<std::vector<std::pair<int, double>>> pe_columns(pe_dim);
            std::vector<double> pe_diag(pe_dim, 0.0);
            for (int local_col = 0; local_col < pe_dim; ++local_col) {
                int col = pe_start + local_col;
                for (SparseMatrix<double>::InnerIterator it(lhs, col); it; ++it) {
                    int row = it.row();
                    double val = it.value();
                    if (row == col) {
                        pe_diag[local_col] = val;
                    } else if (row < pe_start || row >= pe_start + pe_dim) {
                        pe_columns[local_col].push_back({full_to_reduced[row], val});
                    }
                }
            }

            for (int j = 0; j < pe_dim; ++j) {
                double djj = pe_diag[j];
                if (std::abs(djj) < 1e-12) {
                    throw std::runtime_error("Pe absorption failed: singular diagonal block encountered.");
                }

                double rhs_pe_j = rhs(pe_start + j);
                for (const auto& entry : pe_columns[j]) {
                    rhs_reduced(entry.first) -= (entry.second / djj) * rhs_pe_j;
                }

                const auto& col_entries = pe_columns[j];
                for (size_t a = 0; a < col_entries.size(); ++a) {
                    int ra = col_entries[a].first;
                    double va = col_entries[a].second;
                    for (size_t b = a; b < col_entries.size(); ++b) {
                        int rb = col_entries[b].first;
                        double vb = col_entries[b].second;
                        double update = -(va * vb) / djj;
                        reduced_trips.emplace_back(ra, rb, update);
                        if (ra != rb) reduced_trips.emplace_back(rb, ra, update);
                    }
                }
            }

            SparseMatrix<double> lhs_reduced((int)reduced_to_full.size(), (int)reduced_to_full.size());
            lhs_reduced.setFromTriplets(reduced_trips.begin(), reduced_trips.end());
            lhs_reduced.makeCompressed();

            direct_solver = std::make_unique<SimplicialLDLT<SparseMatrix<double>>>();
            direct_solver->compute(lhs_reduced);
            if (direct_solver->info() != Success) throw runtime_error("Reduced Direct Solver Decomposition Failed");

            VectorXd sol_reduced = direct_solver->solve(rhs_reduced);
            sol = VectorXd::Zero(full_dim);
            for (size_t i = 0; i < reduced_to_full.size(); ++i) sol(reduced_to_full[i]) = sol_reduced((int)i);

            for (int j = 0; j < pe_dim; ++j) {
                double rhs_back = rhs(pe_start + j);
                for (const auto& entry : pe_columns[j]) {
                    rhs_back -= entry.second * sol_reduced(entry.first);
                }
                sol(pe_start + j) = rhs_back / pe_diag[j];
            }

            used_pe_absorption = true;
            reduced_dim = (int)reduced_to_full.size();
            LOG_INFO("Applied Pe absorption in final MME solve: dimension reduced from "
                     << full_dim << " to " << reduced_dim << ".");
        } else {
            direct_solver = std::make_unique<SimplicialLDLT<SparseMatrix<double>>>();
            direct_solver->compute(lhs);
            if (direct_solver->info() != Success) throw runtime_error("Direct Solver Decomposition Failed");
            sol = direct_solver->solve(rhs);
        }
    } else {
        if (cfg.rrm_model) {
            LOG_WARN("RRM block-K refinement currently requires explicit MME construction; matrix-free path uses independent coefficient penalties.");
        }
        ImplicitMME implicit_mme(recs, fd, mme_components, mme_lambdas);
        MatrixFreePCG pcg(rhs.size());
        pcg.setMultOp([&](const VectorXd& v, VectorXd& out) { implicit_mme.multiply(v, out); });

        VectorXd diag = implicit_mme.diag;
        pcg.setPrecondOp([&](const VectorXd& v, VectorXd& out) { out = v.cwiseQuotient(diag); });

        if (vce_result.solution.size() == rhs.size()) {
            LOG_INFO("Using VCE solution as warm start for PCG...");
            sol = pcg.solve(rhs, cfg.pcg_tol, cfg.pcg_maxit, vce_result.solution);
        } else {
            sol = pcg.solve(rhs, cfg.pcg_tol, cfg.pcg_maxit);
        }
    }

    // RRM block-K refinement
    if (cfg.rrm_model && build_matrix && !ctx.stcg_mode) {
        auto estimate_rrm_K = [&](const VectorXd& u_vec, int start_idx, int comp_count) -> Eigen::MatrixXd {
            Eigen::MatrixXd K = Eigen::MatrixXd::Zero(comp_count, comp_count);
            if (comp_count <= 0) return K;

            std::vector<int> q_dims(comp_count, 0);
            int q_common = 0;
            bool consistent = true;
            for (int k = 0; k < comp_count; ++k) {
                int comp_idx = start_idx + k;
                if (comp_idx >= (int)mme_components.size() || !mme_components[comp_idx].Qinv) {
                    consistent = false;
                    break;
                }
                q_dims[k] = mme_components[comp_idx].Qinv->rows();
                if (k == 0) q_common = q_dims[k];
                else if (q_dims[k] != q_common) consistent = false;
            }
            if (!consistent || q_common <= 0) return K;

            for (int i_ind = 0; i_ind < q_common; ++i_ind) {
                Eigen::VectorXd coef = Eigen::VectorXd::Zero(comp_count);
                int offset = 0;
                for (int comp = 0; comp < start_idx; ++comp) offset += mme_components[comp].Qinv->rows();
                for (int k = 0; k < comp_count; ++k) {
                    int q_k = q_dims[k];
                    if (i_ind < q_k) coef(k) = u_vec(offset + i_ind);
                    offset += q_k;
                }
                K.noalias() += coef * coef.transpose();
            }
            K /= std::max(1, q_common);

            Eigen::MatrixXd S = 0.5 * (K + K.transpose());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
            if (es.info() == Eigen::Success) {
                Eigen::VectorXd eval = es.eigenvalues();
                double max_eval = std::max(eval.maxCoeff(), 0.0);
                double floor = std::max(1e-10, 1e-8 * std::max(1.0, max_eval));
                for (int i = 0; i < eval.size(); ++i) eval(i) = std::max(eval(i), floor);
                return es.eigenvectors() * eval.asDiagonal() * es.eigenvectors().transpose();
            }

            for (int i = 0; i < comp_count; ++i) K(i, i) = std::max(K(i, i), 1e-8);
            return K;
        };

        VectorXd u_stage1 = sol.tail(sol.size() - fd.p);
        if (ctx.rrm_additive_component_count > 0) {
            Eigen::MatrixXd estimated = estimate_rrm_K(u_stage1, 0, ctx.rrm_additive_component_count);
            if (estimated.rows() == ctx.rrm_additive_component_count) result.rrm_additive_K = estimated;
        }
        if (ctx.rrm_pe_component_count > 0) {
            Eigen::MatrixXd estimated = estimate_rrm_K(u_stage1, ctx.rrm_additive_component_count, ctx.rrm_pe_component_count);
            if (estimated.rows() == ctx.rrm_pe_component_count) result.rrm_pe_K = estimated;
        }

        rrm_mme_lambda = Eigen::MatrixXd::Zero((int)mme_components.size(), (int)mme_components.size());
        if (ctx.rrm_additive_component_count > 0) {
            rrm_mme_lambda.block(0, 0, ctx.rrm_additive_component_count, ctx.rrm_additive_component_count) =
                sigma2_e * result.rrm_additive_K.inverse();
        }
        if (ctx.rrm_pe_component_count > 0) {
            rrm_mme_lambda.block(ctx.rrm_additive_component_count, ctx.rrm_additive_component_count,
                                 ctx.rrm_pe_component_count, ctx.rrm_pe_component_count) =
                sigma2_e * result.rrm_pe_K.inverse();
        }
        result.use_rrm_lambda_matrix = true;

        SparseMatrix<double> lhs_rrm = mme_builder.build_lhs(rrm_mme_lambda);
        direct_solver = std::make_unique<SimplicialLDLT<SparseMatrix<double>>>();
        direct_solver->compute(lhs_rrm);
        if (direct_solver->info() != Success) {
            throw runtime_error("RRM block-K direct solver decomposition failed");
        }
        sol = direct_solver->solve(rhs);
        used_pe_absorption = false;
        reduced_dim = mme_builder.get_dim();
        LOG_INFO("RRM final MME refined with block coefficient covariance matrices (K).");
    }

    // Extract solutions
    int p = fd.p;
    result.beta = sol.head(p);
    result.u = sol.tail(sol.size() - p);

    // Calculate SE if requested
    if (cfg.calc_se) {
        if (direct_solver && direct_solver->info() == Success) {
            result.Cinv_beta.resize(p, p);
            for (int i = 0; i < p; ++i) {
                VectorXd ei = VectorXd::Zero(used_pe_absorption ? reduced_dim : mme_builder.get_dim());
                ei(i) = 1.0;
                VectorXd Cinv_ei = direct_solver->solve(ei);
                for (int j = 0; j < p; ++j) result.Cinv_beta(j, i) = Cinv_ei(j);
                double Cinv_ii = Cinv_ei(i);
                result.beta_se.resize(p);
                result.beta_se(i) = std::sqrt(std::max(0.0, sigma2_e * Cinv_ii));
            }
            result.se_calculated = true;
            result.cov_calculated = true;
        } else {
            LOG_INFO("Using PCG to compute SE for " << p << " fixed effects...");
            if (result.use_rrm_lambda_matrix) {
                LOG_WARN("RRM block-K mode currently skips PCG-based SE calculation for fixed effects.");
            } else {
                ImplicitMME implicit_mme(recs, fd, mme_components, mme_lambdas);
                VectorXd diag = implicit_mme.diag;

                int old_threads = 1;
#ifdef _OPENMP
                old_threads = omp_get_max_threads();
                omp_set_num_threads(1);
#endif
#ifdef COSMICBLUP_HAS_WEAK_OPENBLAS
                if (openblas_set_num_threads) openblas_set_num_threads(1);
#endif

                result.Cinv_beta.resize(p, p);
                result.beta_se.resize(p);

#ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic) num_threads(old_threads)
#endif
                for (int i = 0; i < p; ++i) {
                    VectorXd ei = VectorXd::Zero(mme_builder.get_dim());
                    ei(i) = 1.0;

                    MatrixFreePCG pcg(mme_builder.get_dim());
                    pcg.setQuiet(true);
                    pcg.setMultOp([&](const VectorXd& v, VectorXd& out) { implicit_mme.multiply(v, out); });
                    pcg.setPrecondOp([&](const VectorXd& v, VectorXd& out) { out = v.cwiseQuotient(diag); });

                    VectorXd Cinv_ei = pcg.solve(ei, cfg.pcg_tol, cfg.pcg_maxit);

#ifdef _OPENMP
                    #pragma omp critical
#endif
                    {
                        for (int j = 0; j < p; ++j) result.Cinv_beta(j, i) = Cinv_ei(j);
                        double Cinv_ii = Cinv_ei(i);
                        result.beta_se(i) = std::sqrt(std::max(0.0, sigma2_e * Cinv_ii));
                        if ((i + 1) % 10 == 0 || i == p - 1) {
                            LOG_INFO("  ... " << (i + 1) << "/" << p << " completed.");
                        }
                    }
                }
#ifdef _OPENMP
                omp_set_num_threads(old_threads);
#endif
#ifdef COSMICBLUP_HAS_WEAK_OPENBLAS
                if (openblas_set_num_threads) openblas_set_num_threads(old_threads);
#endif

                result.se_calculated = true;
                result.cov_calculated = true;
            }
        }
    }

    return result;
}

// ============================================================================
// Multi-trait GBLUP MME solve
// ============================================================================
SolveResult GSSolve::solveMultiTrait(
    const VCEResult& vce,
    const Eigen::MatrixXd& G_inv,
    const std::map<std::string, int>& idmap,
    const Config& cfg) {

    SolveResult result;
    const MatrixXd& Vg = vce.mv_Vg;
    const MatrixXd& Ve = vce.mv_Ve;
    const MatrixXd& Y = vce.mv_Y;
    const MatrixXd& X = vce.mv_X;

    int n = Y.rows();
    int t = Y.cols();
    int p = X.cols();
    int nt = n * t;
    int pt = p * t;

    LOG_INFO("Multi-trait MME solve: n=" << n << " t=" << t << " p=" << p
              << " (MME dimension: " << (nt + pt) << ")");

    // Slice G_inv to the retained individuals (matching VCE's complete-case set).
    MatrixXd G_inv_used;
    if (!vce.mv_keep_indices.empty() && (int)vce.mv_keep_indices.size() == n) {
        G_inv_used.resize(n, n);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                G_inv_used(r, c) = G_inv(vce.mv_keep_indices[r], vce.mv_keep_indices[c]);
    } else {
        // No subsetting was performed (or legacy path): use full G_inv.
        if (G_inv.rows() != n || G_inv.cols() != n) {
            LOG_ERROR("Multi-trait MME: G_inv dimension (" << G_inv.rows() << "x" << G_inv.cols()
                      << ") does not match Y rows (" << n << ").");
            return result;
        }
        G_inv_used = G_inv;
    }

    // V_e^{-1} and V_g^{-1} (t×t)
    MatrixXd Ve_inv = Ve.inverse();
    MatrixXd Vg_inv = Vg.inverse();

    // y = vec(Y) in trait-major order: [Y.col(0); Y.col(1); ...; Y.col(t-1)]
    VectorXd y(nt);
    for (int j = 0; j < t; ++j) y.segment(j * n, n) = Y.col(j);

    // Build MME:
    // LHS = [ Ve_inv ⊗ X'X    |  Ve_inv ⊗ X'     ]
    //       [ Ve_inv ⊗ X       |  Ve_inv ⊗ I + Vg_inv ⊗ G_inv ]
    // RHS = [ Ve_inv ⊗ X' y ;  Ve_inv ⊗ y ]

    // Use Kronecker product to build LHS explicitly
    // C11 = Ve_inv ⊗ (X'X)   [pt × pt]
    MatrixXd XtX = X.transpose() * X;
    MatrixXd C11 = kron(Ve_inv, XtX);

    // C12 = Ve_inv ⊗ X'      [pt × nt]
    MatrixXd Xt = X.transpose();
    MatrixXd C12 = kron(Ve_inv, Xt);

    // C22 = Ve_inv ⊗ I_n + Vg_inv ⊗ G_inv  [nt × nt]
    MatrixXd C22 = kron(Ve_inv, MatrixXd::Identity(n, n))
                 + kron(Vg_inv, G_inv_used);

    // Assemble full LHS
    int dim = pt + nt;
    MatrixXd LHS(dim, dim);
    LHS.topLeftCorner(pt, pt) = C11;
    LHS.topRightCorner(pt, nt) = C12;
    LHS.bottomLeftCorner(nt, pt) = C12.transpose();
    LHS.bottomRightCorner(nt, nt) = C22;

    // RHS
    VectorXd RHS(dim);
    // RHS1 = (Ve_inv ⊗ X') y = vec(X' Y Ve_inv')  -- but y is trait-major
    // (Ve_inv ⊗ X') y: reshape y to n×t, compute X' Y Ve_inv, then vec
    MatrixXd Ymat = Y; // n×t
    MatrixXd RHS1_mat = Xt * Ymat * Ve_inv.transpose(); // p×t
    RHS.head(pt) = Map<VectorXd>(RHS1_mat.data(), pt);

    // RHS2 = (Ve_inv ⊗ I_n) y = vec(Y Ve_inv')  -- trait-major
    MatrixXd RHS2_mat = Ymat * Ve_inv.transpose(); // n×t
    RHS.tail(nt) = Map<VectorXd>(RHS2_mat.data(), nt);

    // Solve with LDLT
    LOG_INFO("Solving multi-trait MME (" << dim << "×" << dim << ") via dense LDLT...");
    Eigen::LDLT<MatrixXd> solver(LHS);
    if (solver.info() != Eigen::Success) {
        LOG_ERROR("Multi-trait MME factorization failed.");
        return result;
    }
    VectorXd sol = solver.solve(RHS);

    // Extract beta (pt×1) and u (nt×1)
    result.beta = sol.head(pt);
    result.u = sol.tail(nt);

    LOG_INFO("Multi-trait MME solve completed.");
    return result;
}

} // namespace cosmic
