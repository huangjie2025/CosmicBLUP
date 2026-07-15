#include "stcg_vce.h"
#include "slq_estimator.h"
#include "logger.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

static double safe_pos(double x) { return std::max(x, 1e-12); }
static double clamp_value(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

STCG_VCE::STCG_VCE(const std::vector<GenRecord>& recs_in,
                   const FixedDesignG& fd_in,
                   const std::vector<RandomComponent>& comps_in,
                   const Eigen::VectorXd& y_in,
                   VCEConfig cfg_in)
    : VCESolver(recs_in, fd_in, comps_in, y_in, cfg_in),
      recs(recs_in), fd(fd_in), comps(comps_in), y(y_in), cfg(std::move(cfg_in)) {
    n_records = (int)recs.size();
    p = fd.p;
    n_ind = 0;
    rec_to_ind.resize(n_records, -1);
    for (int i = 0; i < n_records; ++i) {
        int idx = recs[i].aid - 1;
        rec_to_ind[i] = idx;
        if (idx + 1 > n_ind) n_ind = idx + 1;
    }
    n_obs_per_ind.assign(n_ind, 0);
    for (int i = 0; i < n_records; ++i) {
        int idx = rec_to_ind[i];
        if (idx >= 0 && idx < n_ind) n_obs_per_ind[idx] += 1;
    }

    X_dense = Eigen::MatrixXd::Zero(n_records, p);
    for (int i = 0; i < n_records; ++i) {
        for (const auto& kv : fd.rows[i]) {
            if (kv.first >= 0 && kv.first < p) X_dense(i, kv.first) = kv.second;
        }
    }

    for (size_t k = 0; k < comps.size(); ++k) {
        if (comps[k].geno_mat) {
            genetic_idx = (int)k;
            continue;
        }
        const auto* ident = dynamic_cast<const IdentityMatrixAdapter*>(comps[k].Qinv);
        if (ident && comps[k].id_map.empty() && comps[k].covar_map.empty()) pe_idx = (int)k;
    }
    if (genetic_idx < 0) throw std::runtime_error("STCG_VCE requires a genotype matrix component.");
    if (pe_idx < 0) {
        for (size_t k = 0; k < comps.size(); ++k) {
            if ((int)k == genetic_idx) continue;
            if (!comps[k].covar_map.empty()) continue;
            if (!comps[k].Qinv || comps[k].Qinv->rows() < n_ind) continue;
            pe_idx = (int)k;
            break;
        }
    }
    if (pe_idx < 0) throw std::runtime_error("STCG_VCE requires a permanent-environment identity component.");
    geno = comps[genetic_idx].geno_mat;
    if (!geno) throw std::runtime_error("STCG_VCE genotype component not initialized.");
    if (geno->rows() < n_ind) throw std::runtime_error("STCG_VCE: genotype sample size is smaller than phenotype individual count.");

    #ifdef _OPENMP
    n_threads = std::max(1, omp_get_max_threads());
    #endif
}

void STCG_VCE::gather_to_ind(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_ind) {
    out_ind.setZero(geno->rows());
    for (int i = 0; i < n_records; ++i) {
        int idx = rec_to_ind[i];
        if (idx >= 0) out_ind(idx) += v_rec(i);
    }
}

void STCG_VCE::scatter_to_rec(const Eigen::VectorXd& v_ind, Eigen::VectorXd& out_rec) {
    out_rec.resize(n_records);
    for (int i = 0; i < n_records; ++i) {
        int idx = rec_to_ind[i];
        out_rec(i) = (idx >= 0) ? v_ind(idx) : 0.0;
    }
}

void STCG_VCE::apply_G(const Eigen::VectorXd& u, Eigen::VectorXd& out) {
    Eigen::VectorXd tmp;
    tmp.resize(geno->cols());
    geno->multiply_Zt_v(u, tmp, n_threads);
    out.resize(geno->rows());
    geno->multiply_Z_v(tmp, out, n_threads);
    out.array() /= (double)std::max(1, geno->cols());
}

void STCG_VCE::apply_dV_genetic(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_rec) {
    Eigen::VectorXd u_ind;
    gather_to_ind(v_rec, u_ind);
    Eigen::VectorXd gu;
    apply_G(u_ind, gu);
    scatter_to_rec(gu, out_rec);
}

void STCG_VCE::apply_dV_pe(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_rec) {
    Eigen::VectorXd u_ind;
    gather_to_ind(v_rec, u_ind);
    scatter_to_rec(u_ind, out_rec);
}

void STCG_VCE::apply_V(const Eigen::VectorXd& v, Eigen::VectorXd& out) {
    out.setZero(n_records);
    Eigen::VectorXd tmp(n_records);
    if (genetic_idx >= 0) {
        apply_dV_genetic(v, tmp);
        out.noalias() += vars_u[genetic_idx] * tmp;
    }
    if (pe_idx >= 0) {
        apply_dV_pe(v, tmp);
        out.noalias() += vars_u[pe_idx] * tmp;
    }
    out.noalias() += var_e * v;
}

Eigen::VectorXd STCG_VCE::solve_V(const Eigen::VectorXd& b, const Eigen::VectorXd& warm_start) {
    MatrixFreePCG pcg(n_records);
    pcg.setQuiet(true);
    pcg.setMultOp([&](const Eigen::VectorXd& v, Eigen::VectorXd& out) { apply_V(v, out); });
    double diag = safe_pos(var_e);
    if (genetic_idx >= 0) diag += safe_pos(vars_u[genetic_idx]);
    if (pe_idx >= 0) diag += safe_pos(vars_u[pe_idx]);
    pcg.setPrecondOp([&](const Eigen::VectorXd& r, Eigen::VectorXd& z) { z = r / diag; });
    return pcg.solve(b, cfg.pcg_tol, cfg.pcg_max_iter, warm_start);
}

void STCG_VCE::compute_Vinv_X_y(const Eigen::VectorXd& y_in,
                                Eigen::MatrixXd& Vinv_X,
                                Eigen::VectorXd& Vinv_y,
                                Eigen::MatrixXd& XtVinvX_inv,
                                Eigen::VectorXd& beta) {
    Vinv_y = solve_V(y_in);

    Vinv_X.resize(n_records, p);
    for (int j = 0; j < p; ++j) {
        Eigen::VectorXd col = X_dense.col(j);
        Vinv_X.col(j) = solve_V(col);
    }

    Eigen::MatrixXd XtVinvX = X_dense.transpose() * Vinv_X;
    Eigen::LDLT<Eigen::MatrixXd> xsolver(XtVinvX);
    XtVinvX_inv = xsolver.solve(Eigen::MatrixXd::Identity(p, p));
    Eigen::VectorXd XtVinvY = X_dense.transpose() * Vinv_y;
    beta = XtVinvX_inv * XtVinvY;
}

Eigen::VectorXd STCG_VCE::apply_P(const Eigen::VectorXd& v,
                                  const Eigen::MatrixXd& Vinv_X,
                                  const Eigen::MatrixXd& XtVinvX_inv) {
    Eigen::VectorXd Vinv_v = solve_V(v);
    Eigen::VectorXd XtVinvV = X_dense.transpose() * Vinv_v;
    Eigen::VectorXd coef = XtVinvX_inv * XtVinvV;
    return Vinv_v - Vinv_X * coef;
}

void STCG_VCE::solve() {
    vars_u.assign(comps.size(), 0.0);
    double mean_y = y.mean();
    double var_y = (y.array() - mean_y).square().sum() / std::max(1, n_records - 1);
    double var_floor = std::max(1e-10, var_y * 1e-6);

    std::vector<double> sum_y(n_ind, 0.0), sumsq_y(n_ind, 0.0);
    std::vector<int> cnt_y(n_ind, 0);
    for (int i = 0; i < n_records; ++i) {
        int idx = rec_to_ind[i];
        if (idx >= 0 && idx < n_ind) {
            sum_y[idx] += y(i);
            sumsq_y[idx] += y(i) * y(i);
            cnt_y[idx] += 1;
        }
    }

    double within_ss = 0.0;
    int within_df = 0;
    std::vector<double> subj_means;
    subj_means.reserve(n_ind);
    double avg_n = 0.0;
    for (int i = 0; i < n_ind; ++i) {
        if (cnt_y[i] <= 0) continue;
        double mean_i = sum_y[i] / cnt_y[i];
        subj_means.push_back(mean_i);
        avg_n += cnt_y[i];
        if (cnt_y[i] > 1) {
            within_ss += sumsq_y[i] - cnt_y[i] * mean_i * mean_i;
            within_df += cnt_y[i] - 1;
        }
    }
    avg_n = subj_means.empty() ? 1.0 : (avg_n / subj_means.size());

    double within_var = (within_df > 0) ? (within_ss / within_df) : (0.5 * var_y);
    double subj_mean = 0.0;
    for (double m : subj_means) subj_mean += m;
    subj_mean /= std::max<size_t>(1, subj_means.size());
    double between_var = 0.0;
    for (double m : subj_means) between_var += (m - subj_mean) * (m - subj_mean);
    between_var /= std::max<int>(1, (int)subj_means.size() - 1);

    double repeatable_var_guess = std::max(var_floor, between_var - within_var / std::max(1.0, avg_n));
    double pe_guess = std::max(var_floor, 0.35 * within_var);
    double ve_guess = std::max(var_floor, within_var - pe_guess);
    double vg_guess = std::max(var_floor, repeatable_var_guess);
    double total_guess = vg_guess + pe_guess + ve_guess;
    if (total_guess > 0.0 && var_y > var_floor) {
        double scale = var_y / total_guess;
        vg_guess *= scale;
        pe_guess *= scale;
        ve_guess *= scale;
    }

    var_e = ve_guess;
    if (genetic_idx >= 0) vars_u[genetic_idx] = vg_guess;
    if (pe_idx >= 0) vars_u[pe_idx] = pe_guess;

    std::mt19937_64 rng(20250101ULL);
    std::uniform_int_distribution<int> bit(0, 1);

    Eigen::VectorXd last_theta(3);
    last_theta.setConstant(std::numeric_limits<double>::quiet_NaN());
    const int em_warmup_iters = 4;
    const double min_step_factor = 0.5;
    const double max_step_factor = 2.0;
    const double ai_blend = 0.35;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        Eigen::MatrixXd Vinv_X;
        Eigen::VectorXd Vinv_y;
        Eigen::MatrixXd XtVinvX_inv;
        Eigen::VectorXd beta;
        compute_Vinv_X_y(y, Vinv_X, Vinv_y, XtVinvX_inv, beta);

        Eigen::VectorXd Py = Vinv_y - Vinv_X * beta;

        std::vector<Eigen::VectorXd> s(3), q(3);
        s[0].resize(n_records);
        s[1].resize(n_records);
        s[2] = Py;

        apply_dV_genetic(Py, s[0]);
        if (pe_idx >= 0) apply_dV_pe(Py, s[1]);
        else s[1].setZero();

        q[0] = apply_P(s[0], Vinv_X, XtVinvX_inv);
        q[1] = apply_P(s[1], Vinv_X, XtVinvX_inv);
        q[2] = apply_P(s[2], Vinv_X, XtVinvX_inv);

        Eigen::Vector3d tr;
        tr.setZero();
        for (int t = 0; t < cfg.mc_samples; ++t) {
            Eigen::VectorXd z(n_records);
            for (int i = 0; i < n_records; ++i) z(i) = bit(rng) ? 1.0 : -1.0;
            Eigen::VectorXd Pz = apply_P(z, Vinv_X, XtVinvX_inv);

            Eigen::VectorXd dvg_z(n_records), dvp_z(n_records);
            apply_dV_genetic(z, dvg_z);
            if (pe_idx >= 0) apply_dV_pe(z, dvp_z);
            else dvp_z.setZero();

            tr(0) += Pz.dot(dvg_z);
            tr(1) += Pz.dot(dvp_z);
            tr(2) += Pz.dot(z);
        }
        tr.array() /= (double)cfg.mc_samples;

        Eigen::Vector3d grad;
        Eigen::Vector3d quad;
        quad(0) = Py.dot(s[0]);
        quad(1) = Py.dot(s[1]);
        quad(2) = Py.dot(s[2]);
        grad(0) = 0.5 * (quad(0) - tr(0));
        grad(1) = 0.5 * (quad(1) - tr(1));
        grad(2) = 0.5 * (quad(2) - tr(2));

        Eigen::Matrix3d AI;
        AI(0, 0) = 0.5 * s[0].dot(q[0]);
        AI(0, 1) = 0.5 * s[0].dot(q[1]);
        AI(0, 2) = 0.5 * s[0].dot(q[2]);
        AI(1, 0) = AI(0, 1);
        AI(1, 1) = 0.5 * s[1].dot(q[1]);
        AI(1, 2) = 0.5 * s[1].dot(q[2]);
        AI(2, 0) = AI(0, 2);
        AI(2, 1) = AI(1, 2);
        AI(2, 2) = 0.5 * s[2].dot(q[2]);

        Eigen::Vector3d theta;
        theta(0) = (genetic_idx >= 0) ? vars_u[genetic_idx] : 0.0;
        theta(1) = (pe_idx >= 0) ? vars_u[pe_idx] : 0.0;
        theta(2) = var_e;

        Eigen::Vector3d next = theta;
        bool use_em_style = (iter < em_warmup_iters);
        if (!use_em_style) {
            Eigen::LDLT<Eigen::Matrix3d> ai_solver(AI);
            if (ai_solver.info() == Eigen::Success) {
                Eigen::Vector3d delta = ai_solver.solve(grad);
                if (delta.allFinite()) {
                    Eigen::Vector3d cand = theta + ai_blend * delta;
                    for (int k = 0; k < 3; ++k) {
                        double lo = std::max(var_floor, theta(k) * min_step_factor);
                        double hi = std::max(lo, theta(k) * max_step_factor);
                        cand(k) = clamp_value(cand(k), lo, hi);
                    }
                    next = cand;
                } else {
                    use_em_style = true;
                }
            } else {
                use_em_style = true;
            }
        }
        if (use_em_style) {
            // Standard REML EM update: sigma2_k_new = quad_k / tr_k, where
            //   quad_k = y' P (dV/dtheta_k) Py,  tr_k = tr(P dV/dtheta_k).
            // This is the mathematically correct EM step (previously this block
            // used an ad-hoc sqrt-ratio multiplier, which is not standard EM).
            for (int k = 0; k < 3; ++k) {
                double denom = std::max(tr(k), var_floor);
                double em_target = quad(k) / denom;
                if (!std::isfinite(em_target) || em_target < 0.0) em_target = theta(k);
                double lo = std::max(var_floor, theta(k) * min_step_factor);
                double hi = std::max(lo,       theta(k) * max_step_factor);
                next(k) = clamp_value(em_target, lo, hi);
            }
        }
        next(0) = std::max(var_floor, next(0));
        next(1) = std::max(var_floor, next(1));
        next(2) = std::max(var_floor, next(2));

        // Compute LogL using SLQ
        auto op_V = [&](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
            apply_V(v, out);
        };
        auto f_log = [](double x) { return std::log(std::max(x, 1e-12)); };

        // Compute log|V|
        double log_det_V = SLQEstimator::estimate(n_records, op_V, f_log, 30, 15, 42 + iter);

        // Compute log|X' V^-1 X|
        Eigen::LDLT<Eigen::MatrixXd> llt_xtvinvx(X_dense.transpose() * Vinv_X);
        double log_det_XtVinvX = 0;
        if (llt_xtvinvx.info() == Eigen::Success) {
            log_det_XtVinvX = llt_xtvinvx.vectorD().array().log().sum();
        } else {
            log_det_XtVinvX = -std::log(std::max(XtVinvX_inv.determinant(), 1e-12));
        }

        double yPy = y.dot(Py);
        double current_logL = -0.5 * (log_det_V + log_det_XtVinvX + yPy + (n_records - p) * std::log(2 * M_PI));

        if (cfg.verbose) {
            std::ostringstream ss;
            ss << "STCG iter " << (iter + 1) << (use_em_style ? " [EM]" : " [AI]")
               << " | LogL: " << std::fixed << std::setprecision(2) << current_logL
               << " | Vg=" << std::setprecision(5) << next(0)
               << " Vpe=" << next(1)
               << " Ve=" << next(2);
            LOG_INFO(ss.str());

            std::ostringstream h_ss;
            h_ss << "STCG(" << (use_em_style ? "EM" : "AI") << ")\t" << (iter + 1) << "\t"
                 << std::fixed << std::setprecision(2) << current_logL << "\t"
                 << std::setprecision(5) << next(0) << "\t" << next(1) << "\t" << next(2);
            history.push_back(h_ss.str());
        }

        double rel = 0.0;
        if (iter > 0 && last_theta.allFinite()) {
            rel = ((next - last_theta).cwiseAbs().array() / (last_theta.array().abs() + 1e-12)).maxCoeff();
        }
        last_theta = next;

        if (genetic_idx >= 0) vars_u[genetic_idx] = next(0);
        if (pe_idx >= 0) vars_u[pe_idx] = next(1);
        var_e = next(2);

        if (iter > 0 && rel < cfg.tol) break;
    }

    Eigen::MatrixXd Vinv_X;
    Eigen::VectorXd Vinv_y;
    Eigen::MatrixXd XtVinvX_inv;
    Eigen::VectorXd beta_final;
    compute_Vinv_X_y(y, Vinv_X, Vinv_y, XtVinvX_inv, beta_final);

    Eigen::VectorXd Py = Vinv_y - Vinv_X * beta_final;

    Eigen::VectorXd z_ind(geno->rows());
    gather_to_ind(Py, z_ind);

    Eigen::VectorXd a_ind;
    apply_G(z_ind, a_ind);
    a_ind *= vars_u[genetic_idx];

    Eigen::VectorXd pe_ind;
    pe_ind.setZero(geno->rows());
    if (pe_idx >= 0) {
        pe_ind = vars_u[pe_idx] * z_ind;
    }

    Eigen::VectorXd sol(p + n_ind + n_ind);
    sol.setZero();
    sol.head(p) = beta_final;
    sol.segment(p, n_ind) = a_ind.head(n_ind);
    sol.segment(p + n_ind, n_ind) = pe_ind.head(n_ind);

    final_solution = sol;
}

} // namespace cosmic
