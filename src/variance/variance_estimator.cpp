#include "variance_estimator.h"
#include "logger.h"
#include "vce.h"
#include "eigen_vce.h"
#include "matrix_adapter.h"
#include "plink_reader.h"
#include "mme_builder.h"
#include "string_utils.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace cosmic {

using namespace std;
using namespace Eigen;

namespace {

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::vector<std::string> split_multi_trait_token(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(s);
    while (std::getline(ss, cur, ',')) {
        cur = trim_copy(cur);
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

std::vector<std::string> split_pheno_line(const std::string& line, char delim) {
    std::vector<std::string> vals;
    if (delim == ',') {
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) vals.push_back(trim_copy(tok));
    } else {
        std::stringstream ss(line);
        std::string tok;
        while (ss >> tok) vals.push_back(tok);
    }
    return vals;
}

int find_header_col(const std::vector<std::string>& header, const std::string& name) {
    if (name.empty()) return -1;
    if (is_integer_string(name)) return std::stoi(name) - 1;
    const std::string target = lower_ascii(name);
    for (size_t i = 0; i < header.size(); ++i) {
        if (lower_ascii(header[i]) == target) return static_cast<int>(i);
    }
    return -1;
}

double parse_optional_double(const std::string& value) {
    if (is_missing_token(value)) return std::numeric_limits<double>::quiet_NaN();
    try {
        return std::stod(value);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

} // namespace

VCEResult GSVce::run(const Context& ctx, double sigma2_u_init, double sigma2_e_init) {
    VCEResult result;
    const auto& recs = ctx.recs;
    const auto& fd = ctx.fd;
    const auto& cfg = ctx.cfg;
    int n = (int)recs.size();

    // Multi-trait VCE
    if (cfg.multi_trait) {
        LOG_INFO("Using Exact Multi-Trait LMM (MvEigenVce) for VCE...");
        if (!ctx.Qinv || !dynamic_cast<DenseMatrixAdapter*>(ctx.Qinv)) {
            throw std::runtime_error("Multi-trait VCE currently requires dense inverse matrix (e.g. GBLUP/ssGBLUP without --mmap).");
        }
        // G = (G^{-1})^{-1}; use LDLT solve(I) for numerical stability instead of inverse().
        Eigen::MatrixXd Ginv = ctx.Qinv->toDense();
        Eigen::LDLT<Eigen::MatrixXd> ldlt_ginv(Ginv);
        if (ldlt_ginv.info() != Eigen::Success) {
            throw std::runtime_error("GSVce::run (multi-trait): G^{-1} is not positive-definite; cannot recover G.");
        }
        Eigen::MatrixXd G = ldlt_ginv.solve(Eigen::MatrixXd::Identity(Ginv.rows(), Ginv.cols()));

        int n_geno = ctx.Qinv->rows();
        std::vector<std::string> iids(n_geno);
        for (const auto& kv : ctx.idmap) {
            if (kv.second >= 1 && kv.second <= n_geno) {
                iids[kv.second - 1] = kv.first;
            }
        }

        std::ifstream pfs(cfg.pheno_path);
        if (!pfs) throw std::runtime_error("Cannot open phenotype file: " + cfg.pheno_path);
        std::string header_line;
        if (!std::getline(pfs, header_line)) {
            throw std::runtime_error("Empty phenotype file: " + cfg.pheno_path);
        }
        char delim = (header_line.find(',') != std::string::npos) ? ',' : ' ';
        std::vector<std::string> headers = split_pheno_line(header_line, delim);
        if (headers.size() < 3) {
            throw std::runtime_error("Multi-trait phenotype file requires an ID column and at least two trait columns.");
        }

        int iid_col = find_header_col(headers, cfg.id_col_name);
        if (iid_col < 0) iid_col = find_header_col(headers, "IID");
        if (iid_col < 0) iid_col = find_header_col(headers, "id");
        if (iid_col < 0) iid_col = (headers.size() >= 2 ? 1 : 0);

        std::vector<std::string> trait_specs = split_multi_trait_token(cfg.pheno_name);
        std::vector<int> trait_cols;
        std::vector<std::string> trait_names;
        if (!trait_specs.empty()) {
            for (const auto& spec : trait_specs) {
                int idx = find_header_col(headers, spec);
                if (idx < 0 || idx >= (int)headers.size()) {
                    throw std::runtime_error("Cannot find multi-trait phenotype column: " + spec);
                }
                trait_cols.push_back(idx);
                trait_names.push_back(headers[idx]);
            }
        } else if (cfg.pheno_pos > 0) {
            int idx = cfg.pheno_pos - 1;
            if (idx < 0 || idx >= (int)headers.size()) {
                throw std::runtime_error("--pheno-pos is out of range for multi-trait phenotype file.");
            }
            trait_cols.push_back(idx);
            trait_names.push_back(headers[idx]);
        }

        std::vector<int> dcols;
        for (const auto& name : cfg.dcovar_names) {
            int idx = find_header_col(headers, name);
            if (idx < 0) throw std::runtime_error("Cannot find multi-trait discrete covariate column: " + name);
            dcols.push_back(idx);
        }
        for (int pos : cfg.dcovar_cols) {
            int idx = pos - 1;
            if (idx < 0 || idx >= (int)headers.size()) {
                throw std::runtime_error("Multi-trait --dcovar column is out of range: " + std::to_string(pos));
            }
            dcols.push_back(idx);
        }

        std::vector<int> qcols;
        for (const auto& name : cfg.qcovar_names) {
            int idx = find_header_col(headers, name);
            if (idx < 0) throw std::runtime_error("Cannot find multi-trait quantitative covariate column: " + name);
            qcols.push_back(idx);
        }
        for (int pos : cfg.qcovar_cols) {
            int idx = pos - 1;
            if (idx < 0 || idx >= (int)headers.size()) {
                throw std::runtime_error("Multi-trait --qcovar column is out of range: " + std::to_string(pos));
            }
            qcols.push_back(idx);
        }

        if (trait_cols.empty()) {
            std::set<int> excluded;
            excluded.insert(iid_col);
            int fid_col = find_header_col(headers, "FID");
            if (fid_col >= 0) excluded.insert(fid_col);
            for (int idx : dcols) excluded.insert(idx);
            for (int idx : qcols) excluded.insert(idx);
            for (int idx = 0; idx < (int)headers.size(); ++idx) {
                if (excluded.count(idx)) continue;
                trait_cols.push_back(idx);
                trait_names.push_back(headers[idx]);
            }
        }
        if (trait_cols.size() < 2) {
            throw std::runtime_error("Multi-trait mode requires at least two trait columns. Use --pheno-name Trait1,Trait2.");
        }

        struct MultiTraitRow {
            Eigen::VectorXd y;
            std::vector<std::string> cats;
            std::vector<double> nums;
        };
        std::map<std::string, MultiTraitRow> pheno_by_iid;
        std::string line;
        while (std::getline(pfs, line)) {
            std::vector<std::string> vals = split_pheno_line(line, delim);
            if ((int)vals.size() <= iid_col) continue;
            int max_col = iid_col;
            for (int idx : trait_cols) max_col = std::max(max_col, idx);
            for (int idx : dcols) max_col = std::max(max_col, idx);
            for (int idx : qcols) max_col = std::max(max_col, idx);
            if ((int)vals.size() <= max_col) continue;

            MultiTraitRow row;
            row.y.resize((int)trait_cols.size());
            for (size_t j = 0; j < trait_cols.size(); ++j) {
                row.y((int)j) = parse_optional_double(vals[trait_cols[j]]);
            }
            bool bad_covariate = false;
            for (int idx : dcols) {
                if (is_missing_token(vals[idx])) { bad_covariate = true; break; }
                row.cats.push_back(vals[idx]);
            }
            if (bad_covariate) continue;
            for (int idx : qcols) {
                double v = parse_optional_double(vals[idx]);
                if (!std::isfinite(v)) { bad_covariate = true; break; }
                row.nums.push_back(v);
            }
            if (bad_covariate) continue;
            pheno_by_iid[vals[iid_col]] = row;
        }

        Eigen::MatrixXd Y(n_geno, (int)trait_cols.size());
        Y.setConstant(std::numeric_limits<double>::quiet_NaN());
        std::vector<GenRecord> aligned_records(n_geno);
        std::vector<char> has_row(n_geno, 0);
        size_t matched = 0;
        for (int i = 0; i < n_geno; ++i) {
            if (iids[i].empty()) continue;
            auto it = pheno_by_iid.find(iids[i]);
            if (it == pheno_by_iid.end()) continue;
            Y.row(i) = it->second.y;
            GenRecord rec;
            rec.idstr = iids[i];
            rec.aid = i + 1;
            rec.y = it->second.y(0);
            rec.cats = it->second.cats;
            rec.nums = it->second.nums;
            aligned_records[i] = std::move(rec);
            has_row[i] = 1;
            if (!it->second.y.hasNaN()) matched++;
        }

        LOG_INFO("Matched " << matched << " records for multi-trait VCE.");

        // Drop individuals with any missing trait (complete-case analysis).
        // MvEigenVce requires a fully observed Y; NaN rows would propagate
        // through V_i = Vg*D[i] + Ve and break the LDLT factorization.
        std::vector<int> keep_idx;
        keep_idx.reserve(Y.rows());
        for (int i = 0; i < Y.rows(); ++i) {
            bool has_nan = false;
            for (int j = 0; j < Y.cols(); ++j) {
                if (!std::isfinite(Y(i, j))) { has_nan = true; break; }
            }
            if (!has_nan && has_row[i]) keep_idx.push_back(i);
        }
        int n_keep = (int)keep_idx.size();
        if (n_keep == 0) {
            throw std::runtime_error("Multi-trait VCE: no individuals with complete phenotypes across all traits.");
        }
        if (n_keep < Y.rows()) {
            LOG_INFO("Dropping " << (Y.rows() - n_keep) << " individuals with missing traits; "
                      << n_keep << " complete cases retained.");
        }
        Eigen::MatrixXd Y_k(n_keep, Y.cols());
        for (int r = 0; r < n_keep; ++r) Y_k.row(r) = Y.row(keep_idx[r]);
        // Slice G symmetrically to keep order.
        Eigen::MatrixXd G_k(n_keep, n_keep);
        for (int r = 0; r < n_keep; ++r)
            for (int c = 0; c < n_keep; ++c)
                G_k(r, c) = G(keep_idx[r], keep_idx[c]);
        std::vector<GenRecord> keep_records;
        keep_records.reserve(n_keep);
        for (int idx : keep_idx) keep_records.push_back(aligned_records[idx]);
        FixedDesignG fd_m = buildFixedDesignGeneric(keep_records, false, cfg.dcovar_names, cfg.qcovar_names);
        Eigen::MatrixXd X_m = Eigen::MatrixXd::Zero(n_keep, fd_m.p);
        for (int r = 0; r < n_keep; ++r) {
            for (const auto& kv : fd_m.rows[r]) {
                if (kv.first >= 0 && kv.first < fd_m.p) X_m(r, kv.first) = kv.second;
            }
        }
        LOG_INFO("Multi-trait fixed-effect design has " << fd_m.p << " columns.");

        MvEigenVce::Options mv_opts;
        mv_opts.use_projection = true;
        // Multi-trait EM converges slowly; use a larger budget than the
        // single-trait default. Prefer --vce-max-iter when set by the user.
        mv_opts.max_iter = std::max(500, cfg.vce_max_iter * 25);
        mv_opts.tol = cfg.vce_tol;
        MvEigenVce mv_vce(mv_opts);

        mv_vce.prepare(Y_k, X_m, G_k);
        mv_vce.runNullModel();

        LOG_INFO("Multi-trait VCE completed.");
        Eigen::MatrixXd Vg = mv_vce.getVg();
        Eigen::MatrixXd Ve = mv_vce.getVe();

        string vars_out = cfg.out_prefix + ".mv.vars";
        ofstream fv(vars_out);
        fv << "Trait1\tTrait2\tVg\tVe\n";
        for (int i = 0; i < Vg.rows(); ++i) {
            for (int j = 0; j < Vg.cols(); ++j) {
                fv << (i + 1) << "\t" << (j + 1) << "\t" << Vg(i, j) << "\t" << Ve(i, j) << "\n";
            }
        }
        fv.close();
        LOG_INFO("Multi-trait VCE results saved to [" << vars_out << "].");

        // Save multi-trait results for downstream MME solve + EBV output
        result.mv_Vg = Vg;
        result.mv_Ve = Ve;
        result.mv_Y = Y_k;
        result.mv_X = X_m;
        result.mv_keep_indices = keep_idx;
        result.mv_trait_names = trait_names;
        result.mv_effect_names = fd_m.names;
        result.sigma2_e = 1.0;  // not used for multi-trait
        result.multi_trait_exit = false;  // allow downstream solve
        return result;
    }

    VectorXd y(n);
    for (int i = 0; i < n; ++i) y(i) = recs[i].y;

    double ve_init = sigma2_e_init > 0 ? sigma2_e_init : 1.0;
    double vu_init = sigma2_u_init > 0 ? sigma2_u_init : 1.0;

    std::string resolved_vce_mode = cfg.vce_mode;
    if (cfg.rrm_model && (resolved_vce_mode == "exact" || resolved_vce_mode == "stcg")) {
        LOG_WARN("RRM GS currently does not support --vce-mode " << resolved_vce_mode << ". Falling back to AI-REML style iteration.");
        resolved_vce_mode = "ai";
    }

    // Exact LMM via Common's EigenVce
    if (resolved_vce_mode == "exact" && ctx.Qinv && dynamic_cast<DenseMatrixAdapter*>(ctx.Qinv)) {
        LOG_INFO("Using Exact LMM (EigenVce) for Variance Component Estimation...");
        // G = (G^{-1})^{-1}; use LDLT solve(I) for numerical stability instead of inverse().
        Eigen::MatrixXd Ginv = ctx.Qinv->toDense();
        Eigen::LDLT<Eigen::MatrixXd> ldlt_ginv(Ginv);
        if (ldlt_ginv.info() != Eigen::Success) {
            throw std::runtime_error("GSVce::run (exact): G^{-1} is not positive-definite; cannot recover G.");
        }
        Eigen::MatrixXd G = ldlt_ginv.solve(Eigen::MatrixXd::Identity(Ginv.rows(), Ginv.cols()));

        Eigen::MatrixXf X_f = Eigen::MatrixXf::Zero(n, fd.p);
        for (int i = 0; i < n; ++i) {
            for (const auto& kv : fd.rows[i]) {
                if (kv.first >= 0 && kv.first < fd.p) {
                    X_f(i, kv.first) = kv.second;
                }
            }
        }

        EigenVce::Options vce_opts;
        vce_opts.use_covariate_projection = true;
        EigenVce vce(vce_opts);
        vce.prepare(y, X_f, G);
        vce.runNullModel();

        result.sigma2_e = vce.getSigma2e();
        double sigma2_u = vce.getSigma2g();
        result.vars_u.push_back(sigma2_u);
        result.se_u.push_back(0.0);
        result.se_e = 0.0;

        LOG_INFO("Exact VCE completed: Vu = " << sigma2_u << ", Ve = " << result.sigma2_e);
        return result;
    }

    // Default to AI_REML / EM / MCEM
    VCEConfig vce_cfg;
    if (resolved_vce_mode == "ai") vce_cfg.max_iter = cfg.ai_maxit;
    else if (resolved_vce_mode == "em") vce_cfg.max_iter = cfg.em_maxit;
    else if (resolved_vce_mode == "fdiff") vce_cfg.max_iter = cfg.ai_maxit;
    else vce_cfg.max_iter = cfg.vce_max_iter;

    vce_cfg.mc_samples = cfg.vce_mc_samples;
    vce_cfg.tol = cfg.vce_tol;
    vce_cfg.pcg_max_iter = cfg.use_he_pcg ? cfg.pcg_num : cfg.pcg_maxit;
    vce_cfg.pcg_tol = cfg.pcg_tol;
    vce_cfg.set_algorithm_from_string(resolved_vce_mode);
    vce_cfg.force_dense = cfg.force_dense_exact;
    vce_cfg.force_exact = cfg.force_exact;
    vce_cfg.use_implicit = ctx.force_implicit;
    vce_cfg.trace_mode = cfg.trace_mode;
    vce_cfg.solver_mode = cfg.solver_mode;
    vce_cfg.print_report = cfg.print_report;

    if (cfg.dense_solver == "direct") vce_cfg.dense_tier = DenseSolverTier::DirectLDLT;
    else if (cfg.dense_solver == "cholesky") vce_cfg.dense_tier = DenseSolverTier::CholeskyLLT;
    else if (cfg.dense_solver == "eigen") vce_cfg.dense_tier = DenseSolverTier::EigenDecomp;
    else if (cfg.dense_solver == "lowrank") vce_cfg.dense_tier = DenseSolverTier::LowRankSVD;
    else if (cfg.dense_solver == "pcgslq") vce_cfg.dense_tier = DenseSolverTier::PCG_SLQ;
    else vce_cfg.dense_tier = DenseSolverTier::CholeskyLLT;

    std::vector<RandomComponent> components;
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };

    if (to_lower(resolved_vce_mode) == "stcg") {
        if (!ctx.stcg_genotype || !ctx.stcg_ga_dim) {
            throw std::runtime_error("STCG mode requires an initialized genotype matrix and dimension adapter.");
        }
        RandomComponent rc_g;
        rc_g.Qinv = ctx.stcg_ga_dim;
        rc_g.geno_mat = ctx.stcg_genotype;
        components.push_back(rc_g);
        if (ctx.PEinv) {
            RandomComponent rc_pe;
            rc_pe.Qinv = ctx.PEinv;
            components.push_back(rc_pe);
        }
    } else if (cfg.rrm_model) {
        components = ctx.base_components;
    } else {
        if (ctx.Qinv) { RandomComponent rc; rc.Qinv = ctx.Qinv; components.push_back(rc); }
        if (ctx.GDinv) { RandomComponent rc; rc.Qinv = ctx.GDinv; components.push_back(rc); }
        if (ctx.GEinv) { RandomComponent rc; rc.Qinv = ctx.GEinv; components.push_back(rc); }
        if (ctx.PEinv) { RandomComponent rc; rc.Qinv = ctx.PEinv; components.push_back(rc); }
    }

    if (to_lower(resolved_vce_mode) == "stcg") {
        if (cfg.mat_effect || !ctx.generic_rand_invs.empty() || ctx.GDinv || ctx.GEinv) {
            throw std::runtime_error("STCG mode currently supports only additive genomic effect (+ optional Pe).");
        }
    } else {
        if (cfg.mat_effect) {
            RandomComponent rc;
            rc.Qinv = ctx.Qinv;
            rc.id_map = ctx.mat_id_map;
            components.push_back(rc);
        }
        for (size_t i = 0; i < ctx.generic_rand_invs.size(); ++i) {
            RandomComponent rc;
            rc.Qinv = ctx.generic_rand_invs[i].get();
            rc.id_map = ctx.generic_rand_maps[i];
            components.push_back(rc);
        }
    }

    if (components.empty()) {
        throw std::runtime_error("No random effect matrix provided. VCE cannot run.");
    }

    std::string resolved_lower = to_lower(resolved_vce_mode);
    // Path A: MME-path AI-REML now supports multi-component models via stochastic
    // trace estimation (run_aireml_step_multi). No automatic downgrade to EM is
    // needed for ai/emai/hi/fdiff; the solver will fall back to an EM step
    // internally only if a numerical failure occurs during an AI iteration.
    // Users who want exact dense multi-component AI can still request --vce-mode vmatrix.

    if (ctx.PEinv && components.size() > 1) {
        LOG_INFO("AI-REML applied Pe absorption optimization for multi-component repeatability VCE.");
    }

    std::vector<double> vu_inits(components.size(), vu_init);
    if (cfg.var_priors.size() > 1) {
        for (size_t k = 0; k < std::min(components.size(), cfg.var_priors.size() - 1); ++k) {
            vu_inits[k] = cfg.var_priors[k];
        }
    }

    auto vce = VCESolver::create(recs, fd, components, y, vce_cfg);
    vce->setInitialVars(ve_init, vu_inits);
    vce->solve();
    vce->calculate_SE();

    result.sigma2_e = vce->getVarE();
    result.vars_u = vce->getVarsU();
    result.solution = vce->getSolution();
    result.se_u = vce->getSEVarsU();
    result.se_e = vce->getSEVarE();

    return result;
}

} // namespace cosmic
