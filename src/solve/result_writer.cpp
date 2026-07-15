#include "mixed_model_solver.h"
#include "logger.h"
#include "plink_reader.h"
#include "pgen_reader.h"
#include "matrix_adapter.h"
#include "phenotype_loader.h"
#include "string_utils.h"
#include "design.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <fstream>
#include <numeric>

namespace cosmic {

using namespace std;
using namespace Eigen;

static auto calc_pval = [](double val, double se) {
    if (se < 1e-12) return 1.0;
    double z = val / se;
    return std::erfc(z / std::sqrt(2.0));
};

static auto format_pval = [](double p) {
    if (p < 2e-16) return std::string("<2e-16");
    std::ostringstream ss;
    ss << p;
    return ss.str();
};

static auto betai_func = [](double a, double b, double x) -> double {
    auto betacf = [](double a, double b, double x) -> double {
        double qab = a + b, qap = a + 1.0, qam = a - 1.0;
        double c = 1.0, d = 1.0 - qab * x / qap;
        if (std::abs(d) < 1e-30) d = 1e-30;
        d = 1.0 / d;
        double h = d;
        for (int m = 1; m <= 100; ++m) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d; if (std::abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c; if (std::abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d; h *= d * c;
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d; if (std::abs(d) < 1e-30) d = 1e-30;
            c = 1.0 + aa / c; if (std::abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d; double del = d * c; h *= del;
            if (std::abs(del - 1.0) < 3e-7) break;
        }
        return h;
    };
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betacf(a, b, x) / a;
    } else {
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
    }
};

void GSOutput::writeResults(const Context& ctx) {
    const auto& recs = ctx.recs;
    const auto& fd = ctx.fd;
    const auto& cfg = ctx.cfg;
    const auto& sr = ctx.solve_result;
    const auto& vce_result = ctx.vce_result;

    double sigma2_e = ctx.sigma2_e;
    const auto& final_vars_u = vce_result.vars_u;
    const auto& final_se_u = vce_result.se_u;
    double se_e = vce_result.se_e;

    // === Write VCE vars file ===
    if (ctx.run_vce && !final_vars_u.empty()) {
        if (final_se_u.empty() || final_se_u.size() != final_vars_u.size()) {
            const_cast<std::vector<double>&>(final_se_u).assign(final_vars_u.size(), 0.0);
        }

        string vars_out = cfg.out_prefix + ".vars";
        ofstream fv(vars_out);

        string m = auto_model_from_inv(ctx.inv_label, cfg.model);
        string model_tag;
        if (m == "ssgblup") model_tag = "HA";
        else if (m == "gblup") model_tag = "GA";
        else model_tag = "PA";

        std::vector<string> tags;
        if (ctx.stcg_mode) {
            tags.push_back("GA");
            if (ctx.solve_result.mme_components.size() > 1) tags.push_back("Pe");
        } else if (cfg.rrm_model) {
            for (int k = 0; k < ctx.rrm_additive_component_count; ++k) tags.push_back(model_tag + "_L" + std::to_string(k));
            for (int k = 0; k < ctx.rrm_pe_component_count; ++k) tags.push_back("Pe_L" + std::to_string(k));
        } else {
            // Reconstruct tags from mme_components
            for (size_t k = 0; k < sr.mme_components.size(); ++k) {
                if (k == 0 && ctx.solve_result.mme_components[k].Qinv) {
                    tags.push_back(model_tag);
                } else {
                    // Try to identify by checking against known matrices
                    tags.push_back("u" + std::to_string(k));
                }
            }
            // More precise tagging
            tags.clear();
            int comp_idx = 0;
            if (sr.mme_components.size() > comp_idx && sr.mme_components[comp_idx].Qinv) { tags.push_back(model_tag); comp_idx++; }
            if (cfg.dom_effect && sr.mme_components.size() > comp_idx && sr.mme_components[comp_idx].Qinv) { tags.push_back("GD"); comp_idx++; }
            if (cfg.epi_effect && sr.mme_components.size() > comp_idx && sr.mme_components[comp_idx].Qinv) { tags.push_back("GE"); comp_idx++; }
            if (cfg.pe_effect && sr.mme_components.size() > comp_idx && sr.mme_components[comp_idx].Qinv) { tags.push_back("Pe"); comp_idx++; }
            if (cfg.mat_effect && sr.mme_components.size() > comp_idx) { tags.push_back("Mat"); comp_idx++; }
            for (size_t i = 0; i < cfg.rand_names.size() && comp_idx < (int)sr.mme_components.size(); ++i, ++comp_idx) {
                tags.push_back(cfg.rand_names[i]);
            }
        }

        double total_var = sigma2_e;
        for (double v : final_vars_u) total_var += v;

        fv << "Item\tVar\tVar_SE\th2\th2_SE\th2_Pr(Chisq)\n";
        LOG_INFO("Summary of estimated variances and heritability:");
        LOG_INFO("       Var Var_SE     h2  h2_SE h2_Pr(Chisq)");

        for (size_t k = 0; k < final_vars_u.size(); ++k) {
            double h2 = (total_var > 1e-9) ? (final_vars_u[k] / total_var) : 0.0;
            double denom = total_var * total_var;
            double dh_du = (total_var - final_vars_u[k]) / denom;
            double dh_dother = -final_vars_u[k] / denom;
            double var_h2 = dh_du * dh_du * final_se_u[k] * final_se_u[k];
            for (size_t j = 0; j < final_vars_u.size(); ++j) {
                if (j != k) var_h2 += dh_dother * dh_dother * final_se_u[j] * final_se_u[j];
            }
            var_h2 += dh_dother * dh_dother * se_e * se_e;
            double se_h2 = std::sqrt(std::max(0.0, var_h2));
            double chisq_pval_h2 = calc_pval(h2, se_h2);

            string tag = (k < tags.size()) ? tags[k] : "u" + std::to_string(k);
            fv << tag << "\t" << final_vars_u[k] << "\t" << final_se_u[k] << "\t" << h2 << "\t" << se_h2 << "\t" << format_pval(chisq_pval_h2) << "\n";

            char buf[256];
            std::snprintf(buf, sizeof(buf), "%3s %.4f %.4f %.4f %.4f       %s",
                tag.c_str(), final_vars_u[k], final_se_u[k], h2, se_h2, format_pval(chisq_pval_h2).c_str());
            LOG_INFO(buf);
        }

        double h2_e = (total_var > 1e-9) ? (sigma2_e / total_var) : 0.0;
        double denom_e = total_var * total_var;
        double dhe_de = (total_var - sigma2_e) / denom_e;
        double dhe_dother = -sigma2_e / denom_e;
        double var_h2_e = dhe_de * dhe_de * se_e * se_e;
        for (size_t j = 0; j < final_vars_u.size(); ++j) {
            var_h2_e += dhe_dother * dhe_dother * final_se_u[j] * final_se_u[j];
        }
        double se_h2_e = std::sqrt(std::max(0.0, var_h2_e));
        double chisq_pval_e = calc_pval(h2_e, se_h2_e);

        fv << "e\t" << sigma2_e << "\t" << se_e << "\t" << h2_e << "\t" << se_h2_e << "\t" << format_pval(chisq_pval_e) << "\n";

        // Repeatability
        if (sr.mme_components.size() >= 2) {
            int genetic_idx = -1, pe_idx = -1;
            for (size_t k = 0; k < tags.size(); ++k) {
                if ((tags[k] == model_tag || tags[k] == "GA" || tags[k] == "PA" || tags[k] == "HA") && genetic_idx < 0) genetic_idx = (int)k;
                if (tags[k] == "Pe") pe_idx = (int)k;
            }
            if (genetic_idx >= 0 && pe_idx >= 0) {
                double repeatability_var = final_vars_u[genetic_idx] + final_vars_u[pe_idx];
                double repeatability_var_se = std::sqrt(
                    final_se_u[genetic_idx] * final_se_u[genetic_idx] +
                    final_se_u[pe_idx] * final_se_u[pe_idx]);
                double repeatability = (total_var > 1e-9) ? (repeatability_var / total_var) : 0.0;

                double dr_dselected = (total_var - repeatability_var) / (total_var * total_var);
                double dr_dother = -repeatability_var / (total_var * total_var);
                double var_repeatability = dr_dother * dr_dother * se_e * se_e;
                for (size_t k = 0; k < final_vars_u.size(); ++k) {
                    double deriv = (k == (size_t)genetic_idx || k == (size_t)pe_idx) ? dr_dselected : dr_dother;
                    var_repeatability += deriv * deriv * final_se_u[k] * final_se_u[k];
                }
                double se_repeatability = std::sqrt(std::max(0.0, var_repeatability));
                double repeatability_pval = calc_pval(repeatability, se_repeatability);

                fv << "Repeatability\t" << repeatability_var << "\t" << repeatability_var_se
                   << "\t" << repeatability << "\t" << se_repeatability
                   << "\t" << format_pval(repeatability_pval) << "\n";

                char rbuf[256];
                std::snprintf(rbuf, sizeof(rbuf), "%3s %.4f %.4f %.4f %.4f       %s",
                    "r2", repeatability_var, repeatability_var_se, repeatability, se_repeatability, format_pval(repeatability_pval).c_str());
                LOG_INFO(rbuf);
            }
        }

        fv.close();

        char buf[256];
        std::snprintf(buf, sizeof(buf), "  e %.4f %.4f %.4f %.4f       %s",
            sigma2_e, se_e, h2_e, se_h2_e, format_pval(chisq_pval_e).c_str());
        LOG_INFO(buf);
        LOG_INFO("Results of estimated variance components have been saved in the file [" << cfg.out_prefix << ".vars].\n");
    }

    // === Write Beta ===
    LOG_INFO("Summary of estimated coefficients:");
    LOG_INFO("   Levels Estimation     SE");
    int p = fd.p;
    string beta_path = cfg.out_prefix + ".beta";
    ofstream fo(beta_path);
    fo << "Levels\tEstimation\tSE\n";
    for (int i = 0; i < p; ++i) {
        fo << fd.names[i] << "\t" << sr.beta(i);
        if (sr.se_calculated && i < sr.beta_se.size()) {
            fo << "\t" << sr.beta_se(i);
            if (i < 15 || i == p - 1) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%9s %10.4f %6.4f", fd.names[i].c_str(), sr.beta(i), sr.beta_se(i));
                LOG_INFO(buf);
            } else if (i == 15) {
                LOG_INFO("       .          .      .");
                LOG_INFO("       .          .      .");
                LOG_INFO("       .          .      .");
            }
        } else {
            fo << "\tnan";
            if (i < 15 || i == p - 1) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%9s %10.4f    nan", fd.names[i].c_str(), sr.beta(i));
                LOG_INFO(buf);
            } else if (i == 15) {
                LOG_INFO("       .          .      .");
                LOG_INFO("       .          .      .");
                LOG_INFO("       .          .      .");
            }
        }
        fo << "\n";
    }
    fo.close();

    // === ANOVA ===
    if (p > 1 && sr.cov_calculated) {
        string anova_path = cfg.out_prefix + ".anova";
        ofstream fanova(anova_path);
        if (fanova) {
            fanova << "Factors\tDf\tSumSq\tMeanSq\tF\tPr(>F)\n";
            double den_df = std::max(1.0, (double)(recs.size() - p));
            double s2_e = sigma2_e > 0 ? sigma2_e : 1.0;

            for (const auto& factor : fd.factor_names) {
                if (factor == "mu") continue;
                auto it = fd.factor_cols.find(factor);
                if (it == fd.factor_cols.end()) continue;
                const auto& cols = it->second;
                int df = cols.size();
                if (df == 0) continue;

                MatrixXd C_ff(df, df);
                VectorXd beta_f(df);
                for (int i = 0; i < df; ++i) {
                    beta_f(i) = sr.beta(cols[i]);
                    for (int j = 0; j < df; ++j) {
                        C_ff(j, i) = sr.Cinv_beta(cols[j], cols[i]);
                    }
                }

                double W = beta_f.transpose() * C_ff.inverse() * beta_f;
                double sumsq = W;
                double meansq = sumsq / df;
                double F_val = meansq / s2_e;

                double p_val = 1.0;
                if (F_val > 0) {
                    double x = df * F_val / (df * F_val + den_df);
                    p_val = 1.0 - betai_func(df / 2.0, den_df / 2.0, x);
                }

                string p_str = (p_val < 2e-16) ? "<2e-16" : (std::ostringstream() << p_val).str();
                fanova << factor << "\t" << df << "\t"
                       << sumsq << "\t" << meansq << "\t"
                       << F_val << "\t" << p_str << "\n";
            }
            fanova << "e\t" << (int)den_df << "\t"
                   << (den_df * s2_e) << "\t" << s2_e << "\t.\t.\n";
            fanova.close();
            LOG_INFO("Analysis of variance table:");
            std::ifstream fanova_in(anova_path);
            string line;
            while (std::getline(fanova_in, line)) LOG_INFO(line);
            LOG_INFO("Analysis of variance table has been saved in file [" << anova_path << "].");
        }
    }

    // === Write Random Effects ===
    string rand_path = cfg.out_prefix + ".rand";
    ofstream fr(rand_path);

    string m_tag = auto_model_from_inv(ctx.inv_label, cfg.model);
    string model_col_name;
    if (m_tag == "ssgblup") model_col_name = "HA";
    else if (m_tag == "gblup") model_col_name = "GA";
    else model_col_name = "PA";

    std::vector<string> rand_cols;
    if (ctx.stcg_mode) {
        rand_cols.push_back("GA");
        if (sr.mme_components.size() > 1) rand_cols.push_back("Pe");
    } else if (cfg.rrm_model) {
        for (int k = 0; k < ctx.rrm_additive_component_count; ++k) rand_cols.push_back(model_col_name + "_L" + std::to_string(k));
        for (int k = 0; k < ctx.rrm_pe_component_count; ++k) rand_cols.push_back("Pe_L" + std::to_string(k));
    } else {
        int comp_idx = 0;
        if (sr.mme_components.size() > comp_idx && sr.mme_components[comp_idx].Qinv) { rand_cols.push_back(model_col_name); comp_idx++; }
        if (cfg.dom_effect && sr.mme_components.size() > comp_idx) { rand_cols.push_back("GD"); comp_idx++; }
        if (cfg.epi_effect && sr.mme_components.size() > comp_idx) { rand_cols.push_back("GE"); comp_idx++; }
        if (cfg.pe_effect && sr.mme_components.size() > comp_idx) { rand_cols.push_back("Pe"); comp_idx++; }
        if (cfg.mat_effect && sr.mme_components.size() > comp_idx) { rand_cols.push_back("Mat"); comp_idx++; }
        for (size_t i = 0; i < cfg.rand_names.size() && comp_idx < (int)sr.mme_components.size(); ++i, ++comp_idx) {
            rand_cols.push_back(cfg.rand_names[i]);
        }
    }

    fr << "ID";
    for (const auto& col : rand_cols) fr << "\t" << col;
    fr << "\tresiduals\tn_obs\n";

    int n_ind = 0;
    for (const auto& kv : ctx.idmap) {
        if (kv.second > n_ind) n_ind = kv.second;
    }
    if (n_ind == 0) {
        n_ind = sr.mme_components.empty() ? 0 : sr.mme_components[0].Qinv->rows();
    }
    vector<string> id_list(n_ind);
    for (const auto& kv : ctx.idmap) {
        if (kv.second >= 1 && kv.second <= n_ind) id_list[kv.second - 1] = kv.first;
    }

    vector<double> residual_sums(n_ind, 0.0);
    vector<int> obs_counts(n_ind, 0);
    vector<vector<double>> component_sums(sr.mme_components.size(), vector<double>(n_ind, 0.0));
    vector<double> obs_fixed(recs.size(), 0.0);
    vector<double> obs_fitted(recs.size(), 0.0);
    vector<double> obs_residuals(recs.size(), std::numeric_limits<double>::quiet_NaN());
    vector<vector<double>> obs_component_values(sr.mme_components.size(), vector<double>(recs.size(), 0.0));

    for (size_t i = 0; i < recs.size(); ++i) {
        double xb = 0.0;
        for (const auto& kv : fd.rows[i]) {
            if (kv.first >= 0 && kv.first < p) xb += kv.second * sr.beta(kv.first);
        }
        obs_fixed[i] = xb;

        double total_u = 0.0;
        int offset = 0;
        for (size_t k = 0; k < sr.mme_components.size(); ++k) {
            int q_k = sr.mme_components[k].Qinv ? sr.mme_components[k].Qinv->rows() : 0;
            int mapped_idx = -1;
            if (sr.mme_components[k].id_map.empty()) {
                if (recs[i].aid > 0) mapped_idx = recs[i].aid - 1;
            } else if (i < sr.mme_components[k].id_map.size()) {
                mapped_idx = sr.mme_components[k].id_map[i];
            }

            double component_value = 0.0;
            if (mapped_idx >= 0 && mapped_idx < q_k) {
                double covar = sr.mme_components[k].covar_map.empty() ? 1.0 : sr.mme_components[k].covar_map[i];
                component_value = covar * sr.u(offset + mapped_idx);
            }
            obs_component_values[k][i] = component_value;
            total_u += component_value;
            offset += q_k;
        }

        obs_fitted[i] = xb + total_u;
        obs_residuals[i] = recs[i].y - obs_fitted[i];

        int aid_idx = recs[i].aid - 1;
        if (aid_idx >= 0 && aid_idx < n_ind) {
            residual_sums[aid_idx] += obs_residuals[i];
            obs_counts[aid_idx] += 1;
            for (size_t k = 0; k < sr.mme_components.size(); ++k) {
                component_sums[k][aid_idx] += obs_component_values[k][i];
            }
        }
    }

    vector<int> output_order;
    output_order.reserve(n_ind);
    vector<bool> visited(n_ind, false);
    for (const auto& r : recs) {
        int idx = r.aid - 1;
        if (idx >= 0 && idx < n_ind && !visited[idx]) {
            visited[idx] = true;
            output_order.push_back(idx);
        }
    }
    vector<int> remaining;
    remaining.reserve(n_ind - output_order.size());
    for (int i = 0; i < n_ind; ++i) {
        if (!visited[i]) remaining.push_back(i);
    }
    std::sort(remaining.begin(), remaining.end(), [&](int a, int b) {
        return id_list[a] < id_list[b];
    });
    output_order.insert(output_order.end(), remaining.begin(), remaining.end());

    for (int i : output_order) {
        if (i < n_ind) {
            string id = id_list[i].empty() ? "UNK_" + to_string(i + 1) : id_list[i];
            fr << id;

            int offset = 0;
            for (size_t k = 0; k < sr.mme_components.size(); ++k) {
                int q_k = sr.mme_components[k].Qinv ? sr.mme_components[k].Qinv->rows() : 0;
                if (obs_counts[i] > 0) {
                    fr << "\t" << (component_sums[k][i] / obs_counts[i]);
                } else if (sr.mme_components[k].id_map.empty() && i < q_k) {
                    fr << "\t" << sr.u(offset + i);
                } else {
                    fr << "\tnan";
                }
                offset += q_k;
            }

            if (obs_counts[i] > 0) fr << "\t" << (residual_sums[i] / obs_counts[i]);
            else fr << "\tnan";
            fr << "\t" << obs_counts[i] << "\n";
        }
    }
    fr.close();

    // === RRM-specific outputs ===
    if (cfg.rrm_model) {
        // RRM metadata
        string rrm_meta_path = cfg.out_prefix + ".rrm.meta.txt";
        ofstream fmeta(rrm_meta_path);
        fmeta << "Key\tValue\n";
        fmeta << "Mode\tGS\n";
        fmeta << "Effect\t" << model_col_name << "\n";
        fmeta << "TimeColumn\t" << cfg.time_col_name << "\n";
        fmeta << "Order\t" << ctx.rrm_order << "\n";
        fmeta << "BasisCount\t" << (ctx.rrm_order + 1) << "\n";
        fmeta << "TimeMin\t" << ctx.rrm_tmin << "\n";
        fmeta << "TimeMax\t" << ctx.rrm_tmax << "\n";
        fmeta << "CurvePointCount\t" << ctx.rrm_curve_times.size() << "\n";
        fmeta << "AdditiveCoeffCount\t" << ctx.rrm_additive_component_count << "\n";
        fmeta << "PeCoeffCount\t" << ctx.rrm_pe_component_count << "\n";
        fmeta.close();
        LOG_INFO("RRM metadata has been saved in file [" << rrm_meta_path << "].");

        // RRM coefficients
        string rrcoef_path = cfg.out_prefix + ".rrcoef";
        ofstream frr(rrcoef_path);
        frr << "ID\tEffect\tOrder\tCoefficient\n";
        for (int i : output_order) {
            if (i < 0 || i >= n_ind) continue;
            string id = id_list[i].empty() ? "UNK_" + to_string(i + 1) : id_list[i];
            int offset = 0;
            for (int k = 0; k < ctx.rrm_additive_component_count; ++k) {
                int q_k = sr.mme_components[k].Qinv ? sr.mme_components[k].Qinv->rows() : 0;
                if (i < q_k) {
                    frr << id << "\t" << model_col_name << "\t" << k << "\t" << sr.u(offset + i) << "\n";
                }
                offset += q_k;
            }
            for (int k = 0; k < ctx.rrm_pe_component_count; ++k) {
                int comp_idx = ctx.rrm_additive_component_count + k;
                int q_k = sr.mme_components[comp_idx].Qinv ? sr.mme_components[comp_idx].Qinv->rows() : 0;
                if (i < q_k) {
                    frr << id << "\tPe\t" << k << "\t" << sr.u(offset + i) << "\n";
                }
                offset += q_k;
            }
        }
        frr.close();
        LOG_INFO("RRM coefficient table has been saved in file [" << rrcoef_path << "].");

        // RRM K matrix
        string rrk_path = cfg.out_prefix + ".rrk";
        ofstream frk(rrk_path);
        frk << "Effect\tRow\tCol\tValue\tEstimateType\n";
        for (int i = 0; i < ctx.rrm_additive_component_count; ++i) {
            for (int j = 0; j < ctx.rrm_additive_component_count; ++j) {
                double val = (i < sr.rrm_additive_K.rows() && j < sr.rrm_additive_K.cols()) ? sr.rrm_additive_K(i, j) : 0.0;
                frk << model_col_name << "\t" << i << "\t" << j << "\t" << val << "\tblup_cov\n";
            }
        }
        for (int i = 0; i < ctx.rrm_pe_component_count; ++i) {
            for (int j = 0; j < ctx.rrm_pe_component_count; ++j) {
                double val = (i < sr.rrm_pe_K.rows() && j < sr.rrm_pe_K.cols()) ? sr.rrm_pe_K(i, j) : 0.0;
                frk << "Pe\t" << i << "\t" << j << "\t" << val << "\tblup_cov\n";
            }
        }
        frk.close();
        LOG_INFO("RRM coefficient covariance matrix has been saved in file [" << rrk_path << "].");

        // RRM trajectory curves
        if (!ctx.rrm_curve_times.empty()) {
            Eigen::VectorXd curve_time_vec(ctx.rrm_curve_times.size());
            for (size_t g = 0; g < ctx.rrm_curve_times.size(); ++g) curve_time_vec((int)g) = ctx.rrm_curve_times[g];
            Eigen::MatrixXd Phi_curve = buildLegendreMatrix(curve_time_vec, ctx.rrm_tmin, ctx.rrm_tmax, ctx.rrm_order);

            string rrcurve_path = cfg.out_prefix + ".rrcurve";
            ofstream frc(rrcurve_path);
            frc << "ID\tTime\t" << model_col_name;
            if (ctx.rrm_pe_component_count > 0) frc << "\tPe";
            frc << "\tTotal\n";

            for (int i : output_order) {
                if (i < 0 || i >= n_ind) continue;
                string id = id_list[i].empty() ? "UNK_" + to_string(i + 1) : id_list[i];

                std::vector<double> add_coef(ctx.rrm_additive_component_count, 0.0);
                int offset = 0;
                for (int k = 0; k < ctx.rrm_additive_component_count; ++k) {
                    int q_k = sr.mme_components[k].Qinv ? sr.mme_components[k].Qinv->rows() : 0;
                    if (i < q_k) add_coef[k] = sr.u(offset + i);
                    offset += q_k;
                }

                std::vector<double> pe_coef(ctx.rrm_pe_component_count, 0.0);
                for (int k = 0; k < ctx.rrm_pe_component_count; ++k) {
                    int comp_idx = ctx.rrm_additive_component_count + k;
                    int q_k = sr.mme_components[comp_idx].Qinv ? sr.mme_components[comp_idx].Qinv->rows() : 0;
                    if (i < q_k) pe_coef[k] = sr.u(offset + i);
                    offset += q_k;
                }

                for (size_t g = 0; g < ctx.rrm_curve_times.size(); ++g) {
                    double add_curve = 0.0;
                    for (int k = 0; k < ctx.rrm_additive_component_count; ++k) add_curve += add_coef[k] * Phi_curve((int)g, k);
                    double pe_curve = 0.0;
                    for (int k = 0; k < ctx.rrm_pe_component_count; ++k) pe_curve += pe_coef[k] * Phi_curve((int)g, k);

                    frc << id << "\t" << ctx.rrm_curve_times[g] << "\t" << add_curve;
                    if (ctx.rrm_pe_component_count > 0) frc << "\t" << pe_curve;
                    frc << "\t" << (add_curve + pe_curve) << "\n";
                }
            }
            frc.close();
            LOG_INFO("RRM trajectory table has been saved in file [" << rrcurve_path << "].");
        }
    }

    // === Observation-level output ===
    string obsrand_path = cfg.out_prefix + ".obsrand";
    ofstream forow(obsrand_path);
    forow << "Record\tID\tY\tFixed";
    for (const auto& col : rand_cols) forow << "\t" << col;
    forow << "\tFitted\tResidual\n";
    for (size_t i = 0; i < recs.size(); ++i) {
        forow << (i + 1) << "\t" << recs[i].idstr << "\t" << recs[i].y << "\t" << obs_fixed[i];
        for (size_t k = 0; k < sr.mme_components.size(); ++k) {
            forow << "\t" << obs_component_values[k][i];
        }
        forow << "\t" << obs_fitted[i] << "\t" << obs_residuals[i] << "\n";
    }
    forow.close();
    LOG_INFO("Observation-level fitted values have been saved in file [" << obsrand_path << "].");

    // === SNP Effects Back-solving ===
    if (cfg.calc_snp_effect) {
        if (cfg.rrm_model) {
            LOG_WARN("RRM GS currently does not support SNP-effect back-solving in Cosmicsolver. Skipping --snp-effect.");
        } else if (cfg.bfile_path.empty() && cfg.pfile_path.empty()) {
            LOG_WARN("--snp-effect requires --bfile or --pfile to read genotypes. Skipping SNP effects.");
        } else if (!ctx.Ginv && !(ctx.Qinv && !ctx.use_split)) {
            LOG_WARN("Could not locate Ginv for back-solving. Skipping SNP effects.");
        } else {
            LOG_INFO("Back-solving SNP effects (RR-BLUP / SNP-BLUP equivalent)...");

            AbstractMatrix* g_inv = ctx.Ginv ? ctx.Ginv : ctx.Qinv;

            int n_g = g_inv->rows();
            VectorXd u_g(n_g);
            if (ctx.use_split) {
                for (int i = 0; i < n_g; ++i) {
                    if (ctx.genotyped_map[i] >= 0 && ctx.genotyped_map[i] < sr.u.size()) {
                        u_g(i) = sr.u(ctx.genotyped_map[i]);
                    } else {
                        u_g(i) = 0.0;
                    }
                }
            } else {
                u_g = sr.u.head(n_g);
            }

            VectorXd q = g_inv->operator*(u_g);

            int n_threads = cfg.threads > 0 ? cfg.threads : 1;

            if (!cfg.bfile_path.empty()) {
                PlinkReader reader;
                reader.load(cfg.bfile_path);
                reader.computeStats(n_threads);

                VectorXd zt_q;
                reader.multiply_Zt_v(q, zt_q, n_threads);

                auto means = reader.getMeans();
                auto inv_stds = reader.getInvStds();
                auto bim = reader.getBimInfo();

                double total_2pq = 0.0;
                for (size_t j = 0; j < means.size(); ++j) {
                    double p = means[j] / 2.0;
                    double var = 2.0 * p * (1.0 - p);
                    if (var >= 1e-9) total_2pq += var;
                }

                string snp_out = cfg.out_prefix + ".snp_eff";
                ofstream fs(snp_out);
                fs << "CHR\tSNP\tBP\tA1\tA2\tFREQ\tEFFECT\n";
                for (size_t j = 0; j < bim.size(); ++j) {
                    double effect = 0.0;
                    if (inv_stds[j] > 0.0 && total_2pq > 0.0) {
                        effect = (zt_q(j) / inv_stds[j]) / total_2pq;
                    }
                    double freq = means[j] / 2.0;
                    fs << bim[j].chrom << "\t" << bim[j].id << "\t" << bim[j].bp_pos << "\t"
                       << bim[j].alt << "\t" << bim[j].ref << "\t" << freq << "\t" << effect << "\n";
                }
                fs.close();
                LOG_INFO("Saved SNP effects to " << snp_out);
            } else if (!cfg.pfile_path.empty()) {
                PgenReader reader;
                reader.open(cfg.pfile_path);
                reader.computeStats(n_threads);

                VectorXd zt_q;
                reader.multiply_Zt_v(q, zt_q, n_threads);

                auto means = reader.getMeans();
                auto inv_stds = reader.getInvStds();
                auto var_info = reader.getVariants();

                double total_2pq = 0.0;
                for (size_t j = 0; j < means.size(); ++j) {
                    double p = means[j] / 2.0;
                    double var = 2.0 * p * (1.0 - p);
                    if (var >= 1e-9) total_2pq += var;
                }

                string snp_out = cfg.out_prefix + ".snp_eff";
                ofstream fs(snp_out);
                fs << "CHR\tSNP\tBP\tA1\tA2\tFREQ\tEFFECT\n";
                for (size_t j = 0; j < var_info.size(); ++j) {
                    double effect = 0.0;
                    if (inv_stds[j] > 0.0 && total_2pq > 0.0) {
                        effect = (zt_q(j) / inv_stds[j]) / total_2pq;
                    }
                    double freq = means[j] / 2.0;
                    fs << var_info[j].chrom << "\t" << var_info[j].id << "\t" << var_info[j].pos << "\t"
                       << var_info[j].alt << "\t" << var_info[j].ref << "\t" << freq << "\t" << effect << "\n";
                }
                fs.close();
                LOG_INFO("Saved SNP effects to " << snp_out);
            }
        }
    }

    LOG_INFO("Coefficients of all covariates and fixed effects are saved in file [" << beta_path << "].");
    LOG_INFO("Random effects of all individuals are saved in file [" << rand_path << "].\n");
}

// ============================================================================
// Multi-trait EBV/beta output
// ============================================================================
void GSOutput::writeMultiTrait(
    const SolveResult& solve_result,
    const VCEResult& vce,
    const std::map<std::string, int>& idmap,
    const Config& cfg) {

    const MatrixXd& Vg = vce.mv_Vg;
    const MatrixXd& Y = vce.mv_Y;
    int n = Y.rows();
    int t = Vg.rows();
    int p = (vce.mv_X.cols() > 0) ? vce.mv_X.cols() : 1;

    // Trait names
    std::vector<std::string> trait_names = vce.mv_trait_names;
    if ((int)trait_names.size() < t) {
        trait_names.clear();
        for (int j = 0; j < t; ++j) trait_names.push_back("Trait_" + std::to_string(j + 1));
    }

    // Build ID list for the retained individuals.
    // When VCE dropped missing-trait rows, mv_keep_indices maps row r in
    // Y/u back to the 0-based index in the original n_geno G matrix; we
    // then resolve that to an IID via idmap. Otherwise (no subsetting)
    // row r corresponds directly to idmap index r+1.
    std::vector<std::string> ids(n);
    if (!vce.mv_keep_indices.empty() && (int)vce.mv_keep_indices.size() == n) {
        // Build reverse lookup: 0-based G index -> IID
        std::map<int, std::string> idx_to_iid;
        for (const auto& kv : idmap) {
            int idx = kv.second - 1;
            if (idx >= 0) idx_to_iid[idx] = kv.first;
        }
        for (int r = 0; r < n; ++r) {
            auto it = idx_to_iid.find(vce.mv_keep_indices[r]);
            ids[r] = (it != idx_to_iid.end()) ? it->second : ("Idx_" + std::to_string(vce.mv_keep_indices[r]));
        }
    } else {
        for (const auto& kv : idmap) {
            int idx = kv.second - 1;
            if (idx >= 0 && idx < n) ids[idx] = kv.first;
        }
    }

    // --- Write .beta (p × t) ---
    std::string beta_path = cfg.out_prefix + ".beta";
    ofstream fb(beta_path);
    fb << "Effect";
    for (int j = 0; j < t; ++j) fb << "\t" << trait_names[j];
    fb << "\n";
    // beta is pt×1 in trait-major order: [beta_trait1; beta_trait2; ...]
    // Each trait block is p×1
    std::vector<std::string> effect_names;
    if ((int)vce.mv_effect_names.size() == p) {
        effect_names = vce.mv_effect_names;
    } else if (p == 1) {
        effect_names.push_back("Intercept");
    } else {
        for (int k = 0; k < p; ++k) effect_names.push_back("Covar_" + std::to_string(k + 1));
    }
    for (int k = 0; k < p; ++k) {
        fb << effect_names[k];
        for (int j = 0; j < t; ++j) {
            double val = solve_result.beta(j * p + k);
            fb << "\t" << val;
        }
        fb << "\n";
    }
    fb.close();

    // --- Write .rand (n × t, EBV per trait) ---
    std::string rand_path = cfg.out_prefix + ".rand";
    ofstream fr(rand_path);
    fr << "ID";
    for (int j = 0; j < t; ++j) fr << "\t" << trait_names[j];
    fr << "\n";
    // u is nt×1 in trait-major order: [u_trait1; u_trait2; ...]
    // Each trait block is n×1
    for (int i = 0; i < n; ++i) {
        fr << ids[i];
        for (int j = 0; j < t; ++j) {
            double val = solve_result.u(j * n + i);
            fr << "\t" << val;
        }
        fr << "\n";
    }
    fr.close();

    LOG_INFO("Multi-trait beta saved to [" << beta_path << "].");
    LOG_INFO("Multi-trait EBV saved to [" << rand_path << "].");
}

} // namespace cosmic
