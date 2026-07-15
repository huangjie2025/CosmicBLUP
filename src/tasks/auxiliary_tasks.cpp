#include "auxiliary_tasks.h"
#include "logger.h"
#include "qc_engine.h"
#include "plink_reader.h"
#include "pgen_reader.h"
#include "bgen_reader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <Eigen/Dense>

namespace cosmic {

using namespace std;
using namespace Eigen;

// ── QC Mode ─────────────────────────────────────────────────────

void runQcMode(const Config& cfg) {
    LOG_INFO("=================================================");
    LOG_INFO("             Genotype QC Engine                  ");
    LOG_INFO("=================================================");
    LOG_INFO("Min MAF: " << cfg.qc_maf << " | Max MAF: " << cfg.qc_max_maf);
    LOG_INFO("Min HWE p-value: " << cfg.qc_hwe);
    LOG_INFO("Max missing per SNP (GENO): " << cfg.qc_geno);
    LOG_INFO("Max missing per IND (MIND): " << cfg.qc_mind);

    if (cfg.bfile_path.empty() && cfg.pfile_path.empty() && cfg.bgen_path.empty()) {
        throw std::runtime_error("QC requires --bfile, --pfile, or --bgen.");
    }

    QCOptions qc_opts;
    qc_opts.thread_num = cfg.threads > 0 ? cfg.threads : 1;
    qc_opts.min_maf = cfg.qc_maf;
    qc_opts.max_maf = cfg.qc_max_maf;
    qc_opts.min_hwe = cfg.qc_hwe;
    qc_opts.geno_miss = cfg.qc_geno;
    qc_opts.mind_miss = cfg.qc_mind;
    qc_opts.write_stats = true;

    GenotypeQC qc(qc_opts);
    if (!cfg.bfile_path.empty()) {
        PlinkReader reader;
        reader.load(cfg.bfile_path);
        qc.run(reader, cfg.out_prefix);
    } else if (!cfg.pfile_path.empty()) {
        PgenReader reader;
        reader.open(cfg.pfile_path);
        qc.run(reader, cfg.out_prefix);
    } else if (!cfg.bgen_path.empty()) {
        BgenReader reader;
        reader.open(cfg.bgen_path);
        qc.run(reader, cfg.out_prefix);
    }
    LOG_INFO("Genotype QC finished successfully.");
}

// ── SBLUP Mode ──────────────────────────────────────────────────

void runSblupMode(const Config& cfg) {
    LOG_INFO("=================================================");
    LOG_INFO("    Summary-level BLUP (SBLUP) Mode Selected     ");
    LOG_INFO("=================================================");
    if (cfg.sumstat_path.empty()) throw std::runtime_error("--sumstat is required for SBLUP.");
    if (cfg.h2 < 0) throw std::runtime_error("--h2 is required for SBLUP.");
    LOG_INFO("Summary statistics file : " << cfg.sumstat_path);
    LOG_INFO("Heritability (h2)       : " << cfg.h2);
    LOG_INFO("LD block size           : " << (cfg.window_num > 0 ? cfg.window_num : 1000) << " variants");

    if (cfg.bfile_path.empty() && cfg.pfile_path.empty()) {
        throw std::runtime_error("--bfile or --pfile is required to provide reference LD for SBLUP.");
    }

    std::map<std::string, std::tuple<std::string, std::string, double>> sumstat;
    std::ifstream fs(cfg.sumstat_path);
    if (!fs) throw std::runtime_error("Cannot open --sumstat file.");
    std::string line;

    int snp_col = -1, a1_col = -1, a2_col = -1, beta_col = -1;
    if (std::getline(fs, line)) {
        std::istringstream iss(line);
        std::string token;
        int col = 0;
        while (iss >> token) {
            std::string t = token;
            for (auto& c : t) c = std::toupper(c);
            if (t == "SNP" || t == "VARIANT_ID" || t == "ID") snp_col = col;
            else if (t == "A1" || t == "EFFECT_ALLELE" || t == "ALLELE1") a1_col = col;
            else if (t == "A2" || t == "OTHER_ALLELE" || t == "ALLELE2") a2_col = col;
            else if (t == "BETA" || t == "BETA_1" || t == "SCORE" || t == "EFFECT" || t == "SBLUP_EFFECT") beta_col = col;
            col++;
        }
        if (snp_col == -1 || a1_col == -1 || beta_col == -1) {
            LOG_WARN("Could not perfectly identify header in --sumstat. Assuming 1:SNP, 2:A1, 3:A2, 4:BETA.");
            snp_col = 0; a1_col = 1; a2_col = 2; beta_col = 3;
            fs.clear();
            fs.seekg(0, std::ios::beg);
        }
    }

    while (std::getline(fs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (iss >> token) tokens.push_back(token);
        if (tokens.size() <= std::max({snp_col, a1_col, a2_col, beta_col})) continue;

        std::string snp = tokens[snp_col];
        std::string a1 = tokens[a1_col];
        std::string a2 = a2_col >= 0 ? tokens[a2_col] : "";
        double beta = 0.0;
        try { beta = std::stod(tokens[beta_col]); } catch (...) { continue; }

        sumstat[snp] = {a1, a2, beta};
    }
    LOG_INFO("Loaded " << sumstat.size() << " summary statistics.");

    std::vector<std::string> out_snps;
    std::vector<double> sblup_effs;
    std::vector<std::string> a1s, a2s;

    if (!cfg.bfile_path.empty()) {
        PlinkReader reader;
        reader.load(cfg.bfile_path);
        const auto& bim = reader.getBimInfo();
        int n_ref = reader.getFamInfo().size();
        int m = bim.size();

        double lambda = m * (1.0 - cfg.h2) / (cfg.h2);
        LOG_INFO("Calculated regularization lambda = " << lambda);

        int win_size = cfg.window_num > 0 ? cfg.window_num : 1000;
        LOG_INFO("Using LD block size of " << win_size << " SNPs.");

        for (int start = 0; start < m; start += win_size) {
            int end = std::min(m, start + win_size);
            int b_size = end - start;

            Eigen::MatrixXd X(n_ref, b_size);
            Eigen::VectorXd b_marg = Eigen::VectorXd::Zero(b_size);
            std::vector<int> valid_idx;

            for (int j = start; j < end; ++j) {
                auto it = sumstat.find(bim[j].id);
                if (it != sumstat.end()) {
                    double eff = std::get<2>(it->second);
                    std::string eff_a1 = std::get<0>(it->second);
                    if (bim[j].alt != eff_a1) {
                        if (bim[j].ref == eff_a1) eff = -eff;
                        else continue;
                    }

                    Eigen::VectorXf g; reader.getSnpGenotypes(j, g);
                    double mean = 0, count = 0;
                    for (int i = 0; i < n_ref; ++i) {
                        if (!std::isnan(g(i))) { mean += g(i); count++; }
                    }
                    mean = count > 0 ? mean / count : 0.0;

                    double var = 0;
                    for (int i = 0; i < n_ref; ++i) {
                        if (std::isnan(g(i))) X(i, valid_idx.size()) = 0.0;
                        else {
                            double centered = g(i) - mean;
                            X(i, valid_idx.size()) = centered;
                            var += centered * centered;
                        }
                    }
                    double sd = var > 0 ? std::sqrt(var / (n_ref - 1)) : 1.0;
                    if (sd > 0) X.col(valid_idx.size()) /= sd;

                    b_marg(valid_idx.size()) = eff;
                    valid_idx.push_back(j);
                }
            }

            int v_size = valid_idx.size();
            if (v_size > 0) {
                Eigen::MatrixXd X_sub = X.leftCols(v_size);
                Eigen::VectorXd b_sub = b_marg.head(v_size);

                Eigen::MatrixXd R = (X_sub.transpose() * X_sub) / (n_ref - 1);
                R.diagonal().array() += (lambda / n_ref);

                Eigen::VectorXd b_joint = R.llt().solve(b_sub);

                for (int k = 0; k < v_size; ++k) {
                    int j = valid_idx[k];
                    out_snps.push_back(bim[j].id);
                    a1s.push_back(bim[j].alt);
                    a2s.push_back(bim[j].ref);
                    sblup_effs.push_back(b_joint(k));
                }
            }
            if (start % 50000 == 0 && start > 0) LOG_INFO("Processed " << start << " SNPs for SBLUP.");
        }
    } else if (!cfg.pfile_path.empty()) {
        PgenReader reader;
        reader.open(cfg.pfile_path);
        const auto& variants = reader.getVariants();
        int n_ref = static_cast<int>(reader.getNumSamples());
        int m = static_cast<int>(variants.size());

        double lambda = m * (1.0 - cfg.h2) / (cfg.h2);
        LOG_INFO("Calculated regularization lambda = " << lambda);

        int win_size = cfg.window_num > 0 ? cfg.window_num : 1000;
        LOG_INFO("Using LD block size of " << win_size << " SNPs (PGEN sequential mode).");

        int processed = 0;
        while (processed < m) {
            int b_size = std::min(win_size, m - processed);

            Eigen::MatrixXd X(n_ref, b_size);
            Eigen::VectorXd b_marg = Eigen::VectorXd::Zero(b_size);
            std::vector<std::string> valid_ids;
            std::vector<std::string> valid_a1s;
            std::vector<std::string> valid_a2s;
            Eigen::VectorXf dosage;

            for (int j = 0; j < b_size; ++j) {
                std::string chrom, rsid, ref, alt;
                uint32_t pos = 0;
                if (!reader.readVariant(chrom, rsid, pos, ref, alt, dosage)) {
                    LOG_WARN("PGEN reader reached EOF after " << processed << " variants during SBLUP; stopping streaming read.");
                    processed = m;
                    break;
                }
                processed++;
                auto it = sumstat.find(rsid);
                if (it == sumstat.end()) continue;

                double eff = std::get<2>(it->second);
                std::string eff_a1 = std::get<0>(it->second);
                if (alt != eff_a1) {
                    if (ref == eff_a1) eff = -eff;
                    else continue;
                }

                int col_idx = static_cast<int>(valid_ids.size());
                double mean = 0.0, count = 0.0;
                for (int i = 0; i < n_ref; ++i) {
                    if (!std::isnan(dosage(i))) {
                        mean += dosage(i);
                        count += 1.0;
                    }
                }
                mean = count > 0 ? mean / count : 0.0;

                double var = 0.0;
                for (int i = 0; i < n_ref; ++i) {
                    if (std::isnan(dosage(i))) X(i, col_idx) = 0.0;
                    else {
                        double centered = dosage(i) - mean;
                        X(i, col_idx) = centered;
                        var += centered * centered;
                    }
                }
                double sd = var > 0 ? std::sqrt(var / std::max(1, n_ref - 1)) : 1.0;
                if (sd > 0) X.col(col_idx) /= sd;

                b_marg(col_idx) = eff;
                valid_ids.push_back(rsid);
                valid_a1s.push_back(alt);
                valid_a2s.push_back(ref);
            }

            int v_size = static_cast<int>(valid_ids.size());
            if (v_size > 0) {
                Eigen::MatrixXd X_sub = X.leftCols(v_size);
                Eigen::VectorXd b_sub = b_marg.head(v_size);
                Eigen::MatrixXd R = (X_sub.transpose() * X_sub) / std::max(1, n_ref - 1);
                R.diagonal().array() += (lambda / std::max(1, n_ref));

                Eigen::VectorXd b_joint = R.llt().solve(b_sub);
                for (int k = 0; k < v_size; ++k) {
                    out_snps.push_back(valid_ids[k]);
                    a1s.push_back(valid_a1s[k]);
                    a2s.push_back(valid_a2s[k]);
                    sblup_effs.push_back(b_joint(k));
                }
            }
            if (processed % 50000 == 0 && processed > 0) LOG_INFO("Processed " << processed << " PGEN variants for SBLUP.");
        }
    }

    std::string out_file = cfg.out_prefix + ".sblup.eff";
    std::ofstream fout(out_file);
    fout << "SNP\tA1\tA2\tSBLUP_EFFECT\n";
    for (size_t i = 0; i < out_snps.size(); ++i) fout << out_snps[i] << "\t" << a1s[i] << "\t" << a2s[i] << "\t" << sblup_effs[i] << "\n";
    LOG_INFO("SBLUP calculation complete. Effects written to [" << out_file << "].");
}

// ── Prediction Mode ─────────────────────────────────────────────

void runPredictionMode(const Config& cfg) {
    LOG_INFO("=================================================");
    LOG_INFO("     GEBV/GPRS Prediction Mode Selected          ");
    LOG_INFO("=================================================");
    if (cfg.score_path.empty()) throw std::runtime_error("--score is required for GEBV prediction.");
    if (cfg.bfile_path.empty() && cfg.pfile_path.empty()) throw std::runtime_error("--bfile or --pfile is required for GEBV prediction.");

    LOG_INFO("Genotype file       : " << (cfg.bfile_path.empty() ? cfg.pfile_path : cfg.bfile_path));
    LOG_INFO("SNP effects file    : " << cfg.score_path);

    std::map<std::string, std::tuple<std::string, std::string, double>> snp_effects;
    std::ifstream fs(cfg.score_path);
    if (!fs) throw std::runtime_error("Cannot open --score file.");
    std::string line;

    int snp_col = -1, a1_col = -1, a2_col = -1, beta_col = -1;
    if (std::getline(fs, line)) {
        std::istringstream iss(line);
        std::string token;
        int col = 0;
        while (iss >> token) {
            std::string t = token;
            for (auto& c : t) c = std::toupper(c);
            if (t == "SNP" || t == "VARIANT_ID" || t == "ID") snp_col = col;
            else if (t == "A1" || t == "EFFECT_ALLELE" || t == "ALLELE1") a1_col = col;
            else if (t == "A2" || t == "OTHER_ALLELE" || t == "ALLELE2") a2_col = col;
            else if (t == "BETA" || t == "BETA_1" || t == "SCORE" || t == "EFFECT" || t == "SBLUP_EFFECT") beta_col = col;
            col++;
        }
        if (snp_col == -1 || a1_col == -1 || beta_col == -1) {
            LOG_WARN("Could not perfectly identify header in --score. Assuming 1:SNP, 2:A1, 3:A2, 4:EFFECT.");
            snp_col = 0; a1_col = 1; a2_col = 2; beta_col = 3;
            fs.clear();
            fs.seekg(0, std::ios::beg);
        }
    }

    while (std::getline(fs, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (iss >> token) tokens.push_back(token);
        if (tokens.size() <= std::max({snp_col, a1_col, a2_col, beta_col})) continue;

        std::string snp = tokens[snp_col];
        std::string a1 = tokens[a1_col];
        std::string a2 = a2_col >= 0 ? tokens[a2_col] : "";
        double eff = 0.0;
        try { eff = std::stod(tokens[beta_col]); } catch (...) { continue; }

        snp_effects[snp] = {a1, a2, eff};
    }
    LOG_INFO("Loaded " << snp_effects.size() << " valid SNP effects.");

    std::vector<std::string> iids;
    Eigen::VectorXd gebv;
    int matched_snps = 0;

    if (!cfg.bfile_path.empty()) {
        PlinkReader reader;
        reader.load(cfg.bfile_path);
        const auto& bim = reader.getBimInfo();
        const auto& fam = reader.getFamInfo();
        int n = fam.size();
        int m = bim.size();
        for (const auto& f : fam) iids.push_back(f.iid.empty() ? f.fid : f.iid);
        gebv = Eigen::VectorXd::Zero(n);

        for (int j = 0; j < m; ++j) {
            auto it = snp_effects.find(bim[j].id);
            if (it != snp_effects.end()) {
                double eff = std::get<2>(it->second);
                std::string eff_a1 = std::get<0>(it->second);
                if (bim[j].alt != eff_a1) {
                    if (bim[j].ref == eff_a1) eff = -eff;
                    else continue;
                }
                matched_snps++;
                Eigen::VectorXf geno; reader.getSnpGenotypes(j, geno);
                for (int i = 0; i < n; ++i) {
                    if (!std::isnan(geno(i))) gebv(i) += geno(i) * eff;
                }
            }
        }
    } else if (!cfg.pfile_path.empty()) {
        PgenReader reader;
        reader.open(cfg.pfile_path);
        const auto& sample_ids = reader.getSampleIds();
        for (const auto& id : sample_ids) iids.push_back(id);
        gebv = Eigen::VectorXd::Zero(static_cast<int>(iids.size()));

        std::string chrom, rsid, ref, alt;
        uint32_t pos = 0;
        Eigen::VectorXf dosage;
        while (reader.readVariant(chrom, rsid, pos, ref, alt, dosage)) {
            auto it = snp_effects.find(rsid);
            if (it == snp_effects.end()) continue;

            double eff = std::get<2>(it->second);
            std::string eff_a1 = std::get<0>(it->second);
            if (alt != eff_a1) {
                if (ref == eff_a1) eff = -eff;
                else continue;
            }
            matched_snps++;
            for (int i = 0; i < dosage.size() && i < gebv.size(); ++i) {
                if (!std::isnan(dosage(i))) gebv(i) += dosage(i) * eff;
            }
        }
    }

    LOG_INFO("Matched " << matched_snps << " SNPs between effect file and genotype.");
    std::string out_file = cfg.out_prefix + ".profile";
    std::ofstream fout(out_file);
    fout << "ID\tSCORE\n";
    for (size_t i = 0; i < iids.size(); ++i) fout << iids[i] << "\t" << gebv(i) << "\n";
    LOG_INFO("Prediction complete. Scores written to [" << out_file << "].");
}

} // namespace cosmic
