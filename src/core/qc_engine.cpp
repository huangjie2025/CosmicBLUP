#include "qc_engine.h"
#include "qc_utils.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

GenotypeQC::GenotypeQC(const QCOptions& opts) : options(opts) {}

void GenotypeQC::processGenotypes(int n_ind, int n_snp,
                                  std::function<void(int, Eigen::VectorXf&)> get_snp_row,
                                  const std::vector<std::string>& snp_ids,
                                  const std::vector<std::string>& snp_chroms,
                                  const std::vector<std::string>& snp_a1,
                                  const std::vector<std::string>& snp_a2,
                                  const std::vector<std::string>& ind_fids,
                                  const std::vector<std::string>& ind_iids,
                                  const std::string& out_prefix) {

    LOG_INFO("Starting Genotype QC for " << n_ind << " individuals and " << n_snp << " SNPs.");

    // Allocate memory for statistics
    std::vector<float> snp_maf(n_snp, 0.0f);
    std::vector<float> snp_miss(n_snp, 0.0f);
    std::vector<double> snp_hwe(n_snp, 1.0);
    std::vector<int> snp_hom1(n_snp, 0);
    std::vector<int> snp_het(n_snp, 0);
    std::vector<int> snp_hom2(n_snp, 0);

    // Individual-level statistics
    std::vector<int> ind_miss_count(n_ind, 0);
    std::vector<int> ind_het_count(n_ind, 0);
    std::vector<int> ind_valid_count(n_ind, 0);

#ifdef _OPENMP
    omp_set_num_threads(options.thread_num);
#endif

    // Process SNP by SNP sequentially to support variable-length compressed formats (PGEN/BGEN)
    Eigen::VectorXf dosage_buffer(n_ind);

    for (int i = 0; i < n_snp; ++i) {
        get_snp_row(i, dosage_buffer);

        int hom1 = 0, het = 0, hom2 = 0, miss = 0;

        for (int j = 0; j < n_ind; ++j) {
            float d = dosage_buffer(j);
            if (std::isnan(d)) {
                miss++;
                ind_miss_count[j]++;
            } else {
                ind_valid_count[j]++;
                int g = std::round(d);
                if (g == 0) hom1++;
                else if (g == 1) {
                    het++;
                    ind_het_count[j]++;
                }
                else if (g == 2) hom2++;
            }
        }

        snp_hom1[i] = hom1;
        snp_het[i] = het;
        snp_hom2[i] = hom2;

        int valid = hom1 + het + hom2;
        if (valid > 0) {
            double freq1 = (double)(2 * hom1 + het) / (2 * valid);
            snp_maf[i] = (float)std::min(freq1, 1.0 - freq1);
        } else {
            snp_maf[i] = 0.0f;
        }

        snp_miss[i] = (float)miss / n_ind;

        if (options.min_hwe > 0.0f && valid > 0) {
            snp_hwe[i] = calculateHWE(hom1, het, hom2);
        }
    }

    // Apply Filters for Individuals
    valid_inds.clear();
    std::vector<float> ind_miss_rate(n_ind, 0.0f);
    std::vector<float> ind_het_rate(n_ind, 0.0f);
    for (int j = 0; j < n_ind; ++j) {
        ind_miss_rate[j] = (float)ind_miss_count[j] / n_snp;
        if (ind_valid_count[j] > 0) {
            ind_het_rate[j] = (float)ind_het_count[j] / ind_valid_count[j];
        }

        if (ind_miss_rate[j] <= options.mind_miss) {
            valid_inds.push_back(j);
        }
    }

    // Apply Filters for SNPs
    valid_snps.clear();
    for (int i = 0; i < n_snp; ++i) {
        if (snp_miss[i] <= options.geno_miss &&
            snp_maf[i] >= options.min_maf &&
            snp_maf[i] <= options.max_maf &&
            snp_hwe[i] >= options.min_hwe) {
            valid_snps.push_back(i);
        }
    }

    LOG_INFO("QC Summary:");
    LOG_INFO("  Individuals kept: " << valid_inds.size() << " out of " << n_ind << " (failed MIND: " << (n_ind - valid_inds.size()) << ")");
    LOG_INFO("  SNPs kept: " << valid_snps.size() << " out of " << n_snp << " (failed MAF/GENO/HWE: " << (n_snp - valid_snps.size()) << ")");

    // Write Stats if requested
    if (options.write_stats && !out_prefix.empty()) {
        std::string snp_file = out_prefix + ".qc.snp";
        std::ofstream fsnp(snp_file);
        if (fsnp) {
            fsnp << "CHR\tSNP\tA1\tA2\tHOM1\tHET\tHOM2\tMISS\tMAF\tHWE_P\tPASS\n";
            for (int i = 0; i < n_snp; ++i) {
                bool pass = (snp_miss[i] <= options.geno_miss &&
                             snp_maf[i] >= options.min_maf &&
                             snp_maf[i] <= options.max_maf &&
                             snp_hwe[i] >= options.min_hwe);
                fsnp << snp_chroms[i] << "\t" << snp_ids[i] << "\t"
                     << snp_a1[i] << "\t" << snp_a2[i] << "\t"
                     << snp_hom1[i] << "\t" << snp_het[i] << "\t" << snp_hom2[i] << "\t"
                     << snp_miss[i] << "\t" << snp_maf[i] << "\t" << snp_hwe[i] << "\t" << (pass ? "1" : "0") << "\n";
            }
            LOG_INFO("SNP QC stats written to " << snp_file);
        }

        std::string ind_file = out_prefix + ".qc.id";
        std::ofstream find(ind_file);
        if (find) {
            find << "FID\tIID\tMISS_RATE\tHET_RATE\tPASS\n";
            for (int j = 0; j < n_ind; ++j) {
                bool pass = (ind_miss_rate[j] <= options.mind_miss);
                find << ind_fids[j] << "\t" << ind_iids[j] << "\t"
                     << ind_miss_rate[j] << "\t" << ind_het_rate[j] << "\t" << (pass ? "1" : "0") << "\n";
            }
            LOG_INFO("Individual QC stats written to " << ind_file);
        }

        std::string keep_snp = out_prefix + ".keep.snp";
        std::ofstream fksnp(keep_snp);
        if (fksnp) {
            for (int idx : valid_snps) fksnp << snp_ids[idx] << "\n";
            LOG_INFO("Valid SNP IDs written to " << keep_snp);
        }

        std::string keep_id = out_prefix + ".keep.id";
        std::ofstream fkid(keep_id);
        if (fkid) {
            for (int idx : valid_inds) fkid << ind_fids[idx] << "\t" << ind_iids[idx] << "\n";
            LOG_INFO("Valid Individual IDs written to " << keep_id);
        }
    }
}

void GenotypeQC::run(const PlinkReader& reader, const std::string& out_prefix) {
    int n_ind = reader.getNumSamples();
    int n_snp = reader.getNumSnps();

    const auto& bim = reader.getBimInfo();
    const auto& fam = reader.getFamInfo();

    std::vector<std::string> snp_ids(n_snp), snp_chroms(n_snp), snp_a1(n_snp), snp_a2(n_snp);
    for (int i = 0; i < n_snp; ++i) {
        snp_ids[i] = bim[i].id;
        snp_chroms[i] = bim[i].chrom;
        snp_a1[i] = bim[i].alt; // PLINK logic A1
        snp_a2[i] = bim[i].ref; // PLINK logic A2
    }

    std::vector<std::string> ind_fids(n_ind), ind_iids(n_ind);
    for (int i = 0; i < n_ind; ++i) {
        ind_fids[i] = fam[i].fid;
        ind_iids[i] = fam[i].iid;
    }

    auto get_row = [&](int snp_idx, Eigen::VectorXf& buf) {
        reader.getSnpGenotypes(snp_idx, buf);
    };

    processGenotypes(n_ind, n_snp, get_row, snp_ids, snp_chroms, snp_a1, snp_a2, ind_fids, ind_iids, out_prefix);
}

void GenotypeQC::run(PgenReader& reader, const std::string& out_prefix) {
    int n_ind = reader.getNumSamples();
    int n_snp = reader.getNumVariants();

    const auto& pvar = reader.getVariants();
    std::vector<std::string> sample_ids = reader.getSampleIds();

    std::vector<std::string> snp_ids(n_snp), snp_chroms(n_snp), snp_a1(n_snp), snp_a2(n_snp);
    for (int i = 0; i < n_snp; ++i) {
        snp_ids[i] = pvar[i].id;
        snp_chroms[i] = pvar[i].chrom;
        snp_a1[i] = pvar[i].alt;
        snp_a2[i] = pvar[i].ref;
    }

    std::vector<std::string> ind_fids(n_ind, "0"), ind_iids(n_ind);
    for (int i = 0; i < n_ind; ++i) {
        ind_iids[i] = sample_ids[i];
    }

    auto get_row = [&](int snp_idx, Eigen::VectorXf& buf) {
        std::string c, rs, ref, alt;
        uint32_t pos;
        reader.readVariant(c, rs, pos, ref, alt, buf);
    };

    processGenotypes(n_ind, n_snp, get_row, snp_ids, snp_chroms, snp_a1, snp_a2, ind_fids, ind_iids, out_prefix);
}

void GenotypeQC::run(BgenReader& reader, const std::string& out_prefix) {
    int n_ind = reader.getNumSamples();
    int n_snp = reader.getNumVariants();

    const auto& vars = reader.getVariants();
    std::vector<std::string> sample_ids = reader.getSampleIds();

    std::vector<std::string> snp_ids(n_snp), snp_chroms(n_snp), snp_a1(n_snp), snp_a2(n_snp);
    for (int i = 0; i < n_snp; ++i) {
        snp_ids[i] = vars[i].rsid;
        snp_chroms[i] = vars[i].chrom;
        snp_a1[i] = vars[i].alleles.size() > 1 ? vars[i].alleles[1] : "";
        snp_a2[i] = vars[i].alleles.size() > 0 ? vars[i].alleles[0] : "";
    }

    std::vector<std::string> ind_fids(n_ind, "0"), ind_iids(n_ind);
    for (int i = 0; i < n_ind; ++i) {
        ind_iids[i] = sample_ids[i];
    }

    auto get_row = [&](int snp_idx, Eigen::VectorXf& buf) {
        std::string c, rs, ref, alt;
        uint32_t pos;
        reader.readVariant(c, rs, pos, ref, alt, buf);
    };

    processGenotypes(n_ind, n_snp, get_row, snp_ids, snp_chroms, snp_a1, snp_a2, ind_fids, ind_iids, out_prefix);
}

} // namespace cosmic
