#pragma once
#include <string>
#include <vector>
#include <memory>
#include "plink_reader.h"
#include "pgen_reader.h"
#include "bgen_reader.h"

namespace cosmic {

struct QCOptions {
    int thread_num = 1;
    float min_maf = 0.0f;
    float max_maf = 1.0f;
    float min_hwe = 0.0f;
    float geno_miss = 1.0f; // max missing rate per SNP
    float mind_miss = 1.0f; // max missing rate per individual
    bool write_stats = true; // output .qc.snp and .qc.id stats
};

class GenotypeQC {
public:
    GenotypeQC(const QCOptions& opts);

    // Run QC on Plink files
    void run(const PlinkReader& reader, const std::string& out_prefix);

    // Run QC on PGEN files
    void run(PgenReader& reader, const std::string& out_prefix);

    // Run QC on BGEN files
    void run(BgenReader& reader, const std::string& out_prefix);

    // Get lists of valid indices (can be used to pass to GrmBuilder)
    const std::vector<int>& getValidSnps() const { return valid_snps; }
    const std::vector<int>& getValidInds() const { return valid_inds; }

private:
    QCOptions options;
    std::vector<int> valid_snps;
    std::vector<int> valid_inds;

    // Helper to run the generic QC logic given the genotype data provider
    void processGenotypes(int n_ind, int n_snp,
                          std::function<void(int, Eigen::VectorXf&)> get_snp_row,
                          const std::vector<std::string>& snp_ids,
                          const std::vector<std::string>& snp_chroms,
                          const std::vector<std::string>& snp_a1,
                          const std::vector<std::string>& snp_a2,
                          const std::vector<std::string>& ind_fids,
                          const std::vector<std::string>& ind_iids,
                          const std::string& out_prefix);
};

} // namespace cosmic
