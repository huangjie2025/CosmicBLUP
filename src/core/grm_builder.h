#pragma once
#include <string>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include "plink_reader.h"
#include "bgen_reader.h"
#include "pgen_reader.h"

namespace cosmic {

class GrmBuilder {
public:
    struct Options {
        int thread_num = 1;
        size_t block_size = 2048; // Number of SNPs to process at once
        float min_maf = 0.01f;    // Minimum Allele Frequency filter
        float min_hwe = 0.0f;     // Minimum HWE p-value filter (0.0 to disable)
        int algorithm = 1;        // 1: VanRaden, 2: Zeng, 3: Yang, 4: Vitezica, 5: TrueVar (Yang 2010 corrected)
        bool write_gcta_format = true;
        bool sparse = false;      // Enable sparse GRM computation
        bool force_slow = false;  // Force slow (float-based) computation for verification
        float sparse_threshold = 0.05f; // Threshold for sparse GRM
        std::string sparse_out_prefix; // Output prefix for sparse GRM
        std::string chrom_filter = ""; // If not empty, only use SNPs on this chromosome
        bool compute_homo_hete = false; // Whether to compute homo/hete counts
        std::string snp_weight_file = ""; // File containing SNP weights
        std::string pop_class_file = "";  // File containing population classifications
    };

    GrmBuilder();
    GrmBuilder(const Options& opts);

    // Compute GRM from the provided PlinkReader
    // keep_indices: Optional list of sample indices to include in GRM calculation (if empty, use all)
    void compute(const PlinkReader& reader, const std::vector<int>& keep_indices = {});

    // Compute GRM from the provided BgenReader
    // keep_indices: Optional list of sample indices to include
    void compute(BgenReader& reader, const std::vector<int>& keep_indices = {});

    // Compute GRM from the provided PgenReader
    // keep_indices: Optional list of sample indices to include
    void compute(PgenReader& reader, const std::vector<int>& keep_indices = {});

    // Save results to disk (GCTA format)
    void save(const std::string& out_prefix);

    // Save sparse GRM to disk
    void saveSparse(const std::string& out_prefix);

    // Load auxiliary files (snp_weight, pop_class)
    void loadAuxFiles();

    // Accessors
    const Eigen::MatrixXd& getGrm() const { return G; }
    double getSum2pq() const { return total_sum_2pq; }
    long long getTotalSnps() const { return total_valid_snps_count; }
    const std::vector<float>& getSnpCounts() const { return snp_counts; } // N SNPs per pair
    const std::set<std::string>& getObservedChroms() const { return observed_chroms; }
    const std::vector<float>& getAlleleFreqs() const { return allele_freqs; }
    const std::vector<int>& getHomoCounts() const { return homo_counts; }
    const std::vector<int>& getHeteCounts() const { return hete_counts; }
    const std::vector<int>& getMissingCounts() const { return missing_counts; }

private:
    Options options;
    Eigen::MatrixXd G;      // The resulting GRM (n x n)
    Eigen::MatrixXf N_snps; // Number of valid SNPs for each pair (n x n) - using float for easier matrix ops
    std::vector<float> snp_counts; // For saving (flattened lower triangle)
    std::vector<float> allele_freqs; // Stored allele frequencies
    std::vector<int> homo_counts; // Homozygous count per sample
    std::vector<int> hete_counts; // Heterozygous count per sample
    std::vector<int> missing_counts; // Missing count per sample
    double total_sum_2pq = 0.0;
    long long total_valid_snps_count = 0;
    std::set<std::string> observed_chroms;

    std::vector<std::string> sample_fids;
    std::vector<std::string> sample_iids;

    std::unordered_map<std::string, float> snp_weights;
    std::unordered_map<std::string, int> pop_map;
    int num_pops = 0;
    std::vector<int> ind_pop;

public:
    struct SplitGrmResult {
        Eigen::MatrixXd global;     // Normalized global GRM
        std::vector<std::string> chrom_labels;  // Chromosome names in order
        std::vector<Eigen::MatrixXd> per_chrom_raw;  // Raw G = Z_chr * Z_chr'
        std::vector<long long> per_chrom_snp_count;  // Valid SNPs per chromosome
        std::vector<double> per_chrom_sum_2pq;       // Sum 2p(1-p) per chromosome
        double global_sum_2pq = 0.0;
        long long global_snp_count = 0;
    };

    // Compute global + per-chromosome GRMs in a single pass over genotype data.
    // chrom_labels: set of chromosome names to split by.
    SplitGrmResult computeSplit(const PlinkReader& reader,
                                const std::set<std::string>& chrom_labels = {});

    SplitGrmResult computeSplit(BgenReader& reader,
                                const std::set<std::string>& chrom_labels = {},
                                const std::vector<int>& keep_indices = {});

    SplitGrmResult computeSplit(PgenReader& reader,
                                const std::set<std::string>& chrom_labels = {},
                                const std::vector<int>& keep_indices = {});

    // Sparse computation helpers
    void computeSparse(const PlinkReader& reader);
    void computeSparse(BgenReader& reader);
    void computeSparse(PgenReader& reader);

    // Sparse GRM storage (triplets: row, col, value)
    // We can't store all in memory. We'll write to a temporary file or keep in memory if sparse enough?
    // For 500k samples, even sparse matrix is huge.
    // We will write directly to disk in chunks.
};

} // namespace cosmic
