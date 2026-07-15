#pragma once
#include <string>
#include <vector>
#include <memory>
#include "mmap_matrix.h"
#include <Eigen/Dense>
#include "matrix_adapter.h"

namespace cosmic {

struct PlinkBimInfo {
    std::string chrom;
    std::string id;
    double cm_pos;
    long long bp_pos;
    std::string alt; // Allele 1
    std::string ref; // Allele 2
};

struct PlinkFamInfo {
    std::string fid;
    std::string iid;
    std::string father;
    std::string mother;
    int sex;
    double phenotype;
};

class PlinkReader : public GenotypeMatrix {
public:
    PlinkReader();
    ~PlinkReader();

    // Load .bed, .bim, .fam files based on prefix
    void load(const std::string& prefix);

    size_t getNumSamples() const { return num_samples; }
    size_t getNumSnps() const { return num_snps; }

    // GenotypeMatrix interface
    int rows() const override { return (int)num_samples; }
    int cols() const override { return (int)num_snps; }

    const std::vector<PlinkBimInfo>& getBimInfo() const { return bim_data; }
    const std::vector<PlinkFamInfo>& getFamInfo() const { return fam_data; }

    // Read genotype for a specific SNP into an Eigen Vector
    // Returns number of non-missing values
    // Output vector is resized to num_samples
    // Missing values are set to NAN
    size_t getSnpGenotypes(size_t snp_idx, Eigen::VectorXf& out_genotypes) const;

    // Read a block of SNPs (e.g. for matrix multiplication)
    // out_matrix is resized to (num_samples, end_snp_idx - start_snp_idx)
    void getSnpBlock(size_t start_snp_idx, size_t end_snp_idx, Eigen::MatrixXf& out_matrix) const;

    // Read a block of SNPs directly into MatrixXd (for LMM)
    void getSnpBlock(size_t start_snp_idx, size_t end_snp_idx, Eigen::MatrixXd& out_matrix) const;

    // Direct access to raw memory map data
    const uint8_t* getData() const { return static_cast<const uint8_t*>(bed_file.get_data()); }
    size_t getBytesPerSnp() const { return bytes_per_snp; }

private:
    MMapFile bed_file;
    std::vector<PlinkBimInfo> bim_data;
    std::vector<PlinkFamInfo> fam_data;

    size_t num_samples = 0;
    size_t num_snps = 0;
    size_t bytes_per_snp = 0;

    // Look-up table for PLINK 2-bit mapping to float (0, 1, 2, NAN)
    // 00 -> 2 (Homozygous Ref/Allele 1? Check convention)
    // PLINK bed format:
    // 00 -> Homozygous for first allele in .bim
    // 01 -> Missing
    // 10 -> Heterozygous
    // 11 -> Homozygous for second allele in .bim
    // Standard additive coding usually counts copies of Alternative/Minor allele.
    // If we assume Allele 1 is A1 and Allele 2 is A2 in .bim:
    // 00 -> A1/A1
    // 11 -> A2/A2
    // If we code A1 as reference (0) and A2 as effect (1), then 00->0, 11->2.
    // However, PLINK documentation says:
    // 00 -> Homozygous for first allele in .bim
    // 11 -> Homozygous for second allele in .bim
    // So 00 is A1A1, 11 is A2A2.
    // We will clarify the mapping in implementation.
    float lookup_table[4];

    void readBim(const std::string& path);
    void readFam(const std::string& path);
    void openBed(const std::string& path);

    // RHE / PCG Helpers
    // Precomputed stats for fast multiplication
    std::vector<double> snp_means;
    std::vector<double> snp_inv_stds; // 1 / sqrt(2p(1-p))
    bool stats_computed = false;

public:
    // Getters for stats
    const std::vector<double>& getMeans() const { return snp_means; }
    const std::vector<double>& getInvStds() const { return snp_inv_stds; }

    // Compute stats needed for standardized multiplication
    void computeStats(int n_threads = 1);

    // y = Z * v (Z is standardized genotype matrix)
    // v: M x 1 (SNP vector)
    // y: N x 1 (Sample vector)
    void multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, int n_threads = 1) override;

    // Subset support for Z * v
    void multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, const std::vector<int>& keep_indices, int n_threads = 1);

    // y = Z' * v
    // v: N x 1 (Sample vector)
    // y: M x 1 (SNP vector)
    void multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, int n_threads = 1) override;

    // Subset support for Z' * v
    void multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& y, const std::vector<int>& keep_indices, int n_threads = 1);

    // Read a single SNP into a dense vector
    void readSnp(int snp_idx, Eigen::VectorXd& out) const;
    void reset() {}
};

} // namespace cosmic
