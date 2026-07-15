#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "mmap_matrix.h"
#include "matrix_adapter.h"

namespace cosmic {

struct PgenVariant {
    std::string chrom;
    std::string id;
    uint32_t pos;
    std::string ref;
    std::string alt;
};

class PgenReader : public GenotypeMatrix {
public:
    PgenReader();
    ~PgenReader();

    // Open .pgen, .pvar, .psam
    // If pvar/psam are empty, try to infer from pgen prefix
    void open(const std::string& pgen_file, const std::string& pvar_file = "", const std::string& psam_file = "");
    void close();
    void reset();

    uint32_t getNumSamples() const { return num_samples; }
    uint32_t getNumVariants() const { return num_variants; }
    const std::vector<std::string>& getSampleIds() const { return sample_ids; }
    int rows() const override { return static_cast<int>(num_samples); }
    int cols() const override { return static_cast<int>(num_variants); }

    // Read next variant
    // Returns false if EOF
    // Fills variant metadata and dosage vector (size N)
    // Dosage = expected count of Alt allele (0..2)
    bool readVariant(std::string& chrom, std::string& rsid, uint32_t& pos,
                     std::string& ref, std::string& alt,
                     Eigen::VectorXf& dosage);

    // Compute allele frequencies and missingness stats
    // Required for standardized multiplication
    void computeStats(int n_threads = 1);

    // Matrix-vector multiplication: out = Z * v
    // Z is standardized genotype matrix (n x m)
    // v is vector of size m
    // out is vector of size n
    // Requires computeStats() called first
    void multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads = 1) override;

    // Matrix-vector multiplication: out = Z' * v
    // Z is standardized genotype matrix (n x m)
    // v is vector of size n
    // out is vector of size m
    // Requires computeStats() called first
    void multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads = 1) override;

    // Getters for stats
    const std::vector<double>& getMeans() const { return snp_means; }
    const std::vector<double>& getInvStds() const { return snp_inv_stds; }

    // Memory Caching
    // Use Memory Mapping for efficient access
    void cacheAllData() { /* No-op, mmap is already active */ }
    bool isCached() const { return mmap_file.is_open(); }

    // Get Variant Info (BIM-like)
    const std::vector<PgenVariant>& getVariants() const { return variants; }

private:
    std::string pgen_path, pvar_path, psam_path;

    // Memory Mapped File
    MMapFile mmap_file;
    size_t current_offset = 0; // Current read position

    uint32_t num_samples = 0;
    uint32_t num_variants = 0;
    std::vector<std::string> sample_ids;
    std::vector<PgenVariant> variants;

    // Stats
    bool stats_computed = false;
    std::vector<double> snp_means;
    std::vector<double> snp_inv_stds;

    uint32_t current_variant_idx = 0;

    // PGEN Header info
    uint8_t version = 0;
    uint8_t storage_mode = 0;
    uint32_t variant_block_offset = 0;
    uint32_t n_samples_pgen = 0;
    uint32_t n_variants_pgen = 0;

    // Helpers
    void readPsam();
    void readPvar();
    void readPgenHeader();

    // Decompression buffer
    std::vector<char> compressed_buffer;
    std::vector<char> decompressed_buffer;

    // Helper to decompress Zstd
    // Returns true on success
    bool decompressZstd(const char* src, size_t src_len, std::vector<char>& dst, size_t dst_len);
};

} // namespace cosmic
