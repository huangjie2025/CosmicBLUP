#ifndef COSMIC_BGEN_READER_H
#define COSMIC_BGEN_READER_H

#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <Eigen/Dense>
#include "matrix_adapter.h"

namespace cosmic {

// Structure to hold variant metadata from BGEN
struct BgenVariant {
    std::string id;
    std::string rsid;
    std::string chrom;
    uint32_t pos;
    uint16_t n_alleles;
    std::vector<std::string> alleles;
    uint32_t data_offset; // File offset to the start of genotype data block
    uint32_t data_length; // Length of compressed genotype data
};

class BgenReader : public GenotypeMatrix {
public:
    BgenReader();
    ~BgenReader();

    void open(const std::string& filename);
    void close();

    // Read header and sample block
    // Returns number of samples
    uint32_t initialize();

    // Reset reader to the beginning of variant data
    void reset();

    // Iterate through variants
    // Returns false if end of file
    // Fills variant metadata and dosage vector (size N)
    // Dosage = P(AB) + 2*P(BB) (assuming Ref=AA, Alt=BB)
    // Actually standard is: Dosage of Alt allele.
    // If alleles are A1, A2. Dosage of A2 = 1*P(A1A2) + 2*P(A2A2).
    bool readVariant(std::string& chrom, std::string& rsid, uint32_t& pos,
                     std::string& a1, std::string& a2,
                     Eigen::VectorXf& dosage);

    // For random access (requires indexing, which we might skip for now)
    // We will stick to sequential reading for scanning.

    // Compute allele frequencies and missingness stats
    void computeStats(int n_threads = 1);

    // Matrix-vector multiplication: out = Z * v
    void multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads = 1) override;

    // Matrix-vector multiplication: out = Z' * v
    void multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads = 1) override;

    const std::vector<BgenVariant>& getVariants() const { return variants; }
    uint32_t getNumSamples() const { return num_samples; }
    uint32_t getNumVariants() const { return num_variants; }
    const std::vector<std::string>& getSampleIds() const { return sample_ids; }
    int rows() const override { return static_cast<int>(num_samples); }
    int cols() const override { return static_cast<int>(num_variants); }

    // Start background prefetcher (optional)
    void startPrefetch();
    // Stop background prefetcher
    void stopPrefetch();

private:
    std::string filename;
    std::ifstream ifs;

    uint32_t num_samples = 0;
    uint32_t num_variants = 0;
    uint32_t header_length = 0;
    uint32_t flags = 0;
    uint32_t offset = 0; // Offset to first variant block

    std::vector<std::string> sample_ids;
    std::vector<BgenVariant> variants; // To store variant info if we parse them all

    // Stats
    bool stats_computed = false;
    std::vector<double> snp_means;
    std::vector<double> snp_inv_stds;

    // Internal buffers
    std::vector<char> compressed_buffer;
    std::vector<char> decompressed_buffer;

    // Background Prefetcher
    bool prefetch_enabled = false;
    void* prefetcher_ptr = nullptr; // Opaque pointer to implementation

    // Helper to read simple types
    template<typename T> T read(std::ifstream& in);
    std::string readString(std::ifstream& in, uint16_t len_bytes = 2);

    // Decompress using zlib
    bool decompress(const char* src, size_t src_len, std::vector<char>& dst, size_t dst_len);

    // Parse Layout 2 genotype data
    void parseLayout2(const char* data, size_t len, uint32_t n_samples, uint16_t n_alleles, Eigen::VectorXf& dosage);
};

} // namespace cosmic

#endif // COSMIC_BGEN_READER_H
