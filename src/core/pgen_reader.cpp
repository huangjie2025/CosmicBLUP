#include "pgen_reader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <zstd.h>
#include <cstring>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace cosmic {

// PGEN Constants
static const uint16_t kPgenMagic = 0x1B6C; // 0x6C, 0x1B
static const uint8_t kPgenTypeUncompressed = 0;

PgenReader::PgenReader() {}

PgenReader::~PgenReader() {
    close();
}

void PgenReader::open(const std::string& pgen_file, const std::string& pvar_file, const std::string& psam_file) {
    auto has_suffix = [](const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    const std::string prefix = has_suffix(pgen_file, ".pgen")
        ? pgen_file.substr(0, pgen_file.size() - 5)
        : pgen_file;
    pgen_path = has_suffix(pgen_file, ".pgen") ? pgen_file : prefix + ".pgen";
    pvar_path = pvar_file.empty() ? prefix + ".pvar" : pvar_file;
    psam_path = psam_file.empty() ? prefix + ".psam" : psam_file;

    // Open MMap
    mmap_file.open(pgen_path);
    if (!mmap_file.is_open()) {
        throw std::runtime_error("Failed to open PGEN file: " + pgen_path);
    }

    readPgenHeader();
    readPvar();
    readPsam();

    // Validate
    if (variants.size() != n_variants_pgen) {
        std::cerr << "Warning: Number of variants in PGEN header (" << n_variants_pgen
                  << ") does not match PVAR (" << variants.size() << "). Using PVAR count.\n";
        num_variants = (uint32_t)variants.size();
    } else {
        num_variants = n_variants_pgen;
    }

    if (sample_ids.size() != n_samples_pgen) {
        std::cerr << "Warning: Number of samples in PGEN header (" << n_samples_pgen
                  << ") does not match PSAM (" << sample_ids.size() << "). Using PSAM count.\n";
        num_samples = (uint32_t)sample_ids.size();
    } else {
        num_samples = n_samples_pgen;
    }

    reset();
    stats_computed = false;
}

void PgenReader::close() {
    mmap_file.close();
    variants.clear();
    sample_ids.clear();
    snp_means.clear();
    snp_inv_stds.clear();
    stats_computed = false;
}

void PgenReader::reset() {
    current_offset = variant_block_offset;
    current_variant_idx = 0;
}

// Helper to read PGEN length (1-5 bytes)
static uint32_t readPgenLength(const uint8_t*& ptr) {
    uint8_t b1 = *ptr++;
    if (b1 < 253) return b1;
    if (b1 == 253) {
        uint32_t len = *reinterpret_cast<const uint16_t*>(ptr);
        ptr += 2;
        return len;
    }
    if (b1 == 254) {
        uint32_t len = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16);
        ptr += 3;
        return len;
    }
    // b1 == 255
    uint32_t len = *reinterpret_cast<const uint32_t*>(ptr);
    ptr += 4;
    return len;
}

// Helper to get raw variant data (decompressing if needed)
// Returns pointer to data and its length. Updates current_offset.
// type is updated to inner type if decompressed.
// decomp_buf is used for decompression storage.
static const uint8_t* getVariantData(const uint8_t* mmap_data, size_t file_size,
                                     size_t& offset, uint8_t& type, size_t& len,
                                     std::vector<char>& decomp_buf, uint32_t num_samples) {
    if (offset >= file_size) return nullptr;

    type = mmap_data[offset++];

    if (type == kPgenTypeUncompressed) {
        // Uncompressed (Type 0)
        len = (num_samples + 3) / 4;
        if (offset + len > file_size) return nullptr;
        const uint8_t* ptr = mmap_data + offset;
        offset += len;
        return ptr;
    } else {
        // Compressed or Sparse (Type > 0)
        // Read length of following data
        const uint8_t* ptr_len = mmap_data + offset;
        uint32_t data_len = readPgenLength(ptr_len);
        offset = (ptr_len - mmap_data); // Update offset after length read

        if (offset + data_len > file_size) return nullptr;

        const uint8_t* data_ptr = mmap_data + offset;
        offset += data_len;

        // Check for Zstd Compression (Type >= 64)
        if (type >= 64) {
            uint8_t inner_type = type & 63;

            // Decompress
            unsigned long long const dSize = ZSTD_getFrameContentSize(data_ptr, data_len);
            if (dSize == ZSTD_CONTENTSIZE_ERROR || dSize == ZSTD_CONTENTSIZE_UNKNOWN) {
                // Fallback: Use expected size for Type 0 or grow?
                // For Type 0, expected size is (N+3)/4.
                // Let's assume we can resize to expected uncompressed size if Type 0.
                if (inner_type == kPgenTypeUncompressed) {
                    size_t expected = (num_samples + 3) / 4;
                    if (decomp_buf.size() < expected) decomp_buf.resize(expected);
                } else {
                    // Unknown size and unknown type -> unsafe.
                    // Try a reasonable buffer size?
                    if (decomp_buf.size() < 65536) decomp_buf.resize(65536);
                }
            } else {
                if (decomp_buf.size() < dSize) decomp_buf.resize(dSize);
            }

            size_t const decompressed_size = ZSTD_decompress(decomp_buf.data(), decomp_buf.size(), data_ptr, data_len);
            if (ZSTD_isError(decompressed_size)) {
                // std::cerr << "ZSTD Error: " << ZSTD_getErrorName(decompressed_size) << "\n";
                return nullptr;
            }

            type = inner_type;
            len = decompressed_size;
            return (const uint8_t*)decomp_buf.data();
        } else {
            // Not Zstd compressed (e.g. Type 1-63 stored directly)
            // Return raw data
            len = data_len;
            return data_ptr;
        }
    }
}

void PgenReader::computeStats(int n_threads) {
    if (stats_computed) return;

    reset();
    snp_means.assign(num_variants, 0.0);
    snp_inv_stds.assign(num_variants, 0.0);

    const uint8_t* mmap_data = (const uint8_t*)mmap_file.get_data();
    size_t file_size = mmap_file.get_size();

    std::cout << "Computing stats for " << num_variants << " variants in PGEN (MMap, Zstd support)..." << std::endl;

    // Local buffer for decompression (reused)
    // Note: computeStats is single-threaded at variant level, so member decompressed_buffer is safe.
    // Or we can use a local one.

    for (size_t j = 0; j < num_variants; ++j) {
        uint8_t type;
        size_t len;
        const uint8_t* ptr = getVariantData(mmap_data, file_size, current_offset, type, len, decompressed_buffer, num_samples);

        if (!ptr) break;

        if (type == kPgenTypeUncompressed) {
            // Process Uncompressed (Type 0)
            long long sum = 0;
            long long sum_sq = 0;
            long long count = 0;

            #pragma omp parallel num_threads(n_threads) reduction(+:sum, sum_sq, count)
            {
                long long local_sum = 0;
                long long local_sum_sq = 0;
                long long local_count = 0;

                #pragma omp for
                for (long long i = 0; i < (long long)num_samples; ++i) {
                    size_t byte_idx = i / 4;
                    size_t bit_idx = (i % 4) * 2;
                    if (byte_idx < len) {
                        uint8_t val = (ptr[byte_idx] >> bit_idx) & 3;
                        if (val != 3) {
                            local_sum += val;
                            local_sum_sq += (val * val);
                            local_count++;
                        }
                    }
                }
                sum += local_sum;
                sum_sq += local_sum_sq;
                count += local_count;
            }

            if (count > 0) {
                double mean = (double)sum / count;
                double var = (double)sum_sq / count - mean * mean;
                if (var > 1e-9) {
                    snp_means[j] = mean;
                    snp_inv_stds[j] = 1.0 / std::sqrt(var);
                } else {
                    snp_means[j] = mean;
                    snp_inv_stds[j] = 0.0;
                }
            } else {
                snp_means[j] = 0.0;
                snp_inv_stds[j] = 0.0;
            }
        } else {
            // Other types (Sparse, etc.) - Skip for now or implement later
            // std::cerr << "Warning: Skipping unsupported inner type " << (int)type << "\n";
            snp_means[j] = NAN;
            snp_inv_stds[j] = 0.0;
        }
    }

    stats_computed = true;
    reset();
}

void PgenReader::multiply_Z_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads) {
    if (!stats_computed) computeStats(n_threads);
    if (v.size() != (long long)num_variants) throw std::runtime_error("Dimension mismatch in multiply_Z_v");

    reset();
    out.setZero(num_samples);

    const uint8_t* mmap_data = (const uint8_t*)mmap_file.get_data();
    size_t file_size = mmap_file.get_size();

    for (size_t j = 0; j < num_variants; ++j) {
        uint8_t type;
        size_t len;
        const uint8_t* ptr = getVariantData(mmap_data, file_size, current_offset, type, len, decompressed_buffer, num_samples);

        if (!ptr) break;

        if (type == kPgenTypeUncompressed) {
            double inv_std = snp_inv_stds[j];
            if (inv_std == 0.0 || std::isnan(snp_means[j])) continue;

            double mean = snp_means[j];
            double weight = v(j) * inv_std;

            double val0 = (0.0 - mean) * weight;
            double val1 = (1.0 - mean) * weight;
            double val2 = (2.0 - mean) * weight;

            #pragma omp parallel num_threads(n_threads)
            {
                Eigen::VectorXd out_local = Eigen::VectorXd::Zero(num_samples);

                #pragma omp for
                for (long long i = 0; i < (long long)num_samples; ++i) {
                    size_t byte_idx = i / 4;
                    size_t bit_idx = (i % 4) * 2;
                    if (byte_idx < len) {
                        uint8_t val = (ptr[byte_idx] >> bit_idx) & 3;
                        if (val == 0) out_local(i) += val0;
                        else if (val == 1) out_local(i) += val1;
                        else if (val == 2) out_local(i) += val2;
                    }
                }

                #pragma omp critical
                {
                    out += out_local;
                }
            }
        }
        // Skip other types
    }
    reset();
}

void PgenReader::multiply_Zt_v(const Eigen::VectorXd& v, Eigen::VectorXd& out, int n_threads) {
    if (!stats_computed) computeStats(n_threads);
    if (v.size() != (long long)num_samples) throw std::runtime_error("Dimension mismatch in multiply_Zt_v");

    reset();
    out.setZero(num_variants);

    const uint8_t* mmap_data = (const uint8_t*)mmap_file.get_data();
    size_t file_size = mmap_file.get_size();

    for (size_t j = 0; j < num_variants; ++j) {
        uint8_t type;
        size_t len;
        const uint8_t* ptr = getVariantData(mmap_data, file_size, current_offset, type, len, decompressed_buffer, num_samples);

        if (!ptr) break;

        if (type == kPgenTypeUncompressed) {
            double inv_std = snp_inv_stds[j];
            if (inv_std == 0.0 || std::isnan(snp_means[j])) {
                out(j) = 0.0;
                continue;
            }

            double mean = snp_means[j];
            double dot_gv = 0.0;
            double sum_v = 0.0;

            #pragma omp parallel num_threads(n_threads) reduction(+:dot_gv, sum_v)
            {
                double local_dot = 0.0;
                double local_sum = 0.0;

                #pragma omp for
                for (long long i = 0; i < (long long)num_samples; ++i) {
                    size_t byte_idx = i / 4;
                    size_t bit_idx = (i % 4) * 2;
                    if (byte_idx < len) {
                        uint8_t val = (ptr[byte_idx] >> bit_idx) & 3;
                        if (val != 3) {
                            double g = (double)val;
                            local_dot += g * v(i);
                            local_sum += v(i);
                        }
                    }
                }
                dot_gv += local_dot;
                sum_v += local_sum;
            }

            out(j) = (dot_gv - mean * sum_v) * inv_std;
        } else {
            out(j) = 0.0;
        }
    }
    reset();
}

void PgenReader::readPsam() {
    std::ifstream in(psam_path);
    if (!in) return;

    sample_ids.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string fid, iid;
        ss >> fid >> iid; // Assuming FID IID columns
        sample_ids.push_back(fid + ":" + iid);
    }
}

void PgenReader::readPvar() {
    std::ifstream in(pvar_path);
    if (!in) return;

    variants.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        PgenVariant v;
        ss >> v.chrom >> v.pos >> v.id >> v.ref >> v.alt;
        variants.push_back(v);
    }
}

void PgenReader::readPgenHeader() {
    const uint8_t* data = (const uint8_t*)mmap_file.get_data();
    size_t size = mmap_file.get_size();

    if (size < 12) throw std::runtime_error("PGEN file too small");

    size_t offset = 0;

    // Magic
    if (data[0] != 0x6C || data[1] != 0x1B) {
        throw std::runtime_error("Invalid PGEN magic number");
    }
    offset += 2;

    // Storage Mode
    storage_mode = data[offset++];

    // Num Variants
    n_variants_pgen = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;

    // Num Samples
    n_samples_pgen = *reinterpret_cast<const uint32_t*>(data + offset);
    offset += 4;

    // Header flags (1 byte)
    // char flags_byte;
    offset++; // flags_byte

    variant_block_offset = 12;
    current_offset = variant_block_offset;
}

bool PgenReader::decompressZstd(const char* src, size_t src_len, std::vector<char>& dst, size_t dst_len) {
    size_t const dSize = ZSTD_decompress(dst.data(), dst_len, src, src_len);
    if (ZSTD_isError(dSize)) {
        std::cerr << "ZSTD Error: " << ZSTD_getErrorName(dSize) << std::endl;
        return false;
    }
    return dSize == dst_len;
}

bool PgenReader::readVariant(std::string& chrom, std::string& rsid, uint32_t& pos,
                             std::string& ref, std::string& alt,
                             Eigen::VectorXf& dosage) {
    if (current_variant_idx >= num_variants) return false;
    if (current_variant_idx >= variants.size()) return false;

    const auto& v = variants[current_variant_idx];
    chrom = v.chrom;
    rsid = v.id;
    pos = v.pos;
    ref = v.ref;
    alt = v.alt;

    const uint8_t* mmap_data = (const uint8_t*)mmap_file.get_data();
    size_t file_size = mmap_file.get_size();

    uint8_t type;
    size_t len;
    const uint8_t* ptr = getVariantData(mmap_data, file_size, current_offset, type, len, decompressed_buffer, num_samples);

    if (!ptr) return false;

    dosage.resize(num_samples);

    if (type == kPgenTypeUncompressed) {
        for (size_t i = 0; i < num_samples; ++i) {
            size_t byte_idx = i / 4;
            size_t bit_idx = (i % 4) * 2;

            if (byte_idx < len) {
                uint8_t geno = (ptr[byte_idx] >> bit_idx) & 0x03;

                if (geno == 0) dosage[i] = 0.0f;
                else if (geno == 1) dosage[i] = 1.0f;
                else if (geno == 2) dosage[i] = 2.0f;
                else dosage[i] = std::numeric_limits<float>::quiet_NaN();
            } else {
                 dosage[i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    } else {
        // Handle Compressed Types (Skip)
        // std::cerr << "Warning: PGEN compression inner type " << (int)type << " not supported yet. Setting all to NAN.\n";
        dosage.setConstant(std::numeric_limits<float>::quiet_NaN());
    }

    current_variant_idx++;
    return true;
}

} // namespace cosmic
