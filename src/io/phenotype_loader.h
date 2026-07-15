#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include <Eigen/Sparse>
#include "config.h"
#include "matrix_adapter.h"
#include "design.h"
#include "matrix_io.h"
#include "string_utils.h"

namespace cosmic {

class AbstractMatrix;

// ── Data loading result ──────────────────────────────────────────
struct DataLoadResult {
    // Phenotype
    std::vector<GenRecord> records;
    PhenStats phen_stats;

    // Variance components
    double sigma2_u = 1.0;
    double sigma2_e = 1.0;
    bool run_vce = false;

    // ID mapping (from matrix ID file)
    std::map<std::string, int> idmap;

    // Fixed design
    struct FixedDesignG fixed_design;
};

// ── DataLoader ───────────────────────────────────────────────────
// Unified data loading: phenotype, variance, ID mapping, fixed design.
// Matrix loading is handled by MatrixBuilder; this focuses on
// phenotype/variance/design construction.
class DataLoader {
public:
    static DataLoadResult load(const Config& cfg,
                               const std::map<std::string, int>& idmap_from_matrix);
};

// Phenotype reading for CosmicBLUP's tabular phenotype format.
// Matrix, ID, and variance I/O is provided by the integrated core utilities.
std::vector<GenRecord> readPhenGeneric(const std::string& phen_file,
                                       const std::map<std::string,int>& idmap,
                                       int trait_pos_1based,
                                       const std::vector<int>& dcovar_pos_1based,
                                       const std::vector<int>& qcovar_pos_1based,
                                       const std::vector<int>& rand_pos_1based,
                                       std::vector<std::string>& dcovar_names,
                                       std::vector<std::string>& qcovar_names,
                                       std::vector<std::string>& rand_names,
                                       const std::string& trait_name,
                                       const std::string& id_col_name,
                                       PhenStats* stats = nullptr);

} // namespace cosmic
