#pragma once
#include "config.h"
#include "matrix_adapter.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace cosmic {

/// Result of building inverse matrices (A/G/H).
struct MatrixBuildResult {
    std::unique_ptr<AbstractMatrix> Ainv;
    std::unique_ptr<AbstractMatrix> Ginv;
    std::unique_ptr<AbstractMatrix> A22inv;
    std::unique_ptr<AbstractMatrix> Qinv;      // primary inverse used by solver
    std::unique_ptr<AbstractMatrix> GDinv;     // dominance inverse
    std::unique_ptr<AbstractMatrix> GEinv;     // epistatic inverse
    std::unique_ptr<GenotypeMatrix> stcg_genotype;
    std::unique_ptr<IdentityMatrixAdapter> stcg_ga_dim;

    std::map<std::string, int> idmap;
    std::vector<int> genotyped_map;
    std::vector<int> global_dam_map;
    std::string inv_label;
    std::string inv_file;
    std::string id_file_log;
    bool use_split = false;
    bool stcg_mode = false;
};

/// Builds A-inverse, G-inverse, or H-inverse matrices from pedigree, genotype, or files.
class MatrixBuilder {
public:
    /// Build all required inverse matrices based on config.
    /// Returns ownership of all matrix objects via MatrixBuildResult.
    static MatrixBuildResult build(const Config& cfg);
};

} // namespace cosmic
