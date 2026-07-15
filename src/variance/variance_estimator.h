#pragma once
#include "config.h"
#include "design.h"
#include "mme_builder.h"
#include "plink_reader.h"
#include "matrix_adapter.h"
#include <vector>
#include <string>
#include <memory>

namespace cosmic {

struct VCEResult {
    double sigma2_e = 1.0;
    std::vector<double> vars_u;
    std::vector<double> se_u;
    double se_e = 0.0;
    Eigen::VectorXd solution;  // warm-start solution from VCE
    bool multi_trait_exit = false;  // early exit for multi-trait (legacy)
    // Multi-trait VCE results
    Eigen::MatrixXd mv_Vg;       // t x t genetic covariance
    Eigen::MatrixXd mv_Ve;       // t x t residual covariance
    Eigen::MatrixXd mv_Y;        // n_keep x t phenotype matrix (complete cases only)
    Eigen::MatrixXd mv_X;        // n_keep x p fixed-effect design
    std::vector<std::string> mv_trait_names;
    std::vector<std::string> mv_effect_names;
    // Indices (0-based, into the original n_geno G matrix) of individuals
    // retained after dropping rows with missing phenotypes. Used by
    // solveMultiTrait to slice G^{-1} consistently.
    std::vector<int> mv_keep_indices;
};

class GSVce {
public:
    struct Context {
        const std::vector<GenRecord>& recs;
        const FixedDesignG& fd;
        const std::map<std::string, int>& idmap;
        const Config& cfg;
        AbstractMatrix* Qinv = nullptr;
        AbstractMatrix* GDinv = nullptr;
        AbstractMatrix* GEinv = nullptr;
        AbstractMatrix* PEinv = nullptr;
        AbstractMatrix* stcg_ga_dim = nullptr;
        GenotypeMatrix* stcg_genotype = nullptr;
        const std::vector<std::unique_ptr<AbstractMatrix>>& generic_rand_invs;
        const std::vector<std::vector<int>>& generic_rand_maps;
        const std::vector<int>& mat_id_map;
        const std::vector<RandomComponent>& base_components;
        bool force_implicit = false;
        bool stcg_mode = false;
        int rrm_additive_component_count = 0;
        int rrm_pe_component_count = 0;
    };

    static VCEResult run(const Context& ctx, double sigma2_u_init, double sigma2_e_init);
};

} // namespace cosmic
