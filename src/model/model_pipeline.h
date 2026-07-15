#pragma once
#include "variance_estimator.h"
#include "mixed_model_solver.h"
#include "random_regression.h"
#include "config.h"
#include "mme_builder.h"
#include "matrix_adapter.h"
#include <memory>
#include <string>

namespace cosmic {

/// Unified GS model pipeline: VCE → Solve → Output
/// Encapsulates the full GS workflow for any model type (PBLUP/GBLUP/ssGBLUP/RRM).
class GSModel {
public:
    /// Shared data context across all GS models.
    /// Populated once by the caller (SolverApp) before running the pipeline.
    struct SharedData {
        const std::vector<GenRecord>& recs;
        const FixedDesignG& fd;
        const Config& cfg;
        const std::map<std::string, int>& idmap;

        // Inverse matrices (any may be nullptr)
        AbstractMatrix* Qinv = nullptr;
        AbstractMatrix* GDinv = nullptr;
        AbstractMatrix* GEinv = nullptr;
        AbstractMatrix* PEinv = nullptr;
        AbstractMatrix* Ginv = nullptr;

        // STCG mode
        AbstractMatrix* stcg_ga_dim = nullptr;
        GenotypeMatrix* stcg_genotype = nullptr;

        // Generic random effects
        const std::vector<std::unique_ptr<AbstractMatrix>>& generic_rand_invs;
        const std::vector<std::vector<int>>& generic_rand_maps;

        // Maternal effect
        const std::vector<int>& mat_id_map;

        // RRM context (populated by GSRrm::setup if RRM model)
        const std::vector<RandomComponent>& base_components;
        int rrm_additive_component_count = 0;
        int rrm_pe_component_count = 0;
        int rrm_order = 2;
        double rrm_tmin = 0.0;
        double rrm_tmax = 0.0;
        const std::vector<double>& rrm_curve_times;

        // Flags
        bool force_implicit = false;
        bool stcg_mode = false;
        bool use_split = false;

        // Initial variance components
        double sigma2_u_init = 1.0;
        double sigma2_e_init = 1.0;
        bool run_vce = false;

        // Labels
        const std::string& inv_label;
        const std::vector<int>& genotyped_map;
    };

    /// Run the full GS pipeline: VCE (optional) → MME Solve → Output.
    /// Updates sigma2_u/sigma2_e with VCE results if run_vce is true.
    static void run(const SharedData& data, double& sigma2_u, double& sigma2_e);
};

} // namespace cosmic
