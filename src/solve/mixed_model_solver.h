#pragma once
#include "config.h"
#include "design.h"
#include "mme_builder.h"
#include "matrix_adapter.h"
#include "variance_estimator.h"
#include <vector>
#include <string>
#include <memory>

namespace cosmic {

struct SolveResult {
    Eigen::VectorXd beta;
    Eigen::VectorXd u;
    Eigen::VectorXd beta_se;
    bool se_calculated = false;
    Eigen::MatrixXd Cinv_beta;
    bool cov_calculated = false;
    std::vector<RandomComponent> mme_components;
    std::vector<double> mme_lambdas;
    // RRM-specific
    Eigen::MatrixXd rrm_additive_K;
    Eigen::MatrixXd rrm_pe_K;
    bool use_rrm_lambda_matrix = false;
};

class GSSolve {
public:
    struct Context {
        const std::vector<GenRecord>& recs;
        const FixedDesignG& fd;
        const Config& cfg;
        const std::map<std::string, int>& idmap;
        AbstractMatrix* Qinv = nullptr;
        AbstractMatrix* GDinv = nullptr;
        AbstractMatrix* GEinv = nullptr;
        AbstractMatrix* PEinv = nullptr;
        const std::vector<std::unique_ptr<AbstractMatrix>>& generic_rand_invs;
        const std::vector<std::vector<int>>& generic_rand_maps;
        const std::vector<int>& mat_id_map;
        const std::vector<RandomComponent>& base_components;
        bool force_implicit = false;
        bool stcg_mode = false;
        int rrm_additive_component_count = 0;
        int rrm_pe_component_count = 0;
        const VCEResult& vce_result;
        double sigma2_u_init = 1.0;
        double sigma2_e_init = 1.0;
        bool run_vce = false;
        AbstractMatrix* stcg_ga_dim = nullptr;
    };

    static SolveResult solve(const Context& ctx);

    /// Multi-trait GBLUP MME solve.
    /// Uses Vg (t×t), Ve (t×t), Y (n×t), X (n×p) from VCE result.
    /// Builds MME via Kronecker products and solves with dense LDLT.
    /// G_inv: G^{-1} matrix (n×n, dense). idmap: individual ID → 1-based index.
    static SolveResult solveMultiTrait(
        const VCEResult& vce,
        const Eigen::MatrixXd& G_inv,
        const std::map<std::string, int>& idmap,
        const Config& cfg);
};

class GSOutput {
public:
    struct Context {
        const std::vector<GenRecord>& recs;
        const FixedDesignG& fd;
        const Config& cfg;
        const std::map<std::string, int>& idmap;
        const SolveResult& solve_result;
        double sigma2_e = 1.0;
        double sigma2_u = 1.0;
        const VCEResult& vce_result;
        bool run_vce = false;
        const std::string& inv_label;
        bool stcg_mode = false;
        int rrm_additive_component_count = 0;
        int rrm_pe_component_count = 0;
        int rrm_order = 2;
        double rrm_tmin = 0.0;
        double rrm_tmax = 0.0;
        const std::vector<double>& rrm_curve_times;
        AbstractMatrix* Ginv = nullptr;
        AbstractMatrix* Qinv = nullptr;
        bool use_split = false;
        const std::vector<int>& genotyped_map;
    };

    static void writeResults(const Context& ctx);

    /// Multi-trait EBV/beta output.
    /// solve_result.beta: pt×1 vectorized (trait-major), solve_result.u: nt×1 vectorized.
    static void writeMultiTrait(
        const SolveResult& solve_result,
        const VCEResult& vce,
        const std::map<std::string, int>& idmap,
        const Config& cfg);
};

} // namespace cosmic
