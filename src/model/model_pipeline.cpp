#include "model_pipeline.h"
#include "logger.h"
#include "matrix_adapter.h"

namespace cosmic {

void GSModel::run(const SharedData& data, double& sigma2_u, double& sigma2_e) {
    // === Phase 1: VCE (optional) ===
    VCEResult vce_result;
    if (data.run_vce) {
        GSVce::Context vce_ctx{
            data.recs, data.fd, data.idmap, data.cfg,
            data.Qinv, data.GDinv, data.GEinv, data.PEinv,
            data.stcg_ga_dim, data.stcg_genotype,
            data.generic_rand_invs, data.generic_rand_maps,
            data.mat_id_map, data.base_components,
            data.force_implicit, data.stcg_mode,
            data.rrm_additive_component_count, data.rrm_pe_component_count
        };
        vce_result = GSVce::run(vce_ctx, sigma2_u, sigma2_e);
        if (vce_result.multi_trait_exit) return;

        sigma2_e = vce_result.sigma2_e;
        if (!vce_result.vars_u.empty()) sigma2_u = vce_result.vars_u[0];
    }

    // === Multi-trait path: MME solve + EBV output ===
    if (data.cfg.multi_trait) {
        // Build G^{-1} from Qinv (G^{-1} stored as Qinv in GBLUP)
        Eigen::MatrixXd G_inv;
        if (data.Qinv && dynamic_cast<DenseMatrixAdapter*>(data.Qinv)) {
            G_inv = data.Qinv->toDense();
        } else {
            LOG_ERROR("Multi-trait solve requires dense G^{-1} (use GBLUP without --mmap).");
            return;
        }
        SolveResult mv_result = GSSolve::solveMultiTrait(vce_result, G_inv, data.idmap, data.cfg);
        GSOutput::writeMultiTrait(mv_result, vce_result, data.idmap, data.cfg);
        LOG_INFO("Multi-trait breeding values saved to [" << data.cfg.out_prefix << ".rand].");
        LOG_INFO("Multi-trait fixed effects saved to [" << data.cfg.out_prefix << ".beta].\n");
        return;
    }

    // === Phase 2: MME Solve (single-trait) ===
    GSSolve::Context solve_ctx{
        data.recs, data.fd, data.cfg, data.idmap,
        data.Qinv, data.GDinv, data.GEinv, data.PEinv,
        data.generic_rand_invs, data.generic_rand_maps,
        data.mat_id_map, data.base_components,
        data.force_implicit, data.stcg_mode,
        data.rrm_additive_component_count, data.rrm_pe_component_count,
        vce_result, sigma2_u, sigma2_e, data.run_vce,
        data.stcg_ga_dim
    };
    SolveResult solve_result = GSSolve::solve(solve_ctx);

    // === Phase 3: Output ===
    GSOutput::Context out_ctx{
        data.recs, data.fd, data.cfg, data.idmap, solve_result,
        sigma2_e, sigma2_u, vce_result, data.run_vce, data.inv_label,
        data.stcg_mode,
        data.rrm_additive_component_count, data.rrm_pe_component_count,
        data.rrm_order, data.rrm_tmin, data.rrm_tmax, data.rrm_curve_times,
        data.Ginv, data.Qinv, data.use_split, data.genotyped_map
    };
    GSOutput::writeResults(out_ctx);

    LOG_INFO("Coefficients of all covariates and fixed effects are saved in file [" << data.cfg.out_prefix << ".beta].");
    LOG_INFO("Random effects of all individuals are saved in file [" << data.cfg.out_prefix << ".rand].\n");
}

} // namespace cosmic
