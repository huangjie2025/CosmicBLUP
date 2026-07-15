#include "random_regression.h"
#include "logger.h"
#include "design.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace cosmic {

RRMContext GSRrm::setup(const SetupInput& in) {
    const auto& recs = in.recs;
    const auto& cfg = in.cfg;
    RRMContext ctx;
    ctx.order = cfg.cli_present.count("--rrm-order") ? cfg.rrm_order : 2;

    if (!in.Qinv) {
        throw std::runtime_error("RRM GS currently requires a genetic relationship matrix (A/G/H inverse) to be available.");
    }

    LOG_INFO("Setting up Random Regression Model (RRM) components...");

    // Find time covariate index
    int time_qcovar_idx = -1;
    if (cfg.time_col_name.empty() && !cfg.qcovar_names.empty()) {
        time_qcovar_idx = 0;
        LOG_WARN("No --time-col specified. Using the first quantitative covariate '" << cfg.qcovar_names[0] << "' as time.");
    } else {
        for (size_t i = 0; i < cfg.qcovar_names.size(); ++i) {
            if (cfg.qcovar_names[i] == cfg.time_col_name) {
                time_qcovar_idx = i;
                break;
            }
        }
    }
    if (time_qcovar_idx < 0) {
        throw std::runtime_error("RRM requires a valid time covariate (--time-col or first --qcovar).");
    }

    // Extract time vector and range
    int valid_obs = recs.size();
    Eigen::VectorXd time_vec(valid_obs);
    double tmin = std::numeric_limits<double>::max();
    double tmax = std::numeric_limits<double>::lowest();
    for (int i = 0; i < valid_obs; ++i) {
        double t = recs[i].nums[time_qcovar_idx];
        time_vec(i) = t;
        if (t < tmin) tmin = t;
        if (t > tmax) tmax = t;
    }
    ctx.tmin = tmin;
    ctx.tmax = tmax;

    // Build Legendre basis matrix
    Eigen::MatrixXd Phi = buildLegendreMatrix(time_vec, tmin, tmax, ctx.order);
    LOG_INFO("RRM time range normalized from [" << tmin << ", " << tmax << "] with Legendre order " << ctx.order << ".");

    // Determine curve output time points
    std::set<double> unique_times;
    for (int i = 0; i < valid_obs; ++i) unique_times.insert(time_vec(i));
    if (unique_times.size() <= 50) {
        ctx.curve_times.assign(unique_times.begin(), unique_times.end());
        LOG_INFO("RRM curve output will use " << ctx.curve_times.size() << " observed time points.");
    } else {
        const int default_grid = 21;
        ctx.curve_times.resize(default_grid);
        if (std::abs(tmax - tmin) < 1e-12) {
            std::fill(ctx.curve_times.begin(), ctx.curve_times.end(), tmin);
        } else {
            for (int g = 0; g < default_grid; ++g) {
                double frac = static_cast<double>(g) / static_cast<double>(default_grid - 1);
                ctx.curve_times[g] = tmin + frac * (tmax - tmin);
            }
        }
        LOG_INFO("RRM curve output will use a regular grid of " << ctx.curve_times.size() << " time points.");
    }

    // Additive genetic components (one per Legendre coefficient)
    for (int k = 0; k <= ctx.order; ++k) {
        RandomComponent rc;
        rc.Qinv = in.Qinv;
        rc.covar_map.assign(Phi.col(k).data(), Phi.col(k).data() + valid_obs);
        ctx.base_components.push_back(rc);
    }
    ctx.additive_component_count = ctx.order + 1;

    // Permanent environmental components (if enabled)
    if (cfg.pe_effect) {
        if (!in.PEinv) {
            throw std::runtime_error("RRM Pe components require a configured permanent environmental matrix.");
        }
        for (int k = 0; k <= ctx.order; ++k) {
            RandomComponent rc;
            rc.Qinv = in.PEinv;
            rc.covar_map.assign(Phi.col(k).data(), Phi.col(k).data() + valid_obs);
            ctx.base_components.push_back(rc);
        }
        ctx.pe_component_count = ctx.order + 1;
    }

    LOG_INFO("Created " << ctx.base_components.size() << " random components for RRM ("
             << ctx.additive_component_count << " additive"
             << (ctx.pe_component_count > 0 ? ", " + std::to_string(ctx.pe_component_count) + " Pe" : "")
             << ").");

    return ctx;
}

} // namespace cosmic
