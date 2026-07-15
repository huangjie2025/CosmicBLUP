#pragma once
#include "config.h"
#include "mme_builder.h"
#include "matrix_adapter.h"
#include <vector>
#include <string>
#include <memory>

namespace cosmic {

struct RRMContext {
    std::vector<RandomComponent> base_components;
    int additive_component_count = 0;
    int pe_component_count = 0;
    int order = 2;
    double tmin = 0.0;
    double tmax = 0.0;
    std::vector<double> curve_times;
};

class GSRrm {
public:
    struct SetupInput {
        const std::vector<GenRecord>& recs;
        const Config& cfg;
        AbstractMatrix* Qinv = nullptr;
        AbstractMatrix* PEinv = nullptr;
    };

    /// Build RRM random components (Legendre basis), time range, curve grid.
    /// Returns fully populated RRMContext. Throws on invalid config.
    static RRMContext setup(const SetupInput& in);
};

} // namespace cosmic
