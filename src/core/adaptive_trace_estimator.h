#pragma once
#include <Eigen/Core>
#include <functional>
#include <cmath>
#include <iostream>
#include "logger.h"

namespace cosmic {

class AdaptiveTraceEstimator {
public:
    using Operator = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

    // Estimate trace of an implicit matrix using Hutchinson's method with adaptive stopping
    static double estimate(int dim,
                           Operator op,
                           int min_samples = 30,
                           int max_samples = 300,
                           double tol = 0.05,
                           int seed = 42);
};

} // namespace cosmic
