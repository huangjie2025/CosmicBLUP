#pragma once
#include <functional>
#include <Eigen/Dense>

namespace cosmic {

class SLQEstimator {
public:
    // Estimate tr(f(A)) using Stochastic Lanczos Quadrature (SLQ)
    // op_A: function that computes y = A * x
    // n: dimension of A
    // f: scalar function to apply to eigenvalues (e.g., std::log)
    // num_samples: number of random Rademacher vectors
    // m: number of Lanczos steps (Krylov subspace dimension)
    // seed: random seed
    static double estimate(int n,
                           std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> op_A,
                           std::function<double(double)> f,
                           int num_samples = 30,
                           int m = 15,
                           unsigned long seed = 42);
};

} // namespace cosmic
