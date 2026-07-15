#include "slq_estimator.h"
#include "logger.h"
#include <random>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

double SLQEstimator::estimate(int n,
                              std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> op_A,
                              std::function<double(double)> f,
                              int num_samples,
                              int m,
                              unsigned long seed) {
    if (n <= 0 || m <= 0 || num_samples <= 0) return 0.0;

    // For small matrices, limit Lanczos steps to matrix dimension
    m = std::min(m, n);

    double total_estimate = 0.0;

    #pragma omp parallel reduction(+:total_estimate)
    {
        // Thread-local RNG
        unsigned long thread_seed = seed;
        #ifdef _OPENMP
        thread_seed += omp_get_thread_num() * 19937;
        #endif
        std::mt19937_64 rng(thread_seed);
        std::uniform_int_distribution<int> bit(0, 1);

        #pragma omp for
        for (int s = 0; s < num_samples; ++s) {
            // 1. Generate Rademacher vector z
            Eigen::VectorXd z(n);
            for (int i = 0; i < n; ++i) {
                z(i) = bit(rng) ? 1.0 : -1.0;
            }

            // 2. Lanczos algorithm to build tridiagonal matrix T
            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, m + 1);
            Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
            Eigen::VectorXd beta = Eigen::VectorXd::Zero(m);

            // V_0 = z / ||z||. Since z is Rademacher, ||z|| = sqrt(n)
            double norm_z = std::sqrt((double)n);
            V.col(0) = z / norm_z;

            int actual_m = m;
            for (int j = 0; j < m; ++j) {
                Eigen::VectorXd w(n);
                op_A(V.col(j), w);

                if (j > 0) {
                    w -= beta(j-1) * V.col(j-1);
                }

                alpha(j) = w.dot(V.col(j));
                w -= alpha(j) * V.col(j);

                // Full reorthogonalization for numerical stability
                for (int k = 0; k <= j; ++k) {
                    double proj = w.dot(V.col(k));
                    w -= proj * V.col(k);
                }

                beta(j) = w.norm();
                if (beta(j) < 1e-12) {
                    actual_m = j + 1;
                    break;
                }

                V.col(j+1) = w / beta(j);
            }

            // 3. Construct T and find eigenvalues/eigenvectors
            Eigen::MatrixXd T = Eigen::MatrixXd::Zero(actual_m, actual_m);
            for (int j = 0; j < actual_m; ++j) {
                T(j, j) = alpha(j);
                if (j < actual_m - 1) {
                    T(j+1, j) = beta(j);
                    T(j, j+1) = beta(j);
                }
            }

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
            const Eigen::VectorXd& evals = es.eigenvalues();
            const Eigen::MatrixXd& evecs = es.eigenvectors();

            // 4. Compute estimate for this sample
            // tr(f(A)) ≈ ||z||^2 * sum_{i} [ (evecs(0, i))^2 * f(evals(i)) ]
            // Since ||z||^2 = n
            double sample_sum = 0.0;
            for (int i = 0; i < actual_m; ++i) {
                double weight = evecs(0, i) * evecs(0, i);
                sample_sum += weight * f(evals(i));
            }

            total_estimate += n * sample_sum;
        }
    }

    return total_estimate / num_samples;
}

} // namespace cosmic
