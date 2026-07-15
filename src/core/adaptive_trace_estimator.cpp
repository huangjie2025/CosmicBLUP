#include "adaptive_trace_estimator.h"
#include "rng.h"
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

double AdaptiveTraceEstimator::estimate(int dim,
                                        Operator op,
                                        int min_samples,
                                        int max_samples,
                                        double tol,
                                        int seed) {
    if (dim <= 0) return 0.0;
    if (min_samples <= 0) min_samples = 10;
    if (max_samples < min_samples) max_samples = min_samples;

    double trace_sum = 0.0;
    double trace_sq_sum = 0.0;
    int samples_done = 0;

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    // We process in batches to check convergence
    int batch_size = num_threads * 4;
    if (batch_size < 16) batch_size = 16;

    while (samples_done < max_samples) {
        int current_batch = std::min(batch_size, max_samples - samples_done);
        double batch_sum = 0.0;
        double batch_sq_sum = 0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:batch_sum, batch_sq_sum)
#endif
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            RNG local_rng(seed + samples_done + tid * 12345);

#ifdef _OPENMP
#pragma omp for
#endif
            for (int i = 0; i < current_batch; ++i) {
                Eigen::VectorXd z(dim);
                local_rng.fill_rademacher(z);

                Eigen::VectorXd Az = op(z);
                double val = z.dot(Az);

                batch_sum += val;
                batch_sq_sum += val * val;
            }
        }

        trace_sum += batch_sum;
        trace_sq_sum += batch_sq_sum;
        samples_done += current_batch;

        if (samples_done >= min_samples) {
            double mean = trace_sum / samples_done;
            double var = (trace_sq_sum - (trace_sum * trace_sum) / samples_done) / (samples_done - 1.0);
            double se = std::sqrt(std::max(0.0, var) / samples_done);

            if (std::abs(mean) > 1e-12 && (se / std::abs(mean)) < tol) {
                // Converged!
                break;
            }
        }
    }

    return trace_sum / samples_done;
}

} // namespace cosmic
