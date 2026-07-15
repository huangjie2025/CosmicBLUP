#include "mme_builder.h"
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

using namespace Eigen;
using namespace std;

// Legacy Constructor
MMELHSBuilder::MMELHSBuilder(const std::vector<GenRecord>& recs,
                             const FixedDesignG& fd,
                             const AbstractMatrix* Qinv,
                             bool build_matrix)
{
    // Convert single Qinv to vector<RandomComponent>
    RandomComponent rc;
    rc.Qinv = Qinv;
    // id_map empty -> use aid
    components.push_back(rc);

    p = fd.p;

    // Always calculate dimensions
    qs.clear();
    q_offsets.clear();
    q_total = 0;

    for (const auto& c : components) {
        int dim = c.Qinv->rows();
        qs.push_back(dim);
        q_offsets.push_back(q_total);
        q_total += dim;
    }

    if (build_matrix) {
        initialize(recs, fd);
    }
}

// New Constructor
MMELHSBuilder::MMELHSBuilder(const std::vector<GenRecord>& recs,
                             const FixedDesignG& fd,
                             const std::vector<RandomComponent>& comps,
                             bool build_matrix)
    : components(comps)
{
    p = fd.p;

    // Always calculate dimensions
    qs.clear();
    q_offsets.clear();
    q_total = 0;

    for (const auto& c : components) {
        int dim = c.Qinv->rows();
        qs.push_back(dim);
        q_offsets.push_back(q_total);
        q_total += dim;
    }

    if (build_matrix) {
        initialize(recs, fd);
    }
}

void MMELHSBuilder::initialize(const std::vector<GenRecord>& recs, const FixedDesignG& fd) {
    int n_records = (int)recs.size();

    // 1. Calculate dimensions (already done in constructor, but doing again is fine)
    qs.clear();
    q_offsets.clear();
    q_total = 0;

    for (const auto& c : components) {
        int dim = c.Qinv->rows();
        qs.push_back(dim);
        q_offsets.push_back(q_total);
        q_total += dim;
    }

    int total_dim = p + q_total;

    // 2. Estimate Triplets
    size_t est_triplets = n_records * 20; // Fixed part estimate
    for (const auto& c : components) {
        est_triplets += n_records * 2; // Z'Z approx (diagonal)
        est_triplets += c.Qinv->nonZeros();
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    vector<vector<Triplet<double>>> thread_triplets(num_threads);
    for(int t=0; t<num_threads; ++t) {
        thread_triplets[t].reserve(est_triplets / num_threads);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& trips = thread_triplets[tid];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n_records; ++i) {
            const auto& nz = fd.rows[i];

            // Fixed Effects (X)
            // X'X
            for (size_t a = 0; a < nz.size(); ++a) {
                int ci = nz[a].first; double vi = nz[a].second;
                for (size_t b = a; b < nz.size(); ++b) {
                    int cj = nz[b].first; double vj = nz[b].second;
                    trips.emplace_back(ci, cj, vi * vj);
                    if (ci != cj) trips.emplace_back(cj, ci, vi * vj);
                }
            }

            // Random Effects (Z)
            for (size_t k = 0; k < components.size(); ++k) {
                int id_k = -1;
                if (components[k].id_map.empty()) {
                    if (recs[i].aid > 0) id_k = recs[i].aid - 1;
                } else {
                    if (i < (int)components[k].id_map.size()) id_k = components[k].id_map[i];
                }

                if (id_k >= 0 && id_k < qs[k]) {
                    int col_k = p + q_offsets[k] + id_k;
                    double covar_k = components[k].covar_map.empty() ? 1.0 : components[k].covar_map[i];

                    // X'Z_k
                    for (size_t a = 0; a < nz.size(); ++a) {
                        int ci = nz[a].first; double vi = nz[a].second;
                        trips.emplace_back(ci, col_k, vi * covar_k); // X'Z
                        trips.emplace_back(col_k, ci, vi * covar_k); // Z'X
                    }

                    // Z_k'Z_k (Diagonal block contribution from data)
                    trips.emplace_back(col_k, col_k, covar_k * covar_k);

                    // Z_k'Z_j (Cross Random Blocks)
                    // Loop over other components j > k
                    for (size_t j = k + 1; j < components.size(); ++j) {
                        int id_j = -1;
                        if (components[j].id_map.empty()) {
                             if (recs[i].aid > 0) id_j = recs[i].aid - 1;
                        } else {
                             if (i < (int)components[j].id_map.size()) id_j = components[j].id_map[i];
                        }

                        if (id_j >= 0 && id_j < qs[j]) {
                            double covar_j = components[j].covar_map.empty() ? 1.0 : components[j].covar_map[i];
                            int col_j = p + q_offsets[j] + id_j;
                            trips.emplace_back(col_k, col_j, covar_k * covar_j);
                            trips.emplace_back(col_j, col_k, covar_k * covar_j);
                        }
                    }
                }
            }
        }
    }

    // Merge triplets
    vector<Triplet<double>> all_trips;
    size_t total_data_trips = 0;
    for(int t=0; t<num_threads; ++t) total_data_trips += thread_triplets[t].size();

    size_t total_est = total_data_trips;
    for (const auto& c : components) total_est += c.Qinv->nonZeros();

    all_trips.reserve(total_est);
    for(int t=0; t<num_threads; ++t) {
        all_trips.insert(all_trips.end(), thread_triplets[t].begin(), thread_triplets[t].end());
    }

    // Add Qinv structure (with epsilon) for diagonal AND off-diagonal correlated blocks
    for (size_t k = 0; k < components.size(); ++k) {
        for (size_t j = k; j < components.size(); ++j) {
            // Only add structure if they share the SAME Qinv matrix (e.g. Additive & Maternal)
            if (components[k].Qinv == components[j].Qinv) {
                int offset_k = p + q_offsets[k];
                int offset_j = p + q_offsets[j];
                components[k].Qinv->visit_triplets([&](int r, int c, double v) {
                    all_trips.emplace_back(offset_k + r, offset_j + c, 1e-50);
                    if (k != j) {
                        all_trips.emplace_back(offset_j + r, offset_k + c, 1e-50); // Symmetric block
                    }
                });
            }
        }
    }

    lhs.resize(total_dim, total_dim);
    lhs.setFromTriplets(all_trips.begin(), all_trips.end());
    lhs.makeCompressed();

    // Save base values
    lhs_base_values.assign(lhs.valuePtr(), lhs.valuePtr() + lhs.nonZeros());

    // Build update map
    lhs_update_map.clear();

    for (size_t k = 0; k < components.size(); ++k) {
        for (size_t j = k; j < components.size(); ++j) {
            if (components[k].Qinv == components[j].Qinv) {
                int offset_k = p + q_offsets[k];
                int offset_j = p + q_offsets[j];

                components[k].Qinv->visit_triplets([&](int r, int c, double v) {
                    // Update for block (k, j)
                    int lhs_row = offset_k + r;
                    int lhs_col = offset_j + c;

                    int start = lhs.outerIndexPtr()[lhs_col];
                    int end = lhs.outerIndexPtr()[lhs_col+1];
                    for (int idx = start; idx < end; ++idx) {
                        if (lhs.innerIndexPtr()[idx] == lhs_row) {
                            lhs_update_map.push_back({idx, v, (int)k, (int)j});
                            break;
                        }
                    }

                    // Update for symmetric block (j, k) if k != j
                    if (k != j) {
                        lhs_row = offset_j + r;
                        lhs_col = offset_k + c;
                        start = lhs.outerIndexPtr()[lhs_col];
                        end = lhs.outerIndexPtr()[lhs_col+1];
                        for (int idx = start; idx < end; ++idx) {
                            if (lhs.innerIndexPtr()[idx] == lhs_row) {
                                lhs_update_map.push_back({idx, v, (int)j, (int)k});
                                break;
                            }
                        }
                    }
                });
            }
        }
    }

    fast_update_ready = true;
}

const Eigen::SparseMatrix<double>& MMELHSBuilder::build_lhs(const std::vector<double>& lambdas) {
    if (!fast_update_ready) return lhs;
    if (lambdas.size() != components.size()) {
        // Fallback for legacy calls if size 1 matches
        if (components.size() == 1 && lambdas.size() == 1) {
             // OK
        } else {
             throw std::runtime_error("Lambda size mismatch");
        }
    }

    // Restore base
    if (lhs.nonZeros() == lhs_base_values.size()) {
        std::copy(lhs_base_values.begin(), lhs_base_values.end(), lhs.valuePtr());
    }

    double* values = lhs.valuePtr();
    for (const auto& u : lhs_update_map) {
        if (u.comp_i == u.comp_j) {
            values[u.lhs_idx] += lambdas[u.comp_i] * u.qinv_val;
        }
    }

    // Debug check
    bool has_nan = false;
    for(int i=0; i<lhs.nonZeros(); ++i) {
        if (!std::isfinite(values[i])) {
            has_nan = true;
            break;
        }
    }
    if (has_nan) {
        std::cerr << "CRITICAL ERROR: LHS matrix contains NaNs! lambdas[0]=" << lambdas[0] << "\n";
    }

    return lhs;
}

const Eigen::SparseMatrix<double>& MMELHSBuilder::build_lhs(const Eigen::MatrixXd& Lambda) {
    if (!fast_update_ready) return lhs;
    if (Lambda.rows() != components.size() || Lambda.cols() != components.size()) {
        throw std::runtime_error("Lambda matrix size mismatch");
    }

    // Restore base
    if (lhs.nonZeros() == lhs_base_values.size()) {
        std::copy(lhs_base_values.begin(), lhs_base_values.end(), lhs.valuePtr());
    }

    double* values = lhs.valuePtr();
    for (const auto& u : lhs_update_map) {
        values[u.lhs_idx] += Lambda(u.comp_i, u.comp_j) * u.qinv_val;
    }

    return lhs;
}

const Eigen::SparseMatrix<double>& MMELHSBuilder::build_lhs(double lambda) {
    if (components.empty()) return lhs;
    return build_lhs(std::vector<double>{lambda});
}

Eigen::VectorXd MMELHSBuilder::build_rhs(const std::vector<GenRecord>& recs, const FixedDesignG& fd, int total_dim) const {
    int n = (int)recs.size();
    Eigen::VectorXd y(n);
    #pragma omp parallel for schedule(static)
    for(int i=0; i<n; ++i) y(i) = recs[i].y;

    return mult_transpose_design(y, recs, fd);
}

Eigen::VectorXd MMELHSBuilder::mult_transpose_design(const Eigen::VectorXd& v, const std::vector<GenRecord>& recs, const FixedDesignG& fd) const {
    int n_records = (int)recs.size();
    if (v.size() != n_records) throw std::runtime_error("Vector dimension mismatch in mult_transpose_design");

    int total_dim = p + q_total;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total_dim);

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    vector<VectorXd> thread_rhs(num_threads, VectorXd::Zero(total_dim));

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& lrhs = thread_rhs[tid];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < n_records; ++i) {
             const auto& nz = fd.rows[i];
             double val = v(i);
             for (size_t a = 0; a < nz.size(); ++a) {
                int ci = nz[a].first; double vi = nz[a].second;
                lrhs(ci) += vi * val;
             }

             // Random Effects
             for (size_t k = 0; k < components.size(); ++k) {
                int id_k = -1;
                if (components[k].id_map.empty()) {
                    if (recs[i].aid > 0) id_k = recs[i].aid - 1;
                } else {
                    if (i < (int)components[k].id_map.size()) id_k = components[k].id_map[i];
                }

                if (id_k >= 0 && id_k < qs[k]) {
                    double covar_k = components[k].covar_map.empty() ? 1.0 : components[k].covar_map[i];
                    int idx = p + q_offsets[k] + id_k;
                    lrhs(idx) += val * covar_k;
                }
             }
        }
    }

    for(int t=0; t<num_threads; ++t) {
        rhs += thread_rhs[t];
    }
    return rhs;
}

Eigen::VectorXd MMELHSBuilder::mult_design(const Eigen::VectorXd& b, const Eigen::VectorXd& u, const std::vector<GenRecord>& recs, const FixedDesignG& fd) const {
    int n_records = (int)recs.size();
    Eigen::VectorXd res(n_records);

    if (b.size() != p || u.size() != q_total) throw std::runtime_error("Vector dimension mismatch in mult_design");

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n_records; ++i) {
        double val = 0.0;
        const auto& nz = fd.rows[i];
        for (const auto& pr : nz) {
            if (pr.first < p) val += pr.second * b(pr.first);
        }

        // Random Effects
        for (size_t k = 0; k < components.size(); ++k) {
            int id_k = -1;
            if (components[k].id_map.empty()) {
                if (recs[i].aid > 0) id_k = recs[i].aid - 1;
            } else {
                if (i < (int)components[k].id_map.size()) id_k = components[k].id_map[i];
            }

            if (id_k >= 0 && id_k < qs[k]) {
                double covar_k = components[k].covar_map.empty() ? 1.0 : components[k].covar_map[i];
                int u_idx = q_offsets[k] + id_k;
                val += u(u_idx) * covar_k;
            }
        }
        res(i) = val;
    }
    return res;
}

}
