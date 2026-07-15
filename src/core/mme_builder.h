#pragma once
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include "matrix_adapter.h"
#include "design.h"

namespace cosmic {

struct RandomComponent {
    const AbstractMatrix* Qinv;
    std::vector<int> id_map; // If empty, use recs[i].aid - 1. If size > 0, use id_map[i].

    // Covariate values for the random effect (e.g., for Random Regression Model)
    // If empty, assumes coefficient of 1.0 for all mapped individuals.
    // If size > 0, must be same size as the number of records.
    std::vector<double> covar_map;

    // For Genomic Data (Matrix-Free RHE)
    GenotypeMatrix* geno_mat = nullptr;
};

// A unified builder for MME LHS (Explicit Sparse)
// Manages: [X'X  X'Z;  Z'X  Z'Z + lambda * Qinv]
// Supports efficient updates of lambda without rebuilding the pattern.
class MMELHSBuilder {
public:
    // New Generic Constructor
    MMELHSBuilder(const std::vector<GenRecord>& recs,
                  const FixedDesignG& fd,
                  const std::vector<RandomComponent>& components,
                  bool build_matrix = true);

    // Legacy Constructor
    MMELHSBuilder(const std::vector<GenRecord>& recs,
                  const FixedDesignG& fd,
                  const AbstractMatrix* Qinv,
                  bool build_matrix = true);

    // Update the LHS matrix with new lambdas (sigma2_e / sigma2_u_k)
    // Returns reference to the updated internal matrix
    const Eigen::SparseMatrix<double>& build_lhs(const std::vector<double>& lambdas);

    // Advanced: Update LHS with a full Covariance Structure matrix (Lambda = sigma2_e * G0_inv)
    // Supports cross-component covariances (e.g. Maternal-Direct or RRM cov)
    const Eigen::SparseMatrix<double>& build_lhs(const Eigen::MatrixXd& Lambda);

    // Legacy support (single random effect)
    const Eigen::SparseMatrix<double>& build_lhs(double lambda);

    // Get the current LHS (must call build_lhs at least once)
    const Eigen::SparseMatrix<double>& get_lhs() const { return lhs; }

    // Build RHS from data vector y: [X'y; Z'y]
    Eigen::VectorXd build_rhs(const std::vector<GenRecord>& recs, const FixedDesignG& fd, int total_dim) const;

    // Generic transpose multiply: [X'v; Z'v] for any vector v of size n_records
    Eigen::VectorXd mult_transpose_design(const Eigen::VectorXd& v, const std::vector<GenRecord>& recs, const FixedDesignG& fd) const;

    // Generic multiply: X*b + Z*u
    // u contains concatenated random effects [u1; u2; ...]
    Eigen::VectorXd mult_design(const Eigen::VectorXd& b, const Eigen::VectorXd& u, const std::vector<GenRecord>& recs, const FixedDesignG& fd) const;

    // Get dimensions
    int get_p() const { return p; }
    int get_q_total() const { return q_total; }
    int get_dim() const { return p + q_total; }
    const std::vector<int>& get_qs() const { return qs; }

private:
    std::vector<RandomComponent> components;
    std::vector<int> qs; // Size of each random effect
    std::vector<int> q_offsets; // Start index of each random effect in u

    // Internal LHS storage
    Eigen::SparseMatrix<double> lhs;

    // Optimization for fast lambda updates
    std::vector<double> lhs_base_values; // Values of LHS corresponding to data part (X'X, X'Z, Z'Z)

    struct UpdateMap {
        int lhs_idx;
        double qinv_val;
        int comp_i;
        int comp_j;
    };
    std::vector<UpdateMap> lhs_update_map;
    bool fast_update_ready = false;

    int p = 0;
    int q_total = 0;

    void initialize(const std::vector<GenRecord>& recs, const FixedDesignG& fd);
};

}
