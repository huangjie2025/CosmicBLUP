#pragma once
#include "vce.h"
#include <Eigen/Dense>
#include <vector>

namespace cosmic {

// V-Matrix AI-REML Implementation
class VMatrixAI_REML : public VCESolver {
private:
    int n, p;
    Eigen::MatrixXd X_dense;
    std::vector<Eigen::MatrixXd> GZt; // Stores G_k * Z_k^T (dim: q_k x n) for BLUP
    std::vector<Eigen::MatrixXd> ZGZt; // Stores Z_k * G_k * Z_k^T (dim: n x n) for VCE

    bool initialized = false;

    // Helper for computing Log Likelihood and AI Terms
    bool run_aireml_step(int iter);

    // Save last components for BLUP
    Eigen::MatrixXd last_P;
    Eigen::VectorXd last_Py;
    Eigen::MatrixXd last_XtVinvX_inv;
    Eigen::MatrixXd last_Xt_Vinv;

    Eigen::MatrixXd last_AI;
    double last_logL = -1e20;

public:
    using VCESolver::VCESolver;
    void solve() override;
    void calculate_SE() override;

private:
    void initialize();
    void compute_blup();
};

} // namespace cosmic