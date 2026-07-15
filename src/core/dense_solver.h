#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace cosmic {

enum class DenseSolverTier {
    DirectLDLT,
    CholeskyLLT,
    EigenDecomp,
    LowRankSVD,
    PCG_SLQ
};

class DenseSolver {
public:
    DenseSolver(DenseSolverTier tier = DenseSolverTier::DirectLDLT);

    // Set matrix and compute factorization/preconditioner
    bool compute(const Eigen::MatrixXd& mat);

    // Solve Ax = b
    Eigen::VectorXd solve(const Eigen::VectorXd& b) const;

    // Compute log determinant (exact or approximated via SLQ/SVD)
    double logDeterminant() const;

    // Compute inverse trace tr(A^{-1}) (exact or approximated)
    double inverseTrace() const;

private:
    DenseSolverTier tier_;
    Eigen::MatrixXd A_;

    // For DirectLDLT
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;

    // For CholeskyLLT
    Eigen::LLT<Eigen::MatrixXd> llt_;

    // For EigenDecomp
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_;

    // For LowRankSVD
    Eigen::MatrixXd U_svd_;
    Eigen::VectorXd S_svd_;
    double ridge_lambda_ = 1e-6; // Assumed ridge for non-singular approximation

    // For PCG_SLQ
    // (We will use MatrixFreePCG internally for solve)
};

} // namespace cosmic
