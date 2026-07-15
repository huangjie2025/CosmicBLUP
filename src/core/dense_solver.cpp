#include "dense_solver.h"
#include "slq_estimator.h"
#include "rng.h"
#include "pcg_solver.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace cosmic {

DenseSolver::DenseSolver(DenseSolverTier tier) : tier_(tier) {}

bool DenseSolver::compute(const Eigen::MatrixXd& mat) {
    A_ = mat;
    if (tier_ == DenseSolverTier::DirectLDLT) {
        ldlt_.compute(A_);
        return ldlt_.info() == Eigen::Success;
    }
    else if (tier_ == DenseSolverTier::CholeskyLLT) {
        llt_.compute(A_);
        return llt_.info() == Eigen::Success;
    }
    else if (tier_ == DenseSolverTier::EigenDecomp) {
        eigen_.compute(A_);
        return eigen_.info() == Eigen::Success;
    }
    else if (tier_ == DenseSolverTier::LowRankSVD) {
        // Implement Randomized Eigen Decomposition for Symmetric Matrix
        int n = A_.rows();
        int k = std::min(n, 100); // Default low rank size
        int q = 2; // Power iterations

        RNG rng(42);
        Eigen::MatrixXd Omega(n, k);
        rng.fill_normal(Omega);

        Eigen::MatrixXd Y = A_ * Omega;
        for (int i = 0; i < q; ++i) {
            Y = A_ * (A_ * Y); // A is symmetric
        }

        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, k);

        Eigen::MatrixXd B = Q.transpose() * A_ * Q;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
        if (es.info() != Eigen::Success) return false;

        S_svd_ = es.eigenvalues();
        U_svd_ = Q * es.eigenvectors();

        return true;
    }
    else if (tier_ == DenseSolverTier::PCG_SLQ) {
        // Preconditioner could be diagonal
        return true; // Matrix-free just stores A_
    }

    return false;
}

Eigen::VectorXd DenseSolver::solve(const Eigen::VectorXd& b) const {
    if (tier_ == DenseSolverTier::DirectLDLT) {
        return ldlt_.solve(b);
    }
    else if (tier_ == DenseSolverTier::CholeskyLLT) {
        return llt_.solve(b);
    }
    else if (tier_ == DenseSolverTier::EigenDecomp) {
        Eigen::VectorXd s = eigen_.eigenvalues();
        double eps = std::max(1e-12 * s.maxCoeff(), 1e-8);
        Eigen::VectorXd sinv = s.unaryExpr([eps](double val) { return val > eps ? 1.0 / val : 0.0; });
        return eigen_.eigenvectors() * sinv.asDiagonal() * (eigen_.eigenvectors().transpose() * b);
    }
    else if (tier_ == DenseSolverTier::LowRankSVD) {
        // Woodbury or direct inversion of low rank approx
        // A ~ U S U^T + lambda I
        // A^{-1} b = 1/lambda * b - 1/lambda * U * (S^{-1} lambda + I)^{-1} * U^T * b
        Eigen::VectorXd Ub = U_svd_.transpose() * b;
        Eigen::VectorXd middle = Ub.cwiseQuotient(Eigen::VectorXd::Constant(S_svd_.size(), 1.0) + (ridge_lambda_ * S_svd_.cwiseInverse()));
        return (b - U_svd_ * middle) / ridge_lambda_;
    }
    else if (tier_ == DenseSolverTier::PCG_SLQ) {
        Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;
        cg.compute(A_);
        cg.setMaxIterations(1000);
        cg.setTolerance(1e-6);
        return cg.solve(b);
    }
    throw std::runtime_error("Unknown solver tier");
}

double DenseSolver::logDeterminant() const {
    if (tier_ == DenseSolverTier::DirectLDLT) {
        return ldlt_.vectorD().array().log().sum();
    }
    else if (tier_ == DenseSolverTier::CholeskyLLT) {
        return 2.0 * llt_.matrixLLT().diagonal().array().log().sum();
    }
    else if (tier_ == DenseSolverTier::EigenDecomp) {
        double log_det = 0;
        double eps = std::max(1e-12 * eigen_.eigenvalues().maxCoeff(), 1e-8);
        for (int i = 0; i < eigen_.eigenvalues().size(); ++i) {
            double v = eigen_.eigenvalues()(i);
            if (v > eps) log_det += std::log(v);
        }
        return log_det;
    }
    else if (tier_ == DenseSolverTier::LowRankSVD) {
        // log det(U S U^T + lambda I) = sum log(s_i + lambda) + (n-k) log(lambda)
        int n = A_.rows();
        int k = S_svd_.size();
        double log_det = (n - k) * std::log(ridge_lambda_);
        for (int i = 0; i < k; ++i) {
            log_det += std::log(std::max(S_svd_(i) + ridge_lambda_, 1e-8));
        }
        return log_det;
    }
    else if (tier_ == DenseSolverTier::PCG_SLQ) {
        // Implement Stochastic Lanczos Quadrature (SLQ) for log|A|
        int n = A_.rows();
        int num_samples = 30;
        int m = 15;
        auto op_A = [&](const Eigen::VectorXd& v, Eigen::VectorXd& out) {
            out = A_ * v;
        };
        auto f_log = [](double x) { return std::log(std::max(x, 1e-12)); };

        return SLQEstimator::estimate(n, op_A, f_log, num_samples, m, 42);
    }
    return 0;
}

double DenseSolver::inverseTrace() const {
    // Exact or approximate tr(A^{-1})
    if (tier_ == DenseSolverTier::DirectLDLT || tier_ == DenseSolverTier::CholeskyLLT || tier_ == DenseSolverTier::EigenDecomp) {
        // Slow exact trace if we really need it
        double tr = 0;
        int n = A_.rows();
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd ei = Eigen::VectorXd::Zero(n);
            ei(i) = 1.0;
            tr += solve(ei)(i);
        }
        return tr;
    }
    else if (tier_ == DenseSolverTier::LowRankSVD) {
        // tr(A^{-1}) = tr(1/lambda * I - 1/lambda * U * (S^{-1} lambda + I)^{-1} * U^T)
        // = n/lambda - 1/lambda * tr((S^{-1} lambda + I)^{-1} * U^T U)
        // Since U^T U = I_k, it is n/lambda - 1/lambda * sum_i (lambda/s_i + 1)^{-1}
        int n = A_.rows();
        int k = S_svd_.size();
        double tr = (double)n / ridge_lambda_;
        for (int i = 0; i < k; ++i) {
            tr -= 1.0 / (ridge_lambda_ + ridge_lambda_ * ridge_lambda_ / S_svd_(i));
        }
        return tr;
    }
    else if (tier_ == DenseSolverTier::PCG_SLQ) {
        // Hutchinson for tr(A^{-1})
        int n = A_.rows();
        int num_samples = 30;
        double trace_inv = 0.0;
        RNG rng(42);

        for (int i = 0; i < num_samples; ++i) {
            Eigen::VectorXd z(n);
            rng.fill_rademacher(z);
            Eigen::VectorXd x = solve(z);
            trace_inv += z.dot(x);
        }
        return trace_inv / num_samples;
    }
    return 0;
}

} // namespace cosmic
