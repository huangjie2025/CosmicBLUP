#include "eigen_vce.h"
#include "vce.h" // For VCESolver::solveDiagonal
#include <iostream>
#include <stdexcept>

#include <unsupported/Eigen/KroneckerProduct>

namespace cosmic {

// --- EigenVce (Single Trait) ---

EigenVce::EigenVce(const Options& opts) : options(opts) {}

void EigenVce::prepare(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::MatrixXd& G) {
    n_samples = y.size();
    n_covars = X.cols();

    if (X.rows() != n_samples || G.rows() != n_samples || G.cols() != n_samples) {
        throw std::runtime_error("EigenVce::prepare: Dimension mismatch");
    }

    if (options.use_covariate_projection && n_covars > 0) {
        std::cout << "EigenVce: Applying Covariate Projection to y and G..." << std::endl;

        Eigen::MatrixXd X_d = X.cast<double>();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(X_d);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n_samples, n_samples);

        // A is the null space of X (n x (n-c))
        Eigen::MatrixXd A = Q.rightCols(n_samples - n_covars);

        // Project y and G
        Eigen::VectorXd y_reml = A.transpose() * y;
        Eigen::MatrixXd G_reml = A.transpose() * G * A;

        std::cout << "EigenVce: Eigen Decomposition of projected G (" << (n_samples - n_covars) << " x " << (n_samples - n_covars) << ")..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G_reml);
        D = es.eigenvalues();

        for(int i=0; i<D.size(); ++i) if(D[i] < 0) D[i] = 0;

        U = es.eigenvectors();
        Uty = U.transpose() * y_reml;
        W_proj = A * U;
        UtX = Eigen::MatrixXd::Zero(n_samples - n_covars, 0);
    } else {
        std::cout << "EigenVce: Eigen Decomposition of G (" << n_samples << " x " << n_samples << ")..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        D = es.eigenvalues();

        for(int i=0; i<D.size(); ++i) if(D[i] < 0) D[i] = 0;

        U = es.eigenvectors();
        Uty = U.transpose() * y;
        UtX = U.transpose() * X.cast<double>();
        W_proj = U; // Standard rotation
    }
}

void EigenVce::prepareSparse(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::SparseMatrix<double>& G_sparse) {
    std::cout << "EigenVce: Converting sparse GRM to dense for Eigen Decomposition..." << std::endl;
    Eigen::MatrixXd G_dense = Eigen::MatrixXd(G_sparse);
    prepare(y, X, G_dense);
}

void EigenVce::prepareRRM(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& Phi, const Eigen::MatrixXd& G) {
    std::cout << "EigenVce: Preparing RRM specific VCE logic..." << std::endl;
    // For RRM, the random effect covariance is V = Z * (K \otimes G) * Z' + e
    // If K is scalar (order 0), V = Z G Z' * sigma2_g + I * sigma2_e
    // Let's implement the simplified order 0 case first for exact decomposition

    Eigen::MatrixXd ZGZt = Z * G * Z.transpose();

    // Fall back to standard dense preparation
    prepare(y, X, ZGZt);
}

bool EigenVce::runNullModel() {
    std::cout << "EigenVce: Running Null Model (Brent Search)..." << std::endl;
    double var_g, var_e, se_g, se_e;
    Eigen::VectorXd beta, beta_se;

    // We reuse the existing VCESolver::solveDiagonal which implements the Brent search
    VCESolver::solveDiagonal(D, Uty, UtX, var_g, var_e, se_g, se_e, &beta, &beta_se);
    bool success = true;

    sigma2_g = var_g;
    sigma2_e = var_e;
    null_delta = sigma2_e / sigma2_g;

    return success;
}

void EigenVce::fixedComponents(double s2g, double s2e) {
    if (n_samples <= 0) throw std::runtime_error("EigenVce: prepare() must be called before fixedComponents()");
    sigma2_g = s2g;
    sigma2_e = s2e;
    null_delta = sigma2_e / (sigma2_g + 1e-10);
}

// --- MvEigenVce (Multiple Traits) ---

MvEigenVce::MvEigenVce() {}
MvEigenVce::MvEigenVce(const Options& opts) : options(opts) {}

void MvEigenVce::prepare(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X, const Eigen::MatrixXd& G) {
    n_samples = Y.rows();
    n_traits = Y.cols();
    n_covars = X.cols();

    if (X.rows() != n_samples || G.rows() != n_samples || G.cols() != n_samples) {
        throw std::runtime_error("MvEigenVce::prepare: Dimension mismatch");
    }

    if (options.use_projection && n_covars > 0) {
        if (options.verbose) std::cout << "MvEigenVce: Applying Covariate Projection to Y and G..." << std::endl;
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
        Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n_samples, n_samples);

        Eigen::MatrixXd A = Q.rightCols(n_samples - n_covars);

        Eigen::MatrixXd Y_reml = A.transpose() * Y;
        Eigen::MatrixXd G_reml = A.transpose() * G * A;

        if (options.verbose) std::cout << "MvEigenVce: Eigen Decomposition of projected G (" << (n_samples - n_covars) << " x " << (n_samples - n_covars) << ")..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G_reml);
        D = es.eigenvalues();
        for (int i=0; i<D.size(); ++i) if (D[i] < 0) D[i] = 0;

        U = es.eigenvectors();
        W_proj = A * U;

        Uty = U.transpose() * Y_reml;
        UtX = Eigen::MatrixXd::Zero(n_samples - n_covars, 0);

        n_samples = n_samples - n_covars;
        n_covars = 0;
        options.use_projection = false;
    } else {
        if (options.verbose) std::cout << "MvEigenVce: Eigen Decomposition of G (" << G.rows() << " x " << G.cols() << ")..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        D = es.eigenvalues();
        for (int i=0; i<D.size(); ++i) {
            if (D[i] < 0) D[i] = 0;
        }

        if (options.verbose) std::cout << "MvEigenVce: Rotating Y and X..." << std::endl;
        U = es.eigenvectors();
        Uty = U.transpose() * Y;
        UtX = U.transpose() * X;
        W_proj = U;
    }

    // Initialize Parameters
    Eigen::MatrixXd Y_centered = Y.rowwise() - Y.colwise().mean();
    Eigen::MatrixXd CovY = (Y_centered.transpose() * Y_centered) / (double)(n_samples - 1);
    Vg = CovY * 0.5;
    Ve = CovY * 0.5;
    B = Eigen::MatrixXd::Zero(n_covars, n_traits);
}

void MvEigenVce::updateEM(Eigen::MatrixXd& Vg_new, Eigen::MatrixXd& Ve_new) {
    Eigen::MatrixXd S_g = Eigen::MatrixXd::Zero(n_traits, n_traits);
    Eigen::MatrixXd S_e = Eigen::MatrixXd::Zero(n_traits, n_traits);

    int pd = n_covars * n_traits;
    Eigen::MatrixXd LHS_B = Eigen::MatrixXd::Zero(pd, pd);
    Eigen::VectorXd RHS_B = Eigen::VectorXd::Zero(pd);

    for (int i = 0; i < n_samples; ++i) {
        Eigen::MatrixXd V_i = Vg * D[i] + Ve;
        // Use LDLT solve for numerical stability instead of direct inverse().
        Eigen::LDLT<Eigen::MatrixXd> V_i_ldlt(V_i);
        if (V_i_ldlt.info() != Eigen::Success) {
            throw std::runtime_error("MvEigenVce::updateEM: V_i is not positive-definite (LDLT failed)");
        }
        Eigen::MatrixXd I_t = Eigen::MatrixXd::Identity(n_traits, n_traits);
        Eigen::MatrixXd V_i_inv = V_i_ldlt.solve(I_t);

        Eigen::VectorXd y_i = Uty.row(i).transpose();
        Eigen::VectorXd x_i;
        if (n_covars > 0) x_i = UtX.row(i).transpose();

        Eigen::VectorXd resid_i = y_i;
        if (n_covars > 0) {
            resid_i -= B.transpose() * x_i;

            Eigen::MatrixXd X_i_kron = Eigen::kroneckerProduct(x_i.transpose(), Eigen::MatrixXd::Identity(n_traits, n_traits));
            LHS_B += X_i_kron.transpose() * V_i_inv * X_i_kron;
            RHS_B += X_i_kron.transpose() * V_i_inv * y_i;
        }

        Eigen::MatrixXd P_i = V_i_inv;

        // Model (after eigendecomp of G): y_i = sqrt(d_i) * u_i + e_i,
        //   u_i ~ N(0, Vg),  e_i ~ N(0, Ve),  V_i = d_i*Vg + Ve.
        // Posterior:  E[u_i|y]      = sqrt(d_i) * Vg * V_i^{-1} * resid_i
        //             Var[u_i|y]    = Vg - d_i * Vg * V_i^{-1} * Vg
        //             E[u_i u_i'|y] = E[u_i|y]*E[u_i|y]' + Var[u_i|y]
        // Vg update:  Vg_new = (1/n) * sum_i E[u_i u_i'|y].
        //
        // We define W_g_i = d_i * Vg * V_i^{-1}  (= sqrt(d_i) * E[u_i|y] / resid_i),
        // so u_hat_i = W_g_i * resid_i = sqrt(d_i) * E[u_i|y].
        // Therefore E[u_i|y]*E[u_i|y]' = u_hat_i*u_hat_i' / d_i, and the
        // variance term is Vg - d_i*Vg*V_i^{-1}*Vg = Vg - W_g_i*Vg.
        // => S_g += u_hat_i*u_hat_i'/d_i + Vg - W_g_i*Vg.
        Eigen::MatrixXd W_g_i = Vg * D[i] * P_i;
        Eigen::VectorXd u_hat_i = W_g_i * resid_i;

        if (D[i] > 1e-12) {
            S_g += (u_hat_i * u_hat_i.transpose()) / D[i] + Vg - W_g_i * Vg;
        } else {
            // d_i ~= 0: u_i is unidentifiable from y_i; posterior == prior Vg.
            S_g += Vg;
        }

        // Ve update: e_i ~ N(0, Ve), E[e_i|y] = Ve * V_i^{-1} * resid_i,
        // Var[e_i|y] = Ve - Ve * V_i^{-1} * Ve.  No d_i scaling here.
        Eigen::MatrixXd W_e_i = Ve * P_i;
        Eigen::VectorXd e_hat_i = W_e_i * resid_i;

        S_e += e_hat_i * e_hat_i.transpose() + Ve - W_e_i * Ve;
    }

    Vg_new = S_g / n_samples;
    Ve_new = S_e / n_samples;

    if (n_covars > 0) {
        Eigen::VectorXd b_vec = LHS_B.ldlt().solve(RHS_B);
        for (int c = 0; c < n_covars; ++c) {
            for (int t = 0; t < n_traits; ++t) {
                B(c, t) = b_vec(c * n_traits + t);
            }
        }
    }
}

bool MvEigenVce::runNullModel() {
    if (options.verbose) std::cout << "MvEigenVce: Running Multivariate Null Model (EM)..." << std::endl;

    Eigen::MatrixXd Vg_new(n_traits, n_traits);
    Eigen::MatrixXd Ve_new(n_traits, n_traits);

    for (int iter = 0; iter < options.max_iter; ++iter) {
        updateEM(Vg_new, Ve_new);

        // Guard against division by zero when Vg/Ve collapse toward zero.
        double vg_norm = Vg.norm(); if (vg_norm < 1e-12) vg_norm = 1e-12;
        double ve_norm = Ve.norm(); if (ve_norm < 1e-12) ve_norm = 1e-12;
        double diff_g = (Vg_new - Vg).norm() / vg_norm;
        double diff_e = (Ve_new - Ve).norm() / ve_norm;

        Vg = Vg_new;
        Ve = Ve_new;

        if (diff_g < options.tol && diff_e < options.tol) {
            if (options.verbose) std::cout << "MvEigenVce: EM Converged in " << iter + 1 << " iterations." << std::endl;
            return true;
        }
    }

    if (options.verbose) std::cout << "MvEigenVce: Warning: EM did not converge within " << options.max_iter << " iterations." << std::endl;
    return false;
}

} // namespace cosmic
