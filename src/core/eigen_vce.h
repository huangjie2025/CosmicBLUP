#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace cosmic {

// Exact VCE for Single Trait using Eigen Decomposition (Brent Search)
class EigenVce {
public:
    struct Options {
        bool use_covariate_projection = true;
    };

    EigenVce();
    EigenVce(const Options& opts);

    void prepare(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::MatrixXd& G);
    void prepareSparse(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::SparseMatrix<double>& G_sparse);

    // RRM-specific VCE logic for longitudinal data
    void prepareRRM(const Eigen::VectorXd& y, const Eigen::MatrixXf& X, const Eigen::MatrixXd& Z, const Eigen::MatrixXd& Phi, const Eigen::MatrixXd& G);

    bool runNullModel();

    // Skip VCE and use pre-computed variance components directly
    void fixedComponents(double s2g, double s2e);

    double getSigma2g() const { return sigma2_g; }
    double getSigma2e() const { return sigma2_e; }
    double getDelta() const { return null_delta; }
    const Eigen::VectorXd& getEigenvalues() const { return D; }
    const Eigen::MatrixXd& getEigenvectors() const { return U; }
    const Eigen::MatrixXd& getProjectedW() const { return W_proj; }
    const Eigen::VectorXd& getUty() const { return Uty; }
    const Eigen::MatrixXd& getUtX() const { return UtX; }

private:
    Options options;
    int n_samples;
    int n_covars;

    Eigen::VectorXd D;
    Eigen::MatrixXd U;
    Eigen::MatrixXd W_proj;
    Eigen::VectorXd Uty;
    Eigen::MatrixXd UtX;

    double null_delta = 1.0;
    double sigma2_g = 0.0;
    double sigma2_e = 0.0;

    double computeNegLogLikelihood(double delta);
};

// Exact VCE for Multiple Traits using Eigen Decomposition (EM/REML)
class MvEigenVce {
public:
    struct Options {
        int max_iter = 100;
        double tol = 1e-5;
        bool use_projection = true;
        bool verbose = true;
    };

    MvEigenVce();
    MvEigenVce(const Options& opts);

    void prepare(const Eigen::MatrixXd& Y, const Eigen::MatrixXd& X, const Eigen::MatrixXd& G);
    bool runNullModel();

    Eigen::MatrixXd getVg() const { return Vg; }
    Eigen::MatrixXd getVe() const { return Ve; }
    const Eigen::VectorXd& getEigenvalues() const { return D; }
    const Eigen::MatrixXd& getProjectedW() const { return W_proj; }
    const Eigen::MatrixXd& getUty() const { return Uty; }

private:
    Options options;
    int n_samples;
    int n_traits;
    int n_covars;

    Eigen::VectorXd D;
    Eigen::MatrixXd U;
    Eigen::MatrixXd W_proj;
    Eigen::MatrixXd Uty;
    Eigen::MatrixXd UtX;

    Eigen::MatrixXd Vg;
    Eigen::MatrixXd Ve;
    Eigen::MatrixXd B;

    void updateEM(Eigen::MatrixXd& Vg_new, Eigen::MatrixXd& Ve_new);
};

} // namespace cosmic
