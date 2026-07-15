#include "matrix_inversion.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>

namespace cosmic {
namespace grm {

void invertSPD(std::vector<double> &A, size_t n, double ridge, std::vector<double> &Ainv){
#ifdef USE_EIGEN
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M(n, n);
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<n;++j){
            M(static_cast<int>(i), static_cast<int>(j)) = A[i*n + j];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> es(M);
    if(es.info() != Eigen::Success) throw std::runtime_error("Eigen decomposition failed");
    auto eval = es.eigenvalues();
    auto U = es.eigenvectors();
    double maxeig = 0.0;
    for(int i=0;i<eval.size();++i){
        if(eval(i) > maxeig) maxeig = eval(i);
    }
    double eps = std::max(1e-12*maxeig, ridge);
    for(int i=0;i<eval.size();++i){
        if(eval(i) < 0 && -eval(i) > 1e-12*maxeig) throw std::runtime_error("Negative eigenvalue detected");
        if(eval(i) < eps) eval(i) = 0.0;
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Dinv = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(n, n);
    for(int i=0;i<eval.size();++i){
        if(eval(i) > 0.0) Dinv(i,i) = 1.0 / eval(i);
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> AinvM = U * Dinv * U.transpose();
    Ainv.assign(n*n, 0.0);
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<n;++j){
            Ainv[i*n + j] = AinvM(static_cast<int>(i), static_cast<int>(j));
        }
    }
#elif defined(USE_LAPACK)
    std::vector<double> Ac(n*n);
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<n;++j){
            Ac[j*n + i] = A[i*n + j];
        }
    }
    std::vector<double> w(n);
    int nn = static_cast<int>(n);
    int lda = static_cast<int>(n);
    int info = 0;
    char jobz = 'V';
    char uplo = 'U';
    double wkopt;
    int lwork = -1;
    dsyev_(&jobz, &uplo, &nn, Ac.data(), &lda, w.data(), &wkopt, &lwork, &info);
    if(info != 0) throw std::runtime_error("LAPACK dsyev workspace query failed: info=" + std::to_string(info));
    lwork = static_cast<int>(wkopt);
    std::vector<double> work(lwork);
    dsyev_(&jobz, &uplo, &nn, Ac.data(), &lda, w.data(), work.data(), &lwork, &info);
    if(info != 0) throw std::runtime_error("LAPACK dsyev failed: info=" + std::to_string(info));
    double maxeig = 0.0;
    for(size_t i=0;i<n;++i){ if(w[i] > maxeig) maxeig = w[i]; }
    double eps = std::max(1e-12*maxeig, ridge);
    std::vector<double> invw(n);
    for(size_t i=0;i<n;++i){
        double ev = w[i];
        if(ev < 0 && -ev > 1e-12*maxeig) throw std::runtime_error("Negative eigenvalue detected");
        invw[i] = (ev >= eps) ? (1.0/ev) : 0.0;
    }
    std::vector<double> Ainv_c(n*n, 0.0);
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<n;++j){
            double s = 0.0;
            for(size_t k=0;k<n;++k){
                double uik = Ac[k*n + i];
                double ujk = Ac[k*n + j];
                s += uik * invw[k] * ujk;
            }
            Ainv_c[j*n + i] = s;
        }
    }
    Ainv.assign(n*n, 0.0);
    for(size_t i=0;i<n;++i){
        for(size_t j=0;j<n;++j){
            Ainv[i*n + j] = Ainv_c[j*n + i];
        }
    }
#else
    throw std::runtime_error("No eigen decomposition backend available");
#endif
}

Eigen::MatrixXd compute_explicit_apy_inverse(const Eigen::MatrixXd& G, const std::vector<int>& core_status, double ridge) {
    int n_g = G.rows();
    if (n_g == 0 || core_status.size() != n_g) {
        throw std::runtime_error("compute_explicit_apy_inverse: Invalid dimensions.");
    }

    std::vector<int> core_indices, noncore_indices;
    for (int i = 0; i < n_g; ++i) {
        if (core_status[i] == 1) core_indices.push_back(i);
        else noncore_indices.push_back(i);
    }

    int n_c = core_indices.size();
    int n_n = noncore_indices.size();

    if (n_c == 0) {
        throw std::runtime_error("compute_explicit_apy_inverse: No core individuals specified.");
    }

    Eigen::MatrixXd G_cc(n_c, n_c);
    Eigen::MatrixXd G_cn(n_c, n_n);
    Eigen::VectorXd g_nn_diag(n_n);

    for (int i = 0; i < n_c; ++i) {
        for (int j = 0; j < n_c; ++j) G_cc(i, j) = G(core_indices[i], core_indices[j]);
        for (int j = 0; j < n_n; ++j) G_cn(i, j) = G(core_indices[i], noncore_indices[j]);
    }
    for (int i = 0; i < n_n; ++i) {
        g_nn_diag(i) = G(noncore_indices[i], noncore_indices[i]);
    }

    if (ridge > 0.0) {
        G_cc.diagonal().array() += ridge;
        g_nn_diag.array() += ridge;
    }

    Eigen::MatrixXd G_cc_inv;
    Eigen::LLT<Eigen::MatrixXd> llt_Gcc(G_cc);
    if (llt_Gcc.info() == Eigen::Success) {
        G_cc_inv = llt_Gcc.solve(Eigen::MatrixXd::Identity(n_c, n_c));
    } else {
        G_cc_inv = G_cc.completeOrthogonalDecomposition().pseudoInverse();
    }

    Eigen::VectorXd m_nn(n_n);
    Eigen::MatrixXd G_cc_inv_G_cn = G_cc_inv * G_cn;

    for (int i = 0; i < n_n; ++i) {
        double q = G_cn.col(i).dot(G_cc_inv_G_cn.col(i));
        m_nn(i) = g_nn_diag(i) - q;
        if (m_nn(i) < 1e-8) m_nn(i) = 1e-8; // Prevent division by zero
    }
    Eigen::VectorXd m_nn_inv = m_nn.cwiseInverse();

    Eigen::MatrixXd Ginv = Eigen::MatrixXd::Zero(n_g, n_g);

    Eigen::MatrixXd M_nn_inv_mat = m_nn_inv.asDiagonal();
    Eigen::MatrixXd top_left = G_cc_inv + G_cc_inv_G_cn * M_nn_inv_mat * G_cc_inv_G_cn.transpose();
    Eigen::MatrixXd top_right = -G_cc_inv_G_cn * M_nn_inv_mat;

    for (int i = 0; i < n_c; ++i) {
        for (int j = 0; j < n_c; ++j) Ginv(core_indices[i], core_indices[j]) = top_left(i, j);
        for (int j = 0; j < n_n; ++j) {
            Ginv(core_indices[i], noncore_indices[j]) = top_right(i, j);
            Ginv(noncore_indices[j], core_indices[i]) = top_right(i, j);
        }
    }
    for (int i = 0; i < n_n; ++i) {
        Ginv(noncore_indices[i], noncore_indices[i]) = m_nn_inv(i);
    }

    return Ginv;
}

} // namespace grm
} // namespace cosmic
