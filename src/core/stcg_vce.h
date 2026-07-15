#pragma once
#include "vce.h"
#include "matrix_free_pcg.h"

namespace cosmic {

class STCG_VCE : public VCESolver {
public:
    STCG_VCE(const std::vector<GenRecord>& recs,
             const FixedDesignG& fd,
             const std::vector<RandomComponent>& comps,
             const Eigen::VectorXd& y,
             VCEConfig cfg);

    void solve() override;

private:
    const std::vector<GenRecord>& recs;
    const FixedDesignG& fd;
    const std::vector<RandomComponent>& comps;
    Eigen::VectorXd y;
    VCEConfig cfg;

    int n_records = 0;
    int n_ind = 0;
    int p = 0;
    int genetic_idx = -1;
    int pe_idx = -1;

    GenotypeMatrix* geno = nullptr;
    int n_threads = 1;

    std::vector<int> rec_to_ind;
    std::vector<int> n_obs_per_ind;
    Eigen::MatrixXd X_dense;

    Eigen::VectorXd solve_V(const Eigen::VectorXd& b, const Eigen::VectorXd& warm_start = Eigen::VectorXd());
    void apply_V(const Eigen::VectorXd& v, Eigen::VectorXd& out);
    void apply_G(const Eigen::VectorXd& u, Eigen::VectorXd& out);
    void gather_to_ind(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_ind);
    void scatter_to_rec(const Eigen::VectorXd& v_ind, Eigen::VectorXd& out_rec);
    void apply_dV_genetic(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_rec);
    void apply_dV_pe(const Eigen::VectorXd& v_rec, Eigen::VectorXd& out_rec);

    void compute_Vinv_X_y(const Eigen::VectorXd& y_in,
                          Eigen::MatrixXd& Vinv_X,
                          Eigen::VectorXd& Vinv_y,
                          Eigen::MatrixXd& XtVinvX_inv,
                          Eigen::VectorXd& beta);

    Eigen::VectorXd apply_P(const Eigen::VectorXd& v,
                            const Eigen::MatrixXd& Vinv_X,
                            const Eigen::MatrixXd& XtVinvX_inv);
};

} // namespace cosmic
