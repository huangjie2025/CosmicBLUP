#pragma once
#include "config.h"
#include "mme_builder.h"
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace cosmic {

class AbstractMatrix;
class GenotypeMatrix;
class IdentityMatrixAdapter;

class SolverApp {
public:
    SolverApp(const Config& cfg);
    ~SolverApp();  // defined in .cpp for unique_ptr with incomplete types
    SolverApp(SolverApp&&) = default;
    SolverApp& operator=(SolverApp&&) = default;

    void run();

private:
    Config cfg;

    // === Independent functional modes ===
    void runQc();
    void runSblup();
    void runPrediction();

    // === Shared state (populated by MatrixBuilder::build / DataLoader::load) ===
    std::unique_ptr<AbstractMatrix> Ainv_ptr_, Ginv_ptr_, A22inv_ptr_;
    std::unique_ptr<AbstractMatrix> Qinv_ptr_;
    std::unique_ptr<AbstractMatrix> GDinv_ptr_, GEinv_ptr_, PEinv_ptr_;
    std::unique_ptr<GenotypeMatrix> stcg_genotype_;
    std::unique_ptr<IdentityMatrixAdapter> stcg_ga_dim_;
    std::vector<std::unique_ptr<AbstractMatrix>> generic_rand_invs_;
    std::vector<std::vector<int>> generic_rand_maps_;
    std::map<std::string, int> idmap_;
    std::vector<int> genotyped_map_;
    std::vector<int> global_dam_map_;
    std::string inv_label_, inv_file_, id_file_log_;
    bool use_mmap_ = false;
    bool use_split_ = false;
    bool stcg_mode_ = false;
    std::pair<std::string, std::string> invs_;
    double sigma2_u_ = 1.0;
    double sigma2_e_ = 1.0;
    bool run_vce_ = false;
    struct PhenStats phen_stats_;
    std::vector<struct GenRecord> recs_;

    // === GS pipeline phases (delegated to gs_vce/gs_solve/gs_output) ===
    void runVceAndSolve();

    // === Context for VCE/Solve/Output (populated in run() before runVceAndSolve) ===
    struct FixedDesignG fd_;
    std::vector<struct RandomComponent> base_components_;
    std::vector<int> mat_id_map_;
    int rrm_additive_component_count_ = 0;
    int rrm_pe_component_count_ = 0;
    int rrm_order_ = 2;
    double rrm_tmin_ = 0.0;
    double rrm_tmax_ = 0.0;
    std::vector<double> rrm_curve_times_;
    bool force_implicit_ = false;

    // === Helpers ===
    void writeLogHeader(const std::string& inv_file, const std::string& id_file, std::chrono::system_clock::time_point start_wall);
    void parseModelEquation();
};

// Helper function to parse CLI arguments into Config
Config parse_solver_cli(int argc, char** argv);

} // namespace cosmic
