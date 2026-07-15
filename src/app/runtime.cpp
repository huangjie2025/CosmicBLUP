#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

#include "blup_app.h"
#include "phenotype_loader.h"
#include "logger.h"
#include "matrix_adapter.h"
#include "model_pipeline.h"
#include "random_regression.h"
#include "relationship_builder.h"
#include "string_utils.h"
#include "version.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <filesystem>

#if defined(__GNUC__) || defined(__clang__)
extern "C" void openblas_set_num_threads(int num_threads) __attribute__((weak));
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cosmic {

using namespace std;

void SolverApp::writeLogHeader(const std::string& inv_file_, const std::string& id_file,
                               std::chrono::system_clock::time_point start_wall) {
    auto& logger = Logger::getInstance();

    std::time_t ts = std::chrono::system_clock::to_time_t(start_wall);
    std::tm tm_now;
#ifdef _WIN32
    localtime_s(&tm_now, &ts);
#else
    localtime_r(&ts, &tm_now);
#endif

    char buf_start[64];
    std::strftime(buf_start, sizeof(buf_start), "%a %b %d %H:%M:%S %Y", &tm_now);

    char hostname[256];
#ifdef _WIN32
    DWORD size = sizeof(hostname);
    if (!GetComputerNameA(hostname, &size)) {
        strcpy(hostname, "Unknown");
    }
#else
    if (gethostname(hostname, sizeof(hostname)) != 0) {
        strcpy(hostname, "Unknown");
    }
#endif

    logger.logRaw("#===============================================================#\n");
    logger.logRaw("#                   WELCOME TO COSMICBLUP                       #\n");
    logger.logRaw("#           Home: https://github.com/huangjie2025/CosmicBLUP    #\n");
#ifdef _WIN32
    const char* platform = "Windows";
#elif defined(__APPLE__)
    const char* platform = "macOS";
#else
    const char* platform = "Linux";
#endif
    logger.logRaw("#      Software: " + std::string(platform) + " v" + COSMICBLUP_VERSION + "\n");
    logger.logRaw("#    Developers: Jie Huang                                      #\n");
    logger.logRaw("#     Copyright: (c) 2026-present. All rights reserved.         #\n");
    logger.logRaw("#===============================================================#\n");
    logger.logRaw("Analysis started: " + std::string(buf_start) + "\n");
    logger.logRaw("Hostname: " + std::string(hostname) + "\n\n");

    logger.logRaw("Commands:\n" + cfg.full_command + "\n\n");
    logger.logRaw("ThreadsRequested: " + std::to_string(cfg.threads > 0 ? cfg.threads : 1) + "\n");
#ifdef _OPENMP
    logger.logRaw("OpenMP: enabled\n");
#else
    logger.logRaw("OpenMP: disabled\n");
#endif
#ifndef _WIN32
    const char* openblas_threads = std::getenv("OPENBLAS_NUM_THREADS");
    const char* mkl_threads = std::getenv("MKL_NUM_THREADS");
    logger.logRaw("OPENBLAS_NUM_THREADS: " + std::string(openblas_threads ? openblas_threads : "(unset)") + "\n");
    logger.logRaw("MKL_NUM_THREADS: " + std::string(mkl_threads ? mkl_threads : "(unset)") + "\n\n");
#else
    logger.logRaw("OPENBLAS_NUM_THREADS: (platform-managed)\n");
    logger.logRaw("MKL_NUM_THREADS: (platform-managed)\n\n");
#endif

    LOG_INFO("Program will be running on up to " << (cfg.threads > 0 ? cfg.threads : 1) << " threads.");
    LOG_INFO("The log will be recorded in file [" << cfg.out_prefix << ".log].\n");
}

void SolverApp::run() {
    parseModelEquation();

    const std::filesystem::path output_path(cfg.out_prefix);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    if (cfg.threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(cfg.threads);
        Eigen::setNbThreads(cfg.threads);
#endif
#ifndef _WIN32
        setenv("OPENBLAS_NUM_THREADS", std::to_string(cfg.threads).c_str(), 1);
        setenv("MKL_NUM_THREADS", std::to_string(cfg.threads).c_str(), 1);
#else
        _putenv_s("OPENBLAS_NUM_THREADS", std::to_string(cfg.threads).c_str());
        _putenv_s("MKL_NUM_THREADS", std::to_string(cfg.threads).c_str());
#endif
    }

    std::string logp = cfg.out_prefix + ".log";
    Logger::getInstance().setLogFile(logp);

    auto start_wall = std::chrono::system_clock::now();
    auto start_steady = std::chrono::steady_clock::now();
    (void)start_steady;

    invs_ = std::make_pair(std::string(), std::string());
    if (!cfg.inv_matrix_path.empty()) {
        invs_ = detect_inv_files(cfg.inv_matrix_path);
        if (!cfg.inv_id_path.empty()) invs_.second = cfg.inv_id_path;
    } else if (!cfg.xrminv_path.empty()) {
        invs_ = detect_inv_files(cfg.xrminv_path);
    }

    use_split_ = !cfg.inv_A_path.empty() || (!cfg.ped_path.empty() && (!cfg.bfile_path.empty() || !cfg.pfile_path.empty()));
    use_mmap_ = cfg.use_mmap;

    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    stcg_mode_ = to_lower(cfg.vce_mode) == "stcg";

    if (cfg.run_qc) {
        runQc();
        return;
    }

    if (cfg.sblup) {
        runSblup();
        return;
    }

    if (cfg.pred) {
        runPrediction();
        return;
    }

    auto mat_result = MatrixBuilder::build(cfg);
    Ainv_ptr_ = std::move(mat_result.Ainv);
    Ginv_ptr_ = std::move(mat_result.Ginv);
    A22inv_ptr_ = std::move(mat_result.A22inv);
    Qinv_ptr_ = std::move(mat_result.Qinv);
    GDinv_ptr_ = std::move(mat_result.GDinv);
    GEinv_ptr_ = std::move(mat_result.GEinv);
    stcg_genotype_ = std::move(mat_result.stcg_genotype);
    stcg_ga_dim_ = std::move(mat_result.stcg_ga_dim);
    idmap_ = std::move(mat_result.idmap);
    genotyped_map_ = std::move(mat_result.genotyped_map);
    global_dam_map_ = std::move(mat_result.global_dam_map);
    inv_label_ = std::move(mat_result.inv_label);
    inv_file_ = std::move(mat_result.inv_file);
    id_file_log_ = std::move(mat_result.id_file_log);
    use_split_ = mat_result.use_split;
    stcg_mode_ = mat_result.stcg_mode;

    if (cfg.make_xrm && cfg.pheno_path.empty() && !cfg.sblup && !cfg.pred) {
        return;
    }

    auto data_result = DataLoader::load(cfg, idmap_);
    recs_ = std::move(data_result.records);
    phen_stats_ = data_result.phen_stats;
    sigma2_u_ = data_result.sigma2_u;
    sigma2_e_ = data_result.sigma2_e;
    run_vce_ = data_result.run_vce;
    fd_ = std::move(data_result.fixed_design);

    if (cfg.multi_trait) {
        LOG_INFO("Multi-Trait Mode Enabled (VCE + MME Solve + EBV Output)");
        if (!cfg.run_vce) {
            LOG_WARN("Multi-trait requires VCE to estimate Vg/Ve before MME solve. Forcing VCE mode.");
            cfg.run_vce = true;
            run_vce_ = true;
        }
    } else if (cfg.rrm_model) {
        LOG_INFO("Random Regression Model (RRM) GS Mode Enabled");
        if (!cfg.run_vce) {
            LOG_WARN("RRM requires variance setup before final GS solving. Forcing VCE mode.");
            cfg.run_vce = true;
            run_vce_ = true;
        }
    }

    writeLogHeader(inv_file_, id_file_log_, start_wall);

    LOG_INFO("Single Trait Model Fitted:");
    std::string model_str = "  " + (cfg.pheno_name.empty() ? "y" : cfg.pheno_name) + " = 1";
    for (const auto& name : cfg.dcovar_names) {
        model_str += " + " + name + "(F)";
    }
    for (const auto& name : cfg.qcovar_names) {
        model_str += " + " + name + "(C)";
    }
    if (!cfg.ped_path.empty() && cfg.bfile_path.empty()) {
        model_str += cfg.rrm_model ? " + PA(RRM[G])" : " + PA(R[G])";
    } else if (cfg.ped_path.empty() && !cfg.bfile_path.empty()) {
        model_str += cfg.rrm_model ? " + GA(RRM[G])" : " + GA(R[G])";
    } else if (!cfg.ped_path.empty() && !cfg.bfile_path.empty()) {
        model_str += cfg.rrm_model ? " + HA(RRM[G])" : " + HA(R[G])";
    } else {
        model_str += " + Zu";
    }
    if (cfg.pe_effect) {
        model_str += cfg.rrm_model ? " + Pe(RRM[I])" : " + Pe(R[I])";
    }
    for (const auto& name : cfg.rand_names) {
        model_str += " + " + name + "(R[I])";
    }
    if (cfg.mat_effect) {
        model_str += " + Mat(R[G])";
    }
    model_str += " + e\n";
    LOG_INFO(model_str);

    runVceAndSolve();

    auto end_wall = std::chrono::system_clock::now();
    std::time_t ts_end = std::chrono::system_clock::to_time_t(end_wall);
    std::tm tm_end;
#ifdef _WIN32
    localtime_s(&tm_end, &ts_end);
#else
    localtime_r(&ts_end, &tm_end);
#endif
    char buf_end[64];
    std::strftime(buf_end, sizeof(buf_end), "%a %b %d %H:%M:%S %Y", &tm_end);

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_wall - start_wall).count();
    int h = duration / 3600;
    int m = (duration % 3600) / 60;
    int s = duration % 60;

    LOG_INFO("Analysis finished: " << buf_end);
    LOG_INFO("Total running time: " << h << "h" << m << "m" << s << "s");

#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        LOG_INFO("Peak Memory: " << (pmc.PeakWorkingSetSize/1024/1024) << " MB");
    }
#else
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmHWM:") {
            long peak_kb = 0;
            if (sscanf(line.c_str(), "VmHWM: %ld kB", &peak_kb) == 1) {
                LOG_INFO("Peak Memory: " << (peak_kb / 1024) << " MB");
            }
            break;
        }
    }
#endif
}

void SolverApp::runVceAndSolve() {
    if (cfg.pe_effect) {
        int pe_dim = 0;
        if (Qinv_ptr_) {
            pe_dim = Qinv_ptr_->rows();
        } else {
            for (const auto& r : recs_) {
                if (r.aid > pe_dim) pe_dim = r.aid;
            }
        }
        if (pe_dim > 0) {
            PEinv_ptr_ = std::make_unique<IdentityMatrixAdapter>(pe_dim);
            LOG_INFO("Permanent Environmental effect (Pe) matrix configured with dimension " << pe_dim);
        } else {
            throw std::runtime_error("Could not determine dimension for Permanent Environmental effect.");
        }
    }

    for (size_t rc = 0; rc < cfg.rand_names.size(); ++rc) {
        std::map<std::string, int> level_map;
        std::vector<int> current_map(recs_.size(), -1);
        int current_id = 0;
        for (size_t i = 0; i < recs_.size(); ++i) {
            if (rc < recs_[i].rand_cats.size()) {
                const std::string& level = recs_[i].rand_cats[rc];
                if (level_map.find(level) == level_map.end()) {
                    level_map[level] = current_id++;
                }
                current_map[i] = level_map[level];
            }
        }
        generic_rand_maps_.push_back(current_map);
        generic_rand_invs_.push_back(std::make_unique<IdentityMatrixAdapter>(current_id));
        LOG_INFO("Generic Random effect '" << cfg.rand_names[rc] << "' configured with " << current_id << " levels.");
    }

    if (cfg.mat_effect) {
        if (!Qinv_ptr_) {
            throw std::runtime_error("Maternal effect requires a genetic relationship matrix (A or G).");
        }
        if (global_dam_map_.empty()) {
            throw std::runtime_error("Maternal effect requires pedigree information (--ped) to track Dam IDs.");
        }
        mat_id_map_.assign(recs_.size(), -1);
        int valid_mat = 0;
        for (size_t i = 0; i < recs_.size(); ++i) {
            int aid_0based = recs_[i].aid - 1;
            if (aid_0based >= 0 && aid_0based < (int)global_dam_map_.size()) {
                int dam_0based = global_dam_map_[aid_0based];
                if (dam_0based >= 0) {
                    mat_id_map_[i] = dam_0based;
                    valid_mat++;
                }
            }
        }
        LOG_INFO("Maternal effect (Mat) initialized. " << valid_mat << " records mapped to a valid Dam.");
    }

    rrm_order_ = cfg.cli_present.count("--rrm-order") ? cfg.rrm_order : 2;
    rrm_additive_component_count_ = 0;
    rrm_pe_component_count_ = 0;
    rrm_curve_times_.clear();

    if (cfg.rrm_model) {
        if (cfg.mat_effect || !generic_rand_invs_.empty() || GDinv_ptr_ || GEinv_ptr_) {
            throw std::runtime_error("RRM GS first phase currently supports additive genetic effect (+ optional Pe) only.");
        }
        auto rrm_ctx = GSRrm::setup({recs_, cfg, Qinv_ptr_.get(), PEinv_ptr_.get()});
        base_components_ = std::move(rrm_ctx.base_components);
        rrm_additive_component_count_ = rrm_ctx.additive_component_count;
        rrm_pe_component_count_ = rrm_ctx.pe_component_count;
        rrm_order_ = rrm_ctx.order;
        rrm_tmin_ = rrm_ctx.tmin;
        rrm_tmax_ = rrm_ctx.tmax;
        rrm_curve_times_ = std::move(rrm_ctx.curve_times);
    } else {
        if (Qinv_ptr_) {
            RandomComponent rc;
            rc.Qinv = Qinv_ptr_.get();
            base_components_.push_back(rc);
        }
    }

    force_implicit_ = use_mmap_ || use_split_ || cfg.matrix_free;

    GSModel::SharedData gs_data{
        recs_, fd_, cfg, idmap_,
        Qinv_ptr_.get(), GDinv_ptr_.get(), GEinv_ptr_.get(), PEinv_ptr_.get(),
        Ginv_ptr_.get(),
        stcg_ga_dim_.get(), stcg_genotype_.get(),
        generic_rand_invs_, generic_rand_maps_,
        mat_id_map_,
        base_components_,
        rrm_additive_component_count_, rrm_pe_component_count_,
        rrm_order_, rrm_tmin_, rrm_tmax_, rrm_curve_times_,
        force_implicit_, stcg_mode_, use_split_,
        sigma2_u_, sigma2_e_, run_vce_,
        inv_label_, genotyped_map_
    };
    GSModel::run(gs_data, sigma2_u_, sigma2_e_);
}

} // namespace cosmic
