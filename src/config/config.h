#pragma once
#include <string>
#include <vector>
#include <set>
#include <map>

namespace cosmic {

struct Config {
    bool blup_mode = false;
    bool single_trait = true;
    bool multi_trait = false;
    bool rrm_model = false;
    int rrm_order = 2; // Default Legendre polynomial order
    bool show_version = false;
    bool run_vce = false; // VCE Flag
    std::set<std::string> cli_present;
    std::string pheno_path;
    int pheno_pos = -1;
    std::string pheno_name;
    std::vector<int> dcovar_cols;
    std::vector<int> qcovar_cols;
    std::vector<int> rand_cols; // --rand
    std::vector<std::string> dcovar_names;
    std::vector<std::string> qcovar_names;
    std::vector<std::string> rand_names; // --rand-names
    std::string qcovar_names_str;
    std::string time_col_name;
    std::string id_col_name;
    std::string xrminv_path;
    std::string matrix_path;
    std::string inv_matrix_path;
    std::string inv_id_path;
    std::string matrix_id_path;
    std::string vars_path;
    std::string out_prefix;
    std::string model;
    std::string relationship;
    int threads = 1;
    std::string relationship_output_format = "txt";
    double pcg_tol = 1e-10;
    int pcg_maxit = 10000;
    int pcg_report_every = 200;
    std::string pcg_precond = "auto";
    bool calc_se = true;
    bool calc_snp_effect = false; // New flag for SNP effects back-solving
    bool matrix_free = false;
    bool use_mmap = false;

    // VCE params
    int vce_max_iter = 20;
    int ai_maxit = 20; // --ai-maxit
    int em_maxit = 20; // --em-maxit
    int pcg_num = 10000; // --pcg-num
    bool use_he_pcg = false; // --pcg or --PCG
    int vce_mc_samples = 100; // Increased default for stability with GBLUP/PBLUP
    double vce_tol = 1e-4;
    std::string vce_mode = "ai"; // Default to AI (match HIBLUP)
    std::string dense_solver = "cholesky"; // --dense-solver (direct, cholesky, eigen, lowrank, pcgslq)
    std::string trace_mode = "auto";   // --trace-mode (auto, exact, fdiff, hutch, slq)
    std::string solver_mode = "auto";  // --solver (auto, dense, sparse, pcg)
    bool print_report = false;         // --solver-report
    std::string ped_path; // For Pedigree Sampler
    bool force_dense_exact = false;
    bool force_exact = false;

    // Decomposition support
    std::string inv_A_path;
    std::string inv_G_path;
    std::string inv_A22_path;
    std::string id_A_path;
    std::string id_G_path; // A22 usually shares ID with G

    // In-memory matrix build support
    std::string bfile_path; // For PLINK binary files (.bed)
    std::string pfile_path; // For PLINK2 binary files (.pgen)
    std::string bgen_path;  // For BGEN files
    // ped_path is already defined above

    // QC options
    bool run_qc = false;
    float qc_maf = 0.0f;
    float qc_max_maf = 1.0f;
    float qc_hwe = 0.0f;
    float qc_geno = 1.0f;
    float qc_mind = 1.0f;

    double tau = 1.0;
    double omega = 1.0;
    bool skip_vce = false;
    double var_a = -1.0;
    double var_e = -1.0;

    // Original command line
    std::string full_command;

    // Variance priors
    std::vector<double> var_priors;
    std::vector<double> covc_priors;

    // Auxiliary BLUP and relationship parameters
    std::string pop_class;
    bool write_txt = false;
    bool sblup = false;
    std::string sumstat_path; // --sumstat
    double h2 = -1.0; // --h2
    int window_num = -1; // --window-num
    bool pred = false;
    std::string score_path; // --score
    bool allele_freq = false;
    bool homo = false;
    bool hete = false;
    bool ibc = false;
    bool rc = false;
    bool pca = false;
    int npc = 10;
    bool make_xrm = false;
    double ridge_value = 0.0;
    int code_method = 1; // 1: VanRaden, 2: Zeng, 3: Yang, 4: Vitezica
    bool tuneG = true; // Whether to align G to A22 in SSGBLUP
    double alpha = 0.05; // Blending factor for ssGBLUP (H = (1-alpha)*G + alpha*A22)
    bool add_effect = false; // --add
    bool dom_effect = false; // --dom
    bool epi_effect = false; // --epi
    bool pe_effect = false;  // --pe (Permanent Environmental effect)
    bool mat_effect = false; // --mat (Maternal effect)
    bool add_inv = false; // --add-inv
    bool dom_inv = false; // --dom-inv
    bool epi_inv = false; // --epi-inv
    int block_size = 2048; // --step
    std::string snp_weight_file; // --snp-weight
    std::string pop_class_file; // --pop-class

    // APY Support
    bool use_apy = false; // --apy
    int apy_core_size = 0; // Number of core animals
    std::string apy_core_file; // File containing core animal IDs
};

struct PhenStats {
    long long n_rows_total = 0;
    long long n_used = 0;
    long long n_dropped_missing = 0;
    long long n_not_in_id = 0;
};

} // namespace cosmic
