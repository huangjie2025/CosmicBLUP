#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/Cholesky>
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include "rng.h"
#include "pcg_solver.h" // Include full solver definition
#include "mme_builder.h"
#include "matrix_adapter.h"
#include "dense_solver.h"

namespace cosmic {

class ImplicitMME;

// Algorithm Enumeration
enum class VCEAlgorithm {
    AI, // AI-REML (Average Information)
    EM, // EM-REML (Expectation Maximization)
    MC, // MC-EM (Monte Carlo EM)
    HE, // HE (Haseman-Elston Regression)
    EMAI, // EM + AI-REML Hybrid
    HI, // HE + AI-REML Hybrid
    Exact, // Exact LMM via EigenVce
    VMatrix, // Dense Exact V-Matrix AI-REML
    STCG, // Stochastic Trace Conjugate Gradient
    Fdiff // Finite Difference AI-REML
};

// ============================================================================
// Strategy Policy & Report (IASBLUP-style unified decision framework)
// ============================================================================

/// High-level strategy: which mathematical formulation to use
enum class VceStrategy {
    VBased,      ///< Operate on V = ZGZ' + R (phenotype covariance)
    MMEBased,    ///< Operate on MME coefficient matrix C
    MatrixFree   ///< Matrix-free operators (ImplicitMME / STCG)
};

/// Linear solver type
enum class SolverType {
    DenseLDLT,       ///< Eigen::LDLT (dim < 5K)
    SparseLDLT,      ///< Eigen::SimplicialLDLT (5K < dim < 200K)
    PCG,             ///< Preconditioned Conjugate Gradient (dim > 200K)
    DenseVCholesky,  ///< V-based Cholesky/LLT (V route)
    DenseVEigen,     ///< V-based Eigendecomposition (V route)
    DenseVLowRank,   ///< V-based Low-rank SVD (V route)
    DenseVPCG_SLQ,   ///< V-based PCG+SLQ (V route)
    STCG             ///< Stochastic Trace CG (repeatability matrix-free)
};

/// Trace estimation method for AI-REML
enum class TraceMode {
    Exact,    ///< Takahashi inverse from sparse/dense factorization
    Fdiff,    ///< Finite difference approximation
    Hutch,    ///< Hutchinson stochastic estimator
    SLQ       ///< Stochastic Lanczos Quadrature
};

/// Preconditioner type for PCG
enum class PrecondType {
    None,         ///< No preconditioner (set externally)
    Jacobi,       ///< Diagonal preconditioner
    BlockJacobi,  ///< Block-Jacobi (fixed effects exact + random diagonal)
    IC0,          ///< Incomplete Cholesky(0)
    Pedigree      ///< Pedigree-aware (Ainv-based)
};

/// Task type driving the strategy choice
enum class TaskType {
    VCE,             ///< Variance component estimation
    Prediction,      ///< Breeding value prediction
    GWASNull,        ///< GWAS null model
    Repeatability,   ///< Repeatability model
    RRM              ///< Random regression model
};

/// Input parameters for policy decision
struct PolicyInput {
    int n_records = 0;                    ///< Number of observation records
    int n_individuals = 0;                ///< Number of distinct individuals
    int mme_dim = 0;                      ///< MME dimension (p + q)
    double lhs_density = 0.0;             ///< LHS non-zero density (0-1)
    int n_components = 0;                 ///< Number of random effect components
    bool has_pedigree_sparse_inverse = false;  ///< Ainv available as sparse?
    bool has_genotype_operator = false;    ///< Genotype matrix operator available?
    TaskType task_type = TaskType::VCE;
};

/// The chosen policy (output of choose_policy)
struct SolverPolicy {
    VceStrategy strategy = VceStrategy::MMEBased;
    SolverType solver = SolverType::SparseLDLT;
    TraceMode trace_mode = TraceMode::Exact;
    PrecondType preconditioner = PrecondType::Jacobi;
    std::string reason;  ///< Human-readable reason for this choice
};

/// Runtime report (what actually happened)
struct SolverReport {
    TaskType task = TaskType::VCE;
    VceStrategy strategy = VceStrategy::MMEBased;
    SolverType solver = SolverType::SparseLDLT;
    TraceMode trace_mode = TraceMode::Exact;
    PrecondType preconditioner = PrecondType::Jacobi;
    bool fallback_triggered = false;
    std::string fallback_reason;
    std::string strategy_reason;

    // Runtime stats
    int mme_dim = 0;
    double lhs_density = 0.0;
    int n_components = 0;
    int pcg_iterations = 0;
    int vce_iterations = 0;
};

/// Centralized policy decision function
SolverPolicy choose_policy(const PolicyInput& input);

/// Print IASBLUP-style solver report
void print_solver_report(const SolverReport& report, std::ostream& os = std::cout);

// String conversion helpers
std::string to_string(VceStrategy s);
std::string to_string(SolverType s);
std::string to_string(TraceMode m);
std::string to_string(PrecondType p);
std::string to_string(TaskType t);

// ============================================================================
// VCEConfig
// ============================================================================

struct VCEConfig {
    VCEAlgorithm algorithm = VCEAlgorithm::EMAI; // Default to Hybrid
    DenseSolverTier dense_tier = DenseSolverTier::CholeskyLLT; // Default solver tier for dense VCE

    int max_iter = 100;
    int mc_samples = 20; // Number of samples for MC-step (MC-EM usually needs fewer than full MC)
    double tol = 1e-6;
    bool verbose = true;
    int pcg_max_iter = 2000;
    double pcg_tol = 1e-10;
    std::string pcg_precond = "diag";
    std::string vce_mode = "hybrid"; // "ai", "em", "hybrid", "mc", "he", "exact", "vmatrix"
    bool force_dense = false;
    bool force_exact = false; // Force Exact AI-REML (Data-Driven AI + Takahashi/Dense)
    bool use_implicit = false; // Matrix-Free mode
    std::ostream* log_stream = nullptr;

    // Unified strategy parameters (IASBLUP-style auto decision)
    std::string trace_mode = "auto";   // "auto", "exact", "fdiff", "hutch", "slq"
    std::string solver_mode = "auto";  // "auto", "dense", "sparse", "pcg", "vmatrix", "stcg"
    bool print_report = true;          // Print solver report at start

    // Hybrid Strategy Settings
    int em_max_iter = 5; // Number of EM iterations in Hybrid mode
    bool use_he_init = false; // HI Strategy: Use HE Regression for initialization

    void set_algorithm_from_string(const std::string& mode) {
        vce_mode = mode;
        std::string m = mode;
        std::transform(m.begin(), m.end(), m.begin(), ::tolower);
        if (m == "ai") algorithm = VCEAlgorithm::AI;
        else if (m == "em") algorithm = VCEAlgorithm::EM;
        else if (m == "mc") algorithm = VCEAlgorithm::MC;
        else if (m == "he") algorithm = VCEAlgorithm::HE;
        else if (m == "emai") algorithm = VCEAlgorithm::EMAI;
        else if (m == "hi") algorithm = VCEAlgorithm::HI;
        else if (m == "exact") algorithm = VCEAlgorithm::Exact;
        else if (m == "vmatrix") algorithm = VCEAlgorithm::VMatrix;
        else if (m == "stcg") algorithm = VCEAlgorithm::STCG;
        else if (m == "fdiff") algorithm = VCEAlgorithm::Fdiff;
        else algorithm = VCEAlgorithm::EMAI; // Default Hybrid
    }
};

// Abstract base class for Prior Sampler
class PriorSampler {
public:
    virtual ~PriorSampler() = default;
    // Sample u ~ N(0, G)
    // For standard model G = I * var_u, or G = A * var_u
    // We return standardized sample (mean 0, var 1), caller scales by sqrt(var_u)
    virtual void sample(Eigen::VectorXd& u) = 0;
};

class IdentitySampler : public PriorSampler {
    RNG& rng;
public:
    IdentitySampler(RNG& r) : rng(r) {}
    void sample(Eigen::VectorXd& u) override {
        rng.fill_normal(u);
    }
};

// Pedigree Sampler (u = L * z)
// Requires pedigree file to build L implicitly or explicitly
class PedigreeSampler : public PriorSampler {
    RNG& rng;
    // We store pedigree structure: sire, dam for each individual
    // IDs are mapped to 0..n-1
    struct Node { int s = -1, d = -1; };
    std::vector<Node> ped;
    std::vector<double> D_sqrt; // Sqrt of diagonal of D matrix (Mendelian sampling variance)

public:
    PedigreeSampler(RNG& r) : rng(r) {}

    // Load pedigree from file: ID Sire Dam
    // ID map must be consistent with the MME equation order
    void load(const std::string& ped_file, const std::map<std::string, int>& id_map);

    void sample(Eigen::VectorXd& u) override;
};

class VCESolver {
protected:
    const Eigen::SparseMatrix<double>* X;
    const Eigen::SparseMatrix<double>* Z;
    // const AbstractMatrix* Ainv_ptr; // Removed in favor of components list
    const Eigen::VectorXd& y;

    double var_e = 1.0;
    std::vector<double> vars_u; // Multiple variance components

    // Standard Errors
    double var_e_se = 0.0;
    std::vector<double> vars_u_se;

    // Legacy support: if user asks for single var_u, return first
    double var_u_legacy = 1.0;

    VCEConfig config;
    RNG rng;
    std::vector<std::shared_ptr<PriorSampler>> samplers; // Multiple samplers
    bool converged = false;
    int iterations_run = 0;
    double last_diff = std::numeric_limits<double>::quiet_NaN();
    double last_logL = -1e20;

    // Components
    std::vector<RandomComponent> components;

    // Final Solution (BLUP)
    Eigen::VectorXd final_solution;

public:
    // Expose RNG for samplers
    RNG& rng_ref() { return rng; }

    // Unified Constructor using Recs/FD (Multiple Components)
    VCESolver(const std::vector<GenRecord>& recs_ref,
              const FixedDesignG& fd_ref,
              const std::vector<RandomComponent>& comps,
              const Eigen::VectorXd& y_ref,
              VCEConfig cfg = VCEConfig())
        : X(nullptr), Z(nullptr), y(y_ref), config(cfg), recs_ptr(&recs_ref), fd_ptr(&fd_ref), components(comps) {

        // Initialize variances
        if (components.empty()) {
             // Should not happen?
        } else {
             vars_u.resize(components.size(), 1.0);
             // Create default samplers (Identity)
             for(size_t k=0; k<components.size(); ++k) {
                 samplers.push_back(std::make_shared<IdentitySampler>(rng));
             }
        }
        var_u_legacy = 1.0;
    }

    // Legacy Constructor (Single Component)
    VCESolver(const std::vector<GenRecord>& recs_ref,
              const FixedDesignG& fd_ref,
              const AbstractMatrix* Ainv_ref,
              const Eigen::VectorXd& y_ref, // Keep y for compatibility or extract from recs
              VCEConfig cfg = VCEConfig())
        : X(nullptr), Z(nullptr), y(y_ref), config(cfg), recs_ptr(&recs_ref), fd_ptr(&fd_ref) {

        RandomComponent rc;
        rc.Qinv = Ainv_ref;
        components.push_back(rc);

        vars_u.push_back(1.0);
        samplers.push_back(std::make_shared<IdentitySampler>(rng));
        var_u_legacy = 1.0;
    }

    // Legacy Constructor (Matrix mode - Deprecated/Partial support)
    VCESolver(const Eigen::SparseMatrix<double>& X_,
              const Eigen::SparseMatrix<double>& Z_,
              const AbstractMatrix* Ainv_,
              const Eigen::VectorXd& y_,
              VCEConfig cfg = VCEConfig())
        : X(&X_), Z(&Z_), y(y_), config(cfg) {
            // This mode doesn't support generic builder easily.
            // We assume single component.
            RandomComponent rc;
            rc.Qinv = Ainv_;
            components.push_back(rc);
            vars_u.push_back(1.0);
            samplers.push_back(std::make_shared<IdentitySampler>(rng));
    }

    virtual ~VCESolver() = default;

    // Pointers to raw data (optional, for Builder)
    const std::vector<GenRecord>* recs_ptr = nullptr;
    const FixedDesignG* fd_ptr = nullptr;

    virtual void solve() = 0;

    virtual void calculate_SE() {}

    double getVarE() const { return var_e; }
    double getVarU() const { return vars_u.empty() ? 0.0 : vars_u[0]; }
    const std::vector<double>& getVarsU() const { return vars_u; }

    // Standard Error Getters
    double getSEVarE() const { return var_e_se; }
    double getSEVarU() const { return vars_u_se.empty() ? 0.0 : vars_u_se[0]; }
    const std::vector<double>& getSEVarsU() const { return vars_u_se; }

    const std::vector<std::string>& getHistory() const { return history; }
    int getIterationsRun() const { return iterations_run; }
    double getLogLikelihood() const { return last_logL; }

    void setInitialVariances(const std::vector<double>& vu, double ve) {
        vars_u = vu;
        var_e = ve;
    }

    // Legacy setter
    void setInitialVars(double ve, double vu) {
        var_e = ve;
        if (!vars_u.empty()) vars_u[0] = vu;
        var_u_legacy = vu;
    }

    void setInitialVars(double ve, const std::vector<double>& vus) {
        var_e = ve;
        vars_u = vus;
        if (!vars_u.empty()) var_u_legacy = vars_u[0];
    }

    // Set sampler for component k
    void setSampler(int k, std::shared_ptr<PriorSampler> s) {
        if (k >= 0 && k < (int)samplers.size()) samplers[k] = s;
    }

    // Legacy setSampler (sets first)
    void setSampler(std::shared_ptr<PriorSampler> s) {
        if (!samplers.empty()) samplers[0] = s;
    }

    bool isConverged() const { return converged; }
    int getIterations() const { return iterations_run; }
    double getLastDiff() const { return last_diff; }

    // Get final solution (Beta + u)
    const Eigen::VectorXd& getSolution() const { return final_solution; }

    std::vector<std::string> history; // Log of iteration history

    // Static Utility: Solve VCE for Diagonalized System (GWAS Null Model)
    // Model: y ~ N(X*beta, var_g * D + var_e * I)
    static void solveDiagonal(const Eigen::VectorXd& D,
                              const Eigen::VectorXd& Uty,
                              const Eigen::MatrixXd& UtX,
                              double& out_var_g,
                              double& out_var_e,
                              double& out_se_g,
                              double& out_se_e,
                              Eigen::VectorXd* out_beta = nullptr,
                              Eigen::VectorXd* out_beta_se = nullptr);

    // Factory method
    static std::unique_ptr<VCESolver> create(const std::vector<GenRecord>& recs,
                                           const FixedDesignG& fd,
                                           const std::vector<RandomComponent>& comps,
                                           const Eigen::VectorXd& y,
                                           VCEConfig cfg);
};

class AI_REML : public VCESolver {
private:
    // Unified Builder
    std::unique_ptr<MMELHSBuilder> mme_builder;

    // Matrix-Free Support
    std::unique_ptr<ImplicitMME> implicit_mme;

    // RHS Vectors (Needed for solving)
    Eigen::VectorXd Xty;
    Eigen::VectorXd Zty;

    // Legacy members removed (XtX, XtZ, etc.) as we use MMELHSBuilder now.

    bool initialized = false;

    // Optimization: Pre-built LHS for current iteration
    // Builder manages LHS now.
    // Eigen::SparseMatrix<double> current_LHS;
    // Eigen::SparseMatrix<double, Eigen::RowMajor> current_LHS_row;
    Eigen::MatrixXd current_LHS_dense;

    // Warm start support
    Eigen::VectorXd last_solution;
    bool current_LHS_dense_valid = false;

    // Optimization: Fast LHS Update -> Moved to Builder
    // std::vector<double> lhs_base_values;
    // struct UpdateMap { int lhs_idx; double ainv_val; };
    // std::vector<UpdateMap> lhs_update_map;
    // bool fast_lhs_setup = false;

    // Cache for Direct Solver
    std::unique_ptr<Eigen::LDLT<Eigen::MatrixXd>> direct_solver;
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> sparse_direct_solver;
    bool lhs_built = false;
    bool sparse_symbolic_done = false;  ///< True after first symbolic factorization (analyzePattern)

    // Solver report (IASBLUP-style)
    SolverReport solver_report_;

    // Cached PCG components (avoid rebuilding every solve_lhs call)
    Eigen::SparseMatrix<double, Eigen::RowMajor> cached_A_row;
    std::shared_ptr<Preconditioner> cached_precond;
    bool pcg_cache_valid = false;
    int pcg_cache_p_fixed = -1;
    std::string pcg_cache_precond_type;

    // Cholesky factors for Ainv (G = Ainv^-1)
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> Ainv_sparse_chol;
    // Use LDLT for dense to handle indefinite/semi-definite better than LLT
    std::unique_ptr<Eigen::LDLT<Eigen::MatrixXd>> Ainv_dense_chol;
    bool Ainv_factorized = false;
    bool use_dense_Ainv = false;

public:
    // Explicit Constructors to allow incomplete types in header
    AI_REML(const std::vector<GenRecord>& recs_ref,
            const FixedDesignG& fd_ref,
            const std::vector<RandomComponent>& comps,
            const Eigen::VectorXd& y_ref,
            VCEConfig cfg = VCEConfig());

    AI_REML(const std::vector<GenRecord>& recs_ref,
            const FixedDesignG& fd_ref,
            const AbstractMatrix* Ainv_ref,
            const Eigen::VectorXd& y_ref,
            VCEConfig cfg = VCEConfig());

    ~AI_REML();

    void solve() override;

    // Calculate Standard Errors using stochastic AI matrix approximation
    void calculate_SE();

private:
    void initialize();
    void run_ai_iteration(int iter);

    // Build LHS: [X'X X'Z; Z'X Z'Z + Ainv*lambda]
    void build_lhs(const std::vector<double>& lambdas);

    // Solve Linear System using current_LHS
    Eigen::VectorXd solve_lhs(const Eigen::VectorXd& rhs, const Eigen::VectorXd& initial_guess = Eigen::VectorXd());

    // Operator P: returns P * v where P is the projection matrix
    Eigen::VectorXd ApplyP(const Eigen::VectorXd& v);

    // Wrapper for compatibility if needed, but we should use the above
    // Eigen::VectorXd solve_mme(double lambda, const Eigen::VectorXd& rhs_beta, const Eigen::VectorXd& rhs_u);

    // Estimate Tr(Qinv_k * C_uu_k)
    double estimate_trace_Qinv_Cuu(int component_idx, int n_samples);

    // Estimate Tr(C_uu)
    double estimate_trace_Cuu(int n_samples);

    // AI-REML Step
    // Returns true if step was successful
    bool run_aireml_step(int iter);

    // Exact Dense AI-REML Step
    bool run_dense_exact_step(int iter);

    // Dense Matrix Storage for Exact Mode
    Eigen::MatrixXd X_dense, Z_dense, A_dense, W_dense, ZAZt_dense;
    std::vector<int> dense_rec_to_anim;
    bool dense_initialized = false;
    void initialize_dense();

    // Per-instance cache for log|A| used by run_dense_exact_step.
    // Must NOT be static — a stale value would be reused across AI_REML instances.
    double log_det_A_member = 0.0;
    bool   log_det_A_member_valid = false;

    struct AI_Terms {
        double grad_u = 0;
        double grad_e = 0;
        double AI_uu = 0;
        double AI_ue = 0;
        double AI_ee = 0;
        // Multi-component support
        Eigen::VectorXd grad_vec;
        Eigen::MatrixXd AI_mat;
    };

    // Compute Gradient and AI Matrix stochastically
    AI_Terms compute_AI_terms(int n_samples);

    // Multi-component AI step (supports c >= 1 random components + residual).
    // Returns false on numerical failure; caller should fall back to EM step.
    bool run_aireml_step_multi(int iter);

    // Multi-component stochastic AI terms (gradient vector and AI matrix).
    AI_Terms compute_AI_terms_multi(int n_samples);

    // Multi-component Data-Driven AI matrix (Gilmour et al. 1995).
    // Uses current MME solution to form v_i = dV_i * P * y, then AI_ij = 0.5 * v_i' P v_j.
    // Requires (c+1) extra MME solves. Exact (non-stochastic).
    AI_Terms compute_AI_terms_multi_datadriven();

    // last_logL moved to VCESolver base class

    // Save last exact AI matrix for SE calculation
    Eigen::MatrixXd last_AI_mat;

    // Internal helper to calculate LogL for current variances
    double calc_logL_internal();

    // Fdiff-based AI matrix and Gradient calculation for Sparse MME
    AI_Terms compute_AI_terms_fdiff(int n_samples);

    // Helper to compute stochastic gradient with fixed random vectors
    Eigen::Vector2d compute_stochastic_gradient(double vu, double ve, const std::vector<Eigen::VectorXd>& z_samples);
};

class MCEM_REML : public VCESolver {
private:
    // Unified Builder
    std::unique_ptr<MMELHSBuilder> mme_builder;

    // RHS Vectors
    Eigen::VectorXd Xty;
    Eigen::VectorXd Zty;

    // Check if initialized
    bool initialized = false;

public:
    using VCESolver::VCESolver;

    void solve() override;

private:
    void initialize();
    void run_em_iteration(int iter);

    // Solve MME for a given RHS
    // Returns concatenated [beta; u]
    Eigen::VectorXd solve_mme(const Eigen::VectorXd& rhs_beta, const Eigen::VectorXd& rhs_u, const std::vector<double>& lambdas);
};

// Haseman-Elston Regression Implementation
class HE_Regression : public VCESolver {
private:
    std::unique_ptr<MMELHSBuilder> mme_builder;

    // Solvers for Inverse operation (G * z = Ainv^-1 * z)
    std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> sparse_solver;
    std::unique_ptr<Eigen::LDLT<Eigen::MatrixXd>> dense_solver;

    void run_dense_he();
    void run_randomized_he(); // RHE for large/sparse matrices

public:
    using VCESolver::VCESolver;
    void solve() override;
};

// Exact LMM Implementation (Null Model Optimization)
class Exact_LMM : public VCESolver {
private:
    std::unique_ptr<MMELHSBuilder> mme_builder;

    // Eigen Decomposition components
    Eigen::VectorXd D; // Eigenvalues of G
    Eigen::MatrixXd U; // Eigenvectors of G
    bool decomposed = false;

    // Optimization helpers
    double calc_neg_logL(double delta, const Eigen::MatrixXd& X_rot, const Eigen::VectorXd& y_rot, int n, int p);

public:
    using VCESolver::VCESolver;
    void solve() override;

    // Expose decomposition for GWAS reuse
    bool get_eigen_components(Eigen::VectorXd& out_D, Eigen::MatrixXd& out_U) const {
        if (!decomposed) return false;
        out_D = D;
        out_U = U;
        return true;
    }
};



} // namespace cosmic
