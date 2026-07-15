#include "blup_app.h"
#include "string_utils.h"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cosmic {

using namespace std;

// Helper to split comma-separated strings
static vector<string> split_comma(const string& s) {
    vector<string> r; string t; stringstream ss(s);
    while (getline(ss, t, ',')) r.push_back(trim_copy(t));
    return r;
}

static string lower_copy(string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static string detect_help_topic(const vector<string>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        const string& a = args[i];
        if (a.rfind("--help=", 0) == 0) return lower_copy(a.substr(7));
        if (a == "--help" || a == "-h") {
            if (i + 1 < args.size() && !args[i + 1].empty() && args[i + 1][0] != '-') {
                return lower_copy(args[i + 1]);
            }
            return "overview";
        }
    }
    return "overview";
}

static void print_help_overview() {
    std::cout << "Usage: cosmicblup [options]\n\n"
              << "Task-level help:\n"
              << "  cosmicblup --help blup\n"
              << "  cosmicblup --help relationship\n"
              << "  cosmicblup --help vce\n"
              << "  cosmicblup --help release\n\n"
              << "Task entry options:\n"
              << "  --blup                  GS-compatible BLUP task entry\n"
              << "  --make-matrix           GS-compatible relationship matrix export task\n"
              << "  --make-inv              GS-compatible relationship inverse export task\n"
              << "  --relationship <kind>   Relationship kind for export task: a, g, h\n"
              << "  --output-format <fmt>   Relationship export format: txt, tsv, bin\n\n"
              << "Core inputs:\n"
              << "  --pheno <file>          Phenotype file path\n"
              << "  --pheno-name <str>      Phenotype column name\n"
              << "  --ped <file>            Pedigree file\n"
              << "  --bfile <prefix>        PLINK BED/BIM/FAM prefix\n"
              << "  --pfile <prefix>        PLINK2 PGEN/PVAR/PSAM prefix\n"
              << "  --matrix <file>         External relationship matrix\n"
              << "  --matrix-id <file>      ID file for external relationship matrix\n"
              << "  --inv <file>            External inverse matrix\n"
              << "  --inv-id <file>         ID file for external inverse matrix\n"
              << "  --out <prefix>          Output prefix\n\n"
              << "Solver and VCE:\n"
              << "  --vce-mode <mode>       ai, em, mc, he, emai, hi, exact, vmatrix, stcg, fdiff\n"
              << "  --skip-vce              Skip VCE and solve with provided variances or defaults\n"
              << "  --var-a <float>         Additive variance for no-VCE solve\n"
              << "  --var-e <float>         Residual variance for no-VCE solve\n"
              << "  --solver <mode>         auto, dense, sparse, pcg\n"
              << "  --threads <int>         Worker thread count\n\n"
              << "Supported model families:\n"
              << "  PBLUP, GBLUP, ssGBLUP, repeatability, random regression, and multi-trait GBLUP\n\n"
              << "Docs:\n"
              << "  docs/USER_GUIDE_CN.md\n"
              << "  docs/CLI_REFERENCE.md\n"
              << "  docs/RELEASE_NOTES.md\n";
}

static void print_help_blup() {
    std::cout << "BLUP task help\n\n"
              << "Typical commands:\n"
              << "  cosmicblup --blup --model pblup --ped pedigree.txt --pheno pheno.txt --pheno-name trait --out run/pblup\n"
              << "  cosmicblup --blup --model gblup --bfile toy --pheno pheno.txt --pheno-name trait --out run/gblup\n"
              << "  cosmicblup --blup --model ssgblup --ped pedigree.txt --bfile toy --pheno pheno.txt --pheno-name trait --out run/ssgblup\n\n"
              << "Model selection:\n"
              << "  --model pblup|gblup|ssgblup|repeatability|rrm\n"
              << "  --single-trait | --multi-trait | --rrm\n\n"
              << "Input variants:\n"
              << "  Pedigree path: --ped\n"
              << "  Genotype path: --bfile / --pfile / --bgen\n"
              << "  External matrix path: --matrix + --matrix-id\n"
              << "  External inverse path: --inv + --inv-id\n\n"
              << "Variance control:\n"
              << "  --skip-vce / --no-vce\n"
              << "  --var-a <float> --var-e <float>\n"
              << "  --vce-mode ai|em|mc|he|emai|hi|exact|vmatrix|stcg|fdiff\n\n"
              << "Notes:\n"
              << "  Multi-trait currently targets dense GBLUP-oriented workflows.\n"
              << "  SBLUP and prediction remain separate task paths.\n";
}

static void print_help_relationship() {
    std::cout << "Relationship task help\n\n"
              << "Export tasks:\n"
              << "  cosmicblup --make-matrix --relationship a --ped pedigree.txt --output-format txt --out out/a\n"
              << "  cosmicblup --make-matrix --relationship g --bfile toy --output-format bin --out out/g\n"
              << "  cosmicblup --make-inv --relationship h --ped pedigree.txt --bfile toy --output-format txt --out out/hinv\n\n"
              << "Kinds:\n"
              << "  a  pedigree relationship matrix / inverse\n"
              << "  g  genomic relationship matrix / inverse\n"
              << "  h  hybrid relationship matrix / inverse for ssGBLUP-style workflows\n\n"
              << "Formats:\n"
              << "  txt / tsv  lower-triangle triplet text plus .id file\n"
              << "  bin        COSMIC_UPPER_PACKED binary with versioned header plus .id file\n\n"
              << "External reuse:\n"
              << "  cosmicblup --blup --model gblup --matrix out/g.GA.bin --matrix-id out/g.GA.id --pheno pheno.txt --pheno-name trait --skip-vce --var-a 0.3 --var-e 0.7 --out solve/external\n\n"
              << "Important options:\n"
              << "  --code-method / --xrm-algo\n"
              << "  --tau --omega\n"
              << "  --ridge / --ridge-value\n"
              << "  --no-tuneG\n";
}

static void print_help_vce() {
    std::cout << "VCE task help\n\n"
              << "Main controls:\n"
              << "  --vce-mode ai|em|mc|he|emai|hi|exact|vmatrix|stcg|fdiff\n"
              << "  --trace-mode auto|exact|fdiff|hutch|slq\n"
              << "  --vce-max-iter / --max-vce-iter\n"
              << "  --em-maxit / --em-iters\n"
              << "  --vce-samples\n"
              << "  --vce-tol\n\n"
              << "Skip-VCE path:\n"
              << "  --skip-vce or --no-vce\n"
              << "  --var-a <float> --var-e <float>\n"
              << "  Useful for smoke tests, fixed-variance comparisons, and GS-compatible batch runs.\n\n"
              << "Examples:\n"
              << "  cosmicblup --blup --model pblup --ped pedigree.txt --pheno pheno.txt --pheno-name trait --vce-mode ai --out run/ai\n"
              << "  cosmicblup --blup --model gblup --bfile toy --pheno pheno.txt --pheno-name trait --skip-vce --var-a 0.4 --var-e 0.6 --out run/fixed\n";
}

static void print_help_release() {
    std::cout << "Release and packaging help\n\n"
              << "Main script:\n"
              << "  bash scripts/package_linux.sh\n\n"
              << "Release gate:\n"
              << "  1. configure/build Release in build-release\n"
              << "  2. run full ctest --output-on-failure\n"
              << "  3. install into dist/CosmicBLUP-<version>-<platform>-<arch>\n"
              << "  4. run packaged --version, --help, and minimal PBLUP smoke\n"
              << "  5. write docs/package_smoke.log and docs/ldd_report.txt\n"
              << "  6. archive tar.gz and write matching .sha256\n\n"
              << "Release docs:\n"
              << "  docs/RELEASE.md\n"
              << "  docs/RELEASE_NOTES.md\n";
}

static void print_help_topic(const string& topic) {
    if (topic == "overview" || topic == "all") {
        print_help_overview();
    } else if (topic == "blup" || topic == "solve") {
        print_help_blup();
    } else if (topic == "relationship" || topic == "matrix" || topic == "xrm") {
        print_help_relationship();
    } else if (topic == "vce" || topic == "variance") {
        print_help_vce();
    } else if (topic == "release" || topic == "package" || topic == "packaging") {
        print_help_release();
    } else {
        std::cout << "Unknown help topic: " << topic << "\n\n";
        print_help_overview();
    }
}

Config parse_solver_cli(int argc, char** argv) {
    Config cfg;
    vector<string> args;
    for (int i = 0; i < argc; ++i) {
        cfg.full_command += string(argv[i]);
        if (i < argc - 1) cfg.full_command += " \\\n  ";
    }
    for (int i = 1; i < argc; ++i) args.push_back(string(argv[i]));

    bool show_help = false;
    for (const auto& a : args) {
        if (a == "--help" || a == "-h" || a.rfind("--help=", 0) == 0) show_help = true;
    }
    if (show_help || argc <= 1) {
        print_help_topic(detect_help_topic(args));
        exit(0);
    }

    for (size_t i = 0; i < args.size(); ++i) {
        string a = args[i];
        if (a == "--blup") { cfg.cli_present.insert("--blup"); cfg.blup_mode = true; }
        else if (a == "--single-trait") { cfg.cli_present.insert("--single-trait"); cfg.single_trait = true; cfg.multi_trait = false; cfg.rrm_model = false; }
        else if (a == "--multi-trait") { cfg.cli_present.insert("--multi-trait"); cfg.multi_trait = true; cfg.single_trait = false; cfg.rrm_model = false; }
        else if (a == "--rrm") { cfg.cli_present.insert("--rrm"); cfg.rrm_model = true; cfg.single_trait = false; cfg.multi_trait = false; }
        else if (a == "--rrm-order" && i + 1 < args.size()) { cfg.cli_present.insert("--rrm-order"); cfg.rrm_order = stoi(args[++i]); }
        else if (a == "--apy") { cfg.cli_present.insert("--apy"); cfg.use_apy = true; }
        else if (a == "--apy-core-size" && i + 1 < args.size()) { cfg.cli_present.insert("--apy-core-size"); cfg.apy_core_size = stoi(args[++i]); }
        else if (a == "--apy-core-file" && i + 1 < args.size()) { cfg.cli_present.insert("--apy-core-file"); cfg.apy_core_file = args[++i]; }
        else if (a == "--vce") { cfg.cli_present.insert("--vce"); cfg.run_vce = true; }
        else if ((a == "--pheno" || a == "--phe") && i + 1 < args.size()) { cfg.cli_present.insert("--pheno"); cfg.pheno_path = args[++i]; }
        else if ((a == "--pheno-pos" || a == "--mpheno") && i + 1 < args.size()) { cfg.cli_present.insert("--pheno-pos"); cfg.pheno_pos = stoi(args[++i]); }
        else if (a == "--pheno-name" && i + 1 < args.size()) { cfg.cli_present.insert("--pheno-name"); cfg.pheno_name = args[++i]; }
        else if (a == "--time-col" && i + 1 < args.size()) { cfg.cli_present.insert("--time-col"); cfg.time_col_name = args[++i]; }
        else if ((a == "--dcovar-names" || a == "--covar-name" || a == "--factor-names" || a == "--dcovar" || a == "--factors") && i + 1 < args.size()) {
            string val = args[++i];
            if (!val.empty() && isdigit(val[0]) && a != "--covar-name" && a != "--dcovar-names") {
                cfg.cli_present.insert("--dcovar");
                vector<string> parts = split_comma(val);
                for(const auto& p : parts) cfg.dcovar_cols.push_back(stoi(p));
            } else {
                cfg.cli_present.insert("--dcovar-names");
                cfg.dcovar_names = split_comma(val);
            }
        } else if ((a == "--qcovar-names" || a == "--qcovar-name" || a == "--covars" || a == "--qcovar") && i + 1 < args.size()) {
            string val = args[++i];
            if (!val.empty() && isdigit(val[0]) && a != "--qcovar-name" && a != "--qcovar-names") {
                cfg.cli_present.insert("--qcovar");
                vector<string> parts = split_comma(val);
                for(const auto& p : parts) cfg.qcovar_cols.push_back(stoi(p));
            } else {
                cfg.cli_present.insert("--qcovar-names");
                cfg.qcovar_names = split_comma(val);
            }
        } else if (a == "--rand" && i + 1 < args.size()) {
            string val = args[++i];
            if (!val.empty() && isdigit(val[0])) {
                cfg.cli_present.insert("--rand-cols");
                vector<string> parts = split_comma(val);
                for(const auto& p : parts) cfg.rand_cols.push_back(stoi(p));
            } else {
                cfg.cli_present.insert("--rand-names");
                cfg.rand_names = split_comma(val);
            }
        } else if ((a == "--xrminv" || a == "--xrm") && i + 1 < args.size()) { cfg.cli_present.insert("--xrminv"); cfg.xrminv_path = args[++i]; }
        else if (a == "--matrix" && i + 1 < args.size()) { cfg.cli_present.insert("--matrix"); cfg.matrix_path = args[++i]; }
        else if (a == "--matrix-id" && i + 1 < args.size()) { cfg.cli_present.insert("--matrix-id"); cfg.matrix_id_path = args[++i]; }
        else if (a == "--inv" && i + 1 < args.size()) { cfg.cli_present.insert("--inv"); cfg.inv_matrix_path = args[++i]; }
        else if (a == "--inv-A" && i + 1 < args.size()) { cfg.cli_present.insert("--inv-A"); cfg.inv_A_path = args[++i]; }
        else if (a == "--inv-G" && i + 1 < args.size()) { cfg.cli_present.insert("--inv-G"); cfg.inv_G_path = args[++i]; }
        else if (a == "--inv-A22" && i + 1 < args.size()) { cfg.cli_present.insert("--inv-A22"); cfg.inv_A22_path = args[++i]; }
        else if ((a == "--id" || a == "--inv-id") && i + 1 < args.size()) { cfg.cli_present.insert("--id"); cfg.inv_id_path = args[++i]; }
        else if (a == "--id-A" && i + 1 < args.size()) { cfg.cli_present.insert("--id-A"); cfg.id_A_path = args[++i]; }
        else if (a == "--id-G" && i + 1 < args.size()) { cfg.cli_present.insert("--id-G"); cfg.id_G_path = args[++i]; }
        else if (a == "--id-col" && i + 1 < args.size()) { cfg.cli_present.insert("--id-col"); cfg.id_col_name = args[++i]; }
        else if (a == "--vars" && i + 1 < args.size()) { cfg.cli_present.insert("--vars"); cfg.vars_path = args[++i]; }
        else if (a == "--out" && i + 1 < args.size()) { cfg.cli_present.insert("--out"); cfg.out_prefix = args[++i]; }
        else if (a == "--model" && i + 1 < args.size()) { cfg.cli_present.insert("--model"); cfg.model = args[++i]; }
        else if (a == "--relationship" && i + 1 < args.size()) {
            cfg.cli_present.insert("--relationship");
            cfg.relationship = args[++i];
            std::transform(cfg.relationship.begin(), cfg.relationship.end(), cfg.relationship.begin(),
                           [](unsigned char c){ return std::tolower(c); });
        }
        else if ((a == "--threads" || a == "--thread") && i + 1 < args.size()) { cfg.cli_present.insert("--threads"); cfg.threads = stoi(args[++i]); }
        else if (a == "--output-format" && i + 1 < args.size()) {
            cfg.cli_present.insert("--output-format");
            cfg.relationship_output_format = args[++i];
            std::transform(cfg.relationship_output_format.begin(), cfg.relationship_output_format.end(),
                           cfg.relationship_output_format.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            cfg.write_txt = (cfg.relationship_output_format != "bin");
        }
        else if (a == "--pcg-tol" && i + 1 < args.size()) { cfg.cli_present.insert("--pcg-tol"); cfg.pcg_tol = stod(args[++i]); }
        else if (a == "--pcg-maxit" && i + 1 < args.size()) { cfg.cli_present.insert("--pcg-maxit"); cfg.pcg_maxit = stoi(args[++i]); }
        else if (a == "--pcg-report" && i + 1 < args.size()) { cfg.cli_present.insert("--pcg-report"); cfg.pcg_report_every = stoi(args[++i]); }
        else if (a == "--pcg-precond" && i + 1 < args.size()) { cfg.cli_present.insert("--pcg-precond"); cfg.pcg_precond = args[++i]; }
        else if (a == "--no-se") { cfg.cli_present.insert("--no-se"); cfg.calc_se = false; }
        else if (a == "--snp-effect") { cfg.cli_present.insert("--snp-effect"); cfg.calc_snp_effect = true; }
        else if (a == "--fast") { cfg.cli_present.insert("--fast"); cfg.calc_se = false; cfg.matrix_free = true; }
        else if (a == "--matrix-free") { cfg.cli_present.insert("--matrix-free"); cfg.matrix_free = true; }
        else if (a == "--mmap") { cfg.cli_present.insert("--mmap"); cfg.use_mmap = true; }
        else if (a == "--version") { cfg.cli_present.insert("--version"); cfg.show_version = true; }
        else if (a == "--vce-max-iter" && i + 1 < args.size()) { cfg.cli_present.insert("--vce-max-iter"); cfg.vce_max_iter = stoi(args[++i]); }
        else if (a == "--max-vce-iter" && i + 1 < args.size()) { cfg.cli_present.insert("--max-vce-iter"); cfg.vce_max_iter = stoi(args[++i]); }
        else if (a == "--vce-samples" && i + 1 < args.size()) { cfg.cli_present.insert("--vce-samples"); cfg.vce_mc_samples = stoi(args[++i]); }
        else if (a == "--vce-tol" && i + 1 < args.size()) { cfg.cli_present.insert("--vce-tol"); cfg.vce_tol = stod(args[++i]); }
        else if ((a == "--vce-mode" || a == "--vc-method") && i + 1 < args.size()) {
            cfg.cli_present.insert("--vce-mode");
            string mode = args[++i];
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
            cfg.vce_mode = mode;
        }
        else if (a == "--vce-method" && i + 1 < args.size()) {
            cfg.cli_present.insert("--vce-method");
            string mode = args[++i];
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
            cfg.vce_mode = (mode == "hybrid") ? "emai" : mode;
        }
        else if (a == "--em-iters" && i + 1 < args.size()) { cfg.cli_present.insert("--em-iters"); cfg.em_maxit = stoi(args[++i]); }
        else if (a == "--skip-vce" || a == "--no-vce") { cfg.cli_present.insert("--skip-vce"); cfg.skip_vce = true; cfg.run_vce = false; }
        else if (a == "--var-a" && i + 1 < args.size()) { cfg.cli_present.insert("--var-a"); cfg.var_a = stod(args[++i]); }
        else if (a == "--var-e" && i + 1 < args.size()) { cfg.cli_present.insert("--var-e"); cfg.var_e = stod(args[++i]); }
        else if (a == "--force-dense-exact") { cfg.cli_present.insert("--force-dense-exact"); cfg.force_dense_exact = true; }
        else if (a == "--force-exact") { cfg.cli_present.insert("--force-exact"); cfg.force_exact = true; }
        else if (a == "--trace-mode" && i + 1 < args.size()) {
            cfg.cli_present.insert("--trace-mode");
            cfg.trace_mode = args[++i];
        }
        else if (a == "--solver" && i + 1 < args.size()) {
            cfg.cli_present.insert("--solver");
            cfg.solver_mode = args[++i];
        }
        else if (a == "--solver-report") { cfg.cli_present.insert("--solver-report"); cfg.print_report = true; }
        else if (a == "--alg" && i + 1 < args.size()) {
            cfg.cli_present.insert("--vce-mode");
            string mode = args[++i];
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c){ return std::tolower(c); });
            cfg.vce_mode = mode;
        }
        else if (a == "--dense-solver" && i + 1 < args.size()) {
            cfg.cli_present.insert("--dense-solver");
            string solver = args[++i];
            std::transform(solver.begin(), solver.end(), solver.begin(), [](unsigned char c){ return std::tolower(c); });
            cfg.dense_solver = solver;
        }
        else if ((a == "--ped" || a == "--pedigree") && i + 1 < args.size()) { cfg.cli_present.insert("--ped"); cfg.ped_path = args[++i]; }
        else if (a == "--bfile" && i + 1 < args.size()) { cfg.cli_present.insert("--bfile"); cfg.bfile_path = args[++i]; }
        else if (a == "--pfile" && i + 1 < args.size()) { cfg.cli_present.insert("--pfile"); cfg.pfile_path = args[++i]; }
        else if (a == "--bgen" && i + 1 < args.size()) { cfg.cli_present.insert("--bgen"); cfg.bgen_path = args[++i]; }
        else if (a == "--qc") { cfg.cli_present.insert("--qc"); cfg.run_qc = true; }
        else if (a == "--maf" && i + 1 < args.size()) { cfg.cli_present.insert("--maf"); cfg.qc_maf = std::stof(args[++i]); }
        else if (a == "--max-maf" && i + 1 < args.size()) { cfg.cli_present.insert("--max-maf"); cfg.qc_max_maf = std::stof(args[++i]); }
        else if (a == "--hwe" && i + 1 < args.size()) { cfg.cli_present.insert("--hwe"); cfg.qc_hwe = std::stof(args[++i]); }
        else if (a == "--geno" && i + 1 < args.size()) { cfg.cli_present.insert("--geno"); cfg.qc_geno = std::stof(args[++i]); }
        else if (a == "--mind" && i + 1 < args.size()) { cfg.cli_present.insert("--mind"); cfg.qc_mind = std::stof(args[++i]); }
        else if (a == "--tau" && i + 1 < args.size()) { cfg.cli_present.insert("--tau"); cfg.tau = stod(args[++i]); }
        else if (a == "--omega" && i + 1 < args.size()) { cfg.cli_present.insert("--omega"); cfg.omega = stod(args[++i]); }
        else if ((a == "--vc-priors" || a == "--var-priors" || a == "--var-prior") && i + 1 < args.size()) {
            cfg.cli_present.insert("--vc-priors");
            string val = args[++i];
            vector<string> parts = split_comma(val);
            for(const auto& p : parts) cfg.var_priors.push_back(stod(p));
        }
        else if ((a == "--covc-priors" || a == "--cov-priors") && i + 1 < args.size()) {
            cfg.cli_present.insert("--covc-priors");
            string val = args[++i];
            vector<string> parts = split_comma(val);
            for(const auto& p : parts) cfg.covc_priors.push_back(stod(p));
        }
        else if (a == "--ai-maxit" && i + 1 < args.size()) { cfg.cli_present.insert("--ai-maxit"); cfg.ai_maxit = std::stoi(args[++i]); }
        else if (a == "--em-maxit" && i + 1 < args.size()) { cfg.cli_present.insert("--em-maxit"); cfg.em_maxit = std::stoi(args[++i]); }
        else if (a == "--pcg" || a == "--PCG") { cfg.cli_present.insert(a); cfg.use_he_pcg = true; }
        else if ((a == "--pcg-num" || a == "--PCG-num") && i + 1 < args.size()) { cfg.cli_present.insert(a); cfg.pcg_num = std::stoi(args[++i]); }
        else if (a == "--pop-class" && i + 1 < args.size()) { cfg.cli_present.insert("--pop-class"); cfg.pop_class = args[++i]; cfg.pop_class_file = cfg.pop_class; }
        else if (a == "--snp-weight" && i + 1 < args.size()) { cfg.cli_present.insert("--snp-weight"); cfg.snp_weight_file = args[++i]; }
        else if (a == "--write-txt") { cfg.cli_present.insert("--write-txt"); cfg.write_txt = true; }
        else if (a == "--sblup") { cfg.cli_present.insert("--sblup"); cfg.sblup = true; }
        else if (a == "--sumstat" && i + 1 < args.size()) { cfg.cli_present.insert("--sumstat"); cfg.sumstat_path = args[++i]; }
        else if (a == "--h2" && i + 1 < args.size()) { cfg.cli_present.insert("--h2"); cfg.h2 = stod(args[++i]); }
        else if (a == "--window-num" && i + 1 < args.size()) { cfg.cli_present.insert("--window-num"); cfg.window_num = stoi(args[++i]); }
        else if (a == "--pred") { cfg.cli_present.insert("--pred"); cfg.pred = true; }
        else if (a == "--score" && i + 1 < args.size()) { cfg.cli_present.insert("--score"); cfg.score_path = args[++i]; }
        else if (a == "--allele-freq") { cfg.cli_present.insert("--allele-freq"); cfg.allele_freq = true; }
        else if (a == "--homo") { cfg.cli_present.insert("--homo"); cfg.homo = true; }
        else if (a == "--hete") { cfg.cli_present.insert("--hete"); cfg.hete = true; }
        else if (a == "--ibc") { cfg.cli_present.insert("--ibc"); cfg.ibc = true; }
        else if (a == "--rc") { cfg.cli_present.insert("--rc"); cfg.rc = true; }
        else if (a == "--pca") { cfg.cli_present.insert("--pca"); cfg.pca = true; }
        else if (a == "--npc" && i + 1 < args.size()) { cfg.cli_present.insert("--npc"); cfg.npc = stoi(args[++i]); }
        else if (a == "--make-xrm" || a == "--make-xrm-bin") { cfg.cli_present.insert("--make-xrm"); cfg.make_xrm = true; }
        else if (a == "--make-matrix") { cfg.cli_present.insert("--make-matrix"); cfg.make_xrm = true; }
        else if (a == "--make-inv") { cfg.cli_present.insert("--make-inv"); cfg.make_xrm = true; }
        else if ((a == "--ridge-value" || a == "--ridge-xrm" || a == "--ridge") && i + 1 < args.size()) { cfg.cli_present.insert("--ridge-value"); cfg.ridge_value = stod(args[++i]); }
        else if ((a == "--code-method" || a == "--xrm-algo") && i + 1 < args.size()) { cfg.cli_present.insert("--code-method"); cfg.code_method = stoi(args[++i]); }
        else if (a == "--alpha" && i + 1 < args.size()) { cfg.cli_present.insert("--alpha"); cfg.alpha = stod(args[++i]); }
        else if (a == "--no-tuneG") { cfg.cli_present.insert("--no-tuneG"); cfg.tuneG = false; }
        else if (a == "--step" && i + 1 < args.size()) { cfg.cli_present.insert("--step"); cfg.block_size = stoi(args[++i]); }
        else if (a == "--add") { cfg.cli_present.insert("--add"); cfg.add_effect = true; }
        else if (a == "--dom") { cfg.cli_present.insert("--dom"); cfg.dom_effect = true; }
        else if (a == "--epi") { cfg.cli_present.insert("--epi"); cfg.epi_effect = true; }
        else if (a == "--pe") { cfg.cli_present.insert("--pe"); cfg.pe_effect = true; }
        else if (a == "--mat") { cfg.cli_present.insert("--mat"); cfg.mat_effect = true; }
        else if (a == "--add-inv") { cfg.cli_present.insert("--add-inv"); cfg.add_inv = true; }
        else if (a == "--dom-inv") { cfg.cli_present.insert("--dom-inv"); cfg.dom_inv = true; }
        else if (a == "--epi-inv") { cfg.cli_present.insert("--epi-inv"); cfg.epi_inv = true; }
        else {
            throw std::invalid_argument("Unknown option or missing value: " + a);
        }
    }

    const bool has_genotype_input = !cfg.bfile_path.empty() || !cfg.pfile_path.empty() || !cfg.bgen_path.empty();
    const bool wants_relationship_task = cfg.cli_present.count("--make-matrix") || cfg.cli_present.count("--make-inv");
    if (wants_relationship_task) {
        cfg.make_xrm = true;
        if (cfg.relationship.empty()) {
            if (!cfg.ped_path.empty() && !has_genotype_input) cfg.relationship = "a";
            else if (cfg.ped_path.empty() && has_genotype_input) cfg.relationship = "g";
            else if (!cfg.ped_path.empty() && has_genotype_input) cfg.relationship = "h";
        }
        if (cfg.relationship == "g") {
            if (!has_genotype_input) throw std::invalid_argument("--relationship g requires --bfile, --pfile, or --bgen");
            if (cfg.cli_present.count("--make-inv")) cfg.add_inv = true;
        } else if (cfg.relationship == "a") {
            if (cfg.ped_path.empty()) throw std::invalid_argument("--relationship a requires --ped");
            if (cfg.cli_present.count("--make-inv")) cfg.add_inv = true;
        } else if (cfg.relationship == "h") {
            if (cfg.ped_path.empty() || !has_genotype_input) {
                throw std::invalid_argument("--relationship h requires --ped and a genotype input");
            }
        } else if (!cfg.relationship.empty()) {
            throw std::invalid_argument("Unsupported --relationship value '" + cfg.relationship + "'. Expected one of: a, g, h");
        }
    }

    if (!cfg.matrix_path.empty() && cfg.matrix_id_path.empty() && cfg.inv_id_path.empty()) {
        throw std::invalid_argument("External --matrix input requires --matrix-id <file> (or --inv-id as fallback).");
    }
    if (cfg.relationship_output_format != "txt" && cfg.relationship_output_format != "tsv" &&
        cfg.relationship_output_format != "bin") {
        throw std::invalid_argument("Unsupported --output-format '" + cfg.relationship_output_format + "'. Expected txt, tsv, or bin.");
    }
    if (cfg.var_a >= 0.0 || cfg.var_e >= 0.0) {
        if (cfg.var_a <= 0.0 || cfg.var_e <= 0.0) {
            throw std::invalid_argument("--var-a and --var-e must both be positive when either is specified.");
        }
        cfg.var_priors = {cfg.var_a, cfg.var_e};
    }

    if (!cfg.show_version && cfg.out_prefix.empty()) {
        throw std::invalid_argument("Missing required argument: --out <prefix>");
    }
    if (cfg.threads < 1) {
        throw std::invalid_argument("--threads must be a positive integer");
    }
    const std::set<std::string> valid_vce_modes = {
        "ai", "em", "mc", "he", "emai", "hi", "exact", "vmatrix", "stcg", "fdiff"
    };
    if (valid_vce_modes.find(cfg.vce_mode) == valid_vce_modes.end()) {
        throw std::invalid_argument(
            "Invalid --vce-mode '" + cfg.vce_mode +
            "'. Expected one of: ai, em, mc, he, emai, hi, exact, vmatrix, stcg, fdiff"
        );
    }
    return cfg;
}

} // namespace cosmic
