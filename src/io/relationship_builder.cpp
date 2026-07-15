#include "relationship_builder.h"
#include "phenotype_loader.h"
#include "logger.h"
#include "grm_builder.h"
#include "a_matrix.h"
#include "plink_reader.h"
#include "pgen_reader.h"
#include "bgen_reader.h"
#include "matrix_adapter.h"
#include "matrix_io.h"
#include "string_utils.h"
#include "matrix_inversion.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <set>

namespace cosmic {

static bool want_matrix_export(const Config& cfg) {
    return cfg.make_xrm &&
           (cfg.cli_present.count("--make-xrm") || cfg.cli_present.count("--make-matrix") ||
            !cfg.cli_present.count("--make-inv"));
}

static bool want_inverse_export(const Config& cfg) {
    return cfg.cli_present.count("--make-inv") || cfg.add_inv || cfg.dom_inv || cfg.epi_inv;
}

static std::vector<std::string> ordered_ids_from_idmap(const std::map<std::string, int>& idmap) {
    std::vector<std::string> ids(idmap.size());
    for (const auto& kv : idmap) {
        if (kv.second >= 1 && kv.second <= static_cast<int>(ids.size())) {
            ids[kv.second - 1] = kv.first;
        }
    }
    return ids;
}

static void write_named_matrix(const Config& cfg, const std::string& stem,
                               const Eigen::MatrixXd& matrix,
                               const std::vector<std::string>& ids) {
    if (cfg.write_txt) {
        write_matrix_txt(cfg.out_prefix + "." + stem + ".txt", matrix, ids);
        std::ofstream fid(cfg.out_prefix + "." + stem + ".id");
        for (const auto& id : ids) fid << id << "\n";
    } else {
        write_matrix_bin(cfg.out_prefix + "." + stem + ".bin",
                         cfg.out_prefix + "." + stem + ".id", matrix, ids);
    }
}

static Eigen::MatrixXd invert_inverse_matrix(AbstractMatrix* inv) {
    if (!inv) throw std::runtime_error("Cannot recover dense matrix from null inverse pointer.");
    if (auto* sparse = dynamic_cast<SparseMatrixAdapter*>(inv)) {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(sparse->getMatrix());
        if (solver.info() == Eigen::Success) {
            return solver.solve(Eigen::MatrixXd::Identity(sparse->rows(), sparse->cols()));
        }
    }

    Eigen::MatrixXd dense_inv = inv->toDense();
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    if (factorize_with_jitter(dense_inv, ldlt)) {
        return ldlt.solve(Eigen::MatrixXd::Identity(dense_inv.rows(), dense_inv.cols()));
    }
    return invert_psd_matrix(dense_inv, 0.0);
}

static std::string infer_external_relationship(const Config& cfg) {
    if (!cfg.relationship.empty()) return cfg.relationship;
    const bool has_ped = !cfg.ped_path.empty();
    const bool has_genotype = !cfg.bfile_path.empty() || !cfg.pfile_path.empty() || !cfg.bgen_path.empty();
    if (has_ped && has_genotype) return "h";
    if (has_genotype) return "g";
    return "a";
}

// --- Local helper: parse pedigree file ---
static int parse_ped(const std::string& path, std::vector<int>& dam, std::vector<int>& sire, std::vector<std::string>& ids) {
    std::ifstream fin(path);
    if(!fin) return -1;
    std::unordered_map<std::string,int> id2idx;
    std::vector<std::vector<std::string>> rows;
    std::string line;

    auto is_header = [](const std::string& s){
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return lower == "id" || lower == "animal" || lower == "ind";
    };

    while(std::getline(fin,line)){
        if(line.empty()) continue;
        std::istringstream ss(line);
        std::string a,b,c;
        ss>>a>>b>>c;
        if(a.empty()) continue;
        if(is_header(a)) continue;

        rows.push_back({a,b,c});
        if(id2idx.find(a)==id2idx.end()){ id2idx[a]=(int)ids.size(); ids.push_back(a);}
        if(!b.empty() && b!="NA" && b!="0" && b!="."){
            if(id2idx.find(b)==id2idx.end()){ id2idx[b]=(int)ids.size(); ids.push_back(b);}
        }
        if(!c.empty() && c!="NA" && c!="0" && c!="."){
            if(id2idx.find(c)==id2idx.end()){ id2idx[c]=(int)ids.size(); ids.push_back(c);}
        }
    }
    int n=(int)ids.size();
    dam.assign(n,-1);
    sire.assign(n,-1);
    for(auto& r: rows){
        int self=id2idx[r[0]];
        int d=-1,s=-1;
        auto itb=id2idx.find(r[1]); if(itb!=id2idx.end()) d=itb->second;
        auto itc=id2idx.find(r[2]); if(itc!=id2idx.end()) s=itc->second;
        dam[self]=d;
        sire[self]=s;
    }
    return n;
}

MatrixBuildResult MatrixBuilder::build(const Config& cfg) {
    MatrixBuildResult r;
    bool use_mmap = cfg.use_mmap;
    bool use_split = !cfg.inv_A_path.empty() || (!cfg.ped_path.empty() && (!cfg.bfile_path.empty() || !cfg.pfile_path.empty()));
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    };
    bool stcg_mode = to_lower(cfg.vce_mode) == "stcg";
    auto invs = detect_inv_files(cfg.inv_matrix_path.empty() ? cfg.inv_A_path : cfg.inv_matrix_path);

    // ================================================================
    // 1. Auto-build A-inverse from pedigree
    // ================================================================
    if (!cfg.ped_path.empty() && cfg.inv_A_path.empty() && cfg.inv_matrix_path.empty() && cfg.xrminv_path.empty()) {
        LOG_INFO("Loading pedigree file ...");
        std::vector<int> dam, sire;
        std::vector<std::string> a_ids;
        int n_ped = parse_ped(cfg.ped_path, dam, sire, a_ids);
        if (n_ped <= 0) throw std::runtime_error("Failed to read ped file or ped is empty.");
        LOG_INFO(n_ped << " unique individuals have been detected in file [" << cfg.ped_path << "].");
        r.global_dam_map = dam;

        LOG_INFO("Computing Ainv using Henderson's method (direct sparse inverse)...");
        Eigen::SparseMatrix<double> Ainv_sparse = genmatrix::invertA_henderson(dam, sire);
        r.Ainv = std::make_unique<SparseMatrixAdapter>(Ainv_sparse);

        r.idmap.clear();
        for (int i = 0; i < n_ped; ++i) r.idmap[a_ids[i]] = i + 1;
        r.inv_label = "Ainv(Henderson)";
        r.inv_file = "Auto-generated from " + cfg.ped_path;
        r.id_file_log = cfg.ped_path;

        if (cfg.ibc) {
            LOG_INFO("Computing inbreeding coefficients...");
            std::vector<int> d = dam, s = sire;
            genmatrix::complete_missing_parents_add_phantoms(d, s);
            std::vector<double> f, dii;
            genmatrix::ml(d, s, f, dii);
            std::string ibc_file = cfg.out_prefix + ".ibc";
            std::ofstream fout(ibc_file);
            if (fout) {
                fout << "ID\tF\n";
                for (size_t i = 0; i < a_ids.size(); ++i) {
                    fout << a_ids[i] << "\t" << f[i] << "\n";
                }
                LOG_INFO("Inbreeding coefficients saved to " << ibc_file);
            }
        }

        if (cfg.make_xrm) {
            std::vector<std::string> all_ids = a_ids;
            int n_total = Ainv_sparse.rows();
            for (int i = all_ids.size(); i < n_total; ++i) {
                all_ids.push_back("Phantom_" + std::to_string(i - a_ids.size() + 1));
            }
            if (want_matrix_export(cfg)) {
                LOG_INFO("Recovering A matrix for relationship export...");
                Eigen::MatrixXd A_dense = invert_inverse_matrix(r.Ainv.get());
                write_named_matrix(cfg, "PA", A_dense, all_ids);
                LOG_INFO("A matrix saved to " << cfg.out_prefix << ".PA." << (cfg.write_txt ? "txt" : "bin"));
            }
            if (want_inverse_export(cfg) || !want_matrix_export(cfg)) {
                LOG_INFO("Saving A matrix inverse...");
                write_named_matrix(cfg, "PAinv", Eigen::MatrixXd(Ainv_sparse), all_ids);
                LOG_INFO("A inverse matrix saved to " << cfg.out_prefix << ".PAinv." << (cfg.write_txt ? "txt" : "bin"));
            }
        }

        if (!use_split) {
            r.Qinv = std::make_unique<SparseMatrixAdapter>(Ainv_sparse);
        }
    }

    // ================================================================
    // 2. Auto-build G-inverse from genotype (BFILE/PFILE)
    // ================================================================
    if (!stcg_mode && (!cfg.bfile_path.empty() || !cfg.pfile_path.empty()) && cfg.inv_G_path.empty() && cfg.inv_matrix_path.empty() && cfg.xrminv_path.empty()) {
        LOG_INFO("Building G-inverse matrix from genotype...");
        GrmBuilder::Options gopts;
        gopts.thread_num = cfg.threads > 0 ? cfg.threads : 1;
        gopts.compute_homo_hete = cfg.homo || cfg.hete;
        gopts.block_size = cfg.block_size;
        gopts.algorithm = cfg.code_method;
        gopts.snp_weight_file = cfg.snp_weight_file;
        gopts.pop_class_file = cfg.pop_class_file;

        GrmBuilder builder(gopts);
        std::vector<std::string> g_ids;
        if (!cfg.bfile_path.empty()) {
            PlinkReader reader;
            reader.load(cfg.bfile_path);
            builder.compute(reader);
            auto fam = reader.getFamInfo();
            for (auto& f : fam) g_ids.push_back(f.iid.empty() ? f.fid : f.iid);
        } else {
            PgenReader reader;
            reader.open(cfg.pfile_path);
            builder.compute(reader);
            g_ids = reader.getSampleIds();
        }

        // --- make-xrm for G ---
        if (cfg.make_xrm) {
            const Eigen::MatrixXd& G_ref = builder.getGrm();
            if (want_matrix_export(cfg)) {
                LOG_INFO("Saving G matrix...");
                write_named_matrix(cfg, "GA", G_ref, g_ids);
                LOG_INFO("G matrix saved to " << cfg.out_prefix << ".GA." << (cfg.write_txt ? "txt" : "bin"));
            }
            if (cfg.add_inv || !want_matrix_export(cfg)) {
                LOG_INFO("Inverting G matrix for --add-inv...");
                Eigen::MatrixXd Ginv = invert_psd_matrix(builder.getGrm(), cfg.ridge_value);
                write_named_matrix(cfg, "GAinv", Ginv, g_ids);
                LOG_INFO("G inverse matrix saved to " << cfg.out_prefix << ".GAinv." << (cfg.write_txt ? "txt" : "bin"));
            }
        }

        // --- Dominance effect ---
        if (cfg.dom_effect) {
            LOG_INFO("Building Dominance Relationship Matrix (GD)...");
            GrmBuilder::Options dom_opts = gopts;
            dom_opts.algorithm = 4; // Vitezica
            GrmBuilder dom_builder(dom_opts);
            if (!cfg.bfile_path.empty()) {
                PlinkReader reader;
                reader.load(cfg.bfile_path);
                dom_builder.compute(reader);
            } else {
                PgenReader reader;
                reader.open(cfg.pfile_path);
                dom_builder.compute(reader);
            }
            const Eigen::MatrixXd& GD_ref = dom_builder.getGrm();
            if (cfg.make_xrm) {
                if (cfg.write_txt) {
                    LOG_INFO("Saving Dominance matrix (TXT format)...");
                    write_matrix_txt(cfg.out_prefix + ".GD.txt", GD_ref, g_ids);
                } else {
                    LOG_INFO("Saving Dominance matrix (BIN format)...");
                    write_matrix_bin(cfg.out_prefix + ".GD.bin", cfg.out_prefix + ".GD.id", GD_ref, g_ids);
                }
            }
            if (cfg.dom_inv || (!cfg.make_xrm && (cfg.run_vce || !cfg.pheno_path.empty()))) {
                LOG_INFO("Inverting Dominance matrix...");
                Eigen::MatrixXd GDinv = invert_psd_matrix(GD_ref, cfg.ridge_value);
                if (cfg.make_xrm && cfg.dom_inv) {
                    if (cfg.write_txt) {
                        write_matrix_txt(cfg.out_prefix + ".GDinv.txt", GDinv, g_ids);
                    } else {
                        write_matrix_bin(cfg.out_prefix + ".GDinv.bin", cfg.out_prefix + ".GDinv.id", GDinv, g_ids);
                    }
                }
                r.GDinv = std::make_unique<DenseMatrixAdapter>(GDinv);
            }
        }

        // --- Epistatic effect ---
        if (cfg.epi_effect) {
            LOG_INFO("Building Epistatic Relationship Matrix (GE = GA # GA)...");
            const Eigen::MatrixXd& GA_ref = builder.getGrm();
            Eigen::MatrixXd GE_ref = GA_ref.cwiseProduct(GA_ref);
            if (cfg.make_xrm) {
                if (cfg.write_txt) {
                    LOG_INFO("Saving Epistatic matrix (TXT format)...");
                    write_matrix_txt(cfg.out_prefix + ".GE.txt", GE_ref, g_ids);
                } else {
                    LOG_INFO("Saving Epistatic matrix (BIN format)...");
                    write_matrix_bin(cfg.out_prefix + ".GE.bin", cfg.out_prefix + ".GE.id", GE_ref, g_ids);
                }
            }
            if (cfg.epi_inv || (!cfg.make_xrm && (cfg.run_vce || !cfg.pheno_path.empty()))) {
                LOG_INFO("Inverting Epistatic matrix...");
                Eigen::MatrixXd GEinv = invert_psd_matrix(GE_ref, cfg.ridge_value);
                if (cfg.make_xrm && cfg.epi_inv) {
                    if (cfg.write_txt) {
                        write_matrix_txt(cfg.out_prefix + ".GEinv.txt", GEinv, g_ids);
                    } else {
                        write_matrix_bin(cfg.out_prefix + ".GEinv.bin", cfg.out_prefix + ".GEinv.id", GEinv, g_ids);
                    }
                }
                r.GEinv = std::make_unique<DenseMatrixAdapter>(GEinv);
            }
        }

        // --- RC (relationship coefficients) ---
        if (cfg.rc) {
            std::string rc_file = cfg.out_prefix + ".rc";
            std::ofstream fout(rc_file);
            if (fout) {
                fout << "id1\tid2\trc\n";
                const auto& G_ref = builder.getGrm();
                for (size_t i = 0; i < g_ids.size(); ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        fout << g_ids[i] << "\t" << g_ids[j] << "\t" << G_ref(i, j) << "\n";
                    }
                }
                LOG_INFO("Relationship coefficients saved to " << rc_file);
            }
        }

        // --- PCA ---
        if (cfg.pca) {
            LOG_INFO("Computing PCA...");
            const Eigen::MatrixXd& G_ref = builder.getGrm();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G_ref);
            auto eval = es.eigenvalues();
            auto evec = es.eigenvectors();
            std::string pca_file = cfg.out_prefix + ".pca";
            std::ofstream fout(pca_file);
            if (fout) {
                int n = G_ref.rows();
                int npc = std::min(cfg.npc, n);
                fout << "ID";
                for (int i = 1; i <= npc; ++i) fout << "\tPC" << i;
                fout << "\n";
                for (int i = 0; i < n; ++i) {
                    fout << g_ids[i];
                    for (int j = 0; j < npc; ++j) {
                        fout << "\t" << evec(i, n - 1 - j);
                    }
                    fout << "\n";
                }
                LOG_INFO("PCA saved to " << pca_file);
            }
        }

        // --- Allele frequencies ---
        if (cfg.allele_freq) {
            LOG_INFO("Saving allele frequencies...");
            std::string freq_file = cfg.out_prefix + ".frq";
            std::ofstream fout(freq_file);
            if (fout) {
                if (!cfg.bfile_path.empty()) {
                    PlinkReader r_plink;
                    r_plink.load(cfg.bfile_path);
                    const auto& bim = r_plink.getBimInfo();
                    const auto& freqs = builder.getAlleleFreqs();
                    fout << "CHR\tSNP\tA1\tA2\tFREQ_A1\n";
                    for (size_t i = 0; i < freqs.size(); ++i) {
                        if (freqs[i] >= 0.0f) {
                            fout << bim[i].chrom << "\t" << bim[i].id << "\t"
                                 << bim[i].alt << "\t" << bim[i].ref << "\t" << freqs[i] << "\n";
                        }
                    }
                } else if (!cfg.pfile_path.empty()) {
                    PgenReader r_pgen;
                    r_pgen.open(cfg.pfile_path);
                    const auto& pvar = r_pgen.getVariants();
                    const auto& freqs = builder.getAlleleFreqs();
                    fout << "CHR\tSNP\tA1\tA2\tFREQ_A1\n";
                    for (size_t i = 0; i < freqs.size(); ++i) {
                        if (freqs[i] >= 0.0f) {
                            fout << pvar[i].chrom << "\t" << pvar[i].id << "\t"
                                 << pvar[i].ref << "\t" << pvar[i].alt << "\t" << freqs[i] << "\n";
                        }
                    }
                }
                LOG_INFO("Allele frequencies saved to " << freq_file);
            }
        }

        // --- Homozygosity / Heterozygosity ---
        if (cfg.homo || cfg.hete) {
            long long m_total = builder.getTotalSnps();
            if (m_total > 0) {
                if (cfg.homo) {
                    std::string homo_file = cfg.out_prefix + ".homo";
                    std::ofstream fout(homo_file);
                    if (fout) {
                        const auto& counts = builder.getHomoCounts();
                        fout << "ID\tHOMO_COUNT\tTOTAL_VALID\tHOMO_RATE\n";
                        for (size_t i = 0; i < g_ids.size(); ++i) {
                            int valid = m_total - builder.getMissingCounts()[i];
                            double rate = valid > 0 ? (double)counts[i] / valid : 0.0;
                            fout << g_ids[i] << "\t" << counts[i] << "\t" << valid << "\t" << rate << "\n";
                        }
                        LOG_INFO("Homozygosity rates saved to " << homo_file);
                    }
                }
                if (cfg.hete) {
                    std::string hete_file = cfg.out_prefix + ".hete";
                    std::ofstream fout(hete_file);
                    if (fout) {
                        const auto& counts = builder.getHeteCounts();
                        fout << "ID\tHETE_COUNT\tTOTAL_VALID\tHETE_RATE\n";
                        for (size_t i = 0; i < g_ids.size(); ++i) {
                            int valid = m_total - builder.getMissingCounts()[i];
                            double rate = valid > 0 ? (double)counts[i] / valid : 0.0;
                            fout << g_ids[i] << "\t" << counts[i] << "\t" << valid << "\t" << rate << "\n";
                        }
                        LOG_INFO("Heterozygosity rates saved to " << hete_file);
                    }
                }
            }
        }

        // --- Build G-inverse for solver ---
        if (!use_split) {
            LOG_INFO("Inverting G matrix with Bending/Ridge...");
            if (cfg.use_apy) {
                LOG_INFO("=== APY (Algorithm for Proven and Young) Mode Enabled ===");
                std::vector<int> core_status;
                const Eigen::MatrixXd& G_ref = builder.getGrm();
                Eigen::MatrixXd G = G_ref;
                int n_geno = G.rows();
                core_status.assign(n_geno, 0);

                if (!cfg.apy_core_file.empty()) {
                    // Core defined by file
                    std::vector<std::string> core_ids;
                    std::ifstream core_file(cfg.apy_core_file);
                    if (!core_file.is_open()) {
                        throw std::runtime_error("Failed to open APY core file: " + cfg.apy_core_file);
                    }
                    std::string line;
                    while (std::getline(core_file, line)) {
                        std::string id = cosmic::trim_copy(line);
                        if (!id.empty()) core_ids.push_back(id);
                    }
                    LOG_INFO("Read " << core_ids.size() << " core animal IDs from " << cfg.apy_core_file);
                    if (core_ids.empty()) throw std::runtime_error("APY core file is empty.");
                    std::set<std::string> core_set(core_ids.begin(), core_ids.end());
                    int found_cores = 0;
                    for (int i = 0; i < n_geno; ++i) {
                        if (core_set.count(g_ids[i])) { core_status[i] = 1; found_cores++; }
                    }
                    LOG_INFO("Matched " << found_cores << " core animals in the genotype data (out of " << n_geno << " total).");
                    if (found_cores == 0) throw std::runtime_error("No core animals found in the genotype data. Cannot proceed with APY.");
                } else if (cfg.apy_core_size > 0) {
                    // Core defined by first N individuals
                    int ncore = std::min(cfg.apy_core_size, n_geno);
                    for (int i = 0; i < ncore; ++i) core_status[i] = 1;
                    LOG_INFO("Using first " << ncore << " individuals as core (out of " << n_geno << " total).");
                } else {
                    throw std::runtime_error("APY mode requires either --apy-core-file or --apy-core-size.");
                }

                LOG_INFO("Building APY G-inverse approximation...");
                Eigen::MatrixXd Ginv = cosmic::grm::compute_explicit_apy_inverse(G, core_status, cfg.ridge_value);
                r.Ginv = std::make_unique<DenseMatrixAdapter>(Ginv);
                r.Qinv = std::make_unique<DenseMatrixAdapter>(Ginv);
                LOG_INFO("APY G-inverse constructed successfully.");
            } else {
                Eigen::MatrixXd Ginv = invert_psd_matrix(builder.getGrm(), cfg.ridge_value);
                r.Ginv = std::make_unique<DenseMatrixAdapter>(Ginv);
                r.Qinv = std::make_unique<DenseMatrixAdapter>(Ginv);
            }

            r.idmap.clear();
            for (size_t i = 0; i < g_ids.size(); ++i) r.idmap[g_ids[i]] = i + 1;
            r.inv_label = "Ginv(VanRaden)";
            r.inv_file = "Auto-generated from Genotype";
            r.id_file_log = cfg.bfile_path.empty() ? cfg.pfile_path : cfg.bfile_path;
        } else {
            // ================================================================
            // 2b. ssGBLUP split mode: build H-inverse from A + G
            // ================================================================
            r.use_split = true;
            // Map A IDs to G IDs
            r.genotyped_map.resize(g_ids.size(), -1);
            for (size_t i = 0; i < g_ids.size(); ++i) {
                if (r.idmap.count(g_ids[i])) {
                    r.genotyped_map[i] = r.idmap[g_ids[i]] - 1;
                }
            }

            int n_g = g_ids.size();
            Eigen::MatrixXd A22(n_g, n_g);
            Eigen::MatrixXd A_dense;
            LOG_INFO("Extracting A22 block...");
            SparseMatrixAdapter* sparse_ainv = dynamic_cast<SparseMatrixAdapter*>(r.Ainv.get());
            if (sparse_ainv) {
                Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver(sparse_ainv->getMatrix());
                for (int j = 0; j < n_g; ++j) {
                    if (r.genotyped_map[j] >= 0) {
                        Eigen::VectorXd ej = Eigen::VectorXd::Zero(sparse_ainv->rows());
                        ej(r.genotyped_map[j]) = 1.0;
                        Eigen::VectorXd xj = solver.solve(ej);
                        for (int i = 0; i < n_g; ++i) {
                            if (r.genotyped_map[i] >= 0) {
                                A22(i, j) = xj(r.genotyped_map[i]);
                            } else {
                                A22(i, j) = (i == j) ? 1.0 : 0.0;
                            }
                        }
                    } else {
                        for (int i = 0; i < n_g; ++i) A22(i, j) = (i == j) ? 1.0 : 0.0;
                    }
                }
            } else {
                Eigen::MatrixXd Ainv_dense_mat = r.Ainv->toDense();
                Eigen::LLT<Eigen::MatrixXd> lltAinv(Ainv_dense_mat);
                if (lltAinv.info() == Eigen::Success) {
                    A_dense = lltAinv.solve(Eigen::MatrixXd::Identity(Ainv_dense_mat.rows(), Ainv_dense_mat.cols()));
                } else {
                    A_dense = Ainv_dense_mat.inverse();
                }
                for (int i = 0; i < n_g; ++i) {
                    for (int j = 0; j < n_g; ++j) {
                        if (r.genotyped_map[i] >= 0 && r.genotyped_map[j] >= 0) {
                            A22(i, j) = A_dense(r.genotyped_map[i], r.genotyped_map[j]);
                        } else {
                            A22(i, j) = (i == j) ? 1.0 : 0.0;
                        }
                    }
                }
            }

            const Eigen::MatrixXd& G_ref = builder.getGrm();
            Eigen::MatrixXd G = G_ref;

            if (cfg.tuneG) {
                LOG_INFO("Aligning G to A22...");
                double sum_G_diag = 0.0, sum_A_diag = 0.0;
                double sum_G_off = 0.0, sum_A_off = 0.0;
                for (int i = 0; i < n_g; ++i) {
                    sum_G_diag += G(i, i); sum_A_diag += A22(i, i);
                    for (int j = 0; j < n_g; ++j) {
                        if (i != j) { sum_G_off += G(i, j); sum_A_off += A22(i, j); }
                    }
                }
                double mean_G_diag = sum_G_diag / n_g;
                double mean_A_diag = sum_A_diag / n_g;
                double n_off = (double)n_g * (n_g - 1);
                double mean_G_off = n_off > 0 ? sum_G_off / n_off : 0.0;
                double mean_A_off = n_off > 0 ? sum_A_off / n_off : 0.0;
                double b = 1.0;
                if (std::abs(mean_G_diag - mean_G_off) > 1e-12) {
                    b = (mean_A_diag - mean_A_off) / (mean_G_diag - mean_G_off);
                }
                double a = mean_A_diag - b * mean_G_diag;
                LOG_INFO("Mean of diagonal and Off-diagonal of A22: " << mean_A_diag << " " << mean_A_off);
                LOG_INFO("Mean of diagonal and Off-diagonal of G: " << mean_G_diag << " " << mean_G_off);
                LOG_INFO("Alignment factors: a = " << a << ", b = " << b);
                G = b * G; G.array() += a;
            } else {
                LOG_INFO("Skipping G matrix alignment to A22 (tuneG=false).");
            }

            double p_blend = cfg.alpha;
            LOG_INFO("Blending G with A22 (alpha = " << p_blend << ")...");
            G = (1.0 - p_blend) * G + p_blend * A22;

            Eigen::MatrixXd Ginv = invert_psd_matrix(G, cfg.ridge_value);

            LOG_INFO("Inverting A22 block...");
            if (cfg.ridge_value > 0.0) A22.diagonal().array() += cfg.ridge_value;
            Eigen::MatrixXd A22inv;
            Eigen::LLT<Eigen::MatrixXd> lltA22(A22);
            if (lltA22.info() == Eigen::Success) {
                A22inv = lltA22.solve(Eigen::MatrixXd::Identity(n_g, n_g));
            } else {
                A22inv = A22.inverse();
            }

            Ginv *= cfg.tau;
            A22inv *= cfg.omega;

            if (cfg.make_xrm && cfg.relationship == "h") {
                std::vector<std::string> all_ids = ordered_ids_from_idmap(r.idmap);
                if (all_ids.empty()) {
                    throw std::runtime_error("Failed to recover ordered IDs for H matrix export.");
                }

                if (A_dense.size() == 0) {
                    A_dense = invert_inverse_matrix(r.Ainv.get());
                }

                Eigen::MatrixXd H = A_dense;
                for (int i = 0; i < n_g; ++i) {
                    const int row_idx = r.genotyped_map[i];
                    if (row_idx < 0) continue;
                    for (int j = 0; j < n_g; ++j) {
                        const int col_idx = r.genotyped_map[j];
                        if (col_idx < 0) continue;
                        H(row_idx, col_idx) = G(i, j);
                    }
                }

                Eigen::MatrixXd Hinv = r.Ainv->toDense();
                for (int i = 0; i < n_g; ++i) {
                    const int row_idx = r.genotyped_map[i];
                    if (row_idx < 0) continue;
                    for (int j = 0; j < n_g; ++j) {
                        const int col_idx = r.genotyped_map[j];
                        if (col_idx < 0) continue;
                        Hinv(row_idx, col_idx) += Ginv(i, j) - A22inv(i, j);
                    }
                }

                if (want_matrix_export(cfg)) {
                    LOG_INFO("Saving H matrix...");
                    write_named_matrix(cfg, "HA", H, all_ids);
                }
                if (want_inverse_export(cfg) || !want_matrix_export(cfg)) {
                    LOG_INFO("Saving H inverse matrix...");
                    write_named_matrix(cfg, "HAinv", Hinv, all_ids);
                }
            }

            r.Ginv = std::make_unique<DenseMatrixAdapter>(Ginv);
            r.A22inv = std::make_unique<DenseMatrixAdapter>(A22inv);
            r.Qinv.reset(new SplitMatrixAdapter(r.Ainv.get(), r.Ginv.get(), r.A22inv.get(), r.genotyped_map));
            r.inv_label = "Hinv(ssGBLUP_Aligned)";
            r.inv_file = "Auto-generated Hinv";
        }
    }

    // ================================================================
    // 3. Load split matrices from files
    // ================================================================
    if (use_split && !r.Qinv) {
        if (cfg.inv_A_path.empty() || cfg.inv_G_path.empty() || cfg.inv_A22_path.empty()) {
            throw std::runtime_error("Split mode requires --inv-A, --inv-G, and --inv-A22 all to be specified.");
        }
        LOG_INFO("Loading split matrices...");
        if (use_mmap) LOG_INFO("Using Memory Mapping (mmap) for matrices...");

        r.Ainv.reset(readInvMatrixAbstract(cfg.inv_A_path, use_mmap));
        r.Ginv.reset(readInvMatrixAbstract(cfg.inv_G_path, use_mmap));
        r.A22inv.reset(readInvMatrixAbstract(cfg.inv_A22_path, use_mmap));

        if (cfg.id_A_path.empty()) throw std::runtime_error("Split mode requires --id-A (global ID list).");
        r.idmap = readIdList(cfg.id_A_path);
        if (static_cast<int>(r.idmap.size()) != r.Ainv->rows()) {
            throw std::runtime_error("Split mode mismatch: --id-A count (" + std::to_string(r.idmap.size()) +
                                     ") does not match --inv-A dimension (" + std::to_string(r.Ainv->rows()) + ").");
        }

        std::string id_g_file = cfg.id_G_path;
        if (id_g_file.empty()) {
            auto p = detect_inv_files(cfg.inv_G_path);
            if (!p.second.empty()) id_g_file = p.second;
            else throw std::runtime_error("Split mode requires --id-G (genotyped ID list) or auto-detection failed.");
        }
        std::map<std::string, int> idmap__g = readIdList(id_g_file);
        if (static_cast<int>(idmap__g.size()) != r.Ginv->rows()) {
            throw std::runtime_error("Split mode mismatch: genotyped ID count (" + std::to_string(idmap__g.size()) +
                                     ") does not match --inv-G dimension (" + std::to_string(r.Ginv->rows()) + ").");
        }
        if (r.Ginv->rows() != r.Ginv->cols()) {
            throw std::runtime_error("Split mode requires square --inv-G matrix.");
        }
        if (r.A22inv->rows() != r.A22inv->cols()) {
            throw std::runtime_error("Split mode requires square --inv-A22 matrix.");
        }
        if (r.Ginv->rows() != r.A22inv->rows()) {
            throw std::runtime_error("Split mode mismatch: --inv-G dimension (" + std::to_string(r.Ginv->rows()) +
                                     ") must equal --inv-A22 dimension (" + std::to_string(r.A22inv->rows()) + ").");
        }
        std::vector<std::string> g_ids(idmap__g.size());
        for (auto& kv : idmap__g) if (kv.second >= 1 && kv.second <= (int)g_ids.size()) g_ids[kv.second - 1] = kv.first;

        r.genotyped_map.resize(g_ids.size(), -1);
        for (size_t i = 0; i < g_ids.size(); ++i) {
            if (r.idmap.count(g_ids[i])) r.genotyped_map[i] = r.idmap[g_ids[i]] - 1;
        }
        for (size_t i = 0; i < g_ids.size(); ++i) {
            if (g_ids[i].empty()) {
                throw std::runtime_error("Split mode requires contiguous --id-G ordering without blank IDs.");
            }
            if (r.genotyped_map[i] < 0) {
                throw std::runtime_error("Split mode mismatch: genotyped ID '" + g_ids[i] +
                                         "' from --id-G is missing from --id-A.");
            }
        }

        r.Qinv.reset(new SplitMatrixAdapter(r.Ainv.get(), r.Ginv.get(), r.A22inv.get(), r.genotyped_map));
        r.inv_label = make_inv_label(cfg.inv_A_path) + "+split";
        r.inv_file = cfg.inv_A_path;
        r.id_file_log = cfg.id_A_path + " ; " + id_g_file;
        r.use_split = true;
    }

    // ================================================================
    // 4. Load single inverse matrix from file
    // ================================================================
    else if (!cfg.matrix_path.empty()) {
        std::string id_file = cfg.matrix_id_path.empty() ? cfg.inv_id_path : cfg.matrix_id_path;
        if (id_file.empty()) {
            throw std::runtime_error("External --matrix input requires --matrix-id <file> (or --inv-id as fallback).");
        }

        LOG_INFO("Loading external relationship matrix from " << cfg.matrix_path << " ...");
        Eigen::SparseMatrix<double> matrix_sparse = readInvMatrix(cfg.matrix_path, 1e-9, false);
        Eigen::MatrixXd matrix_dense(matrix_sparse);
        Eigen::MatrixXd inv_dense = invert_psd_matrix(matrix_dense, cfg.ridge_value);

        const std::string rel = infer_external_relationship(cfg);
        r.Qinv = std::make_unique<DenseMatrixAdapter>(inv_dense);
        r.idmap = readIdList(id_file);
        r.inv_label = (rel == "h" ? "Hinv" : (rel == "g" ? "Ginv" : "Ainv")) + std::string("(external matrix)");
        r.inv_file = cfg.matrix_path;
        r.id_file_log = id_file;

        if (rel == "g") {
            r.Ginv = std::make_unique<DenseMatrixAdapter>(inv_dense);
        } else if (rel == "a") {
            r.Ainv = std::make_unique<DenseMatrixAdapter>(inv_dense);
        }
    }

    else if (!cfg.inv_matrix_path.empty() || (!cfg.inv_A_path.empty() && !cfg.inv_G_path.empty() && !cfg.inv_A22_path.empty())) {
        if (invs.first.empty()) throw std::runtime_error("Could not find inverse matrix file. Use --inv or --inv-A/--inv-G/--inv-A22.");
        std::string inv_file = invs.first;
        std::string id_file = invs.second;
        if (id_file.empty()) throw std::runtime_error("Could not find ID file for " + inv_file + ". Use --id.");

        if (use_mmap) LOG_INFO("Using Memory Mapping (mmap) for matrix...");
        r.Qinv.reset(readInvMatrixAbstract(inv_file, use_mmap));
        r.idmap = readIdList(id_file);
        r.inv_label = make_inv_label(inv_file);
        r.inv_file = inv_file;
    }

    if (r.id_file_log.empty()) {
        r.id_file_log = use_split ? cfg.id_A_path : invs.second;
    }

    // ================================================================
    // 5. Log matrix info
    // ================================================================
    if (r.Qinv) {
        std::string m = auto_model_from_inv(r.inv_label, cfg.model);
        std::string model_tag;
        if (m == "ssgblup") model_tag = "HA";
        else if (m == "gblup") model_tag = "GA";
        else model_tag = "PA";
        LOG_INFO("Derive " << model_tag << " matrix ...\n");
        LOG_INFO("Total " << r.Qinv->rows() << " individuals will be predicted.\n");
    }

    // ================================================================
    // 6. STCG mode
    // ================================================================
    if (stcg_mode) {
        r.stcg_mode = true;
        if (use_split || !cfg.ped_path.empty()) {
            throw std::runtime_error("STCG mode currently supports GBLUP repeatability without pedigree (--ped).");
        }
        if (cfg.bfile_path.empty() && cfg.pfile_path.empty() && cfg.bgen_path.empty()) {
            throw std::runtime_error("STCG mode requires --bfile, --pfile, or --bgen to access genotype matrix for implicit GRM operations.");
        }

        std::vector<std::string> sample_ids;
        std::string source_label;
        if (!cfg.bfile_path.empty()) {
            auto reader = std::make_unique<PlinkReader>();
            reader->load(cfg.bfile_path);
            reader->computeStats(cfg.threads > 0 ? cfg.threads : 1);
            const auto& fam = reader->getFamInfo();
            sample_ids.reserve(fam.size());
            for (const auto& row : fam) {
                sample_ids.push_back(row.iid.empty() ? row.fid : row.iid);
            }
            source_label = cfg.bfile_path;
            r.stcg_genotype = std::move(reader);
        } else if (!cfg.pfile_path.empty()) {
            auto reader = std::make_unique<PgenReader>();
            reader->open(cfg.pfile_path);
            reader->computeStats(cfg.threads > 0 ? cfg.threads : 1);
            sample_ids = reader->getSampleIds();
            source_label = cfg.pfile_path;
            r.stcg_genotype = std::move(reader);
        } else {
            auto reader = std::make_unique<BgenReader>();
            reader->open(cfg.bgen_path);
            reader->initialize();
            reader->computeStats(cfg.threads > 0 ? cfg.threads : 1);
            sample_ids = reader->getSampleIds();
            source_label = cfg.bgen_path;
            r.stcg_genotype = std::move(reader);
        }

        if (!r.stcg_genotype || r.stcg_genotype->rows() <= 0 || r.stcg_genotype->cols() <= 0) {
            throw std::runtime_error("STCG mode could not initialize a non-empty genotype matrix.");
        }
        if (sample_ids.empty()) {
            throw std::runtime_error("STCG mode requires sample IDs from the genotype input.");
        }
        if ((int)sample_ids.size() != r.stcg_genotype->rows()) {
            throw std::runtime_error("STCG mode sample ID count does not match genotype row count.");
        }

        r.idmap.clear();
        std::map<std::string, int> iid_alias_counts;
        for (const auto& sample_id : sample_ids) {
            size_t sep = sample_id.find(':');
            if (sep != std::string::npos && sep + 1 < sample_id.size()) {
                iid_alias_counts[sample_id.substr(sep + 1)] += 1;
            }
        }
        for (size_t i = 0; i < sample_ids.size(); ++i) {
            if (sample_ids[i].empty()) {
                throw std::runtime_error("STCG mode encountered an empty genotype sample ID.");
            }
            r.idmap[sample_ids[i]] = (int)i + 1;
            size_t sep = sample_ids[i].find(':');
            if (sep != std::string::npos && sep + 1 < sample_ids[i].size()) {
                std::string iid_alias = sample_ids[i].substr(sep + 1);
                if (iid_alias_counts[iid_alias] == 1) {
                    r.idmap[iid_alias] = (int)i + 1;
                }
            }
        }
        r.stcg_ga_dim = std::make_unique<IdentityMatrixAdapter>(r.stcg_genotype->rows());
        r.inv_label = "GA(STCG)";
        r.inv_file = "Implicit GRM via genotype matrix";
        r.id_file_log = source_label;
        LOG_INFO("STCG mode enabled: using implicit GRM from genotype matrix with "
                 << r.stcg_genotype->rows() << " samples and "
                 << r.stcg_genotype->cols() << " variants; skipping explicit G-inverse construction.");
    }

    return r;
}

} // namespace cosmic
