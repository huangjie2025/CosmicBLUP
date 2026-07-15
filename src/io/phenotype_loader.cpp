#include "phenotype_loader.h"
#include "logger.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>

namespace cosmic {

using namespace std;
using namespace Eigen;

// ── Phenotype reading ────────────────────────────────────────────

std::vector<GenRecord> readPhenGeneric(const std::string& phen_file,
                                       const std::map<std::string,int>& idmap,
                                       int trait_pos_1based,
                                       const std::vector<int>& dcovar_pos_1based,
                                       const std::vector<int>& qcovar_pos_1based,
                                       const std::vector<int>& rand_pos_1based,
                                       std::vector<std::string>& dcovar_names,
                                       std::vector<std::string>& qcovar_names,
                                       std::vector<std::string>& rand_names,
                                       const std::string& trait_name,
                                       const std::string& id_col_name,
                                       PhenStats* stats) {
    ifstream file(phen_file);
    if (!file.is_open()) throw runtime_error("Cannot open phenotype file: " + phen_file);
    string header; getline(file, header);
    char delim = detect_delim(header);
    vector<string> headers;
    {
        if (!header.empty() && (header[0] == ' ' || header[0] == '\t')) {
            headers.push_back("ID");
        }
        string tok; stringstream ss(header);
        if (delim == ' ') { while (ss >> tok) headers.push_back(tok); }
        else { while (getline(ss, tok, ',')) headers.push_back(trim_copy(tok)); }
    }
    auto find_col_index = [&](const string& name_or_idx) -> int {
        if (is_integer_string(name_or_idx)) return stoi(name_or_idx);
        for (size_t i = 0; i < headers.size(); ++i) {
            if (headers[i] == name_or_idx) return (int)i;
            string h = headers[i]; string n = name_or_idx;
            transform(h.begin(), h.end(), h.begin(), ::tolower);
            transform(n.begin(), n.end(), n.begin(), ::tolower);
            if (h == n) return (int)i;
        }
        return -1;
    };
    int id_idx = -1;
    if (!id_col_name.empty()) id_idx = find_col_index(id_col_name);
    if (id_idx < 0) {
        id_idx = find_col_index("id");
        if (id_idx < 0) id_idx = find_col_index("ID");
        if (id_idx < 0) id_idx = find_col_index("Id");
        if (id_idx < 0) id_idx = find_col_index("IID");
    }
    if (id_idx < 0) throw runtime_error("Phenotype file missing required ID column");
    int trait_idx = -1;
    if (!trait_name.empty()) trait_idx = find_col_index(trait_name);
    if (trait_idx < 0) trait_idx = trait_pos_1based - 1;
    vector<int> dcols, qcols;
    if (!dcovar_names.empty()) {
        for (const auto& nm : dcovar_names) {
            int idx = find_col_index(nm);
            if (idx < 0) throw runtime_error("Cannot find discrete covariate column: " + nm);
            dcols.push_back(idx);
        }
    } else {
        for (int v : dcovar_pos_1based) {
            dcols.push_back(v - 1);
            if (v - 1 >= 0 && v - 1 < (int)headers.size()) dcovar_names.push_back(headers[v - 1]);
            else dcovar_names.push_back("factor" + to_string(v));
        }
    }
    if (!qcovar_names.empty()) {
        for (const auto& nm : qcovar_names) {
            int idx = find_col_index(nm);
            if (idx < 0) throw runtime_error("Cannot find quantitative covariate column: " + nm);
            qcols.push_back(idx);
        }
    } else {
        for (int v : qcovar_pos_1based) {
            qcols.push_back(v - 1);
            if (v - 1 >= 0 && v - 1 < (int)headers.size()) qcovar_names.push_back(headers[v - 1]);
            else qcovar_names.push_back("cov" + to_string(v));
        }
    }
    vector<int> rcols;
    if (!rand_names.empty()) {
        for (const auto& nm : rand_names) {
            int idx = find_col_index(nm);
            if (idx < 0) throw runtime_error("Cannot find random effect column: " + nm);
            rcols.push_back(idx);
        }
    } else {
        for (int v : rand_pos_1based) {
            rcols.push_back(v - 1);
            if (v - 1 >= 0 && v - 1 < (int)headers.size()) rand_names.push_back(headers[v - 1]);
            else rand_names.push_back("rand" + to_string(v));
        }
    }

    vector<GenRecord> recs; string line;
    PhenStats local;
    while (getline(file, line)) {
        string row = trim_copy(line);
        if (row.empty()) continue;
        local.n_rows_total++;
        vector<string> vals;
        if (delim == ' ') { string v; stringstream ss(row); while (ss >> v) vals.push_back(v); }
        else { string tok; stringstream ss(row); while (getline(ss, tok, ',')) vals.push_back(trim_copy(tok)); }

        if (id_idx < 0 || trait_idx < 0) continue;

        int max_needed = max(id_idx, trait_idx);
        for(int c : dcols) max_needed = max(max_needed, c);
        for(int c : qcols) max_needed = max(max_needed, c);
         for(int c : rcols) max_needed = max(max_needed, c);

        if ((int)vals.size() <= max_needed) { local.n_dropped_missing++; continue; }

        string idstr = vals[id_idx]; if (idstr.empty()) { local.n_dropped_missing++; continue; }
        auto it = idmap.find(idstr); if (it == idmap.end()) { local.n_not_in_id++; continue; }

        string ystr = vals[trait_idx]; if (is_missing_token(ystr)) { local.n_dropped_missing++; continue; }
        double yv = 0.0; try { yv = stod(ystr); } catch (...) { local.n_dropped_missing++; continue; }

        GenRecord r; r.idstr = idstr; r.aid = it->second; r.y = yv;
        bool keep = true;
        for (int dc : dcols) {
            string vs = vals[dc];
            if (is_missing_token(vs)) { keep = false; break; }
            r.cats.push_back(vs);
        }
        if (!keep) { local.n_dropped_missing++; continue; }
         for (int rc : rcols) {
             string vs = vals[rc];
             if (is_missing_token(vs)) { keep = false; break; }
             r.rand_cats.push_back(vs);
         }
         if (!keep) { local.n_dropped_missing++; continue; }

        for (int qc : qcols) {
            string vs = vals[qc];
            if (is_missing_token(vs)) { keep = false; break; }
            try { r.nums.push_back(stod(vs)); } catch (...) { keep = false; break; }
        }
        if (!keep) { local.n_dropped_missing++; continue; }

        recs.push_back(r);
    }
    local.n_used = (long long)recs.size();
    if (stats) {
        *stats = local;
    } else {
        LOG_INFO("Phenotype read: records=" << recs.size() << ", dropped=" << local.n_dropped_missing);
    }
    return recs;
}

// ── DataLoader::load ─────────────────────────────────────────────

DataLoadResult DataLoader::load(const Config& cfg,
                                const std::map<std::string, int>& idmap_from_matrix) {
    DataLoadResult result;
    result.idmap = idmap_from_matrix;

    // 1. Load variance components
    result.sigma2_u = 1.0;
    result.sigma2_e = 1.0;
    result.run_vce = cfg.run_vce && !cfg.skip_vce;

    if (cfg.vars_path.empty()) {
        if (cfg.var_a > 0.0 && cfg.var_e > 0.0) {
            result.sigma2_u = cfg.var_a;
            result.sigma2_e = cfg.var_e;
            LOG_INFO("Using explicit variance inputs: Vu=" << result.sigma2_u << ", Ve=" << result.sigma2_e);
        } else if (!cfg.var_priors.empty() && cfg.var_priors.size() >= 2) {
            result.sigma2_u = cfg.var_priors[0];
            result.sigma2_e = cfg.var_priors[1];
            LOG_INFO("Using provided variance priors: Vu=" << result.sigma2_u << ", Ve=" << result.sigma2_e);
        } else if (result.run_vce) {
            LOG_INFO("No vars file provided. Using default initial variances (1.0, 1.0) for VCE.");
        } else if (cfg.skip_vce) {
            LOG_INFO("No vars file provided. --skip-vce requested, so using default variances (1.0, 1.0) without running VCE.");
        } else {
            LOG_INFO("No vars file provided. Switching to VCE Mode (Normal).");
            result.run_vce = true;
        }
    } else {
        auto vars = readVars(cfg.vars_path);
        result.sigma2_u = vars.first;
        result.sigma2_e = vars.second;
    }

    // 2. Load phenotype
    LOG_INFO("Loading phenotype file ...");
    result.records = readPhenGeneric(cfg.pheno_path, result.idmap,
                                     cfg.pheno_pos, cfg.dcovar_cols, cfg.qcovar_cols, cfg.rand_cols,
                                     const_cast<std::vector<std::string>&>(cfg.dcovar_names),
                                     const_cast<std::vector<std::string>&>(cfg.qcovar_names),
                                     const_cast<std::vector<std::string>&>(cfg.rand_names),
                                     cfg.pheno_name, cfg.id_col_name,
                                     &result.phen_stats);
    LOG_INFO(result.phen_stats.n_rows_total << " individuals have been detected in file [" << cfg.pheno_path << "].");
    long long missing_count = result.phen_stats.n_rows_total - (long long)result.records.size();
    if (missing_count > 0) {
        LOG_INFO("Non-missing phenotypes of " << result.records.size() << " individuals are included.");
        LOG_INFO(missing_count << " individuals have missing phenotypes and are excluded.\n");
    } else {
        LOG_INFO(result.records.size() << " effective records without missing will be used for analysis.\n");
    }

    // 3. Build fixed design matrix
    result.fixed_design = buildFixedDesignGeneric(result.records, false,
                                                  cfg.dcovar_names, cfg.qcovar_names);
    LOG_INFO("Make index matrix for fixed effects and covariates ...");
    LOG_INFO("Make index matrix for random effects ...\n");

    return result;
}

} // namespace cosmic
