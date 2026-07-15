#include "blup_app.h"
#include "auxiliary_tasks.h"
#include <algorithm>
#include <sstream>

namespace cosmic {

using namespace std;

SolverApp::SolverApp(const Config& cfg) : cfg(cfg) {}
SolverApp::~SolverApp() = default;  // needed for unique_ptr with incomplete types

void SolverApp::runQc() { runQcMode(cfg); }

void SolverApp::runSblup() { runSblupMode(cfg); }

void SolverApp::runPrediction() { runPredictionMode(cfg); }

void SolverApp::parseModelEquation() {
    if (cfg.model.empty()) return;
    string model_name = cfg.model;
    transform(model_name.begin(), model_name.end(), model_name.begin(), ::tolower);
    if (model_name == "pblup" || model_name == "gblup" || model_name == "ssgblup" ||
        model_name == "repeatability" || model_name == "rrm") return;
    string s = cfg.model;
    for(char& c : s) if(c=='+' || c=='=') c=' ';
    stringstream ss(s);
    string token;
    vector<string> parts;
    while(ss >> token) parts.push_back(token);

    if (parts.empty()) return;

    if (cfg.pheno_name.empty()) cfg.pheno_name = parts[0];

    for (size_t i = 1; i < parts.size(); ++i) {
        string term = parts[i];
        string lower = term;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "id" || lower == "iid") continue;
        if (term == "1") continue;

        bool exists = false;
        for(const auto& n : cfg.dcovar_names) if(n == term) exists = true;
        if (!exists) cfg.dcovar_names.push_back(term);
    }
}

} // namespace cosmic
