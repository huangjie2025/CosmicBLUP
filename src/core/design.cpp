#include "design.h"
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <cctype>

namespace cosmic {


FixedDesignG buildFixedDesignGeneric(const std::vector<GenRecord>& recs,
                                     bool legacy_names,
                                     const std::vector<std::string>& cat_names,
                                     const std::vector<std::string>& num_names) {
    int n = (int)recs.size();
    std::vector<std::vector<std::string>> cat_levels;
    if (!recs.empty()) {
        int cnum = (int)recs.front().cats.size();
        cat_levels.resize(cnum);
        for (int j = 0; j < cnum; ++j) {
            std::set<std::string> lv;
            for (const auto& r : recs) {
                if (!r.cats[j].empty()) lv.insert(r.cats[j]);
            }
            // Try to sort numerically if all look like integers
            bool all_digits = false; // Disable numerical sorting to match HIBLUP's string sort
            /*
            bool all_digits = true;
            for (const auto& s : lv) {
                if (s.empty()) continue;
                for (char c : s) if (!isdigit(c)) { all_digits = false; break; }
                if (!all_digits) break;
            }
            */
            if (all_digits && !lv.empty()) {
                std::vector<int> ints;
                for (const auto& s : lv) ints.push_back(std::stoi(s));
                std::sort(ints.begin(), ints.end());
                for (int v : ints) cat_levels[j].push_back(std::to_string(v));
            } else {
                for (auto& s : lv) cat_levels[j].push_back(s);
            }
        }
    }
    int p = 1;
    std::vector<std::string> names; names.push_back("mu");
    std::map<std::string, std::vector<int>> factor_cols;
    std::vector<std::string> factor_names;

    factor_cols["mu"].push_back(0);
    factor_names.push_back("mu");

    for (size_t j = 0; j < cat_levels.size(); ++j) {
        auto& lv = cat_levels[j];
        if (lv.empty()) continue;
        std::string base = lv.front();
        std::string prefix;
        if (legacy_names && cat_levels.size() == 4) {
            static const std::vector<std::string> legacy = {"parity","herd","ys","sex"};
            prefix = legacy[j];
        } else if (!cat_names.empty() && j < cat_names.size()) {
            prefix = cat_names[j];
        } else {
            prefix = std::string("factor") + std::to_string(j+1);
        }
        factor_names.push_back(prefix);

        for (size_t k = 0; k < lv.size(); ++k) {
            if (lv[k] == base) continue;
            names.push_back(prefix + std::string("_") + lv[k]);
            factor_cols[prefix].push_back(p);
            ++p;
        }
    }
    if (!recs.empty()) {
        int qn = (int)recs.front().nums.size();
        for (int j = 0; j < qn; ++j) {
            std::string name;
            if (legacy_names && qn == 1) name = "bw";
            else if (!num_names.empty() && j < num_names.size()) name = num_names[j];
            else name = std::string("cov") + std::to_string(j+1);
            names.push_back(name);
            factor_cols[name].push_back(p);
            factor_names.push_back(name);
            ++p;
        }
    }

    FixedDesignG fd; fd.p = p; fd.names = names; fd.rows.resize(n); fd.factor_cols = factor_cols; fd.factor_names = factor_names;
    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<int,double>> nz; nz.emplace_back(0, 1.0);
        int col = 1;
        for (size_t j = 0; j < cat_levels.size(); ++j) {
            auto& lv = cat_levels[j]; if (lv.empty()) continue; std::string base = lv.front(); std::string val = recs[i].cats[j];
            for (size_t k = 0; k < lv.size(); ++k) {
                if (lv[k] == base) continue;
                if (val == lv[k]) nz.emplace_back(col, 1.0);
                ++col;
            }
        }
        if (!recs[i].nums.empty()) {
            for (size_t j = 0; j < recs[i].nums.size(); ++j) { nz.emplace_back(col, recs[i].nums[j]); ++col; }
        }
        fd.rows[i] = std::move(nz);
    }
    return fd;
}

// Helper for factorial
static double factorial(int n) {
    if (n <= 1) return 1.0;
    double res = 1.0;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

Eigen::MatrixXd buildLegendreMatrix(const Eigen::VectorXd& time, double tmin, double tmax, int order) {
    int n = time.size();
    Eigen::MatrixXd pmat(n, order + 1);

    // Scale time to [-1, 1]
    Eigen::VectorXd tvec = 2.0 * (time.array() - tmin) / (tmax - tmin) - 1.0;

    for (int k = 0; k <= order; ++k) {
        int c = k / 2;
        int j = k;
        Eigen::VectorXd p = Eigen::VectorXd::Zero(n);

        for (int r = 0; r <= c; ++r) {
            double coeff = std::sqrt((2.0 * j + 1.0) / 2.0) * std::pow(0.5, j) *
                           (std::pow(-1.0, r) * factorial(2 * j - 2 * r) /
                           (factorial(r) * factorial(j - r) * factorial(j - 2 * r)));

            p.array() += coeff * tvec.array().pow(j - 2 * r);
        }
        pmat.col(k) = p;
    }
    return pmat;
}

}