#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <Eigen/Dense>

namespace cosmic {

struct Record {
    int aid;
    double y;
    int parity;
    int herd;
    int ys;
    int sex;
    double bw;
    std::string idstr;
};

struct GenRecord {
    int aid;
    double y;
    std::vector<std::string> cats;
    std::vector<double> nums;
    std::vector<std::string> rand_cats;
    std::string idstr;
};

struct FixedDesign {
    int p;
    std::vector<std::string> names;
    std::vector<std::vector<std::pair<int,double>>> rows;
    int mu_col;
    int bw_col;
    std::vector<int> parity_levels, herd_levels, ys_levels, sex_levels;
    int parity_base, herd_base, ys_base, sex_base;
    std::unordered_map<int,int> parity_col, herd_col, ys_col, sex_col;
};

struct FixedDesignG {
    int p;
    std::vector<std::string> names;
    std::vector<std::vector<std::pair<int,double>>> rows;
    std::map<std::string, std::vector<int>> factor_cols;
    std::vector<std::string> factor_names;
};


FixedDesignG buildFixedDesignGeneric(const std::vector<GenRecord>& recs,
                                    bool legacy_names,
                                    const std::vector<std::string>& cat_names,
                                    const std::vector<std::string>& num_names);

// Generate Legendre polynomial matrix for Random Regression Model
// Returns a matrix of size N x (order + 1)
Eigen::MatrixXd buildLegendreMatrix(const Eigen::VectorXd& time, double tmin, double tmax, int order);

}