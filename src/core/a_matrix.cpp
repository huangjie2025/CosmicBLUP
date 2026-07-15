#include "a_matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <deque>
#include <numeric>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <map>
#include <set>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>



// GenMatrix.h
// C++ header-only 实现：Meuwissen & Luo 风格的 ml/mml/ga 等
// - -1 表示缺失父母
// - Eigen 矩阵操作
// - buildA 使用 Henderson 递推算法
// - invertA_henderson: Henderson 规则直接构造 A^{-1}（已增强：考虑 inbreeding）
// - 所有稀疏矩阵构造后强制对称化以及对角修正



namespace cosmic {
namespace genmatrix {

using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::MatrixXd;

void complete_missing_parents_add_phantoms(std::vector<int>& dam, std::vector<int>& sire) {
    const int n = static_cast<int>(dam.size());
    if (static_cast<int>(sire.size()) != n) {
        throw std::invalid_argument("complete_missing_parents_add_phantoms: dam/sire length mismatch");
    }
    for (int j = 0; j < n; ++j) {
        if (!(dam[j]  >= 0 && dam[j]  < n)) dam[j]  = -1;
        if (!(sire[j] >= 0 && sire[j] < n)) sire[j] = -1;
    }
}

bool valid_id(int id, int n) {
    return (id >= 0 && id < n);
}

void enforceSymmetryAndDiagonal(SparseMatrix<double>& M,
                                       double eps = 1e-10,
                                       double tiny = 1e-14)
{
    SparseMatrix<double> Mt = M.transpose();
    M = 0.5 * (M + Mt);

    for (int k = 0; k < M.outerSize(); ++k) {
        bool has_diag = false;
        for (SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            if (it.row() == it.col()) {
                has_diag = true;
                if (it.value() < tiny) {
                    const_cast<double&>(it.valueRef()) = eps;
                }
            }
        }
        if (!has_diag) {
            M.coeffRef(k, k) = eps;
        }
    }
}

int safe_stoi(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("safe_stoi: 空字符串不能转换为整数");
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (!(std::isdigit(c) || ((c == '+' || c == '-') && i == 0))) {
            throw std::invalid_argument("safe_stoi: 非法字符 '" + std::string(1, c) + "' in \"" + s + "\"");
        }
    }
    try {
        return std::stoi(s);
    } catch (const std::out_of_range&) {
        throw std::out_of_range("safe_stoi: 数字超出 int 范围 -> " + s);
    }
}

std::vector<double> calculateF_Atemp(const std::vector<int>& dam, const std::vector<int>& sire)
{
    const int n = static_cast<int>(dam.size());
    std::vector<double> f(n, 0.0);

    Eigen::MatrixXd A_temp = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        int d = valid_id(dam[i], n) ? dam[i] : -1;
        int s = valid_id(sire[i], n) ? sire[i] : -1;

        double parental_coeff = 0.0;
        if (d != -1 && s != -1) {
            parental_coeff = A_temp(d, s);
        }

        f[i] = 0.5 * parental_coeff;
        A_temp(i, i) = 1.0 + f[i];

        for (int j = 0; j < i; ++j) {
            double val = 0.0;
            if (s != -1) val += A_temp(j, s);
            if (d != -1) val += A_temp(j, d);
            val *= 0.5;
            A_temp(i, j) = val;
            A_temp(j, i) = val;
        }
    }
    return f;
}

void ml(const vector<int>& dam,
               const vector<int>& sire,
               vector<double>& f,
               vector<double>& dii,
               int g,
               int fmiss)
{
    const int n = static_cast<int>(dam.size());
    if ((int)sire.size() != n) throw std::invalid_argument("ml: dam/sire length mismatch");
    if ((int)f.size() != n) f.assign(n, 0.0);
    if ((int)dii.size() != n) dii.assign(n, 0.0);

    f.assign(n, 0.0);
    dii.assign(n, 1.0);

    std::vector<double> L(n, 0.0);

    for (int i = 0; i < n; ++i) {
        int s = sire[i];
        int d = dam[i];

        bool s_ok = valid_id(s, n);
        bool d_ok = valid_id(d, n);

        double Fs = s_ok ? f[s] : 0.0;
        double Fd = d_ok ? f[d] : 0.0;

        if (s_ok && d_ok) {
            dii[i] = 1.0 - 0.25 * (1.0 + Fs) - 0.25 * (1.0 + Fd);
        } else if (s_ok) {
            dii[i] = 1.0 - 0.25 * (1.0 + Fs);
        } else if (d_ok) {
            dii[i] = 1.0 - 0.25 * (1.0 + Fd);
        } else {
            dii[i] = 1.0;
        }

        if (s_ok && d_ok) {
            L[s] += 0.5;
            L[d] += 0.5;
            double F_calc = 0.0;
            int max_idx = std::max(s, d);
            for (int j = max_idx; j >= 0; --j) {
                if (L[j] > 0.0) {
                    F_calc += L[j] * L[j] * dii[j];
                    int sj = sire[j];
                    int dj = dam[j];
                    if (valid_id(sj, n)) L[sj] += 0.5 * L[j];
                    if (valid_id(dj, n)) L[dj] += 0.5 * L[j];
                    L[j] = 0.0;
                }
            }
            f[i] = F_calc + dii[i] - 1.0;
            // Correct floating point inaccuracies
            if (f[i] < 0.0) f[i] = 0.0;
        } else {
            f[i] = 0.0;
        }
    }
}

void ga(const vector<int>& dam,
               const vector<int>& sire,
               vector<int>& generation)
{
    const int n = static_cast<int>(dam.size());
    if ((int)sire.size() != n) throw std::invalid_argument("ga: dam/sire length mismatch");
    if ((int)generation.size() != n) generation.assign(n, -1);

    int nmiss = 1, cnt = 0;
    while ((nmiss > 0) && (cnt < n)) {
        nmiss = 0;
        for (int k = 0; k < n; ++k) {
            int kdam = dam[k];
            int ksire = sire[k];
            if (generation[k] != -1) continue;
            if (valid_id(kdam, n) && valid_id(ksire, n)) {
                if (generation[kdam] != -1 && generation[ksire] != -1) {
                    generation[k] = std::max(generation[kdam], generation[ksire]) + 1;
                } else nmiss++;
            } else if (valid_id(kdam, n)) {
                if (generation[kdam] != -1) generation[k] = generation[kdam] + 1;
                else nmiss++;
            } else if (valid_id(ksire, n)) {
                if (generation[ksire] != -1) generation[k] = generation[ksire] + 1;
                else nmiss++;
            } else {
                generation[k] = 0;
            }
        }
        ++cnt;
    }
}

SparseMatrix<double> buildA(const vector<int>& dam,
                                   const vector<int>& sire)
{
    if (dam.size() != sire.size()) throw std::invalid_argument("buildA: dam/sire length mismatch");

    std::vector<int> d = dam, s = sire;
    complete_missing_parents_add_phantoms(d, s);
    const int n = static_cast<int>(d.size());
    std::cout << "DEBUG: buildA n after phantoms: " << n << std::endl;
    if ((int)s.size() != n) throw std::invalid_argument("buildA: dam/sire length mismatch after completion");

    std::vector<int> gen(n, -1);
    ga(d, s, gen);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        if (gen[a] != gen[b]) return gen[a] < gen[b];
        return a < b;
    });
    std::vector<int> old_to_new(n);
    for (int i = 0; i < n; ++i) old_to_new[order[i]] = i;
    std::vector<int> d_new(n, -1), s_new(n, -1);
    for (int i = 0; i < n; ++i) {
        int old_idx = order[i];
        int od = d[old_idx];
        int os = s[old_idx];
        d_new[i] = (od >= 0 && od < n) ? old_to_new[od] : -1;
        s_new[i] = (os >= 0 && os < n) ? old_to_new[os] : -1;
    }
    d.swap(d_new);
    s.swap(s_new);

    std::cout << "DEBUG: buildA allocating dense matrix " << n << "x" << n << std::endl;
    MatrixXd Ad = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        if (i % 1000 == 0) std::cout << "DEBUG: buildA row " << i << "/" << n << "\r" << std::flush;
        int di = d[i], sire_i = s[i]; // renamed to sire_i to avoid conflict if any
        double parental_coeff = 0.0;
        if (valid_id(di, n) && valid_id(sire_i, n)) {
            parental_coeff = Ad(di, sire_i);
        }
        Ad(i, i) = 1.0 + 0.5 * parental_coeff;

        for (int j = 0; j < i; ++j) {
            double val = 0.0;
            if (valid_id(sire_i, n)) val += 0.5 * Ad(j, sire_i);
            if (valid_id(di, n)) val += 0.5 * Ad(j, di);
            Ad(i, j) = val;
            Ad(j, i) = val;
        }
    }
    std::cout << "DEBUG: buildA done filling dense matrix" << std::endl;

    SparseMatrix<double> A(n, n);
    std::vector<Triplet<double>> triplets;
    // reserve triplets? n*n is too big for sparse, but A is dense here so...
    // actually if A is dense, why return sparse?
    // ah, the function returns SparseMatrix.

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = Ad(i, j);
            if (v != 0.0) {
                triplets.emplace_back(order[i], order[j], v);
                if (i != j) triplets.emplace_back(order[j], order[i], v);
            }
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    enforceSymmetryAndDiagonal(A);
    return A;
}

Eigen::MatrixXd buildA_dense(const vector<int>& dam,
                                   const vector<int>& sire)
{
    if (dam.size() != sire.size()) throw std::invalid_argument("buildA_dense: dam/sire length mismatch");

    std::vector<int> d = dam, s = sire;
    complete_missing_parents_add_phantoms(d, s);
    const int n = static_cast<int>(d.size());
    if ((int)s.size() != n) throw std::invalid_argument("buildA_dense: dam/sire length mismatch after completion");

    std::vector<int> gen(n, -1);
    ga(d, s, gen);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        if (gen[a] != gen[b]) return gen[a] < gen[b];
        return a < b;
    });
    std::vector<int> old_to_new(n);
    for (int i = 0; i < n; ++i) old_to_new[order[i]] = i;
    std::vector<int> d_new(n, -1), s_new(n, -1);
    for (int i = 0; i < n; ++i) {
        int old_idx = order[i];
        int od = d[old_idx];
        int os = s[old_idx];
        d_new[i] = (od >= 0 && od < n) ? old_to_new[od] : -1;
        s_new[i] = (os >= 0 && os < n) ? old_to_new[os] : -1;
    }
    d.swap(d_new);
    s.swap(s_new);

    MatrixXd Ad = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        int di = d[i], sire_i = s[i];
        double parental_coeff = 0.0;
        if (valid_id(di, n) && valid_id(sire_i, n)) {
            parental_coeff = Ad(di, sire_i);
        }
        Ad(i, i) = 1.0 + 0.5 * parental_coeff;

        for (int j = 0; j < i; ++j) {
            double val = 0.0;
            if (valid_id(sire_i, n)) val += 0.5 * Ad(j, sire_i);
            if (valid_id(di, n)) val += 0.5 * Ad(j, di);
            Ad(i, j) = val;
            Ad(j, i) = val;
        }
    }

    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double v = Ad(i, j);
            if (v != 0.0) {
                A(order[i], order[j]) = v;
                if (i != j) A(order[j], order[i]) = v;
            }
        }
    }
    return A;
}

Eigen::MatrixXd buildA_dense_noninline(const std::vector<int>& dam,
                                       const std::vector<int>& sire) {
    return buildA_dense(dam, sire);
}

// Provide a non-export wrapper to ensure linker symbol is emitted
Eigen::SparseMatrix<double> buildA_noninline(const std::vector<int>& dam,
                                             const std::vector<int>& sire) {
    return buildA(dam, sire);
}

SparseMatrix<double> invertA(const SparseMatrix<double>& A)
{
    if (A.rows() != A.cols()) throw std::invalid_argument("invertA: A must be square");
    const int n = A.rows();

    Eigen::SimplicialLDLT<SparseMatrix<double>> chol;
    chol.compute(A);
    if (chol.info() != Eigen::Success) {
        MatrixXd denseA = MatrixXd(A);
        Eigen::FullPivLU<MatrixXd> lu(denseA);
        if (!lu.isInvertible()) {
            throw std::runtime_error("invertA: matrix not invertible");
        }
        MatrixXd denseInv = lu.inverse();
        SparseMatrix<double> res = denseInv.sparseView();
        enforceSymmetryAndDiagonal(res);
        return res;
    }
    Eigen::MatrixXd Id_mat = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd denseInv = chol.solve(Id_mat);
    SparseMatrix<double> res = denseInv.sparseView();
    enforceSymmetryAndDiagonal(res);
    return res;
}

SparseMatrix<double> invertA_henderson(const vector<int>& dam,
                                              const vector<int>& sire)
{
    if ((int)dam.size() != (int)sire.size()) throw std::invalid_argument("invertA_henderson: dam/sire length mismatch");

    std::vector<int> d = dam, s = sire;
    complete_missing_parents_add_phantoms(d, s);
    const int n = static_cast<int>(d.size());
    if ((int)s.size() != n) throw std::invalid_argument("invertA_henderson: dam/sire length mismatch after completion");

    std::vector<double> F, dii;
    ml(d, s, F, dii);

    std::vector<Triplet<double>> triplets;
    triplets.reserve(n * 9);

    for (int i = 0; i < n; ++i) {
        int di = d[i];
        int si = s[i];
        bool d_ok = valid_id(di, n);
        bool s_ok = valid_id(si, n);

        double d_inv = 1.0 / dii[i];

        triplets.emplace_back(i, i, d_inv);

        if (s_ok) {
            triplets.emplace_back(i, si, -0.5 * d_inv);
            triplets.emplace_back(si, i, -0.5 * d_inv);
            triplets.emplace_back(si, si, 0.25 * d_inv);
        }
        if (d_ok) {
            triplets.emplace_back(i, di, -0.5 * d_inv);
            triplets.emplace_back(di, i, -0.5 * d_inv);
            triplets.emplace_back(di, di, 0.25 * d_inv);
        }
        if (s_ok && d_ok) {
            triplets.emplace_back(si, di, 0.25 * d_inv);
            triplets.emplace_back(di, si, 0.25 * d_inv);
        }
    }

    SparseMatrix<double> Ainv(n, n);
    Ainv.setFromTriplets(triplets.begin(), triplets.end());
    enforceSymmetryAndDiagonal(Ainv, 1e-12, 1e-15);
    Ainv.prune(1e-12);
    return Ainv;
}

Eigen::SparseMatrix<double> invertA_henderson_noninline(const std::vector<int>& dam,
                                                        const std::vector<int>& sire) {
    return invertA_henderson(dam, sire);
}

SparseMatrix<double> invertA_henderson_fixed(const vector<int>& dam,
                                                    const vector<int>& sire)
{
    if ((int)dam.size() != (int)sire.size()) throw std::invalid_argument("invertA_henderson_fixed: dam/sire length mismatch");

    std::vector<int> d = dam, s = sire;
    complete_missing_parents_add_phantoms(d, s);
    const int n = static_cast<int>(d.size());
    if ((int)s.size() != n) throw std::invalid_argument("invertA_henderson_fixed: dam/sire length mismatch after completion");

    std::vector<double> F, dii;
    ml(d, s, F, dii);

    std::vector<std::map<int, double>> rows(n);
    auto add_one_way = [&](int r, int c, double v) {
        if (v == 0.0) return;
        if (valid_id(r, n) && valid_id(c, n)) {
            rows[r][c] += v;
        }
    };

    for (int i = 0; i < n; ++i) {
        int di = d[i], si = s[i];
        bool d_ok = valid_id(di, n), s_ok = valid_id(si, n);

        double d_inv = 1.0 / dii[i];

        add_one_way(i, i, d_inv);

        if (d_ok && s_ok) {
            add_one_way(i, si, -0.5 * d_inv); add_one_way(si, i, -0.5 * d_inv);
            add_one_way(i, di, -0.5 * d_inv); add_one_way(di, i, -0.5 * d_inv);
            add_one_way(si, si, 0.25 * d_inv);
            add_one_way(di, di, 0.25 * d_inv);
            add_one_way(si, di, 0.25 * d_inv); add_one_way(di, si, 0.25 * d_inv);
        } else if (d_ok || s_ok) {
            int p = d_ok ? di : si;
            add_one_way(i, p, -0.5 * d_inv); add_one_way(p, i, -0.5 * d_inv);
            add_one_way(p, p, 0.25 * d_inv);
        }
    }

    std::vector<Triplet<double>> triplets;
    triplets.reserve(n * 4);
    for (int r = 0; r < n; ++r) {
        for (const auto &kv : rows[r]) {
            if (std::abs(kv.second) > 1e-12) triplets.emplace_back(r, kv.first, kv.second);
        }
    }

    SparseMatrix<double> Ainv(n, n);
    Ainv.setFromTriplets(triplets.begin(), triplets.end());
    enforceSymmetryAndDiagonal(Ainv, 1e-12, 1e-15);
    Ainv.prune(1e-12);
    return Ainv;
}

SparseMatrix<double> invert_sparse_matrix(const SparseMatrix<double>& M)
{
    return invertA(M);
}

void reT(const vector<int>& dam,
                const vector<int>& sire,
                vector<int>& i,
                vector<int>& p,
                vector<double>& x,
                int& maxcnt,
                const vector<double>& tx)
{
    const int n = static_cast<int>(dam.size());
    if ((int)sire.size() != n) throw std::invalid_argument("reT: dam/sire length mismatch");
    if ((int)tx.size() < 4) throw std::invalid_argument("reT: tx must have length >= 4");

    i.clear(); x.clear(); p.assign(n+1,0);
    int cnt=0;
    for(int k=0;k<n;++k){
        p[k]=cnt;
        int kdam=dam[k], ksire=sire[k];
        if(kdam==ksire){
            if(valid_id(kdam,n)){i.push_back(kdam);x.push_back(-tx[2]);++cnt;}
        } else {
            if(kdam<ksire){
                if(valid_id(kdam,n)){i.push_back(kdam);x.push_back(-tx[0]);++cnt;}
                if(valid_id(ksire,n)){i.push_back(ksire);x.push_back(-tx[1]);++cnt;}
            } else {
                if(valid_id(ksire,n)){i.push_back(ksire);x.push_back(-tx[1]);++cnt;}
                if(valid_id(kdam,n)){i.push_back(kdam);x.push_back(-tx[0]);++cnt;}
            }
        }
        i.push_back(k);x.push_back(tx[3]);++cnt;
    }
    p[n]=cnt; maxcnt=cnt;
}

void printSparse(const SparseMatrix<double>& M, std::ostream& out = std::cout) {
    for (int k = 0; k < M.outerSize(); ++k) {
        for (typename SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
            out << "(" << it.row() << "," << it.col() << ") = " << it.value() << "\n";
        }
    }
}

} // namespace genmatrix
} // namespace cosmic



using namespace std;
using Eigen::SparseMatrix;
using Eigen::MatrixXd;

void write_matrix_to_csv(const SparseMatrix<double>& M, const std::string& filename) {
    MatrixXd M_dense = MatrixXd(M);
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "❌ 无法写入矩阵文件 " << filename << endl;
        return;
    }
    for (int r = 0; r < M_dense.rows(); ++r) {
        for (int c = 0; c < M_dense.cols(); ++c) {
            fout << M_dense(r, c);
            if (c < M_dense.cols() - 1) fout << ",";
        }
        fout << "\n";
    }
    fout.close();
    cout << "✅ 矩阵已输出到 " << filename << endl;
}

void write_matrix_with_ids_to_csv(const SparseMatrix<double>& M, const std::vector<std::string>& ids, const std::string& filename) {
    MatrixXd M_dense = MatrixXd(M);
    int n = M_dense.rows();
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        cerr << "❌ 无法写入矩阵文件 " << filename << endl;
        return;
    }
    fout << "ID";
    for (int c = 0; c < n; ++c) {
        fout << "," << ids[c];
    }
    fout << "\n";
    for (int r = 0; r < n; ++r) {
        fout << ids[r];
        for (int c = 0; c < n; ++c) {
            fout << "," << M_dense(r, c);
        }
        fout << "\n";
    }
    fout.close();
    cout << "✅ 矩阵(含 ID 标签)已输出到 " << filename << endl;
}

SparseMatrix<double> invert_sparse_matrix(const SparseMatrix<double>& A) {
    int n = A.rows();
    MatrixXd denseA = MatrixXd(A);
    Eigen::LLT<MatrixXd> lltA(denseA);
    MatrixXd denseInv;
    if(lltA.info() == Eigen::Success) {
        denseInv = lltA.solve(MatrixXd::Identity(denseA.rows(), denseA.cols()));
    } else {
        denseInv = denseA.inverse();
    }

    MatrixXd temp = 0.5 * (denseInv + denseInv.transpose());
    for (int i = 0; i < temp.rows(); ++i)
        if (temp(i, i) < 1e-12) temp(i, i) = 1e-12;

    return temp.sparseView();
}
