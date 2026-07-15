#include "matrix_io.h"
#include "mmap_matrix.h"
#include "logger.h"
#include "upper_packed_binary.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace cosmic {

using namespace std;
using namespace Eigen;

// ── Matrix sign correction ───────────────────────────────────────

void fixMatrixSigns(SparseMatrix<double>& A, const string& filename) {
    long long off_pos = 0;
    long long off_neg = 0;
    long long total_off = 0;

    for (int k=0; k<A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it) {
            if (it.row() != it.col()) {
                total_off++;
                if (it.value() > 0) off_pos++;
                else if (it.value() < 0) off_neg++;
            }
        }
    }

    LOG_DEBUG("Matrix sign check: TotalOff=" << total_off << ", Pos=" << off_pos << ", Neg=" << off_neg);

    if (total_off > 0) {
        LOG_DEBUG("Positive Ratio = " << (double)off_pos / total_off);
    }

    string lower_name = filename;
    transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    bool is_Ainv = (lower_name.find("ainv") != string::npos && lower_name.find("ginv") == string::npos);
    bool is_Ginv = (lower_name.find("ginv") != string::npos);

    if (!is_Ginv && total_off > 0 && off_pos > off_neg && (double)off_pos / total_off > 0.6) {
        LOG_WARN("Suspicious matrix sign structure detected (" << off_pos << " positive off-diagonals vs " << off_neg << " negative off-diagonals).");
        LOG_WARN("Likely HIBLUP binary format with inverted signs. Auto-flipping off-diagonal signs to enforce M-matrix property.");
        for (int k=0; k<A.outerSize(); ++k) {
            for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it) {
                if (it.row() != it.col()) {
                    it.valueRef() = -it.value();
                }
            }
        }
    } else if (is_Ainv && total_off > 0 && off_pos > 0) {
        double ratio = (double)off_pos / total_off;
        if (ratio > 0.001) {
             LOG_WARN("A-inverse contains " << off_pos << " (" << (ratio*100) << "%) positive off-diagonals.");
             LOG_WARN("Keeping them as is (assuming valid mate connections). If variance is inflated, check matrix scaling.");
        }
     }
}

// ── Binary matrix reading ────────────────────────────────────────

static SparseMatrix<double> dense_to_sparse(const MatrixXd& dense, double abs_zero_threshold) {
    vector<Triplet<double>> triplets;
    for (int i = 0; i < dense.rows(); ++i) {
        for (int j = 0; j < dense.cols(); ++j) {
            if (fabs(dense(i, j)) > abs_zero_threshold) {
                triplets.emplace_back(i, j, dense(i, j));
            }
        }
    }
    SparseMatrix<double> A(dense.rows(), dense.cols());
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

static bool has_upper_packed_magic(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) return false;
    char magic[32] = {};
    file.read(magic, sizeof(magic));
    return file.gcount() == static_cast<streamsize>(sizeof(magic)) &&
           std::strncmp(magic, "COSMIC_UPPER_PACKED", 19) == 0;
}

static SparseMatrix<double> readInvMatrixBinary(const string& filename,
                                                double abs_zero_threshold,
                                                bool apply_sign_fix) {
    if (has_upper_packed_magic(filename)) {
        LOG_INFO("Detected COSMIC_UPPER_PACKED binary matrix with metadata header.");
        SparseMatrix<double> A = dense_to_sparse(io::read_upper_packed_binary(filename), abs_zero_threshold);
        if (apply_sign_fix) fixMatrixSigns(A, filename);
        return A;
    }

    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open binary matrix file: " + filename);

    file.seekg(0, ios::end);
    size_t fsize = file.tellg();
    file.seekg(0, ios::beg);
    LOG_DEBUG("Reading binary matrix, size=" << fsize << " bytes.");

    size_t entries_dense = fsize / 4;
    double disc = 1.0 + 8.0 * entries_dense;
    double sqrt_disc = sqrt(disc);
    long long n_dense = -1;
    if (abs(sqrt_disc - round(sqrt_disc)) < 1e-9) {
        long long root = (long long)round(sqrt_disc);
        if ((root - 1) % 2 == 0) {
            n_dense = (root - 1) / 2;
        }
    }

    if (n_dense > 0 && fsize == n_dense * (n_dense + 1) * 2) {
         LOG_INFO("Detected Binary Dense Format (Lower Triangle, float), dim=" << n_dense);
         vector<float> buf(entries_dense);
         file.read(reinterpret_cast<char*>(buf.data()), fsize);
         vector<Triplet<double>> triplets;
         triplets.reserve(entries_dense * 2);
         long long idx = 0;
         for (int i = 0; i < n_dense; ++i) {
             for (int j = 0; j <= i; ++j) {
                 float val = buf[idx++];
                 if (val != 0.0) {
                     triplets.emplace_back(i, j, (double)val);
                     if (i != j) triplets.emplace_back(j, i, (double)val);
                 }
             }
         }
         SparseMatrix<double> A(n_dense, n_dense);
         A.setFromTriplets(triplets.begin(), triplets.end());
         A.makeCompressed();
         if (apply_sign_fix) fixMatrixSigns(A, filename);
         return A;
    }

    size_t record_size = 12;
    if (fsize % record_size != 0) {
        LOG_WARN("Binary file size is not multiple of 12 bytes. Trying anyway.");
    }
    size_t n_entries = fsize / record_size;
    vector<Triplet<double>> triplets;
    triplets.reserve(n_entries);

    int max_idx = 0;
    int min_idx = std::numeric_limits<int>::max();

    const size_t chunk_size = 1024 * 1024;
    vector<char> buffer(chunk_size * record_size);

    while (file) {
        file.read(buffer.data(), buffer.size());
        streamsize bytes_read = file.gcount();
        if (bytes_read == 0) break;

        size_t count = bytes_read / record_size;
        for (size_t i = 0; i < count; ++i) {
            const char* ptr = buffer.data() + i * record_size;
            int r = *reinterpret_cast<const int*>(ptr);
            int c = *reinterpret_cast<const int*>(ptr + 4);
            float v = *reinterpret_cast<const float*>(ptr + 8);

            if (r > max_idx) max_idx = r;
            if (c > max_idx) max_idx = c;
            if (r < min_idx) min_idx = r;
            if (c < min_idx) min_idx = c;

            if (abs(v) > 1e-9) {
                triplets.emplace_back(r, c, (double)v);
            }
        }
    }

    LOG_DEBUG("Binary Matrix Range: [" << min_idx << ", " << max_idx << "]");

    if (min_idx == 1) {
        LOG_DEBUG("Detected 1-based indexing in binary matrix. Shifting to 0-based.");
        vector<Triplet<double>> new_triplets;
        new_triplets.reserve(triplets.size());
        for(const auto& t : triplets) {
            new_triplets.emplace_back(t.row()-1, t.col()-1, t.value());
        }
        triplets = std::move(new_triplets);
        max_idx--;
    }

    int dim = max_idx + 1;

    SparseMatrix<double> A(dim, dim);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    bool has_upper = false;
    bool has_lower = false;
    for (int k=0; k<A.outerSize(); ++k) {
        for (SparseMatrix<double>::InnerIterator it(A,k); it; ++it) {
            if (it.row() < it.col()) has_upper = true;
            if (it.row() > it.col()) has_lower = true;
        }
    }

    if (has_upper != has_lower) {
        LOG_DEBUG("Auto-symmetrizing binary matrix (" << (has_lower ? "lower" : "upper") << " triangle detected) - MODIFIED.");
        vector<Triplet<double>> sym_triplets = triplets;
        for(const auto& t : triplets) {
            if (t.row() != t.col()) {
                sym_triplets.emplace_back(t.col(), t.row(), t.value());
            }
        }
        SparseMatrix<double> Asym(dim, dim);
        Asym.setFromTriplets(sym_triplets.begin(), sym_triplets.end());
        Asym.makeCompressed();
        if (apply_sign_fix) fixMatrixSigns(Asym, filename);
        return Asym;
    }

    if (apply_sign_fix) fixMatrixSigns(A, filename);
    return A;
}

// ── Text matrix reading ──────────────────────────────────────────

SparseMatrix<double> readInvMatrix(const string& filename, double abs_zero_threshold, bool apply_sign_fix) {
    if (filename.size() >= 4 && filename.substr(filename.size()-4) == ".bin") {
        return readInvMatrixBinary(filename, abs_zero_threshold, apply_sign_fix);
    }
    ifstream file(filename);
    if (!file.is_open()) throw runtime_error("Cannot open matrix file: " + filename);

    // Helper: trim a string
    auto trim_copy_local = [](const string& s) -> string {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == string::npos) return string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };

    string line;
    vector<string> first_tokens;
    size_t n_lines = 0;

    while (getline(file, line)) {
        string row = trim_copy_local(line);
        if (row.empty()) continue;
        if (n_lines == 0) {
            if (row.find(',') != string::npos) {
                string tok; stringstream ss(row);
                while (getline(ss, tok, ',')) first_tokens.push_back(trim_copy_local(tok));
            } else {
                stringstream ss(row); string tok;
                while (ss >> tok) first_tokens.push_back(tok);
            }
        }
        n_lines++;
    }

    file.clear();
    file.seekg(0, ios::beg);

    size_t ncols = first_tokens.size();
    bool is_dense = (ncols == n_lines);

    vector<Triplet<double>> triplets;

    if (is_dense) {
        int i = 0;
        while (getline(file, line)) {
            string row = trim_copy_local(line);
            if (row.empty()) continue;
            vector<string> tokens;
            if (row.find(',') != string::npos) {
                string tok; stringstream ss(row);
                while (getline(ss, tok, ',')) tokens.push_back(trim_copy_local(tok));
            } else {
                stringstream ss(row); string tok;
                while (ss >> tok) tokens.push_back(tok);
            }
            if (tokens.size() != ncols) throw runtime_error("Dense matrix: inconsistent columns at row " + to_string(i+1));
            for (size_t j = 0; j < tokens.size(); ++j) {
                double v = stod(tokens[j]);
                if (fabs(v) > abs_zero_threshold) triplets.emplace_back(i, (int)j, v);
            }
            ++i;
        }
        SparseMatrix<double> A((int)ncols, (int)ncols);
        A.setFromTriplets(triplets.begin(), triplets.end());
        A.makeCompressed();
        if (apply_sign_fix) fixMatrixSigns(A, filename);
        return A;
    } else if (ncols == 3) {
        int min_idx_found = 1e9;
        int max_idx_found = -1e9;

        while (getline(file, line)) {
            string row = trim_copy_local(line);
            if (row.empty()) continue;
            stringstream ss(row);
            double r_d, c_d, v;
            if (row.find(',') != string::npos) {
                string tok; vector<string> toks; stringstream ss2(row);
                while (getline(ss2, tok, ',')) toks.push_back(trim_copy_local(tok));
                if (toks.size() < 3) continue;
                r_d = stod(toks[0]); c_d = stod(toks[1]); v = stod(toks[2]);
            } else {
                ss >> r_d >> c_d >> v;
            }

            int r = (int)r_d; int c = (int)c_d;
            if (r < min_idx_found) min_idx_found = r;
            if (c < min_idx_found) min_idx_found = c;
            if (r > max_idx_found) max_idx_found = r;
            if (c > max_idx_found) max_idx_found = c;

            triplets.emplace_back(r, c, v);
        }

        int dim = 0;

        file.clear();
        file.seekg(0, ios::beg);
        triplets.clear();

        int offset = (min_idx_found == 0) ? 0 : 1;
        dim = (min_idx_found == 0) ? max_idx_found + 1 : max_idx_found;

        while (getline(file, line)) {
            string row = trim_copy_local(line);
            if (row.empty()) continue;
            double r_d, c_d, v;
            if (row.find(',') != string::npos) {
                string tok; vector<string> toks; stringstream ss2(row);
                while (getline(ss2, tok, ',')) toks.push_back(trim_copy_local(tok));
                if (toks.size() < 3) continue;
                r_d = stod(toks[0]); c_d = stod(toks[1]); v = stod(toks[2]);
            } else {
                stringstream ss(row); ss >> r_d >> c_d >> v;
            }
            int r = (int)r_d - offset;
            int c = (int)c_d - offset;
            if (fabs(v) > abs_zero_threshold) {
                triplets.emplace_back(r, c, v);
            }
        }

        bool has_upper = false;
        bool has_lower = false;
        for(const auto& t : triplets) {
            if (t.row() < t.col()) has_upper = true;
            if (t.row() > t.col()) has_lower = true;
            if (has_upper && has_lower) break;
        }

        if (has_upper != has_lower) {
            size_t n = triplets.size();
            for(size_t i=0; i<n; ++i) {
                if (triplets[i].row() != triplets[i].col()) {
                    triplets.emplace_back(triplets[i].col(), triplets[i].row(), triplets[i].value());
                }
            }
            LOG_DEBUG("Auto-symmetrizing matrix (" << (has_lower ? "lower" : "upper") << " triangle detected).");
        }

        SparseMatrix<double> A(dim, dim);
        A.setFromTriplets(triplets.begin(), triplets.end());

        A.makeCompressed();
        if (apply_sign_fix) fixMatrixSigns(A, filename);
        return A;

    } else {
        throw runtime_error("Unrecognized matrix format: cols=" + to_string(ncols) + ", lines=" + to_string(n_lines) +
                            ". Supported: dense (rows=cols) or triplet (row, col, val).");
    }
}

AbstractMatrix* readInvMatrixAbstract(const string& filename, bool use_mmap) {
    if (use_mmap) {
        auto* mmap_mat = new MMapMatrix();
        mmap_mat->open(filename);
        mmap_mat->scan_dim();
        return new MMapMatrixAdapter(mmap_mat);
    } else {
        auto sparse_mat = readInvMatrix(filename);
        return new SparseMatrixAdapter(sparse_mat);
    }
}

// ── ID list reading ──────────────────────────────────────────────

map<string,int> readIdList(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) throw runtime_error("Cannot open ID list: " + filename);
    map<string,int> idmap; string line; int idx = 1;
    auto trim_copy_local = [](const string& s) -> string {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == string::npos) return string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    while (getline(f, line)) {
        string id = trim_copy_local(line);
        if (id.empty()) continue;
        idmap[id] = idx; ++idx;
    }
    return idmap;
}

// ── Variance reading ─────────────────────────────────────────────

pair<double,double> readVars(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) throw runtime_error("Cannot open variance file: " + filename);
    string line;
    double vu = 0, ve = 0;
    bool found_u = false, found_e = false;
    auto trim_copy_local = [](const string& s) -> string {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == string::npos) return string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    while (getline(f, line)) {
        string s = trim_copy_local(line);
        if (s.empty()) continue;
        stringstream ss(s);
        string key;
        ss >> key;
        string klow = key; transform(klow.begin(), klow.end(), klow.begin(), ::tolower);
        if (klow == "item" || klow == "var") continue;

        double val = 0;
        if (!(ss >> val)) continue;

        if (klow == "e" || klow == "error" || klow == "residual") {
            ve = val; found_e = true;
        } else {
            if (!found_u) {
                vu = val; found_u = true;
            }
        }
    }

    if (!found_u && !found_e) {
        f.clear(); f.seekg(0);
        string k1, k2; double v1 = 0, v2 = 0;
        if (f >> k1 >> v1) {
             string klow = k1; transform(klow.begin(), klow.end(), klow.begin(), ::tolower);
             if (klow == "e") { ve = v1; found_e = true; } else { vu = v1; found_u = true; }
        }
        if (f >> k2 >> v2) {
             string klow = k2; transform(klow.begin(), klow.end(), klow.begin(), ::tolower);
             if (klow == "e") { ve = v2; found_e = true; } else { if (!found_u) { vu = v2; found_u = true; } }
        }
    }

    if (vu == 0 && ve == 0) throw runtime_error("Cannot parse variance components from: " + filename);

    return {vu, ve};
}

// ── Matrix writing ───────────────────────────────────────────────

void write_matrix_txt(const string& path, const Eigen::MatrixXd& M, const vector<string>& ids) {
    ofstream fout(path);
    if (!fout) throw runtime_error("Cannot write matrix file: " + path);
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            fout << (i + 1) << "\t" << (j + 1) << "\t" << M(i, j) << "\n";
        }
    }
}

void write_matrix_bin(const string& bin_path, const string& id_path,
                      const Eigen::MatrixXd& M, const vector<string>& ids) {
    ofstream fid(id_path);
    if (fid) { for (const auto& id : ids) fid << id << "\n"; }
    io::write_upper_packed_binary(bin_path, M);
}

// ── Matrix inversion with eigen-bending ──────────────────────────

Eigen::MatrixXd invert_psd_matrix(Eigen::MatrixXd M, double ridge) {
    if (ridge > 0.0) M.diagonal().array() += ridge;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
    auto eval = es.eigenvalues();
    auto U = es.eigenvectors();
    double maxeig = eval.maxCoeff();
    double eps = std::max(1e-12 * maxeig, 1e-6);
    for (int i = 0; i < eval.size(); ++i) {
        if (eval(i) < eps) eval(i) = eps;
    }
    Eigen::MatrixXd Dinv = Eigen::MatrixXd::Zero(M.rows(), M.cols());
    for (int i = 0; i < eval.size(); ++i) Dinv(i, i) = 1.0 / eval(i);
    return U * Dinv * U.transpose();
}

bool factorize_with_jitter(const Eigen::MatrixXd& matrix,
                           Eigen::LDLT<Eigen::MatrixXd>& solver) {
    const double jitter_values[] = {0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2};
    for (double jitter : jitter_values) {
        Eigen::MatrixXd work = matrix;
        if (jitter > 0.0) work.diagonal().array() += jitter;
        solver.compute(work);
        if (solver.info() == Eigen::Success) {
            Eigen::VectorXd diag = solver.vectorD();
            if (diag.cwiseAbs().minCoeff() > 1e-15) return true;
        }
    }
    return false;
}

} // namespace cosmic
