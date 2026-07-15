#pragma once
#include <Eigen/Core>
#include <functional>
#include <stdexcept>
#include <cstdio>

namespace cosmic {

class MatrixFreePCG {
public:
    using MultOp = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    using PrecondOp = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

    MatrixFreePCG(int n_size) : n(n_size) {}

    void setMultOp(MultOp op) { mult_op = op; }
    void setPrecondOp(PrecondOp op) { precond_op = op; }
    void setQuiet(bool q) { quiet = q; }

    Eigen::VectorXd solve(const Eigen::VectorXd& b, double tol = 1e-8, int max_iter = 1000, const Eigen::VectorXd& initial_guess = Eigen::VectorXd()) {
        if (!mult_op) throw std::runtime_error("MatrixFreePCG: mult_op not set");

        Eigen::VectorXd x;
        if (initial_guess.size() == n) {
            x = initial_guess;
        } else {
            x = Eigen::VectorXd::Zero(n);
        }

        Eigen::VectorXd Ax(n);
        mult_op(x, Ax);
        Eigen::VectorXd r = b - Ax; // r = b - A*x

        if (r.norm() < 1e-20) return x; // Already solved

        Eigen::VectorXd z(n);
        if (precond_op) precond_op(r, z);
        else z = r;

        Eigen::VectorXd p = z;
        double rsold = r.dot(z);

        for (int i = 0; i < max_iter; ++i) {
            Eigen::VectorXd Ap(n);
            mult_op(p, Ap);

            double alpha = rsold / p.dot(Ap);
            x += alpha * p;
            r -= alpha * Ap;

            if (r.norm() < tol) {
                if (!quiet) printf("PCG Converged at iter %d, residual %g\n", i, r.norm());
                return x;
            }

            if (precond_op) precond_op(r, z);
            else z = r;

            double rsnew = r.dot(z);
            p = z + (rsnew / rsold) * p;
            rsold = rsnew;
        }

        if (!quiet) printf("PCG Reached max iter %d, residual %g\n", max_iter, r.norm());
        return x;
    }

private:
    int n;
    MultOp mult_op;
    PrecondOp precond_op;
    bool quiet = false;
};

} // namespace cosmic
