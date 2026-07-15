#pragma once
#include <random>
#include <vector>
#include <cmath>
#include <chrono>

namespace cosmic {

class RNG {
private:
    std::mt19937_64 gen;
public:
    RNG() {
        // Seed with high resolution clock
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        gen.seed(seed);
    }

    explicit RNG(unsigned long seed) {
        gen.seed(seed);
    }

    // Generate standard normal distribution N(0, 1)
    double normal() {
        std::normal_distribution<double> d(0.0, 1.0);
        return d(gen);
    }

    // Fill a vector with standard normal random numbers
    void fill_normal(std::vector<double>& v) {
        std::normal_distribution<double> d(0.0, 1.0);
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = d(gen);
        }
    }

    // Fill Eigen Vector
    template<typename Derived>
    void fill_normal(Eigen::PlainObjectBase<Derived>& v) {
        std::normal_distribution<double> d(0.0, 1.0);
        for (int i = 0; i < v.size(); ++i) {
            v(i) = d(gen);
        }
    }

    // Fill Eigen Vector with Rademacher distribution (+1/-1)
    template<typename Derived>
    void fill_rademacher(Eigen::PlainObjectBase<Derived>& v) {
        std::uniform_int_distribution<int> d(0, 1);
        for (int i = 0; i < v.size(); ++i) {
            v(i) = (d(gen) == 0) ? -1.0 : 1.0;
        }
    }

    // Chi-square distribution
    double chi_square(double df) {
        std::chi_squared_distribution<double> d(df);
        return d(gen);
    }
};

}
