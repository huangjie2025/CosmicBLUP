#include "qc_utils.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cosmic {

// Wigginton et al. 2005 implementation of exact test for HWE
double calculateHWE(long long obs_hom1, long long obs_hets, long long obs_hom2) {
    // Ensure hom1 is the rare homozygote for symmetry in calculation logic
    // (Though the formula is symmetric, standardizing helps debugging)
    long long obs_homc = obs_hom1 < obs_hom2 ? obs_hom2 : obs_hom1;
    long long obs_homr = obs_hom1 < obs_hom2 ? obs_hom1 : obs_hom2;

    long long rare_copies = 2 * obs_homr + obs_hets;
    long long genotypes = obs_hom1 + obs_hets + obs_hom2;

    if (genotypes == 0) return 1.0;

    // Determine the midpoint (mode of the distribution)
    long long mid = rare_copies * (2 * genotypes - rare_copies) / (2 * genotypes);
    if ((mid % 2) != (rare_copies % 2)) mid++;

    std::vector<double> probs(rare_copies + 1, 0.0);

    probs[mid] = 1.0;
    double mysum = 1.0;

    // Calculate probabilities from midpoint down
    long long curr_hets = mid;
    long long curr_homr = (rare_copies - mid) / 2;
    long long curr_homc = genotypes - curr_hets - curr_homr;

    for (curr_hets = mid; curr_hets >= 2; curr_hets -= 2) {
        probs[curr_hets - 2] = probs[curr_hets] * curr_hets * (curr_hets - 1.0)
                             / (4.0 * (curr_homr + 1.0) * (curr_homc + 1.0));
        mysum += probs[curr_hets - 2];

        // 2 fewer hets -> 1 more homr, 1 more homc
        curr_homr++;
        curr_homc++;
    }

    // Calculate probabilities from midpoint up
    curr_hets = mid;
    curr_homr = (rare_copies - mid) / 2;
    curr_homc = genotypes - curr_hets - curr_homr;

    for (curr_hets = mid; curr_hets <= rare_copies - 2; curr_hets += 2) {
        probs[curr_hets + 2] = probs[curr_hets] * 4.0 * curr_homr * curr_homc
                             / ((curr_hets + 2.0) * (curr_hets + 1.0));
        mysum += probs[curr_hets + 2];

        // 2 more hets -> 1 fewer homr, 1 fewer homc
        curr_homr--;
        curr_homc--;
    }

    // P-value calculation
    double target = probs[obs_hets];
    double p_hwe = 0.0;

    // Only sum probabilities for valid heterozygote counts (same parity as rare_copies)
    // Also, due to floating point precision, we use a small epsilon for comparison
    // But Wigginton code simply uses <= target.
    // However, probs are relative.

    for (int i = 0; i <= rare_copies; i++) {
        // Skip invalid heterozygote counts (must have same parity as rare_copies)
        if ((i % 2) != (rare_copies % 2)) continue;

        if (probs[i] <= target) {
            p_hwe += probs[i];
        }
    }

    return std::min(1.0, p_hwe / mysum);
}

double calculateMAF(long long obs_hom1, long long obs_hets, long long obs_hom2) {
    long long total_alleles = 2 * (obs_hom1 + obs_hets + obs_hom2);
    if (total_alleles == 0) return 0.0;

    long long count1 = 2 * obs_hom1 + obs_hets;
    double freq1 = (double)count1 / total_alleles;

    return std::min(freq1, 1.0 - freq1);
}

} // namespace cosmic
