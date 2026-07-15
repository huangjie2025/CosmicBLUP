#pragma once

namespace cosmic {

// Calculate Hardy-Weinberg Equilibrium p-value (Exact Test)
// obs_hom1: Observed count of first homozygote (e.g. AA)
// obs_hets: Observed count of heterozygote (e.g. AB)
// obs_hom2: Observed count of second homozygote (e.g. BB)
double calculateHWE(long long obs_hom1, long long obs_hets, long long obs_hom2);

// Calculate Minor Allele Frequency
// Returns the frequency of the less common allele (0.0 to 0.5)
double calculateMAF(long long obs_hom1, long long obs_hets, long long obs_hom2);

} // namespace cosmic
