# CosmicBLUP

CosmicBLUP is a focused command-line toolkit for best linear unbiased prediction in animal and plant breeding. It provides pedigree, genomic, single-step, longitudinal, random-regression, and multi-trait workflows in one executable without Bayesian regression modules.

Current version: **1.0.0**

## Supported Models

- PBLUP with pedigree relationship matrices
- GBLUP with PLINK BED or PGEN genotype input
- ssGBLUP with pedigree and genotyped subsets
- Repeatability models with permanent environmental effects
- Random regression models using Legendre bases
- Multi-trait dense GBLUP
- AI-REML, EM-REML, HE, Monte Carlo, EMAI, HI, STCG, and related VCE strategies
- A, G, and H relationship matrix construction and inversion
- Genotype QC, summary-level BLUP, and score prediction

## Build

Dependencies are CMake 3.15+, a C++17 compiler, Eigen3, zlib, and zstd. OpenMP is optional but recommended.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Quick Start

```bash
./build/cosmicblup \
  --blup --model pblup \
  --ped tests/fixtures/pblup_minimal/pedigree.txt \
  --pheno tests/fixtures/pblup_minimal/pheno.txt \
  --pheno-name trait \
  --vce --vce-mode ai \
  --out run/pblup
```

The parent directory of `--out` is created automatically.

## Documentation

- [Chinese user guide](docs/USER_GUIDE_CN.md)
- [English user guide](docs/USER_GUIDE_EN.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [Model support](docs/MODEL_SUPPORT.md)
- [Input formats](docs/INPUT_FORMATS.md)
- [Output formats](docs/OUTPUT_FORMATS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Validation](docs/VALIDATION.md)
- [Release process](docs/RELEASE.md)

## License

CosmicBLUP is distributed under the MIT License. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
