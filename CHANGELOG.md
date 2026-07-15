# Changelog

## 1.0.0 - 2026-07-15

- Established CosmicBLUP as a BLUP-only project derived from the CosmicSolver BLUP pipeline.
- Removed all Bayes and variational Bayes source, dispatch, tests, and documentation.
- Renamed the executable to `cosmicblup`.
- Reorganized application sources by responsibility: app, config, IO, model, variance, solve, and tasks.
- Integrated the required numerical and genomics core directly under `src/core`.
- Removed unrelated core conversion, GWAS-support, and unused matrix modules.
- Added repository-owned portable fixtures and a cross-platform Python test runner.
- Fixed unknown-option handling, multi-trait forced VCE, nested output directories, platform detection, and text matrix round trips.
- Added Linux, macOS, and Windows packaging definitions.
