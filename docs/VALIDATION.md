# Validation

The default CTest suite is self-contained and uses only repository-owned synthetic fixtures. It covers:

- CLI version, help, unknown arguments, and missing values
- Text and binary relationship matrix export and reuse
- PBLUP, GBLUP, and ssGBLUP fixed-variance solves
- Repeatability with permanent environment
- Random regression outputs
- Multi-trait GBLUP VCE and solve
- PLINK QC, SBLUP, and score prediction

Run:

```bash
ctest --test-dir build --output-on-failure
```

Scalability, PGEN/BGEN breadth, external software comparisons, and large STCG memory tests are separate validation activities and are not implied by the default smoke suite.
