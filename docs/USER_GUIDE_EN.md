# CosmicBLUP User Guide

CosmicBLUP implements BLUP-focused breeding-value workflows without Bayesian regression modules.

## PBLUP

```bash
cosmicblup --blup --model pblup --ped pedigree.txt \
  --pheno phenotype.txt --pheno-name trait \
  --vce --vce-mode ai --out results/pblup
```

## GBLUP and ssGBLUP

Use `--bfile` for PLINK BED or `--pfile` for PGEN. Add `--ped` and select `--model ssgblup` for single-step analysis. Explicit BGEN GBLUP is not currently supported; BGEN is available for QC and selected STCG workflows.

## Fixed Variances

Use `--skip-vce --var-a <positive> --var-e <positive>` to solve with supplied variance components.

## Longitudinal Models

`--pe` adds a permanent environmental component. Random regression uses `--rrm --time-col <column> --rrm-order <order>` and writes coefficient, covariance, and trajectory outputs.

## Multi-Trait Analysis

Pass comma-separated trait names with `--multi-trait`. The current implementation targets dense GBLUP and complete observations.

## Relationship Matrices

`--make-matrix` and `--make-inv` support A, G, and H relationships. Text output uses one-based numeric triplets plus an ID file. Binary output uses the versioned `COSMIC_UPPER_PACKED` format.

See `CLI_REFERENCE.md`, `INPUT_FORMATS.md`, and `OUTPUT_FORMATS.md` for details.
