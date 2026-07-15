# CLI Reference

## Task Selection

| Option | Purpose |
|---|---|
| `--blup` | Run the BLUP pipeline |
| `--make-matrix` | Export an A, G, or H relationship matrix |
| `--make-inv` | Export a relationship inverse |
| `--qc` | Run genotype quality control |
| `--sblup` | Run summary-level BLUP |
| `--pred` | Score individuals from SNP effects |
| `--single-trait` | Select the single-trait pipeline |
| `--multi-trait` | Select the multi-trait pipeline |
| `--rrm` | Select random regression |

## Main Inputs

| Option | Meaning |
|---|---|
| `--ped <file>` | Pedigree table |
| `--pheno <file>` | Phenotype table |
| `--pheno-name <name[,name]>` | Trait column or multi-trait columns |
| `--bfile <prefix>` | BED/BIM/FAM prefix |
| `--pfile <prefix>` | PGEN/PVAR/PSAM prefix |
| `--bgen <file>` | BGEN input for supported tasks |
| `--matrix <file>` | External relationship matrix |
| `--matrix-id <file>` | IDs corresponding to an external matrix |
| `--inv <file>` | External inverse matrix |
| `--inv-id <file>` | IDs corresponding to an inverse matrix |

## Model and Effects

`--model` accepts `pblup`, `gblup`, `ssgblup`, `repeatability`, or `rrm`. Use `--dcovar` for categorical covariates, `--qcovar` for quantitative covariates, `--pe` for permanent environment, and `--rand` or `--mat` for additional random components.

## Variance Estimation

`--vce-mode` accepts `ai`, `em`, `mc`, `he`, `emai`, `hi`, `exact`, `vmatrix`, `stcg`, or `fdiff`. Control iteration and trace behavior with `--vce-max-iter`, `--vce-tol`, `--vce-samples`, and `--trace-mode`.

To bypass VCE, use `--skip-vce` with both `--var-a` and `--var-e`.

## Output and Runtime

`--out <prefix>` is required for analysis tasks. `--threads` must be positive. Unknown options and missing option values are errors.
