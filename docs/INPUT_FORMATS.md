# Input Formats

## Pedigree

Whitespace-delimited text with a header and three columns identifying individual, sire, and dam. Missing parents are represented by `0`. The accepted parent-column order is detected from the header.

## Phenotype

Whitespace- or tab-delimited text with a header. The ID column defaults to the recognized ID field and may be selected with `--id-col`. Traits and covariates are selected by name.

## Genotype

- PLINK 1 binary: matching `.bed`, `.bim`, and `.fam` files supplied as `--bfile <prefix>`.
- PLINK 2: matching `.pgen`, `.pvar`, and `.psam` supplied as `--pfile <prefix>`.
- BGEN: supported by QC and selected STCG paths.

## Relationship Matrices

Text matrices are dense square numeric tables or three-column one-based numeric triplets. IDs are supplied separately. Binary matrices use the versioned `COSMIC_UPPER_PACKED` header and float64 upper-triangle payload.

## Variance Components

Variance files contain component labels and numeric values. Fixed-variance CLI runs should prefer explicit `--var-a` and `--var-e`.
