# Minimal pBLUP Example

Run from the CosmicBLUP source tree:

```bash
./build/cosmicblup \
  --blup \
  --model pblup \
  --ped tests/fixtures/pblup_minimal/pedigree.txt \
  --pheno tests/fixtures/pblup_minimal/pheno.txt \
  --pheno-name trait \
  --vars tests/fixtures/pblup_minimal/vars.txt \
  --out run/pblup \
  --threads 1 \
  --no-se
```

Expected output files:

```text
run/pblup.beta
run/pblup.rand
run/pblup.anova
run/pblup.log
```

This example uses provided variance components. Omitting `--vars` runs the current AI-REML VCE path and also writes `pblup.vars`.
