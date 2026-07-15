# Minimal PBLUP Example

From an installed CosmicBLUP package root:

```bash
./bin/cosmicblup --blup --model pblup \
  --ped examples/pblup_minimal/pedigree.txt \
  --pheno examples/pblup_minimal/pheno.txt \
  --pheno-name trait --vars examples/pblup_minimal/vars.txt \
  --threads 1 --no-se --out run/pblup
```
