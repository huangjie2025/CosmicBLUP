# Repeatability and RRM toy example

Small pedigree and longitudinal phenotype example for CosmicBLUP.

Repeatability:

```bash
cosmicblup --single-trait --model repeatability \
  --ped repeatability_demo.ped.txt \
  --pheno repeatability_demo.phe.txt \
  --pheno-name milk \
  --dcovar season \
  --qcovar age \
  --pe \
  --vce --vce-mode em \
  --out repeatability
```

RRM:

```bash
cosmicblup --rrm \
  --ped repeatability_demo.ped.txt \
  --pheno repeatability_demo.phe.txt \
  --pheno-name milk \
  --dcovar season \
  --qcovar age \
  --time-col age \
  --rrm-order 1 \
  --vce --vce-mode exact \
  --out rrm
```
