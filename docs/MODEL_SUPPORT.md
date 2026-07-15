# Model Support

| Model | Pedigree | BED | PGEN | BGEN | VCE | Status |
|---|---:|---:|---:|---:|---:|---|
| PBLUP | Yes | No | No | No | Yes | Stable |
| GBLUP | No | Yes | Yes | STCG only | Yes | Stable for BED/PGEN |
| ssGBLUP | Yes | Yes | Yes | No | Yes | Stable |
| Repeatability | Yes | Yes | Yes | STCG only | Yes | Stable |
| Random regression | Yes | Limited | Limited | No | Yes | Additive plus optional Pe |
| Multi-trait GBLUP | No | Yes | Limited | No | Required | Dense complete-case workflow |
| SBLUP | No | Yes | Yes | No | No | Supported |
| Prediction | No | Yes | Yes | No | No | Supported |

STCG currently supports genomic matrix-free workflows and optional permanent environment; it is not a general replacement for pedigree, single-step, or random-regression solvers.
