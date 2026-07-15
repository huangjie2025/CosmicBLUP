# Output Formats

| Suffix | Contents |
|---|---|
| `.log` | Command, environment, progress, and diagnostics |
| `.vars` | Estimated variance components and standard errors when available |
| `.beta` | Fixed-effect estimates |
| `.rand` | Individual random effects and breeding values |
| `.obsrand` | Observation-level fitted values and residuals |
| `.anova` | Model summary statistics |
| `.mv.vars` | Multi-trait genetic and residual covariance matrices |
| `.rrcoef` | Random-regression coefficients |
| `.rrk` | Random-regression covariance estimates |
| `.rrcurve` | Predicted trajectories over observed times |
| `.rrm.meta.txt` | Random-regression metadata |
| `.sblup.eff` | SBLUP-adjusted SNP effects |
| `.profile` | Individual prediction scores |

Relationship exports use model-specific suffixes such as `.PA`, `.GA`, `.HA`, and their inverse forms, followed by `.txt` or `.bin`, with a companion `.id` file.
