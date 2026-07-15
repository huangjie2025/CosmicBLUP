# Architecture

The application source tree is organized by responsibility:

```text
src/
  app/       entry point, CLI, orchestration, runtime setup
  config/    user configuration and generated version interface
  core/      numerical methods, genotype readers, matrices, MME, and VCE kernels
  io/        phenotype loading and relationship construction
  model/     model pipeline and random regression setup
  variance/  variance-component dispatch and result types
  solve/     mixed-model solving and output writing
  tasks/     independent QC, SBLUP, and prediction tasks
```

The numerical and genomics implementation is fully integrated under `src/core/` and built as the internal `cosmicblup_core` target. There is no submodule or nested project boundary. Unrelated GWAS, conversion, and unused matrix modules are intentionally excluded.

The stable runtime pipeline is:

1. Parse and validate CLI options.
2. Build or load relationship structures.
3. Load phenotype records and fixed effects.
4. Configure model-specific random components.
5. Estimate variance components when enabled.
6. Solve mixed-model equations.
7. Write model-specific outputs.
