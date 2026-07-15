# Release Process

1. Confirm `VERSION`, CMake project version, package metadata, and `cosmicblup --version` agree.
2. Configure a fresh Release build with BLAS/LAPACK disabled for portable baseline binaries.
3. Run all CTest cases.
4. Install into a clean staging directory.
5. Run staged `--version`, `--help`, and minimal PBLUP checks.
6. Generate `release_manifest.json`, archive the staged tree, and write SHA-256 checksums.
7. Test the archive in a clean environment without source-tree library paths.
8. Review bundled runtime licenses before public binary distribution.

Use `scripts/package_linux.sh`, `scripts/package_macos.sh`, or `scripts/package_windows.ps1`. Generated artifacts are placed under `dist/` and are never source-controlled.
