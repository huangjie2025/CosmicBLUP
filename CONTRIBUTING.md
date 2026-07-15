# Contributing

Build and run all tests before opening a change:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Keep application code in the responsibility-based `src/` directories. Add new `src/core` sources explicitly to the root `CMakeLists.txt`; source globbing is intentionally not used. Tests must use repository fixtures, write only below the build tree, and must not silently skip required coverage.
