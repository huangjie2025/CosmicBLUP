#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${ROOT}/VERSION")"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;;
esac

BUILD="${ROOT}/build-package-linux"
DIST="${ROOT}/dist"
NAME="CosmicBLUP-${VERSION}-linux-${ARCH}"
STAGE="${DIST}/${NAME}"

GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
fi

cmake -E remove_directory "${BUILD}"
cmake -S "${ROOT}" -B "${BUILD}" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCMAKE_INSTALL_RPATH=\$ORIGIN/../lib \
  -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_LAPACK=ON
cmake --build "${BUILD}" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
ctest --test-dir "${BUILD}" --output-on-failure

cmake -E remove_directory "${STAGE}"
cmake --install "${BUILD}" --prefix "${STAGE}"

mkdir -p "${STAGE}/lib"
while read -r soname arrow library remainder; do
  case "${soname}" in
    libz.so.1|libzstd.so.1)
      resolved="$(readlink -f "${library}")"
      cp "${resolved}" "${STAGE}/lib/"
      ln -sfn "$(basename "${resolved}")" "${STAGE}/lib/${soname}"
      ;;
  esac
done < <(ldd "${STAGE}/bin/cosmicblup")

"${STAGE}/bin/cosmicblup" --version
"${STAGE}/bin/cosmicblup" --help >/dev/null
SMOKE="${BUILD}/package-smoke"
cmake -E remove_directory "${SMOKE}"
mkdir -p "${SMOKE}"
"${STAGE}/bin/cosmicblup" --blup --model pblup \
  --ped "${STAGE}/examples/pblup_minimal/pedigree.txt" \
  --pheno "${STAGE}/examples/pblup_minimal/pheno.txt" \
  --pheno-name trait --vars "${STAGE}/examples/pblup_minimal/vars.txt" \
  --threads 1 --no-se --out "${SMOKE}/pblup" >/dev/null
test -s "${SMOKE}/pblup.beta"
test -s "${SMOKE}/pblup.rand"

python3 -c 'import json, pathlib, platform, sys; root=pathlib.Path(sys.argv[1]); out=root/"release_manifest.json"; out.write_text(json.dumps({"name":"CosmicBLUP","version":sys.argv[2],"platform":"linux","architecture":platform.machine(),"executable":"bin/cosmicblup"}, indent=2)+"\n")' "${STAGE}" "${VERSION}"

tar -C "${DIST}" -czf "${DIST}/${NAME}.tar.gz" "${NAME}"
sha256sum "${DIST}/${NAME}.tar.gz" > "${DIST}/${NAME}.tar.gz.sha256"
echo "Created ${DIST}/${NAME}.tar.gz"
