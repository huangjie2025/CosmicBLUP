param(
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Triplet = "x64-windows"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Version = (Get-Content (Join-Path $Root "VERSION") -Raw).Trim()
$Build = Join-Path $Root "build-package-windows"
$Dist = Join-Path $Root "dist"
$Name = "CosmicBLUP-$Version-windows-x86_64"
$Stage = Join-Path $Dist $Name

$Configure = @("-S", $Root, "-B", $Build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_LAPACK=ON")
if ($VcpkgRoot) {
    $Configure += "-DCMAKE_TOOLCHAIN_FILE=$VcpkgRoot/scripts/buildsystems/vcpkg.cmake"
    $Configure += "-DVCPKG_TARGET_TRIPLET=$Triplet"
}
cmake @Configure
cmake --build $Build --parallel 2
ctest --test-dir $Build --output-on-failure
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
cmake --install $Build --prefix $Stage
& (Join-Path $Stage "bin/cosmicblup.exe") --version

$Manifest = @{ name="CosmicBLUP"; version=$Version; platform="windows"; architecture="x86_64"; executable="bin/cosmicblup.exe" } | ConvertTo-Json
Set-Content -Path (Join-Path $Stage "release_manifest.json") -Value $Manifest
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
Compress-Archive -Path $Stage -DestinationPath (Join-Path $Dist "$Name.zip") -Force
(Get-FileHash (Join-Path $Dist "$Name.zip") -Algorithm SHA256).Hash.ToLower() + "  $Name.zip" | Set-Content (Join-Path $Dist "$Name.zip.sha256")
