param(
    [ValidateSet('Release','Debug')]
    [string]$Config = 'Release',
    [switch]$Clean,
    [switch]$TestOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir  = Join-Path $ScriptDir 'build'
$RepoRoot  = Split-Path -Parent $ScriptDir
$PythonSrc = Join-Path $RepoRoot 'python_src'

# --- Locate Visual Studio 2022 ---
$VsRoot = 'C:\Program Files\Microsoft Visual Studio\2022\Community'
$VsWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $VsWherePath) {
    $found = & $VsWherePath -latest -property installationPath 2>$null
    if ($found) { $VsRoot = $found }
}

if (-not (Test-Path $VsRoot)) {
    Write-Error "Visual Studio 2022 not found at: $VsRoot"
}

$VcVarsAll = Join-Path $VsRoot 'VC\Auxiliary\Build\vcvarsall.bat'
if (-not (Test-Path $VcVarsAll)) {
    Write-Error "vcvarsall.bat not found at: $VcVarsAll"
}

# --- Activate MSVC x64 environment ---
Write-Host "`n[build.ps1] Activating MSVC x64 environment..." -ForegroundColor Cyan

$envDump = cmd /c "`"$VcVarsAll`" x64 >nul 2>&1 && set"
foreach ($line in $envDump) {
    if ($line -match '^([^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

$ClExe = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $ClExe) {
    Write-Error "cl.exe not found after activating MSVC -- check VS installation."
}
Write-Host "[build.ps1] Compiler : $($ClExe.Source)" -ForegroundColor Green

# --- Strip MSYS2/MinGW paths that poison the MSVC include/lib search ---
# These can leak in when the user has MSYS2 on their system PATH.
function Remove-Msys2Paths([string]$envvar) {
    $val = [System.Environment]::GetEnvironmentVariable($envvar, 'Process')
    if (-not $val) { return }
    $clean = ($val -split ';') | Where-Object {
        $_ -notmatch 'msys2|ucrt64|mingw|cygwin' -and $_.Trim() -ne ''
    }
    [System.Environment]::SetEnvironmentVariable($envvar, ($clean -join ';'), 'Process')
}
Remove-Msys2Paths 'INCLUDE'
Remove-Msys2Paths 'LIB'
Remove-Msys2Paths 'LIBPATH'
Remove-Msys2Paths 'PATH'
Write-Host "[build.ps1] MSYS2 paths removed from INCLUDE/LIB/PATH" -ForegroundColor Yellow

# --- Locate Python ---
$VenvPy = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (Test-Path $VenvPy) {
    $PyExe = $VenvPy
} else {
    $PyExe = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "[build.ps1] Python   : $PyExe" -ForegroundColor Green
& $PyExe --version

$NbDir = & $PyExe -m nanobind --cmake_dir 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "nanobind not found. Run: pip install nanobind  (or: uv sync)"
}
Write-Host "[build.ps1] nanobind : $NbDir" -ForegroundColor Green

# --- Locate VS-bundled CMake + Ninja ---
$VsCMakePath = Join-Path $VsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$VsNinjaPath = Join-Path $VsRoot 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'

$CMakeExe = if (Test-Path $VsCMakePath) { $VsCMakePath } else { (Get-Command cmake -ErrorAction Stop).Source }
$NinjaExe = if (Test-Path $VsNinjaPath) { $VsNinjaPath } else { (Get-Command ninja -ErrorAction Stop).Source }

Write-Host "[build.ps1] CMake    : $CMakeExe" -ForegroundColor Green
Write-Host "[build.ps1] Ninja    : $NinjaExe" -ForegroundColor Green

# --- Add Ninja to PATH so CMake can find it ---
$NinjaDir = [System.IO.Path]::GetDirectoryName($NinjaExe)
$env:PATH = $NinjaDir + ';' + $env:PATH

# --- Clean ---
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "`n[build.ps1] Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $PythonSrc | Out-Null

# --- Configure + Build ---
if (-not $TestOnly) {
    Write-Host "`n[build.ps1] Configuring ($Config)..." -ForegroundColor Cyan

    $cmakeArgs = @(
        '-S', $ScriptDir,
        '-B', $BuildDir,
        '-G', 'Ninja',
        "-DCMAKE_BUILD_TYPE=$Config",
        '-DCMAKE_C_COMPILER=cl',
        '-DCMAKE_CXX_COMPILER=cl',
        "-DPython_EXECUTABLE=$PyExe"
    )

    & $CMakeExe @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed." }

    Write-Host "`n[build.ps1] Building..." -ForegroundColor Cyan
    & $CMakeExe --build $BuildDir --config $Config
    if ($LASTEXITCODE -ne 0) { Write-Error "CMake build failed." }

    $PydPath = Join-Path $PythonSrc 'libdsp.pyd'
    if (-not (Test-Path $PydPath)) {
        Write-Error "Build succeeded but libdsp.pyd not found at: $PydPath"
    }
    Write-Host "`n[build.ps1] SUCCESS -- $PydPath" -ForegroundColor Green
}

# --- Run C++ unit tests ---
Write-Host "`n[build.ps1] Running C++ unit tests..." -ForegroundColor Cyan
& $CMakeExe --build $BuildDir --target run_all_tests
$TestExit = $LASTEXITCODE

if ($TestExit -ne 0) {
    Write-Warning "One or more C++ tests failed (exit $TestExit)."
} else {
    Write-Host "[build.ps1] All C++ tests passed." -ForegroundColor Green
}

Write-Host "`n[build.ps1] Done." -ForegroundColor Cyan
Write-Host "  Extension : $PythonSrc\libdsp.pyd"
Write-Host "  C++ tests : $BuildDir\test_dsp.exe"
Write-Host "              $BuildDir\test_channel.exe"
Write-Host "  Python    : .\.venv\Scripts\python.exe -m pytest test_pytest.py -v"
Write-Host "  Editable  : pip install -e .  (or: uv sync)"
