$ErrorActionPreference = "Stop"

Write-Host "Starting Build Process..." -ForegroundColor Green

# 1. Clean Build Directory
if (Test-Path "build") {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build"
}

# 2. Configure with CMake
Write-Host "Configuring CMake..." -ForegroundColor Cyan
cmake -B build -T cuda=12.8
if ($LASTEXITCODE -ne 0) { throw "CMake Configuration Failed" }

# 3. Build Release
Write-Host "Building project..." -ForegroundColor Cyan
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build Failed" }

# 4. Copy DLLs
Write-Host "Copying LibTorch DLLs..." -ForegroundColor Cyan
$torchLib = Join-Path $PSScriptRoot "libtorch\lib"
$destDir = Join-Path $PSScriptRoot "build\Release"

if (Test-Path $torchLib) {
    Copy-Item "$torchLib\*.dll" $destDir -Force
    Write-Host "DLLs copied successfully." -ForegroundColor Green
} else {
    Write-Warning "LibTorch lib directory not found at $torchLib. You might need to add DLLs to PATH manually."
}

Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "Executable is located at: $destDir\nav_screen_capture.exe"
