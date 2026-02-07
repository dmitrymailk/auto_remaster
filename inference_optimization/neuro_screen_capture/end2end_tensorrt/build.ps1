$ErrorActionPreference = "Stop"

Write-Host "Starting Build Process..." -ForegroundColor Green

# 1. Clean Build Directory
if (Test-Path "build") {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build"
}

# 2. Configure with CMake
Write-Host "Configuring CMake..." -ForegroundColor Cyan
# Using -T cuda=12.8 as verified earlier (user has 12.8 installed)
cmake -B build -T cuda=12.8
if ($LASTEXITCODE -ne 0) { throw "CMake Configuration Failed" }

# 3. Build Release
Write-Host "Building project..." -ForegroundColor Cyan
cmake --build build --config Release
if ($LASTEXITCODE -ne 0) { throw "Build Failed" }

# 4. Copy Dependencies
Write-Host "Copying Dependencies..." -ForegroundColor Cyan
$destDir = Join-Path $PSScriptRoot "build\Release"

# LibTorch DLLs
$torchLib = Join-Path $PSScriptRoot "..\libtorch\lib"
if (Test-Path $torchLib) {
    Copy-Item "$torchLib\*.dll" $destDir -Force
    Write-Host "LibTorch DLLs copied." -ForegroundColor Green
} else {
    Write-Warning "LibTorch lib directory not found at $torchLib."
}

# TensorRT DLLs
# Using the path we identified
$trtRoot = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29"
$trtLib = Join-Path $trtRoot "lib" 
# Check if DLLs are in lib or bin. Usually lib for these zips? 
# If previous listing showed only .lib in lib, they might be in lib (and ls just missed them? No, ls shows all).
# Or they are in bin.
# I will check 'bin' in the script or assume 'lib' based on Windows conventions for some packages, 
# but TRT often puts DLLs in lib.
# Actually, let's copy from both just in case, or check.
# The list_dir output for 'lib' previously showed only .lib. I am invoking check on bin now.
# I will write the script to copy from 'lib' AND 'bin' if they exist.

$trtBin = Join-Path $trtRoot "bin" -ErrorAction SilentlyContinue

if (Test-Path $trtLib) {
    Copy-Item "$trtLib\*.dll" $destDir -Force -ErrorAction SilentlyContinue
}
if (Test-Path $trtBin) {
    Copy-Item "$trtBin\*.dll" $destDir -Force -ErrorAction SilentlyContinue
}

# 5. Copy Models
Write-Host "Copying Model Files..." -ForegroundColor Cyan
$modelDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\flux_vae_tiny_trt"
if (Test-Path $modelDir) {
    Copy-Item "$modelDir\*.plan" $destDir -Force
    Write-Host "VAE model files copied." -ForegroundColor Green
} else {
    Write-Warning "VAE model directory not found at $modelDir. Please manually copy .plan files."
}

# Copy UNet Model
$unetDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\unet_trt"
if (Test-Path $unetDir) {
    Copy-Item "$unetDir\*.plan" $destDir -Force
    Write-Host "UNet model files copied." -ForegroundColor Green
} else {
    Write-Warning "UNet model directory not found at $unetDir. Please manually copy unet.plan file."
}

Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "Executable is located at: $destDir\end2end_tensorrt.exe"
