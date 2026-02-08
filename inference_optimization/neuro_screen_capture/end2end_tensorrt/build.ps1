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



# TensorRT DLLs
# Using the path we identified
$trtRoot = "C:\programming\auto_remaster\inference_optimization\TensorRT-10.15.1.29"
$trtLib = Join-Path $trtRoot "lib" 
$trtBin = Join-Path $trtRoot "bin" -ErrorAction SilentlyContinue

# Only copy essential runtime DLLs to avoid huge builder resources
$requiredTrtDlls = @(
    "nvinfer_10.dll", 
    "nvinfer_plugin_10.dll", 
    "nvonnxparser_10.dll", 
    "nvinfer_dispatch_10.dll"
)

if (Test-Path $trtBin) {
    foreach ($dll in $requiredTrtDlls) {
        $srcPath = Join-Path $trtBin $dll
        if (Test-Path $srcPath) {
            Copy-Item $srcPath $destDir -Force
        }
    }
}


Write-Host "Copying Model Files..." -ForegroundColor Cyan
$modelDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\flux_vae_tiny_trt"
if (Test-Path $modelDir) {
    Copy-Item "$modelDir\*.plan" $destDir -Force
    Write-Host "VAE model files copied." -ForegroundColor Green
} else {
    Write-Warning "VAE model directory not found at $modelDir. Please manually copy .plan files."
}

# Copy UNet Model
# $unetDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\unet_trt_v6_upscale_2x"
$unetDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\unet_trt_v6"
# $unetDir = Join-Path $PSScriptRoot "..\..\Model_Optimizer\examples\diffusers\quantization\unet_trt"
if (Test-Path $unetDir) {
    Copy-Item "$unetDir\*.plan" $destDir -Force
    Write-Host "UNet model files copied." -ForegroundColor Green
} else {
    Write-Warning "UNet model directory not found at $unetDir. Please manually copy unet.plan file."
}

# Copy RTX Video SDK DLLs
$rtxSdkBin = "C:\programming\auto_remaster\inference_optimization\RTX_Video_SDK_v1.1.0\bin\Windows\x64\dev" 
# Use 'dev' folders for development/debugging, 'rel' for release. 'dev' usually has valid signatures for development.
# But guide says regular driver installation puts them in System32. 
# However, to be safe and ensure they are found if not in path, copy them.
# Research suggested 'rel' or 'dev'. Let's check if 'dev' exists in the previous file listing? 
# The listing showed: bin\Windows\x64\dev\nvngx_vsr.dll
if (Test-Path $rtxSdkBin) {
    Copy-Item "$rtxSdkBin\nvngx_vsr.dll" $destDir -Force
    Copy-Item "$rtxSdkBin\nvngx_truehdr.dll" $destDir -Force # Just in case
    Write-Host "RTX Video SDK DLLs copied." -ForegroundColor Green
} else {
    Write-Warning "RTX Video SDK bin directory not found at $rtxSdkBin."
}

Write-Host "Build Complete!" -ForegroundColor Green
Write-Host "Executable is located at: $destDir\end2end_tensorrt.exe"
