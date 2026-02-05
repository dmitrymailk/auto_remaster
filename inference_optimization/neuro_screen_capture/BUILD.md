# Build Instructions

## Prerequisites
1.  **Visual Studio 2022** with C++ Desktop Development workload.
2.  **CMake 3.18+** installed and added to PATH.
3.  **CUDA Toolkit 12.8** installed.
4.  **LibTorch (PyTorch C++ API)**:
    -   Download the **Release** version with **CUDA 12.4/12.x** support (or matching your CUDA version).
    -   Extract the `libtorch` folder into the project root:
        `C:\programming\auto_remaster\inference_optimization\neuro_screen_capture\libtorch`

## Method 1: Automatic Build (Recommended)

Run the provided PowerShell script. It handles cleaning, configuration (forcing CUDA 12.8), building, and copying necessary DLLs.

Open PowerShell in the project directory and run:

```powershell
powershell -ExecutionPolicy Bypass -File build.ps1
```

If you see errors about execution policy, you can try:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build.ps1
```

## Method 2: Manual Build

If you prefer to run commands manually, follow these steps:

1.  **Create Build Directory**:
    ```powershell
    mkdir build
    cd build
    ```

2.  **Configure CMake**:
    *Crucial*: You must specify the CUDA toolset version to avoid picking up other installed versions (like v13.1).
    ```powershell
    cmake .. -T cuda=12.8
    ```

3.  **Build the Project**:
    ```powershell
    cmake --build . --config Release
    ```

4.  **Copy Dependencies**:
    You must copy the LibTorch DLLs to the folder where the executable was created (`Release`).
    ```powershell
    copy ..\libtorch\lib\*.dll Release\
    copy ..\libtorch\lib\*.lib Release\
    ```

## Running the Application

The executable will be located at:
`build\Release\nav_screen_capture.exe`

Run it from the console to see FPS output:
```powershell
.\build\Release\nav_screen_capture.exe
```
