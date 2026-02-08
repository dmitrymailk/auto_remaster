cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)

project(end2end_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- Dependencies ---
find_package(CUDAToolkit 12.8 REQUIRED)

# TensorRT - Attempt to find or setup paths manually if FindTensorRT.cmake is missing
# You might need to set TRT_ROOT if it's not in system paths
if (NOT DEFINED ENV{TRT_ROOT})
    set(TRT_ROOT "C:/programming/auto_remaster/inference_optimization/TensorRT-10.15.1.29")
else()
    set(TRT_ROOT $ENV{TRT_ROOT})
endif()

# Quick hack for includes if find_package fails or isn't available
include_directories(${TRT_ROOT}/include)
link_directories(${TRT_ROOT}/lib)


set(SOURCES
    main.cpp
    capture_dxgi.cpp
    capture_wgc.cpp
    window_helper.cpp
    pipeline.cpp
    recorder.cpp
    cuda_utils.cu
    vsr_upscaler.cpp
)

set(HEADERS
    capture_dxgi.h
    pipeline.h
    cuda_utils.h
    utils.h
    vsr_upscaler.h
)

# --- Executable ---
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# --- RTX Video SDK ---
set(RTX_VIDEO_SDK_ROOT "C:/programming/auto_remaster/inference_optimization/RTX_Video_SDK_v1.1.0")

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRT_ROOT}/include # repetitive but safe
    ${RTX_VIDEO_SDK_ROOT}/include
)

# --- Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
    nvinfer_10
    d3d11
    dxgi
    d3dcompiler
    dwmapi
    windowsapp
    ${RTX_VIDEO_SDK_ROOT}/lib/Windows/x64/nvsdk_ngx_d.lib
)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4>")
endif()
