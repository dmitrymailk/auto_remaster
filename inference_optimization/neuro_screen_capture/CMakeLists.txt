cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
# Force CUDA 12.8 paths (via toolset)
# set(CMAKE_CUDA_COMPILER ...) - Removed to let VS generator handle it
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")

project(nav_screen_capture LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- Configuration ---
# Set this to your LibTorch installation path if not found automatically
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
find_package(CUDAToolkit 12.8 REQUIRED)

# --- Sources ---
set(SOURCES
    main.cpp
    capture_dxgi.cpp
    cuda_interop.cu
)

set(HEADERS
    capture_dxgi.h
    cuda_interop.h
    utils.h
)

# --- Executable ---
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# --- Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    d3d11
    dxgi
    d3dcompiler
)

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4>")
endif()
