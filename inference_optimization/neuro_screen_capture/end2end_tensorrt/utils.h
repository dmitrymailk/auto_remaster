#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <windows.h>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// Helper macros for error checking
#define MODEL_SIZE 512

#define DX_CHECK(hr) \
    if (FAILED(hr)) { \
        throw std::runtime_error("DirectX Error at line " + std::to_string(__LINE__) + " (HR: " + std::to_string(hr) + ")"); \
    }

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA Error at line " + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    }

// Helper to check for generic bool errors
#define CHECK(x) \
    if (!(x)) { \
        throw std::runtime_error("Error at line " + std::to_string(__LINE__)); \
    }
