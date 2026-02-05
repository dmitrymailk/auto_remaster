#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <windows.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t error = err; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Check Failed"); \
        } \
    } while (0)

#define DX_CHECK(hr) \
    do { \
        HRESULT res = hr; \
        if (FAILED(res)) { \
            std::cerr << "DirectX Error: HRESULT 0x" << std::hex << res << std::dec << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("DirectX Check Failed"); \
        } \
    } while (0)
