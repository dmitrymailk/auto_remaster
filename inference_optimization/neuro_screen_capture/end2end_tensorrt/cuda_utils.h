#pragma once

#include <cuda_runtime.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

// Launch params: 
// input: Texture object (Screen)
// output: Float ptr (NCHW, 3x512x512)
// original_width, original_height: Screen dims
void launch_preprocess_kernel(
    cudaTextureObject_t tex_obj, 
    float* d_output, 
    int screen_width, 
    int screen_height, 
    cudaStream_t stream = 0
);

// Launch params:
// input: Float ptr (NCHW, 3x512x512)
// output: Surface object (Screen)
// screen_width, screen_height: Screen dims
void launch_postprocess_kernel(const void* d_input, cudaSurfaceObject_t surf_obj, int screen_width, int screen_height, cudaStream_t stream);

// Debug
void launch_debug_tensor_dump(const void* d_input, void* d_debug_output, int width, int height, cudaStream_t stream);

// Setup & Interop
cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture);
void unregister_d3d11_resource(cudaGraphicsResource* resource);
cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource);
void unmap_d3d11_resource(cudaGraphicsResource* resource);

// Kernels
// d_output is half* (FP16) internally
void launch_preprocess_kernel(cudaTextureObject_t tex_obj, void* d_output, int screen_width, int screen_height, cudaStream_t stream);

// d_input is half* (FP16) internally
void launch_postprocess_kernel(const void* d_input, cudaSurfaceObject_t surf_obj, int screen_width, int screen_height, cudaStream_t stream);
