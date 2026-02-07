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
// offset_x, offset_y: Destination top-left
void launch_postprocess_kernel(const void* d_input, cudaSurfaceObject_t surf_obj, int screen_width, int screen_height, int offset_x, int offset_y, cudaStream_t stream);

// Debug
void launch_debug_tensor_dump(const void* d_input, void* d_debug_output, int width, int height, cudaStream_t stream);

// Setup & Interop
cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture);
void unregister_d3d11_resource(cudaGraphicsResource* resource);
cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource);
void unmap_d3d11_resource(cudaGraphicsResource* resource);

// Kernels
// Launch preprocessing (Texture -> FP16 Tensor)
// Optional: out_u8_rgb for recording (3 channels, interleaved, 0-255)
void launch_preprocess_kernel(cudaTextureObject_t tex_obj, void* d_output, int screen_width, int screen_height, cudaStream_t stream, unsigned char* out_u8_rgb = nullptr);

// d_input is half* (FP16) internally
void launch_postprocess_kernel(const void* d_input, cudaSurfaceObject_t surf_obj, int screen_width, int screen_height, int offset_x, int offset_y, cudaStream_t stream);

// ========== NEW: Pipeline support kernels ==========
// Scale latents (FP16): data[i] *= scale
void launch_scale_latents(void* d_data, float scale, size_t count, cudaStream_t stream);

// Concatenate two latent tensors along channel dimension
// src1 (N,C,H,W) + src2 (N,C,H,W) -> dst (N,2C,H,W)
void launch_concat_latents(const void* src1, const void* src2, void* dst,
                           int batch, int channels, int height, int width, cudaStream_t stream);

// FlowMatchEulerDiscreteScheduler step
// sample = sample + model_output * (sigma_next - sigma)
void launch_scheduler_step(void* sample, const void* model_output,
                           float sigma, float sigma_next, size_t count, cudaStream_t stream);

// Convert float to FP16 and write to device buffer
void launch_float_to_half(void* d_output, float value, cudaStream_t stream);
