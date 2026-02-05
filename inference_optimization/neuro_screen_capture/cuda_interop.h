#pragma once
#include <cuda_runtime.h>
#include <d3d11.h>
#include <cstdint>

// Forward declarations
struct cudaGraphicsResource;

// Registers a D3D11 texture with CUDA. 
// Returns the registered graphics resource handle.
cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture);

// Unregisters the resource.
void unregister_d3d11_resource(cudaGraphicsResource* resource);

// Maps the resource and returns a cudaArray pointer.
// This must be called before using the array in a kernel.
cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource);

// Unmaps the resource.
void unmap_d3d11_resource(cudaGraphicsResource* resource);

// Preprocess Kernel: Texture (BGRA) -> Tensor (RGB Float NCHW)
// width, height: dimensions of the image.
// stream: CUDA stream to execute on.
void launch_preprocess_kernel(cudaTextureObject_t tex_obj, float* d_output, int width, int height, cudaStream_t stream = 0);

// Postprocess Kernel: Tensor (RGB Float NCHW) -> Surface (BGRA)
// d_input: Pointer to float tensor data.
// surf_obj: Output surface object.
void launch_postprocess_kernel(const float* d_input, cudaSurfaceObject_t surf_obj, int width, int height, cudaStream_t stream = 0);
