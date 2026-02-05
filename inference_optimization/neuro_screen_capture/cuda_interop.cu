#include "cuda_interop.h"
#include <cuda_d3d11_interop.h>
#include "utils.h"

cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture) {
    cudaGraphicsResource* resource = nullptr;
    // Register with None flags as we might map for read (texture) or write (surface)
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(&resource, texture, cudaGraphicsRegisterFlagsNone));
    return resource;
}

void unregister_d3d11_resource(cudaGraphicsResource* resource) {
    if (resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
    }
}

cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, 0));
    cudaArray_t array;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    return array;
}

void unmap_d3d11_resource(cudaGraphicsResource* resource) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, 0));
}

// =========================================================================
// Kernels
// =========================================================================

__global__ void preprocess_kernel_nchw(cudaTextureObject_t texObj, float* out_tensor, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Read from texture (BGRA usually). 
        // Note: tex2D with uchar4 normalized=false returns 0-255 values in .x, .y, .z, .w
        // DXGI_FORMAT_B8G8R8A8:
        // x = Blue
        // y = Green
        // z = Red
        // w = Alpha
        uchar4 pixel = tex2D<uchar4>(texObj, x, y);

        int area = width * height;
        int idx_base = y * width + x;

        // Planar NCHW: Red Plane, Green Plane, Blue Plane
        out_tensor[idx_base]            = pixel.z / 255.0f; // R
        out_tensor[area + idx_base]     = pixel.y / 255.0f; // G
        out_tensor[2 * area + idx_base] = pixel.x / 255.0f; // B
        // Alpha ignored for now
    }
}

__global__ void postprocess_kernel_nchw(const float* in_tensor, cudaSurfaceObject_t surfObj, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int area = width * height;
        int idx_base = y * width + x;

        float r = in_tensor[idx_base];
        float g = in_tensor[area + idx_base];
        float b = in_tensor[2 * area + idx_base];

        // Saturate/Clamp
        uchar4 pixel;
        pixel.z = (unsigned char)(__saturatef(r) * 255.0f); // R
        pixel.y = (unsigned char)(__saturatef(g) * 255.0f); // G
        pixel.x = (unsigned char)(__saturatef(b) * 255.0f); // B
        pixel.w = 255; // Alpha full

        // Write to surface
        surf2Dwrite(pixel, surfObj, x * sizeof(uchar4), y);
    }
}

void launch_preprocess_kernel(cudaTextureObject_t tex_obj, float* d_output, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    preprocess_kernel_nchw<<<grid, block, 0, stream>>>(tex_obj, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
}

void launch_postprocess_kernel(const float* d_input, cudaSurfaceObject_t surf_obj, int width, int height, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    postprocess_kernel_nchw<<<grid, block, 0, stream>>>(d_input, surf_obj, width, height);
    CUDA_CHECK(cudaGetLastError());
}
