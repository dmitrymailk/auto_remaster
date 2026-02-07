#include "cuda_utils.h"
#include "utils.h"
#include <algorithm>
#include <cuda_fp16.h>


__global__ void preprocess_kernel(cudaTextureObject_t texObj, __half* out_tensor, int crop_w, int crop_h, int offset_x, int offset_y) {
    // Grid matches MODEL_SIZE x MODEL_SIZE
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < MODEL_SIZE && y < MODEL_SIZE) {
        // UV coordinates in Texture (0.0 .. 1.0) relative to Crop Area? 
        // No, we sample from the Crop Area within the Screen.
        // u = (offset_x + x * (crop_w / 512.0)) 
        // But tex2D uses normalized or unnormalized? We set texDesc.normalizedCoords = 0.
        // So we use pixel coords.
        // Screen Crop: [offset_x, offset_y] size [crop_w, crop_h]
        
        // Box Filter Downsampling
        float scale_x = crop_w / (float)MODEL_SIZE;
        float scale_y = crop_h / (float)MODEL_SIZE;
        
        // Determine number of samples (e.g. if scale is 3.125, take 4x4 samples)
        int samples_x = (int)ceilf(scale_x);
        int samples_y = (int)ceilf(scale_y);
        
        // Clamp max samples to avoid performance kill (e.g. max 8x8)
        samples_x = min(samples_x, 8);
        samples_y = min(samples_y, 8);
        
        float r_acc = 0.0f;
        float g_acc = 0.0f;
        float b_acc = 0.0f;
        
        // Start of the source region for this destination pixel
        float src_start_x = offset_x + x * scale_x;
        float src_start_y = offset_y + y * scale_y;
        
        // Sample grid within the source region
        for (int dy = 0; dy < samples_y; ++dy) {
            for (int dx = 0; dx < samples_x; ++dx) {
                // Sample center of the sub-block
                float u = src_start_x + (dx + 0.5f) * (scale_x / samples_x);
                float v = src_start_y + (dy + 0.5f) * (scale_y / samples_y);
                
                float4 pixel = tex2D<float4>(texObj, u, v);
                
                // Accumulate (BGR -> RGB swap happening here)
                r_acc += pixel.z; // R is at .z
                g_acc += pixel.y;
                b_acc += pixel.x; // B is at .x
            }
        }
        
        // Average
        float num_samples = (float)(samples_x * samples_y);
        float r = (r_acc / num_samples) * 2.0f - 1.0f; // [0,1] -> [-1,1]
        float g = (g_acc / num_samples) * 2.0f - 1.0f;
        float b = (b_acc / num_samples) * 2.0f - 1.0f;

        // Write to Output Tensor (NCHW Planar) - FP16
        int area = MODEL_SIZE * MODEL_SIZE;
        int idx = y * MODEL_SIZE + x;
        
        out_tensor[idx] = __float2half(r);
        out_tensor[area + idx] = __float2half(g);
        out_tensor[2 * area + idx] = __float2half(b);
    }
}

// Draw the 512x512 tensor back to the screen center
// Input: Tensor (FP16 NCHW)
// Output: Surface (uchar4)
__global__ void postprocess_kernel(const __half* in_tensor, cudaSurfaceObject_t surfObj, int screen_w, int screen_h, int dst_w, int dst_h, int dst_off_x, int dst_off_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < screen_w && y < screen_h) {
        // Check if inside destination ROI
        int rel_x = x - dst_off_x;
        int rel_y = y - dst_off_y;
        
        if (rel_x >= 0 && rel_x < dst_w && rel_y >= 0 && rel_y < dst_h) {
             // Map Dest (rel_x, rel_y) -> Source (sx, sy in 0..511)
            // Bilinear Interpolation
            float src_x = (rel_x / (float)dst_w) * MODEL_SIZE;
            float src_y = (rel_y / (float)dst_h) * MODEL_SIZE;
            
            // Coordinates of top-left pixel
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = min(x0 + 1, MODEL_SIZE - 1);
            int y1 = min(y0 + 1, MODEL_SIZE - 1);
            
            // Weights
            float dx = src_x - x0;
            float dy = src_y - y0;
            float w00 = (1.0f - dx) * (1.0f - dy);
            float w10 = dx * (1.0f - dy);
            float w01 = (1.0f - dx) * dy;
            float w11 = dx * dy;
            
            // Function to read a pixel (helper lambda not avail here, macro or inline?)
            // Inline reading:
            int idx00 = y0 * MODEL_SIZE + x0;
            int idx10 = y0 * MODEL_SIZE + x1;
            int idx01 = y1 * MODEL_SIZE + x0;
            int idx11 = y1 * MODEL_SIZE + x1;
            int area = MODEL_SIZE * MODEL_SIZE;
            

            
            // Read R
            float r00 = __half2float(in_tensor[idx00]);
            float r10 = __half2float(in_tensor[idx10]);
            float r01 = __half2float(in_tensor[idx01]);
            float r11 = __half2float(in_tensor[idx11]);
            float r = w00 * r00 + w10 * r10 + w01 * r01 + w11 * r11;
            
            // Read G (Area offset)
            float g00 = __half2float(in_tensor[area + idx00]);
            float g10 = __half2float(in_tensor[area + idx10]);
            float g01 = __half2float(in_tensor[area + idx01]);
            float g11 = __half2float(in_tensor[area + idx11]);
            float g = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11;

            // Read B (2*Area offset)
            float b00 = __half2float(in_tensor[2 * area + idx00]);
            float b10 = __half2float(in_tensor[2 * area + idx10]);
            float b01 = __half2float(in_tensor[2 * area + idx01]);
            float b11 = __half2float(in_tensor[2 * area + idx11]);
            float b = w00 * b00 + w10 * b10 + w01 * b01 + w11 * b11;

            // Denormalize [-1, 1] -> [0, 1]
            r = r * 0.5f + 0.5f;
            g = g * 0.5f + 0.5f;
            b = b * 0.5f + 0.5f;
            
            uchar4 pixel;
            pixel.z = (unsigned char)(__saturatef(r) * 255.0f);
            pixel.y = (unsigned char)(__saturatef(g) * 255.0f);
            pixel.x = (unsigned char)(__saturatef(b) * 255.0f);
            pixel.w = 255;
            
            surf2Dwrite(pixel, surfObj, x * sizeof(uchar4), y);
        } else {
            // Draw Black Background
            uchar4 black = make_uchar4(0, 0, 0, 255);
            surf2Dwrite(black, surfObj, x * sizeof(uchar4), y);
        }
    }
}

void launch_preprocess_kernel(cudaTextureObject_t tex_obj, void* d_output, int screen_width, int screen_height, cudaStream_t stream) {
    // Crop logic: Take largest center square
    int crop_size = std::min(screen_width, screen_height);
    int offset_x = (screen_width - crop_size) / 2;
    int offset_y = (screen_height - crop_size) / 2;
    
    static int frame_count = 0;
    if (frame_count++ % 300 == 0) { // Print every 300 frames (~5 sec)
        std::cout << "[Preprocess] Screen: " << screen_width << "x" << screen_height 
                  << " Crop: " << crop_size << "x" << crop_size 
                  << " Offset: " << offset_x << "," << offset_y 
                  << " Step: " << (float)crop_size / MODEL_SIZE << std::endl;
    }

    dim3 block(16, 16);
    dim3 grid((MODEL_SIZE + block.x - 1) / block.x, (MODEL_SIZE + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(tex_obj, (__half*)d_output, crop_size, crop_size, offset_x, offset_y);
    CUDA_CHECK(cudaGetLastError());
}

void launch_postprocess_kernel(const void* d_input, cudaSurfaceObject_t surf_obj, int screen_width, int screen_height, cudaStream_t stream) {
    // Draw fixed 512x512 center square (No upscaling)
    int dst_size = MODEL_SIZE;
    int offset_x = (screen_width - dst_size) / 2;
    int offset_y = (screen_height - dst_size) / 2;

    // Run threads over the FULL SCREEN to allow clearing background
    dim3 block(16, 16);
    dim3 grid((screen_width + block.x - 1) / block.x, (screen_height + block.y - 1) / block.y);
    
    postprocess_kernel<<<grid, block, 0, stream>>>((const __half*)d_input, surf_obj, screen_width, screen_height, dst_size, dst_size, offset_x, offset_y);
    CUDA_CHECK(cudaGetLastError());
}

// Debug: Convert FP16 NCHW tensor to UInt8 NHWC (RGB) for saving
__global__ void debug_nchw_half_to_rgb_kernel(const __half* in_tensor, unsigned char* out_img, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int area = width * height;
        // NCHW Indexing
        int idx = y * width + x;
        int idx_r = idx;
        int idx_g = area + idx;
        int idx_b = 2 * area + idx;

        // Read FP16 [-1, 1]
        float r = __half2float(in_tensor[idx_r]);
        float g = __half2float(in_tensor[idx_g]);
        float b = __half2float(in_tensor[idx_b]);

        // Denormalize to [0, 1] then Scale to [0, 255]
        r = r * 0.5f + 0.5f;
        g = g * 0.5f + 0.5f;
        b = b * 0.5f + 0.5f;

        // RGB Interleaved Indexing
        int out_idx = (y * width + x) * 3;
        out_img[out_idx]     = (unsigned char)(__saturatef(r) * 255.0f);
        out_img[out_idx + 1] = (unsigned char)(__saturatef(g) * 255.0f);
        out_img[out_idx + 2] = (unsigned char)(__saturatef(b) * 255.0f);
    }
}

void launch_debug_tensor_dump(const void* d_input, void* d_debug_output, int width, int height, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    debug_nchw_half_to_rgb_kernel<<<grid, block, 0, stream>>>((const __half*)d_input, (unsigned char*)d_debug_output, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// NEW: Pipeline Support Kernels
// =========================================================================

// Scale latents by a factor (FP16)
__global__ void scale_latents_kernel(__half* data, float scale, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = __half2float(data[idx]);
        data[idx] = __float2half(val * scale);
    }
}

void launch_scale_latents(void* d_data, float scale, size_t count, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    scale_latents_kernel<<<num_blocks, block_size, 0, stream>>>((__half*)d_data, scale, count);
    CUDA_CHECK(cudaGetLastError());
}

// Concatenate two latent tensors along channel dimension (FP16)
// Input: src1 (N,C,H,W), src2 (N,C,H,W)
// Output: dst (N,2C,H,W)
__global__ void concat_latents_kernel(const __half* src1, const __half* src2, __half* dst, 
                                       int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = height * width;
    int total = batch * channels * 2 * spatial;
    
    if (idx < total) {
        int b = idx / (channels * 2 * spatial);
        int remainder = idx % (channels * 2 * spatial);
        int c = remainder / spatial;
        int hw = remainder % spatial;
        
        if (c < channels) {
            // First half: from src1
            int src_idx = b * channels * spatial + c * spatial + hw;
            dst[idx] = src1[src_idx];
        } else {
            // Second half: from src2
            int src_idx = b * channels * spatial + (c - channels) * spatial + hw;
            dst[idx] = src2[src_idx];
        }
    }
}

void launch_concat_latents(const void* src1, const void* src2, void* dst,
                           int batch, int channels, int height, int width, cudaStream_t stream) {
    int total = batch * channels * 2 * height * width;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    concat_latents_kernel<<<num_blocks, block_size, 0, stream>>>(
        (const __half*)src1, (const __half*)src2, (__half*)dst,
        batch, channels, height, width);
    CUDA_CHECK(cudaGetLastError());
}

// FlowMatchEulerDiscreteScheduler step (FP16)
// sample = sample + model_output * (sigma_next - sigma)
__global__ void scheduler_step_kernel(__half* sample, const __half* model_output,
                                       float sigma, float sigma_next, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float s = __half2float(sample[idx]);
        float velocity = __half2float(model_output[idx]);
        
        // FlowMatchEulerDiscreteScheduler step:
        // In flow matching, model directly predicts velocity
        // sample = sample + velocity * dt
        float dt = sigma_next - sigma;  // negative because sigma decreases
        s = s + velocity * dt;
        // s = s + velocity * -1.0f;
        
        sample[idx] = __float2half(s);
    }
}

void launch_scheduler_step(void* sample, const void* model_output,
                           float sigma, float sigma_next, size_t count, cudaStream_t stream) {
    int block_size = 256;
    int num_blocks = (count + block_size - 1) / block_size;
    scheduler_step_kernel<<<num_blocks, block_size, 0, stream>>>(
        (__half*)sample, (const __half*)model_output, sigma, sigma_next, count);
    CUDA_CHECK(cudaGetLastError());
}

// Float to FP16 conversion kernel
__global__ void float_to_half_kernel(__half* output, float value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = __float2half(value);
    }
}

void launch_float_to_half(void* d_output, float value, cudaStream_t stream) {
    float_to_half_kernel<<<1, 1, 0, stream>>>((__half*)d_output, value);
    CUDA_CHECK(cudaGetLastError());
}

// =========================================================================
// Interop Helpers
// =========================================================================
#include <cuda_d3d11_interop.h>
// ... (rest of the file)
cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture) {
    cudaGraphicsResource* res = nullptr;
    CUDA_CHECK(cudaGraphicsD3D11RegisterResource(&res, texture, cudaGraphicsRegisterFlagsNone));
    return res;
}

void unregister_d3d11_resource(cudaGraphicsResource* resource) {
    if (resource) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(resource));
    }
}

cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource) {
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource));
    cudaArray_t array;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
    return array;
}

void unmap_d3d11_resource(cudaGraphicsResource* resource) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource));
}
