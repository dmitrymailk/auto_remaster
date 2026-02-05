#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "utils.h"
#include "capture_dxgi.h"
#include "pipeline.h"
#include "cuda_utils.h"

// cuda_interop.h was in previous but we moved kernel wrappers to cuda_utils.h
// We still need registration functions. 
// Ah, I missed copying `register_d3d11_resource` etc implementation.
// They were in `cuda_interop.cu` in the original.
// I should add them to `cuda_utils.cu` or separate file. 
// I'll add them to `cuda_utils.cu` and declarations to `cuda_utils.h` to avoid another file.
// Or I can just write them inline here if simple? No, reuse or clean code.
// I'll update `cuda_utils` after `main` or correct it now?
// I'll write `main` assuming they exist in `cuda_utils.h`. I will update `cuda_utils` next.

// Forward decls for interop (will be in header)
cudaGraphicsResource* register_d3d11_resource(ID3D11Texture2D* texture);
void unregister_d3d11_resource(cudaGraphicsResource* resource);
cudaArray_t map_d3d11_resource(cudaGraphicsResource* resource);
void unmap_d3d11_resource(cudaGraphicsResource* resource);


// Window dimensions (initial)
HWND g_hwnd = nullptr;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_KEYDOWN && wParam == VK_ESCAPE) {
        PostQuitMessage(0);
        return 0;
    }
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void CreateNativeWindow(HINSTANCE hInstance, int width, int height) {
    const char CLASS_NAME[] = "NeuroScreenCaptureClass";
    
    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    g_hwnd = CreateWindowEx(
        0, CLASS_NAME, "Neuro Screen Capture (TensorRT VAE)",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        NULL, NULL, hInstance, NULL
    );

    if (g_hwnd == NULL) throw std::runtime_error("Failed to create window");

    ShowWindow(g_hwnd, SW_SHOW);
}

int main() {
    try {
        std::cout << "Initializing Neuro Screen Capture (TensorRT VAE)..." << std::endl;
        
        // 0. Display Settings
        DEVMODE dm = { 0 };
        dm.dmSize = sizeof(dm);
        if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm)) {
             throw std::runtime_error("Failed to enum display settings");
        }
        int WIDTH = dm.dmPelsWidth;
        int HEIGHT = dm.dmPelsHeight;
        std::cout << "Screen Resolution: " << WIDTH << "x" << HEIGHT << std::endl;

        // 1. Create Window
        CreateNativeWindow(GetModuleHandle(NULL), WIDTH, HEIGHT);

        // 2. D3D11 Setup
        DXGI_SWAP_CHAIN_DESC scd = {0};
        scd.BufferCount = 2;
        scd.BufferDesc.Width = WIDTH;
        scd.BufferDesc.Height = HEIGHT;
        scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scd.BufferDesc.RefreshRate.Numerator = 60;
        scd.BufferDesc.RefreshRate.Denominator = 1;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.OutputWindow = g_hwnd;
        scd.SampleDesc.Count = 1;
        scd.Windowed = TRUE;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<IDXGISwapChain> swap_chain; // IDXGISwapChain1? 
        // Using standard swapchain for simplicity
        
        UINT createDeviceFlags = 0;
        #ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
        #endif

        DX_CHECK(D3D11CreateDeviceAndSwapChain(
            NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags,
            NULL, 0, D3D11_SDK_VERSION, &scd,
            &swap_chain, &device, NULL, &context
        ));

        // 3. Screen Capture
        ScreenCapture capture(device.Get(), context.Get());
        capture.Initialize();

        // 4. TensorRT Pipeline
        // Hardcoded paths for now - adjust as needed
        // Assuming running from build directory, and models in a known relative path
        // Or simply absolute paths if known.
        // Let's assume adjacent to executable or CWD.
        std::string enc_path = "vae_encoder.plan";
        std::string dec_path = "vae_decoder.plan";
        
        TensorRTPipeline pipeline;
        pipeline.LoadEngines(enc_path, dec_path);

        // 5. Shared Texture (Backbuffer target for CUDA)
        D3D11_TEXTURE2D_DESC sharedTexDesc = {0};
        sharedTexDesc.Width = WIDTH;
        sharedTexDesc.Height = HEIGHT;
        sharedTexDesc.MipLevels = 1;
        sharedTexDesc.ArraySize = 1;
        sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sharedTexDesc.SampleDesc.Count = 1;
        sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
        sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; 
        
        ComPtr<ID3D11Texture2D> shared_texture;
        DX_CHECK(device->CreateTexture2D(&sharedTexDesc, NULL, &shared_texture));

        // Register with CUDA
        cudaGraphicsResource* cuda_res = register_d3d11_resource(shared_texture.Get());
        if (!cuda_res) throw std::runtime_error("Failed to register shared texture with CUDA");

        // 6. Allocate Tensor Memory (512x512)
        // Format: NCHW FP16 (2 bytes per element)
        // Engine expects FP16, so we must provide FP16 buffers.
        
        void *d_input, *d_output;
        size_t tensor_elements = 3 * 512 * 512;
        size_t tensor_bytes = tensor_elements * 2; // 2 bytes for half
        
        CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));
        
        // 7. Loop
        MSG msg = {0};
        auto start_time = std::chrono::high_resolution_clock::now();
        int frames = 0;
        
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Debug Buffer (RGB 512x512)
        unsigned char* d_debug_img = nullptr;
        size_t debug_size = MODEL_SIZE * MODEL_SIZE * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_debug_img, debug_size));
        bool save_requested = false;

        std::cout << "Starting Loop... Press 'S' to save debugging image (debug_input.ppm)." << std::endl;

        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_KEYDOWN && msg.wParam == 'S') {
                    save_requested = true;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                ComPtr<ID3D11Texture2D> current_frame;
                if (capture.AcquireFrame(current_frame, 10)) {
                    
                    // Copy to Shared Texture (Or map directly? Map directly is better if possible)
                    // If we map 'current_frame', it must be created with D3D11_BIND_SHADER_RESOURCE? 
                    // Capture texture usually isn't. So Copy is needed.
                    // Copy to 'shared_texture' first? 
                    // Wait, 'shared_texture' is our Output target.
                    // We need an Input staging texture? 
                    // Actually, we can reuse 'shared_texture' as Input staging if we want, 
                    // but we overwrite it with Output later.
                    // If Preprocess finishes before Postprocess starts, safe.
                    // Let's use 'shared_texture' for both? 
                    // 1. Copy Capture -> Shared.
                    // 2. Map Shared (Read). Preprocess -> TensorIn. Unmap.
                    // 3. Infer.
                    // 4. Map Shared (Write). Postprocess TensorOut -> Shared. Unmap.
                    // 5. Present (Copy Shared -> Backbuffer).
                    // Yes, valid reuse.
                    
                    // CopySubresourceRegion to handle resolution mismatch if any
                    D3D11_TEXTURE2D_DESC desc;
                    current_frame->GetDesc(&desc);
                    
                    D3D11_BOX sourceRegion;
                    sourceRegion.left = 0;
                    sourceRegion.right = std::min<UINT>(desc.Width, WIDTH);
                    sourceRegion.top = 0;
                    sourceRegion.bottom = std::min<UINT>(desc.Height, HEIGHT);
                    sourceRegion.front = 0;
                    sourceRegion.back = 1;
                    
                    context->CopySubresourceRegion(shared_texture.Get(), 0, 0, 0, 0, current_frame.Get(), 0, &sourceRegion);
                    capture.ReleaseFrame(); // Release early to unblock capture

                    // CUDA Map
                    cudaArray_t mapped_array = map_d3d11_resource(cuda_res);
                    
                    // Create Texture Object (Read)
                    cudaResourceDesc resDesc = {};
                    resDesc.resType = cudaResourceTypeArray;
                    resDesc.res.array.array = mapped_array;
                    
                    cudaTextureDesc texDesc = {};
                    texDesc.addressMode[0] = cudaAddressModeClamp;
                    texDesc.addressMode[1] = cudaAddressModeClamp;
                    texDesc.filterMode = cudaFilterModeLinear; // Better quality for resizing
                    texDesc.readMode = cudaReadModeNormalizedFloat;
                    texDesc.normalizedCoords = 0; // Use pixel coords
                    
                    cudaTextureObject_t texObj = 0;
                    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
                    
                    // Create Surface Object (Write) - for later
                    // Be careful creating surface on same array while reading as texture?
                    // Safe if read/write hazards managed. We read then write different pixels? 
                    // No, we write different pass.
                    // Preprocess kernel reads.
                    
                    // 1. Preprocess
                    launch_preprocess_kernel(texObj, d_input, WIDTH, HEIGHT, stream);
                    
                    if (save_requested) {
                        launch_debug_tensor_dump(d_input, d_debug_img, 512, 512, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream)); // Wait for dump
                        
                        std::vector<unsigned char> h_debug(512 * 512 * 3);
                        CUDA_CHECK(cudaMemcpy(h_debug.data(), d_debug_img, debug_size, cudaMemcpyDeviceToHost));
                        
                        // Save PPM
                        std::ofstream ppm("debug_input.ppm", std::ios::binary);
                        ppm << "P6\n512 512\n255\n";
                        ppm.write((char*)h_debug.data(), h_debug.size());
                        ppm.close();
                        
                        std::cout << "Saved debug_input.ppm!" << std::endl;
                        save_requested = false;
                    }

                    CUDA_CHECK(cudaDestroyTextureObject(texObj)); // Destroy after launch (async?) 
                    
                    // 2. Inference
                    pipeline.Inference(stream, d_input, d_output);
                    
                    // 3. Postprocess
                    cudaResourceDesc surfResDesc = {};
                    surfResDesc.resType = cudaResourceTypeArray;
                    surfResDesc.res.array.array = mapped_array;
                    
                    cudaSurfaceObject_t surfObj = 0;
                    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &surfResDesc));
                    
                    launch_postprocess_kernel(d_output, surfObj, WIDTH, HEIGHT, stream);
                    
                    // Destroy objects
                    // We need to wait for stream before destroying objects? 
                    // documentation says "The texture object is defined by... The state is captured... except for the resource..."
                    // "The resource... must not be freed...". mapped_array is valid until unmap.
                    // TextureObject handle itself? 
                    // "cudaDestroyTextureObject() ... destroys the texture object."
                    // If specific hardware state is tied, it might need to wait.
                    // To be safe, verify. But usually destruction is host-side handle release.
                    // Let's defer destruction or Sync.
                    // Syncing every frame is bad for pipelining but okay for latency-sensitive loop?
                    // Actually, let's just destroy at end.
                    
                    // Unmap
                    // Unmap implies synchronization for graphics? 
                    // "Unmap... implicitly synchronizes...?" No.
                    // "Subsequent usage by D3D... will see..."
                    // We must ensure CUDA is done before Unmap? 
                    // Usually yes, or use semaphores.
                    // Simple way: cudaStreamSynchronize(stream).
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    
                    CUDA_CHECK(cudaDestroyTextureObject(texObj));
                    CUDA_CHECK(cudaDestroySurfaceObject(surfObj));
                    
                    unmap_d3d11_resource(cuda_res);

                    // Present
                    ComPtr<ID3D11Texture2D> back_buffer;
                    DX_CHECK(swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
                    context->CopyResource(back_buffer.Get(), shared_texture.Get());
                    
                    DX_CHECK(swap_chain->Present(0, DXGI_PRESENT_ALLOW_TEARING));
                    
                    frames++;
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = now - start_time;
                    if (diff.count() >= 1.0) {
                        std::cout << "FPS: " << frames << std::endl;
                        frames = 0;
                        start_time = now;
                    }
                }
            }
        }
        
        // Cleanup
        cudaStreamDestroy(stream);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_debug_img);
        unregister_d3d11_resource(cuda_res);
        
    } catch (const std::exception& e) {
        MessageBox(NULL, e.what(), "Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}
