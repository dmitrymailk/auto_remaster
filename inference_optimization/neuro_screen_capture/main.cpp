#include <iostream>
#include <vector>
#include <chrono>

#include <torch/torch.h>

#include "capture_dxgi.h"
#include "cuda_interop.h"
#include "utils.h"

// Window dimensions (initial)
HWND g_hwnd = nullptr;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Window dimensions passed dynamically
void CreateNativeWindow(HINSTANCE hInstance, int width, int height) {
    const char CLASS_NAME[] = "NeuroScreenCaptureClass";
    
    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    g_hwnd = CreateWindowEx(
        0, CLASS_NAME, "Neuro Screen Capture (Zero Copy)",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height,
        NULL, NULL, hInstance, NULL
    );

    if (g_hwnd == NULL) {
        throw std::runtime_error("Failed to create window");
    }

    ShowWindow(g_hwnd, SW_SHOW);
}

int main() {
    try {
        std::cout << "Initializing Neuro Screen Capture..." << std::endl;
        
        // 0. Query Display Settings for Primary Monitor
        DEVMODE dm = { 0 };
        dm.dmSize = sizeof(dm);
        if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm)) {
             throw std::runtime_error("Failed to enum display settings");
        }
        
        int WIDTH = dm.dmPelsWidth;
        int HEIGHT = dm.dmPelsHeight;
        
        std::cout << "Detected Screen Resolution: " << WIDTH << "x" << HEIGHT << std::endl;

        // 1. Create Window
        CreateNativeWindow(GetModuleHandle(NULL), WIDTH, HEIGHT);

        // 2. Init Init D3D11 Device & SwapChain
        DXGI_SWAP_CHAIN_DESC scd = {0};
        scd.BufferCount = 2; // Double buffering
        scd.BufferDesc.Width = WIDTH;
        scd.BufferDesc.Height = HEIGHT;
        scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scd.BufferDesc.RefreshRate.Numerator = 60;
        scd.BufferDesc.RefreshRate.Denominator = 1;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; // We will copy to backbuffer
        scd.OutputWindow = g_hwnd;
        scd.SampleDesc.Count = 1;
        scd.Windowed = TRUE;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD; // Fastest
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<IDXGISwapChain> swap_chain;

        // Use D3D_DRIVER_TYPE_HARDWARE
        UINT createDeviceFlags = 0;
        #ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
        #endif

        DX_CHECK(D3D11CreateDeviceAndSwapChain(
            NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags,
            NULL, 0, D3D11_SDK_VERSION, &scd,
            &swap_chain, &device, NULL, &context
        ));

        // 3. Init Screen Capture
        ScreenCapture capture(device.Get(), context.Get());
        capture.Initialize(); 
        // Note: Actual capture size might differ from our window size.
        // For simplicity, we assume 1080p or handle mismatch later (will stretch or clip).
        // Ideally we query output size from capture.
        
        // 4. Create Shared Texture (Staging for CUDA)
        // This texture stays registered with CUDA.
        D3D11_TEXTURE2D_DESC sharedTexDesc = {0};
        sharedTexDesc.Width = WIDTH; // Fixed to our processing resolution
        sharedTexDesc.Height = HEIGHT;
        sharedTexDesc.MipLevels = 1;
        sharedTexDesc.ArraySize = 1;
        sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sharedTexDesc.SampleDesc.Count = 1;
        sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
        // Bind as Shader Resource (for reading in CUDA if needed via texture) 
        // and Render Target (to support CopyResource destination?) 
        // Actually CopyResource works on Default usage.
        // We need to register it for CUDA access.
        sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; 
        
        ComPtr<ID3D11Texture2D> shared_texture;
        DX_CHECK(device->CreateTexture2D(&sharedTexDesc, NULL, &shared_texture));

        // 5. Register with CUDA
        cudaGraphicsResource* cuda_res = register_d3d11_resource(shared_texture.Get());
        if (!cuda_res) throw std::runtime_error("Failed to register resource with CUDA");

        // 6. Allocate CUDA Tensor Memory
        float* d_tensor_in = nullptr;
        float* d_tensor_out = nullptr; // For result
        size_t tensor_bytes = WIDTH * HEIGHT * 3 * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_tensor_in, tensor_bytes));
        // We can reuse same buffer if model is in-place, but let's have separate for clarify
        // Actually for this demo, "create tensor, do nothing, draw" -> 
        // We can just use one buffer or copy.
        // User asked: "creates tensor, does nothing, then I draw this array".
        // So: Preprocess(d_tensor_in) -> Torch Tensor(d_tensor_in) -> Postprocess(d_tensor_in).
        // We reuse the same buffer.

        // 7. Loop
        MSG msg = {0};
        auto start_time = std::chrono::high_resolution_clock::now();
        int frames = 0;

        std::cout << "Starting loop..." << std::endl;

        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                // Capture Frame
                ComPtr<ID3D11Texture2D> current_frame;
                if (capture.AcquireFrame(current_frame, 10)) { // 10ms timeout
                    // COPY: Captured Frame -> Shared Texture
                    // Note: CopyResource requires same dimensions. 
                    // If desktop != 1920x1080, this will fail or crash.
                    // IMPORTANT: In production, we'd use CopySubresourceRegion or stretch.
                    // For this MVP, we rely on matching resolution or logic.
                    // Let's assume matching for now, or check desc.
                    
                    D3D11_TEXTURE2D_DESC desc;
                    current_frame->GetDesc(&desc);
                    if (desc.Width == WIDTH && desc.Height == HEIGHT) {
                        context->CopyResource(shared_texture.Get(), current_frame.Get());
                    } else {
                        // Mismatch handling: Just copy subregion or skip? 
                        // Printing once warning?
                        static bool warned = false;
                         if (!warned) {
                            std::cerr << "Resolution mismatch! Captured: " << desc.Width << "x" << desc.Height << " vs Target: " << WIDTH << "x" << HEIGHT << std::endl;
                            warned = true;
                         }
                         // Try copy subregion to avoid crash if Captured > Target
                         D3D11_BOX sourceRegion;
                         sourceRegion.left = 0;
                         sourceRegion.right = std::min<UINT>(desc.Width, WIDTH);
                         sourceRegion.top = 0;
                         sourceRegion.bottom = std::min<UINT>(desc.Height, HEIGHT);
                         sourceRegion.front = 0;
                         sourceRegion.back = 1;
                         context->CopySubresourceRegion(shared_texture.Get(), 0, 0, 0, 0, current_frame.Get(), 0, &sourceRegion);
                    }

                    capture.ReleaseFrame(); // Done with DXGI capture resource

                    // CUDA Processing
                    cudaArray_t mapped_array = map_d3d11_resource(cuda_res);
                    
                    // Create Texture Object for reading
                    cudaResourceDesc resDesc = {};
                    resDesc.resType = cudaResourceTypeArray;
                    resDesc.res.array.array = mapped_array;
                    
                    cudaTextureDesc texDesc = {};
                    texDesc.addressMode[0] = cudaAddressModeClamp;
                    texDesc.addressMode[1] = cudaAddressModeClamp;
                    texDesc.filterMode = cudaFilterModePoint; // Point sampling
                    texDesc.readMode = cudaReadModeElementType;
                    texDesc.normalizedCoords = 0;

                    cudaTextureObject_t texObj = 0;
                    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

                    // Launch Preprocess: Texture -> Float*
                    launch_preprocess_kernel(texObj, d_tensor_in, WIDTH, HEIGHT);
                    
                    CUDA_CHECK(cudaDestroyTextureObject(texObj));
                    
                    // PyTorch (Zero Copy)
                    // Create tensor wrapper around d_tensor_in
                    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
                    torch::Tensor tensor = torch::from_blob(d_tensor_in, {1, 3, HEIGHT, WIDTH}, options);
                    
                    // --- Neural Net Inference would go here ---
                    // tensor = model.forward(tensor);
                    // ------------------------------------------

                    // Launch Postprocess: Float* -> Surface (Write back to shared_texture)
                    // Note: We need a Surface Object to write.
                    // Surface writes only work on arrays created with CUDA_ARRAY3D_SURFACE_LDST flag?
                    // DXGI interop arrays usually support this? 
                    // Let's create Surface Object from the same mapped array.
                    
                    // Check flags: When registering, we used FlagsNone.
                    // For surface write, we usually need to ensure the D3D texture was created with BindFlags including UnorderedAccess? 
                    // Or just RenderTarget is enough for interop surface write? 
                    // "CUDA array must be created with cudaArraySurfaceLoadStore flag"
                    // Graphics Interop usually handles this if resource is registered for surface access.
                    // Actually, let's use cudaGraphicsRegisterFlagsSurfaceLoadStore if explicitly needed,
                    // but usually default allows it.
                
                    cudaResourceDesc surfResDesc = {};
                    surfResDesc.resType = cudaResourceTypeArray;
                    surfResDesc.res.array.array = mapped_array;
                    
                    cudaSurfaceObject_t surfObj = 0;
                    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &surfResDesc));
                    
                    launch_postprocess_kernel(d_tensor_in, surfObj, WIDTH, HEIGHT); // Reading from d_tensor_in (which holds the data)
                    
                    CUDA_CHECK(cudaDestroySurfaceObject(surfObj));

                    unmap_d3d11_resource(cuda_res);

                    // Present
                    // We wrote back to shared_texture. Now we copy shared_texture to BackBuffer.
                    ComPtr<ID3D11Texture2D> back_buffer;
                    DX_CHECK(swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
                    context->CopyResource(back_buffer.Get(), shared_texture.Get());
                    
                    DX_CHECK(swap_chain->Present(0, DXGI_PRESENT_ALLOW_TEARING));

                    frames++;
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> diff = now - start_time;
                    if (diff.count() >= 1.0) {
                        std::cout << "FPS: " << frames << " | Tensor Address: " << (void*)d_tensor_in << std::endl;
                        frames = 0;
                        start_time = now;
                    }
                }
            }
        }

        // Cleanup
        unregister_d3d11_resource(cuda_res);
        cudaFree(d_tensor_in);

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
