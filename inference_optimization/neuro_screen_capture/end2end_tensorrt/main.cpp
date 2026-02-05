#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>

#include "utils.h"
#include "capture_dxgi.h"
#include "pipeline.h"
#include "cuda_utils.h"
#include <cuda_d3d11_interop.h>

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

    // Adjust window size to client area
    RECT rect = { 0, 0, width, height };
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

    g_hwnd = CreateWindowEx(
        0, CLASS_NAME, "Neuro Screen Capture (TensorRT VAE)",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 
        rect.right - rect.left, rect.bottom - rect.top,
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

        // 1. Create Window (Client area 512x512 initially)
        CreateNativeWindow(GetModuleHandle(NULL), 512, 512);

        // 2. D3D11 Setup
        DXGI_SWAP_CHAIN_DESC scd = {0};
        scd.BufferCount = 2;
        scd.BufferDesc.Width = 0; // Use window size
        scd.BufferDesc.Height = 0;
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
        ComPtr<IDXGISwapChain> swap_chain; 
        
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
        std::string enc_path = "vae_encoder.plan";
        std::string dec_path = "vae_decoder.plan";
        
        TensorRTPipeline pipeline;
        pipeline.LoadEngines(enc_path, dec_path);

        // 5. Textures
        // Input Texture (Screen Resolution) - for Capture Copy
        D3D11_TEXTURE2D_DESC inputDesc = {0};
        inputDesc.Width = WIDTH;
        inputDesc.Height = HEIGHT;
        inputDesc.MipLevels = 1;
        inputDesc.ArraySize = 1;
        inputDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        inputDesc.SampleDesc.Count = 1;
        inputDesc.Usage = D3D11_USAGE_DEFAULT;
        inputDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET; 
        
        ComPtr<ID3D11Texture2D> d3d_input_texture;
        DX_CHECK(device->CreateTexture2D(&inputDesc, NULL, &d3d_input_texture));

        // Output Texture (Fixed 512x512) - for Inference Output
        D3D11_TEXTURE2D_DESC outputDesc = inputDesc;
        outputDesc.Width = 512;
        outputDesc.Height = 512;
        
        ComPtr<ID3D11Texture2D> d3d_output_texture;
        DX_CHECK(device->CreateTexture2D(&outputDesc, NULL, &d3d_output_texture));

        // Register with CUDA
        cudaGraphicsResource* cuda_tex_in = register_d3d11_resource(d3d_input_texture.Get());
        cudaGraphicsResource* cuda_tex_out = register_d3d11_resource(d3d_output_texture.Get());
        if (!cuda_tex_in || !cuda_tex_out) throw std::runtime_error("Failed to register textures with CUDA");

        // 6. Allocate Tensor Memory (512x512)
        // Format: NCHW FP16 (2 bytes per element)
        void *d_input, *d_output;
        size_t tensor_elements = 3 * 512 * 512;
        size_t tensor_bytes = tensor_elements * 2; 
        
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
                RECT clientRect;
                GetClientRect(g_hwnd, &clientRect);
                int win_w = clientRect.right - clientRect.left;
                int win_h = clientRect.bottom - clientRect.top;
                
                // Ensure swapchain matches window size
                static int prev_win_w = 0;
                static int prev_win_h = 0;
                
                if (win_w != prev_win_w || win_h != prev_win_h) {
                    if (win_w > 0 && win_h > 0) {
                        context->OMSetRenderTargets(0, 0, 0);
                        swap_chain->ResizeBuffers(0, win_w, win_h, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH | DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);
                        prev_win_w = win_w;
                        prev_win_h = win_h;
                    }
                }

                ComPtr<ID3D11Texture2D> current_frame;
                if (capture.AcquireFrame(current_frame, 10)) {
                    
                    // 1. Copy Capture -> Input Texture (Screen Size)
                    
                    // Simple full copy as sizes match (Screen Resolution)
                    context->CopyResource(d3d_input_texture.Get(), current_frame.Get());
                    capture.ReleaseFrame();

                    // 2. Preprocess (Input Texture -> Tensor)
                    // Map Input
                    cudaArray_t arr_in = map_d3d11_resource(cuda_tex_in);
                    
                    cudaResourceDesc resDesc = {};
                    resDesc.resType = cudaResourceTypeArray;
                    resDesc.res.array.array = arr_in;
                    
                    cudaTextureDesc texDesc = {};
                    texDesc.addressMode[0] = cudaAddressModeClamp;
                    texDesc.addressMode[1] = cudaAddressModeClamp;
                    texDesc.filterMode = cudaFilterModeLinear;
                    texDesc.readMode = cudaReadModeNormalizedFloat;
                    texDesc.normalizedCoords = 0;
                    
                    cudaTextureObject_t texObj = 0;
                    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
                    
                    launch_preprocess_kernel(texObj, d_input, WIDTH, HEIGHT, stream);
                    
                    if (save_requested) {
                        launch_debug_tensor_dump(d_input, d_debug_img, 512, 512, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        
                        std::vector<unsigned char> h_debug(512 * 512 * 3);
                        CUDA_CHECK(cudaMemcpy(h_debug.data(), d_debug_img, debug_size, cudaMemcpyDeviceToHost));
                        
                        std::ofstream ppm("debug_input.ppm", std::ios::binary);
                        ppm << "P6\n512 512\n255\n";
                        ppm.write((char*)h_debug.data(), h_debug.size());
                        ppm.close();
                        std::cout << "Saved debug_input.ppm!" << std::endl;
                        save_requested = false;
                    }

                    CUDA_CHECK(cudaDestroyTextureObject(texObj));
                    unmap_d3d11_resource(cuda_tex_in); // Unmap Input immediately

                    // 3. Inference
                    pipeline.Inference(stream, d_input, d_output);
                    
                    // 4. Postprocess (Tensor -> Output Texture 512x512)
                    cudaArray_t arr_out = map_d3d11_resource(cuda_tex_out);
                    
                    cudaResourceDesc surfResDesc = {};
                    surfResDesc.resType = cudaResourceTypeArray;
                    surfResDesc.res.array.array = arr_out;
                    
                    cudaSurfaceObject_t surfObj = 0;
                    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &surfResDesc));
                    
                    // Always render to 512x512 fixed output
                    launch_postprocess_kernel(d_output, surfObj, 512, 512, stream);
                    
                    CUDA_CHECK(cudaStreamSynchronize(stream)); // Finish writing to surface logic
                    CUDA_CHECK(cudaDestroySurfaceObject(surfObj));
                    unmap_d3d11_resource(cuda_tex_out);

                    // 5. Present (Copy Output Texture -> Backbuffer Center)
                    ComPtr<ID3D11Texture2D> back_buffer;
                    DX_CHECK(swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
                    
                    ComPtr<ID3D11RenderTargetView> rtv;
                    DX_CHECK(device->CreateRenderTargetView(back_buffer.Get(), NULL, &rtv));
                    
                    // Clear Background (Black)
                    float black[] = {0.0f, 0.0f, 0.0f, 1.0f};
                    context->ClearRenderTargetView(rtv.Get(), black);
                    
                    // Calculate Center Offset
                    int tgt_x = (win_w - 512) / 2;
                    int tgt_y = (win_h - 512) / 2;
                    
                    // Clip if window is smaller than 512
                    int src_x = 0, src_y = 0;
                    int dst_x = tgt_x, dst_y = tgt_y;
                    int copy_w = 512, copy_h = 512;
                    
                    if (tgt_x < 0) { src_x = -tgt_x; dst_x = 0; copy_w = win_w; }
                    if (tgt_y < 0) { src_y = -tgt_y; dst_y = 0; copy_h = win_h; }
                    
                    D3D11_BOX srcBox;
                    srcBox.left = src_x;
                    srcBox.top = src_y;
                    srcBox.front = 0;
                    srcBox.right = src_x + copy_w;
                    srcBox.bottom = src_y + copy_h;
                    srcBox.back = 1;

                    context->CopySubresourceRegion(back_buffer.Get(), 0, dst_x, dst_y, 0, d3d_output_texture.Get(), 0, &srcBox);
                    
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
        
        if (cuda_tex_in) unregister_d3d11_resource(cuda_tex_in);
        if (cuda_tex_out) unregister_d3d11_resource(cuda_tex_out);
        
    } catch (const std::exception& e) {
        MessageBox(NULL, e.what(), "Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}

