#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <memory> 

#include "utils.h"
#include "capture_interface.h"
#include "capture_dxgi.h"
#include "capture_wgc.h" 
#include "window_helper.h"
#include "pipeline.h"
#include "cuda_utils.h"
#include <cuda_d3d11_interop.h>
#include "recorder.h" // Added Recorder

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
        // Initializes COM for WGC
        winrt::init_apartment(winrt::apartment_type::single_threaded);

        std::cout << "Initializing Neuro Screen Capture (TensorRT VAE)..." << std::endl;
        
        // --- Capture Mode Selection ---
        std::cout << "Select Capture Mode:\n"
                  << "[0] Full Screen (Monitor 1)\n"
                  << "[1] Specific Window\n"
                  << "Enter mode: ";
        int mode = 0;
        if (!(std::cin >> mode)) {
            mode = 0; // Default
            std::cin.clear();
            std::cin.ignore(10000, '\n');
        }

        HWND target_window = nullptr;
        if (mode == 1) {
             while (true) {
                auto windows = EnumerateWindows();
                target_window = SelectWindow(windows);
                if (target_window) break;
                
                std::cout << "Retry? (y/n): ";
                char c; std::cin >> c;
                if (c != 'y') return 0;
             }
        }
        
        // 1. Create Window (Client area MODEL_SIZE x MODEL_SIZE initially)
        int window_width = MODEL_SIZE;
        #if SPLIT_SCREEN
        window_width = MODEL_SIZE * 2;
        std::cout << "Split Screen Enabled. Window Width: " << window_width << std::endl;
        #endif
        CreateNativeWindow(GetModuleHandle(NULL), window_width, MODEL_SIZE);

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

        // 3. Screen Capture Setup
        std::unique_ptr<ICapture> capture;

        if (mode == 1 && target_window) {
            auto wgc = std::make_unique<WindowCapture>(device.Get(), context.Get());
            wgc->Start(target_window);
            capture = std::move(wgc);
        } else {
            auto dxgi = std::make_unique<ScreenCapture>(device.Get(), context.Get());
            dxgi->Initialize();
            capture = std::move(dxgi);
        }

        // Get actual capture dimensions
        UINT WIDTH = capture->GetWidth();
        UINT HEIGHT = capture->GetHeight();
        std::cout << "Capture Resolution: " << WIDTH << "x" << HEIGHT << std::endl;

        // 4. TensorRT Pipeline
        // Load TensorRT Models... 
        TensorRTPipeline pipeline;
        pipeline.LoadEngines("vae_encoder.plan", "vae_decoder.plan");
        #if ENABLE_UNET
        pipeline.LoadUNet("unet.plan");
        #endif

        // 6. Textures
        // Input Texture (Capture Resolution)
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

        // Output Texture (Matches Window Size for Split Screen) - for Inference Output
        D3D11_TEXTURE2D_DESC outputDesc = inputDesc;
        outputDesc.Width = MODEL_SIZE;
        #if SPLIT_SCREEN
        outputDesc.Width = MODEL_SIZE * 2;
        #endif
        outputDesc.Height = MODEL_SIZE;
        
        ComPtr<ID3D11Texture2D> d3d_output_texture;
        DX_CHECK(device->CreateTexture2D(&outputDesc, NULL, &d3d_output_texture));

        // Register with CUDA
        cudaGraphicsResource* cuda_tex_in = register_d3d11_resource(d3d_input_texture.Get());
        cudaGraphicsResource* cuda_tex_out = register_d3d11_resource(d3d_output_texture.Get());
        if (!cuda_tex_in || !cuda_tex_out) throw std::runtime_error("Failed to register textures with CUDA");

        // 6. Allocate Tensor Memory (MODEL_SIZE x MODEL_SIZE)
        // Format: NCHW FP16 (2 bytes per element)
        void *d_input, *d_output;
        size_t tensor_elements = 3 * MODEL_SIZE * MODEL_SIZE;
        size_t tensor_bytes = tensor_elements * 2; 
        
        CUDA_CHECK(cudaMalloc(&d_input, tensor_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, tensor_bytes));
        
        // 7. Loop
        MSG msg = {0};
        auto start_time = std::chrono::high_resolution_clock::now();
        int frames = 0;
        

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Recorder Setup
        bool is_split = SPLIT_SCREEN;
        int rec_w = is_split ? MODEL_SIZE * 2 : MODEL_SIZE;
        int rec_h = MODEL_SIZE;
        Recorder recorder(rec_w, rec_h, 24);
        
        void* d_record_buffer = nullptr;
        size_t record_buffer_size = rec_w * rec_h * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_record_buffer, record_buffer_size));
        
        // Recording State
        auto last_record_time = std::chrono::steady_clock::now();
        double record_interval = 1.0 / 24.0; 

        // Debug Buffer (RGB 512x512)
        unsigned char* d_debug_img = nullptr;
        size_t debug_size = MODEL_SIZE * MODEL_SIZE * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_debug_img, debug_size));
        bool save_requested = false;

        bool is_overlay_mode = false;
        bool f9_pressed_last = false; // Debounce F9
        bool f10_pressed_last = false; // Debounce F10

        auto ToggleOverlay = [&](bool enable) {
            is_overlay_mode = enable;
            if (is_overlay_mode) {
                // Overlay Mode: Borderless, TopMost, Transparent to Input
                SetWindowLong(g_hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
                SetWindowLong(g_hwnd, GWL_EXSTYLE, WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST);
                // 255 = Opaque (Visuals visible), but EX_TRANSPARENT makes input fall through
                SetLayeredWindowAttributes(g_hwnd, 0, 255, LWA_ALPHA);
                std::cout << "[Overlay] Enabled. Press F9 to disable." << std::endl;
            } else {
                // Normal Mode: Windowed, Borders, Interactive
                SetWindowLong(g_hwnd, GWL_STYLE, WS_OVERLAPPEDWINDOW | WS_VISIBLE);
                SetWindowLong(g_hwnd, GWL_EXSTYLE, 0);
                SetWindowPos(g_hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED);
                std::cout << "[Overlay] Disabled." << std::endl;
            }
        };

        std::cout << "Starting Loop... \nPress 'S' to save debugging image.\nPress 'F9' to toggle Overlay Mode.\nPress 'F10' to toggle Recording." << std::endl;

        while (msg.message != WM_QUIT) {
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_KEYDOWN && msg.wParam == 'S') {
                    save_requested = true;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                // --- Hotkey Handling ---
                bool f9_down = (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
                if (f9_down && !f9_pressed_last) {
                    ToggleOverlay(!is_overlay_mode);
                }
                f9_pressed_last = f9_down;

                // F10 Recording Toggle
                bool f10_down = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
                if (f10_down && !f10_pressed_last) {
                    if (recorder.IsRecording()) {
                        recorder.Stop();
                    } else {
                        std::string prog = (mode == 1 && target_window) ? "Window" : "Desktop"; // Simplified, ideally get window title
                        recorder.Start(prog);
                    }
                }
                f10_pressed_last = f10_down;

                // --- Overlay Tracking Logic ---
                if (is_overlay_mode && mode == 1 && target_window && IsWindow(target_window)) {
                    RECT targetRect;
                    GetWindowRect(target_window, &targetRect);
                    
                    int target_w = targetRect.right - targetRect.left;
                    int target_h = targetRect.bottom - targetRect.top;
                    
                    // Match the target window size exactly
                    // The rendering logic will automatically center the 512x512 content
                    // and fill the rest with black (ClearRenderTargetView).
                    
                    SetWindowPos(g_hwnd, HWND_TOPMOST, targetRect.left, targetRect.top, target_w, target_h, SWP_NOACTIVATE | SWP_SHOWWINDOW);
                }

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
                if (capture->AcquireFrame(current_frame, 10)) {
                    
                    D3D11_TEXTURE2D_DESC frameDesc;
                    current_frame->GetDesc(&frameDesc);

                    if (frameDesc.Width != WIDTH || frameDesc.Height != HEIGHT) {
                        // Handle Resize
                        std::cout << "Resize Detected: " << WIDTH << "x" << HEIGHT << " -> " << frameDesc.Width << "x" << frameDesc.Height << std::endl;
                        
                        // 1. Unregister old resource
                        if (cuda_tex_in) {
                            unregister_d3d11_resource(cuda_tex_in);
                            cuda_tex_in = nullptr;
                        }
                        
                        // 2. Update Dimensions
                        WIDTH = frameDesc.Width;
                        HEIGHT = frameDesc.Height;
                        
                        // 3. Recreate D3D Texture
                        inputDesc.Width = WIDTH;
                        inputDesc.Height = HEIGHT;
                        d3d_input_texture.Reset(); // Release old
                        DX_CHECK(device->CreateTexture2D(&inputDesc, NULL, &d3d_input_texture));
                        
                        // 4. Register new resource
                        cuda_tex_in = register_d3d11_resource(d3d_input_texture.Get());
                        if (!cuda_tex_in) throw std::runtime_error("Failed to register resized texture with CUDA");
                    }

                    // 1. Copy Capture -> Input Texture
                    context->CopyResource(d3d_input_texture.Get(), current_frame.Get());
                    capture->ReleaseFrame();

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
                    
                    // Launch Preprocess (Computes FP16 Tensor)
                    launch_preprocess_kernel(texObj, d_input, WIDTH, HEIGHT, stream);
                    
                    if (save_requested) {
                        // Save Raw FP16 Tensor (NCHW)
                        size_t tensor_size_bytes = MODEL_SIZE * MODEL_SIZE * 3 * sizeof(unsigned short); // half = 2 bytes
                        std::vector<char> h_tensor(tensor_size_bytes);
                        
                        CUDA_CHECK(cudaMemcpy(h_tensor.data(), d_input, tensor_size_bytes, cudaMemcpyDeviceToHost));
                        CUDA_CHECK(cudaStreamSynchronize(stream));
                        
                        std::ofstream bin("capture_input_fp16.bin", std::ios::binary);
                        bin.write(h_tensor.data(), tensor_size_bytes);
                        bin.close();
                        
                        std::cout << "Saved capture_input_fp16.bin (" << tensor_size_bytes << " bytes)" << std::endl;
                    }

                    CUDA_CHECK(cudaDestroyTextureObject(texObj));
                    unmap_d3d11_resource(cuda_tex_in); // Unmap Input immediately

                    // 3. Inference
                    pipeline.Inference(stream, d_input, d_output, save_requested);
                    if (save_requested) save_requested = false;

                    // 4. Recording Logic (Post-Inference to capture Input + Output)
                    auto now_steady = std::chrono::steady_clock::now();
                    std::chrono::duration<double> time_since_record = now_steady - last_record_time;
                    if (recorder.IsRecording() && (time_since_record.count() >= record_interval)) {
                        last_record_time = now_steady;
                        
                        // Concatenate or Convert to RGB
                        if (is_split) {
                            launch_concat_tensors_kernel(d_input, d_output, d_record_buffer, MODEL_SIZE, stream);
                        } else {
                            // Capture Output (Processed)
                            launch_tensor_to_u8_rgb_kernel(d_output, d_record_buffer, MODEL_SIZE, stream);
                        }
                        
                        // Schedule write
                        recorder.Capture(d_record_buffer, stream);
                    }
                    
                    // 5. Postprocess (Tensor -> Output Texture)
                    // If SPLIT_SCREEN:
                    //   Left (0): Processed (d_output)
                    //   Right (MODEL_SIZE): Original (d_input)

                    cudaArray_t arr_out = map_d3d11_resource(cuda_tex_out);
                    
                    cudaResourceDesc surfResDesc = {};
                    surfResDesc.resType = cudaResourceTypeArray;
                    surfResDesc.res.array.array = arr_out;
                    
                    cudaSurfaceObject_t surfObj = 0;
                    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &surfResDesc));
                    
                    // Always render to MODEL_SIZE x MODEL_SIZE fixed output
                    int processed_off_x = 0;
                    
                    #if SPLIT_SCREEN
                    processed_off_x = MODEL_SIZE; // Right side
                    // Draw Original to Left (0)
                    launch_postprocess_kernel(d_input, surfObj, MODEL_SIZE, MODEL_SIZE, 0, 0, stream);
                    #endif

                    // Draw Processed
                    launch_postprocess_kernel(d_output, surfObj, MODEL_SIZE, MODEL_SIZE, processed_off_x, 0, stream);
                    
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
                    
                    // Recalculate copy params for correct width
                    int tex_w = MODEL_SIZE;
                    #if SPLIT_SCREEN
                    tex_w = MODEL_SIZE * 2;
                    #endif
                    
                    int tgt_x = (win_w - tex_w) / 2;
                    int tgt_y = (win_h - MODEL_SIZE) / 2;
                    
                    int src_x = 0; int src_y = 0;
                    int dst_x = tgt_x; int dst_y = tgt_y;
                    int copy_w = tex_w; 
                    int copy_h = MODEL_SIZE;

                    if (tgt_x < 0) { src_x = -tgt_x; dst_x = 0; copy_w = win_w; }
                    if (tgt_y < 0) { src_y = -tgt_y; dst_y = 0; copy_h = win_h; }

                    D3D11_BOX srcBox;
                    srcBox.left = src_x;
                    srcBox.right = src_x + copy_w;
                    srcBox.top = src_y;
                    srcBox.bottom = src_y + copy_h;
                    srcBox.front = 0;
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
        cudaFree(d_record_buffer); // Free Record Buffer
        
        if (cuda_tex_in) unregister_d3d11_resource(cuda_tex_in);
        if (cuda_tex_out) unregister_d3d11_resource(cuda_tex_out);
        
    } catch (const std::exception& e) {
        MessageBox(NULL, e.what(), "Error", MB_OK | MB_ICONERROR);
        return 1;
    }
    return 0;
}

