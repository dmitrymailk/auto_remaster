#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <memory> 
#include <string>

#include "utils.h"
#include "capture_interface.h"
#include "capture_dxgi.h"
#include "capture_wgc.h" 
#include "window_helper.h"
#include "pipeline.h"
#include "cuda_utils.h"
#include <cuda_d3d11_interop.h>
#include "recorder.h" 

#include "config.h" // Ensure config is included for ENABLE_VSR

#if ENABLE_VSR
#include "vsr_upscaler.h"
#endif

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
        0, CLASS_NAME, "Neuro Screen Capture (TensorRT VAE + VSR)",
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
        
        // 1. Create Window
        // Calculate Base Window Width (Inference Resolution)
        int base_width = MODEL_SIZE;
        #if SPLIT_SCREEN
        base_width = MODEL_SIZE * 2;
        std::cout << "Split Screen Enabled." << std::endl;
        #endif

        // VSR scales are calculated but window starts at base resolution
        // VSR will be enabled dynamically via F11
        int display_width_scaled = base_width;
        int display_height_scaled = MODEL_SIZE;

        #if ENABLE_VSR
        std::cout << "VSR Available. Target Scale: " << VSR_SCALE << "x" << std::endl;
        display_width_scaled = static_cast<int>(base_width * VSR_SCALE);
        display_height_scaled = static_cast<int>(MODEL_SIZE * VSR_SCALE);
        #endif

        // Start with base resolution (VSR starts DISABLED)
        std::cout << "Initial Display Resolution: " << base_width << "x" << MODEL_SIZE << std::endl;
        std::cout << "VSR Scaled Resolution: " << display_width_scaled << "x" << display_height_scaled << std::endl;
        CreateNativeWindow(GetModuleHandle(NULL), base_width, MODEL_SIZE);

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
        
        UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        #ifdef _DEBUG
        createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
        #endif

        // Create Factory to enumerate adapters
        ComPtr<IDXGIFactory1> factory;
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);

        ComPtr<IDXGIAdapter1> adapter;
        ComPtr<IDXGIAdapter1> selectedAdapter;
        DXGI_ADAPTER_DESC1 selectedDesc;

        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            std::wstring name(desc.Description);
            if (name.find(L"NVIDIA") != std::wstring::npos) {
                selectedAdapter = adapter;
                selectedDesc = desc;
                break;
            }
        }

        if (selectedAdapter) {
            std::wcout << L"[Main] Selected Adapter: " << selectedDesc.Description << std::endl;
        } else {
            std::cerr << "[Main] WARNING: NVIDIA Adapter not found! Using default." << std::endl;
        }

        // Use UNKNOWN driver type when adapter is specified
        D3D_DRIVER_TYPE driverType = selectedAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;

        DX_CHECK(D3D11CreateDeviceAndSwapChain(
            selectedAdapter.Get(), driverType, NULL, createDeviceFlags,
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
        
        // 4.1. VSR Init
        #if ENABLE_VSR
        std::cout << "[Main] Initializing VSR..." << std::endl;
        std::unique_ptr<VSRUpscaler> vsr = std::make_unique<VSRUpscaler>(device.Get(), context.Get());
        // Initialize for Base -> Scaled
        // Ensure dimensions are valid
        std::cout << "[Main] VSR Init Params: Base=" << base_width << "x" << MODEL_SIZE << " Display=" << display_width_scaled << "x" << display_height_scaled << std::endl;
        
        if (!vsr->Initialize(base_width, MODEL_SIZE, display_width_scaled, display_height_scaled)) {
            std::cerr << "VSR Initialization Failed! Falling back to standard display." << std::endl;
            // Handle fallback or exit? For now proceeded but vsr will do nothing.
        } else {
             std::cout << "[Main] VSR Initialized successfully." << std::endl;
        }
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

        // Inference Output Texture (Base Resolution)
        // Matches Window Size for Split Screen
        D3D11_TEXTURE2D_DESC outputDesc = inputDesc;
        outputDesc.Width = base_width;
        outputDesc.Height = MODEL_SIZE;
        // Make sure it can be used by VSR (Shader Resource) and by CUDA (Render Target not strictly needed for CUDA write, but UAV might be)
        // D3D11: CUDA interop usually works with default usage.
        
        ComPtr<ID3D11Texture2D> d3d_output_texture;
        DX_CHECK(device->CreateTexture2D(&outputDesc, NULL, &d3d_output_texture));
        
        #if ENABLE_VSR
        // VSR Output Texture (Scaled Resolution)
        D3D11_TEXTURE2D_DESC vsrOutDesc = outputDesc;
        vsrOutDesc.Width = display_width_scaled;
        vsrOutDesc.Height = display_height_scaled;
        vsrOutDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_RENDER_TARGET; 
        // Note: NGX writes via UAV usually.
        
        ComPtr<ID3D11Texture2D> d3d_vsr_output_texture;
        DX_CHECK(device->CreateTexture2D(&vsrOutDesc, NULL, &d3d_vsr_output_texture));
        #endif

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
        
        // If VSR is enabled, do we record the VSR output?
        // User request: "апскелинг результата... дополнительного апскейла результата"
        // Usually recording is done on the native internal resolution to avoid large files, 
        // but if VSR is part of the "look", maybe record it?
        // For now, let's keep recording at inference resolution as implemented before 
        // (logic below uses rec_w/rec_h which are based on MODEL_SIZE).
        // Modifying recorder to support VSR resolution requires resizing buffers.
        // Let's stick to inference resolution for recording to keep it performant.

        Recorder recorder(rec_w, rec_h, 24);
        
        void* d_record_buffer = nullptr;
        size_t record_buffer_size = rec_w * rec_h * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_record_buffer, record_buffer_size));
        
        // Recording State
        auto next_record_time = std::chrono::steady_clock::now();
        bool was_recording = false;
        double record_interval = 1.0 / 24.0; 

        // Debug Buffer (RGB 512x512)
        unsigned char* d_debug_img = nullptr;
        size_t debug_size = MODEL_SIZE * MODEL_SIZE * 3 * sizeof(unsigned char);
        CUDA_CHECK(cudaMalloc(&d_debug_img, debug_size));
        bool save_requested = false;

        bool is_overlay_mode = false;
        bool f9_pressed_last = false; // Debounce F9
        bool f10_pressed_last = false; // Debounce F10
        bool f11_pressed_last = false; // Debounce F11
        bool vsr_enabled_runtime = false; // Start with VSR DISABLED

        auto ResizeWindow = [&](int new_w, int new_h) {
            if (!is_overlay_mode) {
                RECT rect = { 0, 0, new_w, new_h };
                AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
                SetWindowPos(g_hwnd, NULL, 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED);
            }
        };

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
                // Restore correct size based on VSR state
                if (vsr_enabled_runtime) {
                    ResizeWindow(display_width_scaled, display_height_scaled);
                } else {
                    ResizeWindow(base_width, MODEL_SIZE);
                }
                std::cout << "[Overlay] Disabled." << std::endl;
            }
        };

        std::cout << "Starting Loop... \nPress 'S' to save debugging image.\nPress 'F9' to toggle Overlay Mode.\nPress 'F10' to toggle Recording.\nPress 'F11' to toggle VSR (RTX Video Super Resolution)." << std::endl;

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

                // F11 Button Handling (VSR Toggle)
                bool f11_down = (GetAsyncKeyState(VK_F11) & 0x8000) != 0;

                if (f11_down && !f11_pressed_last) {
                    #if ENABLE_VSR
                    vsr_enabled_runtime = !vsr_enabled_runtime;
                    if (vsr_enabled_runtime) {
                        ResizeWindow(display_width_scaled, display_height_scaled);
                        std::cout << "[VSR] ENABLED: Window resized to " << display_width_scaled << "x" << display_height_scaled << std::endl;
                    } else {
                        ResizeWindow(base_width, MODEL_SIZE);
                        std::cout << "[VSR] DISABLED: Window resized to " << base_width << "x" << MODEL_SIZE << std::endl;
                    }
                    #else
                    std::cout << "[VSR] Not available (ENABLE_VSR=0 in config.h)" << std::endl;
                    #endif
                }
                f11_pressed_last = f11_down;

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
                    
                    // Preprocessing Logic: Center Crop Square + Downscale
                    // 1. Determine shortest side
                    int crop_size = min(WIDTH, HEIGHT);
                    // 2. Center the crop
                    int crop_off_x = (WIDTH - crop_size) / 2;
                    int crop_off_y = (HEIGHT - crop_size) / 2;

                    // Launch Preprocess (Computes FP16 Tensor)
                    // Maps the [crop_size x crop_size] region at (crop_off_x, crop_off_y) to the 512x512 Network Input
                    launch_preprocess_kernel(texObj, d_input, crop_size, crop_size, crop_off_x, crop_off_y, stream);
                    
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
                    bool is_rec = recorder.IsRecording();
                    if (is_rec && !was_recording) {
                        next_record_time = std::chrono::steady_clock::now();
                    }
                    was_recording = is_rec;

                    if (is_rec) {
                        auto now_steady = std::chrono::steady_clock::now();
                        
                        if (now_steady >= next_record_time) {
                            // Advance target time by interval (perfect pacing)
                            next_record_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                std::chrono::duration<double>(record_interval)
                            );
                            
                            // Prevent falling too far behind (e.g. stalled for > 1 second)
                            if (now_steady > next_record_time + std::chrono::seconds(1)) {
                                next_record_time = now_steady;
                            }

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
                    }
                    
                    // 5. Postprocess (Tensor -> Buffer)
                    
                    // Logic switch:
                    // If VSR Enabled & Supported & Runtime ON:
                    //    -> Postprocess to Base Resolution (MODEL_SIZE) to d3d_output_texture 
                    //    -> VSR Process to d3d_vsr_output_texture (High Res)
                    //    -> Present High Res
                    // Else:
                    //    -> Postprocess to High Res directly (Bilinear) to d3d_vsr_output_texture (Reuse this texture as target!)
                    //    -> Present High Res
                    
                    bool use_vsr = false;
                    
                    #if ENABLE_VSR
                    if (vsr && vsr->IsSupported() && vsr_enabled_runtime) {
                        use_vsr = true;
                    }
                    #endif
                    
                    ID3D11Texture2D* pTargetTexture = d3d_output_texture.Get(); // Default target if VSR off (base res)
                    // VSR OFF -> use base resolution texture (d3d_output_texture)
                    // VSR ON  -> use VSR upscaled texture (d3d_vsr_output_texture)
                    
                    cudaGraphicsResource* cuda_target_tex = cuda_tex_out; // Default: Output Texture (Base Res)
                    ID3D11Texture2D* pFinalForPresent = d3d_output_texture.Get();

                    #if ENABLE_VSR
                    if (use_vsr && d3d_vsr_output_texture) {
                        // VSR will upscale, final output is the VSR texture
                        pFinalForPresent = d3d_vsr_output_texture.Get();
                    }
                    #endif

                    cudaArray_t arr_out = map_d3d11_resource(cuda_target_tex);
                    
                    cudaResourceDesc surfResDesc = {};
                    surfResDesc.resType = cudaResourceTypeArray;
                    surfResDesc.res.array.array = arr_out;
                    
                    cudaSurfaceObject_t surfObj = 0;
                    CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &surfResDesc));
                    
                    // Always render to base resolution texture
                    // VSR will upscale it if enabled
                    int processed_off_x = 0;
                    #if SPLIT_SCREEN
                    processed_off_x = MODEL_SIZE;
                    // Launch postprocess for original (left side)
                    launch_postprocess_kernel(d_input, surfObj, MODEL_SIZE, MODEL_SIZE, 0, 0, stream);
                    #endif
                    // Launch postprocess for processed (right side or full)
                    launch_postprocess_kernel(d_output, surfObj, MODEL_SIZE, MODEL_SIZE, processed_off_x, 0, stream);
                    
                    CUDA_CHECK(cudaStreamSynchronize(stream)); 
                    CUDA_CHECK(cudaDestroySurfaceObject(surfObj));
                    unmap_d3d11_resource(cuda_target_tex);

                    // 6. VSR Pass (if enabled)
                    // Logic check: use_vsr is runtime toggle, vsr_enabled is compile time flag (represented by pointer existence)
                    if (use_vsr && vsr) {
                        static int vsr_log_counter = 0;
                        if (vsr->Process(d3d_output_texture.Get(), d3d_vsr_output_texture.Get())) {
                            pFinalForPresent = d3d_vsr_output_texture.Get();
                            // if (vsr_log_counter++ % 60 == 0) std::cout << "[Main] VSR Applied to frame." << std::endl;
                        } else {
                            if (vsr_log_counter++ % 60 == 0) std::cerr << "[Main] VSR Process Failed!" << std::endl;
                        }
                    }
                    // When VSR is disabled, we just use base resolution output (no bilinear upscale)

                    // 7. Present (Copy Final Texture -> Backbuffer Center)
                    ComPtr<ID3D11Texture2D> back_buffer;
                    DX_CHECK(swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), &back_buffer));
                    
                    ComPtr<ID3D11RenderTargetView> rtv;
                    DX_CHECK(device->CreateRenderTargetView(back_buffer.Get(), NULL, &rtv));
                    
                    // Clear Background (Black)
                    float black[] = {0.0f, 0.0f, 0.0f, 1.0f};
                    context->ClearRenderTargetView(rtv.Get(), black);
                    
                    // Recalculate copy params based on final texture size
                    D3D11_TEXTURE2D_DESC finalDesc;
                    pFinalForPresent->GetDesc(&finalDesc);
                    
                    int tex_w = finalDesc.Width;
                    int tex_h = finalDesc.Height;
                    
                    int tgt_x = (win_w - tex_w) / 2;
                    int tgt_y = (win_h - tex_h) / 2;
                    
                    int src_x = 0; int src_y = 0;
                    int dst_x = tgt_x; int dst_y = tgt_y;
                    int copy_w = tex_w; 
                    int copy_h = tex_h;

                    if (tgt_x < 0) { src_x = -tgt_x; dst_x = 0; copy_w = win_w; }
                    if (tgt_y < 0) { src_y = -tgt_y; dst_y = 0; copy_h = win_h; }

                    if (copy_w > 0 && copy_h > 0) {
                        D3D11_BOX srcBox;
                        srcBox.left = src_x;
                        srcBox.right = src_x + copy_w;
                        srcBox.top = src_y;
                        srcBox.bottom = src_y + copy_h;
                        srcBox.front = 0;
                        srcBox.back = 1;

                        context->CopySubresourceRegion(back_buffer.Get(), 0, dst_x, dst_y, 0, pFinalForPresent, 0, &srcBox);
                    }
                    
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

