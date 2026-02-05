#include "capture_dxgi.h"

ScreenCapture::ScreenCapture(ID3D11Device* device, ID3D11DeviceContext* context)
    : device_(device), context_(context) {}

ScreenCapture::~ScreenCapture() {
    ReleaseFrame();
}

void ScreenCapture::Initialize() {
    // Get DXGI Device
    ComPtr<IDXGIDevice> dxgi_device;
    DX_CHECK(device_.As(&dxgi_device));

    // Get Adapter
    ComPtr<IDXGIAdapter> idxgi_adapter;
    DX_CHECK(dxgi_device->GetAdapter(&idxgi_adapter));

    // Get Output 0 (Primary Monitor)
    // Note: In multi-monitor setups, we might want to let user choose.
    // For now, we hardcode to output 0.
    ComPtr<IDXGIOutput> idxgi_output;
    DX_CHECK(idxgi_adapter->EnumOutputs(0, &idxgi_output));

    ComPtr<IDXGIOutput1> idxgi_output1;
    DX_CHECK(idxgi_output.As(&idxgi_output1));

    // Create Duplication
    DX_CHECK(idxgi_output1->DuplicateOutput(device_.Get(), &duplication_));
}

bool ScreenCapture::AcquireFrame(ComPtr<ID3D11Texture2D>& captured_texture, UINT timeout_ms) {
    if (frame_acquired_) {
        ReleaseFrame();
    }

    DXGI_OUTDUPL_FRAME_INFO frame_info;
    ComPtr<IDXGIResource> desktop_resource;

    HRESULT hr = duplication_->AcquireNextFrame(timeout_ms, &frame_info, &desktop_resource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false;
    }
    
    if (FAILED(hr)) {
        // Handle device lost or other errors? For now just throw/log via macro
        // But invalid call can happen if we didn't release previous frame?
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            // Re-initialization needed usually, throw for now to crash and restart or handle in main
             std::cerr << "DXGI Access Lost. Re-initialization required." << std::endl;
        }
        DX_CHECK(hr);
    }
    
    // Only process if we actually have an image update? 
    // Usually LastPresentTime > 0 implies a new frame.
    // But for simplicity, if Acquire succeeds, we return true.
    
    frame_acquired_ = true;
    DX_CHECK(desktop_resource.As(&captured_texture));
    return true;
}

void ScreenCapture::ReleaseFrame() {
    if (frame_acquired_ && duplication_) {
        DX_CHECK(duplication_->ReleaseFrame());
        frame_acquired_ = false;
    }
}


