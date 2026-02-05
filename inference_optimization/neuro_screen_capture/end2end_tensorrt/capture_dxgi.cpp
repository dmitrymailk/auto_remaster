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
    ComPtr<IDXGIOutput> idxgi_output;
    DX_CHECK(idxgi_adapter->EnumOutputs(0, &idxgi_output));

    ComPtr<IDXGIOutput1> idxgi_output1;
    DX_CHECK(idxgi_output.As(&idxgi_output1));

    // Create Duplication
    // This might fail if some other app is capturing or if in exclusive full screen
    DX_CHECK(idxgi_output1->DuplicateOutput(device_.Get(), &duplication_));

    // Get Desc
    DXGI_OUTDUPL_DESC desc;
    duplication_->GetDesc(&desc);
    width_ = desc.ModeDesc.Width;
    height_ = desc.ModeDesc.Height;
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
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            std::cerr << "[ScreenCapture] DXGI Access Lost. Re-initializing..." << std::endl;
            duplication_.Reset();
            Sleep(100); // Wait for switch
            Initialize(); 
            return false;
        }
        DX_CHECK(hr);
    }
    
    frame_acquired_ = true;
    DX_CHECK(desktop_resource.As(&captured_texture));
    return true;
}

void ScreenCapture::ReleaseFrame() {
    if (frame_acquired_ && duplication_) {
        // Ignore errors on release
        duplication_->ReleaseFrame();
        frame_acquired_ = false;
    }
}
