// Include these FIRST to avoid conflicts and ensure visibility
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include "capture_wgc.h"

#include <winrt/Windows.Graphics.DirectX.h> 
#include <iostream>

#pragma comment(lib, "windowsapp")

// Manually define a custom interface to avoid conflicts
// This bypasses any SDK versioning issues
extern "C" const IID IID_IMyDirect3DDxgiInterfaceAccess = { 0xA9B3D012, 0x3DF2, 0x4EE3, { 0xB8, 0xD1, 0x86, 0x95, 0xF4, 0x57, 0xD3, 0xC1 } };

MIDL_INTERFACE("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1")
IMyDirect3DDxgiInterfaceAccess : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetInterface( 
        REFIID iid,
        void **p) = 0;
};

// Use specific namespaces, avoid 'using namespace' to prevent collisions
namespace winrt_cap = winrt::Windows::Graphics::Capture;
namespace winrt_d3d = winrt::Windows::Graphics::DirectX::Direct3D11;
namespace winrt_dx = winrt::Windows::Graphics::DirectX;

// Helper to get IGraphicsCaptureItemInterop
inline auto CreateCaptureItemForWindow(HWND hwnd) {
    auto interop_factory = winrt::get_activation_factory<winrt_cap::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    winrt_cap::GraphicsCaptureItem item = { nullptr };
    winrt::check_hresult(interop_factory->CreateForWindow(hwnd, winrt::guid_of<winrt_cap::GraphicsCaptureItem>(), winrt::put_abi(item)));
    return item;
}

inline winrt_d3d::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice* dxgi_device) {
    winrt::com_ptr<IInspectable> d3d_device;
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi_device, d3d_device.put()));
    return d3d_device.as<winrt_d3d::IDirect3DDevice>();
}

WindowCapture::WindowCapture(ID3D11Device* device, ID3D11DeviceContext* context) 
    : device_(device), context_(context) {
    
    // Create WinRT D3D Device wrapper
    ComPtr<IDXGIDevice> dxgi_device;
    DX_CHECK(device_.As(&dxgi_device));
    winrt_device_ = CreateDirect3DDevice(dxgi_device.Get());
}

WindowCapture::~WindowCapture() {
    Stop();
}

void WindowCapture::Start(HWND hwnd) {
    if (is_capturing_) Stop();

    try {
        item_ = CreateCaptureItemForWindow(hwnd);
        auto size = item_.Size();
        width_ = size.Width;
        height_ = size.Height;
        
        // Create Frame Pool
        frame_pool_ = winrt_cap::Direct3D11CaptureFramePool::Create(
            winrt_device_,
            winrt_dx::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            size);

        // Subscribe to FrameArrived
        frame_pool_.FrameArrived({ this, &WindowCapture::OnFrameArrived });

        // Create Session
        session_ = frame_pool_.CreateCaptureSession(item_);
        session_.IsCursorCaptureEnabled(false); // Optional: Disable cursor?
        session_.StartCapture();
        
        is_capturing_ = true;
        std::cout << "[WindowCapture] Started capturing window: " << width_ << "x" << height_ << std::endl;

    } catch (winrt::hresult_error const& ex) {
        std::cerr << "[WindowCapture] Failed to start capture: " << winrt::to_string(ex.message()) << std::endl;
        throw;
    }
}

void WindowCapture::Stop() {
    if (!is_capturing_) return;

    is_capturing_ = false;
    
    if (session_) {
        session_.Close();
        session_ = nullptr;
    }
    
    if (frame_pool_) {
        frame_pool_.Close();
        frame_pool_ = nullptr;
    }
    
    item_ = nullptr;
    
    std::cout << "[WindowCapture] Stopped." << std::endl;
}

void WindowCapture::OnFrameArrived(winrt_cap::Direct3D11CaptureFramePool const&, winrt::Windows::Foundation::IInspectable const&) {
    if (!is_capturing_) return;
    // Notify only
}

bool WindowCapture::AcquireFrame(ComPtr<ID3D11Texture2D>& texture, UINT) {
    if (!is_capturing_) return false;

    // Release previous frame if any
    ReleaseFrame();

    // Try to get the latest frame
    auto frame = frame_pool_.TryGetNextFrame();
    if (!frame) return false;

    // Keep the frame object alive!
    current_frame_object_ = frame;

    // Get the texture from the frame
    auto surface = frame.Surface();
    auto surface_interop = surface.as<IMyDirect3DDxgiInterfaceAccess>();
    
    DX_CHECK(surface_interop->GetInterface(IID_PPV_ARGS(&texture)));

    // Handle Resize (if window size changed)
    auto contentSize = frame.ContentSize();
    if (static_cast<UINT>(contentSize.Width) != width_ || static_cast<UINT>(contentSize.Height) != height_) {
        width_ = static_cast<UINT>(contentSize.Width);
        height_ = static_cast<UINT>(contentSize.Height);
        
        // We must recreate the pool for new size
        frame_pool_.Recreate(
            winrt_device_,
            winrt_dx::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            contentSize);
    }

    return true;
}

void WindowCapture::ReleaseFrame() {
    if (current_frame_object_) {
        current_frame_object_.Close();
        current_frame_object_ = nullptr;
    }
}
