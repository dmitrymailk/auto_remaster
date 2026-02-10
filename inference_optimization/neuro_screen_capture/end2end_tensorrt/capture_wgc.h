#pragma once

#include <d3d11.h>
#include <wrl/client.h>

// Include Windows Foundation first to establish winrt namespace
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>

#include <atomic>
#include <mutex>

#include "utils.h"
#include "capture_interface.h"

using namespace Microsoft::WRL;

class WindowCapture : public ICapture {
public:
    WindowCapture(ID3D11Device* device, ID3D11DeviceContext* context);
    ~WindowCapture();

    // Start capturing a specific window
    void Start(HWND hwnd);
    void Stop();

    // Returns true if a new frame is available. 
    // Captured texture is returned in 'texture'.
    bool AcquireFrame(ComPtr<ID3D11Texture2D>& texture, UINT timeout_ms = 100) override;
    
    // Release the frame after processing
    void ReleaseFrame() override;

    UINT GetWidth() const override { return width_; }
    UINT GetHeight() const override { return height_; }
    bool IsCapturing() const { return is_capturing_; }

private:
    void OnFrameArrived(winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const& sender, winrt::Windows::Foundation::IInspectable const& args);

    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice winrt_device_{ nullptr };

    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item_{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool_{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession session_{ nullptr };

    std::atomic<bool> is_capturing_{ false };
    std::atomic<bool> frame_arrived_{ false };
    
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame current_frame_object_{ nullptr };
    ComPtr<ID3D11Texture2D> current_frame_;
    std::mutex frame_mutex_;

    UINT width_ = 0;
    UINT height_ = 0;
};
