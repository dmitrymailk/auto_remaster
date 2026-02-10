#pragma once

#include "utils.h"
#include "capture_interface.h"

class ScreenCapture : public ICapture {
public:
    ScreenCapture(ID3D11Device* device, ID3D11DeviceContext* context);
    ~ScreenCapture();

    void Initialize();
    // Returns true if frame acquired, false if timeout or other recoverable issue
    bool AcquireFrame(ComPtr<ID3D11Texture2D>& captured_texture, UINT timeout_ms = 100) override;
    void ReleaseFrame() override;

    // Get latest frame dimensions
    UINT GetWidth() const override { return width_; }
    UINT GetHeight() const override { return height_; }

private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    
    ComPtr<IDXGIOutputDuplication> duplication_;
    bool frame_acquired_ = false;

    UINT width_ = 0;
    UINT height_ = 0;
};
