#pragma once

#include "utils.h"

class ScreenCapture {
public:
    ScreenCapture(ID3D11Device* device, ID3D11DeviceContext* context);
    ~ScreenCapture();

    void Initialize();
    // Returns true if frame acquired, false if timeout or other recoverable issue
    bool AcquireFrame(ComPtr<ID3D11Texture2D>& captured_texture, UINT timeout_ms = 100);
    void ReleaseFrame();

    // Get latest frame dimensions
    UINT GetWidth() const { return width_; }
    UINT GetHeight() const { return height_; }

private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    
    ComPtr<IDXGIOutputDuplication> duplication_;
    bool frame_acquired_ = false;

    UINT width_ = 0;
    UINT height_ = 0;
};
