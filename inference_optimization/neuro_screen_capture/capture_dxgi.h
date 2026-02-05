#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include "utils.h"

using Microsoft::WRL::ComPtr;

class ScreenCapture {
public:
    ScreenCapture(ID3D11Device* device, ID3D11DeviceContext* context);
    ~ScreenCapture();

    void Initialize();
    
    // Returns true if a new frame was captured.
    // 'captured_texture' will point to the internal DXGI buffer (do not release).
    bool AcquireFrame(ComPtr<ID3D11Texture2D>& captured_texture, UINT timeout_ms = 0);
    
    void ReleaseFrame();

private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    bool frame_acquired_ = false;
};
