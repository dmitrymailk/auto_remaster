#pragma once
#include <d3d11.h>
#include <wrl/client.h>

using namespace Microsoft::WRL;

class ICapture {
public:
    virtual ~ICapture() = default;
    
    // Returns true if frame acquired
    virtual bool AcquireFrame(ComPtr<ID3D11Texture2D>& captured_texture, UINT timeout_ms = 100) = 0;
    virtual void ReleaseFrame() = 0;
    
    virtual UINT GetWidth() const = 0;
    virtual UINT GetHeight() const = 0;
};
