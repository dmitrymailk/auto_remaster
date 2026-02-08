#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <memory>
#include <string>

// Forward declaration of NGX types to avoid including the massive header here if possible in full
// But for unique_ptr or member variables we might need it or just void* and cast.
// For simplicity, let's include it or just forward declare handles.
// Given strict type checking, including the header is safer.
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_defs_vsr.h"

using Microsoft::WRL::ComPtr;

class VSRUpscaler {
public:
    VSRUpscaler(ID3D11Device* device, ID3D11DeviceContext* context);
    ~VSRUpscaler();

    // Returns true if initialization and feature creation was successful
    bool Initialize(int inputWidth, int inputHeight, int outputWidth, int outputHeight);

    // Processes the input texture and writes to the output texture.
    // Ensure input state is SHADER_RESOURCE and output state is UNORDERED_ACCESS (if using barriers, but D3D11 handles this implicitly often)
    // Actually D3D11 doesn't have explicit barriers like D3D12. 
    // We just need to ensure correct BindFlags.
    bool Process(ID3D11Texture2D* input, ID3D11Texture2D* output);

    // Returns the ideal scratch buffer size if we were managing it manually (NGX manages it usually)
    // But we might need to know if VSR is actually available.
    bool IsSupported() const { return m_isSupported; }

private:
    ComPtr<ID3D11Device> m_pDevice;
    ComPtr<ID3D11DeviceContext> m_pContext;

    NVSDK_NGX_Parameter* m_pNgxParameters = nullptr;
    NVSDK_NGX_Handle* m_pNgxVSRHandle = nullptr;
    
    bool m_isSupported = false;
    bool m_initialized = false;
    
    unsigned long long m_appId = 123456; // Dummy ID
};
