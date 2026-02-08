#include "vsr_upscaler.h"
#include <iostream>
#include <filesystem>

// Link against the lib
// #pragma comment(lib, "nvsdk_ngx_s.lib") // Managed by CMake

// We need to include helper headers for macros? 
// nvsdk_ngx_helpers.h provides NVSDK_NGX_D3D11_Init and other helper macros/functions.
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_defs_vsr.h"
#include "nvsdk_ngx_helpers.h"

VSRUpscaler::VSRUpscaler(ID3D11Device* device, ID3D11DeviceContext* context)
    : m_pDevice(device), m_pContext(context) {
}

VSRUpscaler::~VSRUpscaler() {
    if (m_pNgxVSRHandle) {
        NVSDK_NGX_D3D11_ReleaseFeature(m_pNgxVSRHandle);
        m_pNgxVSRHandle = nullptr;
    }
    
    // Shutdown NGX
    // Note: If you have multiple features or instances, you should likely refactor Shutdown to a global manager.
    // For this single-use case, we shut down here.
    NVSDK_NGX_D3D11_Shutdown1(m_pDevice.Get());
}

bool VSRUpscaler::Initialize(int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    if (m_initialized) return true; // Already initialized

    // Use 0 to allow driver to use Process Name detection (for vlc.exe spoofing)
    unsigned long long appId = 0; 
    
    // Absolute path for logs (using backslashes)
    std::wstring logDir = L"C:\\Users\\legion2025\\.gemini\\antigravity\\brain\\85de904a-a6fd-4d90-bab2-3a14058585b9\\";
    
    std::wcout << L"[VSR] Init NGX LogDir: " << logDir << std::endl;
    // 1. Initialize NGX
    NVSDK_NGX_Result result = NVSDK_NGX_D3D11_Init(appId, logDir.c_str(), m_pDevice.Get());
    
    std::cout << "[VSR] NVSDK_NGX_D3D11_Init Result: " << std::hex << result << std::dec << std::endl;

    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[VSR] Failed to initialize NGX. Result: " << std::hex << result << std::dec << std::endl;
        return false;
    }

    // 2. Get Parameters Interface
    result = NVSDK_NGX_D3D11_GetCapabilityParameters(&m_pNgxParameters);
    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[VSR] Failed to get capability parameters." << std::endl;
        return false;
    }

    int isVSRAvailable = 0;
    int isDLSSAvailable = 0;
    
    // Check DLSS (SuperSampling) to verify NGX is working generally
    m_pNgxParameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &isDLSSAvailable);
    std::cout << "[VSR] SuperSampling (DLSS) Available: " << isDLSSAvailable << std::endl;

    // Check VSR (using correct VSR macro)
    m_pNgxParameters->Get(NVSDK_NGX_Parameter_VSR_Available, &isVSRAvailable);
    std::cout << "[VSR] VSR Available: " << isVSRAvailable << std::endl;

    // Check Driver Update Needed
    int needsUpdate = 0;
    m_pNgxParameters->Get(NVSDK_NGX_Parameter_VSR_NeedsUpdatedDriver, &needsUpdate);
    std::cout << "[VSR] Needs Driver Update: " << needsUpdate << std::endl;
    
    if (needsUpdate) {
        unsigned int minMajor = 0, minMinor = 0;
        m_pNgxParameters->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMajor, &minMajor);
        m_pNgxParameters->Get(NVSDK_NGX_Parameter_VSR_MinDriverVersionMinor, &minMinor);
        std::cout << "[VSR] Min Driver Version: " << minMajor << "." << minMinor << std::endl;
    }

    if (!isVSRAvailable) {
        std::cerr << "[VSR] WARNING: VSR reported as unavailable." << std::endl;
        std::cerr << "[VSR] TIP: Ensure 'RTX Video Enhancement' / 'Super Resolution' is ENABLED in NVIDIA Control Panel." << std::endl;
        std::cerr << "[VSR] TIP: Ensure the laptop is plugged in (Battery mode might disable VSR)." << std::endl;
        std::cerr << "[VSR] Attempting creation anyway for debug..." << std::endl;
        // return false; // FORCE CONTINUE
    }
    
    m_isSupported = true;

    // 4. Create Feature
    // We need create params. 
    // VSR creation handles parameters via m_pNgxParameters.

    // But VSR is not DLSS. Let's see if we can just pass common params.
    // The helper CreateFeature takes NVSDK_NGX_Parameter.
    // We should set parameters on m_pNgxParameters? No, creation handles it.
    // Wait, NVSDK_NGX_D3D11_CreateFeature takes 'InParameters' and returns handle.
    // We need to Setup the parameters before calling Create.
    
    // Common Setup
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Width, inputWidth);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Height, inputHeight);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutWidth, outputWidth);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutHeight, outputHeight);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_VSR_QualityLevel, NVSDK_NGX_VSR_Quality_High); // 1-4. 4 is Ultra? defined as 4 in header.

    // Hint: VSR expects sRGB/Gamma content usually, or at least we should specify.
    // But for "RGB only" use case, we just pass the textures.

    std::cout << "[VSR] Creating Feature... Params: " << inputWidth << "x" << inputHeight << " -> " << outputWidth << "x" << outputHeight << std::endl;
    if (!m_pNgxParameters) {
        std::cerr << "[VSR] Error: m_pNgxParameters is NULL!" << std::endl;
        return false;
    }

    result = NVSDK_NGX_D3D11_CreateFeature(
        m_pContext.Get(), // Device Context
        NVSDK_NGX_Feature_VSR, // Correct VSR Feature ID (16)
        m_pNgxParameters, // Parameters
        &m_pNgxVSRHandle  // Output Handle
    );

    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[VSR] CreateFeature failed. Result: " << std::hex << result << std::dec << std::endl;
        return false;
    }

    std::cout << "[VSR] Initialized Successfully. " << inputWidth << "x" << inputHeight << " -> " << outputWidth << "x" << outputHeight << std::endl;
    m_initialized = true;
    return true;
}

bool VSRUpscaler::Process(ID3D11Texture2D* input, ID3D11Texture2D* output) {
    if (!m_initialized || !m_pNgxVSRHandle) return false;

    // Retrieve resources
    ID3D11Resource* pIn = input;
    ID3D11Resource* pOut = output;

    // Get Dimensions for Validation (and to set rects)
    D3D11_TEXTURE2D_DESC inDesc, outDesc;
    input->GetDesc(&inDesc);
    output->GetDesc(&outDesc);

    // Bind resources to parameters
    // VSR uses Input1, not Color
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Input1, pIn);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Output, pOut);
    
    // Set Rects (Full Frame)
    // Input Rect
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Rect_X, 0);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Rect_Y, 0);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Rect_W, inDesc.Width);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_Rect_H, inDesc.Height);

    // Output Rect
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutRect_X, 0);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutRect_Y, 0);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutRect_W, outDesc.Width);
    m_pNgxParameters->Set(NVSDK_NGX_Parameter_OutRect_H, outDesc.Height);

    // Execute
    NVSDK_NGX_Result result = NVSDK_NGX_D3D11_EvaluateFeature(
        m_pContext.Get(),
        m_pNgxVSRHandle,
        m_pNgxParameters,
        NULL // No Callback
    );

    if (NVSDK_NGX_FAILED(result)) {
        std::cerr << "[VSR] EvaluateFeature failed. Result: " << std::hex << result << std::dec << std::endl;
        std::cerr << "[VSR] Debug: Input " << inDesc.Width << "x" << inDesc.Height << " Format: " << inDesc.Format << std::endl;
        std::cerr << "[VSR] Debug: Output " << outDesc.Width << "x" << outDesc.Height << " Format: " << outDesc.Format << std::endl;
        return false;
    }
    
    return true;
}
