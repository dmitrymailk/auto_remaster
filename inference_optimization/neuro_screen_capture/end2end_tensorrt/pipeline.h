#pragma once

#include "utils.h"
#include <NvInfer.h>
#include <vector>
#include <memory>
#include <map>

// Logger for TensorRT
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

class TensorRTPipeline {
public:
    TensorRTPipeline();
    ~TensorRTPipeline();

    void LoadEngines(const std::string& enc_path, const std::string& dec_path);
    
    // Run VAE: Image -> Hidden Latents -> [UNet] -> Image
    void Inference(cudaStream_t stream, void* d_input_image, void* d_output_image, bool dump_debug = false);

    // Optional: Load UNet
    void LoadUNet(const std::string& unet_path);

private:
    static constexpr int CONTEXT_POOL_SIZE = 3;
    
    struct Model {
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> contexts[CONTEXT_POOL_SIZE];
        int current_context_idx = 0;
        
        nvinfer1::IExecutionContext* GetContext() {
            auto* ctx = contexts[current_context_idx].get();
            current_context_idx = (current_context_idx + 1) % CONTEXT_POOL_SIZE;
            return ctx;
        }
    };

    void LoadSingleEngine(const std::string& path, Model& model);

    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;

    Model encoder_;
    Model decoder_;
    Model unet_;

    // Internal Buffers
    void* d_latents_ = nullptr;       // Current sample (ping buffer)
    void* d_latents_2_ = nullptr;     // Output sample (pong buffer)
    void* d_z_source_ = nullptr;      // Encoded latents (constant reference)
    void* d_unet_input_ = nullptr;    // Concatenated [sample, z_source] (8 channels)
    void* d_timestep_ = nullptr;      // For UNet timestep input
    size_t latents_size_ = 0;
    size_t latents_elements_ = 0;
};
