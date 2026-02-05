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
    
    // Run VAE: Image (512x512) -> Hidden Latents -> Image (512x512)
    void Inference(cudaStream_t stream, void* d_input_image, void* d_output_image);

private:
    struct Model {
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        // Map tensor name -> index
        std::map<std::string, int> bindings;
    };

    void LoadSingleEngine(const std::string& path, Model& model);
    void SetupBindings(Model& model);

    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;

    Model encoder_;
    Model decoder_;

    // Internal Buffer for Latents
    // VAE Tiny: 512x512 image -> 64x64 latent? Or 32x32? 
    // Python script said: 128 channels, 32x32 for standard VAE?
    // "Tiny AutoEncoder" might be different.
    // We will query engine for size.
    void* d_latents_ = nullptr;
    size_t latents_size_ = 0;
};
