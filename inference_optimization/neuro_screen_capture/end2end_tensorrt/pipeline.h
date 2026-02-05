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
    void Inference(cudaStream_t stream, void* d_input_image, void* d_output_image);

    // Optional: Load UNet
    void LoadUNet(const std::string& unet_path);

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
    Model unet_;

    // Internal Buffers
    void* d_latents_ = nullptr;
    void* d_latents_2_ = nullptr; // Ping-pong buffer for UNet
    void* d_timestep_ = nullptr; // For UNet timestep input
    size_t latents_size_ = 0;
};
