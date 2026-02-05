#include "pipeline.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

// Auto-delete for raw TRT pointers
struct TRTDeleter {
    template <typename T> void operator()(T* obj) const { if (obj) delete obj; }
};

TensorRTPipeline::TensorRTPipeline() {
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
}

TensorRTPipeline::~TensorRTPipeline() {
    if (d_latents_) cudaFree(d_latents_);
    if (d_latents_2_) cudaFree(d_latents_2_);
    if (d_timestep_) cudaFree(d_timestep_);
}

void TensorRTPipeline::LoadEngines(const std::string& enc_path, const std::string& dec_path) {
    std::cout << "Loading Encoder: " << enc_path << std::endl;
    LoadSingleEngine(enc_path, encoder_);
    
    std::cout << "Loading Decoder: " << dec_path << std::endl;
    LoadSingleEngine(dec_path, decoder_);
    
    // Allocate Intermediate Latents Buffer
    // Find output size of Encoder
    // Assuming 1 output
    int out_idx = -1;
    for (int i = 0; i < encoder_.engine->getNbIOTensors(); ++i) {
        const char* name = encoder_.engine->getIOTensorName(i);
        if (encoder_.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            out_idx = i;
            break;
        }
    }
    CHECK(out_idx != -1);
    
    auto dims = encoder_.engine->getTensorShape(encoder_.engine->getIOTensorName(out_idx));
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        // If dynamic (-1), we need to set it. But assume static for now or fixed batch 1.
        if (dims.d[i] < 0) vol *= 1; // Assume 1 for batch
        else vol *= dims.d[i];
    }
    
    auto type = encoder_.engine->getTensorDataType(encoder_.engine->getIOTensorName(out_idx));
    size_t elem_size = (type == nvinfer1::DataType::kFLOAT) ? 4 : 2; 

    latents_size_ = vol * elem_size;
    std::cout << "Allocating Latents Buffer: " << vol << " elements (" << latents_size_ << " bytes)" << std::endl;
    CUDA_CHECK(cudaMalloc(&d_latents_, latents_size_));

#if ENABLE_UNET
    // Allocate secondary buffer for ping-pong
    CUDA_CHECK(cudaMalloc(&d_latents_2_, latents_size_));

    // Allocate Timestep (1 element, FP16 usually, check if UNet needs FP32?)
    // Python script says FP16 if half. Assuming FP16 for now (2 bytes).
    CUDA_CHECK(cudaMalloc(&d_timestep_, 2));
    uint16_t h_ts = 0x3c00; // 1.0 in FP16 (approx)
    // Or if FP32:
    // float h_ts_f = 1.0f;
    // Let's assume FP16 based on VAE.
    // 0x3c00 is 1.0 in IEEE 754 half-precision.
    CUDA_CHECK(cudaMemcpy(d_timestep_, &h_ts, 2, cudaMemcpyHostToDevice));
#endif

    // Debug: Print Bindings
    // ... (rest of debug print omitted for brevity if needed)
}

void TensorRTPipeline::LoadUNet(const std::string& path) {
#if ENABLE_UNET
    std::cout << "Loading UNet: " << path << std::endl;
    LoadSingleEngine(path, unet_);
#else
    std::cout << "UNet disabled via config.h" << std::endl;
#endif
}

void TensorRTPipeline::LoadSingleEngine(const std::string& path, Model& model) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    CHECK(file.good());
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read engine file");
    }

    model.engine.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    CHECK(model.engine != nullptr);
    model.context.reset(model.engine->createExecutionContext());
    CHECK(model.context != nullptr);
}

void TensorRTPipeline::Inference(cudaStream_t stream, void* d_input_image, void* d_output_image) {
    // 1. Encoder
    // Bind Inputs/Outputs
    // Simple logic: If name contains "image" -> input. If name contains "latent" -> output.
    // We iterate generic helper style.

    // ENCODER
    for (int i = 0; i < encoder_.engine->getNbIOTensors(); ++i) {
        const char* name = encoder_.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = encoder_.engine->getTensorIOMode(name);
        
        void* ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ptr = d_input_image; // Assuming provided input is ready
        } else {
            ptr = d_latents_;
        }
        encoder_.context->setTensorAddress(name, ptr);
    }
    
    if (!encoder_.context->enqueueV3(stream)) {
        std::cerr << "Encoder Inference Failed!" << std::endl;
    }

    void* current_latents = d_latents_;

#if ENABLE_UNET
    if (unet_.context) {
        // 2. UNet Loop
        int steps = UNET_STEPS;
        void* in_buf = d_latents_;
        void* out_buf = d_latents_2_;

        for (int s = 0; s < steps; ++s) {
            for (int i = 0; i < unet_.engine->getNbIOTensors(); ++i) {
                const char* name = unet_.engine->getIOTensorName(i);
                nvinfer1::TensorIOMode mode = unet_.engine->getTensorIOMode(name);
                
                void* ptr = nullptr;
                // UNet Bindings: "sample" (In), "timestep" (In), "out_sample" (Out)?
                // Or "sample" (Out) if in-place?
                // Python script: set_input_shape("sample", ...), set_tensor_address("sample", in)
                // set_tensor_address("out_sample", out)
                
                std::string sname = name;
                if (sname.find("timestep") != std::string::npos) {
                    ptr = d_timestep_;
                } else if (mode == nvinfer1::TensorIOMode::kINPUT) {
                    ptr = in_buf;
                } else {
                    ptr = out_buf;
                }
                unet_.context->setTensorAddress(name, ptr);
            }
            if (!unet_.context->enqueueV3(stream)) {
                 std::cerr << "UNet Inference Failed!" << std::endl;
            }
            
            // Swap
            std::swap(in_buf, out_buf);
        }
        current_latents = in_buf; // The result is in the last in_buf (which was the last out_buf)
    }
#endif

    // DECODER
    for (int i = 0; i < decoder_.engine->getNbIOTensors(); ++i) {
        const char* name = decoder_.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = decoder_.engine->getTensorIOMode(name);
        
        void* ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ptr = current_latents;
        } else {
            ptr = d_output_image;
        }
        decoder_.context->setTensorAddress(name, ptr);
    }
    
    if (!decoder_.context->enqueueV3(stream)) {
       std::cerr << "Decoder Inference Failed!" << std::endl;
    }
}
