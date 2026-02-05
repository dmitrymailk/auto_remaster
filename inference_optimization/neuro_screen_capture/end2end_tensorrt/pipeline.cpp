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
    
    // Float16 or Float32?
    // Defaulting to Float32 size safely, or check type.
    // Python script used FP16 usually.
    // If engine is FP16, we need FP16 storage.
    // Let's alloc enough for Float32 to be safe, or query type.
    auto type = encoder_.engine->getTensorDataType(encoder_.engine->getIOTensorName(out_idx));
    size_t elem_size = (type == nvinfer1::DataType::kFLOAT) ? 4 : 2; 

    latents_size_ = vol * elem_size;
    std::cout << "Allocating Latents Buffer: " << vol << " elements (" << latents_size_ << " bytes)" << std::endl;
    CUDA_CHECK(cudaMalloc(&d_latents_, latents_size_));

    // Debug: Print Bindings
    std::cout << "--- Encoder Bindings ---" << std::endl;
    for (int i = 0; i < encoder_.engine->getNbIOTensors(); ++i) {
        const char* name = encoder_.engine->getIOTensorName(i);
        auto mode = encoder_.engine->getTensorIOMode(name);
        auto dims = encoder_.engine->getTensorShape(name);
        auto type = encoder_.engine->getTensorDataType(name);
        std::cout << "  " << name << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "In" : "Out") << "] "
                  << "Type: " << (int)type << " Shape: ";
        for(int d=0; d<dims.nbDims; ++d) std::cout << dims.d[d] << " ";
        std::cout << std::endl;
    }
    std::cout << "--- Decoder Bindings ---" << std::endl;
    for (int i = 0; i < decoder_.engine->getNbIOTensors(); ++i) {
        const char* name = decoder_.engine->getIOTensorName(i);
        auto mode = decoder_.engine->getTensorIOMode(name);
        auto dims = decoder_.engine->getTensorShape(name);
        auto type = decoder_.engine->getTensorDataType(name);
        std::cout << "  " << name << " [" << (mode == nvinfer1::TensorIOMode::kINPUT ? "In" : "Out") << "] "
                  << "Type: " << (int)type << " Shape: ";
        for(int d=0; d<dims.nbDims; ++d) std::cout << dims.d[d] << " ";
        std::cout << std::endl;
    }
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

    // DECODER
    for (int i = 0; i < decoder_.engine->getNbIOTensors(); ++i) {
        const char* name = decoder_.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = decoder_.engine->getTensorIOMode(name);
        
        void* ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ptr = d_latents_;
        } else {
            ptr = d_output_image;
        }
        decoder_.context->setTensorAddress(name, ptr);
    }
    
    if (!decoder_.context->enqueueV3(stream)) {
       std::cerr << "Decoder Inference Failed!" << std::endl;
    }
}
