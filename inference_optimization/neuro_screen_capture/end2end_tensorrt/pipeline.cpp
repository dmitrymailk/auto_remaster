#include "pipeline.h"
#include "config.h"
#include "cuda_utils.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

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
    if (d_z_source_) cudaFree(d_z_source_);
    if (d_unet_input_) cudaFree(d_unet_input_);
    if (d_timestep_) cudaFree(d_timestep_);
}

void TensorRTPipeline::LoadEngines(const std::string& enc_path, const std::string& dec_path) {
    std::cout << "Loading Encoder: " << enc_path << std::endl;
    LoadSingleEngine(enc_path, encoder_);
    
    std::cout << "Loading Decoder: " << dec_path << std::endl;
    LoadSingleEngine(dec_path, decoder_);
    
    // Allocate Intermediate Latents Buffer
    // Find output size of Encoder
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
        if (dims.d[i] < 0) vol *= 1; // Assume 1 for batch
        else vol *= dims.d[i];
    }
    
    auto type = encoder_.engine->getTensorDataType(encoder_.engine->getIOTensorName(out_idx));
    size_t elem_size = (type == nvinfer1::DataType::kFLOAT) ? 4 : 2; 

    latents_elements_ = vol;
    latents_size_ = vol * elem_size;
    std::cout << "Allocating Latents Buffer: " << vol << " elements (" << latents_size_ << " bytes)" << std::endl;
    
    CUDA_CHECK(cudaMalloc(&d_latents_, latents_size_));
    CUDA_CHECK(cudaMalloc(&d_z_source_, latents_size_));  // Store encoded latents

#if ENABLE_UNET
    // Allocate secondary buffer for ping-pong
    CUDA_CHECK(cudaMalloc(&d_latents_2_, latents_size_));
    
    // Allocate concatenated input buffer for UNet (8 channels = sample + z_source)
    CUDA_CHECK(cudaMalloc(&d_unet_input_, latents_size_ * 2));  // 2x for 8 channels

    // Allocate Timestep (1 element, FP16 = 2 bytes)
    CUDA_CHECK(cudaMalloc(&d_timestep_, sizeof(uint16_t)));
#endif
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
    
    // Create context pool for stable repeated inference
    for (int i = 0; i < CONTEXT_POOL_SIZE; ++i) {
        model.contexts[i].reset(model.engine->createExecutionContext());
        CHECK(model.contexts[i] != nullptr);
    }
}

void TensorRTPipeline::Inference(cudaStream_t stream, void* d_input_image, void* d_output_image, bool dump_debug) {
    
    // ==== ENCODER ====
    auto* enc_ctx = encoder_.GetContext();
    
    for (int i = 0; i < encoder_.engine->getNbIOTensors(); ++i) {
        const char* name = encoder_.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = encoder_.engine->getTensorIOMode(name);
        
        void* ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ptr = d_input_image;
        } else {
            ptr = d_z_source_;  // Store encoded latents in z_source
        }
        enc_ctx->setTensorAddress(name, ptr);
    }
    
    if (!enc_ctx->enqueueV3(stream)) {
        std::cerr << "Encoder Inference Failed!" << std::endl;
        return;
    }
    
    // Apply scaling factor: z_source = encoder_output * SCALING_FACTOR
    launch_scale_latents(d_z_source_, VAE_SCALING_FACTOR, latents_elements_, stream);

    if (dump_debug) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::cout << "[DEBUG] Dumping C++ Tensors..." << std::endl;
        
        // Dump z_source (Latents after encoding & scaling)
        std::vector<char> h_z(latents_size_);
        CUDA_CHECK(cudaMemcpy(h_z.data(), d_z_source_, latents_size_, cudaMemcpyDeviceToHost));
        std::ofstream f("debug_cpp_z_source.bin", std::ios::binary);
        f.write(h_z.data(), latents_size_);
        f.close();
        std::cout << "  Saved debug_cpp_z_source.bin" << std::endl;
    }


    void* current_latents = d_z_source_;

#if ENABLE_UNET
    if (unet_.contexts[0]) {
        // Copy z_source to sample (starting point)
        CUDA_CHECK(cudaMemcpyAsync(d_latents_, d_z_source_, latents_size_, cudaMemcpyDeviceToDevice, stream));
        
        int steps = UNET_STEPS;
        void* sample_buf = d_latents_;
        void* output_buf = d_latents_2_;
        
        // Compute sigmas... (omitted for brevity, same as before)
        std::vector<float> sigmas(steps + 1);
        if (steps == 1) {
            sigmas[0] = 1.0f;
            sigmas[1] = 0.0f;
        } else {
            for (int i = 0; i < steps; ++i) {
                sigmas[i] = 1.0f - (float)i * (1.0f - 1.0f / (float)steps) / ((float)steps - 1.0f);
            }
            sigmas[steps] = 0.0f;
        }

        // UNet Loop
        for (int s = 0; s < steps; ++s) {
            float sigma = sigmas[s];
            float sigma_next = sigmas[s + 1];
            
            launch_float_to_half(d_timestep_, sigma, stream);
            
            // Concatenate [sample, z_source] -> unet_input (8 channels)
            launch_concat_latents(sample_buf, d_z_source_, d_unet_input_,
                                  1, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, stream);
            
            if (dump_debug && s == 0) {
                 CUDA_CHECK(cudaStreamSynchronize(stream));
                 
                 // Dump UNet Input
                 std::vector<char> h_in(latents_size_ * 2);
                 CUDA_CHECK(cudaMemcpy(h_in.data(), d_unet_input_, latents_size_ * 2, cudaMemcpyDeviceToHost));
                 std::ofstream f1("debug_cpp_unet_input.bin", std::ios::binary);
                 f1.write(h_in.data(), latents_size_ * 2);
                 f1.close();
                 
                 // Dump Timestep
                 unsigned short h_t;
                 CUDA_CHECK(cudaMemcpy(&h_t, d_timestep_, 2, cudaMemcpyDeviceToHost));
                 std::cout << "  Timestep (FP16 bits): " << h_t << std::endl;

                 std::cout << "  Saved debug_cpp_unet_input.bin" << std::endl;
            }

            // Run UNet
            auto* unet_ctx = unet_.GetContext();
            
            for (int i = 0; i < unet_.engine->getNbIOTensors(); ++i) {
                const char* name = unet_.engine->getIOTensorName(i);
                nvinfer1::TensorIOMode mode = unet_.engine->getTensorIOMode(name);
                std::string sname = name;
                
                void* ptr = nullptr;
                if (sname.find("timestep") != std::string::npos) {
                    ptr = d_timestep_;
                } else if (mode == nvinfer1::TensorIOMode::kINPUT) {
                    ptr = d_unet_input_;  // Concatenated input
                } else {
                    ptr = output_buf;
                }
                unet_ctx->setTensorAddress(name, ptr);
            }
            
            if (!unet_ctx->enqueueV3(stream)) {
                std::cerr << "UNet Inference Failed at step " << s << std::endl;
                return;
            }

            if (dump_debug && s == 0) {
                 CUDA_CHECK(cudaStreamSynchronize(stream));
                 
                 // Dump UNet Output
                 std::vector<char> h_out(latents_size_);
                 CUDA_CHECK(cudaMemcpy(h_out.data(), output_buf, latents_size_, cudaMemcpyDeviceToHost));
                 std::ofstream f2("debug_cpp_unet_output.bin", std::ios::binary);
                 f2.write(h_out.data(), latents_size_);
                 f2.close();
                 std::cout << "  Saved debug_cpp_unet_output.bin" << std::endl;
            }
            
            // Scheduler step...
            launch_scheduler_step(sample_buf, output_buf, sigma, sigma_next, latents_elements_, stream);
        }
        
        current_latents = sample_buf;
    }
#endif

    // ==== DECODER ====
    // First divide by scaling factor
    void* decoder_input = current_latents;
    
#if ENABLE_UNET
    // Need to divide by scaling factor before decode
    // Copy to d_latents_2_ and scale
    CUDA_CHECK(cudaMemcpyAsync(d_latents_2_, current_latents, latents_size_, cudaMemcpyDeviceToDevice, stream));
    launch_scale_latents(d_latents_2_, 1.0f / VAE_SCALING_FACTOR, latents_elements_, stream);
    decoder_input = d_latents_2_;
#else
    // Without UNet, z_source already has scaling applied, need to undo it
    launch_scale_latents(d_z_source_, 1.0f / VAE_SCALING_FACTOR, latents_elements_, stream);
    decoder_input = d_z_source_;
#endif

    auto* dec_ctx = decoder_.GetContext();
    
    for (int i = 0; i < decoder_.engine->getNbIOTensors(); ++i) {
        const char* name = decoder_.engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = decoder_.engine->getTensorIOMode(name);
        
        void* ptr = nullptr;
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ptr = decoder_input;
        } else {
            ptr = d_output_image;
        }
        dec_ctx->setTensorAddress(name, ptr);
    }
    
    if (!dec_ctx->enqueueV3(stream)) {
       std::cerr << "Decoder Inference Failed!" << std::endl;
    }
}
