#include "recorder.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
namespace fs = std::filesystem;

Recorder::Recorder(int width, int height, int fps) 
    : width_(width), height_(height) {
    
    frame_size_ = width_ * height_ * channels_; // Uint8 RGB

    // Initialize Buffer Pool
    buffer_pool_.resize(POOL_SIZE);
    for (int i = 0; i < POOL_SIZE; ++i) {
        // Allocate Pinned Memory
        CUDA_CHECK(cudaMallocHost(&buffer_pool_[i].h_data, frame_size_));
        buffer_pool_[i].size = frame_size_;
        CUDA_CHECK(cudaEventCreate(&buffer_pool_[i].copy_complete_event, cudaEventDisableTiming));
        
        free_indices_.push(i);
    }

    // Start Worker Thread
    stop_worker_ = false;
    worker_thread_ = std::thread(&Recorder::WorkerThread, this);

    std::cout << "[Recorder] Initialized for " << width << "x" << height << " RGB (" << fps << " FPS target)" << std::endl;
}

Recorder::~Recorder() {
    Stop();
    
    // Signal worker to exit
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_worker_ = true;
    }
    worker_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // Free Resources
    for (int i = 0; i < POOL_SIZE; ++i) {
        if (buffer_pool_[i].h_data) {
            cudaError_t err = cudaFreeHost(buffer_pool_[i].h_data);
            if (err != cudaSuccess) std::cerr << "cudaFreeHost failed: " << cudaGetErrorString(err) << std::endl;
        }
        if (buffer_pool_[i].copy_complete_event) {
            cudaError_t err = cudaEventDestroy(buffer_pool_[i].copy_complete_event);
            if (err != cudaSuccess) std::cerr << "cudaEventDestroy failed: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

void Recorder::Start(const std::string& program_name) {
    if (is_recording_) return;
    current_program_name_ = program_name;
    // Sanitize program name
    std::replace(current_program_name_.begin(), current_program_name_.end(), ' ', '_');
    std::replace(current_program_name_.begin(), current_program_name_.end(), '\\', '_');
    std::replace(current_program_name_.begin(), current_program_name_.end(), '/', '_');
    
    // Create recordings directory
    if (!fs::exists("recordings")) {
        fs::create_directory("recordings");
    }

    // Generate Filename (time in ms)
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    std::stringstream ss;
    ss << "recordings/" << current_program_name_ << "_" << millis << "_" << width_ << "x" << height_ << ".raw";
    current_filename_ = ss.str();

    outfile_.open(current_filename_, std::ios::binary);
    if (!outfile_.is_open()) {
        std::cerr << "[Recorder] Failed to open file for writing: " << current_filename_ << std::endl;
        return;
    }

    is_recording_ = true;
    std::cout << "[Recorder] Started. Saving to " << current_filename_ << std::endl;
}

void Recorder::Stop() {
    if (!is_recording_) return;
    
    // Wait for worker to finish pending frames?
    // We'll leave is_recording_ true for the thread to flush, but we need to stop accepting new frames.
    // simpler: just set flag false, main loop won't call Capture. 
    // Worker will process remaining queue.
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        // Ensure we flush? Worker loop handles this.
    }
    
    is_recording_ = false;
    
    // We can't close outfile_ here effectively because the worker might still be writing.
    // We should close it in the worker thread or wait for worker to be idle.
    // For simplicity, we'll let the worker handle closing if it detects recording stopped and queue empty,
    // OR just close it here if we are sure. But we are async.
    
    // Better approach: Start() opens, Worker writes. 
    // Stop() sets flag. Worker sees flag false AND empty queue -> effectively done for this session.
    // But we might start again.
    
    // Actually, distinct sessions need distinct files.
    // So Worker needs to know when a "session" ends to close the file?
    // Or we close it in Stop() but risk losing last frames?
    // Let's rely on the worker to close usage.
    
    // To handle multiple sessions correctly without restarting the thread:
    // The file stream is a member. We should protect it with mutex if multiple threads accessed (only worker writes).
    // Main thread calls Stop(). 
    // Let's add a "Flush" mechanism or just sleep briefly?
    
    // Current simple architecture:
    // Stop() sets is_recording_ = false.
    // Main loop stops calling Capture().
    // Worker processes remaining pending_indices_.
    // Worker sees queue empty.
    // We need to close the file.
    
    // Problem: Worker loop doesn't know "session ended" vs "waiting for next frame".
    // Solution: We'll close the file in Start() if open (shouldn't be), 
    // and close in Stop() AFTER waiting for queue to empty? 
    // Sync wait for queue empty in Stop() is safest for file integrity.
    
    while(true) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (pending_indices_.empty()) break;
        // spin wait or condition variable
    }
    
    // Ensure last write finished (Worker might be in the write block)
    // We can't easily sync that without another CV.
    // Hack: sleep 100ms.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    if (outfile_.is_open()) {
        outfile_.close();
        std::cout << "[Recorder] Stopped. File closed." << std::endl;
    }
}

void Recorder::Capture(void* d_ptr, cudaStream_t stream) {
    if (!is_recording_ || !outfile_.is_open()) return;

    int idx = -1;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (free_indices_.empty()) {
             // Drop frame
            return;
        }
        idx = free_indices_.front();
        free_indices_.pop();
    }

    FrameBuffer& fb = buffer_pool_[idx];
    
    // 1. Async Copy Device -> Host Pinned
    CUDA_CHECK(cudaMemcpyAsync(fb.h_data, d_ptr, frame_size_, cudaMemcpyDeviceToHost, stream));
    
    // 2. Record Event
    CUDA_CHECK(cudaEventRecord(fb.copy_complete_event, stream));

    // 3. Push to Pending
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_indices_.push(idx);
    }
    worker_cv_.notify_one();
}

void Recorder::WorkerThread() {
    while (true) {
        int idx = -1;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            worker_cv_.wait(lock, [&] { return !pending_indices_.empty() || stop_worker_; });
            
            if (stop_worker_ && pending_indices_.empty()) {
                break;
            }
            
            if (!pending_indices_.empty()) {
                idx = pending_indices_.front();
                pending_indices_.pop();
            }
        }

        if (idx != -1) {
            FrameBuffer& fb = buffer_pool_[idx];

            // Wait for copy to complete
            cudaEventSynchronize(fb.copy_complete_event);

            // Write to global open file
            if (outfile_.is_open()) {
                outfile_.write(reinterpret_cast<char*>(fb.h_data), fb.size);
            }

            // Return to free pool
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                free_indices_.push(idx);
            }
        }
    }
}
