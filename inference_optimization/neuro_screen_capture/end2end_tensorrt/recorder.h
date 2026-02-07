#pragma once

#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"

// Struct to hold a single frame's data in the ring buffer
struct FrameBuffer {
    unsigned char* h_data = nullptr; // Pinned host memory
    size_t size = 0;
    std::string filename;
    cudaEvent_t copy_complete_event = nullptr;
};

class Recorder {
public:
    Recorder(int width, int height, int fps = 24);
    ~Recorder();

    // Start/Stop recording session (resets internal state if needed)
    void Start(const std::string& program_name);
    void Stop();

    // Schedule a frame for writing
    // d_ptr: Device pointer to the source data (Uint8 RGB)
    // The data is copied asynchronously to a pinned host buffer.
    void Capture(void* d_ptr, cudaStream_t stream);

    bool IsRecording() const { return is_recording_; }

private:
    void WorkerThread();

    int width_;
    int height_;
    int channels_ = 3;
    size_t frame_size_;

    std::atomic<bool> is_recording_{false};
    std::atomic<bool> stop_worker_{false};
    
    // Ring Buffer Resources
    static const int POOL_SIZE = 60; // Enough for ~2.5 seconds of buffering at 24 FPS
    std::vector<FrameBuffer> buffer_pool_;
    
    std::queue<int> free_indices_;
    std::queue<int> pending_indices_;
    
    std::mutex queue_mutex_;
    std::condition_variable worker_cv_;
    
    std::thread worker_thread_;
    std::string current_program_name_;
    
    // Single file output
    std::ofstream outfile_;
    std::string current_filename_;
};
