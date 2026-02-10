import numpy as np
from PIL import Image
import os
import glob
import math

# Configuration
RESOLUTION_W = 1024
RESOLUTION_H = 512
CHANNELS = 3
RECORDINGS_DIR = "recordings"

def inspect_recording():
    # Find all .raw files in recordings dir or current dir
    search_path = os.path.join(RECORDINGS_DIR, "*.raw")
    files = glob.glob(search_path)
    
    # Fallback to current dir if nothing found in recordings
    if not files:
        files = glob.glob("*.raw")
    
    if not files:
        print(f"No .raw files found in {RECORDINGS_DIR} or current directory.")
        return

    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    latest_file = files[0]

    print(f"Inspecting newest recording: {latest_file}")
    
    # Parse resolution from filename (format: ..._WxH.raw)
    try:
        base = os.path.basename(latest_file)
        res_part = base.split('_')[-1].split('.')[0]
        w_str, h_str = res_part.split('x')
        width = int(w_str)
        height = int(h_str)
        print(f"Detected resolution from filename: {width}x{height}")
    except Exception as e:
        print(f"Could not parse resolution from filename, using default {RESOLUTION_W}x{RESOLUTION_H}. Error: {e}")
        width = RESOLUTION_W
        height = RESOLUTION_H
    
    # Load Raw Uint8 Data
    try:
        data = np.fromfile(latest_file, dtype=np.uint8)
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    frame_size = width * height * CHANNELS
    total_frames = data.size // frame_size
    
    print(f"Total size: {data.size} bytes")
    print(f"Frame size: {frame_size} bytes")
    print(f"Total frames: {total_frames}")

    if total_frames == 0:
        print("File is empty or too small for one frame.")
        return

    # Reshape to (N, H, W, C)
    # The C++ recorder saves interleaved RGB
    try:
        video_data = data.reshape((total_frames, height, width, CHANNELS))
    except ValueError:
        print("Warning: Data size is not a perfect multiple of frame size. Truncating...")
        valid_size = total_frames * frame_size
        video_data = data[:valid_size].reshape((total_frames, height, width, CHANNELS))

    # Save first, middle, and last frame
    indices_to_save = [0, total_frames // 2, total_frames - 1]
    # Remove duplicates if only 1 or 2 frames
    indices_to_save = sorted(list(set(indices_to_save)))

    for i in indices_to_save:
        frame = video_data[i]
        
        # Statistics
        print(f"\nFrame {i} Statistics:")
        print(f"  Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.2f}")
        
        try:
            img = Image.fromarray(frame, 'RGB')
            output_filename = f"{latest_file}_frame_{i}.png"
            img.save(output_filename)
            print(f"Saved visualization to {output_filename}")
        except Exception as e:
            print(f"Error saving image: {e}")

if __name__ == "__main__":
    inspect_recording()
