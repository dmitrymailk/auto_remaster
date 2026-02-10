import numpy as np
from PIL import Image
import os

# Configuration
INPUT_FILE = "capture_input_fp16.bin"
RESOLUTION = 512
CHANNELS = 3

def inspect_tensor():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run the C++ app and press 'S' to generate it.")
        return

    print(f"Loading {INPUT_FILE}...")
    
    # Load Raw FP16 Data
    # The C++ app saves: NCHW format, FP16 (2 bytes per element)
    # Total bytes = 512 * 512 * 3 * 2 = 1,572,864 bytes
    data = np.fromfile(INPUT_FILE, dtype=np.float16)
    
    expected_elems = RESOLUTION * RESOLUTION * CHANNELS
    if data.size != expected_elems:
        print(f"Error: Expected {expected_elems} elements, got {data.size}")
        return

    # Reshape to (C, H, W)
    tensor = data.reshape((CHANNELS, RESOLUTION, RESOLUTION))
    
    # Statistics
    print("\nTensor Statistics (Normalized [-1, 1]):")
    print(f"  Min: {tensor.min():.4f}")
    print(f"  Max: {tensor.max():.4f}")
    print(f"  Mean: {tensor.mean():.4f}")
    print(f"  Std: {tensor.std():.4f}")
    
    # Check for NaN/Inf
    if np.isnan(tensor).any():
        print("  WARNING: NaNs detected!")
    if np.isinf(tensor).any():
        print("  WARNING: Infs detected!")

    # Denormalize to [0, 255] for Visualization
    # Image = (Tensor * 0.5 + 0.5) * 255
    image_data = (tensor.astype(np.float32) * 0.5 + 0.5)
    image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)
    
    # Convert NCHW -> HWC for PIL
    image_data = np.transpose(image_data, (1, 2, 0)) # (H, W, C)
    
    # Save Image
    try:
        img = Image.fromarray(image_data, 'RGB')
        img.save("capture_visualization.png")
        print("\nSaved visualization to capture_visualization.png")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    inspect_tensor()
