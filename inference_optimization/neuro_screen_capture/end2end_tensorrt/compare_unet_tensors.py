import numpy as np
import os

CONF = {
    "z_source": (128, 32, 32), # CHW (Batch 1 assumed)
    "unet_input": (256, 32, 32),
    "unet_output": (128, 32, 32)
}

def load_bin(filename, shape):
    if not os.path.exists(filename):
        print(f"MISSING: {filename}")
        return None
    
    try:
        data = np.fromfile(filename, dtype=np.float16)
        expected_len = np.prod(shape)
        if data.size != expected_len:
            print(f"SIZE MISMATCH: {filename}: Expected {expected_len}, got {data.size}")
            return None
        return data.reshape(shape)
    except Exception as e:
        print(f"ERROR reading {filename}: {e}")
        return None

def compare(name):
    print(f"\n--- Comparing {name} ---")
    py_file = f"debug_py_{name}.bin"
    cpp_file = f"debug_cpp_{name}.bin"
    
    py_data = load_bin(py_file, CONF[name])
    cpp_data = load_bin(cpp_file, CONF[name])
    
    if py_data is None or cpp_data is None:
        return

    diff = np.abs(py_data - cpp_data)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"Py Stats: Min={py_data.min():.4f}, Max={py_data.max():.4f}, Mean={py_data.mean():.4f}")
    print(f"Cpp Stats: Min={cpp_data.min():.4f}, Max={cpp_data.max():.4f}, Mean={cpp_data.mean():.4f}")
    print(f"Difference: MAE={mae:.6f}, Max={max_diff:.6f}")
    
    if max_diff > 1e-2:
        print(">> FAIL: Significant difference detected!")
    else:
        print(">> PASS: Tensors match.")

if __name__ == "__main__":
    print("Verifying Tensor Similarity (PyTorch vs TensorRT C++)...")
    for key in CONF.keys():
        compare(key)
