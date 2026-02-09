import os
import argparse
from datasets import load_dataset
from PIL import Image
import multiprocessing

def save_example(example, dataset_name, static_dir):
    try:
        # Assuming the dataset has 'input_image' and 'edited_image' columns based on previous notebook analysis
        # Adjust column names if different. The user mentioned "dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw"
        
        # Typically indices are not explicitly in the row, so we might need to handle naming carefully.
        # However, map doesn't give us easy linear index unless with_indices is True.
        # check if 'cursor' or similar matches. 
        # Actually, let's use a global counter or pass index if possible.
        # dataset.map provides 'idx' if with_indices=True.
        
        idx = example['idx'] # This will be injected by the wrapper function below
        
        input_path = os.path.join(static_dir, dataset_name, "input", f"{idx}.png")
        output_path = os.path.join(static_dir, dataset_name, "output", f"{idx}.png")
        
        if not os.path.exists(input_path):
            example['input_image'].save(input_path)
            
        if not os.path.exists(output_path):
            example['edited_image'].save(output_path)
            
    except Exception as e:
        print(f"Error saving {idx}: {e}")

def process_batch(batch, indices, dataset_name, static_dir):
    # Wrapper to handle batch processing if needed, but for image saving simple map is okay.
    # We will use simple map with indices.
    pass

def main():
    parser = argparse.ArgumentParser(description="Prepare data for labeling tool")
    parser.add_argument("--dataset", type=str, default="dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw", help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for dataset")
    args = parser.parse_args()

    if args.cache_dir is None:
        # Dynamic cache dir based on dataset name pattern from user requirements
        # e.g., dim/nfs_pix2pix_1920_1080_v6 -> /code/dataset/nfs_pix2pix_1920_1080_v6
        dataset_short = args.dataset.split("/")[-1]
        args.cache_dir = f"/code/dataset/{dataset_short}"
        print(f"Using dynamic cache directory: {args.cache_dir}")

    dataset_name_clean = args.dataset.replace("/", "_")
    static_dir = "static"
    
    input_dir = os.path.join(static_dir, dataset_name_clean, "input")
    output_dir = os.path.join(static_dir, dataset_name_clean, "output")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    
    print(f"Dataset loaded. {len(dataset)} examples. Saving to {input_dir} and {output_dir}...")

    # Wrapper function for map to include static args
    def save_wrapper(example, idx):
        # Add idx to example for convenience in the logic
        example['idx'] = idx 
        save_example(example, dataset_name_clean, static_dir)
        return example

    dataset.map(save_wrapper, with_indices=True, num_proc=args.num_proc)

    # Initialize labels file with "bad" default
    labels_dir = "labels"
    os.makedirs(labels_dir, exist_ok=True)
    labels_file = os.path.join(labels_dir, f"labels_{dataset_name_clean}.json")
    if not os.path.exists(labels_file):
        print(f"Initializing {labels_file} with default 'bad' labels...")
        # We need the indices. Since map processes in parallel, we can't easily get them from it.
        # But we know the dataset length and can assume 0..N-1 if it was a standard dataset, 
        # but here we used 'idx' from the dataset itself in save_wrapper? 
        # Wait, save_wrapper used example['idx'].
        # Let's extract indices from the dataset to be sure.
        
        # Reloading or using the dataset object
        # The 'idx' column might not exist if it's not in the source.
        # The previous code in save_wrapper used example['idx'].
        # Let's assume the dataset has an 'idx' column or we added it.
        
        # Actually, simpler approach: just list the output directory files
        indices = []
        if os.path.exists(input_dir):
             files = os.listdir(input_dir)
             for f in files:
                if f.endswith(".png"):
                    try:
                        idx = int(f.replace(".png", ""))
                        indices.append(idx)
                    except ValueError:
                        pass
        
        initial_labels = {str(i): "bad" for i in indices}
        
        import json
        with open(labels_file, "w") as f:
            json.dump(initial_labels, f, indent=2)
            
    print("Done!")

if __name__ == "__main__":
    main()
