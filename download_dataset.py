# from huggingface_hub import snapshot_download
from datasets import load_dataset

load_dataset(
    # "dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw",
    "dim/nfs_pix2pix_1920_1080_v5_upscale_1x_raw",
    cache_dir="dataset/nfs_pix2pix_1920_1080_v5_upscale_1x_raw",
    num_proc=16,
)
