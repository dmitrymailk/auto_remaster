# from huggingface_hub import snapshot_download
from datasets import load_dataset

load_dataset(
    "dim/nfs_pix2pix_1920_1080_v6",
    cache_dir="dataset/nfs_pix2pix_1920_1080_v6",
    num_proc=16,
)
