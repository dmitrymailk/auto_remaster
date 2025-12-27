# from huggingface_hub import snapshot_download
from datasets import load_dataset

load_dataset(
    # "dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw",
    # "dim/nfs_pix2pix_1920_1080_v5_upscale_1x_raw",
    "dim/render_nfs_4screens_6_sdxl_1_wan_mix",
    cache_dir="dataset/render_nfs_4screens_6_sdxl_1_wan_mix",
    num_proc=16,
)
