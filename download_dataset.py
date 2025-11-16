from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="QingyanBai/Ditto-1M",
    repo_type="dataset",
    local_dir="./Ditto-1M",
    allow_patterns=[
        "videos/global_style1/*",
    ],
)
