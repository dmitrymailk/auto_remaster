# vpn for wandb
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES="0"
timestamp=$(date +%s)
folder_name="nfs_pix2pix_$timestamp"
output_dir="output/pix2pix_light/$folder_name"
# accelerate launch --mixed_precision 'bf16' src/train_pix2pix_turbo.py \
accelerate launch src/train_pix2pix_light.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir=$output_dir \
    --dataset_folder="dim/nfs_pix2pix_1920_1080_v5" \
    --train_batch_size=4 \
    --enable_xformers_memory_efficient_attention --viz_freq 15 \
    --track_val_fid \
    --train_image_prep="resized_crop_512" \
    --test_image_prep="resized_crop_512" \
    --lora_rank_unet=128 \
    --lora_rank_vae=64 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=500 \
    --eval_freq=50 \
    --max_train_steps=100000 \
    --report_to "wandb" \
    --diff_ver "v1"
    # --report_to "" 
    # --mixed_precision="bf16" \
    # --resolution=512 