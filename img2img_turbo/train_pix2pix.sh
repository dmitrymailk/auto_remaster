export CUDA_VISIBLE_DEVICES="0"
timestamp=$(date +%s)
folder_name="nfs_pix2pix_$timestamp"
output_dir="/code/img2img_turbo/output/pix2pix_turbo/$folder_name"
# accelerate launch --mixed_precision 'bf16' src/train_pix2pix_turbo.py \
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir=$output_dir \
    --dataset_folder="dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw" \
    --train_batch_size=1 \
    --enable_xformers_memory_efficient_attention --viz_freq 15 \
    --track_val_fid \
    --train_image_prep="resized_crop_512" \
    --test_image_prep="resized_crop_512" \
    --lora_rank_unet=16 \
    --lora_rank_vae=8 \
    --gradient_accumulation_steps=8 \
    --checkpointing_steps=100 \
    --eval_freq=50 \
    --max_train_steps=100000 \
    --report_to "wandb" \
    # --report_to "" 
    # --mixed_precision="bf16" \
    # --resolution=512 \
	# --tracker_project_name "pix2pix_turbo_fill50k"