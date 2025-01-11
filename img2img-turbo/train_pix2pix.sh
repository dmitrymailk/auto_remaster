export CUDA_VISIBLE_DEVICES="0,1"
timestamp=$(date +%s)
folder_name="nfs_pix2pix_$timestamp"
output_dir="output/pix2pix_turbo/$folder_name"
# accelerate launch --mixed_precision 'bf16' src/train_pix2pix_turbo.py \
accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir=$output_dir \
    --dataset_folder="dim/nfs_pix2pix_1920_1080_v6" \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention --viz_freq 100 \
    --track_val_fid \
    --train_image_prep="resized_crop_512" \
    --test_image_prep="resized_crop_512" \
    --lora_rank_unet=128 \
    --eval_freq=500 \
    --max_train_steps=100000 \
    --report_to "wandb" \
    # --report_to "" 
    # --mixed_precision="bf16" \
    # --resolution=512 \
	# --tracker_project_name "pix2pix_turbo_fill50k"