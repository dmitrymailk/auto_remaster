accelerate launch src/train_pix2pix_turbo.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="output/pix2pix_turbo/fill50k" \
    --dataset_folder="data/my_fill50k" \
    --resolution=512 \
    --train_batch_size=2 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --track_val_fid \
    --report_to "" \
    # --report_to "wandb" \
	# --tracker_project_name "pix2pix_turbo_fill50k"