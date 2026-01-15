# vpn for wandb
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0

accelerate launch train_rectified_instaflow.py \
  --pretrained_model_name_or_path="nota-ai/bk-sdm-small"  \
  --generated_dataset_name="Isamu136/bk-sdm-small_generated_images_pokemon_blip" \
  --dataset_name="lambdalabs/pokemon-blip-captions" \
  --output_dir="./rectified_flow_checkpoints" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epoch=5000 \
  --checkpointing_steps=500 \
  --gradient_checkpointing \
  --mixed_precision=no \
  --use_8bit_adam \
  --validation_prompts="A pokemon like an orange cat on a leaf" \
  --validation_epochs=1 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --report_to="wandb" \
  --resume_from_checkpoint="latest"
#   --rank=4 \