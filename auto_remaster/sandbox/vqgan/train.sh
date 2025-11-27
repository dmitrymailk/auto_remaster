export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

accelerate launch train_vqgan.py \
  --dataset_name=huggan/flowers-102-categories \
  --image_column=edited_image \
  --validation_images images/output.png \
  --resolution=128 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --report_to=wandb \
  --allow_tf32