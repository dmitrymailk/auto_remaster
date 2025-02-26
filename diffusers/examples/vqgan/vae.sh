accelerate launch train_vqgan.py \
  --dataset_name=cifar10 \
  --image_column=img \
  --validation_images images/bird.jpg images/car.jpg images/dog.jpg images/frog.jpg \
  --resolution=128 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --report_to=wandb