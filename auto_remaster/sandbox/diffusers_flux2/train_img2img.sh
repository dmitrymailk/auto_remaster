# Proxy settings for WandB
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

accelerate launch train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-base-4B  \
  --output_dir="outputs/flux2_i2i" \
  --dataset_name="dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw" \
  --image_column="edited_image" --cond_image_column="input_image" \
  --cache_dir="/code/dataset/nfs_pix2pix_1920_1080_v6_upscale_2x_raw" \
  --do_fp8_training \
  --resolution=512 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=50 \
  --max_train_steps=10000 \
  --rank=32 \
  --validation_steps=250 \
  --mixed_precision=bf16 \
  --seed="0" \
  --lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2"