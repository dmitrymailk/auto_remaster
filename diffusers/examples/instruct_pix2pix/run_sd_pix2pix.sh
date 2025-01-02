# MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_NAME="stablediffusionapi/juggernaut-reborn"
# export DATASET_ID="fusing/instructpix2pix-1000-samples"
DATASET_ID="dim/nfs_pix2pix_1920_1080_v5"
mkdir -p models
timestamp=$(date +%s)
folder_name="nfs_pix2pix_$timestamp"
WANDB_NAME=$folder_name
output_dir="models/$folder_name"

accelerate launch --mixed_precision="bf16" train_instruct_pix2pix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --enable_xformers_memory_efficient_attention \
    --resolution=1024 --random_flip \
    --train_batch_size=2 --gradient_accumulation_steps=8 \
    --max_train_steps=1500 \
    --checkpointing_steps=500 --checkpoints_total_limit=1 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=bf16 \
    --seed=42 \
    --allow_tf32 \
    --output_dir $output_dir \
    --report_to=wandb
