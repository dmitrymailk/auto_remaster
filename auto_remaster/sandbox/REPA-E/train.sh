export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0

accelerate launch train_repae.py \
    --max-train-steps=400000 \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --data-dir="data" \
    --output-dir="exps" \
    --batch-size=4 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-B/2" \
    --checkpointing-steps=50000 \
    --loss-cfg-path="configs/l1_lpips_kl_gan.yaml" \
    --vae="f8d4" \
    --vae-ckpt="pretrained/sdvae/sdvae-f8d4.pt" \
    --disc-pretrained-ckpt="pretrained/sdvae/sdvae-f8d4-discriminator-ckpt.pt" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --vae-align-proj-coeff=1.5 \
    --bn-momentum=0.1 \
    --exp-name="sit-xl-dinov2-b-enc8-repae-sdvae-0.5-1.5-400k"