# vpn for wandb
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

loglr=-7
width=64
lr=$(python -c "import math; print(2**${loglr})")
run_name="stage_4_msepool-cont-512-1.0-1.0-batch-gradnorm_make_deterministic_test"
echo "Running ${run_name}"

# torchrun --nproc_per_node=1 vae_trainer.py \
torchrun --nproc_per_node=1 vae_trainer_hf_dataset.py \
--learning_rate_vae ${lr} \
--vae_ch ${width} \
--run_name ${run_name} \
--num_epochs 200 \
--max_steps 100000 \
--evaluate_every_n_steps 500 \
--learning_rate_disc 1e-5 \
--batch_size 2 \
--do_clamp \
--do_ganloss \
--use_lecam True \
--project_name "HrDecoderAE" \
--decoder_also_perform_hr True \
--vae_z_channels 4
#--load_path "/home/ubuntu/auravasa/ckpt/stage_3_msepool-cont-512-1.0-1.0-batch-gradnorm/vae_epoch_1_step_23501.pt"
#--load_path "/home/ubuntu/auravasa/ckpt/stage2_msepool-cont-512-1.0-1.0-batch-gradnorm/vae_epoch_0_step_28501.pt"
# --load_path "/home/ubuntu/auravasa/ckpt/exp_vae_ch_256_lr_0.0078125_weighted_percep+f8areapool_l2_0.0/vae_epoch_1_step_27001.pt"  