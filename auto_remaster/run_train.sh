pushd /code/

# vpn for wandb
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0

config_path=/code/auto_remaster/single_gpu.yaml

# hf_train_config=/code/auto_remaster/configs/sd1.5_ddpm.yaml
hf_train_config=/code/auto_remaster/configs/lbm.yaml
# hf_train_config=/code/auto_remaster/configs/vae.yaml

# python -m auto_remaster.train_auto_remaster --config $hf_train_config
# accelerate launch --config_file=$config_path -m auto_remaster.train_auto_remaster --config $hf_train_config
accelerate launch --config_file=$config_path -m auto_remaster.train_auto_remaster_lbm --config $hf_train_config
# accelerate launch --config_file=$config_path -m auto_remaster.train_vae --config $hf_train_config