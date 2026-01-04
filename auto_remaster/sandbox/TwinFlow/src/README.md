## Codebase Structure

```sh
.
├── configs/    # configuration files for distinct operational setups.
├── data/       # dataset definitions and data loading pipelines.
├── methods/    # core algorithm implementations and logic.
├── scripts/    # shell scripts for execution of training and sampling.
├── services/   # auxiliary utilities and shared tooling services.
└── steerers/   # primary control flows for training and sampling.
```

## Quick Start

To deploy TwinFlow on OpenUni, please follow the procedure outlined below.

### Environment Configuration
Begin by setting up the environment prerequisites as detailed in the official [OpenUni repository](https://github.com/wusize/OpenUni). Ensure all dependencies are correctly installed before proceeding.

### Model Configuration

- Generator Backbone

Download the [OpenUni generator backbone](https://huggingface.co/wusize/openuni/blob/main/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth) checkpoint locally. Once downloaded, update the configuration file `configs/openuni_task/openuni_full.yaml` to reflect the local path:

```yaml
model:
  type: ./networks/openuni/openuni_l_internvl3_2b_sana_1_6b_512_hf.py
  path: path/to/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth 
  in_chans: 16
```

- Other Components
  - OpenUni Encoder: [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)
  - SANA 1.6B: [Sana_1600M_512px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers)
  - DC-AE: [dc-ae-f32c32-sana-1.1-diffusers](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers)

Prior to initiating training, define the environment variables pointing to your downloaded component models. You may modify `scripts/openuni/train_ddp.sh` directly or export them in your script:

```sh
export INTERNVL3_PATH="path/to/InternVL3-2B"
export SANA_1600M_512PX_PATH="path/to/Sana_1600M_512px_diffusers"
export DCAE_PATH="path/to/dc-ae-f32c32-sana-1.1-diffusers"
```

### Launch Training

- Standard Training (TwinFlow on OpenUni):

```sh
scripts/openuni/train_ddp.sh configs/openuni_task/openuni_full.yaml
```

- Data-Free Training (No Text-Image Pairs Required):

```sh
scripts/openuni/train_ddp.sh configs/openuni_task/openuni_full_imgfree.yaml
```

### Sampling Images

To directly run sampling:

```sh
scripts/openuni/sample_demo.sh configs/openuni_task/openuni_full.yaml
```

After training, the trained model supports 3 sampling modes: **few-step**, **any-step**, and **standard multi-step**. We provide different sampling configurations for each mode in the yaml for reference:

```yaml
# few-step sampling
sample:
  ckpt: "700" # <- change to the ckpt
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 2 # 1
  stochast_ratio: 1.0 # 0.8
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.0, 1.0, 1.0]
  rfba_gap_steps: [0.001, 0.7]
  sampling_style: few

# any-step sampling
sample:
  ckpt: "700"
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 4 # 8
  stochast_ratio: 0.0
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.0, 1.0, 1.0]
  rfba_gap_steps: [0.001, 0.5]
  sampling_style: any

# multi-step sampling
sample:
  ckpt: "700"
  cfg_scale: 0
  cfg_interval: [0.00, 0.00]
  sampling_steps: 30
  stochast_ratio: 0.0
  extrapol_ratio: 0.0
  sampling_order: 1
  time_dist_ctrl: [1.17, 0.8, 1.1]
  rfba_gap_steps: [0.001, 0.0]
  sampling_style: mul
```