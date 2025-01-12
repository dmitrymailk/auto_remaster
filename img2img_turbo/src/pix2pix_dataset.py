import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import (
    parse_args_paired_training,
    PairedDataset,
    NFSPairedDataset,
)


def main():
    net_pix2pix = Pix2Pix_Turbo(
        lora_rank_unet=8,
        lora_rank_vae=4,
    )
    dataset_folder = "dim/nfs_pix2pix_1920_1080_v5"
    dataset_train = NFSPairedDataset(
        dataset_folder=dataset_folder,
        image_prep="resized_crop_512",
        split="train",
        tokenizer=net_pix2pix.tokenizer,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
        shuffle=True,
        num_workers=1,
    )


if __name__ == "__main__":
    main()
