import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from transformers import TrainingArguments, HfArgumentParser

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
    AutoencoderTiny,
    UNet2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    EMAModel,
    compute_dream_and_update_latents,
    compute_snr,
)
from diffusers.utils import (
    check_min_version,
    deprecate,
    is_wandb_available,
    make_image_grid,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)
from typing import cast, Tuple, Any
from dataclasses import dataclass, field
import wandb
import gc
from PIL import Image
import clip
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import lpips
import torchvision

from auto_remaster.evaluation import ImageEvaluator
import bitsandbytes as bnb


@dataclass
class DiffusionTrainingArguments:
    use_ema: bool = field(default=False)
    non_ema_revision: str = field(default=None)
    resolution: int = field(default=512)
    revision: str = field(default=None)
    variant: str = field(default=None)
    scale_lr: bool = field(default=False)
    cache_dir: str = field(default=None)
    source_image_name: str = field(default="source_image")
    target_image_name: str = field(default="target_image")
    caption_column: str = field(default="caption")
    tracker_project_name: str = field(default="auto_remaster")
    noise_offset: float = field(default=0.0)
    input_perturbation: float = field(default=0.0)
    num_inference_steps: int = field(default=4)
    metrics_list: list[str] = field(default=None)
    lpips_factor: float = field(default=5.0)
    gan_factor: float = field(default=0.5)
    bridge_noise_sigma: float = field(default=0.001)
    timestep_sampling: str = field(
        default="custom_timesteps"
    )  # "uniform", "custom_timesteps"
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    latent_loss_weight: float = field(default=1.0)
    latent_loss_type: str = field(default="l2")  # "l2" or "l1"


unet2d_config = {
    "sample_size": 64,
    # "in_channels": 4,
    # "in_channels": 16,
    "in_channels": 32,
    # "out_channels": 4,
    # "out_channels": 16,
    "out_channels": 32,
    "center_input_sample": False,
    "time_embedding_type": "positional",
    "freq_shift": 0,
    "flip_sin_to_cos": True,
    "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "downsample_padding": 1,
    "downsample_type": "conv",
    "upsample_type": "conv",
    "dropout": 0.0,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-05,
    "resnet_time_scale_shift": "default",
    "add_attention": False,
}

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def make_parser():
    dataclass_types = (
        ScriptArguments,
        SFTConfig,
        ModelConfig,
        DiffusionTrainingArguments,
    )
    parser = TrlParser(dataclass_types)
    return parser


def main():
    parser = make_parser()
    script_args, training_args, model_args, diffusion_args = cast(
        Tuple[
            ScriptArguments,
            SFTConfig,
            ModelConfig,
            DiffusionTrainingArguments,
            Any,
        ],
        parser.parse_args_and_config(),
    )

    logging_dir = os.path.join(training_args.output_dir, training_args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=training_args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        project_config=accelerator_project_config,
        mixed_precision="no",
        # mixed_precision="fp16",
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    # If passed along, set the training seed now.
    set_seed(training_args.seed)
    diffusers.utils.logging.set_verbosity_info()

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)

    # Load scheduler for training and sampling (Flow Matching)
    # Create scheduler from scratch (no need for SDXL)
    # One scheduler is enough - set_timesteps() in validation doesn't affect training
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    num_steps = diffusion_args.num_inference_steps

    # sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    # noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)

    # weight_dtype = torch.float16
    weight_dtype = torch.float32

    # vae = AutoencoderTiny.from_pretrained(
    #     "madebyollin/taesd",
    #     torch_device="cuda",
    #     torch_dtype=weight_dtype,
    # )
    # vae.decoder.ignore_skip = False
    vae = AutoencoderKL.from_pretrained(
        # "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    unet = UNet2DModel(**unet2d_config)
    # unet = UNet2DModel.from_pretrained('checkpoints/auto_remaster/lbm/checkpoint-28800')
    # unet.enable_xformers_memory_efficient_attention()
    unet.set_attention_backend("flash")

    # Freeze VAE (as in LBM)
    vae.requires_grad_(False)
    vae.eval()
    # text_encoder.requires_grad_(False)
    unet.train()

    if training_args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    if diffusion_args.scale_lr:
        training_args.learning_rate = (
            training_args.learning_rate
            * training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
            * accelerator.num_processes
        )

    dataset = load_dataset(
        script_args.dataset_name,
        script_args.dataset_config,
        cache_dir=diffusion_args.cache_dir,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    caption_column = diffusion_args.caption_column
    source_column = diffusion_args.source_image_name
    target_column = diffusion_args.target_image_name

    # Get the specified interpolation method from the args
    interpolation = transforms.InterpolationMode.LANCZOS

    # Data preprocessing transformations

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                diffusion_args.resolution,
                interpolation=interpolation,
            ),
            transforms.CenterCrop(diffusion_args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    def preprocess_train(examples):
        source_images = [image.convert("RGB") for image in examples[source_column]]
        target_images = [image.convert("RGB") for image in examples[target_column]]
        # TODO: при более сложных преобразованиях трансформацию необходимо делать в паре
        # а не независимо
        examples["source_images"] = [train_transforms(image) for image in source_images]
        examples["target_images"] = [train_transforms(image) for image in target_images]
        return examples

    with accelerator.main_process_first():
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        source_images = torch.stack([example["source_images"] for example in examples])
        source_images = source_images.to(memory_format=torch.contiguous_format).to(
            weight_dtype
        )

        target_images = torch.stack(
            [example["target_images"] for example in examples]
        ).to(weight_dtype)
        target_images = target_images.to(memory_format=torch.contiguous_format).to(
            weight_dtype
        )
        return {
            "source_images": source_images,
            "target_images": target_images,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = (
        training_args.warmup_steps * accelerator.num_processes
    )

    num_training_steps_for_scheduler = (
        training_args.max_steps * accelerator.num_processes
    )

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    # Only optimize UNet parameters (VAE is frozen)
    layers_to_opt = []
    for n, _p in unet.named_parameters():
        layers_to_opt.append(_p)

    # optimizer = torch.optim.AdamW(
    optimizer = bnb.optim.AdamW8bit(
        layers_to_opt,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        vae,
    ) = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        vae,
    )

    # Move LPIPS to gpu and cast to weight_dtype
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )

    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(
        training_args.max_steps / num_update_steps_per_epoch
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(training_args))
        accelerator.init_trackers(
            diffusion_args.tracker_project_name,
            tracker_config,
        )

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size
        * accelerator.num_processes
        * training_args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(training_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{training_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            training_args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(training_args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    trainer = TwinFlow()

    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # l_acc = [unet, vae]
            l_acc = [unet]
            with accelerator.accumulate(*l_acc):
                # Convert images to latent space (Bridge Matching approach)
                with torch.no_grad():
                    z_source = vae.encode(
                        batch["source_images"].to(weight_dtype),
                        return_dict=False,
                        # )[0]
                    )[0].sample()
                    z_source = z_source * vae.config.scaling_factor

                    z_target = vae.encode(
                        batch["target_images"].to(weight_dtype),
                        return_dict=False,
                        # )[0]
                    )[0].sample()
                    z_target = z_target * vae.config.scaling_factor

    accelerator.end_training()


if __name__ == "__main__":
    main()
