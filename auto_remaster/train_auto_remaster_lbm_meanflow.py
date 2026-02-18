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
    AutoModel,
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
from typing import cast, Tuple, Any, Optional, Union
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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.unets.unet_2d import UNet2DOutput


# --- MeanFlow UNet2DModel with dual time embedding ---
# Adds a second time embedding for `r` (target time for mean velocity).
# Uses the same subclassing technique as TwinFlowUNet2DModel.


class MeanFlowUNet2DModel(UNet2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        block_out_channels = self.config.block_out_channels
        flip_sin_to_cos = self.config.flip_sin_to_cos
        freq_shift = self.config.freq_shift
        time_embedding_dim = self.config.time_embedding_dim

        self.time_proj_2 = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos,
            freq_shift,
        )
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        timestep_input_dim = block_out_channels[0]
        self.time_embedding_2 = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        timestep2: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # Embed t
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        # Embed r
        timesteps2 = timestep2
        if not torch.is_tensor(timesteps2):
            timesteps2 = torch.tensor(
                [timesteps2], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps2) and len(timesteps2.shape) == 0:
            timesteps2 = timesteps2[None].to(sample.device)
        timesteps2 = timesteps2 * torch.ones(
            sample.shape[0], dtype=timesteps2.dtype, device=timesteps2.device
        )

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        t_emb_2 = self.time_proj_2(timesteps2).to(dtype=self.dtype)

        emb = self.time_embedding(t_emb) + self.time_embedding_2(t_emb_2)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError(
                "class_embedding needs to be initialized in order to use class conditioning"
            )

        skip_sample = sample
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample, res_samples, emb, skip_sample
                )
            else:
                sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)


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
    num_inference_steps: int = field(default=1)
    metrics_list: list[str] = field(default=None)
    lpips_factor: float = field(default=1.0)
    bridge_noise_sigma: float = field(default=0.001)
    latent_loss_weight: float = field(default=1.0)
    latent_loss_type: str = field(default="l2")
    # MeanFlow-specific parameters
    meanflow_p_mean: float = field(default=0.0)
    meanflow_p_std: float = field(default=1.0)
    consistency_ratio: float = field(default=1.0)


unet2d_config = {
    "sample_size": 64,
    "in_channels": 32 * 2,
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


def log_validation(
    vae=None,
    unet=None,
    noise_scheduler=None,
    accelerator=None,
    weight_dtype=None,
    global_step=None,
    training_args=None,
    model_args=None,
    diffusion_args: DiffusionTrainingArguments = None,
    dataset=None,
    train_transforms=None,
    checkpoint_path: str = None,
    **kwargs,
):
    logger.info("Running validation... ")

    vae_val = (
        AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            subfolder="vae",
            torch_device="cuda",
        )
        .to(accelerator.device)
        .to(weight_dtype)
    )
    vae_val.eval()

    unet_val = MeanFlowUNet2DModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    unet_val.eval()

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(
                diffusion_args.resolution,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(diffusion_args.resolution),
        ]
    )

    test_images_ids = list(range(0, len(dataset), 30))
    rng = random.Random(training_args.seed)
    amount = min(30, len(test_images_ids))
    selected_ids = rng.sample(test_images_ids, amount)

    images = []
    originals = []
    generated = []

    for idx in selected_ids:
        item = dataset[idx]

        orig_source_pil = item[diffusion_args.source_image_name].convert("RGB")
        target_pil = item[diffusion_args.target_image_name].convert("RGB")

        source_tensor = valid_transforms(orig_source_pil)
        target_tensor = valid_transforms(target_pil)

        c_t = (
            train_transforms(orig_source_pil)
            .unsqueeze(0)
            .to(vae_val.dtype)
            .to(vae_val.device)
        )

        with torch.no_grad():
            z_source = (
                vae_val.encode(
                    c_t,
                    return_dict=False,
                )[0].sample()
                * vae_val.config.scaling_factor
            )

            # MeanFlow one-step inference: u(z_source, t=1, r=0)
            # z_target = z_source - u
            model_input = torch.cat([z_source, z_source], dim=1)
            t_val = torch.ones(1, device=z_source.device, dtype=z_source.dtype)
            r_val = torch.zeros(1, device=z_source.device, dtype=z_source.dtype)

            u_pred = unet_val(
                model_input,
                t_val,
                r_val,
                return_dict=False,
            )[0]

            sample = z_source - u_pred

            output_image = vae_val.decode(
                sample / vae_val.config.scaling_factor,
                return_dict=False,
            )[0].clamp(-1, 1)

            pred_image_pil = transforms.ToPILImage()(
                output_image[0].cpu().float() * 0.5 + 0.5
            )

        img_h = Image.fromarray(
            np.hstack(
                (
                    np.array(source_tensor),
                    np.array(pred_image_pil),
                    np.array(target_tensor),
                )
            )
        )

        originals.append(item[diffusion_args.source_image_name])
        generated.append(pred_image_pil)
        images.append(img_h)

    for tracker in accelerator.trackers:
        tracker.log(
            {
                "validation": [wandb.Image(image) for image in images],
            }
        )

    del vae_val
    del unet_val
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


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
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()

    set_seed(training_args.seed)
    diffusers.utils.logging.set_verbosity_info()

    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)

    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    unet = MeanFlowUNet2DModel(**unet2d_config)
    unet.set_attention_backend("flash")

    vae.requires_grad_(False)
    vae.eval()
    unet.train()

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

    column_names = dataset["train"].column_names
    caption_column = diffusion_args.caption_column
    source_column = diffusion_args.source_image_name
    target_column = diffusion_args.target_image_name

    interpolation = transforms.InterpolationMode.LANCZOS

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
        examples["source_images"] = [train_transforms(image) for image in source_images]
        examples["target_images"] = [train_transforms(image) for image in target_images]
        return examples

    with accelerator.main_process_first():
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=training_args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = (
        training_args.warmup_steps * accelerator.num_processes
    )

    num_training_steps_for_scheduler = (
        training_args.max_steps * accelerator.num_processes
    )

    layers_to_opt = list(unet.parameters())

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

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )

    training_args.num_train_epochs = math.ceil(
        training_args.max_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        tracker_config = dict(vars(training_args))
        accelerator.init_trackers(
            diffusion_args.tracker_project_name,
            tracker_config,
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

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

    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint != "latest":
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
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

    progress_bar = tqdm(
        range(0, training_args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    z_source = (
                        vae.encode(
                            batch["source_images"].to(weight_dtype),
                            return_dict=False,
                        )[0].sample()
                    )
                    z_source = z_source * vae.config.scaling_factor

                    z_target = (
                        vae.encode(
                            batch["target_images"].to(weight_dtype),
                            return_dict=False,
                        )[0].sample()
                    )
                    z_target = z_target * vae.config.scaling_factor

                bsz = z_source.shape[0]
                device = z_source.device

                # --- MeanFlow Time Sampling ---
                # Sample two times from logit-normal, sort so r < t
                t_pair = torch.randn((2, bsz), device=device)
                t_pair = (
                    t_pair.mul(diffusion_args.meanflow_p_std)
                    .add(diffusion_args.meanflow_p_mean)
                    .sigmoid()
                )
                t_pair = t_pair.sort(dim=0).values
                r = t_pair[0]  # target time (closer to data)
                t = t_pair[1]  # current time (closer to source)

                # Apply consistency ratio: r = t - rand * cor * (t - 0)
                # When cor=1.0, r can be anywhere in [0, t]
                # When cor=0.0, r = t (standard flow matching)
                if diffusion_args.consistency_ratio < 1.0:
                    r = t - torch.rand_like(t) * diffusion_args.consistency_ratio * t
                    r = r.clamp(min=0.0)

                t_expand = t.view(-1, 1, 1, 1)
                r_expand = r.view(-1, 1, 1, 1)

                # Bridge interpolant: z_t = t * z_source + (1-t) * z_target
                # At t=1: z_t = z_source (starting point)
                # At t=0: z_t = z_target (destination)
                z_t = t_expand * z_source + (1.0 - t_expand) * z_target

                # Instantaneous velocity on the bridge
                v = z_source - z_target

                # --- MeanFlow JVP Loss ---
                # We need to compute u(z_t, t, r) and ∂u/∂t simultaneously
                # using the JVP (Jacobian-Vector Product) trick from the paper.
                #
                # The tangent vectors are:
                #   dz/dt = v (instantaneous velocity direction)
                #   dt/dt = 1
                #   dr/dt = 0

                # We need z_source as conditioning (concatenated input).
                # Since z_source is constant w.r.t. the JVP variables,
                # we capture it in the closure.
                z_source_detached = z_source.detach()

                def model_fn(z_in, t_in, r_in):
                    model_input = torch.cat([z_in, z_source_detached], dim=1)
                    return unet(model_input, t_in, r_in, return_dict=False)[0]

                tangents = (
                    v,
                    torch.ones(bsz, device=device),
                    torch.zeros(bsz, device=device),
                )

                u, dudt = torch.autograd.functional.jvp(
                    model_fn,
                    (z_t, t, r),
                    tangents,
                    create_graph=True,
                )

                # MeanFlow identity: u = v - (t - r) * ∂u/∂t
                u_target = v - (t_expand - r_expand) * dudt.detach()

                # Weighted MSE loss (weight by 1/(t-r) to balance different intervals)
                weight = 1.0 / (t_expand - r_expand + 1e-5)
                loss = (weight * (u - u_target).pow(2)).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "meanflow_loss": loss.detach().item(),
                    "t_mean": t.mean().item(),
                    "r_mean": r.mean().item(),
                    "t_minus_r_mean": (t - r).mean().item(),
                }
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if global_step % training_args.save_steps == 0:
                    if accelerator.is_main_process:
                        if training_args.save_total_limit is not None:
                            checkpoints = os.listdir(training_args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            if len(checkpoints) >= training_args.save_total_limit:
                                num_to_remove = (
                                    len(checkpoints)
                                    - training_args.save_total_limit
                                    + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        training_args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            training_args.output_dir, f"checkpoint-{global_step}"
                        )
                        unwrap_model(unet).save_pretrained(
                            os.path.join(save_path, "unet")
                        )
                        logger.info(f"Saved state to {save_path}")

                        log_validation(
                            vae=vae,
                            unet=unet,
                            noise_scheduler=None,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            global_step=global_step,
                            script_args=script_args,
                            training_args=training_args,
                            model_args=model_args,
                            diffusion_args=diffusion_args,
                            dataset=dataset["train"],
                            train_transforms=train_transforms,
                            checkpoint_path=save_path,
                        )

            if accelerator.sync_gradients:
                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step >= training_args.max_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
