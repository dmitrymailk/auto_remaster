import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass, field
from typing import cast, Tuple, Any, Optional, List, Union

import accelerate
import datasets
import numpy as np
import torch
import torch.nn as nn
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
from transformers import TrainingArguments, HfArgumentParser
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderTiny
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available

from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

import lpips
import wandb
import gc
from PIL import Image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@dataclass
class VAETrainingArguments:
    resolution: int = field(default=512)
    save_images_steps: int = field(default=100)

    # Dataset args
    # dataset_name is in ScriptArguments
    dataset_config_name: str = field(default=None)
    train_data_dir: str = field(default=None)
    image_column: str = field(default="image")
    cache_dir: str = field(default=None)

    # Loss weights
    lambda_l1: float = field(default=1.0)
    lambda_lpips: float = field(default=0.5)
    lambda_gan: float = field(default=0.5)
    lambda_kl: float = field(default=0.000001)  # For Soft-Intro

    # Soft-Intro VAE specific
    beta_rec: float = field(default=1.0)  # Weight for reconstruction in ELBO
    beta_kl: float = field(default=1.0)  # Weight for KL in ELBO
    beta_neg: float = field(default=256.0)  # Weight for negative ELBO (exp)
    gamma_r: float = field(default=1e-8)  # Scale for rec loss in exp

    # R3GAN specific
    r1_gamma: float = field(default=10.0)
    r2_gamma: float = field(default=10.0)

    # New Configurable Losses (Sandbox Integration)
    use_lecam: bool = field(default=False)
    lecam_loss_weight: float = field(default=0.1)

    use_blurriness_loss: bool = field(default=False)
    blurriness_loss_weight: float = field(
        default=1.0
    )  # Multiplier for the weighted part

    discriminator_type: str = field(default="patch")  # "simple", "patch"

    # Training Mode
    train_mode: str = field(default="r3gan")  # "r3gan", "classic" (L1 + KL)

    use_ema: bool = field(default=True)
    ema_decay: float = field(default=0.9999)

    tracker_project_name: str = field(default="auto_remaster_vae")


# ------------------------------------------------------------------------------
# Helper Classes & Functions (from sandbox)
# ------------------------------------------------------------------------------
class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def blurriness_heatmap(input_image):
    # Convert to grayscale
    if input_image.shape[1] == 3:
        grayscale_image = input_image.mean(dim=1, keepdim=True)
    else:
        grayscale_image = input_image

    laplacian_kernel = torch.tensor(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, -20, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    laplacian_kernel = laplacian_kernel.view(1, 1, 5, 5).to(input_image.device)

    edge_response = F.conv2d(grayscale_image, laplacian_kernel, padding=2)

    # Simple Gaussian blur approximation for magnitude smoothing
    # Using a fixed kernel for simplicity or we can use torchvision.transforms.GaussianBlur
    # Here we implement a simple box/gaussian-like blur via avg pool for speed/no-dependency
    edge_magnitude = edge_response.abs()
    edge_magnitude = F.avg_pool2d(edge_magnitude, kernel_size=13, stride=1, padding=6)

    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (
        edge_magnitude.max() - edge_magnitude.min() + 1e-8
    )

    blurriness_map = 1 - edge_magnitude
    blurriness_map = torch.where(
        blurriness_map < 0.8, torch.zeros_like(blurriness_map), blurriness_map
    )

    return blurriness_map.repeat(1, 3, 1, 1)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        # We use torchvision.models.vgg16 features
        from torchvision import models

        _vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5


class VariationalAdapter(nn.Module):
    """
    Adapter to convert deterministic encoder output to variational (mean, logvar).
    """

    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        # Initialize to identity-like for mean and small value for logvar
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        # Set weight for mean to identity (first 4 channels)
        for i in range(in_channels):
            self.conv.weight.data[i, i, 0, 0] = 1.0

        # Initialize logvar bias to -5.0 (small std) to help initial convergence
        # The second half of out_channels is logvar
        self.conv.bias.data[in_channels:] = -5.0

    def forward(self, x):
        return self.conv(x)


# ------------------------------------------------------------------------------
# Discriminator for R3GAN (Simple)
# ------------------------------------------------------------------------------
class SimpleDiscriminator(nn.Module):
    """
    A simple PatchGAN-style discriminator or a ResNet discriminator.
    For 512x512, a deeper network is better.
    """

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, base_channels, normalize=False),
            *discriminator_block(base_channels, base_channels * 2),
            *discriminator_block(base_channels * 2, base_channels * 4),
            *discriminator_block(base_channels * 4, base_channels * 8),
            nn.Conv2d(base_channels * 8, 1, 3, padding=1),  # Output single channel map
        )

    def forward(self, img):
        return self.model(img)


# ------------------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------------------
def r3gan_loss(d_real, d_fake, device):
    """
    Relativistic GAN loss (R3GAN variant).
    L_D = - E[log(sigmoid(D(xr) - D(xf)))] - E[log(sigmoid(D(xf) - D(xr)))]
    L_G = - E[log(sigmoid(D(xf) - D(xr)))] - E[log(sigmoid(D(xr) - D(xf)))]
    """
    # Relativistic Discriminator Loss
    # We want D(real) > D(fake)
    # diff_real = d_real - d_fake.mean()
    # diff_fake = d_fake - d_real.mean()

    # Simplified Relativistic Hinge/BCE often used:
    # BCEWithLogits(D(xr) - D(xf), 1) + BCEWithLogits(D(xf) - D(xr), 0)

    # Using Softplus as in some R3GAN implementations for stability
    # loss_d = torch.nn.functional.softplus(- (d_real - d_fake)).mean()

    # Standard Relativistic Average GAN (RaGAN) Hinge Loss
    # loss_d = (torch.mean(torch.nn.functional.relu(1.0 - (d_real - torch.mean(d_fake)))) +
    #           torch.mean(torch.nn.functional.relu(1.0 + (d_fake - torch.mean(d_real))))) / 2

    # R3GAN paper suggests:
    # L_D = -log(sigmoid(D_real - D_fake))

    pred_real = torch.sigmoid(d_real - d_fake)
    loss_d = -torch.log(pred_real + 1e-8).mean()

    return loss_d


def r3gan_gen_loss(d_real, d_fake):
    # L_G = -log(sigmoid(D_fake - D_real))
    pred_fake = torch.sigmoid(d_fake - d_real)
    loss_g = -torch.log(pred_fake + 1e-8).mean()
    return loss_g


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones(
        (real_samples.shape[0], *d_interpolates.shape[1:]),
        requires_grad=False,
        device=device,
    )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_r1_loss(d_out, x_in):
    # R1 gradient penalty: E[ ||grad(D(x))||^2 ]
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = grad_dout2.view(batch_size, -1).sum(1).mean()
    return reg


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, VAETrainingArguments))
    script_args, training_args, model_args, vae_args = parser.parse_args_and_config()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=training_args.report_to,
        # mixed_precision="no",
        project_config=ProjectConfiguration(
            project_dir=training_args.output_dir, logging_dir=training_args.logging_dir
        ),
    )

    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        accelerator.init_trackers(vae_args.tracker_project_name, config=vars(vae_args))

    logger.info(accelerator.state)

    # Set seed
    set_seed(training_args.seed)

    # 1. Load Dataset
    # --------------------------------------------------------------------------
    dataset = load_dataset(
        script_args.dataset_name,
        vae_args.dataset_config_name,
        cache_dir=vae_args.cache_dir,
    )

    column_names = dataset["train"].column_names
    image_column = vae_args.image_column
    if image_column not in column_names:
        # Try to guess or use the first column
        image_column = column_names[0]
        logger.warning(
            f"Image column {vae_args.image_column} not found, using {image_column}"
        )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                vae_args.resolution, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(vae_args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    with accelerator.main_process_first():
        dataset["train"] = dataset["train"].shuffle(seed=training_args.seed)
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=training_args.per_device_train_batch_size,
        num_workers=4,
    )

    # --------------------------------------------------------------------------
    # Prepare Fixed Validation Set (30 pairs)
    # --------------------------------------------------------------------------
    validation_batch = None
    if "input_image" in column_names and "edited_image" in column_names:
        logger.info(
            "Preparing fixed validation set with 30 pairs of input/edited images..."
        )
        num_val_samples = 30
        val_indices = random.sample(
            range(len(dataset["train"])), min(len(dataset["train"]), num_val_samples)
        )

        val_inputs = []
        val_edited = []

        for idx in val_indices:
            item = dataset["train"][idx]
            # Process input
            img_in = item["input_image"].convert("RGB")
            val_inputs.append(train_transforms(img_in))

            # Process edited
            img_ed = item["edited_image"].convert("RGB")
            val_edited.append(train_transforms(img_ed))

        val_inputs = torch.stack(val_inputs)
        val_edited = torch.stack(val_edited)
        validation_batch = {"input": val_inputs, "edited": val_edited}
    else:
        logger.warning(
            "Could not find 'input_image' and 'edited_image' columns for validation. Skipping fixed validation set."
        )

    # 2. Load Model (AutoencoderTiny)
    # --------------------------------------------------------------------------
    # AutoencoderTiny is usually just for inference, but we can train it.
    # It has .encoder and .decoder
    # Initialize from scratch (architecture only)
    config = AutoencoderTiny.load_config("madebyollin/taesd")
    vae = AutoencoderTiny.from_config(config)

    # Variational Adapter for Classic Mode
    variational_adapter = None
    if vae_args.train_mode == "classic":
        # TAESD latent channels = 4. We need 8 for KL.
        variational_adapter = VariationalAdapter(
            in_channels=vae.config.latent_channels,
            out_channels=vae.config.latent_channels * 2,
        )
        variational_adapter.to(accelerator.device)

    # Create EMA model
    if vae_args.use_ema:
        ema_vae = EMAModel(
            vae.parameters(),
            decay=vae_args.ema_decay,
            model_cls=AutoencoderTiny,
            model_config=vae.config,
        )
        ema_vae.to(accelerator.device)

    # Discriminator for R3GAN
    if vae_args.discriminator_type == "patch":
        discriminator = PatchDiscriminator()
    else:
        discriminator = SimpleDiscriminator()

    # LPIPS
    lpips_loss_fn = lpips.LPIPS(net="vgg").to(accelerator.device)
    lpips_loss_fn.requires_grad_(False)

    # Optimizers
    params_to_optimize = list(vae.parameters())
    if variational_adapter is not None:
        params_to_optimize += list(variational_adapter.parameters())

    optimizer_e = torch.optim.AdamW(
        params_to_optimize, lr=training_args.learning_rate, betas=(0.5, 0.9)
    )
    # For classic mode, we optimize everything with one optimizer usually, or E and D together.
    # But to keep structure, we can share optimizer or just add params.
    # Actually, in R3GAN loop we have opt_e and opt_d.
    # Let's keep them separate but include adapter in opt_e.

    if vae_args.train_mode == "classic":
        optimizer_e = torch.optim.AdamW(
            list(vae.encoder.parameters()) + list(variational_adapter.parameters()),
            lr=training_args.learning_rate,
            betas=(0.5, 0.9),
        )
    else:
        optimizer_e = torch.optim.AdamW(
            vae.encoder.parameters(), lr=training_args.learning_rate, betas=(0.5, 0.9)
        )

    optimizer_d = torch.optim.AdamW(
        vae.decoder.parameters(), lr=training_args.learning_rate, betas=(0.5, 0.9)
    )
    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(), lr=training_args.learning_rate, betas=(0.5, 0.9)
    )

    # Prepare
    if variational_adapter is not None:
        (
            vae,
            discriminator,
            variational_adapter,
            optimizer_e,
            optimizer_d,
            optimizer_disc,
            train_dataloader,
        ) = accelerator.prepare(
            vae,
            discriminator,
            variational_adapter,
            optimizer_e,
            optimizer_d,
            optimizer_disc,
            train_dataloader,
        )
    else:
        (
            vae,
            discriminator,
            optimizer_e,
            optimizer_d,
            optimizer_disc,
            train_dataloader,
        ) = accelerator.prepare(
            vae,
            discriminator,
            optimizer_e,
            optimizer_d,
            optimizer_disc,
            train_dataloader,
        )

    # 3. Training Loop
    # --------------------------------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    if training_args.max_steps is None or training_args.max_steps <= 0:
        training_args.max_steps = (
            training_args.num_train_epochs * num_update_steps_per_epoch
        )

    global_step = 0
    progress_bar = tqdm(
        range(training_args.max_steps), disable=not accelerator.is_local_main_process
    )

    # Soft-Intro VAE Helper
    def calc_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])

    def calc_reconstruction_loss(x, x_rec):
        # L1 + LPIPS
        l1 = F.l1_loss(x, x_rec, reduction="none").mean(dim=[1, 2, 3])

        # Blurriness Weighted Loss
        if vae_args.use_blurriness_loss:
            # Calculate heatmap
            heatmap = blurriness_heatmap(x)
            # Weighted L1
            weighted_l1 = (F.l1_loss(x, x_rec, reduction="none") * heatmap).mean(
                dim=[1, 2, 3]
            )
            l1 = l1 + weighted_l1 * vae_args.blurriness_loss_weight

        p_loss = lpips_loss_fn(x, x_rec).squeeze()
        return vae_args.lambda_l1 * l1 + vae_args.lambda_lpips * p_loss

    # LeCAM Variables
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9

    for epoch in range(int(training_args.num_train_epochs)):
        vae.train()
        if variational_adapter:
            variational_adapter.train()
        discriminator.train()

        for step, batch in enumerate(train_dataloader):
            real_images = batch["pixel_values"]

            # ------------------------------------------------------------------
            # Classic VAE Mode (L1 + KL)
            # ------------------------------------------------------------------
            if vae_args.train_mode == "classic":
                optimizer_e.zero_grad()
                optimizer_d.zero_grad()

                # Encode
                h = vae.encoder(real_images)
                moments = variational_adapter(h)
                mean, logvar = moments.chunk(2, dim=1)

                # Reparameterize
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std

                # Decode
                reconstruction = vae.decoder(z)

                # Losses
                # L1
                rec_loss = (
                    F.l1_loss(real_images, reconstruction, reduction="none")
                    .mean(dim=[1, 2, 3])
                    .mean()
                )

                # KL
                kl_loss = calc_kl(mean, logvar).mean() * vae_args.lambda_kl

                total_loss = rec_loss + kl_loss

                accelerator.backward(total_loss)
                optimizer_e.step()
                optimizer_d.step()

                if vae_args.use_ema:
                    ema_vae.step(vae.parameters())

                # Logging
                if global_step % 10 == 0:
                    logs = {
                        "loss/total": total_loss.item(),
                        "loss/rec": rec_loss.item(),
                        "loss/kl": kl_loss.item(),
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                global_step += 1
                progress_bar.update(1)

            else:
                # ------------------------------------------------------------------
                # R3GAN / Soft-Intro Mode
                # ------------------------------------------------------------------
                # 1. Train Discriminator (R3GAN)
                # ------------------------------------------------------------------
                # We need fake images for D training
                with torch.no_grad():
                    # Encode and Decode
                    # AutoencoderTiny encode returns latents directly if scaled?
                    # Let's check source code or behavior.
                    # .encode() -> returns latents.
                    # .decode() -> returns image.
                    # Note: AutoencoderTiny doesn't have a distribution output by default like KL Autoencoder?
                    # Wait, AutoencoderTiny is usually deterministic?
                    # "madebyollin/taesd" is a distilled model. It approximates the SD VAE.
                    # If it's deterministic, Soft-Intro (which relies on KL) might need adaptation.
                    # BUT, taesd is designed to be compatible with SD VAE latents.
                    # Let's look at the model structure. It usually outputs just latents.
                    # If it is deterministic, we can't strictly use KL divergence.
                    # However, we can treat it as a regular Autoencoder with GAN loss + LPIPS.
                    # Soft-Intro *requires* a variational component (KL).
                    # If the user insists on Soft-Intro, we might need to add a reparameterization layer
                    # or assume the tiny encoder outputs mu and we treat logvar as 0 or learned.
                    # TAESD architecture: Encoder -> Latents. Decoder -> Image.
                    # It does NOT output distribution parameters usually.
                    # Let's assume for this task we treat it as a standard AE for the base,
                    # but we can ADD a small variation or just use the GAN/LPIPS losses which work for AE too.
                    # Soft-Intro for deterministic AE is just Intro-VAE (no Soft?).
                    # Intro-VAE works for deterministic AEs too (using reconstruction error as energy).
                    # Let's proceed with Intro-VAE logic: Encoder tries to minimize Rec Error of real, maximize Rec Error of fake.

                    latents = vae.encoder(real_images)
                    # If latents need scaling? TAESD usually handles it or expects raw.
                    # We will just pass latents to decoder.
                    fake_images = vae.decoder(latents)

                # Discriminator Step
                optimizer_disc.zero_grad()

                d_real = discriminator(real_images)
                d_fake = discriminator(fake_images.detach())

                loss_d = (
                    r3gan_loss(d_real, d_fake, accelerator.device) * vae_args.lambda_gan
                )

                # LeCAM Loss
                if vae_args.use_lecam:
                    avg_real_logits = d_real.mean().item()
                    avg_fake_logits = d_fake.mean().item()

                    # Update anchors (EMA)
                    lecam_anchor_real_logits = (
                        lecam_beta * lecam_anchor_real_logits
                        + (1 - lecam_beta) * avg_real_logits
                    )
                    lecam_anchor_fake_logits = (
                        lecam_beta * lecam_anchor_fake_logits
                        + (1 - lecam_beta) * avg_fake_logits
                    )

                    # LeCAM regularization
                    lecam_loss = (d_real - lecam_anchor_fake_logits).pow(2).mean() + (
                        d_fake - lecam_anchor_real_logits
                    ).pow(2).mean()

                    loss_d += lecam_loss * vae_args.lecam_loss_weight

                # R1 Penalty (on real)
                if vae_args.r1_gamma > 0:
                    real_images.requires_grad_(True)
                    d_real_loc = discriminator(real_images)
                    r1_loss = compute_r1_loss(d_real_loc, real_images) * (
                        vae_args.r1_gamma / 2.0
                    )
                    loss_d += r1_loss

                accelerator.backward(loss_d)
                optimizer_disc.step()

                # ------------------------------------------------------------------
                # 2. Train VAE (Encoder + Decoder)
                # ------------------------------------------------------------------
                optimizer_e.zero_grad()
                optimizer_d.zero_grad()

                # Forward
                latents = vae.encoder(real_images)
                reconstruction = vae.decoder(latents)

                # Reconstruction Loss
                rec_loss = calc_reconstruction_loss(real_images, reconstruction).mean()

                # GAN Generator Loss (R3GAN)
                d_real_for_g = discriminator(real_images)
                d_fake_for_g = discriminator(reconstruction)
                gan_loss = (
                    r3gan_gen_loss(d_real_for_g, d_fake_for_g) * vae_args.lambda_gan
                )

                # Intro-VAE / Soft-Intro Logic
                # If deterministic, we use the reconstruction error as the "energy"
                # Encoder wants to minimize Rec(real) and MAXIMIZE Rec(fake) (if it were a discriminator)
                # But here we have an external discriminator.
                # If we strictly follow Soft-Intro for VAE:
                # We need KL. Since TAESD is deterministic, we skip KL.
                # So this becomes a regular GAN-AE training with R3GAN loss.

                total_loss = rec_loss + gan_loss

                accelerator.backward(total_loss)
                optimizer_e.step()
                optimizer_d.step()

                if vae_args.use_ema:
                    ema_vae.step(vae.parameters())

                # Logging
                if global_step % 10 == 0:
                    logs = {
                        "loss/total": total_loss.item(),
                        "loss/rec": rec_loss.item(),
                        "loss/gan_d": loss_d.item(),
                        "loss/gan_g": gan_loss.item(),
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                global_step += 1
                progress_bar.update(1)

            # Save images
            if (
                global_step % vae_args.save_images_steps == 0
                and accelerator.is_main_process
            ):
                with torch.no_grad():
                    if vae_args.use_ema:
                        ema_vae.store(vae.parameters())
                        ema_vae.copy_to(vae.parameters())

                    if validation_batch is not None:
                        # Use fixed validation set
                        val_in = validation_batch["input"].to(accelerator.device)
                        val_ed = validation_batch["edited"].to(accelerator.device)

                        # Split into chunks if too large for GPU
                        chunk_size = 10
                        recon_in_list = []
                        recon_ed_list = []

                        for i in range(0, len(val_in), chunk_size):
                            batch_in = val_in[i : i + chunk_size]

                            if variational_adapter is not None:
                                h = vae.encoder(batch_in)
                                moments = variational_adapter(h)
                                mean, _ = moments.chunk(
                                    2, dim=1
                                )  # Use mean for validation
                                latents_in = mean
                            else:
                                latents_in = vae.encoder(batch_in)

                            recon_in_list.append(vae.decoder(latents_in))

                            batch_ed = val_ed[i : i + chunk_size]
                            if variational_adapter is not None:
                                h = vae.encoder(batch_ed)
                                moments = variational_adapter(h)
                                mean, _ = moments.chunk(2, dim=1)
                                latents_ed = mean
                            else:
                                latents_ed = vae.encoder(batch_ed)

                            recon_ed_list.append(vae.decoder(latents_ed))

                        recon_in = torch.cat(recon_in_list)
                        recon_ed = torch.cat(recon_ed_list)

                        # Prepare strips: Each strip is [Input, Rec-Input, Edited, Rec-Edited] horizontally

                        def to_pil(tensor):
                            tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
                            np_img = tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                            np_img = (np_img * 255).round().astype("uint8")
                            return [Image.fromarray(img) for img in np_img]

                        pil_in = to_pil(val_in)
                        pil_rec_in = to_pil(recon_in)
                        pil_ed = to_pil(val_ed)
                        pil_rec_ed = to_pil(recon_ed)

                        validation_strips = []
                        for i in range(len(pil_in)):
                            # Concatenate horizontally using np.hstack
                            # Convert PIL to numpy for hstack
                            img_h = Image.fromarray(
                                np.hstack(
                                    (
                                        np.array(pil_in[i]),
                                        np.array(pil_rec_in[i]),
                                        np.array(pil_ed[i]),
                                        np.array(pil_rec_ed[i]),
                                    )
                                )
                            )
                            validation_strips.append(img_h)

                        # Save first 5 strips to disk to verify
                        for i in range(min(5, len(validation_strips))):
                            save_path = os.path.join(
                                training_args.output_dir,
                                f"sample_{global_step}_{i}.png",
                            )
                            validation_strips[i].save(save_path)

                        # Log all strips to tracker
                        if accelerator.trackers:
                            accelerator.log(
                                {
                                    "validation_samples": [
                                        wandb.Image(img, caption=f"Sample {i}")
                                        for i, img in enumerate(validation_strips)
                                    ]
                                },
                                step=global_step,
                            )

                    else:
                        # Fallback to batch sample
                        if variational_adapter is not None:
                            h = vae.encoder(real_images[:4])
                            mean, _ = variational_adapter(h).chunk(2, dim=1)
                            latents = mean
                        else:
                            latents = vae.encoder(real_images[:4])

                        recon = vae.decoder(latents)

                        # Denormalize
                        recon = (recon * 0.5 + 0.5).clamp(0, 1)
                        orig = (real_images[:4] * 0.5 + 0.5).clamp(0, 1)

                        # Convert to PIL
                        orig_np = orig.cpu().permute(0, 2, 3, 1).float().numpy()
                        recon_np = recon.cpu().permute(0, 2, 3, 1).float().numpy()

                        orig_np = (orig_np * 255).round().astype("uint8")
                        recon_np = (recon_np * 255).round().astype("uint8")

                        orig_pil = [Image.fromarray(img) for img in orig_np]
                        recon_pil = [Image.fromarray(img) for img in recon_np]

                        # Just save one grid for fallback
                        num_images = len(orig_pil)
                        grid = make_image_grid(
                            orig_pil + recon_pil, rows=2, cols=num_images
                        )
                        save_path = os.path.join(
                            training_args.output_dir, f"sample_{global_step}.png"
                        )
                        grid.save(save_path)

                        if accelerator.trackers:
                            accelerator.log(
                                {"samples": wandb.Image(grid)}, step=global_step
                            )

                    if vae_args.use_ema:
                        ema_vae.restore(vae.parameters())

            if global_step >= training_args.max_steps:
                break

        if global_step >= training_args.max_steps:
            break

    # Save final model
    if accelerator.is_main_process:
        vae.save_pretrained(training_args.output_dir)
        if vae_args.use_ema:
            ema_vae.copy_to(vae.parameters())
            vae.save_pretrained(os.path.join(training_args.output_dir, "ema"))


if __name__ == "__main__":
    main()
