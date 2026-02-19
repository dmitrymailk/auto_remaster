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
from torchvision import models
from collections import namedtuple
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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


# --- GAN Loss and Helper Functions ---


# --- R3GAN Discriminator (ported from R3GAN/R3GAN/Networks.py, pure PyTorch) ---


def _r3gan_msr_init(layer, activation_gain=1):
    fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    layer.weight.data.normal_(0, activation_gain / math.sqrt(fan_in))
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


def _create_lowpass_kernel(weights):
    kernel = np.convolve(weights, [1, 1]).reshape(1, -1)
    kernel = torch.tensor(kernel.T @ kernel, dtype=torch.float32)
    return kernel / kernel.sum()


class R3GANBiasedActivation(torch.nn.Module):
    _GAIN = math.sqrt(2 / (1 + 0.2**2))
    _FUNC = torch.nn.LeakyReLU(0.2)

    def __init__(self, channels):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        b = self.bias.to(x.dtype)
        y = x + b.view(1, -1, 1, 1) if x.dim() == 4 else x + b.view(1, -1)
        return R3GANBiasedActivation._FUNC(y)


class R3GANConvolution(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, groups=1, activation_gain=1):
        super().__init__()
        self.layer = _r3gan_msr_init(
            torch.nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            activation_gain=activation_gain,
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.layer.weight.to(x.dtype),
            padding=self.layer.padding,
            groups=self.layer.groups,
        )


class R3GANInterpolativeDownsampler(torch.nn.Module):
    def __init__(self, filter_weights):
        super().__init__()
        self.register_buffer("kernel", _create_lowpass_kernel(filter_weights))
        self.filter_radius = len(filter_weights) // 2

    def forward(self, x):
        k = self.kernel.view(1, 1, self.kernel.shape[0], self.kernel.shape[1]).to(
            x.dtype
        )
        b, c, h, w = x.shape
        y = F.conv2d(x.view(b * c, 1, h, w), k, stride=2, padding=self.filter_radius)
        return y.view(b, c, y.shape[2], y.shape[3])


class R3GANResidualBlock(torch.nn.Module):
    def __init__(self, channels, cardinality, expansion, kernel_size, var_scale):
        super().__init__()
        n_linear = 3
        expanded = channels * expansion
        gain = R3GANBiasedActivation._GAIN * var_scale ** (-1 / (2 * n_linear - 2))

        self.linear1 = R3GANConvolution(
            channels, expanded, kernel_size=1, activation_gain=gain
        )
        self.linear2 = R3GANConvolution(
            expanded,
            expanded,
            kernel_size=kernel_size,
            groups=cardinality,
            activation_gain=gain,
        )
        self.linear3 = R3GANConvolution(
            expanded, channels, kernel_size=1, activation_gain=0
        )
        self.act1 = R3GANBiasedActivation(expanded)
        self.act2 = R3GANBiasedActivation(expanded)

    def forward(self, x):
        y = self.linear1(x)
        y = self.linear2(self.act1(y))
        y = self.linear3(self.act2(y))
        return x + y


class R3GANDownsampleLayer(torch.nn.Module):
    def __init__(self, in_ch, out_ch, resampling_filter):
        super().__init__()
        self.resampler = R3GANInterpolativeDownsampler(resampling_filter)
        if in_ch != out_ch:
            self.linear = R3GANConvolution(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.resampler(x)
        if hasattr(self, "linear"):
            x = self.linear(x)
        return x


class R3GANDiscriminativeBasis(torch.nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.basis = _r3gan_msr_init(
            torch.nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=4,
                stride=1,
                padding=0,
                groups=in_ch,
                bias=False,
            )
        )
        self.linear = _r3gan_msr_init(torch.nn.Linear(in_ch, out_dim, bias=False))

    def forward(self, x):
        return self.linear(self.basis(x).view(x.shape[0], -1))


class R3GANDiscriminatorStage(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        cardinality,
        n_blocks,
        expansion,
        kernel_size,
        var_scale,
        resampling_filter=None,
    ):
        super().__init__()
        if resampling_filter is None:
            transition = R3GANDiscriminativeBasis(in_ch, out_ch)
        else:
            transition = R3GANDownsampleLayer(in_ch, out_ch, resampling_filter)
        blocks = [
            R3GANResidualBlock(in_ch, cardinality, expansion, kernel_size, var_scale)
            for _ in range(n_blocks)
        ]
        self.layers = torch.nn.ModuleList(blocks + [transition])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class R3GANDiscriminator(torch.nn.Module):
    """R3GAN discriminator — deep ResNet without normalization, MSR init."""

    def __init__(
        self,
        width_per_stage,
        cardinality_per_stage,
        blocks_per_stage,
        expansion=2,
        kernel_size=3,
        resampling_filter=None,
    ):
        super().__init__()
        if resampling_filter is None:
            resampling_filter = [1, 2, 1]

        var_scale = sum(blocks_per_stage)
        n_stages = len(width_per_stage)

        main = []
        for i in range(n_stages - 1):
            main.append(
                R3GANDiscriminatorStage(
                    width_per_stage[i],
                    width_per_stage[i + 1],
                    cardinality_per_stage[i],
                    blocks_per_stage[i],
                    expansion,
                    kernel_size,
                    var_scale,
                    resampling_filter,
                )
            )
        # Final stage (no resampling — discriminative basis)
        main.append(
            R3GANDiscriminatorStage(
                width_per_stage[-1],
                1,
                cardinality_per_stage[-1],
                blocks_per_stage[-1],
                expansion,
                kernel_size,
                var_scale,
                None,
            )
        )

        self.extraction = R3GANConvolution(3, width_per_stage[0], kernel_size=1)
        self.main = torch.nn.ModuleList(main)

    def forward(self, x):
        x = self.extraction(x)
        for stage in self.main:
            x = stage(x)
        return x.view(x.shape[0])


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def zero_centered_gradient_penalty(samples, logits):
    (gradient,) = torch.autograd.grad(
        outputs=logits.sum(), inputs=samples, create_graph=True
    )
    return gradient.square().sum([1, 2, 3])


def gan_disc_loss(real_preds, fake_preds, disc_type="bce"):
    if disc_type == "bce":
        real_loss = F.binary_cross_entropy_with_logits(
            real_preds, torch.ones_like(real_preds)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_preds, torch.zeros_like(fake_preds)
        )
        # eval its online performance
        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

    if disc_type == "hinge":
        real_loss = F.relu(1 - real_preds).mean()
        fake_loss = F.relu(1 + fake_preds).mean()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

    return (real_loss + fake_loss) * 0.5, avg_real_preds, avg_fake_preds, acc


class ScalingLayer(torch.nn.Module):
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


class PatchDiscriminator(torch.nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = models.vgg16(pretrained=True)

        self.slice1 = torch.nn.Sequential(_vgg.features[:4])
        self.slice2 = torch.nn.Sequential(_vgg.features[4:9])
        self.slice3 = torch.nn.Sequential(_vgg.features[9:16])
        self.slice4 = torch.nn.Sequential(_vgg.features[16:23])
        self.slice5 = torch.nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        torch.nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        torch.nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        torch.nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        torch.nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        torch.nn.init.zeros_(self.binary_classifier5[-1].weight)

    def forward(self, x):
        x = self.scaling_layer(x)
        features1 = self.slice1(x)
        features2 = self.slice2(features1)
        features3 = self.slice3(features2)
        features4 = self.slice4(features3)
        features5 = self.slice5(features4)

        # torch.Size([1, 64, 256, 256]) torch.Size([1, 128, 128, 128]) torch.Size([1, 256, 64, 64]) torch.Size([1, 512, 32, 32]) torch.Size([1, 512, 16, 16])

        bc1 = self.binary_classifier1(features1).flatten(1)
        bc2 = self.binary_classifier2(features2).flatten(1)
        bc3 = self.binary_classifier3(features3).flatten(1)
        bc4 = self.binary_classifier4(features4).flatten(1)
        bc5 = self.binary_classifier5(features5).flatten(1)

        return bc1 + bc2 + bc3 + bc4 + bc5


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        grad_output_norm = torch.norm(grad_output).mean().item()

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)


class NLayerDiscriminator(torch.nn.Module):
    """PatchGAN discriminator from pix2pix — learns its own features, orthogonal to LPIPS."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_bn=True):
        super().__init__()
        use_bias = not use_bn

        kw = 4
        padw = 1
        sequence = [
            torch.nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            torch.nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            block = [
                torch.nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
            ]
            if use_bn:
                block.append(torch.nn.BatchNorm2d(ndf * nf_mult))
            block.append(torch.nn.LeakyReLU(0.2, True))
            sequence += block

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        block = [
            torch.nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
        ]
        if use_bn:
            block.append(torch.nn.BatchNorm2d(ndf * nf_mult))
        block.append(torch.nn.LeakyReLU(0.2, True))
        sequence += block

        sequence += [
            torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.main = torch.nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


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
    lpips_factor: float = field(default=1.0)
    gan_factor: float = field(default=0.1)
    bridge_noise_sigma: float = field(default=0.001)
    timestep_sampling: str = field(
        default="custom_timesteps"
    )  # "uniform", "custom_timesteps"
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    latent_loss_weight: float = field(default=1.0)
    latent_loss_type: str = field(default="l2")  # "l2" or "l1"
    learning_rate_disc: float = field(default=5e-5)
    use_lecam: bool = field(default=True)
    lecam_loss_weight: float = field(default=0.1)
    disc_warmup_steps: int = field(default=5000)
    self_generation_prob: float = field(default=0.5)
    self_generation_max_steps: int = field(default=4)
    discriminator_type: str = field(
        default="r3gan"
    )  # "vgg_patch", "patchgan", or "r3gan"
    gan_loss_type: str = field(default="r3gan")  # "bce", "hinge", or "r3gan"
    r3gan_gamma: float = field(default=0.01)
    d_noise_std: float = field(default=0.1)
    d_noise_anneal_steps: int = field(default=10000)
    d_train_every: int = field(default=1)


unet2d_config = {
    # "sample_size": 64,
    "sample_size": 32,
    # "in_channels": 4,
    # "in_channels": 16,
    # "in_channels": 32,
    # "in_channels": 32 * 2,
    "in_channels": 128 * 2,
    # "out_channels": 4,
    # "out_channels": 16,
    # "out_channels": 32,
    "out_channels": 128,
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
    noise_scheduler = FlowMatchEulerDiscreteScheduler()

    # 1. Загрузка VAE (Tiny Autoencoder для скорости и экономии памяти)
    # vae_name = "fal/FLUX.2-Tiny-AutoEncoder"
    vae_name = "dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora"
    vae_val = (
        AutoModel.from_pretrained(vae_name, trust_remote_code=True)
        .to(accelerator.device)
        .to(weight_dtype)
    )
    # vae_val.decoder.ignore_skip = False
    # vae_val = (
    #     AutoencoderKL.from_pretrained(
    #         # "black-forest-labs/FLUX.1-dev",
    #         "black-forest-labs/FLUX.2-dev",
    #         subfolder="vae",
    #         torch_device="cuda",
    #     )
    #     .to(accelerator.device)
    #     .to(weight_dtype)
    # )
    vae_val.eval()

    # 2. Загрузка UNet из чекпоинта
    unet_val = UNet2DModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    unet_val.eval()

    # 3. Подготовка трансформаций
    valid_transforms = transforms.Compose(
        [
            transforms.Resize(
                diffusion_args.resolution,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(diffusion_args.resolution),
        ]
    )

    # 4. Выбор изображений для валидации
    test_images_ids = list(range(0, len(dataset), 30))
    rng = random.Random(training_args.seed)
    amount = min(30, len(test_images_ids))
    selected_ids = rng.sample(test_images_ids, amount)

    images = []
    originals = []
    generated = []

    # Вспомогательная функция для получения сигм (как в основном скрипте)
    def _get_sigmas_val(
        scheduler,
        timesteps,
        n_dim=4,
        dtype=torch.float32,
        device="cpu",
    ):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---------------------------------------------------------
    # НАСТРОЙКА ШЕДУЛЕРА
    # ---------------------------------------------------------
    # num_steps = diffusion_args.num_inference_steps
    num_steps = 1

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)

    for idx in selected_ids:
        item = dataset[idx]

        # Подготовка исходных изображений для визуализации и метрик
        orig_source_pil = item[diffusion_args.source_image_name].convert("RGB")
        target_pil = item[diffusion_args.target_image_name].convert("RGB")

        source_tensor = valid_transforms(orig_source_pil)
        target_tensor = valid_transforms(target_pil)

        # Подготовка латента source
        # Используем train_transforms для кодирования, как в обучении
        c_t = (
            train_transforms(orig_source_pil)
            .unsqueeze(0)
            .to(vae_val.dtype)
            .to(vae_val.device)
        )
        noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)

        with torch.no_grad():
            # Encode source image
            z_source = (
                vae_val.encode(c_t, return_dict=False)
                # vae_val.encode(
                #     c_t,
                #     return_dict=False,
                # )[0].sample()
                * vae_val.config.scaling_factor
            )

            sample = z_source

            # ---------------------------------------------------------
            # ЦИКЛ СЭМПЛИНГА (Адаптировано из sample())
            # ---------------------------------------------------------
            # for i, t in enumerate(noise_scheduler.timesteps):
            for i in range(num_steps):
                t = noise_scheduler.timesteps[i]
                # 1. Масштабирование входа (если требуется шедулером)
                if hasattr(noise_scheduler, "scale_model_input"):
                    denoiser_input = noise_scheduler.scale_model_input(sample, t)
                else:
                    denoiser_input = sample
                
                # Add perturbation for validation as well, to match training distribution
                z_source_cond = z_source
                if diffusion_args.input_perturbation > 0:
                     # For validation we might want to use a fixed seed or just random? 
                     # Let's use random but maybe we should restart generator for each image?
                     # valid_rng = torch.Generator(device=z_source.device).manual_seed(training_args.seed + idx)
                     z_source_cond = z_source + diffusion_args.input_perturbation * torch.randn_like(z_source)

                denoiser_input = torch.cat([denoiser_input, z_source_cond], dim=1)
                # 2. Предсказание направления (UNet)
                # unet_val(x, t) -> output
                # print(i, t, noise_scheduler.timesteps)
                pred = unet_val(
                    denoiser_input,
                    t.to(z_source.device).repeat(denoiser_input.shape[0]),
                    return_dict=False,
                )[0]

                # 3. Шаг диффузии (Reverse Process)
                sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]

                # 4. Добавление стохастичности (Bridge Noise)
                # Не добавляем шум после последнего шага
                if i < len(noise_scheduler.timesteps) - 1:
                    # Получаем таймстемп следующего шага
                    next_timestep = (
                        noise_scheduler.timesteps[i + 1]
                        .to(z_source.device)
                        .repeat(sample.shape[0])
                    )

                    # Получаем сигму для следующего шага
                    sigmas_next = _get_sigmas_val(
                        noise_scheduler,
                        next_timestep,
                        n_dim=4,
                        dtype=weight_dtype,
                        device=z_source.device,
                    )

                    # Формула Bridge Matching: шум пропорционален sqrt(sigma * (1-sigma))
                    noise = torch.randn_like(sample)
                    bridge_factor = (sigmas_next * (1.0 - sigmas_next)) ** 0.5

                    sample = (
                        sample
                        + diffusion_args.bridge_noise_sigma * bridge_factor * noise
                    )
                    sample = sample.to(z_source.dtype)

            # ---------------------------------------------------------

            # Декодирование результата
            output_image = vae_val.decode(
                sample / vae_val.config.scaling_factor,
                return_dict=False,
            ).clamp(-1, 1)
            # )[0].clamp(-1, 1)

            pred_image_pil = transforms.ToPILImage()(
                output_image[0].cpu().float() * 0.5 + 0.5
            )
            # print(pred_image_pil)

        # Сборка горизонтальной полоски [Source | Generated | Target]
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

    # 5. Расчет метрик
    # metrics_result = {}
    # if diffusion_args.metrics_list:
    #     try:
    #         evaluator = ImageEvaluator(
    #             metrics_list=diffusion_args.metrics_list,
    #             device="cuda",
    #             num_workers=4,
    #             prefix_key="eval",
    #         )
    #         metrics_result = evaluator.evaluate(
    #             originals,
    #             generated,
    #             batch_size=16,
    #         )
    #     except Exception as e:
    #         logger.warning(f"Evaluation failed with error: {e}")

    # 6. Логирование в WandB / Accelerator
    for tracker in accelerator.trackers:
        tracker.log(
            {
                "validation": [wandb.Image(image) for image in images],
                # **metrics_result,
            }
        )

    # 7. Очистка памяти
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

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)
    selected_timesteps_tensor = torch.tensor(
        list(range(1000, 0, -1000 // num_steps)),
        # [250, 500, 750, 1000],
        # [1000],
        device=accelerator.device,
    ).long()
    # weight_dtype = torch.float16
    weight_dtype = torch.float32

    # vae = AutoencoderTiny.from_pretrained(
    #     "madebyollin/taesd",
    #     torch_device="cuda",
    #     torch_dtype=weight_dtype,
    # )
    # vae.decoder.ignore_skip = False
    # vae = AutoencoderKL.from_pretrained(
    #     # "black-forest-labs/FLUX.1-dev",
    #     "black-forest-labs/FLUX.2-dev",
    #     subfolder="vae",
    #     torch_dtype=weight_dtype,
    # )
    vae = AutoModel.from_pretrained(
        # "fal/FLUX.2-Tiny-AutoEncoder",
        "dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora",
        trust_remote_code=True,
        torch_dtype=weight_dtype,
    )

    # unet = UNet2DModel(**unet2d_config)
    unet = UNet2DModel.from_pretrained(
        "checkpoints/auto_remaster/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora/checkpoint-225600",
        subfolder="unet",
    )
    # unet.enable_xformers_memory_efficient_attention()
    unet.set_attention_backend("flash")

    # Freeze VAE (as in LBM)
    vae.requires_grad_(False)
    vae.eval()
    # text_encoder.requires_grad_(False)
    unet.train()

    # if training_args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

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

    # Remove discriminator and GAN-related components (not needed for LBM)
    # Only keep LPIPS for optional pixel loss if needed
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips_alex = lpips.LPIPS(net="alex").cuda()
    net_lpips.requires_grad_(False)
    net_lpips_alex.requires_grad_(False)

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

    # Initialize Discriminator and Optimizer
    if diffusion_args.discriminator_type == "latent":
        # Latent-space PatchGAN: operates on 128-channel VAE latents (32×32 for 512px)
        use_bn = diffusion_args.gan_loss_type != "r3gan"
        discriminator = NLayerDiscriminator(
            input_nc=128, ndf=64, n_layers=2, use_bn=use_bn
        ).to(accelerator.device)
    elif diffusion_args.discriminator_type == "r3gan":
        # R3GAN ResNet discriminator — 512→4 in 8 downsample stages
        # Lighter config for practical training speed
        width_per_stage = [16, 32, 64, 64, 128, 128, 128, 128]
        cardinality_per_stage = [1, 1, 2, 2, 2, 4, 4, 4]
        blocks_per_stage = [1, 1, 1, 1, 1, 1, 1, 1]
        discriminator = R3GANDiscriminator(
            width_per_stage=width_per_stage,
            cardinality_per_stage=cardinality_per_stage,
            blocks_per_stage=blocks_per_stage,
            expansion=2,
        ).to(accelerator.device)
    elif diffusion_args.discriminator_type == "patchgan":
        use_bn = diffusion_args.gan_loss_type != "r3gan"
        discriminator = NLayerDiscriminator(
            input_nc=3, ndf=128, n_layers=4, use_bn=use_bn
        ).to(accelerator.device)
    else:
        discriminator = PatchDiscriminator().to(accelerator.device)

    discriminator.requires_grad_(True)
    discriminator.train()

    # R3GAN uses beta1=0 (no momentum) for optimizer stability
    d_beta1 = (
        0.0 if diffusion_args.gan_loss_type == "r3gan" else training_args.adam_beta1
    )
    optimizer_D = torch.optim.AdamW(
        discriminator.parameters(),
        lr=diffusion_args.learning_rate_disc,
        betas=(d_beta1, training_args.adam_beta2),
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
        discriminator,
        optimizer_D,
    ) = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        vae,
        discriminator,
        optimizer_D,
    )

    # Initialize LeCam variables
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9

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

    # dummy_emb = dummy_emb.to(accelerator.device).to(weight_dtype)

    progress_bar = tqdm(
        range(0, training_args.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Helper function to get sigmas from scheduler (similar to LBM)
    def _get_sigmas(scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Helper function for timestep sampling (similar to LBM)
    def _timestep_sampling(n_samples=1, device="cpu"):
        if diffusion_args.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return noise_scheduler.timesteps[idx].to(device=device)
        elif diffusion_args.timestep_sampling == "custom_timesteps":
            # print("selected_timesteps_tensor", selected_timesteps_tensor)
            idx = np.random.choice(len(selected_timesteps_tensor), n_samples)

            return selected_timesteps_tensor[idx]

    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # l_acc = [unet, vae]
            l_acc = [unet, discriminator]
            with accelerator.accumulate(*l_acc):
                # Convert images to latent space (Bridge Matching approach)
                with torch.no_grad():
                    z_source = vae.encode(
                        batch["source_images"].to(weight_dtype),
                        return_dict=False,
                    )
                    # )[0].sample()
                    # z_source = vae.encode(
                    #     batch["source_images"].to(weight_dtype)
                    # ).latent
                    z_source = z_source * vae.config.scaling_factor

                    z_target = vae.encode(
                        batch["target_images"].to(weight_dtype),
                        return_dict=False,
                    )
                    # )[0].sample()
                    # z_target = vae.encode(
                    #     batch["target_images"].to(weight_dtype),
                    # ).latent
                    z_target = z_target * vae.config.scaling_factor

                # Sample timesteps (Bridge Matching)
                timesteps = _timestep_sampling()

                # --- БЛОК ACCUMULATED SELF-GENERATION ---
                perturbation_prob = (
                    diffusion_args.self_generation_prob
                    if global_step > diffusion_args.disc_warmup_steps
                    else 0.0
                )

                # Prepare conditioning latent (perturbed if requested)
                if diffusion_args.input_perturbation > 0:
                    z_source_cond = z_source + diffusion_args.input_perturbation * torch.randn_like(z_source)
                else:
                    z_source_cond = z_source

                if np.random.rand() < perturbation_prob:
                    # 1. Рассчитываем размер шага
                    step_size_int = (
                        1000 // diffusion_args.num_inference_steps
                    )  # 125 для 8 шагов

                    # 2. Определяем, на сколько шагов назад мы можем уйти
                    # Мы не можем уйти дальше 1000 (Source).
                    # (1000 - t) // step_size дает макс. кол-во шагов "вверх" по потоку.

                    # Используем view(-1) для корректной работы с тензорами
                    steps_available_up = (1000 - timesteps) // step_size_int

                    # Выбираем случайную глубину симуляции для каждого элемента батча или общую
                    # Для простоты и скорости выберем общую глубину для батча, но не больше доступного максимума
                    max_sim_steps = diffusion_args.self_generation_max_steps

                    # Нужно найти минимум среди батча, чтобы не выйти за 1000
                    min_available = steps_available_up.min().item()

                    if min_available < 1:
                        # Если мы уже в t=1000 или близко, мы не можем идти назад.
                        # Используем идеальный сэмпл (fallback)
                        n_steps_back = 0
                    else:
                        # Случайно выбираем от 1 до min(3, доступно)
                        limit = min(max_sim_steps, int(min_available))
                        # n_steps_back = np.random.randint(1, limit + 1)
                        n_steps_back = limit

                    if n_steps_back > 0:
                        # 3. Стартовое время симуляции (t_start)
                        t_start_int = timesteps + n_steps_back * step_size_int
                        t_start_int = t_start_int.clamp(max=1000)

                        # Получаем идеальную точку старта (на прямой)
                        sigmas_start = _get_sigmas(
                            noise_scheduler,
                            t_start_int,
                            n_dim=4,
                            dtype=weight_dtype,
                            device=z_source.device,
                        )

                        # current_sim_sample - это "шарик", который мы будем катить вниз
                        current_sim_sample = (
                            sigmas_start * z_source + (1.0 - sigmas_start) * z_target
                        )

                        # 4. ЦИКЛ СИМУЛЯЦИИ (накапливаем ошибку)
                        # Мы идем от t_start вниз к t_target (timesteps)

                        curr_t = t_start_int

                        # with torch.no_grad():
                        if True:
                            for _ in range(n_steps_back):
                                # a. Предсказываем скорость в текущей точке
                                model_input_temp = torch.cat(
                                    [current_sim_sample, z_source_cond],
                                    dim=1,
                                )
                                pred_velocity = unet(
                                    model_input_temp,
                                    curr_t,
                                    return_dict=False,
                                )[0]

                                # b. Определяем следующее время (на шаг вниз)
                                next_t = curr_t - step_size_int

                                # Получаем сигмы для шага
                                s_curr = _get_sigmas(
                                    noise_scheduler,
                                    curr_t,
                                    n_dim=4,
                                    dtype=weight_dtype,
                                    device=z_source.device,
                                )
                                s_next = _get_sigmas(
                                    noise_scheduler,
                                    next_t,
                                    n_dim=4,
                                    dtype=weight_dtype,
                                    device=z_source.device,
                                )
                                dt_sigma = s_next - s_curr  # Отрицательное число

                                # c. Делаем шаг Эйлера
                                current_sim_sample = (
                                    current_sim_sample + pred_velocity * dt_sigma
                                )

                                # d. Добавляем Bridge Noise (SDE шаг)
                                if diffusion_args.bridge_noise_sigma > 0:
                                    noise = torch.randn_like(current_sim_sample)
                                    bridge_factor = (s_next * (1.0 - s_next)).sqrt()
                                    current_sim_sample = (
                                        current_sim_sample
                                        + diffusion_args.bridge_noise_sigma
                                        * bridge_factor
                                        * noise
                                    )

                                # Обновляем время для следующей итерации
                                curr_t = next_t

                        # После цикла current_sim_sample находится во времени timesteps,
                        # но он пришел туда "своим ходом" через 1-3 шага, накопив кривизну.
                        noisy_sample = current_sim_sample

                        # Sigmas нужны для финального расчета лосса (соответствуют времени timesteps)
                        sigmas = _get_sigmas(
                            noise_scheduler,
                            timesteps,
                            n_dim=4,
                            dtype=weight_dtype,
                            device=z_source.device,
                        )

                    else:
                        # Fallback (если t=1000)
                        sigmas = _get_sigmas(
                            noise_scheduler,
                            timesteps,
                            n_dim=4,
                            dtype=weight_dtype,
                            device=z_source.device,
                        )
                        noisy_sample = sigmas * z_source + (1.0 - sigmas) * z_target

                else:
                    # --- СТАНДАРТНАЯ ВЕТКА ---
                    sigmas = _get_sigmas(
                        noise_scheduler,
                        timesteps,
                        n_dim=4,
                        dtype=weight_dtype,
                        device=z_source.device,
                    )

                    noisy_sample = (
                        sigmas * z_source
                        + (1.0 - sigmas) * z_target
                        + diffusion_args.bridge_noise_sigma
                        * (sigmas * (1.0 - sigmas)) ** 0.5
                        * torch.randn_like(z_source)
                    )

                # --- ДАЛЬШЕ БЕЗ ИЗМЕНЕНИЙ ---

                # Get sigmas for the timesteps
                # sigmas = _get_sigmas(
                #     noise_scheduler,
                #     timesteps,
                #     n_dim=4,
                #     dtype=weight_dtype,
                #     device=z_source.device,
                # )
                # # print(sigmas)
                # # Create interpolant (Bridge between z_source and z_target)
                # noisy_sample = (
                #     sigmas * z_source
                #     + (1.0 - sigmas) * z_target
                #     + diffusion_args.bridge_noise_sigma
                #     * (sigmas * (1.0 - sigmas)) ** 0.5
                #     * torch.randn_like(z_source)
                # )
                # noisy_sample = z_source

                # Ensure first timestep uses z_source
                for i, t in enumerate(timesteps):
                    if t.item() == noise_scheduler.timesteps[0]:
                        noisy_sample[i] = z_source[i]

                # Predict direction of transport (target = z_source - z_target)
                model_input = torch.cat([noisy_sample, z_source_cond], dim=1)

                # D step: no grad needed through generator
                with torch.no_grad():
                    model_pred = unet(
                        model_input,
                        timesteps,
                        return_dict=False,
                    )[0]
                    denoised_sample = noisy_sample - model_pred * sigmas
                    denoised_sample = vae.decode(
                        denoised_sample / vae.config.scaling_factor,
                        return_dict=False,
                    ).clamp(-1, 1)
                    # )[0].clamp(-1, 1)

                # --- Discriminator Step ---
                is_latent_disc = diffusion_args.discriminator_type == "latent"
                if is_latent_disc:
                    real_for_d = z_target.detach().float()
                    fake_for_d = (noisy_sample - model_pred * sigmas).detach().float()
                else:
                    with torch.no_grad():
                        reconstructed_targets = vae.decode(
                            z_target / vae.config.scaling_factor,
                            return_dict=False,
                        ).clamp(-1, 1)
                    real_for_d = reconstructed_targets.detach().float()
                    fake_for_d = denoised_sample.detach()

                # Instance noise: blur real/fake boundary, decay over training
                noise_std = diffusion_args.d_noise_std * max(
                    0.0, 1.0 - global_step / diffusion_args.d_noise_anneal_steps
                )
                if noise_std > 0:
                    real_for_d = real_for_d + noise_std * torch.randn_like(real_for_d)
                    fake_for_d = fake_for_d + noise_std * torch.randn_like(fake_for_d)

                # Train D every N steps to give G more time to adapt
                train_d_this_step = global_step % diffusion_args.d_train_every == 0

                if diffusion_args.gan_loss_type == "r3gan":
                    real_gp = real_for_d.detach().requires_grad_(True)
                    fake_gp = fake_for_d.detach().requires_grad_(True)

                    real_preds = discriminator(real_gp)
                    fake_preds = discriminator(fake_gp)

                    # Relativistic loss: D wants real > fake
                    relativistic_logits = real_preds - fake_preds
                    d_adv_loss = F.softplus(-relativistic_logits).mean()
                    
                    # R1 + R2 gradient penalty (per step - NO LAZY REG)
                    r1_penalty = zero_centered_gradient_penalty(real_gp, real_preds)
                    r2_penalty = zero_centered_gradient_penalty(fake_gp, fake_preds)
                    reg_loss = (diffusion_args.r3gan_gamma / 2) * (r1_penalty.mean() + r2_penalty.mean())
                    d_loss = d_adv_loss + reg_loss
                        
                    with torch.no_grad():
                        acc = (relativistic_logits > 0).float().mean().item()
                elif diffusion_args.gan_loss_type == "hinge":
                    real_preds = discriminator(real_for_d)
                    fake_preds = discriminator(fake_for_d)
                    d_loss = hinge_d_loss(real_preds, fake_preds)
                    with torch.no_grad():
                        avg_real_preds = real_preds.mean().item()
                        avg_fake_preds = fake_preds.mean().item()
                        acc = (real_preds > 0).sum().item() + (
                            fake_preds < 0
                        ).sum().item()
                        acc = acc / (real_preds.numel() + fake_preds.numel())
                else:
                    real_preds = discriminator(real_for_d)
                    fake_preds = discriminator(fake_for_d)
                    d_loss, avg_real_preds, avg_fake_preds, acc = gan_disc_loss(
                        real_preds,
                        fake_preds,
                        disc_type="bce",
                    )

                # LeCam Logic (skip for r3gan — R1/R2 replaces it)
                if diffusion_args.use_lecam and diffusion_args.gan_loss_type != "r3gan":
                    lecam_anchor_real_logits = (
                        lecam_beta * lecam_anchor_real_logits
                        + (1 - lecam_beta) * avg_real_preds
                    )
                    lecam_anchor_fake_logits = (
                        lecam_beta * lecam_anchor_fake_logits
                        + (1 - lecam_beta) * avg_fake_preds
                    )

                    lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(
                        2
                    ).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()

                    d_loss = d_loss + lecam_loss * diffusion_args.lecam_loss_weight
                else:
                    lecam_loss = torch.tensor(0.0)

                # Initialize gradient norm for logging
                _d_grad_norm = 0.0

                if train_d_this_step:
                    accelerator.backward(d_loss)

                    # Measure gradient norms BEFORE zero_grad
                    for p in discriminator.parameters():
                        if p.grad is not None:
                            _d_grad_norm += p.grad.data.norm(2).item() ** 2
                    _d_grad_norm = _d_grad_norm**0.5

                    optimizer_D.step()
                    optimizer_D.zero_grad()

                # --- Generator Step ---
                # Recompute forward pass to get a fresh computation graph
                # (the previous graph was freed by accelerator.backward(d_loss))
                model_pred_g = unet(
                    model_input,
                    timesteps,
                    return_dict=False,
                )[0]
                denoised_sample_g = noisy_sample - model_pred_g * sigmas
                denoised_sample_g = vae.decode(
                    denoised_sample_g / vae.config.scaling_factor,
                    return_dict=False,
                ).clamp(-1, 1)

                loss_lpips = net_lpips(
                    denoised_sample_g, batch["target_images"].float().detach()
                ).mean()

                # Prevent gradients from flowing into discriminator during G update
                for p in discriminator.parameters():
                    p.requires_grad = False

                # GAN loss with warmup: skip adversarial signal early in training
                if global_step >= diffusion_args.disc_warmup_steps:
                    # For latent D: pass latent directly
                    if is_latent_disc:
                        z_pred_g = noisy_sample - model_pred_g * sigmas
                        d_input_g = z_pred_g
                        d_input_real = z_target.detach()
                    else:
                        d_input_g = denoised_sample_g
                        d_input_real = reconstructed_targets.detach()

                    if diffusion_args.gan_loss_type == "r3gan":
                        fake_preds_for_g = discriminator(d_input_g)
                        real_preds_for_g = discriminator(d_input_real)
                        relativistic_logits_g = fake_preds_for_g - real_preds_for_g
                        g_gan_loss = F.softplus(-relativistic_logits_g).mean()
                    elif diffusion_args.gan_loss_type == "hinge":
                        fake_preds_for_g = discriminator(d_input_g)
                        g_gan_loss = -torch.mean(fake_preds_for_g)
                    else:
                        if is_latent_disc:
                            fake_preds_for_g = discriminator(d_input_g)
                        else:
                            denoised_for_disc = gradnorm(denoised_sample_g, weight=1.0)
                            fake_preds_for_g = discriminator(denoised_for_disc)
                        g_gan_loss = F.binary_cross_entropy_with_logits(
                            fake_preds_for_g, torch.ones_like(fake_preds_for_g)
                        )
                else:
                    g_gan_loss = torch.tensor(0.0, device=denoised_sample_g.device)

                loss = (
                    loss_lpips * diffusion_args.lpips_factor
                    + g_gan_loss * diffusion_args.gan_factor
                )

                accelerator.backward(loss)

                for p in discriminator.parameters():
                    p.requires_grad = True

                # Measure gradient norms BEFORE zero_grad
                _unet_grad_norm = 0.0
                for p in unet.parameters():
                    if p.grad is not None:
                        _unet_grad_norm += p.grad.data.norm(2).item() ** 2
                _unet_grad_norm = _unet_grad_norm**0.5

                # _d_grad_norm is calculated earlier
                # _d_grad_norm = 0.0
                # for p in discriminator.parameters():
                #     if p.grad is not None:
                #         _d_grad_norm += p.grad.data.norm(2).item() ** 2
                # _d_grad_norm = _d_grad_norm ** 0.5

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {}
                # log the loss
                logs["loss"] = loss.detach().item()
                # logs["latent_loss"] = loss.detach().item()
                logs["loss_lpips"] = loss_lpips.detach().item()

                logs["d_loss"] = d_loss.detach().item()
                logs["g_gan_loss"] = g_gan_loss.detach().item()
                logs["d_acc"] = acc

                # Diagnostic: pixel-space distance between real and fake for D
                with torch.no_grad():
                    logs["real_fake_l2"] = (
                        (real_for_d - fake_for_d).pow(2).mean().item()
                    )

                # Diagnostic: gradient norms (measured before zero_grad)
                logs["unet_grad_norm"] = _unet_grad_norm
                logs["d_grad_norm"] = _d_grad_norm
                if diffusion_args.gan_loss_type == "r3gan":
                    logs["r1_penalty"] = r1_penalty.mean().detach().item()
                    logs["r2_penalty"] = r2_penalty.mean().detach().item()
                if diffusion_args.use_lecam and diffusion_args.gan_loss_type != "r3gan":
                    logs["lecam_loss"] = lecam_loss.detach().item()

                # logs["mse_loss"] = mse_loss.detach().item()
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if global_step % training_args.save_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if training_args.save_total_limit is not None:
                            checkpoints = os.listdir(training_args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
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
                        # accelerator.save_state(save_path)
                        # Сохраняем UNet
                        unwrap_model(unet).save_pretrained(
                            os.path.join(save_path, "unet")
                        )
                        # VAE не сохраняем, так как он заморожен и используется предобученный
                        logger.info(f"Saved state to {save_path}")

                        # start validation
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
