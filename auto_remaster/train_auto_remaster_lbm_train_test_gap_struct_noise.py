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
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


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
    minimal_noise_r: float = field(default=20.0)


unet2d_config = {
    "sample_size": 64,
    # "in_channels": 4,
    # "in_channels": 16,
    # "in_channels": 32,
    "in_channels": 32 * 2,
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


def create_frequency_soft_cutoff_mask(
    height: int,
    width: int,
    cutoff_radius: float,
    transition_width: float = 5.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a smooth frequency cutoff mask for low-pass filtering.

    Args:
        height: Image height
        width: Image width
        cutoff_radius: Frequency cutoff radius (0 = no structure, max_radius = full structure)
        transition_width: Width of smooth transition (smaller = sharper cutoff)
        device: Device to create tensor on

    Returns:
        torch.Tensor: Frequency mask of shape (height, width)
    """
    if device is None:
        device = torch.device("cpu")

    # Create frequency coordinates
    u = torch.arange(height, device=device)
    v = torch.arange(width, device=device)
    u, v = torch.meshgrid(u, v, indexing="ij")

    # Calculate distance from center
    center_u, center_v = height // 2, width // 2
    frequency_radius = torch.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)

    # Create smooth transition mask
    mask = torch.exp(
        -((frequency_radius - cutoff_radius) ** 2) / (2 * transition_width**2)
    )
    mask = torch.where(frequency_radius <= cutoff_radius, torch.ones_like(mask), mask)

    return mask


def clip_frequency_magnitude(noise_magnitudes, clip_percentile=0.95):
    """Clip frequency domain magnitude to prevent large values."""

    # Calculate clipping threshold
    clip_threshold = torch.quantile(noise_magnitudes, clip_percentile)

    # Clip large values
    clipped_magnitudes = torch.clamp(noise_magnitudes, max=clip_threshold)

    return clipped_magnitudes


def generate_structured_noise_batch_vectorized(
    image_batch: torch.Tensor,
    noise_std: float = 1.0,
    pad_factor: float = 1.5,
    cutoff_radius: float = None,
    transition_width: float = 2.0,
    input_noise: torch.Tensor = None,
    sampling_method: str = "fft",
) -> torch.Tensor:
    """
    Generate structured noise for a batch of images using frequency soft cutoff.
    Reduces boundary artifacts by padding images before FFT processing.

    Args:
        image_batch: Batch of image tensors of shape (B, C, H, W)
        noise_std: Standard deviation for Gaussian noise
        pad_factor: Padding factor (1.5 = 50% padding, 2.0 = 100% padding)
        cutoff_radius: Frequency cutoff radius (None = auto-calculate)
        transition_width: Width of smooth transition for frequency cutoff
        input_noise: Optional input noise tensor to use instead of generating new noise.
        sampling_method: Method to sample noise magnitude ('fft', 'cdf', 'two-gaussian')

    Returns:
        torch.Tensor: Batch of structured noise tensors of shape (B, C, H, W)
    """
    assert sampling_method in ["fft", "cdf", "two-gaussian"]
    # Ensure tensor is on the correct device
    batch_size, channels, height, width = image_batch.shape
    dtype = image_batch.dtype
    device = image_batch.device
    image_batch = image_batch.float()

    # Calculate padding size for overlap-add method
    pad_h = int(height * (pad_factor - 1))
    pad_h = pad_h // 2 * 2  # make it even
    pad_w = int(width * (pad_factor - 1))
    pad_w = pad_w // 2 * 2  # make it even

    # Pad images with reflection to reduce boundary artifacts
    padded_images = torch.nn.functional.pad(
        image_batch,
        (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
        mode="reflect",  # Mirror edges for natural transitions
    )

    # Calculate padded dimensions
    padded_height = height + pad_h
    padded_width = width + pad_w

    # Create frequency soft cutoff mask only if cutoff_radius is provided
    if cutoff_radius is not None:
        cutoff_radius = min(min(padded_height / 2, padded_width / 2), cutoff_radius)
        freq_mask = create_frequency_soft_cutoff_mask(
            padded_height, padded_width, cutoff_radius, transition_width, device
        )
    else:
        # No cutoff - preserve all frequencies (full structure preservation)
        freq_mask = torch.ones(padded_height, padded_width, device=device)

    # Apply 2D FFT to padded images
    fft = torch.fft.fft2(padded_images, dim=(-2, -1))

    # Shift zero frequency to center
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

    # Extract phase and magnitude for all images
    image_phases = torch.angle(fft_shifted)
    image_phases = clip_frequency_magnitude(image_phases)
    image_magnitudes = torch.abs(fft_shifted)

    if input_noise is not None:
        # Use provided noise
        noise_batch = torch.nn.functional.pad(
            input_noise,
            (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2),
            mode="reflect",  # Mirror edges for natural transitions
        )
        noise_batch = noise_batch.float()
    else:
        # Generate Gaussian noise for the padded size
        noise_batch = torch.randn_like(padded_images)

    # Extract noise magnitude and phase
    if sampling_method == "fft":
        # Apply 2D FFT to noise batch
        noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
        noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

        noise_magnitudes = torch.abs(noise_fft_shifted)
        noise_phases = torch.angle(noise_fft_shifted)
    elif sampling_method == "cdf":
        # The magnitude of FFT of Gaussian noise follows a Rayleigh distribution.
        # We can sample it directly.
        # The scale of the Rayleigh distribution is related to the std of the Gaussian noise
        # and the size of the FFT.
        # For an N-point FFT of Gaussian noise with variance sigma^2, the variance of
        # the real and imaginary parts of the FFT coefficients is N*sigma^2.
        # The scale parameter for the Rayleigh distribution is sqrt(N*sigma^2 / 2).
        # Here, N = padded_height * padded_width.

        N = padded_height * padded_width
        rayleigh_scale = (N / 2) ** 0.5

        ## Sample from a standard Rayleigh distribution (scale=1) and then scale it.
        uu = torch.rand(size=image_magnitudes.shape, device=device)
        noise_magnitudes = rayleigh_scale * torch.sqrt(-2.0 * torch.log(uu))
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = (
                torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi
                - torch.pi
            )
    elif sampling_method == "two-gaussian":
        N = padded_height * padded_width
        rayleigh_scale = (N / 2) ** 0.5
        # A standard Rayleigh can be generated from two standard normal distributions.
        u1 = torch.randn_like(image_magnitudes)
        u2 = torch.randn_like(image_magnitudes)
        noise_magnitudes = rayleigh_scale * torch.sqrt(u1**2 + u2**2)
        if input_noise is not None:
            noise_fft = torch.fft.fft2(noise_batch, dim=(-2, -1))
            noise_fft_shifted = torch.fft.fftshift(noise_fft, dim=(-2, -1))

            noise_magnitudes = torch.abs(noise_fft_shifted)
            noise_phases = torch.angle(noise_fft_shifted)
        else:
            noise_phases = (
                torch.rand(size=image_magnitudes.shape, device=device) * 2 * torch.pi
                - torch.pi
            )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    noise_magnitudes = clip_frequency_magnitude(noise_magnitudes)

    # Scale noise magnitude by standard deviation
    noise_magnitudes = noise_magnitudes * noise_std

    # Apply frequency soft cutoff to mix phases
    # Low frequencies (within cutoff) use image phase, high frequencies use noise phase
    mixed_phases = (
        freq_mask.unsqueeze(0).unsqueeze(0) * image_phases
        + (1 - freq_mask.unsqueeze(0).unsqueeze(0)) * noise_phases
    )

    # Combine magnitude and mixed phase for all images
    fft_combined = noise_magnitudes * torch.exp(1j * mixed_phases)
    # Shift zero frequency back to corner
    fft_unshifted = torch.fft.ifftshift(fft_combined, dim=(-2, -1))
    # Apply inverse FFT
    structured_noise_padded = torch.fft.ifft2(fft_unshifted, dim=(-2, -1))
    # Take real part
    structured_noise_padded = torch.real(structured_noise_padded)

    clamp_mask = (structured_noise_padded < -5) + (structured_noise_padded > 5)
    clamp_mask = (clamp_mask > 0).float()

    structured_noise_padded = (
        structured_noise_padded * (1 - clamp_mask) + noise_batch * clamp_mask
    )

    # Crop back to original size (remove padding)
    structured_noise_batch = structured_noise_padded[
        :, :, pad_h // 2 : pad_h // 2 + height, pad_w // 2 : pad_w // 2 + width
    ]
    return structured_noise_batch.to(dtype)


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
    # vae_val = AutoencoderTiny.from_pretrained(
    #     "madebyollin/taesd",
    #     torch_device="cuda",
    #     torch_dtype=weight_dtype,
    # ).to(accelerator.device)
    # vae_val.decoder.ignore_skip = False
    vae_val = AutoencoderKL.from_pretrained(
        # "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        torch_device="cuda",
    ).to(accelerator.device)
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
    num_steps = diffusion_args.num_inference_steps

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
                # vae_val.encode(c_t, return_dict=False)[0]
                vae_val.encode(c_t, return_dict=False)[0].sample()
                * vae_val.config.scaling_factor
            )
            structured_noise = generate_structured_noise_batch_vectorized(
                z_source.float(),  # float обязателен для FFT
                noise_std=1.0,
                pad_factor=1.5,
                cutoff_radius=diffusion_args.minimal_noise_r,  # Фиксированный радиус для валидации
                input_noise=torch.randn_like(z_source.float()),
                sampling_method="fft",
            ).to(dtype=z_source.dtype, device=z_source.device)

            sample = z_source + diffusion_args.bridge_noise_sigma * structured_noise

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
                denoiser_input = torch.cat([denoiser_input, z_source], dim=1)
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
                    # noise = torch.randn_like(sample)
                    structured_noise = generate_structured_noise_batch_vectorized(
                        z_source.float(),  # float обязателен для FFT
                        noise_std=1.0,
                        pad_factor=1.5,
                        cutoff_radius=diffusion_args.minimal_noise_r,  # Фиксированный радиус для валидации
                        input_noise=torch.randn_like(z_source.float()),
                        sampling_method="fft",
                    ).to(dtype=sample.dtype, device=sample.device)
                    noise = structured_noise
                    bridge_factor = (sigmas_next * (1.0 - sigmas_next)) ** 0.5

                    sample = (
                        sample
                        + diffusion_args.bridge_noise_sigma * bridge_factor * noise
                    )
                    sample = sample.to(z_source.dtype)

            # ---------------------------------------------------------

            # Декодирование результата
            output_image = (
                vae_val.decode(
                    sample / vae_val.config.scaling_factor,
                    return_dict=False,
                )[0]
            ).clamp(-1, 1)

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
                    # 6.0 взято рандомно, исходя из того что у нас не сильно меняется картинка
                    # всегда и там не так много креатива нужно
                    cutoff_radius = 6.0 + np.random.exponential(scale=1 / 0.1)
                    structured_noise = generate_structured_noise_batch_vectorized(
                        z_source.float(),
                        cutoff_radius=cutoff_radius,
                        input_noise=torch.randn_like(z_source.float()),
                    )

                # Sample timesteps (Bridge Matching)
                timesteps = _timestep_sampling()

                # --- БЛОК ACCUMULATED SELF-GENERATION ---

                # Включаем после разогрева
                # perturbation_prob = 0.5 if global_step > 2000 else 0.0
                perturbation_prob = 0.5

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
                    # Ограничим max_chain, чтобы не замедлять обучение слишком сильно (например, макс 3 шага)
                    max_sim_steps = 4

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

                        with torch.no_grad():
                            for _ in range(n_steps_back):
                                # a. Предсказываем скорость в текущей точке
                                model_input_temp = torch.cat(
                                    [current_sim_sample, z_source],
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
                                # current_sim_sample = (
                                #     current_sim_sample + pred_velocity * dt_sigma
                                # )
                                current_sim_sample = (
                                    current_sim_sample
                                    + pred_velocity * dt_sigma
                                    #
                                )

                                # Обновляем время для следующей итерации
                                curr_t = next_t

                        # После цикла current_sim_sample находится во времени timesteps,
                        # но он пришел туда "своим ходом" через 1-3 шага, накопив кривизну.
                        noisy_sample = (
                            current_sim_sample
                            + diffusion_args.bridge_noise_sigma
                            * (s_next * (1.0 - s_next)) ** 0.5
                            * structured_noise
                        )

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
                        * structured_noise
                    )

                # Ensure first timestep uses z_source
                # for i, t in enumerate(timesteps):
                #     if t.item() == noise_scheduler.timesteps[0]:
                #         noisy_sample[i] = (
                #             z_source[i]
                #             + diffusion_args.bridge_noise_sigma * structured_noise
                #         )

                # Predict direction of transport (target = z_source - z_target)
                model_input = torch.cat([noisy_sample, z_source], dim=1)
                # print(noisy_sample.shape, z_source.shape)
                model_pred = unet(
                    model_input,
                    timesteps,
                    return_dict=False,
                )[0]

                # Target is the direction from z_source to z_target
                target = z_source - z_target

                # Compute loss in latent space
                if diffusion_args.latent_loss_type == "l2":
                    latent_loss = F.mse_loss(
                        model_pred,
                        target.detach(),
                        reduction="mean",
                    )
                elif diffusion_args.latent_loss_type == "l1":
                    latent_loss = F.l1_loss(
                        model_pred,
                        target.detach(),
                        reduction="mean",
                    )
                else:
                    raise ValueError(
                        f"Unknown latent_loss_type: {diffusion_args.latent_loss_type}"
                    )

                denoised_sample = noisy_sample - model_pred * sigmas
                denoised_sample = vae.decode(
                    denoised_sample / vae.config.scaling_factor,
                    return_dict=False,
                )[0].clamp(-1, 1)

                x_tgt = batch["target_images"].float()
                loss_lpips = net_lpips(denoised_sample, x_tgt.to(weight_dtype)).mean()
                loss_lpips_alex = net_lpips_alex(
                    denoised_sample, x_tgt.to(weight_dtype)
                ).mean()
                # pixel_loss = F.l1_loss(
                #     denoised_sample.float(),
                #     x_tgt.to(weight_dtype),
                #     reduction="mean",
                # )

                loss = (
                    # latent_loss * diffusion_args.latent_loss_weight
                    # + loss_lpips * diffusion_args.lpips_factor
                    # + pixel_loss
                    loss_lpips * diffusion_args.lpips_factor
                    + loss_lpips_alex * diffusion_args.lpips_factor
                )

                accelerator.backward(loss)
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
                logs["latent_loss"] = latent_loss.detach().item()
                logs["loss_lpips"] = loss_lpips.detach().item()
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
