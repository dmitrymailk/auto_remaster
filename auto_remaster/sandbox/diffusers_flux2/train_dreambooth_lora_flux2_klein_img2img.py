import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from datasets import load_dataset

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    AutoModel,
    AutoencoderKLFlux2,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    wrap_with_fsdp,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")


logger = get_logger(__name__)


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Flux2TinyAutoEncoder(AutoencoderKLFlux2):
    def __init__(self):
        super().__init__()
        self.vae = AutoModel.from_pretrained(
            "fal/FLUX.2-Tiny-AutoEncoder",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        # это ломает генерацию при помощи пайплайна
        # self._internal_dict = self.vae.config

    def encode(self, x):
        x = torch.nn.functional.pixel_shuffle(self.vae.encode(x).latent, 2)
        return DotDict(latent_dist=DotDict(mode=lambda: x))

    def decode(self, x, return_dict=True):
        if return_dict:
            return DotDict(
                sample=self.vae.decode(
                    torch.nn.functional.pixel_unshuffle(x, 2)
                ).sample.unsqueeze(0)
            )
        return self.vae.decode(
            torch.nn.functional.pixel_unshuffle(x, 2)
        ).sample.unsqueeze(0)


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
    pipeline,
    args,
    accelerator,
    dataset,
    epoch,
    torch_dtype,
    prompt_embeds=None,
    text_ids=None,
    is_final_validation=False,
):
    logger.info("Running validation...")
    # pipeline = pipeline.to(dtype=torch_dtype)
    # vae_dtype = torch.bfloat16
    # pipeline.vae.to(dtype=vae_dtype)
    # pipeline.enable_model_cpu_offload()
    # pipeline.set_progress_bar_config(disable=True)

    # run inference
    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    # Ensure validation uses the same structural noise logic if enabled
    structural_noise_radius = getattr(args, "structural_noise_radius", None)

    autocast_ctx = (
        torch.autocast(accelerator.device.type)
        if not is_final_validation
        else nullcontext()
    )

    # Select 5 random images deterministically
    num_samples = 15
    indices = range(len(dataset))
    num_samples = min(num_samples, len(dataset))

    # Use a local Random instance to ensure the same validation images are selected every time
    # if a seed is provided.
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    selected_indices = rng.sample(indices, num_samples)

    images = []

    # Validation loop
    for idx in selected_indices:
        example = dataset[idx]

        # Get control image and prompt
        cond_image_column = args.cond_image_column
        # caption_column = args.caption_column # Not used, we use single prompt
        image_column = args.image_column

        control_image = example[cond_image_column]
        # prompt = example[caption_column] # Ignored
        target_image = example[image_column]

        # Convert to RGB to ensure 3 channels
        if not control_image.mode == "RGB":
            control_image = control_image.convert("RGB")
        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")

        # Helper validation transform
        validation_transform = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(args.resolution),
            ]
        )

        # Resize and CenterCrop to match training logic exactly using standard Compose
        control_image = validation_transform(control_image)
        target_image = validation_transform(target_image)

        with autocast_ctx:
            # Generate structural noise if required
            latents = None
            if structural_noise_radius is not None and structural_noise_radius > 0:
                # Encode control/source image for structural noise reference
                # We need to replicate what happens in training: VAE encode -> Structure Noise
                # We assume pipeline.vae is available
                with torch.no_grad():
                    # Transform control image to tensor [-1, 1]
                    # Note: validation_transform creates PIL, need to convert to tensor
                    # But wait, validation_transform calls transforms.CenterCrop which returns PIL
                    # Then we immediately pass it to pipeline? No, we need latents.
                    # We can use `control_image` (which is already valid_transformed PIL)
                    pass

                # Need to match VAE device/dtype (VAE might be on CPU due to offloading)
                img_tensor = transforms.ToTensor()(control_image)
                img_tensor = (img_tensor - 0.5) / 0.5
                img_tensor = img_tensor.unsqueeze(0)
                # Use strict vae_dtype (float16) and accelerator device (GPU)
                # pipeline.vae.device might be CPU if offloaded, but forward pass happens on GPU
                img_tensor = img_tensor.to(device=accelerator.device, dtype=torch_dtype)

                # Encode
                z_source = pipeline.vae.encode(img_tensor).latent_dist.mode()

                # Generate noise (perform on GPU for speed, usually)
                # Cast back to torch_dtype (bf16) for structure noise and pipeline
                z_source = z_source.to(device=accelerator.device, dtype=torch_dtype)

                # Generate noise
                noise_spatial = generate_structured_noise_batch_vectorized(
                    z_source,
                    cutoff_radius=structural_noise_radius,
                ).to(dtype=torch_dtype, device=accelerator.device)
                # noise_spatial = torch.randn_like(z_source).to(
                #     dtype=torch_dtype, device=accelerator.device
                # )

                # Patchify latents to match pipeline expectations
                latents = pipeline._patchify_latents(noise_spatial)

            # We strictly use the pre-computed prompt embeddings
            image = pipeline(
                prompt_embeds=prompt_embeds,
                image=control_image,
                latents=latents,
                num_inference_steps=30,  # Use reasonable steps for validation
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]

            # Combine [Original (Condition) | Generated | Target]
            w, h = image.size
            if target_image.size != (w, h):
                target_image = target_image.resize((w, h))

            combined = Image.new("RGB", (w * 3, h))
            combined.paste(control_image, (0, 0))
            combined.paste(image, (w, 0))
            combined.paste(target_image, (w * 2, 0))

            images.append(combined)

    # Generalization Validation
    logger.info("Running generalization validation on test dataset")
    gen_images = []

    generalization_dataset_name = "dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw"

    # Load generalization dataset on the fly
    gen_dataset = load_dataset(
        generalization_dataset_name,
        split="train",
        cache_dir="/code/dataset/" + generalization_dataset_name.split("/")[-1],
    )

    # Select 5 random images deterministically (using same seed logic)
    gen_indices = range(len(gen_dataset))
    gen_num_samples = min(num_samples, len(gen_dataset))
    gen_selected_indices = rng.sample(gen_indices, gen_num_samples)

    for idx in gen_selected_indices:
        example = gen_dataset[idx]

        # Assuming same column names: input_image, edited_image
        control_image = example[args.cond_image_column]
        target_image = example[args.image_column]

        # Convert to RGB to ensure 3 channels
        if not control_image.mode == "RGB":
            control_image = control_image.convert("RGB")
        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")

        # Reuse same transform
        control_image = validation_transform(control_image)
        target_image = validation_transform(target_image)

        with autocast_ctx:
            # Generate structural noise if required (Same logic as above)
            latents = None
            if structural_noise_radius is not None and structural_noise_radius > 0:
                img_tensor = transforms.ToTensor()(control_image)
                img_tensor = (img_tensor - 0.5) / 0.5
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor.to(device=accelerator.device, dtype=torch_dtype)

                z_source = pipeline.vae.encode(img_tensor).latent_dist.mode()
                z_source = z_source.to(device=accelerator.device, dtype=torch_dtype)

                noise_spatial = generate_structured_noise_batch_vectorized(
                    z_source,
                    cutoff_radius=structural_noise_radius,
                ).to(dtype=torch_dtype, device=accelerator.device)
                # noise_spatial = torch.randn_like(z_source).to(
                #     dtype=torch_dtype, device=accelerator.device
                # )

                # Patchify latents
                latents = pipeline._patchify_latents(noise_spatial)

            image = pipeline(
                prompt_embeds=prompt_embeds,
                image=control_image,
                latents=latents,
                num_inference_steps=30,
                generator=generator,
                guidance_scale=args.guidance_scale,
            ).images[0]

            w, h = image.size
            if target_image.size != (w, h):
                target_image = target_image.resize((w, h))

            combined = Image.new("RGB", (w * 3, h))
            combined.paste(control_image, (0, 0))
            combined.paste(image, (w, 0))
            combined.paste(target_image, (w * 2, 0))

            gen_images.append(combined)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if tracker.name == "wandb":
            log_dict = {
                phase_name: [
                    wandb.Image(image, caption=f"Val Epoch {epoch}")
                    for i, image in enumerate(images)
                ]
            }
            if gen_images:
                log_dict["generalization"] = [
                    wandb.Image(image, caption=f"Gen Val Epoch {epoch}")
                    for i, image in enumerate(gen_images)
                ]
            tracker.log(log_dict)

    del pipeline
    free_memory()

    return images


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default=None,
        help="Column in the dataset containing the condition image. Must be specified when performing I2I fine-tuning",
    )
    parser.add_argument(
        "--structural_noise_radius",
        type=float,
        default=None,
        help="Radius for structural noise soft cutoff. If not set, standard Gaussian noise is used.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=250,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Enabla Flash Attention for NPU",
    )
    parser.add_argument(
        "--fsdp_text_encoder", action="store_true", help="Use FSDP for text encoder"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.cond_image_column is None:
        raise ValueError(
            "you must provide --cond_image_column for image-to-image training. Otherwise please see Flux2 text-to-image training example."
        )
    else:
        assert args.image_column is not None

    if args.dataset_name is None:
        raise ValueError("Specify `--dataset_name`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    use_normal_vae = True
    if use_normal_vae:
        vae = AutoencoderKLFlux2.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
            # torch_dtype=weight_dtype,
        )
    else:
        vae = Flux2TinyAutoEncoder().to(dtype=weight_dtype)

    if use_normal_vae:
        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(accelerator.device)

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        quantization_config=quantization_config,
        torch_dtype=weight_dtype,
    )

    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

    transformer.enable_xformers_memory_efficient_attention(
        attention_op=MemoryEfficientAttentionFlashAttentionOp
    )
    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(
            transformer, use_gradient_checkpointing=False
        )

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if not use_normal_vae:
        vae.vae.requires_grad_(False)

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = (
        {"dtype": weight_dtype, "device": accelerator.device}
        if not args.offload
        else {"dtype": weight_dtype}
    )
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer,
            module_filter_fn=module_filter_fn,
            config=Float8LinearConfig(pad_inner_dim=True),
        )

    # Text encoder init on CPU
    text_encoder.to(dtype=weight_dtype)
    text_encoder.to("cpu")
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    print("*****")
    print("*****")
    print("*****")
    print(args.lora_layers)
    print("*****")
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        transformer_cls = type(unwrap_model(transformer))

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        state_dict = accelerator.get_state_dict(model) if is_fsdp else None

        # 3) Only main process materializes the LoRA state dict
        transformer_lora_layers_to_save = None
        if accelerator.is_main_process:
            peft_kwargs = {}
            if is_fsdp:
                peft_kwargs["state_dict"] = state_dict

            transformer_lora_layers_to_save = get_peft_model_state_dict(
                unwrap_model(transformer_model) if is_fsdp else transformer_model,
                **peft_kwargs,
            )

            if is_fsdp:
                transformer_lora_layers_to_save = _to_cpu_contiguous(
                    transformer_lora_layers_to_save
                )

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not is_fsdp:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
        # pass

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir="/code/dataset/" + args.dataset_name.split("/")[-1],
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.LANCZOS
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        all_images = [image.convert("RGB") for image in examples[args.image_column]]
        all_cond_images = [
            image.convert("RGB") for image in examples[args.cond_image_column]
        ]

        # User requested single prompt for all images, so we don't need dataset captions.
        # Captions will be handled via args.instance_prompt globally.

        pixel_values = []
        cond_pixel_values = []

        for img, cond_img in zip(all_images, all_cond_images):
            # Apply composed transforms
            img = train_transform(img)
            cond_img = train_transform(cond_img)

            pixel_values.append(img)
            cond_pixel_values.append(cond_img)

        return {
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
        }

    def collate_fn_internal(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        cond_pixel_values = torch.stack(
            [example["cond_pixel_values"] for example in examples]
        )
        cond_pixel_values = cond_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        # No prompts returned here

        return {
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
        }

    train_dataset = dataset["train"].with_transform(preprocess_train)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_internal,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt, max_sequence_length=args.max_sequence_length
            )
        return prompt_embeds, text_ids

    # Force single prompt logic
    # Hardcoded prompt as per user request
    args.instance_prompt = "make this image photorealistic"

    # We encode the instance prompt once and reuse it.
    # Calculate embeddings on CPU to avoid OOM with the Transformer on GPU
    instance_prompt_hidden_states, instance_text_ids = compute_text_embeddings(
        args.instance_prompt, text_encoding_pipeline
    )

    # Move embeddings to accelerator device
    instance_prompt_hidden_states = instance_prompt_hidden_states.to(accelerator.device)
    instance_text_ids = instance_text_ids.to(accelerator.device)

    # Save prompt embeddings to disk
    if accelerator.is_main_process:
        torch.save(
            instance_prompt_hidden_states.to("cpu"),
            os.path.join(args.output_dir, "prompt_embeds.pt"),
        )
        torch.save(
            instance_text_ids.to("cpu"), os.path.join(args.output_dir, "text_ids.pt")
        )

    # Validation prompts are redundant as we use the single instance prompt for validation too.
    # The instance_prompt_hidden_states are reused in log_validation.

    # Delete text encoder to save memory as we only use cached embeddings
    del text_encoding_pipeline

    free_memory()

    prompt_embeds = instance_prompt_hidden_states
    text_ids = instance_text_ids

    # We do NOT run latent caching loop as per user request.
    # Images are encoded on the fly in the training loop.

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * accelerator.num_processes
            * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux2-image2img-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            # prompts = batch["prompts"] # No longer available

            with accelerator.accumulate(models_to_accumulate):
                # Подготовка батча: дублируем эмбеддинги под размер текущего батча
                # We assume args.train_batch_size is the batch size (or batch["pixel_values"].shape[0])
                current_batch_size = batch["pixel_values"].shape[0]
                batch_prompt_embeds = prompt_embeds.repeat(current_batch_size, 1, 1)
                batch_text_ids = text_ids.repeat(current_batch_size, 1, 1)

                # Кодирование VAE на лету (вместо кэширования), работает с .to(dtype=vae.dtype)
                # We always encode on the fly as per user request (and removed cache logic)
                with offload_models(
                    vae, device=accelerator.device, offload=args.offload
                ):
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)

                # VAE Encoder: получаем сжатые латентные представления для таргета и кондишна
                model_input = vae.encode(pixel_values).latent_dist.mode()
                # Capture spatial latents for structural noise
                cond_model_input_spatial = vae.encode(
                    cond_pixel_values
                ).latent_dist.mode()

                # "Патчификация" латентов и нормализация статистиками VAE
                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                if use_normal_vae:
                    model_input = (model_input - latents_bn_mean) / latents_bn_std

                cond_model_input = Flux2KleinPipeline._patchify_latents(
                    cond_model_input_spatial
                )
                if use_normal_vae:
                    cond_model_input = (
                        cond_model_input - latents_bn_mean
                    ) / latents_bn_std

                # Подготовка ID для Rotary Positional Embeddings (RoPE)
                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(
                    model_input
                ).to(device=model_input.device)
                cond_model_input_list = [
                    cond_model_input[i].unsqueeze(0)
                    for i in range(cond_model_input.shape[0])
                ]
                cond_model_input_ids = Flux2KleinPipeline._prepare_image_ids(
                    cond_model_input_list
                ).to(device=cond_model_input.device)
                cond_model_input_ids = cond_model_input_ids.view(
                    cond_model_input.shape[0], -1, model_input_ids.shape[-1]
                )

                # Генерируем случайный шум той же размерности, что и вход
                noise_spatial = generate_structured_noise_batch_vectorized(
                    cond_model_input_spatial,
                    cutoff_radius=args.structural_noise_radius,
                ).to(dtype=model_input.dtype, device=model_input.device)
                # noise_spatial = torch.randn_like(
                #     cond_model_input_spatial,
                # ).to(dtype=model_input.dtype, device=model_input.device)

                # Patchify structural noise
                noise = Flux2KleinPipeline._patchify_latents(noise_spatial)

                bsz = model_input.shape[0]

                # Сэмплируем временные шаги t (timesteps) для Flow Matching
                # for weighting schemes where we sample timesteps non-uniformly
                # Weighting Scheme (схема взвешивания) нужна для того, чтобы модель не училась на всех уровнях шума одинаково,
                # а уделяла больше внимания самым важным этапам восстановления изображения (обычно средним уровням шума,
                # где формируется структура). Она регулирует как частоту выбора определенных t (timestep sampling),
                # так и вес ошибки loss scaling.
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )

                # Flow Matching: смешиваем картинку и шум.
                # zt = (1 - t) * x + t * noise ("траектория" от данных к шуму)
                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                # Упаковываем 2D латенты в 1D последовательность токенов
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(
                    noisy_model_input
                )
                packed_cond_model_input = Flux2KleinPipeline._pack_latents(
                    cond_model_input
                )
                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                # Конкатенация: склеиваем зашумленный вход (target) и условие (condition)
                packed_noisy_model_input = torch.cat(
                    [packed_noisy_model_input, packed_cond_model_input], dim=1
                )
                model_input_ids = torch.cat(
                    [model_input_ids, cond_model_input_ids], dim=1
                )

                # Обработка guidance scale (если используется)
                if transformer.config.guidance_embeds:
                    guidance = torch.full(
                        [1], args.guidance_scale, device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                # Предикт Модели: трансформер предсказывает "скорость" (velocity) v
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=batch_prompt_embeds,
                    txt_ids=batch_text_ids,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                # Pruning: отрезаем часть выхода, относящуюся к condition (она не нужна для лосса)
                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(
                    model_pred, model_input_ids
                )

                # Расчет весов для лосса (weighting scheme)
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # Целевой вектор для Flow Matching: разница (шум - данные)
                target = noise - model_input

                # MSE Loss: разница между предсказанным вектором и реальным
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or is_fsdp:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
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
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Validation logic in training loop - INSIDE sync_gradients
                    if global_step == 20 or (global_step % args.validation_steps == 0):
                        # create pipeline
                        pipeline = Flux2KleinPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            text_encoder=None,
                            tokenizer=None,
                            transformer=unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=weight_dtype,
                            vae=vae,
                        )
                        images = log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            dataset=dataset["train"],
                            epoch=epoch,
                            torch_dtype=weight_dtype,
                            prompt_embeds=prompt_embeds,
                            text_ids=text_ids,
                        )

                        del pipeline
                        free_memory()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Save the lora layers
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v
                        for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v
                        for k, v in state_dict.items()
                    }

            transformer_lora_layers = get_peft_model_state_dict(
                transformer,
                state_dict=state_dict,
            )
            transformer_lora_layers = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in transformer_lora_layers.items()
            }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

        modules_to_save["transformer"] = transformer

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # Final "validation" is just saving the model card to point to the last validation images
        # Since we log to wandb, we might not need to manually generate images here again.
        # However, saving the LoRA was the critical part.

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
