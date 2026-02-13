import torch
import os
import argparse
import re
from diffusers import Flux2KleinPipeline
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Tuple
import sys

# from auto_remaster.train_auto_remaster_lbm_train_test_gap_struct_noise import (
#     generate_structured_noise_batch_vectorized,
# )
import more_itertools
from torchao.float8 import Float8LinearConfig, convert_to_float8_training


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


# --- Configuration ---
@dataclass
class GenerationConfig:
    lora_path: str = field(metadata={"help": "Path to LoRA checkpoint directory"})
    prompt_embeds_path: str = field(metadata={"help": "Path to prompt_embeds.pt"})
    model_name: str = field(
        default="black-forest-labs/FLUX.2-klein-base-4B",
        metadata={"help": "Model name"},
    )
    weight_name: str = field(
        default="pytorch_lora_weights.safetensors",
        metadata={"help": "LoRA weight filename"},
    )
    adapter_name: str = field(
        default="my_first_lora_v5", metadata={"help": "Adapter name"}
    )
    dataset_name: str = field(
        default="dim/nfs_pix2pix_1920_1080_v6", metadata={"help": "Dataset name"}
    )
    resolution: int = field(default=768, metadata={"help": "Resolution for generation"})
    cutoff_radius: float = field(
        default=1200.0, metadata={"help": "Cutoff radius for noise generation"}
    )
    batch_size: int = field(default=4, metadata={"help": "Batch size for generation"})


# --- Noise Generation Functions (from snippet) ---


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


def generate_sample(
    pipeline,
    source_image,
    img_tensor,  # expects (B, 3, H, W) normalized
    prompt_embeds,
    cutoff_radius,
    device,
    weight_dtype,
    generator_seed=0,
):
    batch_size = img_tensor.shape[0]
    with torch.no_grad():
        img_tensor = img_tensor.to(pipeline.vae.dtype).to(device)

        z_source = pipeline.vae.encode(img_tensor).latent_dist.mode()
        z_source = z_source.to(device=device, dtype=weight_dtype)

        noise_spatial = generate_structured_noise_batch_vectorized(
            z_source.float(),
            cutoff_radius=cutoff_radius,
        ).to(dtype=weight_dtype, device=device)

        # Patchify latents
        latents = pipeline._patchify_latents(noise_spatial).to(weight_dtype)

        # Repeat prompt embeds for batch
        batch_prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        # Generate
        image = pipeline(
            prompt_embeds=batch_prompt_embeds.to(weight_dtype),
            image=source_image,
            latents=latents,
            guidance_scale=1.0,
            num_inference_steps=30,
            generator=torch.Generator(device=device).manual_seed(generator_seed),
        ).images  # returns list of PIL images
    return image


# --- Main Logic ---


def get_start_index(output_dir):
    """
    Scans the output directory for files named '{index}.png' and returns the next index.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 0

    # Check 'edited_image' subdirectory for progress
    scan_dir = os.path.join(output_dir, "edited_image")
    if not os.path.exists(scan_dir):
        return 0

    max_idx = -1
    for filename in os.listdir(scan_dir):
        if filename.endswith(".png"):
            # Try to match simple integer filenames like '0.png', '100.png'
            match = re.match(r"^(\d+)\.png$", filename)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx

    return max_idx + 1


from trl import TrlParser


def main():
    parser = TrlParser((GenerationConfig))
    config = parser.parse_args_and_config()[0]
    print(config)

    device = "cuda"
    weight_dtype = torch.bfloat16

    pipeline = Flux2KleinPipeline.from_pretrained(
        config.model_name,
        torch_dtype=weight_dtype,
    )
    # convert_to_float8_training(
    #     pipeline.transformer,
    #     module_filter_fn=module_filter_fn,
    #     config=Float8LinearConfig(pad_inner_dim=True),
    # )

    pipeline.load_lora_weights(
        config.lora_path,
        weight_name=config.weight_name,
        adapter_name=config.adapter_name,
    )
    pipeline.fuse_lora(lora_scale=1.0)
    pipeline.unload_lora_weights()

    pipeline = pipeline.to(device)

    # Block-wise compilation
    if hasattr(pipeline.transformer, "transformer_blocks"):
        for i, block in enumerate(pipeline.transformer.transformer_blocks):
            pipeline.transformer.transformer_blocks[i] = torch.compile(block)

    if hasattr(pipeline.transformer, "single_transformer_blocks"):
        for i, block in enumerate(pipeline.transformer.single_transformer_blocks):
            pipeline.transformer.single_transformer_blocks[i] = torch.compile(block)

    pipeline.set_progress_bar_config(disable=True)

    # 2. Prompt Embeds
    prompt_embeds = torch.load(config.prompt_embeds_path)
    prompt_embeds = prompt_embeds.to(weight_dtype).to(device)
    dataset_name = config.dataset_name.split("/")[-1]
    # 3. Dataset
    dataset = load_dataset(
        config.dataset_name,
        cache_dir=f"/code/dataset/{dataset_name}",
    )
    dataset = dataset["train"]

    # 4. transforms
    # Preprocessing for batching
    preprocess = transforms.Compose(
        [
            transforms.Resize(
                config.resolution,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(config.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Transform for saving (without normalization)
    resize_crop = transforms.Compose(
        [
            transforms.Resize(
                config.resolution,
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(config.resolution),
        ]
    )

    # 5. Determine start and setup folders
    output_base_path = (
        f"/code/auto_remaster/sandbox/diffusers_flux2/dataset/{dataset_name}/"
    )
    input_dir = output_base_path + "input_image/"
    edited_dir = output_base_path + "edited_image/"
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(edited_dir, exist_ok=True)

    start_idx = get_start_index(output_base_path)
    print(f"Resuming from index {start_idx} (output dir: {output_base_path})")

    total_items = len(dataset)

    metadata_path = os.path.join(output_base_path, "metadata.csv")

    columns = ["input_image_file_name", "edit_prompt", "edited_image_file_name"]

    # Initialize CSV with header if it doesn't exist
    if not os.path.exists(metadata_path):
        pd.DataFrame(columns=columns).to_csv(metadata_path, index=False)

    # --- Warmup Phase ---
    print("Starting warmup...")
    # Run one batch for warmup if possible
    batch_original_pils_temp = []
    if total_items > 0:
        warmup_batch_size = config.batch_size
        indices = list(range(0, min(config.batch_size, total_items)))
        dummy_pixel_values = []
        for i in indices:
            item = dataset[i]
            orig_source_pil = item["input_image"].convert("RGB")
            dummy_pixel_values.append(preprocess(orig_source_pil))
            resized_crop_img = resize_crop(orig_source_pil)
            batch_original_pils_temp.append(resized_crop_img)

        if dummy_pixel_values:
            dummy_batch = torch.stack(dummy_pixel_values).to(device)
            print(f"Warmup with batch size {len(dummy_batch)}...")
            _ = generate_sample(
                pipeline,
                batch_original_pils_temp,
                dummy_batch,
                prompt_embeds,
                config.cutoff_radius,
                device,
                weight_dtype,
            )
    print("Warmup complete.")

    # --- Main Generation Loop ---
    batch_size = config.batch_size

    # Iterate over chunks of indices using more_itertools
    indices = range(start_idx, total_items)
    chunks = more_itertools.chunked(indices, batch_size)

    # Estimate number of chunks for progress bar if possible, or total checks
    # len(indices) is known.

    pbar = tqdm(total=total_items - start_idx, initial=start_idx, desc="Generating")

    for batch_indices in chunks:
        # batch_indices is already a list of indices [i, i+1, ... i+batch_size-1]

        batch_pixel_values = []
        batch_original_pils = []

        for idx in batch_indices:
            item = dataset[idx]
            orig_source_pil = item["input_image"].convert("RGB")

            # For saving, we need the cropped version but in PIL
            # Using same resize/crop as preprocess
            resized_crop_img = resize_crop(orig_source_pil)
            batch_original_pils.append(resized_crop_img)

            # For model
            batch_pixel_values.append(preprocess(orig_source_pil))

        if not batch_pixel_values:
            break

        batch_tensor = torch.stack(batch_pixel_values).to(device)

        # Generator
        images = generate_sample(
            pipeline,
            batch_original_pils,
            batch_tensor,
            prompt_embeds,
            config.cutoff_radius,
            device,
            weight_dtype,
        )

        # Save batch results
        new_rows = []
        for b_idx, (idx, generated_img, input_pil) in enumerate(
            zip(
                batch_indices,
                images,
                batch_original_pils,
            )
        ):
            input_path = os.path.join(input_dir, f"{idx}.png")
            edited_path = os.path.join(edited_dir, f"{idx}.png")

            input_pil.save(input_path)
            generated_img.save(edited_path)

            row = {}
            row["input_image_file_name"] = f"input_image/{idx}.png"
            row["edit_prompt"] = " "
            row["edited_image_file_name"] = f"edited_image/{idx}.png"
            new_rows.append(row)

        pd.DataFrame(new_rows).to_csv(
            metadata_path,
            mode="a",
            header=False,
            index=False,
        )
        pbar.update(len(batch_indices))

    pbar.close()
    print("Generation complete.")


if __name__ == "__main__":
    main()
