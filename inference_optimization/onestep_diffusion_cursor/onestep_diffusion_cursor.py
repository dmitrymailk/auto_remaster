import torch
from diffusers import (
    UNet2DModel,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    AutoencoderTiny,
)
from typing import Tuple, Optional, Dict, Any
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """Base model class with common functionality"""

    def __init__(self):
        super().__init__()

    def get_params_count(self) -> Dict[str, Any]:
        """Get parameter counts"""
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class LBMEmbedder(BaseModel):
    """Embedder network for LBM"""

    def __init__(self, in_channels: int, emb_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(emb_channels, emb_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LBMBridge(BaseModel):
    """Bridge network for latent matching"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels

        # Ensure number of heads divides channels evenly
        assert (
            channels % num_heads == 0
        ), f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.scale = self.head_dim**-0.5

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = source.shape

        # Reshape and normalize
        source = source.permute(0, 2, 3, 1).reshape(B, H * W, C)
        target = target.permute(0, 2, 3, 1).reshape(B, H * W, C)
        source = self.norm(source)
        target = self.norm(target)

        # Multi-head attention
        q = (
            self.q_proj(source)
            .reshape(B, H * W, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(target)
            .reshape(B, H * W, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(target)
            .reshape(B, H * W, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)

        # Reshape back to image format
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class LBM(BaseModel):
    """Main LBM model"""

    def __init__(
        self,
        unet: UNet2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        bridge_scale: float = 0.1,
        min_bridge_scale: float = 0.05,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.device = device
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.scheduler = scheduler

        # Bridge scaling parameters
        self.max_bridge_scale = bridge_scale
        self.min_bridge_scale = min_bridge_scale
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Initialize embedder and bridge
        # For TAESDXL, latent channels is 4
        latent_channels = (
            4
            if isinstance(self.vae, AutoencoderTiny)
            else self.vae.config.latent_channels
        )
        self.embedder = LBMEmbedder(latent_channels, latent_channels).to(device)

        # Ensure number of heads is appropriate for latent channels
        num_heads = min(
            4, latent_channels // 4
        )  # Use at most 4 heads, and ensure head_dim >= 4
        self.bridge = LBMBridge(latent_channels, num_heads=num_heads).to(device)

        # Initialize scheduler with proper settings
        self.scheduler.set_timesteps(1000, device=device)

        # Set evaluation mode
        self.eval()

    def get_bridge_scale(self) -> float:
        """Get current bridge scale factor with warmup"""
        if self.warmup_steps > 0:
            scale = self.min_bridge_scale + (
                self.max_bridge_scale - self.min_bridge_scale
            ) * (min(self.current_step, self.warmup_steps) / self.warmup_steps)
            self.current_step += 1
            return scale
        return self.max_bridge_scale

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        with torch.no_grad():
            # For TAESDXL, we don't need distribution sampling or scaling
            if isinstance(self.vae, AutoencoderTiny):
                latents = self.vae.encode(image, return_dict=False)[0]
                # TAESDXL doesn't need additional scaling
                return latents
            else:
                # Standard VAE path with distribution sampling
                encoder_output = self.vae.encode(image)
                latents = encoder_output.latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space"""
        with torch.no_grad():
            # For TAESDXL, we don't need scaling
            if isinstance(self.vae, AutoencoderTiny):
                images = self.vae.decode(latents, return_dict=False)[0]
            else:
                # Standard VAE path
                latents = 1 / self.vae.config.scaling_factor * latents
                images = self.vae.decode(latents).sample

            # Ensure output is in [-1, 1] range
            return images.clamp(-1, 1)

    @torch.no_grad()
    def bridge_latents(
        self, source_latents: torch.Tensor, target_latents: torch.Tensor
    ) -> torch.Tensor:
        """Apply latent bridge matching"""
        # Get embeddings
        source_emb = self.embedder(source_latents)
        target_emb = self.embedder(target_latents)

        # Apply bridge attention with dynamic scaling
        bridged = self.bridge(source_emb, target_emb)
        return bridged

    def get_flow_weights(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get flow weights for velocity prediction with improved scaling"""
        # Reshape sigma for broadcasting
        sigma = sigma.view(-1, 1, 1, 1)

        # Use log-space for better numerical stability
        log_sigma = torch.log(sigma)

        # Modified weight schedule for better flow matching
        w = torch.exp(-0.5 * log_sigma)  # Reduced weight decay
        v = 1.0 / (1.0 + sigma)  # Smoother velocity scaling

        return w, v

    @torch.no_grad()
    def predict_velocity(
        self, x: torch.Tensor, t: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field with improved stability"""
        # Get flow weights
        w, v = self.get_flow_weights(sigma)

        # Scale input with improved numerical stability
        x_in = w * x

        # Get UNet prediction
        eps = self.unet(x_in, t, return_dict=False)[0]

        # Scale velocity prediction
        velocity = v * eps

        # Add small noise for stability
        noise_scale = 0.01 * sigma.view(-1, 1, 1, 1)
        velocity = velocity + noise_scale * torch.randn_like(velocity)

        return velocity

    @torch.no_grad()
    def denoise_one_step(
        self,
        noisy_latents: torch.Tensor,
        source_latents: Optional[torch.Tensor],
        noise_level: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step denoising with improved flow matching"""
        # Ensure noise_level is valid and not at the boundary
        noise_level = min(max(noise_level, 0), len(self.scheduler.timesteps) - 3)
        timestep = torch.tensor(
            [self.scheduler.timesteps[noise_level]], device=self.device
        )
        sigma = self.scheduler.sigmas[noise_level].to(self.device)

        if source_latents is not None:
            # Apply bridge matching with dynamic scale
            bridge_latents = self.bridge_latents(source_latents, noisy_latents)
            bridge_scale = self.get_bridge_scale()
            noisy_latents = noisy_latents + bridge_scale * bridge_latents

        # Predict velocity field
        velocity_pred = self.predict_velocity(noisy_latents, timestep, sigma)

        # Use scheduler step with improved stability
        denoised = self.scheduler.step(
            velocity_pred, timestep[0], noisy_latents, return_dict=False
        )[0]

        # Only apply refinement if we're not at the end of the schedule
        if noise_level < len(self.scheduler.timesteps) - 2:
            next_timestep = torch.tensor(
                [self.scheduler.timesteps[noise_level + 1]], device=self.device
            )
            next_sigma = self.scheduler.sigmas[noise_level + 1].to(self.device)

            # Get refined velocity prediction
            refined_velocity = self.predict_velocity(
                denoised, next_timestep, next_sigma
            )

            # Apply refinement step
            denoised = self.scheduler.step(
                refined_velocity, next_timestep[0], denoised, return_dict=False
            )[0]

        return denoised, velocity_pred

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        latent_shape: Tuple[int, ...] = (4, 64, 64),
        noise_level: int = None,
        source_image: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples with improved flow matching"""
        if seed is not None:
            torch.manual_seed(seed)

        # Use higher noise level for better quality, but avoid boundary
        if noise_level is None:
            noise_level = len(self.scheduler.timesteps) // 2
        else:
            noise_level = min(noise_level, len(self.scheduler.timesteps) - 3)

        # Generate initial noise
        shape = (batch_size,) + latent_shape
        noisy_latents = torch.randn(shape, device=self.device)
        # Scale noise with improved initialization
        sigma = self.scheduler.sigmas[noise_level]
        noisy_latents = noisy_latents * sigma * 0.8  # Reduced initial noise

        # Get source latents if provided
        source_latents = None
        if source_image is not None:
            source_latents = self.encode_image(source_image)

        # Apply denoising with improved flow matching
        denoised_latents, _ = self.denoise_one_step(
            noisy_latents, source_latents, noise_level
        )

        # Decode to image space
        images = self.decode_latents(denoised_latents)
        return images

    @torch.no_grad()
    def debug_vae(self, image: torch.Tensor) -> torch.Tensor:
        """Debug VAE by directly encoding and decoding an image without any diffusion process.

        Args:
            image: Input image tensor in [-1, 1] range with shape [B, C, H, W]

        Returns:
            Reconstructed image tensor in [-1, 1] range
        """
        # Encode
        latents = self.encode_image(image)

        # Decode
        reconstructed = self.decode_latents(latents)

        return reconstructed


# Example usage:
if __name__ == "__main__":
    # Initialize models
    unet = UNet2DModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D",) * 4,
        up_block_types=("UpBlock2D",) * 4,
    )

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    )

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    # Create LBM model
    lbm = LBM(unet, vae, scheduler)
    print("Model parameters:", lbm.get_params_count())

    # Generate samples
    samples = lbm.generate(
        batch_size=4,
        latent_shape=(4, 64, 64),
        noise_level=999,
    )
    print("Generated samples shape:", samples.shape)

    # For image translation:
    # source_image = ... # [B, C, H, W] tensor
    # translated = lbm.generate(
    #     batch_size=source_image.shape[0],
    #     latent_shape=(4, source_image.shape[2]//8, source_image.shape[3]//8),
    #     noise_level=999,
    #     source_image=source_image
    # )
