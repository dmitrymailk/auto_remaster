import torch
from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL
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
        scheduler: DDPMScheduler,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.unet = unet.to(device)
        self.vae = vae.to(device)
        self.scheduler = scheduler

        # Initialize embedder and bridge
        latent_channels = self.vae.config.latent_channels
        self.embedder = LBMEmbedder(latent_channels, latent_channels).to(device)

        # Ensure number of heads is appropriate for latent channels
        num_heads = min(
            4, latent_channels // 4
        )  # Use at most 4 heads, and ensure head_dim >= 4
        self.bridge = LBMBridge(latent_channels, num_heads=num_heads).to(device)

        # Set evaluation mode
        self.eval()

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space"""
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space"""
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def bridge_latents(
        self, source_latents: torch.Tensor, target_latents: torch.Tensor
    ) -> torch.Tensor:
        """Apply latent bridge matching"""
        # Get embeddings
        source_emb = self.embedder(source_latents)
        target_emb = self.embedder(target_latents)

        # Apply bridge attention
        bridged = self.bridge(source_emb, target_emb)
        return bridged

    @torch.no_grad()
    def denoise_one_step(
        self,
        noisy_latents: torch.Tensor,
        source_latents: Optional[torch.Tensor],
        noise_level: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step denoising with bridge matching"""
        timestep = torch.tensor([noise_level], device=self.device)

        if source_latents is not None:
            # Apply bridge matching
            bridge_latents = self.bridge_latents(source_latents, noisy_latents)
            noisy_latents = noisy_latents + bridge_latents

        # Predict noise and denoise
        noise_pred = self.unet(noisy_latents, timestep, return_dict=False)[0]
        denoised = self.scheduler.step(
            noise_pred, noise_level, noisy_latents, return_dict=False
        )[0]

        return denoised, noise_pred

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        latent_shape: Tuple[int, ...] = (4, 64, 64),
        noise_level: int = 999,
        source_image: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate samples with one-step diffusion and LBM"""
        if seed is not None:
            torch.manual_seed(seed)

        # Initial noise
        shape = (batch_size,) + latent_shape
        noisy_latents = torch.randn(shape, device=self.device)
        noisy_latents = noisy_latents * self.scheduler.init_noise_sigma

        # Get source latents if provided
        source_latents = None
        if source_image is not None:
            source_latents = self.encode_image(source_image)

        # One-step denoising with bridge matching
        denoised_latents, _ = self.denoise_one_step(
            noisy_latents, source_latents, noise_level
        )

        # Decode to image space
        images = self.decode_latents(denoised_latents)
        return images


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

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

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
