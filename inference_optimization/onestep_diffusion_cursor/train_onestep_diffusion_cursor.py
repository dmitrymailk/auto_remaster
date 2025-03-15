import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from onestep_diffusion_cursor import LBM, UNet2DModel, FlowMatchEulerDiscreteScheduler
from diffusers import AutoencoderTiny
import torch.optim as optim
from tqdm import tqdm
import os
import wandb
import argparse
from PIL import Image
import torchvision.transforms.functional as F
from datasets import load_dataset
import multiprocessing as mp
import torchvision.utils as vutils


def build_transform():
    """Build transform for 512x512 images"""
    return transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
        ]
    )


class NFSPairedDatasetPureImages(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):
        super().__init__()
        # Load dataset with reduced number of processes for Docker
        dataset = load_dataset(dataset_folder, num_proc=2)
        dataset = dataset["train"]
        dataset = dataset.train_test_split(test_size=40, shuffle=True, seed=42)
        self.dataset = dataset
        self.T = build_transform()
        self.split = split

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        input_img = self.dataset[self.split][idx]["input_image"].convert("RGB")
        output_img = self.dataset[self.split][idx]["edited_image"].convert("RGB")

        # Transform images to 512x512
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        img_t = F.normalize(img_t, mean=[0.5], std=[0.5])  # Normalize to [-1, 1]

        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])  # Normalize to [-1, 1]

        return {
            "conditioning_pixel_values": img_t,
            "output_pixel_values": output_t,
        }


def log_images_to_wandb(
    source_images, target_images, generated_images, step, return_images=False
):
    """Log images to wandb for visualization"""

    # Denormalize images from [-1, 1] to [0, 1]
    def denorm(x):
        return (x + 1) / 2

    B = source_images.shape[0]
    images_dict = {
        "source_images": [
            wandb.Image(
                denorm(source_images[idx]).float().detach().cpu(),
                caption=f"Source {idx}",
            )
            for idx in range(min(B, 3))
        ],
        "target_images": [
            wandb.Image(
                denorm(target_images[idx]).float().detach().cpu(),
                caption=f"Target {idx}",
            )
            for idx in range(min(B, 3))
        ],
        "generated_images": [
            wandb.Image(
                denorm(generated_images[idx]).float().detach().cpu(),
                caption=f"Generated {idx}",
            )
            for idx in range(min(B, 3))
        ],
    }

    if return_images:
        return images_dict
    else:
        wandb.log(images_dict, step=step)
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)  # Reduced for Docker
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    # Wandb logging options
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="onestep-diffusion-cursor")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # Default to 0 workers in Docker to avoid memory issues
    parser.add_argument("--num_workers", type=int, default=0)
    # Add gradient accumulation for smaller batch sizes
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # Add image logging frequency
    parser.add_argument(
        "--log_images_every",
        type=int,
        default=100,
        help="Log images to wandb every N steps",
    )
    return parser.parse_args()


def get_dataloader(batch_size, split="train", num_workers=0):
    """Get dataloader with Docker-friendly settings"""
    # Set PyTorch to use spawn method for multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    dataset = NFSPairedDatasetPureImages(
        "dim/nfs_pix2pix_1920_1080_v5_upscale_2x_raw", split=split
    )

    # Use simpler DataLoader settings for Docker
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )
    return dataloader


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    gradient_accumulation_steps,
    args,
):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch
    global_step = epoch * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        current_step = global_step + batch_idx
        # Get source and target images
        source_images = batch["conditioning_pixel_values"].to(device)
        target_images = batch["output_pixel_values"].to(device)
        batch_size = source_images.shape[0]

        # Encode images to latent space
        with torch.no_grad():  # Don't need gradients for encoding since VAE is frozen
            source_latents = model.encode_image(source_images)
            target_latents = model.encode_image(target_images)

        # Add noise to target latents
        noise = torch.randn_like(target_latents, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)

        # For FlowMatch, we scale noise directly using sigmas
        sigma = scheduler.sigmas[timesteps]
        sigma = sigma.reshape(-1, 1, 1, 1)
        noisy_latents = target_latents + sigma * noise

        # Predict noise
        noise_pred = model.unet(noisy_latents, timesteps, return_dict=False)[0]

        # Apply bridge matching
        bridge_latents = model.bridge_latents(source_latents, noisy_latents)
        noisy_latents_bridged = (
            noisy_latents + 0.1 * bridge_latents
        )  # Scale bridge influence

        # Predict noise again with bridged latents
        noise_pred_bridged = model.unet(
            noisy_latents_bridged, timesteps, return_dict=False
        )[0]

        # Calculate combined loss using FlowMatch loss
        # For FlowMatch, we predict the velocity field (negative score)
        target_velocity = -noise / sigma
        loss = torch.nn.functional.mse_loss(
            noise_pred, target_velocity
        ) + torch.nn.functional.mse_loss(noise_pred_bridged, target_velocity)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        current_loss = loss.item() * gradient_accumulation_steps
        total_loss += current_loss
        pbar.set_postfix({"loss": current_loss})

        # Log metrics
        if not args.no_wandb:
            metrics = {
                "batch_loss": current_loss,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            # Periodically generate and log images
            if current_step % args.log_images_every == 0:
                model.eval()
                with torch.no_grad():
                    # Generate images
                    generated_images = model.generate(
                        batch_size=source_images.shape[0],
                        source_image=source_images,
                        noise_level=999,
                    )

                    # Get VAE reconstructions for debugging
                    source_recon = model.debug_vae(source_images)
                    target_recon = model.debug_vae(target_images)

                    # Add images to metrics dictionary
                    vis_images = {
                        "source_images": [
                            wandb.Image(
                                ((source_images[idx] + 1) / 2).float().cpu(),
                                caption=f"Source {idx}",
                            )
                            for idx in range(min(batch_size, 3))
                        ],
                        "source_reconstructions": [
                            wandb.Image(
                                ((source_recon[idx] + 1) / 2).float().cpu(),
                                caption=f"Source Recon {idx}",
                            )
                            for idx in range(min(batch_size, 3))
                        ],
                        "target_images": [
                            wandb.Image(
                                ((target_images[idx] + 1) / 2).float().cpu(),
                                caption=f"Target {idx}",
                            )
                            for idx in range(min(batch_size, 3))
                        ],
                        "target_reconstructions": [
                            wandb.Image(
                                ((target_recon[idx] + 1) / 2).float().cpu(),
                                caption=f"Target Recon {idx}",
                            )
                            for idx in range(min(batch_size, 3))
                        ],
                        "generated_images": [
                            wandb.Image(
                                ((generated_images[idx] + 1) / 2).float().cpu(),
                                caption=f"Generated {idx}",
                            )
                            for idx in range(min(batch_size, 3))
                        ],
                    }
                    metrics.update(vis_images)
                model.train()

            # Log all metrics together
            wandb.log(metrics, step=current_step)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()

    # Initialize wandb by default unless --no_wandb is specified
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "log_images_every": args.log_images_every,
                "effective_batch_size": args.batch_size
                * args.gradient_accumulation_steps,
                "model_config": {
                    "unet_sample_size": 64,
                    "unet_in_channels": 4,
                    "unet_out_channels": 4,
                    "unet_layers_per_block": 2,
                    "scheduler_timesteps": 1000,
                    "vae": "madebyollin/taesdxl",  # Log VAE model being used
                },
            },
        )
        print(
            f"Wandb logging enabled. Project: {args.wandb_project}, Run: {wandb.run.name}"
        )
    else:
        print("Wandb logging disabled")

    # Initialize models
    unet = UNet2DModel(
        sample_size=64,  # 64x64 latent space
        in_channels=4,  # TAESDXL uses 4 channels
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=("DownBlock2D",) * 4,
        up_block_types=("UpBlock2D",) * 4,
    )

    # Initialize TAESDXL with proper configuration
    vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    print("Using TAESDXL for VAE")

    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False

    # Initialize scheduler with fixed timesteps
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    # Set timesteps to 1000 and move tensors to device
    scheduler.set_timesteps(1000, device="cuda")

    # Create LBM model
    model = LBM(unet, vae, scheduler, device=args.device)
    model = model.to(args.device)

    # Set training modes
    model.unet.train()
    model.embedder.train()
    model.bridge.train()
    model.vae.eval()

    print("Model parameters:", model.get_params_count())

    try:
        # Setup training
        train_dataloader = get_dataloader(
            args.batch_size, split="train", num_workers=args.num_workers
        )
        val_dataloader = get_dataloader(
            args.batch_size, split="test", num_workers=args.num_workers
        )

        # Only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=args.lr)

        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)

        # Training loop
        best_loss = float("inf")
        for epoch in range(args.epochs):
            try:
                # Training
                train_loss = train_one_epoch(
                    model,
                    train_dataloader,
                    optimizer,
                    scheduler,
                    args.device,
                    epoch,
                    args.gradient_accumulation_steps,
                    args,
                )

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        source_images = batch["conditioning_pixel_values"].to(
                            args.device
                        )
                        target_images = batch["output_pixel_values"].to(args.device)

                        # Generate images and calculate validation loss
                        generated_images = model.generate(
                            batch_size=source_images.shape[0],
                            source_image=source_images,
                            noise_level=0,  # Use first timestep for FlowMatch
                        )
                        val_loss += torch.nn.functional.mse_loss(
                            generated_images, target_images
                        ).item()

                val_loss /= len(val_dataloader)

                print(
                    f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                if not args.no_wandb:
                    wandb.log(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "epoch": epoch,
                        }
                    )

                # Save checkpoint if best validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, "best_model.pt"))

                # Save latest checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, os.path.join(args.save_dir, "latest_model.pt"))

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory, clearing cache and skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise e

    except Exception as e:
        print(f"Training failed: {str(e)}")
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        raise e

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
