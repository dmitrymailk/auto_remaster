import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from onestep_diffusion_cursor import LBM, UNet2DModel, AutoencoderKL, DDPMScheduler
import torch.optim as optim
from tqdm import tqdm
import os
import wandb
import argparse
from PIL import Image
import torchvision.transforms.functional as F
from datasets import load_dataset
import multiprocessing as mp


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)  # Reduced for Docker
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_wandb", action="store_true")
    # Default to 0 workers in Docker to avoid memory issues
    parser.add_argument("--num_workers", type=int, default=0)
    # Add gradient accumulation for smaller batch sizes
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
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
    model, dataloader, optimizer, device, epoch, gradient_accumulation_steps
):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at start of epoch

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
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
        noisy_latents = model.scheduler.add_noise(target_latents, noise, timesteps)

        # Predict noise
        noise_pred = model.unet(noisy_latents, timesteps, return_dict=False)[0]

        # Apply bridge matching
        bridge_latents = model.bridge_latents(source_latents, noisy_latents)
        noisy_latents_bridged = noisy_latents + bridge_latents

        # Predict noise again with bridged latents
        noise_pred_bridged = model.unet(
            noisy_latents_bridged, timesteps, return_dict=False
        )[0]

        # Calculate combined loss
        loss = torch.nn.functional.mse_loss(
            noise_pred, noise
        ) + torch.nn.functional.mse_loss(noise_pred_bridged, noise)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every gradient_accumulation_steps batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += (
            loss.item() * gradient_accumulation_steps
        )  # Scale back for logging
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        if wandb.run is not None:
            wandb.log(
                {
                    "batch_loss": loss.item() * gradient_accumulation_steps,
                    "epoch": epoch,
                }
            )

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()

    if args.log_wandb:
        wandb.init(project="onestep-diffusion-cursor")

    # Initialize models
    unet = UNet2DModel(
        sample_size=64,  # This is for latent space size
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
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

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
                    args.device,
                    epoch,
                    args.gradient_accumulation_steps,
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
                            noise_level=999,
                        )
                        val_loss += torch.nn.functional.mse_loss(
                            generated_images, target_images
                        ).item()

                val_loss /= len(val_dataloader)

                print(
                    f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

                if wandb.run is not None:
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
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise e

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
