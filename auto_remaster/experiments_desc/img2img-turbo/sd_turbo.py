"""
Обучение стандартного sd turbo с моей модификацией.
модель обучается с нуля, vae тоже обучается. предсказание за 1 шаг.
"""

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
import vision_aided_loss
import torchvision

from auto_remaster.evaluation import ImageEvaluator


@dataclass
class DiffusionTrainingArguments:
    use_ema: bool = field(default=False)
    non_ema_revision: str = field(default=None)
    resolution: int = field(default=320)
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
    noise_scheduler_type: str = field(default="stabilityai/sd-turbo")
    metrics_list: list[str] = field(default=None)
    lpips_factor: float = field(default=5.0)
    gan_factor: float = field(default=0.5)


unet2d_config = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
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
    # text_encoder=None,
    # tokenizer=None,
    unet=None,
    noise_scheduler=None,
    accelerator: Accelerator = None,
    weight_dtype=None,
    global_step=None,
    script_args: ScriptArguments = None,
    training_args: SFTConfig = None,
    model_args: ModelConfig = None,
    diffusion_args: DiffusionTrainingArguments = None,
    dataset=None,
    dummy_emb=None,
    train_transforms=None,
    checkpoint_path: str = None,
):
    vae = AutoencoderTiny.from_pretrained(
        checkpoint_path,
        torch_device="cuda",
        subfolder="vae",
        torch_dtype=weight_dtype,
    ).to(vae.device)
    vae.decoder.ignore_skip = False

    unet = UNet2DModel.from_pretrained(
        checkpoint_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    ).to(vae.device)

    logger.info("Running validation... ")

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
    timesteps = torch.tensor([999], device="cuda:0")
    # сохраняем для эвалюации
    originals = []
    generated = []
    for idx in selected_ids:
        item = dataset[idx]
        orig_source = item[diffusion_args.source_image_name].convert("RGB")
        source = valid_transforms(orig_source)
        target = valid_transforms(item[diffusion_args.target_image_name].convert("RGB"))
        c_t = train_transforms(orig_source).unsqueeze(0).to(vae.dtype).to(vae.device)
        with torch.no_grad():
            encoded_control = vae.encode(c_t, False)[0] * vae.config.scaling_factor
            model_pred = unet(
                encoded_control,
                timesteps,
                return_dict=False,
            )[0]
            x_denoised = noise_scheduler.step(
                model_pred,
                timesteps,
                encoded_control,
                return_dict=False,
            )[0].to(vae.dtype)
            output_image = (
                vae.decode(
                    x_denoised / vae.config.scaling_factor,
                    return_dict=False,
                )[0]
            ).clamp(-1, 1)
            pred_image = transforms.ToPILImage()(
                output_image[0].cpu().float() * 0.5 + 0.5
            )
        img_h = Image.fromarray(
            np.hstack(
                (
                    np.array(source),
                    np.array(pred_image),
                    np.array(target),
                )
            )
        )
        originals.append(item[diffusion_args.source_image_name])
        generated.append(pred_image)

        images.append(img_h)

    evaluator = ImageEvaluator(
        metrics_list=diffusion_args.metrics_list,
        device="cuda",
        num_workers=4,
        prefix_key="eval",
    )
    metrics_result = evaluator.evaluate(
        originals,
        generated,
        batch_size=16,
    )

    for tracker in accelerator.trackers:
        tracker.log(
            {
                "validation": [wandb.Image(image) for i, image in enumerate(images)],
                **metrics_result,
            }
        )

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
        # mixed_precision="no",
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        diffusion_args.noise_scheduler_type,
        subfolder="scheduler",
    )
    noise_scheduler.set_timesteps(1, device="cuda")

    # weight_dtype = torch.float16
    weight_dtype = torch.float32
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=accelerator.device, dtype=weight_dtype
    )
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=accelerator.device, dtype=weight_dtype
    )
    noise_scheduler.betas = noise_scheduler.betas.to(
        device=accelerator.device, dtype=weight_dtype
    )
    noise_scheduler.alphas = noise_scheduler.alphas.to(
        device=accelerator.device, dtype=weight_dtype
    )

    vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        torch_device="cuda",
        torch_dtype=weight_dtype,
    )
    vae.decoder.ignore_skip = False

    unet = UNet2DModel(**unet2d_config)
    # unet.enable_xformers_memory_efficient_attention()
    unet.set_attention_backend("flash")

    vae.requires_grad_(True)
    vae.train()
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

    net_disc = vision_aided_loss.Discriminator(
        cv_type="clip",
        loss_type="multilevel_sigmoid",
        device="cuda",
    )

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    layers_to_opt = []
    for n, _p in unet.named_parameters():
        layers_to_opt.append(_p)
    layers_to_opt += list(unet.conv_in.parameters())
    for n, _p in vae.named_parameters():
        layers_to_opt.append(_p)

    optimizer = torch.optim.AdamW(
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
    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer_disc,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_disc,
        lr_scheduler_disc,
        net_disc,
    ) = accelerator.prepare(
        unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_disc,
        lr_scheduler_disc,
        net_disc,
    )

    # Move text_encode and vae to gpu and cast to weight_dtype
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

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

    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc, vae]
            with accelerator.accumulate(*l_acc):
                # Convert images to latent space
                source_latents = vae.encode(batch["source_images"].to(weight_dtype))[0]
                source_latents = source_latents * vae.config.scaling_factor

                # одинаковый последний степ для каждого изображения
                timesteps = timesteps = torch.tensor(
                    [999], device=source_latents.device
                ).long()

                # Predict the noise residual and compute loss
                model_pred = unet(
                    source_latents,
                    timesteps,
                    # encoder_hidden_states,
                    return_dict=False,
                )[0]
                x_denoised = noise_scheduler.step(
                    model_pred,
                    timesteps,
                    source_latents,
                    return_dict=False,
                )[0]
                x_denoised = x_denoised.to(weight_dtype)
                x_tgt_pred = (
                    vae.decode(
                        x_denoised / vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                ).clamp(-1, 1)
                x_tgt = batch["target_images"].float()
                loss_l2 = F.mse_loss(
                    x_tgt_pred.float(),
                    x_tgt,
                    reduction="mean",
                )

                loss_lpips = (
                    net_lpips(x_tgt_pred, x_tgt.to(weight_dtype)).mean()
                    * diffusion_args.lpips_factor
                )
                loss = loss_l2 + loss_lpips

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                """
                Generator loss: fool the discriminator
                """
                source_latents = vae.encode(
                    batch["source_images"].to(weight_dtype),
                )[0]
                source_latents = source_latents * vae.config.scaling_factor
                model_pred = unet(
                    source_latents,
                    timesteps,
                    # encoder_hidden_states,
                    return_dict=False,
                )[0]
                x_denoised = noise_scheduler.step(
                    model_pred,
                    timesteps,
                    source_latents,
                    return_dict=False,
                )[0]
                x_denoised = x_denoised.to(weight_dtype)
                x_tgt_pred = (
                    vae.decode(
                        x_denoised / vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                ).clamp(-1, 1)
                lossG = (
                    net_disc(x_tgt_pred, for_G=True).mean() * diffusion_args.gan_factor
                )
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = (
                    net_disc(x_tgt.detach(), for_real=True).mean()
                    * diffusion_args.gan_factor
                )
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), training_args.max_grad_norm
                    )
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)
                # fake image
                lossD_fake = (
                    net_disc(x_tgt_pred.detach(), for_real=False).mean()
                    * diffusion_args.gan_factor
                )
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(), training_args.max_grad_norm
                    )
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=True)
                lossD = lossD_real + lossD_fake

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {}
                # log all the losses
                logs["lossG"] = lossG.detach().item()
                logs["lossD"] = lossD.detach().item()
                logs["loss_l2"] = loss_l2.detach().item()
                logs["loss_lpips"] = loss_lpips.detach().item()
                # accelerator.log({"train_loss": train_loss}, step=global_step)
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
                        # Сохраняем VAE
                        unwrap_model(vae).save_pretrained(
                            os.path.join(save_path, "vae")
                        )
                        logger.info(f"Saved state to {save_path}")

                        # start validation
                        log_validation(
                            vae=vae,
                            unet=unet,
                            noise_scheduler=noise_scheduler,
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
