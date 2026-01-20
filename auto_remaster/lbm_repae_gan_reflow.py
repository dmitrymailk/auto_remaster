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

from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from einops import rearrange
from torch.cuda.amp import autocast
from collections import OrderedDict, defaultdict
import copy

# from .discriminator import NLayerDiscriminator, weights_init
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """
    Step the EMA model parameters towards the current model parameters.

    Args:
        ema_model (nn.Module): The exponential moving average model (Teacher).
        model (nn.Module): The current training model (Student).
        decay (float): The decay rate for the moving average.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            # param_ema = decay * param_ema + (1 - decay) * param_curr
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # по умолчанию всегда используется батчнорм из торча
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        # else:
        #     norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model_name: str = "convnext_s"):
        """Initializes the PerceptualLoss class.

        Args:
            model_name: A string, the name of the perceptual loss model to use.

        Raise:
            ValueError: If the model_name does not contain "lpips" or "convnext_s".
        """
        super().__init__()
        if ("lpips" not in model_name) and ("convnext_s" not in model_name):
            raise ValueError(f"Unsupported Perceptual Loss model name {model_name}")
        self.lpips = None
        self.convnext = None
        self.loss_weight_lpips = None
        self.loss_weight_convnext = None

        # Parsing the model name. We support name formatted in
        # "lpips-convnext_s-{float_number}-{float_number}", where the
        # {float_number} refers to the loss weight for each component.
        # E.g., lpips-convnext_s-1.0-2.0 refers to compute the perceptual loss
        # using both the convnext_s and lpips, and average the final loss with
        # (1.0 * loss(lpips) + 2.0 * loss(convnext_s)) / (1.0 + 2.0).

        # lpips by defaults in repa-e
        if "lpips" in model_name:
            self.lpips = lpips.LPIPS(net="vgg").eval()

        # if "convnext_s" in model_name:
        #     self.convnext = models.convnext_small(
        #         weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        #     ).eval()

        if "lpips" in model_name and "convnext_s" in model_name:
            loss_config = model_name.split("-")[-2:]
            self.loss_weight_lpips, self.loss_weight_convnext = float(
                loss_config[0]
            ), float(loss_config[1])
            print(
                f"self.loss_weight_lpips, self.loss_weight_convnext: {self.loss_weight_lpips}, {self.loss_weight_convnext}"
            )

        self.register_buffer(
            "imagenet_mean", torch.Tensor(_IMAGENET_MEAN)[None, :, None, None]
        )
        self.register_buffer(
            "imagenet_std", torch.Tensor(_IMAGENET_STD)[None, :, None, None]
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the perceptual loss.

        Args:
            input: A tensor of shape (B, C, H, W), the input image. Normalized to [0, 1].
            target: A tensor of shape (B, C, H, W), the target image. Normalized to [0, 1].

        Returns:
            A scalar tensor, the perceptual loss.
        """
        # Always in eval mode.
        self.eval()
        loss = 0.0
        num_losses = 0.0
        lpips_loss = 0.0
        convnext_loss = 0.0
        # Computes LPIPS loss, if available.
        # True by default
        if self.lpips is not None:
            lpips_loss = self.lpips(input, target)
            # True by default
            if self.loss_weight_lpips is None:
                loss += lpips_loss
                num_losses += 1
            else:
                num_losses += self.loss_weight_lpips
                loss += self.loss_weight_lpips * lpips_loss
        # False by default
        # if self.convnext is not None:
        #     # Computes ConvNeXt-s loss, if available.
        #     input = torch.nn.functional.interpolate(input, size=224, mode="bilinear", align_corners=False, antialias=True)
        #     target = torch.nn.functional.interpolate(target, size=224, mode="bilinear", align_corners=False, antialias=True)
        #     pred_input = self.convnext((input - self.imagenet_mean) / self.imagenet_std)
        #     pred_target = self.convnext((target - self.imagenet_mean) / self.imagenet_std)
        #     convnext_loss = torch.nn.functional.mse_loss(
        #         pred_input,
        #         pred_target,
        #         reduction="mean")

        #     if self.loss_weight_convnext is None:
        #         num_losses += 1
        #         loss += convnext_loss
        #     else:
        #         num_losses += self.loss_weight_convnext
        #         loss += self.loss_weight_convnext * convnext_loss

        # weighted avg.
        loss = loss / num_losses
        return loss


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20

    Нет, не совсем так. Это более тонкий момент, который часто вызывает путаницу, особенно когда речь идет о логитах, а не о вероятностях.

    В данной реализации `hinge_d_loss`:

    *   **Позитивное число (например, > 1.0) для `logits_real`:** Дискриминатор считает картинку **настоящей**.
    *   **Негативное число (например, < -1.0) для `logits_fake`:** Дискриминатор считает картинку **фейковой**.

    **Почему так?**

    `logits` - это необработанные выходные значения нейронной сети, *до* применения какой-либо функции активации, которая бы их сжала в диапазон [0, 1] (как Sigmoid) или [0, N] (как Softmax).

    *   В традиционной бинарной классификации (например, с Sigmoid и BCE Loss), если бы мы хотели получить вероятности, мы бы пропустили логиты через Sigmoid: `P = sigmoid(logit)`. Тогда:
        *   `logit = 0` -> `P = 0.5`
        *   `logit = большое_положительное` -> `P = ~1`
        *   `logit = большое_отрицательное` -> `P = ~0`

    *   **Но в Hinge Loss для дискриминатора, мы работаем непосредственно с логитами и устанавливаем "маржу" или "запас":**

        *   **Для `logits_real` (реальные картинки):** Мы хотим, чтобы дискриминатор выдавал значения, **большие 1.0**.
            *   Если `logits_real > 1.0`, то `1.0 - logits_real` будет отрицательным, и `F.relu` вернет 0 (нет штрафа).
            *   Если `logits_real <= 1.0`, то `1.0 - logits_real` будет положительным, и `F.relu` вернет это положительное значение (есть штраф).

        *   **Для `logits_fake` (фейковые картинки):** Мы хотим, чтобы дискриминатор выдавал значения, **меньшие -1.0**.
            *   Если `logits_fake < -1.0`, то `1.0 + logits_fake` будет отрицательным, и `F.relu` вернет 0 (нет штрафа).
            *   Если `logits_fake >= -1.0`, то `1.0 + logits_fake` будет положительным, и `F.relu` вернет это положительное значение (есть штраф).

    **Таким образом, пороги для Hinge Loss составляют:**

    *   **`> 1.0`** для **реальных** (True)
    *   **`< -1.0`** для **фейковых** (False)

    Все, что находится между `-1.0` и `1.0`, а также значения, которые "не дотягивают" до этих порогов, будут приводить к штрафу.

    Это не `0` и `1` как метки классов, а скорее целевые диапазоны для выходных значений сети.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class ReconstructionLoss_Stage2(torch.nn.Module):
    """
    model:
        vq_model:
            quantize_mode: vae

    losses:
        discriminator_start: 0
        discriminator_factor: 1.0
        discriminator_weight: 0.1
        quantizer_weight: 1.0
        perceptual_loss: "lpips"
        perceptual_weight: 1.0
        reconstruction_loss: "l1"
        reconstruction_weight: 1.0
        lecam_regularization_weight: 0.0
        kl_weight: 1e-6
        logvar_init: 0.0
    """

    def __init__(self, config):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config
        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            n_layers=3,
            use_actnorm=False,
        ).apply(weights_init)

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        # просто переусложненное вычисление LPIPS с VGG,
        # который обычно пишется в одну строчку
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.lecam_ema_decay
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

        loss_config = config
        # quantize_mode=vae
        self.quantize_mode = config.quantize_mode

        # by default True
        if self.quantize_mode == "vae":
            self.kl_weight = loss_config.kl_weight
            logvar_init = loss_config.logvar_init
            self.logvar = nn.Parameter(
                torch.ones(size=()) * logvar_init,
                requires_grad=False,
            )

        # coefficient for the projection alignment loss
        self.proj_coef = loss_config.proj_coef

    @autocast(enabled=False)
    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(
                inputs,
                reconstructions,
                extra_result_dict,
                global_step,
            )
        elif mode == "discriminator":
            return self._forward_discriminator(
                inputs,
                reconstructions,
                global_step,
            )
        # elif mode == "generator_alignment":
        #     return self._forward_generator_alignment(
        #         inputs,
        #         reconstructions,
        #         extra_result_dict,
        #         global_step,
        #     )
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int):
        return global_step >= self.discriminator_iter_start

    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step.
        inputs - оригинальные изображения
        reconstructions - реконструированные изображения с прошлого шага
        """
        # дискриминатор стартует с нуля, значит фактор тоже применяется сразу
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True
        # открепляем инпуты и реконструиванные изображения
        real_images = inputs.detach().requires_grad_(True)
        # предсказываем для реальных
        logits_real = self.discriminator(real_images)
        # предсказываем для сгенерированных
        logits_fake = self.discriminator(reconstructions.detach())
        # вычисляем hinge_d_loss, который заставляет быть логиты настоящих картинок положительными
        # ненастоящих отрицательными
        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        # if self.lecam_regularization_weight > 0.0:
        #     lecam_loss = (
        #         compute_lecam_loss(
        #             torch.mean(logits_real),
        #             torch.mean(logits_fake),
        #             self.ema_real_logits_mean,
        #             self.ema_fake_logits_mean,
        #         )
        #         * self.lecam_regularization_weight
        #     )

        #     self.ema_real_logits_mean = (
        #         self.ema_real_logits_mean * self.lecam_ema_decay
        #         + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
        #     )
        #     self.ema_fake_logits_mean = (
        #         self.ema_fake_logits_mean * self.lecam_ema_decay
        #         + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
        #     )

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict

    def _forward_generator(
        self,
        original_image: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        original_image = original_image.contiguous()
        reconstructions = reconstructions.contiguous()
        # by default
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(
                original_image, reconstructions, reduction="mean"
            )
        # elif self.reconstruction_loss == "l2":
        #     reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        # else:
        #     raise ValueError(
        #         f"Unsuppored reconstruction_loss {self.reconstruction_loss}"
        #     )
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(original_image, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=original_image.device)
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0.0
        )
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        if self.quantize_mode == "vq":
            # Compute quantizer loss.
            # quantizer_loss = extra_result_dict["quantizer_loss"]
            # total_loss = (
            #     reconstruction_loss
            #     + self.perceptual_weight * perceptual_loss
            #     + self.quantizer_weight * quantizer_loss
            #     + d_weight * discriminator_factor * generator_loss
            # )
            # loss_dict = dict(
            #     total_loss=total_loss.clone().detach(),
            #     reconstruction_loss=reconstruction_loss.detach(),
            #     perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            #     quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            #     weighted_gan_loss=(
            #         d_weight * discriminator_factor * generator_loss
            #     ).detach(),
            #     discriminator_factor=torch.tensor(discriminator_factor),
            #     commitment_loss=extra_result_dict["commitment_loss"].detach(),
            #     codebook_loss=extra_result_dict["codebook_loss"].detach(),
            #     d_weight=d_weight,
            #     gan_loss=generator_loss.detach(),
            # )
            pass
        elif self.quantize_mode == "vae":
            # Compute kl loss.
            reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            posteriors = extra_result_dict
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            total_loss = (
                reconstruction_loss
                + self.perceptual_weight * perceptual_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            # total_loss = (
            #     self.perceptual_weight * perceptual_loss
            #     + self.kl_weight * kl_loss
            #     + d_weight * discriminator_factor * generator_loss
            # )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                reconstruction_loss=reconstruction_loss.detach(),
                perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
                kl_loss=(self.kl_weight * kl_loss).detach(),
                weighted_gan_loss=(
                    d_weight * discriminator_factor * generator_loss
                ).detach(),
                discriminator_factor=torch.tensor(discriminator_factor).to(
                    generator_loss.device
                ),
                d_weight=torch.tensor(d_weight).to(generator_loss.device),
                gan_loss=generator_loss.detach(),
            )
        else:
            raise NotImplementedError

        return total_loss, loss_dict


def vae_forward(vae: AutoencoderKL, x: Any, return_recon=True):
    posterior = vae.encode(x).latent_dist
    z = posterior.sample()

    recon = vae.decode(
        z / vae.config.scaling_factor,
        False,
    )[0]
    return posterior, z, recon


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


@dataclass
class RepaETrainingArguments:
    quantize_mode = "vae"
    discriminator_start = 0
    discriminator_factor = 1.0
    discriminator_weight = 0.1
    quantizer_weight = 1.0
    perceptual_loss = "lpips"
    perceptual_weight = 1.0
    reconstruction_loss = "l1"
    reconstruction_weight = 1.0
    lecam_regularization_weight = 0.0
    kl_weight = 1e-6
    logvar_init = 0.0
    lecam_ema_decay = 0.99
    proj_coef = 0.0


unet2d_config = {
    "sample_size": 64,
    # "in_channels": 4,
    # "in_channels": 16,
    "in_channels": 32,
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
        # "black-forest-labs/FLUX.2-dev",
        checkpoint_path,
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
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

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
        [250, 500, 750, 1000],
        # [1000],
        device=accelerator.device,
    ).long()
    # weight_dtype = torch.float16
    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        # "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.2-dev",
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    unet = UNet2DModel(**unet2d_config)
    unet.set_attention_backend("flash")

    unet.train()

    # ema = copy.deepcopy(unet).to(
    #     accelerator.device,
    # )

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

    # Only optimize UNet parameters (VAE is frozen)
    layers_to_opt = []
    for n, _p in unet.named_parameters():
        layers_to_opt.append(_p)

    # optimizer = torch.optim.AdamW(
    optimizer_model = bnb.optim.AdamW8bit(
        layers_to_opt,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer_model,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    loss_cfg = RepaETrainingArguments()
    vae_loss_fn = ReconstructionLoss_Stage2(loss_cfg).to(accelerator.device)
    optimizer_vae = bnb.optim.AdamW8bit(
        vae.parameters(),
    )
    optimizer_loss_fn = bnb.optim.AdamW8bit(
        vae_loss_fn.parameters(),
    )
    # Prepare everything with our `accelerator`.
    (
        unet,
        optimizer_model,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_vae,
        optimizer_loss_fn,
    ) = accelerator.prepare(
        unet,
        optimizer_model,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_vae,
        optimizer_loss_fn,
    )

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
            idx = np.random.choice(len(selected_timesteps_tensor), n_samples)

            return selected_timesteps_tensor[idx]

    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, vae, vae_loss_fn]
            # l_acc = [unet]
            with accelerator.accumulate(*l_acc):
                # Convert images to latent space (Bridge Matching approach)
                # with torch.no_grad():
                #     z_source = vae.encode(
                #         batch["source_images"].to(weight_dtype),
                #         return_dict=False,
                #         # )[0]
                #     )[0].sample()
                #     z_source = z_source * vae.config.scaling_factor

                #     z_target = vae.encode(
                #         batch["target_images"].to(weight_dtype),
                #         return_dict=False,
                #         # )[0]
                #     )[0].sample()
                #     z_target = z_target * vae.config.scaling_factor
                vae.train()

                z_source_post, z_source_std, z_source_recon = vae_forward(
                    vae,
                    batch["source_images"].to(weight_dtype),
                )
                z_source_std = z_source_std * vae.config.scaling_factor

                z_target_post, z_target_std, z_target_recon = vae_forward(
                    vae,
                    batch["target_images"].to(weight_dtype),
                )
                z_target_std = z_target_std * vae.config.scaling_factor

                # Turn off grads for the SiT model (avoid REPA gradient on the SiT model)
                requires_grad(unet, False)
                # Avoid BN stats to be updated by the VAE
                unet.eval()
                vae_loss, vae_loss_dict = vae_loss_fn.forward(
                    batch["target_images"],
                    z_target_recon,
                    z_target_post,
                    global_step,
                    "generator",
                )
                vae_loss = vae_loss.mean()

                # Sample timesteps (Bridge Matching)
                timesteps = _timestep_sampling()

                # Get sigmas for the timesteps
                sigmas = _get_sigmas(
                    noise_scheduler,
                    timesteps,
                    n_dim=4,
                    dtype=weight_dtype,
                    device=z_source_std.device,
                )
                # print(sigmas)
                # Create interpolant (Bridge between z_source and z_target)
                # noisy_sample = (
                #     sigmas * z_source_std
                #     + (1.0 - sigmas) * z_target_std
                #     + diffusion_args.bridge_noise_sigma
                #     * (sigmas * (1.0 - sigmas)) ** 0.5
                #     * torch.randn_like(z_source_std)
                # )
                # noisy_sample = z_source

                # Ensure first timestep uses z_source
                # for i, t in enumerate(timesteps):
                #     if t.item() == noise_scheduler.timesteps[0]:
                #         noisy_sample[i] = z_source_std[i]

                # Predict direction of transport (target = z_source - z_target)
                # model_pred = unet(
                #     noisy_sample,
                #     timesteps,
                #     return_dict=False,
                # )[0]

                # Target is the direction from z_source to z_target
                # target = z_source_std - z_target_std

                # denoising_loss = F.mse_loss(
                #     model_pred,
                #     target.detach(),
                #     reduction="mean",
                # )

                # vae_loss = vae_loss + denoising_loss  # * 0.5
                vae_loss = vae_loss
                accelerator.backward(vae_loss)
                if accelerator.sync_gradients:
                    grad_norm_vae = accelerator.clip_grad_norm_(
                        vae.parameters(),
                        1.0,
                    )
                optimizer_vae.step()
                optimizer_vae.zero_grad(set_to_none=True)

                d_loss, d_loss_dict = vae_loss_fn(
                    batch["target_images"],
                    z_target_recon,
                    z_target_post,
                    global_step,
                    "discriminator",
                )
                d_loss = d_loss.mean()
                # обновляем дискриминатор
                accelerator.backward(d_loss)
                if accelerator.sync_gradients:
                    grad_norm_disc = accelerator.clip_grad_norm_(
                        vae_loss_fn.parameters(),
                        1.0,
                    )
                optimizer_loss_fn.step()
                # очищаем дискриминатор
                optimizer_loss_fn.zero_grad(set_to_none=True)

                requires_grad(unet, True)
                unet.train()
                # with torch.no_grad():
                #     noisy_sample = (
                #         sigmas * z_source_std
                #         + (1.0 - sigmas) * z_target_std
                #         + diffusion_args.bridge_noise_sigma
                #         * (sigmas * (1.0 - sigmas)) ** 0.5
                #         * torch.randn_like(z_source_std)
                #     )
                noisy_sample = (
                    sigmas * z_source_std
                    + (1.0 - sigmas) * z_target_std
                    + diffusion_args.bridge_noise_sigma
                    * (sigmas * (1.0 - sigmas)) ** 0.5
                    * torch.randn_like(z_source_std)
                )

                model_pred = unet(
                    noisy_sample.detach(),
                    timesteps,
                    return_dict=False,
                )[0]
                # target = z_source_std.detach() - z_target_std.detach()
                target = z_source_std - z_target_std
                denoising_loss = F.mse_loss(
                    model_pred,
                    target.detach(),
                    # target,
                    reduction="mean",
                )
                # обновляем теперь диффузионную модель
                accelerator.backward(denoising_loss)
                if accelerator.sync_gradients:
                    grad_norm_sit = (
                        accelerator.clip_grad_norm_(unet.parameters(), 1.0),
                    )
                optimizer_model.step()
                lr_scheduler.step()
                optimizer_model.zero_grad(set_to_none=True)

                # if accelerator.sync_gradients:
                #     unwrapped_model = accelerator.unwrap_model(unet)
                #     update_ema(
                #         ema,
                #         unwrapped_model,
                #     )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {}
                # log the loss
                logs["loss"] = denoising_loss.detach().item()
                logs["d_loss"] = d_loss.detach().item()
                # logs["loss_lpips"] = loss_lpips.detach().item()
                logs["vae_loss"] = vae_loss.detach().item()
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
                        unwrap_model(vae).save_pretrained(
                            os.path.join(save_path, "vae")
                        )
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
                    "step_loss": denoising_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step >= training_args.max_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
