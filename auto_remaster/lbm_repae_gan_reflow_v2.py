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
from collections import namedtuple

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
    AutoencoderKLFlux2,
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
from diffusers.models.unets.unet_2d import UNet2DOutput
from typing import List, Callable, Union, Literal, Optional
from torchvision.transforms import Normalize
import torch.distributed as dist


def vae_loss_function(x, x_reconstructed, encoder_latent, do_pool=True, do_recon=False):
    # downsample images by factor of 8
    # всегда False, значит эта ветка не работает
    if do_recon:
        # if do_pool:
        #     x_reconstructed_down = F.interpolate(
        #         x_reconstructed, scale_factor=1 / 16, mode="area"
        #     )
        #     x_down = F.interpolate(x, scale_factor=1 / 16, mode="area")
        #     recon_loss = ((x_reconstructed_down - x_down)).abs().mean()
        # else:
        #     x_reconstructed_down = x_reconstructed
        #     x_down = x

        #     recon_loss = (
        #         ((x_reconstructed_down - x_down) * blurriness_heatmap(x_down))
        #         .abs()
        #         .mean()
        #     )
        #     recon_loss_item = recon_loss.item()
        pass
    else:
        recon_loss = 0
        recon_loss_item = 0

    elewise_mean_loss = encoder_latent.pow(2)
    zloss = elewise_mean_loss.mean()

    with torch.no_grad():
        actual_mean_loss = elewise_mean_loss.mean()
        actual_ks_loss = actual_mean_loss.mean()
    # такой трюк когда мы возводим латенты в степень, а затем придаем им большой вес
    # лишь говорят о том чтобы латенты держались близко около 0, по абсолютному значению мы не
    # хотим чтобы они улетали слишком далеко
    vae_loss = recon_loss * 0.0 + zloss * 0.1
    return vae_loss, {
        "recon_loss": recon_loss_item,
        "kl_loss": actual_ks_loss.item(),
        "average_of_abs_z": encoder_latent.abs().mean().item(),
        "std_of_abs_z": encoder_latent.abs().std().item(),
        "average_of_logvar": 0.0,
        "std_of_logvar": 0.0,
    }


@torch.no_grad()
def avg_scalar_over_nodes(value: float, device):
    value = torch.tensor(value, device=device)
    # dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value.item()


class GradNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(weight)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        weight = ctx.saved_tensors[0]

        grad_output_norm = torch.norm(grad_output).mean().item()
        # nccl over all nodes
        grad_output_norm = avg_scalar_over_nodes(
            grad_output_norm, device=grad_output.device
        )

        grad_output_normalized = weight * grad_output / (grad_output_norm + 1e-8)

        return grad_output_normalized, None


def gradnorm(x, weight=1.0):
    weight = torch.tensor(weight, device=x.device)
    return GradNormFunction.apply(x, weight)


def gan_disc_loss(real_preds, fake_preds, disc_type="bce"):
    if disc_type == "bce":
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_preds, torch.ones_like(real_preds)
        )
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_preds, torch.zeros_like(fake_preds)
        )
        # eval its online performance
        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

    if disc_type == "hinge":
        real_loss = nn.functional.relu(1 - real_preds).mean()
        fake_loss = nn.functional.relu(1 + fake_preds).mean()

        with torch.no_grad():
            acc = (real_preds > 0).sum().item() + (fake_preds < 0).sum().item()
            acc = acc / (real_preds.numel() + fake_preds.numel())

        avg_real_preds = real_preds.mean().item()
        avg_fake_preds = fake_preds.mean().item()

    return (real_loss + fake_loss) * 0.5, avg_real_preds, avg_fake_preds, acc


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=pretrained
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        try:
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))
        except:
            print("Failed to load vgg.pth, downloading...")
            os.system(
                # "wget https://heibox.uni-heidelberg.de/seafhttp/files/9535cbee-6558-4c0c-8743-78f5e56ea75e/vgg.pth"
                "wget https://huggingface.co/xingjianleng/pretrained_vae/resolve/main/lpips/vgg.pth"
            )
            data = torch.load("vgg.pth", map_location=torch.device("cpu"))

        self.load_state_dict(
            data,
            strict=False,
        )

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
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


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.scaling_layer = ScalingLayer()

        _vgg = torchvision.models.vgg16(pretrained=True)

        self.slice1 = nn.Sequential(_vgg.features[:4])
        self.slice2 = nn.Sequential(_vgg.features[4:9])
        self.slice3 = nn.Sequential(_vgg.features[9:16])
        self.slice4 = nn.Sequential(_vgg.features[16:23])
        self.slice5 = nn.Sequential(_vgg.features[23:30])

        self.binary_classifier1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=4, stride=4, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier1[-1].weight)

        self.binary_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=4, stride=4, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier2[-1].weight)

        self.binary_classifier3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier3[-1].weight)

        self.binary_classifier4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=2, stride=2, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier4[-1].weight)

        self.binary_classifier5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        nn.init.zeros_(self.binary_classifier5[-1].weight)

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


# from .discriminator import NLayerDiscriminator, weights_init
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_raw_image(x, enc_type="dinov2"):
    # dinov2-vit-b
    resolution = x.shape[-1]
    if "dinov2" in enc_type:
        # x = x / 255.0
        x = x * 0.5 + 0.5
        x = Normalize(_IMAGENET_MEAN, _IMAGENET_STD)(x)
        x = torch.nn.functional.interpolate(
            # x, 224 * (resolution // 256), mode="bicubic"
            x,
            224,
            mode="bicubic",
        )

    return x


def build_mlp(hidden_size, projector_dim, z_dim):
    # hidden_size=1152, projector_dim=2048,z_dim=768
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class REPAEUNet2DModel(UNet2DModel):
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            center_input_sample=center_input_sample,
            time_embedding_type=time_embedding_type,
            time_embedding_dim=time_embedding_dim,
            freq_shift=freq_shift,
            flip_sin_to_cos=flip_sin_to_cos,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            mid_block_scale_factor=mid_block_scale_factor,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            dropout=dropout,
            act_fn=act_fn,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            attn_norm_num_groups=attn_norm_num_groups,
            norm_eps=norm_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=add_attention,
            class_embed_type=class_embed_type,
            num_class_embeds=num_class_embeds,
            num_train_timesteps=num_train_timesteps,
        )

        z_dims = [768]
        self.repa_projector = build_mlp(block_out_channels[-1], 2048, 768)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        use_repa: bool = False,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )
        # reverse time

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError(
                "class_embedding needs to be initialized in order to use class conditioning"
            )

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid # sample [1, 1280, 16, 16]
        # ---
        if use_repa:
            # sample сейчас имеет форму [Batch, 1280, 16, 16] (например)

            # 1. Превращаем "картинку" в последовательность токенов
            # Permute: [B, C, H, W] -> [B, H, W, C]
            h, w = sample.shape[-2], sample.shape[-1]
            hidden = sample.permute(0, 2, 3, 1)

            # Flatten: [B, H, W, C] -> [B, H*W, C] -> [B, 256, 1280]
            # Но лучше сохранить форму [B, H, W, C] до интерполяции,
            # либо интерполировать ДО permute (это эффективнее).
            hidden = hidden.reshape(-1, 256, 1280)
            # ВАРИАНТ: Проецируем сразу каждый пиксель (так как MLP работает с последней размерностью)
            repa_features = self.repa_projector(hidden)  # Выход: [B, H, W, DinoDim]

            # Теперь у нас [B, 16, 16, 768].
            # Нам нужно будет выровнять это с DINO снаружи.

        # ---

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample, res_samples, emb, skip_sample
                )
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        if not return_dict:
            if use_repa:
                return (sample, repa_features)
            return (sample,)

        return UNet2DOutput(sample=sample)


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

        # lpips by defaults in repa-e
        if "lpips" in model_name:
            self.lpips = lpips.LPIPS(net="vgg").eval()

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
            # n_layers=1,
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
        # if self.reconstruction_loss == "l1":
        #     reconstruction_loss = F.l1_loss(
        #         original_image, reconstructions, reduction="mean"
        #     )
        # elif self.reconstruction_loss == "l2":
        #     reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        # else:
        #     raise ValueError(
        #         f"Unsuppored reconstruction_loss {self.reconstruction_loss}"
        #     )
        # reconstruction_loss *= self.reconstruction_weight

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
            # reconstruction_loss = reconstruction_loss / torch.exp(self.logvar)
            posteriors = extra_result_dict
            kl_loss = posteriors.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            # total_loss = (
            #     reconstruction_loss
            #     + self.perceptual_weight * perceptual_loss
            #     + self.kl_weight * kl_loss
            #     + d_weight * discriminator_factor * generator_loss
            # )
            total_loss = (
                self.perceptual_weight * perceptual_loss
                + self.kl_weight * kl_loss
                + d_weight * discriminator_factor * generator_loss
            )
            loss_dict = dict(
                total_loss=total_loss.clone().detach(),
                # reconstruction_loss=reconstruction_loss.detach(),
                reconstruction_loss=torch.tensor(0.0).detach(),
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
        z,
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
# unet2d_config = {
#     "sample_size": 64,  # Для латента FLUX (1024 / 16 = 64) или (512 / 8 = 64)
#     "in_channels": 32,  # FLUX VAE channels
#     "out_channels": 32,
#     "center_input_sample": False,
#     "time_embedding_type": "positional",
#     "freq_shift": 0,
#     "flip_sin_to_cos": True,
#     # 1. УЗКАЯ ШИРИНА: Экономим память и шину данных.
#     # 3060/4060 любят, когда каналов не слишком много (до 512 - ок).
#     "block_out_channels": (256, 256, 512, 512),
#     # 2. ПОЛНЫЙ ОТКАЗ ОТ ВНИМАНИЯ (Down/Up Blocks)
#     # Используем только сверточные блоки. Никаких AttnDownBlock2D.
#     "down_block_types": (
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",
#         "DownBlock2D",  # Даже на самом дне только свертки
#     ),
#     "up_block_types": (
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#         "UpBlock2D",
#     ),
#     # Обычный мид-блок, внимание отключим флагом ниже
#     "mid_block_type": "UNetMidBlock2D",
#     # 3. БОЛЬШАЯ ГЛУБИНА: Компенсируем отсутствие ширины и внимания.
#     # layers_per_block=3 означает, что на каждом разрешении будет 3 ResNet блока.
#     # Это дает сети "время подумать" (эмуляция шагов ODE).
#     "layers_per_block": 3,
#     "mid_block_scale_factor": 1,
#     "downsample_padding": 1,
#     "downsample_type": "conv",  # Conv лучше сохраняет детали при сжатии
#     "upsample_type": "conv",
#     "dropout": 0.0,
#     "act_fn": "silu",
#     "norm_num_groups": 32,  # GroupNorm отлично работает с батчами
#     "norm_eps": 1e-05,
#     "resnet_time_scale_shift": "scale_shift",  # Лучше для генерации, чем default
#     # !!! ГЛАВНЫЙ БУСТ СКОРОСТИ !!!
#     # Выключает внимание в MidBlock. В Down/Up его и так нет из-за типов блоков выше.
#     "add_attention": False,
# }

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
        torch_dtype=weight_dtype,
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
                sample = unet_val(
                    denoiser_input,
                    t.to(z_source.device).repeat(denoiser_input.shape[0]),
                    return_dict=False,
                )[0]

                # 3. Шаг диффузии (Reverse Process)
                # sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]

                # 4. Добавление стохастичности (Bridge Noise)
                # Не добавляем шум после последнего шага
                # if i < len(noise_scheduler.timesteps) - 1:
                #     # Получаем таймстемп следующего шага
                #     next_timestep = (
                #         noise_scheduler.timesteps[i + 1]
                #         .to(z_source.device)
                #         .repeat(sample.shape[0])
                #     )

                #     # Получаем сигму для следующего шага
                #     sigmas_next = _get_sigmas_val(
                #         noise_scheduler,
                #         next_timestep,
                #         n_dim=4,
                #         dtype=weight_dtype,
                #         device=z_source.device,
                #     )

                #     # Формула Bridge Matching: шум пропорционален sqrt(sigma * (1-sigma))
                #     noise = torch.randn_like(sample)
                #     bridge_factor = (sigmas_next * (1.0 - sigmas_next)) ** 0.5

                #     sample = (
                #         sample
                #         + diffusion_args.bridge_noise_sigma * bridge_factor * noise
                #     )
                #     sample = sample.to(z_source.dtype)

            # ---------------------------------------------------------

            # Декодирование результата
            output_image = (
                vae_val.decode(
                    (z_source - sample) / vae_val.config.scaling_factor,
                    # sample,
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
        # mixed_precision="bf16",
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
    noise_scheduler = FlowMatchEulerDiscreteScheduler()
    num_steps = diffusion_args.num_inference_steps

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)
    selected_timesteps_tensor = torch.tensor(
        # [250, 500, 750, 1000],
        [1000],
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
    vae.requires_grad_(False)
    vae = vae.eval()
    # print(vae)
    # new_state = {}
    # state = torch.load(
    #     "/code/auto_remaster/sandbox/vqgan_training/ckpt/stage_4_msepool-cont-512-1.0-1.0-batch-gradnorm_make_deterministic_test_flux2/vae_epoch_14_step_12001.pt",
    #     weights_only=True,
    # )
    # for key in state.keys():
    #     new_state[key.replace("module.", "")] = state[key]
    # vae.load_state_dict(
    #     new_state,
    #     strict=False,
    # )

    # unet = UNet2DModel(**unet2d_config)
    unet = REPAEUNet2DModel(**unet2d_config)
    # unet.set_attention_backend("flash")
    unet = unet.to(weight_dtype)
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
    # vae_loss_fn = ReconstructionLoss_Stage2(loss_cfg).to(accelerator.device)
    optimizer_vae = bnb.optim.AdamW8bit(
        vae.parameters(),
    )
    # optimizer_loss_fn = bnb.optim.AdamW8bit(
    #     vae_loss_fn.parameters(),
    # )

    discriminator = PatchDiscriminator()
    discriminator.requires_grad_(True)
    lpips = LPIPS().cuda()
    lpips.requires_grad_(True)

    optimizer_D = bnb.optim.AdamW8bit(
        discriminator.parameters(),
        lr=1e-5,
        weight_decay=1e-3,
        betas=(0.9, 0.95),
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        optimizer_model,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_vae,
        # optimizer_loss_fn,
        discriminator,
        optimizer_D,
    ) = accelerator.prepare(
        unet,
        optimizer_model,
        train_dataloader,
        lr_scheduler,
        vae,
        optimizer_vae,
        # optimizer_loss_fn,
        discriminator,
        optimizer_D,
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

    import timm

    encoder = torch.hub.load("facebookresearch/dinov2", f"dinov2_vitb14")
    del encoder.head
    patch_resolution = 16 * (512 // 256)
    encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
        encoder.pos_embed.data,
        [patch_resolution, patch_resolution],
    )
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(accelerator.device).to(weight_dtype)
    encoder.eval()
    encoder.requires_grad_(False)

    lecam_loss_weight = 0.1
    lecam_anchor_real_logits = 0.0
    lecam_anchor_fake_logits = 0.0
    lecam_beta = 0.9
    disc_type = "bce"
    for epoch in range(first_epoch, training_args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # l_acc = [unet, vae, discriminator]

            # with torch.no_grad():
            #     dino_features = []

            #     z = encoder.forward_features(
            #         preprocess_raw_image(
            #             batch["target_images"].to(weight_dtype),
            #         )
            #     )
            #     z = z["x_norm_patchtokens"].to(weight_dtype)
            #     # torch.Size([1, 1024, 768])
            #     dino_features.append(z)
            # with accelerator.accumulate(*l_acc):
            #     vae.train()

            #     z_source_post, z_source_std, z_source_recon = vae_forward(
            #         vae,
            #         batch["source_images"].to(weight_dtype),
            #     )
            #     z_source_std = z_source_std * vae.config.scaling_factor

            #     z_target_post, z_target_std, z_target_recon = vae_forward(
            #         vae,
            #         batch["target_images"].to(weight_dtype),
            #     )
            #     z_target_std = z_target_std * vae.config.scaling_factor

            #     target = z_source_std - z_target_std

            #     # получаем предсказание для реальных
            #     real_preds = discriminator(batch["target_images"])
            #     ###----
            #     requires_grad(unet, False)
            #     # # Avoid BN stats to be updated by the VAE
            #     unet.eval()
            #     # Sample timesteps (Bridge Matching)
            #     timesteps = _timestep_sampling()

            #     # Get sigmas for the timesteps
            #     sigmas = _get_sigmas(
            #         noise_scheduler,
            #         timesteps,
            #         n_dim=4,
            #         dtype=weight_dtype,
            #         device=z_source_std.device,
            #     )

            #     noisy_sample = z_source_std

            #     model_pred, repa_mlp_features = unet(
            #         noisy_sample,
            #         timesteps,
            #         return_dict=False,
            #         use_repa=True,
            #     )

            #     proj_loss = -F.cosine_similarity(
            #         dino_features[0], repa_mlp_features, dim=-1
            #     ).mean()

            #     ###----

            #     # предсказание для декодированных
            #     # открепляем, чтобы при обучении дискриминатора не обучался сам
            #     # декодер
            #     fake_preds = discriminator(z_target_recon.detach())
            #     # disc_type=BCE по умолчанию
            #     # Дискриминатор штрафуют, если он говорит, что реальная картинка — фейк,
            #     # или фейковая — реал.
            #     d_loss, avg_real_logits, avg_fake_logits, disc_acc = gan_disc_loss(
            #         real_preds,
            #         fake_preds,
            #     )

            #     # Regularizing Generative Adversarial Networks under Limited Data https://arxiv.org/pdf/2104.03310
            #     # это техника не дает дискриминатору «зазнаваться» и выдавать слишком уверенные ответы
            #     # Что это: Это Экспоненциальное Скользящее Среднее (EMA).
            #     # Зачем: Мы запоминаем, какое среднее значение дискриминатор обычно выдает для реальных и фейковых
            #     # картинок на протяжении всего обучения. Это создает стабильные «якоря» или ориентиры.
            #     # тоже самое для фейков
            #     # lecam_loss_weight = 0.1
            #     # lecam_anchor_real_logits = 0.0
            #     # lecam_anchor_fake_logits = 0.0
            #     # lecam_beta = 0.9
            #     lecam_anchor_real_logits = (
            #         lecam_beta * lecam_anchor_real_logits
            #         + (1 - lecam_beta) * avg_real_logits
            #     )
            #     lecam_anchor_fake_logits = (
            #         lecam_beta * lecam_anchor_fake_logits
            #         + (1 - lecam_beta) * avg_fake_logits
            #     )
            #     total_d_loss = d_loss.mean()
            #     d_loss_item = total_d_loss.item()
            #     # по умолчанию True
            #     if True:
            #         # penalize the real logits to fake and fake logits to real.
            #         lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(
            #             2
            #         ).mean() + (fake_preds - lecam_anchor_real_logits).pow(2).mean()
            #         lecam_loss_item = lecam_loss.item()
            #         total_d_loss = total_d_loss + lecam_loss * lecam_loss_weight

            #     optimizer_D.zero_grad()
            #     # сохраняем градиенты, если real_preds будут использоваться дальше
            #     # так как после вызова backward весь граф уничтожается
            #     total_d_loss += proj_loss
            #     # total_d_loss.backward(retain_graph=True)
            #     accelerator.backward(total_d_loss, retain_graph=True)
            #     optimizer_D.step()

            #     # unnormalize the images, and perceptual loss
            #     _recon_for_perceptual = gradnorm(z_target_recon)

            #     # Мы не хотим попиксельного совпадения (это дает мыло). Мы хотим, чтобы нейросеть VGG (внутри LPIPS)
            #     # "видела" на обеих картинках одинаковые объекты и текстуры.
            #     percep_rec_loss = lpips(
            #         _recon_for_perceptual, batch["target_images"]
            #     ).mean()

            #     # mse, vae loss.
            #     recon_for_mse = gradnorm(z_target_recon, weight=0.001)
            #     # ничего не делаем, кроме того что заставляем латенты быть близко к 0 по абслютному значению
            #     vae_loss, loss_data = vae_loss_function(
            #         batch["target_images"],
            #         recon_for_mse,
            #         z_target_std,
            #     )

            # ### ---
            # requires_grad(unet, True)
            # unet.train()

            # model_pred, repa_mlp_features = unet(
            #     noisy_sample.detach(),
            #     timesteps,
            #     return_dict=False,
            #     use_repa=True,
            # )

            # denoising_loss = (
            #     F.mse_loss(
            #         model_pred,
            #         target.detach(),
            #         # target,
            #         reduction="mean",
            #     )
            #     * 0.5
            # )
            # proj_loss = -F.cosine_similarity(
            #     dino_features[0],
            #     repa_mlp_features,
            #     dim=-1,
            # ).mean()
            # denoising_loss += proj_loss * 0.1

            # recon = vae.decode(
            #     z_source_std - model_pred,
            #     False,
            # )[0]
            # percep_rec_loss = lpips(recon, batch["target_images"]).mean()
            # denoising_loss += percep_rec_loss

            # # обновляем теперь диффузионную модель
            # accelerator.backward(denoising_loss, retain_graph=True)

            # # gan loss
            # if global_step >= 0:
            #     # нормализуем реконструкцию с обучаемого декодера
            #     recon_for_gan = gradnorm(z_target_recon, weight=1.0)
            #     # пропускаем нормализованную через дискриминатор
            #     fake_preds = discriminator(recon_for_gan)
            #     real_preds_const = real_preds.clone().detach()
            #     # по умолчанию данная ветка
            #     if disc_type == "bce":
            #         # заставляем предсказывать только для фейковых
            #         g_gan_loss = nn.functional.binary_cross_entropy_with_logits(
            #             fake_preds, torch.ones_like(fake_preds)
            #         )
            #     # elif disc_type == "hinge":
            #     #     g_gan_loss = -fake_preds.mean()
            #     # соединяем лосс с LPIPS, с дискриминатора и просто возведение в степень латентов
            #     # для регуляризации
            #     overall_vae_loss = percep_rec_loss + g_gan_loss + vae_loss
            #     g_gan_loss = g_gan_loss.item()
            # else:
            #     overall_vae_loss = percep_rec_loss + vae_loss
            #     g_gan_loss = 0.0

            # # overall_vae_loss.backward()
            # accelerator.backward(overall_vae_loss)
            # optimizer_vae.step()
            # optimizer_model.step()
            # optimizer_vae.zero_grad()
            # # optimizer_model.zero_grad(set_to_none=True)
            # optimizer_model.zero_grad()
            # lr_scheduler.step()
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            ############
            #################
            #################
            ############
            with torch.no_grad():
                # 1. Получаем латент source (условие/вход для UNet)
                # Предполагаем, что UNet делает mapping: Source Latent -> Target Latent за 1 шаг
                z_source_dist = vae.encode(
                    batch["source_images"].to(weight_dtype)
                ).latent_dist
                z_source = z_source_dist.sample() * vae.config.scaling_factor

                # 2. Подготовка таймстепов для одношаговой генерации
                # Используем t=0 (или минимальный t), так как предсказываем чистое изображение
                timesteps = (
                    torch.ones(
                        z_source.shape[0],
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    * 1000
                )

            # --- Шаг 1: Обучение Дискриминатора ---
            # Замораживаем UNet, размораживаем Дискриминатор
            # requires_grad(unet, False)
            requires_grad(discriminator, True)

            # Генерация фейка (без градиентов для генератора)

            # Предсказание латента из UNet
            z_pred_for_d = unet(
                z_source,
                timesteps,
                return_dict=False,
                use_repa=False,
            )[0]
            # Накапливаем градиенты для D (если gradient_accumulation > 1)
            with accelerator.accumulate(discriminator):
                # Декодирование в пиксели для дискриминатора
                # Используем .sample, так как AutoencoderKL возвращает DecoderOutput
                with torch.no_grad():
                    fake_image_d = vae.decode(
                        (z_source - z_pred_for_d) / vae.config.scaling_factor,
                        return_dict=False,
                    )[0]

                # Предсказания дискриминатора
                real_preds = discriminator(batch["target_images"])
                fake_preds = discriminator(
                    fake_image_d.detach()
                )  # Detach на всякий случай, хотя мы в no_grad

                # Расчет GAN Loss (Hinge или BCE)
                d_loss, avg_real_logits, avg_fake_logits, disc_acc = gan_disc_loss(
                    real_preds, fake_preds, disc_type=disc_type
                )

                # LeCam Regularization (из VAE скрипта)
                # Обновляем EMA якоря
                lecam_anchor_real_logits = (
                    lecam_beta * lecam_anchor_real_logits
                    + (1 - lecam_beta) * avg_real_logits
                )
                lecam_anchor_fake_logits = (
                    lecam_beta * lecam_anchor_fake_logits
                    + (1 - lecam_beta) * avg_fake_logits
                )

                total_d_loss = d_loss.mean()

                # Добавляем LeCam loss
                lecam_loss = (real_preds - lecam_anchor_fake_logits).pow(2).mean() + (
                    fake_preds - lecam_anchor_real_logits
                ).pow(2).mean()
                total_d_loss += lecam_loss * lecam_loss_weight

                # Обратное распространение для D
                accelerator.backward(total_d_loss)
                optimizer_D.step()
                optimizer_D.zero_grad()

            # --- Шаг 2: Обучение Генератора (UNet) ---
            # Размораживаем UNet, замораживаем Дискриминатор (чтобы не тратить память на его градиенты)
            requires_grad(unet, True)
            requires_grad(discriminator, False)

            with accelerator.accumulate(unet):
                # Прямой проход UNet (с градиентами)
                z_pred = unet(
                    z_source,
                    timesteps,
                    return_dict=False,
                    use_repa=False,
                )[0]

                # Декодирование (градиенты текут СКВОЗЬ декодер к z_pred, но веса декодера не меняются)
                fake_image_g = vae.decode(
                    (z_source - z_pred) / vae.config.scaling_factor, return_dict=False
                )[0]

                # Применяем GradNorm для стабилизации (как в VAE скрипте)
                # Это нормализует градиенты перед попаданием в перцептуал/GAN лоссы
                fake_image_g_norm = gradnorm(fake_image_g, weight=1.0)

                # 1. LPIPS Loss (Основная метрика качества)
                # Сравниваем декодированный предсказанный латент с реальной целевой картинкой
                loss_lpips = lpips(fake_image_g_norm, batch["target_images"]).mean()

                # 2. Generator GAN Loss (Обман дискриминатора)
                # Нормируем вход для дискриминатора так же, как в VAE скрипте
                recon_for_gan = gradnorm(fake_image_g, weight=1.0)
                g_fake_preds = discriminator(recon_for_gan)

                if disc_type == "bce":
                    loss_gan = nn.functional.binary_cross_entropy_with_logits(
                        g_fake_preds, torch.ones_like(g_fake_preds)
                    )

                loss_z_reg = z_pred.pow(2).mean()

                # Итоговый лосс генератора
                # Мы НЕ используем MSE на латентах, как и требовалось.
                # LPIPS тянет текстуры, GAN тянет реализм
                total_gen_loss = loss_lpips * 1.0 + loss_gan * 0.1 + loss_z_reg * 0.0001

                # Обратное распространение для G
                accelerator.backward(total_gen_loss)

                # Клиппинг градиентов UNet
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer_model.step()
                optimizer_model.zero_grad()
                lr_scheduler.step()
            #####
            ########d#######
            ################
            #####

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {}
                # log the loss
                logs["loss"] = total_gen_loss.detach().item()
                # logs["d_loss"] = d_loss.detach().item()
                logs["loss_lpips"] = loss_lpips.detach().item()
                # logs["vae_loss"] = vae_loss.detach().item()
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
                    # "step_loss": denoising_loss.detach().item(),
                    "step_loss": total_gen_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step >= training_args.max_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
