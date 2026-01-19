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

import torch
import numpy as np
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from typing import List, Callable, Union, Dict
from collections import OrderedDict, defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# from torchvision import datasets, transforms, utils
import numpy as np
import random
import os
import argparse
import json
from torch.backends import cudnn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from typing import Optional, Tuple, Union
from diffusers.models.unets.unet_2d import UNet2DOutput
from typing import List, Callable, Union, Literal


class Linear:
    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1


class UnifiedSampler(torch.nn.Module):
    """
    UCGM-S: https://arxiv.org/abs/2505.07447
    Credit to https://github.com/LINs-lab/UCGM/blob/main/methodes/unigen.py
    """

    def __init__(self):
        super().__init__()

        transport = Linear()
        self.alpha_in, self.gamma_in = transport.alpha_in, transport.gamma_in
        self.alpha_to, self.gamma_to = transport.alpha_to, transport.gamma_to

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
            self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
            self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def forward(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        tt = tt.flatten()  # if tt is not None else t.clone().flatten()
        dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        q = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        q = q if self.integ_st == 1 else 1 - q
        F_t = (-1) ** (1 - self.integ_st) * model(
            x_t,
            timestep=q,
            timestep2=tt,
            **model_kwargs,
        ).sample
        t = torch.abs(t)
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c

    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 1,
        stochast_ratio: float = 0.8,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.001, 0.7],
        **model_kwargs,
    ):
        """
        Performs unified sampling to generate data samples from the learned distribution.
        """
        input_dtype = inital_noise_z.dtype
        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization.
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])

        # Prepare the buffer for the first order prediction
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # First order prediction
            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                torch.zeros_like(t_cur),
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Apply extrapolation for prediction (extrapolating z is not nessary?)
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            if stochast_ratio == "SDE":
                stochast_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.50), min=0, max=1)
                noi = torch.randn(x_cur.size()).to(x_cur)
            else:
                noi = torch.randn(x_cur.size()).to(x_cur) if stochast_ratio > 0 else 0.0
            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
            )

            # Apply second order correction, Heun-like
            if sampling_order == 2 and i < num_steps - 1:
                x_pri, z_pri, _, _ = self.forward(
                    sampling_model,
                    x_next.to(input_dtype),
                    t_next.to(input_dtype),
                    **model_kwargs,
                )
                x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * z_hat + 0.5 * z_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(input_dtype)


class TwinFlowUNet2DModel(UNet2DModel):
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

        self.time_proj_2 = Timesteps(
            block_out_channels[0],
            flip_sin_to_cos,
            freq_shift,
        )
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        timestep_input_dim = block_out_channels[0]
        self.time_embedding_2 = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    def init_time_proj_2_weights(self):
        missing, unexpected = self.time_proj_2.load_state_dict(
            self.time_proj.state_dict()
        )
        if len(missing) > 0:
            logger.warning(f"Missing keys in time_text_embed state dict: {missing}")
        if len(unexpected) > 0:
            logger.warning(
                f"Unexpected keys in time_text_embed state dict: {unexpected}"
            )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        timestep2: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
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
        # --- new content
        timesteps2 = timestep2
        if not torch.is_tensor(timesteps2):
            timesteps2 = torch.tensor(
                [timesteps2], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps2) and len(timesteps2.shape) == 0:
            timesteps2 = timesteps2[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps2 = timesteps2 * torch.ones(
            sample.shape[0], dtype=timesteps2.dtype, device=timesteps2.device
        )
        # --- new content

        t_emb = self.time_proj(timesteps)
        # ---
        t_emb_2 = self.time_proj_2(timesteps2)
        # ---

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        # ---
        emb_2 = self.time_embedding_2(t_emb_2)
        emb = emb + emb_2 * timestep.unsqueeze(1)
        # ---

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

        # 4. mid
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
            return (sample,)

        return UNet2DOutput(sample=sample)


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


class TwinFlow(torch.nn.Module):
    def __init__(
        self,
        # --- Training Strategy & Consistency Control ---
        consistc_ratio: float = 1.0,
        ema_decay_rate: float = 0.99,
        # --- Enhanced Target Score Mechanism ---
        enhanced_ratio: float = 0.5,
        enhanced_range: List[float] = [0.00, 1.00],
        # --- Time Discretization & Distribution ---
        time_dist_ctrl: List[float] = [1.0, 1.0, 1.0],
        estimate_order: int = 2,
        loss_func_type: dict = {"type": "barron_reweighting"},
        dist_match_cof: float = 0.5,
        use_image_free: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.cor = consistc_ratio
        self.enr = enhanced_ratio
        self.emd = ema_decay_rate
        self.eng = enhanced_range
        self.tdc = time_dist_ctrl
        self.estimate_order = estimate_order
        loss_func_args = loss_func_type.copy()
        self.lft = loss_func_args.pop("type")
        self.lfa = loss_func_args
        self.dmc = dist_match_cof
        self.uif = use_image_free

        assert self.estimate_order >= 0, "Only support estimate_order >= 0"

        self.cmd = 0
        self.l2p = 0
        self.step = 0
        self.mod = None
        self.lsw = None
        self.nge = None

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
            self.alpha_in, self.gamma_in = self.gamma_in, self.alpha_in
            self.alpha_to, self.gamma_to = self.gamma_to, self.alpha_to
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1

    def sample_beta(self, theta_1, theta_2, x):
        size = [x.size(0)] + [1] * (len(x.shape) - 1)
        beta_dist = torch.distributions.Beta(theta_1, theta_2)
        beta_samples = beta_dist.sample(size)
        return beta_samples.to(x)

    def l2_loss(self, pred, target):
        """
        Standard l2 Loss.
        """
        loss = (pred.float() - target.float()) ** 2
        return loss.flatten(1).mean(dim=1).to(pred.dtype)

    def barron_reweighting_loss(self, pred, target, alpha=1.0, c=1e-3):
        """
        Barron-Reweighted L2 Loss (Sample-level + Detach).
        """
        mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.ndim)))
        rmse_sq_norm = mse / (c**2 + 1e-8)  # (x/c)^2

        if abs(alpha - 2.0) < 1e-5:
            weight = torch.ones_like(mse)
        else:
            abs_a_m_2 = abs(alpha - 2.0)
            weight = (rmse_sq_norm / abs_a_m_2 + 1.0).pow((alpha - 2.0) / 2.0)
        loss = mse * weight.data

        return loss / c

    def loss_func(self, pred, target):
        loss_func = {
            "l2": self.l2_loss,
            "barron_reweighting": self.barron_reweighting_loss,
        }[self.lft]

        return loss_func(pred, target, **self.lfa)

    def prepare_inputs(
        self,
        model: nn.Module,
        z: torch.Tensor,
        x: torch.Tensor,
        # c: List[torch.Tensor],
        # e: List[torch.Tensor],
    ):
        # 1. Init time and containers
        bsz, device = x.shape[0], x.device
        assert (
            bsz >= 4
        )  # we need minimal batch=4 to assign different target time, see L132
        t = self.sample_beta(self.tdc[0], self.tdc[1], x).clamp_(0, 1) * self.tdc[2]
        # c = [torch.zeros_like(t)] if c is None else c
        # e = [torch.zeros_like(t)] if e is None else e

        # Aux time variables
        tt = t - torch.rand_like(t) * self.cor * t
        t_min = t - torch.rand_like(t) * t * min(0.05, self.cor)

        probs = {"e2e": 1, "mul": 1, "any": 1, "adv": 1}
        # 3. Partition batch & Create masks
        keys, vals = list(probs.keys()), list(probs.values())
        lens = [int(round(bsz * v / (sum(vals) or 1))) for v in vals]
        lens[-1] = bsz - sum(lens[:-1])  # Adjust last to match bsz

        masks, cursor = {}, 0
        for k, l in zip(keys, lens):
            m = torch.zeros(bsz, dtype=torch.bool, device=device)
            if l > 0:
                m[cursor : cursor + l] = True
            masks[k] = m
            cursor += l

        # 4. Apply specific logic (using walrus operator for compactness)
        if (m := masks.get("e2e")) is not None and m.any():
            t[m], tt[m] = 1.0, 0.0
        if (m := masks.get("mul")) is not None and m.any():
            tt[m] = t_min[m]

        if self.uif:
            x_fake, _, _, _ = self.forward(
                model,
                # torch.randn_like(x),
                # x,
                z,
                torch.ones_like(t),
                torch.zeros_like(t),
                # **dict(c=c),
            )
            z = x_fake.data

        # 5. Adversarial Generation Phase
        if (m := masks.get("adv")) is not None and m.any():
            tt[m] = -t[m]  # Flag logic
            # Generate fake samples (t=1 -> t=0) to replace inputs
            x_fake, _, _, _ = self.forward(
                model,
                # torch.randn_like(x[m]),
                z[m],
                torch.ones_like(t[m]),
                torch.zeros_like(t[m]),
                # **dict(c=[ic[m] for ic in c]),
            )
            x[m] = x_fake.data

        # 6. Construct Training Targets
        # z = torch.randn_like(x)
        # x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        # target = z * self.alpha_to(t) + x * self.gamma_to(t)
        target = z * self.alpha_to(t) + x * self.gamma_to(t)

        # Restore adv time sign for loss/matching logic
        if masks["adv"].any():
            t[masks["adv"]] *= -1
        # return x_t, t, tt, c, e, target, masks
        return x_t, t, tt, target, masks

    @torch.no_grad()
    def get_refer_predc(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        # c: List[torch.Tensor],
        # e: List[torch.Tensor],
    ):
        torch.cuda.set_rng_state(rng_state)
        refer_x, refer_z, refer_v, _ = self.forward(
            model,
            x_t,
            t,
            tt,
            # **dict(c=e),
        )
        torch.cuda.set_rng_state(rng_state)
        predc_x, predc_z, predc_v, _ = self.forward(
            model,
            x_t,
            t,
            tt,
            # **dict(c=c),
        )
        return refer_x, refer_z, refer_v, predc_x, predc_z, predc_v

    # @torch.no_grad()
    # def enhance_target(
    #     self,
    #     target: torch.Tensor,
    #     idx: torch.Tensor,
    #     ratio: float,
    #     pred_w_c: torch.Tensor,
    #     pred_wo_c: torch.Tensor,
    # ):
    #     target[idx] = target[idx] + ratio * (pred_w_c[idx] - pred_wo_c[idx])
    #     # target[~idx] = (target[~idx] + pred_w_c[~idx]) * 0.50
    #     return target

    @torch.no_grad()
    def multi_fwd(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        # c: List[torch.Tensor],
        N: int,
    ):
        pred = 0
        ts = [t * (1 - i / (N)) + tt * (i / (N)) for i in range(N + 1)]
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            torch.cuda.set_rng_state(rng_state)
            hx, hz, F_c, _ = self.forward(
                model,
                x_t,
                t_c,
                t_n,
                # **dict(c=c),
            )
            x_t = self.alpha_in(t_n) * hz + self.gamma_in(t_n) * hx
            pred = pred + F_c * (t_c - t_n)
        return hx, hz, pred

    @torch.no_grad()
    def get_rcgm_target(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        # c: List[torch.Tensor],
        N: int,
    ):
        """
        References:
        - RCGM: https://github.com/LINs-lab/RCGM/blob/main/assets/paper.pdf
        """
        t_m = (t - 0.01).clamp_min(tt)
        x_t = x_t - target * 0.01
        _, _, Ft_tar = self.multi_fwd(
            rng_state,
            model,
            x_t,
            t_m,
            tt,
            # c,
            N,
        )
        mask = t < (tt + 0.01)
        cof_l = torch.where(mask, torch.ones_like(t), 100 * (t - tt))
        cof_r = torch.where(mask, 1 / (t - tt), torch.ones_like(t) * 100)
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - (Ft_tar).clamp(min=-1.0, max=1.0)
        return Ft_tar

    # @torch.no_grad()
    # def update_ema(
    #     self,
    #     model: nn.Module,
    # ):
    #     if self.emd > 0.0 and self.emd < 1.0:
    #         assert hasattr(model, "ema_transformer")
    #         self.mod = self.mod or model.ema_transformer
    #         update_ema(self.mod, model.transformer, decay=self.cmd)
    #         self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
    #     elif self.emd == 0.0:
    #         self.mod = model.transformer
    #     elif self.emd == 1.0:
    #         self.mod = model.ema_transformer
    @torch.no_grad()
    def update_ema(
        self,
        model: nn.Module,
    ):
        """Updates the EMA (Teacher) model."""
        if self.emd > 0.0 and self.emd < 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()
            update_ema(self.mod, model, decay=self.cmd)
            # Warmup logic for EMA decay
            self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
        elif self.emd == 0.0:
            self.mod = model
        elif self.emd == 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()

    @torch.no_grad()
    def dist_match(
        self,
        model: nn.Module,
        z: torch.Tensor,
        x: torch.Tensor,
        # c: List[torch.Tensor],
    ):
        # z = torch.randn_like(x)
        t = self.sample_beta(1.0, 1.0, x)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        fake_s, _, fake_v, _ = self.forward(
            model,
            x_t,
            -t,
            -t,
            # **dict(c=c),
        )
        real_s, _, real_v, _ = self.forward(
            model,
            x_t,
            t,
            t,
            # **dict(c=c),
        )
        F_grad = fake_v - real_v
        x_grad = fake_s - real_s

        F_grad = F_grad.to(torch.float32).clamp(min=-1.0, max=1.0)
        x_grad = x_grad.to(torch.float32).clamp(min=-1.0, max=1.0)
        return x_grad, F_grad

    def training_step(
        self,
        model: Union[nn.Module, Callable],
        x: torch.Tensor,
        edited_image: torch.Tensor,
        # c: List[torch.Tensor],
        # e: List[torch.Tensor],
        # step: int,
        # v: torch.Tensor,
    ):
        with torch.no_grad():
            # x_t, t, tt, c, e, target, sample_masks = self.prepare_inputs(
            x_t, t, tt, target, sample_masks = self.prepare_inputs(
                model,
                x,
                edited_image,
                # c,
                # e,
            )

        loss, rng_state = 0, torch.cuda.get_rng_state()
        x_wc_t, z_wc_t, F_th_t, den_t = self.forward(
            model,
            x_t,
            t,
            tt,
            # **dict(c=c),
        )

        # Start update the state of ema model
        if (self.enr != 0.0) or (self.cor != 0.0) or (self.emd != 0.0):
            self.update_ema(model)

        # Start enhancing target
        # classifier free guidance
        # if (self.estimate_order >= 0) and (self.enr != 0.0):
        #     _, _, refer_v, _, _, predc_v = self.get_refer_predc(
        #         rng_state,
        #         self.mod,
        #         x_t,
        #         t,
        #         t,
        #         c,
        #         e,
        #     )
        #     idx = (t.flatten() < self.eng[1]) & (t.flatten() > self.eng[0])
        #     idx = idx & (~sample_masks["adv"])
        #     target_ = target.clone()
        #     target = self.enhance_target(target, idx, self.enr, predc_v, refer_v)
        #     if self.emd == 1.0:
        #         target[idx] = (target - target_)[idx] + refer_v[idx]

        # Start optimizing RCGM-based or multi-step generation (if self.eso = 0)
        rcgm_idx = sample_masks["e2e"] | sample_masks["any"]
        if (self.estimate_order >= 1) and (self.cor != 0.0) and (rcgm_idx.any()):
            model_4_rcgm = self.mod if (self.emd != 1.0) else model
            rcgm_target = self.get_rcgm_target(
                rng_state,
                model_4_rcgm,
                F_th_t,
                target,
                x_t,
                t,
                tt,
                # c,
                self.estimate_order,
            )
            rcgm_idx = ~(sample_masks["adv"])
            target[rcgm_idx] = rcgm_target[rcgm_idx]

        weighting = (torch.tan((1 - (t.abs() - tt.abs())) * np.pi / 2.5) + 1).flatten()
        weighting[sample_masks["adv"]] = 1
        loss = self.loss_func(F_th_t, target) * weighting
        # Only calculate the multi-step generation loss (if self.eso = 0)
        mul_adv_idx = sample_masks["mul"] | sample_masks["adv"]
        if self.estimate_order >= 1:
            loss = loss.mean()
        elif mul_adv_idx.any():
            loss = loss[mul_adv_idx].mean()
        else:
            loss = 0

        # Start optimizing one or few-step generation
        opt_idx = sample_masks["e2e"]
        if (self.cor == 1.0) and (opt_idx.any()):
            # c_e2e = [ic[opt_idx] for ic in c]
            x_e2e = x_wc_t[opt_idx]
            # _, F_grad = self.dist_match(model, x_e2e, c_e2e)
            _, F_grad = self.dist_match(
                model,
                x_e2e,
                edited_image,
            )
            F_fake = F_th_t[opt_idx]
            e2e_loss = self.loss_func(F_fake, (F_fake - F_grad).data)
            loss = loss + e2e_loss.mean() * self.dmc
        return loss

    def forward(
        self,
        model: TwinFlowUNet2DModel,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        tt = tt.flatten()  # if tt is not None else t.clone().flatten()
        # dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        dent = -1
        q = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        q = q if self.integ_st == 1 else 1 - q
        F_t = (-1) ** (1 - self.integ_st) * model(
            x_t,
            timestep=q,
            timestep2=tt,
            **model_kwargs,
        ).sample
        t = torch.abs(t)
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c

    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 20,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.0, 0.0],
        sampling_style: Literal["few", "mul", "any"] = "few",
        **model_kwargs,
    ):
        """
        References:
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447 (Unified Continuous Generative Models)
        """
        input_dtype = inital_noise_z.dtype
        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization.
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])

        # Prepare the buffer for the first order prediction
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # First order prediction
            if sampling_style == "few":
                t_tgt = torch.zeros_like(t_cur)
            elif sampling_style == "mul":
                t_tgt = t_cur
            elif sampling_style == "any":
                t_tgt = t_next
            else:
                raise ValueError(f"Unknown sampling style: {sampling_style}")

            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                t_tgt.to(input_dtype),
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Apply extrapolation for prediction (extrapolating z is not nessary?)
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            if stochast_ratio == "SDE":
                stochast_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.50), min=0, max=1)
                noi = torch.randn(x_cur.size()).to(x_cur)
            else:
                noi = torch.randn(x_cur.size()).to(x_cur) if stochast_ratio > 0 else 0.0
            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
            )

            # Apply second order correction, Heun-like
            if sampling_order == 2 and i < num_steps - 1:
                x_pri, z_pri, _, _ = self.forward(
                    sampling_model,
                    x_next.to(input_dtype),
                    t_next.to(input_dtype),
                    **model_kwargs,
                )
                x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * z_hat + 0.5 * z_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(input_dtype)


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
    trainer: TwinFlow = None,
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
    # unet_val = UNet2DModel.from_pretrained(
    unet_val = TwinFlowUNet2DModel.from_pretrained(
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
            # for i in range(num_steps):
            #     t = noise_scheduler.timesteps[i]
            #     # 1. Масштабирование входа (если требуется шедулером)
            #     if hasattr(noise_scheduler, "scale_model_input"):
            #         denoiser_input = noise_scheduler.scale_model_input(sample, t)
            #     else:
            #         denoiser_input = sample

            #     # 2. Предсказание направления (UNet)
            #     # unet_val(x, t) -> output
            #     # print(i, t, noise_scheduler.timesteps)
            #     pred = unet_val(
            #         denoiser_input,
            #         t.to(z_source.device).repeat(denoiser_input.shape[0]),
            #         torch.zeros_like(
            #             t.to(z_source.device).repeat(denoiser_input.shape[0])
            #         ),
            #         return_dict=False,
            #     )[0]

            #     # 3. Шаг диффузии (Reverse Process)
            #     sample = noise_scheduler.step(pred, t, sample, return_dict=False)[0]

            #     # 4. Добавление стохастичности (Bridge Noise)
            #     # Не добавляем шум после последнего шага
            #     if i < len(noise_scheduler.timesteps) - 1:
            #         # Получаем таймстемп следующего шага
            #         next_timestep = (
            #             noise_scheduler.timesteps[i + 1]
            #             .to(z_source.device)
            #             .repeat(sample.shape[0])
            #         )

            #         # Получаем сигму для следующего шага
            #         sigmas_next = _get_sigmas_val(
            #             noise_scheduler,
            #             next_timestep,
            #             n_dim=4,
            #             dtype=weight_dtype,
            #             device=z_source.device,
            #         )

            #         # Формула Bridge Matching: шум пропорционален sqrt(sigma * (1-sigma))
            #         noise = torch.randn_like(sample)
            #         bridge_factor = (sigmas_next * (1.0 - sigmas_next)) ** 0.5

            #         sample = (
            #             sample
            #             + diffusion_args.bridge_noise_sigma * bridge_factor * noise
            #         )
            #         sample = sample.to(z_source.dtype)

            # ---------------------------------------------------------
            # sample = (
            #     trainer.sampling_loop(
            #         inital_noise_z=sample,
            #         sampling_model=unet_val,
            #         sampling_steps=1,
            #     )[-1]
            #     .to(vae_val.dtype)
            #     .to(vae_val.device)
            # )
            sample = (
                UnifiedSampler()
                .sampling_loop(
                    inital_noise_z=sample,
                    sampling_model=unet_val,
                )[-1]
                .to(vae_val.dtype)
                .to(vae_val.device)
            )

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

    # sigmas = np.linspace(1.0, 1 / num_steps, num_steps)

    # noise_scheduler.set_timesteps(sigmas=sigmas, device=accelerator.device)

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

    # unet = UNet2DModel(**unet2d_config)
    unet = TwinFlowUNet2DModel(**unet2d_config)
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
        drop_last=True,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = (
        training_args.warmup_steps * accelerator.num_processes
    )

    num_training_steps_for_scheduler = (
        training_args.max_steps * accelerator.num_processes
    )

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

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

    trainer = TwinFlow()

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

                loss = trainer.training_step(
                    unet,
                    z_source,
                    z_target,
                )
                # вычисляем градиент
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        training_args.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {}
                # log the loss
                logs["loss"] = loss.detach().item()

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
                            trainer=trainer,
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
