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
from torchvision import datasets, transforms, utils
import numpy as np
import random
import os
import argparse
import json
from torch.backends import cudnn


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
    """
    Recursive Consistent Generation Model (RCGM).

    This class implements the backbone for 'Any-step Generation via N-th Order
    Recursive Consistent Velocity Field Estimation'. It serves as the foundation
    for consistency training by estimating higher-order trajectories.

    References:
        - RCGM: https://github.com/LINs-lab/RCGM/blob/main/assets/paper.pdf
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447 (Unified Continuous Generative Models)
    """

    def __init__(
        self,
        ema_decay_rate: float = 0.99,  # Recomended: >=0.99 for estimate_order >=2
        estimate_order: int = 2,
        enhanced_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.emd = ema_decay_rate
        # default 2
        self.estimation_order = estimate_order  # N-th order estimation (RCGM paper)

        assert self.estimation_order >= 1, "Only support estimate_order >= 1"

        self.cmd = 0
        self.mod = None  # EMA Model container
        self.enr = enhanced_ratio  # CFG Guidance ratio
        self.utf = True

    # ------------------------------------------------------------------
    # Flow Matching Schedule / OT-Flow Coefficients
    # Interpolation: x_t = alpha(t) * z + gamma(t) * x
    # ------------------------------------------------------------------
    def alpha_in(self, t):
        return t  # Coefficient for noise z

    def gamma_in(self, t):
        return 1 - t  # Coefficient for data x

    @torch.no_grad()
    def dist_match(
        self,
        model: nn.Module,
        x: torch.Tensor,
        c: List[torch.Tensor],
    ):
        """
        Distribution Matching (L_rectify helper).

        Matches the distribution of the generated 'fake' flow (reverse time)
        against the 'real' flow (forward time) to align velocity fields.
        """
        z = torch.randn_like(x)
        size = [x.size(0)] + [1] * (len(x.shape) - 1)
        t = torch.rand(size=size).to(x)
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)

        # Forward passes for fake (negative time) and real (positive time) trajectories
        fake_s, _, fake_v, _ = self.forward(model, x_t, -t, -t, **dict(c=c))
        real_s, _, real_v, _ = self.forward(model, x_t, t, t, **dict(c=c))

        F_grad = fake_v - real_v
        x_grad = fake_s - real_s
        return x_grad, F_grad

    def alpha_to(self, t):
        return 1  # d(alpha)/dt

    def gamma_to(self, t):
        return -1  # d(gamma)/dt

    def l2_loss(self, pred, target):
        """Standard L2 (MSE) Loss flattened over spatial dimensions."""
        loss = (pred.float() - target.float()) ** 2
        return loss.flatten(1).mean(dim=1).to(pred.dtype)

    def loss_func(self, pred, target):
        return self.l2_loss(pred, target)

    @torch.no_grad()
    def get_refer_predc(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
    ):
        """
        Get reference predictions with and without conditions (Classifier-Free Guidance).
        Restores RNG state to ensure noise consistency between forward passes.
        """
        torch.cuda.set_rng_state(rng_state)
        # Unconditional forward (using empty condition 'e')
        refer_x, refer_z, refer_v, _ = self.forward(model, x_t, t, tt, **dict(c=e))

        torch.cuda.set_rng_state(rng_state)
        # Conditional forward (using condition 'c')
        predc_x, predc_z, predc_v, _ = self.forward(model, x_t, t, tt, **dict(c=c))

        return refer_x, refer_z, refer_v, predc_x, predc_z, predc_v

    @torch.no_grad()
    def enhance_target(
        self,
        target: torch.Tensor,
        ratio: float,
        pred_w_c: torch.Tensor,
        pred_wo_c: torch.Tensor,
    ):
        """
        Enhance the training target using Classifier-Free Guidance (CFG).
        Target' = Target + w * (Prediction_cond - Prediction_uncond)
        """
        target = target + ratio * (pred_w_c - pred_wo_c)
        return target

    @torch.no_grad()
    def prepare_inputs(
        self,
        model: nn.Module,
        real_image: torch.Tensor,
        labels: List[torch.Tensor],
    ):
        """
        Prepare inputs for Flow Matching training.
        Constructs x_t (noisy data) and target vector field.
        """
        # тут получается что лен будет 4-1=3, по итогу shape будет
        # [batch, 1, 1, 1], это надо для бродкастинга
        size = [real_image.size(0)] + [1] * (len(real_image.shape) - 1)
        # создаем рандомный т
        t = torch.rand(size=size).to(real_image)
        # лейблы нулевые если их нет
        labels = [torch.zeros_like(t)] if labels is None else labels

        # Aux time variable tt < t for consistency estimation
        # создаем еще какое-то рандомное время тт которое строго меньше чем т
        tt = t - torch.rand_like(t) * t

        # Construct Flow Matching Targets
        # генерируем рандомную величину из которой стартует семплирование
        z = torch.randn_like(real_image)
        # x_t = t * z + (1-t) * x
        # просто смешиваем по формуле выше, функции тут чтобы запутать видимо
        # получаем инпут для нейросети
        x_t = z * self.alpha_in(t) + real_image * self.gamma_in(t)
        # v_t = z - x (Target velocity)
        target = z * self.alpha_to(t) + real_image * self.gamma_to(t)

        return x_t, z, real_image, t, tt, labels, target

    @torch.no_grad()
    def multi_fwd(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        N: int,
    ):
        """
        Used to calculate the recursive consistency target.
        """
        pred = 0
        # N=2
        # t * (1 - i / (N)) + tt * (i / (N))
        # t * (1 - i / 2) + tt * (i / 2)
        ts = [t * (1 - i / (N)) + tt * (i / (N)) for i in range(N + 1)]

        # Euler integration loop
        # я не очень понимаю что тут происходит на самом деле
        # типа решается проблема расхождения test time inference?
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            torch.cuda.set_rng_state(rng_state)
            predicted_image, predicted_noise, F_c, _ = self.forward(
                model, x_t, t_c, t_n, **dict(c=c)
            )
            x_t = (
                self.alpha_in(t_n) * predicted_noise
                + self.gamma_in(t_n) * predicted_image
            )
            pred = pred + F_c * (t_c - t_n)

        return predicted_image, predicted_noise, pred

    @torch.no_grad()
    def get_rcgm_target(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        noised_image_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        labels: List[torch.Tensor],
        estimation_order: int,
    ):
        """
        Calculates the RCGM consistency target using N-th order estimation.

        Ref: 'Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation'
        Uses a small temporal perturbation (Delta t = 0.01) to enforce local consistency.
        """
        # Delta t = 0.01 as mentioned in RCGM paper
        # хз что делает эта фунция, наверное при минусе, мы выбираем максимум среди данных чисел
        # тем самым говорим что t_m не может быть меньше чем tt
        # что сам по себе строго меньше чем t
        t_m = (t - 0.01).clamp_min(tt)
        # из зашумленного изображения вычитаем то которое должны получить умноженное на какой-то коэфициент
        # наверное это оговаривается в статье
        noised_image_t = noised_image_t - target * 0.01  # First order step

        # N-step integration from t_m to tt
        """
        через предсказание оригинальных данных и шума в разные стороны
        получаем предсказание для изображения за 2 прохода
        """
        _, _, Ft_tar = self.multi_fwd(
            rng_state,
            model,
            noised_image_t,
            t_m,
            tt,
            labels,
            estimation_order,
        )

        # Weighting for boundary conditions near t=tt
        mask = t < (tt + 0.01)
        cof_l = torch.where(mask, torch.ones_like(t), 100 * (t - tt))
        cof_r = torch.where(mask, 1 / (t - tt), torch.ones_like(t) * 100)

        # Reconstruct velocity field target from integral
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - (Ft_tar).clamp(min=-1.0, max=1.0)
        return Ft_tar

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

    def training_step(
        self,
        model: Union[nn.Module, Callable],
        real_image: torch.Tensor,
        labels: List[torch.Tensor],
        e: List[torch.Tensor] = None,
    ):
        """
        TwinFlow Training Step.
        Combines RCGM loss with TwinFlow-specific losses (L_adv, L_rectify).
        """
        noised_image_t, noise_z, real_image, t, tt, labels, target = (
            self.prepare_inputs(
                model,
                real_image,
                labels,
            )
        )
        

        loss = 0
        rng_state = torch.cuda.get_rng_state()
        # F_th_t - velocity field (предсказание модели)
        _, _, F_th_t, _ = self.forward(
            model,
            noised_image_t,
            t,
            tt,
            **dict(c=labels),
        )
        # обновляем модель через EMA
        self.update_ema(model)

        # Enhance Target (CFG Guidance)
        # это мы выключаем
        # if self.enr > 0.0:
        #     _, _, refer_v, _, _, predc_v = self.get_refer_predc(
        #         rng_state, self.mod, noised_image_t, t, t, labels, e
        #     )
        #     target = self.enhance_target(target, self.enr, predc_v, refer_v)

        # -----------------------------------------------------------
        # 1. RCGM Base Loss (L_base)
        # -----------------------------------------------------------
        """
        делаем тут какую-то сложную 2 эвалюацию для нашей картинки.
        я бы думал о ней как о предикте 2 порядка на данный момент(но не уверен надо читать статью)
        """
        rcgm_target = self.get_rcgm_target(
            rng_state,
            self.mod,
            F_th_t,
            target.clone(),
            noised_image_t,
            t,
            tt,
            labels,
            self.estimation_order,
        )
        # заставляем модель предсказания из обычной модели
        # быть по MSE похожей на предсказание с двунаправленным
        # предсказанием
        loss = self.loss_func(F_th_t, rcgm_target).mean()

        # -----------------------------------------------------------
        # TwinFlow Specific Losses
        # -----------------------------------------------------------

        # [Optional] Real Velocity Loss
        # Ensures real velocity is learned well (redundant if RCGM loss is perfect)
        _, _, F_pred, _ = self.forward(model, noised_image_t, t, t, **dict(c=labels))
        # а тут мы сразу говорим, за один проход сеть должна предсказать
        # нам итоговое velocity
        loss += self.loss_func(F_pred, target).mean()

        # One-step generation forward pass (z -> x_fake)
        # z = torch.randn_like(z); t = rand... (Re-sampling noise/time if needed)
        x_fake, _, F_fake, _ = self.forward(
            model,
            noise_z,
            torch.ones_like(t),
            torch.zeros_like(t),
            **dict(c=labels),
        )

        # 2. Fake Velocity Loss / Self-Adversarial (L_adv)
        # Training fake velocity at t in [-1, 0] to match target flow
        x_t_fake = noise_z * self.alpha_in(t) + x_fake.detach() * self.gamma_in(t)
        target_fake = noise_z * self.alpha_to(t) + x_fake.detach() * self.gamma_to(t)
        _, _, F_th_t_fake, _ = self.forward(model, x_t_fake, -t, -t, **dict(c=labels))
        loss += self.loss_func(F_th_t_fake, target_fake).mean()

        # 3. Distribution Matching / Rectification Loss (L_rectify)
        # Aligns the generated flow with the 'correct' gradient direction
        _, F_grad = self.dist_match(model, x_fake, labels)
        loss += self.loss_func(F_fake, (F_fake - F_grad).detach()).mean()

        # [Optional] Consistency mapping (t=1 to tt=0)
        rcgm_target = self.get_rcgm_target(
            rng_state,
            self.mod,
            F_fake,
            target.clone(),
            noise_z,
            torch.ones_like(t),
            torch.zeros_like(t),
            labels,
            self.estimation_order,
        )
        loss += self.loss_func(F_fake, rcgm_target).mean()

        """
        NOTE ON EFFICIENCY:
        The code above demonstrates the complete TwinFlow logic with all loss terms 
        (RCGM L_base, L_adv, L_rectify) calculated in a single step.
        
        In practice, calculating multiple forward passes with gradients is computationally expensive.
        For large-scale training, it is recommended to:
        1. Split the batch into sub-batches.
        2. Apply different loss terms to different sub-batches (e.g. 50% Base, 25% Adv, 25% Rectify).
        3. Optimize redundant calculations.
        """
        return loss

    def forward(
        self,
        model: Union[nn.Module, Callable],
        noised_image_t: torch.Tensor,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
        **model_kwargs,
    ):
        """
        Forward pass.
        Returns:
            x_hat: Reconstructed data (x0)
            z_hat: Reconstructed noise (x1)
            F_t: Predicted velocity field v_t
            dent: Denominator (normalization term)
        """
        dent = -1  # dent = alpha(t)*gamma'(t) - gamma(t)*alpha'(t) for linear flow
        """
        помещаем в модель зашумленное изображение
        время т
        время тт
        там это все внутри превращается в вектор, и 
        просто конкатенируется с изображением, далее происходит стандартный форвард unet
        и вот эта вся магия, на выходе мы имеем наше предсказанное значение
        которое вычитается из оригинальной картинки F_t = x-unet_pred
        """
        F_t = model(
            noised_image_t,
            t=torch.ones(
                noised_image_t.size(0),
                device=noised_image_t.device,
            )
            * (t).flatten(),
            tt=torch.ones(
                noised_image_t.size(0),
                device=noised_image_t.device,
            )
            * tt.flatten(),
            **model_kwargs,
        )
        t = torch.abs(t)

        # Invert flow to recover x and z
        z_hat = (noised_image_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - noised_image_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def kumaraswamy_transform(self, t, a, b, c):
        """
        Kumaraswamy distribution transform for time step discretization.
        Used to concentrate sampling steps in regions of high curvature.
        """
        return (1 - (1 - t**a) ** b) ** c

    """
    Sampler: UCGM (Unified Continuous Generative Models)
    
    This sampler is highly compatible with RCGM and TwinFlow frameworks.
    Since RCGM and TwinFlow are designed to train "Any-step" models (capable of 
    functioning effectively as both one-step/few-step generators and multi-step 
    generators), this unified sampler enables seamless switching between these 
    regimes without structural changes.
    
    Reference: https://arxiv.org/abs/2505.07447
    Adapted from: https://github.com/LINs-lab/UCGM/blob/main/methodes/unigen.py
    """

    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 1,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.0, 0.0],
        **model_kwargs,
    ):
        """
        Executes the UCGM sampling loop.

        Args:
            inital_noise_z: Initial Gaussian noise.
            sampling_model: The trained Any-step model (RCGM/TwinFlow).
            sampling_steps: 1 for One-step generation, >1 for Multi-step refinement.
            ...
        """
        input_dtype = inital_noise_z.dtype
        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization (with Kumaraswamy transform)
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])

        # Buffer for extrapolation
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            # 1. First order prediction (Euler)
            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                torch.zeros_like(t_cur),  # tt=0 for few-step (one-step)
                # t_next.to(input_dtype), # any-step mode
                # t_cur.to(input_dtype),  # multi-step mode
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Extrapolation logic (optional)
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            # Stochastic injection (SDE-like behavior)
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

            # 2. Second order correction (Heun)
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

    unet = UNet2DModel(**unet2d_config)
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

    optimizer = optim.Adam(unet.parameters())

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
            loss.backward()

    accelerator.end_training()


if __name__ == "__main__":
    main()
