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


class DiffusionUNet(nn.Module):
    def __init__(
        self,
        data_dim,
        conv_hidden_dim,
        time_embed_dim,
        num_classes=10,
        label_embed_dim=32,
    ):
        super(DiffusionUNet, self).__init__()
        self.data_dim = data_dim
        self.side_len = int(np.sqrt(data_dim))  # MNIST: 28

        # 1. Time Embedding
        # t and tt use separate embedding layers as they are distinct time scalars
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.target_time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 2. Label Embedding
        self.label_embedding = nn.Embedding(num_classes, label_embed_dim)

        # 3. Encoder
        self.enc1 = nn.Sequential(
            (nn.Conv2d(1, conv_hidden_dim, kernel_size=3, padding=1)),
            nn.ReLU(),
            (
                nn.Conv2d(
                    conv_hidden_dim, conv_hidden_dim, kernel_size=4, stride=2, padding=1
                )
            ),
        )
        self.enc2 = nn.Sequential(
            nn.ReLU(),
            (nn.Conv2d(conv_hidden_dim, conv_hidden_dim * 2, kernel_size=3, padding=1)),
            nn.ReLU(),
            (
                nn.Conv2d(
                    conv_hidden_dim * 2,
                    conv_hidden_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
        )

        # 4. Bottleneck
        self.bottleneck = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_dim * 2, conv_hidden_dim * 4, kernel_size=3, padding=1
            ),
        )

        # 5. Condition Projection Layer [Correction]
        # Input = t_emb + tt_emb + label_emb
        total_cond_dim = time_embed_dim * 2 + label_embed_dim
        self.cond_projection = nn.Linear(total_cond_dim, conv_hidden_dim * 4)

        # 6. Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                conv_hidden_dim * 4 + conv_hidden_dim * 2,
                conv_hidden_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_hidden_dim * 2, conv_hidden_dim * 2, kernel_size=3, padding=1
            ),
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(
                conv_hidden_dim * 2 + conv_hidden_dim,
                conv_hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(conv_hidden_dim, conv_hidden_dim, kernel_size=3, padding=1),
        )

        self.output_conv = nn.Conv2d(conv_hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_flat, t, tt=None, c=None):
        # x_flat: [Batch, 784] -> x: [Batch, 1, 28, 28]
        x = x_flat.view(-1, 1, self.side_len, self.side_len)

        # --- Process t ---
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]
        t_emb = self.time_embedding(t)  # [B, time_dim]

        # --- Process tt [Correction] ---
        if tt is not None:
            if tt.dim() == 1:
                tt = tt.unsqueeze(1)
            tt_emb = self.target_time_embedding(tt)  # [B, time_dim]
        else:
            # If tt is not provided, fill with zeros (for robustness, though TwinFlow should provide it)
            tt_emb = torch.zeros_like(t_emb)

        # --- Process c (Labels) ---
        if c is not None:
            label_feat = self.label_embedding(c[0])  # [B, label_dim]
        else:
            label_feat = torch.zeros(x.size(0), self.label_embedding.embedding_dim).to(
                x.device
            )

        # --- Concatenate all conditions ---
        # [B, time_dim + time_dim + label_dim]
        cond_feat = torch.cat([t_emb, tt_emb, label_feat], dim=1)

        # --- U-Net Forward ---
        skip1 = self.enc1(x)
        skip2 = self.enc2(skip1)
        b = self.bottleneck(skip2)

        # Inject conditions (Bias injection)
        # Project dimensions and reshape to [B, C, 1, 1] for broadcasting addition
        cond_b = self.cond_projection(cond_feat).view(-1, b.shape[1], 1, 1)
        b = b + cond_b

        up1 = self.dec1(torch.cat([b, skip2], dim=1))
        up2 = self.dec2(torch.cat([up1, skip1], dim=1))

        output = self.output_conv(up2)
        # более глобальный skip connection, чтобы в этом примере работал лучше
        # но это не меняет задачу на денойзинг, просто меняется линейная комбинация,
        # потому что таргет по преждему моделирование производной
        return x_flat - output.view(-1, self.data_dim)
        # return output.view(-1, self.data_dim)


class MLP(nn.Module):

    def __init__(self, in_dim, context_dim, h, out_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim + context_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x, t, tt=None, c=None):
        t = t.flatten().unsqueeze(1)
        tt = tt.flatten().unsqueeze(1)
        if t is not None and tt is None:
            input = torch.cat((x, t), dim=1)
        elif t is not None and tt is not None:
            input = torch.cat((x, t, tt), dim=1)
        else:
            input = x
        return self.network(input)


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


def get_args():
    parser = argparse.ArgumentParser(description="TwinFlow MNIST Training")

    # --- Core Parameters ---
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")

    # --- Model Parameters ---
    parser.add_argument(
        "--conv_hidden_dim", type=int, default=64, help="UNet hidden dimension"
    )
    parser.add_argument(
        "--time_embed_dim", type=int, default=64, help="Time embedding dimension"
    )

    # --- TwinFlow Parameters ---
    parser.add_argument(
        "--estimate_order", type=int, default=2, help="Estimate order for RCGM loss"
    )
    parser.add_argument(
        "--ema_decay_rate", type=float, default=0.99, help="EMA decay rate"
    )
    parser.add_argument(
        # "--enhanced_ratio", type=float, default=0.5, help="Training time CFG ratio"
        "--enhanced_ratio",
        type=float,
        default=0.0,
        help="Training time CFG ratio",
    )
    parser.add_argument(
        "--using_twinflow",
        action="store_true",
        help="If using TwinFlow loss (or only RCGM loss)",
    )

    # --- Miscellaneous ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./outputs/mnist",
        help="Directory to save logs and images",
    )
    parser.add_argument(
        "--data_root", type=str, default="../buffers", help="Path to MNIST dataset"
    )

    return parser.parse_args()


def get_experiment_names(args):
    """
    Generate experiment names. Short name format strictly aligns with Moons example.
    """

    def fmt(val):
        return str(val).replace(".", "p")

    short_name = (
        f"lr={fmt(args.lr)}_edr={fmt(args.ema_decay_rate)}_eo={args.estimate_order}"
    )

    # Long name includes more details for Log indexing
    long_name = (
        f"TwinFlow_MNIST_{short_name}_"
        f"ep={args.epochs}_bs={args.batch_size}_"
        f"seed={args.seed}"
    )

    return long_name, short_name


def save_experiment_log(long_name, result_data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[long_name] = result_data

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Log updated: {filepath}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = get_args()
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    long_exp_name, short_exp_name = get_experiment_names(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Short Name: {short_exp_name}")
    print(f"Long Name:  {long_exp_name}")

    # 1. Data Preparation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        transform=transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # 2. Model and Trainer
    model = DiffusionUNet(
        data_dim=784,
        # 64
        conv_hidden_dim=args.conv_hidden_dim,
        # 64
        time_embed_dim=args.time_embed_dim,
        num_classes=10,
        label_embed_dim=64,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = TwinFlow(
        ema_decay_rate=args.ema_decay_rate,
        # 2
        estimate_order=args.estimate_order,
        # 0.0
        enhanced_ratio=args.enhanced_ratio,
        # True
        using_twinflow=args.using_twinflow,
    )

    # 3. Training Loop
    print("Starting training...")
    final_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (real_img, labels) in enumerate(train_loader):
            # зануляем оптимайзер
            optimizer.zero_grad()
            # вытягиваем картинку?
            real_x = real_img.view(real_img.size(0), -1).to(device)
            # переводим лейблы на девайс
            labels = labels.to(device)
            # зачем-то перемешивает лейблы
            uncond = labels[torch.randperm(labels.size(0))]
            # считаем лосс
            loss = trainer.training_step(
                model,
                real_x,
                [labels],
                None,
                # [uncond],
            )
            # вычисляем градиент
            loss.backward()

            # Gradient handling
            # фиксим наны, бесконечности, зануляем их
            # хз зачем, видимо в этом трейн сетапе так избегаем ошибок
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

            # обновляем оптимизатор
            optimizer.step()
            epoch_loss += loss.item()

        final_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {final_loss:.4f}")

    # 4. Save Log
    result_data = {"final_loss": final_loss, "args": vars(args)}
    save_experiment_log(
        long_exp_name, result_data, os.path.join(args.save_dir, "experiments_log.json")
    )

    # 5. Visualization (10x10 Ordered Grid)
    # Uses save_image directly to avoid matplotlib margins and titles
    print("Generating 10x10 Ordered Visualization...")
    n_classes = 10
    n_samples_per_class = 10
    total_vis = n_classes * n_samples_per_class
    # генерируем картинки
    with torch.no_grad():
        # Construct ordered labels: 10 of 0, 10 of 1...
        vis_labels = (
            torch.arange(n_classes).repeat_interleave(n_samples_per_class).to(device)
        )

        z_vis = torch.randn(total_vis, 784).to(device)

        # Condition Generation using UCGM sampler
        gen_vis = trainer.sampling_loop(z_vis, model, **dict(c=[vis_labels]))[-1]

        # Restore to image space
        # Map from [-1, 1] back to [0, 1]
        gen_vis = (gen_vis + 1) / 2.0
        gen_vis = gen_vis.clamp(0, 1).view(total_vis, 1, 28, 28)

        save_path = os.path.join(args.save_dir, f"{short_exp_name}.png")

        # Save image directly using torchvision utils
        # nrow=10 ensures 10 digits per row
        # padding=2 sets the interval between digits
        utils.save_image(gen_vis, save_path, nrow=n_samples_per_class, padding=2)

        print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    main()
