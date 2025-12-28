#!/usr/bin/env python3

import random

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(n_in, n_out, **kwargs):
    kwargs["bias"] = False
    khw = 4 if kwargs.get("stride", 1) == 2 else 3
    return nn.Sequential(
        nn.Conv2d(n_in, n_out, khw, padding=1, **kwargs),
        nn.BatchNorm2d(n_out),
        nn.ReLU(inplace=True),
    )


class PatchDiscWithContext(nn.Module):
    def __init__(self, c_im=3, c_ctx=4):
        super().__init__()
        self.blocks = nn.Sequential(
            conv_bn_act(c_im + c_ctx, 64),
            conv_bn_act(64, 128, stride=2),
            conv_bn_act(128, 256, stride=2),
            conv_bn_act(256, 512, stride=2),
            conv_bn_act(512, 512),
            conv_bn_act(512, 512),
        )
        self.proj = nn.ModuleList(
            nn.Conv2d(block[0].out_channels, 1, 1, bias=False) for block in self.blocks
        )

    def forward(self, x, ctx):
        """
        x - изначальная картинка
        ctx - латент данной картинки
        """
        out_hw = tuple(hw // 8 for hw in x.shape[-2:])
        # растягиваем латент до размера исходного изображения и конкатенируем с оригиналом
        x = th.cat([x.mul(2).sub(1), F.interpolate(ctx, x.shape[-2:])], 1)
        out = []
        for block, proj in zip(self.blocks, self.proj):
            x = block(x)
            # Дискриминатор выдает оценку не только в самом конце, но и после каждого блока сверток.
            # Это позволяет находить ошибки как на уровне мелких текстур (первые слои), так и на уровне
            # глобальной структуры (глубокие слои).
            out.append(F.adaptive_avg_pool2d(proj(x), out_hw))
        return th.cat(out, 1)


class Seraena(nn.Module):
    def __init__(self, c_im, c_ctx, use_amp=True, max_buff_len=16384):
        """An adversarial trainer / corrector for conditional patch-based generative models.

        Internally, Seraena trains a patch-based discriminator to distinguish real / fake images,
        conditioned on corresponding context (latent) images. The gradients of this discriminator
        are used to create corrected fake images.

        Seraena maximizes quality / stability of the provided corrections by using a few GAN training strategies:
            * Using a replay buffer to make sure the discriminator handles all sorts of fake images
              (not just the latest fake images from the current generator)
            * Making "relativistic" corrections that always push real / fake closer together
              (regardless of how good the discriminator is doing)
            * Scaling corrections to a sensible range manually
              (to prevent extremely small / large gradients from going to the generator)
            * Predicting scores from multiple layers of the discriminator
              (rather than just at the end)

        Args:
            c_im: number of channels in the input images.
            c_ctx: number of channels in the latent / context images.
            use_amp: if True, use mixed precision for forward / backward passes.
                Enabling mixed precision should reduce memory usage & improve speed.
            max_buff_len: maximum number of fake samples to store in memory.
                Higher values will use more memory but should improve the quality of
                the corrections and the stability of training.
        """
        super().__init__()
        self.use_amp = use_amp
        # discriminator
        self.disc = PatchDiscWithContext(c_im=c_im, c_ctx=c_ctx)
        self.scaler = th.cuda.amp.GradScaler(enabled=use_amp)
        self.opt = th.optim.AdamW(self.disc.parameters(), 3e-4, betas=(0.9, 0.99))
        # replay buffer of recent fake images
        self.buff = []
        self.max_buff_len = max_buff_len

    def _disc_train_step(self, real, fake, pretrained_latents):
        """
        Выполняет один шаг обновления весов дискриминатора, обучая его отличать реальные изображения от сгенерированных.
        Использует стратегии стабилизации GAN для повышения качества обучения.
        Аргументы:
            real: Тензор реальных изображений из датасета (Ground Truth).
            fake: Тензор сгенерированных изображений, полученных от текущей версии декодера.
            pretrained_latents: Контекст (латентный код), служащий условием генерации.

        Ключевые этапы алгоритма:
        Experience Replay (Буфер опыта):
            Текущие фейковые изображения сохраняются в циклический буфер self.buff.
            Для обучения формируется комбинированный набор фейков: 50% свежих (от текущего генератора) и
            50% исторических (случайно выбранных из буфера). Это предотвращает "зацикливание" обучения и
            забывание старых типов ошибок.
        Batch Mixing (Смешивание батча):
            Реальные и фейковые изображения (вместе с их контекстом) перемешиваются в один тензор с помощью случайной маски.
            Это гарантирует, что слои BatchNorm внутри дискриминатора вычисляют статистику по смешанным
            данным, предотвращая утечку информации о классе (real/fake) через метаданные распределения.
        Расчет LSGAN Loss:
            Дискриминатор делает предсказание для смешанного батча.
            Вычисляется ошибка MSE (Least Squares Loss), где целевыми значениями выступают метки
            классов (например, 1 для фейка, -1 для реального).
        Возвращает:
            Словарь debug с текущим значением лосса (disc_loss), входными данными и предсказаниями
            для визуализации и мониторинга процесса обучения.
        """
        self.disc.train()

        # add new fake / pretrained_latents to buffer
        # pretrained_latents это оригинальные латенты из предобученного энкодера
        # fake это декодированная картинка из обучаемого декодера
        for fake_i, ctx_i in zip(fake, pretrained_latents):
            if len(self.buff) >= self.max_buff_len:
                i = random.randrange(0, len(self.buff))
                self.buff[i][0].copy_(fake_i)
                self.buff[i][1].copy_(ctx_i)
            else:
                self.buff.append((fake_i.clone(), ctx_i.clone()))

        # sample half of fake / ctx new, half from buffer
        n = len(fake) // 2
        # семплируем из буфера фейки и латенты, что является половиной от текущего батча
        # фейки создаются динамически, поэтому желательно их сохранять для того чтобы алгоритм получал
        # не только актуальные ошибки которые он совершал, но и несколько более ранние
        fake_shuf, fake_shuf_ctx = (
            th.stack(items, 0)
            for items in zip(*(random.choice(self.buff) for _ in range(n)))
        )
        # объединяем пришедший батч с историческим
        fake_shuf = th.cat([fake[:n], fake_shuf], 0)
        fake_shuf_ctx = th.cat([pretrained_latents[:n], fake_shuf_ctx], 0)

        with th.cuda.amp.autocast(enabled=self.use_amp):
            # хитрым образом создаем маску выбора примеров для обучения
            # это типа должно быть быстрее
            fake_mask = th.rand_like(real[:, :1, :1, :1]) < 0.5
            # тут мы зануляем какую-то часть массива, соответственно при сложении
            # они встанут в нужное место
            in_ims = fake_mask * fake_shuf + ~fake_mask * real
            in_ctxs = fake_mask * fake_shuf_ctx + ~fake_mask * pretrained_latents
            # получаем фичи и предсказание с дискриминатора
            scores = self.disc(in_ims, in_ctxs)
            # переводим таргеты в диапазон от -1 до 1
            # так как у нас нет сигмоиды, которая вызывает нестабильность при обучении, тут мы используем
            # MSE loss, который как показано в работе работает более стабильно https://arxiv.org/pdf/1611.04076
            targets = fake_mask.float().mul(2).sub(1).expand(scores.shape)
            loss = F.mse_loss(scores, targets)
        self.opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        return {
            "disc_loss": loss.item(),
            "disc_in": in_ims,
            "disc_in_ctx": in_ctxs,
            "disc_pred": scores.detach(),
            "disc_targets": targets,
        }

    def _make_correction(self, real, fake, ctx):
        self.disc.eval()

        def featurizer(x):
            with th.cuda.amp.autocast(enabled=self.use_amp):
                return self.disc(x, ctx)

        correction = th.zeros_like(fake).requires_grad_(True)
        with th.no_grad():
            # берем фичи из обучаемого дискриминатора
            ref_feats = featurizer(real)
        # fake это изображение декодированное до обучения дискриминатора
        # ref_feats это фичи полученные из дискриминатора из реальных изображений
        # такой лосс скорее всего нужен чтобы при помощи дискриминатора выучить только
        # correction
        # Вычисляется аддитивная поправка к пикселям fake, которая приближает их к real в пространстве признаков.
        loss = F.mse_loss(
            ref_feats,
            featurizer(fake + correction),
            reduction="none",
        ).mean((1, 2, 3), keepdim=True)
        loss.sum().backward(inputs=[correction])
        # негативный градиент направляет нас в сторону уменьшения
        correction = correction.grad.detach().neg()
        # Нормируем (делим на стандартное отклонение), чтобы масштаб поправок был стабильным 
        # и не зависел от текущей крутизны функции потерь.
        correction.div_(correction.std(correction=0).add(1e-5))
        return correction

    def step_and_make_correction_targets(self, real, fake, ctx):
        """Run one Seraena step on the provided real / fake images.

        This function is expected to be used in your training loop, e.g.
            def compute_loss_on_batch(generator, real, ctx):
                fake = generator(ctx)
                targets, _ = seraena.step_and_make_correction_targets(real, fake, ctx)
                return F.mse_loss(fake, targets)

        Args:
            real (NCHW tensor): batch of real images
            fake (NCHW tensor): batch of corresponding fake images
            ctx (NCHW tensor): batch of context images (e.g. latents)
                corresponding to each real / fake pair.

        Returns a tuple of:
            output (NCHW tensor): batch of corrected fake images
            debug (dict): dictionary of debug information
        """
        real, fake, ctx = real.detach(), fake.detach(), ctx.detach()
        debug = self._disc_train_step(real, fake, ctx)
        correction = self._make_correction(real, fake, ctx)
        return fake + correction, debug
