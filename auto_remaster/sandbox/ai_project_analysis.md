# Анализ проекта Auto Game Remaster

> Дата анализа: 08.03.2026  
> Охват: 13 шоукейсов (ноябрь 2024 — февраль 2026), история коммитов и тренировочные скрипты

---

## Описание проекта в одной фразе

**Auto Remaster** — это попытка создать real-time нейронный рендер, который на лету конвертирует кадры старых видеоигр в фотореалистичную картинку, используя только входное изображение (без доступа к движку, геометрии, или G-буферам).

---

## Хронология ключевых этапов

| Шоукейс | Дата | Ключевая идея |
|---------|------|---------------|
| 1 | 03.11.24 | SDXL + ControlNet + IP-Adapter + LoRAs в ComfyUI |
| 2–3 | 05–07.11.24 | Добавление VEnhancer для стабилизации видео. Reshade canny-шейдер |
| 4 | 11.11.24 | Разделение экрана на 4 части, Flux upscaler, VEnhancer optimized |
| 5 | 17.11.24 | SAM2 + Flux для исправления машин; смешивание дорого по маске |
| 6 | 22.11.24 | Reshade normal map canny-шейдер; более стабильные контуры авто |
| 7 | 25.11.24 | Понижен уровень шума в VEnhancer, смешан асфальт из SDXL |
| 8 | 02.01.25 | Переход к one-step diffusion (img2img-turbo, SD-Turbo); 15.6 FPS на 4090 |
| 9 | 02.01.25 | Датасет 49.5k пар, InstructPix2Pix, LCM-дистилляция |
| 10 | 26.02.25 | Tiny UNet без attention, TAESD; 100 FPS на 4090; double-upscale датасет |
| 11 | 05.03.25 | Интеграция с Reshade через shared memory + Python ↔ C++ IPC; 70 FPS |
| 12 | 01.01.26 | SDXL → WAN VACE → SeedVR2 → WAN v2v pipeline; LBM-обучение |
| 13 | 02.19.26 | Windows API capture → CUDA → TensorRT pipeline; 56–60 FPS на RTX 4060 Laptop; обученная пользователем VAE; LoRA 503 KB на Flux Klein 4B |

---

## Достижения

### 1. Конец-в-конец production-ready пайплайн (showcase 13)
Самое важное достижение всего проекта: рабочий real-time нейронный рендер на ноутбучной RTX 4060.

- **Windows API screen capture → CUDA (без CPU-копий)** — устранил главный боттлнек IPC, который убивал FPS ещё в showcase 11.
- **TensorRT pipeline**: VAE-encode → UNet → VAE-decode за один проход; 56–60 FPS в NFS.
- **Собственная обученная VAE** на целевом датасете (fork `vqgan-training`). Стандартный `fal/FLUX.2-Tiny-AutoEncoder` давал мыло и убивал детали; кастомная VAE вернула чёткость и мелкие текстуры.
- **LoRA 503 КБ** поверх Flux Klein 4B для синтеза датасета — практически без деградации на тестовых данных.

### 2. Итеративное построение синтетического датасета
Самостоятельное понимание того, что *data is the bottleneck*, а не архитектура:

- Автоматизация ComfyUI через Python API (batch-обработка, фоновое выполнение).
- Прогрессия качества данных: raw 852 фото → double-upscale SDXL (v5) → 49.5k (v6) → ручная фильтрация 4000 → 241 пара высокого качества + structured noise (v6-filtered).
- Применение `structured noise` (PPD) для борьбы со скачками цветов — значительное улучшение стабильности картинки.

### 3. Архитектурные эксперименты: путь к one-step inference
- Удаление всех attention-блоков из UNet, замена на Conv → 100 FPS @ 512×512 (showcase 10).
- Изучение альтернатив scheduler: DDPMScheduler → FlowMatchEulerDiscreteScheduler; попытки LBM (Latent Bridge Matching).
- Итоговый вывод: "Данный размер модели — это предел без хитрых кернелов или алгоритмов" — **верный вывод**, позволяющий сосредоточить поиск.

### 4. Инфраструктура
- IPC через shared memory C++ ↔ Python (showcase 11–13).
- Понимание ограничений x32 vs x64: Reshade-аддон в x32 несовместим с PyBind.
- Метрики: FID, LPIPS, CLIP; система `ImageEvaluator` в `evaluation.py`.
- Wandb для логирования.
- LeCam regularization, gradient norm logging, discriminator warmup в GAN-скрипте.

### 5. Скорость итераций и широкий охват методов
За полгода проверены: InstructPix2Pix, img2img-turbo, LBM, flow matching, structured noise, TwinFlow, MeanFlow (в коде), VEnhancer, HunyuanVideo, LTX-V, SAM2+Flux, GAN (R3GAN + PatchGAN + VGG-Patch). Это серьёзный объём самостоятельного R&D.

---

## Ошибки и системные проблемы

### 1. GAN-обучение так и не было стабилизировано
Это **самая дорогостоящая ошибка проекта**. Несколько месяцев (5+ разговоров в истории) потрачено на попытки заставить работать GAN-дискриминатор в `train_auto_remaster_lbm_train_test_gap_gan.py`.

**Симптомы:**
- Gradient norm дискриминатора = 0 (баг с detach/attach в вычислительном графе).
- Взрыв потерь генератора.
- Discriminator overpowering: дискриминатор побеждал генератор за несколько сотен шагов.
- Training collapse.

**Коренные причины (наблюдаемые в коде):**
- В `train_auto_remaster_lbm_train_test_gap_gan.py` дискриминатор работает на декодированных VAE-картинках (`taesd.decode(pred.detach())`), но было несколько итераций, когда `.detach()` применялся неправильно или вообще отсутствовал — это обнуляло градиенты.
- В коде находится **три разных дискриминатора** (`R3GANDiscriminator`, `NLayerDiscriminator`, `PatchDiscriminator`) + два типа GAN-loss (`hinge`, `r3gan`) + лецам-регуляризация. Такая сложность без стабильной базовой версии — anti-pattern.
- `self_generation_prob=0.5` и `disc_warmup_steps=5000` — попытка мягко ввести GAN, но без тестов на синтетических данных.

**Что следовало сделать:**
- Проверить GAN на простом синтетическом датасете (например, перекраска цвета) *до* интеграции во flow matching pipeline.
- Использовать один дискриминатор (например, NLayerDiscriminator) с hinge loss и убедиться, что он сходится, прежде чем пробовать R3GAN.
- Ввести GAN только после того, как base flow matching показал стабильный LPIPS < 0.1 на тесте.

### 2. Данные несли артефакты слишком долго
Датасеты v5 (double-upscale SDXL) содержали явные галлюцинации на далёких объектах. Факт: "апскейл 852 картинок занял 36–48 часов на 4090" говорит о том, что батчинг и оптимизация apscale-pipeline не применялись. Результат: showcase 10 дал видео, где картинка "стала хуже" после перехода на более качественный датасет — из-за нестабильных галлюцинаций в данных.

**Системная ошибка:** датасет не валидировался перед обучением. Ручная фильтрация (4000 → 241) была сделана *после* обнаружения проблемы, а не как стандартная практика.

### 3. Переключение методов до глубокого понимания текущего
Есть паттерн: метод не работает → переключиться на следующий. Примеры:
- InstructPix2Pix обнаружил проблему "засвеченных картинок на 20–30 inference steps" → заключили "больше вопросов чем ответов" → перешли на img2img-turbo.
- Img2img-turbo не выучил мелкие детали → переключились на LBM.
- LBM-GAN нестабилен → переключились на MeanFlow/TwinFlow.

Ни одна проблема не была доведена до понимания **root cause** перед переходом. Это размывает эксперименты.

### 4. Мыльная картинка как следствие MSE без GAN
Вывод в README: "модель превратилась в просто покрасочный фильтр". Это **предсказуемый результат** обучения только на MSE/LPIPS без adversarial loss. Проблема была известна из литературы (Pix2Pix, SDXL-Turbo, LCM), но попытки добавить GAN проваливались — см. пункт 1. Получился порочный круг.

### 5. Attention-блоки выброшены слишком рано
В showcase 10 для достижения 100 FPS удалены все attention-слои. Это означает, что модель не может делать нелокальные операции — она не може "видеть" контекст дальних объектов. Именно поэтому "детальный асфальт или листву она не создает". Правильный путь — linear attention или эффективное window attention (как в Sana или HART), а не полное удаление.

### 6. Отсутствие тест-трейн протокола с самого начала
В showcase 10: "... модель почти идеально научилась воспроизводить обучающую выборку" — а потом оказывалось, что на тесте картинка хуже. Тест/валидационный сплит, метрики на hold-out, и ранняя остановка не были встроены в процесс с самого начала.

---

## Технические долги (актуальные на март 2026)

1. **GAN не стабилен** — главный незакрытый вопрос для качества текстур.
2. **VAE bottleneck** — в showcase 13 указано, что больше времени занимает VAE, а не UNet. Путь: `seraena`/`Reducio-VAE`/`DC-Gen` для сжатия латентного пространства.
3. **Attention отсутствует** → модель не создаёт высокочастотные детали в сложных сценах.
4. **Данные всё ещё имеют артефакты** — LoRA обучена на 241 паре (малый размер), и "есть ещё артефакты на дальние автомобили".
5. **Нет автоматического теста качества** — каждая оценка визуальная, нет автоматического CI на метрики.

---

## Что работает хорошо и стоит зафиксировать

| Компонент | Статус | Рекомендация |
|-----------|--------|--------------|
| Windows API capture → CUDA → TensorRT | ✅ Production | Зафиксировать как базовый inference pipeline |
| Кастомная VAE (vqgan-training fork) | ✅ Работает | Может нуждаться в доообучении по мере роста датасета |
| LoRA 503KB на Flux Klein 4B для синтеза | ✅ Работает | Расширить датасет, убрать оставшиеся артефакты |
| Structured noise (PPD) | ✅ Помогает | Обязательно при генерации следующих датасетов |
| LBM base (без GAN) | ⚠️ Работает, но мыльный | Нужен adversarial loss или другой perceptual loss |
| R3GAN + hinge loss | ❌ Нестабилен | Требует отдельного исследования на простом датасете |
| MeanFlow / TwinFlow | 🔬 В коде, не обучался | Перспективно, требует тщательного тестирования |

---

## Рекомендации для следующего этапа

### Приоритет 1: Зафиксировать GAN или отказаться
Либо:
- Prototyping GAN на простом датасете (e.g., edges-to-shoes) и довести до стабильной сходимости.
- Или использовать LADD (Latent Adversarial Diffusion Distillation) из `Nitro-1` repo, которая уже решила этот пузырь.

### Приоритет 2: Расширить и очистить датасет
241 пара — очень мало. Нужно:
- Устранить галлюцинации дальних объектов из pipeline (или исключить их маской из loss).
- Масштабировать до хотя бы 5000–10000 чистых пар.

### Приоритет 3: Добраить attention
Заменить полное отсутствие attention на linear attention (LinAttention из LinFusion или HART) для поддержки 30+ FPS с нелокальным контекстом.

### Приоритет 4: Автоматизировать evaluation
Добавить автоматический запуск LPIPS/FID на hold-out сете при каждом checkpoint — без этого невозможно сравнивать runs объективно.

---

## Итог

**Auto Remaster — это реальный R&D проект**, который прошёл путь от ComfyUI-воркфлоу до production-ready real-time нейронного рендера. Главное достижение — рабочий демо на ноутбуке. Главная незакрытая проблема — качество картинки ограничено отсутствием стабильного adversarial loss. Ключевой урок: **data quality and validation first**, архитектурные эксперименты — потом.

---

## Технический разбор: детали из тренировочных скриптов

> Источники: `run_train.sh`, `train_auto_remaster_lbm_train_test_gap_struct_noise.py`,
> `train_auto_remaster_lbm_train_test_gap_gan.py`, `train_auto_remaster_lbm_train_test_gap.py`,
> `train_auto_remaster_flow_struct_noise.py` + конфиги `configs/*.yaml`

---

### Эволюция скриптов (хронология в run_train.sh)

```
train_auto_remaster.py                    # ранний DDPM/SD1.5
train_auto_remaster_lbm.py               # первый LBM, FLUX.2-dev VAE
train_vae.py                             # обучение кастомной VAE
twinflow.py                              # эксперимент TwinFlow
lbm_repae_gan_reflow.py / _v2           # GAN + reflow
train_auto_remaster_lbm_train_test_gap.py        # LBM + custom tiny VAE
train_auto_remaster_lbm_train_test_gap_struct_noise.py   # + structured noise
train_auto_remaster_flow_struct_noise.py          # чистый flow matching + struct noise
train_auto_remaster_lbm_train_test_gap_gan.py    # ТЕКУЩИЙ: GAN + LeCam + self-gen
```

Активный `run_train.sh` запускает последний `_gan`-скрипт с конфигом `lbm_train_test_gap_gan.yaml`.

---

### UNet — два разных конфига в разных скриптах

**Конфиг A** (struct_noise, flow_struct_noise):
```python
in_channels = 32 * 2 = 64   # 32-канальный латент FLUX.2-dev VAE × 2 (concat)
out_channels = 32
sample_size = 64             # 64×64 латент для 512px изображений
block_out_channels = [320, 640, 1280]
add_attention = False        # нет attention
```

**Конфиг B** (lbm_train_test_gap, GAN-скрипт) — с **кастомной Tiny VAE** (`dim/fal_FLUX.2-Tiny-AutoEncoder_v6_...`):
```python
in_channels = 128 * 2 = 256  # 128-канальный латент Tiny VAE × 2
out_channels = 128
sample_size = 32              # 32×32 латент для 512px изображений
block_out_channels = [320, 640, 1280]
add_attention = False         # всё ещё нет attention
```

> **Ключевое наблюдение:** переход с FLUX.2-dev VAE (32 каналов) на кастомную Tiny VAE (128 каналов) разрезал латентный размер с 64×64 до 32×32 при сохранении числа каналов, что уменьшает пространственное разрешение вдвое. `sample_size` соответственно уменьшился с 64 до 32.

Оба конфига: 3 уровня (`DownBlock2D` × 3, `UpBlock2D` × 3), без attention, `act_fn=silu`, `AdamW8bit` оптимизатор (bitsandbytes).

---

### VAE — эволюция

| Скрипт | VAE |
|--------|-----|
| struct_noise | `black-forest-labs/FLUX.2-dev` VAE, `AutoencoderKL`, 32 канала |
| flow_struct_noise | то же |
| lbm_train_test_gap | `dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora_v2` (кастомная), 128 каналов |
| GAN-скрипт (текущий) | `dim/fal_FLUX.2-Tiny-AutoEncoder_v6_2x_flux_klein_4B_lora` (v1) |

Во всех скриптах VAE заморожен (`vae.requires_grad_(False)`). Обучается только UNet.

---

### Датасеты — evolution

| Конфиг | Датасет |
|--------|---------|
| lbm.yaml | `dim/render_nfs_4screens_6_sdxl_1_wan_mix` |
| lbm_train_test_gap_struct_noise.yaml | `dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw` |
| lbm_train_test_gap.yaml | `dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora` |
| lbm_train_test_gap_gan.yaml (текущий) | `dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora` |

---

### Гиперпараметры обучения

| Параметр | Значение |
|----------|---------|
| `learning_rate` (UNet) | `5e-6` |
| `lr_scheduler_type` | `constant` (без decay) |
| `optimizer` | `AdamW8bit`, `weight_decay=1e-2`, `max_grad_norm=1.0` |
| `per_device_train_batch_size` | 2 |
| `max_steps` | 300 000 |
| `save_steps` | 1600 |
| `precision` | `float32` (fp16 закомментирован) |
| `resolution` | 512×512 |
| `num_inference_steps` | 8 (struct_noise), **1** (GAN-скрипт) |
| `bridge_noise_sigma` | 0.001 (struct_noise) / 0.01 (GAN) |
| `latent_loss_type` | `l1` (во всех последних скриптах) |
| `lpips_factor` | 10.0 (lbm/struct_noise) / 1.0 (GAN) |
| `timestep_sampling` | `custom_timesteps`: `[1000, 875, 750, 625, 500, 375, 250, 125]` |

> **О num_inference_steps=1 в GAN-скрипте:** это попытка дистиллировать модель в one-step — тренируется на t=1000 (source → target за один шаг, как consistency model). В struct_noise — 8 шагов для лучшей итеративной оценки.

---

### Structured Noise — технические детали

Реализован в `generate_structured_noise_batch_vectorized()`, используется как замена Gaussian noise для bridge matching.

**Алгоритм:**
1. Padding изображения с `pad_factor=1.5` (50% mirror padding для устранения boundary artifacts).
2. `FFT2D` → сдвиг к центру → извлечение fase и magnitude изображения.
3. Генерация Gaussian noise → `FFT2D` → извлечение noise magnitude/phase.
4. **Частотная маска** `create_frequency_soft_cutoff_mask`:
   - Гауссова маска перехода со скользящим cutoff_radius.
   - `freq ≤ cutoff_radius`: используется фаза **изображения** (сохраняем структуру).
   - `freq > cutoff_radius`: используется фаза **шума** (рандомизируем высокие частоты).
5. Комбинирование: `magnitude_noise * exp(i * mixed_phase)` → `iFFT` → crop до исходного размера.
6. Экстремальные значения (> ±5) заменяются исходным Gaussian noise.

**В train loop (struct_noise):**
```python
cutoff_radius = 6.0 + np.random.exponential(scale=1/0.1)  # ~6–16, рандом
```

Это означает, что при маленьком `cutoff_radius` шум почти полностью случайный; когда он большой (~16) — структурированный, похожий на оригинал. Экспоненциальное распределение смещает выборку в сторону малых отклонений.

**Назначение:** предотвратить цветовые скачки при инференсе, обеспечив плавный переход между кадрами — ключевое техническое решение для временной стабильности.

---

### Self-Generation / Rollout для борьбы с Train-Test Gap

Реализован в обоих скриптах (struct_noise и GAN). Включается с вероятностью `perturbation_prob=0.5`.

**Идея (Train-Test Gap):** при обучении модель видит идеальные интерполяции `σ·z_source + (1–σ)·z_target`, но при инференсе — накопленные ошибки нескольких шагов. Self-generation имитирует эти ошибки при обучении.

**Алгоритм:**
1. Выбрать таймстемп `t_target`.
2. Выбрать `n_steps_back` (до 4 шагов в struct_noise, до `self_generation_max_steps` в GAN).
3. Начать симуляцию с `t_start = t_target + n_steps_back * step_size`.
4. Запустить цикл Эйлерских шагов `с torch.no_grad()` (в struct_noise) **или с градиентами** (в GAN — `if True:` вместо `with torch.no_grad():`).
5. Добавить bridge noise на каждом шаге: `bridge_factor = sqrt(σ * (1-σ))`.
6. В конце обучать на этом "накопившем ошибки" состоянии.

> **Критический баг в GAN-скрипте (строки 1358–1410):** симуляция запущена `if True:` вместо `with torch.no_grad():` — это означает, что градиенты вычислений внутри цикла симуляции **накапливаются** через все n_steps_back итераций UNet. При `n_steps_back=4` это квадратично увеличивает граф вычислений и почти наверняка вызывает взрывы градиентов или OOM при больших батчах.

---

### GAN-компонент — технические детали

**Конфиг `lbm_train_test_gap_gan.yaml`:**
```yaml
discriminator_type: r3gan
gan_loss_type: r3gan
r3gan_gamma: 1.0
learning_rate_disc: 1e-5       # в 2x меньше lr UNet
disc_warmup_steps: 100         # очень короткий warmup
d_noise_std: 0.0               # шум на вход D
d_train_every: 1               # обновлять D каждый шаг
self_generation_prob: 0.5
use_lecam: true
lecam_loss_weight: 0.1
input_perturbation: 0.1        # шум к z_source_cond
```

**R3GAN дискриминатор (облегчённый конфиг):**
```python
width_per_stage = [16, 32, 64, 64, 128, 128, 128, 128]
cardinality_per_stage = [1, 1, 2, 2, 2, 4, 4, 4]
blocks_per_stage = [1, 1, 1, 1, 1, 1, 1, 1]
expansion = 2
```
Всего 8 стадий, 512→1 даунсэмплинг. AdamW (`beta1=0.0` для R3GAN — как в референс-репозитории).

**D оптимизатор специально:**
```python
d_beta1 = 0.0 if gan_loss_type == "r3gan" else training_args.adam_beta1
```

**LeCam регуляризация (EMA на логиты):**
```python
lecam_anchor_real_logits = 0.0   # инициализация
lecam_beta = 0.9                  # EMA коэффициент
# обновление: anchor = beta * anchor + (1-beta) * mean(logits)
lecam_loss = (real_logits - fake_anchor)² + (fake_logits - real_anchor)²
```

**Режимы дискриминатора:**
- `latent`: принимает 128-канальные латенты напрямую (без VAE-decode).
- `r3gan`: принимает RGB-картинки (после VAE-decode).
- `patchgan`: принимает RGB, `NLayerDiscriminator(input_nc=3, ndf=128, n_layers=4)`.

В лupe D работает на `denoised_sample` — предсказанном `z_target`, декодированном через VAE в RGB.

---

### Функция потерь: финальная версия (struct_noise скрипт)

```python
# latent loss (не используется в финале — закомментирован)
latent_loss = F.l1_loss(model_pred, z_source - z_target)

# Decode predicted latent
denoised = vae.decode(predicted / scaling_factor).clamp(-1, 1)

# LPIPS с двумя backbones
loss_lpips     = lpips_vgg(denoised, target)   # factor=10
loss_lpips_alex = lpips_alex(denoised, target)  # factor=10

total_loss = loss_lpips * 10 + loss_lpips_alex * 10
# latent_loss НЕ входит в финальный лосс struct_noise скрипта
```

> `latent_loss` закомментирован в финальном лоссе — используется только как метрика в wandb.

---

### Проблема .detach() в GAN train loop

В строке ~1489–1493 GAN-скрипта:
```python
with torch.no_grad():
    model_pred = unet(model_input, timesteps)[0]
    denoised_sample = noisy_sample - model_pred * sigmas
    denoised_sample = vae.decode(denoised_sample / vae.config.scaling_factor)[0].clamp(-1, 1)
    # ↑ Для D-step это правильно: UNet без градиентов
```

Но далее в G-step того же скрипта (позже в коде) — модель предсказывает снова **с градиентами**, что корректно. Однако в **earlier iteration** (конец 2025) этот `torch.no_grad()` отсутствовал для D-step, из-за чего градиенты дискриминатора были нулевыми (граф вёл в детачированный тензор). Это был главный диагностированный баг.

---

### flow_struct_noise — ключевое отличие

В `train_auto_remaster_flow_struct_noise.py` (классический flow matching, не LBM):
- Начальная точка `x_start = structured_noise` (не `z_source + noise`).
- Цель: научить модель переводить structured noise → z_target напрямую.
- На инференсе: `sample = structured_noise` → итеративный sampling → `z_target`.
- Bridge noise в инференс-цикле **отключён** (в отличие от LBM).
- VAE: `FLUX.2-dev` (не кастомная Tiny), т.е. это более ранний скрипт.

Это концептуально другая постановка задачи: модель не "улучшает" source, а "генерирует target из шума, обусловленного source". Мост Лебега здесь не источник → цель, а шум (с фазовой структурой source) → цель.

---

## Технические ограничения деплоя: C++ TensorRT Pipeline

> Источник: `inference_optimization/neuro_screen_capture/end2end_tensorrt/`  
> Файлы: `main.cpp`, `cuda_utils.cu`, `config.h`, `pipeline.cpp`, `vsr_upscaler.cpp`, `ISSUE_SUMMARY.md`, `video_sdk_research_1.md`, `video_sdk_research_2.md`

---

### Архитектура production inference pipeline

```
Игра (NFS) → Windows Screen Capture (WGC/DXGI)
  → D3D11 Texture (BGRA, native res)
  → CUDA D3D11 Interop (без CPU-копии!)
  → preprocess_kernel (crop + box-filter downscale → FP16 NCHW [-1,1])
  → TensorRT VAE Encoder (.plan)   → latents FP16 [1, 128, 32, 32]
  → scale * VAE_SCALING_FACTOR     → scaled latents
  → concat_latents (latent ‖ latent)  → [1, 256, 32, 32]  ← UNet вход
  → TensorRT UNet (1 step, UNET_STEPS=1) → velocity [1, 128, 32, 32]
  → scheduler_step_kernel (Euler: sample += velocity * dt)
  → TensorRT VAE Decoder (.plan)   → decoded RGB FP16 NCHW
  → postprocess_kernel (bilinear → D3D11 Surface BGRA)
  → [опционально] VSR Upscaler (D3D11 Video API, NGX)
  → Present (DXGI_PRESENT_ALLOW_TEARING, VSync=0)
```

**Режимы работы:**
- `SPLIT_SCREEN=1`: два квадрата рядом — оригинал | нейросеть (ширина окна `MODEL_SIZE * 2`).
- `ENABLE_VSR=1`, `VSR_SCALE=1.5`: нейросеть → VSR апскейл `512→768px`. Переключается `F11`.
- `ENABLE_UNET=0`: работает только VAE encode→decode (диагностический режим).

---

### Жёсткие аппаратные ограничения

| Требование | Значение | Причина |
|------------|---------|---------|
| GPU | NVIDIA RTX 20+ (Turing/Ampere/Ada) | TensorRT + VSR требуют tensor cores |
| Драйвер | ≥ 530.xx (рекоменд. 550.xx) | VSR модели появились в 530.xx |
| ОС | Windows 10 Build 2004+ / Windows 11 x64 | WGC API, NGX, D3D11 Video |
| VRAM | ≥ 6 GB | VAE 128ch + UNet + VSR scratch buffer |
| VSR scratch buffer | сотни MB для 4K выхода | внутренние тензоры NGX |
| DirectX | D3D11 (с поддержкой BGRA и Video) | interop с CUDA и NGX |

> **GTX карты не поддерживаются** — нет Tensor Cores → VSR не работает. Нет `nvngx_dlvsr.dll`.

---

### Критические баги, найденные в С++ pipeline

#### Баг 1: Неверное число каналов VAE (`config.h`)

```cpp
// Было (наследие от SD 1.5 VAE, 4-канальный латент):
#define LATENT_CHANNELS 4   // → UNet получал [1, 8, 32, 32] вместо [1, 256, 32, 32]

// Стало (FLUX.2-Tiny-AutoEncoder, 128-канальный латент × 2):
#define LATENT_CHANNELS 128  // → конкатенация даёт [1, 256, 32, 32] ✓
```

Симптом: UNet выдавал garbage или чёрный экран. Граф вычислений был корректным, но размерности не совпадали с `.plan` файлом.

#### Баг 2: Знак шага Эйлера в `scheduler_step_kernel`

```cpp
// Было (двойное отрицание):
float dt = sigma_next - sigma;  // dt < 0 (sigma убывает)
s = s + velocity * -dt;         // → s += velocity * |dt|  → движение в НЕВЕРНОМ направлении!

// Стало:
s = s + velocity * dt;          // → s += velocity * (отрицательное) → очистка шума ✓
```

Симптом: модель шла по потоку в обратную сторону — изображение накапливало шум вместо очистки.

> **Оба бага были нетривиальны**: первый из-за несоответствия с устаревшей константой, второй из-за неочевидной семантики `dt` в flow matching (dt уже отрицательное, дополнительный минус был ошибкой).

---

### CUDA Kernels: детали реализации

#### `preprocess_kernel` (capture → network input)
- Box-filter downsample: для каждого пикселя выхода сэмплирует до `8×8` пикселей входа.
- Center crop: `crop_size = min(WIDTH, HEIGHT)`, оффсет выравнивает по центру.
- Нормализация: `[0,1] → [-1,1]` (стандарт для SD-совместимых моделей).
- Формат вывода: FP16 NCHW planar (3 × 512 × 512).
- **BGR→RGB swap** происходит здесь же: `pixel.x=B, pixel.z=R` в BGRA текстуре.

#### `scheduler_step_kernel` (Euler step)
- Полностью на GPU, FP16 end-to-end.
- Реализует: `sample = sample + velocity * (sigma_next - sigma)`.
- Соответствует Python `FlowMatchEulerDiscreteScheduler.step()`.

#### `concat_latents_kernel` (UNet conditioning)
- Конкатенирует два тензора `[1, 128, 32, 32]` → `[1, 256, 32, 32]` по оси C.
- Аналог Python `torch.cat([noisy_sample, z_source_cond], dim=1)`.

#### `postprocess_kernel` (network output → display)
- Bilinear interpolation на GPU при маппинге в D3D surface.
- Denormalize: `[-1,1] → [0,1]`, clamp via `__saturatef`, → uchar4 BGRA.
- В `SPLIT_SCREEN` запускается дважды: слева — `d_input`, справа — `d_output`.

---

### VSR: архитектурные ограничения интеграции

VSR работает через **NVIDIA NGX API** (`nvngx_dlvsr.dll`) в режиме `NVSDK_NGX_Feature_VideoSuperResolution`.

**Проблема цветового пространства:**
- NGX VSR ожидает данные в формате NV12 (YUV 4:2:0), как видеопоток.
- Игровой пайплайн даёт `DXGI_FORMAT_B8G8R8A8_UNORM` (BGRA).
- Требуется Compute Shader для конвертации `BGRA → NV12` перед VSR.

**Потери при конвертации RGB→NV12:**
- 75% цветовой информации теряется (chroma subsampling 4:2:0).
- Для геймерских текстур/HUD это достаточно заметно при высокой CSP.
- Рекомендовано: VSR применять после tone mapping, до UI overlay.

**Производительность VSR (бенчмарки Maxine/NGX):**

| GPU | Задержка 1080p→4K | Допустимость |
|-----|-------------------|-------------|
| RTX 4090 | ~0.52 мс | ✅ даже для 144 FPS |
| RTX 3080/3090 | ~1.57 мс | ✅ для 60-90 FPS |
| RTX 2060 | ~3.5 мс | ⚠️ только для 60 FPS |
| RTX 4060 Laptop | ~2–3 мс (est.) | ✅ для 56–60 FPS (showcase 13) |

**VSR потребление VRAM:** `sотни МБ` для scratch buffer при 4K выходе. Для 768px выхода (scale=1.5) — существенно меньше.

**Критический риск Energy Budget:**
- VSR активирует Tensor Cores когда GPU уже загружен 3D рендером.
- На ноутбучных GPU (RTX 4060 Laptop) это почти всегда вызывает `Power Limit Throttle`.
- При throttle частота ядра падает → итоговый FPS может **упасть** при включении VSR несмотря на апскейл.
- В showcase 13 это обходится тем, что UNet очень лёгкий и GPU недогружен.

**HDR ограничение:**
- VSR + HDR одновременно требует формат P010 (10-бит YUV) вместо NV12 (8-бит).
- При несоответствии форматов — чёрный экран или некорректные цвета.
- В текущем коде используется только SDR (BGRA 8-бит).

---

### Жёсткие ограничения текущего pipeline по размерам

```
MODEL_SIZE = 512                          ← фиксировано в config.h
LATENT_SIZE = 512 / 16 = 32              ← latent spatial resolution
LATENT_CHANNELS = 128                    ← Tiny VAE channels
UNet input = [1, 256, 32, 32]            ← 128*2 (concat source+noisy)
UNet output = [1, 128, 32, 32]           ← velocity prediction
VAE_SCALING_FACTOR = 0.13025             ← must match training config
UNET_STEPS = 1                           ← one-step inference (consistency)
```

**Важно:** все `.plan` файлы (TensorRT engines) скомпилированы для фиксированных batch=1 и размерности `512→32`. Смена MODEL_SIZE требует:
1. Перекомпиляции `.plan` из ONNX.
2. Обновления `config.h`.
3. Проверки VAE_SCALING_FACTOR (зависит от модели, не от размера).

---

### Latency budget при 56–60 FPS на RTX 4060 Laptop

| Шаг | Оценка времени |
|-----|---------------|
| WGC Screen Capture | ~0.5–1 мс |
| D3D11→CUDA interop + preprocess | ~0.2 мс |
| VAE Encoder (TRT FP16) | ~3–5 мс |
| UNet 1-step (TRT FP16) | ~1–2 мс |
| VAE Decoder (TRT FP16) | ~3–5 мс |
| postprocess + D3D present | ~0.5 мс |
| VSR (если включен, scale=1.5) | ~2–3 мс |
| **Итого без VSR** | **~8–14 мс → 71–120 FPS** |
| **Итого с VSR** | **~10–17 мс → 56–100 FPS** |

> 56–60 FPS на showcase 13 соответствует бюджету ~16–17 мс при включённом VSR. **VAE — основной боттлнек** (encode + decode = ~6–10 мс), на что и указывает README.

---

### Почему VAE — главный боттлнек (и что с этим делать)

Кастомная Tiny VAE с 128 каналами платит за высокое качество реконструкции большим латентным пространством `[1, 128, 32, 32]`. При этом:
- Encoder: сжимает `512×512×3 → 32×32×128` = в 32 раза уменьшает spatial, умножает каналы в 42 раза.
- Decoder: обратный путь целиком на GPU, но 128-канальные операции тяжелее чем у стандартного SD 1.5 (4ch).

**Пути оптимизации:**
- `Reducio-VAE`: VAE с уменьшенным числом каналов при сохранении качества.
- `DC-Gen` / `seraena`: ещё более агрессивное сжатие.
- Квантизация VAE в INT8 через TensorRT (с калибровочным датасетом).
- Использовать TAESD (4 канала) — но showcase 13 показал потерю качества.

---

### Несоответствия между Python обучением и C++ деплоем

| Аспект | Python (обучение) | C++ (деплой) |
|--------|-----------------|--------------|
| Precision | float32 | FP16 (TensorRT) |
| Normalization input | `(x - 0.5) / 0.5` | `x * 2 - 1` (эквивалентно) |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | custom `scheduler_step_kernel` |
| latent scaling | `* vae.config.scaling_factor` (Python) | `launch_scale_latents(0.13025)` |
| UNet steps | 1 (GAN config), 8 (struct_noise) | `UNET_STEPS=1` |
| Self-generation | есть в train loop | **нет** в inference (не нужно) |

> Несоответствие точности FP32 vs FP16 — потенциальный источник расхождений. Особенно критично для VAE decoder, где ошибки квантизации накапливаются в 128 каналах. Это отчасти объясняет оставшиеся артефакты в showcase 13.

---

## Прорыв в данных: Безлимитный синтетический датасет через Flux2Klein

> Источник: `auto_remaster/sandbox/diffusers_flux2/`  
> Файлы: `generate_dataset.py`, `generate_dataset.yaml`, `train_img2img.sh`

### Проблема, которую это решает

До этого датасет был ограничен вручную собранными скриншотами игр (пары `input_image` / `edited_image`). Масштабировать его вручную невозможно. Ключевой прорыв — генерировать `edited_image` (фотореалистичные версии кадров NFS) **автоматически** с помощью FLUX.2-klein-base-4B + LoRA, тем самым сделав датасет неограниченным.

### Как работает pipeline генерации

```
dim/nfs_pix2pix_1920_1080_v6  ← базовый датасет (скриншоты NFS)
  → resize+CenterCrop 512px (или 768px)   ← нормализуем входной кадр
  → VAE encode (FLUX.2-Tiny-AutoEncoder)  → z_source [1, 128, 32, 32]
  → generate_structured_noise (cutoff_radius=1200, FFT)
       noise имеет фазовую структуру z_source,
       high freq → чистый шум, low freq → сохраняют layout
  → pipeline._patchify_latents(noise)     ← patchify для FLUX.2
  → Flux2KleinPipeline (img2img, 30 шагов, guidance=1.0)
       с fused LoRA (rank=1 из 10000 шагов обучения)
       prompt_embeds из обученного checkpoint
  → generated PIL images
  → сохраняем: input_image/{idx}.png + edited_image/{idx}.png + metadata.csv
```

### LoRA обучение (train_dreambooth_lora_flux2_klein_img2img.py)

```bash
# train_img2img.sh
accelerate launch train_dreambooth_lora_flux2_klein_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-klein-base-4B \
  --dataset_name="dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw_filtered" \
  --image_column="edited_image" --cond_image_column="input_image" \
  --resolution=768 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --rank=1 \
  --max_train_steps=10000 \
  --lora_layers="attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,..." \
  --structural_noise_radius=100   # structured noise во время обучения тоже!
```

**Ключевые технические детали:**
- Модель: `FLUX.2-klein-base-4B` (4B параметров, img2img версия, не text2img).
- LoRA rank=1 — **минимальная инвазия**. Задача: доменная адаптация NFS→реализм, не обучение с нуля.
- `--structural_noise_radius=100` при обучении, `cutoff_radius=1200` при генерации — **важное расхождение**: при генерации используется бо́льший radius, давая больше "свободы" для деталей.
- `guidance_scale=1.0` — без classifier-free guidance. Модель работает как детерминированный img2img.
- Модель компилируется поблочно (`torch.compile(block)` для каждого `transformer_blocks[i]`).

### Итог: что это дало

| До | После |
|----|-------|
| ~несколько тысяч ручных скриншотов | ∞ (генерируется auto) |
| Только оффлайн кадры | Любые кадры из NFS датасета → auto-remasterизация |
| Узкое распределение данных | Разнообразие ракурсов, освещения, погоды |

Результирующий датасет загружен в HuggingFace как:  
- `dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora` — основной синтетический  
- `dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw` — сырые апскейлы (HR пары)

---

## Кастомный VAE: Устранение Артефактов и Улучшение Качества

> Источник: `auto_remaster/sandbox/vqgan_training/vae_trainer_hf_dataset_flux2_vae.py`

### Проблема, которую это решает

`FLUX.2-Tiny-AutoEncoder` (fal) — хороший общий VAE, но не оптимизирован под специфический домен **NFS → реализм**. Основные проблемы:
- Артефакты VAE-реконструкции в деталях (текстуры асфальта, листва, отражения).
- Стандартный KL-дивергенс усредняет детали → мыло.
- VAE не обучался на синтетических парах domain-specific.

**Решение:** файн-тюнинг FLUX.2-Tiny-AutoEncoder на **нашем датасете** с GAN-лоссом, LPIPS и специальной регуляризацией латентов.

### Архитектура Flux2TinyAutoEncoder (враппер для файн-тюнинга)

```python
class Flux2TinyAutoEncoder(AutoencoderKLFlux2):
    def __init__(self):
        super().__init__()
        self.vae = AutoModel.from_pretrained("fal/FLUX.2-Tiny-AutoEncoder")
        self.encoder = None   # переопределяем стандартный
        self.decoder = None

    def encode(self, x):
        return DotDict(latent_dist=DotDict(sample=lambda: self.vae.encode(x).latent))

    def decode(self, x):
        return DotDict(sample=self.vae.decode(x).sample)
```

Это DDP-совместимый враппер над `fal/FLUX.2-Tiny-AutoEncoder`, позволяющий файн-тюнить его через стандартный обучающий цикл.

### Алгоритм обучения VAE (детально)

```
Данные: dim/nfs_pix2pix_1920_1080_v6_2x_flux_klein_4B_lora
        + dim/nfs_pix2pix_1920_1080_v6_upscale_2x_raw (concatenated)
FlattenedDataset: каждый sample даёт 2 изображения (input_image + edited_image)
→ RandomCrop/Resize до 512px, normalize [-1,1]
```

**Шаг 1 — Encode:**
```python
encoder_latent = vae.encode(real_images_512px).latent_dist.sample()
encoder_latent = encoder_latent.clamp(-8, 8)  # защита от взрывов
```

**Шаг 2 — Decode:**
```python
dec_reconstructed = vae.decode(encoder_latent).sample  # input=512, output=512
```

**Шаг 3 — Discriminator step (LeCam + BCE):**
```python
real_preds = discriminator(real_images_512px)       # PatchDiscriminator
fake_preds = discriminator(dec_reconstructed.detach())
d_loss = BCE(real_preds, 1) + BCE(fake_preds, 0)

# LeCam EMA (β=0.9):
lecam_anchor_real = 0.9 * lecam_anchor_real + 0.1 * mean(real_preds)
lecam_anchor_fake = 0.9 * lecam_anchor_fake + 0.1 * mean(fake_preds)
lecam_loss = (real_preds - lecam_anchor_fake)² + (fake_preds - lecam_anchor_real)²
total_d_loss = d_loss + 0.1 * lecam_loss
# → backward(retain_graph=True) → optimizer_D.step()
```

**Шаг 4 — Generator step (LPIPS + zloss + GAN adv):**
```python
_recon_normed = gradnorm(dec_reconstructed)       # нормируем градиенты для баланса
percep_loss = lpips(_recon_normed, real_images_512px)  # VGG features

# latent regularization: принуждаем z² быть близко к 0
zloss = (encoder_latent ** 2).mean() * 0.1

# adversarial: обманываем дискриминатор
fake_preds_g = discriminator(_recon_normed)       # без detach()
g_adv_loss = BCE(fake_preds_g, 1)                 # говорим что это реал

total_g_loss = percep_loss + zloss + g_adv_loss
# → backward() → optimizer_G.step()
```

### Ключевые технические решения

| Решение | Почему |
|---------|--------|
| LPIPS вместо MSE | MSE даёт мыло; LPIPS сравнивает VGG features → текстуры |
| GAN (PatchDiscriminator) | Заставляет декодер восстанавливать high-freq детали |
| LeCam regularization | Предотвращает дестабилизацию дискриминатора при малых данных |
| z² регуляризация (zloss=0.1) | Латенты компактны → диффузионная модель лучше их моделирует |
| gradnorm() | Балансирует вклад LPIPS и GAN без explicit веса |
| clamp[-8,8] | Предотвращает взрывы латентов → stable diffusion training после |
| lr_vae = lr / vae_ch | Adaptive LR для conv_in vs остальных слоёв |
| DDP multi-GPU (NCCL) | Обучение на нескольких GPU, avg loss через all_reduce |

### Мониторинг Posterior Collapse

В обучающем цикле явно отслеживаются:
- Квантили z (0%, 20%, 40%, 60%, 80%, 100%) — защита от collapse и взрывов.
- Skewness (асимметрия) и Kurtosis (эксцесс) распределения z.
- `std → 0` = posterior collapse (модель игнорирует encoder).
- `max → ∞` = gradient explosion.

### Результат

Файн-тюн Tiny VAE на domain-specific данных с GAN:
- **Убраны VAE-артефакты** в текстурах дороги, деревьев, машин.
- **Улучшено качество реконструкции** без роста размера латентного пространства.
- VAE остаётся совместимым с UNet (те же 128 каналов, те же размерности).
- Улучшенный VAE загружается в C++ pipeline через тот же TensorRT `.plan` (после реэкспорта ONNX → TRT).

---

## Главная боль: модель — просто покрасочный фильтр

> Контекст (README, showcase 13): при попытке убрать галлюцинации модель перестала изменять геометрию — не создаёт детальный асфальт, листву, не меняет структуру. Устойчивость достигнута, смысл потерян. GAN взрывается. Изучаются: reflow, MeanFlow, PiFlow, TwinFlow, SenseFlow, LADD.

---

### Почему так произошло технически

Проблема возникла из суперпозиции трёх факторов:

1. **LPIPS с большим весом** — LPIPS наказывает за любое отклонение от ground truth в feature space VGG. Это буквально учит модель «не галлюцинировать» — то есть быть консервативной.
2. **Один шаг Эйлера (UNET_STEPS=1)** — за один шаг модель физически не может «решиться» изменить геометрию: риск слишком высок, любая ошибка в геометрии стоит дорого. При многошаговом процессе ошибки на ранних шагах исправляются позже.
3. **Structured noise с маленьким `cutoff_radius`** — чем меньше cutoff, тем больше высокочастотных компонент source сохраняется в шуме → модель видит «подсказку» исходной геометрии и просто красит её.

Фактически модель нашла оптимум: **копировать структуру source, менять только цвет** — это идеальный LPIPS при нулевом GAN.

---

### Идеи по приоритету: «быстро попробовать → скорее всего работает»

---

#### 🥇 Идея 1: Просто дать GAN работать — R1 Gradient Penalty вместо LeCam

**Проблема с текущим GAN:** он взрывается потому что дискриминатор учится слишком быстро → generator loss → ∞. Нужен не новый алгоритм, а одна строчка.

**Самое простое что работает стабильно везде** — R1 regularization:
```python
# Вместо всего что было — просто добавить к d_loss:
r1_weight = 10.0
gp = (torch.autograd.grad(real_preds.sum(), real_images, create_graph=True)[0] ** 2).sum()
d_loss = d_loss + r1_weight * 0.5 * gp
```

R1 penalty — стандарт de facto. Буквально используется в StyleGAN, LDM, SD3. Дискриминатор физически не может уйти слишком далеко — его Lipschitz константа ограничена. При `r1_weight=10` GAN обычно стабилен.

**Дополнительно:** сделать discriminator warmup — первые 1000 шагов обновлять только discriminator, generator заморожен. Дать disc «обучиться» до того, как generator начнёт с ним конкурировать.

**Ожидание:** GAN стабилизируется, начнёт давать текстуры асфальта и листвы. Это самая дешёвая по времени попытка.

---

#### 🥇 Идея 2: Перестать штрафовать за геометрию в основном лоссе

Текущий лосс: `LPIPS(decoded, target)` — это полное изображение. LPIPS видит геометрическую ошибку и штрафует.

**Трюк:** считать LPIPS только на low-frequency части (downsampled), а GAN пусть отвечает за high-frequency детали:

```python
# Основной лосс — только на downsampled версии (цвет/структура верны)
lpips_low = lpips(
    F.interpolate(decoded, scale_factor=0.25, mode='bilinear'),
    F.interpolate(target, scale_factor=0.25, mode='bilinear')
)

# GAN — только на кропах 128x128 (текстуры/детали)
patch = random_crop(decoded, 128)
real_patch = random_crop(target, 128)
g_adv_loss = disc(patch)

total = lpips_low * 1.0 + g_adv_loss * 0.1
```

Модель перестанет бояться геометрических изменений — LPIPS больше не штрафует за них напрямую, только за глобальную цветовую структуру.

---

#### 🥈 Идея 3: Latent Adversarial Diffusion Distillation (LADD)

**Что это:** [Sauer et al., 2024](https://arxiv.org/abs/2403.12015). Используется в SD3-Turbo и FLUX-Schnell. Основная идея — дискриминатор работает не в pixel space, а в latent space «учителя» (большой диффузионной модели).

**Почему лучше для нас:**
- Дискриминатор в latent space (`[1, 128, 32, 32]`) — намного проще обучить, чем в pixel space.
- Учитель (FLUX.2-klein) уже знает, как выглядят «правильные» текстуры асфальта → дискриминатор это унаследует.
- Нет проблемы «GAN видит пиксели и взрывается» — латенты гладкие.

**Конкретный план:**
```python
# В train loop:
# 1. Encode decoded image через FROZEN FLUX VAE → student_latent
student_latent = flux_vae.encode(decoded).latent

# 2. Encode real target через FROZEN FLUX VAE → real_latent
real_latent = flux_vae.encode(target).latent

# 3. Discriminator в latent space
real_preds = latent_disc(real_latent)
fake_preds = latent_disc(student_latent.detach())
d_loss = hinge_loss(real_preds, fake_preds)

# 4. Generator adversarial loss
g_adv = latent_disc(student_latent).mean()  # latent disc вместо pixel disc
```

У тебя уже есть инфраструктура для latent discriminator в `lbm_train_test_gap_gan.yaml` — это минимальное изменение.

---

#### 🥈 Идея 4: MeanFlow — заменить LBM целиком

**Репо:** [inclusionAI/MeanFlow](https://github.com/inclusionAI/MeanFlow)

**Почему это могло бы помочь:** MeanFlow решает ту же проблему иначе — вместо предсказания мгновенной velocity в точке, модель предсказывает **среднюю velocity по всей траектории** от t до r:

```
v_mean(x_t, t, r) = (x_r - x_t) / (r - t)
```

Это принципиально другое обучение:
- Одношаговый inference **теоретически обоснован** — не heuristic как в LBM.
- Модель оптимизирована для r=0 (полная очистка за 1 шаг) явно.
- JVP (Jacobian-vector product) в loss function обеспечивает согласованность trajectory.

**Проблема:** нужно переписать training loop полностью. Логика: `train_auto_remaster_lbm_meanflow.py` уже начат (из conversation summaries).

**Когда пробовать:** после того как GAN stabilized — MeanFlow даст лучшее качество single-step, но не решает проблему «нет текстур» сам по себе.

---

#### 🥉 Идея 5: Увеличить cutoff_radius во время обучения

**Самое простое из всего.** Текущий structured noise с маленьким `cutoff_radius` → шум содержит фазовую структуру source → модель «видит подсказку» геометрии и не учится её изменять.

**Попробовать:** динамически увеличивать `cutoff_radius` в процессе обучения:
```python
# Текущая логика (уже есть в коде):
cutoff_radius = min(base_cutoff + step * increase_rate, max_cutoff)

# Попробовать более агрессивный schedule — к концу обучения cutoff=∞ (чистый гауссовый шум)
# Это заставит модель научиться «создавать» геометрию, а не копировать её
```

При `cutoff_radius → ∞` шум полностью гауссовый → модель должна создавать текстуры с нуля, опираясь только на source как condition. Именно так работают generation-capable модели.

**Риск:** нестабильность на больших cutoff. Можно делать curriculum: сначала маленький cutoff (стабильность), потом постепенно увеличивать.

---

#### 🥉 Идея 6: TwinFlow / PiFlow — parallel branch для деталей

[TwinFlow](https://github.com/inclusionAI/TwinFlow) предлагает два параллельных flow: один для грубой структуры, второй для деталей. Для нашей задачи это означает:
- **Branch 1 (structure):** low-freq, консервативный, стабильный.
- **Branch 2 (detail):** high-freq, aggressive, отвечает за листву/асфальт.

**Почему это релевантно:** сейчас один UNet пытается делать и то, и другое под единым лоссом. Разделение позволяет отдельно регуляризовать каждый branch.

**Сложность:** существенная переработка архитектуры. Пробовать после GAN fix.

---

### Рекомендуемый порядок попыток

```
1. [1-2 дня] R1 penalty + discriminator warmup
   → если GAN стабилен → добавить patch-based LPIPS разделение

2. [3-5 дней] LADD (latent discriminator через FLUX VAE)
   → latent disc уже есть в конфиге, нужно только изменить target

3. [1 неделя] Curriculum cutoff_radius: 50 → 500 → ∞
   → посмотреть меняется ли геометрия при больших cutoff

4. [2+ недели] MeanFlow — полная замена LBM
   → только если предыдущие дали частичный результат

5. [месяц+] TwinFlow/PiFlow — если нужна финальная полировка
```

---

### Один совет поверх всего

Прежде чем пробовать новый алгоритм — **замерить одно число** на фиксированных 20 кадрах. Например, `LPIPS(decoded, target)` на тест-сете. Это занимает 30 минут кода один раз, и потом каждый эксперимент даёт конкретный ответ: лучше или хуже. Без числа каждый эксперимент — субъективная оценка.

