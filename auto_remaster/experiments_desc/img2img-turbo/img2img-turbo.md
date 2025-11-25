### Описание экспериментов над сетапом img2img-turbo (custom)

Данная работа основана на https://github.com/GaParmar/img2img-turbo. One-Step Image Translation with Text-to-Image Models. https://arxiv.org/abs/2403.12036

In this work, we address two limitations of existing conditional diffusion models: their slow inference speed due to the iterative denoising process and their reliance on paired data for model fine-tuning. To tackle these issues, we introduce a general method for adapting a single-step diffusion model to new tasks and domains through adversarial learning objectives. Specifically, we consolidate various modules of the vanilla latent diffusion model into a single end-to-end generator network with small trainable weights, enhancing its ability to preserve the input image structure while reducing overfitting. We demonstrate that, for unpaired settings, our model CycleGAN-Turbo outperforms existing GAN-based and diffusion-based methods for various scene translation tasks, such as day-to-night conversion and adding/removing weather effects like fog, snow, and rain. We extend our method to paired settings, where our model pix2pix-Turbo is on par with recent works like Control-Net for Sketch2Photo and Edge2Image, but with a single-step inference. This work suggests that single-step diffusion models can serve as strong backbones for a range of GAN learning objectives. Our code and models are available at this https URL.

Ключевые изменения это замена UNet2DConditionModel на стандартный UNet2DModel, чтобы убрать лишние операции, так как предполагается что данная модель должна работать как фильтр по входному изображению в реальном времени для улучшения графики в играх. Также я убрал все улучшения по типу skip connection или lora finetuning. Такое упрощение делает модель совместимой со многими модификациями из других работ. В данном сетапе обучается сразу vae и unet. Входные Timesteps установлены максимальными 999. Добавлена автоматическая эвалюация на "lpips", "mse", "ssim", "dists", "psnr", "fid". 

**Техническое описание алгоритма**

Скрипт реализует обучение модели Image-to-Image (pix2pix) с архитектурой латентной диффузии для одношагового инференса. Обучение производится с учителем (supervised) на парном датасете.

### 1. Архитектура модели
*   **Backbone:** `UNet2DModel` (инициализируется с нуля по конфигу `unet2d_config`). В отличие от стандартного SD, используется безусловная архитектура (без Cross-Attention и Text Encoder). Вход/выход: 4 канала.
*   **VAE:** `AutoencoderTiny` (TAESD). Инициализируется весами `madebyollin/taesd`, но **полностью обучается** (`requires_grad=True`) вместе с UNet для адаптации декодера к целевому домену.
*   **Дискриминатор:** `vision_aided_loss.Discriminator` на базе CLIP (ViT-B/32) для состязательного обучения.

### 2. Алгоритм обучения (Training Loop)
Процесс представляет собой комбинацию дистилляции диффузии и GAN-обучения.

1.  **Входные данные:** Батч парных изображений $(x_{src}, x_{tgt})$ разрешением 512x512.
2.  **Проекция в латентное пространство:**
    *   Исходное изображение кодируется обучаемым VAE: $z_{src} = \mathcal{E}(x_{src})$.
3.  **Прямой проход (Forward Pass):**
    *   В UNet подается тензор $z_{src}$ и фиксированная временная метка $t=999$ (эмуляция полного шума/начала генерации).
    *   UNet предсказывает выходной тензор (шум/скорость).
    *   `DDPMScheduler` выполняет **один шаг** интеграции, преобразуя выход UNet в предсказанный латент $z_{pred}$.
    *   VAE декодирует результат в изображение: $\hat{x}_{tgt} = \mathcal{D}(z_{pred})$.
4.  **Вычисление функции потерь (Loss Function):**
    Итоговый лосс генератора состоит из трех компонентов:
    *   **Pixel Loss (MSE):** $||\hat{x}_{tgt} - x_{tgt}||^2_2$. Отвечает за низкочастотное соответствие.
    *   **Perceptual Loss (LPIPS):** Расстояние в пространстве признаков VGG ($\lambda_{lpips}=5.0$). Отвечает за структурную целостность.
    *   **Adversarial Loss (Generator):** Ошибка дискриминатора на сгенерированных изображениях ($\lambda_{gan}=0.5$).
5.  **Шаг Дискриминатора:**
    *   Вычисляется классический GAN-лосс (hinge или sigmoid) для различения реальных $x_{tgt}$ и сгенерированных $\hat{x}_{tgt}$ изображений.
6.  **Оптимизация:**
    *   Обновляются веса UNet и VAE (по лоссу генератора).
    *   Обновляются веса Дискриминатора (по лоссу дискриминатора).
    *   Используется AdamW (lr=5e-6), warmup 500 шагов.

### 3. Алгоритм инференса (Inference)
Инференс строго детерминирован и выполняется за один проход нейронной сети.

1.  Входное изображение $x_{in}$ кодируется VAE: $z_{in} = \mathcal{E}(x_{in})$.
2.  UNet принимает $z_{in}$ и $t=999$, генерируя латентное представление целевого изображения $z_{out}$ за **один вызов** (без итеративного денойзинга).
3.  VAE декодирует результат: $x_{out} = \mathcal{D}(z_{out})$.

```yaml
# Model configuration
# model_name_or_path: stablediffusionapi/juggernaut-reborn 
model_name_or_path: stabilityai/sd-turbo
# dataset_name: lambdalabs/naruto-blip-captions
dataset_name: dim/nfs_pix2pix_1920_1080_v5
seed: 2025

# Output configuration
output_dir: checkpoints/auto_remaster/sd1.5_ddpm

# Training optimization parameters
learning_rate: 5e-6
lr_scheduler_type: "constant"
warmup_steps: 500
weight_decay: 1e-2
max_grad_norm: 1.0
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
gradient_checkpointing: false

# Training duration
max_steps: 15000
save_steps: 400
resume_from_checkpoint: false
eval_on_start: false

# Reporting
report_to: wandb

# Diffusion specific parameters
resolution: 512
use_ema: false
noise_scheduler_type: stabilityai/sd-turbo

# Dataset column names
source_image_name: input_image
target_image_name: edited_image
caption_column: edit_prompt

# Cache directory
cache_dir: dataset/nfs_pix2pix_1920_1080_v5

# Inference parameters
num_inference_steps: 1

# Metrics for evaluation
metrics_list: [
    "lpips",
    "mse",
    "ssim",
    "dists",
    "psnr",
    "fid",
]

# Loss weights
lpips_factor: 5.0
gan_factor: 0.5
```