"""
RunDiffusion/Juggernaut-XL-v9 — raw one-step benchmark
Замеряет только UNet + VAE, без pipeline.__call__.
Текстовые эмбеддинги вычисляются один раз заранее.
"""

import time
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from torch.nn.attention import sdpa_kernel, SDPBackend

TAESD_PATH = "taesd"
sys.path.insert(0, TAESD_PATH)
from taesd import TAESD

MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"

PROMPT = "A cat walks on the grass, realistic style, photorealistic, 4k."
NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, deformed, ugly, extra fingers, "
    "poorly drawn hands, poorly drawn face, disfigured, watermark"
)

NUM_INFERENCE_STEPS = 1
HEIGHT = 512
WIDTH  = 512
GUIDANCE_SCALE = 0.0   # отключён CFG

OUTPUT_IMAGE    = "sdxl_output.png"
NUM_WARMUP_RUNS = 1
NUM_BENCHMARK_RUNS = 50

DEVICE = "cuda"
DTYPE  = torch.bfloat16

print("=" * 60)
print("Juggernaut-XL-v9  |  Raw one-step benchmark")
print("=" * 60)

# ── Загрузка компонентов ─────────────────────────────────────
print(f"\n[1/3] Loading model from '{MODEL_ID}' ...")
t0 = time.time()

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16",
    use_safetensors=True,
).to(DEVICE)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

unet      = pipe.unet
scheduler = pipe.scheduler

print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

# ── TAESD decoder вместо полного VAE ──────────────────
print(f"   Loading TAESD from 'taesd/taesdxl_decoder.pth' ...")
taesd = TAESD(encoder_path=None, decoder_path="taesd/taesdxl_decoder.pth").to(DEVICE, DTYPE)
taesd.eval()
print(f"   ✓ TAESD ready")

# ── Текстовые эмбеддинги (один раз) ─────────────────────────
print("\n[2/3] Pre-computing text embeddings ...")
with torch.no_grad():
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=None,
        negative_prompt=NEGATIVE_PROMPT,
        negative_prompt_2=None,
        device=DEVICE,
        do_classifier_free_guidance=False,  # guidance_scale=0 → CFG выключен
    )
# SDXL требует add_time_ids
add_time_ids = torch.tensor(
    [[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]],
    dtype=DTYPE, device=DEVICE,
)
print("   ✓ Embeddings ready")

# ── Бенчмарк ─────────────────────────────────────────────────
print(f"\n[3/3] Benchmark: {NUM_WARMUP_RUNS} warmup + {NUM_BENCHMARK_RUNS} measured runs")
print(f"      steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH}")
print(f"      measuring: UNet + VAE only\n")

LATENT_H = HEIGHT // 8
LATENT_W = WIDTH  // 8
LATENT_CHANNELS = unet.config.in_channels  # 4

# ── Один шаг без шедулера: предвычислим sigma, timestep, c_in ───────
# EulerAncestral epsilon 1-step: x_prev = latent - sigma * noise_pred
scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
_t_fixed   = scheduler.timesteps[0].to(DEVICE)
_sigma     = scheduler.sigmas[0].to(device=DEVICE, dtype=DTYPE)
_c_in      = (1.0 / (_sigma ** 2 + 1) ** 0.5)            # scale_model_input коэффициент
_init_scale = scheduler.init_noise_sigma              # масштаб начального шума


def run_inference():
    generator = torch.Generator(DEVICE).manual_seed(42)
    latents = torch.randn(
        1, LATENT_CHANNELS, LATENT_H, LATENT_W,
        device=DEVICE, dtype=DTYPE, generator=generator,
    ) * _init_scale

    scaled = latents * _c_in      # scale_model_input без шедулера

    # ── UNet ──────────────────────────────────────────────────────────
    torch.cuda.synchronize()
    _t = time.time()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.no_grad():
        noise_pred = unet(
            scaled, _t_fixed,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    t_unet = time.time() - _t

    # 1-step euler: x_prev = latent - sigma * noise_pred  (sigma_prev=0)
    latents = latents - _sigma * noise_pred

    # VAE decode ── TAESD (без /scaling_factor, напрямую сырые latents)
    torch.cuda.synchronize()
    _t = time.time()
    with torch.no_grad():
        image = taesd.decoder(latents).clamp(0, 1)
    torch.cuda.synchronize()
    t_vae = time.time() - _t

    # image: (1, 3, H, W) in [0, 1]
    image = image.cpu().float().numpy()[0].transpose(1, 2, 0)
    return (image * 255).astype(np.uint8), t_unet, t_vae

# Warmup
print("  Warmup...")
for i in range(NUM_WARMUP_RUNS):
    torch.cuda.synchronize()
    t_s = time.time()
    out, _, _ = run_inference()
    torch.cuda.synchronize()
    print(f"    warmup {i+1}: {time.time() - t_s:.2f}s")

# Прогоны
print("\n  Measuring...")
times, times_unet, times_vae = [], [], []
for i in range(NUM_BENCHMARK_RUNS):
    torch.cuda.synchronize()
    t_s = time.time()
    out, t_unet, t_vae = run_inference()
    torch.cuda.synchronize()
    times.append(time.time() - t_s)
    times_unet.append(t_unet)
    times_vae.append(t_vae)

Image.fromarray(out).save(OUTPUT_IMAGE)
print(f"\n  Image saved → {OUTPUT_IMAGE}")

avg      = sum(times) / len(times)
avg_unet = sum(times_unet) / len(times_unet)
avg_vae  = sum(times_vae) / len(times_vae)
print("\n" + "=" * 60)
print(f"RESULTS  (steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH})")
print(f"  runs           : {NUM_BENCHMARK_RUNS}")
print(f"  avg total      : {avg * 1000:.1f} ms  ({1.0 / avg:.2f} img/s)")
print(f"  avg unet       : {avg_unet * 1000:.1f} ms  ({avg_unet / avg * 100:.1f}%)")
print(f"  avg vae decode : {avg_vae  * 1000:.1f} ms  ({avg_vae  / avg * 100:.1f}%)")
print(f"  min / max      : {min(times) * 1000:.1f} / {max(times) * 1000:.1f} ms")
print(f"  GPU mem        : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("=" * 60)
