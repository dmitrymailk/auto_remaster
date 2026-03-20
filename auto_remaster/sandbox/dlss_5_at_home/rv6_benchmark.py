"""
SG161222/Realistic_Vision_V6.0_B1_noVAE — raw one-step benchmark
SD 1.5 pipeline. Замеряет только UNet + VAE (TAESD).
"""

import time
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from torch.nn.attention import sdpa_kernel, SDPBackend

TAESD_PATH = "taesd"
sys.path.insert(0, TAESD_PATH)
from taesd import TAESD

MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"

PROMPT = "A cat walks on the grass, realistic style, photorealistic, 4k."
NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, deformed, ugly, extra fingers, "
    "poorly drawn hands, poorly drawn face, disfigured, watermark"
)

NUM_INFERENCE_STEPS = 1
HEIGHT = 512
WIDTH  = 512

OUTPUT_IMAGE    = "rv6_output.png"
NUM_WARMUP_RUNS = 1
NUM_BENCHMARK_RUNS = 50

DEVICE = "cuda"
DTYPE  = torch.bfloat16

print("=" * 60)
print("Realistic_Vision_V6 (SD1.5) |  Raw one-step benchmark")
print("=" * 60)

# ── Загрузка ─────────────────────────────────────────────────
print(f"\n[1/3] Loading model from '{MODEL_ID}' ...")
t0 = time.time()

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
).to(DEVICE)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

unet      = pipe.unet
scheduler = pipe.scheduler

print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

# ── TAESD decoder ────────────────────────────────────────────
print(f"   Loading TAESD from 'taesd/taesd_decoder.pth' ...")
taesd = TAESD(encoder_path=None, decoder_path="taesd/taesd_decoder.pth").to(DEVICE, DTYPE)
taesd.eval()
print(f"   ✓ TAESD ready")

# ── Embeddings (один раз) ────────────────────────────────────
print("\n[2/3] Pre-computing text embeddings ...")
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=PROMPT,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=NEGATIVE_PROMPT,
    )
print("   ✓ Embeddings ready")

# ── Бенчмарк ─────────────────────────────────────────────────
print(f"\n[3/3] Benchmark: {NUM_WARMUP_RUNS} warmup + {NUM_BENCHMARK_RUNS} measured runs")
print(f"      steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH}, measuring: UNet + VAE\n")

LATENT_H = HEIGHT // 8
LATENT_W = WIDTH  // 8
LATENT_CHANNELS = unet.config.in_channels  # 4

# Предвычислим sigma и timestep
scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
_t_fixed    = scheduler.timesteps[0].to(DEVICE)
_sigma      = scheduler.sigmas[0].to(device=DEVICE, dtype=DTYPE)
_c_in       = (1.0 / (_sigma ** 2 + 1) ** 0.5)
_init_scale = scheduler.init_noise_sigma


def run_inference():
    generator = torch.Generator(DEVICE).manual_seed(42)
    latents = torch.randn(
        1, LATENT_CHANNELS, LATENT_H, LATENT_W,
        device=DEVICE, dtype=DTYPE, generator=generator,
    ) * _init_scale

    scaled = latents * _c_in

    # ── UNet ──────────────────────────────────────────────────
    torch.cuda.synchronize()
    _t = time.time()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.no_grad():
        noise_pred = unet(
            scaled, _t_fixed,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    t_unet = time.time() - _t

    # 1-step euler
    latents = latents - _sigma * noise_pred

    # ── TAESD decode ───────────────────────────────────────────
    torch.cuda.synchronize()
    _t = time.time()
    with torch.no_grad():
        image = taesd.decoder(latents).clamp(0, 1)
    torch.cuda.synchronize()
    t_vae = time.time() - _t

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
print(f"RESULTS  (steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH}, SD1.5)")
print(f"  runs           : {NUM_BENCHMARK_RUNS}")
print(f"  avg total      : {avg * 1000:.1f} ms  ({1.0 / avg:.2f} img/s)")
print(f"  avg unet       : {avg_unet * 1000:.1f} ms  ({avg_unet / avg * 100:.1f}%)")
print(f"  avg vae decode : {avg_vae  * 1000:.1f} ms  ({avg_vae  / avg * 100:.1f}%)")
print(f"  min / max      : {min(times) * 1000:.1f} / {max(times) * 1000:.1f} ms")
print(f"  GPU mem        : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("=" * 60)
