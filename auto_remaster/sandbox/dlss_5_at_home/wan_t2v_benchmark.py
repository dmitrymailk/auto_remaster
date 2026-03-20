"""
Wan2.1-VACE-1.3B — raw one-step benchmark
Замеряет только Transformer + VAE, без pipeline.__call__.
Раздельный замер трансформера и VAE.
"""

import time
import sys
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from torch.nn.attention import sdpa_kernel, SDPBackend

TAEHV_PATH = "taehv/taew2_1.pth"
sys.path.insert(0, "taehv")
from taehv import TAEHV

MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"

PROMPT = "A cat walks on the grass, realistic style."
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residual, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn face, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

NUM_INFERENCE_STEPS = 1
HEIGHT     = 512
WIDTH      = 512
NUM_FRAMES = 1
GUIDANCE_SCALE = 1.0

OUTPUT_IMAGE       = "wan_output.png"
NUM_WARMUP_RUNS    = 1
NUM_BENCHMARK_RUNS = 50

DEVICE = "cuda"
DTYPE  = torch.bfloat16

print("=" * 60)
print("Wan2.1-VACE-1.3B  |  Raw one-step T2I benchmark")
print("=" * 60)

# ── Загрузка ─────────────────────────────────────────────────
print(f"\n[1/3] Loading model from '{MODEL_ID}' ...")
t0 = time.time()
vae  = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE)
pipe = WanVACEPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE).to(DEVICE)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

transformer = pipe.transformer
scheduler   = pipe.scheduler
print(f"   ✓ Loaded in {time.time() - t0:.1f}s")

# ── TAEHV вместо полного VAE ────────────────────
print(f"   Loading TAEHV from '{TAEHV_PATH}' ...")
taehv = TAEHV(checkpoint_path=TAEHV_PATH).to(DEVICE, DTYPE)
taehv.eval()
print(f"   ✓ TAEHV ready")

# ── Embeddings + video conditioning (один раз) ───────────────
print("\n[2/3] Pre-computing embeddings & conditioning ...")
with torch.no_grad():
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        do_classifier_free_guidance=False,
        device=DEVICE,
    )

with torch.no_grad():
    video, mask, reference_images = pipe.preprocess_conditions(
        video=None, mask=None, reference_images=None,
        batch_size=1, height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        dtype=torch.float32, device=DEVICE,
    )
    conditioning_latents = pipe.prepare_video_latents(video, mask, reference_images, generator=None, device=DEVICE)
    mask_latents = pipe.prepare_masks(mask, reference_images, generator=None)
    conditioning_latents = torch.cat([conditioning_latents, mask_latents], dim=1).to(DTYPE)

vace_layers        = transformer.config.vace_layers
conditioning_scale = torch.ones(len(vace_layers), device=DEVICE, dtype=DTYPE)
print("   ✓ Ready")


# ── Один шаг без шедулера: предвычислим sigma и timestep ────────
# Wan использует flow_prediction: x0 = latent - sigma * noise_pred
scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)
_t0 = scheduler.timesteps[0]                    # единственный timestep
_sigma = scheduler.sigmas[0].to(device=DEVICE, dtype=DTYPE)  # σ в bf16 на GPU
_timestep_fixed = _t0.expand(1).to(DEVICE)      # shape [1]

lat_channels   = transformer.config.in_channels
vae_spatial    = pipe.vae_scale_factor_spatial
vae_temporal   = pipe.vae_scale_factor_temporal
num_lat_frames = (NUM_FRAMES - 1) // vae_temporal + 1
latent_h       = HEIGHT // vae_spatial
latent_w       = WIDTH  // vae_spatial

print(f"\n[3/3] Benchmark: {NUM_WARMUP_RUNS} warmup + {NUM_BENCHMARK_RUNS} measured runs")
print(f"      steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH}, σ={_sigma.item():.4f}, measuring: transformer + VAE\n")


def run_inference():
    generator = torch.Generator(DEVICE).manual_seed(42)
    latents = torch.randn(
        1, lat_channels, num_lat_frames, latent_h, latent_w,
        device=DEVICE, dtype=DTYPE, generator=generator,
    )

    # ── Transformer ─────────────────────────────────────────────
    torch.cuda.synchronize()
    _t = time.time()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.no_grad():
        noise_pred = transformer(
            hidden_states=latents,
            timestep=_timestep_fixed,
            encoder_hidden_states=prompt_embeds,
            control_hidden_states=conditioning_latents,
            control_hidden_states_scale=conditioning_scale,
            return_dict=False,
        )[0]
    torch.cuda.synchronize()
    t_transformer = time.time() - _t

    # 1-step flow: x0 = latent - sigma * noise_pred  (без вызова шедулера)
    latents = latents - _sigma * noise_pred

    # VAE decode ── TAEHV (сырые latents без скейлинга)
    # TAEHV ждёт NTCHW, diffusers дает NCTHW → переставим
    latents_ntchw = latents.transpose(1, 2)   # (1, T, C, H, W)
    torch.cuda.synchronize()
    _t = time.time()
    with torch.no_grad():
        frames_ntchw = taehv.decode_video(latents_ntchw, parallel=True, show_progress_bar=False)
    torch.cuda.synchronize()
    t_vae = time.time() - _t

    # frames_ntchw: (1, T_out, 3, H, W) in [0, 1]
    frame = frames_ntchw[0, 0]   # (3, H, W)
    frame = frame.cpu().float().numpy().transpose(1, 2, 0)
    return (frame.clip(0, 1) * 255).astype(np.uint8), t_transformer, t_vae


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
times, times_tr, times_vae = [], [], []
for i in range(NUM_BENCHMARK_RUNS):
    torch.cuda.synchronize()
    t_s = time.time()
    out, t_tr, t_vae = run_inference()
    torch.cuda.synchronize()
    times.append(time.time() - t_s)
    times_tr.append(t_tr)
    times_vae.append(t_vae)

Image.fromarray(out).save(OUTPUT_IMAGE)
print(f"\n  Image saved → {OUTPUT_IMAGE}")

avg     = sum(times) / len(times)
avg_tr  = sum(times_tr) / len(times_tr)
avg_vae = sum(times_vae) / len(times_vae)
print("\n" + "=" * 60)
print(f"RESULTS  (steps={NUM_INFERENCE_STEPS}, {HEIGHT}x{WIDTH}, T2I)")
print(f"  runs           : {NUM_BENCHMARK_RUNS}")
print(f"  avg total      : {avg * 1000:.1f} ms  ({1.0 / avg:.2f} img/s)")
print(f"  avg transformer: {avg_tr  * 1000:.1f} ms  ({avg_tr  / avg * 100:.1f}%)")
print(f"  avg vae decode : {avg_vae * 1000:.1f} ms  ({avg_vae / avg * 100:.1f}%)")
print(f"  min / max      : {min(times) * 1000:.1f} / {max(times) * 1000:.1f} ms")
print(f"  GPU mem        : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("=" * 60)
