# Benchmarks

> Методология: raw denoising loop (без pipeline overhead), embeddings вычислены заранее.  
> 1 шаг инференса, 512×512, bfloat16, 50 прогонов после 1 warmup.

## Wan2.1-VACE-1.3B

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **116.6 ms (8.57 img/s)** |
| avg transformer | 43.6 ms (37.3%) |
| avg VAE decode | 64.2 ms (55.1%) |
| min / max | 107.9 / 121.0 ms |
| GPU memory | 15.12 GB |
| Notes | Full AutoencoderKLWan; scheduler убран; VAE = основной bottleneck |

### Wan2.1-VACE-1.3B + TAEHV (taew2_1)

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **55.7 ms (17.94 img/s) ↑ 2.09×** |
| avg transformer | 43.3 ms (77.6%) |
| avg VAE decode | **4.9 ms (8.7%)** ↓ 13× vs full VAE |
| min / max | 50.4 / 60.0 ms |
| GPU memory | 15.42 GB |
| Notes | TAEHV tiny VAE; Transformer — единственный bottleneck |

### Wan2.1-VACE-1.3B + TAEHV + Flash Attention

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **56.3 ms (17.76 img/s)** |
| avg transformer | 43.7 ms (77.6%) |
| avg VAE decode | 4.8 ms (8.6%) |
| min / max | 52.6 / 59.5 ms |
| GPU memory | 15.42 GB |
| Notes | sdpa_kernel(FLASH_ATTENTION) на transformer; прирост незначительный (≈±0.6ms) |

---

## Juggernaut-XL-v9 (SDXL)

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **60.4 ms (16.54 img/s)** |
| avg UNet | 26.0 ms (43.1%) |
| avg VAE decode | 26.9 ms (44.5%) |
| min / max | 54.8 / 64.6 ms |
| GPU memory | 7.17 GB |
| Notes | UNet + VAE почти пополам, scheduler убран, параметров в 2× больше чем Wan 1.3B |

### Juggernaut-XL-v9 + TAESD + Flash Attention

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **32.5 ms (30.80 img/s) ↑ 1.86×** |
| avg UNet | 24.5 ms (75.6%) |
| avg VAE decode | **2.2 ms (6.8%)** ↓ 12× vs full VAE |
| min / max | 26.8 / 38.5 ms |
| GPU memory | 6.74 GB |
| Notes | TAESD taesdxl decoder; UNet — единственный bottleneck |

---

## segmind/SSD-1B

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **22.4 ms (44.64 img/s)** |
| avg UNet | 15.6 ms (69.7%) |
| avg VAE decode | 2.2 ms (9.8%) TAESD |
| min / max | 20.7 / 25.8 ms |
| GPU memory | 4.38 GB |
| Notes | TAESD taesdxl; Flash Attention; в 2× меньше параметров чем Juggernaut-XL |

---

## segmind/Segmind-Vega

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **18.0 ms (55.55 img/s)** |
| avg UNet | 10.6 ms (59.0%) |
| avg VAE decode | 2.2 ms (12.2%) TAESD |
| min / max | 13.4 / 23.3 ms |
| GPU memory | 3.27 GB |
| Notes | TAESD taesdxl; Flash Attention; самый быстрый результат |

---

## SG161222/Realistic_Vision_V6.0_B1 (SD 1.5)

| Параметр | Значение |
|---|---|
| Дата | 2026-03-21 |
| Разрешение | 512×512 |
| Шаги | 1 |
| Прецизионность | bfloat16 |
| **avg total** | **20.8 ms (48.05 img/s)** |
| avg UNet | 13.7 ms (65.9%) |
| avg VAE decode | 2.2 ms (10.6%) TAESD |
| min / max | 17.7 / 24.2 ms |
| GPU memory | **2.18 GB** |
| Notes | TAESD taesd_decoder; Flash Attention; 860M параметров, SD1.5 pipeline |
