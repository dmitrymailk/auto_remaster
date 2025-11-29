### Порядок перебора методов для image2image
- img2img-turbo
- LBM
- VAE(обучить его конкретно под мой датасет)
- DDPM
- flow matching
- rectified flow
- DMD2
- I2SB: Image-to-Image Schrödinger Bridge
- Piecewise Rectified Flow
- LCM
- control net(обучение на синтетически сглаженных датасетах, затем перегенерация и снова сглаживание)
- [DREAM](https://github.com/jinxinzhou/dream)
- [Min-SNR Weighting Strategy](https://huggingface.co/papers/2303.09556)
- RMT-diffusion?

### Базовые методы которые стоит попробовать 
- аугментация через сдвиг, поворот и кроп
- EMA
- обучение на 8 шагов, затем дистиляция
- можно создать синтетический датасет с нужной позицией камеры при помощи qwen edit relight и гаусианов, так как мы можем свободно двигать камеру при просмотре гаусианнов. 
- инициализация с других моделей

### Идеи (которые не сработают)
- Так как задача супер узкая, можно попробовать сделать так чтобы латенты пар картинок лежали на одной прямой+- шум.

### Best models
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952)
- [Qwen-Image Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf)
- [FLUX.1](https://cdn.sanity.io/files/2gpum2i6/production/880b072208997108f87e5d2729d8a8be481310b5.pdf)
- [FLUX.2: Analyzing and Enhancing the Latent Space of FLUX – Representation Comparison](https://bfl.ai/research/representation-comparison)
- [Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752)
- [Stable Diffusion 3](https://arxiv.org/pdf/2403.03206) https://github.com/Stability-AI/sd3.5

### Fast diffusion models
- [Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer](https://github.com/NVlabs/Sana)
- [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://github.com/mit-han-lab/hart)
- [One-step image-to-image with Stable Diffusion turbo: sketch2image, day2night, and more](https://github.com/GaParmar/img2img-turbo)
- [SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training](https://github.com/ByteDance-Seed/SeedVR)
- [DiffO: Single-step Diffusion for Image Compression at Ultra-Low Bitrates](https://arxiv.org/pdf/2506.16572v1)
- [fast-DiT Scalable Diffusion Models with Transformers (DiT)](https://github.com/chuanyangjin/fast-DiT)

### GANs
- [The GAN is dead; long live the GAN! A Modern Baseline GAN](https://arxiv.org/pdf/2501.05441)

### Fast vae
- [Tiny AutoEncoder for Stable Diffusion](https://github.com/madebyollin/taesd)
- [DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space](https://github.com/dc-ai-projects/DC-Gen)
- [Notes / Links about Stable Diffusion VAE](https://gist.github.com/madebyollin/ff6aeadf27b2edbc51d05d5f97a595d9)
- [LiteVAE: Lightweight and Efficient Variational Autoencoder](https://arxiv.org/pdf/2405.14477)

### Quality vae
- [[ICCV 2025] Official implementation of the paper: REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](https://github.com/End2End-Diffusion/REPA-E)
- [[CVPR 2025 Oral] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://github.com/hustvl/LightningDiT)
- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://sihyun.me/REPA/)
- [[CVPR 2021 Oral] Soft-IntroVAE: Analyzing and Improving Introspective Variational Autoencoders](https://github.com/taldatech/soft-intro-vae-pytorch)
- [DIFFUSION TRANSFORMERS WITH REPRESENTATION AUTOENCODERS](https://arxiv.org/pdf/2510.11690)

### Diffusion Distillation
- [Knowledge-distilled, smaller versions of Stable Diffusion Segmind Distilled diffusion](https://github.com/segmind/distill-sd)
- [LATENT CONSISTENCY MODELS: SYNTHESIZING HIGH-RESOLUTION IMAGES WITH FEW-STEP INFERENCE](https://arxiv.org/pdf/2310.04378)
- [\[ICCV2025\] "Di\[M\]O: Distilling Masked Diffusion Models into One-step Generator](https://github.com/yuanzhi-zhu/DiMO)
- [One-Step Diffusion via Shortcut Models](https://github.com/kvfrans/shortcut-models)
- [rCM: Score-Regularized Continuous-Time Consistency Model](https://github.com/NVlabs/rcm)

### Diffusion Distillation (no code)
- [Diffusion Adversarial Post-Training for One-Step Video Generation](https://arxiv.org/pdf/2501.08316)
- [OSV: One Step is Enough for High-Quality Image to Video Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_OSV_One_Step_is_Enough_for_High-Quality_Image_to_Video_CVPR_2025_paper.pdf)

### hyperparameter tuning
- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/pdf/2312.02696)

### Vae train repos
- https://github.com/cloneofsimo/vqgan-training https://huggingface.co/fal/AuraEquiVAE
- https://github.com/KohakuBlueleaf/HakuLatent
- https://github.com/zelaki/eqvae
- https://huggingface.co/AiArtLab/sdxl_vae/blob/main/src/train_sdxl_vae.py
- https://github.com/hustvl/Turbo-VAED
- https://github.com/microsoft/Reducio-VAE
- https://github.com/VideoVerses/VideoVAEPlus
- https://github.com/bytetriper/RAE
- https://github.com/snap-research/alphaflow

### Losses
- https://github.com/richzhang/PerceptualSimilarity
- https://github.com/sypsyp97/convnext_perceptual_loss

### Theory
- [Generative modelling in latent space](https://sander.ai/2025/04/15/latents.html)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103)
- https://www.practical-diffusion.org/schedule/
- https://diffusion.csail.mit.edu/
- https://mbernste.github.io/posts/vae/
- https://d2l.ai/chapter_generative-adversarial-networks/gan.html

## TAESD reimlementation deconstruction
https://github.com/madebyollin/taesd/issues/16
- TAESD weights are initialized from scratch, but TAESD is supervised using outputs from the SD VAE.
- TAESD is just trained to directly mimic the SD encoder / decoder, without using any KL loss or whatever. In pseudocode:
```python
def taesd_training_loss(real_images, taesd, sdvae):
    taesd_latents = taesd.encoder(real_images)
    sdvae_latents = apply_scaling_factor(sdvae.encoder(real_images))
    taesd_images = taesd.decoder(sdvae_latents)
    
    # in the ideal case latent_loss and image_loss would be F.mse_loss,
    # but in practice they're some mix of adversarial and perceptual loss terms
    return latent_loss(taesd_latents, sdvae_latents) + image_loss(taesd_images, real_images)
```
- TAESD is entirely deterministic and doesn't use any KL loss.
- https://github.com/madebyollin/taesd/issues/2
- The TAESD encoder is just trained with MSE (against the SD encoder results), and the TAESD decoder is trained as a conditional GAN (conditioned on the SD encoder results).
- I only used discriminator, MSE and LPIPS
- Real img → encoder → Latent → decoder → recon img
- loss = a * mse(recon, real) + b * lpips(recon, real) + c * discrininator(recon, real). And we use very small a here so we said "tiny bit of mse"
- https://github.com/madebyollin/taesd/issues/11
- my optimizer was th.optim.Adam(model.parameters(), 3e-4, betas=(0.9, 0.9), amsgrad=True)
- My adversarial loss was "relativistic" (penalizing distance(disc(real).mean(), disc(fake).mean()), rather than just disc(fake).mean()) and I used a replay buffer of fakes for the discriminator training, both of which may have helped with stability. I also used several auxiliary losses (LPIPS + frequency-amplitude matching + MSE at 1/8 res) for the most recent model, which helped reduce the dependence on the adversarial loss a bit. I don't remember any persistent instability issues with this setup.

- https://github.com/madebyollin/taesd/issues/11#issuecomment-1914990359
Adding some more info here.

Changes in 1.2
For TAESD 1.2, I removed the LPIPS and other icky hand-coded losses (now just using adversarial + very faint lowres MSE). I also added adversarial loss to the encoder training as well (though I'm not sure it made a difference).

Various questions I've seen
Are the decoder targets GT images or SD-decoded images? GT; TAESD's decoder is a standalone conditional GAN, not a distilled model.
What dataset was used? Depends on model version, but usually some mix of photos (e.g. laion-aesthetic) and illustrations (e.g. danbooru2021), with some color / geometric augmentations
Do you delay adversarial loss until a certain number of steps (like the SD VAE does)? I usually prefer to start from a pretrained decoder model, but I don't have some specific number of steps in mind.
- What do you mean by low-res MSE loss? like F.mse_loss(F.avg_pool2d(decoded, 8), F.avg_pool2d(real_images, 8)). Just making sure that the color of each 8x8 patch is approximately correct.
Which reference VAEs do you use? https://huggingface.co/stabilityai/sd-vae-ft-ema and https://huggingface.co/madebyollin/sdxl-vae-fp16-fix - these are used to supervise the encoder and also as a gold standard for decoder quality.
Various figures
Color Augmentation
Color augmentation (occasional hue / saturation shifting of input images) helped improve reproduction of saturated colors (which are otherwise rare in aesthetic datasets)

image
Downsides of different losses
MSE/MAE can make everything very smooth (top is GT, bottom is a simple MSE-only decoder)
image

LPIPS can cause recognizable artifacts on faces & eyes (top is from a run with LPIPS, bottom is a run without it)

image
Adversarial loss can cause divergence if not handled properly:

image
Blue eyes
I don't remember what caused this

image

- I posted https://github.com/madebyollin/seraena/blob/main/TAESDXL_Training_Example.ipynb which should work as a starting point (and it does most of the complicated adversarial loss part). You can try adding additional pooled-MSE / LPIPS losses to speed up convergence