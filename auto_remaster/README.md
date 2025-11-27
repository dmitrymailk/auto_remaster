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

### Best models
- [Qwen-Image Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf)
- [FLUX2]()

### Fast diffusion models
- [Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer](https://github.com/NVlabs/Sana)
- [HART: Efficient Visual Generation with Hybrid Autoregressive Transformer](https://github.com/mit-han-lab/hart)
- [One-step image-to-image with Stable Diffusion turbo: sketch2image, day2night, and more](https://github.com/GaParmar/img2img-turbo)
- [SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training](https://github.com/ByteDance-Seed/SeedVR)
- [DiffO: Single-step Diffusion for Image Compression at Ultra-Low Bitrates](https://arxiv.org/pdf/2506.16572v1)

### Fast vae
- [Tiny AutoEncoder for Stable Diffusion](https://github.com/madebyollin/taesd)
- [DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space](https://github.com/dc-ai-projects/DC-Gen)


### Quality vae
- [[ICCV 2025] Official implementation of the paper: REPA-E: Unlocking VAE for End-to-End Tuning of Latent Diffusion Transformers](https://github.com/End2End-Diffusion/REPA-E)
- [[CVPR 2025 Oral] Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models](https://github.com/hustvl/LightningDiT)
- [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://sihyun.me/REPA/)

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
- https://github.com/cloneofsimo/vqgan-training?tab=readme-ov-file#details
- https://github.com/SonyResearch/micro_diffusion
- https://github.com/microsoft/Reducio-VAE

### Theory
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103)