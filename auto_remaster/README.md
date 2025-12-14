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
- шеринг параметров и повторяющихся блоков или MOE из более малых частей(должно сработать лучше чем шеринг).
- для сохранения структуры сделать canny loss
- увеличения лосса для центра ослабление для краев
- Так как задача супер узкая, можно попробовать сделать так чтобы латенты пар картинок лежали на одной прямой+- шум.

### Самые близкие работы к моей
- [Cosmos Transfer 2.5 Sim2Real for Simulator Videos](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.html)
- [Enhancing Photorealism Enhancement](https://arxiv.org/pdf/2105.04619)
- [Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://opendrivelab.com/Vista/)
- [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://drivedreamer.github.io/)
- [DriveGAN](https://github.com/nv-tlabs/DriveGAN_code)
- [Genie 2: A large-scale foundation world model](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)
- [Genie 3: A new frontier for world models](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)

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
- [Reconstruct Anything Model (RAM)](https://github.com/matthieutrs/ram)

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
- [SimFlow: Simplified and End-to-End Training of Latent Normalizing Flows](https://github.com/Qinyu-Allen-Zhao/SimFlow)

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
- https://github.com/madebyollin/seraena (official taesd like vae distillation)
- https://github.com/KohakuBlueleaf/HakuLatent
- https://github.com/zelaki/eqvae
- https://huggingface.co/AiArtLab/sdxl_vae/blob/main/src/train_sdxl_vae.py
- https://github.com/hustvl/Turbo-VAED
- https://github.com/microsoft/Reducio-VAE
- https://github.com/VideoVerses/VideoVAEPlus
- https://github.com/bytetriper/RAE
- https://github.com/snap-research/alphaflow
- https://github.com/segmind/distill-sd

### Losses
- https://github.com/richzhang/PerceptualSimilarity
- https://github.com/sypsyp97/convnext_perceptual_loss

### Theory
- [Generative modelling in latent space](https://sander.ai/2025/04/15/latents.html)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103)
- https://lilianweng.github.io/posts/2018-08-12-vae/
- https://www.practical-diffusion.org/schedule/
- https://diffusion.csail.mit.edu/
- https://mbernste.github.io/posts/vae/
- https://d2l.ai/chapter_generative-adversarial-networks/gan.html

### Ссылки на датасеты
- [GenAD: OpenDV Dataset The largest driving video dataset to date, containing more than 1700 hours of real-world driving videos.](https://opendrivelab.com/datasets)
