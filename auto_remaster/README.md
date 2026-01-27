### Порядок перебора методов для image2image
- img2img-turbo
- LBM
- vqgan-training
- seraena
- REPA-E
- [TwinFlow](https://github.com/inclusionAI/TwinFlow)
- flow matching
- rectified flow
- InstaFlow https://github.com/gnobitab/InstaFlow https://github.com/isamu-isozaki/diffusers/blob/rectified_flow/examples/rectified_insta_flow/train_rectified_instaflow.py https://colab.research.google.com/drive/13GmggtFLnVj55i2XnVp-E_-tLgv2ThSm#scrollTo=9QpxgPmqXZQd (нет никакого смысла воспроизводить его, это просто rectified flow)
- [NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation](https://arxiv.org/pdf/2512.05106)
- pi-flow https://github.com/Lakonik/piFlow
- https://github.com/guandeh17/Self-Forcing
- alphaflow https://github.com/snap-research/alphaflow
- [FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner (NeurIPS 2024)](https://github.com/shiml20/FlowTurbo)
- [Straighter and Faster: Efficient One-Step Generative Modeling via Meanflow on Rectified Trajectories](https://github.com/Xinxi-Zhang/Re-MeanFlow)
- [Mean Flows for One-step Generative Modeling](https://github.com/haidog-yaqub/MeanFlow)
- https://github.com/Candibulldog/MeanFlow-Edge2Image
- [Diffusion-GAN — Official PyTorch implementation](https://github.com/Zhendong-Wang/Diffusion-GAN)
- LCM https://github.com/huggingface/diffusers/blob/main/docs/source/en/training/lcm_distill.md
- DMD2 
- https://zsyoaoa.github.io/projects/resshift/
- [rCM](https://github.com/NVlabs/rcm)
- LightningDiT
- The GAN is dead
- I2SB: Image-to-Image Schrödinger Bridge
- GigaGAN
- shortcut-models
- DDPM
- Piecewise Rectified Flow
- control net(обучение на синтетически сглаженных датасетах, затем перегенерация и снова сглаживание)
- [DREAM](https://github.com/jinxinzhou/dream)
- [Min-SNR Weighting Strategy](https://huggingface.co/papers/2303.09556)
- https://github.com/zhuyu-cs/MeanFlow
- https://github.com/gnobitab/FlowGrad
- RMT-diffusion?
- TRM-block?

### Оценка генеративных моделей
- DreamStyle: A Unified Framework for Video Stylization

### Базовые методы которые стоит попробовать 
- FlowTurbo ОБЯЗАТЕЛЬНО нужно сделать, но это все равно стоит рассматривать только как метод ускорения. он требует всего 4% параметров для значимого ускорения.
- для создания крутой базовой модели можно попробовать просто сделать lora на qwen image, из нее уже потом пытаться что-то дистиллировать(это на самом деле очень хорошая идея, примерно так сделали SnapGen++)
- сделать лосс для нижней половины экрана более высоким чем для остальной картинки
- учиться на кропах в высоком разрешении, а не только на всей картинке сразу
- аугментация через сдвиг, поворот и кроп
- EMA
- обучение на 8 шагов, затем дистиляция
- можно создать синтетический датасет с нужной позицией камеры при помощи qwen edit relight и гаусианов, так как мы можем свободно двигать камеру при просмотре гаусианнов. 
- инициализация с других моделей
- пареллельно в сети моделировать низкие частоты и высокие, и в конце их объединять как в https://arxiv.org/pdf/2508.19789 StableIntrinsic: Detail-preserving One-step Diffusion Model for Multi-view Material Estimation

### Идеи (которые не сработают)
- HRM модель, упор не на 1 шаг, а на множество, но очень легкой моделью
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
- https://research.google/blog/mobilediffusion-rapid-text-to-image-generation-on-device/


### Лоссы для текстуры
- LPIPS
- Focal Frequency Loss

### Neural rendering
- [DIFFUSIONRENDERER: Neural Inverse and Forward Rendering with Video Diffusion Models](https://arxiv.org/pdf/2501.18590)
- [StableIntrinsic: Detail-preserving One-step Diffusion Model for Multi-view Material Estimation](https://arxiv.org/pdf/2508.19789)
- [Materialist: Physically Based Editing Using Single-Image Inverse Rendering](https://arxiv.org/pdf/2501.03717)
- [MegaDepth: Learning Single-View Depth Prediction from Internet Photos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf)
- [NeAR: Coupled Neural Asset–Renderer Stack](https://arxiv.org/pdf/2511.18600)
- [WHAT MATTERS WHEN REPURPOSING DIFFUSIONMODELS FOR GENERAL DENSE PERCEPTION TASKS?](https://arxiv.org/pdf/2403.06090)
- [Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation](https://github.com/Daniellli/DKT)

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
- [Official PyTorch and Diffusers Implementation of "LinFusion: 1 GPU, 1 Minute, 16K Image"](https://github.com/fla-org/flash-bidirectional-linear-attention)
- [SnapGen++: Unleashing Diffusion Transformers for Efficient High-Fidelity Image Generation on Edge Devices](https://arxiv.org/pdf/2601.08303)
- [SnapGen: Taming High-Resolution Text-to-Image Models for Mobile Devices with Efficient Architectures and Training](https://arxiv.org/pdf/2412.09619)
- [PocketSR: The Super-Resolution Expert in Your Pocket Mobiles](https://arxiv.org/pdf/2510.03012)
- https://github.com/horseee/DeepCache
- [BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion](https://arxiv.org/pdf/2305.15798)
- [Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers](https://crowsonkb.github.io/hourglass-diffusion-transformers/)

### GANs
- [The GAN is dead; long live the GAN! A Modern Baseline GAN](https://arxiv.org/pdf/2501.05441)
- [Implementation of GigaGAN, new SOTA GAN out of Adobe.](https://github.com/lucidrains/gigagan-pytorch)

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

- [Phased Consistency Models](https://github.com/G-U-N/Phased-Consistency-Model)
- [Latent Consistency Distillation Example](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)

## Flow matching distillation  
- [InstaFlow! One-Step Stable Diffusion with Rectified Flow](https://github.com/gnobitab/InstaFlow) (это обычный rectified flow с дистиляцией на втором шаге, оригинального кода нет, лучше просто рассмотреть стандартный rectified flow)
- [pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation](https://github.com/Lakonik/piFlow)
- [SiD-DiT Score Distillation of Flow Matching Models](https://yigu1008.github.io/SiD-DiT/)
- [SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](https://github.com/XingtongGe/SenseFlow)
- [Improved Distribution Matching Distillation for Fast Image Synthesis](https://github.com/tianweiy/DMD2.git)
- [This repo provides a working re-implementation of Latent Adversarial Diffusion Distillation by AMD](https://github.com/AMD-AGI/Nitro-1)
- [Official codebase for "Efficient Distillation of Classifier-Free Guidance using Adapters."](https://github.com/cristianpjensen/agd)
- [Consistency Distillation with Target Timestep Selection and Decoupled Guidance](https://github.com/FireRedTeam/Target-Driven-Distillation)
- [rCM: Score-Regularized Continuous-Time Consistency Model](https://github.com/NVlabs/rcm)
- [Official Implementation for "Consistency Flow Matching: Defining Straight Flows with Velocity Consistency"](https://github.com/YangLing0818/consistency_flow_matching) 
- [Official PyTorch Implementation of "Flow Map Distillation Without Data"](https://github.com/ShangyuanTong/FreeFlow)

### Diffusion Distillation (no code)
- [Diffusion Adversarial Post-Training for One-Step Video Generation](https://arxiv.org/pdf/2501.08316)
- [OSV: One Step is Enough for High-Quality Image to Video Generation](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_OSV_One_Step_is_Enough_for_High-Quality_Image_to_Video_CVPR_2025_paper.pdf)

### stochastic interpolants
- [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571)
- [Flow map matching with stochastic interpolants: A mathematical framework for consistency models](https://arxiv.org/pdf/2406.07507v2)

### hyperparameter tuning
- [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/pdf/2312.02696)

### Vae train repos
- https://github.com/cloneofsimo/vqgan-training https://huggingface.co/fal/AuraEquiVAE
- https://github.com/madebyollin/seraena (official taesd like vae distillation)
- https://github.com/LINs-lab/UCGM
- https://github.com/KohakuBlueleaf/HakuLatent
- https://github.com/zelaki/eqvae
- https://huggingface.co/AiArtLab/sdxl_vae/blob/main/src/train_sdxl_vae.py
- https://github.com/hustvl/Turbo-VAED
- https://github.com/microsoft/Reducio-VAE
- https://github.com/VideoVerses/VideoVAEPlus
- https://github.com/bytetriper/RAE
- https://github.com/snap-research/alphaflow
- https://github.com/segmind/distill-sd

### Super Resolution
- https://zsyoaoa.github.io/projects/resshift/
- https://github.com/wyf0912/SinSR
- https://github.com/XPixelGroup/BasicSR
- https://github.com/hongyuanyu/SPAN
- https://github.com/neosr-project/neosr
- https://github.com/XPixelGroup/HAT
- https://github.com/zsyOAOA/InvSR
- [One-Step Effective Diffusion Network for Real-World Image Super-Resolution](https://github.com/cswry/OSEDiff)

### Losses
- https://github.com/richzhang/PerceptualSimilarity
- https://github.com/sypsyp97/convnext_perceptual_loss

### Train Frameworks
- https://github.com/bghira/SimpleTuner
- https://github.com/modelscope/DiffSynth-Studio
- https://github.com/tdrussell/diffusion-pipe
- https://github.com/aigc-apps/VideoX-Fun


### Theory

- [An Introduction to Flow Matching and Diffusion Models](https://arxiv.org/pdf/2506.02070)
- [A Visual Dive into Conditional Flow Matching](https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/)
- [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929)
- [Generative modelling in latent space](https://sander.ai/2025/04/15/latents.html)
- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/pdf/2403.18103)
- [Diffusion models from scratch, from a new theoretical perspective](https://www.chenyang.co/diffusion.html)
- [Diffusion Meets Flow Matching: Two Sides of the Same Coin](https://diffusionflow.github.io/)
- [Interpreting and Improving Diffusion Models from an Optimization Perspective](https://arxiv.org/pdf/2306.04848)
- [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)
- https://lilianweng.github.io/posts/2018-08-12-vae/
- https://www.practical-diffusion.org/schedule/
- https://diffusion.csail.mit.edu/
- https://mbernste.github.io/posts/vae/
- https://d2l.ai/chapter_generative-adversarial-networks/gan.html
- https://www.physicsbaseddeeplearning.org/probmodels-intro.html
- [Flowing Through Continuous-Time Generative Models: A Clear and Systematic Tour](https://icml.cc/virtual/2025/40011)
- [Let us Flow Together](https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf)
- https://rectifiedflow.github.io/

### Normal Map, Depth Estimation
- [NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](https://normalcrafter.github.io/)
- [UniGeo: Taming Video Diffusion for Unified Consistent Geometry Estimation](https://github.com/SunYangtian/UniGeo)
- [\[CVPR 2025 Highlight\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](https://github.com/DepthAnything/Video-Depth-Anything)
- [RollingDepth: Video Depth without Video Models](https://github.com/prs-eth/rollingdepth)
- [Diffusion Knows Transparency : Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation](https://daniellli.github.io/projects/DKT/)
- [Light of Normals: Unified Feature Representation for Universal Photometric Stereo](https://github.com/houyuanchen111/LINO_UniPS)
- [Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction](https://lotus3d.github.io/)

### Нормали из гаусианов
- 

### Ссылки на датасеты с машинами
- [GenAD: OpenDV Dataset The largest driving video dataset to date, containing more than 1700 hours of real-world driving videos.](https://opendrivelab.com/datasets)
- https://www.shutterstock.com/video/search/rally-car-racing
- https://www.kaggle.com/datasets/manideep1108/culane
- https://www.kaggle.com/datasets/manideep1108/tusimple
- https://www.nuscenes.org/
- https://bair.berkeley.edu/blog/2018/05/30/bdd/
- https://www.a2d2.audi/en/dataset/ - хороший кажется
- [VIL-100: A New Dataset and A Baseline Model for Video Instance Lane Detection](https://github.com/yujun0-0/MMA-Net)
- [comma2k19 comma.ai presents comma2k19, a dataset of over 33 hours of commute in California's 280 highway.](https://github.com/commaai/comma2k19)
- [OpenLKA: An Open Dataset for Lane Keeping Assist Systems](https://github.com/OpenLKA/OpenLKA)

