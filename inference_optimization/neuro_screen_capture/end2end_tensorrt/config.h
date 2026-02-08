#pragma once

// Resolution Configuration
// Change this value to modify the rendering resolution (e.g. 512, 1024)
#define MODEL_SIZE 512

// Enable Split Screen (Parallel Rendering: Original | Processed)
// #define SPLIT_SCREEN 0
#define SPLIT_SCREEN 1

// VAE Scaling Factor (from diffusers/FLUX.2-Tiny-AutoEncoder)
// Applied after encoder output, divided before decoder input
#define VAE_SCALING_FACTOR 0.13025f

// UNet Configuration
#define ENABLE_UNET 1
#define UNET_STEPS 1

// Latent dimensions (MODEL_SIZE / 16 for Flux Tiny VAE)
#define LATENT_SIZE (MODEL_SIZE / 16)
#define LATENT_CHANNELS 128

// RTX Video Super Resolution (VSR) Configuration
#define ENABLE_VSR 1 // 0 = Disable, 1 = Enable
#define VSR_SCALE 1.5  // Scale factor (e.g., 2 for 2x upscale, 1.5 for 1.5x)
