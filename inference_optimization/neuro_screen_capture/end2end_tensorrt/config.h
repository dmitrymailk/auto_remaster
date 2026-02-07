#pragma once

// Resolution Configuration
// Change this value to modify the rendering resolution (e.g. 512, 1024)
#define MODEL_SIZE 512

// VAE Scaling Factor (from diffusers/FLUX.2-Tiny-AutoEncoder)
// Applied after encoder output, divided before decoder input
#define VAE_SCALING_FACTOR 0.13025f

// UNet Configuration
#define ENABLE_UNET 1
#define UNET_STEPS 1

// Latent dimensions (MODEL_SIZE / 16 for Flux Tiny VAE)
#define LATENT_SIZE (MODEL_SIZE / 16)
#define LATENT_CHANNELS 128
