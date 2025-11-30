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

- I posted https://github.com/madebyollin/seraena/blob/main/TAESDXL_Training_Example.ipynb which should work as a starting point (and it does most of the complicated adversarial loss part). You can try adding additional pooled-MSE / LPIPS losses to speed up convergence'


- Color augmentation details
- Nothing particularly exciting - this is the code I used (assuming [0, 1] images, applied to 5-10% of samples)
```python
def color_augment(im):
    scale = 0.5 + th.randn(3, 3)
    blend = th.rand(3, 1, 1)
    return (scale @ im.flatten(1)).view(im.shape).clamp(0, 1) * blend + (1 - blend) * im
```

The data range conventions are:

taesd.py: images are in [0, 1], latents are gaussian-distributed
diffusers.AutoencoderTiny images are in [-1, 1], latents are unit-normalized (you could apply the scale factor, but it's just 1.0)
diffusers.AutoencoderKL images are in [-1, 1], latents are not unit-normalized until you apply the scale factor
These ranges apply to both inputs and outputs. So your examples need to scale images before sending them to the SD VAE encoder. I think the correct pseudocode would be:

```python
'''training with taesd.py'''
x # [0,1]
SD_latent = SD_vae.encoder(x.mul(2).sub(1)) * vae_factor
taesd_latent = taesd.encoder(x) 
enc_loss = L2(SD_latent, taesd_latent)

taesd_output = taesd.decoder(SD_latent)
dec_loss = L2(x, taesd_output)

'''training with diffusers.AutoencoderTiny'''
x # [0,1]
SD_latent = SD_vae.encoder(x.mul(2).sub(1)) * vae_factor
taesd_latent = autoencodertiny_encoder(x.mul(2).sub(1)) 
enc_loss = L2(SD_latent, taesd_latent)

taesd_output = autoencodertiny_decoder(SD_latent) # auto convert to [-1,1]
# convert x to [-1,1]
dec_loss = L2(x.mul(2).sub(1), taesd_output)
```
I posted example TAESDXL training code here BTW, should be useful reference (specifically the DiffusersVAEWrapper portion).