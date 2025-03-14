import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig


from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "stabilityai/sd-turbo", subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="text_encoder",
            # torch_dtype=torch.bfloat16,
        ).cuda()
        self.sched = make_1step_sched()
        # self.sched.betas = self.sched.betas.to(torch.bfloat16).cuda()
        # self.sched.alphas = self.sched.alphas.to(torch.bfloat16).cuda()
        # self.sched.one = self.sched.one.to(torch.bfloat16).cuda()
        # self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(torch.bfloat16).cuda()

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="vae",
            # variant="fp16",
            # torch_dtype=torch.bfloat16,
        )
        # это можно пофиксить если задать другие ключи для Sequential, тогда он будет правильно выбирать адаптеры
        # https://github.com/huggingface/peft/blob/b345a6e41521b977793cbdcaf932280081b18141/docs/source/developer_guides/custom_models.md?plain=1#L69
        # vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(
        #     device="cuda",
        #     dtype=torch.bfloat16,
        # )
        vae.encoder.forward = my_vae_encoder_fwd.__get__(
            vae.encoder, vae.encoder.__class__
        )
        vae.decoder.forward = my_vae_decoder_fwd.__get__(
            vae.decoder, vae.decoder.__class__
        )
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(
            512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(
            256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(
            128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(
            128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False
        ).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="unet",
            # variant="fp16",
            # torch_dtype=torch.bfloat16,
        )

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )
                with open(outf, "wb") as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"],
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"],
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = [
                "conv1",
                "conv2",
                "conv_in",
                "conv_shortcut",
                "conv",
                "conv_out",
                "skip_conv_1",
                "skip_conv_2",
                "skip_conv_3",
                "skip_conv_4",
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
            ]
            vae_lora_config = LoraConfig(
                r=lora_rank_vae,
                init_lora_weights="gaussian",
                target_modules=target_modules_vae,
            )
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "conv",
                "conv1",
                "conv2",
                "conv_shortcut",
                "conv_out",
                "proj_in",
                "proj_out",
                "ff.net.2",
                "ff.net.0.proj",
            ]
            unet_lora_config = LoraConfig(
                r=lora_rank_unet,
                init_lora_weights="gaussian",
                target_modules=target_modules_unet,
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.text_encoder.requires_grad_(False)

        self.cache_prompts = {}

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(
        self,
        c_t,
        prompt=None,
        prompt_tokens=None,
        deterministic=True,
        r=1.0,
        noise_map=None,
    ):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (
            prompt_tokens is None
        ), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        if deterministic:
            encoded_control = (
                self.vae.encode(c_t).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            model_pred = self.unet(
                encoded_control,
                self.timesteps,
                encoder_hidden_states=caption_enc,
            ).sample
            x_denoised = self.sched.step(
                model_pred, self.timesteps, encoded_control, return_dict=True
            ).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (
                self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
            ).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = (
                self.vae.encode(c_t).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(
                unet_input,
                self.timesteps,
                encoder_hidden_states=caption_enc,
            ).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(
                unet_output, self.timesteps, unet_input, return_dict=True
            ).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (
                self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
            ).clamp(-1, 1)
        return output_image

    def custom_forward(
        self,
        c_t,
        prompt=None,
        prompt_tokens=None,
        deterministic=True,
        r=1.0,
        noise_map=None,
    ):

        if prompt in self.cache_prompts:
            caption_enc = self.cache_prompts[prompt]
        else:
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
            self.cache_prompts[prompt] = caption_enc

        encoded_control = (
            # torch.Size([1, 4, 64, 64])
            self.vae.encode(c_t, return_dict=False)[0].sample()
            # self.vae.encode(c_t, return_dict=False)[0]
            * self.vae.config.scaling_factor
        )
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            encoder_hidden_states=caption_enc,
            return_dict=False,
        )[0]
        x_denoised = self.sched.step(
            model_pred,
            self.timesteps,
            encoded_control,
            return_dict=False,
        )[0]
        x_denoised = x_denoised.to(model_pred.dtype)
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (
            self.vae.decode(
                x_denoised / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        ).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {
            k: v
            for k, v in self.unet.state_dict().items()
            if "lora" in k or "conv_in" in k
        }
        sd["state_dict_vae"] = {
            k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k
        }
        torch.save(sd, outf)


unet2d_config = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "time_embedding_type": "positional",
    "freq_shift": 0,
    "flip_sin_to_cos": True,
    "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "downsample_padding": 1,
    "downsample_type": "conv",
    "upsample_type": "conv",
    "dropout": 0.0,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-05,
    "resnet_time_scale_shift": "default",
    "add_attention": False,
}

from diffusers import UNet2DModel
from diffusers import AutoencoderTiny
from diffusers import DDPMScheduler


class Pix2PixLight(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        sched = DDPMScheduler.from_pretrained(
            "stabilityai/sd-turbo",
            subfolder="scheduler",
        )
        sched.set_timesteps(1, device="cuda")
        sched.alphas_cumprod = sched.alphas_cumprod.cuda()
        sched.betas = sched.betas.to(dtype).cuda()
        sched.alphas = sched.alphas.to(dtype).cuda()
        sched.one = sched.one.to(dtype).cuda()
        sched.alphas_cumprod = sched.alphas_cumprod.to(dtype).cuda()
        self.sched = sched

        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd",
            torch_device="cuda",
            torch_dtype=dtype,
        ).cuda()

        vae.decoder.ignore_skip = False
        unet = UNet2DModel(**unet2d_config).to("cuda").to(dtype)

        # vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.unet = unet
        self.vae = vae

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()

    def forward(self, c_t):
        encoded_control = (
            self.vae.encode(c_t, False)[0] * self.vae.config.scaling_factor
        )
        model_pred = self.unet(
            encoded_control,
            self.timesteps,
            return_dict=False,
        )[0]
        x_denoised = self.sched.step(
            model_pred,
            self.timesteps,
            encoded_control,
            return_dict=False,
        )[0]
        output_image = (
            self.vae.decode(
                x_denoised / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        ).clamp(-1, 1)

        return output_image

    def save_model(self, outf):
        self.unet.save_pretrained(outf + "unet")
        self.vae.save_pretrained(outf + "vae")


unet2d_config_v2 = {
    "sample_size": 64,
    "in_channels": 4 * 4,
    "out_channels": 4 * 4,
    "center_input_sample": False,
    "time_embedding_type": "positional",
    "freq_shift": 0,
    "flip_sin_to_cos": True,
    "down_block_types": ("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 1,
    "mid_block_scale_factor": 1,
    "downsample_padding": 1,
    "downsample_type": "conv",
    "upsample_type": "conv",
    "dropout": 0.0,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-05,
    "resnet_time_scale_shift": "default",
    "add_attention": False,
}

# from diffusers import AutoencoderDC


# class Pix2PixLightV2(Pix2PixLight):
#     def __init__(self, dtype=torch.bfloat16):
#         super().__init__()
#         sched = DDPMScheduler.from_pretrained(
#             "stabilityai/sd-turbo",
#             subfolder="scheduler",
#         )
#         sched.set_timesteps(1, device="cuda")
#         sched.alphas_cumprod = sched.alphas_cumprod.cuda()
#         sched.betas = sched.betas.to(dtype).cuda()
#         sched.alphas = sched.alphas.to(dtype).cuda()
#         sched.one = sched.one.to(dtype).cuda()
#         sched.alphas_cumprod = sched.alphas_cumprod.to(dtype).cuda()
#         self.sched = sched

#         vae = (
#             AutoencoderDC(
#                 in_channels=3,
#                 latent_channels=16,
#                 attention_head_dim=32,
#                 encoder_block_types=[
#                     "ResBlock",
#                     "ResBlock",
#                     "ResBlock",
#                     "EfficientViTBlock",
#                     "EfficientViTBlock",
#                     "EfficientViTBlock",
#                 ],
#                 decoder_block_types=[
#                     "ResBlock",
#                     "ResBlock",
#                     "ResBlock",
#                     "EfficientViTBlock",
#                     "EfficientViTBlock",
#                     "EfficientViTBlock",
#                 ],
#                 encoder_block_out_channels=[32, 32, 32, 32, 32, 32],
#                 decoder_block_out_channels=(32, 32, 32, 32, 32, 32),
#                 encoder_layers_per_block=(1, 2, 2, 3, 3, 3),
#                 decoder_layers_per_block=(3, 3, 3, 3, 3, 1),
#                 encoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
#                 decoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
#                 upsample_block_type="interpolate",
#                 downsample_block_type="Conv",
#                 decoder_norm_types="rms_norm",
#                 decoder_act_fns="silu",
#                 scaling_factor=0.41407,
#             )
#             .to(dtype)
#             .cuda()
#         )

#         unet = UNet2DModel(**unet2d_config_v2).to("cuda").to(dtype)

#         self.timesteps = torch.tensor([999], device="cuda").long()
#         self.unet = unet
#         self.vae = vae
