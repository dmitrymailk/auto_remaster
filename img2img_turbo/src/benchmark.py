import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from img2img_turbo.src.pix2pix_turbo import Pix2Pix_Turbo
from img2img_turbo.src.image_prep import canny_from_pil
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from img2img_turbo.src.pix2pix_turbo import Pix2Pix_Turbo
from img2img_turbo.src.image_prep import canny_from_pil
import time

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


def merge_loras(model):
    model_modules = dict(model.named_modules())
    for module_key in model_modules:
        if "base_layer" in module_key:
            parent_module = ".".join(module_key.split(".")[:-1])
            prev_parent_module = ".".join(module_key.split(".")[:-2])
            if hasattr(model_modules[parent_module], "base_layer"):
                model_modules[parent_module].merge()
                setattr(
                    model_modules[prev_parent_module],
                    parent_module.split(".")[-1],
                    model_modules[module_key],
                )


def single_image(model, dataset, T, prompt):

    input_image = dataset[190]["input_image"].convert("RGB")
    i_t = T(input_image)
    c_t = F.to_tensor(i_t).unsqueeze(0).cuda()
    # c_t = c_t.half()
    c_t = c_t.to(torch.bfloat16)

    start = time.time()
    with torch.no_grad():
        # output_image = model(c_t, prompt)
        output_image = model.custom_forward(c_t, prompt)

        # output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    print("single image", time.time() - start)


def multiple_images(model, dataset, T, prompt):

    input_image = dataset[190]["input_image"].convert("RGB")
    total_images = 140
    images = [
        dataset[190 + i]["input_image"].convert("RGB") for i in range(total_images)
    ]
    images = [
        F.to_tensor(T(item)).unsqueeze(0).cuda().to(torch.bfloat16) for item in images
    ]

    start = time.time()
    for input_image in images:
        with torch.no_grad():
            # i_t = T(input_image)
            # c_t = F.to_tensor(i_t).unsqueeze(0).cuda()
            # c_t = c_t.half()
            # output_image = model(c_t, prompt)
            # output_image = model.custom_forward(c_t, prompt)
            output_image = model.custom_forward(input_image, prompt)

            # output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
    full_time = time.time() - start
    print("multiple_images", full_time)
    print("multiple_images fps", 1 / (full_time / 140))


if __name__ == "__main__":
    from datasets import load_dataset

    dataset_name = "dim/nfs_pix2pix_1920_1080_v5"
    # dataset_name = "dim/nfs_pix2pix_1920_1080_v6"
    dataset = load_dataset(dataset_name, num_proc=4)
    dataset = dataset["train"]

    model_name = ""
    model_path = "/code/img2img_turbo/models/model_20001.pkl"
    use_fp16 = not False

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=model_name, pretrained_path=model_path)
    merge_loras(model=model)
    model.set_eval()
    # if use_fp16:
    model.to(torch.bfloat16)
    model.unet.to(torch.bfloat16)
    model.vae.to(torch.bfloat16)
    model.unet.fuse_qkv_projections()
    # model.timesteps = 1
    # model.unet.to(memory_format=torch.channels_last)
    # model.vae.to(memory_format=torch.channels_last)
    # model.unet = torch.compile(model.unet, mode="reduce-overhead", fullgraph=not True)
    # model.vae.config.force_upcast = False
    # model.vae.decode = torch.compile(
    #     model.vae.decode, mode="reduce-overhead", fullgraph=not True
    # )

    T = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ]
    )
    prompt = dataset[0]["edit_prompt"]

    single_image(model, dataset, T, prompt)
    single_image(model, dataset, T, prompt)
    single_image(model, dataset, T, prompt)
    single_image(model, dataset, T, prompt)
    multiple_images(model, dataset, T, prompt)
    multiple_images(model, dataset, T, prompt)
    multiple_images(model, dataset, T, prompt)
    multiple_images(model, dataset, T, prompt)
    """
    single image 511.1411769390106
    single image 1.0843024253845215
    single image 0.03383207321166992
    single image 0.0336606502532959
    multiple_images 8.789534568786621
    multiple_images fps 15.928033379283555
    multiple_images 8.79971957206726
    multiple_images fps 15.909597897232844
    multiple_images 8.794561862945557
    multiple_images fps 15.918928331139158
    multiple_images 8.796127080917358
    multiple_images fps 15.916095653474715
    """
